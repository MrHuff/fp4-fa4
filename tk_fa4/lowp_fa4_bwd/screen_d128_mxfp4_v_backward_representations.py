#!/usr/bin/env python3
"""Screen D128 MXFP4 V representations for the backward dP operand.

This is a diagnostic, not a production route or a timing benchmark.  It calls
the production NVFP4 QKV projection twice on identical B2 inputs: once with
the current E4M3 backward publications and once with the already-implemented
backward-oriented MXFP4 publication.  It authenticates invariant BF16 and
forward-MX outputs byte-for-byte before decoding either backward operand.

The screen distinguishes two physically different MXFP4 V layouts:

* forward max-6: one E8M0 scale for 32 sequence values of one feature;
* backward max-6: one E8M0 scale for 32 depth values of one sequence row.

Readable variants preserve the backward grouping while changing only E2M1
rounding, normalization endpoint, or the E8M0 exponent.  Their decoded V is
then passed through an exact, readable causal GQA Jacobian with fixed Q, K,
dO, and probability.  Therefore V changes dP/dS/dQ/dK, but cannot change dV.
No claim about packed-MX kernel latency is allowed from this experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from tk_fa4 import (
    _C_b300_lowp_bwd,
    b300_pack_gqa_d128_rope,
    b300_pair_interleave_gqa_d128_qk_projection_weights,
    b300_prepare_nvfp4_projection_operand,
    b300_prepare_nvfp4_projection_weight,
    b300_project_qkv_gqa_d128_unified_lowp_nvfp4,
    b300_stack_gqa_d128_qkv_projection_weights,
)
from tk_fa4.lowp_fa4_bwd.projection_quantization_reference import (
    SIGNED_E2M1_LEVELS,
    decode_native_mxfp4_v,
)


E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--sequence", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--sr-draws", type=int, default=4)
    parser.add_argument(
        "--backward-endpoint",
        type=float,
        choices=(4.0, 6.0),
        default=6.0,
        help=(
            "E2M1 normalization endpoint used by the authenticated binary; "
            "use 4 only for the historical pre-width-six build"
        ),
    )
    parser.add_argument(
        "--backward-scale-selector",
        choices=("rte", "mse_1d"),
        default="mse_1d",
        help=(
            "E8M0 selector used by the authenticated binary; use rte only "
            "for the historical pre-MSE build"
        ),
    )
    parser.add_argument("--expected-extension", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_identity(root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    dirty = tuple(
        line
        for line in run("status", "--porcelain=v1").splitlines()
        if line
    )
    return {
        "root": str(root.resolve()),
        "head": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(dirty),
        "dirty_paths": list(dirty),
    }


def _write_create_only(path: Path, rendered: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(rendered)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _byte_comparison(
    left: torch.Tensor,
    right: torch.Tensor,
) -> dict[str, int | bool]:
    left_bytes = left.contiguous().view(torch.uint8)
    right_bytes = right.contiguous().view(torch.uint8)
    if tuple(left_bytes.shape) != tuple(right_bytes.shape):
        return {
            "equal": False,
            "left_bytes": left_bytes.numel(),
            "right_bytes": right_bytes.numel(),
            "mismatches": max(left_bytes.numel(), right_bytes.numel()),
        }
    mismatch = left_bytes != right_bytes
    return {
        "equal": bool(not mismatch.any()),
        "left_bytes": left_bytes.numel(),
        "right_bytes": right_bytes.numel(),
        "mismatches": int(mismatch.sum()),
    }


def _storage_overlap(left: torch.Tensor, right: torch.Tensor) -> bool:
    left_begin = left.data_ptr()
    left_end = left_begin + left.numel() * left.element_size()
    right_begin = right.data_ptr()
    right_end = right_begin + right.numel() * right.element_size()
    return max(left_begin, right_begin) < min(left_end, right_end)


def _decode_packed_e2m1(payload: torch.Tensor) -> torch.Tensor:
    packed = payload.contiguous().view(torch.uint8)
    levels = torch.tensor(
        SIGNED_E2M1_LEVELS,
        device=payload.device,
        dtype=torch.float32,
    )
    return torch.stack(
        (
            levels[(packed & 0x0F).long()],
            levels[(packed >> 4).long()],
        ),
        dim=-1,
    ).flatten(-2)


def _logical_backward_scale_codes(
    scales: torch.Tensor,
    *,
    sequence: int,
    depth: int,
) -> torch.Tensor:
    """Return the swizzled scale page as logical [B,S,H,D/32] bytes."""
    if scales.ndim != 4 or sequence % 128 or depth % 32:
        raise ValueError("invalid backward MX scale geometry")
    batch, pages, heads, page_values = scales.shape
    if pages != sequence // 128 or page_values != 512:
        raise ValueError("invalid backward MX scale pages")
    row = torch.arange(sequence, device=scales.device)
    depth_group = torch.arange(depth // 32, device=scales.device)
    page = row // 128
    offset = (
        (row % 32)[:, None] * 16
        + ((row // 32) % 4)[:, None] * 4
        + depth_group[None, :]
    )
    scale_bytes = scales.contiguous().view(torch.uint8)
    logical = torch.empty(
        batch,
        sequence,
        heads,
        depth // 32,
        device=scales.device,
        dtype=torch.uint8,
    )
    for batch_index in range(batch):
        for head_index in range(heads):
            logical[batch_index, :, head_index] = scale_bytes[
                batch_index,
                page[:, None],
                head_index,
                offset,
            ]
    return logical


def _logical_forward_scale_codes(
    scales: torch.Tensor,
    *,
    sequence: int,
    depth: int,
) -> torch.Tensor:
    """Return sequence-major scale pages as logical [B,H,D,S/32]."""
    batch, pages, heads, page_values = scales.shape
    if pages != sequence // 128 or page_values != 512 or depth != 128:
        raise ValueError("invalid forward MX scale pages")
    sequence_group = torch.arange(sequence // 32, device=scales.device)
    depth_index = torch.arange(depth, device=scales.device)
    page = (sequence_group // 4)[None].expand(depth, -1)
    offset = (
        (depth_index % 32)[:, None] * 16
        + (depth_index // 32)[:, None] * 4
        + (sequence_group % 4)[None]
    )
    scale_bytes = scales.contiguous().view(torch.uint8)
    logical = torch.empty(
        batch,
        heads,
        depth,
        sequence // 32,
        device=scales.device,
        dtype=torch.uint8,
    )
    for batch_index in range(batch):
        for head_index in range(heads):
            logical[batch_index, head_index] = scale_bytes[
                batch_index,
                page,
                head_index,
                offset,
            ]
    return logical


def _forward_scale_codes_from_bf16_amax(values: torch.Tensor) -> torch.Tensor:
    """Recompute the production 1x32 sequence-group MSE selector."""
    batch, sequence, heads, depth = values.shape
    blocks = (
        values.bfloat16()
        .float()
        .permute(0, 2, 3, 1)
        .contiguous()
        .reshape(batch, heads, depth, sequence // 32, 32)
    )
    amax = blocks.abs().amax(dim=-1)
    positive = amax > 0
    safe = amax.clamp_min(torch.finfo(torch.float32).tiny)
    lower_exponent = torch.floor(torch.log2(safe))
    normalized = safe / torch.exp2(lower_exponent)
    code = lower_exponent + 127.0 + (normalized >= 1.203125).float()
    code = code.clamp_(1.0, 254.0).to(torch.uint8)
    return torch.where(positive, code, torch.zeros_like(code))


def decode_native_backward_mxfp4_v(
    payload: torch.Tensor,
    scales: torch.Tensor,
    *,
    endpoint: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode production backward MXFP4 into logical [B,S,H,D]."""
    if payload.ndim != 4:
        raise ValueError("backward V payload must have shape [B,S,H,D/2]")
    batch, sequence, heads, packed_depth = payload.shape
    depth = packed_depth * 2
    if depth != 128:
        raise ValueError("this D128 diagnostic requires depth 128")
    scale_codes = _logical_backward_scale_codes(
        scales,
        sequence=sequence,
        depth=depth,
    )
    code_values = _decode_packed_e2m1(payload).reshape(
        batch,
        sequence,
        heads,
        depth,
    )
    exponent = scale_codes.float() - 127.0
    decode = torch.where(
        scale_codes > 0,
        torch.exp2(exponent) / endpoint,
        torch.zeros_like(exponent),
    )
    values = (
        code_values.reshape(batch, sequence, heads, depth // 32, 32)
        * decode[..., None]
    ).reshape(batch, sequence, heads, depth)
    return values, code_values, scale_codes


def _quantize_e2m1_rne(values: torch.Tensor) -> torch.Tensor:
    levels = torch.tensor(
        E2M1_LEVELS,
        device=values.device,
        dtype=torch.float32,
    )
    magnitude = values.float().abs()
    upper_index = torch.searchsorted(levels, magnitude).clamp_(1, 7)
    lower_index = upper_index - 1
    lower = levels[lower_index]
    upper = levels[upper_index]
    lower_distance = magnitude - lower
    upper_distance = upper - magnitude
    tie = lower_distance == upper_distance
    choose_upper = (upper_distance < lower_distance) | (
        tie & ((upper_index & 1) == 0)
    )
    selected = torch.where(choose_upper, upper_index, lower_index)
    selected = torch.where(
        magnitude >= levels[-1],
        torch.full_like(selected, 7),
        selected,
    )
    return levels[selected].copysign(values.float())


def _quantize_e2m1_stochastic(
    values: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    levels = torch.tensor(
        E2M1_LEVELS,
        device=values.device,
        dtype=torch.float32,
    )
    magnitude = values.float().abs()
    upper_index = torch.searchsorted(levels, magnitude).clamp_(1, 7)
    lower_index = upper_index - 1
    lower = levels[lower_index]
    upper = levels[upper_index]
    probability_upper = ((magnitude - lower) / (upper - lower)).clamp_(0, 1)
    choose_upper = torch.rand(
        magnitude.shape,
        device=magnitude.device,
        generator=generator,
        dtype=torch.float32,
    ) < probability_upper
    selected = lower_index + choose_upper.to(torch.long)
    selected = torch.where(
        magnitude >= levels[-1],
        torch.full_like(selected, 7),
        selected,
    )
    return levels[selected].copysign(values.float())


def _scale_codes_from_bf16_amax(
    values: torch.Tensor,
    *,
    selector: str,
) -> torch.Tensor:
    """Select E8M0 bytes for backward [B,S,H,D/32,32] blocks."""
    blocks = values.bfloat16().float().reshape(
        *values.shape[:-1],
        values.shape[-1] // 32,
        32,
    )
    amax = blocks.abs().amax(dim=-1)
    positive = amax > 0
    safe = amax.clamp_min(torch.finfo(torch.float32).tiny)
    lower_exponent = torch.floor(torch.log2(safe))
    normalized = safe / torch.exp2(lower_exponent)
    biased_lower = lower_exponent + 127.0
    if selector == "rte":
        round_up = (normalized > 1.5) | (
            (normalized == 1.5)
            & (biased_lower.to(torch.int32).bitwise_and(1) == 1)
        )
    elif selector == "mse_1d":
        round_up = normalized >= 1.203125
    elif selector == "ceil":
        round_up = normalized > 1.0
    elif selector == "floor":
        round_up = torch.zeros_like(positive)
    else:
        raise ValueError(f"unknown E8M0 selector {selector!r}")
    code = biased_lower + round_up.float()
    code = code.clamp_(1.0, 254.0).to(torch.uint8)
    return torch.where(positive, code, torch.zeros_like(code))


def _qdq_backward_mxfp4(
    values: torch.Tensor,
    scale_codes: torch.Tensor,
    *,
    endpoint: float,
    rounding: str,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | str]]:
    batch, sequence, heads, depth = values.shape
    expected_scales = (batch, sequence, heads, depth // 32)
    if tuple(scale_codes.shape) != expected_scales:
        raise ValueError(
            f"expected logical scales {expected_scales}, got "
            f"{tuple(scale_codes.shape)}"
        )
    blocks = values.float().reshape(batch, sequence, heads, depth // 32, 32)
    exponent = scale_codes.float() - 127.0
    decode = torch.where(
        scale_codes > 0,
        torch.exp2(exponent) / endpoint,
        torch.zeros_like(exponent),
    )
    normalized = blocks / decode.clamp_min(
        torch.finfo(torch.float32).tiny
    )[..., None]
    if rounding == "rne":
        codes = _quantize_e2m1_rne(normalized)
    elif rounding == "stochastic":
        if generator is None:
            raise ValueError("stochastic rounding requires a generator")
        codes = _quantize_e2m1_stochastic(normalized, generator=generator)
    else:
        raise ValueError("rounding must be rne or stochastic")
    decoded = (codes * decode[..., None]).reshape_as(values)
    nonzero = blocks != 0
    zeroed = nonzero & (codes == 0)
    difference = decoded.reshape_as(blocks) - blocks
    diagnostics: dict[str, float | str] = {
        "group_geometry": "one_sequence_row_by_32_depth_values",
        "endpoint": endpoint,
        "rounding": rounding,
        "payload_zero_fraction": float((codes == 0).float().mean()),
        "nonzero_input_zeroed_fraction": float(
            zeroed.sum() / nonzero.sum().clamp_min(1)
        ),
        "payload_saturation_fraction": float(
            (codes.abs() == 6).float().mean()
        ),
        "payload_clipping_fraction": float(
            (normalized.abs() > 6).float().mean()
        ),
        "signed_error_mean": float(difference.mean()),
        "rmse": float(difference.square().mean().sqrt()),
    }
    return decoded, codes.reshape_as(values), diagnostics


def _metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, float]:
    expected = reference.float().reshape(-1)
    actual = candidate.float().reshape(-1)
    difference = actual - expected
    expected_norm = torch.linalg.vector_norm(expected).clamp_min(1.0e-30)
    actual_norm = torch.linalg.vector_norm(actual).clamp_min(1.0e-30)
    return {
        "relative_l2": float(torch.linalg.vector_norm(difference) / expected_norm),
        "cosine": float(
            torch.dot(expected, actual) / (expected_norm * actual_norm)
        ),
        "norm_ratio": float(actual_norm / expected_norm),
        "rmse": float(difference.square().mean().sqrt()),
        "mae": float(difference.abs().mean()),
        "max_abs_error": float(difference.abs().max()),
        "signed_error_mean": float(difference.mean()),
    }


def _zero_diagnostics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, float]:
    reference_nonzero = reference != 0
    return {
        "zero_fraction": float((candidate == 0).float().mean()),
        "nonzero_reference_zeroed_fraction": float(
            (reference_nonzero & (candidate == 0)).sum()
            / reference_nonzero.sum().clamp_min(1)
        ),
    }


def _make_rope(
    batch: int,
    sequence: int,
    device: torch.device,
) -> torch.Tensor:
    positions = torch.arange(sequence, device=device, dtype=torch.float32)
    frequencies = 1.0 / (
        10_000.0
        ** (torch.arange(64, device=device, dtype=torch.float32) / 64.0)
    )
    angles = positions[:, None] * frequencies[None]
    cosine = angles.cos()[None].repeat(batch, 1, 1).bfloat16().contiguous()
    sine = angles.sin()[None].repeat(batch, 1, 1).bfloat16().contiguous()
    return b300_pack_gqa_d128_rope(cosine, sine)


def _causal_probability(
    q: torch.Tensor,
    k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_heads = q.shape[2]
    kv_heads = k.shape[2]
    group_size = q_heads // kv_heads
    q_h = q.float().permute(0, 2, 1, 3).contiguous()
    k_h = k.float().permute(0, 2, 1, 3).contiguous()
    k_expanded = k_h.repeat_interleave(group_size, dim=1)
    scores = torch.matmul(q_h, k_expanded.transpose(-1, -2)) / math.sqrt(
        q.shape[-1]
    )
    causal = torch.ones(
        q.shape[1],
        q.shape[1],
        device=q.device,
        dtype=torch.bool,
    ).triu_(1)
    scores.masked_fill_(causal, -torch.inf)
    probability = torch.softmax(scores, dim=-1)
    return probability, q_h, k_expanded


def _backward_v_dependent_state(
    probability: torch.Tensor,
    q_h: torch.Tensor,
    k_expanded: torch.Tensor,
    v: torch.Tensor,
    dout_h: torch.Tensor,
    *,
    kv_heads: int,
) -> dict[str, torch.Tensor]:
    batch, q_heads, sequence, depth = q_h.shape
    group_size = q_heads // kv_heads
    v_expanded = (
        v.float()
        .permute(0, 2, 1, 3)
        .contiguous()
        .repeat_interleave(group_size, dim=1)
    )
    dp = torch.matmul(dout_h, v_expanded.transpose(-1, -2))
    ds = probability * (
        dp - (dp * probability).sum(dim=-1, keepdim=True)
    )
    inverse_depth_scale = 1.0 / math.sqrt(depth)
    dq = torch.matmul(ds, k_expanded) * inverse_depth_scale
    dk_expanded = torch.matmul(ds.transpose(-1, -2), q_h) * inverse_depth_scale
    dk = dk_expanded.reshape(
        batch,
        kv_heads,
        group_size,
        sequence,
        depth,
    ).sum(dim=2)
    return {"dp": dp, "ds": ds, "dq": dq, "dk": dk}


def _state_metrics(
    reference: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
) -> dict[str, dict[str, float]]:
    return {name: _metrics(reference[name], candidate[name]) for name in reference}


@torch.no_grad()
def main() -> None:
    args = _parse_args()
    if args.batch <= 0 or args.sequence <= 0 or args.hidden <= 0:
        raise ValueError("batch, sequence, and hidden must be positive")
    if args.sequence % 256 or args.hidden % 256:
        raise ValueError("sequence and hidden must be divisible by 256")
    if args.q_heads <= 0 or args.kv_heads <= 0:
        raise ValueError("head counts must be positive")
    if args.q_heads % args.kv_heads:
        raise ValueError("q-heads must be divisible by kv-heads")
    if args.sr_draws <= 0:
        raise ValueError("sr-draws must be positive")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    started = time.time()
    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda", args.gpu)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats(device)

    extension_path = Path(_C_b300_lowp_bwd.__file__).resolve()
    if args.expected_extension is not None:
        expected = args.expected_extension.resolve()
        if extension_path != expected:
            raise RuntimeError(
                f"loaded extension {extension_path}, expected {expected}"
            )

    activation = torch.randn(
        args.batch,
        args.sequence,
        args.hidden,
        device=device,
        dtype=torch.float32,
    )
    activation.mul_(
        torch.rsqrt(activation.square().mean(dim=-1, keepdim=True) + 1.0e-5)
    )
    activation = activation.bfloat16().contiguous()
    depth = 128
    q_weight = (
        torch.randn(
            args.q_heads * depth,
            args.hidden,
            device=device,
            dtype=torch.float32,
        )
        * 0.02
    ).bfloat16()
    k_weight = (
        torch.randn(
            args.kv_heads * depth,
            args.hidden,
            device=device,
            dtype=torch.float32,
        )
        * 0.02
    ).bfloat16()
    v_weight = torch.randn_like(k_weight.float()).mul_(0.02).bfloat16()
    q_weight, k_weight = b300_pair_interleave_gqa_d128_qk_projection_weights(
        q_weight.contiguous(),
        k_weight.contiguous(),
    )
    qkv_weight = b300_stack_gqa_d128_qkv_projection_weights(
        q_weight,
        k_weight,
        v_weight.contiguous(),
    )
    activation_operand = tuple(
        b300_prepare_nvfp4_projection_operand(
            activation.reshape(args.batch * args.sequence, args.hidden)
        )
    )
    weight_operand = tuple(b300_prepare_nvfp4_projection_weight(qkv_weight))
    qk_scales = torch.zeros(
        args.batch,
        args.q_heads,
        7,
        device=device,
        dtype=torch.float32,
    )
    qk_scales[..., 0] = 2.25
    qk_scales[..., 1] = 2.0
    common = {
        "input_operand": activation_operand,
        "qkv_weight_operand": weight_operand,
        "qk_scales": qk_scales,
        "batch": args.batch,
        "seqlen": args.sequence,
        "q_heads": args.q_heads,
        "kv_heads": args.kv_heads,
        "store_bf16": True,
        "v_mxfp4_scale_2d": False,
        # The legacy MX-backward publication specialization predates the
        # independent per-block Q/K selector.  Keep this false in both calls
        # so the V publication policy is the only changed launch argument.
        "per_block_qk_scales": False,
        # V is not transformed by RoPE.  The no-RoPE specialization keeps
        # this diagnostic focused on the V epilogue and also exercises the
        # same payload code without clustered scheduling confounders.
        "cluster_cap": 0,
        "cache_packed_rope": False,
        "cache_adaptive_qk_scale": False,
    }
    fp8_bundle = b300_project_qkv_gqa_d128_unified_lowp_nvfp4(
        **common,
        publish_fp8_backward=True,
    )
    mx_bundle = b300_project_qkv_gqa_d128_unified_lowp_nvfp4(
        **common,
        publish_fp8_backward=False,
    )
    torch.cuda.synchronize(device)

    repeat_fields = {
        "fp8_policy": (
            fp8_bundle,
            True,
            (
                "v_forward_fp4",
                "v_forward_scales",
                "v_backward_fp8",
                "q_backward_fp8",
                "k_backward_fp8",
                "v",
            ),
        ),
        "mx_policy": (
            mx_bundle,
            False,
            (
                "v_forward_fp4",
                "v_forward_scales",
                "v_backward_fp4",
                "v_backward_scales",
                "v",
            ),
        ),
    }
    deterministic_repeats: dict[str, Any] = {}
    for policy_name, (
        reference_bundle,
        publish_fp8_backward,
        field_names,
    ) in repeat_fields.items():
        comparisons: list[dict[str, dict[str, int | bool]]] = []
        for _ in range(2):
            repeated = b300_project_qkv_gqa_d128_unified_lowp_nvfp4(
                **common,
                publish_fp8_backward=publish_fp8_backward,
            )
            field_comparisons: dict[str, dict[str, int | bool]] = {}
            for field_name in field_names:
                reference_tensor = getattr(reference_bundle, field_name)
                repeated_tensor = getattr(repeated, field_name)
                if reference_tensor is None or repeated_tensor is None:
                    raise RuntimeError(
                        f"repeat omitted {policy_name}.{field_name}"
                    )
                field_comparisons[field_name] = _byte_comparison(
                    reference_tensor,
                    repeated_tensor,
                )
            comparisons.append(field_comparisons)
            del repeated
        deterministic_repeats[policy_name] = {
            "calls": 3,
            "repeat_comparisons_to_first": comparisons,
            "all_fields_bitwise_deterministic": all(
                bool(comparison["equal"])
                for repeat in comparisons
                for comparison in repeat.values()
            ),
        }

    if fp8_bundle.v_backward_fp8 is None:
        raise RuntimeError("FP8 projection omitted backward V")
    if mx_bundle.v_backward_fp4.numel() == 0:
        raise RuntimeError("MX projection omitted backward V")
    storage_alias_checks = {
        "mx_forward_backward_payload_overlap": _storage_overlap(
            mx_bundle.v_forward_fp4,
            mx_bundle.v_backward_fp4,
        ),
        "mx_forward_backward_scale_overlap": _storage_overlap(
            mx_bundle.v_forward_scales,
            mx_bundle.v_backward_scales,
        ),
        "fp8_forward_backward_v_overlap": (
            False
            if fp8_bundle.v_forward_fp8 is None
            else _storage_overlap(
                fp8_bundle.v_forward_fp8,
                fp8_bundle.v_backward_fp8,
            )
        ),
    }
    invariant_pairs = {
        "bf16_q": (fp8_bundle.q, mx_bundle.q),
        "bf16_k": (fp8_bundle.k, mx_bundle.k),
        "bf16_v": (fp8_bundle.v, mx_bundle.v),
        "forward_q_payload": (
            fp8_bundle.q_forward_fp4,
            mx_bundle.q_forward_fp4,
        ),
        "forward_q_scales": (
            fp8_bundle.q_forward_scales,
            mx_bundle.q_forward_scales,
        ),
        "forward_k_payload": (
            fp8_bundle.k_forward_fp4,
            mx_bundle.k_forward_fp4,
        ),
        "forward_k_scales": (
            fp8_bundle.k_forward_scales,
            mx_bundle.k_forward_scales,
        ),
        "forward_v_payload": (
            fp8_bundle.v_forward_fp4,
            mx_bundle.v_forward_fp4,
        ),
        "forward_v_scales": (
            fp8_bundle.v_forward_scales,
            mx_bundle.v_forward_scales,
        ),
    }
    invariants: dict[str, dict[str, int | bool]] = {}
    for name, (left, right) in invariant_pairs.items():
        if left is None or right is None:
            raise RuntimeError(f"projection omitted invariant {name}")
        invariants[name] = _byte_comparison(left, right)
    # Changing only the backward V representation must not alter any forward
    # operand.  In particular, this catches a physical [B,H,S/128,512] scale
    # page being decoded through the public [B,S/128,H,512] contract.
    strict_invariant_names = tuple(invariants)
    if not all(
        bool(invariants[name]["equal"]) for name in strict_invariant_names
    ):
        changed = {
            name: comparison
            for name, comparison in invariants.items()
            if name in strict_invariant_names and not bool(comparison["equal"])
        }
        raise RuntimeError(
            "backward publication policy changed forward outputs: "
            + json.dumps(changed, sort_keys=True)
        )

    assert fp8_bundle.q is not None
    assert fp8_bundle.k is not None
    assert fp8_bundle.v is not None
    v_bf16 = fp8_bundle.v.float()
    v_e4m3 = fp8_bundle.v_backward_fp8.float().mul_(0.25)
    v_forward_max6 = decode_native_mxfp4_v(
        fp8_bundle.v_forward_fp4,
        fp8_bundle.v_forward_scales,
    )
    v_forward_max6_mx_policy = decode_native_mxfp4_v(
        mx_bundle.v_forward_fp4,
        mx_bundle.v_forward_scales,
    )
    expected_forward_scale_codes = _forward_scale_codes_from_bf16_amax(
        v_bf16
    )
    fp8_policy_logical_forward_scales = _logical_forward_scale_codes(
        fp8_bundle.v_forward_scales,
        sequence=args.sequence,
        depth=depth,
    )
    mx_policy_logical_forward_scales = _logical_forward_scale_codes(
        mx_bundle.v_forward_scales,
        sequence=args.sequence,
        depth=depth,
    )
    fp8_policy_forward_scale_bytes = (
        fp8_bundle.v_forward_scales.contiguous().view(torch.uint8)
    )
    mx_policy_forward_scale_bytes = (
        mx_bundle.v_forward_scales.contiguous().view(torch.uint8)
    )
    forward_scale_mismatch = (
        fp8_policy_forward_scale_bytes != mx_policy_forward_scale_bytes
    )
    forward_scale_mismatch_grid = forward_scale_mismatch.reshape(
        args.batch,
        args.sequence // 128,
        args.kv_heads,
        32,
        4,
        4,
    ).sum(dim=(0, 1, 2, 3))
    forward_scale_policy_isolation = {
        "tensor_storage_alias": (
            fp8_bundle.v_forward_scales.data_ptr()
            == mx_bundle.v_forward_scales.data_ptr()
        ),
        "mismatches_by_depth_group_then_sequence_quarter": (
            forward_scale_mismatch_grid.cpu().tolist()
        ),
        "fp8_policy_zero_scale_byte_fraction": float(
            (fp8_policy_forward_scale_bytes == 0).float().mean()
        ),
        "mx_policy_zero_scale_byte_fraction": float(
            (mx_policy_forward_scale_bytes == 0).float().mean()
        ),
        "fp8_policy_ff_scale_byte_fraction": float(
            (fp8_policy_forward_scale_bytes == 0xFF).float().mean()
        ),
        "mx_policy_ff_scale_byte_fraction": float(
            (mx_policy_forward_scale_bytes == 0xFF).float().mean()
        ),
        "decoded_forward_v_policy_delta": _metrics(
            v_forward_max6,
            v_forward_max6_mx_policy,
        ),
        "decoded_mx_policy_isfinite": bool(
            torch.isfinite(v_forward_max6_mx_policy).all()
        ),
        "fp8_policy_matches_independent_1d_mse_selector": _byte_comparison(
            fp8_policy_logical_forward_scales,
            expected_forward_scale_codes,
        ),
        "mx_policy_matches_independent_1d_mse_selector": _byte_comparison(
            mx_policy_logical_forward_scales,
            expected_forward_scale_codes,
        ),
    }
    (
        v_backward_actual,
        backward_code_values,
        backward_scale_codes,
    ) = decode_native_backward_mxfp4_v(
        mx_bundle.v_backward_fp4,
        mx_bundle.v_backward_scales,
        endpoint=args.backward_endpoint,
    )

    expected_backward_scale_codes = _scale_codes_from_bf16_amax(
        v_bf16,
        selector=args.backward_scale_selector,
    )
    scale_authentication = _byte_comparison(
        backward_scale_codes,
        expected_backward_scale_codes,
    )
    proxy_actual, proxy_actual_codes, proxy_actual_diagnostics = (
        _qdq_backward_mxfp4(
            v_bf16,
            backward_scale_codes,
            endpoint=args.backward_endpoint,
            rounding="rne",
        )
    )
    payload_authentication = {
        "e2m1_code_values_equal": bool(
            torch.equal(backward_code_values, proxy_actual_codes)
        ),
        "e2m1_code_value_mismatches": int(
            (backward_code_values != proxy_actual_codes).sum()
        ),
        "decoded_values_equal": bool(
            torch.equal(v_backward_actual, proxy_actual)
        ),
        "decoded_value_mismatches": int(
            (v_backward_actual != proxy_actual).sum()
        ),
    }
    if not bool(scale_authentication["equal"]):
        raise RuntimeError(
            "backward MX scale bytes do not match configured "
            f"{args.backward_scale_selector} selector"
        )
    if not payload_authentication["e2m1_code_values_equal"]:
        raise RuntimeError("backward MX payload does not match readable RNE")

    variants: dict[str, tuple[torch.Tensor, dict[str, Any]]] = {
        "e4m3_current": (
            v_e4m3,
            {"source": "authenticated_projection_e4m3_x4_decoded_x0.25"},
        ),
        "forward_mx_max6_actual": (
            v_forward_max6,
            {
                "source": "authenticated_forward_feature_major_mx",
                "group_geometry": "one_feature_by_32_sequence_values",
                "endpoint": 6.0,
            },
        ),
        "backward_mx_actual": (
            v_backward_actual,
            {
                "source": "authenticated_backward_sequence_major_mx",
                **proxy_actual_diagnostics,
            },
        ),
    }

    shifted_down = torch.where(
        backward_scale_codes > 1,
        backward_scale_codes - 1,
        backward_scale_codes,
    )
    shifted_up = torch.where(
        (backward_scale_codes > 0) & (backward_scale_codes < 254),
        backward_scale_codes + 1,
        backward_scale_codes,
    )
    rte_scale_codes = _scale_codes_from_bf16_amax(
        v_bf16,
        selector="rte",
    )
    mse_scale_codes = _scale_codes_from_bf16_amax(
        v_bf16,
        selector="mse_1d",
    )
    endpoint_label = int(args.backward_endpoint)
    readable_specs = {
        f"backward_mx_max{endpoint_label}_scale_minus1_rne": (
            shifted_down,
            args.backward_endpoint,
            "rne",
        ),
        f"backward_mx_max{endpoint_label}_scale_plus1_rne": (
            shifted_up,
            args.backward_endpoint,
            "rne",
        ),
        "backward_mx_max4_rte_scale_rne": (
            rte_scale_codes,
            4.0,
            "rne",
        ),
        "backward_mx_max6_rte_scale_rne": (
            rte_scale_codes,
            6.0,
            "rne",
        ),
        "backward_mx_max6_mse_scale_rne": (
            mse_scale_codes,
            6.0,
            "rne",
        ),
    }
    for name, (scale_codes, endpoint, rounding) in readable_specs.items():
        values, _, diagnostics = _qdq_backward_mxfp4(
            v_bf16,
            scale_codes,
            endpoint=endpoint,
            rounding=rounding,
        )
        variants[name] = (
            values,
            {
                "source": "readable_backward_geometry_proxy",
                **diagnostics,
            },
        )

    sr_values: list[torch.Tensor] = []
    sr_v_metrics: list[dict[str, float]] = []
    sr_diagnostics: list[dict[str, float | str]] = []
    for draw in range(args.sr_draws):
        generator = torch.Generator(device=device).manual_seed(
            args.seed + 1_000_003 + draw * 97
        )
        values, _, diagnostics = _qdq_backward_mxfp4(
            v_bf16,
            backward_scale_codes,
            endpoint=args.backward_endpoint,
            rounding="stochastic",
            generator=generator,
        )
        sr_values.append(values)
        sr_v_metrics.append(_metrics(v_e4m3, values))
        sr_diagnostics.append(diagnostics)
    variants[f"backward_mx_max{endpoint_label}_sr_draw0"] = (
        sr_values[0],
        {
            "source": "readable_backward_geometry_sr_proxy",
            "seed": args.seed + 1_000_003,
            **sr_diagnostics[0],
        },
    )
    variants[
        f"backward_mx_max{endpoint_label}_sr_mean{args.sr_draws}"
    ] = (
        torch.stack(sr_values).mean(dim=0),
        {
            "source": "nondeployable_mean_of_multiple_sr_draws",
            "draws": args.sr_draws,
        },
    )

    q_e4m3 = fp8_bundle.q_backward_fp8
    k_e4m3 = fp8_bundle.k_backward_fp8
    if q_e4m3 is None or k_e4m3 is None:
        raise RuntimeError("FP8 projection omitted backward Q/K")
    q = q_e4m3.float().mul_(0.25)
    k = k_e4m3.float().mul_(0.25)
    probability, q_h, k_expanded = _causal_probability(q, k)
    dout = (
        torch.randn(
            args.batch,
            args.sequence,
            args.q_heads,
            depth,
            device=device,
            dtype=torch.float32,
        )
        * 0.02
    ).bfloat16()
    # Match the current backward E4M3 operand lift while keeping dO fixed.
    dout_e4m3 = (
        (dout.float() * 4.0).to(torch.float8_e4m3fn).float() * 0.25
    )
    dout_h = dout_e4m3.permute(0, 2, 1, 3).contiguous()
    reference_state = _backward_v_dependent_state(
        probability,
        q_h,
        k_expanded,
        v_e4m3,
        dout_h,
        kv_heads=args.kv_heads,
    )

    results: dict[str, Any] = {}
    for name, (values, diagnostics) in variants.items():
        state = (
            reference_state
            if name == "e4m3_current"
            else _backward_v_dependent_state(
                probability,
                q_h,
                k_expanded,
                values,
                dout_h,
                kv_heads=args.kv_heads,
            )
        )
        results[name] = {
            "diagnostics": diagnostics,
            "v_vs_bf16_projection": _metrics(v_bf16, values),
            "v_vs_current_e4m3": _metrics(v_e4m3, values),
            "zeros_vs_bf16_projection": _zero_diagnostics(v_bf16, values),
            "backward_vs_current_e4m3": (
                {
                    tensor_name: {
                        "relative_l2": 0.0,
                        "cosine": 1.0,
                        "norm_ratio": 1.0,
                        "rmse": 0.0,
                        "mae": 0.0,
                        "max_abs_error": 0.0,
                        "signed_error_mean": 0.0,
                    }
                    for tensor_name in reference_state
                }
                if name == "e4m3_current"
                else _state_metrics(reference_state, state)
            ),
        }
        if state is not reference_state:
            del state

    sr_summary: dict[str, dict[str, float]] = {}
    for metric_name in sr_v_metrics[0]:
        values = [metrics[metric_name] for metrics in sr_v_metrics]
        sr_summary[metric_name] = {
            "mean": statistics.fmean(values),
            "minimum": min(values),
            "maximum": max(values),
            "standard_deviation": statistics.pstdev(values),
        }

    publication_bytes = {
        "forward_mx_payload": fp8_bundle.v_forward_fp4.numel()
        * fp8_bundle.v_forward_fp4.element_size(),
        "forward_mx_scales": fp8_bundle.v_forward_scales.numel()
        * fp8_bundle.v_forward_scales.element_size(),
        "backward_mx_payload": mx_bundle.v_backward_fp4.numel()
        * mx_bundle.v_backward_fp4.element_size(),
        "backward_mx_scales": mx_bundle.v_backward_scales.numel()
        * mx_bundle.v_backward_scales.element_size(),
        "backward_e4m3_v": fp8_bundle.v_backward_fp8.numel()
        * fp8_bundle.v_backward_fp8.element_size(),
        "feature_major_e4m3_v": (
            0
            if fp8_bundle.v_forward_fp8 is None
            else fp8_bundle.v_forward_fp8.numel()
            * fp8_bundle.v_forward_fp8.element_size()
        ),
    }
    publication_bytes["mx_backward_instead_of_e4m3_bytes_saved"] = (
        publication_bytes["backward_e4m3_v"]
        - publication_bytes["backward_mx_payload"]
        - publication_bytes["backward_mx_scales"]
    )

    torch.cuda.synchronize(device)
    repo_root = Path(__file__).resolve().parents[2]
    result = {
        "schema": "d128_mxfp4_v_backward_representations_v2",
        "status": "diagnostic_not_production_timing",
        "configuration": {
            "seed": args.seed,
            "batch": args.batch,
            "sequence": args.sequence,
            "hidden": args.hidden,
            "q_heads": args.q_heads,
            "kv_heads": args.kv_heads,
            "head_dim": depth,
            "sr_draws": args.sr_draws,
            "backward_endpoint": args.backward_endpoint,
            "backward_scale_selector": args.backward_scale_selector,
        },
        "interpretation_contract": {
            "current_reference": "projection_published_E4M3_x4_decoded_x0.25",
            "backward_mx_actual": (
                "projection_published_BSHD_packed_E2M1_with_one_E8M0_scale_"
                "per_sequence_row_by_depth32_group_configured_endpoint_and_"
                "configured_scale_selector"
            ),
            "forward_mx_actual": (
                "projection_published_BHDS_packed_E2M1_with_one_E8M0_scale_"
                "per_feature_by_sequence32_group_and_endpoint6"
            ),
            "fixed_operands": ["Q", "K", "dO", "causal probability"],
            "v_affects": ["dP", "dS", "dQ", "dK"],
            "v_does_not_affect": ["dV"],
            "ste": "quantizer derivatives are not taken",
            "timing_claim_allowed": False,
            "sr_mean_is_deployable_one_pass": False,
        },
        "authentication": {
            "extension_symbol": (
                "project_qkv_gqa_d128_unified_fp4_nvfp4"
            ),
            "call_flag_delta": {
                "identical_flags": {
                    "store_bf16": True,
                    "v_mxfp4_scale_2d": False,
                    "per_block_qk_scales": False,
                    "rope": False,
                },
                "only_changed_flag": "publish_fp8_backward",
            },
            "deterministic_three_call_repeats": deterministic_repeats,
            "storage_alias_checks": storage_alias_checks,
            "publication_policy_invariants": invariants,
            "forward_v_scale_policy_isolation": (
                forward_scale_policy_isolation
            ),
            "backward_scale_selector": scale_authentication,
            "backward_payload_rne": payload_authentication,
        },
        "publication_bytes": publication_bytes,
        "sr_draw_v_vs_current_e4m3_summary": sr_summary,
        "variants": results,
        "provenance": {
            "git": _git_identity(repo_root),
            "extension": {
                "path": str(extension_path),
                "sha256": _sha256(extension_path),
                "bytes": extension_path.stat().st_size,
            },
            "script": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256(Path(__file__).resolve()),
            },
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
        },
        "resources": {
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device)
            / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
            "elapsed_seconds": time.time() - started,
        },
    }
    _write_create_only(args.output, json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
