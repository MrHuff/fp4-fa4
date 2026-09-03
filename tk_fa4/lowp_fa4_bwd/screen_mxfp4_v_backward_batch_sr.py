#!/usr/bin/env python3
"""Screen batch aggregation and true SR for represented MXFP4 backward V.

The deployed MXFP4-PV training route deliberately retains direct E4M3 V for
the canonical backward.  This diagnostic asks a narrower question without
changing that production ABI: if backward instead sees the forward MXFP4 V
representation lifted into E4M3, does its V-dependent dP -> dQ/dK error
average at larger physical batches, and does true stochastic E2M1 rounding
reduce bias or zeroed signal?

Projection publications come from the production CUDA extension.  The
deterministic represented-MX case is therefore exact.  The stochastic case is
an explicitly labeled numerical proxy: direct projection-accumulator E4M3 V
is QDQ'd with the production 1x32 E8M0 scale policy and true random E2M1
rounding, then consumed by a readable causal-GQA backward.  It is not a timing
result and it is not a packed-MX backward kernel.
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
    B300UnifiedLowpQKV,
    b300_pack_gqa_d64_paired_rope,
    b300_prepare_e4m3_projection_operand,
    b300_prepare_e4m3_projection_weight,
    b300_project_qkv_gqa_d64_paired_unified_lowp_e4m3,
)
from tk_fa4.lowp_fa4_bwd.projection_quantization_reference import (
    fake_quantize_mxfp4_v_1d,
    tensor_error_metrics,
)
from tk_fa4.lowp_fa4_bwd.validate_experimental_split_v_publication import (
    BACKWARD_QK_FIELDS,
    FORWARD_FIELDS,
    _byte_comparison,
    _contract_view,
    _extension_identity,
    _publications,
)


AUTHENTICATED_PRODUCTION_BATCHES = (1, 2, 8, 16)


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
        line for line in run("status", "--porcelain=v1").splitlines() if line
    )
    return {
        "root": str(root.resolve()),
        "head": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(dirty),
        "dirty_paths": list(dirty),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _make_rope(
    batch: int,
    sequence: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positions = torch.arange(sequence, device=device, dtype=torch.float32)
    frequencies = 1.0 / (
        10_000.0
        ** (
            torch.arange(32, device=device, dtype=torch.float32)
            / 32.0
        )
    )
    angles = positions[:, None] * frequencies[None, :]
    cosine = angles.cos()[None].repeat(batch, 1, 1).bfloat16().contiguous()
    sine = angles.sin()[None].repeat(batch, 1, 1).bfloat16().contiguous()
    packed = b300_pack_gqa_d64_paired_rope(cosine, sine)
    return cosine, sine, packed


def _project(
    input_operand: tuple[torch.Tensor, torch.Tensor],
    weight_operand: tuple[torch.Tensor, torch.Tensor],
    qk_scales: torch.Tensor,
    rope: torch.Tensor,
    *,
    batch: int,
    sequence: int,
    q_heads: int,
    kv_heads: int,
    split_v_backward: bool,
) -> B300UnifiedLowpQKV:
    return b300_project_qkv_gqa_d64_paired_unified_lowp_e4m3(
        input_operand,
        weight_operand,
        qk_scales,
        rope,
        batch=batch,
        seqlen=sequence,
        q_heads=q_heads,
        kv_heads=kv_heads,
        publish_mxfp4_v=True,
        represented_backward=True,
        per_block_qk_scales=True,
        experimental_split_v_backward=split_v_backward,
        v_mxfp4_scale_2d=False,
    )


def _inverse_pair_rope(
    gradient: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    depth = gradient.shape[-1]
    pairs = gradient.float().reshape(*gradient.shape[:-1], depth // 2, 2)
    first, second = pairs[..., 0], pairs[..., 1]
    cosine_f = cosine.float().unsqueeze(2)
    sine_f = sine.float().unsqueeze(2)
    return torch.stack(
        (
            first * cosine_f + second * sine_f,
            -first * sine_f + second * cosine_f,
        ),
        dim=-1,
    ).flatten(-2)


def _causal_probability(
    q: torch.Tensor,
    k: torch.Tensor,
) -> torch.Tensor:
    q_heads = q.shape[2]
    kv_heads = k.shape[2]
    if q_heads % kv_heads:
        raise ValueError("Q heads must be divisible by KV heads")
    q_h = q.float().permute(0, 2, 1, 3)
    k_h = k.float().permute(0, 2, 1, 3).repeat_interleave(
        q_heads // kv_heads,
        dim=1,
    )
    scores = torch.matmul(q_h, k_h.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    causal = torch.ones(
        q.shape[1],
        q.shape[1],
        device=q.device,
        dtype=torch.bool,
    ).triu_(1)
    return torch.softmax(scores.masked_fill(causal, -torch.inf), dim=-1)


def _backward_state(
    probability: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    rows: torch.Tensor,
    qkv_weight: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> dict[str, torch.Tensor]:
    batch, sequence, q_heads, depth = q.shape
    kv_heads = k.shape[2]
    group_size = q_heads // kv_heads
    q_h = q.float().permute(0, 2, 1, 3)
    k_h = k.float().permute(0, 2, 1, 3)
    v_h = v.float().permute(0, 2, 1, 3)
    dout_h = dout.float().permute(0, 2, 1, 3)
    k_expanded = k_h.repeat_interleave(group_size, dim=1)
    v_expanded = v_h.repeat_interleave(group_size, dim=1)

    dp = torch.matmul(dout_h, v_expanded.transpose(-1, -2))
    ds = probability * (
        dp - (dp * probability).sum(dim=-1, keepdim=True)
    )
    inverse_depth_scale = 1.0 / math.sqrt(depth)
    dq_h = torch.matmul(ds, k_expanded) * inverse_depth_scale
    dk_expanded = torch.matmul(ds.transpose(-1, -2), q_h) * inverse_depth_scale
    dv_expanded = torch.matmul(probability.transpose(-1, -2), dout_h)
    dk_h = dk_expanded.reshape(
        batch,
        kv_heads,
        group_size,
        sequence,
        depth,
    ).sum(dim=2)
    dv_h = dv_expanded.reshape(
        batch,
        kv_heads,
        group_size,
        sequence,
        depth,
    ).sum(dim=2)
    dq = dq_h.permute(0, 2, 1, 3).contiguous()
    dk = dk_h.permute(0, 2, 1, 3).contiguous()
    dv = dv_h.permute(0, 2, 1, 3).contiguous()
    dq_projection = _inverse_pair_rope(dq, cosine, sine)
    dk_projection = _inverse_pair_rope(dk, cosine, sine)

    q_width = q_heads * depth
    kv_width = kv_heads * depth
    wq = qkv_weight[:q_width].float()
    wk = qkv_weight[q_width : q_width + kv_width].float()
    wv = qkv_weight[q_width + kv_width :].float()
    rows_f = rows.float().reshape(batch * sequence, -1)
    normalizer = float(batch * sequence)
    dq_rows = dq_projection.reshape(batch * sequence, q_width)
    dk_rows = dk_projection.reshape(batch * sequence, kv_width)
    dv_rows = dv.reshape(batch * sequence, kv_width)
    grad_wq = torch.matmul(dq_rows.T, rows_f) / normalizer
    grad_wk = torch.matmul(dk_rows.T, rows_f) / normalizer
    grad_wv = torch.matmul(dv_rows.T, rows_f) / normalizer
    dx = (
        torch.matmul(dq_rows, wq)
        + torch.matmul(dk_rows, wk)
        + torch.matmul(dv_rows, wv)
    ).reshape(batch, sequence, -1)
    return {
        "dq": dq,
        "dk": dk,
        "dv": dv,
        "dx": dx,
        "grad_wq": grad_wq,
        "grad_wk": grad_wk,
        "grad_wv": grad_wv,
    }


def _metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, float]:
    result = tensor_error_metrics(reference, candidate)
    difference = candidate.float() - reference.float()
    reference_rms = reference.float().square().mean().sqrt().clamp_min(1.0e-30)
    result.update(
        {
            "signed_error_mean": float(difference.mean()),
            "signed_bias_over_reference_rms": float(
                difference.mean() / reference_rms
            ),
        }
    )
    return result


def _state_metrics(
    reference: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
) -> dict[str, dict[str, float]]:
    return {
        name: _metrics(reference[name], candidate[name])
        for name in reference
    }


def _mean_state(
    accumulated: dict[str, torch.Tensor],
    draws: int,
) -> dict[str, torch.Tensor]:
    return {name: value / float(draws) for name, value in accumulated.items()}


def _summarize_draw_metrics(
    draws: list[dict[str, dict[str, float]]],
) -> dict[str, dict[str, dict[str, float]]]:
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for tensor_name in draws[0]:
        summary[tensor_name] = {}
        for metric_name in draws[0][tensor_name]:
            values = [draw[tensor_name][metric_name] for draw in draws]
            summary[tensor_name][metric_name] = {
                "mean": statistics.mean(values),
                "standard_deviation": statistics.pstdev(values),
                "minimum": min(values),
                "maximum": max(values),
            }
    return summary


def _decoded(bundle: B300UnifiedLowpQKV, name: str) -> torch.Tensor:
    tensor = getattr(bundle, name)
    if tensor is None or not tensor.numel():
        raise RuntimeError(f"projection omitted {name}")
    return tensor.float().mul_(0.25)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=(1, 2, 4, 8, 16),
    )
    parser.add_argument("--sequence", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--sr-draws", type=int, default=16)
    parser.add_argument("--q-quant-scale", type=float, default=2.25)
    parser.add_argument("--k-quant-scale", type=float, default=2.0)
    parser.add_argument("--expected-projection-extension", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    batches = tuple(sorted(set(args.batches)))
    if not batches or batches[0] <= 0:
        parser.error("--batches must contain positive integers")
    if args.sequence <= 0 or args.sequence % 256:
        parser.error("--sequence must be positive and divisible by 256")
    if args.hidden <= 0 or args.hidden % 128:
        parser.error("--hidden must be positive and divisible by 128")
    if args.sr_draws <= 0:
        parser.error("--sr-draws must be positive")
    if (
        args.q_heads <= 0
        or args.kv_heads <= 0
        or args.q_heads % args.kv_heads
        or args.q_heads % 2
        or args.kv_heads % 2
    ):
        parser.error("D64 GQA requires positive even Hq/Hkv and Hq % Hkv == 0")
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output {args.output}")

    started = time.time()
    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda", args.gpu)
    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats(device)
    extension = _extension_identity(args.expected_projection_extension)

    maximum_batch = max(batches)
    rows_bank = torch.randn(
        maximum_batch,
        args.sequence,
        args.hidden,
        device=device,
        dtype=torch.float32,
    ).bfloat16()
    total_width = (args.q_heads + 2 * args.kv_heads) * 64
    weight = (
        torch.randn(
            total_width,
            args.hidden,
            device=device,
            dtype=torch.float32,
        )
        * 0.02
    ).bfloat16()
    weight_operand = tuple(b300_prepare_e4m3_projection_weight(weight))
    dout_bank = (
        torch.randn(
            maximum_batch,
            args.sequence,
            args.q_heads,
            64,
            device=device,
            dtype=torch.float32,
        )
        * 0.02
    )

    batch_results: dict[str, Any] = {}
    previous_publications: dict[str, torch.Tensor] | None = None
    previous_batch = 0
    for batch in batches:
        rows = rows_bank[:batch].contiguous()
        input_operand = tuple(
            b300_prepare_e4m3_projection_operand(
                rows.reshape(batch * args.sequence, args.hidden)
            )
        )
        qk_scales = torch.zeros(
            batch,
            args.q_heads // 2,
            7,
            device=device,
            dtype=torch.float32,
        )
        qk_scales[..., 0] = args.q_quant_scale
        qk_scales[..., 1] = args.k_quant_scale
        cosine, sine, rope = _make_rope(batch, args.sequence, device)
        common = {
            "input_operand": input_operand,
            "weight_operand": weight_operand,
            "qk_scales": qk_scales,
            "rope": rope,
            "batch": batch,
            "sequence": args.sequence,
            "q_heads": args.q_heads,
            "kv_heads": args.kv_heads,
        }
        represented = _project(**common, split_v_backward=False)
        direct = _project(**common, split_v_backward=True)
        torch.cuda.synchronize(device)
        represented_publications = _publications(represented)
        direct_publications = _publications(direct)
        unchanged = {
            field: _byte_comparison(
                _contract_view(field, represented_publications[field]),
                _contract_view(field, direct_publications[field]),
            )
            for field in (*FORWARD_FIELDS, *BACKWARD_QK_FIELDS)
        }
        if not all(bool(comparison["equal"]) for comparison in unchanged.values()):
            raise RuntimeError(
                f"split-V changed a supposedly invariant publication at B{batch}"
            )

        q = _decoded(direct, "q_backward_fp8")
        k = _decoded(direct, "k_backward_fp8")
        direct_v = _decoded(direct, "v_backward_fp8")
        represented_v = _decoded(represented, "v_backward_fp8")
        current_publications = {
            "q": direct.q_backward_fp8,
            "k": direct.k_backward_fp8,
            "direct_v": direct.v_backward_fp8,
            "represented_v": represented.v_backward_fp8,
        }
        prefix_checks: dict[str, dict[str, Any]] | None = None
        if previous_publications is not None:
            prefix_checks = {
                name: _byte_comparison(
                    previous,
                    current[:previous_batch],
                )
                for name, previous in previous_publications.items()
                for current in (current_publications[name],)
            }
            if not all(bool(check["equal"]) for check in prefix_checks.values()):
                raise RuntimeError(
                    f"production projection changed prefix values at B{batch}"
                )
        previous_publications = {
            name: tensor.detach().clone()
            for name, tensor in current_publications.items()
        }
        previous_batch = batch

        probability = _causal_probability(q, k)
        dout = dout_bank[:batch].contiguous()
        reference_state = _backward_state(
            probability,
            q,
            k,
            direct_v,
            dout,
            rows,
            weight,
            cosine,
            sine,
        )
        represented_state = _backward_state(
            probability,
            q,
            k,
            represented_v,
            dout,
            rows,
            weight,
            cosine,
            sine,
        )
        proxy_rne = fake_quantize_mxfp4_v_1d(direct_v, rounding="rne")
        proxy_rne_state = _backward_state(
            probability,
            q,
            k,
            proxy_rne.values,
            dout,
            rows,
            weight,
            cosine,
            sine,
        )

        sr_draw_metrics: list[dict[str, dict[str, float]]] = []
        sr_diagnostics: list[dict[str, Any]] = []
        accumulated: dict[str, torch.Tensor] | None = None
        for draw in range(args.sr_draws):
            sr_seed = args.seed + batch * 1_000_003 + draw * 97
            generator = torch.Generator(device=device).manual_seed(sr_seed)
            sr_v = fake_quantize_mxfp4_v_1d(
                direct_v,
                rounding="stochastic",
                generator=generator,
            )
            sr_state = _backward_state(
                probability,
                q,
                k,
                sr_v.values,
                dout,
                rows,
                weight,
                cosine,
                sine,
            )
            sr_draw_metrics.append(_state_metrics(reference_state, sr_state))
            sr_diagnostics.append(sr_v.diagnostics)
            if accumulated is None:
                accumulated = {
                    name: value.detach().clone()
                    for name, value in sr_state.items()
                }
            else:
                for name, value in sr_state.items():
                    accumulated[name].add_(value)
        assert accumulated is not None
        mean_sr_state = _mean_state(accumulated, args.sr_draws)

        direct_nonzero = direct_v != 0.0
        actual_zeroed = direct_nonzero & (represented_v == 0.0)
        batch_results[str(batch)] = {
            "production_runtime_batch_authenticated": (
                batch in AUTHENTICATED_PRODUCTION_BATCHES
            ),
            "nested_fixed_bank_prefix_checks": prefix_checks,
            "split_v_invariant_publications": unchanged,
            "v_representation": {
                "native_represented_mx_vs_direct_e4m3": _metrics(
                    direct_v,
                    represented_v,
                ),
                "e4m3_derived_proxy_rne_vs_direct_e4m3": _metrics(
                    direct_v,
                    proxy_rne.values,
                ),
                "e4m3_derived_proxy_rne_vs_native_represented_mx": _metrics(
                    represented_v,
                    proxy_rne.values,
                ),
                "native_represented_nonzero_input_zeroed_fraction": float(
                    actual_zeroed.sum() / direct_nonzero.sum().clamp_min(1)
                ),
                "proxy_rne_diagnostics": proxy_rne.diagnostics,
                "sr_diagnostics_mean": {
                    name: statistics.mean(
                        float(diagnostic[name]) for diagnostic in sr_diagnostics
                    )
                    for name in (
                        "payload_saturation_fraction",
                        "payload_clipping_fraction",
                        "payload_zero_fraction",
                        "nonzero_input_zeroed_fraction",
                        "signed_error_mean",
                        "error_rmse",
                    )
                },
            },
            "backward": {
                "native_represented_mx_rne": _state_metrics(
                    reference_state,
                    represented_state,
                ),
                "e4m3_derived_proxy_rne": _state_metrics(
                    reference_state,
                    proxy_rne_state,
                ),
                "e4m3_derived_proxy_sr_draw_distribution": (
                    _summarize_draw_metrics(sr_draw_metrics)
                ),
                "e4m3_derived_proxy_sr_mean_estimator": _state_metrics(
                    reference_state,
                    mean_sr_state,
                ),
            },
        }
        del (
            probability,
            reference_state,
            represented_state,
            proxy_rne_state,
            accumulated,
            mean_sr_state,
        )

    torch.cuda.synchronize(device)
    repo_root = Path(__file__).resolve().parents[2]
    result = {
        "schema": "mxfp4_v_backward_batch_sr_screen_v1",
        "status": "diagnostic_not_production_timing",
        "configuration": {
            "seed": args.seed,
            "batches": list(batches),
            "sequence": args.sequence,
            "hidden": args.hidden,
            "q_heads": args.q_heads,
            "kv_heads": args.kv_heads,
            "head_dim": 64,
            "sr_draws": args.sr_draws,
            "q_quant_scale": args.q_quant_scale,
            "k_quant_scale": args.k_quant_scale,
            "sample_bank": "nested_prefixes_of_one_fixed_max_batch_bank",
        },
        "interpretation_contract": {
            "reference_v": "projection_accumulator_e4m3_decoded_x0.25",
            "native_rne_v": "forward_mxfp4_codes_lifted_to_e4m3_decoded_x0.25",
            "sr_v": (
                "direct_e4m3_qdq_with_production_1x32_scale_selector_and_true_"
                "random_e2m1_rounding"
            ),
            "backward": "readable_exact_causal_gqa_jacobian_with_fixed_q_k_do",
            "ste": (
                "identity_for_projection_gradient_interpretation; quantizer_"
                "derivatives_are_not_taken"
            ),
            "native_packed_mx_backward": False,
            "timing_claim_allowed": False,
            "v_affects": ["dP", "dS", "dQ", "dK", "Q/K weight gradients"],
            "v_does_not_affect_with_fixed_p_do": ["dV", "V weight gradient"],
        },
        "provenance": {
            "git": _git_identity(repo_root),
            "projection_extension": extension,
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
        "batches": batch_results,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    _write_create_only(args.output, rendered)
    print(rendered)


if __name__ == "__main__":
    main()
