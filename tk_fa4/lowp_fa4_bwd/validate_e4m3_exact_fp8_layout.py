#!/usr/bin/env python3
"""Validate the E4M3 projection layout used by exact FP8-PV forward.

The exact FP8-PV kernel consumes Q, K, and V in ordinary sequence order.
The MXFP4-PV kernel instead consumes the retained quarter-interleaved K/MX-V
layout.  This validator exercises both projection specializations with the
same inputs, checks their byte-level relationship, and compares exact FP8-PV
against attention evaluated from the exact represented operands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

import tk_fa4.interface as tk_interface
from tk_fa4 import (
    b300_pack_gqa_d64_paired_rope,
    b300_prepare_e4m3_projection_operand,
    b300_prepare_e4m3_projection_weight,
    b300_project_qkv_gqa_d64_paired_unified_lowp_e4m3,
)
from tk_fa4.lowp_fa4_bwd import benchmark_llama12b_e2e as benchmark
from tk_fa4.lowp_fa4_bwd.benchmark_llama12b_e2e import (
    Config,
    _load_forward,
    _make_rope,
)


SIGNED_E2M1_LEVELS = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, Any]:
    reference_f = reference.detach().float().reshape(-1)
    actual_f = actual.detach().float().reshape(-1)
    reference_norm = reference_f.norm().clamp_min(1.0e-30)
    actual_norm = actual_f.norm().clamp_min(1.0e-30)
    difference = actual_f - reference_f
    return {
        "cosine": float(
            torch.dot(reference_f, actual_f)
            / (reference_norm * actual_norm)
        ),
        "relative_l2": float(difference.norm() / reference_norm),
        "max_abs": float(difference.abs().max()),
        "finite": bool(torch.isfinite(actual_f).all()),
    }


def _byte_comparison(
    reference: torch.Tensor,
    actual: torch.Tensor,
) -> dict[str, int | float | bool]:
    reference_bytes = reference.contiguous().view(torch.uint8)
    actual_bytes = actual.contiguous().view(torch.uint8)
    if reference_bytes.shape != actual_bytes.shape:
        raise ValueError(
            f"byte comparison shape mismatch: {reference_bytes.shape} != "
            f"{actual_bytes.shape}"
        )
    mismatch = reference_bytes != actual_bytes
    mismatches = int(mismatch.sum())
    values = int(mismatch.numel())
    return {
        "equal": mismatches == 0,
        "mismatches": mismatches,
        "values": values,
        "mismatch_fraction": mismatches / values,
    }


def _physical_to_logical_index(
    sequence: int,
    device: torch.device,
) -> torch.Tensor:
    """Map each quarter-interleaved physical row to its logical row."""
    physical = torch.arange(sequence, device=device)
    block_base = (physical // 128) * 128
    local = physical & 127
    return block_base + (local // 32) + 4 * (local & 31)


def _deinterleave_k_bytes(payload: torch.Tensor) -> torch.Tensor:
    payload_bytes = payload.contiguous().view(torch.uint8)
    mapping = _physical_to_logical_index(
        payload_bytes.shape[2], payload_bytes.device
    )
    logical = torch.empty_like(payload_bytes)
    logical[:, :, mapping] = payload_bytes
    return logical


def _decode_e2m1(payload: torch.Tensor) -> torch.Tensor:
    packed = payload.contiguous().view(torch.uint8)
    levels = torch.tensor(
        SIGNED_E2M1_LEVELS,
        device=payload.device,
        dtype=torch.float32,
    )
    return torch.stack(
        (levels[(packed & 0x0F).long()], levels[(packed >> 4).long()]),
        dim=-1,
    ).flatten(-2)


def _decode_nvfp4_qk(
    payload: torch.Tensor,
    prepared_scale: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    scale_tile_rows: int,
) -> torch.Tensor:
    """Decode a logical-order native NVFP4 operand to contiguous [B,H,S,D]."""
    batch, heads, rows, packed_columns = payload.shape
    columns = packed_columns * 2
    if columns != 64 or rows % scale_tile_rows:
        raise ValueError("expected a packed D64 NVFP4 operand")
    row_tiles = rows // scale_tile_rows
    scales = (
        prepared_scale.float()
        .reshape(batch, row_tiles, heads, 32, 16)
        .permute(0, 2, 1, 3, 4)
        .contiguous()
    )
    decoded = _decode_e2m1(payload).reshape(
        batch, heads, rows, columns // 16, 16
    )
    row = torch.arange(rows, device=payload.device)
    block = torch.arange(columns // 16, device=payload.device)
    tile_index = (row // scale_tile_rows)[:, None]
    row_lane = (row % 32)[:, None]
    scale_slot = (
        ((row % scale_tile_rows) // 32)[:, None] * (columns // 16)
        + block[None, :]
    )
    for batch_index in range(batch):
        for head_index in range(heads):
            local_scale = scales[
                batch_index,
                head_index,
                tile_index,
                row_lane,
                scale_slot,
            ]
            decoded[batch_index, head_index].mul_(
                local_scale[..., None]
                * global_scale[batch_index, head_index]
            )
    return decoded.reshape(batch, heads, rows, columns)


def _default_module(path: Path) -> str:
    return path.name.split(".", 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--forward-extension",
        type=Path,
        required=True,
        help="exact causal GQA NVFP4-QK/E4M3-PV forward extension",
    )
    parser.add_argument("--forward-module")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--sequence", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--q-quant-scale", type=float, default=2.25)
    parser.add_argument("--k-quant-scale", type=float, default=2.0)
    parser.add_argument(
        "--per-block-qk-scales",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "exercise the represented-backward specialization with one "
            "E4M3 scale per logical Q/K row x K16"
        ),
    )
    parser.add_argument("--minimum-layout-cosine", type=float, default=0.98)
    parser.add_argument("--minimum-cosine", type=float, default=0.999)
    parser.add_argument("--maximum-relative-l2", type=float, default=0.03)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.sequence % 256:
        parser.error("--sequence must be divisible by 256")
    if args.hidden % 128:
        parser.error("--hidden must be divisible by 128")
    if args.q_heads % args.kv_heads or args.q_heads % 2 or args.kv_heads % 2:
        parser.error("paired D64 requires even Hq/Hkv and Hq divisible by Hkv")

    torch.cuda.set_device(args.gpu)
    torch.manual_seed(args.seed)
    config = Config(
        layers=1,
        sequence=args.sequence,
        hidden=args.hidden,
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
    )
    forward_path = args.forward_extension.resolve()
    forward_module = args.forward_module or _default_module(forward_path)
    forward, topology = _load_forward(
        forward_path, forward_module, config
    )
    if topology.get("pv_format") != "e4m3_fp8":
        raise RuntimeError("validator requires the exact E4M3 FP8-PV route")
    if bool(topology.get("causal_interleaved_kv", False)):
        raise RuntimeError("exact FP8 forward topology must use logical K/V")

    # RMSNorm feeds the real QKV projection with unit-RMS rows.  Retain that
    # scale here so QK quantization and the causal score path are exercised in
    # their production range rather than an artificially uniform-softmax one.
    rows = torch.randn(
        args.sequence,
        args.hidden,
        device="cuda",
        dtype=torch.float32,
    ).bfloat16()
    total_width = (args.q_heads + 2 * args.kv_heads) * 64
    weight = (
        torch.randn(
            total_width,
            args.hidden,
            device="cuda",
            dtype=torch.float32,
        )
        * 0.02
    ).bfloat16()
    qk_scales = torch.zeros(
        1,
        args.q_heads // 2,
        7,
        device="cuda",
        dtype=torch.float32,
    )
    qk_scales[..., 0] = args.q_quant_scale
    qk_scales[..., 1] = args.k_quant_scale
    paired_rope = b300_pack_gqa_d64_paired_rope(
        *_make_rope(args.sequence, 64)
    )
    input_operand = tuple(b300_prepare_e4m3_projection_operand(rows))
    weight_operand = tuple(b300_prepare_e4m3_projection_weight(weight))

    # Exercise route-selected defaults: exact emits logical K/V, MX retains
    # the quarter-interleaved K/MX-V contract.
    exact = b300_project_qkv_gqa_d64_paired_unified_lowp_e4m3(
        input_operand,
        weight_operand,
        qk_scales,
        paired_rope,
        batch=1,
        seqlen=args.sequence,
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
        publish_mxfp4_v=False,
        represented_backward=args.per_block_qk_scales,
        per_block_qk_scales=args.per_block_qk_scales,
    )
    mx = b300_project_qkv_gqa_d64_paired_unified_lowp_e4m3(
        input_operand,
        weight_operand,
        qk_scales,
        paired_rope,
        batch=1,
        seqlen=args.sequence,
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
        publish_mxfp4_v=True,
        represented_backward=args.per_block_qk_scales,
        per_block_qk_scales=args.per_block_qk_scales,
    )
    if exact.v_forward_fp8 is None:
        raise RuntimeError("exact FP8 projection did not publish feature-major V")
    if (
        exact.q_backward_fp8 is None
        or exact.k_backward_fp8 is None
        or exact.v_backward_fp8 is None
        or mx.v_backward_fp8 is None
    ):
        raise RuntimeError("projection did not publish backward E4M3 Q/K/V")

    exact_q = exact.backward.score_q_fp4
    exact_k = exact.backward.score_k_fp4
    mx_q = mx.backward.score_q_fp4
    mx_k = mx.backward.score_k_fp4
    layout = {
        "q_exact_vs_mx": _byte_comparison(exact_q, mx_q),
        "k_exact_vs_mx_direct": _byte_comparison(exact_k, mx_k),
        "k_exact_vs_deinterleaved_mx": _byte_comparison(
            exact_k, _deinterleave_k_bytes(mx_k)
        ),
        "v_exact_feature_vs_backward_transpose": _byte_comparison(
            exact.v_forward_fp8,
            exact.v_backward_fp8.permute(0, 2, 3, 1).contiguous(),
        ),
    }

    q_represented = _decode_nvfp4_qk(
        exact_q,
        exact.q_forward_scales,
        exact.q_forward_global_scale,
        scale_tile_rows=128,
    ).permute(0, 2, 1, 3).contiguous()
    k_represented = _decode_nvfp4_qk(
        exact_k,
        (
            exact.k_forward_scales[:, ::2].contiguous()
            if args.per_block_qk_scales
            else exact.k_forward_scales
        ),
        exact.k_forward_global_scale,
        scale_tile_rows=128 if args.per_block_qk_scales else 64,
    ).permute(0, 2, 1, 3).contiguous()
    v_represented = (
        exact.v_forward_fp8.permute(0, 3, 1, 2)
        .contiguous()
        .float()
        .mul(0.25)
    )
    q_projection_e4m3 = exact.q_backward_fp8.float().mul(0.25)
    k_projection_e4m3 = exact.k_backward_fp8.float().mul(0.25)
    v_projection_e4m3 = exact.v_backward_fp8.float().mul(0.25)
    represented_operand_metrics = {
        "q_nvfp4_vs_projection_e4m3": _metrics(
            q_projection_e4m3, q_represented
        ),
        "k_nvfp4_vs_projection_e4m3": _metrics(
            k_projection_e4m3, k_represented
        ),
    }
    if args.per_block_qk_scales:
        layout.update(
            {
                "k_scale_pages_duplicated": _byte_comparison(
                    exact.k_forward_scales[:, 0::2],
                    exact.k_forward_scales[:, 1::2],
                ),
                "q_represented_backward_bytes": _byte_comparison(
                    (q_represented * 4.0).to(torch.float8_e4m3fn),
                    exact.q_backward_fp8,
                ),
                "k_represented_backward_bytes": _byte_comparison(
                    (k_represented * 4.0).to(torch.float8_e4m3fn),
                    exact.k_backward_fp8,
                ),
            }
        )
    reference_result = benchmark.flash_attn_func(
        q_represented.bfloat16(),
        k_represented.bfloat16(),
        v_represented.bfloat16(),
        causal=True,
        return_lse=True,
    )
    if not isinstance(reference_result, tuple) or len(reference_result) != 2:
        raise RuntimeError("FlashAttention reference did not return output/LSE")
    reference_output, reference_lse = reference_result
    projection_reference_result = benchmark.flash_attn_func(
        q_projection_e4m3.bfloat16(),
        k_projection_e4m3.bfloat16(),
        v_projection_e4m3.bfloat16(),
        causal=True,
    )
    projection_reference_output = (
        projection_reference_result[0]
        if isinstance(projection_reference_result, tuple)
        else projection_reference_result
    )

    output = torch.empty_like(reference_output)
    lse = torch.empty(
        1,
        args.q_heads,
        1,
        args.sequence,
        device="cuda",
        dtype=torch.float32,
    )
    os.environ["TK_FA4_FP4PV_FWD_CONFIG"] = str(topology["route"])
    forward.forward_hao_direct_fp8pv(
        exact_q.view(torch.float4_e2m1fn_x2),
        exact.q_forward_scales,
        exact.q_forward_global_scale,
        exact_k.view(torch.float4_e2m1fn_x2),
        exact.k_forward_scales,
        exact.k_forward_global_scale,
        exact.v_forward_fp8,
        output,
        lse,
        0,
        True,
        True,
    )
    torch.cuda.synchronize()
    forward_metrics = {
        "output_vs_exact_represented_inputs": _metrics(
            reference_output, output
        ),
        "output_vs_projection_e4m3_reference": _metrics(
            projection_reference_output, output
        ),
        "lse_vs_exact_represented_inputs": _metrics(
            reference_lse.unsqueeze(2), lse
        ),
    }

    checks = {
        "q_layout_unchanged": bool(layout["q_exact_vs_mx"]["equal"]),
        "mx_k_is_distinct_physical_order": not bool(
            layout["k_exact_vs_mx_direct"]["equal"]
        ),
        "mx_k_deinterleaves_exactly_to_exact_k": bool(
            layout["k_exact_vs_deinterleaved_mx"]["equal"]
        ),
        "exact_v_views_are_byte_identical": bool(
            layout["v_exact_feature_vs_backward_transpose"]["equal"]
        ),
        "logical_k_operand_cosine": (
            represented_operand_metrics["k_nvfp4_vs_projection_e4m3"][
                "cosine"
            ]
            >= args.minimum_layout_cosine
        ),
        "logical_forward_output_cosine": (
            forward_metrics["output_vs_projection_e4m3_reference"]["cosine"]
            >= args.minimum_layout_cosine
        ),
        "represented_output_cosine": (
            forward_metrics["output_vs_exact_represented_inputs"]["cosine"]
            >= args.minimum_cosine
        ),
        "represented_output_relative_l2": (
            forward_metrics["output_vs_exact_represented_inputs"][
                "relative_l2"
            ]
            <= args.maximum_relative_l2
        ),
    }
    if args.per_block_qk_scales:
        checks.update(
            {
                "per_block_k_scale_pages_duplicated": bool(
                    layout["k_scale_pages_duplicated"]["equal"]
                ),
                "per_block_q_backward_matches_represented_codes": bool(
                    layout["q_represented_backward_bytes"]["equal"]
                ),
                "per_block_k_backward_matches_represented_codes": bool(
                    layout["k_represented_backward_bytes"]["equal"]
                ),
                "per_block_q_global_scale_is_one": bool(
                    torch.all(exact.q_forward_global_scale == 1.0)
                ),
                "per_block_k_global_scale_is_one": bool(
                    torch.all(exact.k_forward_global_scale == 1.0)
                ),
                "per_block_q_scales_are_nonuniform": (
                    exact.q_forward_scales.view(torch.uint8).unique().numel()
                    > 1
                ),
                "per_block_k_scales_are_nonuniform": (
                    exact.k_forward_scales.view(torch.uint8).unique().numel()
                    > 1
                ),
            }
        )
    result = {
        "schema": "e4m3_exact_fp8_layout_validator_v1",
        "configuration": {
            "seed": args.seed,
            "sequence": args.sequence,
            "hidden": args.hidden,
            "q_heads": args.q_heads,
            "kv_heads": args.kv_heads,
            "head_dim": 64,
            "q_quant_scale": args.q_quant_scale,
            "k_quant_scale": args.k_quant_scale,
            "per_block_qk_scales": args.per_block_qk_scales,
            "minimum_layout_cosine": args.minimum_layout_cosine,
            "minimum_cosine": args.minimum_cosine,
            "maximum_relative_l2": args.maximum_relative_l2,
            "forward_topology": topology,
        },
        "extensions": {
            "forward": {
                "path": str(forward_path),
                "module": forward_module,
                "sha256": _sha256(forward_path),
            },
            "projection": {
                "path": str(
                    Path(tk_interface._C_b300_lowp_bwd.__file__).resolve()
                ),
                "sha256": _sha256(
                    Path(tk_interface._C_b300_lowp_bwd.__file__).resolve()
                ),
            },
        },
        "layout": layout,
        "represented_operands": represented_operand_metrics,
        "forward": forward_metrics,
        "checks": checks,
        "passed": all(checks.values()),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
