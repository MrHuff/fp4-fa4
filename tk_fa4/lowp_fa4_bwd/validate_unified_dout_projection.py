#!/usr/bin/env python3
"""Validate and time the projection-native MXFP4 dO epilogue."""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable

import torch

from tk_fa4 import (
    _C_b300_lowp_bwd as lowp,
    b300_prepare_nvfp4_projection_operand,
    b300_project_dout_unified_lowp_nvfp4,
)


def byte_equal(left: torch.Tensor, right: torch.Tensor) -> dict[str, int | bool]:
    mismatch = left.view(torch.uint8) != right.view(torch.uint8)
    return {
        "equal": bool(not mismatch.any()),
        "mismatches": int(mismatch.sum()),
        "bytes": int(mismatch.numel()),
    }


def metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float | bool]:
    reference_f = reference.float()
    actual_f = actual.float()
    difference = actual_f - reference_f
    reference_norm = torch.linalg.vector_norm(reference_f).clamp_min(1.0e-20)
    actual_norm = torch.linalg.vector_norm(actual_f).clamp_min(1.0e-20)
    return {
        "finite": bool(torch.isfinite(actual_f).all()),
        "cosine": float(
            torch.sum(reference_f * actual_f) / (reference_norm * actual_norm)
        ),
        "relative_l2": float(torch.linalg.vector_norm(difference) / reference_norm),
        "max_abs": float(difference.abs().max()),
    }


def time_rotated(
    candidates: dict[str, Callable[[], object]], warmups: int, samples: int
) -> dict[str, dict[str, float]]:
    names = list(candidates)
    for iteration in range(warmups):
        for offset in range(len(names)):
            candidates[names[(iteration + offset) % len(names)]]()
    torch.cuda.synchronize()
    elapsed = {name: [] for name in names}
    for iteration in range(samples):
        for offset in range(len(names)):
            name = names[(iteration + offset) % len(names)]
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            candidates[name]()
            end.record()
            end.synchronize()
            elapsed[name].append(float(start.elapsed_time(end)))
    return {
        name: {
            "median_ms": statistics.median(values),
            "mean_ms": statistics.fmean(values),
            "minimum_ms": min(values),
        }
        for name, values in elapsed.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--seed", type=int, default=2026081302)
    args = parser.parse_args()

    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    batch = 1
    rows = batch * args.sequence
    width = args.heads * 128
    grad = (torch.randn(rows, args.hidden, device="cuda") * 0.1).bfloat16()
    weight = (torch.randn(width, args.hidden, device="cuda") * 0.02).bfloat16()
    attention_output = (
        torch.randn(batch, args.sequence, args.heads, 128, device="cuda") * 0.1
    ).bfloat16()
    lse = torch.randn(
        batch, args.sequence, args.heads, device="cuda", dtype=torch.float32
    )
    packed_grad = tuple(b300_prepare_nvfp4_projection_operand(grad))
    packed_weight = tuple(b300_prepare_nvfp4_projection_operand(weight))

    def bf16_projection() -> torch.Tensor:
        return torch.mm(grad, weight.T).reshape(
            batch, args.sequence, args.heads, 128
        )

    def split_projection_and_pack() -> object:
        dout = bf16_projection()
        return lowp.prepare_mxfp4_backward_operands(
            attention_output, dout, torch.zeros_like(dout), lse
        )

    def unified_store() -> object:
        return b300_project_dout_unified_lowp_nvfp4(
            packed_grad,
            packed_weight,
            attention_output,
            lse,
            batch=batch,
            seqlen=args.sequence,
            heads=args.heads,
            store_bf16=True,
        )

    def unified_no_store() -> object:
        return b300_project_dout_unified_lowp_nvfp4(
            packed_grad,
            packed_weight,
            attention_output,
            lse,
            batch=batch,
            seqlen=args.sequence,
            heads=args.heads,
            store_bf16=False,
        )

    fused = unified_store()
    assert fused.dout is not None
    standalone = lowp.prepare_mxfp4_backward_operands(
        attention_output, fused.dout, torch.zeros_like(fused.dout), lse
    )
    no_store = unified_no_store()
    checks = {
        "dout_dp_payload": byte_equal(fused.dout_dp_fp4, standalone[0]),
        "dout_dp_scales": byte_equal(fused.dout_dp_scales, standalone[2]),
        "dout_dv_payload": byte_equal(fused.dout_dv_fp4, standalone[4]),
        "dout_dv_scales": byte_equal(fused.dout_dv_scales, standalone[5]),
        "no_store_dout_dp_payload": byte_equal(
            fused.dout_dp_fp4, no_store.dout_dp_fp4
        ),
        "no_store_dout_dp_scales": byte_equal(
            fused.dout_dp_scales, no_store.dout_dp_scales
        ),
        "no_store_dout_dv_payload": byte_equal(
            fused.dout_dv_fp4, no_store.dout_dv_fp4
        ),
        "no_store_dout_dv_scales": byte_equal(
            fused.dout_dv_scales, no_store.dout_dv_scales
        ),
    }
    quality = {
        "projection": metrics(bf16_projection(), fused.dout),
        "dpsum": metrics(standalone[6], fused.dpsum),
        "lse_log2": metrics(standalone[7], fused.lse_log2),
        "no_store_dpsum": metrics(fused.dpsum, no_store.dpsum),
        "no_store_lse_log2": metrics(fused.lse_log2, no_store.lse_log2),
    }
    timing = time_rotated(
        {
            "bf16_projection": bf16_projection,
            "bf16_projection_then_pack": split_projection_and_pack,
            "unified_nvfp4_store_bf16": unified_store,
            "unified_nvfp4_no_bf16_store": unified_no_store,
        },
        args.warmups,
        args.samples,
    )
    result = {
        "shape": {
            "sequence": args.sequence,
            "heads": args.heads,
            "hidden": args.hidden,
        },
        "payload_checks": checks,
        "quality": quality,
        "timing": timing,
        "passed": bool(
            all(check["equal"] for check in checks.values())
            and quality["projection"]["cosine"] > 0.985
            and quality["dpsum"]["relative_l2"] < 1.0e-6
            and quality["lse_log2"]["relative_l2"] < 1.0e-6
            and quality["no_store_dpsum"]["relative_l2"] < 1.0e-6
            and quality["no_store_lse_log2"]["relative_l2"] < 1.0e-6
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
