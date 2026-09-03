#!/usr/bin/env python3
"""Validate and time the upstream MXFP4 dO publication contract."""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable

import torch

from tk_fa4.fp4_pv_experiments import (
    _load_backward_experiments_ext,
    _load_mxfp4_v3_quant,
)


def _allocate(batch: int, sequence: int, heads: int) -> tuple[torch.Tensor, ...]:
    device = torch.device("cuda")
    return (
        torch.empty(
            (batch, heads, sequence, 64),
            device=device,
            dtype=torch.float4_e2m1fn_x2,
        ),
        torch.empty(
            (batch, heads, sequence // 128, 1, 32, 16),
            device=device,
            dtype=torch.uint8,
        ),
        torch.empty(
            (batch, heads, 128, sequence // 2),
            device=device,
            dtype=torch.float4_e2m1fn_x2,
        ),
        torch.empty(
            (batch, heads, 1, sequence // 128, 32, 16),
            device=device,
            dtype=torch.uint8,
        ),
        torch.empty(
            (batch, heads, sequence),
            device=device,
            dtype=torch.float32,
        ),
    )


def _time_rotated(
    candidates: dict[str, Callable[[], object]],
    *,
    warmups: int,
    samples: int,
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
    parser.add_argument("--sequence", type=int, default=8192)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=31)
    parser.add_argument("--seed", type=int, default=2026081207)
    args = parser.parse_args()
    if args.sequence % 128:
        raise ValueError("sequence must be divisible by 128")

    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    batch = 1
    dout = (
        torch.randn(
            batch,
            args.sequence,
            args.heads,
            128,
            device="cuda",
        )
        * 0.1
    ).bfloat16()
    output = (torch.randn_like(dout.float()) * 0.1).bfloat16()
    quantizer = _load_mxfp4_v3_quant()
    backward = _load_backward_experiments_ext()
    if not hasattr(quantizer, "mxfp4_quantize_bshd128_row_out"):
        raise RuntimeError("the quantizer is missing the row-only publication entry point")
    if not hasattr(backward, "mxfp4_bshd128_delta_out"):
        raise RuntimeError("the backward extension is missing the standalone delta entry point")

    split_outputs = _allocate(batch, args.sequence, args.heads)
    fused_outputs = _allocate(batch, args.sequence, args.heads)

    def split() -> tuple[torch.Tensor, ...]:
        row, row_scales, column, column_scales, delta = split_outputs
        quantizer.mxfp4_quantize_bshd128_row_out(
            dout, row, row_scales, 1
        )
        quantizer.mxfp4_quantize_bshd128_col_out(
            dout, column, column_scales, 1
        )
        backward.mxfp4_bshd128_delta_out(dout, output, delta)
        return split_outputs

    def fused() -> tuple[torch.Tensor, ...]:
        row, row_scales, column, column_scales, delta = fused_outputs
        return quantizer.mxfp4_quantize_bshd128_row_and_col_with_delta_out(
            dout,
            output,
            row,
            row_scales,
            column,
            column_scales,
            delta,
            1,
        )

    split()
    fused()
    torch.cuda.synchronize()
    names = ("row_payload", "row_scales", "column_payload", "column_scales")
    exact = {}
    for name, split_tensor, fused_tensor in zip(
        names, split_outputs[:4], fused_outputs[:4], strict=True
    ):
        if split_tensor.dtype == torch.float4_e2m1fn_x2:
            split_tensor = split_tensor.view(torch.uint8)
            fused_tensor = fused_tensor.view(torch.uint8)
        mismatch = split_tensor != fused_tensor
        exact[name] = {
            "equal": bool(not mismatch.any()),
            "mismatches": int(mismatch.sum()),
            "elements": int(mismatch.numel()),
        }

    reference_delta = (
        (output.float() * dout.float()).sum(dim=-1).permute(0, 2, 1).contiguous()
    )
    fused_delta = fused_outputs[4]
    delta_difference = fused_delta - reference_delta
    timing = _time_rotated(
        {
            "split_row_column_delta": split,
            "fused_row_column_delta": fused,
        },
        warmups=args.warmups,
        samples=args.samples,
    )
    split_median = timing["split_row_column_delta"]["median_ms"]
    fused_median = timing["fused_row_column_delta"]["median_ms"]
    result = {
        "shape": {
            "batch": batch,
            "sequence": args.sequence,
            "heads": args.heads,
        },
        "payload_checks": exact,
        "delta": {
            "finite": bool(torch.isfinite(fused_delta).all()),
            "max_abs": float(delta_difference.abs().max()),
            "relative_l2": float(
                torch.linalg.vector_norm(delta_difference)
                / torch.linalg.vector_norm(reference_delta).clamp_min(1.0e-20)
            ),
        },
        "timing": timing,
        "fused_over_split": fused_median / split_median,
        "saved_ms": split_median - fused_median,
    }
    result["passed"] = bool(
        all(check["equal"] for check in exact.values())
        and result["delta"]["finite"]
        and result["delta"]["relative_l2"] < 1.0e-6
        and fused_median < split_median
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
