#!/usr/bin/env python3
"""Validate and time the producer-native pure-MXFP4 backward contract."""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable

import torch

from tk_fa4 import _C_b300_lowp_bwd as lowp
from tk_fa4.interface import b300_mha_fwd


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
    callables: dict[str, Callable[[], object]], warmups: int, samples: int
) -> dict[str, dict[str, float]]:
    names = list(callables)
    for iteration in range(warmups):
        for offset in range(len(names)):
            callables[names[(iteration + offset) % len(names)]]()
    torch.cuda.synchronize()
    values = {name: [] for name in names}
    for iteration in range(samples):
        for offset in range(len(names)):
            name = names[(iteration + offset) % len(names)]
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            callables[name]()
            end.record()
            end.synchronize()
            values[name].append(float(start.elapsed_time(end)))
    return {
        name: {
            "median_ms": statistics.median(samples_),
            "mean_ms": statistics.fmean(samples_),
            "minimum_ms": min(samples_),
        }
        for name, samples_ in values.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=int, default=8192)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=7)
    parser.add_argument("--samples", type=int, default=31)
    parser.add_argument("--seed", type=int, default=2026081301)
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("validation requires exactly one visible GPU")

    torch.manual_seed(args.seed)
    q = (torch.randn(1, args.sequence, args.heads, 192, device="cuda") * 0.1).bfloat16()
    k = (torch.randn_like(q.float()) * 0.1).bfloat16()
    v = (
        torch.randn(1, args.sequence, args.heads, 128, device="cuda") * 0.1
    ).bfloat16()
    dout = (torch.randn_like(v.float()) * 0.1).bfloat16()
    out, lse = b300_mha_fwd(q, k, v, causal=True, return_lse=True)
    scale = float(192**-0.5)
    packed_qk = tuple(lowp.quantize_fp4_dual_qk_blockscale(q, k, 16.0, 16.0))
    producer_operands = tuple(lowp.prepare_mxfp4_backward_operands(out, dout, v, lse))

    def inline() -> tuple[torch.Tensor, ...]:
        return tuple(
            lowp.backward_fp4_mxfp4dpdvdsdqdk_forward_log_split_q_x32_native(
                q,
                k,
                v,
                out,
                lse,
                dout,
                *packed_qk,
                16.0,
                16.0,
                4096.0,
                True,
                scale,
                False,
            )
        )

    def producer_native() -> tuple[torch.Tensor, ...]:
        return tuple(
            lowp.backward_fp4_mxfp4dpdvdsdqdk_producer_native_x32(
                q,
                k,
                v,
                out,
                lse,
                dout,
                *packed_qk,
                *producer_operands,
                16.0,
                16.0,
                4096.0,
                True,
                scale,
                False,
            )
        )

    def producer_then_native() -> tuple[torch.Tensor, ...]:
        operands = tuple(lowp.prepare_mxfp4_backward_operands(out, dout, v, lse))
        return tuple(
            lowp.backward_fp4_mxfp4dpdvdsdqdk_producer_native_x32(
                q,
                k,
                v,
                out,
                lse,
                dout,
                *packed_qk,
                *operands,
                16.0,
                16.0,
                4096.0,
                True,
                scale,
                False,
            )
        )

    inline_output = inline()
    producer_output = producer_native()
    torch.cuda.synchronize()
    quality = {
        name: metrics(reference, actual)
        for name, reference, actual in zip(
            ("dq", "dk", "dv"), inline_output, producer_output, strict=True
        )
    }
    timing = time_rotated(
        {
            "inline_quantization": inline,
            "producer_native": producer_native,
            "producer_then_native": producer_then_native,
        },
        args.warmups,
        args.samples,
    )
    inline_ms = timing["inline_quantization"]["median_ms"]
    native_ms = timing["producer_native"]["median_ms"]
    result = {
        "shape": {"sequence": args.sequence, "heads": args.heads},
        "quality_vs_inline": quality,
        "timing": timing,
        "producer_native_saved_ms": inline_ms - native_ms,
        "producer_native_speedup": inline_ms / native_ms,
        "passed": bool(
            all(item["finite"] and item["cosine"] > 0.99999 for item in quality.values())
            and native_ms < inline_ms
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
