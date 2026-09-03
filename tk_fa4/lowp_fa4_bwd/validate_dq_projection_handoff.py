#!/usr/bin/env python3
"""Validate and time the head-pipelined dQ projection-backward handoff."""

from __future__ import annotations

import argparse
import json
import statistics
from typing import Callable

import torch

from tk_fa4 import _C_b300_lowp_bwd as lowp
from tk_fa4.interface import b300_mha_fwd


def metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float | bool]:
    reference_f = reference.float()
    actual_f = actual.float()
    difference = actual_f - reference_f
    reference_norm = torch.linalg.vector_norm(reference_f).clamp_min(1e-20)
    actual_norm = torch.linalg.vector_norm(actual_f).clamp_min(1e-20)
    return {
        "finite": bool(torch.isfinite(actual_f).all()),
        "cosine": float(
            torch.sum(reference_f * actual_f) / (reference_norm * actual_norm)
        ),
        "relative_l2": float(torch.linalg.vector_norm(difference) / reference_norm),
        "max_abs": float(difference.abs().max()),
    }


def time_rotated(
    callables: dict[str, Callable[[], object]],
    warmups: int,
    samples: int,
) -> dict[str, dict[str, float]]:
    names = list(callables)
    for iteration in range(warmups):
        order = names[iteration % len(names) :] + names[: iteration % len(names)]
        for name in order:
            callables[name]()
    torch.cuda.synchronize()
    values = {name: [] for name in names}
    for iteration in range(samples):
        order = names[iteration % len(names) :] + names[: iteration % len(names)]
        for name in order:
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
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--repeat-checks", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026081205)
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("validation requires exactly one visible GPU")

    torch.manual_seed(args.seed)
    batch = 1
    sequence = args.sequence
    heads = args.heads
    hidden = heads * 128
    q = (torch.randn(batch, sequence, heads, 192, device="cuda") * 0.1).bfloat16()
    k = (torch.randn_like(q.float()) * 0.1).bfloat16()
    v = (torch.randn(batch, sequence, heads, 128, device="cuda") * 0.1).bfloat16()
    dout = (torch.randn_like(v.float()) * 0.1).bfloat16()
    out, lse = b300_mha_fwd(q, k, v, causal=True, return_lse=True)
    softmax_scale = float(192**-0.5)
    packed = lowp.quantize_fp4_dual_qk_adaptive(
        q,
        k,
        16.0,
        2.0**-12,
        0.325,
        2.75,
        softmax_scale,
        4096.0,
    )
    weight = (
        torch.randn(heads * 192, hidden, device="cuda") * 0.02
    ).bfloat16()
    weight_t = weight.T.contiguous()

    def attention_bf16():
        return lowp.backward_fp4_fp8dpdv_x32_split_dk_adaptive_bf16dq_native(
            q,
            k,
            v,
            out,
            lse,
            dout,
            *packed,
            4096.0,
            True,
            softmax_scale,
            False,
        )

    def baseline_chain():
        dq, dk, dv = attention_bf16()
        dx = torch.mm(dq.reshape(sequence, heads * 192), weight)
        return dx, dk, dv

    def direct_chain():
        return lowp.backward_fp4_fp8dpdv_x32_split_dk_adaptive_direct_dq_projection_native(
            q,
            k,
            v,
            out,
            lse,
            dout,
            *packed,
            weight_t,
            4096.0,
            True,
            softmax_scale,
            False,
        )

    dq, reference_dk, reference_dv = attention_bf16()
    reference_dx = torch.mm(dq.reshape(sequence, heads * 192), weight)
    direct_dx, direct_dk, direct_dv = direct_chain()
    baseline_repeats = [baseline_chain()[0] for _ in range(args.repeat_checks)]
    direct_repeats = [direct_chain()[0] for _ in range(args.repeat_checks)]
    torch.cuda.synchronize()
    baseline_repeat_metrics = [
        metrics(reference_dx, repeat.reshape(sequence, hidden))
        for repeat in baseline_repeats
    ]
    direct_repeat_metrics = [
        metrics(direct_dx.reshape(sequence, hidden), repeat.reshape(sequence, hidden))
        for repeat in direct_repeats
    ]
    result = {
        "shape": {
            "sequence": sequence,
            "heads": heads,
            "hidden": hidden,
        },
        "quality": {
            "dx": metrics(reference_dx, direct_dx.reshape(sequence, hidden)),
            "dk": metrics(reference_dk, direct_dk),
            "dv": metrics(reference_dv, direct_dv),
            "baseline_repeatability": baseline_repeat_metrics,
            "direct_repeatability": direct_repeat_metrics,
        },
        "timings": time_rotated(
            {
                "attention_bf16_only": attention_bf16,
                "attention_then_cublas_dq_projection": baseline_chain,
                "head_pipelined_dq_projection": direct_chain,
            },
            args.warmups,
            args.samples,
        ),
    }
    baseline = result["timings"]["attention_then_cublas_dq_projection"][
        "median_ms"
    ]
    direct = result["timings"]["head_pipelined_dq_projection"]["median_ms"]
    result["direct_speedup"] = baseline / direct
    result["direct_saved_ms"] = baseline - direct
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
