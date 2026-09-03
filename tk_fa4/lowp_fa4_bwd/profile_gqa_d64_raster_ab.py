#!/usr/bin/env python3
"""A/B the retained D64 causal GQA raster against head-fast launch order."""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from tk_fa4.lowp_fa4_bwd.profile_gqa_d128_chain import CompiledGqaBackward
from tk_fa4.lowp_fa4_bwd.tune_d64_gqa_cute import _load_control


def _summary(values: list[float]) -> dict[str, Any]:
    return {
        "median_us": statistics.median(values),
        "minimum_us": min(values),
        "samples_us": values,
    }


def _time_rotated(
    runners: dict[str, Callable[[], None]],
    *,
    prepares: dict[str, Callable[[], None]] | None = None,
    warmups: int,
    samples: int,
) -> dict[str, dict[str, Any]]:
    names = tuple(runners)
    for iteration in range(warmups):
        for offset in range(len(names)):
            name = names[(iteration + offset) % len(names)]
            if prepares is not None:
                prepares[name]()
            runners[name]()
    torch.cuda.synchronize()
    timings = {name: [] for name in names}
    for iteration in range(samples):
        for offset in range(len(names)):
            name = names[(iteration + offset) % len(names)]
            if prepares is not None:
                prepares[name]()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            runners[name]()
            end.record()
            end.synchronize()
            timings[name].append(float(start.elapsed_time(end) * 1000.0))
    return {name: _summary(values) for name, values in timings.items()}


def _metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    candidate_f = candidate.float()
    reference_f = reference.float()
    difference = candidate_f - reference_f
    return {
        "candidate_finite": bool(torch.isfinite(candidate_f).all()),
        "reference_finite": bool(torch.isfinite(reference_f).all()),
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                candidate_f.flatten(), reference_f.flatten(), dim=0
            )
        ),
        "relative_l2": float(
            difference.norm() / reference_f.norm().clamp_min(1.0e-30)
        ),
        "max_abs": float(difference.abs().max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=int, default=8192)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=7)
    parser.add_argument("--samples", type=int, default=41)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--exp2-degree", type=int, choices=(1, 2))
    parser.add_argument(
        "--exp2-period",
        type=int,
        choices=tuple(range(17)),
        help="explicit selective packed-ALU EX2 cadence",
    )
    parser.add_argument(
        "--detached-fp8-p-tmem",
        action="store_true",
        help=(
            "compare head-fast score-aliasing P against the optional "
            "detached FP8 P TMEM schedule"
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise RuntimeError("the A/B requires exactly one visible GPU")
    if args.sequence <= 0 or args.q_heads <= 0 or args.kv_heads <= 0:
        raise ValueError("sequence and head counts must be positive")
    if args.warmups < 0 or args.samples <= 0:
        raise ValueError("warmups must be nonnegative and samples must be positive")
    if (args.exp2_degree is None) != (args.exp2_period is None):
        raise ValueError("EX2 degree and period must be supplied together")
    if args.sequence % 128 or args.q_heads % args.kv_heads:
        raise ValueError(
            "sequence must be divisible by 128 and Hq must be divisible by Hkv"
        )

    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    batch = 1
    depth = 64

    def fp8(shape: tuple[int, ...]) -> torch.Tensor:
        return (
            torch.randn(shape, device="cuda", dtype=torch.float32)
            .mul_(0.25)
            .to(torch.float8_e4m3fn)
        )

    q_shape = (batch, args.sequence, args.q_heads, depth)
    kv_shape = (batch, args.sequence, args.kv_heads, depth)
    q = fp8(q_shape)
    k = fp8(kv_shape)
    v = fp8(kv_shape)
    dout = fp8(q_shape)
    stats = torch.randn(
        batch, args.q_heads, 1, args.sequence, device="cuda"
    ).mul_(0.125)
    lse = torch.randn_like(stats).mul_(0.125)
    control = _load_control(fp8_p_storage="tmem", direct_tma_dkdv=True)
    candidate_control = (
        _load_control(
            fp8_p_storage="tmem",
            direct_tma_dkdv=True,
            detached_fp8_p_tmem=True,
        )
        if args.detached_fp8_p_tmem
        else control
    )
    common = dict(
        q=q,
        k=k,
        v=v,
        o_or_sum=stats,
        dout=dout,
        lse_or_scaled_lse=lse,
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
        lowp=True,
        precomputed_stats=True,
        workspace_stats=True,
        scale_softmax=(depth**-0.5) / 16.0,
        reuse_quantized_p=False,
        fp8_ds_lift=16,
        lowp_do_stages=1,
        direct_tma_dkdv=True,
    )
    if args.exp2_degree is not None:
        common["exp2_degree"] = args.exp2_degree
        common["exp2_period"] = args.exp2_period
    baseline = CompiledGqaBackward(
        control,
        head_fast_raster=args.detached_fp8_p_tmem,
        **common,
    )
    head_fast = CompiledGqaBackward(
        candidate_control,
        head_fast_raster=True,
        **common,
    )
    baseline_name = (
        "head_fast_alias_p"
        if args.detached_fp8_p_tmem
        else "key_fast_alias_p"
    )
    candidate_name = (
        "head_fast_detached_p"
        if args.detached_fp8_p_tmem
        else "head_fast_alias_p"
    )

    baseline.run(reset=True)
    head_fast.run(reset=True)
    torch.cuda.synchronize()
    correctness = {
        name: _metrics(candidate, reference)
        for name, candidate, reference in zip(
            ("dq", "dk", "dv"),
            (head_fast.dq, head_fast.dk, head_fast.dv),
            (baseline.dq, baseline.dk, baseline.dv),
            strict=True,
        )
    }
    for gradient, metrics in correctness.items():
        if (
            not metrics["candidate_finite"]
            or not metrics["reference_finite"]
            or metrics["cosine"] < 0.999
            or metrics["relative_l2"] > 0.02
        ):
            raise RuntimeError(
                f"head-fast correctness gate failed for {gradient}: {metrics}"
            )

    with_clear = _time_rotated(
        {
            baseline_name: lambda: baseline.run(reset=True),
            candidate_name: lambda: head_fast.run(reset=True),
        },
        warmups=args.warmups,
        samples=args.samples,
    )
    kernel_after_clear = _time_rotated(
        {
            baseline_name: lambda: baseline.run(reset=False),
            candidate_name: lambda: head_fast.run(reset=False),
        },
        prepares={
            baseline_name: baseline.reset,
            candidate_name: head_fast.reset,
        },
        warmups=args.warmups,
        samples=args.samples,
    )
    baseline_us = with_clear[baseline_name]["median_us"]
    head_fast_us = with_clear[candidate_name]["median_us"]
    document = {
        "shape": {
            "sequence": args.sequence,
            "q_heads": args.q_heads,
            "kv_heads": args.kv_heads,
            "head_dim": depth,
        },
        "policy": {
            "exp2": baseline.exp2_policy,
            "fp8_ds_lift": 16,
            "lowp_do_stages": 1,
            "probability_storage": "tmem",
            "direct_tma_dkdv": True,
            "baseline_head_fast": args.detached_fp8_p_tmem,
            "candidate_head_fast": True,
            "candidate_detached_fp8_p_tmem": args.detached_fp8_p_tmem,
            "baseline_raster_policy": baseline.raster_policy,
            "candidate_raster_policy": head_fast.raster_policy,
        },
        "protocol": {
            "purpose": (
                "detached-P scheduling candidate-versus-head-fast control A/B"
                if args.detached_fp8_p_tmem
                else "head-fast scheduling candidate-versus-key-fast control A/B"
            ),
            "statistics": "synthetic identical pages for both rasters",
            "authoritative_accuracy": (
                "benchmark_causal_backward_matrix.py uses matched "
                "projection-native statistics and a BF16 reference"
            ),
        },
        "correctness": correctness,
        "timing": {
            "with_required_clear": with_clear,
            "kernel_after_required_clear": kernel_after_clear,
            "speedup": baseline_us / head_fast_us,
            "saved_us": baseline_us - head_fast_us,
        },
    }
    rendered = json.dumps(document, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
