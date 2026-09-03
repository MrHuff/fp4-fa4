#!/usr/bin/env python3
"""A/B the D64 GQA partial-reduction launch against direct TMA-add."""

from __future__ import annotations

import argparse
import json
import statistics

import torch

from tk_fa4.lowp_fa4_bwd.profile_gqa_d128_chain import CompiledGqaBackward
from tk_fa4.lowp_fa4_bwd.tune_d64_gqa_cute import _load_control


def _metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    candidate_f = candidate.float()
    reference_f = reference.float()
    return {
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                candidate_f.flatten(), reference_f.flatten(), dim=0
            )
        ),
        "max_abs": float((candidate_f - reference_f).abs().max()),
        "mean_abs": float((candidate_f - reference_f).abs().mean()),
    }


def _median(values: list[float]) -> dict[str, object]:
    return {
        "median_us": statistics.median(values),
        "minimum_us": min(values),
        "samples_us": values,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=int, default=4096)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=6)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise RuntimeError("profiling requires exactly one visible GPU")
    if args.sequence % 128 or args.q_heads % args.kv_heads:
        raise ValueError("sequence must be divisible by 128 and Hq by Hkv")

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

    shape_q = (batch, args.sequence, args.q_heads, depth)
    shape_kv = (batch, args.sequence, args.kv_heads, depth)
    q, k, v, dout = fp8(shape_q), fp8(shape_kv), fp8(shape_kv), fp8(shape_q)
    stats = torch.randn(
        batch, args.q_heads, 1, args.sequence, device="cuda"
    ).mul_(0.125)
    lse = torch.randn_like(stats).mul_(0.125)

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
    )
    baseline = CompiledGqaBackward(
        _load_control(fp8_p_storage="tmem"),
        **common,
    )
    direct = CompiledGqaBackward(
        _load_control(fp8_p_storage="tmem", direct_tma_dkdv=True),
        direct_tma_dkdv=True,
        **common,
    )

    baseline.run(reset=True)
    direct.run(reset=True)
    torch.cuda.synchronize()
    correctness = {
        "dq": _metrics(direct.dq, baseline.dq),
        "dk": _metrics(direct.dk, baseline.dk),
        "dv": _metrics(direct.dv, baseline.dv),
        "finite": bool(
            torch.isfinite(direct.dq).all()
            and torch.isfinite(direct.dk).all()
            and torch.isfinite(direct.dv).all()
        ),
    }

    for _ in range(args.warmups):
        baseline.run(reset=True)
        direct.run(reset=True)
    torch.cuda.synchronize()

    timings: dict[str, list[float]] = {"baseline": [], "direct_tma": []}
    runners = {"baseline": baseline, "direct_tma": direct}
    for sample in range(args.samples):
        order = ("baseline", "direct_tma")
        if sample % 2:
            order = tuple(reversed(order))
        for name in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            runners[name].run(reset=True)
            end.record()
            end.synchronize()
            timings[name].append(float(start.elapsed_time(end) * 1000.0))

    baseline_us = statistics.median(timings["baseline"])
    direct_us = statistics.median(timings["direct_tma"])
    print(
        json.dumps(
            {
                "shape": {
                    "sequence": args.sequence,
                    "q_heads": args.q_heads,
                    "kv_heads": args.kv_heads,
                    "head_dim": depth,
                },
                "correctness": correctness,
                "timing": {
                    "baseline": _median(timings["baseline"]),
                    "direct_tma": _median(timings["direct_tma"]),
                    "speedup": baseline_us / direct_us,
                    "saved_us": baseline_us - direct_us,
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
