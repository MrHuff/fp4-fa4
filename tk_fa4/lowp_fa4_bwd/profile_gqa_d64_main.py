#!/usr/bin/env python3
"""Profile the retained FP8 D64 causal GQA backward main path."""

from __future__ import annotations

import argparse
import json

import torch

from tk_fa4.lowp_fa4_bwd.profile_gqa_d128_chain import (
    CompiledGqaBackward,
    _time_cuda,
)
from tk_fa4.lowp_fa4_bwd.tune_d64_gqa_cute import _load_control


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=int, default=4096)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=8)
    parser.add_argument("--samples", type=int, default=31)
    parser.add_argument("--single", action="store_true")
    parser.add_argument(
        "--direct-tma-dkdv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reduce per-query-head dK/dV tiles directly into final KV heads",
    )
    parser.add_argument(
        "--head-fast-raster",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="put split query heads in physical grid-x and key tiles in grid-y",
    )
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

    q = fp8((batch, args.sequence, args.q_heads, depth))
    k = fp8((batch, args.sequence, args.kv_heads, depth))
    v = fp8((batch, args.sequence, args.kv_heads, depth))
    dout = fp8((batch, args.sequence, args.q_heads, depth))
    stats = torch.zeros(
        batch, args.q_heads, 1, args.sequence, device="cuda"
    )
    lse = torch.zeros_like(stats)
    control = _load_control(
        fp8_p_storage="tmem",
        direct_tma_dkdv=args.direct_tma_dkdv,
    )
    backward = CompiledGqaBackward(
        control,
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
        head_fast_raster=args.head_fast_raster,
        direct_tma_dkdv=args.direct_tma_dkdv,
    )
    backward.run(reset=True)
    torch.cuda.synchronize()

    if args.single:
        for _ in range(args.warmups):
            backward.run(reset=True)
        torch.cuda.synchronize()
        cudart = torch.cuda.cudart()
        cudart.cudaProfilerStart()
        backward.run(reset=True)
        torch.cuda.synchronize()
        cudart.cudaProfilerStop()
        print(json.dumps({"single_profile": True}))
        return

    with_clear = _time_cuda(
        lambda: backward.run(reset=True),
        warmups=args.warmups,
        samples=args.samples,
    )
    required_clear = _time_cuda(
        backward.reset,
        warmups=args.warmups,
        samples=args.samples,
    )
    print(
        json.dumps(
            {
                "shape": {
                    "sequence": args.sequence,
                    "q_heads": args.q_heads,
                    "kv_heads": args.kv_heads,
                    "head_dim": depth,
                },
                "direct_tma_dkdv": args.direct_tma_dkdv,
                "head_fast_raster": backward.head_fast_raster,
                "raster_policy": backward.raster_policy,
                "with_clear": with_clear,
                "required_clear": required_clear,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
