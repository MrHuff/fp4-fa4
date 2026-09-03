#!/usr/bin/env python3
"""Benchmark native HAO NVFP4-QK/FP8-PV and BF16 on one shape."""

from __future__ import annotations

import argparse
import json
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--seqlen", type=int, required=True)
    parser.add_argument("--heads", type=int, required=True)
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--warmup-ms", type=int, default=10)
    parser.add_argument("--rep-ms", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def comparison(a: Any, b: Any) -> dict[str, float]:
    import torch

    delta = a.float() - b.float()
    return {
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                a.float().flatten().unsqueeze(0),
                b.float().flatten().unsqueeze(0),
            ).item()
        ),
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().mean().item()),
        "rmse": float(delta.square().mean().sqrt().item()),
    }


def main() -> None:
    args = parse_args()

    import cutlass
    import torch
    import triton.testing
    from flash_attn.cute import interface
    from flash_attn.cute.benchmarks import bench_fp4

    if args.batch < 1 or args.heads < 1:
        raise ValueError("batch and heads must be positive")
    if args.seqlen % 128 or args.dim not in (64, 128):
        raise ValueError("seqlen must be divisible by 128 and dim must be 64 or 128")

    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    (
        q_fp4,
        k_fp4,
        v_fp8,
        q_scale,
        k_scale,
        v_scale,
        q_ref,
        k_ref,
        v_ref,
    ) = bench_fp4.create_nvfp4_attention_tensors(
        args.batch,
        args.seqlen,
        args.seqlen,
        args.heads,
        args.heads,
        args.dim,
        args.dim,
        device="cuda",
        dtype_gen=torch.bfloat16,
        pv_mode="fp8",
        pv_fp8_dtype=cutlass.Float8E4M3FN,
    )
    if v_scale is not None:
        raise RuntimeError("plain FP8 V unexpectedly produced scale factors")
    q_bf16 = q_ref.to(torch.bfloat16)
    k_bf16 = k_ref.to(torch.bfloat16)
    v_bf16 = v_ref.to(torch.bfloat16)

    def run_fp8() -> Any:
        return interface.flash_attn_func(
            q_fp4,
            k_fp4,
            v_fp8,
            causal=False,
            mSFQ=q_scale,
            mSFK=k_scale,
            mSFV=None,
        )

    def run_bf16() -> Any:
        return interface.flash_attn_func(
            q_bf16,
            k_bf16,
            v_bf16,
            causal=False,
        )

    direct = {
        "causal": False,
        "return_lse": True,
        "num_splits": 1,
        "pack_gqa": False,
        "_compute_capability": 10,
    }
    fp8_output, fp8_lse = interface._flash_attn_fwd(
        q_fp4,
        k_fp4,
        v_fp8,
        mSFQ=q_scale,
        mSFK=k_scale,
        mSFV=None,
        **direct,
    )
    bf16_output, bf16_lse = interface._flash_attn_fwd(
        q_bf16,
        k_bf16,
        v_bf16,
        **direct,
    )
    torch.cuda.synchronize()
    timing = {
        "hao_native_nvfp4_fp8pv": float(
            triton.testing.do_bench(
                run_fp8,
                warmup=args.warmup_ms,
                rep=args.rep_ms,
                return_mode="median",
            )
        ),
        "hao_native_bf16": float(
            triton.testing.do_bench(
                run_bf16,
                warmup=args.warmup_ms,
                rep=args.rep_ms,
                return_mode="median",
            )
        ),
    }
    flops = (
        args.batch
        * args.heads
        * 2
        * args.seqlen
        * args.seqlen
        * (args.dim + args.dim)
    )
    result = {
        "shape": {
            "batch": args.batch,
            "seqlen": args.seqlen,
            "heads": args.heads,
            "dim": args.dim,
        },
        "protocol": {
            "factory": "HAO create_nvfp4_attention_tensors",
            "timer": "triton.testing.do_bench median",
            "warmup_ms": args.warmup_ms,
            "rep_ms": args.rep_ms,
            "seed": args.seed,
        },
        "timing_ms": timing,
        "tflops": {
            name: flops / (milliseconds * 1e-3) / 1e12
            for name, milliseconds in timing.items()
        },
        "correctness": {
            "hao_fp8_vs_bf16_output": comparison(fp8_output, bf16_output),
            "hao_fp8_vs_bf16_lse": comparison(fp8_lse, bf16_lse),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
