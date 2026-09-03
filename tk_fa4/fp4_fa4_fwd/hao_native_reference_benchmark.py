#!/usr/bin/env python3
"""Benchmark native HAO NVFP4-QK/FP8-PV and BF16 on one shape."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
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
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def comparison(a: Any, b: Any) -> dict[str, float]:
    import torch

    delta = a.float() - b.float()
    rmse = delta.square().mean().sqrt()
    reference_rms = b.float().square().mean().sqrt()
    return {
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                a.float().flatten().unsqueeze(0),
                b.float().flatten().unsqueeze(0),
            ).item()
        ),
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().mean().item()),
        "rmse": float(rmse.item()),
        "reference_rms": float(reference_rms.item()),
        "relative_l2": float((rmse / reference_rms).item()),
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
    if args.rounds < 1:
        raise ValueError("rounds must be positive")
    if args.seqlen % 128 or args.dim not in (64, 128):
        raise ValueError("seqlen must be divisible by 128 and dim must be 64 or 128")

    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    low_precision_supported = args.dim == 128
    if low_precision_supported:
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

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        (
            q_nvfp4,
            k_nvfp4,
            v_nvfp4,
            q_nvfp4_scale,
            k_nvfp4_scale,
            v_nvfp4_scale,
            q_nvfp4_ref,
            k_nvfp4_ref,
            v_nvfp4_ref,
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
            pv_mode="fp4",
        )
        if v_nvfp4_scale is None:
            raise RuntimeError("NVFP4 V unexpectedly lacks scale factors")
        if not all(
            torch.equal(a, b)
            for a, b in (
                (q_ref, q_nvfp4_ref),
                (k_ref, k_nvfp4_ref),
                (v_ref, v_nvfp4_ref),
            )
        ):
            raise RuntimeError(
                "FP8 and NVFP4 factories did not reproduce one input"
            )
    else:
        # HAO's low-precision factory assumes D is a multiple of 128. The
        # BF16 kernel supports D64, so construct its reference distribution
        # directly and omit unsupported HAO low-precision providers.
        q_ref = torch.randn(
            args.batch,
            args.seqlen,
            args.heads,
            args.dim,
            device="cuda",
            dtype=torch.float32,
        )
        k_ref = torch.randn_like(q_ref)
        v_ref = torch.randn_like(q_ref)

    q_bf16 = q_ref.to(torch.bfloat16)
    k_bf16 = k_ref.to(torch.bfloat16)
    v_bf16 = v_ref.to(torch.bfloat16)
    if low_precision_supported:
        fp8_timed_output = torch.empty_like(q_bf16)
        nvfp4_timed_output = torch.empty_like(q_bf16)
    bf16_timed_output = torch.empty_like(q_bf16)
    timed_direct = {
        "causal": False,
        "return_lse": False,
        "num_splits": 1,
        "pack_gqa": False,
        "_compute_capability": 10,
    }

    def run_fp8() -> Any:
        return interface._flash_attn_fwd(
            q_fp4,
            k_fp4,
            v_fp8,
            mSFQ=q_scale,
            mSFK=k_scale,
            mSFV=None,
            out=fp8_timed_output,
            **timed_direct,
        )

    def run_bf16() -> Any:
        return interface._flash_attn_fwd(
            q_bf16,
            k_bf16,
            v_bf16,
            out=bf16_timed_output,
            **timed_direct,
        )

    def run_nvfp4() -> Any:
        return interface._flash_attn_fwd(
            q_nvfp4,
            k_nvfp4,
            v_nvfp4,
            mSFQ=q_nvfp4_scale,
            mSFK=k_nvfp4_scale,
            mSFV=v_nvfp4_scale,
            out=nvfp4_timed_output,
            **timed_direct,
        )

    direct = {
        "causal": False,
        "return_lse": True,
        "num_splits": 1,
        "pack_gqa": False,
        "_compute_capability": 10,
    }
    if low_precision_supported:
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
    if low_precision_supported:
        nvfp4_output, nvfp4_lse = interface._flash_attn_fwd(
            q_nvfp4,
            k_nvfp4,
            v_nvfp4,
            mSFQ=q_nvfp4_scale,
            mSFK=k_nvfp4_scale,
            mSFV=v_nvfp4_scale,
            **direct,
        )
    torch.cuda.synchronize()
    if low_precision_supported:
        providers = (
            ("hao_native_nvfp4_fp8pv", run_fp8),
            ("hao_native_nvfp4_nvfp4pv", run_nvfp4),
            ("hao_native_bf16", run_bf16),
        )
    else:
        providers = (("hao_native_bf16", run_bf16),)
    timing_samples = {name: [] for name, _ in providers}
    provider_by_name = dict(providers)
    provider_names = tuple(provider_by_name)
    if low_precision_supported:
        balanced_orders = (
            (provider_names[0], provider_names[1], provider_names[2]),
            (provider_names[1], provider_names[2], provider_names[0]),
            (provider_names[2], provider_names[0], provider_names[1]),
            (provider_names[2], provider_names[1], provider_names[0]),
            (provider_names[1], provider_names[0], provider_names[2]),
            (provider_names[0], provider_names[2], provider_names[1]),
        )
    else:
        balanced_orders = (provider_names,)
    timing_round_orders = []
    for round_idx in range(args.rounds):
        order = balanced_orders[round_idx % len(balanced_orders)]
        timing_round_orders.append(list(order))
        for name in order:
            function = provider_by_name[name]
            timing_samples[name].append(
                float(
                    triton.testing.do_bench(
                        function,
                        warmup=args.warmup_ms,
                        rep=args.rep_ms,
                        return_mode="median",
                    )
                )
            )
    timing = {
        name: median(samples)
        for name, samples in timing_samples.items()
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
            "factory": (
                "HAO create_nvfp4_attention_tensors"
                if low_precision_supported
                else "HAO-compatible Gaussian D64 fallback"
            ),
            "timer": "triton.testing.do_bench median",
            "warmup_ms": args.warmup_ms,
            "rep_ms": args.rep_ms,
            "rounds": args.rounds,
            "provider_order": (
                "balanced six-permutation cycle"
                if low_precision_supported
                else "single BF16 provider"
            ),
            "seed": args.seed,
        },
        "timing_ms": timing,
        "timing_samples_ms": timing_samples,
        "timing_round_orders": timing_round_orders,
        "tflops": {
            name: flops / (milliseconds * 1e-3) / 1e12
            for name, milliseconds in timing.items()
        },
        "correctness": (
            {
                "hao_fp8_vs_bf16_output": comparison(
                    fp8_output, bf16_output
                ),
                "hao_fp8_vs_bf16_lse": comparison(fp8_lse, bf16_lse),
                "hao_nvfp4_vs_bf16_output": comparison(
                    nvfp4_output, bf16_output
                ),
                "hao_nvfp4_vs_bf16_lse": comparison(
                    nvfp4_lse, bf16_lse
                ),
            }
            if low_precision_supported
            else {}
        ),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
