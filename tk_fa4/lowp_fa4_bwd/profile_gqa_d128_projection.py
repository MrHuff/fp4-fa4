#!/usr/bin/env python3
"""Profile the projection-native D128 GQA forward/backward producer."""

from __future__ import annotations

import argparse
import json
import statistics

import torch

from tk_fa4 import (
    b300_pack_gqa_d128_rope,
    b300_pair_interleave_gqa_d128_qk_projection_weights,
    b300_prepare_nvfp4_projection_operand,
    b300_prepare_nvfp4_projection_weight,
    b300_project_qkv_gqa_d128_unified_lowp_nvfp4,
    b300_stack_gqa_d128_qkv_projection_weights,
)


def make_rope(sequence: int) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(sequence, device="cuda", dtype=torch.float32)
    frequencies = 1.0 / (
        10_000.0
        ** (
            torch.arange(64, device="cuda", dtype=torch.float32) / 64.0
        )
    )
    angles = positions[:, None] * frequencies[None, :]
    return angles.cos()[None].bfloat16(), angles.sin()[None].bfloat16()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=int, default=4096)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--store-bf16", action="store_true")
    parser.add_argument("--no-rope", action="store_true")
    parser.add_argument("--split-rope", action="store_true")
    parser.add_argument(
        "--cluster-cap",
        type=int,
        default=-1,
        help="resident cluster cap; -1 selects the production heuristic",
    )
    parser.add_argument(
        "--shared-rope-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--cache-adaptive-qk-scale",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--nvtx-only", action="store_true")
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise RuntimeError("profiling requires exactly one visible GPU")
    if args.q_heads <= 0 or args.q_heads % args.kv_heads:
        raise ValueError("q-heads must be divisible by kv-heads")
    for name, value in (
        ("sequence", args.sequence),
        ("hidden", args.hidden),
    ):
        if value % 256:
            raise ValueError(f"{name} must be divisible by 256")

    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    rows = args.sequence
    depth = 128
    x = (torch.randn(rows, args.hidden, device="cuda") * 0.1).bfloat16()
    q_weight = (
        torch.randn(
            args.q_heads * depth,
            args.hidden,
            device="cuda",
        )
        * 0.02
    ).bfloat16()
    k_weight = (
        torch.randn(
            args.kv_heads * depth,
            args.hidden,
            device="cuda",
        )
        * 0.02
    ).bfloat16()
    v_weight = torch.randn_like(k_weight.float()).mul_(0.02).bfloat16()
    q_weight, k_weight = (
        b300_pair_interleave_gqa_d128_qk_projection_weights(
            q_weight,
            k_weight,
        )
    )
    qkv_weight = b300_stack_gqa_d128_qkv_projection_weights(
        q_weight,
        k_weight,
        v_weight,
    )
    input_operand = tuple(b300_prepare_nvfp4_projection_operand(x))
    weight_operand = tuple(
        b300_prepare_nvfp4_projection_weight(qkv_weight)
    )
    qk_scales = torch.zeros(
        1,
        args.q_heads,
        7,
        device="cuda",
        dtype=torch.float32,
    )
    qk_scales[:, :, :2] = 16.0
    rope_cos, rope_sin = make_rope(args.sequence)
    rope_packed = b300_pack_gqa_d128_rope(rope_cos, rope_sin)

    def run() -> object:
        return b300_project_qkv_gqa_d128_unified_lowp_nvfp4(
            input_operand,
            weight_operand,
            qk_scales,
            batch=1,
            seqlen=args.sequence,
            q_heads=args.q_heads,
            kv_heads=args.kv_heads,
            store_bf16=args.store_bf16,
            publish_fp8_backward=True,
            rope_cos=(
                rope_cos if args.split_rope and not args.no_rope else None
            ),
            rope_sin=(
                rope_sin if args.split_rope and not args.no_rope else None
            ),
            rope_packed=(
                rope_packed
                if not args.split_rope and not args.no_rope
                else None
            ),
            cluster_cap=(None if args.cluster_cap < 0 else args.cluster_cap),
            cache_packed_rope=(
                args.shared_rope_cache and
                not args.no_rope and
                not args.split_rope
            ),
            cache_adaptive_qk_scale=(
                args.cache_adaptive_qk_scale and
                not args.no_rope and
                not args.split_rope
            ),
        )

    for _ in range(args.warmups):
        run()
    torch.cuda.synchronize()
    if args.nvtx_only:
        torch.cuda.nvtx.range_push("d128_gqa_projection")
        result = run()
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        print(
            json.dumps(
                {
                    "nvtx": "d128_gqa_projection",
                    "forward_shapes": [
                        list(tensor.shape)
                        for tensor in result.forward_operands()
                    ],
                },
                sort_keys=True,
            )
        )
        return

    timings: list[float] = []
    for _ in range(args.samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run()
        end.record()
        end.synchronize()
        timings.append(float(start.elapsed_time(end) * 1000.0))
    print(
        json.dumps(
            {
                "shape": {
                    "sequence": args.sequence,
                    "q_heads": args.q_heads,
                    "kv_heads": args.kv_heads,
                    "head_dim": depth,
                    "hidden": args.hidden,
                },
                "store_bf16": args.store_bf16,
                "rope": not args.no_rope,
                "packed_rope": not args.no_rope and not args.split_rope,
                "cluster_cap": (
                    "auto" if args.cluster_cap < 0 else args.cluster_cap
                ),
                "shared_rope_cache": args.shared_rope_cache,
                "cache_adaptive_qk_scale": args.cache_adaptive_qk_scale,
                "timing_us": {
                    "median": statistics.median(timings),
                    "minimum": min(timings),
                    "samples": timings,
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
