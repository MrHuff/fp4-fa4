#!/usr/bin/env python3
"""Profile a paired-head D64 GQA projection-backward consumer.

Two adjacent D64 heads form one physical D128 tile.  This lets the established
D128 inverse-RoPE/NVFP4 producer consume materialized D64 dQ/dK/dV without a
concatenation kernel and preserves the standard [all Q | all K | all V]
projection-reduction order.
"""

from __future__ import annotations

import argparse
import json

import torch

from tk_fa4 import (
    b300_pack_gqa_d128_hierarchical_qkv_gradient_nvfp4_tiles,
    b300_pack_gqa_d64_paired_rope,
    b300_prepare_nvfp4_projection_operand,
    b300_prepare_nvfp4_projection_weight,
    b300_project_gqa_d64_paired_qkv_gradient_nvfp4,
    b300_project_nvfp4,
)
from tk_fa4.lowp_fa4_bwd.profile_gqa_d128_chain import (
    CompiledGqaBackward,
    _inverse_rope_pair_native,
    _make_rope,
    _metrics,
    _time_cuda,
)
from tk_fa4.lowp_fa4_bwd.tune_d64_gqa_cute import _load_control


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=int, default=4096)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument(
        "--direct-tma-dkdv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reduce dK/dV directly into final KV-head outputs",
    )
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise RuntimeError("profiling requires exactly one visible GPU")
    if args.q_heads % 2 or args.kv_heads % 2:
        raise ValueError("paired D64 publication requires even Q and KV heads")
    if args.q_heads <= 0 or args.q_heads % args.kv_heads:
        raise ValueError("q-heads must be positive and divisible by kv-heads")
    reduction = (args.q_heads + 2 * args.kv_heads) * 64
    if args.sequence % 256 or args.hidden % 256 or reduction % 256:
        raise ValueError("sequence, hidden, and reduction must divide 256")

    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    batch = 1
    depth = 64
    device = torch.device("cuda")

    def fp8(shape: tuple[int, ...]) -> torch.Tensor:
        return (
            torch.randn(shape, device=device, dtype=torch.float32)
            .mul_(0.25)
            .to(torch.float8_e4m3fn)
        )

    q = fp8((batch, args.sequence, args.q_heads, depth))
    k = fp8((batch, args.sequence, args.kv_heads, depth))
    v = fp8((batch, args.sequence, args.kv_heads, depth))
    dout = fp8((batch, args.sequence, args.q_heads, depth))
    stats = torch.zeros(
        batch, args.q_heads, 1, args.sequence, device=device
    )
    lse = torch.zeros_like(stats)
    rope = _make_rope(args.sequence, depth)
    paired_rope = b300_pack_gqa_d64_paired_rope(*rope)

    control = _load_control(
        fp8_p_storage="tmem",
        owner_fused_dq_scale=False,
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
        direct_tma_dkdv=args.direct_tma_dkdv,
    )
    backward.run(reset=True)
    torch.cuda.synchronize()

    reference_gradient = torch.cat(
        (
            _inverse_rope_pair_native(backward.dq, *rope).reshape(
                args.sequence, -1
            ),
            _inverse_rope_pair_native(backward.dk, *rope).reshape(
                args.sequence, -1
            ),
            backward.dv.reshape(args.sequence, -1),
        ),
        dim=1,
    ).contiguous()
    gradient_global_scale = b300_prepare_nvfp4_projection_operand(
        reference_gradient
    )[2]
    projection_weight = (
        torch.randn(args.hidden, reduction, device=device) * 0.02
    ).bfloat16()
    projection_weight_operand = tuple(
        b300_prepare_nvfp4_projection_weight(projection_weight)
    )

    def project_paired() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        result = b300_project_gqa_d64_paired_qkv_gradient_nvfp4(
            backward.dq,
            backward.dk,
            backward.dv,
            projection_weight_operand,
            gradient_global_scale,
            paired_rope,
            return_operand=True,
        )
        assert isinstance(result, tuple)
        return result

    paired_projection, paired_payload, paired_scales = project_paired()
    staged_payload = torch.empty_like(paired_payload)
    staged_scales = torch.empty_like(paired_scales)
    staged_arrivals = torch.zeros(
        batch,
        args.q_heads // 2,
        args.sequence // 128,
        device=device,
        dtype=torch.int32,
    )

    def pack_paired_only() -> None:
        b300_pack_gqa_d128_hierarchical_qkv_gradient_nvfp4_tiles(
            backward.dq.view(
                batch, args.sequence, args.q_heads // 2, 128
            ),
            backward.dk.view(
                batch, args.sequence, args.kv_heads // 2, 128
            ),
            backward.dv.view(
                batch, args.sequence, args.kv_heads // 2, 128
            ),
            gradient_global_scale,
            paired_rope,
            (staged_payload, staged_scales),
            staged_arrivals,
            row_tile_begin=0,
            row_tile_end=args.sequence // 128,
            col_tile_begin=0,
            col_tile_end=(args.q_heads + 2 * args.kv_heads) // 2,
        )

    pack_paired_only()
    staged_operand = (
        staged_payload,
        staged_scales,
        gradient_global_scale,
    )
    reference_operand = tuple(
        b300_prepare_nvfp4_projection_operand(
            reference_gradient,
            global_scale=gradient_global_scale,
        )
    )
    reference_projection = b300_project_nvfp4(
        reference_operand,
        projection_weight_operand,
    )
    bf16_projection = torch.mm(reference_gradient, projection_weight.T)
    torch.cuda.synchronize()

    def project_generic_materialized() -> torch.Tensor:
        operand = tuple(
            b300_prepare_nvfp4_projection_operand(
                reference_gradient,
                global_scale=gradient_global_scale,
            )
        )
        return b300_project_nvfp4(operand, projection_weight_operand)

    def backward_and_paired_projection() -> object:
        backward.run(reset=True)
        return project_paired()

    timing = {
        "attention_backward_with_clear": _time_cuda(
            lambda: backward.run(reset=True),
            warmups=args.warmups,
            samples=args.samples,
        ),
        "paired_inverse_rope_pack_and_projection": _time_cuda(
            project_paired,
            warmups=args.warmups,
            samples=args.samples,
        ),
        "paired_inverse_rope_pack_only": _time_cuda(
            pack_paired_only,
            warmups=args.warmups,
            samples=args.samples,
        ),
        "prepacked_projection_only": _time_cuda(
            lambda: b300_project_nvfp4(
                staged_operand, projection_weight_operand
            ),
            warmups=args.warmups,
            samples=args.samples,
        ),
        "bf16_projection_only": _time_cuda(
            lambda: torch.mm(reference_gradient, projection_weight.T),
            warmups=args.warmups,
            samples=args.samples,
        ),
        "generic_pack_and_projection_no_inverse_rope": _time_cuda(
            project_generic_materialized,
            warmups=args.warmups,
            samples=args.samples,
        ),
        "attention_backward_and_paired_projection": _time_cuda(
            backward_and_paired_projection,
            warmups=args.warmups,
            samples=args.samples,
        ),
    }
    quality = {
        "paired_vs_materialized_projection": _metrics(
            reference_projection, paired_projection
        ),
        "paired_vs_bf16_projection": _metrics(
            bf16_projection, paired_projection
        ),
        "materialized_vs_bf16_projection": _metrics(
            bf16_projection, reference_projection
        ),
        "payload_byte_match": float(
            (
                paired_payload.view(torch.uint8)
                == reference_operand[0].view(torch.uint8)
            )
            .float()
            .mean()
        ),
        "scale_byte_match": float(
            (
                paired_scales.view(torch.uint8)
                == reference_operand[1].view(torch.uint8)
            )
            .float()
            .mean()
        ),
    }
    print(
        json.dumps(
            {
                "shape": {
                    "batch": batch,
                    "sequence": args.sequence,
                    "q_heads": args.q_heads,
                    "kv_heads": args.kv_heads,
                    "head_dim": depth,
                    "hidden": args.hidden,
                },
                "direct_tma_dkdv": args.direct_tma_dkdv,
                "quality": quality,
                "timing": timing,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
