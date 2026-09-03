#!/usr/bin/env python3
"""Validate owner-CTA dQ reduction and projection-native NVFP4 output."""

from __future__ import annotations

import argparse

import torch


def _metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float | bool]:
    reference_f = reference.float().reshape(-1)
    actual_f = actual.float().reshape(-1)
    difference = actual_f - reference_f
    reference_norm = reference_f.norm().clamp_min(1.0e-20)
    actual_norm = actual_f.norm().clamp_min(1.0e-20)
    return {
        "finite": bool(torch.isfinite(actual_f).all()),
        "cosine": float(
            torch.dot(reference_f, actual_f) / (reference_norm * actual_norm)
        ),
        "relative_l2": float(difference.norm() / reference_norm),
        "norm_ratio": float(actual_norm / reference_norm),
        "max_abs": float(difference.abs().max()),
    }


def run_attention_owner(
    *,
    sequence: int = 256,
    q_heads: int = 4,
    kv_heads: int = 1,
    depth: int = 128,
    fuse_kv: bool = False,
    validate: bool = False,
    fused_dq_scale: bool = False,
    warmups: int = 0,
    samples: int = 0,
) -> None:
    from tk_fa4.lowp_fa4_bwd.profile_gqa_d128_chain import (
        CompiledGqaBackward,
        _inverse_rope_pair_native,
        _make_rope,
        _time_cuda,
    )
    from tk_fa4.lowp_fa4_bwd.tune_d64_gqa_cute import _load_control

    batch = 1
    if depth not in (64, 128):
        raise ValueError("depth must be 64 or 128")
    reduction = (q_heads + 2 * kv_heads) * depth
    device = torch.device("cuda")
    torch.manual_seed(20260816)
    q = (torch.randn(batch, sequence, q_heads, depth, device=device) * 0.25).to(
        torch.float8_e4m3fn
    )
    k = (torch.randn(batch, sequence, kv_heads, depth, device=device) * 0.25).to(
        torch.float8_e4m3fn
    )
    v = (torch.randn(batch, sequence, kv_heads, depth, device=device) * 0.25).to(
        torch.float8_e4m3fn
    )
    dout = (
        torch.randn(batch, sequence, q_heads, depth, device=device) * 0.25
    ).to(torch.float8_e4m3fn)
    stats = torch.zeros(batch, q_heads, 1, sequence, device=device)
    lse = torch.zeros_like(stats)
    payload = torch.empty(
        sequence,
        reduction // 2,
        device=device,
        dtype=torch.float4_e2m1fn_x2,
    )
    payload.view(torch.uint8).zero_()
    scales = torch.empty(
        sequence // 128,
        reduction // 64,
        512,
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    scales.view(torch.uint8).zero_()
    rope = _make_rope(sequence, depth)
    from tk_fa4 import (
        b300_pack_gqa_d128_hierarchical_qkv_gradient_nvfp4_tiles,
        b300_pack_gqa_d128_rope,
        b300_prepare_nvfp4_projection_operand,
        b300_project_nvfp4,
    )

    if depth == 128:
        rope_packed = b300_pack_gqa_d128_rope(*rope)
    else:
        rope_packed = (
            torch.stack(rope, dim=-1)
            .contiguous()
            .view(torch.int32)
            .reshape(batch, sequence, depth // 2)
            .contiguous()
        )
    control = _load_control(
        fp8_p_storage="tmem" if depth == 64 else "shared",
        owner_fused_dq_scale=fused_dq_scale,
    )
    reference_gradient = None
    if validate:
        direct = CompiledGqaBackward(
            control,
            q=q,
            k=k,
            v=v,
            o_or_sum=stats,
            dout=dout,
            lse_or_scaled_lse=lse,
            q_heads=q_heads,
            kv_heads=kv_heads,
            lowp=True,
            precomputed_stats=True,
            workspace_stats=True,
            scale_softmax=(depth**-0.5) / 16.0,
        )
        direct.run(reset=True)
        torch.cuda.synchronize()
        reference_gradient = torch.cat(
            (
                _inverse_rope_pair_native(direct.dq, *rope).reshape(
                    sequence, -1
                ),
                _inverse_rope_pair_native(direct.dk, *rope).reshape(
                    sequence, -1
                ),
                direct.dv.reshape(sequence, -1),
            ),
            dim=1,
        ).contiguous()
        global_scale = b300_prepare_nvfp4_projection_operand(
            reference_gradient
        )[2]
    else:
        # Keep the local E4M3 decode scales in range for the small synthetic
        # gradients used by this smoke test.
        global_scale = torch.full((1,), 2.0**-12, device=device)
    owner = CompiledGqaBackward(
        control,
        q=q,
        k=k,
        v=v,
        o_or_sum=stats,
        dout=dout,
        lse_or_scaled_lse=lse,
        q_heads=q_heads,
        kv_heads=kv_heads,
        lowp=True,
        precomputed_stats=True,
        workspace_stats=True,
        scale_softmax=(depth**-0.5) / 16.0,
        owner_output_operand=(payload, scales),
        owner_gradient_global_scale=global_scale,
        owner_rope=rope_packed,
        owner_quantize_kv=fuse_kv,
    )
    owner.run(reset=True)
    torch.cuda.synchronize()

    projection_weight = (
        torch.randn(256, reduction, device=device) * 0.02
    ).bfloat16()
    projection_weight_operand = tuple(
        b300_prepare_nvfp4_projection_operand(projection_weight)
    )
    projection = b300_project_nvfp4(
        (payload, scales, global_scale),
        projection_weight_operand,
    )
    assert owner.owner_dq_clear is not None
    assert owner.owner_dq_ready is not None
    if not fuse_kv and depth == 128:
        b300_pack_gqa_d128_hierarchical_qkv_gradient_nvfp4_tiles(
            owner.owner_dq_clear,
            owner.dk,
            owner.dv,
            global_scale,
            rope_packed,
            (payload, scales),
            owner.owner_dq_ready,
            row_tile_begin=0,
            row_tile_end=sequence // 128,
            col_tile_begin=q_heads,
            col_tile_end=q_heads + 2 * kv_heads,
        )
    projection_with_kv = b300_project_nvfp4(
        (payload, scales, global_scale),
        projection_weight_operand,
    )
    validation = {}
    if reference_gradient is not None:
        reference_operand = tuple(
            b300_prepare_nvfp4_projection_operand(
                reference_gradient,
                global_scale=global_scale,
            )
        )
        reference_projection = b300_project_nvfp4(
            reference_operand,
            projection_weight_operand,
        )
        bf16_projection = torch.mm(reference_gradient, projection_weight.T)
        validation = {
            "owner_vs_materialized_projection": _metrics(
                reference_projection,
                projection_with_kv,
            ),
            "owner_vs_bf16_projection": _metrics(
                bf16_projection,
                projection_with_kv,
            ),
            "materialized_vs_bf16_projection": _metrics(
                bf16_projection,
                reference_projection,
            ),
            "payload_byte_match": float(
                (
                    payload.view(torch.uint8)
                    == reference_operand[0].view(torch.uint8)
                )
                .float()
                .mean()
            ),
            "scale_byte_match": float(
                (
                    scales.view(torch.uint8)
                    == reference_operand[1].view(torch.uint8)
                )
                .float()
                .mean()
            ),
        }
    timing = {}
    if samples:
        timing = {
            "owner_backward_with_clear": _time_cuda(
                lambda: owner.run(reset=True),
                warmups=warmups,
                samples=samples,
            ),
            "owner_backward_and_projection": _time_cuda(
                lambda: (
                    owner.run(reset=True),
                    b300_project_nvfp4(
                        (payload, scales, global_scale),
                        projection_weight_operand,
                    ),
                ),
                warmups=warmups,
                samples=samples,
            ),
        }
    torch.cuda.synchronize()
    assert owner.owner_dq_ready is not None
    print(
        {
            "ready": owner.owner_dq_ready.cpu().tolist(),
            "payload_nonzero": int(torch.count_nonzero(payload.view(torch.uint8))),
            "scale_nonzero": int(torch.count_nonzero(scales.view(torch.uint8))),
            "scale_nan": int(torch.isnan(scales.float()).sum()),
            "scale_inf": int(torch.isinf(scales.float()).sum()),
            "scale_max": float(scales.float().abs().max()),
            "scale_saturated": int((scales.float().abs() == 448.0).sum()),
            "scale_nan_by_row_tile": torch.isnan(scales.float())
            .reshape(sequence // 128, -1)
            .sum(dim=1)
            .cpu()
            .tolist(),
            "projection_finite": bool(torch.isfinite(projection).all()),
            "projection_max": float(projection.float().abs().max()),
            "projection_with_kv_finite": bool(
                torch.isfinite(projection_with_kv).all()
            ),
            "projection_with_kv_max": float(
                projection_with_kv.float().abs().max()
            ),
            **validation,
            "timing": timing,
            "fused_dq_scale": fused_dq_scale,
            "global_scale": float(global_scale),
            "dq_acc_nonzero": int(torch.count_nonzero(owner.owner_dq_acc)),
            "dq_acc_finite": bool(torch.isfinite(owner.owner_dq_acc).all()),
            "dk_nonzero": int(torch.count_nonzero(owner.dk)),
            "dk_finite": bool(torch.isfinite(owner.dk).all()),
            "dv_nonzero": int(torch.count_nonzero(owner.dv)),
            "dv_finite": bool(torch.isfinite(owner.dv).all()),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=int, default=256)
    parser.add_argument("--q-heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=1)
    parser.add_argument("--depth", type=int, choices=(64, 128), default=128)
    parser.add_argument("--fuse-kv", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--fused-dq-scale", action="store_true")
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--samples", type=int, default=0)
    args = parser.parse_args()
    run_attention_owner(
        sequence=args.sequence,
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
        depth=args.depth,
        fuse_kv=args.fuse_kv,
        validate=args.validate,
        fused_dq_scale=args.fused_dq_scale,
        warmups=args.warmups,
        samples=args.samples,
    )


if __name__ == "__main__":
    main()
