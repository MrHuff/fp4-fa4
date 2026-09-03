#!/usr/bin/env python3
"""Measure the D64 GQA backward statistics-publication boundary.

This isolates the D128 chain's producer-native statistics contract at the
Llama-3.2-1B D64 geometry.  The precomputed route consumes the exact two FP32
pages that a projection epilogue can publish and therefore skips the standalone
O*dO / scaled-LSE preprocessing kernel.
"""

from __future__ import annotations

import argparse
import json

import torch

from tk_fa4 import (
    b300_prepare_nvfp4_projection_operand,
    b300_project_dout_unified_lowp_nvfp4,
)
from tk_fa4.lowp_fa4_bwd.profile_gqa_d128_chain import (
    CompiledGqaBackward,
    _time_cuda,
)
from tk_fa4.lowp_fa4_bwd.tune_d64_gqa_cute import _load_control


def metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    reference_f = reference.float().reshape(-1)
    actual_f = actual.float().reshape(-1)
    difference = actual_f - reference_f
    reference_norm = reference_f.norm().clamp_min(1.0e-20)
    actual_norm = actual_f.norm().clamp_min(1.0e-20)
    return {
        "cosine": float(
            torch.dot(reference_f, actual_f) / (reference_norm * actual_norm)
        ),
        "relative_l2": float(difference.norm() / reference_norm),
        "max_abs": float(difference.abs().max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=int, default=4096)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument(
        "--fp8-ds-lift",
        type=int,
        choices=(16, 32, 64, 128, 256),
        default=None,
    )
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--samples", type=int, default=51)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()

    fp8_ds_lift = args.fp8_ds_lift
    if fp8_ds_lift is None:
        if args.sequence <= 512:
            fp8_ds_lift = 32
        elif args.sequence <= 1024:
            fp8_ds_lift = 64
        elif args.sequence <= 2048:
            fp8_ds_lift = 128
        else:
            fp8_ds_lift = 256

    if torch.cuda.device_count() != 1:
        raise RuntimeError("profiling requires exactly one visible GPU")
    if args.q_heads <= 0 or args.q_heads % args.kv_heads:
        raise ValueError("q-heads must be divisible by kv-heads")
    if args.sequence <= 0 or args.sequence % 128:
        raise ValueError("sequence must be a positive multiple of 128")
    if args.hidden <= 0 or args.hidden % 256:
        raise ValueError("hidden must be a positive multiple of 256")

    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    batch = 1
    depth = 64
    scale_softmax = depth**-0.5

    def fp8(shape: tuple[int, ...]) -> torch.Tensor:
        return (
            torch.randn(shape, device="cuda", dtype=torch.float32)
            .mul_(0.25)
            .to(torch.float8_e4m3fn)
        )

    q = fp8((batch, args.sequence, args.q_heads, depth))
    k = fp8((batch, args.sequence, args.kv_heads, depth))
    v = fp8((batch, args.sequence, args.kv_heads, depth))
    output = fp8((batch, args.sequence, args.q_heads, depth))
    attention_output = (output.float() / 4.0).bfloat16()
    lse = (
        torch.rand(
            batch,
            args.q_heads,
            1,
            args.sequence,
            device="cuda",
            dtype=torch.float32,
        )
        + 8.0
    )
    lse_bsh = lse[:, :, 0].permute(0, 2, 1).contiguous()
    rows = batch * args.sequence
    width = args.q_heads * depth
    grad = (
        torch.randn(rows, args.hidden, device="cuda") * 0.1
    ).bfloat16()
    weight = (
        torch.randn(width, args.hidden, device="cuda") * 0.02
    ).bfloat16()
    grad_operand = tuple(b300_prepare_nvfp4_projection_operand(grad))
    weight_operand = tuple(b300_prepare_nvfp4_projection_operand(weight))

    def project_dout(
        *,
        publish_stats: bool,
        stats_workspace: torch.Tensor | None = None,
        dq_clear: torch.Tensor | None = None,
        store_bf16: bool = False,
    ):
        return b300_project_dout_unified_lowp_nvfp4(
            grad_operand,
            weight_operand,
            attention_output,
            lse_bsh,
            batch=batch,
            seqlen=args.sequence,
            heads=args.q_heads,
            store_bf16=store_bf16,
            publish_fp8_backward=True,
            publish_stats=publish_stats,
            stats_workspace=stats_workspace,
            dq_clear=dq_clear,
        )

    projected = project_dout(publish_stats=True, store_bf16=True)
    assert projected.dout_backward_fp8 is not None

    control = _load_control(fp8_p_storage="tmem")
    common = dict(
        control=control,
        q=q,
        k=k,
        v=v,
        dout=projected.dout_backward_fp8,
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
        lowp=True,
        scale_softmax=scale_softmax,
        exp2_degree=1,
        exp2_period=2,
        reuse_quantized_p=False,
        fp8_ds_lift=fp8_ds_lift,
        lowp_do_stages=1,
    )
    internal = CompiledGqaBackward(
        **common,
        o_or_sum=output,
        lse_or_scaled_lse=lse,
        precomputed_stats=False,
    )
    direct = CompiledGqaBackward(
        **common,
        o_or_sum=projected.dpsum,
        lse_or_scaled_lse=projected.lse_log2,
        precomputed_stats=True,
        workspace_stats=True,
    )

    internal.run(reset=True)
    project_dout(
        publish_stats=True,
        stats_workspace=direct.workspace_torch,
        dq_clear=direct.dq,
    )
    direct.run(reset=False)
    torch.cuda.synchronize()
    gradient_equivalence = {
        "dq": metrics(internal.dq, direct.dq),
        "dk": metrics(internal.dk, direct.dk),
        "dv": metrics(internal.dv, direct.dv),
    }

    def stats_free_projection() -> None:
        project_dout(publish_stats=False)

    def stats_free_projection_with_clear() -> None:
        project_dout(publish_stats=False, dq_clear=internal.dq)

    def direct_projection() -> None:
        project_dout(
            publish_stats=True,
            stats_workspace=direct.workspace_torch,
            dq_clear=direct.dq,
        )

    def separated_projection_and_backward() -> None:
        stats_free_projection()
        internal.run(reset=True)

    def clear_fused_projection_and_internal_backward() -> None:
        stats_free_projection_with_clear()
        internal.run(reset=False)

    def fused_projection_and_backward() -> None:
        direct_projection()
        direct.run(reset=False)

    timing = {
        "internal_stats_and_clear": _time_cuda(
            lambda: internal.run(reset=True),
            warmups=args.warmups,
            samples=args.samples,
        ),
        "precomputed_stats_and_clear": _time_cuda(
            lambda: direct.run(reset=True),
            warmups=args.warmups,
            samples=args.samples,
        ),
        "stats_free_projection": _time_cuda(
            stats_free_projection,
            warmups=args.warmups,
            samples=args.samples,
        ),
        "stats_free_projection_with_dq_clear": _time_cuda(
            stats_free_projection_with_clear,
            warmups=args.warmups,
            samples=args.samples,
        ),
        "direct_stats_and_clear_projection": _time_cuda(
            direct_projection,
            warmups=args.warmups,
            samples=args.samples,
        ),
        "separated_projection_and_backward": _time_cuda(
            separated_projection_and_backward,
            warmups=args.warmups,
            samples=args.samples,
        ),
        "clear_fused_projection_and_internal_backward": _time_cuda(
            clear_fused_projection_and_internal_backward,
            warmups=args.warmups,
            samples=args.samples,
        ),
        "fused_projection_and_backward": _time_cuda(
            fused_projection_and_backward,
            warmups=args.warmups,
            samples=args.samples,
        ),
    }
    internal_us = timing["internal_stats_and_clear"]["median_us"]
    precomputed_us = timing["precomputed_stats_and_clear"]["median_us"]
    separated_us = timing["separated_projection_and_backward"]["median_us"]
    clear_fused_us = timing[
        "clear_fused_projection_and_internal_backward"
    ]["median_us"]
    fused_us = timing["fused_projection_and_backward"]["median_us"]
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
                "policy": {
                    "input": "E4M3",
                    "output": "BF16",
                    "direct_compact_dq": True,
                    "exp2_degree": 1,
                    "exp2_period": 2,
                    "reuse_quantized_p": False,
                    "fp8_ds_lift": fp8_ds_lift,
                    "fused_block_seq": 32,
                },
                "precomputed_vs_internal": gradient_equivalence,
                "timing": timing,
                "speedup": {
                    "precomputed_stats": internal_us / precomputed_us,
                    "fused_clear_only_boundary": separated_us / clear_fused_us,
                    "fused_projection_boundary": separated_us / fused_us,
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
