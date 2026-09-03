#!/usr/bin/env python3
"""Validate and benchmark the fused Q/K projection FP4 epilogue."""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable

import torch

from tk_fa4 import _C_b300_lowp_bwd as lowp
from tk_fa4 import (
    b300_prepare_nvfp4_projection_operand,
    b300_project_qk_adaptive_lowp_nvfp4,
)


def tensor_metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    reference_f = reference.float()
    actual_f = actual.float()
    difference = actual_f - reference_f
    reference_norm = torch.linalg.vector_norm(reference_f).clamp_min(1.0e-20)
    actual_norm = torch.linalg.vector_norm(actual_f).clamp_min(1.0e-20)
    return {
        "cosine": float(
            torch.sum(reference_f * actual_f) / (reference_norm * actual_norm)
        ),
        "relative_l2": float(torch.linalg.vector_norm(difference) / reference_norm),
        "max_abs": float(difference.abs().max()),
    }


def time_rotated(
    candidates: dict[str, Callable[[], object]],
    warmups: int,
    samples: int,
) -> dict[str, dict[str, float]]:
    names = list(candidates)
    for iteration in range(warmups):
        for offset in range(len(names)):
            candidates[names[(iteration + offset) % len(names)]]()
    torch.cuda.synchronize()

    elapsed = {name: [] for name in names}
    for iteration in range(samples):
        for offset in range(len(names)):
            name = names[(iteration + offset) % len(names)]
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            candidates[name]()
            end.record()
            end.synchronize()
            elapsed[name].append(float(start.elapsed_time(end)))
    return {
        name: {
            "median_ms": statistics.median(values),
            "mean_ms": statistics.fmean(values),
            "minimum_ms": min(values),
        }
        for name, values in elapsed.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.set_device(0)
    batch = 1
    output_width = args.heads * 192
    input_tensor = (
        torch.randn(batch, args.sequence, args.hidden, device="cuda") * 0.1
    ).bfloat16()
    q_weight = (
        torch.randn(output_width, args.hidden, device="cuda") * 0.02
    ).bfloat16()
    k_weight = (
        torch.randn(output_width, args.hidden, device="cuda") * 0.02
    ).bfloat16()
    qk_weight = torch.cat((q_weight, k_weight), dim=0).contiguous()
    input_nvfp4 = b300_prepare_nvfp4_projection_operand(
        input_tensor.reshape(batch * args.sequence, args.hidden)
    )
    weight_nvfp4 = b300_prepare_nvfp4_projection_operand(qk_weight)

    q_reference = torch.mm(input_tensor.reshape(-1, args.hidden), q_weight.T)
    k_reference = torch.mm(input_tensor.reshape(-1, args.hidden), k_weight.T)
    q_reference = q_reference.reshape(batch, args.sequence, args.heads, 192)
    k_reference = k_reference.reshape(batch, args.sequence, args.heads, 192)
    adaptive = lowp.quantize_fp4_dual_qk_adaptive(
        q_reference,
        k_reference,
        16.0,
        2.0**-12,
        0.325,
        2.75,
        float(192**-0.5),
        4096.0,
    )
    scales = adaptive[4]

    fused = lowp.project_qk_adaptive_fp4_nvfp4(
        *input_nvfp4,
        *weight_nvfp4,
        scales,
        batch,
        args.sequence,
        args.heads,
        True,
    )
    q_fused, k_fused = fused[:2]
    repacked = lowp.quantize_fp4_dual_qk_precomputed_scales(
        q_fused,
        k_fused,
        scales,
    )
    projection_only_output = lowp.project_qk_adaptive_fp4_nvfp4(
        *input_nvfp4,
        *weight_nvfp4,
        scales,
        batch,
        args.sequence,
        args.heads,
        False,
    )
    auto_output = lowp.project_qk_adaptive_fp4_nvfp4_dispatch(
        *input_nvfp4,
        *weight_nvfp4,
        scales,
        batch,
        args.sequence,
        args.heads,
        0,
    )
    public_q, public_k, public_operands = b300_project_qk_adaptive_lowp_nvfp4(
        tuple(input_nvfp4),
        tuple(weight_nvfp4),
        scales,
        batch=batch,
        seqlen=args.sequence,
        heads=args.heads,
    )
    public_output = (
        public_q,
        public_k,
        public_operands.q_fp4,
        public_operands.score_q_fp4,
        public_operands.k_fp4,
        public_operands.score_k_fp4,
        public_operands.qk_scales,
    )
    torch.cuda.synchronize()

    layout_names = (
        "q_sequence_aligned",
        "q_depth_packed",
        "k_depth_aligned",
        "k_depth_packed",
    )
    layout_checks = {}
    for layout_idx, (name, fused_layout, reference_layout) in enumerate(
        zip(
            layout_names,
            fused[2:6],
            repacked[:4],
            strict=True,
        )
    ):
        mismatch = fused_layout != reference_layout
        layout_checks[name] = {
            "equal": bool(not mismatch.any()),
            "mismatches": int(mismatch.sum()),
            "elements": int(mismatch.numel()),
        }
        auto_mismatch = auto_output[2 + layout_idx] != reference_layout
        layout_checks[name]["auto_equal"] = bool(not auto_mismatch.any())
        layout_checks[name]["auto_mismatches"] = int(auto_mismatch.sum())
        public_mismatch = public_output[2 + layout_idx] != reference_layout
        layout_checks[name]["public_api_equal"] = bool(not public_mismatch.any())
        layout_checks[name]["public_api_mismatches"] = int(public_mismatch.sum())

    def baseline() -> object:
        q = torch.mm(input_tensor.reshape(-1, args.hidden), q_weight.T)
        k = torch.mm(input_tensor.reshape(-1, args.hidden), k_weight.T)
        q = q.reshape(batch, args.sequence, args.heads, 192)
        k = k.reshape(batch, args.sequence, args.heads, 192)
        return lowp.quantize_fp4_dual_qk_precomputed_scales(q, k, scales)

    def nvfp4_plus_packer() -> object:
        projection = lowp.project_qk_adaptive_fp4_nvfp4(
            *input_nvfp4,
            *weight_nvfp4,
            scales,
            batch,
            args.sequence,
            args.heads,
            False,
        )
        return lowp.quantize_fp4_dual_qk_precomputed_scales(
            projection[0], projection[1], scales
        )

    def nvfp4_projection_only() -> object:
        return lowp.project_qk_adaptive_fp4_nvfp4(
            *input_nvfp4,
            *weight_nvfp4,
            scales,
            batch,
            args.sequence,
            args.heads,
            False,
        )

    def packer_only() -> object:
        return lowp.quantize_fp4_dual_qk_precomputed_scales(
            projection_only_output[0], projection_only_output[1], scales
        )

    def fused_call() -> object:
        return lowp.project_qk_adaptive_fp4_nvfp4(
            *input_nvfp4,
            *weight_nvfp4,
            scales,
            batch,
            args.sequence,
            args.heads,
            True,
        )

    def auto_call() -> object:
        return lowp.project_qk_adaptive_fp4_nvfp4_dispatch(
            *input_nvfp4,
            *weight_nvfp4,
            scales,
            batch,
            args.sequence,
            args.heads,
            0,
        )

    def input_preparation() -> object:
        return b300_prepare_nvfp4_projection_operand(
            input_tensor.reshape(batch * args.sequence, args.hidden)
        )

    def bf16_input_auto_call() -> object:
        prepared_input = b300_prepare_nvfp4_projection_operand(
            input_tensor.reshape(batch * args.sequence, args.hidden)
        )
        return lowp.project_qk_adaptive_fp4_nvfp4_dispatch(
            *prepared_input,
            *weight_nvfp4,
            scales,
            batch,
            args.sequence,
            args.heads,
            0,
        )

    timing = time_rotated(
        {
            "bf16_cublas_plus_packer": baseline,
            "nvfp4_projection_only": nvfp4_projection_only,
            "standalone_packer_only": packer_only,
            "nvfp4_projection_plus_packer": nvfp4_plus_packer,
            "fused_nvfp4_epilogue": fused_call,
            "auto_dispatched_nvfp4": auto_call,
            "nvfp4_input_preparation": input_preparation,
            "bf16_input_to_auto_nvfp4": bf16_input_auto_call,
        },
        args.warmups,
        args.samples,
    )
    result = {
        "shape": {
            "batch": batch,
            "sequence": args.sequence,
            "heads": args.heads,
            "hidden": args.hidden,
        },
        "q_metrics": tensor_metrics(q_reference, q_fused),
        "k_metrics": tensor_metrics(k_reference, k_fused),
        "layouts": layout_checks,
        "timing": timing,
        "speedup_vs_bf16_cublas_plus_packer": (
            timing["bf16_cublas_plus_packer"]["median_ms"]
            / timing["fused_nvfp4_epilogue"]["median_ms"]
        ),
        "auto_speedup_vs_bf16_cublas_plus_packer": (
            timing["bf16_cublas_plus_packer"]["median_ms"]
            / timing["auto_dispatched_nvfp4"]["median_ms"]
        ),
        "speedup_vs_nvfp4_plus_packer": (
            timing["nvfp4_projection_plus_packer"]["median_ms"]
            / timing["fused_nvfp4_epilogue"]["median_ms"]
        ),
        "auto_publication_route": "fused" if args.hidden >= 3072 else "separate",
        "auto_speedup_vs_forced_other": (
            (
                timing["nvfp4_projection_plus_packer"]["median_ms"]
                if args.hidden >= 3072
                else timing["fused_nvfp4_epilogue"]["median_ms"]
            )
            / timing["auto_dispatched_nvfp4"]["median_ms"]
        ),
        "bf16_input_speedup_vs_bf16_cublas_plus_packer": (
            timing["bf16_cublas_plus_packer"]["median_ms"]
            / timing["bf16_input_to_auto_nvfp4"]["median_ms"]
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))

    if not all(
        check["equal"] and check["auto_equal"] and check["public_api_equal"]
        for check in layout_checks.values()
    ):
        raise SystemExit("fused FP4 layouts differ from the standalone packer")
    if min(result["q_metrics"]["cosine"], result["k_metrics"]["cosine"]) < 0.98:
        raise SystemExit("fused projection does not match the BF16 GEMM")


if __name__ == "__main__":
    main()
