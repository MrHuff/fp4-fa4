#!/usr/bin/env python3
"""Validate the D64/D128 projection-native FP8 dO/statistics publication."""

from __future__ import annotations

import argparse
import json
import math
import statistics

import torch

from tk_fa4 import (
    b300_prepare_nvfp4_projection_operand,
    b300_project_dout_unified_lowp_nvfp4,
)


def metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float | bool]:
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
        "max_abs": float(difference.abs().max()),
    }


def time_cuda(function, *, warmups: int, samples: int) -> dict[str, float]:
    for _ in range(warmups):
        function()
    torch.cuda.synchronize()
    values: list[float] = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end) * 1000.0))
    return {
        "median_us": statistics.median(values),
        "minimum_us": min(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, choices=(64, 128), default=64)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument(
        "--d128-probability-log2-lift",
        type=float,
        choices=(0.0, 8.0),
        default=0.0,
        help="request the native-TK D128 +8 lstat ABI",
    )
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise RuntimeError("validation requires exactly one visible GPU")
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    batch = 1
    depth = args.head_dim
    if depth != 128 and args.d128_probability_log2_lift:
        raise ValueError("the explicit probability lift is D128-only")
    rows = batch * args.sequence
    width = args.heads * depth
    grad = (
        torch.randn(rows, args.hidden, device="cuda") * 0.1
    ).bfloat16()
    weight = (
        torch.randn(width, args.hidden, device="cuda") * 0.02
    ).bfloat16()
    attention_output = (
        torch.randn(
            batch,
            args.sequence,
            args.heads,
            depth,
            device="cuda",
        ) * 0.1
    ).bfloat16()
    lse = torch.randn(
        batch,
        args.sequence,
        args.heads,
        device="cuda",
        dtype=torch.float32,
    )
    grad_operand = tuple(b300_prepare_nvfp4_projection_operand(grad))
    weight_operand = tuple(b300_prepare_nvfp4_projection_operand(weight))
    stats_bytes = 2 * batch * args.heads * args.sequence * 4
    workspace = torch.empty(stats_bytes, device="cuda", dtype=torch.uint8)
    dq_clear = torch.full(
        (batch, args.sequence, args.heads, depth),
        1.0,
        device="cuda",
        dtype=torch.bfloat16,
    )
    dq_clear_only = torch.full_like(dq_clear, 1.0)

    def run():
        return b300_project_dout_unified_lowp_nvfp4(
            grad_operand,
            weight_operand,
            attention_output,
            lse,
            batch=batch,
            seqlen=args.sequence,
            heads=args.heads,
            store_bf16=True,
            publish_fp8_backward=True,
            publish_stats=True,
            stats_workspace=workspace,
            dq_clear=dq_clear,
            probability_log2_lift=args.d128_probability_log2_lift,
        )

    def run_clear_only():
        return b300_project_dout_unified_lowp_nvfp4(
            grad_operand,
            weight_operand,
            attention_output,
            lse,
            batch=batch,
            seqlen=args.sequence,
            heads=args.heads,
            store_bf16=False,
            publish_fp8_backward=True,
            publish_stats=False,
            dq_clear=dq_clear_only,
        )

    bundle = run()
    clear_only_bundle = run_clear_only()
    assert bundle.dout is not None
    assert bundle.dout_backward_fp8 is not None
    torch.cuda.synchronize()
    bf16_reference = torch.mm(grad, weight.T).reshape_as(attention_output)
    # dP consumes fixed-scale E4M3 dO with an x4 operand lift.  The centering
    # statistic must be derived from that exact rounded/saturated operand, not
    # from the pre-publication BF16 projection fragment.
    dpsum_reference = -4.0 * (
        attention_output.float() * bundle.dout_backward_fp8.float()
    ).sum(dim=-1).permute(0, 2, 1).unsqueeze(2)
    lse_reference = (
        -math.log2(math.e) * lse.permute(0, 2, 1).unsqueeze(2)
    )
    if depth == 64:
        lse_reference = lse_reference + 8.0
    elif args.d128_probability_log2_lift:
        lse_reference = lse_reference + args.d128_probability_log2_lift

    lift_control: dict[str, float | bool] | None = None
    if depth == 128 and args.d128_probability_log2_lift:
        control_workspace = torch.empty_like(workspace)
        control = b300_project_dout_unified_lowp_nvfp4(
            grad_operand,
            weight_operand,
            attention_output,
            lse,
            batch=batch,
            seqlen=args.sequence,
            heads=args.heads,
            store_bf16=True,
            publish_fp8_backward=True,
            publish_stats=True,
            stats_workspace=control_workspace,
            probability_log2_lift=0.0,
        )
        assert control.dout is not None
        assert control.dout_backward_fp8 is not None
        lstat_delta = bundle.lse_log2 - control.lse_log2
        lift_control = {
            "dout_bitwise_equal": bool(torch.equal(bundle.dout, control.dout)),
            "dout_fp8_bitwise_equal": bool(
                torch.equal(
                    bundle.dout_backward_fp8,
                    control.dout_backward_fp8,
                )
            ),
            "dstat_bitwise_equal": bool(
                torch.equal(bundle.dpsum, control.dpsum)
            ),
            "lstat_delta_max_abs_error": float(
                (lstat_delta - args.d128_probability_log2_lift).abs().max()
            ),
        }
        lift_control["passed"] = bool(
            lift_control["dout_bitwise_equal"]
            and lift_control["dout_fp8_bitwise_equal"]
            and lift_control["dstat_bitwise_equal"]
            and lift_control["lstat_delta_max_abs_error"] <= 1.0e-6
        )

    # Exercise the saturation cliff explicitly.  This mimics the 2^16 loss
    # scaling used by training while keeping the ordinary projection-quality
    # checks on their original distribution.
    stress_grad = (grad.float() * 2048.0).bfloat16()
    stress_grad_operand = tuple(
        b300_prepare_nvfp4_projection_operand(stress_grad)
    )
    stress_workspace = torch.empty_like(workspace)
    stress_bundle = b300_project_dout_unified_lowp_nvfp4(
        stress_grad_operand,
        weight_operand,
        attention_output,
        lse,
        batch=batch,
        seqlen=args.sequence,
        heads=args.heads,
        store_bf16=True,
        publish_fp8_backward=True,
        publish_stats=True,
        stats_workspace=stress_workspace,
        probability_log2_lift=args.d128_probability_log2_lift,
    )
    assert stress_bundle.dout is not None
    assert stress_bundle.dout_backward_fp8 is not None
    stress_dpsum_reference = -4.0 * (
        attention_output.float()
        * stress_bundle.dout_backward_fp8.float()
    ).sum(dim=-1).permute(0, 2, 1).unsqueeze(2)
    stress_saturated = int((stress_bundle.dout.abs() > 112.0).sum())
    stress_published_saturated = int(
        (stress_bundle.dout_backward_fp8.float().abs() == 448.0).sum()
    )
    quality = {
        "projection_vs_bf16": metrics(bf16_reference, bundle.dout),
        "fp8_publication_vs_projection": metrics(
            bundle.dout,
            bundle.dout_backward_fp8.float() / 4.0,
        ),
        "dpsum": metrics(dpsum_reference, bundle.dpsum),
        "saturated_dpsum": metrics(
            stress_dpsum_reference,
            stress_bundle.dpsum,
        ),
        "saturated_projection_values": stress_saturated,
        "saturated_published_fp8_values": stress_published_saturated,
        "scaled_lse": metrics(lse_reference, bundle.lse_log2),
        "dq_clear_zero": bool(not dq_clear.count_nonzero()),
        "stats_free_dq_clear_zero": bool(not dq_clear_only.count_nonzero()),
        "stats_free_outputs_empty": bool(
            not clear_only_bundle.dpsum.numel()
            and not clear_only_bundle.lse_log2.numel()
        ),
        "d128_lift_control": lift_control,
    }
    timing = {
        "bf16_projection": time_cuda(
            lambda: torch.mm(grad, weight.T),
            warmups=args.warmups,
            samples=args.samples,
        ),
        "unified_nvfp4_projection": time_cuda(
            run,
            warmups=args.warmups,
            samples=args.samples,
        ),
        "unified_nvfp4_projection_clear_only": time_cuda(
            run_clear_only,
            warmups=args.warmups,
            samples=args.samples,
        ),
    }
    passed = bool(
        quality["projection_vs_bf16"]["cosine"] > 0.985
        and quality["fp8_publication_vs_projection"]["cosine"] > 0.999
        and quality["dpsum"]["relative_l2"] < 5.0e-5
        and quality["saturated_dpsum"]["relative_l2"] < 5.0e-5
        and quality["saturated_projection_values"] > 0
        and quality["saturated_published_fp8_values"] > 0
        and quality["scaled_lse"]["relative_l2"] < 1.0e-6
        and quality["dq_clear_zero"]
        and quality["stats_free_dq_clear_zero"]
        and quality["stats_free_outputs_empty"]
        and (lift_control is None or lift_control["passed"])
    )
    print(
        json.dumps(
            {
                "shape": {
                    "batch": batch,
                    "sequence": args.sequence,
                    "heads": args.heads,
                    "head_dim": depth,
                    "hidden": args.hidden,
                },
                "quality": quality,
                "timing": timing,
                "passed": passed,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
