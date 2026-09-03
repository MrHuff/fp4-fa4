#!/usr/bin/env python3
"""Validate and time the adaptive FP4/FP8 backward BF16-dQ endpoint."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from tk_fa4 import _C_b300_lowp_bwd as lowp
from tk_fa4.interface import (
    b300_adaptive_lowp_operands_from_projection,
    b300_mha_bwd_adaptive_lowp,
    b300_mha_fwd,
)


def parse_shapes(value: str) -> list[tuple[int, int]]:
    shapes = []
    for item in value.split(","):
        sequence, heads = item.lower().split("x", maxsplit=1)
        shapes.append((int(sequence), int(heads)))
    return shapes


def component_metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float | bool]:
    reference_f = reference.float()
    actual_f = actual.float()
    reference_norm = torch.linalg.vector_norm(reference_f).clamp_min(1e-20)
    actual_norm = torch.linalg.vector_norm(actual_f).clamp_min(1e-20)
    difference = actual_f - reference_f
    return {
        "finite": bool(torch.isfinite(actual_f).all()),
        "cosine": float(
            torch.sum(reference_f * actual_f) / (reference_norm * actual_norm)
        ),
        "relative_l2": float(torch.linalg.vector_norm(difference) / reference_norm),
        "max_abs": float(difference.abs().max()),
    }


def time_rotated(
    callables: dict[str, object],
    warmups: int,
    samples: int,
) -> dict[str, dict[str, float]]:
    names = list(callables)
    for iteration in range(warmups):
        order = names[iteration % len(names) :] + names[: iteration % len(names)]
        for name in order:
            callables[name]()
    torch.cuda.synchronize()

    raw = {name: [] for name in names}
    for iteration in range(samples):
        order = names[iteration % len(names) :] + names[: iteration % len(names)]
        for name in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            callables[name]()
            end.record()
            end.synchronize()
            raw[name].append(float(start.elapsed_time(end)))
    return {
        name: {
            "median_ms": statistics.median(values),
            "mean_ms": statistics.fmean(values),
            "minimum_ms": min(values),
        }
        for name, values in raw.items()
    }


def build_problem(sequence: int, heads: int, seed: int):
    torch.manual_seed(seed)
    qk_shape = (1, sequence, heads, 192)
    value_shape = (1, sequence, heads, 128)
    q = (torch.randn(qk_shape, device="cuda") * 0.1).bfloat16()
    k = (torch.randn(qk_shape, device="cuda") * 0.1).bfloat16()
    v = (torch.randn(value_shape, device="cuda") * 0.1).bfloat16()
    dout = (torch.randn(value_shape, device="cuda") * 0.1).bfloat16()
    out, lse = b300_mha_fwd(q, k, v, causal=True, return_lse=True)
    softmax_scale = float(192**-0.5)
    packed = lowp.quantize_fp4_dual_qk_adaptive(
        q,
        k,
        16.0,
        2.0**-12,
        0.325,
        2.75,
        softmax_scale,
        4096.0,
    )
    operands = b300_adaptive_lowp_operands_from_projection(q, k, *packed)

    def bf16_control():
        return lowp.backward_bf16_control(
            q, k, v, out, lse, dout, True, softmax_scale, False
        )

    def adaptive_fp32():
        return lowp.backward_fp4_fp8dpdv_x32_split_dk_adaptive_native(
            q,
            k,
            v,
            out,
            lse,
            dout,
            *packed,
            4096.0,
            True,
            softmax_scale,
            False,
        )

    def adaptive_bf16_dq():
        return lowp.backward_fp4_fp8dpdv_x32_split_dk_adaptive_bf16dq_native(
            q,
            k,
            v,
            out,
            lse,
            dout,
            *packed,
            4096.0,
            True,
            softmax_scale,
            False,
        )

    def interface_bf16_dq():
        return b300_mha_bwd_adaptive_lowp(
            q,
            k,
            v,
            out,
            lse,
            dout,
            operands,
            causal=True,
            softmax_scale=softmax_scale,
            return_bf16_dq=True,
        )

    return (
        (q, k, v, out, lse, dout, operands),
        {
            "bf16_control": bf16_control,
            "adaptive_fp32_dq": adaptive_fp32,
            "adaptive_bf16_dq": adaptive_bf16_dq,
            "interface_bf16_dq": interface_bf16_dq,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shapes",
        default="4096x24,8192x8,8192x24,8192x64,16384x24,16384x64",
    )
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--repeat-calls", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026081201)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("validation requires exactly one visible GPU")

    result: dict[str, object] = {
        "configuration": {
            "shapes": parse_shapes(args.shapes),
            "warmups": args.warmups,
            "samples": args.samples,
            "repeat_calls": args.repeat_calls,
            "seed": args.seed,
        },
        "shapes": {},
    }
    for shape_index, (sequence, heads) in enumerate(parse_shapes(args.shapes)):
        retained, routes = build_problem(
            sequence,
            heads,
            args.seed + shape_index,
        )
        reference = routes["bf16_control"]()
        fp32_result = routes["adaptive_fp32_dq"]()
        bf16_result = routes["adaptive_bf16_dq"]()
        interface_result = routes["interface_bf16_dq"]()
        torch.cuda.synchronize()

        quality = {}
        for name, reference_component, actual_component in zip(
            ("dq", "dk", "dv"), reference, bf16_result
        ):
            quality[name] = component_metrics(reference_component, actual_component)
        direct_vs_fp32 = component_metrics(fp32_result[0], bf16_result[0])
        interface_match = [
            component_metrics(direct, interface)
            for direct, interface in zip(bf16_result, interface_result)
        ]
        timings = time_rotated(
            {
                "bf16_control": routes["bf16_control"],
                "adaptive_fp32_dq": routes["adaptive_fp32_dq"],
                "adaptive_bf16_dq": routes["adaptive_bf16_dq"],
            },
            args.warmups,
            args.samples,
        )
        result["shapes"][f"s{sequence}_h{heads}"] = {
            "dtypes": [str(component.dtype) for component in bf16_result],
            "quality_vs_bf16": quality,
            "direct_dq_vs_fp32_route": direct_vs_fp32,
            "interface_match": interface_match,
            "timings": timings,
        }
        del retained, reference, fp32_result, bf16_result, interface_result
        torch.cuda.empty_cache()

    repeat_retained, repeat_routes = build_problem(8192, 8, args.seed + 1000)
    repeat_reference = repeat_routes["adaptive_bf16_dq"]()
    repeat_max_abs = [0.0, 0.0, 0.0]
    repeat_all_finite = True
    for _ in range(args.repeat_calls):
        current = repeat_routes["adaptive_bf16_dq"]()
        for index, (baseline, actual) in enumerate(zip(repeat_reference, current)):
            repeat_all_finite &= bool(torch.isfinite(actual).all())
            repeat_max_abs[index] = max(
                repeat_max_abs[index],
                float((baseline.float() - actual.float()).abs().max()),
            )
    torch.cuda.synchronize()
    result["repeat_stability_s8192_h8"] = {
        "calls": args.repeat_calls,
        "all_finite": repeat_all_finite,
        "maximum_absolute_drift": repeat_max_abs,
    }
    del repeat_retained, repeat_reference

    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
