#!/usr/bin/env python3
"""Evaluate the retained low-precision backward speed/quality frontier.

This intentionally varies the activation regime as well as the random seed.
The historical 0.1-std benchmark is useful for timing, but on its own it can
hide score approximations that fail once attention logits have normal scale.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import torch

from tk_fa4 import _C_b300_lowp_bwd as lowp_ext
from tk_fa4.interface import (
    b300_adaptive_lowp_operands_from_projection,
    b300_mha_fwd,
    b300_mha_fwd_with_adaptive_lowp,
    b300_mha_fwd_with_mixed_v,
)


SCENARIOS = {
    "calibrated": ("normal", 0.1, 0.1, 0.1),
    "qk_medium": ("normal", 0.25, 0.1, 0.1),
    "qk_unit": ("normal", 1.0, 0.1, 0.1),
    "qk_uniform": ("uniform", 0.25, 0.1, 0.1),
    "qk_outliers": ("outliers", 0.25, 0.1, 0.1),
    "value_unit": ("normal", 0.1, 1.0, 0.1),
    "dout_unit": ("normal", 0.1, 0.1, 1.0),
}


def _parse_csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def _parse_shapes(value: str) -> list[tuple[int, int]]:
    result = []
    for item in value.split(","):
        sequence, heads = item.lower().split("x", maxsplit=1)
        result.append((int(sequence), int(heads)))
    return result


def _standard_sample(
    shape: tuple[int, ...],
    distribution: str,
    device: torch.device,
) -> torch.Tensor:
    if distribution == "normal":
        return torch.randn(shape, device=device)
    if distribution == "uniform":
        return (2.0 * torch.rand(shape, device=device) - 1.0) * math.sqrt(3.0)
    if distribution == "outliers":
        values = torch.randn(shape, device=device)
        multipliers = torch.where(
            torch.rand(shape, device=device) < 0.01,
            torch.full((), 8.0, device=device),
            torch.ones((), device=device),
        )
        # Preserve unit RMS while changing kurtosis.
        return values * multipliers / math.sqrt(0.99 + 0.01 * 64.0)
    raise ValueError(f"unknown distribution {distribution}")


def _make_inputs(
    sequence: int,
    heads: int,
    seed: int,
    scenario: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    distribution, qk_scale, value_scale, dout_scale = SCENARIOS[scenario]
    torch.manual_seed(seed)
    qk_shape = (1, sequence, heads, 192)
    value_shape = (1, sequence, heads, 128)
    q = (_standard_sample(qk_shape, distribution, device) * qk_scale).bfloat16()
    k = (_standard_sample(qk_shape, distribution, device) * qk_scale).bfloat16()
    v = (torch.randn(value_shape, device=device) * value_scale).bfloat16()
    dout = (torch.randn(value_shape, device=device) * dout_scale).bfloat16()
    return q, k, v, dout


def _time_rotated(
    callables: dict[str, object], warmup: int, iterations: int
) -> dict[str, list[float]]:
    names = list(callables)
    for iteration in range(warmup):
        order = names[iteration % len(names) :] + names[: iteration % len(names)]
        for name in order:
            callables[name]()
    torch.cuda.synchronize()
    samples = {name: [] for name in names}
    for iteration in range(iterations):
        order = names[iteration % len(names) :] + names[: iteration % len(names)]
        for name in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            callables[name]()
            end.record()
            end.synchronize()
            samples[name].append(float(start.elapsed_time(end)))
    return samples


def _component_metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float | bool]:
    reference_f = reference.float()
    actual_f = actual.float()
    difference = actual_f - reference_f
    reference_norm = torch.linalg.vector_norm(reference_f).clamp_min(1e-20)
    actual_norm = torch.linalg.vector_norm(actual_f)
    cosine = torch.sum(reference_f * actual_f) / (reference_norm * actual_norm).clamp_min(1e-20)

    reference_rows = reference_f.reshape(-1, reference_f.shape[-1])
    actual_rows = actual_f.reshape(-1, actual_f.shape[-1])
    row_dot = torch.sum(reference_rows * actual_rows, dim=1)
    row_denominator = (
        torch.linalg.vector_norm(reference_rows, dim=1)
        * torch.linalg.vector_norm(actual_rows, dim=1)
    ).clamp_min(1e-20)
    row_cosine = row_dot / row_denominator
    quantiles = torch.quantile(
        row_cosine,
        torch.tensor((0.01, 0.05, 0.5), device=row_cosine.device),
    )
    result = {
        "finite": bool(torch.isfinite(actual_f).all()),
        "cosine": float(cosine),
        "relative_l2": float(torch.linalg.vector_norm(difference) / reference_norm),
        "norm_ratio": float(actual_norm / reference_norm),
        "max_abs": float(difference.abs().max()),
        "sign_agreement": float((torch.signbit(reference_f) == torch.signbit(actual_f)).float().mean()),
        "row_cosine_p01": float(quantiles[0]),
        "row_cosine_p05": float(quantiles[1]),
        "row_cosine_p50": float(quantiles[2]),
    }
    del reference_f, actual_f, difference, reference_rows, actual_rows
    return result


def _quality_metrics(
    reference: tuple[torch.Tensor, ...],
    actual: tuple[torch.Tensor, ...],
) -> dict[str, object]:
    result = {}
    aggregate_dot = torch.zeros((), device=reference[0].device)
    aggregate_reference_sq = torch.zeros_like(aggregate_dot)
    aggregate_actual_sq = torch.zeros_like(aggregate_dot)
    for name, reference_component, actual_component in zip(
        ("dq", "dk", "dv"), reference, actual
    ):
        result[name] = _component_metrics(reference_component, actual_component)
        reference_f = reference_component.float()
        actual_f = actual_component.float()
        aggregate_dot += torch.sum(reference_f * actual_f)
        aggregate_reference_sq += torch.sum(reference_f.square())
        aggregate_actual_sq += torch.sum(actual_f.square())
    result["aggregate_cosine"] = float(
        aggregate_dot
        / torch.sqrt(aggregate_reference_sq * aggregate_actual_sq).clamp_min(1e-20)
    )
    return result


def _build_problem(
    sequence: int,
    heads: int,
    seed: int,
    scenario: str,
    device: torch.device,
):
    q, k, v, dout = _make_inputs(sequence, heads, seed, scenario, device)
    out, lse = b300_mha_fwd(q, k, v, causal=True, return_lse=True)
    scale = float(192**-0.5)
    q_fp8 = (q.float() * 256.0).to(torch.float8_e4m3fn).permute(0, 2, 3, 1).contiguous()
    k_fp8 = (k.float() * 256.0).to(torch.float8_e4m3fn).contiguous()
    qk_lowp = lowp_ext.quantize_fp4_dual_qk_blockscale(q, k, 16.0, 16.0)
    q_fp4, score_q_fp4, k_fp4, score_k_fp4 = qk_lowp[:4]
    adaptive_lowp = lowp_ext.quantize_fp4_dual_qk_adaptive(
        q, k, 16.0, 2.0**-12, 0.325, 2.75, scale, 4096.0
    )
    (
        adaptive_q_fp4,
        adaptive_score_q_fp4,
        adaptive_k_fp4,
        adaptive_score_k_fp4,
        adaptive_qk_scales,
    ) = adaptive_lowp
    mixed_v = lowp_ext.prepack_mixed_v(v)
    producer_mxfp4 = lowp_ext.prepare_mxfp4_backward_operands(
        out, dout, v, lse
    )

    def bf16():
        return lowp_ext.backward_bf16_control(
            q, k, v, out, lse, dout, True, scale, False
        )

    def fp8():
        return lowp_ext.backward_fp8_native(
            q, k, v, out, lse, dout, q_fp8, k_fp8,
            256.0, 256.0, 4096.0, True, scale, False,
        )

    def winner():
        return lowp_ext.backward_fp4_fp8dpdv_x32_split_dk_native(
            q, k, v, out, lse, dout,
            q_fp4, score_q_fp4, k_fp4, score_k_fp4,
            16.0, 16.0, 4096.0, True, scale, False,
        )

    def fp4_bf16_dpdv():
        return lowp_ext.backward_fp4_native(
            q, k, v, out, lse, dout,
            q_fp4, score_q_fp4, k_fp4, score_k_fp4,
            16.0, 16.0, 4096.0, True, scale, False,
        )

    def mixed():
        return lowp_ext.backward_fp4_mixed_fp8_mxfp4dp_fp8dv_x32_split_dk_prepacked_v_native(
            q, k, v, out, lse, dout,
            q_fp4, score_q_fp4, k_fp4, score_k_fp4,
            16.0, 16.0, 4096.0, True, scale, False, mixed_v,
        )

    def adaptive_winner():
        return lowp_ext.backward_fp4_fp8dpdv_x32_split_dk_adaptive_native(
            q, k, v, out, lse, dout,
            adaptive_q_fp4, adaptive_score_q_fp4,
            adaptive_k_fp4, adaptive_score_k_fp4,
            adaptive_qk_scales,
            4096.0, True, scale, False,
        )

    def adaptive_mixed():
        return lowp_ext.backward_fp4_mixed_fp8_mxfp4dp_fp8dv_x32_split_dk_adaptive_prepacked_v_native(
            q, k, v, out, lse, dout,
            adaptive_q_fp4, adaptive_score_q_fp4,
            adaptive_k_fp4, adaptive_score_k_fp4,
            adaptive_qk_scales,
            4096.0, True, scale, False, mixed_v,
        )

    def pure_mxfp4():
        return lowp_ext.backward_fp4_mxfp4dpdvdsdqdk_producer_native_x32(
            q,
            k,
            v,
            out,
            lse,
            dout,
            *qk_lowp,
            *producer_mxfp4,
            16.0,
            16.0,
            4096.0,
            True,
            scale,
            False,
        )

    routes = {
        "bf16": bf16,
        "fp4_fp8": winner,
        "mixed_prepacked": mixed,
        "adaptive_fp4_fp8": adaptive_winner,
        "adaptive_mixed_prepacked": adaptive_mixed,
        "pure_mxfp4": pure_mxfp4,
    }
    if sequence == 8192 and heads in (8, 16):
        routes["fp4_bf16_dpdv"] = fp4_bf16_dpdv
        routes["fp8"] = fp8
    return (q, k, v, dout), routes


def _end_to_end_callables(inputs: tuple[torch.Tensor, ...]):
    q, k, v, dout = inputs
    scale = float(192**-0.5)
    projection_packed = lowp_ext.quantize_fp4_dual_qk_adaptive(
        q,
        k,
        16.0,
        2.0**-12,
        0.325,
        2.75,
        scale,
        4096.0,
    )
    projection_operands = b300_adaptive_lowp_operands_from_projection(
        q,
        k,
        *projection_packed,
    )

    def bf16():
        out, lse = b300_mha_fwd(q, k, v, causal=True, return_lse=True)
        return lowp_ext.backward_bf16_control(
            q, k, v, out, lse, dout, True, scale, False
        )

    def winner():
        out, lse = b300_mha_fwd(q, k, v, causal=True, return_lse=True)
        q_fp4, score_q_fp4, k_fp4, score_k_fp4, *_ = (
            lowp_ext.quantize_fp4_dual_qk_blockscale(q, k, 16.0, 16.0)
        )
        return lowp_ext.backward_fp4_fp8dpdv_x32_split_dk_native(
            q, k, v, out, lse, dout,
            q_fp4, score_q_fp4, k_fp4, score_k_fp4,
            16.0, 16.0, 4096.0, True, scale, False,
        )

    def mixed():
        out, lse, mixed_v = b300_mha_fwd_with_mixed_v(
            q, k, v, causal=True, return_lse=True
        )
        q_fp4, score_q_fp4, k_fp4, score_k_fp4, *_ = (
            lowp_ext.quantize_fp4_dual_qk_blockscale(q, k, 16.0, 16.0)
        )
        return lowp_ext.backward_fp4_mixed_fp8_mxfp4dp_fp8dv_x32_split_dk_prepacked_v_native(
            q, k, v, out, lse, dout,
            q_fp4, score_q_fp4, k_fp4, score_k_fp4,
            16.0, 16.0, 4096.0, True, scale, False, mixed_v,
        )

    def adaptive_winner():
        out, lse, operands = b300_mha_fwd_with_adaptive_lowp(
            q,
            k,
            v,
            causal=True,
            return_lse=True,
            prepare_mixed_v=False,
        )
        return lowp_ext.backward_fp4_fp8dpdv_x32_split_dk_adaptive_native(
            q, k, v, out, lse, dout,
            operands.q_fp4, operands.score_q_fp4,
            operands.k_fp4, operands.score_k_fp4, operands.qk_scales,
            4096.0, True, scale, False,
        )

    def adaptive_projection_fused():
        out, lse, operands = b300_mha_fwd_with_adaptive_lowp(
            q,
            k,
            v,
            causal=True,
            return_lse=True,
            prepare_mixed_v=False,
            prepared_operands=projection_operands,
        )
        return lowp_ext.backward_fp4_fp8dpdv_x32_split_dk_adaptive_native(
            q, k, v, out, lse, dout,
            operands.q_fp4, operands.score_q_fp4,
            operands.k_fp4, operands.score_k_fp4, operands.qk_scales,
            4096.0, True, scale, False,
        )

    def adaptive_mixed_projection_fused():
        out, lse, operands = b300_mha_fwd_with_adaptive_lowp(
            q,
            k,
            v,
            causal=True,
            return_lse=True,
            prepare_mixed_v=True,
            prepared_operands=projection_operands,
        )
        return lowp_ext.backward_fp4_mixed_fp8_mxfp4dp_fp8dv_x32_split_dk_adaptive_prepacked_v_native(
            q, k, v, out, lse, dout,
            operands.q_fp4, operands.score_q_fp4,
            operands.k_fp4, operands.score_k_fp4, operands.qk_scales,
            4096.0, True, scale, False, operands.mixed_v,
        )

    return {
        "bf16": bf16,
        "fp4_fp8": winner,
        "mixed_prepacked": mixed,
        "adaptive_fp4_fp8": adaptive_winner,
        "adaptive_projection_fused": adaptive_projection_fused,
        "adaptive_mixed_projection_fused": adaptive_mixed_projection_fused,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", default="4096x24,8192x8")
    parser.add_argument("--seeds", default="2026081201,2026081202,2026081203")
    parser.add_argument(
        "--scenarios",
        default="calibrated,qk_medium,qk_unit,qk_outliers,value_unit,dout_unit",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=9)
    parser.add_argument("--end-to-end", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise RuntimeError("evaluation requires exactly one visible GPU")
    scenarios = [item for item in args.scenarios.split(",") if item]
    unknown = sorted(set(scenarios) - set(SCENARIOS))
    if unknown:
        raise ValueError(f"unknown scenarios: {unknown}")

    device = torch.device("cuda")
    records = []
    for sequence, heads in _parse_shapes(args.shapes):
        for seed in _parse_csv_ints(args.seeds):
            for scenario in scenarios:
                inputs, routes = _build_problem(
                    sequence, heads, seed, scenario, device
                )
                reference = tuple(output.clone() for output in routes["bf16"]())
                torch.cuda.synchronize()
                actual = {}
                for name, route in routes.items():
                    if name == "bf16":
                        continue
                    actual[name] = tuple(output.clone() for output in route())
                    # Some native routes reuse internal output storage.  Make
                    # an owned snapshot and finish it before another route can
                    # overwrite that storage.
                    torch.cuda.synchronize()
                quality = {
                    name: _quality_metrics(reference, output)
                    for name, output in actual.items()
                }
                timing_samples = _time_rotated(
                    routes, args.warmup, args.iterations
                )
                timing = {
                    name: {
                        "median_ms": statistics.median(samples),
                        "min_ms": min(samples),
                        "max_ms": max(samples),
                    }
                    for name, samples in timing_samples.items()
                }
                record = {
                    "sequence": sequence,
                    "heads": heads,
                    "seed": seed,
                    "scenario": scenario,
                    "timing": timing,
                    "quality": quality,
                }
                if args.end_to_end:
                    e2e_samples = _time_rotated(
                        _end_to_end_callables(inputs),
                        args.warmup,
                        args.iterations,
                    )
                    record["end_to_end"] = {
                        name: {
                            "median_ms": statistics.median(samples),
                            "min_ms": min(samples),
                            "max_ms": max(samples),
                        }
                        for name, samples in e2e_samples.items()
                    }
                records.append(record)
                fp8_timing = (
                    f"fp8={timing['fp8']['median_ms']:.6f} "
                    if "fp8" in timing
                    else ""
                )
                fp4_bf16_timing = (
                    f"fp4+bf16_dpdv={timing['fp4_bf16_dpdv']['median_ms']:.6f} "
                    if "fp4_bf16_dpdv" in timing
                    else ""
                )
                fp8_quality = (
                    f"fp8={quality['fp8']['aggregate_cosine']:.6f} "
                    if "fp8" in quality
                    else ""
                )
                fp4_bf16_quality = (
                    "fp4+bf16_dpdv="
                    f"{quality['fp4_bf16_dpdv']['aggregate_cosine']:.6f} "
                    if "fp4_bf16_dpdv" in quality
                    else ""
                )
                print(
                    f"S{sequence} H{heads} seed={seed} {scenario}: "
                    f"bf16={timing['bf16']['median_ms']:.6f} "
                    f"{fp8_timing}"
                    f"{fp4_bf16_timing}"
                    f"fp4+fp8={timing['fp4_fp8']['median_ms']:.6f} "
                    f"mixed={timing['mixed_prepacked']['median_ms']:.6f} ms; "
                    f"adaptive_fp4+fp8={timing['adaptive_fp4_fp8']['median_ms']:.6f} "
                    f"adaptive_mixed={timing['adaptive_mixed_prepacked']['median_ms']:.6f} "
                    f"pure_mxfp4={timing['pure_mxfp4']['median_ms']:.6f} ms; "
                    f"aggregate cosine {fp8_quality}"
                    f"{fp4_bf16_quality}"
                    f"fp4+fp8={quality['fp4_fp8']['aggregate_cosine']:.6f} "
                    f"mixed={quality['mixed_prepacked']['aggregate_cosine']:.6f} "
                    f"adaptive_fp4+fp8={quality['adaptive_fp4_fp8']['aggregate_cosine']:.6f} "
                    f"adaptive_mixed={quality['adaptive_mixed_prepacked']['aggregate_cosine']:.6f} "
                    f"pure_mxfp4={quality['pure_mxfp4']['aggregate_cosine']:.6f} "
                    "(dQ/dK/dV "
                    f"{quality['pure_mxfp4']['dq']['cosine']:.6f}/"
                    f"{quality['pure_mxfp4']['dk']['cosine']:.6f}/"
                    f"{quality['pure_mxfp4']['dv']['cosine']:.6f})"
                )
                del reference, actual, routes, inputs
                torch.cuda.empty_cache()

    payload = {
        "config": {
            "shapes": _parse_shapes(args.shapes),
            "seeds": _parse_csv_ints(args.seeds),
            "scenarios": scenarios,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "end_to_end": args.end_to_end,
        },
        "records": records,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
