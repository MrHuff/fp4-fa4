#!/usr/bin/env python3
"""Validate and benchmark projection-ready adaptive attention backward."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Callable

import torch

from tk_fa4 import _C_b300_lowp_bwd as lowp
from tk_fa4.interface import (
    b300_adaptive_lowp_operands_from_projection,
    b300_interleave_qkv_projection_weights,
    b300_mha_bwd_adaptive_lowp_projection_dgrad,
    b300_mha_fwd,
)


def parse_shapes(value: str) -> list[tuple[int, int]]:
    shapes = []
    for item in value.split(","):
        sequence, heads = item.lower().split("x", maxsplit=1)
        shapes.append((int(sequence), int(heads)))
    return shapes


def metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float | bool]:
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
    callables: dict[str, Callable[[], object]],
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


def time_paired_chains(
    chains: dict[
        str,
        tuple[Callable[[], object], Callable[[object], torch.Tensor]],
    ],
    warmups: int,
    samples: int,
) -> dict[str, dict[str, float]]:
    names = list(chains)
    for iteration in range(warmups):
        order = names[iteration % len(names) :] + names[: iteration % len(names)]
        for name in order:
            attention, projection = chains[name]
            projection(attention())
    torch.cuda.synchronize()

    raw = {
        name: {"attention_ms": [], "projection_ms": [], "total_ms": []}
        for name in names
    }
    for iteration in range(samples):
        order = names[iteration % len(names) :] + names[: iteration % len(names)]
        for name in order:
            attention, projection = chains[name]
            start = torch.cuda.Event(enable_timing=True)
            boundary = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            gradients = attention()
            boundary.record()
            dx = projection(gradients)
            end.record()
            end.synchronize()
            raw[name]["attention_ms"].append(float(start.elapsed_time(boundary)))
            raw[name]["projection_ms"].append(float(boundary.elapsed_time(end)))
            raw[name]["total_ms"].append(float(start.elapsed_time(end)))
            del gradients, dx
    return {
        name: {
            f"{component}_{statistic}": reducer(values)
            for component, values in components.items()
            for statistic, reducer in (
                ("median", statistics.median),
                ("mean", statistics.fmean),
                ("minimum", min),
            )
        }
        for name, components in raw.items()
    }


def _project_three_from_fresh(
    gradients: object,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    sequence: int,
    heads: int,
) -> torch.Tensor:
    dq, dk, dv = gradients
    dx = torch.mm(dq.reshape(sequence, heads * 192), q_weight)
    dx.addmm_(dk.reshape(sequence, heads * 192), k_weight)
    dx.addmm_(dv.reshape(sequence, heads * 128), v_weight)
    return dx


def _project_interleaved_from_fresh(
    gradients: object,
    weight: torch.Tensor,
    sequence: int,
    heads: int,
    splits: int,
) -> torch.Tensor:
    projection_input = gradients[0].reshape(sequence, heads * 512)
    split_width = heads * 512 // splits
    dx = torch.mm(
        projection_input[:, :split_width],
        weight[:split_width],
    )
    for split in range(1, splits):
        start = split * split_width
        end = start + split_width
        dx.addmm_(projection_input[:, start:end], weight[start:end])
    return dx


def build_problem(sequence: int, heads: int, seed: int):
    torch.manual_seed(seed)
    batch = 1
    hidden = heads * 128
    q = (torch.randn(batch, sequence, heads, 192, device="cuda") * 0.1).bfloat16()
    k = (torch.randn_like(q.float()) * 0.1).bfloat16()
    v = (
        torch.randn(batch, sequence, heads, 128, device="cuda") * 0.1
    ).bfloat16()
    dout = (torch.randn_like(v.float()) * 0.1).bfloat16()
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

    q_weight = (
        torch.randn(heads * 192, hidden, device="cuda") * 0.02
    ).bfloat16()
    k_weight = (
        torch.randn(heads * 192, hidden, device="cuda") * 0.02
    ).bfloat16()
    v_weight = (
        torch.randn(heads * 128, hidden, device="cuda") * 0.02
    ).bfloat16()
    interleaved_weight = b300_interleave_qkv_projection_weights(
        q_weight,
        k_weight,
        v_weight,
    )
    projection_splits = (
        4
        if heads == 64
        else 2
        if heads == 16 or (heads == 24 and sequence >= 8192)
        else 1
    )

    def attention_three():
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

    def attention_interleaved():
        backward = getattr(
            lowp,
            "backward_fp4_fp8dpdv_x32_split_dk_adaptive_qkv_bf16_native",
        )
        return backward(
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

    reference_grads = attention_three()
    (reference_interleaved,) = attention_interleaved()

    def projection_three():
        dq, dk, dv = reference_grads
        dx = torch.mm(dq.reshape(batch * sequence, heads * 192), q_weight)
        dx.addmm_(dk.reshape(batch * sequence, heads * 192), k_weight)
        dx.addmm_(dv.reshape(batch * sequence, heads * 128), v_weight)
        return dx

    def projection_interleaved():
        return _project_interleaved_from_fresh(
            (reference_interleaved,),
            interleaved_weight,
            sequence,
            heads,
            projection_splits,
        )

    def chain_three():
        dq, dk, dv = attention_three()
        dx = torch.mm(dq.reshape(batch * sequence, heads * 192), q_weight)
        dx.addmm_(dk.reshape(batch * sequence, heads * 192), k_weight)
        dx.addmm_(dv.reshape(batch * sequence, heads * 128), v_weight)
        return dx

    def chain_one():
        return b300_mha_bwd_adaptive_lowp_projection_dgrad(
            q,
            k,
            v,
            out,
            lse,
            dout,
            operands,
            interleaved_weight,
            causal=True,
            softmax_scale=softmax_scale,
        )

    return {
        "retained": (
            q,
            k,
            v,
            out,
            lse,
            dout,
            packed,
            operands,
            q_weight,
            k_weight,
            v_weight,
            interleaved_weight,
        ),
        "hidden": hidden,
        "projection_splits": projection_splits,
        "reference_grads": reference_grads,
        "reference_interleaved": reference_interleaved,
        "attention_three": attention_three,
        "attention_interleaved": attention_interleaved,
        "projection_three": projection_three,
        "projection_interleaved": projection_interleaved,
        "chain_three": chain_three,
        "chain_one": chain_one,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", default="4096x24,8192x8")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--repeat-calls", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026081203)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("validation requires exactly one visible GPU")

    result: dict[str, object] = {
        "configuration": vars(args)
        | {"output": None if args.output is None else str(args.output)},
        "shapes": {},
    }
    for shape_index, (sequence, heads) in enumerate(parse_shapes(args.shapes)):
        problem = build_problem(sequence, heads, args.seed + shape_index)
        reference_grads = problem["reference_grads"]
        reference_interleaved = problem["reference_interleaved"]
        slices = (
            reference_interleaved[..., :192],
            reference_interleaved[..., 192:384],
            reference_interleaved[..., 384:],
        )
        quality = {
            name: metrics(reference, actual)
            for name, reference, actual in zip(
                ("dq", "dk", "dv"), reference_grads, slices
            )
        }
        projection_quality = metrics(
            problem["projection_three"](),
            problem["projection_interleaved"](),
        )
        timings = time_rotated(
            {
                "attention_three_outputs": problem["attention_three"],
                "attention_interleaved": problem["attention_interleaved"],
                "projection_three_gemms": problem["projection_three"],
                "projection_interleaved": problem["projection_interleaved"],
            },
            args.warmups,
            args.samples,
        )
        paired_chains = time_paired_chains(
            {
                "three_outputs_three_gemms": (
                    problem["attention_three"],
                    lambda gradients: _project_three_from_fresh(
                        gradients,
                        problem["retained"][8],
                        problem["retained"][9],
                        problem["retained"][10],
                        sequence,
                        heads,
                    ),
                ),
                "interleaved_projection": (
                    problem["attention_interleaved"],
                    lambda gradients: _project_interleaved_from_fresh(
                        gradients,
                        problem["retained"][11],
                        sequence,
                        heads,
                        problem["projection_splits"],
                    ),
                ),
            },
            args.warmups,
            args.samples,
        )
        repeat_reference = problem["attention_interleaved"]()[0]
        repeat_max_abs = 0.0
        repeat_finite = True
        for _ in range(args.repeat_calls):
            repeat = problem["attention_interleaved"]()[0]
            repeat_finite = repeat_finite and bool(torch.isfinite(repeat).all())
            repeat_max_abs = max(
                repeat_max_abs,
                float((repeat.float() - repeat_reference.float()).abs().max()),
            )
        torch.cuda.synchronize()

        old_chain = paired_chains["three_outputs_three_gemms"][
            "total_ms_median"
        ]
        new_chain = paired_chains["interleaved_projection"]["total_ms_median"]
        result["shapes"][f"s{sequence}_h{heads}"] = {
            "hidden": problem["hidden"],
            "projection_splits": problem["projection_splits"],
            "interleaved_shape": list(reference_interleaved.shape),
            "quality_vs_three_outputs": quality,
            "projection_quality_interleaved_vs_three": projection_quality,
            "repeat": {
                "finite": repeat_finite,
                "max_abs": repeat_max_abs,
            },
            "timings": timings,
            "paired_chain_timings": paired_chains,
            "chain_speedup": old_chain / new_chain,
            "chain_saved_ms": old_chain - new_chain,
        }
        del problem, reference_grads, reference_interleaved, slices
        torch.cuda.empty_cache()

    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
