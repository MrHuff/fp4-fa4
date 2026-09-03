#!/usr/bin/env python3
"""Validate the exact two-command score for a model-native rank-128 Q/K."""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable

import torch

from tk_fa4 import _C_b300_lowp_bwd as lowp
from tk_fa4.interface import b300_mha_bwd, b300_mha_fwd


def metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    reference_f = reference.float()
    actual_f = actual.float()
    reference_norm = torch.linalg.vector_norm(reference_f).clamp_min(1.0e-20)
    actual_norm = torch.linalg.vector_norm(actual_f).clamp_min(1.0e-20)
    difference = actual_f - reference_f
    return {
        "cosine": float(
            torch.sum(reference_f * actual_f) / (reference_norm * actual_norm)
        ),
        "relative_l2": float(torch.linalg.vector_norm(difference) / reference_norm),
        "max_abs": float(difference.abs().max()),
        "reference_rms": float(reference_f.square().mean().sqrt()),
        "actual_rms": float(actual_f.square().mean().sqrt()),
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
    elapsed: dict[str, list[float]] = {name: [] for name in names}
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
    parser.add_argument("--sequence", type=int, default=8192)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=7)
    parser.add_argument("--samples", type=int, default=31)
    parser.add_argument("--seed", type=int, default=2026081302)
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("validation requires exactly one visible GPU")

    torch.manual_seed(args.seed)
    q = torch.zeros(
        1,
        args.sequence,
        args.heads,
        192,
        device="cuda",
        dtype=torch.bfloat16,
    )
    k = torch.zeros_like(q)
    q[..., :128] = (torch.randn_like(q[..., :128].float()) * 0.1).bfloat16()
    k[..., :128] = (torch.randn_like(k[..., :128].float()) * 0.1).bfloat16()
    v = (
        torch.randn(1, args.sequence, args.heads, 128, device="cuda") * 0.1
    ).bfloat16()
    dout = (torch.randn_like(v.float()) * 0.1).bfloat16()
    out, lse = b300_mha_fwd(q, k, v, causal=True, return_lse=True)
    scale = float(192**-0.5)
    packed_qk = tuple(lowp.quantize_fp4_dual_qk_blockscale(q, k, 16.0, 16.0))
    producer = tuple(lowp.prepare_mxfp4_backward_operands(out, dout, v, lse))

    def full_d192() -> tuple[torch.Tensor, ...]:
        return tuple(
            lowp.backward_fp4_mxfp4dpdvdsdqdk_producer_native_x32(
                q,
                k,
                v,
                out,
                lse,
                dout,
                *packed_qk,
                *producer,
                16.0,
                16.0,
                4096.0,
                True,
                scale,
                False,
            )
        )

    def rank128() -> tuple[torch.Tensor, ...]:
        return tuple(
            lowp.backward_fp4_rank128_mxfp4dpdvdsdqdk_producer_native_x32(
                q,
                k,
                v,
                out,
                lse,
                dout,
                *packed_qk,
                *producer,
                16.0,
                16.0,
                4096.0,
                True,
                scale,
                False,
            )
        )

    reference = tuple(
        b300_mha_bwd(
            q,
            k,
            v,
            out,
            lse,
            dout,
            causal=True,
        )
    )
    full_output = full_d192()
    rank_output = rank128()
    torch.cuda.synchronize()
    names = ("dq", "dk", "dv")
    rank_vs_full = {
        name: metrics(full, rank)
        for name, full, rank in zip(names, full_output, rank_output, strict=True)
    }
    full_vs_bf16 = {
        name: metrics(ref, full)
        for name, ref, full in zip(names, reference, full_output, strict=True)
    }
    rank_vs_bf16 = {
        name: metrics(ref, rank)
        for name, ref, rank in zip(names, reference, rank_output, strict=True)
    }
    timing = time_rotated(
        {"full_d192_score": full_d192, "model_rank128_score": rank128},
        args.warmups,
        args.samples,
    )
    full_ms = timing["full_d192_score"]["median_ms"]
    rank_ms = timing["model_rank128_score"]["median_ms"]
    result = {
        "shape": {"sequence": args.sequence, "heads": args.heads},
        "rank128_vs_full_d192": rank_vs_full,
        "full_d192_vs_bf16": full_vs_bf16,
        "rank128_vs_bf16": rank_vs_bf16,
        "timing": timing,
        "saved_ms": full_ms - rank_ms,
        "speedup": full_ms / rank_ms,
        "producer_scale_ranges": {
            name: {
                "minimum": int(tensor.min()),
                "maximum": int(tensor.max()),
                "zero_count": int((tensor == 0).sum()),
            }
            for name, tensor in zip(
                ("dout_dp", "v_dp", "dout_dv"),
                (producer[2], producer[3], producer[5]),
                strict=True,
            )
        },
        "passed": bool(
            all(item["cosine"] > 0.99999 for item in rank_vs_full.values())
            and rank_ms < full_ms
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
