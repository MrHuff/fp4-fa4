#!/usr/bin/env python3
"""Measure compact dQ projection handoffs on the retained adaptive backward."""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from pathlib import Path

import torch

from tk_fa4 import (
    _C_b300_lowp_bwd as lowp,
    b300_adaptive_lowp_operands_from_projection,
    b300_mha_fwd,
    b300_mha_bwd_adaptive_lowp_nvfp4_projection_dgrad,
    b300_prepare_nvfp4_projection_operand,
    b300_project_nvfp4,
)
from tk_fa4.fp4_pv_experiments import (
    _mxfp4_bh_gemm,
    _quantize_rows_2d_mxfp4,
)
from tk_fa4.lowp_fa4_bwd.evaluate_frontier import _make_inputs


def _parse_shapes(value: str) -> list[tuple[int, int]]:
    result = []
    for item in value.split(","):
        sequence, heads = item.lower().split("x", maxsplit=1)
        result.append((int(sequence), int(heads)))
    return result


def _metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
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
        "norm_ratio": float(actual_norm / reference_norm),
        "max_abs": float(difference.abs().max()),
    }


def _time_rotated(
    candidates: dict[str, Callable[[], object]],
    *,
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


def _evaluate_shape(
    sequence: int,
    heads: int,
    seed: int,
    warmups: int,
    samples: int,
) -> dict[str, object]:
    device = torch.device("cuda")
    q, k, v, dout = _make_inputs(sequence, heads, seed, "calibrated", device)
    out, lse = b300_mha_fwd(q, k, v, causal=True, return_lse=True)
    softmax_scale = float(192**-0.5)
    adaptive = lowp.quantize_fp4_dual_qk_adaptive(
        q,
        k,
        16.0,
        2.0**-12,
        0.325,
        2.75,
        softmax_scale,
        4096.0,
    )
    adaptive_operands = b300_adaptive_lowp_operands_from_projection(
        q,
        k,
        *adaptive,
    )
    calibration = (
        lowp.backward_fp4_fp8dpdv_x32_split_dk_adaptive_bf16dq_native(
            q,
            k,
            v,
            out,
            lse,
            dout,
            *adaptive,
            4096.0,
            True,
            softmax_scale,
            False,
        )
    )
    reduction = heads * 192
    hidden = heads * 128
    dq = calibration[0].reshape(sequence, reduction).contiguous()
    torch.manual_seed(seed + 1000)
    weight = (torch.randn(hidden, reduction, device=device) * 0.02).bfloat16()
    reference = torch.mm(dq, weight.T)

    nvfp4_dq_full = b300_prepare_nvfp4_projection_operand(dq)
    torch.cuda.synchronize()
    dq_global_scale = nvfp4_dq_full[2]
    nvfp4_dq = b300_prepare_nvfp4_projection_operand(
        dq,
        global_scale=dq_global_scale,
    )
    nvfp4_weight = b300_prepare_nvfp4_projection_operand(weight)
    nvfp4_output = b300_project_nvfp4(nvfp4_dq, nvfp4_weight)
    public_compact_output = (
        b300_mha_bwd_adaptive_lowp_nvfp4_projection_dgrad(
            q,
            k,
            v,
            out,
            lse,
            dout,
            adaptive_operands,
            nvfp4_weight,
            dq_global_scale,
            causal=True,
            softmax_scale=softmax_scale,
            deterministic=False,
        )
    )
    torch.cuda.synchronize()

    mxfp4_dq = _quantize_rows_2d_mxfp4(dq, backend="mxfp4")
    mxfp4_weight = _quantize_rows_2d_mxfp4(weight, backend="mxfp4")
    mxfp4_output = _mxfp4_bh_gemm(
        mxfp4_dq["fp4"][None, None],
        mxfp4_dq["scales"][None, None],
        mxfp4_weight["fp4"][None, None],
        mxfp4_weight["scales"][None, None],
        name="dq_projection",
    ).view(sequence, hidden)

    dq_encode_scale = 448.0 / dq.float().abs().max().clamp_min(1.0e-12)
    weight_encode_scale = (
        448.0 / weight.float().abs().max().clamp_min(1.0e-12)
    )
    dq_fp8 = (dq.float() * dq_encode_scale).to(torch.float8_e4m3fn)
    weight_fp8 = (weight.float() * weight_encode_scale).to(
        torch.float8_e4m3fn
    )
    fp8_output = torch._scaled_mm(
        dq_fp8,
        weight_fp8.T,
        scale_a=dq_encode_scale.reciprocal(),
        scale_b=weight_encode_scale.reciprocal(),
        out_dtype=torch.bfloat16,
    )

    def attention_bf16_dq():
        return lowp.backward_fp4_fp8dpdv_x32_split_dk_adaptive_bf16dq_native(
            q, k, v, out, lse, dout, *adaptive,
            4096.0, True, softmax_scale, False,
        )

    def bf16_chain():
        gradients = attention_bf16_dq()
        projected = torch.mm(
            gradients[0].reshape(sequence, reduction),
            weight.T,
        ).view(1, sequence, hidden)
        return projected, gradients[1], gradients[2]

    def direct_bf16_chain():
        return (
            lowp.
            backward_fp4_fp8dpdv_x32_split_dk_adaptive_direct_dq_projection_native(
                q, k, v, out, lse, dout, *adaptive, weight,
                4096.0, True, softmax_scale, False,
            )
        )

    def compact_nvfp4_chain():
        return (
            lowp.
            backward_fp4_fp8dpdv_x32_split_dk_adaptive_nvfp4_dq_projection_native(
                q, k, v, out, lse, dout, *adaptive,
                *nvfp4_weight, dq_global_scale,
                4096.0, True, softmax_scale, False,
            )
        )

    def fp8_gemm():
        return torch._scaled_mm(
            dq_fp8,
            weight_fp8.T,
            scale_a=dq_encode_scale.reciprocal(),
            scale_b=weight_encode_scale.reciprocal(),
            out_dtype=torch.bfloat16,
        )

    def fp8_quant_gemm():
        scale = 448.0 / dq.float().abs().max().clamp_min(1.0e-12)
        packed = (dq.float() * scale).to(torch.float8_e4m3fn)
        return torch._scaled_mm(
            packed,
            weight_fp8.T,
            scale_a=scale.reciprocal(),
            scale_b=weight_encode_scale.reciprocal(),
            out_dtype=torch.bfloat16,
        )

    def nvfp4_gemm():
        return b300_project_nvfp4(nvfp4_dq, nvfp4_weight)

    def nvfp4_delayed_quant_gemm():
        operand = b300_prepare_nvfp4_projection_operand(
            dq,
            global_scale=dq_global_scale,
        )
        return b300_project_nvfp4(operand, nvfp4_weight)

    def nvfp4_full_quant_gemm():
        return b300_project_nvfp4(
            b300_prepare_nvfp4_projection_operand(dq),
            nvfp4_weight,
        )

    def mxfp4_gemm():
        return _mxfp4_bh_gemm(
            mxfp4_dq["fp4"][None, None],
            mxfp4_dq["scales"][None, None],
            mxfp4_weight["fp4"][None, None],
            mxfp4_weight["scales"][None, None],
            name="dq_projection",
        ).view(sequence, hidden)

    def mxfp4_quant_gemm():
        operand = _quantize_rows_2d_mxfp4(dq, backend="mxfp4")
        return _mxfp4_bh_gemm(
            operand["fp4"][None, None],
            operand["scales"][None, None],
            mxfp4_weight["fp4"][None, None],
            mxfp4_weight["scales"][None, None],
            name="dq_projection",
        ).view(sequence, hidden)

    projection_timing = _time_rotated(
        {
            "bf16_gemm": lambda: torch.mm(dq, weight.T),
            "fp8_gemm": fp8_gemm,
            "fp8_quant_gemm": fp8_quant_gemm,
            "nvfp4_gemm": nvfp4_gemm,
            "nvfp4_delayed_quant_gemm": nvfp4_delayed_quant_gemm,
            "nvfp4_full_quant_gemm": nvfp4_full_quant_gemm,
            "mxfp4_gemm": mxfp4_gemm,
            "mxfp4_quant_gemm": mxfp4_quant_gemm,
        },
        warmups=warmups,
        samples=samples,
    )
    chain_timing = _time_rotated(
        {
            "attention_bf16_dq": attention_bf16_dq,
            "bf16_dq_then_cublas": bf16_chain,
            "direct_bf16_projection": direct_bf16_chain,
            "compact_nvfp4_projection": compact_nvfp4_chain,
        },
        warmups=warmups,
        samples=samples,
    )
    chain_reference = bf16_chain()
    compact_chain = compact_nvfp4_chain()
    bf16_median = chain_timing["bf16_dq_then_cublas"]["median_ms"]
    nvfp4_median = chain_timing["compact_nvfp4_projection"]["median_ms"]
    return {
        "shape": {"batch": 1, "sequence": sequence, "heads": heads},
        "dq_statistics": {
            "amax": float(dq.float().abs().max()),
            "rms": float(dq.float().square().mean().sqrt()),
            "nvfp4_global_decode_scale": float(dq_global_scale),
        },
        "format_quality": {
            "fp8": _metrics(reference, fp8_output),
            "nvfp4": _metrics(reference, nvfp4_output),
            "mxfp4": _metrics(reference, mxfp4_output),
        },
        "projection_timing": projection_timing,
        "chain_timing": chain_timing,
        "compact_nvfp4_speedup_vs_bf16_chain": bf16_median / nvfp4_median,
        "compact_chain_quality": {
            "dx": _metrics(chain_reference[0], compact_chain[0]),
            "dk": _metrics(chain_reference[1], compact_chain[1]),
            "dv": _metrics(chain_reference[2], compact_chain[2]),
        },
        "public_api_smoke": {
            "all_finite": all(
                bool(torch.isfinite(tensor).all())
                for tensor in public_compact_output
            ),
            "dx": _metrics(chain_reference[0], public_compact_output[0]),
            "dk": _metrics(chain_reference[1], public_compact_output[1]),
            "dv": _metrics(chain_reference[2], public_compact_output[2]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", default="8192x8,4096x24")
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--warmups", type=int, default=7)
    parser.add_argument("--samples", type=int, default=31)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    torch.cuda.set_device(0)
    result = {
        "device": torch.cuda.get_device_name(0),
        "seed_base": args.seed,
        "warmups": args.warmups,
        "samples": args.samples,
        "results": {},
    }
    for sequence, heads in _parse_shapes(args.shapes):
        result["results"][f"S{sequence}H{heads}"] = _evaluate_shape(
            sequence,
            heads,
            args.seed + heads,
            args.warmups,
            args.samples,
        )
        torch.cuda.empty_cache()
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
