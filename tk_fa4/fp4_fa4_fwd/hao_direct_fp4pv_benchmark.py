#!/usr/bin/env python3
"""Apples-to-apples benchmark for the isolated HAO-structured TK port."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
EXTENSIONS = {
    ("nvfp4", "nvfp4"): (
        Path("/tmp/_C_tk_hao_direct_fp4pv.cpython-312-aarch64-linux-gnu.so"),
        "_C_tk_hao_direct_fp4pv",
    ),
    ("nvfp4", "mxfp4"): (
        Path(
            "/tmp/_C_tk_hao_direct_mxfp4pv"
            ".cpython-312-aarch64-linux-gnu.so"
        ),
        "_C_tk_hao_direct_mxfp4pv",
    ),
    ("mxfp4", "nvfp4"): (
        Path(
            "/tmp/_C_tk_hao_direct_mxqk_nvfp4pv"
            ".cpython-312-aarch64-linux-gnu.so"
        ),
        "_C_tk_hao_direct_mxqk_nvfp4pv",
    ),
    ("mxfp4", "mxfp4"): (
        Path(
            "/tmp/_C_tk_hao_direct_mxqk_mxfp4pv"
            ".cpython-312-aarch64-linux-gnu.so"
        ),
        "_C_tk_hao_direct_mxqk_mxfp4pv",
    ),
}
DEFAULT_ADAPTER = (
    REPO_ROOT
    / "results"
    / "mxfp4_fa4_upstream_strategy_20260722"
    / "active_b1s4096h24"
    / "active_b1s4096h24_upstream_hao_native_tk_port_adapter.py"
)
ROUTES = {
    ("nvfp4", "nvfp4"): (
        "real_fwd_tk_hao_direct_nvfp4_nvfp4pv"
    ),
    ("nvfp4", "mxfp4"): (
        "real_fwd_tk_hao_direct_nvfp4_mxfp4pv"
    ),
    ("mxfp4", "nvfp4"): (
        "real_fwd_tk_hao_direct_mxfp4_nvfp4pv"
    ),
    ("mxfp4", "mxfp4"): (
        "real_fwd_tk_hao_direct_mxfp4_mxfp4pv"
    ),
}


def load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def comparison(a: Any, b: Any) -> dict[str, float | int]:
    import torch

    a32 = a.float()
    b32 = b.float()
    actual_nonfinite = int((~torch.isfinite(a32)).sum().item())
    reference_nonfinite = int((~torch.isfinite(b32)).sum().item())
    delta = a32 - b32
    error_rms = delta.square().mean().sqrt()
    reference_rms = b32.square().mean().sqrt()
    return {
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                a32.flatten().unsqueeze(0),
                b32.flatten().unsqueeze(0),
            ).item()
        ),
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().mean().item()),
        "rmse": float(error_rms.item()),
        "reference_rms": float(reference_rms.item()),
        "relative_l2": float((error_rms / reference_rms).item()),
        "actual_nonfinite": actual_nonfinite,
        "reference_nonfinite": reference_nonfinite,
    }


def denominator_analysis(
    actual: Any,
    reference: Any,
    *,
    actual_lse: Any | None = None,
    reference_lse: Any | None = None,
    query_tile_rows: int = 128,
) -> dict[str, Any]:
    """Measure the accuracy ceiling of denominator-only row rescaling."""
    import torch

    if actual.shape != reference.shape or actual.ndim != 4:
        raise ValueError(
            "denominator analysis requires matching BSHD outputs"
        )
    actual32 = actual.float()
    reference32 = reference.float()
    batch, seqlen, heads, dim = actual32.shape

    def scale_stats(scale: Any) -> dict[str, float]:
        return {
            "min": float(scale.min().item()),
            "mean": float(scale.mean().item()),
            "max": float(scale.max().item()),
            "std": float(scale.std().item()),
        }

    def scaled_result(scale: Any) -> dict[str, Any]:
        expanded = scale.unsqueeze(-1)
        return {
            "scale": scale_stats(scale),
            "error": comparison(actual32 * expanded, reference32),
        }

    # A denominator changes one scalar per output row. This positive
    # least-squares scale is therefore the best result any denominator-only
    # correction can produce without changing the output direction.
    row_numerator = (actual32 * reference32).sum(dim=-1)
    row_denominator = actual32.square().sum(dim=-1).clamp_min_(1.0e-20)
    row_scale = (row_numerator / row_denominator).clamp_min_(0.0)

    global_scale = (
        (actual32 * reference32).sum()
        / actual32.square().sum().clamp_min_(1.0e-20)
    ).reshape(1, 1, 1).expand(batch, seqlen, heads)

    head_numerator = (actual32 * reference32).sum(dim=(1, 3))
    head_denominator = actual32.square().sum(dim=(1, 3)).clamp_min_(1.0e-20)
    head_scale = (head_numerator / head_denominator).clamp_min_(0.0)
    head_scale = head_scale[:, None, :].expand(batch, seqlen, heads)

    if seqlen % query_tile_rows != 0:
        raise ValueError(
            f"seqlen {seqlen} is not divisible by {query_tile_rows}"
        )
    tiles = seqlen // query_tile_rows
    actual_tiles = actual32.reshape(
        batch, tiles, query_tile_rows, heads, dim
    )
    reference_tiles = reference32.reshape_as(actual_tiles)
    tile_numerator = (actual_tiles * reference_tiles).sum(dim=(2, 4))
    tile_denominator = actual_tiles.square().sum(dim=(2, 4)).clamp_min_(
        1.0e-20
    )
    tile_scale = (tile_numerator / tile_denominator).clamp_min_(0.0)
    tile_scale = tile_scale[:, :, None, :].expand(
        batch, tiles, query_tile_rows, heads
    ).reshape(batch, seqlen, heads)

    result = {
        "uncorrected": comparison(actual32, reference32),
        "global_scalar_oracle": scaled_result(global_scale),
        "per_head_oracle": scaled_result(head_scale),
        "per_query_tile_head_oracle": scaled_result(tile_scale),
        "per_row_head_oracle": scaled_result(row_scale),
    }

    def fitted_scale_result(features: list[Any]) -> dict[str, Any]:
        design = torch.stack(
            [feature.expand_as(row_scale) for feature in features],
            dim=-1,
        ).reshape(-1, len(features))
        target = row_scale.reshape(-1, 1)
        coefficients = torch.linalg.lstsq(design, target).solution[:, 0]
        prediction = (design @ coefficients).reshape_as(row_scale)
        prediction = prediction.clamp(0.25, 2.0)
        residual = target[:, 0] - prediction.reshape(-1)
        centered = target[:, 0] - target.mean()
        r_squared = 1.0 - (
            residual.square().sum()
            / centered.square().sum().clamp_min_(1.0e-20)
        )
        return {
            "coefficients": [
                float(value) for value in coefficients.cpu().tolist()
            ],
            "r_squared_vs_row_oracle": float(r_squared.item()),
            **scaled_result(prediction),
        }

    ones = torch.ones_like(row_scale)
    row_index = torch.arange(
        1,
        seqlen + 1,
        device=actual32.device,
        dtype=torch.float32,
    ).reshape(1, seqlen, 1)
    log_position = row_index.log2() / math.log2(float(seqlen))
    result["fitted_log_position"] = fitted_scale_result(
        [ones, log_position]
    )

    if actual_lse is not None and reference_lse is not None:
        def lse_to_bsh(lse: Any) -> Any:
            value = lse.float()
            if value.ndim == 4 and value.shape[2] == 1:
                value = value.squeeze(2)
            if value.ndim != 3:
                raise ValueError(f"unsupported LSE shape {tuple(lse.shape)}")
            if value.shape == (batch, heads, seqlen):
                return value.transpose(1, 2)
            if value.shape == (batch, seqlen, heads):
                return value
            raise ValueError(f"unsupported LSE shape {tuple(lse.shape)}")

        actual_lse_bsh = lse_to_bsh(actual_lse)
        reference_lse_bsh = lse_to_bsh(reference_lse)
        log_scale = actual_lse_bsh - reference_lse_bsh
        lse_scale = torch.exp(log_scale.clamp(-20.0, 20.0))
        result["bf16_lse_substitution"] = {
            "log_scale": scale_stats(log_scale),
            **scaled_result(lse_scale),
        }
        normalized_lse = (
            actual_lse_bsh - actual_lse_bsh.mean()
        ) / actual_lse_bsh.std().clamp_min_(1.0e-20)
        result["fitted_actual_lse"] = fitted_scale_result(
            [ones, normalized_lse]
        )
        result["fitted_log_position_and_actual_lse"] = (
            fitted_scale_result([ones, log_position, normalized_lse])
        )
    return result


def localized_comparison(
    a: Any,
    b: Any,
    *,
    query_tile_rows: int = 128,
) -> dict[str, Any]:
    import torch

    if a.shape != b.shape:
        raise ValueError(f"comparison shape mismatch: {a.shape} != {b.shape}")
    if a.ndim != 4:
        raise ValueError(f"expected BSHD output, got shape {a.shape}")

    a32 = a.float()
    b32 = b.float()
    delta = a32 - b32
    abs_delta = delta.abs()
    _, seqlen, heads, _ = a.shape
    if seqlen % query_tile_rows != 0:
        raise ValueError(
            f"seqlen {seqlen} is not divisible by {query_tile_rows}"
        )

    query_tiles = []
    for tile in range(seqlen // query_tile_rows):
        begin = tile * query_tile_rows
        end = begin + query_tile_rows
        query_tiles.append(
            {
                "tile": tile,
                "query_begin": begin,
                **comparison(a[:, begin:end], b[:, begin:end]),
            }
        )

    per_head = []
    for head in range(heads):
        per_head.append(
            {
                "head": head,
                **comparison(a[:, :, head], b[:, :, head]),
            }
        )

    row_rmse = delta.square().mean(dim=(0, 2, 3)).sqrt()
    worst_count = min(16, seqlen)
    worst_values, worst_indices = torch.topk(row_rmse, worst_count)
    worst_rows = [
        {"query_row": int(index), "rmse": float(value)}
        for value, index in zip(
            worst_values.cpu().tolist(),
            worst_indices.cpu().tolist(),
        )
    ]

    magnitude_edges = (0.0, 0.005, 0.01, 0.02, 0.04, float("inf"))
    magnitude_buckets = []
    reference_abs = b32.abs()
    for low, high in zip(magnitude_edges[:-1], magnitude_edges[1:]):
        mask = (reference_abs >= low) & (reference_abs < high)
        count = int(mask.sum().item())
        if count == 0:
            continue
        selected = delta[mask]
        magnitude_buckets.append(
            {
                "reference_abs_low": low,
                "reference_abs_high": high,
                "count": count,
                "mean_abs": float(selected.abs().mean().item()),
                "rmse": float(selected.square().mean().sqrt().item()),
            }
        )

    quantile_levels = torch.tensor(
        [0.5, 0.9, 0.95, 0.99, 0.999],
        device=abs_delta.device,
        dtype=torch.float32,
    )
    quantile_input = abs_delta.flatten()
    max_quantile_samples = 8 * 1024 * 1024
    quantile_stride = max(
        1,
        (quantile_input.numel() + max_quantile_samples - 1)
        // max_quantile_samples,
    )
    quantile_values = torch.quantile(
        quantile_input[::quantile_stride],
        quantile_levels,
    )
    return {
        "global": comparison(a, b),
        "query_tile_even": comparison(
            a.reshape(
                a.shape[0],
                seqlen // query_tile_rows,
                query_tile_rows,
                heads,
                a.shape[3],
            )[:, 0::2],
            b.reshape(
                b.shape[0],
                seqlen // query_tile_rows,
                query_tile_rows,
                heads,
                b.shape[3],
            )[:, 0::2],
        ),
        "query_tile_odd": comparison(
            a.reshape(
                a.shape[0],
                seqlen // query_tile_rows,
                query_tile_rows,
                heads,
                a.shape[3],
            )[:, 1::2],
            b.reshape(
                b.shape[0],
                seqlen // query_tile_rows,
                query_tile_rows,
                heads,
                b.shape[3],
            )[:, 1::2],
        ),
        "abs_error_quantiles": {
            str(level): float(value)
            for level, value in zip(
                quantile_levels.cpu().tolist(),
                quantile_values.cpu().tolist(),
            )
        },
        "abs_error_quantile_sample_stride": quantile_stride,
        "magnitude_buckets": magnitude_buckets,
        "per_head": per_head,
        "per_query_tile": query_tiles,
        "worst_query_rows": worst_rows,
    }


def tensor_stats(tensor: Any) -> dict[str, Any]:
    value = tensor.float()
    batch, seqlen, heads, dim = value.shape
    midpoint = max(1, seqlen // 2)

    def zero_fraction(part: Any) -> float:
        return float((part == 0).float().mean().item())

    row_zero = (value == 0).float().mean(dim=(2, 3))[0]
    head_zero = (value == 0).float().mean(dim=(1, 3))[0]
    dim_zero = (value == 0).float().mean(dim=(1, 2))[0]
    return {
        "min": float(value.min().item()),
        "max": float(value.max().item()),
        "mean": float(value.mean().item()),
        "zero_fraction": float((value == 0).float().mean().item()),
        "query_first_half_mean": float(
            value[:, :midpoint].mean().item()
        ),
        "query_second_half_mean": float(
            value[:, midpoint:].mean().item()
        ),
        "d_first_half_mean": float(value[..., :64].mean().item()),
        "d_second_half_mean": float(value[..., 64:].mean().item()),
        "d_even_zero_fraction": zero_fraction(value[..., 0::2]),
        "d_odd_zero_fraction": zero_fraction(value[..., 1::2]),
        "row_even_zero_fraction": zero_fraction(value[:, 0::2]),
        "row_odd_zero_fraction": zero_fraction(value[:, 1::2]),
        "tile_even_zero_fraction": zero_fraction(
            value.reshape(
                batch, seqlen // 128, 128, heads, dim
            )[:, 0::2]
        ),
        "tile_odd_zero_fraction": zero_fraction(
            value.reshape(
                batch, seqlen // 128, 128, heads, dim
            )[:, 1::2]
        ),
        "head_even_zero_fraction": zero_fraction(value[:, :, 0::2]),
        "head_odd_zero_fraction": zero_fraction(value[:, :, 1::2]),
        "row_zero_fraction_min": float(row_zero.min().item()),
        "row_zero_fraction_max": float(row_zero.max().item()),
        "row_zero_fraction_first_16": row_zero[:16].tolist(),
        "head_zero_fractions": head_zero.tolist(),
        "dim_zero_fraction_first_16": dim_zero[:16].tolist(),
        "sample_q8_d8": value[0, :8, 0, :8].tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path)
    parser.add_argument(
        "--extension-module",
        help="Python module name compiled into a custom extension",
    )
    parser.add_argument(
        "--compare-extension",
        type=Path,
        help="Optional second TK extension run on the same inputs",
    )
    parser.add_argument(
        "--compare-extension-module",
        help="Python module name compiled into --compare-extension",
    )
    parser.add_argument(
        "--qk-format",
        choices=("nvfp4", "mxfp4"),
        default="nvfp4",
    )
    parser.add_argument(
        "--pv-format",
        choices=("nvfp4", "mxfp4"),
        default="nvfp4",
    )
    parser.add_argument(
        "--nv-qk-global-scale",
        choices=("identity", "te"),
        default="identity",
        help=(
            "NVFP4 Q/K tensor-scale policy. 'te' maps each tensor amax "
            "to the E2M1*E4M3 range and supplies the reciprocal decode "
            "scale to the existing QK scalar path."
        ),
    )
    parser.add_argument(
        "--nv-qk-fold-k64-scales",
        nargs="?",
        const="both",
        choices=("q", "k", "both"),
        help=(
            "Requantize Q, K, or both so corresponding 16-value blocks in "
            "the two K64 halves share one E4M3 scale. The optimized folded "
            "kernel currently requires 'both'; q/k are accuracy probes that "
            "remain valid with the regular kernel."
        ),
    )
    parser.add_argument(
        "--nv-qk-fold-scale-select",
        choices=("max", "mse"),
        default="max",
        help=(
            "Select a shared folded scale from the pair maximum or by a "
            "small offline E4M3 reconstruction-error search. This changes "
            "input quantization only, not kernel timing."
        ),
    )
    parser.add_argument(
        "--nv-qk-fold-scale-multiplier",
        type=float,
        default=1.0,
        help=(
            "Multiply the shared pair-max scale before E4M3 rounding. "
            "Values below one deliberately clip outliers for more precision."
        ),
    )
    parser.add_argument(
        "--nv-v-global-scale",
        choices=("identity", "te"),
        default="identity",
        help="NVFP4 V tensor-scale policy.",
    )
    parser.add_argument(
        "--nv-e4m3-max",
        type=float,
        default=448.0,
        help=(
            "Replace 448 with C in the Transformer Engine global encode "
            "formula s_enc = 6*C/amax for every TE-scaled NVFP4 tensor."
        ),
    )
    parser.add_argument(
        "--nv-qk-e4m3-max",
        type=float,
        help="Override --nv-e4m3-max for Q and K.",
    )
    parser.add_argument(
        "--nv-v-e4m3-max",
        type=float,
        help="Override --nv-e4m3-max for V.",
    )
    parser.add_argument(
        "--nv-q-global-encode",
        type=float,
        help="Override the NVFP4 Q global encode scale.",
    )
    parser.add_argument(
        "--nv-k-global-encode",
        type=float,
        help="Override the NVFP4 K global encode scale.",
    )
    parser.add_argument(
        "--nv-v-global-encode",
        type=float,
        help="Override the NVFP4 V global encode scale.",
    )
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--warmup-ms", type=int, default=10)
    parser.add_argument("--rep-ms", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--causal",
        action="store_true",
        help="Benchmark a causal extension and causal BF16 references",
    )
    parser.add_argument(
        "--causal-leakage-check",
        action="store_true",
        help=(
            "Overwrite the latter half of V and require all earlier causal "
            "outputs to remain bit-identical"
        ),
    )
    parser.add_argument(
        "--constant-v",
        action="store_true",
        help="Replace V with quantized ones to isolate the P/PV path",
    )
    parser.add_argument(
        "--profile-provider",
        choices=("tk", "hao", "bf16"),
        help="Warm the selected provider, launch it once, and exit",
    )
    parser.add_argument(
        "--tk-only",
        action="store_true",
        help="Benchmark only the TK extension without compiling HAO kernels",
    )
    parser.add_argument(
        "--skip-hao-fp4",
        action="store_true",
        help=(
            "Skip HAO's full-FP4 provider while retaining its BF16 "
            "reference and the standard TK-vs-BF16 measurements"
        ),
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print only timing, global error, and topology",
    )
    parser.add_argument(
        "--denominator-analysis",
        action="store_true",
        help=(
            "Report oracle row-rescaling ceilings and, when available, "
            "the result of substituting the BF16 LSE denominator"
        ),
    )
    parser.add_argument(
        "--interleave-kv-quarters",
        action="store_true",
        help=(
            "Apply the same stride-4 K/V permutation within every "
            "128-token tile before quantization."
        ),
    )
    parser.add_argument(
        "--global-anchor-kv",
        action="store_true",
        help=(
            "Place globally distributed K/V rows at the start of the "
            "physical sequence before quantization."
        ),
    )
    parser.add_argument(
        "--global-anchor-samples",
        type=int,
        choices=(32, 64, 128),
        default=32,
        help="Number of globally distributed K/V rows placed first.",
    )
    parser.add_argument(
        "--mx-v-effective-max",
        type=float,
        default=6.0,
        help=(
            "Effective positive E2M1 endpoint used to quantize MXFP4 V. "
            "Values below 6 trade the coarse 4-to-6 code for range."
        ),
    )
    return parser.parse_args()


def interleave_kv_quarters(ref: Any) -> Any:
    """Make the first physical quarter sample every fourth logical key."""
    batch, seqlen, heads, dim = ref.shape
    if seqlen % 128:
        raise ValueError("K/V quarter interleave requires S divisible by 128")
    return (
        ref.reshape(batch, seqlen // 128, 32, 4, heads, dim)
        .transpose(2, 3)
        .contiguous()
        .reshape(batch, seqlen, heads, dim)
    )


def global_anchor_kv(ref: Any, samples: int) -> Any:
    """Move evenly distributed logical keys to the first physical rows."""
    import torch

    _, seqlen, _, _ = ref.shape
    if seqlen % 128:
        raise ValueError("global K/V anchor requires S divisible by 128")
    if samples > seqlen:
        raise ValueError("global K/V anchor cannot exceed S")
    anchor = (
        torch.linspace(
            0,
            seqlen - 1,
            samples,
            device=ref.device,
            dtype=torch.float32,
        )
        .round()
        .to(torch.long)
    )
    selected = torch.zeros(seqlen, device=ref.device, dtype=torch.bool)
    selected[anchor] = True
    remainder = torch.arange(seqlen, device=ref.device)[~selected]
    return ref.index_select(1, torch.cat((anchor, remainder))).contiguous()


def prepare_native_inputs(
    q_fp4: Any,
    k_fp4: Any,
    v_fp4: Any,
    q_scale: Any,
    k_scale: Any,
    v_scale: Any,
    qk_format: str,
    pv_format: str,
    batch: int,
    seqlen: int,
    heads: int,
    dqk: int,
    dvo: int,
    kv_heads: int | None = None,
    key_tile: int = 128,
    sparse_pv: bool = False,
    compact_folded_qk_scales: bool = False,
    q_global_decode: float = 1.0,
    k_global_decode: float = 1.0,
) -> Any:
    import torch

    if kv_heads is None:
        kv_heads = heads

    if qk_format == "nvfp4":
        q_local = (
            q_fp4.view(torch.uint8)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(torch.float4_e2m1fn_x2)
        )
        k_local = (
            k_fp4.view(torch.uint8)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(torch.float4_e2m1fn_x2)
        )
        query_tiles = seqlen // 128
        key_tiles = (seqlen + key_tile - 1) // key_tile
        wide_key_scale_pages = 2 if key_tile in (192, 256) else 1
        k_scale_tiles = key_tiles * wide_key_scale_pages
        expected_q_scale = (
            32, 4, seqlen // 128, 4, dqk // 64, heads, batch
        )
        expected_k_scale = (
            32, 4, k_scale_tiles, 4, dqk // 64, kv_heads, batch
        )
        if tuple(q_scale.shape) != expected_q_scale:
            raise ValueError(
                f"native Q scale shape {tuple(q_scale.shape)} != "
                f"{expected_q_scale}"
            )
        if tuple(k_scale.shape) != expected_k_scale:
            raise ValueError(
                f"native K scale shape {tuple(k_scale.shape)} != "
                f"{expected_k_scale}"
            )
        q_sc_local = (
            q_scale.permute(6, 2, 5, 4, 0, 1, 3)
            .contiguous()
            .reshape(
                batch,
                query_tiles,
                heads * (dqk // 64),
                512,
            )
        )
        k_sc_local = (
            k_scale.permute(6, 2, 5, 4, 0, 1, 3)
            .contiguous()
            .reshape(
                batch,
                k_scale_tiles,
                kv_heads * (dqk // 64),
                512,
            )
        )
        if compact_folded_qk_scales:
            q_sc_local = (
                q_sc_local.reshape(
                    batch, query_tiles, heads, 2, 512
                )[:, :, :, 0, :]
                .contiguous()
            )
            k_sc_local = (
                k_sc_local.reshape(
                    batch, k_scale_tiles, kv_heads, 2, 512
                )[:, :, :, 0, :]
                .contiguous()
            )
        if key_tile not in (192, 256):
            k_sc_local = k_sc_local.repeat_interleave(
                2, dim=1
            ).contiguous()
        if q_sc_local.dtype == torch.uint8:
            q_sc_local = q_sc_local.view(torch.float8_e4m3fn)
        if k_sc_local.dtype == torch.uint8:
            k_sc_local = k_sc_local.view(torch.float8_e4m3fn)
        q_global_scale = q_global_decode
        k_global_scale = k_global_decode
    else:
        expected_q_payload = (batch, heads, seqlen, dqk // 2)
        expected_k_payload = (batch, kv_heads, seqlen, dqk // 2)
        expected_q_scale = (
            batch, seqlen // 128, heads, 512
        )
        expected_k_scale = (
            batch, seqlen // 128, kv_heads, 512
        )
        if tuple(q_fp4.shape) != expected_q_payload:
            raise ValueError(
                f"MXFP4 Q shape {tuple(q_fp4.shape)} != "
                f"{expected_q_payload}"
            )
        if tuple(k_fp4.shape) != expected_k_payload:
            raise ValueError(
                f"MXFP4 K shape {tuple(k_fp4.shape)} != "
                f"{expected_k_payload}"
            )
        if tuple(q_scale.shape) != expected_q_scale:
            raise ValueError(
                f"MXFP4 Q scale shape {tuple(q_scale.shape)} != "
                f"{expected_q_scale}"
            )
        if tuple(k_scale.shape) != expected_k_scale:
            raise ValueError(
                f"MXFP4 K scale shape {tuple(k_scale.shape)} != "
                f"{expected_k_scale}"
            )
        q_local = q_fp4
        k_local = k_fp4
        q_sc_local = q_scale
        k_sc_local = k_scale.repeat_interleave(2, dim=1)
        # The local quantizer encodes code = x * 6 / E8M0_scale.
        # Fold the resulting 6x factor for each QK operand into the scalar.
        q_global_scale = 1.0 / 6.0
        k_global_scale = 1.0 / 6.0

    if pv_format == "nvfp4":
        if sparse_pv:
            expected_v_scale = (
                batch,
                seqlen // 128,
                kv_heads * 2,
                512,
            )
            if tuple(v_scale.shape) != expected_v_scale:
                raise ValueError(
                    "prepared sparse NVFP4 V scale shape "
                    f"{tuple(v_scale.shape)} != {expected_v_scale}"
                )
            v_sc_local = v_scale
        else:
            expected_v_scale = (
                32,
                4,
                1,
                4,
                seqlen // 64,
                kv_heads,
                batch,
            )
            if tuple(v_scale.shape) != expected_v_scale:
                raise ValueError(
                    f"native V scale shape {tuple(v_scale.shape)} != "
                    f"{expected_v_scale}"
                )
            v_sc_local = (
                v_scale.permute(6, 4, 5, 2, 0, 1, 3)
                .contiguous()
                .reshape(batch, seqlen // 64, kv_heads, 512)
                .reshape(batch, seqlen // 128, 2, kv_heads, 512)
                .permute(0, 1, 3, 2, 4)
                .contiguous()
                .reshape(batch, seqlen // 128, kv_heads * 2, 512)
            )
    else:
        v_scale_segments = (
            2 if key_tile == 192 else 3 if key_tile == 256 else 1
        )
        v_scale_tiles = (
            (seqlen + key_tile - 1) // key_tile
        ) * v_scale_segments
        expected_v_scale = (
            batch, v_scale_tiles, kv_heads, 512
        )
        if tuple(v_scale.shape) != expected_v_scale:
            raise ValueError(
                f"prepared MXFP4 V scale shape {tuple(v_scale.shape)} != "
                f"{expected_v_scale}"
            )
        v_sc_local = v_scale
    if v_sc_local.dtype == torch.uint8:
        v_sc_local = v_sc_local.view(torch.float8_e4m3fn)
    q_sg = torch.full(
        (batch, heads),
        q_global_scale,
        device=q_fp4.device,
        dtype=torch.float32,
    )
    k_sg = torch.full(
        (batch, kv_heads),
        k_global_scale,
        device=k_fp4.device,
        dtype=torch.float32,
    )
    return SimpleNamespace(
        q_fp4_bhsd=q_local,
        q_scale_prepared=q_sc_local,
        q_global_scale=q_sg,
        k_fp4_bhsd=k_local,
        k_scale_prepared=k_sc_local,
        k_global_scale=k_sg,
        v_fp4_bhds=v_fp4,
        v_scale_prepared=v_sc_local,
    )


def nvfp4_te_global_encode_scale(
    ref: Any,
    e4m3_max: float = 448.0,
) -> float:
    import torch

    quantized_source = ref.to(torch.bfloat16).float()
    amax = float(quantized_source.abs().amax().item())
    if not amax > 0.0:
        return 1.0
    return (6.0 * e4m3_max) / amax


def quantize_nvfp4_qk(
    rows_ref: Any,
    global_encode_scale: float,
) -> tuple[Any, Any]:
    import torch
    from flashinfer.quantization import SfLayout, nvfp4_quantize

    batch, seqlen, heads, dqk = rows_ref.shape
    if dqk not in (64, 128) or seqlen % 128 != 0:
        raise ValueError(
            "NVFP4 QK sweep requires D64/D128 and S divisible by 128"
        )
    source = rows_ref.to(torch.bfloat16).reshape(
        batch * seqlen,
        heads * dqk,
    )
    encode = torch.full(
        (1,),
        global_encode_scale,
        device=rows_ref.device,
        dtype=torch.float32,
    )
    fp4_data, scale_data = nvfp4_quantize(
        source,
        encode,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
    )
    fp4 = (
        fp4_data.reshape(batch, seqlen, heads, dqk // 2)
        .view(torch.uint8)
        .view(torch.float4_e2m1fn_x2)
    )
    scale = (
        scale_data.reshape(
            batch,
            seqlen // 128,
            heads,
            dqk // 64,
            32,
            4,
            4,
        )
        .permute(0, 2, 1, 3, 4, 5, 6)
        .contiguous()
        .permute(4, 5, 2, 6, 3, 1, 0)
    )
    return fp4, scale


def quantize_nvfp4_qk_folded_k64_scales(
    rows_ref: Any,
    global_encode_scale: float,
    scale_select: str = "max",
    scale_multiplier: float = 1.0,
    tile_rows: int = 128,
) -> tuple[Any, Any]:
    """Quantize matching D[0:64] and D[64:128] blocks with shared scales."""
    import torch
    from flashinfer.quantization import block_scale_interleave

    batch, seqlen, heads, dqk = rows_ref.shape
    if dqk != 128 or tile_rows not in (96, 128, 192, 256):
        raise ValueError(
            "folded NVFP4 QK requires D128 and N96/N128/N192/N256 tiles"
        )
    if not global_encode_scale > 0.0:
        raise ValueError("NVFP4 global encode scale must be positive")
    if not scale_multiplier > 0.0:
        raise ValueError("folded NVFP4 scale multiplier must be positive")

    source = rows_ref.to(torch.bfloat16).float().reshape(
        batch * seqlen,
        heads,
        8,
        16,
    )
    paired_source = torch.stack(
        (source[:, :, :4], source[:, :, 4:]),
        dim=-2,
    )
    pair_amax = paired_source.abs().amax(dim=(-1, -2))
    base_scale = pair_amax * (global_encode_scale / 6.0)
    if scale_select == "max":
        pair_scale = (
            base_scale * scale_multiplier
        ).to(torch.float8_e4m3fn)
    elif scale_select == "mse":
        thresholds = torch.tensor(
            (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0),
            device=source.device,
            dtype=torch.float32,
        )
        grid = torch.tensor(
            (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
            device=source.device,
            dtype=torch.float32,
        )
        best_error = torch.full_like(pair_amax, float("inf"))
        best_scale = base_scale.to(torch.float8_e4m3fn)
        for multiplier in (
            0.25,
            0.3125,
            0.375,
            0.4375,
            0.5,
            0.625,
            0.75,
            0.875,
            1.0,
            1.125,
            1.25,
        ):
            candidate = (
                base_scale * multiplier
            ).to(torch.float8_e4m3fn)
            actual = candidate.float()[..., None, None]
            encoded = torch.where(
                actual > 0.0,
                paired_source * global_encode_scale / actual,
                torch.zeros_like(paired_source),
            )
            indices = (
                encoded.abs()[..., None] > thresholds
            ).sum(dim=-1)
            reconstructed = (
                torch.copysign(grid[indices], encoded) *
                actual / global_encode_scale
            )
            error = (
                reconstructed - paired_source
            ).square().sum(dim=(-1, -2))
            replace = error < best_error
            best_error = torch.where(replace, error, best_error)
            best_scale = torch.where(replace, candidate, best_scale)
        pair_scale = best_scale
    else:
        raise ValueError(f"unknown folded scale selector: {scale_select}")
    block_scale = torch.cat((pair_scale, pair_scale), dim=2)
    actual_scale = block_scale.float()
    scaled = torch.where(
        actual_scale[..., None] > 0.0,
        source * global_encode_scale / actual_scale[..., None],
        torch.zeros_like(source),
    )

    magnitude = scaled.abs()
    # E2M1 round-to-nearest thresholds. Exact midpoint behavior has negligible
    # effect on this offline accuracy experiment and does not affect timing.
    thresholds = torch.tensor(
        (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0),
        device=source.device,
        dtype=torch.float32,
    )
    codes = (magnitude[..., None] > thresholds).sum(dim=-1).to(torch.uint8)
    codes |= (scaled < 0.0).to(torch.uint8) << 3
    codes = codes.reshape(batch * seqlen, heads * dqk)
    packed = (
        codes[:, 0::2] | (codes[:, 1::2] << 4)
    ).contiguous()
    fp4 = (
        packed.reshape(batch, seqlen, heads, dqk // 2)
        .view(torch.float4_e2m1fn_x2)
    )

    scale_tiles = (seqlen + tile_rows - 1) // tile_rows
    tiled_scale = torch.zeros(
        batch,
        scale_tiles,
        tile_rows,
        heads,
        dqk // 16,
        device=block_scale.device,
        dtype=block_scale.dtype,
    )
    tiled_scale.reshape(
        batch, scale_tiles * tile_rows, heads, dqk // 16
    )[:, :seqlen].copy_(
        block_scale.reshape(batch, seqlen, heads, dqk // 16)
    )
    if tile_rows == 128:
        interleave_source = tiled_scale
    elif tile_rows in (192, 256):
        # Wide tiles consume two ordinary 128-row scale pages. N192 zero-pads
        # the unused 64 rows in its second page.
        padded_scale = torch.zeros(
            batch,
            scale_tiles,
            256,
            heads,
            dqk // 16,
            device=block_scale.device,
            dtype=block_scale.dtype,
        )
        padded_scale[:, :, :tile_rows].copy_(tiled_scale)
        interleave_source = padded_scale.reshape(
            batch, scale_tiles * 2, 128, heads, dqk // 16
        )
    else:
        interleave_source = torch.zeros(
            (batch, scale_tiles, 128, heads, dqk // 16),
            device=block_scale.device,
            dtype=block_scale.dtype,
        )
        interleave_source[:, :, :tile_rows].copy_(tiled_scale)
    scale_pages = interleave_source.shape[1]
    linear_scale = interleave_source.reshape(
        batch * scale_pages * 128,
        heads * (dqk // 16),
    ).contiguous()
    scale_data = block_scale_interleave(
        linear_scale.view(torch.uint8)
    ).view(torch.float8_e4m3fn)
    scale = (
        scale_data.reshape(
            batch,
            scale_pages,
            heads,
            dqk // 64,
            32,
            4,
            4,
        )
        .permute(0, 2, 1, 3, 4, 5, 6)
        .contiguous()
        .permute(4, 5, 2, 6, 3, 1, 0)
    )
    return fp4, scale


def quantize_nvfp4_v(
    v_ref: Any,
    global_encode_scale: float = 1.0,
) -> tuple[Any, Any]:
    import torch
    from flashinfer.quantization import SfLayout, nvfp4_quantize

    batch, seqlen, heads, dvo = v_ref.shape
    if dvo not in (64, 128) or seqlen % 128 != 0:
        raise ValueError(
            "NVFP4 V sweep requires D64/D128 and S divisible by 128"
        )
    if dvo == 64:
        # FlashInfer's 128x4 scale layout requires a complete 128-row tile.
        # Quantize each head in its own zero-padded tile so no scale page is
        # shared with the adjacent head, then retain the real D64 payload.
        v_fp4 = torch.empty(
            (batch, heads, dvo, seqlen // 2),
            device=v_ref.device,
            dtype=torch.float4_e2m1fn_x2,
        )
        v_scale = torch.empty(
            (32, 4, 1, 4, seqlen // 64, heads, batch),
            device=v_ref.device,
            dtype=torch.uint8,
        )
        encode = torch.full(
            (1,),
            global_encode_scale,
            device=v_ref.device,
            dtype=torch.float32,
        )
        for batch_idx in range(batch):
            for head in range(heads):
                rows = (
                    v_ref[batch_idx, :, head]
                    .transpose(0, 1)
                    .to(torch.bfloat16)
                    .contiguous()
                )
                padded = torch.cat(
                    (rows, torch.zeros_like(rows)), dim=0
                )
                payload, scales = nvfp4_quantize(
                    padded,
                    encode,
                    sfLayout=SfLayout.layout_128x4,
                    do_shuffle=False,
                )
                v_fp4[batch_idx, head].view(torch.uint8).copy_(
                    payload[:dvo]
                )
                native_scale = (
                    scales.reshape(
                        1, seqlen // 64, 32, 4, 4
                    )
                    .permute(2, 3, 0, 4, 1)
                    .contiguous()
                )
                v_scale[:, :, :, :, :, head, batch_idx].copy_(
                    native_scale
                )
        return v_fp4, v_scale
    v_kmajor = (
        v_ref.to(torch.bfloat16)
        .permute(0, 2, 3, 1)
        .contiguous()
        .reshape(batch * heads * dvo, seqlen)
    )
    encode = torch.full(
        (1,),
        global_encode_scale,
        device=v_ref.device,
        dtype=torch.float32,
    )
    v_data, v_scale_data = nvfp4_quantize(
        v_kmajor,
        encode,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
    )
    v_fp4 = (
        v_data.reshape(batch, heads, dvo, seqlen // 2)
        .view(torch.uint8)
        .view(torch.float4_e2m1fn_x2)
    )
    v_scale = (
        v_scale_data.reshape(
            batch * heads,
            seqlen // 64,
            32,
            4,
            4,
        )
        .reshape(
            batch,
            heads,
            dvo // 128,
            seqlen // 64,
            32,
            4,
            4,
        )
        .permute(4, 5, 2, 6, 3, 1, 0)
    )
    return v_fp4, v_scale


def quantize_sparse_nvfp4_v(v_ref: Any) -> tuple[Any, Any]:
    import torch

    batch, seqlen, heads, dvo = v_ref.shape
    if dvo != 128 or seqlen % 128 != 0:
        raise ValueError(
            "sparse NVFP4 V requires D128 and S divisible by 128"
        )
    values = (
        v_ref.to(torch.float32)
        .permute(0, 2, 3, 1)
        .contiguous()
    )
    blocks = values.reshape(
        batch,
        heads,
        dvo,
        seqlen // 32,
        32,
    )
    scale = (
        blocks.abs().amax(dim=-1) * (1.0 / 6.0)
    ).to(torch.float8_e4m3fn)
    scale_float = scale.to(torch.float32)
    normalized = torch.where(
        scale_float[..., None] > 0.0,
        blocks / scale_float[..., None],
        torch.zeros_like(blocks),
    )
    magnitude = normalized.abs()
    code = torch.zeros_like(magnitude, dtype=torch.uint8)
    for threshold in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0):
        code += (magnitude >= threshold).to(torch.uint8)
    code |= (normalized < 0.0).to(torch.uint8) << 3
    code = code.reshape(batch, heads, dvo, seqlen)
    packed = code[..., 0::2] | (code[..., 1::2] << 4)
    v_fp4 = packed.contiguous().view(torch.float4_e2m1fn_x2)

    n_tiles = seqlen // 128
    scale_tiles = (
        scale.reshape(batch, heads, 4, 32, n_tiles, 4)
        .permute(0, 4, 1, 3, 2, 5)
        .contiguous()
        .reshape(batch, n_tiles, heads, 512)
    )
    scale_pages = torch.empty(
        (batch, n_tiles, heads, 2, 512),
        device=v_ref.device,
        dtype=torch.float8_e4m3fn,
    )
    scale_pages[:, :, :, 0].copy_(scale_tiles)
    scale_pages[:, :, :, 1].copy_(scale_tiles)
    return (
        v_fp4,
        scale_pages.reshape(batch, n_tiles, heads * 2, 512),
    )


def select_mxfp4_l2_scales(
    rows: Any,
    ceil_payload: Any,
    ceil_scales: Any,
    floor_payload: Any,
    floor_scales: Any,
) -> tuple[Any, Any]:
    import torch

    row_count, column_count = rows.shape
    blocks = column_count // 32
    grouped = rows.float().reshape(row_count, blocks, 32)
    amax = grouped.abs().amax(dim=-1).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    log2_amax = torch.log2(amax)
    floor_scale = torch.exp2(torch.floor(log2_amax)) / 6.0
    ceil_scale = torch.exp2(torch.ceil(log2_amax)) / 6.0
    levels = torch.tensor(
        (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
        device=rows.device,
        dtype=torch.float32,
    )
    midpoints = torch.tensor(
        (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0),
        device=rows.device,
        dtype=torch.float32,
    )

    def reconstruction_error(scale: Any) -> Any:
        normalized = grouped.abs() / scale.unsqueeze(-1)
        code = (
            normalized.unsqueeze(-1) > midpoints
        ).sum(dim=-1)
        reconstructed = (
            levels[code] * scale.unsqueeze(-1) * grouped.sign()
        )
        return (reconstructed - grouped).square().sum(dim=-1)

    choose_ceil = (
        reconstruction_error(ceil_scale)
        <= reconstruction_error(floor_scale)
    )
    payload_mask = choose_ceil.repeat_interleave(16, dim=-1)
    payload_bytes = floor_payload.contiguous().view(torch.uint8).clone()
    ceil_bytes = ceil_payload.contiguous().view(torch.uint8)
    payload_bytes[payload_mask] = ceil_bytes[payload_mask]
    payload = payload_bytes.view(torch.float4_e2m1fn_x2)

    row = torch.arange(row_count, device=rows.device)[:, None]
    block = torch.arange(blocks, device=rows.device)[None, :]
    scale_mask = torch.zeros_like(ceil_scales, dtype=torch.bool)
    scale_mask[
        row // 128,
        block // 4,
        row % 32,
        ((row % 128) // 32) * 4 + block % 4,
    ] = choose_ceil
    scales = torch.where(scale_mask, ceil_scales, floor_scales)
    return payload, scales


def quantize_mxfp4_v_effective_max(
    v_ref: Any,
    effective_max: float,
) -> tuple[Any, Any]:
    """Quantize N128 V while reserving only part of the E2M1 range.

    The stored E8M0 amplitude is enlarged by 6/effective_max. The payload
    therefore targets ``effective_max`` at the original block amax while the
    hardware decode remains unchanged. E8M0 rounding can undershoot that
    target, so the payload code is capped explicitly to make the experiment
    a strict effective-range ablation.
    """
    import torch

    batch, seqlen, heads, dvo = v_ref.shape
    if dvo not in (64, 128) or seqlen % 128:
        raise ValueError(
            "effective-max MXFP4 V requires D64/D128 and S divisible by 128"
        )
    if not 0.5 <= effective_max <= 6.0:
        raise ValueError("MXFP4 V effective max must be in [0.5, 6]")

    levels = torch.tensor(
        (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
        device=v_ref.device,
        dtype=torch.float32,
    )
    midpoints = torch.tensor(
        (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0),
        device=v_ref.device,
        dtype=torch.float32,
    )
    n_tiles = seqlen // 128
    v_fp4 = torch.empty(
        (batch, heads, dvo, seqlen // 2),
        device=v_ref.device,
        dtype=torch.float4_e2m1fn_x2,
    )
    v_scale = torch.empty(
        (batch, heads, 1, n_tiles, 32, 16),
        device=v_ref.device,
        dtype=torch.uint8,
    )

    for batch_idx in range(batch):
        for head in range(heads):
            rows = (
                v_ref[batch_idx, :, head]
                .transpose(0, 1)
                .to(torch.bfloat16)
                .float()
                .contiguous()
            )
            if dvo == 64:
                rows = torch.cat((rows, torch.zeros_like(rows)), dim=0)
            blocks = rows.reshape(128, seqlen // 32, 32)
            amax = blocks.abs().amax(dim=-1)
            amplitude_target = amax * (6.0 / effective_max)

            floor_exp = torch.floor(
                torch.log2(amplitude_target.clamp_min(torch.finfo(torch.float32).tiny))
            )
            floor_amp = torch.exp2(floor_exp)
            significand = amplitude_target / floor_amp
            biased_floor = floor_exp.to(torch.int32) + 127
            round_up = (significand > 1.5) | (
                (significand == 1.5) & ((biased_floor & 1) != 0)
            )
            scale_byte = (biased_floor + round_up.to(torch.int32)).clamp(0, 254)
            scale_byte = torch.where(
                amax > 0.0,
                scale_byte,
                torch.zeros_like(scale_byte),
            ).to(torch.uint8)
            amplitude = torch.exp2(scale_byte.to(torch.float32) - 127.0)

            normalized = blocks * (6.0 / amplitude.unsqueeze(-1))
            magnitude_code = (
                normalized.abs().unsqueeze(-1) >= midpoints
            ).sum(dim=-1)
            if effective_max < 6.0:
                magnitude_code.clamp_max_(6)
            code = magnitude_code.to(torch.uint8)
            code |= torch.signbit(normalized).to(torch.uint8) << 3
            code = code.reshape(128, seqlen)
            packed = code[:, 0::2] | (code[:, 1::2] << 4)
            v_fp4[batch_idx, head].view(torch.uint8).copy_(
                packed[:dvo]
            )

            linear_scale = scale_byte.reshape(128, n_tiles, 4)
            swizzled = v_scale[batch_idx, head, 0]
            for group in range(4):
                rows32 = linear_scale[group * 32:(group + 1) * 32]
                for block in range(4):
                    swizzled[:, :, group * 4 + block].copy_(
                        rows32[:, :, block].transpose(0, 1)
                    )

    prepared_scale = (
        v_scale[:, :, 0]
        .permute(0, 2, 1, 3, 4)
        .contiguous()
        .reshape(batch, n_tiles, heads, 512)
        .view(torch.float8_e4m3fn)
    )
    return v_fp4, prepared_scale


def quantize_mxfp4_v(
    v_ref: Any,
    mode: int = 0,
    tile_keys: int = 128,
    compact_n256_payload: bool = False,
) -> tuple[Any, Any]:
    import torch

    batch, seqlen, heads, dvo = v_ref.shape
    if dvo not in (64, 128) or tile_keys not in (96, 128, 192, 256):
        raise ValueError(
            "MXFP4 V sweep requires D64/D128 and N96/N128/N192/N256 tiles"
        )
    key_tiles = (seqlen + tile_keys - 1) // tile_keys
    segments_per_tile = (
        2 if tile_keys == 192 else 3 if tile_keys == 256 else 1
    )
    scale_tiles = key_tiles * segments_per_tile
    quant_root = REPO_ROOT / "TK_quantisation" / "mxfp4_v3"
    sys.path.insert(0, str(quant_root))
    try:
        import mxfp4_quant_v3
    finally:
        sys.path.pop(0)

    v_fp4 = torch.empty(
        (
            batch,
            heads,
            dvo,
            seqlen // 2
            if compact_n256_payload
            else (
                scale_tiles * 64
                if tile_keys in (96, 192, 256)
                else seqlen // 2
            ),
        ),
        device="cuda",
        dtype=torch.float4_e2m1fn_x2,
    )
    v_scale = torch.empty(
        (batch, heads, 1, scale_tiles, 32, 16),
        device="cuda",
        dtype=torch.uint8,
    )
    for batch_idx in range(batch):
        for head in range(heads):
            rows = (
                v_ref[batch_idx, :, head]
                .transpose(0, 1)
                .to(torch.bfloat16)
                .contiguous()
            )
            quant_rows = (
                rows
                if dvo == 128
                else torch.cat((rows, torch.zeros_like(rows)), dim=0)
            )
            if tile_keys in (96, 192, 256):
                payload_u8 = v_fp4[
                    batch_idx, head
                ].view(torch.uint8)
                for tile in range(key_tiles):
                    segment_specs = (
                        ((0, 96), (96, 96), (192, 64))
                        if tile_keys == 256
                        else ((0, 96), (96, 96))
                        if tile_keys == 192
                        else ((0, 96),)
                    )
                    for segment, (offset, width) in enumerate(
                        segment_specs
                    ):
                        first = tile * tile_keys + offset
                        tile_rows = rows[:, first:first + width]
                        padded_rows = torch.cat(
                            (
                                tile_rows,
                                torch.zeros(
                                    (
                                        tile_rows.size(0),
                                        128 - tile_rows.size(1),
                                    ),
                                    device=tile_rows.device,
                                    dtype=tile_rows.dtype,
                                ),
                            ),
                            dim=1,
                        )
                        tile_payload, tile_scales = (
                            mxfp4_quant_v3.mxfp4_quantize_for_gemm(
                                padded_rows, mode
                            )
                        )
                        scale_index = (
                            tile * segments_per_tile + segment
                        )
                        payload_start = (
                            (tile * tile_keys + offset) // 2
                            if compact_n256_payload
                            else scale_index * 64
                        )
                        payload_width = (
                            width // 2
                            if compact_n256_payload
                            else 64
                        )
                        payload_u8[
                            :,
                            payload_start:payload_start + payload_width,
                        ].copy_(
                            tile_payload[:dvo]
                            .contiguous().view(torch.uint8)[
                                :, :payload_width
                            ]
                        )
                        v_scale[
                            batch_idx, head, 0, scale_index
                        ].copy_(
                            tile_scales.reshape(-1, 32, 16)[0]
                        )
                continue
            if mode == 6:
                ceil_payload, ceil_scales = (
                    mxfp4_quant_v3.mxfp4_quantize_for_gemm(
                        quant_rows, 1
                    )
                )
                floor_payload, floor_scales = (
                    mxfp4_quant_v3.mxfp4_quantize_for_gemm(
                        quant_rows, 2
                    )
                )
                payload, scales = select_mxfp4_l2_scales(
                    quant_rows,
                    ceil_payload,
                    ceil_scales,
                    floor_payload,
                    floor_scales,
                )
            else:
                payload, scales = (
                    mxfp4_quant_v3.mxfp4_quantize_for_gemm(
                        quant_rows, mode
                    )
                )
            v_fp4[batch_idx, head].view(torch.uint8).copy_(
                payload[:dvo].contiguous().view(torch.uint8)
            )
            v_scale[batch_idx, head].copy_(scales)
    prepared_scale = (
        v_scale[:, :, 0]
        .permute(0, 2, 1, 3, 4)
        .contiguous()
        .reshape(batch, scale_tiles, heads, 512)
        .view(torch.float8_e4m3fn)
    )
    return v_fp4, prepared_scale


def quantize_mxfp4_qk(
    rows_ref: Any,
    mode: int = 0,
) -> tuple[Any, Any]:
    import torch

    batch, seqlen, heads, dqk = rows_ref.shape
    if dqk != 128 or seqlen % 128 != 0:
        raise ValueError("MXFP4 QK sweep requires D128 and S divisible by 128")
    quant_root = REPO_ROOT / "TK_quantisation" / "mxfp4_v3"
    sys.path.insert(0, str(quant_root))
    try:
        import mxfp4_quant_v3
    finally:
        sys.path.pop(0)

    payload = torch.empty(
        (batch, heads, seqlen, dqk // 2),
        device="cuda",
        dtype=torch.float4_e2m1fn_x2,
    )
    scales = torch.empty(
        (batch, heads, seqlen // 128, 32, 16),
        device="cuda",
        dtype=torch.uint8,
    )
    for batch_idx in range(batch):
        for head in range(heads):
            head_rows = (
                rows_ref[batch_idx, :, head]
                .to(torch.bfloat16)
                .contiguous()
            )
            if mode == 6:
                ceil_payload, ceil_scales = (
                    mxfp4_quant_v3.mxfp4_quantize_for_gemm(
                        head_rows, 1
                    )
                )
                floor_payload, floor_scales = (
                    mxfp4_quant_v3.mxfp4_quantize_for_gemm(
                        head_rows, 2
                    )
                )
                head_payload, head_scales = select_mxfp4_l2_scales(
                    head_rows,
                    ceil_payload,
                    ceil_scales,
                    floor_payload,
                    floor_scales,
                )
            else:
                head_payload, head_scales = (
                    mxfp4_quant_v3.mxfp4_quantize_for_gemm(
                        head_rows, mode
                    )
                )
            payload[batch_idx, head].view(torch.uint8).copy_(
                head_payload.contiguous().view(torch.uint8)
            )
            scales[batch_idx, head].copy_(head_scales[:, 0])
    prepared_scale = (
        scales.permute(0, 2, 1, 3, 4)
        .contiguous()
        .reshape(batch, seqlen // 128, heads, 512)
        .view(torch.float8_e4m3fn)
    )
    return payload, prepared_scale


def main() -> None:
    args = parse_args()
    lifecycle_poll = os.environ.get("TK_HAO_DIRECT_N96_POLL") == "1"

    def lifecycle_checkpoint(name: str) -> None:
        if lifecycle_poll:
            print(
                json.dumps({"host_checkpoint": name}),
                flush=True,
            )

    lifecycle_checkpoint("parsed_args")
    for name in (
        "nv_q_global_encode",
        "nv_k_global_encode",
        "nv_v_global_encode",
    ):
        value = getattr(args, name)
        if value is not None and (
            not math.isfinite(value) or value <= 0.0
        ):
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "nv_e4m3_max",
        "nv_qk_e4m3_max",
        "nv_v_e4m3_max",
    ):
        value = getattr(args, name)
        if value is not None and (
            not math.isfinite(value)
            or value <= 0.0
            or value > 448.0
        ):
            raise ValueError(
                f"--{name.replace('_', '-')} must be in (0, 448]"
            )
    qk_e4m3_max = (
        args.nv_e4m3_max
        if args.nv_qk_e4m3_max is None
        else args.nv_qk_e4m3_max
    )
    v_e4m3_max = (
        args.nv_e4m3_max
        if args.nv_v_e4m3_max is None
        else args.nv_v_e4m3_max
    )

    import torch
    import triton.testing
    interface = None
    bench_fp4 = None
    if not args.tk_only:
        from flash_attn.cute import interface as hao_interface
        from flash_attn.cute.benchmarks import bench_fp4 as hao_bench_fp4

        interface = hao_interface
        bench_fp4 = hao_bench_fp4

    lifecycle_checkpoint("imported_runtime")

    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    extension_path, module_name = EXTENSIONS[
        (args.qk_format, args.pv_format)
    ]
    if args.extension is not None:
        extension_path = args.extension
    if args.extension_module is not None:
        module_name = args.extension_module
    extension = load(extension_path, module_name)
    lifecycle_checkpoint("loaded_extension")
    topology = dict(extension.read_hao_direct_topology())
    if bool(topology.get("causal", False)) != args.causal:
        raise ValueError(
            "extension causal mode does not match --causal: "
            f"extension={bool(topology.get('causal', False))}, "
            f"requested={args.causal}"
        )
    if bool(topology.get("causal_interleaved_kv", False)) != bool(
        args.interleave_kv_quarters
    ):
        raise ValueError(
            "extension K/V layout does not match "
            "--interleave-kv-quarters: "
            f"extension={bool(topology.get('causal_interleaved_kv', False))}, "
            f"requested={args.interleave_kv_quarters}"
        )
    if (
        topology.get("nv_qk_folded_k64_scales", False)
        and args.nv_qk_fold_k64_scales != "both"
    ):
        raise ValueError(
            "extension requires --nv-qk-fold-k64-scales both"
        )
    lifecycle_checkpoint("read_topology")
    key_tile = int(topology.get("key_tile", 128))
    sparse_pv = bool(topology.get("sparse_pv", False))
    compare_extension = None
    compare_module_name = args.compare_extension_module
    if args.compare_extension is not None:
        if compare_module_name is None:
            raise ValueError(
                "--compare-extension-module is required with "
                "--compare-extension"
            )
        compare_extension = load(
            args.compare_extension,
            compare_module_name,
        )
        compare_topology = dict(
            compare_extension.read_hao_direct_topology()
        )
        topology_shape_keys = (
            "batch",
            "heads",
            "kv_heads",
            "seqlen",
            "dqk",
            "dvo",
            "qk_format",
            "pv_format",
            "nv_qk_compact_folded_scales",
        )
        topology_shape = {
            key: (
                topology.get(key, False)
                if key == "nv_qk_compact_folded_scales"
                else topology[key]
            )
            for key in topology_shape_keys
        }
        compare_topology_shape = {
            key: (
                compare_topology.get(key, False)
                if key == "nv_qk_compact_folded_scales"
                else compare_topology[key]
            )
            for key in topology_shape_keys
        }
        if compare_topology_shape != topology_shape:
            raise ValueError(
                "extension topology mismatch: "
                f"{compare_topology_shape} != {topology_shape}"
            )
    batch = int(topology["batch"])
    seqlen = int(topology["seqlen"])
    heads = int(topology["heads"])
    kv_heads = int(topology.get("kv_heads", heads))
    dqk = int(topology["dqk"])
    dvo = int(topology["dvo"])
    if (
        batch < 1
        or dqk not in (64, 128)
        or dvo != dqk
    ):
        raise ValueError(
            "shape sweep requires positive batch and matching D64/D128"
        )

    hao_full_fp4_supported = dqk == 128 and dvo == 128
    run_hao_full_fp4 = (
        hao_full_fp4_supported
        and not args.tk_only
        and not args.skip_hao_fp4
    )
    if run_hao_full_fp4:
        assert bench_fp4 is not None
        (
            q_fp4,
            k_fp4,
            v_fp4,
            q_scale,
            k_scale,
            v_scale,
            q_ref,
            k_ref,
            v_ref,
        ) = bench_fp4.create_nvfp4_attention_tensors(
            batch,
            seqlen,
            seqlen,
            heads,
            kv_heads,
            dqk,
            dvo,
            device="cuda",
            dtype_gen=torch.bfloat16,
            pv_mode="fp4",
        )
        lifecycle_checkpoint("created_hao_tensors")
    else:
        # Match HAO's random input distribution, but quantize lazily below.
        # A TK-only NV/MX run otherwise builds and discards a full NV/NV set
        # before constructing its actual folded-QK and MX-V operands.
        q_ref = torch.randn(
            batch, seqlen, heads, dqk,
            device="cuda", dtype=torch.float32,
        )
        k_ref = torch.randn_like(q_ref)
        if kv_heads != heads:
            k_ref = torch.randn(
                batch, seqlen, kv_heads, dqk,
                device="cuda", dtype=torch.float32,
            )
        v_ref = torch.randn(
            batch, seqlen, kv_heads, dvo,
            device="cuda", dtype=torch.float32,
        )
        q_fp4 = q_scale = None
        k_fp4 = k_scale = None
        v_fp4 = v_scale = None
        lifecycle_checkpoint("created_native_tensors")
    if run_hao_full_fp4 and v_scale is None:
        raise RuntimeError("NVFP4 V is missing its block scales")
    hao_q_fp4 = q_fp4
    hao_k_fp4 = k_fp4
    hao_q_scale = q_scale
    hao_k_scale = k_scale
    hao_v_fp4 = v_fp4
    hao_v_scale = v_scale
    if args.constant_v:
        v_ref = torch.ones(
            (batch, seqlen, kv_heads, dvo),
            device="cuda",
            dtype=torch.float32,
        )
        hao_v_fp4, hao_v_scale = quantize_nvfp4_v(v_ref)
    logical_k_ref = k_ref
    logical_v_ref = v_ref
    if args.interleave_kv_quarters and args.global_anchor_kv:
        raise ValueError("select only one K/V permutation")
    if args.interleave_kv_quarters:
        k_ref = interleave_kv_quarters(k_ref)
        v_ref = interleave_kv_quarters(v_ref)
        hao_k_fp4, hao_k_scale = quantize_nvfp4_qk(k_ref, 1.0)
        hao_v_fp4, hao_v_scale = quantize_nvfp4_v(v_ref)
    elif args.global_anchor_kv:
        k_ref = global_anchor_kv(k_ref, args.global_anchor_samples)
        v_ref = global_anchor_kv(v_ref, args.global_anchor_samples)
        hao_k_fp4, hao_k_scale = quantize_nvfp4_qk(k_ref, 1.0)
        hao_v_fp4, hao_v_scale = quantize_nvfp4_v(v_ref)

    q_global_encode = 1.0
    k_global_encode = 1.0
    if args.qk_format == "nvfp4":
        if args.nv_qk_global_scale == "te":
            q_global_encode = nvfp4_te_global_encode_scale(
                q_ref,
                qk_e4m3_max,
            )
            k_global_encode = nvfp4_te_global_encode_scale(
                k_ref,
                qk_e4m3_max,
            )
        if args.nv_q_global_encode is not None:
            q_global_encode = args.nv_q_global_encode
        if args.nv_k_global_encode is not None:
            k_global_encode = args.nv_k_global_encode
        fold_q_scale = args.nv_qk_fold_k64_scales in ("q", "both")
        fold_k_scale = args.nv_qk_fold_k64_scales in ("k", "both")
        if q_fp4 is None or q_global_encode != 1.0 or fold_q_scale:
            if fold_q_scale:
                q_fp4, q_scale = quantize_nvfp4_qk_folded_k64_scales(
                    q_ref,
                    q_global_encode,
                    args.nv_qk_fold_scale_select,
                    args.nv_qk_fold_scale_multiplier,
                    128,
                )
            else:
                q_fp4, q_scale = quantize_nvfp4_qk(
                    q_ref,
                    q_global_encode,
                )
        if (
            k_fp4 is None
            or k_global_encode != 1.0
            or args.interleave_kv_quarters
            or args.global_anchor_kv
            or fold_k_scale
        ):
            if fold_k_scale:
                k_fp4, k_scale = quantize_nvfp4_qk_folded_k64_scales(
                    k_ref,
                    k_global_encode,
                    args.nv_qk_fold_scale_select,
                    args.nv_qk_fold_scale_multiplier,
                    key_tile,
                )
            else:
                k_fp4, k_scale = quantize_nvfp4_qk(
                    k_ref,
                    k_global_encode,
                )
    elif (
        args.nv_q_global_encode is not None
        or args.nv_k_global_encode is not None
        or args.nv_qk_global_scale != "identity"
        or args.nv_qk_fold_k64_scales
    ):
        raise ValueError("NVFP4 Q/K global scaling requires NVFP4 QK")

    v_global_encode = 1.0
    if args.pv_format == "nvfp4":
        if args.nv_v_global_scale == "te":
            v_global_encode = nvfp4_te_global_encode_scale(
                v_ref,
                v_e4m3_max,
            )
        if args.nv_v_global_encode is not None:
            v_global_encode = args.nv_v_global_encode
    elif (
        args.nv_v_global_encode is not None
        or args.nv_v_global_scale != "identity"
    ):
        raise ValueError("NVFP4 V global scaling requires NVFP4 PV")

    if args.pv_format == "nvfp4":
        if sparse_pv:
            if v_global_encode != 1.0:
                raise ValueError(
                    "sparse NVFP4 V does not support global scaling"
                )
            v_fp4, v_scale = quantize_sparse_nvfp4_v(v_ref)
        elif (
            v_fp4 is None
            or v_global_encode != 1.0
            or args.interleave_kv_quarters
            or args.global_anchor_kv
        ):
            v_fp4, v_scale = quantize_nvfp4_v(
                v_ref,
                v_global_encode,
            )
        else:
            v_fp4, v_scale = hao_v_fp4, hao_v_scale
    else:
        if args.mx_v_effective_max != 6.0:
            if key_tile != 128:
                raise ValueError(
                    "MXFP4 V effective-max experiment currently requires N128"
                )
            v_fp4, v_scale = quantize_mxfp4_v_effective_max(
                v_ref,
                args.mx_v_effective_max,
            )
        else:
            v_fp4, v_scale = quantize_mxfp4_v(
                v_ref,
                tile_keys=key_tile,
                compact_n256_payload=bool(
                    topology.get("sm103_n256_compact_v", False)
                ),
            )
    if v_scale is None:
        raise RuntimeError("FP4 V is missing its block scales")
    if args.qk_format == "mxfp4":
        q_fp4, q_scale = quantize_mxfp4_qk(q_ref)
        k_fp4, k_scale = quantize_mxfp4_qk(k_ref)

    lifecycle_checkpoint("quantized_inputs")

    prepared = prepare_native_inputs(
        q_fp4,
        k_fp4,
        v_fp4,
        q_scale,
        k_scale,
        v_scale,
        args.qk_format,
        args.pv_format,
        batch,
        seqlen,
        heads,
        dqk,
        dvo,
        kv_heads=kv_heads,
        key_tile=key_tile,
        sparse_pv=sparse_pv,
        compact_folded_qk_scales=bool(
            topology.get("nv_qk_compact_folded_scales", False)
        ),
        q_global_decode=1.0 / q_global_encode,
        k_global_decode=1.0 / k_global_encode,
    )
    lifecycle_checkpoint("prepared_inputs")
    tk_output = torch.empty(
        (batch, seqlen, heads, dvo),
        device="cuda",
        dtype=torch.bfloat16,
    )
    tk_lse = torch.empty(
        (batch, heads, 1, seqlen),
        device="cuda",
        dtype=torch.float32,
    )
    compare_output = None
    compare_lse = None
    if compare_extension is not None:
        compare_output = torch.empty_like(tk_output)
        compare_lse = torch.empty_like(tk_lse)
    direct = {
        "causal": args.causal,
        "return_lse": True,
        "num_splits": 1,
        "pack_gqa": False,
        "_compute_capability": 10,
    }
    q_bf16 = q_ref.to(torch.bfloat16)
    k_bf16 = logical_k_ref.to(torch.bfloat16)
    v_bf16 = logical_v_ref.to(torch.bfloat16)
    hao_timed_output = torch.empty_like(tk_output)
    bf16_timed_output = torch.empty_like(tk_output)
    timed_direct = {
        "causal": args.causal,
        "return_lse": False,
        "num_splits": 1,
        "pack_gqa": False,
        "_compute_capability": 10,
    }

    def run_tk(
        *,
        store_lse: bool,
        v_fp4: Any | None = None,
        output: Any | None = None,
        lse: Any | None = None,
    ) -> None:
        if v_fp4 is None:
            v_fp4 = prepared.v_fp4_bhds
        if output is None:
            output = tk_output
        if lse is None:
            lse = tk_lse
        arguments = (
            prepared.q_fp4_bhsd,
            prepared.q_scale_prepared,
            prepared.q_global_scale,
            prepared.k_fp4_bhsd,
            prepared.k_scale_prepared,
            prepared.k_global_scale,
            v_fp4,
            prepared.v_scale_prepared,
        )
        if v_global_encode == 1.0:
            extension.forward_hao_direct_fp4pv(
                *arguments,
                output,
                lse,
                0,
                True,
                store_lse,
            )
        else:
            if not topology.get("nv_v_global_decode", False):
                raise RuntimeError(
                    "extension was not built with NV V global decode"
                )
            extension.forward_hao_direct_fp4pv_vscale(
                *arguments,
                1.0 / v_global_encode,
                output,
                lse,
                0,
                True,
                store_lse,
            )

    def run_tk_timed() -> None:
        run_tk(store_lse=False)

    def run_causal_leakage_check() -> dict[str, Any] | None:
        if not args.causal_leakage_check:
            return None
        if not args.causal:
            raise ValueError("--causal-leakage-check requires --causal")
        cutoff = (seqlen // 2 // 128) * 128
        if cutoff <= 0 or cutoff >= seqlen:
            raise ValueError(
                "causal leakage check requires at least two 128-row tiles"
            )
        perturbed_v = prepared.v_fp4_bhds.clone()
        perturbed_v.view(torch.uint8)[..., cutoff // 2 :] = 0
        perturbed_output = torch.empty_like(tk_output)
        perturbed_lse = torch.empty_like(tk_lse)
        run_tk(
            store_lse=True,
            v_fp4=perturbed_v,
            output=perturbed_output,
            lse=perturbed_lse,
        )
        torch.cuda.synchronize()
        prefix_equal = torch.equal(
            tk_output[:, :cutoff],
            perturbed_output[:, :cutoff],
        )
        result = {
            "cutoff": cutoff,
            "prefix_bitwise_equal": prefix_equal,
            "prefix": comparison(
                perturbed_output[:, :cutoff],
                tk_output[:, :cutoff],
            ),
            "suffix_bitwise_equal": torch.equal(
                tk_output[:, cutoff:],
                perturbed_output[:, cutoff:],
            ),
            "suffix": comparison(
                perturbed_output[:, cutoff:],
                tk_output[:, cutoff:],
            ),
            "lse_bitwise_equal": torch.equal(tk_lse, perturbed_lse),
        }
        if not prefix_equal:
            raise RuntimeError(
                "causal leakage detected: future V changed an earlier output"
            )
        return result

    def run_compare(*, store_lse: bool) -> None:
        if compare_extension is None:
            raise RuntimeError("compare extension is not loaded")
        arguments = (
            prepared.q_fp4_bhsd,
            prepared.q_scale_prepared,
            prepared.q_global_scale,
            prepared.k_fp4_bhsd,
            prepared.k_scale_prepared,
            prepared.k_global_scale,
            prepared.v_fp4_bhds,
            prepared.v_scale_prepared,
        )
        if v_global_encode == 1.0:
            compare_extension.forward_hao_direct_fp4pv(
                *arguments,
                compare_output,
                compare_lse,
                0,
                True,
                store_lse,
            )
        else:
            if not compare_topology.get(
                "nv_v_global_decode",
                False,
            ):
                raise RuntimeError(
                    "compare extension lacks NV V global decode"
                )
            compare_extension.forward_hao_direct_fp4pv_vscale(
                *arguments,
                1.0 / v_global_encode,
                compare_output,
                compare_lse,
                0,
                True,
                store_lse,
            )

    def run_compare_timed() -> None:
        run_compare(store_lse=False)

    def run_hao_timed() -> Any:
        return interface._flash_attn_fwd(
            hao_q_fp4,
            hao_k_fp4,
            hao_v_fp4,
            mSFQ=hao_q_scale,
            mSFK=hao_k_scale,
            mSFV=hao_v_scale,
            out=hao_timed_output,
            **timed_direct,
        )

    def run_hao_correctness() -> Any:
        return interface._flash_attn_fwd(
            hao_q_fp4,
            hao_k_fp4,
            hao_v_fp4,
            mSFQ=hao_q_scale,
            mSFK=hao_k_scale,
            mSFV=hao_v_scale,
            **direct,
        )

    def run_bf16_timed() -> Any:
        return interface._flash_attn_fwd(
            q_bf16,
            k_bf16,
            v_bf16,
            out=bf16_timed_output,
            **timed_direct,
        )

    def run_bf16_correctness() -> Any:
        return interface._flash_attn_fwd(
            q_bf16,
            k_bf16,
            v_bf16,
            **direct,
        )

    previous_route = os.environ.get("TK_FA4_FP4PV_FWD_CONFIG")
    os.environ["TK_FA4_FP4PV_FWD_CONFIG"] = str(
        topology["route"]
    )
    causal_leakage = None
    try:
        if lifecycle_poll:
            lifecycle_checkpoint("before_lifecycle_reset")
            extension.reset_hao_direct_n96_lifecycle()
            lifecycle_checkpoint("after_lifecycle_reset")
            lifecycle_checkpoint("before_cuda_synchronize")
            torch.cuda.synchronize()
            lifecycle_checkpoint("after_cuda_synchronize")
            kernel_stream = torch.cuda.Stream()
            lifecycle_checkpoint("before_kernel_launch")
            with torch.cuda.stream(kernel_stream):
                run_tk(store_lse=False)
            lifecycle_checkpoint("after_kernel_launch")
            print(
                json.dumps(
                    {"launched_topology": extension.read_hao_direct_topology()},
                    sort_keys=True,
                ),
                flush=True,
            )
            lifecycle_polls = int(
                os.environ.get("TK_HAO_DIRECT_LIFECYCLE_POLLS", "20")
            )
            for poll in range(lifecycle_polls):
                time.sleep(0.1)
                try:
                    stream_ready = kernel_stream.query()
                    stream_error = None
                except Exception as exc:
                    stream_ready = False
                    stream_error = str(exc)
                values = extension.read_hao_direct_n96_lifecycle()
                nonzero = {
                    str(index): int(value)
                    for index, value in enumerate(values)
                    if value != 0.0
                }
                print(
                    json.dumps(
                        {
                            "poll": poll,
                            "lifecycle": nonzero,
                            "stream_ready": stream_ready,
                            "stream_error": stream_error,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            sys.stdout.flush()
            os._exit(0)
        if os.environ.get("TK_HAO_DIRECT_TIMING_ONLY") == "1":
            timing_warmups = int(
                os.environ.get("TK_HAO_DIRECT_TIMING_WARMUPS", "10")
            )
            timing_iterations = int(
                os.environ.get("TK_HAO_DIRECT_TIMING_ITERATIONS", "100")
            )
            for _ in range(timing_warmups):
                run_tk_timed()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(timing_iterations):
                run_tk_timed()
            end.record()
            end.synchronize()
            timing = float(start.elapsed_time(end)) / timing_iterations
            print(
                json.dumps(
                    {
                        "timing_ms": timing,
                        "timing_iterations": timing_iterations,
                        "topology": dict(
                            extension.read_hao_direct_topology()
                        ),
                        "tk_output_stats": tensor_stats(tk_output),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return
        if args.profile_provider is not None:
            if (
                args.profile_provider == "hao"
                and not run_hao_full_fp4
            ):
                raise ValueError(
                    "HAO full-FP4 is unsupported or disabled"
                )
            profile_function = {
                "tk": run_tk_timed,
                "hao": run_hao_timed,
                "bf16": run_bf16_timed,
            }[args.profile_provider]
            profile_warmup_launches = int(
                os.environ.get("TK_HAO_DIRECT_PROFILE_WARMUP_LAUNCHES", "3")
            )
            profile_launch_trace = (
                os.environ.get("TK_HAO_DIRECT_PROFILE_LAUNCH_TRACE") == "1"
            )
            for launch in range(profile_warmup_launches):
                if profile_launch_trace:
                    print(
                        json.dumps({"profile_launch": launch}),
                        flush=True,
                    )
                profile_function()
                if profile_launch_trace:
                    torch.cuda.synchronize()
                    print(
                        json.dumps({"profile_launch_done": launch}),
                        flush=True,
                    )
            torch.cuda.synchronize()
            if profile_launch_trace:
                print(json.dumps({"profile_launch": "measured"}), flush=True)
            profile_function()
            torch.cuda.synchronize()
            if profile_launch_trace:
                print(
                    json.dumps({"profile_launch_done": "measured"}),
                    flush=True,
                )
            print(
                json.dumps(
                    {
                        "profile_provider": args.profile_provider,
                        "topology": dict(
                            extension.read_hao_direct_topology()
                        ),
                        "tk_output_stats": (
                            tensor_stats(tk_output)
                            if args.profile_provider == "tk"
                            else None
                        ),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return
        if args.tk_only:
            run_tk(store_lse=True)
            if compare_extension is not None:
                run_compare(store_lse=True)
            torch.cuda.synchronize()
            reference_output = torch.nn.functional.scaled_dot_product_attention(
                q_bf16.transpose(1, 2),
                k_bf16.transpose(1, 2),
                v_bf16.transpose(1, 2),
                is_causal=args.causal,
                enable_gqa=heads != kv_heads,
            ).transpose(1, 2)
            timing = float(
                triton.testing.do_bench(
                    run_tk_timed,
                    warmup=args.warmup_ms,
                    rep=args.rep_ms,
                    return_mode="median",
                )
            )
            compare_timing = None
            if compare_extension is not None:
                compare_timing = float(
                    triton.testing.do_bench(
                        run_compare_timed,
                        warmup=args.warmup_ms,
                        rep=args.rep_ms,
                        return_mode="median",
                    )
                )
            causal_leakage = run_causal_leakage_check()
            compare_result = None
            if compare_extension is not None:
                compare_result = {
                    "module": compare_module_name,
                    "timing_ms": compare_timing,
                    "vs_torch_bf16": localized_comparison(
                        compare_output,
                        reference_output,
                    ),
                    "vs_primary_tk": localized_comparison(
                        compare_output,
                        tk_output,
                    ),
                }
            if args.summary_only:
                global_error = localized_comparison(
                    tk_output,
                    reference_output,
                )["global"]
                result = {
                    "protocol": {
                        "provider": "tk",
                        "warmup_ms": args.warmup_ms,
                        "rep_ms": args.rep_ms,
                        "seed": args.seed,
                        "causal": args.causal,
                        "nv_e4m3_max": args.nv_e4m3_max,
                        "qk_e4m3_max": qk_e4m3_max,
                        "v_e4m3_max": v_e4m3_max,
                        "q_global_encode": q_global_encode,
                        "k_global_encode": k_global_encode,
                        "v_global_encode": v_global_encode,
                        "mx_v_effective_max": args.mx_v_effective_max,
                        "interleave_kv_quarters": (
                            args.interleave_kv_quarters
                        ),
                        "global_anchor_kv": args.global_anchor_kv,
                        "global_anchor_samples": (
                            args.global_anchor_samples
                        ),
                        "nv_qk_fold_k64_scales": (
                            args.nv_qk_fold_k64_scales or "none"
                        ),
                        "nv_qk_fold_scale_select": (
                            args.nv_qk_fold_scale_select
                        ),
                    },
                    "timing_ms": timing,
                    "correctness_global": global_error,
                    "topology": dict(
                        extension.read_hao_direct_topology()
                    ),
                }
            else:
                result = {
                    "protocol": {
                        "provider": "tk",
                        "warmup_ms": args.warmup_ms,
                        "rep_ms": args.rep_ms,
                        "seed": args.seed,
                        "causal": args.causal,
                        "mx_v_effective_max": args.mx_v_effective_max,
                        "interleave_kv_quarters": (
                            args.interleave_kv_quarters
                        ),
                        "global_anchor_kv": args.global_anchor_kv,
                        "global_anchor_samples": (
                            args.global_anchor_samples
                        ),
                        "nv_qk_fold_k64_scales": (
                            args.nv_qk_fold_k64_scales or "none"
                        ),
                        "nv_qk_fold_scale_select": (
                            args.nv_qk_fold_scale_select
                        ),
                    },
                    "timing_ms": timing,
                    "compare": compare_result,
                    "correctness": {
                        "tk_vs_torch_bf16_output": localized_comparison(
                            tk_output,
                            reference_output,
                        ),
                    },
                    "topology": dict(
                        extension.read_hao_direct_topology()
                    ),
                    "tk_output_stats": tensor_stats(tk_output),
                    "reference_output_stats": tensor_stats(
                        reference_output
                    ),
                }
            if causal_leakage is not None:
                result["causal_leakage"] = causal_leakage
            if args.denominator_analysis:
                result["denominator_analysis"] = denominator_analysis(
                    tk_output,
                    reference_output,
                )
            print(json.dumps(result, indent=2, sort_keys=True))
            return

        run_tk(store_lse=True)
        hao_output = None
        hao_lse = None
        if run_hao_full_fp4:
            hao_output, hao_lse = run_hao_correctness()
        bf16_output, bf16_lse = run_bf16_correctness()
        torch.cuda.synchronize()

        timings = {}
        tk_name = (
            f"tk_hao_direct_{args.qk_format}_{args.pv_format}pv"
        )
        providers = [
            (tk_name, run_tk_timed),
            ("hao_native_bf16", run_bf16_timed),
        ]
        if run_hao_full_fp4:
            providers.insert(
                1,
                ("hao_native_nvfp4_nvfp4pv", run_hao_timed),
            )
        for name, function in providers:
            timings[name] = float(
                triton.testing.do_bench(
                    function,
                    warmup=args.warmup_ms,
                    rep=args.rep_ms,
                    return_mode="median",
                )
            )
        causal_leakage = run_causal_leakage_check()
    finally:
        if previous_route is None:
            os.environ.pop("TK_FA4_FP4PV_FWD_CONFIG", None)
        else:
            os.environ["TK_FA4_FP4PV_FWD_CONFIG"] = previous_route

    correctness = {
        "tk_vs_bf16_output": comparison(tk_output, bf16_output),
        "tk_vs_bf16_lse": comparison(tk_lse.squeeze(2), bf16_lse),
    }
    if run_hao_full_fp4:
        correctness.update(
            {
                "tk_vs_hao_output": comparison(tk_output, hao_output),
                "hao_vs_bf16_output": comparison(hao_output, bf16_output),
                "tk_vs_hao_lse": comparison(tk_lse.squeeze(2), hao_lse),
                "hao_vs_bf16_lse": comparison(hao_lse, bf16_lse),
            }
        )

    result = {
        "shape": {
            "batch": batch,
            "seqlen": seqlen,
            "heads": heads,
            "kv_heads": kv_heads,
            "dim": dqk,
        },
        "protocol": {
            "factory": "HAO create_nvfp4_attention_tensors",
            "tk_qk_format": args.qk_format,
            "tk_pv_format": args.pv_format,
            "nv_qk_global_scale": args.nv_qk_global_scale,
            "nv_v_global_scale": args.nv_v_global_scale,
            "nv_e4m3_max": args.nv_e4m3_max,
            "qk_e4m3_max": qk_e4m3_max,
            "v_e4m3_max": v_e4m3_max,
            "q_global_encode": q_global_encode,
            "k_global_encode": k_global_encode,
            "v_global_encode": v_global_encode,
            "constant_v": args.constant_v,
            "causal": args.causal,
            "interleave_kv_quarters": args.interleave_kv_quarters,
            "global_anchor_kv": args.global_anchor_kv,
            "global_anchor_samples": args.global_anchor_samples,
            "nv_qk_fold_k64_scales": (
                args.nv_qk_fold_k64_scales or "none"
            ),
            "nv_qk_fold_scale_select": args.nv_qk_fold_scale_select,
            "nv_qk_fold_scale_multiplier": (
                args.nv_qk_fold_scale_multiplier
            ),
            "timer": "triton.testing.do_bench median",
            "warmup_ms": args.warmup_ms,
            "rep_ms": args.rep_ms,
        },
        "topology": dict(extension.read_hao_direct_topology()),
        "timing_ms": timings,
        "speedup_vs_hao_bf16": {
            name: timings["hao_native_bf16"] / timing
            for name, timing in timings.items()
            if name != "hao_native_bf16"
        },
        "correctness": correctness,
    }
    if causal_leakage is not None:
        result["causal_leakage"] = causal_leakage
    if args.denominator_analysis:
        result["denominator_analysis"] = denominator_analysis(
            tk_output,
            bf16_output,
            actual_lse=tk_lse,
            reference_lse=bf16_lse,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
