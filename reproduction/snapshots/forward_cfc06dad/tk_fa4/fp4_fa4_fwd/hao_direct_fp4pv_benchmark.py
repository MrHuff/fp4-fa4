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


def comparison(a: Any, b: Any) -> dict[str, float]:
    import torch

    a32 = a.float()
    b32 = b.float()
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
    }


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
        "--summary-only",
        action="store_true",
        help="Print only timing, global error, and topology",
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
    sparse_pv: bool = False,
    compact_folded_qk_scales: bool = False,
    q_global_decode: float = 1.0,
    k_global_decode: float = 1.0,
) -> Any:
    import torch

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
        expected_scale = (
            32, 4, seqlen // 128, 4, dqk // 64, heads, batch
        )
        if tuple(q_scale.shape) != expected_scale:
            raise ValueError(
                f"native Q scale shape {tuple(q_scale.shape)} != "
                f"{expected_scale}"
            )
        if tuple(k_scale.shape) != expected_scale:
            raise ValueError(
                f"native K scale shape {tuple(k_scale.shape)} != "
                f"{expected_scale}"
            )
        q_sc_local = (
            q_scale.permute(6, 2, 5, 4, 0, 1, 3)
            .contiguous()
            .reshape(
                batch,
                seqlen // 128,
                heads * (dqk // 64),
                512,
            )
        )
        k_sc_local = (
            k_scale.permute(6, 2, 5, 4, 0, 1, 3)
            .contiguous()
            .reshape(
                batch,
                seqlen // 128,
                heads * (dqk // 64),
                512,
            )
        )
        if compact_folded_qk_scales:
            q_sc_local = (
                q_sc_local.reshape(
                    batch, seqlen // 128, heads, 2, 512
                )[:, :, :, 0, :]
                .contiguous()
            )
            k_sc_local = (
                k_sc_local.reshape(
                    batch, seqlen // 128, heads, 2, 512
                )[:, :, :, 0, :]
                .contiguous()
            )
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
        expected_payload = (batch, heads, seqlen, dqk // 2)
        expected_scale = (
            batch, seqlen // 128, heads, 512
        )
        if tuple(q_fp4.shape) != expected_payload:
            raise ValueError(
                f"MXFP4 Q shape {tuple(q_fp4.shape)} != "
                f"{expected_payload}"
            )
        if tuple(k_fp4.shape) != expected_payload:
            raise ValueError(
                f"MXFP4 K shape {tuple(k_fp4.shape)} != "
                f"{expected_payload}"
            )
        if tuple(q_scale.shape) != expected_scale:
            raise ValueError(
                f"MXFP4 Q scale shape {tuple(q_scale.shape)} != "
                f"{expected_scale}"
            )
        if tuple(k_scale.shape) != expected_scale:
            raise ValueError(
                f"MXFP4 K scale shape {tuple(k_scale.shape)} != "
                f"{expected_scale}"
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
                heads * 2,
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
                dvo // 128,
                4,
                seqlen // 64,
                heads,
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
                .reshape(batch, seqlen // 64, heads, 512)
                .reshape(batch, seqlen // 128, 2, heads, 512)
                .permute(0, 1, 3, 2, 4)
                .contiguous()
                .reshape(batch, seqlen // 128, heads * 2, 512)
            )
    else:
        expected_v_scale = (
            batch, seqlen // 128, heads, 512
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
        (batch, heads),
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
    if dqk != 128 or seqlen % 128 != 0:
        raise ValueError("NVFP4 QK sweep requires D128 and S divisible by 128")
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
) -> tuple[Any, Any]:
    """Quantize matching D[0:64] and D[64:128] blocks with shared scales."""
    import torch
    from flashinfer.quantization import block_scale_interleave

    batch, seqlen, heads, dqk = rows_ref.shape
    if dqk != 128 or seqlen % 128 != 0:
        raise ValueError("folded NVFP4 QK requires D128 and S divisible by 128")
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

    linear_scale = block_scale.reshape(
        batch * seqlen,
        heads * (dqk // 16),
    ).contiguous()
    scale_data = block_scale_interleave(
        linear_scale.view(torch.uint8)
    ).view(torch.float8_e4m3fn)
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


def quantize_nvfp4_v(
    v_ref: Any,
    global_encode_scale: float = 1.0,
) -> tuple[Any, Any]:
    import torch
    from flashinfer.quantization import SfLayout, nvfp4_quantize

    batch, seqlen, heads, dvo = v_ref.shape
    if dvo != 128 or seqlen % 128 != 0:
        raise ValueError("NVFP4 V sweep requires D128 and S divisible by 128")
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


def quantize_mxfp4_v(
    v_ref: Any,
    mode: int = 0,
) -> tuple[Any, Any]:
    import torch

    batch, seqlen, heads, dvo = v_ref.shape
    if dvo != 128 or seqlen % 128 != 0:
        raise ValueError("MXFP4 V sweep requires D128 and S divisible by 128")
    quant_root = REPO_ROOT / "TK_quantisation" / "mxfp4_v3"
    sys.path.insert(0, str(quant_root))
    try:
        import mxfp4_quant_v3
    finally:
        sys.path.pop(0)

    v_fp4 = torch.empty(
        (batch, heads, dvo, seqlen // 2),
        device="cuda",
        dtype=torch.float4_e2m1fn_x2,
    )
    v_scale = torch.empty(
        (batch, heads, dvo // 128, seqlen // 128, 32, 16),
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
            if mode == 6:
                ceil_payload, ceil_scales = (
                    mxfp4_quant_v3.mxfp4_quantize_for_gemm(rows, 1)
                )
                floor_payload, floor_scales = (
                    mxfp4_quant_v3.mxfp4_quantize_for_gemm(rows, 2)
                )
                payload, scales = select_mxfp4_l2_scales(
                    rows,
                    ceil_payload,
                    ceil_scales,
                    floor_payload,
                    floor_scales,
                )
            else:
                payload, scales = (
                    mxfp4_quant_v3.mxfp4_quantize_for_gemm(rows, mode)
                )
            v_fp4[batch_idx, head].view(torch.uint8).copy_(
                payload.contiguous().view(torch.uint8)
            )
            v_scale[batch_idx, head].copy_(scales)
    prepared_scale = (
        v_scale[:, :, 0]
        .permute(0, 2, 1, 3, 4)
        .contiguous()
        .reshape(batch, seqlen // 128, heads, 512)
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
    from flash_attn.cute import interface
    from flash_attn.cute.benchmarks import bench_fp4

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
    topology = dict(extension.read_hao_direct_topology())
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
    dqk = int(topology["dqk"])
    dvo = int(topology["dvo"])
    if batch < 1 or dqk != 128 or dvo != 128:
        raise ValueError("shape sweep requires positive batch and D128")

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
        heads,
        dqk,
        dvo,
        device="cuda",
        dtype_gen=torch.bfloat16,
        pv_mode="fp4",
    )
    if v_scale is None:
        raise RuntimeError("NVFP4 V is missing its block scales")
    hao_q_fp4 = q_fp4
    hao_k_fp4 = k_fp4
    hao_q_scale = q_scale
    hao_k_scale = k_scale
    hao_v_fp4 = v_fp4
    hao_v_scale = v_scale
    if args.constant_v:
        v_ref = torch.ones(
            (batch, seqlen, heads, dvo),
            device="cuda",
            dtype=torch.float32,
        )
        hao_v_fp4, hao_v_scale = quantize_nvfp4_v(v_ref)
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
        if q_global_encode != 1.0 or fold_q_scale:
            if fold_q_scale:
                q_fp4, q_scale = quantize_nvfp4_qk_folded_k64_scales(
                    q_ref,
                    q_global_encode,
                    args.nv_qk_fold_scale_select,
                    args.nv_qk_fold_scale_multiplier,
                )
            else:
                q_fp4, q_scale = quantize_nvfp4_qk(
                    q_ref,
                    q_global_encode,
                )
        if (
            k_global_encode != 1.0
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
            v_global_encode != 1.0
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
        v_fp4, v_scale = quantize_mxfp4_v(v_ref)
    if args.qk_format == "mxfp4":
        q_fp4, q_scale = quantize_mxfp4_qk(q_ref)
        k_fp4, k_scale = quantize_mxfp4_qk(k_ref)

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
        sparse_pv=sparse_pv,
        compact_folded_qk_scales=bool(
            topology.get("nv_qk_compact_folded_scales", False)
        ),
        q_global_decode=1.0 / q_global_encode,
        k_global_decode=1.0 / k_global_encode,
    )
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
        "causal": False,
        "return_lse": True,
        "num_splits": 1,
        "pack_gqa": False,
        "_compute_capability": 10,
    }
    q_bf16 = q_ref.to(torch.bfloat16)
    k_bf16 = k_ref.to(torch.bfloat16)
    v_bf16 = v_ref.to(torch.bfloat16)

    def run_tk(*, store_lse: bool) -> None:
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
            extension.forward_hao_direct_fp4pv(
                *arguments,
                tk_output,
                tk_lse,
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
                tk_output,
                tk_lse,
                0,
                True,
                store_lse,
            )

    def run_tk_timed() -> None:
        run_tk(store_lse=False)

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
        return interface.flash_attn_func(
            hao_q_fp4,
            hao_k_fp4,
            hao_v_fp4,
            causal=False,
            mSFQ=hao_q_scale,
            mSFK=hao_k_scale,
            mSFV=hao_v_scale,
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
        return interface.flash_attn_func(
            q_bf16,
            k_bf16,
            v_bf16,
            causal=False,
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
    try:
        if args.profile_provider is not None:
            profile_function = {
                "tk": run_tk_timed,
                "hao": run_hao_timed,
                "bf16": run_bf16_timed,
            }[args.profile_provider]
            for _ in range(3):
                profile_function()
            torch.cuda.synchronize()
            profile_function()
            torch.cuda.synchronize()
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
                is_causal=False,
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
                        "nv_e4m3_max": args.nv_e4m3_max,
                        "qk_e4m3_max": qk_e4m3_max,
                        "v_e4m3_max": v_e4m3_max,
                        "q_global_encode": q_global_encode,
                        "k_global_encode": k_global_encode,
                        "v_global_encode": v_global_encode,
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
            print(json.dumps(result, indent=2, sort_keys=True))
            return

        run_tk(store_lse=True)
        hao_output, hao_lse = run_hao_correctness()
        bf16_output, bf16_lse = run_bf16_correctness()
        torch.cuda.synchronize()

        timings = {}
        tk_name = (
            f"tk_hao_direct_{args.qk_format}_{args.pv_format}pv"
        )
        for name, function in (
            (tk_name, run_tk_timed),
            ("hao_native_nvfp4_nvfp4pv", run_hao_timed),
            ("hao_native_bf16", run_bf16_timed),
        ):
            timings[name] = float(
                triton.testing.do_bench(
                    function,
                    warmup=args.warmup_ms,
                    rep=args.rep_ms,
                    return_mode="median",
                )
            )
    finally:
        if previous_route is None:
            os.environ.pop("TK_FA4_FP4PV_FWD_CONFIG", None)
        else:
            os.environ["TK_FA4_FP4PV_FWD_CONFIG"] = previous_route

    result = {
        "shape": {
            "batch": batch,
            "seqlen": seqlen,
            "heads": heads,
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
        "correctness": {
            "tk_vs_bf16_output": comparison(tk_output, bf16_output),
            "tk_vs_hao_output": comparison(tk_output, hao_output),
            "hao_vs_bf16_output": comparison(hao_output, bf16_output),
            "tk_vs_bf16_lse": comparison(tk_lse.squeeze(2), bf16_lse),
            "tk_vs_hao_lse": comparison(tk_lse.squeeze(2), hao_lse),
            "hao_vs_bf16_lse": comparison(hao_lse, bf16_lse),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
