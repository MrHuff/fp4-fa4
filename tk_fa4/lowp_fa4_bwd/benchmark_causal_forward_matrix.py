#!/usr/bin/env python3
"""Matched causal FA4 forward comparison on projection-derived operands.

This is a one-shape worker for a shape matrix.  Invoke it once for every
``(S, Hq, Hkv, D)`` tuple, pointing it at shape-specialized MXFP4-PV and
exact E4M3-FP8-PV extensions.  The worker deliberately does not build either
extension: build provenance and benchmark provenance therefore remain
separate, and a matrix driver can cap build parallelism independently.

All three providers start from one deterministic activation/weight draw:

* CuTe BF16 consumes dense BF16 Q/K/V projected from that draw;
* NVFP4-QK/MXFP4-PV consumes fused projection publications from the draw;
* NVFP4-QK/E4M3-FP8-PV consumes fused projection publications from the draw.

The timed scope is prepared causal attention only.  Projection, quantization,
RoPE, allocation, V-layout conversion, and JSON serialization are outside the
CUDA-event interval and are described explicitly in the result manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import platform
import socket
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F

# Keep the file directly executable as well as ``python -m`` executable.
_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

import tk_fa4.interface as tk_interface
from tk_fa4 import (
    b300_pack_gqa_d128_rope,
    b300_pack_gqa_d64_paired_rope,
    b300_pair_interleave_gqa_d128_qk_projection_weights,
    b300_prepare_e4m3_projection_operand,
    b300_prepare_e4m3_projection_weight,
    b300_prepare_nvfp4_projection_operand,
    b300_prepare_nvfp4_projection_weight,
    b300_project_qkv_gqa_d128_unified_lowp_nvfp4,
    b300_project_qkv_gqa_d64_paired_unified_lowp_e4m3,
    b300_project_qkv_gqa_d64_paired_unified_lowp_nvfp4,
    b300_stack_gqa_d128_qkv_projection_weights,
)
from tk_fa4.lowp_fa4_bwd.profile_gqa_d128_chain import (
    _load_extension,
    _make_rope,
)
from tk_fa4.lowp_fa4_bwd.projection_quantization_reference import (
    decode_native_mxfp4_v,
    decode_native_nvfp4_qk,
)


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
FLASH_ATTN_ROOT = REPO_ROOT / "flash-attention"
PROVIDERS = ("bf16_cute", "nvfp4_qk_mxfp4_pv", "nvfp4_qk_fp8_pv_exact")
GIB = 1 << 30


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mx-extension",
        type=Path,
        required=True,
        help="shape-specialized causal NVFP4-QK/MXFP4-PV extension",
    )
    parser.add_argument("--mx-module")
    parser.add_argument(
        "--fp8-extension",
        type=Path,
        required=True,
        help="shape-specialized exact causal NVFP4-QK/E4M3-FP8-PV extension",
    )
    parser.add_argument("--fp8-module")
    parser.add_argument(
        "--projection-extension",
        type=Path,
        help=(
            "optional _C_b300_lowp_bwd artifact used to publish Q/K/V; "
            "required when it is not installed in this worktree"
        ),
    )
    parser.add_argument("--projection-module", default="_C_b300_lowp_bwd")
    parser.add_argument("--sequence", type=int, default=4096)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, choices=(64, 128), default=64)
    parser.add_argument(
        "--hidden",
        type=int,
        help="projection input width (default: 32 * head-dim)",
    )
    parser.add_argument(
        "--projection-format",
        choices=("auto", "e4m3", "nvfp4"),
        default="auto",
        help=(
            "projection publisher; auto selects E4M3 for an interleaved D64 "
            "MX route and NVFP4 otherwise; D128 currently requires NVFP4"
        ),
    )
    parser.add_argument("--q-quant-scale", type=float)
    parser.add_argument("--k-quant-scale", type=float)
    parser.add_argument(
        "--per-block-qk-scales",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="publish native D128 row-by-K16 Q/K scales",
    )
    parser.add_argument("--input-std", type=float, default=1.0)
    parser.add_argument("--weight-std", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--minimum-free-gib",
        type=float,
        default=8.0,
        help="fail before allocating if the selected GPU has less free memory",
    )
    parser.add_argument(
        "--causal-leakage-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="zero future V and require prefix outputs to remain bitwise equal",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    if args.sequence <= 0 or args.sequence % 256:
        parser.error("--sequence must be positive and divisible by 256")
    if args.q_heads <= 0 or args.kv_heads <= 0:
        parser.error("head counts must be positive")
    if args.q_heads % args.kv_heads:
        parser.error("--q-heads must be divisible by --kv-heads")
    if args.head_dim == 64 and (args.q_heads % 2 or args.kv_heads % 2):
        parser.error("paired D64 projection requires even Hq and Hkv")
    if args.per_block_qk_scales and args.head_dim != 128:
        parser.error("--per-block-qk-scales is a D128 matrix option")
    if args.hidden is None:
        args.hidden = 32 * args.head_dim
    if args.hidden <= 0 or args.hidden % 128:
        parser.error("--hidden must be positive and divisible by 128")
    if args.head_dim == 128 and args.projection_format == "e4m3":
        parser.error("D128 projection-native publication currently requires NVFP4")
    if args.warmups < 0 or args.samples <= 0:
        parser.error("--warmups must be non-negative and --samples positive")
    for name in ("input_std", "weight_std", "minimum_free_gib"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    for name in ("q_quant_scale", "k_quant_scale"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0.0):
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    return args


def _default_module(path: Path) -> str:
    return path.name.split(".", 1)[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_provenance(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _git_output(root: Path, *arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ("git", "-C", str(root), *arguments),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.rstrip("\n")


def _git_provenance(root: Path) -> dict[str, Any]:
    status = _git_output(root, "status", "--porcelain=v1", "--untracked-files=no")
    return {
        "root": str(root.resolve()),
        "head": _git_output(root, "rev-parse", "HEAD"),
        "branch": _git_output(root, "branch", "--show-current"),
        "tracked_dirty": bool(status) if status is not None else None,
        "tracked_status": status,
    }


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return repr(value)


def _tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    values = tensor.float()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "mean": float(values.mean()),
        "rms": float(values.square().mean().sqrt()),
        "max_abs": float(values.abs().max()),
    }


def _apply_pair_rope(
    tensor: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    depth = tensor.shape[-1]
    pairs = tensor.float().reshape(*tensor.shape[:-1], depth // 2, 2)
    cosine_f = cosine.float().unsqueeze(2)
    sine_f = sine.float().unsqueeze(2)
    first = pairs[..., 0]
    second = pairs[..., 1]
    return torch.stack(
        (
            first * cosine_f - second * sine_f,
            first * sine_f + second * cosine_f,
        ),
        dim=-1,
    ).flatten(-2).bfloat16()


def _split_half_to_adjacent_pairs(tensor: torch.Tensor) -> torch.Tensor:
    """Permute standard Llama rotary coordinates into producer-native pairs."""
    first, second = tensor.chunk(2, dim=-1)
    return torch.stack((first, second), dim=-1).flatten(-2).contiguous()


def _chunked_metrics(
    reference: torch.Tensor,
    actual: torch.Tensor,
    *,
    chunk_elements: int = 1 << 20,
) -> dict[str, Any]:
    if reference.shape != actual.shape:
        raise ValueError(
            f"metric shape mismatch: {tuple(reference.shape)} != {tuple(actual.shape)}"
        )
    reference_flat = reference.detach().reshape(-1)
    actual_flat = actual.detach().reshape(-1)
    reference_sq = 0.0
    actual_sq = 0.0
    difference_sq = 0.0
    dot = 0.0
    absolute_sum = 0.0
    maximum = 0.0
    finite = True
    for start in range(0, reference_flat.numel(), chunk_elements):
        stop = min(start + chunk_elements, reference_flat.numel())
        reference_chunk = reference_flat[start:stop].float()
        actual_chunk = actual_flat[start:stop].float()
        difference = actual_chunk - reference_chunk
        reference_sq += float(reference_chunk.square().sum())
        actual_sq += float(actual_chunk.square().sum())
        difference_sq += float(difference.square().sum())
        dot += float((reference_chunk * actual_chunk).sum())
        absolute_sum += float(difference.abs().sum())
        maximum = max(maximum, float(difference.abs().max()))
        finite = finite and bool(torch.isfinite(actual_chunk).all())
    reference_norm = math.sqrt(max(reference_sq, 1.0e-40))
    actual_norm = math.sqrt(max(actual_sq, 1.0e-40))
    return {
        "finite": finite,
        "cosine": dot / (reference_norm * actual_norm),
        "relative_l2": math.sqrt(difference_sq) / reference_norm,
        "norm_ratio": actual_norm / reference_norm,
        "mean_abs": absolute_sum / max(reference_flat.numel(), 1),
        "max_abs": maximum,
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _timing_summary(values: Sequence[float]) -> dict[str, Any]:
    return {
        "unit": "microseconds",
        "median_us": statistics.median(values),
        "mean_us": statistics.fmean(values),
        "minimum_us": min(values),
        "p10_us": _percentile(values, 0.10),
        "p90_us": _percentile(values, 0.90),
        "maximum_us": max(values),
        "samples_us": list(values),
    }


def _time_interleaved(
    functions: dict[str, Callable[[], object]],
    *,
    warmups: int,
    samples: int,
) -> tuple[dict[str, dict[str, Any]], list[list[str]]]:
    names = list(PROVIDERS)
    if set(names) != set(functions):
        raise ValueError("timing providers do not match the fixed comparison set")
    for round_index in range(warmups):
        order = names[round_index % len(names) :] + names[: round_index % len(names)]
        for name in order:
            functions[name]()
        torch.cuda.synchronize()

    values: dict[str, list[float]] = {name: [] for name in names}
    orders: list[list[str]] = []
    for round_index in range(samples):
        offset = round_index % len(names)
        order = names[offset:] + names[:offset]
        orders.append(order)
        events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        for name in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            functions[name]()
            end.record()
            events.append((name, start, end))
        events[-1][2].synchronize()
        for name, start, end in events:
            values[name].append(float(start.elapsed_time(end) * 1000.0))
    return ({name: _timing_summary(values[name]) for name in names}, orders)


def _load_cute_interface() -> Any:
    sys.path.insert(0, str(FLASH_ATTN_ROOT))
    try:
        interface = importlib.import_module("flash_attn.cute.interface")
    finally:
        sys.path.pop(0)
    source = Path(interface.__file__).resolve()
    try:
        source.relative_to(FLASH_ATTN_ROOT.resolve())
    except ValueError as error:
        raise RuntimeError(
            f"CuTe BF16 resolved outside the repository: {source}"
        ) from error
    return interface


def _validate_topology(
    label: str,
    topology: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    expected = {
        "batch": 1,
        "heads": args.q_heads,
        "kv_heads": args.kv_heads,
        "seqlen": args.sequence,
        "dqk": args.head_dim,
        "dvo": args.head_dim,
        "causal": True,
        "qk_format": "nvfp4_e4m3_block16",
        "fixed_route_fastpath": True,
        "route_env_guard_per_launch": False,
        "kernel_attribute_init": "once_per_host_thread_and_cuda_device",
        "tma_descriptor_cache": "bounded_thread_local_gl_descriptors",
        "tma_descriptor_cache_capacity": 256,
        "tma_descriptor_cache_lookup": (
            "splitmix64_device_pointer_four_way_set_associative"
        ),
        "tma_descriptor_cache_set_hash": "splitmix64_device_pointer_v1",
        "tma_descriptor_cache_sets": 64,
        "tma_descriptor_cache_ways": 4,
        "tma_descriptor_cache_capacity_scope": "per_compile_time_gl_slot",
        "tma_descriptor_cache_key": (
            "cuda_device_data_ptr_and_compile_time_gl_slot"
        ),
        "tma_descriptor_cache_owns_tensors": False,
        "tma_descriptor_cache_counter_scope": "calling_host_thread",
    }
    for key, value in expected.items():
        if topology.get(key) != value:
            raise ValueError(
                f"{label} topology {key}={topology.get(key)!r} != {value!r}"
            )
    expected_gl_slots = 10 if label == "mx" else 9
    if topology.get("tma_descriptor_cache_gl_slots") != expected_gl_slots:
        raise ValueError(
            f"{label} topology tma_descriptor_cache_gl_slots="
            f"{topology.get('tma_descriptor_cache_gl_slots')!r} != "
            f"{expected_gl_slots!r}"
        )
    expected_entry_ceiling = 256 * expected_gl_slots
    if (
        topology.get("tma_descriptor_cache_total_entry_ceiling")
        != expected_entry_ceiling
    ):
        raise ValueError(
            f"{label} topology tma_descriptor_cache_total_entry_ceiling="
            f"{topology.get('tma_descriptor_cache_total_entry_ceiling')!r}"
            f" != {expected_entry_ceiling!r}"
        )
    if bool(topology.get("fixed_p_ceiling", False)) or bool(
        topology.get("score_pack_ceiling", False)
    ):
        raise ValueError(f"{label} extension is a diagnostic ceiling build")

    pv_format = str(topology.get("pv_format", ""))
    if label == "mx":
        if pv_format != "mxfp4_e8m0_block32":
            raise ValueError(f"MX extension has unexpected PV format {pv_format!r}")
        if args.head_dim == 128 and args.per_block_qk_scales:
            folded = bool(
                topology.get("nv_qk_folded_k64_scales", False)
            )
            folded_mask = int(
                topology.get(
                    "nv_qk_folded_k64_scale_mask",
                    3 if folded else 0,
                )
            )
            compact_folded = bool(
                topology.get("nv_qk_compact_folded_scales", False)
            )
            preload_mask = int(
                topology.get("nv_qk_preload_page_mask", 0)
            )
            if folded or folded_mask or compact_folded or preload_mask != 3:
                raise ValueError(
                    "D128 per-block Q/K scales require a non-folded MX "
                    "consumer that reads both K64 scale pages"
                )
    else:
        if pv_format != "e4m3_fp8":
            raise ValueError(f"FP8 extension has unexpected PV format {pv_format!r}")
        if int(topology.get("shiftless_fp8_mode", -1)) != 0:
            raise ValueError("FP8 extension is not the exact shiftless-mode-0 route")
        if bool(topology.get("causal_interleaved_kv", False)):
            raise ValueError("exact FP8 extension must consume logical-order K/V")


def _projection_format(
    requested: str,
    depth: int,
    mx_topology: dict[str, Any],
) -> str:
    if requested != "auto":
        return requested
    if depth == 64 and bool(mx_topology.get("causal_interleaved_kv", False)):
        return "e4m3"
    return "nvfp4"


def _make_projection_state(
    args: argparse.Namespace,
    mx_topology: dict[str, Any],
) -> dict[str, Any]:
    depth = args.head_dim
    q_scale = args.q_quant_scale
    k_scale = args.k_quant_scale
    if q_scale is None:
        q_scale = 2.25 if depth == 64 else 16.0
    if k_scale is None:
        k_scale = 2.0 if depth == 64 else 16.0
    projection_format = _projection_format(
        args.projection_format, depth, mx_topology
    )
    if depth == 128 and projection_format != "nvfp4":
        raise ValueError("D128 requires the NVFP4 projection publisher")

    rows = torch.randn(
        args.sequence,
        args.hidden,
        device="cuda",
        dtype=torch.bfloat16,
    ).mul_(args.input_std)
    q_weight = torch.randn(
        args.q_heads * depth,
        args.hidden,
        device="cuda",
        dtype=torch.bfloat16,
    ).mul_(args.weight_std)
    k_weight = torch.randn(
        args.kv_heads * depth,
        args.hidden,
        device="cuda",
        dtype=torch.bfloat16,
    ).mul_(args.weight_std)
    v_weight = torch.randn_like(k_weight).mul_(args.weight_std)
    source_summary = {
        "rows": _tensor_summary(rows),
        "q_weight": _tensor_summary(q_weight),
        "k_weight": _tensor_summary(k_weight),
        "v_weight": _tensor_summary(v_weight),
    }

    rope_cos, rope_sin = _make_rope(args.sequence, depth)
    q_bf16 = F.linear(rows, q_weight).reshape(
        1, args.sequence, args.q_heads, depth
    )
    k_bf16 = F.linear(rows, k_weight).reshape(
        1, args.sequence, args.kv_heads, depth
    )
    if depth == 128:
        q_bf16 = _split_half_to_adjacent_pairs(q_bf16)
        k_bf16 = _split_half_to_adjacent_pairs(k_bf16)
    q_bf16 = _apply_pair_rope(
        q_bf16,
        rope_cos,
        rope_sin,
    ).contiguous()
    k_bf16 = _apply_pair_rope(
        k_bf16,
        rope_cos,
        rope_sin,
    ).contiguous()
    v_bf16 = F.linear(rows, v_weight).reshape(
        1, args.sequence, args.kv_heads, depth
    ).contiguous()

    mx_interleaved = bool(mx_topology.get("causal_interleaved_kv", False))
    if depth == 64:
        qkv_weight = torch.cat((q_weight, k_weight, v_weight), dim=0).contiguous()
        qk_scales = torch.zeros(
            1,
            args.q_heads // 2,
            7,
            device="cuda",
            dtype=torch.float32,
        )
        qk_scales[..., 0] = q_scale
        qk_scales[..., 1] = k_scale
        packed_rope = b300_pack_gqa_d64_paired_rope(rope_cos, rope_sin)
        if projection_format == "e4m3":
            if not mx_interleaved:
                raise ValueError(
                    "D64 E4M3 MX publication requires an interleaved-causal "
                    "MX extension; select --projection-format nvfp4 for an "
                    "older logical-order extension"
                )
            input_operand = tuple(b300_prepare_e4m3_projection_operand(rows))
            weight_operand = tuple(
                b300_prepare_e4m3_projection_weight(qkv_weight)
            )
            exact_bundle = b300_project_qkv_gqa_d64_paired_unified_lowp_e4m3(
                input_operand,
                weight_operand,
                qk_scales,
                packed_rope,
                batch=1,
                seqlen=args.sequence,
                q_heads=args.q_heads,
                kv_heads=args.kv_heads,
                publish_mxfp4_v=False,
                interleave_causal_kv=False,
                represented_backward=True,
                per_block_qk_scales=True,
            )
            mx_bundle = b300_project_qkv_gqa_d64_paired_unified_lowp_e4m3(
                input_operand,
                weight_operand,
                qk_scales,
                packed_rope,
                batch=1,
                seqlen=args.sequence,
                q_heads=args.q_heads,
                kv_heads=args.kv_heads,
                publish_mxfp4_v=True,
                interleave_causal_kv=True,
                represented_backward=True,
                per_block_qk_scales=True,
                experimental_split_v_backward=True,
            )
        else:
            input_operand = tuple(b300_prepare_nvfp4_projection_operand(rows))
            weight_operand = tuple(
                b300_prepare_nvfp4_projection_weight(qkv_weight)
            )
            exact_bundle = b300_project_qkv_gqa_d64_paired_unified_lowp_nvfp4(
                input_operand,
                weight_operand,
                qk_scales,
                packed_rope,
                batch=1,
                seqlen=args.sequence,
                q_heads=args.q_heads,
                kv_heads=args.kv_heads,
                store_bf16=False,
                publish_fp8_backward=True,
                interleave_causal_kv=False,
            )
            mx_bundle = b300_project_qkv_gqa_d64_paired_unified_lowp_nvfp4(
                input_operand,
                weight_operand,
                qk_scales,
                packed_rope,
                batch=1,
                seqlen=args.sequence,
                q_heads=args.q_heads,
                kv_heads=args.kv_heads,
                store_bf16=False,
                publish_fp8_backward=True,
                interleave_causal_kv=mx_interleaved,
            )
    else:
        if mx_interleaved:
            raise ValueError(
                "the retained D128 projection publisher does not emit the "
                "quarter-interleaved causal K/V layout"
            )
        q_weight_paired, k_weight_paired = (
            b300_pair_interleave_gqa_d128_qk_projection_weights(
                q_weight, k_weight
            )
        )
        qkv_weight = b300_stack_gqa_d128_qkv_projection_weights(
            q_weight_paired, k_weight_paired, v_weight
        )
        input_operand = tuple(b300_prepare_nvfp4_projection_operand(rows))
        weight_operand = tuple(
            b300_prepare_nvfp4_projection_weight(qkv_weight)
        )
        qk_scales = torch.zeros(
            1,
            args.q_heads,
            7,
            device="cuda",
            dtype=torch.float32,
        )
        qk_scales[..., 0] = q_scale
        qk_scales[..., 1] = k_scale
        packed_rope = b300_pack_gqa_d128_rope(rope_cos, rope_sin)
        exact_bundle = b300_project_qkv_gqa_d128_unified_lowp_nvfp4(
            input_operand,
            weight_operand,
            qk_scales,
            batch=1,
            seqlen=args.sequence,
            q_heads=args.q_heads,
            kv_heads=args.kv_heads,
            store_bf16=False,
            publish_fp8_backward=True,
            per_block_qk_scales=args.per_block_qk_scales,
            rope_packed=packed_rope,
        )
        mx_bundle = exact_bundle

    if exact_bundle.v_backward_fp8 is None:
        raise RuntimeError("projection did not publish exact E4M3 V")
    exact_v = exact_bundle.v_forward_fp8
    if exact_v is None:
        raise RuntimeError(
            "exact FP8 forward requires projection-native feature-major V; "
            "refusing an unfused permute/contiguous fallback"
        )

    input_identity = {
        "same_source_rows_and_weights": True,
        "same_prepared_input_operand": True,
        "same_prepared_weight_operand": True,
        "q_payload_bitwise_equal": torch.equal(
            exact_bundle.backward.score_q_fp4,
            mx_bundle.backward.score_q_fp4,
        ),
        "q_scale_pages_bitwise_equal": torch.equal(
            exact_bundle.q_forward_scales,
            mx_bundle.q_forward_scales,
        ),
        "k_payload_direct_bitwise_equal": torch.equal(
            exact_bundle.backward.score_k_fp4,
            mx_bundle.backward.score_k_fp4,
        ),
        "k_scale_pages_direct_bitwise_equal": torch.equal(
            exact_bundle.k_forward_scales,
            mx_bundle.k_forward_scales,
        ),
        "expected_k_layout_difference": mx_interleaved,
    }
    projection_boundary = None
    decoded_reference = None
    if depth == 128:
        k_scales_for_decode = exact_bundle.k_forward_scales
        k_scale_tile_rows = 64
        if args.per_block_qk_scales:
            # The per-K16 publisher duplicates each logical S128 K scale
            # page for the consumer's two physical S64 traversal rows.
            k_scales_for_decode = k_scales_for_decode[:, 0::2]
            k_scale_tile_rows = 128
        decoded_q = decode_native_nvfp4_qk(
            exact_bundle.q_forward_fp4,
            exact_bundle.q_forward_scales,
            exact_bundle.q_forward_global_scale,
            scale_tile_rows=128,
        ).movedim(1, 2)
        decoded_k = decode_native_nvfp4_qk(
            exact_bundle.k_forward_fp4,
            k_scales_for_decode,
            exact_bundle.k_forward_global_scale,
            scale_tile_rows=k_scale_tile_rows,
        ).movedim(1, 2)
        decoded_fp8_v = (
            exact_v.movedim(3, 1).contiguous().float().mul(0.25)
        )
        decoded_mx_v = decode_native_mxfp4_v(
            exact_bundle.v_forward_fp4,
            exact_bundle.v_forward_scales,
        )
        projection_boundary = {
            "reference_qk_layout": "adjacent_rotary_pairs",
            "projection_weight_scaling": "2d_16x16",
            "decoded_q_vs_dense_bf16": _chunked_metrics(q_bf16, decoded_q),
            "decoded_k_vs_dense_bf16": _chunked_metrics(k_bf16, decoded_k),
            "decoded_fp8_v_vs_dense_bf16": _chunked_metrics(
                v_bf16, decoded_fp8_v
            ),
            "decoded_mx_v_vs_dense_bf16": _chunked_metrics(
                v_bf16, decoded_mx_v
            ),
            "decoded_mx_v_vs_decoded_fp8_v": _chunked_metrics(
                decoded_fp8_v, decoded_mx_v
            ),
            "q_global_scales_all_one": bool(
                torch.all(exact_bundle.q_forward_global_scale == 1.0)
            ) if args.per_block_qk_scales else None,
            "k_global_scales_all_one": bool(
                torch.all(exact_bundle.k_forward_global_scale == 1.0)
            ) if args.per_block_qk_scales else None,
            "k_even_odd_scale_pages_equal": torch.equal(
                exact_bundle.k_forward_scales[:, 0::2],
                exact_bundle.k_forward_scales[:, 1::2],
            ),
        }
        decoded_reference = {
            "q": decoded_q.bfloat16().contiguous(),
            "k": decoded_k.bfloat16().contiguous(),
            "fp8_v": decoded_fp8_v.bfloat16().contiguous(),
            "mx_v": decoded_mx_v.bfloat16().contiguous(),
        }
    return {
        "q_bf16": q_bf16,
        "k_bf16": k_bf16,
        "v_bf16": v_bf16,
        "mx_bundle": mx_bundle,
        "exact_bundle": exact_bundle,
        "exact_v": exact_v,
        "projection_format": projection_format,
        "publication_contract": {
            "represented_backward": projection_format == "e4m3",
            "per_block_qk_scales": (
                projection_format == "e4m3" or
                (depth == 128 and args.per_block_qk_scales)
            ),
            "projection_weight_scaling": (
                "2d_16x16"
                if projection_format == "nvfp4"
                else "e4m3_per_output_channel"
            ),
            "mx_experimental_split_v_backward": (
                depth == 64 and projection_format == "e4m3"
            ),
        },
        "q_quant_scale": q_scale,
        "k_quant_scale": k_scale,
        "source_summary": source_summary,
        "input_identity": input_identity,
        "projection_boundary": projection_boundary,
        "decoded_reference": decoded_reference,
        "exact_v_materialized_transpose": False,
    }


def _future_v_perturbation(tensor: torch.Tensor, cutoff: int) -> torch.Tensor:
    perturbed = tensor.clone()
    if tensor.dtype == torch.float4_e2m1fn_x2:
        # Feature-major packed E2M1: one byte contains two sequence values.
        perturbed.view(torch.uint8)[..., cutoff // 2 :] = 0
    else:
        # BF16 logical V is [B,S,H,D]; E4M3 forward V is [B,H,D,S].
        if tensor.dtype == torch.bfloat16:
            perturbed[:, cutoff:] = 0
        elif tensor.dtype == torch.float8_e4m3fn:
            perturbed[..., cutoff:] = 0
        else:
            raise TypeError(f"unsupported V dtype for leakage check: {tensor.dtype}")
    return perturbed


def _leakage_result(
    baseline: torch.Tensor,
    perturbed: torch.Tensor,
    cutoff: int,
) -> dict[str, Any]:
    prefix = baseline[:, :cutoff]
    perturbed_prefix = perturbed[:, :cutoff]
    suffix = baseline[:, cutoff:]
    perturbed_suffix = perturbed[:, cutoff:]
    prefix_equal = torch.equal(prefix, perturbed_prefix)
    return {
        "cutoff": cutoff,
        "future_v_prefix_bitwise_equal": prefix_equal,
        "passed": prefix_equal,
        "prefix": _chunked_metrics(prefix, perturbed_prefix),
        "suffix_bitwise_equal": torch.equal(suffix, perturbed_suffix),
        "suffix": _chunked_metrics(suffix, perturbed_suffix),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.mx_extension.is_file():
        raise FileNotFoundError(args.mx_extension)
    if not args.fp8_extension.is_file():
        raise FileNotFoundError(args.fp8_extension)
    explicit_projection_path = None
    if args.projection_extension is not None:
        explicit_projection_path = args.projection_extension.resolve(strict=True)
        projection_extension = _load_extension(
            explicit_projection_path, args.projection_module
        )
        # The public projection wrappers resolve this module-global binding.
        # Installing an explicitly hashed artifact here avoids copying a
        # binary into the clean worktree merely to satisfy Python discovery.
        tk_interface._C_b300_lowp_bwd = projection_extension
        tk_interface._LOWP_BWD_IMPORT_ERROR = None
    if tk_interface._C_b300_lowp_bwd is None:
        raise RuntimeError(
            "the QKV projection extension is unavailable; pass "
            "--projection-extension /path/to/_C_b300_lowp_bwd.so"
        )
    if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
        raise RuntimeError(
            f"GPU index {args.gpu} is unavailable ({torch.cuda.device_count()} visible)"
        )
    torch.cuda.set_device(args.gpu)
    free_bytes, total_bytes = torch.cuda.mem_get_info(args.gpu)
    if free_bytes < args.minimum_free_gib * GIB:
        raise RuntimeError(
            f"GPU {args.gpu} has {free_bytes / GIB:.2f} GiB free; "
            f"--minimum-free-gib requires {args.minimum_free_gib:.2f} GiB"
        )
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    mx_path = args.mx_extension.resolve()
    fp8_path = args.fp8_extension.resolve()
    mx_module = args.mx_module or _default_module(mx_path)
    fp8_module = args.fp8_module or _default_module(fp8_path)
    mx = _load_extension(mx_path, mx_module)
    fp8 = _load_extension(fp8_path, fp8_module)
    mx_topology = dict(mx.read_hao_direct_topology())
    fp8_topology = dict(fp8.read_hao_direct_topology())
    _validate_topology("mx", mx_topology, args)
    _validate_topology("fp8", fp8_topology, args)
    cute = _load_cute_interface()

    state = _make_projection_state(args, mx_topology)
    q_bf16 = state["q_bf16"]
    k_bf16 = state["k_bf16"]
    v_bf16 = state["v_bf16"]
    mx_bundle = state["mx_bundle"]
    exact_bundle = state["exact_bundle"]
    exact_v = state["exact_v"]
    mx_operands = mx_bundle.forward_operands()
    exact_qk_operands = exact_bundle.qk_forward_operands()

    output_shape = (1, args.sequence, args.q_heads, args.head_dim)
    lse_shape = (1, args.q_heads, 1, args.sequence)
    outputs = {
        name: torch.empty(output_shape, device="cuda", dtype=torch.bfloat16)
        for name in PROVIDERS
    }
    lowp_lse = {
        name: torch.empty(lse_shape, device="cuda", dtype=torch.float32)
        for name in PROVIDERS[1:]
    }

    def run_bf16(*, store_lse: bool, v: torch.Tensor = v_bf16) -> Any:
        return cute._flash_attn_fwd(
            q_bf16,
            k_bf16,
            v,
            causal=True,
            return_lse=store_lse,
            out=None if store_lse else outputs["bf16_cute"],
            num_splits=1,
            pack_gqa=False,
            _arch=100,
        )

    def run_mx(
        *,
        store_lse: bool,
        v: torch.Tensor | None = None,
        output: torch.Tensor | None = None,
        lse: torch.Tensor | None = None,
    ) -> None:
        arguments = list(mx_operands)
        if v is not None:
            arguments[6] = v
        mx.forward_hao_direct_fp4pv(
            *arguments,
            output if output is not None else outputs["nvfp4_qk_mxfp4_pv"],
            lse if lse is not None else lowp_lse["nvfp4_qk_mxfp4_pv"],
            0,
            True,
            store_lse,
        )

    def run_fp8(
        *,
        store_lse: bool,
        v: torch.Tensor = exact_v,
        output: torch.Tensor | None = None,
        lse: torch.Tensor | None = None,
    ) -> None:
        fp8.forward_hao_direct_fp8pv(
            *exact_qk_operands,
            v,
            output if output is not None else outputs["nvfp4_qk_fp8_pv_exact"],
            lse if lse is not None else lowp_lse["nvfp4_qk_fp8_pv_exact"],
            0,
            True,
            store_lse,
        )

    try:
        bf16_output, bf16_lse = run_bf16(store_lse=True)
        outputs["bf16_cute"].copy_(bf16_output)
        run_mx(store_lse=True)
        run_fp8(store_lse=True)
        torch.cuda.synchronize()

        # Refresh after the first real launch so runtime-populated topology
        # (grid, shared memory, and validity) is evidence from the exact
        # artifact that will be timed, not its zero-initialized load state.
        mx_topology = dict(mx.read_hao_direct_topology())
        fp8_topology = dict(fp8.read_hao_direct_topology())
        _validate_topology("mx", mx_topology, args)
        _validate_topology("fp8", fp8_topology, args)
        if int(mx_topology.get("valid", 0)) != 1 or int(
            fp8_topology.get("valid", 0)
        ) != 1:
            raise RuntimeError("forward topology was not populated by launch")

        timing_functions = {
            "bf16_cute": lambda: run_bf16(store_lse=True),
            "nvfp4_qk_mxfp4_pv": lambda: run_mx(store_lse=True),
            "nvfp4_qk_fp8_pv_exact": lambda: run_fp8(store_lse=True),
        }
        timings, timing_orders = _time_interleaved(
            timing_functions,
            warmups=args.warmups,
            samples=args.samples,
        )
        no_lse_timings, no_lse_timing_orders = _time_interleaved(
            {
                "bf16_cute": lambda: run_bf16(store_lse=False),
                "nvfp4_qk_mxfp4_pv": lambda: run_mx(store_lse=False),
                "nvfp4_qk_fp8_pv_exact": lambda: run_fp8(store_lse=False),
            },
            warmups=args.warmups,
            samples=args.samples,
        )

        # Re-establish one internally consistent correctness sample before
        # pairing outputs with stored LSE tensors or perturbing V.
        bf16_output, bf16_lse = run_bf16(store_lse=True)
        outputs["bf16_cute"].copy_(bf16_output)
        run_mx(store_lse=True)
        run_fp8(store_lse=True)
        torch.cuda.synchronize()

        mx_output = outputs["nvfp4_qk_mxfp4_pv"]
        fp8_output = outputs["nvfp4_qk_fp8_pv_exact"]
        correctness = {
            "nvfp4_qk_mxfp4_pv_vs_bf16": {
                "output": _chunked_metrics(bf16_output, mx_output),
                "lse": _chunked_metrics(
                    bf16_lse, lowp_lse["nvfp4_qk_mxfp4_pv"].squeeze(2)
                ),
            },
            "nvfp4_qk_fp8_pv_exact_vs_bf16": {
                "output": _chunked_metrics(bf16_output, fp8_output),
                "lse": _chunked_metrics(
                    bf16_lse, lowp_lse["nvfp4_qk_fp8_pv_exact"].squeeze(2)
                ),
            },
            "mxfp4_pv_vs_exact_fp8_pv": {
                "output": _chunked_metrics(fp8_output, mx_output),
                "lse": _chunked_metrics(
                    lowp_lse["nvfp4_qk_fp8_pv_exact"],
                    lowp_lse["nvfp4_qk_mxfp4_pv"],
                ),
            },
        }
        decoded_reference = state["decoded_reference"]
        if decoded_reference is not None:
            decoded_fp8_output, decoded_fp8_lse = cute._flash_attn_fwd(
                decoded_reference["q"],
                decoded_reference["k"],
                decoded_reference["fp8_v"],
                causal=True,
                return_lse=True,
                num_splits=1,
                pack_gqa=False,
                _arch=100,
            )
            decoded_mx_output, decoded_mx_lse = cute._flash_attn_fwd(
                decoded_reference["q"],
                decoded_reference["k"],
                decoded_reference["mx_v"],
                causal=True,
                return_lse=True,
                num_splits=1,
                pack_gqa=False,
                _arch=100,
            )
            correctness.update(
                {
                    "native_fp8_pv_vs_cute_decoded_operands": {
                        "output": _chunked_metrics(
                            decoded_fp8_output, fp8_output
                        ),
                        "lse": _chunked_metrics(
                            decoded_fp8_lse,
                            lowp_lse[
                                "nvfp4_qk_fp8_pv_exact"
                            ].squeeze(2),
                        ),
                    },
                    "native_mxfp4_pv_vs_cute_decoded_operands": {
                        "output": _chunked_metrics(
                            decoded_mx_output, mx_output
                        ),
                        "lse": _chunked_metrics(
                            decoded_mx_lse,
                            lowp_lse[
                                "nvfp4_qk_mxfp4_pv"
                            ].squeeze(2),
                        ),
                    },
                }
            )

        leakage: dict[str, Any] | None = None
        if args.causal_leakage_check:
            cutoff = (args.sequence // 2 // 128) * 128
            if cutoff <= 0 or cutoff >= args.sequence:
                raise ValueError("leakage check requires at least two S128 tiles")
            perturbed_bf16_v = _future_v_perturbation(v_bf16, cutoff)
            perturbed_mx_v = _future_v_perturbation(mx_operands[6], cutoff)
            perturbed_fp8_v = _future_v_perturbation(exact_v, cutoff)
            perturbed_outputs = {
                name: torch.empty_like(outputs[name]) for name in PROVIDERS
            }
            perturbed_lse = {
                name: torch.empty_like(lowp_lse[name]) for name in PROVIDERS[1:]
            }
            perturbed_bf16_output, perturbed_bf16_lse = run_bf16(
                store_lse=True, v=perturbed_bf16_v
            )
            run_mx(
                store_lse=True,
                v=perturbed_mx_v,
                output=perturbed_outputs["nvfp4_qk_mxfp4_pv"],
                lse=perturbed_lse["nvfp4_qk_mxfp4_pv"],
            )
            run_fp8(
                store_lse=True,
                v=perturbed_fp8_v,
                output=perturbed_outputs["nvfp4_qk_fp8_pv_exact"],
                lse=perturbed_lse["nvfp4_qk_fp8_pv_exact"],
            )
            torch.cuda.synchronize()
            leakage = {
                "contract": (
                    "future V is zeroed at an S128-aligned midpoint; output "
                    "rows before that point must remain bitwise unchanged"
                ),
                "bf16_cute": _leakage_result(
                    bf16_output, perturbed_bf16_output, cutoff
                ),
                "nvfp4_qk_mxfp4_pv": _leakage_result(
                    mx_output,
                    perturbed_outputs["nvfp4_qk_mxfp4_pv"],
                    cutoff,
                ),
                "nvfp4_qk_fp8_pv_exact": _leakage_result(
                    fp8_output,
                    perturbed_outputs["nvfp4_qk_fp8_pv_exact"],
                    cutoff,
                ),
                "lse_invariance": {
                    "bf16_cute": torch.equal(bf16_lse, perturbed_bf16_lse),
                    "nvfp4_qk_mxfp4_pv": torch.equal(
                        lowp_lse["nvfp4_qk_mxfp4_pv"],
                        perturbed_lse["nvfp4_qk_mxfp4_pv"],
                    ),
                    "nvfp4_qk_fp8_pv_exact": torch.equal(
                        lowp_lse["nvfp4_qk_fp8_pv_exact"],
                        perturbed_lse["nvfp4_qk_fp8_pv_exact"],
                    ),
                },
            }
            leakage["all_passed"] = all(
                leakage[name]["passed"] for name in PROVIDERS
            )

        bf16_us = timings["bf16_cute"]["median_us"]
        mx_us = timings["nvfp4_qk_mxfp4_pv"]["median_us"]
        fp8_us = timings["nvfp4_qk_fp8_pv_exact"]["median_us"]
        lowp_extension_path = getattr(
            tk_interface._C_b300_lowp_bwd, "__file__", None
        )
        properties = torch.cuda.get_device_properties(args.gpu)
        result = {
            "schema": "matched_causal_forward_matrix_v1",
            "shape": {
                "batch": 1,
                "sequence": args.sequence,
                "q_heads": args.q_heads,
                "kv_heads": args.kv_heads,
                "head_dim": args.head_dim,
                "hidden": args.hidden,
                "causal": True,
            },
            "scope": {
                "timed": "prepared attention forward only",
                "included": [
                    "causal attention kernel",
                    "output store",
                ],
                "excluded": [
                    "dense QKV projection",
                    "projection-input and projection-weight quantization",
                    "RoPE",
                    "Q/K/V publication",
                    "FP8 V feature-major transpose fallback",
                    "allocation",
                ],
                "store_lse_during_timing": True,
                "matches_e2e_lowp_attention_abi": True,
                "accuracy_inputs": (
                    "all providers derive from one BF16 activation/weight draw; "
                    "BF16 uses dense projected Q/K/V while low-precision routes "
                    "use their fused projection publications"
                ),
            },
            "projection": {
                "format": state["projection_format"],
                "publication_contract": state["publication_contract"],
                "q_quant_scale": state["q_quant_scale"],
                "k_quant_scale": state["k_quant_scale"],
                "exact_v_materialized_transpose": state[
                    "exact_v_materialized_transpose"
                ],
                "source_summary": state["source_summary"],
                "matched_input_audit": state["input_identity"],
                "boundary_audit": state["projection_boundary"],
            },
            "topology": {
                "bf16_cute": {
                    "provider": "flash_attn.cute.interface._flash_attn_fwd",
                    "causal": True,
                    "num_splits": 1,
                    "pack_gqa": False,
                    "arch": 100,
                },
                "nvfp4_qk_mxfp4_pv": _jsonable(mx_topology),
                "nvfp4_qk_fp8_pv_exact": _jsonable(fp8_topology),
            },
            "timing": {
                "method": (
                    "rotating-provider CUDA-event samples with production "
                    "LSE publication"
                ),
                "warmups_per_provider": args.warmups,
                "samples_per_provider": args.samples,
                "providers": timings,
                "sample_orders": timing_orders,
            },
            "timing_without_lse_diagnostic": {
                "method": "rotating-provider CUDA-event samples",
                "warmups_per_provider": args.warmups,
                "samples_per_provider": args.samples,
                "providers": no_lse_timings,
                "sample_orders": no_lse_timing_orders,
            },
            "speedup": {
                "mxfp4_pv_over_bf16": bf16_us / mx_us,
                "exact_fp8_pv_over_bf16": bf16_us / fp8_us,
                "mxfp4_pv_over_exact_fp8_pv": fp8_us / mx_us,
            },
            "correctness": correctness,
            "causal_leakage": leakage,
            "provenance": {
                "argv": list(sys.argv if argv is None else argv),
                "seed": args.seed,
                "repository": _git_provenance(REPO_ROOT),
                "worker": _file_provenance(Path(__file__)),
                "mx_extension": {
                    **_file_provenance(mx_path),
                    "module": mx_module,
                },
                "fp8_extension": {
                    **_file_provenance(fp8_path),
                    "module": fp8_module,
                },
                "projection_extension": (
                    {
                        **_file_provenance(Path(lowp_extension_path)),
                        "module": args.projection_module,
                        "explicitly_loaded": explicit_projection_path is not None,
                    }
                    if lowp_extension_path is not None
                    else None
                ),
                "flash_attention": {
                    "source": _file_provenance(Path(cute.__file__)),
                    "repository": _git_provenance(FLASH_ATTN_ROOT),
                },
                "runtime": {
                    "python": sys.version,
                    "platform": platform.platform(),
                    "hostname": socket.gethostname(),
                    "torch": torch.__version__,
                    "cuda_runtime": torch.version.cuda,
                    "cuda_device": args.gpu,
                    "cuda_device_name": properties.name,
                    "cuda_capability": [properties.major, properties.minor],
                    "cuda_total_memory_bytes": total_bytes,
                    "cuda_free_memory_bytes_at_start": free_bytes,
                },
            },
        }
        encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
        print(encoded, end="")
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded)
        if leakage is not None and not leakage["all_passed"]:
            return 2
        return 0
    finally:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
