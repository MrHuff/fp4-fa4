#!/usr/bin/env python3
"""Validate a D64 exact-MX replay producer/control A/B.

This is a fixed-shape, projection-native gate for the Llama-1.2B attention
geometry B1/S4096/Hq32/Hkv8/D64.  It compares the authenticated replay
control with either a pre-exponentiated-row-normalizer optimization or an
exact-math all-lane handoff while holding all operands constant.  The forward
probability-scale page is produced by the selected causal D4ALL MXFP4-PV
extension; it is not synthesized.

The timed scopes are deliberately separate:

* output-gradient projection/statistics producer only;
* attention backward only, with each route's statistics already published;
* the training-order reset + producer + attention-backward chain.

Both routes consume represented per-block NVFP4 Q/K, represented MXFP4 V,
the same projection-native E4M3 dO, and the same forward-published probability
scales.  Correctness is captured twice per route so exact-math cross-route
differences can be judged against observed within-route atomic repeat noise.
CUDA timing order rotates between baseline and optimized routes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import socket
import stat
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

# Keep the file directly executable as well as ``python -m`` executable.
_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

import tk_fa4.interface as tk_interface
from tk_fa4 import (
    b300_pack_gqa_d64_paired_rope,
    b300_prepare_e4m3_projection_operand,
    b300_prepare_e4m3_projection_weight,
    b300_prepare_nvfp4_projection_operand,
    b300_prepare_nvfp4_projection_weight,
    b300_project_dout_unified_lowp_nvfp4,
    b300_project_qkv_gqa_d64_paired_unified_lowp_e4m3,
)
from tk_fa4.lowp_fa4_bwd.benchmark_llama12b_e2e import (
    D4ALL_FORWARD_PROBABILITY_REPLAY_TOPOLOGY,
)
from tk_fa4.lowp_fa4_bwd.profile_gqa_d128_chain import (
    CompiledGqaBackward,
    _attention_cute_tensor,
    _load_extension,
    _make_rope,
)
from tk_fa4.lowp_fa4_bwd.tune_d64_gqa_cute import _load_control


BATCH = 1
SEQUENCE = 4096
Q_HEADS = 32
KV_HEADS = 8
HEAD_DIM = 64
HIDDEN = 2048
Q_QUANT_SCALE = 2.25
K_QUANT_SCALE = 2.0
FP8_DS_LIFT = 16
EXP2_DEGREE = 2
EXP2_PERIOD = 0
SEED = 20260820
PROVIDERS = ("baseline", "optimized")
LOG2_E = math.log2(math.e)
LOG2_SIX = math.log2(6.0)
PREEXP_BALANCE_MARKER = "signed_8_10"
ALL_LANE_EXACT_MARKER = (
    "TK_FORWARD_MX_PROBABILITY_REPLAY_ALL_LANE_EXACT_MATH"
)
SHARED_METADATA_MARKER = (
    "TK_FORWARD_MX_PROBABILITY_REPLAY_SHARED_METADATA"
)
SHARED_METADATA_CONTRACT = "soa_2x512_f32"
LOG_CLASSIFIER_MARKER = (
    "TK_FORWARD_MX_PROBABILITY_REPLAY_LOG_CLASSIFIER"
)
LOG_CLASSIFIER_CONTRACT = "gb200_native_first_high_3level_v1"
PSCALE_SENTINEL = 0x12345678


@dataclass
class Route:
    """One independently compiled control and its private output workspace."""

    label: str
    control: Any
    backward: CompiledGqaBackward
    preexp_replay_lse: bool
    last_projection: Any | None = None


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mx-extension",
        type=Path,
        required=True,
        help="shape-specialized causal D4ALL MXFP4-PV extension",
    )
    parser.add_argument("--mx-module", required=True)
    parser.add_argument("--mx-sha256", required=True)
    parser.add_argument("--mx-bytes", type=int, required=True)
    parser.add_argument("--projection-extension-sha256", required=True)
    parser.add_argument(
        "--projection-extension-bytes", type=int, required=True
    )
    parser.add_argument(
        "--optimized-contract",
        choices=(
            "signed_8_10",
            "exact_math_all_lane",
            "exact_math_shared_metadata",
            "exact_math_log_classifier",
        ),
        default="signed_8_10",
        help="contract exported by the optimized precomposed control",
    )
    for route in PROVIDERS:
        parser.add_argument(
            f"--{route}-control",
            type=Path,
            required=True,
            help=f"authenticated precomposed {route} CuTe control",
        )
        parser.add_argument(f"--{route}-control-sha256", required=True)
        parser.add_argument(
            f"--{route}-control-bytes", type=int, required=True
        )
    parser.add_argument(
        "--optimized-patch",
        type=Path,
        help="patch used to derive the optimized precomposed control",
    )
    parser.add_argument("--optimized-patch-sha256")
    parser.add_argument("--optimized-patch-bytes", type=int)
    parser.add_argument("--warmups", type=int, default=9)
    parser.add_argument("--samples", type=int, default=41)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--minimum-free-gib",
        type=float,
        default=32.0,
        help="fail before allocation unless the visible GPU has this much free",
    )
    parser.add_argument(
        "--gradient-cosine-min", type=float, default=0.99999
    )
    parser.add_argument(
        "--gradient-relative-l2-max", type=float, default=5.0e-4
    )
    parser.add_argument(
        "--repeat-noise-multiplier",
        type=float,
        default=2.0,
        help=(
            "exact-math all-lane cross-route tolerance multiplier applied "
            "to the larger within-route repeat difference"
        ),
    )
    parser.add_argument(
        "--baseline-lse-max-abs-error", type=float, default=2.0e-5
    )
    parser.add_argument(
        "--normalizer-log2-max-abs-error", type=float, default=5.0e-4
    )
    parser.add_argument(
        "--normalizer-relative-max-error", type=float, default=5.0e-4
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    patch_identity_args = (
        args.optimized_patch,
        args.optimized_patch_sha256,
        args.optimized_patch_bytes,
    )
    if any(value is not None for value in patch_identity_args) and not all(
        value is not None for value in patch_identity_args
    ):
        parser.error(
            "--optimized-patch, --optimized-patch-sha256, and "
            "--optimized-patch-bytes must be supplied together"
        )
    if (
        args.optimized_contract in (
            "exact_math_shared_metadata",
            "exact_math_log_classifier",
        )
        and args.optimized_patch is None
    ):
        parser.error(
            "the selected experimental contract requires an authenticated "
            "--optimized-patch"
        )
    if (
        args.optimized_patch_bytes is not None
        and args.optimized_patch_bytes <= 0
    ):
        parser.error("--optimized-patch-bytes must be positive")

    if args.warmups < 0 or args.samples < 1:
        parser.error("--warmups must be non-negative and --samples positive")
    if args.mx_bytes <= 0 or args.projection_extension_bytes <= 0:
        parser.error("extension byte sizes must be positive")
    for route in PROVIDERS:
        if getattr(args, f"{route}_control_bytes") <= 0:
            parser.error(f"--{route}-control-bytes must be positive")
    for name in (
        "minimum_free_gib",
        "gradient_cosine_min",
        "gradient_relative_l2_max",
        "repeat_noise_multiplier",
        "baseline_lse_max_abs_error",
        "normalizer_log2_max_abs_error",
        "normalizer_relative_max_error",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            parser.error(
                f"--{name.replace('_', '-')} must be finite and positive"
            )
    if args.gradient_cosine_min > 1.0:
        parser.error("--gradient-cosine-min cannot exceed one")
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_file(
    path: Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
    label: str,
) -> dict[str, Any]:
    try:
        path_stat = path.lstat()
    except OSError as error:
        raise FileNotFoundError(f"cannot stat {label}: {path}") from error
    if not stat.S_ISREG(path_stat.st_mode) or path.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    resolved = path.resolve(strict=True)
    actual_sha256 = _sha256(resolved)
    actual_bytes = resolved.stat().st_size
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{label} SHA256 {actual_sha256} != {expected_sha256}"
        )
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"{label} byte size {actual_bytes} != {expected_bytes}"
        )
    return {
        "path": str(resolved),
        "sha256": actual_sha256,
        "bytes": actual_bytes,
    }


def _projection_extension_identity(
    *,
    expected_sha256: str,
    expected_bytes: int,
    require_preexp: bool,
) -> dict[str, Any]:
    environment_name = tk_interface.LOWP_BWD_EXTENSION_SOURCE_ENV
    requested_source = os.environ.get(environment_name)
    if requested_source is None:
        raise RuntimeError(
            f"set {environment_name} to the freshly built extension; "
            "the gate refuses implicit worktree extension discovery"
        )
    requested = Path(requested_source)
    if not requested.is_absolute():
        raise RuntimeError(f"{environment_name} must be an absolute path")
    requested_stat = requested.lstat()
    if not stat.S_ISREG(requested_stat.st_mode) or requested.is_symlink():
        raise RuntimeError(
            f"{environment_name} must name a regular non-symlink file"
        )
    loaded = tk_interface._C_b300_lowp_bwd
    if loaded is None:
        raise RuntimeError("the low-precision backward extension did not load")
    loaded_path = Path(loaded.__file__).resolve(strict=True)
    resolved = requested.resolve(strict=True)
    if loaded_path != resolved:
        raise RuntimeError(
            f"loaded projection extension {loaded_path} != requested {resolved}"
        )
    actual_sha256 = _sha256(resolved)
    actual_bytes = resolved.stat().st_size
    if actual_sha256 != expected_sha256 or actual_bytes != expected_bytes:
        raise RuntimeError(
            "loaded projection extension identity mismatch: "
            f"sha256={actual_sha256}, bytes={actual_bytes}; expected "
            f"sha256={expected_sha256}, bytes={expected_bytes}"
        )
    required_bindings = [
        "project_dout_unified_fp4_nvfp4",
        (
            "project_qkv_gqa_d64_paired_unified_fp8_e4m3_rope_packed_"
            "interleaved_causal_represented_backward_perblock_qk"
        ),
    ]
    if require_preexp:
        required_bindings.append(
            "project_dout_unified_fp4_nvfp4_replay_preexp"
        )
    missing = [name for name in required_bindings if not hasattr(loaded, name)]
    if missing:
        raise RuntimeError(
            "the selected projection extension lacks required bindings: "
            + ", ".join(missing)
        )
    return {
        "environment": environment_name,
        "path": str(resolved),
        "sha256": actual_sha256,
        "bytes": actual_bytes,
        "required_bindings": list(required_bindings),
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


def _git_identity(root: Path) -> dict[str, Any]:
    status = _git_output(root, "status", "--porcelain=v1")
    return {
        "root": str(root.resolve()),
        "head": _git_output(root, "rev-parse", "HEAD"),
        "branch": _git_output(root, "branch", "--show-current"),
        "dirty": bool(status) if status is not None else None,
        "status": status,
    }


def _tensor_sha256(tensor: torch.Tensor) -> str:
    values = tensor.detach().contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(values).hexdigest()


def _tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    values = tensor.float()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "finite": bool(torch.isfinite(values).all()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "rms": float(values.square().mean().sqrt()),
    }


def _chunked_metrics(
    reference: torch.Tensor,
    actual: torch.Tensor,
    *,
    chunk_elements: int = 1 << 20,
) -> dict[str, Any]:
    if reference.shape != actual.shape:
        raise ValueError(
            f"metric shape mismatch: {tuple(reference.shape)} != "
            f"{tuple(actual.shape)}"
        )
    reference_flat = reference.detach().reshape(-1)
    actual_flat = actual.detach().reshape(-1)
    reference_sq = 0.0
    actual_sq = 0.0
    difference_sq = 0.0
    dot = 0.0
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
        maximum = max(maximum, float(difference.abs().max()))
        finite = finite and bool(torch.isfinite(actual_chunk).all())
    reference_norm = math.sqrt(max(reference_sq, 1.0e-40))
    actual_norm = math.sqrt(max(actual_sq, 1.0e-40))
    return {
        "finite": finite,
        "bitwise_equal": bool(torch.equal(reference, actual)),
        "cosine": dot / (reference_norm * actual_norm),
        "relative_l2": math.sqrt(difference_sq) / reference_norm,
        "norm_ratio": actual_norm / reference_norm,
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


def _time_rotated(
    functions: dict[str, Callable[[], object]],
    *,
    warmups: int,
    samples: int,
) -> dict[str, Any]:
    if set(functions) != set(PROVIDERS):
        raise ValueError("timing providers must be baseline and optimized")
    names = list(PROVIDERS)
    warmup_orders: list[list[str]] = []
    for round_index in range(warmups):
        offset = round_index % len(names)
        order = names[offset:] + names[:offset]
        warmup_orders.append(order)
        for name in order:
            functions[name]()
            torch.cuda.synchronize()

    values: dict[str, list[float]] = {name: [] for name in names}
    sample_orders: list[list[str]] = []
    for round_index in range(samples):
        offset = round_index % len(names)
        order = names[offset:] + names[:offset]
        sample_orders.append(order)
        for name in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            functions[name]()
            end.record()
            end.synchronize()
            values[name].append(float(start.elapsed_time(end) * 1000.0))

    timing = {name: _timing_summary(values[name]) for name in names}
    baseline_us = timing["baseline"]["median_us"]
    optimized_us = timing["optimized"]["median_us"]
    return {
        "providers": timing,
        "comparison": {
            "optimized_minus_baseline_us": optimized_us - baseline_us,
            "baseline_over_optimized_speedup": baseline_us / optimized_us,
        },
        "warmup_orders": warmup_orders,
        "sample_orders": sample_orders,
    }


def _validate_forward_topology(topology: dict[str, Any]) -> None:
    expected_shape = {
        "batch": BATCH,
        "seqlen": SEQUENCE,
        "heads": Q_HEADS,
        "kv_heads": KV_HEADS,
        "dqk": HEAD_DIM,
        "dvo": HEAD_DIM,
        "qk_format": "nvfp4_e4m3_block16",
        "p_scale_publication_supported": True,
    }
    expected = {
        **expected_shape,
        **D4ALL_FORWARD_PROBABILITY_REPLAY_TOPOLOGY,
    }
    mismatches = {
        key: {"actual": topology.get(key), "expected": value}
        for key, value in expected.items()
        if topology.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "the selected forward extension is not the exact D4ALL replay "
            f"topology: {json.dumps(mismatches, sort_keys=True)}"
        )


def _make_projection_state(mx: Any, topology: dict[str, Any]) -> dict[str, Any]:
    rows = torch.randn(
        SEQUENCE, HIDDEN, device="cuda", dtype=torch.bfloat16
    )
    q_weight = torch.randn(
        Q_HEADS * HEAD_DIM,
        HIDDEN,
        device="cuda",
        dtype=torch.bfloat16,
    ).mul_(0.02)
    k_weight = torch.randn(
        KV_HEADS * HEAD_DIM,
        HIDDEN,
        device="cuda",
        dtype=torch.bfloat16,
    ).mul_(0.02)
    v_weight = torch.randn_like(k_weight).mul_(0.02)
    qkv_weight = torch.cat((q_weight, k_weight, v_weight), dim=0).contiguous()

    rope_cos, rope_sin = _make_rope(SEQUENCE, HEAD_DIM)
    packed_rope = b300_pack_gqa_d64_paired_rope(rope_cos, rope_sin)
    qk_scales = torch.zeros(
        BATCH,
        Q_HEADS // 2,
        7,
        device="cuda",
        dtype=torch.float32,
    )
    qk_scales[..., 0] = Q_QUANT_SCALE
    qk_scales[..., 1] = K_QUANT_SCALE
    qkv = b300_project_qkv_gqa_d64_paired_unified_lowp_e4m3(
        tuple(b300_prepare_e4m3_projection_operand(rows)),
        tuple(b300_prepare_e4m3_projection_weight(qkv_weight)),
        qk_scales,
        packed_rope,
        batch=BATCH,
        seqlen=SEQUENCE,
        q_heads=Q_HEADS,
        kv_heads=KV_HEADS,
        publish_mxfp4_v=True,
        v_mxfp4_scale_2d=False,
        interleave_causal_kv=True,
        represented_backward=True,
        per_block_qk_scales=True,
        experimental_split_v_backward=False,
    )
    if (
        qkv.q_backward_fp8 is None
        or qkv.k_backward_fp8 is None
        or qkv.v_backward_fp8 is None
    ):
        raise RuntimeError("projection did not publish represented Q/K/V")

    output = torch.empty(
        BATCH,
        SEQUENCE,
        Q_HEADS,
        HEAD_DIM,
        device="cuda",
        dtype=torch.bfloat16,
    )
    lse = torch.empty(
        BATCH,
        Q_HEADS,
        1,
        SEQUENCE,
        device="cuda",
        dtype=torch.float32,
    )
    probability_scales = torch.full(
        (BATCH, Q_HEADS, SEQUENCE // 128, SEQUENCE),
        PSCALE_SENTINEL,
        device="cuda",
        dtype=torch.int32,
    )
    route = topology.get("route")
    if not isinstance(route, str) or not route:
        raise RuntimeError("the forward topology does not export its route")
    os.environ["TK_FA4_FP4PV_FWD_CONFIG"] = route
    mx.forward_hao_direct_fp4pv_with_p_scales(
        *qkv.forward_operands(),
        output,
        lse,
        probability_scales,
        0,
        True,
        True,
    )
    torch.cuda.synchronize()
    key_tile = torch.arange(
        SEQUENCE // 128, device="cuda", dtype=torch.int64
    ).view(1, 1, -1, 1)
    query = torch.arange(
        SEQUENCE, device="cuda", dtype=torch.int64
    ).view(1, 1, 1, -1)
    active_causal_tile = query >= key_tile * 128
    sentinel = probability_scales == PSCALE_SENTINEL
    active_sentinel_remaining = int(
        sentinel.expand_as(probability_scales)[
            active_causal_tile.expand_as(probability_scales)
        ].sum().item()
    )
    inactive_sentinel_remaining = int(
        sentinel.expand_as(probability_scales)[
            (~active_causal_tile).expand_as(probability_scales)
        ].sum().item()
    )
    if active_sentinel_remaining:
        raise RuntimeError(
            "D4ALL probability-scale publication left "
            f"{active_sentinel_remaining} active causal sentinel values"
        )
    if not bool(torch.isfinite(output.float()).all()):
        raise RuntimeError("D4ALL forward output is non-finite")
    if not bool(torch.isfinite(lse).all()):
        raise RuntimeError("D4ALL forward LSE is non-finite")

    grad_output = torch.randn(
        SEQUENCE, HIDDEN, device="cuda", dtype=torch.bfloat16
    ).mul_(0.1)
    out_weight_transposed = torch.randn(
        Q_HEADS * HEAD_DIM,
        HIDDEN,
        device="cuda",
        dtype=torch.bfloat16,
    ).mul_(0.02)
    grad_operand = tuple(b300_prepare_nvfp4_projection_operand(grad_output))
    out_weight_operand = tuple(
        b300_prepare_nvfp4_projection_weight(out_weight_transposed)
    )
    return {
        "qkv": qkv,
        "output": output,
        "lse": lse,
        "probability_scales": probability_scales,
        "grad_operand": grad_operand,
        "out_weight_operand": out_weight_operand,
        "metadata": {
            "qkv_projection_format": "e4m3",
            "qk_backward_source": "represented_nvfp4_codes_per_row_k16",
            "v_backward_source": "represented_mxfp4_codes",
            "v_mxfp4_scaling": "1d",
            "projection_weight_scaling": "2d",
            "q_quant_scale": Q_QUANT_SCALE,
            "k_quant_scale": K_QUANT_SCALE,
            "probability_scale_shape": list(probability_scales.shape),
            "probability_scale_dtype": str(probability_scales.dtype),
            "probability_scale_sha256": _tensor_sha256(probability_scales),
            "probability_scale_active_sentinel_remaining": (
                active_sentinel_remaining
            ),
            "probability_scale_inactive_sentinel_remaining": (
                inactive_sentinel_remaining
            ),
            "forward_output": _tensor_summary(output),
            "forward_lse": _tensor_summary(lse),
        },
    }


def _load_route_control(
    path: Path,
    sha256: str,
    size_bytes: int,
    *,
    optimized: bool,
    optimized_contract: str,
) -> Any:
    control = _load_control(
        fp8_p_storage="tmem",
        direct_tma_dkdv=True,
        detached_fp8_p_tmem=False,
        precomposed_control_source=path,
        precomposed_control_sha256=sha256,
        precomposed_control_bytes=size_bytes,
    )
    capability = bool(
        getattr(
            control,
            "TK_FORWARD_MX_PROBABILITY_REPLAY_PREEXP_NORMALIZER",
            False,
        )
    )
    balance = getattr(
        control,
        "TK_FORWARD_MX_PROBABILITY_REPLAY_PREEXP_BALANCE",
        None,
    )
    all_lane_exact = bool(getattr(control, ALL_LANE_EXACT_MARKER, False))
    shared_metadata = getattr(control, SHARED_METADATA_MARKER, None)
    log_classifier = getattr(control, LOG_CLASSIFIER_MARKER, None)
    if optimized:
        if optimized_contract == "signed_8_10":
            if (
                not capability
                or balance != PREEXP_BALANCE_MARKER
                or all_lane_exact
                or shared_metadata is not None
                or log_classifier is not None
            ):
                raise RuntimeError(
                    "optimized control must exclusively declare the "
                    "signed_8_10 pre-exp normalizer contract"
                )
        elif optimized_contract == "exact_math_all_lane":
            if (
                not all_lane_exact
                or capability
                or balance is not None
                or shared_metadata is not None
                or log_classifier is not None
            ):
                raise RuntimeError(
                    "optimized control must exclusively declare the "
                    "exact-math all-lane replay contract"
                )
        elif optimized_contract == "exact_math_shared_metadata":
            if (
                not all_lane_exact
                or capability
                or balance is not None
                or shared_metadata != SHARED_METADATA_CONTRACT
                or log_classifier is not None
            ):
                raise RuntimeError(
                    "optimized control must declare the exact-math shared "
                    f"metadata contract {SHARED_METADATA_CONTRACT!r}"
                )
        elif (
            not all_lane_exact
            or capability
            or balance is not None
            or shared_metadata is not None
            or log_classifier != LOG_CLASSIFIER_CONTRACT
        ):
            raise RuntimeError(
                "optimized control must declare the exact-math log "
                f"classifier contract {LOG_CLASSIFIER_CONTRACT!r}"
            )
    elif optimized_contract == "exact_math_log_classifier":
        if (
            capability
            or balance is not None
            or not all_lane_exact
            or shared_metadata is not None
            or log_classifier is not None
        ):
            raise RuntimeError(
                "log-classifier baseline must be the exact all-lane control"
            )
    elif (
        capability
        or balance is not None
        or all_lane_exact
        or shared_metadata is not None
        or log_classifier is not None
    ):
        raise RuntimeError(
            "baseline control unexpectedly declares an optimized contract"
        )
    return control


def _make_backward(control: Any, state: dict[str, Any]) -> CompiledGqaBackward:
    qkv = state["qkv"]
    stats = torch.zeros(
        BATCH,
        Q_HEADS,
        1,
        SEQUENCE,
        device="cuda",
        dtype=torch.float32,
    )
    placeholder_dout = torch.empty_like(qkv.q_backward_fp8)
    return CompiledGqaBackward(
        control,
        q=qkv.q_backward_fp8,
        k=qkv.k_backward_fp8,
        v=qkv.v_backward_fp8,
        o_or_sum=stats,
        dout=placeholder_dout,
        lse_or_scaled_lse=stats,
        q_heads=Q_HEADS,
        kv_heads=KV_HEADS,
        lowp=True,
        precomputed_stats=True,
        workspace_stats=True,
        scale_softmax=(HEAD_DIM**-0.5) / 16.0,
        exp2_degree=EXP2_DEGREE,
        exp2_period=EXP2_PERIOD,
        reuse_quantized_p=False,
        forward_mx_probability_replay=True,
        forward_mx_probability_scales=state["probability_scales"],
        use_forward_mx_probability_scales=True,
        fp8_ds_lift=FP8_DS_LIFT,
        lowp_do_stages=1,
        direct_tma_dkdv=True,
    )


def _project_dout(route: Route, state: dict[str, Any]) -> Any:
    bundle = b300_project_dout_unified_lowp_nvfp4(
        state["grad_operand"],
        state["out_weight_operand"],
        state["output"],
        state["lse"],
        batch=BATCH,
        seqlen=SEQUENCE,
        heads=Q_HEADS,
        store_bf16=False,
        publish_fp8_backward=True,
        publish_stats=True,
        stats_workspace=route.backward.workspace_torch,
        preexp_replay_lse=route.preexp_replay_lse,
    )
    if bundle.dout_backward_fp8 is None:
        raise RuntimeError(f"{route.label} producer did not publish E4M3 dO")
    route.last_projection = bundle
    return bundle


def _bind_dout(route: Route, dout: torch.Tensor) -> None:
    arguments = list(route.backward.arguments)
    arguments[8] = _attention_cute_tensor(
        route.control,
        dout,
        q_heads=Q_HEADS,
        kv_heads=KV_HEADS,
    )
    route.backward.arguments = tuple(arguments)


def _run_chain(route: Route, state: dict[str, Any]) -> None:
    # This is the production order: clear reductions, publish dO/statistics,
    # bind the new dO descriptor, then run backward without a second reset.
    route.backward.reset()
    bundle = _project_dout(route, state)
    _bind_dout(route, bundle.dout_backward_fp8)
    route.backward.run(reset=False)


def _workspace_pages(backward: CompiledGqaBackward) -> tuple[torch.Tensor, ...]:
    stats_numel = BATCH * Q_HEADS * SEQUENCE
    pages = backward.workspace_torch[: 2 * stats_numel * 4].view(torch.float32)
    dpsum = pages[:stats_numel].view(BATCH, Q_HEADS, 1, SEQUENCE)
    lse_page = pages[stats_numel:].view(BATCH, Q_HEADS, 1, SEQUENCE)
    return dpsum, lse_page


def _capture_correctness(
    route: Route, state: dict[str, Any]
) -> dict[str, torch.Tensor]:
    route.backward.reset()
    bundle = _project_dout(route, state)
    _bind_dout(route, bundle.dout_backward_fp8)
    dpsum, lse_page = _workspace_pages(route.backward)
    captured = {
        "dout": bundle.dout_backward_fp8.clone(),
        "dpsum": dpsum.clone(),
        "lse_page": lse_page.clone(),
    }
    route.backward.run(reset=False)
    torch.cuda.synchronize()
    captured.update(
        {
            "dq": route.backward.dq.clone(),
            "dk": route.backward.dk.clone(),
            "dv": route.backward.dv.clone(),
        }
    )
    torch.cuda.synchronize()
    return captured


def _normalizer_metrics(
    baseline_lse: torch.Tensor,
    optimized_normalizer: torch.Tensor,
) -> dict[str, Any]:
    log2_normalizer = baseline_lse.float() - LOG2_SIX
    high = log2_normalizer > 117.0
    balance = torch.where(
        high,
        torch.full_like(log2_normalizer, 8.0),
        torch.full_like(log2_normalizer, 10.0),
    )
    expected_magnitude = torch.exp2(log2_normalizer + balance)
    expected = torch.where(high, -expected_magnitude, expected_magnitude)
    actual = optimized_normalizer.float()
    actual_magnitude = actual.abs()
    relative_error = (actual - expected).abs() / expected_magnitude.clamp_min(
        torch.finfo(torch.float32).tiny
    )
    reconstructed_log2 = torch.log2(actual_magnitude) - balance
    log2_error = (reconstructed_log2 - log2_normalizer).abs()
    finite_nonzero = torch.isfinite(actual) & (actual_magnitude > 0.0)
    return {
        "finite_nonzero": bool(finite_nonzero.all()),
        "sign_tags_match": bool(torch.equal(torch.signbit(actual), high)),
        "high_balance_rows": int(high.sum()),
        "low_balance_rows": int((~high).sum()),
        "maximum_relative_error": float(relative_error.max()),
        "mean_relative_error": float(relative_error.mean()),
        "maximum_log2_abs_error": float(log2_error.max()),
        "mean_log2_abs_error": float(log2_error.mean()),
        "expected_minimum": float(expected.min()),
        "expected_maximum": float(expected.max()),
        "actual_minimum": float(actual.min()),
        "actual_maximum": float(actual.max()),
    }


def _correctness_report(
    baseline: dict[str, torch.Tensor],
    optimized: dict[str, torch.Tensor],
    baseline_repeat: dict[str, torch.Tensor],
    optimized_repeat: dict[str, torch.Tensor],
    state: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, bool]]:
    gradient_names = ("dq", "dk", "dv")
    cross_route_metrics = {
        name: _chunked_metrics(baseline[name], optimized[name])
        for name in gradient_names
    }
    cross_route_repeat_metrics = {
        name: _chunked_metrics(
            baseline_repeat[name], optimized_repeat[name]
        )
        for name in gradient_names
    }
    repeat_metrics = {
        "baseline": {
            name: _chunked_metrics(baseline[name], baseline_repeat[name])
            for name in gradient_names
        },
        "optimized": {
            name: _chunked_metrics(optimized[name], optimized_repeat[name])
            for name in gradient_names
        },
    }
    expected_baseline_lse = (
        state["lse"].float().mul(-LOG2_E).add(8.0)
    )
    baseline_lse_errors = (
        (baseline["lse_page"].float() - expected_baseline_lse).abs(),
        (
            baseline_repeat["lse_page"].float()
            - expected_baseline_lse
        ).abs(),
    )
    all_captures = (
        baseline,
        baseline_repeat,
        optimized,
        optimized_repeat,
    )
    checks = {
        "dout_bitwise_equal": bool(
            all(
                torch.equal(baseline["dout"], capture["dout"])
                for capture in all_captures[1:]
            )
        ),
        "dpsum_bitwise_equal": bool(
            all(
                torch.equal(baseline["dpsum"], capture["dpsum"])
                for capture in all_captures[1:]
            )
        ),
        "baseline_lse_matches_forward": bool(
            max(float(error.max()) for error in baseline_lse_errors)
            <= args.baseline_lse_max_abs_error
        ),
        "baseline_lse_repeat_bitwise_equal": bool(
            torch.equal(
                baseline["lse_page"], baseline_repeat["lse_page"]
            )
        ),
        "optimized_lse_repeat_bitwise_equal": bool(
            torch.equal(
                optimized["lse_page"], optimized_repeat["lse_page"]
            )
        ),
    }
    normalizer = None
    if args.optimized_contract == "signed_8_10":
        normalizer = {
            "primary": _normalizer_metrics(
                baseline["lse_page"], optimized["lse_page"]
            ),
            "repeat": _normalizer_metrics(
                baseline_repeat["lse_page"],
                optimized_repeat["lse_page"],
            ),
        }
        checks.update(
            {
                "optimized_normalizer_finite_nonzero": all(
                    value["finite_nonzero"]
                    for value in normalizer.values()
                ),
                "optimized_normalizer_sign_tags_match": all(
                    value["sign_tags_match"]
                    for value in normalizer.values()
                ),
                "optimized_normalizer_log2_matches": bool(
                    max(
                        value["maximum_log2_abs_error"]
                        for value in normalizer.values()
                    )
                    <= args.normalizer_log2_max_abs_error
                ),
                "optimized_normalizer_relative_matches": bool(
                    max(
                        value["maximum_relative_error"]
                        for value in normalizer.values()
                    )
                    <= args.normalizer_relative_max_error
                ),
            }
        )
    else:
        checks["optimized_lse_bitwise_equal"] = bool(
            all(
                torch.equal(baseline["lse_page"], capture["lse_page"])
                for capture in all_captures[1:]
            )
        )
    repeat_noise_envelopes: dict[str, Any] | None = None
    exact_math_contract = args.optimized_contract in (
        "exact_math_all_lane",
        "exact_math_shared_metadata",
        "exact_math_log_classifier",
    )
    if exact_math_contract:
        repeat_noise_envelopes = {}
    for name in gradient_names:
        cross_metrics = (
            cross_route_metrics[name],
            cross_route_repeat_metrics[name],
        )
        within_metrics = (
            repeat_metrics["baseline"][name],
            repeat_metrics["optimized"][name],
        )
        checks[f"{name}_finite"] = bool(
            all(
                torch.isfinite(capture[name].float()).all()
                for capture in all_captures
            )
        )
        if exact_math_contract:
            repeat_relative_l2 = max(
                metrics["relative_l2"] for metrics in within_metrics
            )
            repeat_cosine_distance = max(
                max(0.0, 1.0 - metrics["cosine"])
                for metrics in within_metrics
            )
            relative_l2_envelope = max(
                args.gradient_relative_l2_max,
                args.repeat_noise_multiplier * repeat_relative_l2,
            )
            cosine_distance_envelope = max(
                1.0 - args.gradient_cosine_min,
                args.repeat_noise_multiplier * repeat_cosine_distance,
            )
            maximum_cross_relative_l2 = max(
                metrics["relative_l2"] for metrics in cross_metrics
            )
            maximum_cross_cosine_distance = max(
                max(0.0, 1.0 - metrics["cosine"])
                for metrics in cross_metrics
            )
            checks[f"{name}_cosine"] = bool(
                maximum_cross_cosine_distance
                <= cosine_distance_envelope
            )
            checks[f"{name}_relative_l2"] = bool(
                maximum_cross_relative_l2 <= relative_l2_envelope
            )
            assert repeat_noise_envelopes is not None
            repeat_noise_envelopes[name] = {
                "formula": (
                    "max(fixed_threshold, repeat_noise_multiplier * "
                    "max(baseline_repeat, optimized_repeat))"
                ),
                "repeat_noise_multiplier": args.repeat_noise_multiplier,
                "within_route_max_relative_l2": repeat_relative_l2,
                "within_route_max_cosine_distance": (
                    repeat_cosine_distance
                ),
                "relative_l2_envelope": relative_l2_envelope,
                "cosine_distance_envelope": cosine_distance_envelope,
                "maximum_cross_route_relative_l2": (
                    maximum_cross_relative_l2
                ),
                "maximum_cross_route_cosine_distance": (
                    maximum_cross_cosine_distance
                ),
                "passed_relative_l2": checks[f"{name}_relative_l2"],
                "passed_cosine": checks[f"{name}_cosine"],
            }
        else:
            checks[f"{name}_cosine"] = bool(
                min(metrics["cosine"] for metrics in cross_metrics)
                >= args.gradient_cosine_min
            )
            checks[f"{name}_relative_l2"] = bool(
                max(metrics["relative_l2"] for metrics in cross_metrics)
                <= args.gradient_relative_l2_max
            )
    report = {
        "producer": {
            "dout_bitwise_equal": checks["dout_bitwise_equal"],
            "dpsum_bitwise_equal": checks["dpsum_bitwise_equal"],
            "baseline_lse_max_abs_error": max(
                float(error.max()) for error in baseline_lse_errors
            ),
            "baseline_lse_mean_abs_error": max(
                float(error.mean()) for error in baseline_lse_errors
            ),
            "baseline_lse_repeat_bitwise_equal": checks[
                "baseline_lse_repeat_bitwise_equal"
            ],
            "optimized_lse_repeat_bitwise_equal": checks[
                "optimized_lse_repeat_bitwise_equal"
            ],
            "optimized_lse_bitwise_equal": bool(
                all(
                    torch.equal(
                        baseline["lse_page"], capture["lse_page"]
                    )
                    for capture in all_captures[1:]
                )
            ),
            "preexp_normalizer": normalizer,
        },
        "gradients": {
            "cross_route_primary": cross_route_metrics,
            "cross_route_repeat": cross_route_repeat_metrics,
            "within_route_repeats": repeat_metrics,
            "repeat_noise_envelopes": repeat_noise_envelopes,
        },
        "thresholds": {
            "gradient_cosine_min": args.gradient_cosine_min,
            "gradient_relative_l2_max": args.gradient_relative_l2_max,
            "repeat_noise_multiplier": args.repeat_noise_multiplier,
            "baseline_lse_max_abs_error": args.baseline_lse_max_abs_error,
            "normalizer_log2_max_abs_error": (
                args.normalizer_log2_max_abs_error
            ),
            "normalizer_relative_max_error": (
                args.normalizer_relative_max_error
            ),
        },
    }
    return report, checks


def _route_policy(route: Route) -> dict[str, Any]:
    return {
        "preexp_replay_lse": route.preexp_replay_lse,
        "control_provenance": getattr(
            route.control, "TK_PRECOMPOSED_CONTROL_PROVENANCE", None
        ),
        "preexp_capability": bool(
            getattr(
                route.control,
                "TK_FORWARD_MX_PROBABILITY_REPLAY_PREEXP_NORMALIZER",
                False,
            )
        ),
        "preexp_balance": getattr(
            route.control,
            "TK_FORWARD_MX_PROBABILITY_REPLAY_PREEXP_BALANCE",
            None,
        ),
        "all_lane_exact_math": bool(
            getattr(route.control, ALL_LANE_EXACT_MARKER, False)
        ),
        "shared_metadata": getattr(
            route.control, SHARED_METADATA_MARKER, None
        ),
        "log_classifier": getattr(
            route.control, LOG_CLASSIFIER_MARKER, None
        ),
        "exp2_degree": route.backward.exp2_degree,
        "exp2_period": route.backward.exp2_period,
        "exp2_policy": route.backward.exp2_policy,
        "direct_tma_dkdv": route.backward.direct_tma_dkdv,
        "detached_fp8_p_tmem": route.backward.detached_fp8_p_tmem,
        "head_fast_raster": route.backward.head_fast_raster,
        "raster_policy": route.backward.raster_policy,
        "forward_mx_probability_replay": bool(
            route.backward.kernel.forward_mx_probability_replay
        ),
        "forward_mx_probability_scale_handoff": bool(
            route.backward.kernel.use_forward_mx_probability_scales
        ),
    }


def _run(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("expose exactly one GPU with CUDA_VISIBLE_DEVICES")
    torch.cuda.set_device(0)
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    minimum_free_bytes = int(args.minimum_free_gib * (1 << 30))
    if free_bytes < minimum_free_bytes:
        raise RuntimeError(
            f"visible GPU has {free_bytes / (1 << 30):.2f} GiB free; "
            f"the gate requires {args.minimum_free_gib:.2f} GiB"
        )

    projection_extension = _projection_extension_identity(
        expected_sha256=args.projection_extension_sha256,
        expected_bytes=args.projection_extension_bytes,
        require_preexp=args.optimized_contract == "signed_8_10",
    )
    mx_identity = _validated_file(
        args.mx_extension,
        expected_sha256=args.mx_sha256,
        expected_bytes=args.mx_bytes,
        label="MX forward extension",
    )
    control_identities = {
        route: _validated_file(
            getattr(args, f"{route}_control"),
            expected_sha256=getattr(args, f"{route}_control_sha256"),
            expected_bytes=getattr(args, f"{route}_control_bytes"),
            label=f"{route} control",
        )
        for route in PROVIDERS
    }
    optimized_patch_identity = None
    if args.optimized_patch is not None:
        optimized_patch_identity = _validated_file(
            args.optimized_patch,
            expected_sha256=args.optimized_patch_sha256,
            expected_bytes=args.optimized_patch_bytes,
            label="optimized patch",
        )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    mx = _load_extension(args.mx_extension.resolve(), args.mx_module)
    if not hasattr(mx, "forward_hao_direct_fp4pv_with_p_scales"):
        raise RuntimeError(
            "the MX forward extension lacks probability-scale publication"
        )
    topology = dict(mx.read_hao_direct_topology())
    _validate_forward_topology(topology)
    state = _make_projection_state(mx, topology)

    baseline_control = _load_route_control(
        args.baseline_control,
        args.baseline_control_sha256,
        args.baseline_control_bytes,
        optimized=False,
        optimized_contract=args.optimized_contract,
    )
    optimized_control = _load_route_control(
        args.optimized_control,
        args.optimized_control_sha256,
        args.optimized_control_bytes,
        optimized=True,
        optimized_contract=args.optimized_contract,
    )
    routes = {
        "baseline": Route(
            "baseline",
            baseline_control,
            _make_backward(baseline_control, state),
            False,
        ),
        "optimized": Route(
            "optimized",
            optimized_control,
            _make_backward(optimized_control, state),
            args.optimized_contract == "signed_8_10",
        ),
    }

    baseline_capture = _capture_correctness(routes["baseline"], state)
    optimized_capture = _capture_correctness(routes["optimized"], state)
    baseline_repeat_capture = _capture_correctness(
        routes["baseline"], state
    )
    optimized_repeat_capture = _capture_correctness(
        routes["optimized"], state
    )
    correctness, checks = _correctness_report(
        baseline_capture,
        optimized_capture,
        baseline_repeat_capture,
        optimized_repeat_capture,
        state,
        args,
    )

    producer_timing = _time_rotated(
        {
            name: (lambda route=route: _project_dout(route, state))
            for name, route in routes.items()
        },
        warmups=args.warmups,
        samples=args.samples,
    )
    # Republish and bind once so backward-only timings never consume the
    # other route's statistics or a stale output allocation.
    for route in routes.values():
        bundle = _project_dout(route, state)
        _bind_dout(route, bundle.dout_backward_fp8)
    torch.cuda.synchronize()
    backward_timing = _time_rotated(
        {
            name: (lambda route=route: route.backward.run(reset=True))
            for name, route in routes.items()
        },
        warmups=args.warmups,
        samples=args.samples,
    )
    chain_timing = _time_rotated(
        {
            name: (lambda route=route: _run_chain(route, state))
            for name, route in routes.items()
        },
        warmups=args.warmups,
        samples=args.samples,
    )
    torch.cuda.synchronize()

    script_path = Path(__file__).resolve()
    properties = torch.cuda.get_device_properties(0)
    payload = {
        "schema": "gqa_d64_replay_optimization_ab_v3",
        "passed": bool(all(checks.values())),
        "checks": checks,
        "shape": {
            "batch": BATCH,
            "sequence": SEQUENCE,
            "q_heads": Q_HEADS,
            "kv_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "hidden": HIDDEN,
        },
        "policy": {
            "causal": True,
            "qkv_projection_format": "e4m3",
            "represented_per_block_qk": True,
            "represented_mxfp4_v": True,
            "v_mxfp4_scaling": "1d",
            "projection_weight_scaling": "2d",
            "forward_probability_replay": True,
            "forward_probability_scale_handoff": True,
            "optimized_contract": args.optimized_contract,
            "correctness_captures_per_route": 2,
            "correctness_capture_order": [
                "baseline",
                "optimized",
                "baseline",
                "optimized",
            ],
            "repeat_noise_multiplier": args.repeat_noise_multiplier,
            "fp8_ds_lift": FP8_DS_LIFT,
            "exp2_degree": EXP2_DEGREE,
            "exp2_period": EXP2_PERIOD,
            "reuse_quantized_p": False,
            "timing_order": "rotated baseline/optimized",
            "timing_synchronization": "per-provider",
            "timing_warmups": args.warmups,
            "timing_samples": args.samples,
        },
        "correctness": correctness,
        "timing": {
            "producer_only": producer_timing,
            "backward_only_including_reset": backward_timing,
            "training_order_reset_producer_backward": chain_timing,
        },
        "routes": {
            name: _route_policy(route) for name, route in routes.items()
        },
        "forward": {
            "extension": mx_identity,
            "module": args.mx_module,
            "topology": topology,
            "state": state["metadata"],
        },
        "provenance": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "command": [sys.executable, *sys.argv],
            "script": {
                "path": str(script_path),
                "sha256": _sha256(script_path),
                "bytes": script_path.stat().st_size,
            },
            "git": _git_identity(_BOOTSTRAP_ROOT),
            "projection_extension": projection_extension,
            "controls": control_identities,
            "optimized_patch": optimized_patch_identity,
            "host": {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python": sys.version,
                "torch": torch.__version__,
            },
            "gpu": {
                "name": properties.name,
                "major": properties.major,
                "minor": properties.minor,
                "multiprocessor_count": properties.multi_processor_count,
                "total_bytes": total_bytes,
                "free_bytes_at_start": free_bytes,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
            },
        },
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(serialized)
        handle.write("\n")
    print(serialized)
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(_run())
