#!/usr/bin/env python3
"""Run a matched causal BF16/low-precision backward matrix on SM100.

The comparison deliberately starts from one represented E4M3 Q/K/V/dO
state.  The BF16 control decodes that state and the retained low-precision
routes consume the bytes directly.  Consequently, the reported gradient
error belongs to backward rather than to a different forward approximation.

The primary timing includes every destination/workspace clear required by a
route.  Clear-only and kernel-after-clear measurements are also emitted so a
future handoff fusion cannot silently change the comparison boundary.

This harness does *not* claim to implement the route-complete MXFP4
probability replay used by the final MX-PV training experiments.  The clean
source line has exploratory MX replay probes, but no retained adapter from a
forward-published E8M0 probability-scale page into ``CompiledGqaBackward``.
``mx_exact_replay`` therefore remains an explicit unavailable route in the
registry below; it must not be substituted with the ordinary E4M3 probability
reconstruction measured by ``retained_lowp``.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import hmac
import json
import math
import os
import platform
import re
import stat
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

try:
    from tk_fa4.lowp_fa4_bwd.backward_policy import (
        BACKWARD_EXP2_POLICY_VERSION,
        BACKWARD_P_TMEM_POLICY_VERSION,
        BACKWARD_RASTER_POLICY_VERSION,
        D64_DETACHED_FP8_P_TMEM_VERIFIED_SHAPES,
        D64_HEAD_FAST_RASTER_VERIFIED_SHAPES,
        D64_SELECTIVE_EXP2_MIN_SEQUENCE,
        D64_SELECTIVE_EXP2_VERIFIED_SHAPES,
        resolve_backward_exp2_policy,
        resolve_backward_probability_tmem_policy,
        resolve_backward_raster_policy,
    )
except ModuleNotFoundError:  # direct script execution without repo on sys.path
    from backward_policy import (  # type: ignore[no-redef]
        BACKWARD_EXP2_POLICY_VERSION,
        BACKWARD_P_TMEM_POLICY_VERSION,
        BACKWARD_RASTER_POLICY_VERSION,
        D64_DETACHED_FP8_P_TMEM_VERIFIED_SHAPES,
        D64_HEAD_FAST_RASTER_VERIFIED_SHAPES,
        D64_SELECTIVE_EXP2_MIN_SEQUENCE,
        D64_SELECTIVE_EXP2_VERIFIED_SHAPES,
        resolve_backward_exp2_policy,
        resolve_backward_probability_tmem_policy,
        resolve_backward_raster_policy,
    )

SCHEMA = "fp4_fa4_causal_backward_matrix_v3"
CONTROL_PROVENANCE_SCHEMA = "fp4_fa4_backward_control_provenance_v1"
GIB = 1 << 30
MAX_PRECOMPOSED_CONTROL_BYTES = 8 * 1024 * 1024
S4096_PRECOMPOSED_CONTROL_SHAPE = {
    "batch": 1,
    "sequence": 4096,
    "q_heads": 32,
    "kv_heads": 8,
    "head_dim": 64,
}


@dataclass(frozen=True, order=True)
class Shape:
    """One fixed-length causal GQA problem."""

    sequence: int
    q_heads: int
    kv_heads: int
    head_dim: int
    batch: int = 1

    def validate(self) -> None:
        if self.batch != 1:
            raise ValueError("the retained causal GQA extension requires B=1")
        if self.sequence <= 0 or self.sequence % 128:
            raise ValueError("sequence must be a positive multiple of 128")
        if self.q_heads <= 0 or self.kv_heads <= 0:
            raise ValueError("head counts must be positive")
        if self.q_heads % self.kv_heads:
            raise ValueError("q-heads must be divisible by kv-heads")
        if self.head_dim not in (64, 128):
            raise ValueError("the retained extension supports D64 and D128")

    def as_dict(self) -> dict[str, int]:
        return {
            "batch": self.batch,
            "sequence": self.sequence,
            "q_heads": self.q_heads,
            "kv_heads": self.kv_heads,
            "head_dim": self.head_dim,
            "gqa_ratio": self.q_heads // self.kv_heads,
        }


@dataclass(frozen=True)
class PrecomposedControlSpec:
    """One caller-authenticated control with an intentionally narrow scope."""

    source: Path
    sha256: str
    size_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": "precomposed",
            "source": {
                "path": str(self.source),
                "sha256": self.sha256,
                "bytes": self.size_bytes,
            },
            "scope": dict(S4096_PRECOMPOSED_CONTROL_SHAPE),
            "route": "retained_lowp",
            "authentication": (
                "SHA256 and byte size verified before CUDA initialization and "
                "reverified immediately before import"
            ),
        }


@dataclass
class Runtime:
    torch: Any
    compiled_backward: Any
    load_control: Callable[..., Any]
    flash_attention: Callable[..., Any]
    lowp_extension_path: Path


@dataclass
class RepresentedState:
    q_fp8: Any
    k_fp8: Any
    v_fp8: Any
    dout_fp8: Any
    q_bf16: Any
    k_bf16: Any
    v_bf16: Any
    dout_bf16: Any
    output_bf16: Any
    lse_bh1s: Any
    direct_dpsum: Any
    direct_lse_log2: Any


@dataclass
class BuiltRoute:
    name: str
    backward: Any
    decode_scale: float
    policy: dict[str, Any]
    control_provenance: dict[str, Any]


@dataclass
class CompiledCase:
    """One pointer-stable BF16/lowp pair reused across accuracy seeds."""

    state: RepresentedState
    reference: BuiltRoute
    route: BuiltRoute
    seed: int


class RouteUnavailable(RuntimeError):
    """Raised when a named numerical contract is not reconstructable."""


ROUTE_CAPABILITIES: dict[str, dict[str, Any]] = {
    "retained_lowp": {
        "available": True,
        "scope": "isolated_attention_backward_shared_core",
        "input_contract": "represented_e4m3_q_k_v_dout_x4",
        "probability_contract": (
            "score/LSE reconstruction with retained E4M3 probability path"
        ),
        "supported_head_dims": [64, 128],
        "d64_exp2_contract": (
            "automatic d1/p2 only for measured S/Hq/Hkv shapes: "
            "4096/16/4, 4096/32/8, 4096/64/16, 8192/32/8, and "
            "16384/32/8; all other shapes retain native d2/p0"
        ),
        "d64_raster_contract": (
            "the measured one-lane direct-TMA TMEM S8192 and S16384 "
            "Hq/Hkv=32/8 routes use the head-fast/key-wavefront raster; named "
            "comparison controls inherit that raster so they isolate only "
            "their advertised direct-TMA, EX2, or detached-P change"
        ),
        "d64_probability_tmem_contract": (
            "S8192 and S16384 Hq/Hkv=32/8 store FP8 P in the unused D64 "
            "TMEM tail; other retained shapes preserve score aliasing"
        ),
    },
    "retained_lowp_no_direct_tma": {
        "available": True,
        "scope": "isolated_attention_backward_shared_core",
        "input_contract": "represented_e4m3_q_k_v_dout_x4",
        "probability_contract": (
            "score/LSE reconstruction with retained E4M3 probability path"
        ),
        "supported_head_dims": [64],
        "comparison_control_for": "retained_lowp",
        "only_policy_change": {
            "control_module_direct_tma_dkdv": [True, False],
            "compiled_backward_direct_tma_dkdv": [True, False],
        },
        "reason_d64_only": (
            "D128 retained_lowp already uses direct_tma_dkdv=False; this "
            "control isolates the D64 direct-TMA dK/dV epilogue only."
        ),
    },
    "retained_lowp_detached_p": {
        "available": True,
        "scope": "isolated_attention_backward_shared_core",
        "input_contract": "represented_e4m3_q_k_v_dout_x4",
        "probability_contract": (
            "score/LSE reconstruction with retained E4M3 probability path"
        ),
        "supported_head_dims": [64],
        "supported_shapes": [
            {
                "sequence": sequence,
                "q_heads": q_heads,
                "kv_heads": kv_heads,
            }
            for sequence, q_heads, kv_heads in (
                D64_DETACHED_FP8_P_TMEM_VERIFIED_SHAPES
            )
        ],
        "comparison_control_for": "retained_lowp_alias_p",
        "explicit_detached_fp8_p_tmem": True,
        "only_policy_change": {
            "control_module_detached_fp8_p_tmem": [False, True],
        },
        "reason_d64_only": (
            "This control moves FP8 P from the score alias to unused D64 "
            "TMEM while inheriting the retained route's physical raster."
        ),
    },
    "retained_lowp_alias_p": {
        "available": True,
        "scope": "isolated_attention_backward_shared_core",
        "input_contract": "represented_e4m3_q_k_v_dout_x4",
        "probability_contract": (
            "score/LSE reconstruction with retained E4M3 probability path"
        ),
        "supported_head_dims": [64],
        "supported_shapes": [
            {
                "sequence": sequence,
                "q_heads": q_heads,
                "kv_heads": kv_heads,
            }
            for sequence, q_heads, kv_heads in (
                D64_DETACHED_FP8_P_TMEM_VERIFIED_SHAPES
            )
        ],
        "comparison_control_for": "retained_lowp",
        "explicit_detached_fp8_p_tmem": False,
        "only_policy_change": {
            "control_module_detached_fp8_p_tmem": [True, False],
        },
        "reason_d64_only": (
            "This control retains score-aliasing FP8 P while inheriting the "
            "production raster so detached-P speedup remains measurable."
        ),
    },
    "retained_lowp_native_exp2": {
        "available": True,
        "scope": "isolated_attention_backward_shared_core",
        "input_contract": "represented_e4m3_q_k_v_dout_x4",
        "probability_contract": (
            "score/LSE reconstruction with retained E4M3 probability path"
        ),
        "supported_head_dims": [64],
        "comparison_control_for": "retained_lowp",
        "explicit_exp2_policy": {"degree": 2, "period": 0},
        "reason_d64_only": (
            "This control forces the pre-dispatch native EX2 implementation."
        ),
    },
    "retained_lowp_exp2_d1_p2": {
        "available": True,
        "scope": "isolated_attention_backward_shared_core",
        "input_contract": "represented_e4m3_q_k_v_dout_x4",
        "probability_contract": (
            "score/LSE reconstruction with retained E4M3 probability path"
        ),
        "supported_head_dims": [64],
        "comparison_control_for": "retained_lowp_native_exp2",
        "explicit_exp2_policy": {"degree": 1, "period": 2},
        "only_policy_change": {
            "exp2_degree": [2, 1],
            "exp2_period": [0, 2],
        },
        "reason_d64_only": (
            "This is a controlled long-sequence D64 screen of the selective "
            "packed-ALU EX2 cadence; it is not a D128 policy."
        ),
    },
    "retained_lowp_exp2_d1_p3": {
        "available": True,
        "scope": "isolated_attention_backward_shared_core",
        "input_contract": "represented_e4m3_q_k_v_dout_x4",
        "probability_contract": (
            "score/LSE reconstruction with retained E4M3 probability path"
        ),
        "supported_head_dims": [64],
        "comparison_control_for": "retained_lowp_native_exp2",
        "explicit_exp2_policy": {"degree": 1, "period": 3},
        "only_policy_change": {
            "exp2_degree": [2, 1],
            "exp2_period": [0, 3],
        },
        "reason_d64_only": (
            "This is a controlled long-sequence D64 screen of the selective "
            "packed-ALU EX2 cadence; it is not a D128 policy."
        ),
    },
    "mx_exact_replay": {
        "available": False,
        "scope": "route_complete_mxfp4_probability_replay_and_backward",
        "reason": (
            "No retained callable adapter in the clean source line consumes "
            "the causal forward's published MXFP4 probability payload/E8M0 "
            "scale page at the CompiledGqaBackward boundary. Exploratory "
            "fp4_pv_experiments.py probes are not a production substitute."
        ),
        "required_adapter_contract": {
            "inputs": [
                "represented E4M3 Q/K/V/dO",
                "forward MXFP4 probability payload or deterministic replay",
                "forward-published E8M0 block32 probability scales",
                "forward LSE/statistics",
            ],
            "outputs": ["BF16 dQ", "BF16 dK", "BF16 dV"],
            "required_methods": ["reset", "run"],
            "timing_boundary": (
                "replay/publication handoff plus every required clear and "
                "attention backward"
            ),
        },
    },
}


def _parse_positive_ints(value: str, *, argument: str) -> tuple[int, ...]:
    parsed: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            number = int(item)
        except ValueError as error:
            raise ValueError(f"{argument} contains a non-integer: {item!r}") from error
        if number <= 0:
            raise ValueError(f"{argument} values must be positive")
        parsed.append(number)
    if not parsed:
        raise ValueError(f"{argument} must contain at least one value")
    return tuple(dict.fromkeys(parsed))


def _parse_head_pairs(value: str) -> tuple[tuple[int, int], ...]:
    pairs: list[tuple[int, int]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        fields = item.split("/")
        if len(fields) != 2:
            raise ValueError(
                "head pairs must use Q/KV notation, for example 32/8"
            )
        try:
            q_heads, kv_heads = (int(field) for field in fields)
        except ValueError as error:
            raise ValueError(f"invalid head pair: {item!r}") from error
        if q_heads <= 0 or kv_heads <= 0 or q_heads % kv_heads:
            raise ValueError(
                f"head pair {item!r} must be positive with integral GQA ratio"
            )
        pairs.append((q_heads, kv_heads))
    if not pairs:
        raise ValueError("--head-pairs must contain at least one Q/KV pair")
    return tuple(dict.fromkeys(pairs))


def _parse_routes(value: str) -> tuple[str, ...]:
    routes = tuple(
        dict.fromkeys(item.strip() for item in value.split(",") if item.strip())
    )
    if not routes:
        raise ValueError("--routes must contain at least one route")
    unknown = sorted(set(routes) - set(ROUTE_CAPABILITIES))
    if unknown:
        raise ValueError(f"unknown routes: {', '.join(unknown)}")
    return routes


def _validate_route_shape_support(
    routes: Sequence[str],
    shapes: Sequence[Shape],
) -> None:
    """Reject shape-specific controls before importing a CUDA runtime."""

    requested_head_dims = {shape.head_dim for shape in shapes}
    for route in routes:
        capability = ROUTE_CAPABILITIES[route]
        if not capability["available"]:
            continue
        supported = capability.get("supported_head_dims")
        if supported is None:
            continue
        unsupported = sorted(requested_head_dims - set(supported))
        if unsupported:
            supported_text = ", ".join(str(value) for value in supported)
            unsupported_text = ", ".join(str(value) for value in unsupported)
            raise ValueError(
                f"route {route!r} supports only head dimensions "
                f"{supported_text}; incompatible requested head dimensions: "
                f"{unsupported_text}"
            )
        supported_shapes = capability.get("supported_shapes")
        if supported_shapes is None:
            continue
        supported_shape_keys = {
            (
                int(candidate["sequence"]),
                int(candidate["q_heads"]),
                int(candidate["kv_heads"]),
            )
            for candidate in supported_shapes
        }
        incompatible_shapes = [
            shape
            for shape in shapes
            if (shape.sequence, shape.q_heads, shape.kv_heads)
            not in supported_shape_keys
        ]
        if incompatible_shapes:
            requested_text = ", ".join(
                f"{shape.sequence}/{shape.q_heads}/{shape.kv_heads}"
                for shape in incompatible_shapes
            )
            supported_text = ", ".join(
                f"{sequence}/{q_heads}/{kv_heads}"
                for sequence, q_heads, kv_heads in sorted(
                    supported_shape_keys
                )
            )
            raise ValueError(
                f"route {route!r} supports only S/Hq/Hkv shapes "
                f"{supported_text}; incompatible requested shapes: "
                f"{requested_text}"
            )


def _build_shapes(
    sequences: Sequence[int],
    head_dims: Sequence[int],
    head_pairs: Sequence[tuple[int, int]],
) -> tuple[Shape, ...]:
    shapes = []
    for head_dim in head_dims:
        for sequence in sequences:
            for q_heads, kv_heads in head_pairs:
                shape = Shape(
                    sequence=sequence,
                    q_heads=q_heads,
                    kv_heads=kv_heads,
                    head_dim=head_dim,
                )
                shape.validate()
                shapes.append(shape)
    return tuple(shapes)


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot take a percentile of an empty sequence")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("percentile fraction must lie in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(samples: Sequence[float]) -> dict[str, Any]:
    values = [float(value) for value in samples]
    if not values:
        raise ValueError("timing summary requires at least one sample")
    return {
        "median_us": statistics.median(values),
        "minimum_us": min(values),
        "maximum_us": max(values),
        "mean_us": statistics.fmean(values),
        "stdev_us": statistics.stdev(values) if len(values) > 1 else 0.0,
        "p05_us": _percentile(values, 0.05),
        "p95_us": _percentile(values, 0.95),
        "samples_us": values,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    display = resolved
    if root is not None:
        try:
            display = resolved.relative_to(root.resolve())
        except ValueError:
            pass
    result: dict[str, Any] = {
        "path": str(display),
        "exists": resolved.is_file(),
    }
    if resolved.is_file():
        result["size_bytes"] = resolved.stat().st_size
        result["sha256"] = _sha256(resolved)
    return result


def _source_content_identity(path: Path | str) -> dict[str, Any]:
    """Return a stable identity without recording an ephemeral module path."""

    source = Path(path)
    if not source.is_file():
        raise RuntimeError(f"backward control source is unavailable: {source}")
    return {
        "bytes": source.stat().st_size,
        "sha256": _sha256(source),
    }


def _resolve_precomposed_control_spec(
    source: Path | None,
    expected_sha256: str | None,
    expected_bytes: int | None,
) -> PrecomposedControlSpec | None:
    """Authenticate the optional S4096 control without importing its code."""

    supplied = (
        source is not None,
        expected_sha256 is not None,
        expected_bytes is not None,
    )
    if not any(supplied):
        return None
    if not all(supplied):
        raise ValueError(
            "--backward-control-source, --backward-control-sha256, and "
            "--backward-control-bytes must be supplied together"
        )
    assert source is not None
    assert expected_sha256 is not None
    assert expected_bytes is not None
    if re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256) is None:
        raise ValueError("--backward-control-sha256 must contain 64 hex digits")
    if expected_bytes <= 0:
        raise ValueError("--backward-control-bytes must be positive")
    if expected_bytes > MAX_PRECOMPOSED_CONTROL_BYTES:
        raise ValueError("precomposed backward control exceeds the 8 MiB limit")
    try:
        source_stat = source.lstat()
    except OSError as error:
        raise ValueError(f"unable to stat backward control: {source}") from error
    if not stat.S_ISREG(source_stat.st_mode):
        raise ValueError(
            "precomposed backward control must be a regular, non-symlink file"
        )
    resolved = source.resolve(strict=True)
    if resolved.suffix != ".py":
        raise ValueError("precomposed backward control must be a Python source")
    if source_stat.st_size != expected_bytes:
        raise ValueError(
            "precomposed backward control size mismatch: "
            f"expected {expected_bytes}, found {source_stat.st_size}"
        )
    actual_sha256 = _sha256(resolved)
    normalized_sha256 = expected_sha256.lower()
    if not hmac.compare_digest(actual_sha256, normalized_sha256):
        raise ValueError(
            "precomposed backward control SHA256 mismatch: "
            f"expected {normalized_sha256}, found {actual_sha256}"
        )
    return PrecomposedControlSpec(
        source=resolved,
        sha256=actual_sha256,
        size_bytes=expected_bytes,
    )


def _precomposed_control_applies(
    spec: PrecomposedControlSpec | None,
    route: str,
    shape: Shape,
) -> bool:
    return (
        spec is not None
        and route == "retained_lowp"
        and all(
            getattr(shape, field) == expected
            for field, expected in S4096_PRECOMPOSED_CONTROL_SHAPE.items()
        )
    )


def _planned_control_binding(
    spec: PrecomposedControlSpec | None,
    route: str,
    shape: Shape,
) -> dict[str, Any]:
    reference = {
        "mode": "generated_patch_chain",
        "consumer": "cute_bf16_reference",
        "final_source_identity_recorded_at_runtime": True,
    }
    if _precomposed_control_applies(spec, route, shape):
        assert spec is not None
        candidate = spec.as_dict()
    else:
        candidate = {
            "mode": "generated_patch_chain",
            "consumer": route,
            "final_source_identity_recorded_at_runtime": True,
        }
        if spec is not None:
            candidate["precomposed_control_not_selected"] = (
                "the authenticated artifact is restricted to retained_lowp "
                "at B1/S4096/Hq32/Hkv8/D64; this case is composed locally"
            )
    return {"reference": reference, "route": candidate}


def _run_git(root: Path, *arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _source_provenance(
    root: Path,
    script: Path,
    extension: Path | None,
) -> dict[str, Any]:
    patch_root = root / "tk_fa4" / "lowp_fa4_bwd"
    control_source = (
        root
        / "qutlass"
        / "third_party"
        / "cutlass"
        / "examples"
        / "python"
        / "CuTeDSL"
        / "blackwell"
        / "fmha_bwd.py"
    )
    inputs = [
        control_source,
        patch_root / "backward_policy.py",
        patch_root / "profile_gqa_d128_chain.py",
        patch_root / "tune_d64_gqa_cute.py",
        root / "flash-attention" / "flash_attn" / "cute" / "interface.py",
    ]
    inputs.extend(sorted(patch_root.glob("d64_gqa_*.patch")))
    provenance = {
        "git": {
            "head": _run_git(root, "rev-parse", "HEAD"),
            "branch": _run_git(root, "branch", "--show-current"),
            "dirty": bool(_run_git(root, "status", "--porcelain")),
        },
        "harness": _artifact(script, root=root),
        "control_inputs": [_artifact(path, root=root) for path in inputs],
        "lowp_extension": (
            _artifact(extension) if extension is not None else None
        ),
    }
    return provenance


def _host_available_bytes() -> int | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return None
    for line in meminfo.read_text().splitlines():
        if line.startswith("MemAvailable:"):
            fields = line.split()
            return int(fields[1]) * 1024
    return None


def _estimated_live_tensor_bytes(shape: Shape) -> int:
    """Conservative tensor/workspace estimate, excluding compiler memory."""

    q_elements = shape.batch * shape.sequence * shape.q_heads * shape.head_dim
    kv_elements = shape.batch * shape.sequence * shape.kv_heads * shape.head_dim
    stats_elements = shape.batch * shape.sequence * shape.q_heads
    # Represented inputs, decoded controls, outputs, two backward instances,
    # GQA partials, and scratch.  The multiplier intentionally overestimates
    # the explicit tensors while the fixed reserve in _memory_snapshot covers
    # CUDA/CuTe compiler allocations and allocator fragmentation.
    return 20 * q_elements + 24 * kv_elements + 32 * stats_elements


def _memory_snapshot(runtime: Runtime, shape: Shape) -> dict[str, Any]:
    torch = runtime.torch
    free_device, total_device = torch.cuda.mem_get_info()
    available_host = _host_available_bytes()
    return {
        "device_free_bytes": int(free_device),
        "device_total_bytes": int(total_device),
        "host_available_bytes": available_host,
        "estimated_live_tensor_bytes": _estimated_live_tensor_bytes(shape),
    }


def _guard_memory(
    snapshot: dict[str, Any],
    *,
    minimum_device_free_gib: float,
    minimum_host_free_gib: float,
) -> None:
    device_floor = int(minimum_device_free_gib * GIB)
    host_floor = int(minimum_host_free_gib * GIB)
    estimated = int(snapshot["estimated_live_tensor_bytes"])
    # Retain the requested reserve after the conservative live-tensor estimate.
    required_device = device_floor + estimated
    if int(snapshot["device_free_bytes"]) < required_device:
        raise RuntimeError(
            "device-memory guard refused the shape: "
            f"free={snapshot['device_free_bytes'] / GIB:.2f} GiB, "
            f"estimated tensors={estimated / GIB:.2f} GiB, "
            f"required reserve={minimum_device_free_gib:.2f} GiB"
        )
    available_host = snapshot["host_available_bytes"]
    if available_host is not None and int(available_host) < host_floor:
        raise RuntimeError(
            "host-memory guard refused compilation: "
            f"available={int(available_host) / GIB:.2f} GiB, "
            f"required reserve={minimum_host_free_gib:.2f} GiB"
        )


def _load_runtime(root: Path) -> Runtime:
    import torch

    import tk_fa4.interface as tk_interface
    from tk_fa4.lowp_fa4_bwd.profile_gqa_d128_chain import (
        CompiledGqaBackward,
    )
    from tk_fa4.lowp_fa4_bwd.tune_d64_gqa_cute import _load_control

    flash_root = root / "flash-attention"
    sys.path.insert(0, str(flash_root))
    try:
        from flash_attn.cute.interface import flash_attn_func
    finally:
        sys.path.pop(0)

    extension = Path(tk_interface._C_b300_lowp_bwd.__file__).resolve()
    return Runtime(
        torch=torch,
        compiled_backward=CompiledGqaBackward,
        load_control=_load_control,
        flash_attention=flash_attn_func,
        lowp_extension_path=extension,
    )


class ControlCache:
    def __init__(
        self,
        runtime: Runtime,
        s4096_precomposed_control: PrecomposedControlSpec | None = None,
    ) -> None:
        self.runtime = runtime
        self.s4096_precomposed_control = s4096_precomposed_control
        self._controls: dict[str, Any] = {}
        self._construction: dict[int, dict[str, Any]] = {}

    def _remember(
        self,
        control: Any,
        *,
        fp8_p_storage: str,
        direct_tma_dkdv: bool,
        detached_fp8_p_tmem: bool,
        mode: str,
    ) -> Any:
        self._construction[id(control)] = {
            "fp8_p_storage": fp8_p_storage,
            "direct_tma_dkdv": direct_tma_dkdv,
            "detached_fp8_p_tmem": detached_fp8_p_tmem,
            "owner_fused_dq_scale": False,
            "mode": mode,
        }
        return control

    def provenance(
        self,
        control: Any,
        *,
        shape: Shape,
        consumer: str,
    ) -> dict[str, Any]:
        construction = self._construction.get(id(control))
        if construction is None:
            raise RuntimeError("backward control is not owned by this cache")
        generated_identity = _source_content_identity(control.__file__)
        authenticated = getattr(
            control,
            "TK_PRECOMPOSED_CONTROL_PROVENANCE",
            None,
        )
        if authenticated is None:
            result = {
                "schema": CONTROL_PROVENANCE_SCHEMA,
                "mode": "generated_patch_chain",
                "generated_source": generated_identity,
                "construction": copy.deepcopy(construction),
            }
        else:
            result = copy.deepcopy(authenticated)
            result["schema"] = CONTROL_PROVENANCE_SCHEMA
            result["loaded_verified_copy"] = generated_identity
            result["construction"] = copy.deepcopy(construction)
        result["binding"] = {
            "consumer": consumer,
            "shape": shape.as_dict(),
        }
        return result

    def bf16(self) -> Any:
        if "bf16" not in self._controls:
            self._controls["bf16"] = self._remember(
                self.runtime.load_control(
                    fp8_p_storage="shared",
                    direct_tma_dkdv=False,
                ),
                fp8_p_storage="shared",
                direct_tma_dkdv=False,
                detached_fp8_p_tmem=False,
                mode="generated_patch_chain",
            )
        return self._controls["bf16"]

    def retained_lowp(
        self,
        head_dim: int,
        *,
        direct_tma_dkdv: bool,
        detached_fp8_p_tmem: bool = False,
        shape: Shape | None = None,
        allow_s4096_precomposed: bool = False,
    ) -> Any:
        use_precomposed = (
            allow_s4096_precomposed
            and shape is not None
            and _precomposed_control_applies(
                self.s4096_precomposed_control,
                "retained_lowp",
                shape,
            )
            and direct_tma_dkdv
            and not detached_fp8_p_tmem
        )
        mode = "precomposed" if use_precomposed else "generated_patch_chain"
        key = (
            f"retained_d{head_dim}_direct_tma_{int(direct_tma_dkdv)}"
            f"_detached_p_{int(detached_fp8_p_tmem)}_{mode}"
        )
        if key not in self._controls:
            if head_dim == 64:
                load_kwargs: dict[str, Any] = {
                    "fp8_p_storage": "tmem",
                    "direct_tma_dkdv": direct_tma_dkdv,
                    "detached_fp8_p_tmem": detached_fp8_p_tmem,
                }
                if use_precomposed:
                    spec = self.s4096_precomposed_control
                    assert spec is not None
                    load_kwargs.update(
                        {
                            "precomposed_control_source": spec.source,
                            "precomposed_control_sha256": spec.sha256,
                            "precomposed_control_bytes": spec.size_bytes,
                        }
                    )
                control = self.runtime.load_control(
                    **load_kwargs,
                )
                self._controls[key] = self._remember(
                    control,
                    fp8_p_storage="tmem",
                    direct_tma_dkdv=direct_tma_dkdv,
                    detached_fp8_p_tmem=detached_fp8_p_tmem,
                    mode=mode,
                )
            elif head_dim == 128:
                if direct_tma_dkdv or detached_fp8_p_tmem or use_precomposed:
                    raise ValueError(
                        "the retained D128 route has no direct-TMA or detached-P policy"
                    )
                self._controls[key] = self._remember(
                    self.runtime.load_control(
                        fp8_p_storage="shared",
                        direct_tma_dkdv=False,
                        detached_fp8_p_tmem=False,
                    ),
                    fp8_p_storage="shared",
                    direct_tma_dkdv=False,
                    detached_fp8_p_tmem=False,
                    mode="generated_patch_chain",
                )
            else:
                raise ValueError(f"unsupported head dimension: {head_dim}")
        return self._controls[key]


def _e4m3_represented(
    torch: Any,
    shape: tuple[int, ...],
    *,
    standard_deviation: float,
) -> tuple[Any, Any]:
    source = (
        torch.randn(shape, device="cuda", dtype=torch.float32)
        .mul_(standard_deviation)
        .bfloat16()
    )
    encoded = source.float().mul_(4.0).to(torch.float8_e4m3fn)
    represented = encoded.float().mul_(0.25).bfloat16()
    return encoded, represented


def _make_state(runtime: Runtime, shape: Shape, seed: int) -> RepresentedState:
    torch = runtime.torch
    torch.manual_seed(seed)
    q_shape = (
        shape.batch,
        shape.sequence,
        shape.q_heads,
        shape.head_dim,
    )
    kv_shape = (
        shape.batch,
        shape.sequence,
        shape.kv_heads,
        shape.head_dim,
    )
    q_fp8, q_bf16 = _e4m3_represented(
        torch, q_shape, standard_deviation=0.25
    )
    k_fp8, k_bf16 = _e4m3_represented(
        torch, kv_shape, standard_deviation=0.25
    )
    v_fp8, v_bf16 = _e4m3_represented(
        torch, kv_shape, standard_deviation=0.25
    )
    dout_fp8, dout_bf16 = _e4m3_represented(
        torch, q_shape, standard_deviation=0.10
    )
    output_bf16, lse = runtime.flash_attention(
        q_bf16,
        k_bf16,
        v_bf16,
        causal=True,
        return_lse=True,
    )
    if lse.ndim == 3:
        lse_bh1s = lse.unsqueeze(2).contiguous()
    elif lse.ndim == 4 and lse.shape[2] == 1:
        lse_bh1s = lse.contiguous()
    else:
        raise RuntimeError(f"unexpected LSE shape: {tuple(lse.shape)}")

    # The projection-native producer stores -16*sum(O*dO) because both O and
    # dO operands are decoded by four.  dout_fp8 already carries one factor of
    # four, hence the explicit factor below is the other one.  The D64 kernel
    # pre-lifts P by 2^8 and therefore consumes -LSE*log2(e)+8; D128 does not.
    direct_dpsum = (
        -4.0
        * (output_bf16.float() * dout_fp8.float())
        .sum(dim=-1)
        .permute(0, 2, 1)
        .unsqueeze(2)
    ).contiguous()
    direct_lse_log2 = (-math.log2(math.e) * lse_bh1s).contiguous()
    if shape.head_dim == 64:
        direct_lse_log2 = (direct_lse_log2 + 8.0).contiguous()
    return RepresentedState(
        q_fp8=q_fp8,
        k_fp8=k_fp8,
        v_fp8=v_fp8,
        dout_fp8=dout_fp8,
        q_bf16=q_bf16,
        k_bf16=k_bf16,
        v_bf16=v_bf16,
        dout_bf16=dout_bf16,
        output_bf16=output_bf16,
        lse_bh1s=lse_bh1s,
        direct_dpsum=direct_dpsum,
        direct_lse_log2=direct_lse_log2,
    )


def _refresh_state_in_place(
    runtime: Runtime,
    shape: Shape,
    state: RepresentedState,
    seed: int,
) -> None:
    """Materialize a fresh deterministic state into compiled pointer storage."""

    torch = runtime.torch
    torch.manual_seed(seed)
    q_shape = (
        shape.batch,
        shape.sequence,
        shape.q_heads,
        shape.head_dim,
    )
    kv_shape = (
        shape.batch,
        shape.sequence,
        shape.kv_heads,
        shape.head_dim,
    )
    represented_fields = (
        ("q_fp8", "q_bf16", q_shape, 0.25),
        ("k_fp8", "k_bf16", kv_shape, 0.25),
        ("v_fp8", "v_bf16", kv_shape, 0.25),
        ("dout_fp8", "dout_bf16", q_shape, 0.10),
    )
    for encoded_name, decoded_name, tensor_shape, standard_deviation in (
        represented_fields
    ):
        encoded, decoded = _e4m3_represented(
            torch,
            tensor_shape,
            standard_deviation=standard_deviation,
        )
        getattr(state, encoded_name).copy_(encoded)
        getattr(state, decoded_name).copy_(decoded)

    output_bf16, lse = runtime.flash_attention(
        state.q_bf16,
        state.k_bf16,
        state.v_bf16,
        causal=True,
        return_lse=True,
    )
    state.output_bf16.copy_(output_bf16)
    if lse.ndim == 3:
        lse_bh1s = lse.unsqueeze(2)
    elif lse.ndim == 4 and lse.shape[2] == 1:
        lse_bh1s = lse
    else:
        raise RuntimeError(f"unexpected LSE shape: {tuple(lse.shape)}")
    state.lse_bh1s.copy_(lse_bh1s)

    state.direct_dpsum.copy_(
        -4.0
        * (state.output_bf16.float() * state.dout_fp8.float())
        .sum(dim=-1)
        .permute(0, 2, 1)
        .unsqueeze(2)
    )
    state.direct_lse_log2.copy_(-math.log2(math.e) * state.lse_bh1s)
    if shape.head_dim == 64:
        state.direct_lse_log2.add_(8.0)


def _publish_workspace_statistics(
    shape: Shape,
    state: RepresentedState,
    route: BuiltRoute,
) -> None:
    """Refresh constructor-published D64 statistics after rebinding a seed."""

    if not route.policy.get("workspace_stats", False):
        return
    stats_numel = shape.batch * shape.q_heads * shape.sequence
    # Use the source statistics' dtype rather than importing torch here; this
    # keeps the pointer-refresh helper independently CPU-testable.
    workspace_stats = route.backward.workspace_torch[
        : 2 * stats_numel * state.direct_dpsum.element_size()
    ].view(state.direct_dpsum.dtype)
    workspace_stats[:stats_numel].copy_(state.direct_dpsum.reshape(-1))
    workspace_stats[stats_numel:].copy_(state.direct_lse_log2.reshape(-1))


def _build_bf16(
    runtime: Runtime,
    controls: ControlCache,
    shape: Shape,
    state: RepresentedState,
) -> BuiltRoute:
    control = controls.bf16()
    backward = runtime.compiled_backward(
        control,
        q=state.q_bf16,
        k=state.k_bf16,
        v=state.v_bf16,
        o_or_sum=state.output_bf16,
        dout=state.dout_bf16,
        lse_or_scaled_lse=state.lse_bh1s,
        q_heads=shape.q_heads,
        kv_heads=shape.kv_heads,
        lowp=False,
        precomputed_stats=False,
        workspace_stats=False,
        scale_softmax=shape.head_dim**-0.5,
        direct_tma_dkdv=False,
    )
    return BuiltRoute(
        name="cute_bf16",
        backward=backward,
        decode_scale=1.0,
        policy={
            "input": "BF16 decode of the represented E4M3 state",
            "probability": "BF16 score/softmax",
            "statistics": "internal O*dO and LSE preprocessing",
            "output": "BF16 dQ/dK/dV",
            "causal": True,
        },
        control_provenance=controls.provenance(
            control,
            shape=shape,
            consumer="cute_bf16_reference",
        ),
    )


def _retained_settings(
    shape: Shape,
    *,
    direct_tma_dkdv_override: bool | None = None,
    detached_fp8_p_tmem_override: bool | None = None,
    exp2_degree_override: int | None = None,
    exp2_period_override: int | None = None,
) -> dict[str, Any]:
    is_d64 = shape.head_dim == 64
    if direct_tma_dkdv_override is not None and not is_d64:
        raise RouteUnavailable(
            "retained_lowp_no_direct_tma is a D64-only control; the retained "
            "D128 policy already has direct_tma_dkdv=False"
        )
    if detached_fp8_p_tmem_override is not None and not is_d64:
        raise RouteUnavailable(
            "retained_lowp_detached_p is a D64-only control"
        )
    if (
        detached_fp8_p_tmem_override is not None
        and (shape.sequence, shape.q_heads, shape.kv_heads)
        not in D64_DETACHED_FP8_P_TMEM_VERIFIED_SHAPES
    ):
        raise RouteUnavailable(
            "detached/alias P controls are verified only for measured "
            "S/Hq/Hkv shapes"
        )
    if (exp2_degree_override is None) != (exp2_period_override is None):
        raise ValueError("EX2 degree and period overrides must be supplied together")
    retained_degree = 2 if is_d64 else 1
    retained_period = None if is_d64 else 0
    retained_exp2_policy = resolve_backward_exp2_policy(
        sequence=shape.sequence,
        head_dim=shape.head_dim,
        q_heads=shape.q_heads,
        kv_heads=shape.kv_heads,
        lowp=True,
        exp2_degree=retained_degree,
        exp2_period=retained_period,
    )
    if exp2_degree_override is None:
        exp2_policy = retained_exp2_policy
    else:
        exp2_policy = resolve_backward_exp2_policy(
            sequence=shape.sequence,
            head_dim=shape.head_dim,
            q_heads=shape.q_heads,
            kv_heads=shape.kv_heads,
            lowp=True,
            exp2_degree=exp2_degree_override,
            exp2_period=exp2_period_override,
        )
    if is_d64:
        settings = {
            "workspace_stats": True,
            "direct_tma_dkdv": True,
            "detached_fp8_p_tmem": False,
            "reuse_quantized_p": False,
            "fp8_ds_lift": 16,
            "lowp_do_stages": 1,
            "probability_storage": "tmem",
            "probability_lift": 256,
            "retained_source": (
                "0db D64 coherent training policy with sequence-resolved EX2"
            ),
        }
    else:
        settings = {
            "workspace_stats": False,
            "direct_tma_dkdv": False,
            "detached_fp8_p_tmem": False,
            "reuse_quantized_p": True,
            "fp8_ds_lift": 256,
            "lowp_do_stages": 2,
            "probability_storage": "shared_coordinate_preserving_128b",
            "probability_lift": 256,
            "retained_source": "D128 corrected E4M3 probability route",
        }
    retained_schedule_auto_eligible = (
        is_d64
        and settings["direct_tma_dkdv"]
        and settings["workspace_stats"]
        and not settings["reuse_quantized_p"]
        and settings["fp8_ds_lift"] == 16
        and settings["lowp_do_stages"] == 1
        and settings["probability_storage"] == "tmem"
        and retained_exp2_policy.effective_degree == 1
        and retained_exp2_policy.effective_period == 2
    )
    retained_probability_tmem_policy = resolve_backward_probability_tmem_policy(
        sequence=shape.sequence,
        head_dim=shape.head_dim,
        q_heads=shape.q_heads,
        kv_heads=shape.kv_heads,
        batch=shape.batch,
        lowp=True,
        detached_fp8_p_tmem=None,
        auto_eligible=retained_schedule_auto_eligible,
    )
    retained_raster_policy = resolve_backward_raster_policy(
        sequence=shape.sequence,
        head_dim=shape.head_dim,
        q_heads=shape.q_heads,
        kv_heads=shape.kv_heads,
        batch=shape.batch,
        lowp=True,
        head_fast_raster=None,
        auto_eligible=retained_schedule_auto_eligible,
    )
    if direct_tma_dkdv_override is not None:
        settings["direct_tma_dkdv"] = direct_tma_dkdv_override
    controlled_comparison = (
        direct_tma_dkdv_override is not None
        or detached_fp8_p_tmem_override is not None
        or exp2_degree_override is not None
    )
    if controlled_comparison:
        comparison_auto_eligible = (
            retained_schedule_auto_eligible
            and settings["direct_tma_dkdv"]
            and exp2_policy.effective_degree == 1
            and exp2_policy.effective_period == 2
        )
        probability_tmem_policy = resolve_backward_probability_tmem_policy(
            sequence=shape.sequence,
            head_dim=shape.head_dim,
            q_heads=shape.q_heads,
            kv_heads=shape.kv_heads,
            batch=shape.batch,
            lowp=True,
            detached_fp8_p_tmem=(
                detached_fp8_p_tmem_override
                if detached_fp8_p_tmem_override is not None
                else retained_probability_tmem_policy.effective_detached
            ),
            auto_eligible=comparison_auto_eligible,
        )
        probability_tmem_policy_record = probability_tmem_policy.as_dict()
        probability_tmem_policy_record["comparison_control"] = (
            "inherits the retained route's P placement unless detached-P is "
            "the explicitly named comparison variable"
        )
        probability_tmem_policy_record["retained_policy"] = (
            retained_probability_tmem_policy.as_dict()
        )
        raster_policy = resolve_backward_raster_policy(
            sequence=shape.sequence,
            head_dim=shape.head_dim,
            q_heads=shape.q_heads,
            kv_heads=shape.kv_heads,
            batch=shape.batch,
            lowp=True,
            head_fast_raster=(
                retained_raster_policy.effective_head_fast
            ),
            auto_eligible=comparison_auto_eligible,
        )
        raster_policy_record = raster_policy.as_dict()
        raster_policy_record["comparison_control"] = (
            "inherits the retained route's physical raster so the named "
            "direct-TMA, EX2, or detached-P policy is the only kernel change"
        )
        raster_policy_record["retained_policy"] = (
            retained_raster_policy.as_dict()
        )
    else:
        probability_tmem_policy = retained_probability_tmem_policy
        probability_tmem_policy_record = probability_tmem_policy.as_dict()
        raster_policy = retained_raster_policy
        raster_policy_record = raster_policy.as_dict()
    settings["exp2_degree"] = exp2_policy.effective_degree
    settings["exp2_period"] = exp2_policy.effective_period
    settings["exp2_policy"] = exp2_policy.as_dict()
    settings["detached_fp8_p_tmem"] = (
        probability_tmem_policy.effective_detached
    )
    settings["probability_tmem_policy"] = probability_tmem_policy_record
    settings["head_fast_raster"] = raster_policy.effective_head_fast
    settings["raster_policy"] = raster_policy_record
    return settings


def _build_retained_lowp(
    runtime: Runtime,
    controls: ControlCache,
    shape: Shape,
    state: RepresentedState,
    *,
    name: str = "retained_lowp",
    direct_tma_dkdv_override: bool | None = None,
    detached_fp8_p_tmem_override: bool | None = None,
    exp2_degree_override: int | None = None,
    exp2_period_override: int | None = None,
) -> BuiltRoute:
    settings = _retained_settings(
        shape,
        direct_tma_dkdv_override=direct_tma_dkdv_override,
        detached_fp8_p_tmem_override=detached_fp8_p_tmem_override,
        exp2_degree_override=exp2_degree_override,
        exp2_period_override=exp2_period_override,
    )
    controlled_comparison = (
        direct_tma_dkdv_override is not None
        or detached_fp8_p_tmem_override is not None
        or exp2_degree_override is not None
    )
    raster_constructor_override = (
        settings["head_fast_raster"] if controlled_comparison else None
    )
    control = controls.retained_lowp(
        shape.head_dim,
        direct_tma_dkdv=settings["direct_tma_dkdv"],
        detached_fp8_p_tmem=settings["detached_fp8_p_tmem"],
        shape=shape,
        allow_s4096_precomposed=name == "retained_lowp",
    )
    backward = runtime.compiled_backward(
        control,
        q=state.q_fp8,
        k=state.k_fp8,
        v=state.v_fp8,
        o_or_sum=state.direct_dpsum,
        dout=state.dout_fp8,
        lse_or_scaled_lse=state.direct_lse_log2,
        q_heads=shape.q_heads,
        kv_heads=shape.kv_heads,
        lowp=True,
        precomputed_stats=True,
        workspace_stats=settings["workspace_stats"],
        scale_softmax=(shape.head_dim**-0.5) / 16.0,
        exp2_degree=settings["exp2_degree"],
        exp2_period=settings["exp2_period"],
        reuse_quantized_p=settings["reuse_quantized_p"],
        fp8_ds_lift=settings["fp8_ds_lift"],
        lowp_do_stages=settings["lowp_do_stages"],
        head_fast_raster=raster_constructor_override,
        direct_tma_dkdv=settings["direct_tma_dkdv"],
    )
    if (
        hasattr(backward, "head_fast_raster")
        and backward.head_fast_raster != settings["head_fast_raster"]
    ):
        raise RuntimeError(
            "compiled backward raster disagrees with the planned route policy"
        )
    if (
        hasattr(backward, "detached_fp8_p_tmem")
        and backward.detached_fp8_p_tmem
        != settings["detached_fp8_p_tmem"]
    ):
        raise RuntimeError(
            "compiled backward P placement disagrees with the planned route "
            "policy"
        )
    return BuiltRoute(
        name=name,
        backward=backward,
        decode_scale=0.25,
        policy={
            "input": "E4M3 Q/K/V/dO with fixed x4 publication scale",
            "score_scale": f"(1/sqrt({shape.head_dim}))/16",
            "statistics": (
                "negative projection-native pages in backward workspace"
                if settings["workspace_stats"]
                else "negative precomputed pages copied by backward preprocess"
            ),
            "output": "BF16 dQ/dK/dV with common x4 decode",
            "causal": True,
            **settings,
        },
        control_provenance=controls.provenance(
            control,
            shape=shape,
            consumer=name,
        ),
    )


def _build_route(
    name: str,
    runtime: Runtime,
    controls: ControlCache,
    shape: Shape,
    state: RepresentedState,
) -> BuiltRoute:
    if name == "retained_lowp":
        return _build_retained_lowp(runtime, controls, shape, state)
    if name == "retained_lowp_no_direct_tma":
        return _build_retained_lowp(
            runtime,
            controls,
            shape,
            state,
            name=name,
            direct_tma_dkdv_override=False,
        )
    if name == "retained_lowp_detached_p":
        return _build_retained_lowp(
            runtime,
            controls,
            shape,
            state,
            name=name,
            detached_fp8_p_tmem_override=True,
        )
    if name == "retained_lowp_alias_p":
        return _build_retained_lowp(
            runtime,
            controls,
            shape,
            state,
            name=name,
            detached_fp8_p_tmem_override=False,
        )
    if name == "retained_lowp_native_exp2":
        return _build_retained_lowp(
            runtime,
            controls,
            shape,
            state,
            name=name,
            exp2_degree_override=2,
            exp2_period_override=0,
        )
    if name == "retained_lowp_exp2_d1_p2":
        return _build_retained_lowp(
            runtime,
            controls,
            shape,
            state,
            name=name,
            exp2_degree_override=1,
            exp2_period_override=2,
        )
    if name == "retained_lowp_exp2_d1_p3":
        return _build_retained_lowp(
            runtime,
            controls,
            shape,
            state,
            name=name,
            exp2_degree_override=1,
            exp2_period_override=3,
        )
    if name == "mx_exact_replay":
        raise RouteUnavailable(ROUTE_CAPABILITIES[name]["reason"])
    raise ValueError(f"unknown route: {name}")


def _planned_route_policy(name: str, shape: Shape) -> dict[str, Any] | None:
    if not ROUTE_CAPABILITIES[name]["available"]:
        return None
    if name == "retained_lowp":
        return _retained_settings(shape)
    if name == "retained_lowp_no_direct_tma":
        return _retained_settings(shape, direct_tma_dkdv_override=False)
    if name == "retained_lowp_detached_p":
        return _retained_settings(
            shape,
            detached_fp8_p_tmem_override=True,
        )
    if name == "retained_lowp_alias_p":
        return _retained_settings(
            shape,
            detached_fp8_p_tmem_override=False,
        )
    if name == "retained_lowp_native_exp2":
        return _retained_settings(
            shape,
            exp2_degree_override=2,
            exp2_period_override=0,
        )
    if name == "retained_lowp_exp2_d1_p2":
        return _retained_settings(
            shape,
            exp2_degree_override=1,
            exp2_period_override=2,
        )
    if name == "retained_lowp_exp2_d1_p3":
        return _retained_settings(
            shape,
            exp2_degree_override=1,
            exp2_period_override=3,
        )
    raise ValueError(f"unknown route: {name}")


def _gradient_metrics(torch: Any, reference: Any, actual: Any) -> dict[str, Any]:
    reference_f = reference.float()
    actual_f = actual.float()
    difference = actual_f - reference_f
    reference_norm = reference_f.norm().clamp_min(1.0e-30)
    actual_norm = actual_f.norm().clamp_min(1.0e-30)
    reference_rows = reference_f.reshape(-1, reference_f.shape[-1])
    actual_rows = actual_f.reshape(-1, actual_f.shape[-1])
    row_denominator = (
        reference_rows.norm(dim=-1) * actual_rows.norm(dim=-1)
    ).clamp_min(1.0e-30)
    row_cosine = (
        (reference_rows * actual_rows).sum(dim=-1) / row_denominator
    )
    quantiles = torch.quantile(
        row_cosine,
        torch.tensor((0.01, 0.05, 0.50), device=row_cosine.device),
    )
    return {
        "reference_finite": bool(torch.isfinite(reference_f).all()),
        "actual_finite": bool(torch.isfinite(actual_f).all()),
        "cosine": float(
            (reference_f * actual_f).sum() / (reference_norm * actual_norm)
        ),
        "relative_l2": float(difference.norm() / reference_norm),
        "norm_ratio": float(actual_norm / reference_norm),
        "max_abs": float(difference.abs().max()),
        "row_cosine_p01": float(quantiles[0]),
        "row_cosine_p05": float(quantiles[1]),
        "row_cosine_p50": float(quantiles[2]),
    }


def _aggregate_metrics(torch: Any, pairs: Iterable[tuple[Any, Any]]) -> dict[str, Any]:
    dot = torch.zeros((), device="cuda", dtype=torch.float64)
    reference_square = torch.zeros_like(dot)
    actual_square = torch.zeros_like(dot)
    difference_square = torch.zeros_like(dot)
    maximum = torch.zeros((), device="cuda", dtype=torch.float32)
    reference_finite = True
    actual_finite = True
    for reference, actual in pairs:
        reference_f = reference.float()
        actual_f = actual.float()
        difference = actual_f - reference_f
        # Accumulate in FP64 without materializing full FP64 copies of these
        # large gradient tensors.  This keeps the accuracy pass well below the
        # memory footprint of a second backward workspace.
        dot += (reference_f * actual_f).sum(dtype=torch.float64)
        reference_square += reference_f.square().sum(dtype=torch.float64)
        actual_square += actual_f.square().sum(dtype=torch.float64)
        difference_square += difference.square().sum(dtype=torch.float64)
        maximum = torch.maximum(maximum, difference.abs().max())
        reference_finite = reference_finite and bool(torch.isfinite(reference_f).all())
        actual_finite = actual_finite and bool(torch.isfinite(actual_f).all())
    reference_norm = reference_square.sqrt().clamp_min(1.0e-30)
    actual_norm = actual_square.sqrt().clamp_min(1.0e-30)
    return {
        "reference_finite": reference_finite,
        "actual_finite": actual_finite,
        "cosine": float(dot / (reference_norm * actual_norm)),
        "relative_l2": float(difference_square.sqrt() / reference_norm),
        "norm_ratio": float(actual_norm / reference_norm),
        "max_abs": float(maximum),
    }


def _accuracy(
    runtime: Runtime,
    reference: BuiltRoute,
    route: BuiltRoute,
) -> dict[str, Any]:
    torch = runtime.torch
    reference.backward.run(reset=True)
    route.backward.run(reset=True)
    torch.cuda.synchronize()
    reference_values = (
        reference.backward.dq,
        reference.backward.dk,
        reference.backward.dv,
    )
    actual_values = tuple(
        value.float().mul(route.decode_scale)
        for value in (
            route.backward.dq,
            route.backward.dk,
            route.backward.dv,
        )
    )
    metrics = {
        name: _gradient_metrics(torch, reference_value, actual_value)
        for name, reference_value, actual_value in zip(
            ("dq", "dk", "dv"),
            reference_values,
            actual_values,
            strict=True,
        )
    }
    metrics["aggregate"] = _aggregate_metrics(
        torch, zip(reference_values, actual_values, strict=True)
    )
    return metrics


def _time_rotated_with_clear(
    torch: Any,
    runners: dict[str, Callable[[], None]],
    *,
    warmups: int,
    samples: int,
) -> dict[str, dict[str, Any]]:
    names = tuple(runners)
    for iteration in range(warmups):
        for offset in range(len(names)):
            runners[names[(iteration + offset) % len(names)]]()
    torch.cuda.synchronize()
    timings = {name: [] for name in names}
    for iteration in range(samples):
        for offset in range(len(names)):
            name = names[(iteration + offset) % len(names)]
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            runners[name]()
            end.record()
            end.synchronize()
            timings[name].append(float(start.elapsed_time(end) * 1000.0))
    return {name: _summary(values) for name, values in timings.items()}


def _time_cuda(
    torch: Any,
    function: Callable[[], None],
    *,
    prepare: Callable[[], None] | None,
    warmups: int,
    samples: int,
) -> dict[str, Any]:
    for _ in range(warmups):
        if prepare is not None:
            prepare()
        function()
    torch.cuda.synchronize()
    values: list[float] = []
    for _ in range(samples):
        if prepare is not None:
            prepare()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end) * 1000.0))
    return _summary(values)


def _timing(
    runtime: Runtime,
    reference: BuiltRoute,
    route: BuiltRoute,
    *,
    warmups: int,
    samples: int,
) -> dict[str, Any]:
    torch = runtime.torch
    paired = _time_rotated_with_clear(
        torch,
        {
            reference.name: lambda: reference.backward.run(reset=True),
            route.name: lambda: route.backward.run(reset=True),
        },
        warmups=warmups,
        samples=samples,
    )
    breakdown: dict[str, Any] = {}
    for built in (reference, route):
        breakdown[built.name] = {
            "required_clear_only": _time_cuda(
                torch,
                built.backward.reset,
                prepare=None,
                warmups=warmups,
                samples=samples,
            ),
            "kernel_after_required_clear": _time_cuda(
                torch,
                lambda backward=built.backward: backward.run(reset=False),
                prepare=built.backward.reset,
                warmups=warmups,
                samples=samples,
            ),
        }
    reference_us = paired[reference.name]["median_us"]
    route_us = paired[route.name]["median_us"]
    return {
        "primary_boundary": "backward_with_every_required_clear",
        "rotated_order": True,
        "paired_with_required_clear": paired,
        "speedup_vs_bf16": reference_us / route_us,
        "breakdown": breakdown,
    }


def _cleanup_shape(runtime: Runtime) -> None:
    gc.collect()
    runtime.torch.cuda.empty_cache()
    runtime.torch.cuda.synchronize()


def _atomic_write(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _run_shape_seed(
    runtime: Runtime,
    controls: ControlCache,
    shape: Shape,
    route_name: str,
    seed: int,
    *,
    compiled_case: CompiledCase | None,
    measure_timing: bool,
    warmups: int,
    samples: int,
) -> tuple[dict[str, Any], CompiledCase]:
    torch = runtime.torch
    torch.cuda.reset_peak_memory_stats()
    reused = compiled_case is not None
    if compiled_case is None:
        state = _make_state(runtime, shape, seed)
        reference = _build_bf16(runtime, controls, shape, state)
        route = _build_route(route_name, runtime, controls, shape, state)
        compiled_case = CompiledCase(
            state=state,
            reference=reference,
            route=route,
            seed=seed,
        )
    else:
        if compiled_case.route.name != route_name:
            raise RuntimeError(
                "compiled backward case cannot be reused for another route"
            )
        _refresh_state_in_place(runtime, shape, compiled_case.state, seed)
        _publish_workspace_statistics(
            shape,
            compiled_case.state,
            compiled_case.route,
        )
        compiled_case.seed = seed
    reference = compiled_case.reference
    route = compiled_case.route
    result = {
        "seed": seed,
        "route": route_name,
        "compiled_specialization_reused": reused,
        "reference_policy": reference.policy,
        "route_policy": route.policy,
        "reference_control_provenance": reference.control_provenance,
        "route_control_provenance": route.control_provenance,
        "accuracy": _accuracy(runtime, reference, route),
        "timing": (
            _timing(
                runtime,
                reference,
                route,
                warmups=warmups,
                samples=samples,
            )
            if measure_timing
            else None
        ),
        "peak_device_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_device_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
    return result, compiled_case


def _device_provenance(runtime: Runtime) -> dict[str, Any]:
    torch = runtime.torch
    properties = torch.cuda.get_device_properties(0)
    return {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device_name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "total_memory_bytes": int(properties.total_memory),
        "visible_device_count": torch.cuda.device_count(),
    }


def _new_document(
    *,
    root: Path,
    script: Path,
    extension: Path | None,
    args: argparse.Namespace,
    shapes: Sequence[Shape],
    seeds: Sequence[int],
    routes: Sequence[str],
    device: dict[str, Any] | None,
    s4096_precomposed_control: PrecomposedControlSpec | None,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": "planned" if args.dry_run else "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": [str(script), *sys.argv[1:]],
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "device": device,
        "source": _source_provenance(root, script, extension),
        "protocol": {
            "causal": True,
            "fixed_length": True,
            "comparison": (
                "CuTe BF16 and retained lowp backward on one exact E4M3-"
                "represented Q/K/V/dO state"
            ),
            "forward_for_statistics": "CuTe BF16 causal flash attention",
            "forward_timing_included": False,
            "quantization_timing_included": False,
            "primary_timing_includes_required_clears": True,
            "warmups": args.warmups,
            "samples": args.samples,
            "timing_seed": seeds[0],
            "accuracy_seeds": list(seeds),
            "compiled_specialization_reuse": {
                "scope": "one BF16/route pair per exact shape and route",
                "input_refresh": (
                    "each later seed is regenerated deterministically into "
                    "the same pointer-stable input/statistics storage"
                ),
                "workspace_statistics_refresh": (
                    "constructor-published D64 dPsum/LSE pages are recopied "
                    "for every later seed"
                ),
            },
            "minimum_device_free_gib": args.min_device_free_gib,
            "minimum_host_free_gib": args.min_host_free_gib,
            "backward_control_provenance": {
                "schema": CONTROL_PROVENANCE_SCHEMA,
                "generated_control_contract": (
                    "the final composed Python source SHA256/bytes and its "
                    "exact shape/consumer binding are recorded in every "
                    "completed seed record"
                ),
                "authenticated_s4096_control": (
                    s4096_precomposed_control.as_dict()
                    if s4096_precomposed_control is not None
                    else None
                ),
                "selection_rule": (
                    "an authenticated control is eligible only for the "
                    "retained_lowp B1/S4096/Hq32/Hkv8/D64 case; every BF16 "
                    "reference, comparison route, S8192 case, and S16384 "
                    "case uses a locally generated control"
                ),
            },
            "backward_exp2_dispatch": {
                "version": BACKWARD_EXP2_POLICY_VERSION,
                "scope": "D64 B1 causal retained lowp",
                "threshold_sequence": D64_SELECTIVE_EXP2_MIN_SEQUENCE,
                "default": {"degree": 2, "period": 0},
                "verified_shape_policy": {"degree": 1, "period": 2},
                "verified_shapes": [
                    {
                        "sequence": sequence,
                        "q_heads": q_heads,
                        "kv_heads": kv_heads,
                    }
                    for sequence, q_heads, kv_heads in (
                        D64_SELECTIVE_EXP2_VERIFIED_SHAPES
                    )
                ],
            },
            "backward_raster_dispatch": {
                "version": BACKWARD_RASTER_POLICY_VERSION,
                "scope": (
                    "D64 B1 causal one-lane direct-TMA retained lowp with "
                    "TMEM probability storage and workspace statistics"
                ),
                "default": "key_fast",
                "verified_shape_policy": "head_fast",
                "verified_shapes": [
                    {
                        "sequence": sequence,
                        "q_heads": q_heads,
                        "kv_heads": kv_heads,
                    }
                    for sequence, q_heads, kv_heads in (
                        D64_HEAD_FAST_RASTER_VERIFIED_SHAPES
                    )
                ],
            },
            "backward_probability_tmem_dispatch": {
                "version": BACKWARD_P_TMEM_POLICY_VERSION,
                "scope": (
                    "D64 B1 causal one-lane direct-TMA retained lowp with "
                    "TMEM probability storage and workspace statistics"
                ),
                "default": "score_alias_p",
                "verified_shape_policy": "detached_fp8_p_tmem",
                "verified_shapes": [
                    {
                        "sequence": sequence,
                        "q_heads": q_heads,
                        "kv_heads": kv_heads,
                    }
                    for sequence, q_heads, kv_heads in (
                        D64_DETACHED_FP8_P_TMEM_VERIFIED_SHAPES
                    )
                ],
            },
        },
        "route_capabilities": ROUTE_CAPABILITIES,
        "requested_routes": list(routes),
        "planned_shapes": [shape.as_dict() for shape in shapes],
        "planned_cases": [
            {
                "shape": shape.as_dict(),
                "route": route,
                "available": ROUTE_CAPABILITIES[route]["available"],
                "route_policy": _planned_route_policy(route, shape),
                "control_plan": _planned_control_binding(
                    s4096_precomposed_control,
                    route,
                    shape,
                ),
            }
            for shape in shapes
            for route in routes
        ],
        "unavailable_routes": [
            {
                "route": route,
                "reason": ROUTE_CAPABILITIES[route]["reason"],
            }
            for route in routes
            if not ROUTE_CAPABILITIES[route]["available"]
        ],
        "records": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequences",
        default="4096",
        help="comma-separated sequence lengths (positive multiples of 128)",
    )
    parser.add_argument(
        "--head-dims",
        default="64",
        help="comma-separated retained head dimensions: 64 and/or 128",
    )
    parser.add_argument(
        "--head-pairs",
        default="32/8",
        help="comma-separated Q/KV head pairs, for example 16/4,32/8,64/16",
    )
    parser.add_argument(
        "--seeds",
        default="20260820",
        help="accuracy seeds; timing is measured only for the first seed",
    )
    parser.add_argument(
        "--routes",
        default="retained_lowp",
        help=(
            "comma-separated routes: retained_lowp, the D64-only "
            "retained_lowp_no_direct_tma, retained_lowp_alias_p, "
            "retained_lowp_detached_p, and retained_lowp_native_exp2 controls, "
            "the D64 selective-EX2 controls "
            "retained_lowp_exp2_d1_p2 and "
            "retained_lowp_exp2_d1_p3, and (once reconstructed) "
            "mx_exact_replay"
        ),
    )
    parser.add_argument(
        "--require-all-routes",
        action="store_true",
        help="fail before GPU initialization if any requested route is unavailable",
    )
    parser.add_argument("--warmups", type=int, default=9)
    parser.add_argument("--samples", type=int, default=41)
    parser.add_argument(
        "--min-device-free-gib",
        type=float,
        default=16.0,
        help="free-device-memory reserve retained beyond the shape estimate",
    )
    parser.add_argument(
        "--min-host-free-gib",
        type=float,
        default=16.0,
        help="minimum MemAvailable before compiling another specialization",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="write the resolved matrix/capability manifest without importing CUDA",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="persist an error record and continue with the next shape/seed",
    )
    parser.add_argument(
        "--backward-control-source",
        type=Path,
        help=(
            "authenticated precomposed Python control used only for the exact "
            "retained_lowp B1/S4096/Hq32/Hkv8/D64 case"
        ),
    )
    parser.add_argument(
        "--backward-control-sha256",
        help="required SHA256 for --backward-control-source",
    )
    parser.add_argument(
        "--backward-control-bytes",
        type=int,
        help="required byte size for --backward-control-source",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmups < 0 or args.samples <= 0:
        parser.error("--warmups must be non-negative and --samples positive")
    if args.min_device_free_gib < 0 or args.min_host_free_gib < 0:
        parser.error("memory reserves must be non-negative")
    try:
        sequences = _parse_positive_ints(args.sequences, argument="--sequences")
        head_dims = _parse_positive_ints(args.head_dims, argument="--head-dims")
        seeds = _parse_positive_ints(args.seeds, argument="--seeds")
        head_pairs = _parse_head_pairs(args.head_pairs)
        routes = _parse_routes(args.routes)
        shapes = _build_shapes(sequences, head_dims, head_pairs)
        _validate_route_shape_support(routes, shapes)
        s4096_precomposed_control = _resolve_precomposed_control_spec(
            args.backward_control_source,
            args.backward_control_sha256,
            args.backward_control_bytes,
        )
        if s4096_precomposed_control is not None and not any(
            _precomposed_control_applies(
                s4096_precomposed_control,
                route,
                shape,
            )
            for shape in shapes
            for route in routes
        ):
            raise ValueError(
                "the authenticated backward control is scoped only to "
                "retained_lowp B1/S4096/Hq32/Hkv8/D64, but the requested "
                "matrix contains no such case"
            )
    except ValueError as error:
        parser.error(str(error))
    unavailable = [
        route for route in routes if not ROUTE_CAPABILITIES[route]["available"]
    ]
    if args.require_all_routes and unavailable:
        parser.error(
            "requested routes are unavailable: " + ", ".join(unavailable)
        )

    script = Path(__file__).resolve()
    root = script.parents[2]
    if args.dry_run:
        document = _new_document(
            root=root,
            script=script,
            extension=None,
            args=args,
            shapes=shapes,
            seeds=seeds,
            routes=routes,
            device=None,
            s4096_precomposed_control=s4096_precomposed_control,
        )
        _atomic_write(args.output, document)
        print(json.dumps(document, indent=2, sort_keys=True))
        return

    runtime = _load_runtime(root)
    torch = runtime.torch
    if torch.cuda.device_count() != 1:
        raise RuntimeError("expose exactly one otherwise-idle GPU")
    torch.cuda.set_device(0)
    document = _new_document(
        root=root,
        script=script,
        extension=runtime.lowp_extension_path,
        args=args,
        shapes=shapes,
        seeds=seeds,
        routes=routes,
        device=_device_provenance(runtime),
        s4096_precomposed_control=s4096_precomposed_control,
    )
    _atomic_write(args.output, document)
    controls = ControlCache(runtime, s4096_precomposed_control)
    available_routes = [
        route for route in routes if ROUTE_CAPABILITIES[route]["available"]
    ]
    try:
        for shape in shapes:
            snapshot = _memory_snapshot(runtime, shape)
            _guard_memory(
                snapshot,
                minimum_device_free_gib=args.min_device_free_gib,
                minimum_host_free_gib=args.min_host_free_gib,
            )
            for route_name in available_routes:
                shape_record: dict[str, Any] = {
                    "shape": shape.as_dict(),
                    "route": route_name,
                    "route_policy": _planned_route_policy(route_name, shape),
                    "status": "running",
                    "memory_before": snapshot,
                    "seed_records": [],
                }
                document["records"].append(shape_record)
                _atomic_write(args.output, document)
                compiled_case: CompiledCase | None = None
                for seed_index, seed in enumerate(seeds):
                    try:
                        seed_record, compiled_case = _run_shape_seed(
                            runtime,
                            controls,
                            shape,
                            route_name,
                            seed,
                            compiled_case=compiled_case,
                            measure_timing=seed_index == 0,
                            warmups=args.warmups,
                            samples=args.samples,
                        )
                        shape_record["seed_records"].append(seed_record)
                        print(
                            f"D={shape.head_dim} S={shape.sequence} "
                            f"H={shape.q_heads}/{shape.kv_heads} seed={seed} "
                            f"route={route_name} "
                            f"cos={seed_record['accuracy']['aggregate']['cosine']:.6f}",
                            flush=True,
                        )
                    except Exception as error:  # persist expensive partial work
                        shape_record["seed_records"].append(
                            {
                                "seed": seed,
                                "status": "error",
                                "error_type": type(error).__name__,
                                "error": str(error),
                            }
                        )
                        shape_record["status"] = "error"
                        _atomic_write(args.output, document)
                        _cleanup_shape(runtime)
                        if not args.continue_on_error:
                            raise
                    _atomic_write(args.output, document)
                if shape_record["status"] != "error":
                    shape_record["status"] = "complete"
                _atomic_write(args.output, document)
                del compiled_case
                _cleanup_shape(runtime)
        document["status"] = "complete"
    except Exception:
        document["status"] = "error"
        _atomic_write(args.output, document)
        raise
    _atomic_write(args.output, document)


if __name__ == "__main__":
    main()
