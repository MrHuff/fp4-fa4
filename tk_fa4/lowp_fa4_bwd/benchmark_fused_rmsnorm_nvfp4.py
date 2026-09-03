#!/usr/bin/env python3
"""Benchmark the fixed B16 RMSNorm -> native-NVFP4 preparation boundary.

This harness is deliberately narrow.  It authenticates one low-precision
extension and runs only the production B16, S4096, H2048 activation shape.
The forward control is the eager FP32 RMSNorm formula followed by the same
exact native-NVFP4 packer.  The backward control is the closed-form FP32
RMSNorm derivative evaluated with the identical saved inverse RMS operand.

The output is a caller-selected, create-only JSON document.  Correctness is
fail closed: a document is still emitted when a numerical threshold fails,
but the process exits with status 2.
"""

from __future__ import annotations

import argparse
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
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
EXTENSION_ENVIRONMENT_VARIABLE = "TK_FA4_LOWP_BWD_EXTENSION_SOURCE"
EXTENSION_MODULE = "tk_fa4._C_b300_lowp_bwd"

BATCH = 16
SEQUENCE = 4096
HIDDEN = 2048
ROWS = BATCH * SEQUENCE
EPSILON = 1.0e-5
GIB = 1 << 30

DEFAULT_SEEDS = (20260826, 20260827)
DEFAULT_MIN_NORMALIZED_COSINE = 0.9999
DEFAULT_MAX_NORMALIZED_RELATIVE_L2 = 2.0e-4
DEFAULT_MAX_INV_RMS_ABS = 2.0e-5
DEFAULT_MIN_DX_COSINE = 0.9999
DEFAULT_MAX_DX_RELATIVE_L2 = 2.0e-3
DEFAULT_MIN_DGAMMA_COSINE = 0.999
DEFAULT_MAX_DGAMMA_RELATIVE_L2 = 2.0e-2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _absolute_without_resolving(path: Path) -> Path:
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _observed_file_identity(path: Path) -> dict[str, Any]:
    selected = _absolute_without_resolving(path)
    try:
        selected_stat = selected.lstat()
    except OSError as error:
        raise ValueError(f"unable to stat extension: {selected}") from error
    if not stat.S_ISREG(selected_stat.st_mode):
        raise ValueError(
            "extension must be a regular, non-symlink file: " f"{selected}"
        )
    resolved = selected.resolve(strict=True)
    resolved_stat = resolved.stat()
    return {
        "selected_path": str(selected),
        "resolved_path": str(resolved),
        "sha256": _sha256(resolved),
        "bytes": int(resolved_stat.st_size),
        "mtime_ns": int(resolved_stat.st_mtime_ns),
    }


def _validate_expected_sha256(value: str) -> str:
    if re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError("extension SHA-256 must contain exactly 64 hex digits")
    return value.lower()


def _authenticate_extension_candidate(
    path: Path,
    expected_sha256: str,
    expected_bytes: int,
) -> dict[str, Any]:
    expected_sha256 = _validate_expected_sha256(expected_sha256)
    if expected_bytes <= 0:
        raise ValueError("extension byte count must be positive")
    identity = _observed_file_identity(path)
    if identity["bytes"] != expected_bytes or not hmac.compare_digest(
        identity["sha256"], expected_sha256
    ):
        raise ValueError(
            "extension identity mismatch: expected "
            f"sha256={expected_sha256}, bytes={expected_bytes}; observed "
            f"sha256={identity['sha256']}, bytes={identity['bytes']}"
        )
    return {
        **identity,
        "expected_sha256": expected_sha256,
        "expected_bytes": expected_bytes,
        "authenticated": True,
    }


def _authenticate_loaded_extension(
    module: ModuleType | Any,
    expected_identity: dict[str, Any],
) -> dict[str, Any]:
    loaded_file = getattr(module, "__file__", None)
    if not loaded_file:
        raise RuntimeError("loaded low-precision extension has no __file__")
    observed = _authenticate_extension_candidate(
        Path(loaded_file),
        str(expected_identity["expected_sha256"]),
        int(expected_identity["expected_bytes"]),
    )
    if Path(observed["resolved_path"]) != Path(
        expected_identity["resolved_path"]
    ):
        raise RuntimeError(
            "loaded extension path differs from the authenticated selection: "
            f"selected {expected_identity['resolved_path']}, loaded "
            f"{observed['resolved_path']}"
        )
    return {
        **observed,
        "module": EXTENSION_MODULE,
        "loaded_file": str(loaded_file),
        "post_load_authenticated": True,
    }


def _output_path(value: str) -> Path:
    path = _absolute_without_resolving(Path(value))
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite output: {path}")
    if path.suffix.lower() != ".json":
        raise ValueError("--output must name a JSON file")
    return path


def _write_new_json(path: Path, document: dict[str, Any]) -> None:
    """Atomically publish JSON without ever replacing an existing node."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(
        document,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n").encode()
    temporary_fd, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.tmp.",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(temporary_fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        # A hard link is an atomic create-only publication.  It fails with
        # EEXIST for files, directories, symlinks, and dangling symlinks.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("timing samples must not be empty")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(
        ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
    )


def _timing_summary(values_us: Sequence[float]) -> dict[str, Any]:
    if not values_us:
        raise ValueError("timing samples must not be empty")
    mean = statistics.fmean(values_us)
    return {
        "unit": "microseconds",
        "samples": len(values_us),
        "minimum_us": min(values_us),
        "p10_us": _percentile(values_us, 0.10),
        "p50_us": statistics.median(values_us),
        "mean_us": mean,
        "p90_us": _percentile(values_us, 0.90),
        "maximum_us": max(values_us),
        "cv": statistics.stdev(values_us) / mean
        if len(values_us) > 1 and mean
        else 0.0,
        "samples_us": list(values_us),
    }


def _rotating_orders(names: Sequence[str], rounds: int) -> list[list[str]]:
    if not names:
        raise ValueError("at least one timing provider is required")
    if rounds < 0:
        raise ValueError("round count must not be negative")
    ordered = tuple(names)
    return [
        list(ordered[offset:] + ordered[:offset])
        for offset in (index % len(ordered) for index in range(rounds))
    ]


def _time_interleaved(
    torch: Any,
    runners: dict[str, Callable[[], None]],
    *,
    warmups: int,
    samples: int,
) -> tuple[dict[str, dict[str, Any]], list[list[str]], list[list[str]]]:
    names = tuple(runners)
    warmup_orders = _rotating_orders(names, warmups)
    sample_orders = _rotating_orders(names, samples)
    for order in warmup_orders:
        for name in order:
            runners[name]()
    torch.cuda.synchronize()

    values: dict[str, list[float]] = {name: [] for name in names}
    for order in sample_orders:
        events = []
        for name in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            runners[name]()
            end.record()
            events.append((name, start, end))
        events[-1][2].synchronize()
        for name, start, end in events:
            values[name].append(float(start.elapsed_time(end) * 1000.0))
    return (
        {name: _timing_summary(values[name]) for name in names},
        warmup_orders,
        sample_orders,
    )


def _tensor_metrics(torch: Any, reference: Any, actual: Any) -> dict[str, Any]:
    if reference.shape != actual.shape:
        raise ValueError(
            f"metric shape mismatch: {reference.shape} versus {actual.shape}"
        )
    reference_flat = reference.reshape(-1)
    actual_flat = actual.reshape(-1)
    dot = 0.0
    reference_square = 0.0
    actual_square = 0.0
    difference_square = 0.0
    absolute_sum = 0.0
    maximum = 0.0
    reference_finite = True
    actual_finite = True
    chunk_size = 1 << 20
    for offset in range(0, reference_flat.numel(), chunk_size):
        reference_part = reference_flat[offset : offset + chunk_size].float()
        actual_part = actual_flat[offset : offset + chunk_size].float()
        difference = actual_part - reference_part
        dot += float(torch.dot(reference_part, actual_part))
        reference_square += float(torch.dot(reference_part, reference_part))
        actual_square += float(torch.dot(actual_part, actual_part))
        difference_square += float(torch.dot(difference, difference))
        absolute_sum += float(difference.abs().sum())
        maximum = max(maximum, float(difference.abs().max()))
        reference_finite = reference_finite and bool(
            torch.isfinite(reference_part).all()
        )
        actual_finite = actual_finite and bool(torch.isfinite(actual_part).all())
    reference_norm = math.sqrt(max(reference_square, 1.0e-40))
    actual_norm = math.sqrt(max(actual_square, 1.0e-40))
    def finite_or_none(value: float) -> float | None:
        return float(value) if math.isfinite(value) else None

    return {
        "reference_finite": reference_finite,
        "actual_finite": actual_finite,
        "cosine": finite_or_none(
            dot / max(reference_norm * actual_norm, 1.0e-40)
        ),
        "relative_l2": finite_or_none(
            math.sqrt(difference_square) / reference_norm
        ),
        "norm_ratio": finite_or_none(actual_norm / reference_norm),
        "mean_abs": finite_or_none(
            absolute_sum / max(reference_flat.numel(), 1)
        ),
        "max_abs": finite_or_none(maximum),
    }


def _byte_metrics(torch: Any, reference: Any, actual: Any) -> dict[str, Any]:
    if reference.shape != actual.shape or reference.dtype != actual.dtype:
        return {
            "byte_equal": False,
            "metadata_equal": False,
            "reference_shape": list(reference.shape),
            "actual_shape": list(actual.shape),
            "reference_dtype": str(reference.dtype),
            "actual_dtype": str(actual.dtype),
        }
    reference_bytes = reference.contiguous().view(torch.uint8).reshape(-1)
    actual_bytes = actual.contiguous().view(torch.uint8).reshape(-1)
    mismatch_count = 0
    chunk_size = 1 << 22
    for offset in range(0, reference_bytes.numel(), chunk_size):
        mismatch_count += int(
            torch.count_nonzero(
                reference_bytes[offset : offset + chunk_size]
                != actual_bytes[offset : offset + chunk_size]
            )
        )
    byte_count = reference_bytes.numel()
    return {
        "byte_equal": mismatch_count == 0,
        "metadata_equal": True,
        "bytes": byte_count,
        "mismatch_bytes": mismatch_count,
        "mismatch_fraction": mismatch_count / max(byte_count, 1),
    }


def _tensor_description(tensor: Any) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "contiguous": bool(tensor.is_contiguous()),
        "logical_bytes": int(tensor.numel() * tensor.element_size()),
    }


def _nested_tensor_bytes(value: Any) -> int:
    if hasattr(value, "numel") and hasattr(value, "element_size"):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(_nested_tensor_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_nested_tensor_bytes(item) for item in value)
    return 0


def _memory_snapshot(torch: Any) -> dict[str, int]:
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "driver_free_bytes": int(free_bytes),
        "driver_total_bytes": int(total_bytes),
    }


def _available_system_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _git_output(*arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ("git", "-C", str(REPO_ROOT), *arguments),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.rstrip("\n")


def _harness_provenance() -> dict[str, Any]:
    source = Path(__file__).resolve()
    return {
        "path": str(source),
        "sha256": _sha256(source),
        "bytes": source.stat().st_size,
        "git_head": _git_output("rev-parse", "HEAD"),
        "git_branch": _git_output("branch", "--show-current"),
        "git_status": _git_output(
            "status", "--short", "--", str(source.relative_to(REPO_ROOT))
        ),
    }


def _eager_forward_preparation(
    torch: Any,
    tensor: Any,
    gamma: Any,
    epsilon: float,
    pack: Callable[[Any], tuple[Any, Any, Any]],
) -> tuple[Any, Any, Any, Any, Any]:
    values = tensor.float()
    inv_rms = torch.rsqrt(values.square().mean(dim=1, keepdim=True) + epsilon)
    normalized = (values * inv_rms * gamma.float().unsqueeze(0)).bfloat16()
    normalized = normalized.contiguous()
    packed, scales, global_decode = pack(normalized)
    return (
        packed,
        scales,
        global_decode,
        inv_rms.reshape(-1).contiguous(),
        normalized,
    )


def _eager_rmsnorm_backward(
    torch: Any,
    tensor: Any,
    gamma: Any,
    inv_rms: Any,
    gradient: Any,
) -> tuple[Any, Any]:
    values = tensor.float()
    gradient_f = gradient.float()
    inverse = inv_rms.float().unsqueeze(1)
    weighted_gradient = gradient_f * gamma.float().unsqueeze(0)
    projection = (weighted_gradient * values).mean(dim=1, keepdim=True)
    correction = inverse.square() * projection
    input_gradient = (
        inverse * (weighted_gradient - values * correction)
    ).bfloat16().contiguous()
    gamma_gradient = (
        gradient_f * (values * inverse)
    ).sum(dim=0).bfloat16().contiguous()
    return input_gradient, gamma_gradient


def _comparison_checks(
    correctness: dict[str, Any],
    thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    normalized = correctness["forward"]["normalized_vs_eager"]
    inverse = correctness["forward"]["inv_rms_vs_eager"]
    dx = correctness["backward"]["dx_same_saved_inv_rms"]
    dgamma = correctness["backward"]["dgamma_same_saved_inv_rms"]
    pack = correctness["forward"]["pack_on_fused_normalized"]
    candidates = (
        (
            "normalized_finite",
            bool(normalized["reference_finite"] and normalized["actual_finite"]),
            "is_true",
            True,
        ),
        (
            "normalized_cosine",
            normalized["cosine"],
            ">=",
            thresholds["min_normalized_cosine"],
        ),
        (
            "normalized_relative_l2",
            normalized["relative_l2"],
            "<=",
            thresholds["max_normalized_relative_l2"],
        ),
        (
            "inv_rms_finite",
            bool(inverse["reference_finite"] and inverse["actual_finite"]),
            "is_true",
            True,
        ),
        (
            "inv_rms_max_abs",
            inverse["max_abs"],
            "<=",
            thresholds["max_inv_rms_abs"],
        ),
        (
            "native_pack_payload_byte_exact",
            bool(pack["payload"]["byte_equal"]),
            "is_true",
            True,
        ),
        (
            "native_pack_scales_byte_exact",
            bool(pack["scales"]["byte_equal"]),
            "is_true",
            True,
        ),
        (
            "native_pack_global_decode_byte_exact",
            bool(pack["global_decode"]["byte_equal"]),
            "is_true",
            True,
        ),
        (
            "backward_dx_finite",
            bool(dx["reference_finite"] and dx["actual_finite"]),
            "is_true",
            True,
        ),
        (
            "backward_dx_cosine",
            dx["cosine"],
            ">=",
            thresholds["min_dx_cosine"],
        ),
        (
            "backward_dx_relative_l2",
            dx["relative_l2"],
            "<=",
            thresholds["max_dx_relative_l2"],
        ),
        (
            "backward_dgamma_finite",
            bool(dgamma["reference_finite"] and dgamma["actual_finite"]),
            "is_true",
            True,
        ),
        (
            "backward_dgamma_cosine",
            dgamma["cosine"],
            ">=",
            thresholds["min_dgamma_cosine"],
        ),
        (
            "backward_dgamma_relative_l2",
            dgamma["relative_l2"],
            "<=",
            thresholds["max_dgamma_relative_l2"],
        ),
    )
    checks = []
    for name, observed, operator, threshold in candidates:
        if operator == "is_true":
            passed = observed is True
        elif operator == ">=":
            passed = bool(
                isinstance(observed, (int, float))
                and not isinstance(observed, bool)
                and math.isfinite(float(observed))
                and observed >= threshold
            )
        elif operator == "<=":
            passed = bool(
                isinstance(observed, (int, float))
                and not isinstance(observed, bool)
                and math.isfinite(float(observed))
                and observed <= threshold
            )
        else:  # pragma: no cover - fixed table above.
            raise AssertionError(operator)
        checks.append(
            {
                "name": name,
                "observed": observed,
                "operator": operator,
                "threshold": threshold,
                "passed": passed,
            }
        )
    return checks


def _run_seed(
    torch: Any,
    *,
    seed: int,
    epsilon: float,
    warmups: int,
    samples: int,
    thresholds: dict[str, float],
    fused_forward: Callable[[Any, Any, float], tuple[Any, ...]],
    native_pack: Callable[[Any], tuple[Any, Any, Any]],
    fused_backward: Callable[[Any, Any, Any, Any], tuple[Any, Any]],
) -> dict[str, Any]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    tensor = torch.randn(
        (ROWS, HIDDEN),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    gamma_noise = torch.randn(
        (HIDDEN,),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    gamma = (1.0 + 0.05 * gamma_noise).bfloat16().contiguous()
    del gamma_noise
    gradient = torch.randn(
        tensor.shape,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    memory: dict[str, Any] = {
        "after_inputs": _memory_snapshot(torch),
        "input_logical_bytes": {
            "tensor": _nested_tensor_bytes(tensor),
            "gamma": _nested_tensor_bytes(gamma),
            "gradient": _nested_tensor_bytes(gradient),
        },
    }

    with torch.inference_mode():
        fused_forward_outputs = fused_forward(tensor, gamma, epsilon)
        eager_forward_outputs = _eager_forward_preparation(
            torch, tensor, gamma, epsilon, native_pack
        )
        exact_pack_on_fused_normalized = native_pack(fused_forward_outputs[4])
        fused_backward_outputs = fused_backward(
            tensor, gamma, fused_forward_outputs[3], gradient
        )
        eager_backward_same_inverse = _eager_rmsnorm_backward(
            torch, tensor, gamma, fused_forward_outputs[3], gradient
        )
        eager_backward_independent_inverse = _eager_rmsnorm_backward(
            torch, tensor, gamma, eager_forward_outputs[3], gradient
        )
        torch.cuda.synchronize()

        correctness = {
            "forward": {
                "normalized_vs_eager": _tensor_metrics(
                    torch, eager_forward_outputs[4], fused_forward_outputs[4]
                ),
                "inv_rms_vs_eager": _tensor_metrics(
                    torch, eager_forward_outputs[3], fused_forward_outputs[3]
                ),
                "pack_on_fused_normalized": {
                    name: _byte_metrics(torch, reference, actual)
                    for name, reference, actual in zip(
                        ("payload", "scales", "global_decode"),
                        exact_pack_on_fused_normalized,
                        fused_forward_outputs[:3],
                        strict=True,
                    )
                },
                "fused_pack_vs_independent_eager": {
                    name: _byte_metrics(torch, reference, actual)
                    for name, reference, actual in zip(
                        ("payload", "scales", "global_decode"),
                        eager_forward_outputs[:3],
                        fused_forward_outputs[:3],
                        strict=True,
                    )
                },
                "fused_outputs": [
                    _tensor_description(value) for value in fused_forward_outputs
                ],
            },
            "backward": {
                "dx_same_saved_inv_rms": _tensor_metrics(
                    torch,
                    eager_backward_same_inverse[0],
                    fused_backward_outputs[0],
                ),
                "dgamma_same_saved_inv_rms": _tensor_metrics(
                    torch,
                    eager_backward_same_inverse[1],
                    fused_backward_outputs[1],
                ),
                "dx_independent_eager_inv_rms": _tensor_metrics(
                    torch,
                    eager_backward_independent_inverse[0],
                    fused_backward_outputs[0],
                ),
                "dgamma_independent_eager_inv_rms": _tensor_metrics(
                    torch,
                    eager_backward_independent_inverse[1],
                    fused_backward_outputs[1],
                ),
                "fused_outputs": [
                    _tensor_description(value) for value in fused_backward_outputs
                ],
            },
        }
        torch.cuda.synchronize()
        memory["correctness_peak"] = _memory_snapshot(torch)

    del (
        fused_forward_outputs,
        eager_forward_outputs,
        exact_pack_on_fused_normalized,
        fused_backward_outputs,
        eager_backward_same_inverse,
        eager_backward_independent_inverse,
    )
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    forward_state: dict[str, tuple[Any, ...]] = {}

    def run_eager_forward() -> None:
        forward_state["eager"] = _eager_forward_preparation(
            torch, tensor, gamma, epsilon, native_pack
        )

    def run_fused_forward() -> None:
        forward_state["fused"] = fused_forward(tensor, gamma, epsilon)

    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        forward_timing, forward_warmup_orders, forward_sample_orders = (
            _time_interleaved(
                torch,
                {"eager_formula_plus_exact_pack": run_eager_forward,
                 "fused_rmsnorm_nvfp4": run_fused_forward},
                warmups=warmups,
                samples=samples,
            )
        )
    memory["forward_timing_peak"] = {
        **_memory_snapshot(torch),
        "retained_output_logical_bytes": {
            name: _nested_tensor_bytes(value)
            for name, value in forward_state.items()
        },
    }
    forward_speedup = {
        "mean": (
            forward_timing["eager_formula_plus_exact_pack"]["mean_us"]
            / forward_timing["fused_rmsnorm_nvfp4"]["mean_us"]
        ),
        "median": (
            forward_timing["eager_formula_plus_exact_pack"]["p50_us"]
            / forward_timing["fused_rmsnorm_nvfp4"]["p50_us"]
        ),
    }
    del forward_state
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Both timed backward providers consume the same forward-saved inverse RMS.
    saved_inv_rms = fused_forward(tensor, gamma, epsilon)[3]
    backward_state: dict[str, tuple[Any, Any]] = {}

    def run_eager_backward() -> None:
        backward_state["eager"] = _eager_rmsnorm_backward(
            torch, tensor, gamma, saved_inv_rms, gradient
        )

    def run_fused_backward() -> None:
        backward_state["fused"] = fused_backward(
            tensor, gamma, saved_inv_rms, gradient
        )

    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        backward_timing, backward_warmup_orders, backward_sample_orders = (
            _time_interleaved(
                torch,
                {"eager_closed_form": run_eager_backward,
                 "fused_rmsnorm_backward": run_fused_backward},
                warmups=warmups,
                samples=samples,
            )
        )
    memory["backward_timing_peak"] = {
        **_memory_snapshot(torch),
        "retained_output_logical_bytes": {
            name: _nested_tensor_bytes(value)
            for name, value in backward_state.items()
        },
    }
    backward_speedup = {
        "mean": (
            backward_timing["eager_closed_form"]["mean_us"]
            / backward_timing["fused_rmsnorm_backward"]["mean_us"]
        ),
        "median": (
            backward_timing["eager_closed_form"]["p50_us"]
            / backward_timing["fused_rmsnorm_backward"]["p50_us"]
        ),
    }
    del backward_state, saved_inv_rms, tensor, gamma, gradient
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    memory["after_cleanup"] = _memory_snapshot(torch)

    checks = _comparison_checks(correctness, thresholds)
    return {
        "seed": seed,
        "rng": {
            "generator": "torch.Generator(device='cuda')",
            "draw_order": [
                "BF16 normal input [65536,2048]",
                "FP32 normal gamma noise [2048], transformed as 1+0.05*x",
                "BF16 normal output gradient [65536,2048]",
            ],
        },
        "correctness": correctness,
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
        "timing": {
            "forward": {
                "providers": forward_timing,
                "warmup_orders": forward_warmup_orders,
                "sample_orders": forward_sample_orders,
                "fused_speedup_vs_eager": forward_speedup,
            },
            "backward": {
                "providers": backward_timing,
                "warmup_orders": backward_warmup_orders,
                "sample_orders": backward_sample_orders,
                "fused_speedup_vs_eager": backward_speedup,
            },
        },
        "memory": memory,
    }


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _finite_positive(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def _probability(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be in [0, 1]")
    return parsed


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension-source", type=Path, required=True)
    parser.add_argument("--extension-sha256", required=True)
    parser.add_argument("--extension-bytes", type=_positive_integer, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--warmups", type=_positive_integer, default=8)
    parser.add_argument("--samples", type=_positive_integer, default=100)
    parser.add_argument("--epsilon", type=_finite_positive, default=EPSILON)
    parser.add_argument(
        "--minimum-free-device-gib", type=_finite_positive, default=16.0
    )
    parser.add_argument(
        "--minimum-free-system-gib", type=_finite_positive, default=8.0
    )
    parser.add_argument(
        "--min-normalized-cosine",
        type=_probability,
        default=DEFAULT_MIN_NORMALIZED_COSINE,
    )
    parser.add_argument(
        "--max-normalized-relative-l2",
        type=_finite_positive,
        default=DEFAULT_MAX_NORMALIZED_RELATIVE_L2,
    )
    parser.add_argument(
        "--max-inv-rms-abs",
        type=_finite_positive,
        default=DEFAULT_MAX_INV_RMS_ABS,
    )
    parser.add_argument(
        "--min-dx-cosine", type=_probability, default=DEFAULT_MIN_DX_COSINE
    )
    parser.add_argument(
        "--max-dx-relative-l2",
        type=_finite_positive,
        default=DEFAULT_MAX_DX_RELATIVE_L2,
    )
    parser.add_argument(
        "--min-dgamma-cosine",
        type=_probability,
        default=DEFAULT_MIN_DGAMMA_COSINE,
    )
    parser.add_argument(
        "--max-dgamma-relative-l2",
        type=_finite_positive,
        default=DEFAULT_MAX_DGAMMA_RELATIVE_L2,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    output = _output_path(args.output)
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must not contain duplicates")

    candidate_identity = _authenticate_extension_candidate(
        args.extension_source,
        args.extension_sha256,
        args.extension_bytes,
    )
    os.environ[EXTENSION_ENVIRONMENT_VARIABLE] = candidate_identity[
        "resolved_path"
    ]

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(args.device)
    capability = torch.cuda.get_device_capability()
    if capability != (10, 0):
        raise RuntimeError(
            f"this benchmark requires SM100, found compute capability {capability}"
        )

    from tk_fa4 import (
        b300_prepare_nvfp4_projection_operand,
        b300_prepare_nvfp4_projection_operand_rmsnorm,
        b300_rmsnorm_backward,
    )
    from tk_fa4 import interface as tk_interface

    module = getattr(tk_interface, "_C_b300_lowp_bwd", None)
    if module is None or sys.modules.get(EXTENSION_MODULE) is not module:
        raise RuntimeError(
            "the authenticated low-precision extension is not the active "
            f"{EXTENSION_MODULE} module"
        )
    loaded_identity = _authenticate_loaded_extension(module, candidate_identity)
    required_symbols = (
        "quantize_nvfp4_projection_operand",
        "quantize_nvfp4_projection_operand_rmsnorm",
        "rmsnorm_backward_bf16",
    )
    missing_symbols = [name for name in required_symbols if not hasattr(module, name)]
    if missing_symbols:
        raise RuntimeError(f"extension is missing symbols: {missing_symbols}")

    free_device_bytes, total_device_bytes = torch.cuda.mem_get_info()
    minimum_device_bytes = int(args.minimum_free_device_gib * GIB)
    if free_device_bytes < minimum_device_bytes:
        raise RuntimeError(
            "device-memory guard refused benchmark: "
            f"{free_device_bytes / GIB:.2f} GiB free, "
            f"{args.minimum_free_device_gib:.2f} GiB required"
        )
    available_system_bytes = _available_system_bytes()
    minimum_system_bytes = int(args.minimum_free_system_gib * GIB)
    if (
        available_system_bytes is not None
        and available_system_bytes < minimum_system_bytes
    ):
        raise RuntimeError(
            "system-memory guard refused benchmark: "
            f"{available_system_bytes / GIB:.2f} GiB available, "
            f"{args.minimum_free_system_gib:.2f} GiB required"
        )

    thresholds = {
        "min_normalized_cosine": args.min_normalized_cosine,
        "max_normalized_relative_l2": args.max_normalized_relative_l2,
        "max_inv_rms_abs": args.max_inv_rms_abs,
        "min_dx_cosine": args.min_dx_cosine,
        "max_dx_relative_l2": args.max_dx_relative_l2,
        "min_dgamma_cosine": args.min_dgamma_cosine,
        "max_dgamma_relative_l2": args.max_dgamma_relative_l2,
    }
    started = datetime.now(timezone.utc)
    seed_results = [
        _run_seed(
            torch,
            seed=seed,
            epsilon=args.epsilon,
            warmups=args.warmups,
            samples=args.samples,
            thresholds=thresholds,
            fused_forward=b300_prepare_nvfp4_projection_operand_rmsnorm,
            native_pack=b300_prepare_nvfp4_projection_operand,
            fused_backward=b300_rmsnorm_backward,
        )
        for seed in args.seeds
    ]

    # Re-hash the loaded file at the end as well; the benchmark must not bless
    # a result if its extension changed after import.
    final_identity = _authenticate_loaded_extension(module, candidate_identity)
    passed = all(result["passed"] for result in seed_results)
    properties = torch.cuda.get_device_properties(args.device)
    document = {
        "schema": "tk_fa4.fused_rmsnorm_nvfp4_component.v1",
        "status": "pass" if passed else "fail",
        "fail_closed": True,
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "shape_contract": {
            "batch": BATCH,
            "sequence": SEQUENCE,
            "rows": ROWS,
            "hidden": HIDDEN,
            "dtype": "torch.bfloat16",
            "epsilon": args.epsilon,
            "exact_shape_authenticated": True,
        },
        "reference_contract": {
            "forward": (
                "eager FP32 RMSNorm with BF16 publication, followed by the "
                "authenticated extension's exact native-NVFP4 packer"
            ),
            "backward": (
                "closed-form FP32 RMSNorm dx and dgamma using the identical "
                "forward-saved inv_rms; independent eager inv_rms is also reported"
            ),
            "pack_semantics": (
                "fused payload, scale pages, and global decode must be byte-exact "
                "to exact packing of the fused BF16 normalized publication"
            ),
        },
        "configuration": {
            "seeds": list(args.seeds),
            "warmups": args.warmups,
            "samples": args.samples,
            "timing_method": (
                "rotating/interleaved providers measured with CUDA events on "
                "the current stream"
            ),
            "thresholds": thresholds,
            "minimum_free_device_gib": args.minimum_free_device_gib,
            "minimum_free_system_gib": args.minimum_free_system_gib,
            "output": str(output),
        },
        "results": seed_results,
        "passed": passed,
        "provenance": {
            "extension_pre_load": candidate_identity,
            "extension_post_load": loaded_identity,
            "extension_post_benchmark": final_identity,
            "required_extension_symbols": list(required_symbols),
            "harness": _harness_provenance(),
            "command": list(sys.argv if argv is None else [sys.argv[0], *argv]),
            "python": {
                "executable": sys.executable,
                "version": platform.python_version(),
            },
            "torch": {
                "version": torch.__version__,
                "cuda_version": torch.version.cuda,
            },
            "device": {
                "selected_index": args.device,
                "name": properties.name,
                "compute_capability": list(capability),
                "total_memory_bytes": int(total_device_bytes),
                "initial_free_memory_bytes": int(free_device_bytes),
            },
            "system_available_memory_bytes": available_system_bytes,
        },
    }
    _write_new_json(output, document)
    print(json.dumps({
        "output": str(output),
        "status": document["status"],
        "seeds": list(args.seeds),
    }, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
