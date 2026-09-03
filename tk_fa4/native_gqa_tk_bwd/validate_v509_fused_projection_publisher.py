#!/usr/bin/env python3
"""Create-only gate for the fused v509 projection E5M2 dO publisher.

This validator is deliberately restricted to the experimental
B1/S4096/H32/D128 route.  It compares the fused publisher with both the
retained BF16 projection result and the authenticated standalone BF16->E5M2
producer.  It never updates a model, optimizer, checkpoint, or job.

The projection extension and standalone producer are mandatory authenticated
inputs.  The v509 extension and the two natural layer-12 captures form one
optional all-or-nothing group; when supplied, the gate additionally compares
one v509 backward using fused versus standalone E5M2 publication.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import re
import stat
import statistics
import struct
import sys
from collections.abc import Callable, Iterable
from typing import Any

import torch


THIS_FILE = pathlib.Path(__file__).resolve()
THIS_DIR = THIS_FILE.parent
REPO_ROOT = THIS_DIR.parents[1]
PROJECTION_ENV = "TK_FA4_LOWP_BWD_EXTENSION_SOURCE"
PROJECTION_MODULE_NAME = "tk_fa4._C_b300_lowp_bwd"
PRODUCER_MODULE_NAME = "_C_sm100_e5m2_dout_producer_microgate_20260831"
V509_MODULE_NAME = (
    "_C_sm100_gqa_tk_v509_d128_nvfp4_score_e4m3_qkv_e5m2_dout_b1_s4096"
)

BATCH = 1
SEQUENCE = 4096
QUERY_HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128
HIDDEN = QUERY_HEADS * HEAD_DIM
ROWS = BATCH * SEQUENCE
OUTPUT_SHAPE = (BATCH, SEQUENCE, QUERY_HEADS, HEAD_DIM)
STATS_SHAPE = (BATCH, QUERY_HEADS, 1, SEQUENCE)
STATS_NUMEL = BATCH * QUERY_HEADS * SEQUENCE
STATS_BYTES = 2 * STATS_NUMEL * torch.float32.itemsize
E5_PAYLOAD_BYTES = 16_777_216
LOG2_E = math.log2(math.e)
# The projection epilogue evaluates ``fmaf(lse, -log2(e), 8)`` with a
# binary32 multiplier and one final binary32 rounding.  Eager Torch's
# ``8 - lse * LOG2_E`` rounds the multiply first, which differs by one ULP
# for some captured rows.  A binary64 multiply/add is exact for these
# binary32 operands, so casting its result once reproduces CUDA ``fmaf``.
LOG2_E_F32 = struct.unpack("<f", struct.pack("<f", LOG2_E))[0]

EXPECTED_PRODUCER_METADATA: dict[str, object] = {
    "schema": "tkfa4.e5m2_dout_producer_microgate.v1",
    "source_identity": "e5m2_dout_producer_microgate_20260831_v1",
    "standalone_fail_closed": True,
    "production_dispatch_enabled": False,
    "input_dtype": "bfloat16",
    "payload_dtype": "float8_e5m2",
    "encode": "(BF16.float()*4).to(float8_e5m2)",
    "encode_scale": 4.0,
    "decode_scale": 0.25,
    "logical_decode": "published_E5_bytes.float()*0.25",
    "dstat_source": "decoded_bytes_actually_published",
    "physical_dstat": "-4*sum(O*published_E5_bytes.float())",
    "physical_dstat_scale": -4.0,
    "logical_dstat": "-16*sum(O*(published_E5_bytes.float()*0.25))",
    "logical_dstat_scale": -16.0,
    "depth": HEAD_DIM,
    "rows_per_cta": 4,
    "threads": 128,
    "caller_owned_output_api": True,
}
# The authenticated microgate was compiled from its source directory, so its
# __FILE__ metadata is intentionally the exact basename rather than an
# absolute repository path.
PRODUCER_SOURCE_SUFFIX = "e5m2_dout_producer_microgate_20260831.cu"

NATURAL_BOUNDARY_FIELDS = (
    "q_fp8",
    "k_fp8",
    "v_fp8",
    "lstat",
    "forward_lse",
    "forward_attention_output",
    "q_forward_payload_uint8",
    "k_forward_payload_uint8",
    "q_forward_scale_pages_workspace",
    "k_forward_scale_pages_workspace",
    "q_forward_global_scale_workspace",
    "k_forward_global_scale_workspace",
)
NATURAL_DOUT_FIELDS = (
    "dout_bf16_scaled",
    "forward_lse",
    "forward_attention_output",
)

SOURCE_PATHS = (
    THIS_FILE,
    REPO_ROOT / "tk_fa4/interface.py",
    REPO_ROOT / "tk_fa4/__init__.py",
    REPO_ROOT / "tk_fa4/lowp_fa4_bwd/lowp_fa4_bwd.cu",
    REPO_ROOT / "tk_fa4/lowp_fa4_bwd/projection_fp4_epilogue.cuh",
    REPO_ROOT
    / "tk_fa4/lowp_fa4_bwd/"
    "native_tk_d128_nvfp4_score_e5m2_dout_backward.py",
)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_argument(value: str) -> str:
    normalized = value.lower()
    if re.fullmatch(r"[0-9a-f]{64}", normalized) is None:
        raise argparse.ArgumentTypeError("expected exactly 64 hexadecimal digits")
    return normalized


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return result


def stable_regular_file_identity(path: pathlib.Path) -> dict[str, int | str]:
    requested = path.expanduser()
    if not requested.is_absolute():
        raise RuntimeError(f"artifact path must be absolute: {requested}")
    before = requested.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError(f"artifact must be a regular non-symlink file: {requested}")
    resolved = requested.resolve(strict=True)
    digest = sha256(resolved)
    after = resolved.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in fields):
        raise RuntimeError(f"artifact changed while hashing: {resolved}")
    return {
        "path": str(resolved),
        "sha256": digest,
        "bytes": after.st_size,
        "device": after.st_dev,
        "inode": after.st_ino,
        "mtime_ns": after.st_mtime_ns,
    }


def authenticate_file(
    path: pathlib.Path,
    expected_sha256: str,
    expected_bytes: int,
    *,
    label: str,
) -> dict[str, int | str]:
    identity = stable_regular_file_identity(path)
    mismatches: dict[str, object] = {}
    if identity["sha256"] != expected_sha256:
        mismatches["sha256"] = {
            "actual": identity["sha256"],
            "expected": expected_sha256,
        }
    if identity["bytes"] != expected_bytes:
        mismatches["bytes"] = {
            "actual": identity["bytes"],
            "expected": expected_bytes,
        }
    if mismatches:
        raise RuntimeError(f"fail-closed {label} identity mismatch: {mismatches}")
    return identity


def load_extension(
    identity: dict[str, int | str], module_name: str
) -> Any:
    path = pathlib.Path(str(identity["path"]))
    before = stable_regular_file_identity(path)
    if before != identity:
        raise RuntimeError(f"extension changed before import: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot construct import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    after = stable_regular_file_identity(path)
    if after != identity:
        sys.modules.pop(module_name, None)
        raise RuntimeError(f"extension changed while importing: {path}")
    return module


def require_metadata(
    actual: dict[str, object],
    expected: dict[str, object],
    *,
    label: str,
    source_suffix: str | None = None,
) -> None:
    missing = set(expected) - set(actual)
    mismatches = {
        key: {"actual": actual.get(key), "expected": value}
        for key, value in expected.items()
        if actual.get(key) != value or type(actual.get(key)) is not type(value)
    }
    if missing:
        mismatches["missing"] = sorted(missing)
    if source_suffix is not None:
        source = actual.get("source_file")
        normalized = source.replace("\\", "/") if isinstance(source, str) else ""
        if not (
            normalized == source_suffix.removeprefix("/")
            or normalized.endswith(source_suffix)
        ):
            mismatches["source_file"] = {
                "actual": source,
                "expected_suffix": source_suffix,
            }
    if mismatches:
        raise RuntimeError(f"fail-closed {label} metadata mismatch: {mismatches}")


def load_capture(
    identity: dict[str, int | str], required_fields: Iterable[str], *, label: str
) -> dict[str, Any]:
    path = pathlib.Path(str(identity["path"]))
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must contain a tensor dictionary")
    missing = set(required_fields) - set(payload)
    if missing:
        raise RuntimeError(f"{label} is missing fields: {sorted(missing)}")
    if stable_regular_file_identity(path) != identity:
        raise RuntimeError(f"{label} changed while loading")
    return payload


def tensor_contract(
    tensor: torch.Tensor,
    *,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    device: torch.device | None = None,
    name: str,
) -> None:
    if (
        tensor.dtype != dtype
        or tuple(tensor.shape) != shape
        or not tensor.is_contiguous()
        or (device is not None and tensor.device != device)
    ):
        raise RuntimeError(
            f"{name} ABI mismatch: got {tensor.dtype} {tuple(tensor.shape)} "
            f"contiguous={tensor.is_contiguous()} on {tensor.device}; expected "
            f"{dtype} {shape} contiguous"
            + ("" if device is None else f" on {device}")
        )


def byte_comparison(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    if actual.numel() != expected.numel():
        return {
            "equal": False,
            "actual_bytes": actual.numel() * actual.element_size(),
            "expected_bytes": expected.numel() * expected.element_size(),
            "mismatches": None,
        }
    actual_bytes = actual.contiguous().view(torch.uint8)
    expected_bytes = expected.contiguous().view(torch.uint8)
    mismatch = actual_bytes != expected_bytes
    count = int(torch.count_nonzero(mismatch))
    return {
        "equal": count == 0,
        "bytes": mismatch.numel(),
        "mismatches": count,
    }


def float_comparison(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    actual_f = actual.float()
    expected_f = expected.float()
    difference = actual_f - expected_f
    tiny = torch.finfo(torch.float32).tiny
    actual_norm = torch.linalg.vector_norm(actual_f)
    expected_norm = torch.linalg.vector_norm(expected_f)
    denominator = (actual_norm * expected_norm).clamp_min(tiny)
    return {
        "finite": bool(torch.isfinite(actual_f).all()),
        "reference_finite": bool(torch.isfinite(expected_f).all()),
        "relative_l2": float(
            torch.linalg.vector_norm(difference) / expected_norm.clamp_min(tiny)
        ),
        "cosine": float((actual_f * expected_f).sum() / denominator),
        "max_abs_error": float(difference.abs().max()),
        "mean_abs_error": float(difference.abs().mean()),
        "actual_abs_max": float(actual_f.abs().max()),
        "reference_abs_max": float(expected_f.abs().max()),
    }


def tensor_summary(tensor: torch.Tensor, *, decode: float = 1.0) -> dict[str, Any]:
    decoded = tensor.float().mul(decode)
    finite = torch.isfinite(decoded)
    nonzero = int(torch.count_nonzero(decoded))
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "contiguous": bool(tensor.is_contiguous()),
        "numel": tensor.numel(),
        "finite": bool(finite.all()),
        "nonzero": nonzero,
        "zero_fraction": 1.0 - nonzero / max(tensor.numel(), 1),
        "abs_max": float(decoded.abs().max()),
        "mean_abs": float(decoded.abs().mean()),
    }


def byte_range(tensor: torch.Tensor) -> tuple[int, int]:
    return int(tensor.data_ptr()), tensor.numel() * tensor.element_size()


def ranges_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    left_start, left_bytes = left
    right_start, right_bytes = right
    if left_bytes == 0 or right_bytes == 0:
        return False
    return (
        left_start < right_start + right_bytes
        and right_start < left_start + left_bytes
    )


def pointer_contract(
    publication: Any,
    *,
    attention_output: torch.Tensor,
    stats_workspace: torch.Tensor,
    dq_clear: torch.Tensor,
    protected: dict[str, torch.Tensor],
) -> dict[str, Any]:
    payload = publication.dout_backward_e5m2
    dstat = publication.dpsum
    lstat = publication.lse_log2
    expected_lstat_ptr = stats_workspace.data_ptr() + STATS_NUMEL * 4
    aliases = {
        "dout_storage_aliases_attention_output": (
            publication.dout_storage.data_ptr() == attention_output.data_ptr()
        ),
        "dstat_is_workspace_page0": dstat.data_ptr() == stats_workspace.data_ptr(),
        "lstat_is_workspace_page1": lstat.data_ptr() == expected_lstat_ptr,
        "dstat_lstat_pages_do_not_overlap": not ranges_overlap(
            byte_range(dstat), byte_range(lstat)
        ),
    }
    payload_range = byte_range(payload)
    disjoint = {
        name: not ranges_overlap(payload_range, byte_range(tensor))
        for name, tensor in {
            "attention_output": attention_output,
            "stats_workspace": stats_workspace,
            "dq_clear": dq_clear,
            **protected,
        }.items()
    }
    return {
        "pointers": {
            "attention_output": attention_output.data_ptr(),
            "dout_storage": publication.dout_storage.data_ptr(),
            "e5m2_payload": payload.data_ptr(),
            "stats_workspace": stats_workspace.data_ptr(),
            "dstat": dstat.data_ptr(),
            "lstat": lstat.data_ptr(),
            "dq_clear": dq_clear.data_ptr(),
        },
        "aliases": aliases,
        "e5_payload_disjoint_from_inputs_and_workspaces": disjoint,
        "passed": all(aliases.values()) and all(disjoint.values()),
    }


def dstat_reference(
    attention_output: torch.Tensor, payload: torch.Tensor
) -> torch.Tensor:
    # The fused epilogue iterates B,S,H rows; v509 consumes the B,H,1,S page.
    bsh = -4.0 * (attention_output.float() * payload.float()).sum(dim=-1)
    return bsh.permute(0, 2, 1).unsqueeze(2).contiguous()


def lstat_reference(lse: torch.Tensor) -> torch.Tensor:
    if tuple(lse.shape) == STATS_SHAPE:
        head_major = lse
    elif tuple(lse.shape) == (BATCH, SEQUENCE, QUERY_HEADS):
        head_major = lse.permute(0, 2, 1).unsqueeze(2).contiguous()
    else:
        raise RuntimeError(f"unexpected LSE shape: {tuple(lse.shape)}")
    return (head_major.double() * -LOG2_E_F32 + 8.0).float()


def dstat_gate(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    comparison = float_comparison(actual, expected)
    tolerance = max(1.0e-6, comparison["reference_abs_max"] * 1.0e-5)
    checks = {
        "finite": comparison["finite"],
        "reference_finite": comparison["reference_finite"],
        "relative_l2_le_3e-6": comparison["relative_l2"] <= 3.0e-6,
        "max_abs_within_dynamic_tolerance": (
            comparison["max_abs_error"] <= tolerance
        ),
    }
    return {
        "comparison": comparison,
        "max_abs_tolerance": tolerance,
        "checks": checks,
        "passed": all(checks.values()),
    }


def lstat_gate(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    comparison = float_comparison(actual, expected)
    checks = {
        "finite": comparison["finite"],
        "reference_finite": comparison["reference_finite"],
        "relative_l2_le_2e-7": comparison["relative_l2"] <= 2.0e-7,
        "max_abs_le_2e-5": comparison["max_abs_error"] <= 2.0e-5,
    }
    return {"comparison": comparison, "checks": checks, "passed": all(checks.values())}


def producer_publish(
    producer: Any,
    dout_bf16: torch.Tensor,
    attention_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    tensor_contract(
        dout_bf16,
        dtype=torch.bfloat16,
        shape=OUTPUT_SHAPE,
        device=attention_output.device,
        name="retained BF16 dO",
    )
    payload = torch.empty(
        OUTPUT_SHAPE,
        device=dout_bf16.device,
        dtype=torch.float8_e5m2,
    )
    flat_dstat = torch.empty(STATS_NUMEL, device=dout_bf16.device, dtype=torch.float32)
    producer.produce_out(
        dout_bf16.view(-1, HEAD_DIM),
        attention_output.view(-1, HEAD_DIM),
        payload.view(-1, HEAD_DIM),
        flat_dstat,
    )
    dstat = (
        flat_dstat.view(BATCH, SEQUENCE, QUERY_HEADS)
        .permute(0, 2, 1)
        .unsqueeze(2)
        .contiguous()
    )
    return payload, dstat


def make_workspaces(
    device: torch.device, *, dq_sentinel: float
) -> tuple[torch.Tensor, torch.Tensor]:
    stats = torch.empty(STATS_BYTES, device=device, dtype=torch.uint8)
    stats.fill_(0xA5)
    dq = torch.full(OUTPUT_SHAPE, dq_sentinel, device=device, dtype=torch.bfloat16)
    return stats, dq


def run_fused(
    interface: Any,
    input_operand: tuple[torch.Tensor, ...],
    weight_operand: tuple[torch.Tensor, ...],
    attention_output: torch.Tensor,
    lse: torch.Tensor,
    *,
    dq_sentinel: float,
) -> tuple[Any, torch.Tensor, torch.Tensor]:
    stats, dq = make_workspaces(attention_output.device, dq_sentinel=dq_sentinel)
    publication = interface.b300_project_dout_unified_lowp_nvfp4_v509_e5m2(
        input_operand,
        weight_operand,
        attention_output,
        lse,
        stats_workspace=stats,
        dq_clear=dq,
    )
    return publication, stats, dq


def run_e4(
    interface: Any,
    input_operand: tuple[torch.Tensor, ...],
    weight_operand: tuple[torch.Tensor, ...],
    attention_output: torch.Tensor,
    lse: torch.Tensor,
    *,
    store_bf16: bool,
    dq_sentinel: float,
) -> tuple[Any, torch.Tensor, torch.Tensor]:
    stats, dq = make_workspaces(attention_output.device, dq_sentinel=dq_sentinel)
    publication = interface.b300_project_dout_unified_lowp_nvfp4(
        input_operand,
        weight_operand,
        attention_output,
        lse,
        batch=BATCH,
        seqlen=SEQUENCE,
        heads=QUERY_HEADS,
        store_bf16=store_bf16,
        publish_fp8_backward=True,
        publish_stats=True,
        stats_workspace=stats,
        dq_clear=dq,
        probability_log2_lift=8.0,
    )
    return publication, stats, dq


def validate_publication_case(
    *,
    name: str,
    interface: Any,
    producer: Any,
    input_operand: tuple[torch.Tensor, ...],
    weight_operand: tuple[torch.Tensor, ...],
    attention_output: torch.Tensor,
    lse: torch.Tensor,
    require_zero: bool = False,
    require_e5_range: bool = False,
    check_old_e4: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    attention_before = attention_output.clone()
    lse_before = lse.clone()

    retained, _, retained_dq = run_e4(
        interface,
        input_operand,
        weight_operand,
        attention_output,
        lse,
        store_bf16=True,
        dq_sentinel=37.0,
    )
    if retained.dout is None:
        raise RuntimeError("retained projection did not return its BF16 oracle")
    fused, fused_stats, fused_dq = run_fused(
        interface,
        input_operand,
        weight_operand,
        attention_output,
        lse,
        dq_sentinel=-41.0,
    )
    standalone_e5, standalone_dstat = producer_publish(
        producer, retained.dout, attention_output
    )
    repeated, repeated_stats, repeated_dq = run_fused(
        interface,
        input_operand,
        weight_operand,
        attention_output,
        lse,
        dq_sentinel=123.0,
    )
    torch.cuda.synchronize()

    expected_e5 = (retained.dout.float() * 4.0).to(torch.float8_e5m2)
    actual_dstat_reference = dstat_reference(
        attention_output, fused.dout_backward_e5m2
    )
    expected_lstat = lstat_reference(lse)
    e5_vs_bf16 = byte_comparison(fused.dout_backward_e5m2, expected_e5)
    e5_vs_standalone = byte_comparison(
        fused.dout_backward_e5m2, standalone_e5
    )
    deterministic = {
        "e5_payload": byte_comparison(
            fused.dout_backward_e5m2, repeated.dout_backward_e5m2
        ),
        "dstat": byte_comparison(fused.dpsum, repeated.dpsum),
        "lstat": byte_comparison(fused.lse_log2, repeated.lse_log2),
        "dq_first_exact_zero": int(torch.count_nonzero(fused_dq)) == 0,
        "dq_repeat_exact_zero": int(torch.count_nonzero(repeated_dq)) == 0,
    }
    pointer = pointer_contract(
        fused,
        attention_output=attention_output,
        stats_workspace=fused_stats,
        dq_clear=fused_dq,
        protected={
            **{
                f"input_operand_{index}": tensor
                for index, tensor in enumerate(input_operand)
            },
            **{
                f"weight_operand_{index}": tensor
                for index, tensor in enumerate(weight_operand)
            },
            "lse": lse,
        },
    )
    dstat_actual = dstat_gate(fused.dpsum, actual_dstat_reference)
    dstat_standalone = dstat_gate(fused.dpsum, standalone_dstat)
    lstat_result = lstat_gate(fused.lse_log2, expected_lstat)

    dtype_layout = {
        "e5_dtype": fused.dout_backward_e5m2.dtype == torch.float8_e5m2,
        "e5_shape": tuple(fused.dout_backward_e5m2.shape) == OUTPUT_SHAPE,
        "e5_stride": tuple(fused.dout_backward_e5m2.stride())
        == (SEQUENCE * QUERY_HEADS * HEAD_DIM, QUERY_HEADS * HEAD_DIM, HEAD_DIM, 1),
        "e5_contiguous": fused.dout_backward_e5m2.is_contiguous(),
        "dstat_dtype_shape": (
            fused.dpsum.dtype == torch.float32
            and tuple(fused.dpsum.shape) == STATS_SHAPE
        ),
        "lstat_dtype_shape": (
            fused.lse_log2.dtype == torch.float32
            and tuple(fused.lse_log2.shape) == STATS_SHAPE
        ),
        "all_same_device": all(
            tensor.device == attention_output.device
            for tensor in (
                fused.dout_storage,
                fused.dout_backward_e5m2,
                fused.dpsum,
                fused.lse_log2,
                fused_dq,
            )
        ),
    }

    case_checks: dict[str, bool] = {
        "e5_exact_vs_retained_bf16": bool(e5_vs_bf16["equal"]),
        "e5_exact_vs_standalone_producer": bool(e5_vs_standalone["equal"]),
        "actual_byte_dstat": bool(dstat_actual["passed"]),
        "standalone_dstat_equivalence": bool(dstat_standalone["passed"]),
        "lstat_plus_8": bool(lstat_result["passed"]),
        "deterministic_e5": bool(deterministic["e5_payload"]["equal"]),
        "deterministic_dstat": bool(deterministic["dstat"]["equal"]),
        "deterministic_lstat": bool(deterministic["lstat"]["equal"]),
        "dq_clear_first": bool(deterministic["dq_first_exact_zero"]),
        "dq_clear_repeat": bool(deterministic["dq_repeat_exact_zero"]),
        "retained_dq_clear": int(torch.count_nonzero(retained_dq)) == 0,
        "pointer_and_nonoverlap_contract": bool(pointer["passed"]),
        "dtype_layout_contract": all(dtype_layout.values()),
        "attention_output_unmodified": bool(
            torch.equal(attention_output, attention_before)
        ),
        "lse_unmodified": bool(torch.equal(lse, lse_before)),
        "repeat_stats_workspace_is_distinct": (
            repeated_stats.data_ptr() != fused_stats.data_ptr()
        ),
    }

    special: dict[str, Any] = {}
    if require_zero:
        zero_checks = {
            "retained_bf16_exact_zero": int(torch.count_nonzero(retained.dout)) == 0,
            "fused_e5_exact_zero": int(
                torch.count_nonzero(fused.dout_backward_e5m2.view(torch.uint8) & 0x7F)
            )
            == 0,
            "fused_dstat_exact_zero": int(torch.count_nonzero(fused.dpsum)) == 0,
        }
        special["zero"] = zero_checks
        case_checks["exact_zero_case"] = all(zero_checks.values())
    if require_e5_range:
        payload_float = fused.dout_backward_e5m2.float()
        e5_range_checks = {
            "payload_finite": bool(torch.isfinite(payload_float).all()),
            "exercises_values_beyond_e4m3_finite_range": int(
                torch.count_nonzero(payload_float.abs() > 448.0)
            )
            > 0,
            "retained_bf16_finite": bool(torch.isfinite(retained.dout.float()).all()),
        }
        special["e5_range"] = e5_range_checks
        case_checks["finite_e5_range_case"] = all(e5_range_checks.values())

    old_e4: dict[str, Any] | None = None
    if check_old_e4:
        expected_e4 = (retained.dout.float() * 4.0).to(torch.float8_e4m3fn)
        old_no_store, _, old_no_store_dq = run_e4(
            interface,
            input_operand,
            weight_operand,
            attention_output,
            lse,
            store_bf16=False,
            dq_sentinel=-77.0,
        )
        torch.cuda.synchronize()
        if retained.dout_backward_fp8 is None or old_no_store.dout_backward_fp8 is None:
            raise RuntimeError("legacy E4 projection omitted its FP8 publication")
        old_checks = {
            "retained_e4_exact_vs_bf16": byte_comparison(
                retained.dout_backward_fp8, expected_e4
            ),
            "no_store_e4_exact_vs_retained": byte_comparison(
                old_no_store.dout_backward_fp8, retained.dout_backward_fp8
            ),
            "no_store_dstat_exact_vs_retained": byte_comparison(
                old_no_store.dpsum, retained.dpsum
            ),
            "no_store_lstat_exact_vs_retained": byte_comparison(
                old_no_store.lse_log2, retained.lse_log2
            ),
            "actual_byte_dstat": dstat_gate(
                retained.dpsum,
                dstat_reference(attention_output, retained.dout_backward_fp8),
            ),
            "lstat_plus_8": lstat_gate(retained.lse_log2, expected_lstat),
            "no_store_dq_exact_zero": int(torch.count_nonzero(old_no_store_dq)) == 0,
        }
        old_passed = (
            bool(old_checks["retained_e4_exact_vs_bf16"]["equal"])
            and bool(old_checks["no_store_e4_exact_vs_retained"]["equal"])
            and bool(old_checks["no_store_dstat_exact_vs_retained"]["equal"])
            and bool(old_checks["no_store_lstat_exact_vs_retained"]["equal"])
            and bool(old_checks["actual_byte_dstat"]["passed"])
            and bool(old_checks["lstat_plus_8"]["passed"])
            and bool(old_checks["no_store_dq_exact_zero"])
        )
        old_e4 = {"checks": old_checks, "passed": old_passed}
        case_checks["old_e4_regression_hook"] = old_passed

    result = {
        "name": name,
        "passed": all(case_checks.values()),
        "checks": case_checks,
        "e5_payload": tensor_summary(fused.dout_backward_e5m2, decode=0.25),
        "e5_exact_vs_retained_bf16": e5_vs_bf16,
        "e5_exact_vs_standalone": e5_vs_standalone,
        "actual_byte_dstat": dstat_actual,
        "standalone_dstat_equivalence": dstat_standalone,
        "lstat_plus_8": lstat_result,
        "determinism": deterministic,
        "dtype_layout": dtype_layout,
        "pointer_contract": pointer,
        "special": special,
        "old_e4_regression": old_e4,
    }
    runtime = {
        "retained_bf16_dout": retained.dout,
        "fused": fused,
        "standalone_e5": standalone_e5,
        "standalone_dstat": standalone_dstat,
    }
    return result, runtime


def rotated_timing(
    candidates: dict[str, Callable[[], object]], warmups: int, samples: int
) -> dict[str, Any]:
    names = list(candidates)
    for iteration in range(warmups):
        order = names[iteration % len(names) :] + names[: iteration % len(names)]
        for name in order:
            candidates[name]()
    torch.cuda.synchronize()
    elapsed: dict[str, list[float]] = {name: [] for name in names}
    for iteration in range(samples):
        order = names[iteration % len(names) :] + names[: iteration % len(names)]
        for name in order:
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            candidates[name]()
            stop.record()
            stop.synchronize()
            elapsed[name].append(float(start.elapsed_time(stop)))
    return {
        name: {
            "median_ms": statistics.median(values),
            "mean_ms": statistics.fmean(values),
            "minimum_ms": min(values),
            "p10_ms": sorted(values)[int(0.10 * (len(values) - 1))],
            "p90_ms": sorted(values)[int(0.90 * (len(values) - 1))],
            "samples": len(values),
        }
        for name, values in elapsed.items()
    }


def run_timing_gate(
    *,
    interface: Any,
    input_operand: tuple[torch.Tensor, ...],
    weight_operand: tuple[torch.Tensor, ...],
    attention_output: torch.Tensor,
    lse: torch.Tensor,
    warmups: int,
    samples: int,
    maximum_median_ratio: float,
    maximum_p90_ratio: float,
) -> dict[str, Any]:
    e4_stats, e4_dq = make_workspaces(attention_output.device, dq_sentinel=0.0)
    e5_stats, e5_dq = make_workspaces(attention_output.device, dq_sentinel=0.0)

    def e4_call() -> object:
        return interface.b300_project_dout_unified_lowp_nvfp4(
            input_operand,
            weight_operand,
            attention_output,
            lse,
            batch=BATCH,
            seqlen=SEQUENCE,
            heads=QUERY_HEADS,
            store_bf16=False,
            publish_fp8_backward=True,
            publish_stats=True,
            stats_workspace=e4_stats,
            dq_clear=e4_dq,
            probability_log2_lift=8.0,
        )

    def e5_call() -> object:
        return interface.b300_project_dout_unified_lowp_nvfp4_v509_e5m2(
            input_operand,
            weight_operand,
            attention_output,
            lse,
            stats_workspace=e5_stats,
            dq_clear=e5_dq,
        )

    timing = rotated_timing(
        {"legacy_e4_no_bf16_store": e4_call, "fused_e5_no_bf16_store": e5_call},
        warmups,
        samples,
    )
    median_ratio = (
        timing["fused_e5_no_bf16_store"]["median_ms"]
        / timing["legacy_e4_no_bf16_store"]["median_ms"]
    )
    p90_ratio = (
        timing["fused_e5_no_bf16_store"]["p90_ms"]
        / timing["legacy_e4_no_bf16_store"]["p90_ms"]
    )
    checks = {
        "median_ratio_within_limit": median_ratio <= maximum_median_ratio,
        "p90_ratio_within_limit": p90_ratio <= maximum_p90_ratio,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "fused_e5_over_legacy_e4_median": median_ratio,
        "fused_e5_over_legacy_e4_p90": p90_ratio,
        "maximum_allowed_median_ratio": maximum_median_ratio,
        "maximum_allowed_p90_ratio": maximum_p90_ratio,
        "rotated_order": True,
        "scope": (
            "projection plus fused dO/stat publication and dQ clear only; "
            "not an end-to-end throughput claim"
        ),
        "timing": timing,
    }


def run_live_allocation_gate(
    *,
    interface: Any,
    input_operand: tuple[torch.Tensor, ...],
    weight_operand: tuple[torch.Tensor, ...],
    attention_output: torch.Tensor,
    lse: torch.Tensor,
) -> dict[str, Any]:
    """Measure only live PyTorch tensor storage retained by one fused call."""
    stats, dq = make_workspaces(attention_output.device, dq_sentinel=0.0)

    def call() -> Any:
        return interface.b300_project_dout_unified_lowp_nvfp4_v509_e5m2(
            input_operand,
            weight_operand,
            attention_output,
            lse,
            stats_workspace=stats,
            dq_clear=dq,
        )

    # Exercise lazy module and allocator setup, then release every returned
    # publication. memory_allocated counts live tensor storage rather than
    # cached blocks, so allocator caching does not enter the exact delta.
    warmup_calls = 3
    for _ in range(warmup_calls):
        warm = call()
        torch.cuda.synchronize()
        del warm
    gc.collect()
    torch.cuda.synchronize()

    device = attention_output.device
    before_allocated = torch.cuda.memory_allocated(device)
    before_reserved = torch.cuda.memory_reserved(device)
    held = call()
    torch.cuda.synchronize()
    after_allocated = torch.cuda.memory_allocated(device)
    after_reserved = torch.cuda.memory_reserved(device)
    allocated_delta = after_allocated - before_allocated
    expected_payload_bytes = E5_PAYLOAD_BYTES
    payload_contract = {
        "dtype": held.dout_backward_e5m2.dtype == torch.float8_e5m2,
        "shape": tuple(held.dout_backward_e5m2.shape) == OUTPUT_SHAPE,
        "contiguous": held.dout_backward_e5m2.is_contiguous(),
        "element_size_one_byte": held.dout_backward_e5m2.element_size() == 1,
        "nbytes_exact": (
            held.dout_backward_e5m2.numel()
            * held.dout_backward_e5m2.element_size()
            == E5_PAYLOAD_BYTES
        ),
    }
    view_contract = {
        "descriptor_storage_aliases_attention_output": (
            held.dout_storage.data_ptr() == attention_output.data_ptr()
        ),
        "dstat_is_preallocated_workspace_page0": (
            held.dpsum.data_ptr() == stats.data_ptr()
        ),
        "lstat_is_preallocated_workspace_page1": (
            held.lse_log2.data_ptr() == stats.data_ptr() + STATS_NUMEL * 4
        ),
    }
    exact_delta = allocated_delta == expected_payload_bytes

    # Keep the publication alive through the after-reading above, then prove
    # that releasing it returns process-local live allocation to the baseline.
    del held
    gc.collect()
    torch.cuda.synchronize()
    released_allocated = torch.cuda.memory_allocated(device)
    released_reserved = torch.cuda.memory_reserved(device)
    released_to_baseline = released_allocated == before_allocated
    checks = {
        "payload_contract": all(payload_contract.values()),
        "descriptor_and_stats_are_storage_aliases": all(
            view_contract.values()
        ),
        "exactly_one_e5_payload_allocated": exact_delta,
        "publication_release_returns_to_baseline": released_to_baseline,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "payload_contract": payload_contract,
        "view_contract": view_contract,
        "warmup_calls": warmup_calls,
        "expected_live_allocation_delta_bytes": expected_payload_bytes,
        "observed_live_allocation_delta_bytes": allocated_delta,
        "memory_allocated_bytes": {
            "before": before_allocated,
            "while_publication_held": after_allocated,
            "after_publication_release": released_allocated,
        },
        "memory_reserved_bytes_non_gating": {
            "before": before_reserved,
            "while_publication_held": after_reserved,
            "after_publication_release": released_reserved,
        },
        "measurement_scope": (
            "Exact process-local torch.cuda.memory_allocated delta after three "
            "warmup calls, with caller-owned stats and dQ storage preallocated "
            "and the returned publication held. Cached allocator blocks and "
            "non-PyTorch driver allocations are deliberately excluded."
        ),
    }


def sanitized_runtime_error(error: RuntimeError) -> str:
    first_line = str(error).splitlines()[0]
    without_addresses = re.sub(
        r"0x[0-9a-fA-F]+",
        "<address>",
        first_line,
    )
    return without_addresses[:512]


def run_alias_rejection_probes(
    *,
    interface: Any,
    input_operand: tuple[torch.Tensor, ...],
    weight_operand: tuple[torch.Tensor, ...],
    attention_output: torch.Tensor,
    lse: torch.Tensor,
) -> dict[str, Any]:
    """Require raw C++ range preflight to reject two high-level aliases."""
    attention_before = attention_output.clone()

    safe_stats = torch.full(
        (STATS_BYTES,),
        0x3C,
        device=attention_output.device,
        dtype=torch.uint8,
    )
    dq_error: RuntimeError | None = None
    dq_returned = None
    try:
        dq_returned = (
            interface.b300_project_dout_unified_lowp_nvfp4_v509_e5m2(
                input_operand,
                weight_operand,
                attention_output,
                lse,
                stats_workspace=safe_stats,
                dq_clear=attention_output,
            )
        )
    except RuntimeError as error:
        dq_error = error
    torch.cuda.synchronize()
    if dq_returned is not None:
        del dq_returned
    dq_message = "" if dq_error is None else sanitized_runtime_error(dq_error)
    dq_probe = {
        "raised_runtime_error": dq_error is not None,
        "message": dq_message,
        "message_identifies_overlap": (
            "attention_output must not overlap" in dq_message
            and "dq_clear" in dq_message
        ),
        "attention_output_unmodified": bool(
            torch.equal(attention_output, attention_before)
        ),
        "safe_stats_unmodified": bool(torch.all(safe_stats == 0x3C)),
    }
    dq_probe["passed"] = all(
        value for key, value in dq_probe.items() if key != "message"
    )
    if not dq_probe["attention_output_unmodified"]:
        attention_output.copy_(attention_before)

    stats_alias = (
        attention_output.view(torch.uint8)
        .reshape(-1)
        .narrow(0, 0, STATS_BYTES)
    )
    safe_dq = torch.full(
        OUTPUT_SHAPE,
        29.0,
        device=attention_output.device,
        dtype=torch.bfloat16,
    )
    stats_error: RuntimeError | None = None
    stats_returned = None
    try:
        stats_returned = (
            interface.b300_project_dout_unified_lowp_nvfp4_v509_e5m2(
                input_operand,
                weight_operand,
                attention_output,
                lse,
                stats_workspace=stats_alias,
                dq_clear=safe_dq,
            )
        )
    except RuntimeError as error:
        stats_error = error
    torch.cuda.synchronize()
    if stats_returned is not None:
        del stats_returned
    stats_message = (
        "" if stats_error is None else sanitized_runtime_error(stats_error)
    )
    stats_probe = {
        "raised_runtime_error": stats_error is not None,
        "message": stats_message,
        "message_identifies_overlap": (
            "attention_output must not overlap" in stats_message
            and "stats_workspace" in stats_message
        ),
        "attention_output_unmodified": bool(
            torch.equal(attention_output, attention_before)
        ),
        "safe_dq_unmodified": bool(torch.all(safe_dq == 29.0)),
    }
    stats_probe["passed"] = all(
        value for key, value in stats_probe.items() if key != "message"
    )
    if not stats_probe["attention_output_unmodified"]:
        attention_output.copy_(attention_before)
    torch.cuda.synchronize()
    gc.collect()

    return {
        "passed": bool(dq_probe["passed"] and stats_probe["passed"]),
        "call_surface": (
            "high-level exact v509 API reaching the raw C++ storage-range "
            "preflight; unchanged write targets prove rejection before launch"
        ),
        "messages_are_first_line_address_scrubbed": True,
        "dq_clear_aliases_attention_output": dq_probe,
        "stats_workspace_aliases_attention_output": stats_probe,
    }


def validate_natural_capture_contracts(
    boundary: dict[str, Any], dout_capture: dict[str, Any]
) -> dict[str, bool]:
    expected_boundary = {
        "q_fp8": (torch.float8_e4m3fn, (1, 4096, 32, 128)),
        "k_fp8": (torch.float8_e4m3fn, (1, 4096, 8, 128)),
        "v_fp8": (torch.float8_e4m3fn, (1, 4096, 8, 128)),
        "lstat": (torch.float32, STATS_SHAPE),
        "forward_lse": (torch.float32, STATS_SHAPE),
        "forward_attention_output": (torch.bfloat16, OUTPUT_SHAPE),
        "q_forward_payload_uint8": (torch.uint8, (1, 32, 4096, 64)),
        "k_forward_payload_uint8": (torch.uint8, (1, 8, 4096, 64)),
        "q_forward_scale_pages_workspace": (
            torch.float8_e4m3fn,
            (1, 32, 64, 512),
        ),
        "k_forward_scale_pages_workspace": (
            torch.float8_e4m3fn,
            (1, 64, 16, 512),
        ),
        "q_forward_global_scale_workspace": (torch.float32, (1, 32)),
        "k_forward_global_scale_workspace": (torch.float32, (1, 8)),
    }
    for name, (dtype, shape) in expected_boundary.items():
        tensor_contract(
            boundary[name],
            dtype=dtype,
            shape=shape,
            name=f"boundary.{name}",
        )
    tensor_contract(
        dout_capture["dout_bf16_scaled"],
        dtype=torch.bfloat16,
        shape=OUTPUT_SHAPE,
        name="dout_capture.dout_bf16_scaled",
    )
    tensor_contract(
        dout_capture["forward_lse"],
        dtype=torch.float32,
        shape=STATS_SHAPE,
        name="dout_capture.forward_lse",
    )
    tensor_contract(
        dout_capture["forward_attention_output"],
        dtype=torch.bfloat16,
        shape=OUTPUT_SHAPE,
        name="dout_capture.forward_attention_output",
    )
    shared = {
        name: bool(torch.equal(boundary[name], dout_capture[name]))
        for name in ("forward_lse", "forward_attention_output")
    }
    if not all(shared.values()):
        raise RuntimeError(
            "natural captures disagree at shared forward boundary: "
            f"{shared}"
        )
    return shared


def run_v509(
    module: Any,
    boundary: dict[str, torch.Tensor],
    *,
    dout_e5: torch.Tensor,
    lstat: torch.Tensor,
    dstat: torch.Tensor,
    sentinel: float,
) -> dict[str, torch.Tensor]:
    q = boundary["q_fp8"]
    k = boundary["k_fp8"]
    outputs = {
        "dq": torch.full(q.shape, sentinel, device=q.device, dtype=torch.bfloat16),
        "dk": torch.full(k.shape, sentinel, device=q.device, dtype=torch.bfloat16),
        "dv": torch.full(k.shape, sentinel, device=q.device, dtype=torch.bfloat16),
    }
    module.backward_nvfp4_score_e4m3_qkv_e5m2_dout_bshd_precomputed_out(
        q,
        k,
        boundary["v_fp8"],
        dout_e5,
        lstat,
        dstat,
        outputs["dq"],
        outputs["dk"],
        outputs["dv"],
        boundary["q_native"],
        boundary["k_native"],
        boundary["q_scale"],
        boundary["k_scale"],
        boundary["q_global"],
        boundary["k_global"],
        HEAD_DIM**-0.5,
    )
    torch.cuda.synchronize()
    return outputs


def run_optional_v509_equivalence(
    *,
    module: Any,
    interface: Any,
    boundary_cpu: dict[str, Any],
    runtime: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    metadata = dict(module.native_tk_d128_backward_metadata())
    route = interface.b300_require_v509_e5m2_dout_route(metadata)
    boundary = {
        "q_fp8": boundary_cpu["q_fp8"].to(device),
        "k_fp8": boundary_cpu["k_fp8"].to(device),
        "v_fp8": boundary_cpu["v_fp8"].to(device),
        "q_native": boundary_cpu["q_forward_payload_uint8"]
        .to(device)
        .view(torch.float4_e2m1fn_x2),
        "k_native": boundary_cpu["k_forward_payload_uint8"]
        .to(device)
        .view(torch.float4_e2m1fn_x2),
        "q_scale": boundary_cpu["q_forward_scale_pages_workspace"].to(device),
        "k_scale": boundary_cpu["k_forward_scale_pages_workspace"].to(device),
        "q_global": boundary_cpu["q_forward_global_scale_workspace"].to(device),
        "k_global": boundary_cpu["k_forward_global_scale_workspace"].to(device),
    }
    fused = runtime["fused"]
    standalone_e5 = runtime["standalone_e5"]
    standalone_dstat = runtime["standalone_dstat"]
    fused_outputs = run_v509(
        module,
        boundary,
        dout_e5=fused.dout_backward_e5m2,
        lstat=fused.lse_log2,
        dstat=fused.dpsum,
        sentinel=float("nan"),
    )
    standalone_outputs = run_v509(
        module,
        boundary,
        dout_e5=standalone_e5,
        lstat=fused.lse_log2,
        dstat=standalone_dstat,
        sentinel=321.0,
    )
    outputs: dict[str, Any] = {}
    passed = True
    for name in ("dq", "dk", "dv"):
        comparison = float_comparison(fused_outputs[name], standalone_outputs[name])
        checks = {
            "fused_finite": comparison["finite"],
            "standalone_finite": comparison["reference_finite"],
            "fused_nontrivial": int(torch.count_nonzero(fused_outputs[name])) > 0,
            "standalone_nontrivial": int(
                torch.count_nonzero(standalone_outputs[name])
            )
            > 0,
            "relative_l2_le_1e-4": comparison["relative_l2"] <= 1.0e-4,
            "cosine_ge_0.999999": comparison["cosine"] >= 0.999999,
        }
        item_passed = all(checks.values())
        passed = passed and item_passed
        outputs[name] = {
            "comparison": comparison,
            "fused": tensor_summary(fused_outputs[name], decode=0.25),
            "standalone": tensor_summary(standalone_outputs[name], decode=0.25),
            "bitwise_equal": bool(
                torch.equal(fused_outputs[name], standalone_outputs[name])
            ),
            "checks": checks,
            "passed": item_passed,
        }
    capture_lstat = boundary_cpu["lstat"].to(device)
    lstat_vs_capture = lstat_gate(fused.lse_log2, capture_lstat)
    passed = passed and lstat_vs_capture["passed"]
    return {
        "passed": bool(passed),
        "scope": (
            "authenticated natural layer-12 Q/K/V, native NVFP4 score payloads, "
            "attention output, and LSE; dO is the NVFP4 identity-projection of "
            "the authenticated natural BF16 dO seed"
        ),
        "route": route,
        "metadata": metadata,
        "payload_identity": byte_comparison(
            fused.dout_backward_e5m2, standalone_e5
        ),
        "dstat_equivalence": dstat_gate(fused.dpsum, standalone_dstat),
        "publisher_lstat_vs_captured_lstat": lstat_vs_capture,
        "outputs": outputs,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the fail-closed fused v509 E5M2 projection publisher "
            "and create one immutable JSON receipt."
        )
    )
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--projection-module", type=pathlib.Path, required=True)
    parser.add_argument("--projection-sha256", type=sha256_argument, required=True)
    parser.add_argument("--projection-bytes", type=positive_int, required=True)
    parser.add_argument("--standalone-producer", type=pathlib.Path, required=True)
    parser.add_argument(
        "--standalone-producer-sha256",
        type=sha256_argument,
        required=True,
    )
    parser.add_argument("--standalone-producer-bytes", type=positive_int, required=True)
    parser.add_argument("--warmups", type=int, default=7)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument(
        "--maximum-e5-over-e4-median",
        type=float,
        default=1.03,
    )
    parser.add_argument(
        "--maximum-e5-over-e4-p90",
        type=float,
        default=1.05,
    )
    parser.add_argument("--seed", type=int, default=20260831)

    optional = parser.add_argument_group("optional authenticated natural v509 gate")
    optional.add_argument("--v509-module", type=pathlib.Path)
    optional.add_argument("--v509-sha256", type=sha256_argument)
    optional.add_argument("--v509-bytes", type=positive_int)
    optional.add_argument("--natural-boundary", type=pathlib.Path)
    optional.add_argument("--natural-boundary-sha256", type=sha256_argument)
    optional.add_argument("--natural-boundary-bytes", type=positive_int)
    optional.add_argument("--natural-dout-capture", type=pathlib.Path)
    optional.add_argument("--natural-dout-capture-sha256", type=sha256_argument)
    optional.add_argument("--natural-dout-capture-bytes", type=positive_int)
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    if not (0 <= args.warmups <= 40 and 3 <= args.samples <= 101):
        raise RuntimeError("bounded timing requires warmups 0..40 and samples 3..101")
    if not (1.0 <= args.maximum_e5_over_e4_median <= 1.25):
        raise RuntimeError(
            "maximum median E5/E4 ratio must be in [1.0, 1.25]"
        )
    if not (1.0 <= args.maximum_e5_over_e4_p90 <= 1.25):
        raise RuntimeError(
            "maximum p90 E5/E4 ratio must be in [1.0, 1.25]"
        )
    if args.maximum_e5_over_e4_median > args.maximum_e5_over_e4_p90:
        raise RuntimeError(
            "maximum median E5/E4 ratio cannot exceed the p90 limit"
        )
    output = args.output.expanduser().resolve()
    if output.exists():
        raise RuntimeError(f"create-only receipt already exists: {output}")
    if not output.parent.is_dir():
        raise RuntimeError("receipt parent must already exist")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("fail-closed: CUDA_VISIBLE_DEVICES must be exactly 0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("fail-closed: exactly one visible CUDA GPU is required")
    capability = torch.cuda.get_device_capability(0)
    if capability[0] != 10:
        raise RuntimeError(
            "fail-closed: exact SM100-class GPU required, "
            f"got {capability}"
        )

    optional_values = (
        args.v509_module,
        args.v509_sha256,
        args.v509_bytes,
        args.natural_boundary,
        args.natural_boundary_sha256,
        args.natural_boundary_bytes,
        args.natural_dout_capture,
        args.natural_dout_capture_sha256,
        args.natural_dout_capture_bytes,
    )
    if any(value is not None for value in optional_values) and not all(
        value is not None for value in optional_values
    ):
        raise RuntimeError(
            "v509 module, natural boundary, and natural dO capture identities "
            "must be supplied together"
        )
    run_natural = all(value is not None for value in optional_values)

    source_hashes_before = {str(path): sha256(path) for path in SOURCE_PATHS}
    projection_identity = authenticate_file(
        args.projection_module,
        args.projection_sha256,
        args.projection_bytes,
        label="projection extension",
    )
    producer_identity = authenticate_file(
        args.standalone_producer,
        args.standalone_producer_sha256,
        args.standalone_producer_bytes,
        label="standalone E5M2 producer",
    )
    optional_identities: dict[str, dict[str, int | str]] = {}
    if run_natural:
        assert args.v509_module is not None
        assert args.v509_sha256 is not None
        assert args.v509_bytes is not None
        assert args.natural_boundary is not None
        assert args.natural_boundary_sha256 is not None
        assert args.natural_boundary_bytes is not None
        assert args.natural_dout_capture is not None
        assert args.natural_dout_capture_sha256 is not None
        assert args.natural_dout_capture_bytes is not None
        optional_identities = {
            "v509_module": authenticate_file(
                args.v509_module,
                args.v509_sha256,
                args.v509_bytes,
                label="v509 extension",
            ),
            "natural_boundary": authenticate_file(
                args.natural_boundary,
                args.natural_boundary_sha256,
                args.natural_boundary_bytes,
                label="natural boundary capture",
            ),
            "natural_dout_capture": authenticate_file(
                args.natural_dout_capture,
                args.natural_dout_capture_sha256,
                args.natural_dout_capture_bytes,
                label="natural dO capture",
            ),
        }

    # Set the authenticated projection image before importing tk_fa4.  The
    # interface's own stable-file loader then verifies the same image again.
    os.environ[PROJECTION_ENV] = str(projection_identity["path"])
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import tk_fa4.interface as interface
    from tk_fa4.lowp_fa4_bwd.native_tk_d128_nvfp4_score_e5m2_dout_backward import (
        EXPECTED_EXTENSION_METADATA as expected_v509_metadata,
    )

    loaded_projection = sys.modules.get(PROJECTION_MODULE_NAME)
    loaded_identity = getattr(
        loaded_projection, "_tk_fa4_loaded_artifact_identity", None
    )
    if loaded_identity != projection_identity:
        raise RuntimeError(
            "tk_fa4 did not load the exact authenticated projection artifact"
        )
    producer = load_extension(producer_identity, PRODUCER_MODULE_NAME)
    producer_metadata = dict(producer.metadata())
    require_metadata(
        producer_metadata,
        EXPECTED_PRODUCER_METADATA,
        label="standalone producer",
        source_suffix=PRODUCER_SOURCE_SUFFIX,
    )

    v509_module = None
    boundary_cpu = None
    dout_capture_cpu = None
    shared_capture_identity: dict[str, bool] | None = None
    if run_natural:
        v509_module = load_extension(
            optional_identities["v509_module"], V509_MODULE_NAME
        )
        actual_v509_metadata = dict(v509_module.native_tk_d128_backward_metadata())
        require_metadata(
            actual_v509_metadata,
            expected_v509_metadata,
            label="v509 backward",
        )
        boundary_cpu = load_capture(
            optional_identities["natural_boundary"],
            NATURAL_BOUNDARY_FIELDS,
            label="natural boundary capture",
        )
        dout_capture_cpu = load_capture(
            optional_identities["natural_dout_capture"],
            NATURAL_DOUT_FIELDS,
            label="natural dO capture",
        )
        shared_capture_identity = validate_natural_capture_contracts(
            boundary_cpu, dout_capture_cpu
        )

    # This also authenticates the fused publisher metadata.  In the non-v509
    # mode, the exact reviewed v509 contract is used solely as a route receipt;
    # no backward kernel is loaded or launched.
    route_receipt = interface.b300_require_v509_e5m2_dout_route(
        dict(expected_v509_metadata)
        if v509_module is None
        else dict(v509_module.native_tk_d128_backward_metadata())
    )

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if dout_capture_cpu is None:
        projection_input = torch.empty(
            (ROWS, HIDDEN), device=device, dtype=torch.bfloat16
        ).normal_(mean=0.0, std=0.04)
        projection_weight = torch.empty(
            (HIDDEN, HIDDEN), device=device, dtype=torch.bfloat16
        ).normal_(mean=0.0, std=0.02)
        attention_output = torch.empty(
            OUTPUT_SHAPE, device=device, dtype=torch.bfloat16
        ).normal_(mean=0.0, std=0.10)
        lse = torch.empty(STATS_SHAPE, device=device, dtype=torch.float32).normal_(
            mean=6.0, std=0.5
        )
        projection_seed_scope = "deterministic synthetic projection"
    else:
        projection_input = (
            dout_capture_cpu["dout_bf16_scaled"]
            .reshape(ROWS, HIDDEN)
            .to(device)
            .contiguous()
        )
        projection_weight = torch.eye(HIDDEN, device=device, dtype=torch.bfloat16)
        attention_output = dout_capture_cpu["forward_attention_output"].to(device)
        lse = dout_capture_cpu["forward_lse"].to(device)
        projection_seed_scope = (
            "authenticated natural BF16 dO seed through an NVFP4 identity "
            "projection"
        )

    input_operand = tuple(
        interface.b300_prepare_nvfp4_projection_operand(projection_input)
    )
    weight_operand = tuple(
        interface.b300_prepare_nvfp4_projection_weight(projection_weight)
    )
    del projection_weight

    base_result, base_runtime = validate_publication_case(
        name="base",
        interface=interface,
        producer=producer,
        input_operand=input_operand,
        weight_operand=weight_operand,
        attention_output=attention_output,
        lse=lse,
        check_old_e4=True,
    )

    zero_input = torch.zeros_like(projection_input)
    zero_operand = tuple(
        interface.b300_prepare_nvfp4_projection_operand(zero_input)
    )
    zero_result, zero_runtime = validate_publication_case(
        name="exact_zero",
        interface=interface,
        producer=producer,
        input_operand=zero_operand,
        weight_operand=weight_operand,
        attention_output=attention_output,
        lse=lse,
        require_zero=True,
    )
    del zero_input, zero_operand, zero_runtime

    extreme_input = torch.empty_like(projection_input).normal_(
        mean=0.0,
        std=80.0,
    )
    extreme_operand = tuple(
        interface.b300_prepare_nvfp4_projection_operand(extreme_input)
    )
    extreme_result, extreme_runtime = validate_publication_case(
        name="finite_e5_range",
        interface=interface,
        producer=producer,
        input_operand=extreme_operand,
        weight_operand=weight_operand,
        attention_output=attention_output,
        lse=lse,
        require_e5_range=True,
    )
    del extreme_input, extreme_operand, extreme_runtime

    alias_rejections = run_alias_rejection_probes(
        interface=interface,
        input_operand=input_operand,
        weight_operand=weight_operand,
        attention_output=attention_output,
        lse=lse,
    )
    live_allocation = run_live_allocation_gate(
        interface=interface,
        input_operand=input_operand,
        weight_operand=weight_operand,
        attention_output=attention_output,
        lse=lse,
    )

    timing = run_timing_gate(
        interface=interface,
        input_operand=input_operand,
        weight_operand=weight_operand,
        attention_output=attention_output,
        lse=lse,
        warmups=args.warmups,
        samples=args.samples,
        maximum_median_ratio=args.maximum_e5_over_e4_median,
        maximum_p90_ratio=args.maximum_e5_over_e4_p90,
    )

    natural_v509: dict[str, Any] = {
        "ran": False,
        "passed": True,
        "reason": "optional authenticated v509/capture group not supplied",
    }
    if v509_module is not None:
        assert boundary_cpu is not None
        natural_v509 = {
            "ran": True,
            **run_optional_v509_equivalence(
                module=v509_module,
                interface=interface,
                boundary_cpu=boundary_cpu,
                runtime=base_runtime,
                device=device,
            ),
        }

    torch.cuda.synchronize()
    artifact_identities_after = {
        "projection_module": stable_regular_file_identity(
            pathlib.Path(str(projection_identity["path"]))
        ),
        "standalone_producer": stable_regular_file_identity(
            pathlib.Path(str(producer_identity["path"]))
        ),
        **{
            name: stable_regular_file_identity(pathlib.Path(str(identity["path"])))
            for name, identity in optional_identities.items()
        },
    }
    artifact_identities_before = {
        "projection_module": projection_identity,
        "standalone_producer": producer_identity,
        **optional_identities,
    }
    artifacts_unchanged = artifact_identities_after == artifact_identities_before
    source_hashes_after = {str(path): sha256(path) for path in SOURCE_PATHS}
    sources_unchanged = source_hashes_after == source_hashes_before

    passed = bool(
        base_result["passed"]
        and zero_result["passed"]
        and extreme_result["passed"]
        and alias_rejections["passed"]
        and live_allocation["passed"]
        and timing["passed"]
        and natural_v509["passed"]
        and artifacts_unchanged
        and sources_unchanged
    )
    receipt = {
        "schema": "tkfa4.v509_fused_projection_publisher.gate.v1",
        "passed": passed,
        "scope": {
            "shape": "B1/S4096/H32/D128",
            "device_selection": "CUDA_VISIBLE_DEVICES=0 and exactly one visible GPU",
            "mutation": (
                "create-only JSON receipt; no model, optimizer, checkpoint, "
                "repository, binary, job, or scheduler mutation"
            ),
            "projection_seed": projection_seed_scope,
            "performance": "bounded rotated projection-only comparison",
            "allocation": (
                "process-local live PyTorch CUDA tensor storage only"
            ),
        },
        "configuration": {
            "seed": args.seed,
            "warmups": args.warmups,
            "samples": args.samples,
            "maximum_e5_over_e4_median": (
                args.maximum_e5_over_e4_median
            ),
            "maximum_e5_over_e4_p90": args.maximum_e5_over_e4_p90,
            "natural_v509_requested": run_natural,
        },
        "device": {
            "name": torch.cuda.get_device_name(0),
            "capability": list(capability),
            "visible_devices": torch.cuda.device_count(),
        },
        "artifacts_before": artifact_identities_before,
        "artifacts_after": artifact_identities_after,
        "artifacts_unchanged": artifacts_unchanged,
        "sources_before": source_hashes_before,
        "sources_after": source_hashes_after,
        "sources_unchanged": sources_unchanged,
        "publisher_backward_route": route_receipt,
        "standalone_producer_metadata": producer_metadata,
        "shared_natural_capture_identity": shared_capture_identity,
        "cases": {
            "base": base_result,
            "exact_zero": zero_result,
            "finite_e5_range": extreme_result,
        },
        "alias_rejection_probes": alias_rejections,
        "live_allocation_contract": live_allocation,
        "rotated_e4_e5_timing": timing,
        "natural_v509_equivalence": natural_v509,
        "claims_excluded": (
            "This receipt is not a whole-step throughput, optimizer, resume, "
            "long-run convergence, or production-dispatch result."
        ),
    }
    if sha256(THIS_FILE) != source_hashes_before[str(THIS_FILE)]:
        raise RuntimeError("validator changed while running")
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(output)
    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())
