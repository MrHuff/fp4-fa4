#!/usr/bin/env python3
"""Authenticated natural-capture gate for experimental dense-score v510.

This script is deliberately fail-closed to B1/S4096/Hq32/Hkv8/D128. It does
not quantize operands or synthesize statistics: the capture must contain the
exact represented E4M3 Q/K/V, represented E5M2 dO, and matched lstat/dstat
pages consumed by the candidate. Reference gradients are logical (unscaled)
BF16/FP32 tensors; v510's represented-x4 BF16 outputs are decoded by 0.25
before comparison.

The gate is read-only apart from an explicitly requested JSON receipt. It
does not update models, optimizers, checkpoints, jobs, or extension artifacts.
No GPU work happens at import time or for ``--help``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import pathlib
import re
import stat
import sys
from typing import Any

import torch


THIS_FILE = pathlib.Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
MODULE_NAME = (
    "_C_sm100_gqa_tk_v510_d128_e4m3_score_qkv_e5m2_dout_b1_s4096"
)
CAPTURE_FIELDS = (
    "q_fp8",
    "k_fp8",
    "v_fp8",
    "dout_e5m2",
    "lstat",
    "dstat",
    "dq_reference",
    "dk_reference",
    "dv_reference",
)


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_argument(value: str) -> str:
    normalized = value.lower()
    if re.fullmatch(r"[0-9a-f]{64}", normalized) is None:
        raise argparse.ArgumentTypeError(
            "expected exactly 64 hexadecimal digits"
        )
    return normalized


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return result


def _unit_interval(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise argparse.ArgumentTypeError("expected a finite value in [0,1]")
    return result


def _nonnegative_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise argparse.ArgumentTypeError("expected a finite nonnegative value")
    return result


def _stable_regular_file_identity(path: pathlib.Path) -> dict[str, int | str]:
    requested = path.expanduser()
    if not requested.is_absolute():
        raise RuntimeError(f"artifact path must be absolute: {requested}")
    before = requested.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError(
            f"artifact must be a regular non-symlink file: {requested}"
        )
    resolved = requested.resolve(strict=True)
    digest = _sha256(resolved)
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


def _authenticate_file(
    path: pathlib.Path,
    expected_sha256: str,
    expected_bytes: int,
    *,
    label: str,
) -> dict[str, int | str]:
    identity = _stable_regular_file_identity(path)
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


def _load_extension(identity: dict[str, int | str]) -> Any:
    path = pathlib.Path(str(identity["path"]))
    if _stable_regular_file_identity(path) != identity:
        raise RuntimeError(f"extension changed before import: {path}")
    spec = importlib.util.spec_from_file_location(MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot construct import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(MODULE_NAME, None)
        raise
    if _stable_regular_file_identity(path) != identity:
        sys.modules.pop(MODULE_NAME, None)
        raise RuntimeError(f"extension changed while importing: {path}")
    module._tk_fa4_loaded_artifact_identity = dict(identity)
    return module


def _load_capture(identity: dict[str, int | str]) -> dict[str, torch.Tensor]:
    path = pathlib.Path(str(identity["path"]))
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(payload, dict):
        raise RuntimeError("capture must contain a tensor dictionary")
    missing = set(CAPTURE_FIELDS) - set(payload)
    if missing:
        raise RuntimeError(f"capture is missing fields: {sorted(missing)}")
    if any(not isinstance(payload[field], torch.Tensor) for field in CAPTURE_FIELDS):
        raise RuntimeError("every required capture field must be a tensor")
    if _stable_regular_file_identity(path) != identity:
        raise RuntimeError(f"capture changed while loading: {path}")
    return {field: payload[field] for field in CAPTURE_FIELDS}


def _metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    actual64 = actual.double().reshape(-1)
    reference64 = reference.double().reshape(-1)
    difference = actual64 - reference64
    denominator = actual64.norm() * reference64.norm()
    cosine = (
        float(torch.dot(actual64, reference64) / denominator)
        if float(denominator) > 0.0
        else float(actual64.equal(reference64))
    )
    return {
        "max_abs": float(difference.abs().max()),
        "mean_abs": float(difference.abs().mean()),
        "cosine": cosine,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=pathlib.Path, required=True)
    parser.add_argument(
        "--extension-sha256", type=_sha256_argument, required=True
    )
    parser.add_argument("--extension-bytes", type=_positive_int, required=True)
    parser.add_argument("--capture", type=pathlib.Path, required=True)
    parser.add_argument("--capture-sha256", type=_sha256_argument, required=True)
    parser.add_argument("--capture-bytes", type=_positive_int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--max-abs-tolerance", type=_nonnegative_float, required=True
    )
    parser.add_argument("--min-cosine", type=_unit_interval, required=True)
    parser.add_argument("--json-out", type=pathlib.Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    sys.path.insert(0, str(REPO_ROOT))
    from tk_fa4.lowp_fa4_bwd.native_tk_d128_dense_score_e5m2_dout_backward import (
        BATCH,
        NativeTkD128DenseE4M3ScoreQKVE5M2DoutBackward,
        _require_extension_metadata,
    )

    extension_identity = _authenticate_file(
        args.extension,
        args.extension_sha256,
        args.extension_bytes,
        label="v510 extension",
    )
    capture_identity = _authenticate_file(
        args.capture,
        args.capture_sha256,
        args.capture_bytes,
        label="natural capture",
    )
    extension = _load_extension(extension_identity)
    extension_metadata = _require_extension_metadata(extension)
    capture = _load_capture(capture_identity)

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("v510 validation requires an available CUDA device")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.get_device_capability(device)[0] != 10:
        raise RuntimeError("v510 validation requires an SM100 device")

    runner = NativeTkD128DenseE4M3ScoreQKVE5M2DoutBackward(
        extension, batch=BATCH, device=device
    )
    q = capture["q_fp8"].to(device=device)
    k = capture["k_fp8"].to(device=device)
    v = capture["v_fp8"].to(device=device)
    dout = capture["dout_e5m2"].to(device=device)
    runner.bind_inputs(q, k, v, dout)
    runner.lstat.copy_(capture["lstat"].to(device=device, dtype=torch.float32))
    runner.dstat.copy_(capture["dstat"].to(device=device, dtype=torch.float32))
    runner.run(reset=True)
    torch.cuda.synchronize(device)

    results: dict[str, dict[str, float]] = {}
    for name, actual, reference_name in (
        ("dq", runner.dq, "dq_reference"),
        ("dk", runner.dk, "dk_reference"),
        ("dv", runner.dv, "dv_reference"),
    ):
        decoded = actual.float().mul(0.25).cpu()
        reference = capture[reference_name].float()
        if tuple(decoded.shape) != tuple(reference.shape):
            raise RuntimeError(
                f"{name} reference shape mismatch: "
                f"{tuple(decoded.shape)} versus {tuple(reference.shape)}"
            )
        if not torch.isfinite(decoded).all():
            raise RuntimeError(f"v510 emitted nonfinite {name}")
        metric = _metrics(decoded, reference)
        if (
            metric["max_abs"] > args.max_abs_tolerance
            or metric["cosine"] < args.min_cosine
        ):
            raise RuntimeError(f"v510 {name} tolerance failure: {metric}")
        results[name] = metric

    zero_dout = torch.zeros_like(dout)
    runner.bind_inputs(q, k, v, zero_dout)
    runner.dstat.zero_()
    runner.run(reset=True)
    torch.cuda.synchronize(device)
    zero_gate = {
        name: bool(torch.count_nonzero(tensor).item() == 0)
        for name, tensor in (("dq", runner.dq), ("dk", runner.dk), ("dv", runner.dv))
    }
    if not all(zero_gate.values()):
        raise RuntimeError(f"v510 exact-zero-dO gate failed: {zero_gate}")

    receipt: dict[str, object] = {
        "schema": "tkfa4.validate_v510_dense_score_e5m2_dout.v1",
        "validator": str(THIS_FILE),
        "extension": extension_identity,
        "capture": capture_identity,
        "extension_metadata": extension_metadata,
        "thresholds": {
            "max_abs_tolerance": args.max_abs_tolerance,
            "min_cosine": args.min_cosine,
        },
        "metrics": results,
        "exact_zero_dout": zero_gate,
        "passed": True,
    }
    rendered = json.dumps(receipt, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out is not None:
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
