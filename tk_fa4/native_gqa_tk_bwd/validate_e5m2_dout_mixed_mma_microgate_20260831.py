#!/usr/bin/env python3
"""Validate the standalone SM100 E4M3 x E5M2 mixed-MMA microgate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import statistics
import sys
from typing import Any, Callable

import torch


THIS_DIR = pathlib.Path(__file__).resolve().parent
SOURCE_NAMES = (
    "e5m2_dout_mixed_mma_microgate_20260831.cuh",
    "e5m2_dout_mixed_mma_microgate_20260831.cu",
    "Makefile.e5m2_dout_mixed_mma_microgate_20260831",
    "validate_e5m2_dout_mixed_mma_microgate_20260831.py",
)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: pathlib.Path) -> Any:
    name = "_C_sm100_e5m2_dout_mixed_mma_microgate_20260831"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def decoded_reference(
    operation: str,
    a0: torch.Tensor,
    b0: torch.Tensor,
    a1: torch.Tensor | None = None,
    b1: torch.Tensor | None = None,
) -> torch.Tensor:
    # CPU FP32 prevents TF32 or another low-precision GPU path from entering
    # the oracle.  Inputs are decoded from their actual FP8 payloads first.
    left0 = a0.float().cpu()
    right0 = b0.float().cpu()
    result = left0 @ (right0.T if operation == "dp" else right0)
    if a1 is not None and b1 is not None:
        left1 = a1.float().cpu()
        right1 = b1.float().cpu()
        result = result + left1 @ (
            right1.T if operation == "dp" else right1
        )
    return result


def error_metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    actual_cpu = actual.float().cpu()
    delta = actual_cpu - reference
    reference_norm = torch.linalg.vector_norm(reference).item()
    delta_norm = torch.linalg.vector_norm(delta).item()
    cosine = torch.nn.functional.cosine_similarity(
        actual_cpu.reshape(1, -1),
        reference.reshape(1, -1),
    ).item()
    return {
        "finite": bool(torch.isfinite(actual_cpu).all().item()),
        "sentinel_fully_overwritten": bool(
            torch.isfinite(actual_cpu).all().item()
        ),
        "max_abs_error": float(delta.abs().max().item()),
        "mean_abs_error": float(delta.abs().mean().item()),
        "relative_l2_error": float(delta_norm / max(reference_norm, 1.0e-30)),
        "cosine_similarity": float(cosine),
        "actual_abs_max": float(actual_cpu.abs().max().item()),
        "reference_abs_max": float(reference.abs().max().item()),
    }


def benchmark(
    function: Callable[..., None],
    arguments: tuple[torch.Tensor, ...],
    warmup: int,
    repetitions: int,
) -> dict[str, float]:
    for _ in range(warmup):
        function(*arguments)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
    stops = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
    for start, stop in zip(starts, stops):
        start.record()
        function(*arguments)
        stop.record()
    torch.cuda.synchronize()
    samples_us = [start.elapsed_time(stop) * 1000.0 for start, stop in zip(starts, stops)]
    ordered = sorted(samples_us)
    return {
        "p10_us": float(ordered[int(0.10 * (len(ordered) - 1))]),
        "p50_us": float(statistics.median(ordered)),
        "p90_us": float(ordered[int(0.90 * (len(ordered) - 1))]),
        "mean_us": float(statistics.fmean(ordered)),
        "samples": float(len(ordered)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", type=pathlib.Path, required=True)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--repetitions", type=int, default=120)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError(
            "fail-closed: run this microgate with CUDA_VISIBLE_DEVICES=0"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("fail-closed: exactly one visible CUDA device is required")
    capability = torch.cuda.get_device_capability(0)
    if capability[0] < 10:
        raise RuntimeError(f"fail-closed: SM100-class GPU required, got {capability}")
    if not args.module.is_file():
        raise FileNotFoundError(args.module)

    torch.cuda.set_device(0)
    module = load_module(args.module.resolve())
    metadata = dict(module.metadata())

    torch.manual_seed(20260831)
    device = torch.device("cuda:0")

    def make_a(scale: float) -> torch.Tensor:
        values = torch.randn((128, 128), device=device, dtype=torch.float32)
        values = (values * scale).clamp(-96.0, 96.0)
        return values.to(torch.float8_e4m3fn)

    def make_b(scale: float) -> torch.Tensor:
        values = torch.randn((128, 128), device=device, dtype=torch.float32)
        values = values * scale
        # Exercise E5M2 encodings outside E4M3 range.  If bit 10 is absent,
        # this payload becomes grossly wrong or nonfinite instead of silently
        # passing a narrow-range test.
        values[:, ::17] *= 96.0
        values[::19, :] *= 8.0
        values = values.clamp(-8192.0, 8192.0)
        return values.to(torch.float8_e5m2)

    a0 = make_a(0.75)
    b0 = make_b(5.0)
    a1 = make_a(0.40)
    b1 = make_b(2.5)
    output = torch.empty((128, 128), device=device, dtype=torch.float32)

    cases: dict[str, tuple[Callable[..., None], tuple[torch.Tensor, ...], torch.Tensor]] = {
        "dp_overwrite": (
            module.dp_overwrite_out,
            (a0, b0, output),
            decoded_reference("dp", a0, b0),
        ),
        "dp_accumulate": (
            module.dp_accumulate_out,
            (a0, b0, a1, b1, output),
            decoded_reference("dp", a0, b0, a1, b1),
        ),
        "dv_overwrite": (
            module.dv_overwrite_out,
            (a0, b0, output),
            decoded_reference("dv", a0, b0),
        ),
        "dv_accumulate": (
            module.dv_accumulate_out,
            (a0, b0, a1, b1, output),
            decoded_reference("dv", a0, b0, a1, b1),
        ),
    }

    results: dict[str, Any] = {}
    passed = True
    for name, (function, arguments, reference) in cases.items():
        output.fill_(math.nan)
        function(*arguments)
        torch.cuda.synchronize()
        first = output.clone()
        metrics = error_metrics(first, reference)

        output.fill_(-1234567.0)
        function(*arguments)
        torch.cuda.synchronize()
        deterministic = bool(torch.equal(first, output))
        metrics["deterministic_repeat"] = deterministic
        metrics["performance"] = benchmark(
            function,
            arguments,
            args.warmup,
            args.repetitions,
        )

        case_pass = (
            metrics["finite"]
            and metrics["sentinel_fully_overwritten"]
            and deterministic
            and metrics["relative_l2_error"] <= 5.0e-5
            and metrics["cosine_similarity"] >= 0.999999
        )
        metrics["passed"] = bool(case_pass)
        passed = passed and bool(case_pass)
        results[name] = metrics

    source_hashes = {
        name: sha256(THIS_DIR / name)
        for name in SOURCE_NAMES
    }
    receipt = {
        "schema": "tkfa4.e5m2_dout_mixed_mma_microgate.receipt.v1",
        "passed": bool(passed),
        "module": {
            "path": str(args.module.resolve()),
            "bytes": args.module.stat().st_size,
            "sha256": sha256(args.module),
        },
        "sources": source_hashes,
        "device": {
            "visible_device_count": torch.cuda.device_count(),
            "name": torch.cuda.get_device_name(0),
            "capability": list(capability),
        },
        "metadata": metadata,
        "input_ranges": {
            "a0_decoded_abs_max": float(a0.float().abs().max().item()),
            "b0_decoded_abs_max": float(b0.float().abs().max().item()),
            "a1_decoded_abs_max": float(a1.float().abs().max().item()),
            "b1_decoded_abs_max": float(b1.float().abs().max().item()),
            "b0_values_outside_e4m3_range": int(
                (b0.float().abs() > 448.0).sum().item()
            ),
        },
        "gates": {
            "relative_l2_max": 5.0e-5,
            "cosine_min": 0.999999,
            "finite": True,
            "deterministic_repeat": True,
            "output_sentinel_overwritten": True,
        },
        "cases": results,
        "performance_scope": (
            "standalone one-CTA kernel including two TMA input loads, "
            "FP32 TMEM drain, and TMA output store; not a production claim"
        ),
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())
