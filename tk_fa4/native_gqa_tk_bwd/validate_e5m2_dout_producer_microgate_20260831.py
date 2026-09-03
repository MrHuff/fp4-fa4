#!/usr/bin/env python3
"""Validate fused BF16 dO x4 -> E5M2 payload plus represented dstat."""

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
from typing import Any

import torch


THIS_DIR = pathlib.Path(__file__).resolve().parent
SOURCE_NAMES = (
    "e5m2_dout_producer_microgate_20260831.cuh",
    "e5m2_dout_producer_microgate_20260831.cu",
    "Makefile.e5m2_dout_producer_microgate_20260831",
    "validate_e5m2_dout_producer_microgate_20260831.py",
)
EXPECTED_CAPTURE_SHA256 = (
    "62eb53246a351955983fe166034c126bf0f71c5afde84e04cb5765d7f35e41e8"
)
EXPECTED_CAPTURE_KEYS = (
    "dout_bf16_scaled",
    "dout_fp8",
    "forward_attention_output",
)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: pathlib.Path) -> Any:
    name = "_C_sm100_e5m2_dout_producer_microgate_20260831"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def patterned_bf16(values: list[float], rows: int, device: torch.device) -> torch.Tensor:
    base = torch.tensor(values, dtype=torch.float32)
    repeats = math.ceil((rows * 128) / base.numel())
    return base.repeat(repeats)[: rows * 128].reshape(rows, 128).to(
        device=device,
        dtype=torch.bfloat16,
    )


def patterned_bf16_bits(
    values: list[int], rows: int, device: torch.device
) -> torch.Tensor:
    base = torch.tensor(values, dtype=torch.uint16).view(torch.bfloat16)
    repeats = math.ceil((rows * 128) / base.numel())
    return base.repeat(repeats)[: rows * 128].reshape(rows, 128).to(device)


def references(
    dout_bf16: torch.Tensor,
    output_bf16: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    payload = (dout_bf16.float() * 4.0).to(torch.float8_e5m2)
    logical = -16.0 * (
        output_bf16.float() * (payload.float() * 0.25)
    ).sum(dim=-1)
    physical = -4.0 * (
        output_bf16.float() * payload.float()
    ).sum(dim=-1)
    return payload, logical, physical


def dstat_metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    delta = actual.float() - reference.float()
    reference_norm = torch.linalg.vector_norm(reference.float()).item()
    delta_norm = torch.linalg.vector_norm(delta).item()
    return {
        "finite": bool(torch.isfinite(actual).all().item()),
        "max_abs_error": float(delta.abs().max().item()),
        "mean_abs_error": float(delta.abs().mean().item()),
        "relative_l2_error": float(delta_norm / max(reference_norm, 1.0e-30)),
        "actual_abs_max": float(actual.abs().max().item()),
        "reference_abs_max": float(reference.abs().max().item()),
    }


def benchmark(
    module: Any,
    arguments: tuple[torch.Tensor, ...],
    warmup: int,
    repetitions: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        module.produce_out(*arguments)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
    stops = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
    for start, stop in zip(starts, stops):
        start.record()
        module.produce_out(*arguments)
        stop.record()
    torch.cuda.synchronize()
    samples = sorted(
        start.elapsed_time(stop) * 1000.0
        for start, stop in zip(starts, stops)
    )
    return {
        "p10_us": float(samples[int(0.10 * (len(samples) - 1))]),
        "p50_us": float(statistics.median(samples)),
        "p90_us": float(samples[int(0.90 * (len(samples) - 1))]),
        "mean_us": float(statistics.fmean(samples)),
        "samples": len(samples),
    }


def run_case(
    module: Any,
    name: str,
    dout_bf16: torch.Tensor,
    output_bf16: torch.Tensor,
    *,
    warmup: int,
    repetitions: int,
    benchmark_case: bool,
) -> dict[str, Any]:
    if dout_bf16.dtype != torch.bfloat16 or output_bf16.dtype != torch.bfloat16:
        raise TypeError(f"{name}: inputs must be BF16")
    if dout_bf16.shape != output_bf16.shape or dout_bf16.shape[-1] != 128:
        raise ValueError(f"{name}: expected matched [rows,128]")
    dout_bf16 = dout_bf16.contiguous()
    output_bf16 = output_bf16.contiguous()
    rows = dout_bf16.shape[0]
    payload = torch.empty_like(dout_bf16, dtype=torch.float8_e5m2)
    dstat = torch.empty((rows,), device=dout_bf16.device, dtype=torch.float32)
    reference_payload, logical_reference, physical_reference = references(
        dout_bf16,
        output_bf16,
    )

    payload.view(torch.uint8).fill_(0xA5)
    dstat.fill_(math.nan)
    module.produce_out(dout_bf16, output_bf16, payload, dstat)
    torch.cuda.synchronize()
    first_payload = payload.view(torch.uint8).clone()
    first_dstat = dstat.clone()

    payload.view(torch.uint8).fill_(0x5A)
    dstat.fill_(-1234567.0)
    module.produce_out(dout_bf16, output_bf16, payload, dstat)
    torch.cuda.synchronize()

    expected_bytes = reference_payload.view(torch.uint8)
    byte_mismatch = first_payload != expected_bytes
    logical_physical_equal = torch.equal(logical_reference, physical_reference)
    metrics = dstat_metrics(first_dstat, logical_reference)
    dstat_tolerance = max(
        1.0e-6,
        metrics["reference_abs_max"] * 1.0e-5,
    )
    case_pass = (
        int(byte_mismatch.sum().item()) == 0
        and torch.equal(first_payload, payload.view(torch.uint8))
        and torch.equal(first_dstat, dstat)
        and logical_physical_equal
        and metrics["finite"]
        and metrics["relative_l2_error"] <= 3.0e-6
        and metrics["max_abs_error"] <= dstat_tolerance
    )
    result: dict[str, Any] = {
        "passed": bool(case_pass),
        "shape": list(dout_bf16.shape),
        "payload_byte_mismatches": int(byte_mismatch.sum().item()),
        "payload_bytes": expected_bytes.numel(),
        "payload_deterministic": bool(
            torch.equal(first_payload, payload.view(torch.uint8))
        ),
        "dstat_deterministic": bool(torch.equal(first_dstat, dstat)),
        "logical_physical_reference_bitwise_equal": bool(
            logical_physical_equal
        ),
        "dstat_tolerance": float(dstat_tolerance),
        "dstat": metrics,
        "payload": {
            "decoded_abs_max": float(reference_payload.float().abs().max().item()),
            "nonzero": int((expected_bytes & 0x7F != 0).sum().item()),
            "positive_zero": int((expected_bytes == 0x00).sum().item()),
            "negative_zero": int((expected_bytes == 0x80).sum().item()),
            "infinite": int(torch.isinf(reference_payload.float()).sum().item()),
            "nan": int(torch.isnan(reference_payload.float()).sum().item()),
        },
    }
    if benchmark_case:
        result["performance"] = benchmark(
            module,
            (dout_bf16, output_bf16, payload, dstat),
            warmup,
            repetitions,
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", type=pathlib.Path, required=True)
    parser.add_argument("--capture", type=pathlib.Path, required=True)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repetitions", type=int, default=100)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("fail-closed: CUDA_VISIBLE_DEVICES must be exactly 0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("fail-closed: exactly one visible CUDA GPU required")
    capability = torch.cuda.get_device_capability(0)
    if capability[0] < 10:
        raise RuntimeError(f"fail-closed: SM100 required, got {capability}")
    if not args.module.is_file() or not args.capture.is_file():
        raise FileNotFoundError("module and authenticated capture must exist")

    capture_hash_before = sha256(args.capture)
    if capture_hash_before != EXPECTED_CAPTURE_SHA256:
        raise RuntimeError(
            "fail-closed: capture SHA256 mismatch: " + capture_hash_before
        )

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    module = load_module(args.module.resolve())
    metadata = dict(module.metadata())
    torch.manual_seed(20260831)

    rows = 64
    random_output = (
        torch.randn((rows, 128), device=device, dtype=torch.float32) * 1.5
    ).to(torch.bfloat16)
    cases: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
        "zero": (
            torch.zeros((rows, 128), device=device, dtype=torch.bfloat16),
            random_output,
        ),
        "bf16_subnormal": (
            patterned_bf16_bits(
                [
                    0x0001, 0x0002, 0x0003, 0x003F, 0x0040, 0x007F,
                    0x8001, 0x8002, 0x803F, 0x8040, 0x807F,
                ],
                rows,
                device,
            ),
            random_output,
        ),
        "e5m2_subnormal_boundaries": (
            patterned_bf16(
                [
                    2.0**-20, 2.0**-19, 2.0**-18, 2.0**-17,
                    3.0 * 2.0**-18, -2.0**-20, -2.0**-19,
                    -2.0**-18, -2.0**-17, -3.0 * 2.0**-18,
                ],
                rows,
                device,
            ),
            random_output,
        ),
        "large_finite": (
            patterned_bf16(
                [
                    1.0, -1.0, 3584.0, -3584.0, 7168.0, -7168.0,
                    10000.0, -10000.0, 14000.0, -14000.0,
                    14336.0, -14336.0, 14976.0, -14976.0,
                ],
                rows,
                device,
            ),
            patterned_bf16(
                [0.0, 2.0**-8, -2.0**-8, 0.25, -0.25, 1.0, -1.0],
                rows,
                device,
            ),
        ),
    }
    exponents = torch.randint(-18, 13, (rows, 128), device=device)
    mantissas = torch.empty((rows, 128), device=device).uniform_(0.5, 1.0)
    signs = torch.where(
        torch.rand((rows, 128), device=device) < 0.5,
        -torch.ones((), device=device),
        torch.ones((), device=device),
    )
    random_dout = torch.ldexp(mantissas * signs, exponents).to(torch.bfloat16)
    cases["random_log_uniform"] = (random_dout, random_output)

    results: dict[str, Any] = {}
    for name, (dout, output) in cases.items():
        results[name] = run_case(
            module,
            name,
            dout,
            output,
            warmup=args.warmup,
            repetitions=args.repetitions,
            benchmark_case=name == "random_log_uniform",
        )

    capture = torch.load(
        args.capture,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if not isinstance(capture, dict) or any(
        key not in capture for key in EXPECTED_CAPTURE_KEYS
    ):
        raise RuntimeError("authenticated capture is missing required tensors")
    captured_dout_cpu = capture["dout_bf16_scaled"]
    captured_output_cpu = capture["forward_attention_output"]
    captured_e4_cpu = capture["dout_fp8"]
    expected_shape = (1, 4096, 32, 128)
    if (
        tuple(captured_dout_cpu.shape) != expected_shape
        or tuple(captured_output_cpu.shape) != expected_shape
        or tuple(captured_e4_cpu.shape) != expected_shape
        or captured_dout_cpu.dtype != torch.bfloat16
        or captured_output_cpu.dtype != torch.bfloat16
        or captured_e4_cpu.dtype != torch.float8_e4m3fn
    ):
        raise RuntimeError("authenticated capture tensor ABI mismatch")

    captured_dout = captured_dout_cpu.reshape(-1, 128).to(device)
    captured_output = captured_output_cpu.reshape(-1, 128).to(device)
    captured_e4 = captured_e4_cpu.reshape(-1, 128).to(device)
    captured_e4_expected = (captured_dout.float() * 4.0).to(
        torch.float8_e4m3fn
    )
    captured_e4_source_exact = torch.equal(
        captured_e4.view(torch.uint8),
        captured_e4_expected.view(torch.uint8),
    )
    results["authenticated_step5000_layer12"] = run_case(
        module,
        "authenticated_step5000_layer12",
        captured_dout,
        captured_output,
        warmup=args.warmup,
        repetitions=args.repetitions,
        benchmark_case=True,
    )
    results["authenticated_step5000_layer12"][
        "captured_e4_confirms_pre_x4_bf16_source"
    ] = bool(captured_e4_source_exact)
    results["authenticated_step5000_layer12"]["passed"] = bool(
        results["authenticated_step5000_layer12"]["passed"]
        and captured_e4_source_exact
    )

    capture_hash_after = sha256(args.capture)
    capture_unchanged = capture_hash_after == capture_hash_before
    passed = capture_unchanged and all(
        bool(result["passed"]) for result in results.values()
    )
    receipt = {
        "schema": "tkfa4.e5m2_dout_producer_microgate.receipt.v1",
        "passed": bool(passed),
        "device": {
            "visible_device_count": torch.cuda.device_count(),
            "name": torch.cuda.get_device_name(0),
            "capability": list(capability),
        },
        "module": {
            "path": str(args.module.resolve()),
            "bytes": args.module.stat().st_size,
            "sha256": sha256(args.module),
        },
        "sources": {
            name: sha256(THIS_DIR / name) for name in SOURCE_NAMES
        },
        "capture": {
            "path": str(args.capture.resolve()),
            "bytes": args.capture.stat().st_size,
            "expected_sha256": EXPECTED_CAPTURE_SHA256,
            "sha256_before": capture_hash_before,
            "sha256_after": capture_hash_after,
            "unchanged": bool(capture_unchanged),
            "checkpoint_step": 5000,
            "layer": 12,
            "loaded_read_only_mmap": True,
        },
        "metadata": metadata,
        "gates": {
            "payload_byte_mismatches": 0,
            "payload_deterministic": True,
            "dstat_deterministic": True,
            "logical_physical_reference_bitwise_equal": True,
            "dstat_relative_l2_max": 3.0e-6,
            "dstat_max_abs_dynamic": "max(1e-6,reference_abs_max*1e-5)",
            "capture_sha_before_equals_after": True,
        },
        "cases": results,
        "performance_scope": (
            "standalone fused BF16-x4 quantize, genuine E5M2 byte store, "
            "decode-from-published-register payload, and per-row dstat; "
            "not integrated projection performance"
        ),
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())
