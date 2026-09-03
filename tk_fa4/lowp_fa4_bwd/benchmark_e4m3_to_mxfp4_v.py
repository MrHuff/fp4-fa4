#!/usr/bin/env python3
"""Gate split E4M3(x4)->MXFP4 V publication against its measured premium.

The matching B16/S4096 Llama-1.2B factorial measured native-NVFP4 QKV
projection publication at 848.927 us for exact FP8-PV and 886.553 us for
direct MXFP4-PV.  A converter fed by the exact route's already-published
feature-major E4M3 V is useful only if it fits inside that 37.626 us premium.

This harness times the caller-owned converter at the identical V shape,
validates its byte layout and E8M0 x4 compensation on a small tile, and
reports ``exact_projection + converter`` versus direct MX publication.  It
does not claim to measure attention or backward.  ``--dry-run`` neither
imports Torch nor touches CUDA.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_FACTORIAL = (
    REPO_ROOT / "results" / "llama12b_b16_forward_factorial_20260826.json"
)
DEFAULT_BATCH = 16
DEFAULT_HEADS = 8
DEFAULT_DEPTH = 64
DEFAULT_SEQUENCE = 4096
CONVERTER_SYMBOL = "convert_e4m3_x4_v_bhds_to_causal_mxfp4_out"
FACTORIAL_SCHEMA = "b16_s4096_qkv_projection_pv_forward_factorial_summary_v2"
FACTORIAL_SHAPE = {
    "batch": 16,
    "sequence": 4096,
    "hidden": 2048,
    "q_heads": 32,
    "kv_heads": 8,
    "head_dim": 64,
    "causal": True,
}
PROCESS_CHECK_INTERVAL = 25


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=_positive_int, default=DEFAULT_BATCH)
    parser.add_argument("--heads", type=_positive_int, default=DEFAULT_HEADS)
    parser.add_argument(
        "--sequence",
        type=_positive_int,
        default=DEFAULT_SEQUENCE,
    )
    parser.add_argument("--warmup", type=_positive_int, default=20)
    parser.add_argument("--iterations", type=_positive_int, default=200)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument(
        "--factorial-result",
        type=Path,
        default=DEFAULT_FACTORIAL,
        help="Matched factorial JSON providing exact/direct projection times.",
    )
    parser.add_argument(
        "--extension",
        type=Path,
        help="Optional explicit _C_b300_lowp_bwd extension path.",
    )
    parser.add_argument(
        "--module",
        default="tk_fa4._C_b300_lowp_bwd",
        help="Import name used when --extension is omitted.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--require-neutral",
        action="store_true",
        help="Exit nonzero when converter median exceeds the direct premium.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.sequence % 128:
        parser.error("--sequence must be divisible by 128")
    return args


def _invocation_shape(args: argparse.Namespace) -> dict[str, int]:
    return {
        "batch": args.batch,
        "heads": args.heads,
        "depth": DEFAULT_DEPTH,
        "sequence": args.sequence,
    }


def _load_projection_baseline(
    path: Path,
    invocation_shape: dict[str, int],
) -> dict[str, Any]:
    document = json.loads(path.read_text())
    if not isinstance(document, dict):
        raise ValueError("factorial result must be a JSON object")
    if document.get("schema") != FACTORIAL_SCHEMA:
        raise ValueError(
            "factorial schema mismatch: expected "
            f"{FACTORIAL_SCHEMA!r}, got {document.get('schema')!r}"
        )
    factorial_shape = document.get("shape")
    shape_types_match = isinstance(factorial_shape, dict) and all(
        key in factorial_shape
        and type(factorial_shape[key]) is type(expected_value)
        for key, expected_value in FACTORIAL_SHAPE.items()
    )
    if not shape_types_match or factorial_shape != FACTORIAL_SHAPE:
        raise ValueError(
            "factorial shape mismatch: expected exact matched shape "
            f"{FACTORIAL_SHAPE}, got {factorial_shape!r}"
        )
    matched_invocation_shape = {
        "batch": factorial_shape["batch"],
        "heads": factorial_shape["kv_heads"],
        "depth": factorial_shape["head_dim"],
        "sequence": factorial_shape["sequence"],
    }
    if invocation_shape != matched_invocation_shape:
        raise ValueError(
            "converter invocation does not match the factorial baseline: "
            f"expected {matched_invocation_shape}, got {invocation_shape}"
        )
    try:
        timings = document["two_process_mean_device_us"]
        exact = float(
            timings["nvfp4_qkv__fp8_pv"]["projection_publication"]
        )
        direct = float(
            timings["nvfp4_qkv__mx_pv"]["projection_publication"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "factorial result lacks the required native-NVFP4 projection "
            "timings"
        ) from error
    premium = direct - exact
    if not all(math.isfinite(value) and value > 0 for value in (exact, direct)):
        raise ValueError("factorial projection timings must be finite and positive")
    if premium <= 0:
        raise ValueError("direct MX publication must have a positive premium")
    return {
        "exact_fp8_projection_us": exact,
        "direct_mx_projection_us": direct,
        "direct_mx_premium_us": premium,
        "factorial_schema": document["schema"],
        "factorial_shape": dict(factorial_shape),
    }


def _load_extension(path: Path | None, module_name: str) -> Any:
    if path is None:
        return importlib.import_module(module_name)
    selected = path.expanduser().resolve()
    if not selected.is_file():
        raise FileNotFoundError(f"extension not found: {selected}")
    dynamic_name = selected.name.split(".", 1)[0]
    spec = importlib.util.spec_from_file_location(dynamic_name, selected)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load extension spec from {selected}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"artifact must be a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return {
        "path": str(resolved),
        "sha256": digest.hexdigest(),
        "bytes": resolved.stat().st_size,
    }


def _authenticate_loaded_interface(
    expected_identity: dict[str, Any],
) -> dict[str, Any]:
    module = sys.modules.get("tk_fa4.interface")
    if module is None:
        return {
            "imported": False,
            "expected": expected_identity,
        }
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str) or not module_file:
        raise RuntimeError("loaded tk_fa4.interface has no source path")
    loaded_identity = _file_identity(Path(module_file))
    if loaded_identity != expected_identity:
        raise RuntimeError(
            "loaded tk_fa4.interface does not match the expected worktree "
            f"source: expected {expected_identity}, got {loaded_identity}"
        )
    return {
        "imported": True,
        "expected": expected_identity,
        "loaded": loaded_identity,
    }


def _parse_gpu_process_report(report: Any) -> list[int]:
    if not isinstance(report, str) or not report.strip():
        raise RuntimeError("GPU process inventory returned an empty report")
    lines = report.splitlines()
    if not re.fullmatch(r"GPU:\d+", lines[0]):
        raise RuntimeError(f"malformed GPU process inventory header: {report!r}")
    process_lines = lines[1:]
    if process_lines == ["no processes are running"]:
        return []
    if not process_lines:
        raise RuntimeError(f"incomplete GPU process inventory: {report!r}")
    pattern = re.compile(
        r"process\s+(\d+)\s+uses\s+"
        r"(?:\d+(?:\.\d+)?|\.\d+)\s+MB GPU memory"
    )
    process_ids = []
    for line in process_lines:
        match = pattern.fullmatch(line)
        if match is None:
            raise RuntimeError(f"malformed GPU process inventory: {report!r}")
        process_ids.append(int(match.group(1)))
    if len(process_ids) != len(set(process_ids)):
        raise RuntimeError(f"duplicate GPU process IDs in inventory: {report!r}")
    return process_ids


def _require_exclusive_visible_gpu(torch: Any) -> dict[str, Any]:
    """Reject timing on a visible GPU used by another compute process."""
    try:
        report = torch.cuda.list_gpu_processes(0)
    except Exception as error:
        raise RuntimeError("GPU process inventory is unavailable") from error
    process_ids = _parse_gpu_process_report(report)
    own_pid = os.getpid()
    foreign = [pid for pid in process_ids if pid != own_pid]
    if foreign:
        raise RuntimeError(
            "converter timing requires an exclusive visible GPU; "
            f"foreign compute PIDs: {foreign}"
        )
    return {
        "report": report,
        "observed_process_ids": process_ids,
        "own_pid": own_pid,
        "foreign_process_ids": foreign,
    }


def _physical_e8m0(physical_amax: float, torch: Any) -> int:
    bits = int(
        torch.tensor([physical_amax], dtype=torch.bfloat16)
        .view(torch.uint16)
        .item()
    )
    exponent = (bits >> 7) & 0xFF
    if exponent == 0:
        return 0
    return exponent + int((bits & 0x7F) >= 0x1A and exponent < 0xFE)


def _e2m1_code(value: float) -> int:
    levels = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    magnitude = abs(value)
    distances = [abs(magnitude - level) for level in levels]
    minimum = min(distances)
    candidates = [
        code
        for code, distance in enumerate(distances)
        if math.isclose(distance, minimum, rel_tol=0.0, abs_tol=1.0e-12)
    ]
    # PTX cvt.rn uses ties-to-even; the nibble value is the significand code.
    code = min(candidates, key=lambda candidate: (candidate & 1, candidate))
    if math.copysign(1.0, value) < 0.0:
        code |= 0x8
    return code


def _nan_correctness_preflight(torch: Any, extension: Any) -> dict[str, Any]:
    """Authenticate the documented sentinel-and-zero nonfinite policy."""
    logical = torch.full(
        (1, 1, DEFAULT_DEPTH, 128),
        1.0,
        device="cuda",
        dtype=torch.bfloat16,
    )
    affected_groups = ((7, 2), (41, 3))
    logical[0, 0, affected_groups[0][0], affected_groups[0][1]] = float(
        "nan"
    )
    logical[0, 0, affected_groups[1][0], affected_groups[1][1]] = -float(
        "nan"
    )
    source = (logical.float() * 4.0).to(torch.float8_e4m3fn)
    payload = torch.empty(
        (1, 1, DEFAULT_DEPTH, 64),
        device="cuda",
        dtype=torch.float4_e2m1fn_x2,
    )
    scales = torch.empty(
        (1, 1, 1, 512),
        device="cuda",
        dtype=torch.float8_e4m3fn,
    )
    getattr(extension, CONVERTER_SYMBOL)(source, payload, scales)
    torch.cuda.synchronize()
    payload_bytes = payload.view(torch.uint8).cpu().reshape(DEFAULT_DEPTH, 64)
    scale_bytes = scales.view(torch.uint8).cpu().reshape(512)

    affected_offsets = {
        (depth & 31) * 16 + (depth >> 5) * 4 + quarter
        for depth, quarter in affected_groups
    }
    valid_offsets = {
        (depth & 31) * 16 + (depth >> 5) * 4 + quarter
        for depth in range(DEFAULT_DEPTH)
        for quarter in range(4)
    }
    sentinel_offsets = {
        offset for offset in valid_offsets if int(scale_bytes[offset]) == 0xFF
    }
    zero_payload_groups = {
        (depth, quarter)
        for depth, quarter in affected_groups
        if int(
            payload_bytes[depth, quarter * 16 : (quarter + 1) * 16]
            .count_nonzero()
            .item()
        )
        == 0
    }
    neighboring_group_nonzero = int(
        payload_bytes[affected_groups[0][0], 0:16].count_nonzero().item()
    ) > 0
    passed = (
        sentinel_offsets == affected_offsets
        and zero_payload_groups == set(affected_groups)
        and neighboring_group_nonzero
    )
    return {
        "affected_groups": [list(group) for group in affected_groups],
        "expected_sentinel_offsets": sorted(affected_offsets),
        "observed_sentinel_offsets": sorted(sentinel_offsets),
        "zero_payload_groups": [
            list(group) for group in sorted(zero_payload_groups)
        ],
        "finite_neighbor_payload_nonzero": neighboring_group_nonzero,
        "passed": passed,
    }


def _correctness_preflight(torch: Any, extension: Any) -> dict[str, Any]:
    element_count = DEFAULT_DEPTH * 128
    logical_bf16 = ((
        (torch.arange(element_count, dtype=torch.float32) * 37 % 257) - 128
    ).reshape(1, 1, DEFAULT_DEPTH, 128) / 32.0).to(torch.bfloat16)
    source = (logical_bf16.float() * 4.0).to(
        device="cuda",
        dtype=torch.float8_e4m3fn,
    )
    payload = torch.empty(
        (1, 1, DEFAULT_DEPTH, 64),
        device="cuda",
        dtype=torch.float4_e2m1fn_x2,
    )
    scales = torch.empty(
        (1, 1, 1, 512),
        device="cuda",
        dtype=torch.float8_e4m3fn,
    )
    getattr(extension, CONVERTER_SYMBOL)(source, payload, scales)
    torch.cuda.synchronize()

    physical = source.float().cpu().reshape(DEFAULT_DEPTH, 128)
    direct_bf16 = logical_bf16.float().reshape(DEFAULT_DEPTH, 128)
    payload_bytes = payload.contiguous().view(torch.uint8).cpu().reshape(
        DEFAULT_DEPTH,
        64,
    )
    scale_bytes = scales.contiguous().view(torch.uint8).cpu().reshape(512)
    payload_mismatches = 0
    scale_mismatches = 0
    source_vs_direct_payload_mismatches = 0
    source_vs_direct_scale_mismatches = 0
    for depth in range(DEFAULT_DEPTH):
        for quarter in range(4):
            values = [
                float(physical[depth, quarter + 4 * lane])
                for lane in range(32)
            ]
            physical_e8m0 = _physical_e8m0(max(abs(v) for v in values), torch)
            logical_e8m0 = 0 if physical_e8m0 == 0 else physical_e8m0 - 2
            scale_offset = (depth & 31) * 16 + (depth >> 5) * 4 + quarter
            scale_mismatches += int(int(scale_bytes[scale_offset]) != logical_e8m0)
            multiplier = (
                0.0
                if physical_e8m0 == 0
                else 6.0 * math.ldexp(1.0, 127 - physical_e8m0)
            )
            direct_values = [
                float(direct_bf16[depth, quarter + 4 * lane])
                for lane in range(32)
            ]
            direct_e8m0 = _physical_e8m0(
                max(abs(v) for v in direct_values),
                torch,
            )
            source_vs_direct_scale_mismatches += int(
                logical_e8m0 != direct_e8m0
            )
            direct_multiplier = (
                0.0
                if direct_e8m0 == 0
                else 6.0 * math.ldexp(1.0, 127 - direct_e8m0)
            )
            for pair in range(16):
                low = _e2m1_code(values[2 * pair] * multiplier)
                high = _e2m1_code(values[2 * pair + 1] * multiplier)
                expected = low | (high << 4)
                actual = int(payload_bytes[depth, quarter * 16 + pair])
                payload_mismatches += int(actual != expected)
                direct_low = _e2m1_code(
                    direct_values[2 * pair] * direct_multiplier
                )
                direct_high = _e2m1_code(
                    direct_values[2 * pair + 1] * direct_multiplier
                )
                source_vs_direct_payload_mismatches += int(
                    expected != (direct_low | (direct_high << 4))
                )
    finite_passed = payload_mismatches == 0 and scale_mismatches == 0
    nan_policy = _nan_correctness_preflight(torch, extension)
    return {
        "payload_bytes_checked": DEFAULT_DEPTH * 64,
        "scale_bytes_checked": DEFAULT_DEPTH * 4,
        "payload_mismatches": payload_mismatches,
        "scale_mismatches": scale_mismatches,
        "source_vs_direct_bf16": {
            "payload_byte_mismatches": source_vs_direct_payload_mismatches,
            "payload_bytes_compared": DEFAULT_DEPTH * 64,
            "scale_mismatches": source_vs_direct_scale_mismatches,
            "scales_compared": DEFAULT_DEPTH * 4,
            "interpretation": (
                "diagnostic only: converter correctness is defined by "
                "MX(E4M3(4V)/4), not byte equality to MX(BF16 V)"
            ),
        },
        "finite_passed": finite_passed,
        "nan_policy": nan_policy,
        "passed": finite_passed and nan_policy["passed"],
    }


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _measure(
    torch: Any,
    function: Any,
    source: Any,
    payload: Any,
    scales: Any,
    warmup: int,
    iterations: int,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    for _ in range(warmup):
        function(source, payload, scales)
    torch.cuda.synchronize()
    process_checks = [
        {
            "phase": "after_warmup",
            **_require_exclusive_visible_gpu(torch),
        }
    ]
    samples = []
    chunk_index = 0
    for offset in range(0, iterations, PROCESS_CHECK_INTERVAL):
        count = min(PROCESS_CHECK_INTERVAL, iterations - offset)
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(count)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(count)]
        for start, end in zip(starts, ends):
            start.record()
            function(source, payload, scales)
            end.record()
        torch.cuda.synchronize()
        samples.extend(
            start.elapsed_time(end) * 1000.0
            for start, end in zip(starts, ends)
        )
        process_checks.append(
            {
                "phase": f"after_timed_chunk_{chunk_index}",
                "first_iteration": offset,
                "iteration_count": count,
                **_require_exclusive_visible_gpu(torch),
            }
        )
        chunk_index += 1
    timing = {
        "median_us": statistics.median(samples),
        "mean_us": statistics.fmean(samples),
        "p10_us": _percentile(samples, 0.10),
        "p90_us": _percentile(samples, 0.90),
        "minimum_us": min(samples),
        "maximum_us": max(samples),
    }
    return timing, process_checks


def _plan(
    args: argparse.Namespace,
    baseline: dict[str, Any],
    baseline_identity: dict[str, Any],
) -> dict[str, Any]:
    elements = args.batch * args.heads * DEFAULT_DEPTH * args.sequence
    scale_writes = (
        args.batch
        * (args.sequence // 128)
        * args.heads
        * DEFAULT_DEPTH
        * 4
    )
    return {
        "shape": {
            "batch": args.batch,
            "heads": args.heads,
            "depth": DEFAULT_DEPTH,
            "sequence": args.sequence,
        },
        "source_contract": "contiguous E4M3(x4) [B,H,64,S]",
        "output_contract": {
            "payload": "packed E2M1 [B,H,64,S/2], quarter-interleaved",
            "scales": "raw E8M0 [B,S/128,H,512]",
        },
        "kernel_bytes": {
            "input_read": elements,
            "payload_write": elements // 2,
            "addressable_scale_write": scale_writes,
            "total": elements + elements // 2 + scale_writes,
        },
        "protocol": {
            "seed": args.seed,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "require_neutral": args.require_neutral,
            "process_check_interval_iterations": PROCESS_CHECK_INTERVAL,
        },
        "baseline": {
            **baseline,
            "identity": baseline_identity,
            "applicability": {
                "applies_to_this_invocation": True,
                "invocation_shape": _invocation_shape(args),
                "matched_factorial_shape": baseline["factorial_shape"],
                "neutrality_gate_enforced": args.require_neutral,
                "interpretation": (
                    "the exact matched factorial premium is the enforced "
                    "pass/fail threshold"
                    if args.require_neutral
                    else "the exact matched factorial premium is applicable "
                    "and reported, but is not an enforced exit-status gate"
                ),
            },
        },
        "neutrality_rule": "converter median <= direct MX projection premium",
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    invocation_shape = _invocation_shape(args)
    baseline = _load_projection_baseline(
        args.factorial_result,
        invocation_shape,
    )
    baseline_identity = _file_identity(args.factorial_result)
    document = _plan(args, baseline, baseline_identity)
    if args.dry_run:
        document["dry_run"] = True
        print(json.dumps(document, indent=2, sort_keys=True))
        return 0
    if args.output is not None and os.path.lexists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")

    source_identities_before = {
        "benchmark_source": _file_identity(Path(__file__)),
        "kernel_source": _file_identity(HERE / "e4m3_to_mxfp4_v.cuh"),
        "wrapper_source": _file_identity(HERE / "lowp_fa4_bwd.cu"),
        "projection_epilogue": _file_identity(
            HERE / "projection_fp4_epilogue.cuh"
        ),
        "interface_source": _file_identity(
            REPO_ROOT / "tk_fa4" / "interface.py"
        ),
        "factorial_result": baseline_identity,
    }

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required unless --dry-run is selected")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("expose exactly one GPU to the converter benchmark")
    capability = list(torch.cuda.get_device_capability())
    if capability != [10, 0]:
        raise RuntimeError("the converter benchmark requires SM100")
    gpu_exclusivity_before = _require_exclusive_visible_gpu(torch)
    torch.manual_seed(args.seed)
    extension = _load_extension(args.extension, args.module)
    extension_path = Path(extension.__file__)
    extension_identity_before = _file_identity(extension_path)
    interface_authentication_before = _authenticate_loaded_interface(
        source_identities_before["interface_source"]
    )
    if not hasattr(extension, CONVERTER_SYMBOL):
        raise AttributeError(f"extension does not export {CONVERTER_SYMBOL}")
    correctness = _correctness_preflight(torch, extension)
    if not correctness["passed"]:
        raise RuntimeError(f"converter correctness failed: {correctness}")

    temporary = torch.randn(
        (args.batch, args.heads, DEFAULT_DEPTH, args.sequence),
        device="cuda",
        dtype=torch.float16,
    )
    temporary.mul_(4.0)
    source = temporary.to(torch.float8_e4m3fn)
    del temporary
    payload = torch.empty(
        (args.batch, args.heads, DEFAULT_DEPTH, args.sequence // 2),
        device="cuda",
        dtype=torch.float4_e2m1fn_x2,
    )
    scales = torch.empty(
        (args.batch, args.sequence // 128, args.heads, 512),
        device="cuda",
        dtype=torch.float8_e4m3fn,
    )
    timing, measurement_process_checks = _measure(
        torch,
        getattr(extension, CONVERTER_SYMBOL),
        source,
        payload,
        scales,
        args.warmup,
        args.iterations,
    )
    gpu_exclusivity_after = _require_exclusive_visible_gpu(torch)
    interface_authentication_after = _authenticate_loaded_interface(
        source_identities_before["interface_source"]
    )
    converter = timing["median_us"]
    exact = baseline["exact_fp8_projection_us"]
    direct = baseline["direct_mx_projection_us"]
    premium = baseline["direct_mx_premium_us"]
    total_bytes = document["kernel_bytes"]["total"]
    document.update(
        {
            "dry_run": False,
            "artifacts": {
                "extension_before": extension_identity_before,
                "extension_after": _file_identity(extension_path),
                "interface_module_before_timing": (
                    interface_authentication_before
                ),
                "interface_module_after_timing": (
                    interface_authentication_after
                ),
                "sources_before": source_identities_before,
                "sources_after": {
                    "benchmark_source": _file_identity(Path(__file__)),
                    "kernel_source": _file_identity(
                        HERE / "e4m3_to_mxfp4_v.cuh"
                    ),
                    "wrapper_source": _file_identity(
                        HERE / "lowp_fa4_bwd.cu"
                    ),
                    "projection_epilogue": _file_identity(
                        HERE / "projection_fp4_epilogue.cuh"
                    ),
                    "interface_source": _file_identity(
                        REPO_ROOT / "tk_fa4" / "interface.py"
                    ),
                    "factorial_result": _file_identity(args.factorial_result),
                },
            },
            "hardware": {
                "device_name": torch.cuda.get_device_name(),
                "compute_capability": capability,
            },
            "gpu_exclusivity": {
                "checks": [
                    {"phase": "before_measurement", **gpu_exclusivity_before},
                    *measurement_process_checks,
                    {"phase": "after_measurement", **gpu_exclusivity_after},
                ],
            },
            "correctness": correctness,
            "timing": timing,
            "comparison": {
                "split_exact_plus_converter_us": exact + converter,
                "split_minus_direct_mx_us": exact + converter - direct,
                "premium_headroom_us": premium - converter,
                "converter_fraction_of_premium": converter / premium,
                "neutral": converter <= premium,
                "effective_kernel_gb_per_s": total_bytes / converter / 1.0e3,
            },
        }
    )
    if (
        document["artifacts"]["extension_before"]
        != document["artifacts"]["extension_after"]
    ):
        raise RuntimeError("projection extension changed during measurement")
    if (
        document["artifacts"]["sources_before"]
        != document["artifacts"]["sources_after"]
    ):
        raise RuntimeError("benchmark source artifact changed during measurement")
    if interface_authentication_before != interface_authentication_after:
        raise RuntimeError(
            "tk_fa4.interface import identity changed during measurement"
        )
    gpu_checks = document["gpu_exclusivity"]["checks"]
    document["gpu_exclusivity"]["check_count"] = len(gpu_checks)
    document["gpu_exclusivity"]["foreign_process_ids"] = sorted(
        {
            pid
            for check in gpu_checks
            for pid in check["foreign_process_ids"]
        }
    )
    rendered = json.dumps(document, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            args.output,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
        with os.fdopen(descriptor, "w") as output_file:
            output_file.write(rendered)
            output_file.flush()
            os.fsync(output_file.fileno())
    if args.require_neutral and not document["comparison"]["neutral"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
