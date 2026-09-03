#!/usr/bin/env python3
"""Compare direct and inline-E4M3-derived native-NVFP4 MX publication.

This is a projection-only, same-process A/B at the authenticated saturated
B16/S4096/Hq32/Hkv8/D64 Llama-1.2B shape. Both providers consume the same prepared
native-NVFP4 activation/weight operands and own disjoint caller-allocated
publication workspaces. First-use allocating/checked authentication is kept
outside timing; measured calls use only the unchecked out-parameter symbols.

The correctness gate is byte-level. Q/K forward publications and all E4M3
backward publications must remain identical between routes. The derived
forward-MX payload must equal the standalone conversion of its exact E4M3(x4)
backward V. Only the 256 addressable bytes in each D64 512-byte scale page are
compared. Provider order alternates for every CUDA-event sample. The receipt
is create-only and the benchmark requires exactly one visible SM100 GPU.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import random
import re
import stat
import statistics
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
BATCH = 16
SEQUENCE = 4096
HIDDEN = 2048
Q_HEADS = 32
KV_HEADS = 8
HEAD_DIM = 64
QKV_WIDTH = (Q_HEADS + 2 * KV_HEADS) * HEAD_DIM
MINIMUM_SAMPLES = 100
MINIMUM_BOOTSTRAP_DRAWS = 1_000
DERIVED_SYMBOL = (
    "project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_"
    "interleaved_causal_represented_backward_perblock_qk_"
    "e4m3_derived_mx_forward_out"
)
CONVERTER_SYMBOL = "convert_e4m3_x4_v_bhds_to_causal_mxfp4"
VALID_D64_SCALE_INDICES = tuple(
    depth_lane * 16 + depth_group * 4 + sequence_quarter
    for depth_lane in range(32)
    for depth_group in range(2)
    for sequence_quarter in range(4)
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--warmups", type=_nonnegative_int, default=12)
    parser.add_argument("--samples", type=_positive_int, default=120)
    parser.add_argument(
        "--bootstrap-draws",
        type=_positive_int,
        default=20_000,
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.samples < MINIMUM_SAMPLES:
        parser.error(f"--samples must be at least {MINIMUM_SAMPLES}")
    if args.samples % 2:
        parser.error("--samples must be even to form balanced order blocks")
    if args.bootstrap_draws < MINIMUM_BOOTSTRAP_DRAWS:
        parser.error(
            "--bootstrap-draws must be at least "
            f"{MINIMUM_BOOTSTRAP_DRAWS}"
        )
    return args


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(path.expanduser()))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    selected = _absolute(path)
    observed = selected.lstat()
    if not stat.S_ISREG(observed.st_mode):
        raise RuntimeError(f"artifact must be a regular non-symlink file: {selected}")
    resolved = selected.resolve(strict=True)
    resolved_stat = resolved.stat()
    return {
        "selected_path": str(selected),
        "resolved_path": str(resolved),
        "sha256": _sha256(resolved),
        "bytes": resolved_stat.st_size,
        "mtime_ns": resolved_stat.st_mtime_ns,
    }


def _authenticate_loaded_interface(
    interface: Any,
    expected_path: Path,
) -> dict[str, Any]:
    """Prove that the imported interface is this worktree's source file."""
    loaded_file = getattr(interface, "__file__", None)
    if not isinstance(loaded_file, (str, os.PathLike)):
        raise RuntimeError("loaded tk_fa4.interface has no filesystem __file__")
    expected_identity = _file_identity(expected_path)
    loaded_identity = _file_identity(Path(loaded_file))
    if (
        loaded_identity["resolved_path"]
        != expected_identity["resolved_path"]
    ):
        raise RuntimeError(
            "loaded tk_fa4.interface is shadowed: "
            f"{loaded_identity['resolved_path']} != "
            f"{expected_identity['resolved_path']}"
        )
    if loaded_identity["sha256"] != expected_identity["sha256"]:
        raise RuntimeError(
            "loaded tk_fa4.interface bytes differ from the expected source"
        )
    return {
        "module": "tk_fa4.interface",
        "expected_file_identity": expected_identity,
        "loaded_file_identity": loaded_identity,
        "resolved_path_matches_expected": True,
        "sha256_matches_expected": True,
    }


def _git_output(*arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ("git", "-C", str(REPO_ROOT), *arguments),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.rstrip("\n")


def _git_receipt() -> dict[str, Any]:
    status = _git_output("status", "--porcelain=v1")
    return {
        "root": str(REPO_ROOT),
        "head": _git_output("rev-parse", "HEAD"),
        "branch": _git_output("branch", "--show-current"),
        "dirty": bool(status) if status is not None else None,
        "status": status,
    }


def _parse_gpu_process_report(report: Any) -> dict[str, Any]:
    """Parse only the documented Torch/NVML process-report grammar."""
    if not isinstance(report, str) or not report.strip():
        raise RuntimeError("CUDA process report is unavailable or empty")
    lines = report.strip().splitlines()
    if any(not line.strip() for line in lines):
        raise RuntimeError("CUDA process report contains an empty record")
    header = re.fullmatch(r"GPU:(\d+)", lines[0].strip())
    if header is None:
        raise RuntimeError(
            "CUDA process report has an unrecognized GPU header"
        )
    body = [line.strip() for line in lines[1:]]
    if not body:
        raise RuntimeError("CUDA process report omitted process state")
    if body == ["no processes are running"]:
        process_ids: list[int] = []
    else:
        process_ids = []
        process_pattern = re.compile(
            r"process\s+(\d+)\s+uses\s+"
            r"\d+(?:\.\d+)?\s+MB GPU memory"
        )
        for line in body:
            match = process_pattern.fullmatch(line)
            if match is None:
                raise RuntimeError(
                    "CUDA process report contains an unrecognized record"
                )
            process_ids.append(int(match.group(1)))
        if len(set(process_ids)) != len(process_ids):
            raise RuntimeError("CUDA process report contains duplicate PIDs")
    return {
        "reported_gpu_index": int(header.group(1)),
        "observed_process_ids": process_ids,
        "process_count": len(process_ids),
        "report_parsed": True,
    }


def _require_exclusive_visible_gpu(torch: Any) -> dict[str, Any]:
    try:
        report = torch.cuda.list_gpu_processes(0)
    except Exception as error:
        raise RuntimeError("CUDA process enumeration failed") from error
    parsed = _parse_gpu_process_report(report)
    process_ids = parsed["observed_process_ids"]
    own_pid = os.getpid()
    foreign = [pid for pid in process_ids if pid != own_pid]
    if foreign:
        raise RuntimeError(
            "projection A/B requires an exclusive visible GPU; "
            f"foreign compute PIDs: {foreign}"
        )
    return {
        **parsed,
        "own_pid": own_pid,
        "foreign_process_ids": foreign,
    }


def _plan(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": "native_inline_e4m3_derived_mx_projection_plan_v1",
        "dry_run": True,
        "touches_cuda": False,
        "shape": {
            "batch": BATCH,
            "sequence": SEQUENCE,
            "hidden": HIDDEN,
            "q_heads": Q_HEADS,
            "kv_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "qkv_width": QKV_WIDTH,
        },
        "parameters": {
            "seed": args.seed,
            "warmups": args.warmups,
            "samples": args.samples,
            "bootstrap_draws": args.bootstrap_draws,
        },
        "providers": ["direct_native_mx", "inline_e4m3_derived_native_mx"],
        "timing": {
            "method": "alternating-order same-stream CUDA events",
            "warmups_per_provider": args.warmups,
            "samples_per_provider": args.samples,
            "balanced_order_blocks": args.samples // 2,
            "bootstrap_draws": args.bootstrap_draws,
            "minimum_samples_per_provider": MINIMUM_SAMPLES,
            "first_use_authentication_excluded": True,
        },
        "correctness": {
            "direct_vs_derived_bitwise": [
                "Q/K payloads and scales",
                "Q/K global scales",
                "E4M3 backward V/Q/K",
            ],
            "derived_mx_reference": (
                "standalone MX(E4M3(4V)/4) converter"
            ),
            "scale_page_valid_bytes": len(VALID_D64_SCALE_INDICES),
            "scale_page_reserved_bytes_excluded": (
                512 - len(VALID_D64_SCALE_INDICES)
            ),
            "fail_closed": True,
        },
        "extension": str(_absolute(args.extension)),
        "output": str(_absolute(args.output)),
    }


def _make_rope(torch: Any) -> tuple[Any, Any]:
    positions = torch.arange(SEQUENCE, device="cuda", dtype=torch.float32)
    frequencies = 1.0 / (
        500_000.0
        ** (
            torch.arange(HEAD_DIM // 2, device="cuda", dtype=torch.float32)
            / (HEAD_DIM // 2)
        )
    )
    angles = positions[:, None] * frequencies[None, :]
    cosine = angles.cos()[None].repeat(BATCH, 1, 1).bfloat16().contiguous()
    sine = angles.sin()[None].repeat(BATCH, 1, 1).bfloat16().contiguous()
    return cosine, sine


def _allocate_workspace(torch: Any, interface: Any) -> Any:
    device = torch.device("cuda")
    q_payload = torch.empty(
        BATCH,
        Q_HEADS,
        SEQUENCE,
        HEAD_DIM // 2,
        device=device,
        dtype=torch.uint8,
    )
    k_payload = torch.empty(
        BATCH,
        KV_HEADS,
        SEQUENCE,
        HEAD_DIM // 2,
        device=device,
        dtype=torch.uint8,
    )
    return interface.B300E4M3QKVForwardWorkspace(
        q_payload=q_payload,
        k_payload=k_payload,
        q_scale_pages=torch.empty(
            BATCH,
            SEQUENCE // 128,
            Q_HEADS,
            512,
            device=device,
            dtype=torch.float8_e4m3fn,
        ),
        q_global_scale=torch.empty(
            BATCH, Q_HEADS, device=device, dtype=torch.float32
        ),
        k_scale_pages=torch.empty(
            BATCH,
            SEQUENCE // 64,
            KV_HEADS,
            512,
            device=device,
            dtype=torch.float8_e4m3fn,
        ),
        k_global_scale=torch.empty(
            BATCH, KV_HEADS, device=device, dtype=torch.float32
        ),
        v_mxfp4_payload=torch.empty(
            BATCH,
            KV_HEADS,
            HEAD_DIM,
            SEQUENCE // 2,
            device=device,
            dtype=torch.float4_e2m1fn_x2,
        ),
        v_mxfp4_scale_pages=torch.empty(
            BATCH,
            SEQUENCE // 128,
            KV_HEADS,
            512,
            device=device,
            dtype=torch.float8_e4m3fn,
        ),
        v_fp8_payload=torch.empty(
            BATCH,
            KV_HEADS,
            HEAD_DIM,
            SEQUENCE,
            device=device,
            dtype=torch.float8_e4m3fn,
        ),
        v_backward_fp8=torch.empty(
            BATCH,
            SEQUENCE,
            KV_HEADS,
            HEAD_DIM,
            device=device,
            dtype=torch.float8_e4m3fn,
        ),
        q_backward_fp8=torch.empty(
            BATCH,
            SEQUENCE,
            Q_HEADS,
            HEAD_DIM,
            device=device,
            dtype=torch.float8_e4m3fn,
        ),
        k_backward_fp8=torch.empty(
            BATCH,
            SEQUENCE,
            KV_HEADS,
            HEAD_DIM,
            device=device,
            dtype=torch.float8_e4m3fn,
        ),
        q_payload_fp4=q_payload.view(torch.float4_e2m1fn_x2),
        k_payload_fp4=k_payload.view(torch.float4_e2m1fn_x2),
        empty_bf16=torch.empty(0, device=device, dtype=torch.bfloat16),
        empty_byte=torch.empty(0, device=device, dtype=torch.uint8),
        empty_fp8=torch.empty(0, device=device, dtype=torch.float8_e4m3fn),
        empty_fp4=torch.empty(0, device=device, dtype=torch.float4_e2m1fn_x2),
    )


def _workspace_pointers(workspace: Any) -> dict[str, int]:
    return {
        name: int(getattr(workspace, name).data_ptr())
        for name in (
            "q_payload",
            "k_payload",
            "q_scale_pages",
            "q_global_scale",
            "k_scale_pages",
            "k_global_scale",
            "v_mxfp4_payload",
            "v_mxfp4_scale_pages",
            "v_fp8_payload",
            "v_backward_fp8",
            "q_backward_fp8",
            "k_backward_fp8",
        )
    }


def _require_checked_input_output_alias_rejection(
    torch: Any,
    projector: Any,
    workspace: Any,
    input_operand: tuple[Any, Any, Any],
    weight_operand: tuple[Any, Any, Any],
    qk_scales: Any,
    rope: Any,
) -> dict[str, Any]:
    """Exercise the checked raw ABI with Q deliberately aliasing input."""
    aliased_q = input_operand[0].view(torch.uint8).view(
        BATCH,
        Q_HEADS,
        SEQUENCE,
        HEAD_DIM // 2,
    )
    aliased_workspace = replace(
        workspace,
        q_payload=aliased_q,
        q_payload_fp4=aliased_q.view(torch.float4_e2m1fn_x2),
    )
    try:
        projector._project_checked(
            *input_operand,
            *weight_operand,
            qk_scales,
            rope,
            BATCH,
            SEQUENCE,
            Q_HEADS,
            KV_HEADS,
            False,
            *aliased_workspace.compact_outputs(),
        )
    except RuntimeError as error:
        message = str(error)
        expected = "must not overlap read operand input_fp4"
        if expected not in message:
            raise RuntimeError(
                "checked raw ABI rejected the alias for an unexpected reason: "
                f"{message}"
            ) from error
        return {
            "passed": True,
            "aliased_output": "q_depth_packed_out",
            "aliased_read_operand": "input_fp4",
            "expected_error_fragment": expected,
        }
    raise RuntimeError("checked raw ABI accepted an input/output storage alias")


def _bitwise_comparison(
    torch: Any,
    left: Any,
    right: Any,
    *,
    valid_last_dim_indices: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    metadata_equal = bool(
        left.dtype == right.dtype
        and tuple(left.shape) == tuple(right.shape)
        and left.device == right.device
        and left.is_contiguous()
        and right.is_contiguous()
    )
    if not metadata_equal:
        return {
            "metadata_equal": False,
            "bytes_compared": 0,
            "byte_mismatches": None,
            "passed": False,
        }
    left_bytes = left.view(torch.uint8)
    right_bytes = right.view(torch.uint8)
    if valid_last_dim_indices is not None:
        indices = torch.tensor(
            valid_last_dim_indices,
            device=left.device,
            dtype=torch.long,
        )
        left_bytes = left_bytes.index_select(-1, indices)
        right_bytes = right_bytes.index_select(-1, indices)
    mismatches = int((left_bytes != right_bytes).sum().item())
    return {
        "metadata_equal": True,
        "bytes_compared": left_bytes.numel(),
        "byte_mismatches": mismatches,
        "passed": mismatches == 0,
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


def _bootstrap_mean_interval(
    values: Sequence[float],
    *,
    draws: int,
    seed: int,
) -> tuple[float, float]:
    """Return a deterministic percentile CI over balanced-order blocks."""
    if len(values) < 2:
        raise ValueError("bootstrap requires at least two blocks")
    if draws < MINIMUM_BOOTSTRAP_DRAWS:
        raise ValueError(
            f"bootstrap requires at least {MINIMUM_BOOTSTRAP_DRAWS} draws"
        )
    generator = random.Random(seed)
    count = len(values)
    means = [
        statistics.fmean(
            values[generator.randrange(count)] for _ in range(count)
        )
        for _ in range(draws)
    ]
    return _percentile(means, 0.025), _percentile(means, 0.975)


def _measure(
    torch: Any,
    functions: dict[str, Callable[[], Any]],
    *,
    warmups: int,
    samples: int,
    bootstrap_draws: int,
    seed: int,
) -> dict[str, Any]:
    names = ("direct_native_mx", "inline_e4m3_derived_native_mx")
    if tuple(functions) != names:
        raise ValueError("projection providers are not in canonical A/B order")
    if samples % 2:
        raise ValueError("samples must form complementary two-sample blocks")
    for iteration in range(warmups):
        order = names if iteration % 2 == 0 else names[::-1]
        retained = [functions[name]() for name in order]
        torch.cuda.synchronize()
        del retained

    values = {name: [] for name in names}
    orders = []
    periodic_gpu_checks = []
    for iteration in range(samples):
        order = names if iteration % 2 == 0 else names[::-1]
        orders.append(list(order))
        events = []
        retained = []
        for name in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            retained.append(functions[name]())
            end.record()
            events.append((name, start, end))
        events[-1][2].synchronize()
        for name, start, end in events:
            values[name].append(float(start.elapsed_time(end) * 1000.0))
        del retained
        if iteration % 2 == 1:
            # This check happens after both complementary samples have fully
            # synchronized and their event intervals have been read. NVML
            # enumeration therefore cannot enter a measured CUDA interval.
            periodic_gpu_checks.append(
                {
                    "balanced_order_block": iteration // 2,
                    **_require_exclusive_visible_gpu(torch),
                }
            )
    direct = values[names[0]]
    derived = values[names[1]]
    paired_delta = [
        derived_value - direct_value
        for direct_value, derived_value in zip(direct, derived, strict=True)
    ]
    # Consecutive iterations use complementary provider order. Averaging
    # their oriented deltas cancels the large first/second launch-position
    # effect before treating blocks as independent sampling units.
    balanced_order_block_deltas = [
        statistics.fmean(paired_delta[start : start + 2])
        for start in range(0, len(paired_delta), 2)
    ]
    balanced_interval = _bootstrap_mean_interval(
        balanced_order_block_deltas,
        draws=bootstrap_draws,
        seed=seed,
    )
    summaries = {name: _timing_summary(values[name]) for name in names}
    direct_median = summaries[names[0]]["median_us"]
    derived_median = summaries[names[1]]["median_us"]
    periodic_foreign_process_ids = sorted(
        {
            pid
            for check in periodic_gpu_checks
            for pid in check["foreign_process_ids"]
        }
    )
    return {
        "providers": summaries,
        "paired_derived_minus_direct": _timing_summary(paired_delta),
        "balanced_order_block_derived_minus_direct": {
            **_timing_summary(balanced_order_block_deltas),
            "bootstrap_mean_95_percent_us": list(balanced_interval),
            "bootstrap_draws": bootstrap_draws,
            "block_definition": (
                "mean of two consecutive complementary provider orders"
            ),
        },
        "sample_orders": orders,
        "periodic_gpu_exclusivity": {
            "placement": (
                "after each synchronized complementary two-sample block, "
                "outside CUDA event intervals"
            ),
            "check_count": len(periodic_gpu_checks),
            "expected_check_count": samples // 2,
            "foreign_process_ids": periodic_foreign_process_ids,
            "checks": periodic_gpu_checks,
        },
        "comparison": {
            "derived_minus_direct_median_us": (
                derived_median - direct_median
            ),
            "direct_over_derived_median_speedup": (
                direct_median / derived_median
            ),
            "derived_faster": derived_median < direct_median,
            "balanced_mean_derived_slower": balanced_interval[0] > 0.0,
            "balanced_mean_derived_faster": balanced_interval[1] < 0.0,
        },
    }


def _write_create_only(path: Path, content: str) -> None:
    destination = _absolute(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    with os.fdopen(descriptor, "w") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _run(args: argparse.Namespace) -> int:
    output = _absolute(args.output)
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite benchmark output: {output}")
    extension_before = _file_identity(args.extension)
    source_paths = {
        "benchmark": Path(__file__),
        "interface": REPO_ROOT / "tk_fa4" / "interface.py",
        "projection_translation_unit": HERE / "lowp_fa4_bwd.cu",
        "projection_epilogue": HERE / "projection_fp4_epilogue.cuh",
        "standalone_converter": HERE / "e4m3_to_mxfp4_v.cuh",
    }
    sources_before = {
        name: _file_identity(path) for name, path in source_paths.items()
    }
    os.environ["TK_FA4_LOWP_BWD_EXTENSION_SOURCE"] = extension_before[
        "resolved_path"
    ]

    import torch
    import tk_fa4.interface as interface

    loaded_interface_before = _authenticate_loaded_interface(
        interface,
        source_paths["interface"],
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("expose exactly one GPU to this benchmark")
    torch.cuda.set_device(0)
    capability = list(torch.cuda.get_device_capability(0))
    if capability != [10, 0]:
        raise RuntimeError("this benchmark requires an SM100 GPU")
    gpu_processes_before = _require_exclusive_visible_gpu(torch)
    loaded_extension = Path(interface._C_b300_lowp_bwd.__file__).resolve(strict=True)
    if loaded_extension != Path(extension_before["resolved_path"]):
        raise RuntimeError("interface loaded a different projection extension")
    for symbol in (
        DERIVED_SYMBOL,
        DERIVED_SYMBOL + "_unchecked",
        CONVERTER_SYMBOL,
    ):
        if getattr(interface._C_b300_lowp_bwd, symbol, None) is None:
            raise RuntimeError(f"projection extension does not export {symbol}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed)
    rows = torch.empty(
        BATCH * SEQUENCE,
        HIDDEN,
        device="cuda",
        dtype=torch.bfloat16,
    ).normal_(mean=0.0, std=1.0, generator=generator)
    qkv_weight = torch.empty(
        QKV_WIDTH,
        HIDDEN,
        device="cuda",
        dtype=torch.bfloat16,
    ).normal_(mean=0.0, std=0.02, generator=generator)
    qk_scales = torch.zeros(
        BATCH,
        Q_HEADS // 2,
        7,
        device="cuda",
        dtype=torch.float32,
    )
    qk_scales[..., 0] = 2.25
    qk_scales[..., 1] = 2.0
    rope_cosine, rope_sine = _make_rope(torch)
    rope = interface.b300_pack_gqa_d64_paired_rope(rope_cosine, rope_sine)
    input_operand = tuple(interface.b300_prepare_nvfp4_projection_operand(rows))
    weight_operand = tuple(
        interface.b300_prepare_nvfp4_projection_weight(qkv_weight)
    )

    direct_workspace = _allocate_workspace(torch, interface)
    derived_workspace = _allocate_workspace(torch, interface)
    initial_pointers = {
        "direct_native_mx": _workspace_pointers(direct_workspace),
        "inline_e4m3_derived_native_mx": _workspace_pointers(
            derived_workspace
        ),
    }
    direct = interface.b300_bind_qkv_gqa_d64_paired_unified_lowp_nvfp4_projection(
        batch=BATCH,
        seqlen=SEQUENCE,
        q_heads=Q_HEADS,
        kv_heads=KV_HEADS,
        publish_mxfp4_v=True,
        v_mxfp4_scale_2d=False,
    )
    derived = interface.b300_bind_qkv_gqa_d64_paired_unified_lowp_nvfp4_projection(
        batch=BATCH,
        seqlen=SEQUENCE,
        q_heads=Q_HEADS,
        kv_heads=KV_HEADS,
        publish_mxfp4_v=True,
        v_mxfp4_scale_2d=False,
        experimental_e4m3_derived_mxfp4_v=True,
    )
    if direct.checked_symbol == derived.checked_symbol:
        raise RuntimeError("direct and derived binders selected the same symbol")
    if derived.checked_symbol != DERIVED_SYMBOL:
        raise RuntimeError("derived binder selected an unexpected checked symbol")
    if direct.experimental_e4m3_derived_mxfp4_v:
        raise RuntimeError("default direct binder unexpectedly enabled derived MX")
    if derived.experimental_split_v_backward:
        raise RuntimeError("derived binder unexpectedly enabled split-V publication")

    checked_alias_rejection = _require_checked_input_output_alias_rejection(
        torch,
        derived,
        derived_workspace,
        input_operand,
        weight_operand,
        qk_scales,
        rope,
    )

    def project(projector: Any, workspace: Any) -> Any:
        return projector(
            input_operand,
            weight_operand,
            qk_scales,
            rope,
            forward_workspace=workspace,
        )

    # Both first calls run the allocating reference and checked out ABI.
    # Binder-internal bitwise authentication is deliberately outside timing.
    with torch.no_grad():
        project(direct, direct_workspace)
        project(derived, derived_workspace)
        torch.cuda.synchronize()
    for name, projector in (("direct", direct), ("derived", derived)):
        if not projector.forward_workspace_abi_validated:
            raise RuntimeError(f"{name} binder did not authenticate its workspace")
        if projector.validated_forward_workspace_count != 1:
            raise RuntimeError(f"{name} binder authenticated an unexpected count")

    comparisons = {
        "q_payload": _bitwise_comparison(
            torch, direct_workspace.q_payload, derived_workspace.q_payload
        ),
        "k_payload": _bitwise_comparison(
            torch, direct_workspace.k_payload, derived_workspace.k_payload
        ),
        "q_scale_pages": _bitwise_comparison(
            torch,
            direct_workspace.q_scale_pages,
            derived_workspace.q_scale_pages,
        ),
        "q_global_scale": _bitwise_comparison(
            torch,
            direct_workspace.q_global_scale,
            derived_workspace.q_global_scale,
        ),
        "k_scale_pages": _bitwise_comparison(
            torch,
            direct_workspace.k_scale_pages,
            derived_workspace.k_scale_pages,
        ),
        "k_global_scale": _bitwise_comparison(
            torch,
            direct_workspace.k_global_scale,
            derived_workspace.k_global_scale,
        ),
        "backward_v_e4m3": _bitwise_comparison(
            torch,
            direct_workspace.v_backward_fp8,
            derived_workspace.v_backward_fp8,
        ),
        "backward_q_e4m3": _bitwise_comparison(
            torch,
            direct_workspace.q_backward_fp8,
            derived_workspace.q_backward_fp8,
        ),
        "backward_k_e4m3": _bitwise_comparison(
            torch,
            direct_workspace.k_backward_fp8,
            derived_workspace.k_backward_fp8,
        ),
    }
    feature_major_v = (
        derived_workspace.v_backward_fp8.permute(0, 2, 3, 1).contiguous()
    )
    reference_payload, reference_scales = getattr(
        interface._C_b300_lowp_bwd, CONVERTER_SYMBOL
    )(feature_major_v)
    reference_comparisons = {
        "derived_payload_vs_standalone_converter": _bitwise_comparison(
            torch,
            derived_workspace.v_mxfp4_payload,
            reference_payload,
        ),
        "derived_valid_scales_vs_standalone_converter": _bitwise_comparison(
            torch,
            derived_workspace.v_mxfp4_scale_pages,
            reference_scales,
            valid_last_dim_indices=VALID_D64_SCALE_INDICES,
        ),
    }
    diagnostic_direct_vs_derived = {
        "payload": _bitwise_comparison(
            torch,
            direct_workspace.v_mxfp4_payload,
            derived_workspace.v_mxfp4_payload,
        ),
        "valid_scale_bytes": _bitwise_comparison(
            torch,
            direct_workspace.v_mxfp4_scale_pages,
            derived_workspace.v_mxfp4_scale_pages,
            valid_last_dim_indices=VALID_D64_SCALE_INDICES,
        ),
        "required_to_match": False,
        "reason": "MX(BF16 V) and MX(E4M3(4V)/4) are distinct semantics",
    }
    correctness_passed = all(
        comparison["passed"]
        for comparison in (*comparisons.values(), *reference_comparisons.values())
    )
    if not correctness_passed:
        raise RuntimeError(
            "projection correctness authentication failed before timing: "
            f"{comparisons!r}, {reference_comparisons!r}"
        )

    functions = {
        "direct_native_mx": lambda: project(direct, direct_workspace),
        "inline_e4m3_derived_native_mx": lambda: project(
            derived, derived_workspace
        ),
    }
    with torch.no_grad():
        # One explicit second call guarantees both binders have moved to their
        # unchecked symbols before any warmup or measured event.
        functions["direct_native_mx"]()
        functions["inline_e4m3_derived_native_mx"]()
        torch.cuda.synchronize()
        timing = _measure(
            torch,
            functions,
            warmups=args.warmups,
            samples=args.samples,
            bootstrap_draws=args.bootstrap_draws,
            seed=args.seed + 104729,
        )
    final_pointers = {
        "direct_native_mx": _workspace_pointers(direct_workspace),
        "inline_e4m3_derived_native_mx": _workspace_pointers(
            derived_workspace
        ),
    }
    pointers_stable = initial_pointers == final_pointers
    if not pointers_stable:
        raise RuntimeError("caller-owned publication workspace pointer changed")

    gpu_processes_after = _require_exclusive_visible_gpu(torch)
    extension_after = _file_identity(args.extension)
    if extension_before != extension_after:
        raise RuntimeError("projection extension changed during measurement")
    sources_after = {
        name: _file_identity(path) for name, path in source_paths.items()
    }
    if sources_before != sources_after:
        raise RuntimeError("projection benchmark source changed during measurement")
    loaded_interface_after = _authenticate_loaded_interface(
        interface,
        source_paths["interface"],
    )
    if loaded_interface_before != loaded_interface_after:
        raise RuntimeError(
            "loaded tk_fa4.interface identity changed during measurement"
        )
    periodic_gpu_exclusivity = timing["periodic_gpu_exclusivity"]
    all_foreign_process_ids = sorted(
        {
            *gpu_processes_before["foreign_process_ids"],
            *periodic_gpu_exclusivity["foreign_process_ids"],
            *gpu_processes_after["foreign_process_ids"],
        }
    )
    total_gpu_exclusivity_check_count = (
        2 + periodic_gpu_exclusivity["check_count"]
    )
    document = {
        **_plan(args),
        "schema": "native_inline_e4m3_derived_mx_projection_v1",
        "dry_run": False,
        "touches_cuda": True,
        "created_utc": (
            dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
        ),
        "hardware": {
            "name": torch.cuda.get_device_name(0),
            "compute_capability": capability,
            "visible_device_count": torch.cuda.device_count(),
        },
        "artifacts": {
            "extension_before": extension_before,
            "extension_after": extension_after,
            "sources_before": sources_before,
            "sources_after": sources_after,
            "loaded_interface_before": loaded_interface_before,
            "loaded_interface_after": loaded_interface_after,
            "extension_and_sources_unchanged_across_timing": True,
            "loaded_interface_unchanged_across_timing": True,
        },
        "git": _git_receipt(),
        "projection_contracts": {
            "direct_native_mx": {
                "checked_symbol": direct.checked_symbol,
                "unchecked_symbol": direct.unchecked_symbol,
                "derived_opt_in": direct.experimental_e4m3_derived_mxfp4_v,
                "split_v_backward": direct.experimental_split_v_backward,
                "authenticated_workspace_count": (
                    direct.validated_forward_workspace_count
                ),
            },
            "inline_e4m3_derived_native_mx": {
                "checked_symbol": derived.checked_symbol,
                "unchecked_symbol": derived.unchecked_symbol,
                "derived_opt_in": derived.experimental_e4m3_derived_mxfp4_v,
                "split_v_backward": derived.experimental_split_v_backward,
                "authenticated_workspace_count": (
                    derived.validated_forward_workspace_count
                ),
            },
            "first_use_authentication_excluded_from_timing": True,
            "steady_state_uses_unchecked_out_parameter_symbols": True,
            "checked_input_output_alias_rejection": checked_alias_rejection,
            "caller_owned_workspace_pointers_stable": pointers_stable,
            "exclusive_visible_gpu_before": gpu_processes_before,
            "exclusive_visible_gpu_periodic": periodic_gpu_exclusivity,
            "exclusive_visible_gpu_after": gpu_processes_after,
            "exclusive_visible_gpu_check_count": (
                total_gpu_exclusivity_check_count
            ),
            "foreign_gpu_process_ids_across_checks": (
                all_foreign_process_ids
            ),
        },
        "correctness": {
            "direct_vs_derived": comparisons,
            "derived_vs_standalone_converter": reference_comparisons,
            "direct_vs_derived_forward_mx_diagnostic": (
                diagnostic_direct_vs_derived
            ),
            "scale_page_valid_last_dim_indices": list(
                VALID_D64_SCALE_INDICES
            ),
            "reserved_scale_page_bytes_compared": False,
            "passed": correctness_passed,
        },
        "timing_result": timing,
    }
    rendered = json.dumps(document, indent=2, sort_keys=True) + "\n"
    _write_create_only(output, rendered)
    print(rendered, end="")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.dry_run:
        print(json.dumps(_plan(args), indent=2, sort_keys=True))
        return 0
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
