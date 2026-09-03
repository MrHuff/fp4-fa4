from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import itertools
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FLASH_ATTENTION = REPO / "flash-attention"
LOCAL_CUTLASS_DSL = (
    REPO
    / "results"
    / ".artifacts"
    / "python"
    / "nvidia_cutlass_dsl"
    / "python_packages"
)
SCOPED_PATHS = (
    "tk_fa4/b300_bwd_cute16_candidate.cuh",
    "tk_fa4/b300_bwd_cute16_kernel_candidate.cuh",
    "tk_fa4/tk_fa4.cu",
    "results/tk_bf16_bwd_s65536h128_harness.py",
    "results/tk_bf16_bwd_broad_sweep_20260711.md",
)
ERROR_LIMITS = {
    "dq": {"rel_l2": 0.01, "max_abs": 0.05},
    "dk": {"rel_l2": 0.01, "max_abs": 0.05},
    "dv": {"rel_l2": 0.001, "max_abs": 0.02},
}
DRIFT_LIMITS = {"dq": 2.0e-7, "dk": 0.0, "dv": 0.0}

torch = None
_flash_attn_bwd = None
flash_attn_func = None
cutlass_module = None
flash_attn_cute_interface_module = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_command(*arguments: str, binary: bool = False):
    return subprocess.run(
        ("git", *arguments),
        cwd=REPO,
        check=True,
        capture_output=True,
        text=not binary,
    ).stdout


def build_manifest(args: argparse.Namespace) -> dict:
    extension = args.extension.resolve()
    harness = Path(__file__).resolve()
    if not extension.is_file():
        raise FileNotFoundError(f"extension artifact does not exist: {extension}")
    scoped_diff = git_command("diff", "--binary", "--", *SCOPED_PATHS, binary=True)
    diff_check = subprocess.run(
        ("git", "diff", "--check", "--", *SCOPED_PATHS),
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    return {
        "complete": False,
        "gate_pass": False,
        "state": "initialized",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "cwd": str(Path.cwd().resolve()),
        "python_executable": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "extension": str(extension),
        "extension_sha256": sha256_file(extension),
        "extension_size_bytes": extension.stat().st_size,
        "harness": str(harness),
        "harness_sha256": sha256_file(harness),
        "git_head": git_command("rev-parse", "HEAD").strip(),
        "scoped_paths": list(SCOPED_PATHS),
        "scoped_status": git_command(
            "status", "--short", "--", *SCOPED_PATHS
        ).splitlines(),
        "scoped_diff_sha256": hashlib.sha256(scoped_diff).hexdigest(),
        "scoped_diff_size_bytes": len(scoped_diff),
        "scoped_diff_check_pass": diff_check.returncode == 0,
        "scoped_diff_check_output": diff_check.stdout + diff_check.stderr,
        "mode": args.mode,
        "seqlen": args.seqlen,
        "heads": args.heads,
        "seed": args.seed,
        "parent": args.parent,
        "child": args.child,
        "thresholds": {
            "error_limits": ERROR_LIMITS,
            "repeat_drift_limits": DRIFT_LIMITS,
            "timing_child_over_parent": 1.0,
        },
    }


def load_runtime() -> None:
    global torch, _flash_attn_bwd, flash_attn_func
    global cutlass_module, flash_attn_cute_interface_module

    for path in (LOCAL_CUTLASS_DSL, REPO, FLASH_ATTENTION):
        if path.exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
    warnings.filterwarnings(
        "ignore",
        message="Use explicit `struct.scalar.ptr` for pointer instead.",
        category=DeprecationWarning,
    )
    import torch as torch_module
    import cutlass as imported_cutlass_module
    import flash_attn.cute.interface as imported_flash_attn_cute_interface
    from flash_attn.cute.interface import (
        _flash_attn_bwd as flash_attn_bwd_module,
    )
    from flash_attn.cute.interface import (
        flash_attn_func as flash_attn_func_module,
    )

    torch = torch_module
    cutlass_module = imported_cutlass_module
    flash_attn_cute_interface_module = imported_flash_attn_cute_interface
    _flash_attn_bwd = flash_attn_bwd_module
    flash_attn_func = flash_attn_func_module


def distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def local_artifact_distribution_version(name: str) -> str | None:
    normalized = name.replace("-", "_")
    artifact_root = REPO / "results" / ".artifacts" / "python"
    matches = sorted(artifact_root.glob(f"{normalized}-*.dist-info/METADATA"))
    if len(matches) != 1:
        return None
    for line in matches[0].read_text().splitlines():
        if line.startswith("Version: "):
            return line.removeprefix("Version: ")
    return None


def runtime_module_provenance() -> dict:
    flash_attention_head = subprocess.run(
        ("git", "-C", str(FLASH_ATTENTION), "rev-parse", "HEAD"),
        capture_output=True,
        text=True,
    )
    return {
        "flash_attn_cute_interface": {
            "module_path": str(
                Path(flash_attn_cute_interface_module.__file__).resolve()
            ),
            "repository_head": (
                flash_attention_head.stdout.strip()
                if flash_attention_head.returncode == 0
                else None
            ),
        },
        "cutlass_dsl": {
            "module_path": str(Path(cutlass_module.__file__).resolve()),
            "module_version": getattr(cutlass_module, "__version__", None),
            "resolved_distribution_version": distribution_version(
                "nvidia-cutlass-dsl"
            ),
            "resolved_libs_base_distribution_version": distribution_version(
                "nvidia-cutlass-dsl-libs-base"
            ),
            "local_artifact_distribution_version": (
                local_artifact_distribution_version("nvidia-cutlass-dsl")
            ),
            "local_artifact_libs_base_distribution_version": (
                local_artifact_distribution_version(
                    "nvidia-cutlass-dsl-libs-base"
                )
            ),
        },
    }


def load_extension(path: Path):
    spec = importlib.util.spec_from_file_location("_C", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load extension from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path | None, payload: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def make_inputs(seed: int, seqlen: int, heads: int):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    shape_qk = (1, seqlen, heads, 192)
    shape_v = (1, seqlen, heads, 128)
    q = torch.randn(
        shape_qk, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    k = torch.randn(
        shape_qk, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    v = torch.randn(
        shape_v, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    dout = torch.randn(
        shape_v, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    out, lse_raw = flash_attn_func(q, k, v, causal=True, return_lse=True)
    if lse_raw.shape == (1, heads, seqlen):
        lse_tk = lse_raw.permute(0, 2, 1).contiguous()
    elif lse_raw.shape == (1, seqlen, heads):
        lse_tk = lse_raw.contiguous()
    else:
        raise RuntimeError(f"unexpected LSE shape: {tuple(lse_raw.shape)}")
    return q, k, v, dout, out, lse_raw, lse_tk


def call_tk(fn, data, seqlen: int):
    q, k, v, dout, out, _, lse_tk = data
    return fn(
        q,
        k,
        v,
        out,
        lse_tk,
        dout,
        True,
        192.0**-0.5,
        seqlen,
        False,
    )


def call_cute(data):
    q, k, v, dout, out, lse_raw, _ = data
    return _flash_attn_bwd(
        q, k, v, out, dout, lse_raw, causal=True, deterministic=False
    )


def metrics(actual, reference):
    rows = []
    for name, value, expected in zip(("dq", "dk", "dv"), actual, reference):
        value_f = value.float()
        expected_f = expected.float()
        difference = value_f - expected_f
        rows.append(
            {
                "name": name,
                "finite": bool(torch.isfinite(value).all().item()),
                "rel_l2": float(
                    torch.linalg.vector_norm(difference)
                    / torch.linalg.vector_norm(expected_f)
                ),
                "max_abs": float(difference.abs().max()),
            }
        )
    return rows


def common_payload(args: argparse.Namespace) -> dict:
    payload = dict(args.manifest)
    payload["runtime"] = args.runtime
    return payload


def error_metrics_pass(rows: list[dict]) -> bool:
    return all(
        row["finite"]
        and row["rel_l2"] <= ERROR_LIMITS[row["name"]]["rel_l2"]
        and row["max_abs"] <= ERROR_LIMITS[row["name"]]["max_abs"]
        for row in rows
    )


def run_onecall(args: argparse.Namespace, extension) -> dict:
    data = make_inputs(args.seed, args.seqlen, args.heads)
    parent_fn = getattr(extension, args.parent)
    child_fn = getattr(extension, args.child)
    reference = call_cute(data)
    torch.cuda.synchronize()
    parent = call_tk(parent_fn, data, args.seqlen)
    torch.cuda.synchronize()
    child = call_tk(child_fn, data, args.seqlen)
    torch.cuda.synchronize()
    parent_rows = metrics(parent, reference)
    child_rows = metrics(child, reference)
    comparison_rows = metrics(child, parent)
    child_error_pass = error_metrics_pass(child_rows)
    return {
        **common_payload(args),
        "complete": True,
        "state": "finished",
        "gate_pass": child_error_pass,
        "gate_checks": {"child_error_pass": child_error_pass},
        "parent_vs_cute": parent_rows,
        "child_vs_cute": child_rows,
        "child_vs_parent": comparison_rows,
    }


def run_stress(args: argparse.Namespace, extension) -> dict:
    data = make_inputs(args.seed, args.seqlen, args.heads)
    child_fn = getattr(extension, args.child)
    first_raw = call_tk(child_fn, data, args.seqlen)
    torch.cuda.synchronize()
    first = tuple(value.clone() for value in first_raw)
    torch.cuda.synchronize()
    finite = [bool(torch.isfinite(value).all().item()) for value in first]
    max_drift = [0.0, 0.0, 0.0]
    first_drift = [None, None, None]
    progress = {
        **common_payload(args),
        "calls_requested": args.calls,
        "calls_completed": 1,
        "finite_dq_dk_dv": finite,
        "max_repeat_drift_dq_dk_dv": max_drift,
        "first_drift_dq_dk_dv": first_drift,
    }
    write_json(args.output, progress)

    for call_idx in range(1, args.calls):
        current = call_tk(child_fn, data, args.seqlen)
        torch.cuda.synchronize()
        for index, (value, baseline) in enumerate(zip(current, first)):
            finite[index] &= bool(torch.isfinite(value).all().item())
            difference = (value.float() - baseline.float()).abs()
            drift = float(difference.max())
            max_drift[index] = max(max_drift[index], drift)
            if drift > 1.0e-6 and first_drift[index] is None:
                flat_index = int(difference.argmax())
                remaining = flat_index
                coordinates = []
                for dimension in reversed(value.shape):
                    coordinates.append(remaining % dimension)
                    remaining //= dimension
                coordinates.reverse()
                first_drift[index] = {
                    "call": call_idx,
                    "max_abs": drift,
                    "index": coordinates,
                    "first": float(baseline.flatten()[flat_index]),
                    "current": float(value.flatten()[flat_index]),
                    "changed_gt_1e6": int((difference > 1.0e-6).sum()),
                }
        torch.cuda.synchronize()
        if (call_idx + 1) % args.progress_every == 0:
            write_json(
                args.output,
                {
                    **progress,
                    "calls_completed": call_idx + 1,
                    "finite_dq_dk_dv": finite,
                    "max_repeat_drift_dq_dk_dv": max_drift,
                    "first_drift_dq_dk_dv": first_drift,
                },
            )

    reference = call_cute(data)
    torch.cuda.synchronize()
    reference_rows = metrics(first, reference)
    drift_pass = all(
        drift <= DRIFT_LIMITS[name]
        for name, drift in zip(("dq", "dk", "dv"), max_drift)
    )
    finite_pass = all(finite)
    reference_error_pass = error_metrics_pass(reference_rows)
    return {
        **progress,
        "complete": True,
        "state": "finished",
        "gate_pass": finite_pass and drift_pass and reference_error_pass,
        "gate_checks": {
            "finite_pass": finite_pass,
            "repeat_drift_pass": drift_pass,
            "reference_error_pass": reference_error_pass,
        },
        "calls_completed": args.calls,
        "finite_dq_dk_dv": finite,
        "max_repeat_drift_dq_dk_dv": max_drift,
        "first_drift_dq_dk_dv": first_drift,
        "first_vs_cute": reference_rows,
    }


def summarize_times(values: list[float]) -> dict:
    return {
        "median_us": statistics.median(values),
        "min_us": min(values),
        "max_us": max(values),
    }


def run_timing(args: argparse.Namespace, extension) -> dict:
    data = make_inputs(args.seed, args.seqlen, args.heads)
    parent_fn = getattr(extension, args.parent)
    child_fn = getattr(extension, args.child)
    routes = {
        "parent": lambda: call_tk(parent_fn, data, args.seqlen),
        "child": lambda: call_tk(child_fn, data, args.seqlen),
        "cute": lambda: call_cute(data),
    }
    for _ in range(args.warmups):
        for fn in routes.values():
            result = fn()
            del result
    torch.cuda.synchronize()

    values = {name: [] for name in routes}
    orders = list(itertools.permutations(routes))
    raw_samples = []
    for sample_index in range(args.samples):
        order = list(orders[sample_index % len(orders)])
        sample = {
            "sample_index": sample_index,
            "execution_order": order,
            "measurements": [],
        }
        for name in order:
            torch.cuda.synchronize()
            started = time.perf_counter_ns()
            result = routes[name]()
            torch.cuda.synchronize()
            elapsed_us = (time.perf_counter_ns() - started) / 1000.0
            values[name].append(elapsed_us)
            sample["measurements"].append(
                {"route": name, "elapsed_us": elapsed_us}
            )
            del result
        raw_samples.append(sample)
        if (sample_index + 1) % args.progress_every == 0:
            write_json(
                args.output,
                {
                    **common_payload(args),
                    "samples_requested": args.samples,
                    "samples_completed": sample_index + 1,
                    "partial": {
                        name: summarize_times(times) for name, times in values.items()
                    },
                    "raw_samples": raw_samples,
                },
            )

    summary = {name: summarize_times(times) for name, times in values.items()}
    child_over_parent = (
        summary["child"]["median_us"] / summary["parent"]["median_us"]
    )
    child_over_cute = summary["child"]["median_us"] / summary["cute"]["median_us"]
    retention_pass = child_over_parent < 1.0
    objective_beats_cute = child_over_cute < 1.0
    return {
        **common_payload(args),
        "complete": True,
        "state": "finished",
        "gate_pass": retention_pass,
        "gate_checks": {
            "retention_pass": retention_pass,
            "objective_beats_cute": objective_beats_cute,
        },
        "retention_pass": retention_pass,
        "objective_beats_cute": objective_beats_cute,
        "timing_protocol": {
            "clock": "time.perf_counter_ns",
            "device_sync_immediately_before_call": True,
            "device_sync_immediately_after_call": True,
            "progress_write_after_complete_three_route_sample": True,
            "route_order": "six permutations selected by sample_index modulo six",
        },
        "samples": args.samples,
        "warmups": args.warmups,
        **summary,
        "child_over_parent": child_over_parent,
        "child_over_cute": child_over_cute,
        "raw_samples": raw_samples,
    }


def run_compare(args: argparse.Namespace, extension) -> dict:
    data = make_inputs(args.seed, args.seqlen, args.heads)
    child_fn = getattr(extension, args.child)

    reference = call_cute(data)
    torch.cuda.synchronize()
    child_result = call_tk(child_fn, data, args.seqlen)
    torch.cuda.synchronize()
    child_rows = metrics(child_result, reference)
    child_error_pass = error_metrics_pass(child_rows)
    del child_result, reference

    routes = {
        "tk": lambda: call_tk(child_fn, data, args.seqlen),
        "cute": lambda: call_cute(data),
    }
    for _ in range(args.warmups):
        for fn in routes.values():
            result = fn()
            del result
    torch.cuda.synchronize()

    values = {name: [] for name in routes}
    orders = list(itertools.permutations(routes))
    raw_samples = []
    for sample_index in range(args.samples):
        order = list(orders[sample_index % len(orders)])
        sample = {
            "sample_index": sample_index,
            "execution_order": order,
            "measurements": [],
        }
        for name in order:
            torch.cuda.synchronize()
            started = time.perf_counter_ns()
            result = routes[name]()
            torch.cuda.synchronize()
            elapsed_us = (time.perf_counter_ns() - started) / 1000.0
            values[name].append(elapsed_us)
            sample["measurements"].append(
                {"route": name, "elapsed_us": elapsed_us}
            )
            del result
        raw_samples.append(sample)
        if (sample_index + 1) % args.progress_every == 0:
            write_json(
                args.output,
                {
                    **common_payload(args),
                    "samples_requested": args.samples,
                    "samples_completed": sample_index + 1,
                    "child_vs_cute": child_rows,
                    "partial": {
                        name: summarize_times(times)
                        for name, times in values.items()
                    },
                    "raw_samples": raw_samples,
                },
            )

    summary = {name: summarize_times(times) for name, times in values.items()}
    tk_over_cute = summary["tk"]["median_us"] / summary["cute"]["median_us"]
    objective_beats_cute = tk_over_cute < 1.0
    return {
        **common_payload(args),
        "complete": True,
        "state": "finished",
        "gate_pass": child_error_pass,
        "gate_checks": {
            "child_error_pass": child_error_pass,
            "objective_beats_cute": objective_beats_cute,
        },
        "objective_beats_cute": objective_beats_cute,
        "timing_protocol": {
            "clock": "time.perf_counter_ns",
            "device_sync_immediately_before_call": True,
            "device_sync_immediately_after_call": True,
            "progress_write_after_complete_two_route_sample": True,
            "route_order": "two permutations selected by sample parity",
        },
        "samples": args.samples,
        "warmups": args.warmups,
        "child_vs_cute": child_rows,
        **summary,
        "tk_over_cute": tk_over_cute,
        "raw_samples": raw_samples,
    }


def run_profile(args: argparse.Namespace, extension) -> dict:
    data = make_inputs(args.seed, args.seqlen, args.heads)
    parent_fn = getattr(extension, args.parent)
    child_fn = getattr(extension, args.child)
    routes = {
        "parent": lambda: call_tk(parent_fn, data, args.seqlen),
        "child": lambda: call_tk(child_fn, data, args.seqlen),
        "cute": lambda: call_cute(data),
    }
    fn = routes[args.route]
    for _ in range(args.warmups):
        result = fn()
        del result
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    result = fn()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    finite = [bool(torch.isfinite(value).all()) for value in result]
    return {
        **common_payload(args),
        "complete": True,
        "state": "finished",
        "gate_pass": all(finite),
        "route": args.route,
        "finite_dq_dk_dv": finite,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=("onecall", "stress", "timing", "compare", "profile")
    )
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--child", required=True)
    parser.add_argument("--seqlen", type=int, default=65536)
    parser.add_argument("--heads", type=int, default=128)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--calls", type=int, default=50)
    parser.add_argument("--samples", type=int, default=201)
    parser.add_argument("--warmups", type=int, default=12)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--route", choices=("parent", "child", "cute"), default="child"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in (
        "seqlen",
        "heads",
        "calls",
        "samples",
        "warmups",
        "progress_every",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {args.output}")
    args.manifest = build_manifest(args)
    args.runtime = None
    write_json(args.output, args.manifest)

    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
            raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 1")
        load_runtime()
        torch.cuda.set_device(0)
        args.runtime = {
            "torch_version": str(torch.__version__),
            "torch_cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(0),
            "device_capability": list(torch.cuda.get_device_capability(0)),
            **runtime_module_provenance(),
        }
        extension = load_extension(args.extension)
        runners = {
            "onecall": run_onecall,
            "stress": run_stress,
            "timing": run_timing,
            "compare": run_compare,
            "profile": run_profile,
        }
        result = runners[args.mode](args, extension)
        write_json(args.output, result)
        print(json.dumps(result, sort_keys=True), flush=True)
    except BaseException as error:
        failure = {
            **common_payload(args),
            "complete": False,
            "gate_pass": False,
            "state": "failed",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        write_json(args.output, failure)
        raise
    if not result.get("gate_pass", False):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
