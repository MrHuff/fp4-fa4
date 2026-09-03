from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
BASE_HARNESS = REPO / "results" / "tk_bf16_bwd_s65536h128_harness.py"
EXPECTED_EXTENSION_SHA256 = (
    "46b241d0557e86c97eb411b93e71ffafa74ecaf0b798302f2532491a856f3bf9"
)
EXPECTED_BASE_HARNESS_SHA256 = (
    "63e19fde901d714168c663ecee7f11d65081d903541d3871c1c8b52f79431db3"
)
ROUTE = (
    "b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv_"
    "pipelined_tmem_dq_double_buffer_split2_dq_first_internal"
)
SCOPED_PATHS = (
    "tk_fa4/b300_bwd_cute16_candidate.cuh",
    "tk_fa4/b300_bwd_cute16_kernel_candidate.cuh",
    "tk_fa4/tk_fa4.cu",
    "results/tk_bf16_bwd_split2_bf16_materialize_harness.py",
    "results/tk_bf16_bwd_broad_sweep_20260711.md",
)
ERROR_LIMITS = {
    "dq": {"rel_l2": 0.01, "max_abs": 0.05},
    "dk": {"rel_l2": 0.01, "max_abs": 0.05},
    "dv": {"rel_l2": 0.001, "max_abs": 0.02},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*arguments: str, binary: bool = False):
    return subprocess.run(
        ("git", *arguments),
        cwd=REPO,
        check=True,
        capture_output=True,
        text=not binary,
    ).stdout


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def environment_truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() not in {
        "",
        "0",
        "false",
        "no",
        "off",
    }


def timing_environment() -> dict:
    values = {
        name: os.environ.get(name)
        for name in (
            "TK_FA4_SPLIT_TIMING",
            "TK_FA4_CLUSTERED_DQ_TIMING",
        )
    }
    return {
        "values": values,
        "truthy_variables": [
            name for name, value in values.items() if environment_truthy(value)
        ],
    }


def safe_file_record(path: Path) -> dict:
    try:
        return {
            "path": str(path.resolve()),
            "exists": path.is_file(),
            "size_bytes": path.stat().st_size if path.is_file() else None,
            "sha256": sha256_file(path) if path.is_file() else None,
        }
    except BaseException as error:
        return {
            "path": str(path),
            "exists": None,
            "size_bytes": None,
            "sha256": None,
            "inspection_error": f"{type(error).__name__}: {error}",
        }


def fallback_manifest(args: argparse.Namespace) -> dict:
    return {
        "schema_version": 2,
        "complete": False,
        "state": "initialization_error",
        "gate_pass": False,
        "gate_scope": "initialization",
        "correctness_pass": None,
        "full_backward_correctness_pass": None,
        "retention_evaluated": False,
        "retention_pass": None,
        "objective_evaluated": False,
        "objective_beats_cute": None,
        "overall_contract_scope": None,
        "overall_contract_pass": None,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "cwd": str(Path.cwd().resolve()),
        "python_executable": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "mode": args.mode,
        "seqlen": args.seqlen,
        "heads": args.heads,
        "seed": args.seed,
        "route": ROUTE,
        "requested_output": str(args.output),
        "extension": safe_file_record(args.extension),
        "harness": safe_file_record(Path(__file__)),
        "base_harness": safe_file_record(BASE_HARNESS),
        "expected_extension_sha256": args.expected_extension_sha256,
        "expected_base_harness_sha256": EXPECTED_BASE_HARNESS_SHA256,
        "timing_environment": timing_environment(),
    }


def initial_manifest(args: argparse.Namespace) -> dict:
    extension = args.extension.resolve()
    harness = Path(__file__).resolve()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    if not extension.is_file():
        raise FileNotFoundError(f"extension does not exist: {extension}")
    extension_sha = sha256_file(extension)
    if extension_sha != args.expected_extension_sha256:
        raise RuntimeError(
            "extension SHA mismatch: "
            f"expected {args.expected_extension_sha256}, got {extension_sha}"
        )
    base_harness_sha = sha256_file(BASE_HARNESS)
    if base_harness_sha != EXPECTED_BASE_HARNESS_SHA256:
        raise RuntimeError(
            "base harness SHA mismatch: "
            f"expected {EXPECTED_BASE_HARNESS_SHA256}, got {base_harness_sha}"
        )
    timing_env = timing_environment()
    if timing_env["truthy_variables"]:
        raise RuntimeError(
            "timing instrumentation environment must be disabled: "
            + ", ".join(timing_env["truthy_variables"])
        )
    scoped_diff = git_output("diff", "--binary", "--", *SCOPED_PATHS, binary=True)
    diff_check = subprocess.run(
        ("git", "diff", "--check", "--", *SCOPED_PATHS),
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    return {
        "schema_version": 2,
        "complete": False,
        "state": "initialized",
        "gate_pass": False,
        "gate_scope": "not_evaluated",
        "correctness_pass": None,
        "full_backward_correctness_pass": None,
        "retention_evaluated": False,
        "retention_pass": None,
        "objective_evaluated": False,
        "objective_beats_cute": None,
        "overall_contract_scope": None,
        "overall_contract_pass": None,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "cwd": str(Path.cwd().resolve()),
        "python_executable": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "mode": args.mode,
        "seqlen": args.seqlen,
        "heads": args.heads,
        "seed": args.seed,
        "route": ROUTE,
        "extension": str(extension),
        "extension_sha256": extension_sha,
        "extension_size_bytes": extension.stat().st_size,
        "harness": str(harness),
        "harness_sha256": sha256_file(harness),
        "base_harness": str(BASE_HARNESS),
        "base_harness_sha256": base_harness_sha,
        "expected_base_harness_sha256": EXPECTED_BASE_HARNESS_SHA256,
        "timing_environment": timing_env,
        "git_head": git_output("rev-parse", "HEAD").strip(),
        "git_head_tree": git_output("rev-parse", "HEAD^{tree}").strip(),
        "scoped_paths": list(SCOPED_PATHS),
        "scoped_status": git_output(
            "status", "--short", "--", *SCOPED_PATHS
        ).splitlines(),
        "scoped_diff_sha256": hashlib.sha256(scoped_diff).hexdigest(),
        "scoped_diff_size_bytes": len(scoped_diff),
        "scoped_diff_check_pass": diff_check.returncode == 0,
        "scoped_diff_check_output": diff_check.stdout + diff_check.stderr,
        "thresholds": {"error_limits": ERROR_LIMITS},
        "materialization_contract": {
            "dq": "unchanged FP32 split-2 output",
            "dk": "materialized exactly once from FP32 to BF16",
            "dv": "materialized exactly once from FP32 to BF16",
            "primary_timing_intermediate_sync": False,
            "ordering": (
                "frontier event joins the caller stream; FP32 merge and Python "
                "casts execute on that same caller stream"
            ),
        },
    }


def load_base_harness():
    spec = importlib.util.spec_from_file_location(
        "tk_bf16_bwd_schema_v2_base", BASE_HARNESS
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load base harness: {BASE_HARNESS}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.load_runtime()
    return module


def tensor_record(tensor) -> dict:
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "contiguous": bool(tensor.is_contiguous()),
        "device": str(tensor.device),
        "finite": bool(tensor.isfinite().all().item()),
    }


def tensor_gate(tensor, shape: tuple[int, ...], dtype) -> bool:
    return (
        tuple(tensor.shape) == shape
        and tensor.dtype == dtype
        and tensor.is_contiguous()
        and tensor.device.type == "cuda"
        and bool(tensor.isfinite().all().item())
    )


def rows_pass(rows: list[dict], names: set[str]) -> bool:
    selected = [row for row in rows if row.get("name") in names]
    if len(selected) != len(names):
        return False
    if {row.get("name") for row in selected} != names:
        return False
    return all(
        row["finite"]
        and row["rel_l2"] <= ERROR_LIMITS[row["name"]]["rel_l2"]
        and row["max_abs"] <= ERROR_LIMITS[row["name"]]["max_abs"]
        for row in selected
    )


def summarize(values: list[float]) -> dict:
    return {
        "median_us": statistics.median(values),
        "min_us": min(values),
        "max_us": max(values),
    }


def screen(base, extension, args: argparse.Namespace):
    torch = base.torch
    data = base.make_inputs(args.seed, args.seqlen, args.heads)
    fn = getattr(extension, ROUTE)

    reference = base.call_cute(data)
    torch.cuda.synchronize()
    if not isinstance(reference, (tuple, list)) or len(reference) != 3:
        raise RuntimeError(
            f"CuTe backward must return exactly three outputs, got {type(reference)} "
            f"with arity {len(reference) if hasattr(reference, '__len__') else None}"
        )
    raw = base.call_tk(fn, data, args.seqlen)
    if not isinstance(raw, (tuple, list)) or len(raw) != 3:
        raise RuntimeError(
            f"TK backward must return exactly three outputs, got {type(raw)} "
            f"with arity {len(raw) if hasattr(raw, '__len__') else None}"
        )

    cast_alloc = (
        raw[0],
        raw[1].to(dtype=torch.bfloat16),
        raw[2].to(dtype=torch.bfloat16),
    )
    cast_dk = torch.empty_like(data[1])
    cast_dv = torch.empty_like(data[2])
    cast_dk.copy_(raw[1])
    cast_dv.copy_(raw[2])
    cast_preallocated = (raw[0], cast_dk, cast_dv)
    torch.cuda.synchronize()

    q, k, v, dout, out, lse_raw, lse_tk = data
    shapes = (tuple(q.shape), tuple(k.shape), tuple(v.shape))
    layout_gates = {
        "inputs_bf16_bshd_contiguous": (
            tensor_gate(q, shapes[0], torch.bfloat16)
            and tensor_gate(k, shapes[1], torch.bfloat16)
            and tensor_gate(v, shapes[2], torch.bfloat16)
            and tensor_gate(dout, shapes[2], torch.bfloat16)
        ),
        "forward_out_bf16_contiguous": tensor_gate(out, shapes[2], torch.bfloat16),
        "lse_raw_bhs_fp32_cuda_finite_contiguous": tensor_gate(
            lse_raw,
            (1, args.heads, args.seqlen),
            torch.float32,
        ),
        "lse_tk_bsh_fp32_cuda_finite_contiguous": tensor_gate(
            lse_tk,
            (1, args.seqlen, args.heads),
            torch.float32,
        ),
        "raw_fp32_shapes_contiguous": all(
            tensor_gate(tensor, shape, torch.float32)
            for tensor, shape in zip(raw, shapes)
        ),
        "cast_alloc_dq_fp32_dkdv_bf16_shapes_contiguous": (
            tensor_gate(cast_alloc[0], shapes[0], torch.float32)
            and tensor_gate(cast_alloc[1], shapes[1], torch.bfloat16)
            and tensor_gate(cast_alloc[2], shapes[2], torch.bfloat16)
        ),
        "cast_preallocated_dq_fp32_dkdv_bf16_shapes_contiguous": (
            tensor_gate(cast_preallocated[0], shapes[0], torch.float32)
            and tensor_gate(cast_preallocated[1], shapes[1], torch.bfloat16)
            and tensor_gate(cast_preallocated[2], shapes[2], torch.bfloat16)
        ),
        "cute_bf16_shapes_contiguous": (
            tensor_gate(reference[0], shapes[0], torch.bfloat16)
            and tensor_gate(reference[1], shapes[1], torch.bfloat16)
            and tensor_gate(reference[2], shapes[2], torch.bfloat16)
        ),
        "allocated_matches_preallocated_bitwise": (
            torch.equal(cast_alloc[1], cast_preallocated[1])
            and torch.equal(cast_alloc[2], cast_preallocated[2])
        ),
    }

    raw_rows = base.metrics(raw, reference)
    cast_alloc_rows = base.metrics(cast_alloc, reference)
    cast_preallocated_rows = base.metrics(cast_preallocated, reference)
    cast_equivalence_rows = base.metrics(cast_alloc, cast_preallocated)
    raw_dkdv_pass = rows_pass(raw_rows, {"dk", "dv"})
    cast_dkdv_pass = rows_pass(cast_alloc_rows, {"dk", "dv"})
    full_backward_pass = rows_pass(cast_alloc_rows, {"dq", "dk", "dv"})
    correctness_pass = all(layout_gates.values()) and cast_dkdv_pass

    payload = {
        "tensor_contracts": {
            "q": tensor_record(q),
            "k": tensor_record(k),
            "v": tensor_record(v),
            "dout": tensor_record(dout),
            "out": tensor_record(out),
            "lse_raw": tensor_record(lse_raw),
            "lse_tk": tensor_record(lse_tk),
            "raw": {
                name: tensor_record(value)
                for name, value in zip(("dq", "dk", "dv"), raw)
            },
            "cast_alloc": {
                name: tensor_record(value)
                for name, value in zip(("dq", "dk", "dv"), cast_alloc)
            },
            "cast_preallocated": {
                name: tensor_record(value)
                for name, value in zip(
                    ("dq", "dk", "dv"), cast_preallocated
                )
            },
            "cute": {
                name: tensor_record(value)
                for name, value in zip(("dq", "dk", "dv"), reference)
            },
        },
        "layout_gates": layout_gates,
        "raw_vs_cute": raw_rows,
        "cast_alloc_vs_cute": cast_alloc_rows,
        "cast_preallocated_vs_cute": cast_preallocated_rows,
        "cast_alloc_vs_preallocated": cast_equivalence_rows,
        "raw_dkdv_correctness_pass": raw_dkdv_pass,
        "cast_dkdv_correctness_pass": cast_dkdv_pass,
        "full_backward_correctness_pass": full_backward_pass,
        "dq_contract_invalid": not rows_pass(cast_alloc_rows, {"dq"}),
        "correctness_pass": correctness_pass,
    }
    del reference, raw, cast_alloc, cast_preallocated
    return payload, data, fn


def time_routes(base, routes: dict, samples: int, warmups: int):
    torch = base.torch
    for _ in range(warmups):
        for fn in routes.values():
            torch.cuda.synchronize()
            result = fn()
            torch.cuda.synchronize()
            del result

    values = {name: [] for name in routes}
    raw_samples = []
    orders = list(itertools.permutations(routes))
    for sample_index in range(samples):
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
    return (
        {name: summarize(samples_) for name, samples_ in values.items()},
        raw_samples,
    )


def run_screen(base, extension, args: argparse.Namespace) -> dict:
    screened, _, _ = screen(base, extension, args)
    correctness_pass = screened["correctness_pass"]
    return {
        **args.manifest,
        **screened,
        "runtime": base.runtime_module_provenance(),
        "complete": True,
        "state": "finished",
        "gate_pass": correctness_pass,
        "gate_scope": "dk_dv_materialization_correctness",
        "correctness_pass": correctness_pass,
        "objective_evaluated": False,
        "objective_beats_cute": None,
        "overall_contract_scope": None,
        "overall_contract_pass": None,
    }


def run_timing(base, extension, args: argparse.Namespace) -> dict:
    torch = base.torch
    screened, data, fn = screen(base, extension, args)
    if not screened["correctness_pass"]:
        return {
            **args.manifest,
            **screened,
            "runtime": base.runtime_module_provenance(),
            "complete": True,
            "state": "finished",
            "gate_pass": False,
            "gate_scope": "dk_dv_materialization_correctness",
            "correctness_pass": False,
            "objective_evaluated": False,
            "objective_beats_cute": None,
            "overall_contract_scope": None,
            "overall_contract_pass": False,
            "timing_skipped": True,
        }

    def raw_route():
        return base.call_tk(fn, data, args.seqlen)

    def cast_alloc_route():
        raw = base.call_tk(fn, data, args.seqlen)
        return (
            raw[0],
            raw[1].to(dtype=torch.bfloat16),
            raw[2].to(dtype=torch.bfloat16),
        )

    routes = {
        "raw_p": raw_route,
        "cast_alloc_p": cast_alloc_route,
        "cute": lambda: base.call_cute(data),
    }
    summaries, raw_samples = time_routes(
        base, routes, args.samples, args.warmups
    )

    completed_raw = raw_route()
    torch.cuda.synchronize()
    overhead_dk = torch.empty_like(data[1])
    overhead_dv = torch.empty_like(data[2])

    def cast_alloc_only():
        return (
            completed_raw[1].to(dtype=torch.bfloat16),
            completed_raw[2].to(dtype=torch.bfloat16),
        )

    def cast_preallocated_only():
        overhead_dk.copy_(completed_raw[1])
        overhead_dv.copy_(completed_raw[2])
        return overhead_dk, overhead_dv

    overhead_summaries, overhead_samples = time_routes(
        base,
        {
            "cast_alloc_only": cast_alloc_only,
            "cast_preallocated_only": cast_preallocated_only,
        },
        args.samples,
        args.warmups,
    )

    raw_over_cute = (
        summaries["raw_p"]["median_us"] / summaries["cute"]["median_us"]
    )
    cast_alloc_over_cute = (
        summaries["cast_alloc_p"]["median_us"]
        / summaries["cute"]["median_us"]
    )
    performance_only_beats_cute = cast_alloc_over_cute < 1.0
    full_backward_pass = screened["full_backward_correctness_pass"]
    objective_beats_cute = full_backward_pass and performance_only_beats_cute

    return {
        **args.manifest,
        **screened,
        "runtime": base.runtime_module_provenance(),
        "complete": True,
        "state": "finished",
        "gate_pass": True,
        "gate_scope": "dk_dv_materialization_correctness_and_timing_complete",
        "correctness_pass": True,
        "objective_evaluated": True,
        "objective_beats_cute": objective_beats_cute,
        "overall_contract_scope": (
            "full_backward_correctness_and_allocated_materialized_speed"
        ),
        "overall_contract_pass": objective_beats_cute,
        "performance_only_beats_cute": performance_only_beats_cute,
        "timing_protocol": {
            "clock": "time.perf_counter_ns",
            "device_sync_immediately_before_route": True,
            "device_sync_immediately_after_route": True,
            "intermediate_sync_before_primary_casts": False,
            "primary_routes": ["raw_p", "cast_alloc_p", "cute"],
            "route_order": "six permutations, cycled by sample index",
            "progress_io_inside_intervals": False,
        },
        "samples": args.samples,
        "warmups": args.warmups,
        "route_timings": summaries,
        "cast_only_timings": overhead_summaries,
        "ratios": {
            "raw_p_over_cute": raw_over_cute,
            "cast_alloc_p_over_cute": cast_alloc_over_cute,
            "cast_alloc_p_over_raw_p": (
                summaries["cast_alloc_p"]["median_us"]
                / summaries["raw_p"]["median_us"]
            ),
        },
        "raw_samples": raw_samples,
        "cast_only_raw_samples": overhead_samples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("screen", "timing"))
    parser.add_argument("--extension", required=True, type=Path)
    parser.add_argument(
        "--expected-extension-sha256",
        default=EXPECTED_EXTENSION_SHA256,
    )
    parser.add_argument("--seqlen", required=True, type=int)
    parser.add_argument("--heads", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--samples", type=int, default=201)
    parser.add_argument("--warmups", type=int, default=12)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.seqlen <= 0 or args.seqlen % 512 != 0:
        parser.error("seqlen must be positive and divisible by 512")
    if args.heads <= 0 or args.seed <= 0:
        parser.error("heads and seed must be positive")
    if args.samples <= 0 or args.warmups < 0:
        parser.error("samples must be positive and warmups non-negative")
    return args


def main() -> None:
    args = parse_args()
    args.output = args.output.resolve()
    try:
        bootstrap = fallback_manifest(args)
    except BaseException as bootstrap_error:
        bootstrap = {
            "schema_version": 2,
            "complete": False,
            "state": "initialization_error",
            "gate_pass": False,
            "gate_scope": "initialization",
            "command": sys.argv,
            "requested_output": str(args.output),
            "bootstrap_error": (
                f"{type(bootstrap_error).__name__}: {bootstrap_error}"
            ),
        }
    try:
        manifest = initial_manifest(args)
    except BaseException as initialization_error:
        failed = {
            **bootstrap,
            "complete": False,
            "state": "initialization_error",
            "error_type": type(initialization_error).__name__,
            "error": str(initialization_error),
            "traceback": traceback.format_exc(),
        }
        error_output = args.output
        if error_output.exists():
            error_output = error_output.with_name(
                error_output.name + ".initialization_error.json"
            )
        write_json(error_output, failed)
        raise
    args.manifest = manifest
    write_json(args.output, manifest)
    try:
        base = load_base_harness()
        extension = base.load_extension(args.extension.resolve())
        if not hasattr(extension, ROUTE):
            raise AttributeError(f"extension does not expose route: {ROUTE}")
        result = (
            run_screen(base, extension, args)
            if args.mode == "screen"
            else run_timing(base, extension, args)
        )
        write_json(args.output, result)
        print(
            json.dumps(
                {
                    key: result.get(key)
                    for key in (
                        "complete",
                        "gate_pass",
                        "correctness_pass",
                        "full_backward_correctness_pass",
                        "dq_contract_invalid",
                        "performance_only_beats_cute",
                        "objective_beats_cute",
                        "ratios",
                    )
                },
                sort_keys=True,
            )
        )
        if not result["gate_pass"]:
            raise SystemExit(2)
    except BaseException as error:
        if isinstance(error, SystemExit):
            raise
        failed = {
            **manifest,
            "complete": False,
            "state": "error",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        write_json(args.output, failed)
        raise


if __name__ == "__main__":
    main()
