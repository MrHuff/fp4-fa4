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
REVIEWED_MATERIALIZATION_HARNESS = (
    REPO / "results" / "tk_bf16_bwd_split2_bf16_materialize_harness.py"
)
EXPECTED_BASE_HARNESS_SHA256 = (
    "63e19fde901d714168c663ecee7f11d65081d903541d3871c1c8b52f79431db3"
)
EXPECTED_REVIEWED_HARNESS_SHA256 = (
    "62462b1a0840c344f046643b84c008a969bbec3c863a964b72e778d1f571534a"
)
EXPECTED_EXTENSION_SHA256 = (
    "55e96426d1fe039f0028ea88a39f524c395125e932c74b71d1851d3a84b69efa"
)
PARENT_ROUTE = (
    "b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv_"
    "pipelined_tmem_dq_double_buffer_split2_dq_first_internal"
)
CHILD_ROUTE = PARENT_ROUTE.replace(
    "split2_dq_first_internal", "split2_dq_first_bf16_internal"
)
SCOPED_PATHS = (
    "tk_fa4/b300_bwd_cute16_candidate.cuh",
    "tk_fa4/b300_bwd_cute16_kernel_candidate.cuh",
    "tk_fa4/tk_fa4.cu",
    "results/tk_bf16_bwd_split2_bf16_fused_store_harness.py",
    "results/tk_bf16_bwd_broad_sweep_20260711.md",
)
ERROR_LIMITS = {
    "dq": {"rel_l2": 0.01, "max_abs": 0.05},
    "dk": {"rel_l2": 0.01, "max_abs": 0.05},
    "dv": {"rel_l2": 0.001, "max_abs": 0.02},
}
DRIFT_LIMITS = {"dq": 2.0e-7, "dk": 0.0, "dv": 0.0}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def git_output(*arguments: str, binary: bool = False):
    return subprocess.run(
        ("git", *arguments),
        cwd=REPO,
        check=True,
        capture_output=True,
        text=not binary,
    ).stdout


def timing_environment() -> dict:
    values = {
        name: os.environ.get(name)
        for name in ("TK_FA4_SPLIT_TIMING", "TK_FA4_CLUSTERED_DQ_TIMING")
    }
    truthy = {
        "", "0", "false", "no", "off"
    }
    return {
        "values": values,
        "truthy_variables": [
            name
            for name, value in values.items()
            if value is not None and value.strip().lower() not in truthy
        ],
    }


def safe_file_record(path: Path) -> dict:
    try:
        exists = path.is_file()
        return {
            "path": str(path.resolve()),
            "exists": exists,
            "size_bytes": path.stat().st_size if exists else None,
            "sha256": sha256_file(path) if exists else None,
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
        "extension": safe_file_record(args.extension),
        "harness": safe_file_record(Path(__file__)),
        "base_harness": safe_file_record(BASE_HARNESS),
        "reviewed_materialization_harness": safe_file_record(
            REVIEWED_MATERIALIZATION_HARNESS
        ),
        "timing_environment": timing_environment(),
    }


def initial_manifest(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    extension = args.extension.resolve()
    if not extension.is_file():
        raise FileNotFoundError(f"extension does not exist: {extension}")
    extension_sha = sha256_file(extension)
    if extension_sha != EXPECTED_EXTENSION_SHA256:
        raise RuntimeError(
            f"extension SHA mismatch: expected {EXPECTED_EXTENSION_SHA256}, "
            f"got {extension_sha}"
        )
    base_sha = sha256_file(BASE_HARNESS)
    if base_sha != EXPECTED_BASE_HARNESS_SHA256:
        raise RuntimeError(
            f"base harness SHA mismatch: expected {EXPECTED_BASE_HARNESS_SHA256}, "
            f"got {base_sha}"
        )
    reviewed_harness_sha = sha256_file(REVIEWED_MATERIALIZATION_HARNESS)
    if reviewed_harness_sha != EXPECTED_REVIEWED_HARNESS_SHA256:
        raise RuntimeError(
            "reviewed materialization harness SHA mismatch: expected "
            f"{EXPECTED_REVIEWED_HARNESS_SHA256}, got {reviewed_harness_sha}"
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
    harness = Path(__file__).resolve()
    return {
        **fallback_manifest(args),
        "state": "initialized",
        "gate_scope": "not_evaluated",
        "extension": str(extension),
        "extension_sha256": extension_sha,
        "extension_size_bytes": extension.stat().st_size,
        "harness": str(harness),
        "harness_sha256": sha256_file(harness),
        "base_harness": str(BASE_HARNESS),
        "base_harness_sha256": base_sha,
        "reviewed_materialization_harness": str(
            REVIEWED_MATERIALIZATION_HARNESS
        ),
        "reviewed_materialization_harness_sha256": reviewed_harness_sha,
        "parent_route": PARENT_ROUTE,
        "child_route": CHILD_ROUTE,
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
        "thresholds": {
            "error_limits": ERROR_LIMITS,
            "drift_limits": DRIFT_LIMITS,
            "timing_child_over_cute": 1.0,
        },
    }


def load_base_harness():
    spec = importlib.util.spec_from_file_location(
        "tk_bf16_bwd_fused_store_reviewed_base",
        REVIEWED_MATERIALIZATION_HARNESS,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            "cannot load reviewed materialization harness: "
            f"{REVIEWED_MATERIALIZATION_HARNESS}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_base_harness()


def rows_pass(rows: list[dict]) -> bool:
    names = {"dq", "dk", "dv"}
    return (
        len(rows) == 3
        and {row.get("name") for row in rows} == names
        and all(
            row["finite"]
            and row["rel_l2"] <= ERROR_LIMITS[row["name"]]["rel_l2"]
            and row["max_abs"] <= ERROR_LIMITS[row["name"]]["max_abs"]
            for row in rows
        )
    )


def output_contract(base, outputs, data, *, child: bool) -> dict:
    torch = base.torch
    q, k, v = data[:3]
    expected_shapes = (tuple(q.shape), tuple(k.shape), tuple(v.shape))
    expected_dtypes = (
        (torch.float32, torch.bfloat16, torch.bfloat16)
        if child
        else (torch.float32, torch.float32, torch.float32)
    )
    exact_arity = isinstance(outputs, (tuple, list)) and len(outputs) == 3
    rows = []
    if exact_arity:
        for name, value, shape, dtype in zip(
            ("dq", "dk", "dv"), outputs, expected_shapes, expected_dtypes
        ):
            rows.append(
                {
                    "name": name,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "device": str(value.device),
                    "contiguous": bool(value.is_contiguous()),
                    "finite": bool(value.isfinite().all().item()),
                    "pass": (
                        tuple(value.shape) == shape
                        and value.dtype == dtype
                        and value.device.type == "cuda"
                        and value.is_contiguous()
                        and bool(value.isfinite().all().item())
                    ),
                }
            )
    return {
        "exact_arity_three": exact_arity,
        "rows": rows,
        "pass": exact_arity and len(rows) == 3 and all(row["pass"] for row in rows),
    }


def tensor_contract(tensor, shape: tuple[int, ...], dtype) -> dict:
    finite = bool(tensor.isfinite().all().item())
    passed = (
        tuple(tensor.shape) == shape
        and tensor.dtype == dtype
        and tensor.device.type == "cuda"
        and tensor.is_contiguous()
        and finite
    )
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "contiguous": bool(tensor.is_contiguous()),
        "finite": finite,
        "pass": passed,
    }


def input_reference_contract(base, data, reference, args) -> dict:
    torch = base.torch
    q, k, v, dout, out, lse_raw, lse_tk = data
    qk_shape = (1, args.seqlen, args.heads, 192)
    v_shape = (1, args.seqlen, args.heads, 128)
    inputs = {
        "q": tensor_contract(q, qk_shape, torch.bfloat16),
        "k": tensor_contract(k, qk_shape, torch.bfloat16),
        "v": tensor_contract(v, v_shape, torch.bfloat16),
        "dout": tensor_contract(dout, v_shape, torch.bfloat16),
        "out": tensor_contract(out, v_shape, torch.bfloat16),
        "lse_raw": tensor_contract(
            lse_raw,
            (1, args.heads, args.seqlen),
            torch.float32,
        ),
        "lse_tk": tensor_contract(
            lse_tk,
            (1, args.seqlen, args.heads),
            torch.float32,
        ),
    }
    exact_reference_arity = (
        isinstance(reference, (tuple, list)) and len(reference) == 3
    )
    reference_rows = {}
    if exact_reference_arity:
        for name, tensor, shape in zip(
            ("dq", "dk", "dv"),
            reference,
            (qk_shape, qk_shape, v_shape),
        ):
            reference_rows[name] = tensor_contract(
                tensor,
                shape,
                torch.bfloat16,
            )
    return {
        "inputs": inputs,
        "cute_exact_arity_three": exact_reference_arity,
        "cute_outputs": reference_rows,
        "pass": (
            all(row["pass"] for row in inputs.values())
            and exact_reference_arity
            and len(reference_rows) == 3
            and all(row["pass"] for row in reference_rows.values())
        ),
    }


def materialize_parent(base, parent):
    torch = base.torch
    return (
        parent[0],
        parent[1].to(dtype=torch.bfloat16),
        parent[2].to(dtype=torch.bfloat16),
    )


def contract_fields(
    correctness_pass: bool,
    retention_pass: bool | None = None,
    objective_beats_cute: bool | None = None,
) -> dict:
    objective_evaluated = objective_beats_cute is not None
    retention_evaluated = retention_pass is not None
    overall = None
    scope = None
    if objective_evaluated:
        scope = (
            "correctness_retention_and_cute_objective"
            if retention_evaluated
            else "correctness_and_cute_objective"
        )
        overall = (
            correctness_pass
            and (not retention_evaluated or bool(retention_pass))
            and bool(objective_beats_cute)
        )
    return {
        "correctness_pass": correctness_pass,
        "retention_evaluated": retention_evaluated,
        "retention_pass": retention_pass,
        "objective_evaluated": objective_evaluated,
        "objective_beats_cute": objective_beats_cute,
        "overall_contract_scope": scope,
        "overall_contract_pass": overall,
    }


def make_context(base, extension, args):
    data = base.make_inputs(args.seed, args.seqlen, args.heads)
    return (
        data,
        getattr(extension, PARENT_ROUTE),
        getattr(extension, CHILD_ROUTE),
    )


def run_onecall(base, extension, args) -> dict:
    torch = base.torch
    data, parent_fn, child_fn = make_context(base, extension, args)
    reference = base.call_cute(data)
    torch.cuda.synchronize()
    parent = base.call_tk(parent_fn, data, args.seqlen)
    parent_cast = materialize_parent(base, parent)
    child = base.call_tk(child_fn, data, args.seqlen)
    torch.cuda.synchronize()
    parent_contract = output_contract(base, parent, data, child=False)
    child_contract = output_contract(base, child, data, child=True)
    input_reference = input_reference_contract(base, data, reference, args)
    child_rows = base.metrics(child, reference)
    parent_rows = base.metrics(parent_cast, reference)
    child_vs_parent = base.metrics(child, parent_cast)
    bitwise = {
        "dk": bool(torch.equal(child[1], parent_cast[1])),
        "dv": bool(torch.equal(child[2], parent_cast[2])),
    }
    correctness = (
        parent_contract["pass"]
        and child_contract["pass"]
        and input_reference["pass"]
        and rows_pass(child_rows)
        and all(bitwise.values())
    )
    return {
        **args.manifest,
        "runtime": base.runtime_module_provenance(),
        "complete": True,
        "state": "finished",
        "gate_pass": correctness,
        "gate_scope": "correctness_and_bitwise_materialization",
        **contract_fields(correctness),
        "gate_checks": {
            "parent_contract": parent_contract["pass"],
            "child_contract": child_contract["pass"],
            "input_reference_contract": input_reference["pass"],
            "child_error_pass": rows_pass(child_rows),
            "dk_dv_bitwise_parent_cast": all(bitwise.values()),
        },
        "parent_contract": parent_contract,
        "child_contract": child_contract,
        "input_reference_contract": input_reference,
        "parent_cast_vs_cute": parent_rows,
        "child_vs_cute": child_rows,
        "child_vs_parent_cast": child_vs_parent,
        "bitwise_equal_parent_cast": bitwise,
    }


def run_stress(base, extension, args) -> dict:
    torch = base.torch
    data, _, child_fn = make_context(base, extension, args)
    first_raw = base.call_tk(child_fn, data, args.seqlen)
    torch.cuda.synchronize()
    contract = output_contract(base, first_raw, data, child=True)
    first = tuple(value.clone() for value in first_raw)
    torch.cuda.synchronize()
    finite = [bool(value.isfinite().all().item()) for value in first]
    max_drift = [0.0, 0.0, 0.0]
    first_drift = [None, None, None]
    progress = {
        **args.manifest,
        "calls_requested": args.calls,
        "calls_completed": 1,
        "finite_dq_dk_dv": finite,
        "max_repeat_drift_dq_dk_dv": max_drift,
        "first_drift_dq_dk_dv": first_drift,
    }
    write_json(args.output, progress)
    for call_index in range(1, args.calls):
        current = base.call_tk(child_fn, data, args.seqlen)
        torch.cuda.synchronize()
        for index, (name, value, baseline) in enumerate(
            zip(("dq", "dk", "dv"), current, first)
        ):
            finite[index] &= bool(value.isfinite().all().item())
            difference = (value.float() - baseline.float()).abs()
            drift = float(difference.max())
            max_drift[index] = max(max_drift[index], drift)
            if drift > DRIFT_LIMITS[name] and first_drift[index] is None:
                first_drift[index] = {
                    "call": call_index,
                    "max_abs": drift,
                    "drift_limit": DRIFT_LIMITS[name],
                    "changed": int((difference > DRIFT_LIMITS[name]).sum()),
                }
        if (call_index + 1) % args.progress_every == 0:
            write_json(
                args.output,
                {
                    **progress,
                    "calls_completed": call_index + 1,
                    "finite_dq_dk_dv": finite,
                    "max_repeat_drift_dq_dk_dv": max_drift,
                    "first_drift_dq_dk_dv": first_drift,
                },
            )
    parent = base.call_tk(
        getattr(extension, PARENT_ROUTE), data, args.seqlen
    )
    parent_cast = materialize_parent(base, parent)
    torch.cuda.synchronize()
    parent_contract = output_contract(base, parent, data, child=False)
    bitwise = {
        "dk": bool(torch.equal(first[1], parent_cast[1])),
        "dv": bool(torch.equal(first[2], parent_cast[2])),
    }
    reference = base.call_cute(data)
    torch.cuda.synchronize()
    input_reference = input_reference_contract(base, data, reference, args)
    reference_rows = base.metrics(first, reference)
    drift_pass = all(
        drift <= DRIFT_LIMITS[name]
        for name, drift in zip(("dq", "dk", "dv"), max_drift)
    )
    correctness = (
        contract["pass"]
        and parent_contract["pass"]
        and input_reference["pass"]
        and all(finite)
        and drift_pass
        and rows_pass(reference_rows)
        and all(bitwise.values())
    )
    return {
        **progress,
        "runtime": base.runtime_module_provenance(),
        "complete": True,
        "state": "finished",
        "gate_pass": correctness,
        "gate_scope": "correctness_and_repeatability",
        **contract_fields(correctness),
        "gate_checks": {
            "output_contract": contract["pass"],
            "raw_parent_contract": parent_contract["pass"],
            "input_reference_contract": input_reference["pass"],
            "finite_pass": all(finite),
            "repeat_drift_pass": drift_pass,
            "reference_error_pass": rows_pass(reference_rows),
            "dk_dv_bitwise_raw_cast": all(bitwise.values()),
        },
        "calls_completed": args.calls,
        "finite_dq_dk_dv": finite,
        "max_repeat_drift_dq_dk_dv": max_drift,
        "first_drift_dq_dk_dv": first_drift,
        "first_vs_cute": reference_rows,
        "input_reference_contract": input_reference,
        "bitwise_equal_raw_cast": bitwise,
    }


def summarize(values: list[float]) -> dict:
    return {
        "median_us": statistics.median(values),
        "min_us": min(values),
        "max_us": max(values),
    }


def run_timing(base, extension, args) -> dict:
    torch = base.torch
    data, parent_fn, child_fn = make_context(base, extension, args)
    reference = base.call_cute(data)
    torch.cuda.synchronize()
    parent_check = base.call_tk(parent_fn, data, args.seqlen)
    parent_cast = materialize_parent(base, parent_check)
    child_check = base.call_tk(child_fn, data, args.seqlen)
    torch.cuda.synchronize()
    parent_contract = output_contract(base, parent_check, data, child=False)
    child_contract = output_contract(base, child_check, data, child=True)
    input_reference = input_reference_contract(base, data, reference, args)
    child_rows = base.metrics(child_check, reference)
    bitwise = {
        "dk": bool(torch.equal(child_check[1], parent_cast[1])),
        "dv": bool(torch.equal(child_check[2], parent_cast[2])),
    }
    correctness = (
        parent_contract["pass"]
        and child_contract["pass"]
        and input_reference["pass"]
        and rows_pass(child_rows)
        and all(bitwise.values())
    )
    if not correctness:
        return {
            **args.manifest,
            "runtime": base.runtime_module_provenance(),
            "complete": True,
            "state": "finished",
            "gate_pass": False,
            "gate_scope": "correctness",
            **contract_fields(False),
            "child_vs_cute": child_rows,
            "input_reference_contract": input_reference,
            "bitwise_equal_raw_cast": bitwise,
            "timing_skipped": True,
        }

    del reference, parent_check, parent_cast, child_check

    routes = {
        "raw_p": lambda: base.call_tk(parent_fn, data, args.seqlen),
        "compiled_bf16_child": lambda: base.call_tk(
            child_fn, data, args.seqlen
        ),
        "cute": lambda: base.call_cute(data),
    }
    for _ in range(args.warmups):
        for function in routes.values():
            torch.cuda.synchronize()
            result = function()
            torch.cuda.synchronize()
            del result
    values = {name: [] for name in routes}
    raw_samples = []
    orders = list(itertools.permutations(routes))
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
            elapsed = (time.perf_counter_ns() - started) / 1000.0
            values[name].append(elapsed)
            sample["measurements"].append({"route": name, "elapsed_us": elapsed})
            del result
        raw_samples.append(sample)
        if (sample_index + 1) % args.progress_every == 0:
            write_json(
                args.output,
                {
                    **args.manifest,
                    "samples_requested": args.samples,
                    "samples_completed": sample_index + 1,
                    "partial": {name: summarize(times) for name, times in values.items()},
                    "raw_samples": raw_samples,
                },
            )
    summaries = {name: summarize(times) for name, times in values.items()}
    child_over_parent = (
        summaries["compiled_bf16_child"]["median_us"]
        / summaries["raw_p"]["median_us"]
    )
    child_over_cute = (
        summaries["compiled_bf16_child"]["median_us"]
        / summaries["cute"]["median_us"]
    )
    raw_control_faster = child_over_parent < 1.0
    objective = child_over_cute < 1.0
    return {
        **args.manifest,
        "runtime": base.runtime_module_provenance(),
        "complete": True,
        "state": "finished",
        "gate_pass": correctness and objective,
        "gate_scope": "correctness_and_cute_objective",
        **contract_fields(correctness, None, objective),
        "child_vs_cute": child_rows,
        "input_reference_contract": input_reference,
        "bitwise_equal_raw_cast": bitwise,
        "timing_protocol": {
            "clock": "time.perf_counter_ns",
            "device_sync_immediately_before_route": True,
            "device_sync_immediately_after_route": True,
            "primary_routes": list(routes),
            "route_order": "six permutations, cycled by sample index",
            "progress_io_inside_intervals": False,
        },
        "samples": args.samples,
        "warmups": args.warmups,
        "route_timings": summaries,
        "ratios": {
            "compiled_bf16_child_over_raw_p": child_over_parent,
            "child_over_cute": child_over_cute,
        },
        "raw_p_control": {
            "api_contract_valid": False,
            "compiled_bf16_child_faster": raw_control_faster,
            "gate_role": "diagnostic_only",
        },
        "raw_samples": raw_samples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("onecall", "stress", "timing"))
    parser.add_argument("--extension", required=True, type=Path)
    parser.add_argument("--seqlen", required=True, type=int)
    parser.add_argument("--heads", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--calls", type=int, default=50)
    parser.add_argument("--samples", type=int, default=201)
    parser.add_argument("--warmups", type=int, default=12)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if (args.seqlen, args.heads) not in ((1024, 8), (4096, 1)):
        parser.error("shape must be S1024/H8 or S4096/H1")
    for name in ("seed", "calls", "samples", "progress_every"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmups < 0:
        parser.error("--warmups must be non-negative")
    if args.mode == "stress":
        if args.calls not in (50, 2000):
            parser.error("stress mode requires exactly 50 or 2000 calls")
        if args.calls == 2000 and args.seed != 20260768:
            parser.error("the 2000-call race gate requires seed 20260768")
    if args.mode == "timing":
        if args.samples != 201:
            parser.error("timing mode requires exactly 201 samples")
        if args.warmups != 12:
            parser.error("timing mode requires exactly 12 warmups")
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
            "bootstrap_error": f"{type(bootstrap_error).__name__}: {bootstrap_error}",
        }
    try:
        manifest = initial_manifest(args)
    except BaseException as error:
        failure = {
            **bootstrap,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        output = args.output
        if output.exists():
            output = output.with_name(output.name + ".initialization_error.json")
        write_json(output, failure)
        raise
    args.manifest = manifest
    write_json(args.output, manifest)
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
            raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 1")
        base = load_base_harness()
        base.torch.cuda.set_device(0)
        extension = base.load_extension(args.extension)
        if not hasattr(extension, PARENT_ROUTE) or not hasattr(extension, CHILD_ROUTE):
            raise AttributeError("extension does not expose both private routes")
        result = {
            "onecall": run_onecall,
            "stress": run_stress,
            "timing": run_timing,
        }[args.mode](base, extension, args)
        write_json(args.output, result)
        print(
            json.dumps(
                {
                    key: result.get(key)
                    for key in (
                        "complete",
                        "gate_pass",
                        "correctness_pass",
                        "retention_pass",
                        "objective_beats_cute",
                        "overall_contract_pass",
                        "ratios",
                    )
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if not result["gate_pass"]:
            raise SystemExit(2)
    except BaseException as error:
        if isinstance(error, SystemExit):
            raise
        failure = {
            **manifest,
            "complete": False,
            "state": "error",
            "gate_pass": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        write_json(args.output, failure)
        raise


if __name__ == "__main__":
    main()
