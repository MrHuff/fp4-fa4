from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace


REPO = Path(__file__).resolve().parents[1]
AUDITED_FUSED_HARNESS = (
    REPO / "results" / "tk_bf16_bwd_split2_bf16_fused_store_harness.py"
)
EXPECTED_FUSED_HARNESS_SHA256 = (
    "4808cd12fb50ea9640282212b6f522994b11d1976482b48e2a99cfccc19d2e3f"
)
AUDITED_MATERIALIZATION_HARNESS = (
    REPO / "results" / "tk_bf16_bwd_split2_bf16_materialize_harness.py"
)
EXPECTED_MATERIALIZATION_HARNESS_SHA256 = (
    "62462b1a0840c344f046643b84c008a969bbec3c863a964b72e778d1f571534a"
)
AUDITED_BASE_HARNESS = (
    REPO / "results" / "tk_bf16_bwd_s65536h128_harness.py"
)
EXPECTED_BASE_HARNESS_SHA256 = (
    "63e19fde901d714168c663ecee7f11d65081d903541d3871c1c8b52f79431db3"
)
EXTENSION = (
    REPO
    / "results"
    / ".artifacts"
    / "tk_bf16_bwd_split2_bf16_cached_acc_call_entry_ready"
    / "_C.cpython-312-aarch64-linux-gnu.so"
)
EXPECTED_EXTENSION_SHA256 = (
    "d47b48949189ee6c45265f4a64cb2a87d2f60290fb19d8d5c1db2c2cdde4cde6"
)
EXPECTED_EXTENSION_SIZE = 10_443_120
RAW_ROUTE = (
    "b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv_"
    "pipelined_tmem_dq_double_buffer_split2_dq_first_internal"
)
UNCACHED_ROUTE = RAW_ROUTE.replace(
    "split2_dq_first_internal", "split2_dq_first_bf16_internal"
)
CACHED_ROUTE = RAW_ROUTE.replace(
    "split2_dq_first_internal", "split2_dq_first_bf16_cached_internal"
)
REQUIRED_EXPORTS = (RAW_ROUTE, UNCACHED_ROUTE, CACHED_ROUTE)
SUPPORTED_SHAPES = ((1024, 8), (4096, 1))
STRESS_CALL_COUNTS = (50, 2000)
STRESS_2000_SEED = 20260768
TIMING_SAMPLES = 201
TIMING_WARMUPS = 12
DQ_MATCH_LIMIT = 2.0e-7
SCOPED_PATHS = (
    "tk_fa4/b300_bwd_cute16_candidate.cuh",
    "tk_fa4/b300_bwd_cute16_kernel_candidate.cuh",
    "tk_fa4/tk_fa4.cu",
    "results/tk_bf16_bwd_split2_bf16_fused_store_harness.py",
    "results/tk_bf16_bwd_split2_bf16_cached_acc_harness.py",
    "results/tk_bf16_bwd_broad_sweep_20260711.md",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def safe_expected_file_record(path: Path, expected_sha256: str) -> dict:
    record = safe_file_record(path)
    return {
        **record,
        "expected_sha256": expected_sha256,
        "hash_gate_pass": (
            record.get("exists") is True
            and record.get("sha256") == expected_sha256
        ),
    }


def gate_audited_dependency_files() -> dict:
    records = {
        "materialization_harness": safe_expected_file_record(
            AUDITED_MATERIALIZATION_HARNESS,
            EXPECTED_MATERIALIZATION_HARNESS_SHA256,
        ),
        "base_harness": safe_expected_file_record(
            AUDITED_BASE_HARNESS,
            EXPECTED_BASE_HARNESS_SHA256,
        ),
    }
    failed = [
        name for name, record in records.items() if not record["hash_gate_pass"]
    ]
    if failed:
        details = "; ".join(
            f"{name}: expected {records[name]['expected_sha256']}, "
            f"got {records[name].get('sha256')}"
            for name in failed
        )
        raise RuntimeError("audited dependency hash gate failed: " + details)
    return {"files": records, "pass": True}


def atomic_write_json(path: Path, payload: dict, *, replace: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            temporary.unlink()
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def git_output(*arguments: str, binary: bool = False):
    return subprocess.run(
        ("git", *arguments),
        cwd=REPO,
        check=True,
        capture_output=True,
        text=not binary,
    ).stdout


def tk_fa4_environment() -> dict:
    values = {
        name: value
        for name, value in sorted(os.environ.items())
        if name.startswith("TK_FA4_")
    }
    return {
        "values": values,
        "present_variables": list(values),
        "pass": not values,
    }


def shape_record(args: argparse.Namespace) -> dict:
    if args.mode == "transitions":
        return {
            "transition_datasets": {
                "primary": {
                    "seed": args.seed,
                    "seqlen": 1024,
                    "heads": 8,
                },
                "shape_change": {
                    "seed": args.seed + 1,
                    "seqlen": 4096,
                    "heads": 1,
                },
                "peer": {
                    "seed": args.seed + 2,
                    "seqlen": 1024,
                    "heads": 8,
                },
            },
            "same_stream_dataset_sequence": [
                "primary",
                "shape_change",
                "primary",
            ],
            "active_stream_dataset_sequence": [
                "primary",
                "peer",
                "primary",
            ],
        }
    return {"seqlen": args.seqlen, "heads": args.heads}


def timing_pair_contract(args: argparse.Namespace):
    if args.mode not in ("timing_ab", "timing_canonical"):
        return None
    paired_offsets = [0, 1] if args.mode == "timing_ab" else [0, 3]
    expected_offset = paired_offsets[args.replicate - 1]
    return {
        "replicate": args.replicate,
        "replicate_count": 2,
        "paired_offsets": paired_offsets,
        "selected_offset": args.order_offset,
        "expected_offset": expected_offset,
        "pass": args.order_offset == expected_offset,
    }


def fallback_manifest(args: argparse.Namespace) -> dict:
    return {
        "schema_version": 2,
        "complete": False,
        "state": "initializing",
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
        "seed": args.seed,
        **shape_record(args),
        "timing_pair_contract": timing_pair_contract(args),
        "extension": safe_file_record(EXTENSION),
        "harness": safe_file_record(Path(__file__)),
        "audited_fused_harness": safe_file_record(AUDITED_FUSED_HARNESS),
        "audited_dependency_files": {
            "materialization_harness": safe_expected_file_record(
                AUDITED_MATERIALIZATION_HARNESS,
                EXPECTED_MATERIALIZATION_HARNESS_SHA256,
            ),
            "base_harness": safe_expected_file_record(
                AUDITED_BASE_HARNESS,
                EXPECTED_BASE_HARNESS_SHA256,
            ),
        },
        "tk_fa4_environment": tk_fa4_environment(),
    }


def initial_manifest(args: argparse.Namespace) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 1")
    environment = tk_fa4_environment()
    if not environment["pass"]:
        raise RuntimeError(
            "all TK_FA4 timing/probe environment variables must be absent: "
            + ", ".join(environment["present_variables"])
        )
    extension = EXTENSION.resolve()
    if not extension.is_file():
        raise FileNotFoundError(f"cached extension does not exist: {extension}")
    if extension.stat().st_size != EXPECTED_EXTENSION_SIZE:
        raise RuntimeError(
            f"cached extension size mismatch: expected {EXPECTED_EXTENSION_SIZE}, "
            f"got {extension.stat().st_size}"
        )
    extension_sha = sha256_file(extension)
    if extension_sha != EXPECTED_EXTENSION_SHA256:
        raise RuntimeError(
            f"cached extension SHA mismatch: expected {EXPECTED_EXTENSION_SHA256}, "
            f"got {extension_sha}"
        )
    fused_sha = sha256_file(AUDITED_FUSED_HARNESS)
    if fused_sha != EXPECTED_FUSED_HARNESS_SHA256:
        raise RuntimeError(
            "audited fused harness SHA mismatch before import: expected "
            f"{EXPECTED_FUSED_HARNESS_SHA256}, got {fused_sha}"
        )
    dependency_gate = gate_audited_dependency_files()
    scoped_diff = git_output("diff", "--binary", "--", *SCOPED_PATHS, binary=True)
    diff_check = subprocess.run(
        ("git", "diff", "--check", "--", *SCOPED_PATHS),
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    if diff_check.returncode != 0:
        raise RuntimeError(
            "scoped git diff --check failed: "
            + diff_check.stdout
            + diff_check.stderr
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
        "audited_fused_harness": str(AUDITED_FUSED_HARNESS),
        "audited_fused_harness_sha256": fused_sha,
        "expected_audited_fused_harness_sha256": EXPECTED_FUSED_HARNESS_SHA256,
        "audited_dependency_hash_gate_before_fused_import": dependency_gate,
        "raw_route": RAW_ROUTE,
        "uncached_route": UNCACHED_ROUTE,
        "cached_route": CACHED_ROUTE,
        "required_exports": list(REQUIRED_EXPORTS),
        "git_head": git_output("rev-parse", "HEAD").strip(),
        "git_head_tree": git_output("rev-parse", "HEAD^{tree}").strip(),
        "scoped_paths": list(SCOPED_PATHS),
        "scoped_status": git_output(
            "status", "--short", "--", *SCOPED_PATHS
        ).splitlines(),
        "scoped_diff_sha256": hashlib.sha256(scoped_diff).hexdigest(),
        "scoped_diff_size_bytes": len(scoped_diff),
        "scoped_diff_check_pass": True,
        "scoped_diff_check_output": diff_check.stdout + diff_check.stderr,
        "thresholds": {
            "dq_match_max_abs": DQ_MATCH_LIMIT,
            "stress_drift": {"dq": DQ_MATCH_LIMIT, "dk": 0.0, "dv": 0.0},
            "timing_cached_over_uncached": 1.0,
            "timing_cached_over_cute": 1.0,
        },
    }


def load_audited_fused_harness():
    dependency_gate = gate_audited_dependency_files()
    fused_sha = sha256_file(AUDITED_FUSED_HARNESS)
    if fused_sha != EXPECTED_FUSED_HARNESS_SHA256:
        raise RuntimeError(
            "audited fused harness changed before import: expected "
            f"{EXPECTED_FUSED_HARNESS_SHA256}, got {fused_sha}"
        )
    spec = importlib.util.spec_from_file_location(
        "tk_bf16_bwd_split2_bf16_fused_store_audited",
        AUDITED_FUSED_HARNESS,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"cannot import audited fused harness: {AUDITED_FUSED_HARNESS}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, dependency_gate


def verify_fused_dependency_contract(fused) -> dict:
    required_attributes = (
        "REVIEWED_MATERIALIZATION_HARNESS",
        "EXPECTED_REVIEWED_HARNESS_SHA256",
        "BASE_HARNESS",
        "EXPECTED_BASE_HARNESS_SHA256",
    )
    missing = [name for name in required_attributes if not hasattr(fused, name)]
    if missing:
        raise RuntimeError(
            "audited fused harness is missing dependency attributes: "
            + ", ".join(missing)
        )
    actual = {
        "materialization_harness_path": str(
            Path(fused.REVIEWED_MATERIALIZATION_HARNESS).resolve()
        ),
        "materialization_harness_expected_sha256": (
            fused.EXPECTED_REVIEWED_HARNESS_SHA256
        ),
        "base_harness_path": str(Path(fused.BASE_HARNESS).resolve()),
        "base_harness_expected_sha256": fused.EXPECTED_BASE_HARNESS_SHA256,
    }
    expected = {
        "materialization_harness_path": str(
            AUDITED_MATERIALIZATION_HARNESS.resolve()
        ),
        "materialization_harness_expected_sha256": (
            EXPECTED_MATERIALIZATION_HARNESS_SHA256
        ),
        "base_harness_path": str(AUDITED_BASE_HARNESS.resolve()),
        "base_harness_expected_sha256": EXPECTED_BASE_HARNESS_SHA256,
    }
    checks = {
        name: actual[name] == expected[name]
        for name in expected
    }
    if not all(checks.values()):
        mismatches = "; ".join(
            f"{name}: expected {expected[name]}, got {actual[name]}"
            for name, passed in checks.items()
            if not passed
        )
        raise RuntimeError("fused dependency contract mismatch: " + mismatches)
    return {
        "required_attributes": list(required_attributes),
        "actual": actual,
        "expected": expected,
        "checks": checks,
        "pass": True,
    }


def load_cached_extension(base):
    if "_C" in sys.modules:
        raise RuntimeError("refusing to load cached artifact over a preloaded _C module")
    extension = base.load_extension(EXTENSION.resolve())
    module_file = getattr(extension, "__file__", None)
    if module_file is None:
        raise RuntimeError("loaded extension has no __file__")
    loaded_path = Path(module_file).resolve()
    expected_path = EXTENSION.resolve()
    if loaded_path != expected_path:
        raise RuntimeError(
            f"loaded extension path mismatch: expected {expected_path}, got {loaded_path}"
        )
    loaded_sha = sha256_file(loaded_path)
    if loaded_sha != EXPECTED_EXTENSION_SHA256:
        raise RuntimeError(
            f"loaded extension SHA mismatch: expected {EXPECTED_EXTENSION_SHA256}, "
            f"got {loaded_sha}"
        )
    missing = [
        name
        for name in REQUIRED_EXPORTS
        if not hasattr(extension, name) or not callable(getattr(extension, name))
    ]
    if missing:
        raise AttributeError("cached extension missing exports: " + ", ".join(missing))
    return extension, {
        "module_file": str(loaded_path),
        "module_file_exact_match": True,
        "module_name": getattr(extension, "__name__", None),
        "sha256": loaded_sha,
        "size_bytes": loaded_path.stat().st_size,
        "required_exports": {name: True for name in REQUIRED_EXPORTS},
    }


def shape_args(seqlen: int, heads: int) -> SimpleNamespace:
    return SimpleNamespace(seqlen=seqlen, heads=heads)


def capture_input_snapshot(data) -> tuple:
    return tuple(tensor.detach().cpu().clone() for tensor in data)


def input_immutability_contract(base, data, snapshots) -> dict:
    names = ("q", "k", "v", "dout", "out", "lse_raw", "lse_tk")
    rows = []
    for name, tensor, snapshot in zip(names, data, snapshots):
        unchanged = bool(base.torch.equal(tensor.detach().cpu(), snapshot))
        shape_match = tuple(tensor.shape) == tuple(snapshot.shape)
        dtype_match = tensor.dtype == snapshot.dtype
        rows.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "contiguous": bool(tensor.is_contiguous()),
                "shape_match": shape_match,
                "dtype_match": dtype_match,
                "unchanged": unchanged,
                "pass": (
                    tensor.device.type == "cuda"
                    and tensor.is_contiguous()
                    and shape_match
                    and dtype_match
                    and unchanged
                ),
            }
        )
    return {"rows": rows, "pass": len(rows) == 7 and all(row["pass"] for row in rows)}


def storage_record(tensor) -> dict:
    storage = tensor.untyped_storage()
    return {
        "data_ptr": int(storage.data_ptr()),
        "nbytes": int(storage.nbytes()),
        "device": str(storage.device),
        "tensor_shape": list(tensor.shape),
        "tensor_dtype": str(tensor.dtype),
        "tensor_storage_offset": int(tensor.storage_offset()),
    }


def tensors_share_storage(left, right) -> bool:
    left_storage = storage_record(left)
    right_storage = storage_record(right)
    return (
        left_storage["device"] == right_storage["device"]
        and left_storage["data_ptr"] == right_storage["data_ptr"]
    )


def flatten_tensor_groups(groups: dict) -> list:
    flattened = []
    for group_name, group in groups.items():
        if len(group) == 3:
            names = ("dq", "dk", "dv")
        elif len(group) == 7:
            names = ("q", "k", "v", "dout", "out", "lse_raw", "lse_tk")
        else:
            names = tuple(f"tensor_{index}" for index in range(len(group)))
        flattened.extend(
            (f"{group_name}.{name}", tensor)
            for name, tensor in zip(names, group)
        )
    return flattened


def non_alias_contract(output_groups: dict, protected_groups: dict) -> dict:
    outputs = flatten_tensor_groups(output_groups)
    protected = flatten_tensor_groups(protected_groups)
    conflicts = []
    for (left_name, left), (right_name, right) in itertools.combinations(outputs, 2):
        if tensors_share_storage(left, right):
            conflicts.append([left_name, right_name])
    for output_name, output in outputs:
        for protected_name, protected_tensor in protected:
            if tensors_share_storage(output, protected_tensor):
                conflicts.append([output_name, protected_name])
    return {
        "output_tensor_count": len(outputs),
        "protected_tensor_count": len(protected),
        "output_storages": {
            name: storage_record(tensor) for name, tensor in outputs
        },
        "protected_storages": {
            name: storage_record(tensor) for name, tensor in protected
        },
        "conflicts": conflicts,
        "pass": not conflicts,
    }


def cross_group_non_alias_contract(groups: dict) -> dict:
    grouped = {
        group_name: flatten_tensor_groups({group_name: tensors})
        for group_name, tensors in groups.items()
    }
    cross_group_conflicts = []
    cross_group_comparison_count = 0
    for (left_group, left_tensors), (right_group, right_tensors) in (
        itertools.combinations(grouped.items(), 2)
    ):
        for left_name, left in left_tensors:
            for right_name, right in right_tensors:
                cross_group_comparison_count += 1
                if tensors_share_storage(left, right):
                    cross_group_conflicts.append(
                        {
                            "left_group": left_group,
                            "left_tensor": left_name,
                            "right_group": right_group,
                            "right_tensor": right_name,
                        }
                    )
    intra_group_aliases = []
    for group_name, tensors in grouped.items():
        for (left_name, left), (right_name, right) in itertools.combinations(
            tensors, 2
        ):
            if tensors_share_storage(left, right):
                intra_group_aliases.append(
                    {
                        "group": group_name,
                        "left_tensor": left_name,
                        "right_tensor": right_name,
                        "gate_role": "documented_non_gating",
                    }
                )
    return {
        "comparison_scope": "cross_group_only",
        "intra_group_alias_policy": "allowed_and_recorded_non_gating",
        "group_count": len(grouped),
        "group_tensor_counts": {
            group_name: len(tensors) for group_name, tensors in grouped.items()
        },
        "group_storages": {
            group_name: {
                tensor_name: storage_record(tensor)
                for tensor_name, tensor in tensors
            }
            for group_name, tensors in grouped.items()
        },
        "cross_group_comparison_count": cross_group_comparison_count,
        "cross_group_conflicts": cross_group_conflicts,
        "intra_group_aliases": intra_group_aliases,
        "pass": not cross_group_conflicts,
    }


def output_snapshot_contract(base, outputs: dict, snapshots: dict) -> dict:
    rows = []
    for group_name, output in outputs.items():
        snapshot = snapshots[group_name]
        for index, name in enumerate(("dq", "dk", "dv")):
            unchanged = bool(base.torch.equal(output[index], snapshot[index]))
            rows.append(
                {
                    "group": group_name,
                    "output": name,
                    "unchanged": unchanged,
                }
            )
    return {
        "comparisons": rows,
        "pass": len(rows) == 3 * len(outputs) and all(
            row["unchanged"] for row in rows
        ),
    }


def dq_match_contract(base, groups: dict) -> dict:
    comparisons = []
    for (left_name, left), (right_name, right) in itertools.combinations(
        groups.items(), 2
    ):
        max_abs = float((left[0].float() - right[0].float()).abs().max())
        comparisons.append(
            {
                "left": left_name,
                "right": right_name,
                "max_abs": max_abs,
                "limit": DQ_MATCH_LIMIT,
                "pass": max_abs <= DQ_MATCH_LIMIT,
            }
        )
    return {
        "comparisons": comparisons,
        "pass": bool(comparisons) and all(row["pass"] for row in comparisons),
    }


def dkdv_bitwise_contract(base, children: dict, raw_cast) -> dict:
    rows = []
    for child_name, child in children.items():
        for index, name in ((1, "dk"), (2, "dv")):
            equal = bool(base.torch.equal(child[index], raw_cast[index]))
            rows.append(
                {
                    "left": child_name,
                    "right": "raw_cast",
                    "output": name,
                    "equal": equal,
                }
            )
    for (left_name, left), (right_name, right) in itertools.combinations(
        children.items(), 2
    ):
        for index, name in ((1, "dk"), (2, "dv")):
            equal = bool(base.torch.equal(left[index], right[index]))
            rows.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "output": name,
                    "equal": equal,
                }
            )
    return {"comparisons": rows, "pass": bool(rows) and all(row["equal"] for row in rows)}


def evaluate_materialized_outputs(
    base,
    fused,
    args,
    data,
    snapshots,
    reference,
    raw,
    children: dict,
) -> dict:
    raw_cast = fused.materialize_parent(base, raw)
    base.torch.cuda.synchronize()
    raw_contract = fused.output_contract(base, raw, data, child=False)
    raw_cast_contract = fused.output_contract(base, raw_cast, data, child=True)
    child_contracts = {
        name: fused.output_contract(base, output, data, child=True)
        for name, output in children.items()
    }
    input_reference = fused.input_reference_contract(
        base, data, reference, shape_args(args.seqlen, args.heads)
    )
    metrics = {
        "raw_cast": base.metrics(raw_cast, reference),
        **{
            name: base.metrics(output, reference)
            for name, output in children.items()
        },
    }
    metrics_pass = all(fused.rows_pass(rows) for rows in metrics.values())
    bitwise = dkdv_bitwise_contract(base, children, raw_cast)
    dq_match = dq_match_contract(base, {"raw": raw, **children})
    aliases = non_alias_contract(
        {"raw": raw, **children, "cute_reference": reference},
        {"inputs": data},
    )
    immutable = input_immutability_contract(base, data, snapshots)
    correctness = (
        raw_contract["pass"]
        and raw_cast_contract["pass"]
        and len(child_contracts) == len(children)
        and all(contract["pass"] for contract in child_contracts.values())
        and input_reference["pass"]
        and metrics_pass
        and bitwise["pass"]
        and dq_match["pass"]
        and aliases["pass"]
        and immutable["pass"]
    )
    return {
        "correctness_pass": correctness,
        "gate_checks": {
            "raw_output_contract": raw_contract["pass"],
            "raw_cast_contract": raw_cast_contract["pass"],
            "child_output_contracts": all(
                contract["pass"] for contract in child_contracts.values()
            ),
            "input_reference_contract": input_reference["pass"],
            "reference_error_pass": metrics_pass,
            "dk_dv_bitwise": bitwise["pass"],
            "dq_match": dq_match["pass"],
            "non_alias": aliases["pass"],
            "input_immutability": immutable["pass"],
        },
        "raw_contract": raw_contract,
        "raw_cast_contract": raw_cast_contract,
        "child_contracts": child_contracts,
        "input_reference_contract": input_reference,
        "metrics_vs_cute": metrics,
        "dk_dv_bitwise": bitwise,
        "dq_match": dq_match,
        "non_alias_contract": aliases,
        "input_immutability_contract": immutable,
    }


def make_context(base, extension, seed: int, seqlen: int, heads: int):
    data = base.make_inputs(seed, seqlen, heads)
    return (
        data,
        getattr(extension, RAW_ROUTE),
        getattr(extension, UNCACHED_ROUTE),
        getattr(extension, CACHED_ROUTE),
    )


def result_base(args, base) -> dict:
    return {
        **args.manifest,
        "runtime": base.runtime_module_provenance(),
        "complete": True,
        "state": "finished",
    }


def run_onecall(base, fused, extension, args) -> dict:
    torch = base.torch
    data, raw_fn, uncached_fn, cached_fn = make_context(
        base, extension, args.seed, args.seqlen, args.heads
    )
    snapshots = capture_input_snapshot(data)
    reference = base.call_cute(data)
    torch.cuda.synchronize()
    cold_cached = base.call_tk(cached_fn, data, args.seqlen)
    uncached = base.call_tk(uncached_fn, data, args.seqlen)
    raw = base.call_tk(raw_fn, data, args.seqlen)
    warm_cached = base.call_tk(cached_fn, data, args.seqlen)
    torch.cuda.synchronize()
    evaluation = evaluate_materialized_outputs(
        base,
        fused,
        args,
        data,
        snapshots,
        reference,
        raw,
        {
            "cold_cached": cold_cached,
            "uncached": uncached,
            "warm_cached": warm_cached,
        },
    )
    correctness = evaluation["correctness_pass"]
    return {
        **result_base(args, base),
        "gate_pass": correctness,
        "gate_scope": "cold_warm_cache_correctness",
        **fused.contract_fields(correctness),
        "call_order": ["cold_cached", "uncached", "raw", "warm_cached"],
        "no_intermediate_device_sync": True,
        **evaluation,
    }


def stress_sample(base, output, baseline, call_index: int) -> dict:
    finite = [bool(value.isfinite().all().item()) for value in output]
    drift = []
    for value, expected in zip(output, baseline):
        value_drift = float((value.float() - expected.float()).abs().max())
        drift.append(value_drift if math.isfinite(value_drift) else None)
    return {
        "call_index": call_index,
        "finite_dq_dk_dv": finite,
        "drift_dq_dk_dv": drift,
    }


def run_stress(base, fused, extension, args) -> dict:
    torch = base.torch
    data, raw_fn, uncached_fn, cached_fn = make_context(
        base, extension, args.seed, args.seqlen, args.heads
    )
    snapshots = capture_input_snapshot(data)
    reference = base.call_cute(data)
    torch.cuda.synchronize()
    first = base.call_tk(cached_fn, data, args.seqlen)
    torch.cuda.synchronize()
    baseline = tuple(value.clone() for value in first)
    torch.cuda.synchronize()
    first_contract = fused.output_contract(base, first, data, child=True)
    raw_samples = [
        {
            "call_index": 0,
            "finite_dq_dk_dv": [bool(value.isfinite().all().item()) for value in first],
            "drift_dq_dk_dv": [0.0, 0.0, 0.0],
        }
    ]
    finite = list(raw_samples[0]["finite_dq_dk_dv"])
    max_drift = list(raw_samples[0]["drift_dq_dk_dv"])
    first_drift = [None, None, None]
    first_non_alias = non_alias_contract(
        {"cached_first": first, "cute_reference": reference},
        {"inputs": data, "cached_first_snapshot": baseline},
    )
    non_alias_pass = first_non_alias["pass"]
    all_output_contracts = first_contract["pass"]
    progress = {
        **args.manifest,
        "calls_requested": args.calls,
        "calls_completed": 1,
        "raw_samples": raw_samples,
    }
    atomic_write_json(args.output, progress, replace=True)
    limits = (DQ_MATCH_LIMIT, 0.0, 0.0)
    for call_index in range(1, args.calls):
        current = base.call_tk(cached_fn, data, args.seqlen)
        torch.cuda.synchronize()
        contract = fused.output_contract(base, current, data, child=True)
        all_output_contracts &= contract["pass"]
        alias_row = non_alias_contract(
            {
                "cached_first": first,
                "cached_current": current,
                "cute_reference": reference,
            },
            {"inputs": data, "cached_first_snapshot": baseline},
        )
        non_alias_pass &= alias_row["pass"]
        sample = stress_sample(base, current, baseline, call_index)
        raw_samples.append(sample)
        for index, (is_finite, drift, limit) in enumerate(
            zip(
                sample["finite_dq_dk_dv"],
                sample["drift_dq_dk_dv"],
                limits,
            )
        ):
            finite[index] &= is_finite
            if drift is None:
                if first_drift[index] is None:
                    first_drift[index] = {
                        "call_index": call_index,
                        "max_abs": None,
                        "limit": limit,
                        "reason": "non-finite drift",
                    }
                continue
            max_drift[index] = max(max_drift[index], drift)
            if drift > limit and first_drift[index] is None:
                first_drift[index] = {
                    "call_index": call_index,
                    "max_abs": drift,
                    "limit": limit,
                }
        del current
        if (call_index + 1) % args.progress_every == 0:
            atomic_write_json(
                args.output,
                {
                    **progress,
                    "calls_completed": call_index + 1,
                    "finite_dq_dk_dv": finite,
                    "max_repeat_drift_dq_dk_dv": max_drift,
                    "first_drift_dq_dk_dv": first_drift,
                    "raw_samples": raw_samples,
                },
                replace=True,
            )
    uncached = base.call_tk(uncached_fn, data, args.seqlen)
    raw = base.call_tk(raw_fn, data, args.seqlen)
    torch.cuda.synchronize()
    evaluation = evaluate_materialized_outputs(
        base,
        fused,
        args,
        data,
        snapshots,
        reference,
        raw,
        {"cached_first": first, "uncached": uncached},
    )
    drift_pass = all(
        drift <= limit for drift, limit in zip(max_drift, limits)
    )
    correctness = (
        evaluation["correctness_pass"]
        and all_output_contracts
        and all(finite)
        and drift_pass
        and non_alias_pass
        and len(raw_samples) == args.calls
    )
    return {
        **result_base(args, base),
        "gate_pass": correctness,
        "gate_scope": "cached_correctness_repeatability_and_bitwise",
        **fused.contract_fields(correctness),
        "gate_checks": {
            **evaluation["gate_checks"],
            "all_output_contracts": all_output_contracts,
            "finite_pass": all(finite),
            "repeat_drift_pass": drift_pass,
            "per_call_non_alias": non_alias_pass,
            "raw_sample_count": len(raw_samples) == args.calls,
        },
        "calls_requested": args.calls,
        "calls_completed": args.calls,
        "finite_dq_dk_dv": finite,
        "max_repeat_drift_dq_dk_dv": max_drift,
        "first_drift_dq_dk_dv": first_drift,
        "first_call_non_alias_contract": first_non_alias,
        "raw_samples": raw_samples,
        "evaluation": evaluation,
    }


def evaluate_cached_sequence(
    base,
    fused,
    data_by_dataset: dict,
    input_snapshots: dict,
    references: dict,
    reference_snapshots: dict,
    outputs: dict,
    output_snapshots: dict,
    output_dataset_keys: dict,
    controls: dict,
    control_snapshots: dict,
    stream_handles_distinct: bool,
    dataset_seeds_distinct: bool,
) -> dict:
    datasets = {
        "primary": {"seqlen": 1024, "heads": 8},
        "shape_change": {"seqlen": 4096, "heads": 1},
        "peer": {"seqlen": 1024, "heads": 8},
    }
    contracts = {}
    metrics = {}
    for name, output in outputs.items():
        dataset_key = output_dataset_keys[name]
        data = data_by_dataset[dataset_key]
        reference = references[dataset_key]
        contracts[name] = fused.output_contract(base, output, data, child=True)
        metrics[name] = base.metrics(output, reference)
    input_reference = {
        dataset_key: fused.input_reference_contract(
            base,
            data_by_dataset[dataset_key],
            references[dataset_key],
            shape_args(dataset["seqlen"], dataset["heads"]),
        )
        for dataset_key, dataset in datasets.items()
    }
    control_contracts = {
        dataset_key: {
            "raw": fused.output_contract(
                base,
                dataset_controls["raw"],
                data_by_dataset[dataset_key],
                child=False,
            ),
            "raw_cast": fused.output_contract(
                base,
                dataset_controls["raw_cast"],
                data_by_dataset[dataset_key],
                child=True,
            ),
            "uncached": fused.output_contract(
                base,
                dataset_controls["uncached"],
                data_by_dataset[dataset_key],
                child=True,
            ),
        }
        for dataset_key, dataset_controls in controls.items()
    }
    control_metrics = {
        f"{dataset_key}_{control_name}": base.metrics(
            controls[dataset_key][control_name], references[dataset_key]
        )
        for dataset_key in datasets
        for control_name in ("raw_cast", "uncached")
    }

    bitwise_rows = []
    dq_rows = []
    for output_name, output in outputs.items():
        dataset_key = output_dataset_keys[output_name]
        for target_name in ("uncached", "raw_cast"):
            target = controls[dataset_key][target_name]
            for index, tensor_name in ((1, "dk"), (2, "dv")):
                bitwise_rows.append(
                    {
                        "comparison_scope": "dataset_control_relative",
                        "left": output_name,
                        "right": f"{dataset_key}_{target_name}",
                        "output": tensor_name,
                        "equal": bool(base.torch.equal(output[index], target[index])),
                    }
                )
        for target_name in ("uncached", "raw"):
            target = controls[dataset_key][target_name]
            max_abs = float((output[0].float() - target[0].float()).abs().max())
            dq_rows.append(
                {
                    "comparison_scope": "dataset_control_relative",
                    "left": output_name,
                    "right": f"{dataset_key}_{target_name}",
                    "max_abs": max_abs,
                    "limit": DQ_MATCH_LIMIT,
                    "pass": math.isfinite(max_abs) and max_abs <= DQ_MATCH_LIMIT,
                }
            )
    for dataset_key in datasets:
        uncached = controls[dataset_key]["uncached"]
        raw = controls[dataset_key]["raw"]
        raw_cast = controls[dataset_key]["raw_cast"]
        for index, tensor_name in ((1, "dk"), (2, "dv")):
            bitwise_rows.append(
                {
                    "comparison_scope": "dataset_control_relative",
                    "left": f"{dataset_key}_uncached",
                    "right": f"{dataset_key}_raw_cast",
                    "output": tensor_name,
                    "equal": bool(base.torch.equal(uncached[index], raw_cast[index])),
                }
            )
        max_abs = float((uncached[0].float() - raw[0].float()).abs().max())
        dq_rows.append(
            {
                "comparison_scope": "dataset_control_relative",
                "left": f"{dataset_key}_uncached",
                "right": f"{dataset_key}_raw",
                "max_abs": max_abs,
                "limit": DQ_MATCH_LIMIT,
                "pass": math.isfinite(max_abs) and max_abs <= DQ_MATCH_LIMIT,
            }
        )
    primary_outputs = {
        name: output
        for name, output in outputs.items()
        if output_dataset_keys[name] == "primary"
    }
    primary_repeat_bitwise_rows = []
    primary_repeat_dq_rows = []
    for (left_name, left), (right_name, right) in itertools.combinations(
        primary_outputs.items(), 2
    ):
        for index, tensor_name in ((1, "dk"), (2, "dv")):
            row = {
                "comparison_scope": "primary_cached_repeat",
                "left": left_name,
                "right": right_name,
                "output": tensor_name,
                "equal": bool(base.torch.equal(left[index], right[index])),
            }
            primary_repeat_bitwise_rows.append(row)
            bitwise_rows.append(row)
        max_abs = float((left[0].float() - right[0].float()).abs().max())
        row = {
            "comparison_scope": "primary_cached_repeat",
            "left": left_name,
            "right": right_name,
            "max_abs": max_abs,
            "limit": DQ_MATCH_LIMIT,
            "pass": math.isfinite(max_abs) and max_abs <= DQ_MATCH_LIMIT,
        }
        primary_repeat_dq_rows.append(row)
        dq_rows.append(row)
    primary_repeat_consistency = {
        "primary_output_count": len(primary_outputs),
        "expected_primary_output_count": 4,
        "dk_dv_comparisons": primary_repeat_bitwise_rows,
        "expected_dk_dv_comparison_count": 12,
        "dq_comparisons": primary_repeat_dq_rows,
        "expected_dq_comparison_count": 6,
        "pass": (
            len(primary_outputs) == 4
            and len(primary_repeat_bitwise_rows) == 12
            and all(row["equal"] for row in primary_repeat_bitwise_rows)
            and len(primary_repeat_dq_rows) == 6
            and all(row["pass"] for row in primary_repeat_dq_rows)
        ),
    }
    bitwise = {
        "comparisons": bitwise_rows,
        "expected_comparison_count": 42,
        "pass": len(bitwise_rows) == 42
        and all(row["equal"] for row in bitwise_rows),
    }
    dq_match = {
        "comparisons": dq_rows,
        "expected_comparison_count": 21,
        "pass": len(dq_rows) == 21 and all(row["pass"] for row in dq_rows),
    }

    transition_snapshots_immutable = output_snapshot_contract(
        base, outputs, output_snapshots
    )
    references_immutable = output_snapshot_contract(
        base, references, reference_snapshots
    )
    controls_with_snapshots = {
        f"{dataset_key}_{control_name}": controls[dataset_key][control_name]
        for dataset_key in datasets
        for control_name in ("raw", "raw_cast", "uncached")
    }
    control_snapshots_immutable = output_snapshot_contract(
        base, controls_with_snapshots, control_snapshots
    )
    alias_control_outputs = {
        f"{dataset_key}_{control_name}": controls[dataset_key][control_name]
        for dataset_key in datasets
        for control_name in ("raw", "uncached")
    }
    alias_outputs = {
        **outputs,
        **alias_control_outputs,
        **{
            f"cute_reference_{dataset_key}": references[dataset_key]
            for dataset_key in datasets
        },
    }
    alias_protected = {
        **{
            f"inputs_{dataset_key}": data_by_dataset[dataset_key]
            for dataset_key in datasets
        },
        **{
            f"transition_snapshot_{name}": snapshot
            for name, snapshot in output_snapshots.items()
        },
        **{
            f"control_snapshot_{name}": snapshot
            for name, snapshot in control_snapshots.items()
        },
        **{
            f"cute_reference_snapshot_{dataset_key}": reference_snapshots[
                dataset_key
            ]
            for dataset_key in datasets
        },
    }
    aliases = non_alias_contract(alias_outputs, alias_protected)
    raw_cast_bf16_aliases = non_alias_contract(
        {
            f"{dataset_key}_raw_cast_bf16": controls[dataset_key]["raw_cast"][
                1:
            ]
            for dataset_key in datasets
        },
        {**alias_outputs, **alias_protected},
    )
    input_dataset_storage = cross_group_non_alias_contract(
        {
            dataset_key: data_by_dataset[dataset_key]
            for dataset_key in datasets
        }
    )
    input_dataset_storage["cross_group_only_pass"] = input_dataset_storage[
        "pass"
    ]
    input_dataset_storage["expected_group_count"] = 3
    input_dataset_storage["expected_tensors_per_group"] = 7
    input_dataset_storage["expected_cross_group_comparison_count"] = 147
    input_dataset_storage["pass"] = (
        input_dataset_storage["cross_group_only_pass"]
        and input_dataset_storage["group_count"] == 3
        and all(
            count == 7
            for count in input_dataset_storage["group_tensor_counts"].values()
        )
        and input_dataset_storage["cross_group_comparison_count"] == 147
    )
    raw_cast_storage_rows = []
    for dataset_key in datasets:
        raw = controls[dataset_key]["raw"]
        raw_cast = controls[dataset_key]["raw_cast"]
        row = {
            "dataset_key": dataset_key,
            "dq_intentionally_aliases_raw": tensors_share_storage(
                raw_cast[0], raw[0]
            ),
            "dk_shares_raw_storage": tensors_share_storage(raw_cast[1], raw[1]),
            "dv_shares_raw_storage": tensors_share_storage(raw_cast[2], raw[2]),
        }
        row["pass"] = (
            row["dq_intentionally_aliases_raw"]
            and not row["dk_shares_raw_storage"]
            and not row["dv_shares_raw_storage"]
        )
        raw_cast_storage_rows.append(row)
    raw_cast_storage = {
        "policy": "raw_cast dQ aliases raw; BF16 dK/dV use new storage",
        "rows": raw_cast_storage_rows,
        "pass": len(raw_cast_storage_rows) == 3
        and all(row["pass"] for row in raw_cast_storage_rows)
        and raw_cast_bf16_aliases["pass"],
        "bf16_dk_dv_non_alias_contract": raw_cast_bf16_aliases,
    }
    immutable = {
        key: input_immutability_contract(
            base, data_by_dataset[key], input_snapshots[key]
        )
        for key in datasets
    }
    metrics_pass = all(fused.rows_pass(rows) for rows in metrics.values())
    control_metrics_pass = all(
        fused.rows_pass(rows) for rows in control_metrics.values()
    )
    control_contracts_pass = all(
        contract["pass"]
        for dataset_contracts in control_contracts.values()
        for contract in dataset_contracts.values()
    )
    expected_dataset_keys = set(datasets)
    dataset_coverage_pass = all(
        set(mapping) == expected_dataset_keys
        for mapping in (
            data_by_dataset,
            input_snapshots,
            references,
            reference_snapshots,
            controls,
        )
    )
    output_map_pass = (
        len(outputs) == 6
        and set(outputs) == set(output_dataset_keys)
        and set(output_snapshots) == set(outputs)
        and set(output_dataset_keys.values()) == expected_dataset_keys
        and set(control_snapshots)
        == {
            f"{dataset_key}_{control_name}"
            for dataset_key in datasets
            for control_name in ("raw", "raw_cast", "uncached")
        }
    )
    correctness = (
        stream_handles_distinct
        and dataset_seeds_distinct
        and dataset_coverage_pass
        and output_map_pass
        and all(contract["pass"] for contract in contracts.values())
        and metrics_pass
        and control_contracts_pass
        and control_metrics_pass
        and all(contract["pass"] for contract in input_reference.values())
        and bitwise["pass"]
        and dq_match["pass"]
        and primary_repeat_consistency["pass"]
        and transition_snapshots_immutable["pass"]
        and control_snapshots_immutable["pass"]
        and references_immutable["pass"]
        and aliases["pass"]
        and input_dataset_storage["pass"]
        and raw_cast_storage["pass"]
        and all(contract["pass"] for contract in immutable.values())
    )
    return {
        "correctness_pass": correctness,
        "gate_checks": {
            "stream_handles_distinct": stream_handles_distinct,
            "dataset_seeds_distinct": dataset_seeds_distinct,
            "all_three_datasets_covered": dataset_coverage_pass,
            "output_contracts": all(
                contract["pass"] for contract in contracts.values()
            ),
            "output_count_and_dataset_map": output_map_pass,
            "reference_error_pass": metrics_pass,
            "control_output_contracts": control_contracts_pass,
            "control_reference_error_pass": control_metrics_pass,
            "input_reference_contracts": all(
                contract["pass"] for contract in input_reference.values()
            ),
            "dk_dv_bitwise": bitwise["pass"],
            "dq_match": dq_match["pass"],
            "primary_cached_repeat_consistency": primary_repeat_consistency[
                "pass"
            ],
            "transition_output_immutability": transition_snapshots_immutable[
                "pass"
            ],
            "control_output_immutability": control_snapshots_immutable["pass"],
            "reference_immutability": references_immutable["pass"],
            "non_alias": aliases["pass"],
            "input_dataset_storage_distinct": input_dataset_storage["pass"],
            "raw_cast_storage_policy": raw_cast_storage["pass"],
            "input_immutability": all(
                contract["pass"] for contract in immutable.values()
            ),
        },
        "output_dataset_metadata": {
            name: {
                "dataset_key": dataset_key,
                **datasets[dataset_key],
                "dq_dk_dv_shapes": [list(tensor.shape) for tensor in outputs[name]],
            }
            for name, dataset_key in output_dataset_keys.items()
        },
        "output_contracts": contracts,
        "metrics_vs_cute": metrics,
        "control_output_contracts": control_contracts,
        "control_metrics_vs_cute": control_metrics,
        "input_reference_contracts": input_reference,
        "dk_dv_bitwise": bitwise,
        "dq_match": dq_match,
        "primary_cached_repeat_consistency": primary_repeat_consistency,
        "transition_output_immutability": transition_snapshots_immutable,
        "control_output_immutability": control_snapshots_immutable,
        "reference_immutability": references_immutable,
        "non_alias_contract": aliases,
        "input_dataset_storage_contract": input_dataset_storage,
        "raw_cast_storage_contract": raw_cast_storage,
        "input_immutability_contracts": immutable,
    }


def run_transitions(base, fused, extension, args) -> dict:
    torch = base.torch
    data_primary, raw_fn, uncached_fn, cached_fn = make_context(
        base, extension, args.seed, 1024, 8
    )
    data_shape_change, _, _, _ = make_context(
        base, extension, args.seed + 1, 4096, 1
    )
    data_peer, _, _, _ = make_context(
        base, extension, args.seed + 2, 1024, 8
    )
    dataset_seeds = {
        "primary": args.seed,
        "shape_change": args.seed + 1,
        "peer": args.seed + 2,
    }
    dataset_seeds_distinct = len(set(dataset_seeds.values())) == 3
    if not dataset_seeds_distinct:
        raise RuntimeError("transition datasets must use three distinct seeds")
    data_by_dataset = {
        "primary": data_primary,
        "shape_change": data_shape_change,
        "peer": data_peer,
    }
    input_snapshots = {
        key: capture_input_snapshot(data)
        for key, data in data_by_dataset.items()
    }
    current_stream = torch.cuda.current_stream()

    references = {}
    reference_snapshots = {}
    with torch.cuda.stream(current_stream):
        for dataset_key, data in data_by_dataset.items():
            reference = base.call_cute(data)
            references[dataset_key] = reference
            reference_snapshots[dataset_key] = tuple(
                tensor.clone() for tensor in reference
            )
    current_stream.synchronize()

    with torch.cuda.stream(current_stream):
        same_primary_first = base.call_tk(cached_fn, data_primary, 1024)
        same_primary_first_snapshot = tuple(
            tensor.clone() for tensor in same_primary_first
        )
        same_shape_change = base.call_tk(
            cached_fn, data_shape_change, 4096
        )
        same_shape_change_snapshot = tuple(
            tensor.clone() for tensor in same_shape_change
        )
        same_primary_second = base.call_tk(cached_fn, data_primary, 1024)
        same_primary_second_snapshot = tuple(
            tensor.clone() for tensor in same_primary_second
        )
    current_stream.synchronize()

    stream_a = torch.cuda.Stream()
    stream_b = torch.cuda.Stream()
    stream_a_handle = int(stream_a.cuda_stream)
    stream_b_handle = int(stream_b.cuda_stream)
    stream_handles_distinct = stream_a_handle != stream_b_handle
    if not stream_handles_distinct:
        raise RuntimeError("transition streams must have distinct CUDA stream handles")
    with torch.cuda.stream(stream_a):
        active_primary_first = base.call_tk(cached_fn, data_primary, 1024)
        active_primary_first_snapshot = tuple(
            tensor.clone() for tensor in active_primary_first
        )
    stream_a.synchronize()
    with torch.cuda.stream(stream_b):
        active_peer = base.call_tk(cached_fn, data_peer, 1024)
        active_peer_snapshot = tuple(tensor.clone() for tensor in active_peer)
    stream_b.synchronize()
    with torch.cuda.stream(stream_a):
        active_primary_second = base.call_tk(cached_fn, data_primary, 1024)
        active_primary_second_snapshot = tuple(
            tensor.clone() for tensor in active_primary_second
        )
    stream_a.synchronize()

    outputs = {
        "same_primary_first": same_primary_first,
        "same_shape_change": same_shape_change,
        "same_primary_second": same_primary_second,
        "active_primary_first": active_primary_first,
        "active_peer": active_peer,
        "active_primary_second": active_primary_second,
    }
    output_snapshots = {
        "same_primary_first": same_primary_first_snapshot,
        "same_shape_change": same_shape_change_snapshot,
        "same_primary_second": same_primary_second_snapshot,
        "active_primary_first": active_primary_first_snapshot,
        "active_peer": active_peer_snapshot,
        "active_primary_second": active_primary_second_snapshot,
    }
    output_dataset_keys = {
        "same_primary_first": "primary",
        "same_shape_change": "shape_change",
        "same_primary_second": "primary",
        "active_primary_first": "primary",
        "active_peer": "peer",
        "active_primary_second": "primary",
    }

    controls = {}
    control_snapshots = {}
    with torch.cuda.stream(current_stream):
        for dataset_key, data, seqlen in (
            ("primary", data_primary, 1024),
            ("shape_change", data_shape_change, 4096),
            ("peer", data_peer, 1024),
        ):
            raw = base.call_tk(raw_fn, data, seqlen)
            control_snapshots[f"{dataset_key}_raw"] = tuple(
                tensor.clone() for tensor in raw
            )
            raw_cast = fused.materialize_parent(base, raw)
            control_snapshots[f"{dataset_key}_raw_cast"] = tuple(
                tensor.clone() for tensor in raw_cast
            )
            uncached = base.call_tk(uncached_fn, data, seqlen)
            control_snapshots[f"{dataset_key}_uncached"] = tuple(
                tensor.clone() for tensor in uncached
            )
            controls[dataset_key] = {
                "raw": raw,
                "raw_cast": raw_cast,
                "uncached": uncached,
            }
    current_stream.synchronize()

    evaluation = evaluate_cached_sequence(
        base,
        fused,
        data_by_dataset,
        input_snapshots,
        references,
        reference_snapshots,
        outputs,
        output_snapshots,
        output_dataset_keys,
        controls,
        control_snapshots,
        stream_handles_distinct,
        dataset_seeds_distinct,
    )
    correctness = evaluation["correctness_pass"]
    return {
        **result_base(args, base),
        "gate_pass": correctness,
        "gate_scope": "cache_shape_and_owner_stream_transitions",
        **fused.contract_fields(correctness),
        "stream_handles": {
            "same_stream": int(current_stream.cuda_stream),
            "stream_a": stream_a_handle,
            "stream_b": stream_b_handle,
            "stream_a_b_distinct": stream_handles_distinct,
        },
        "transition_datasets": {
            dataset_key: {
                "seed": dataset_seeds[dataset_key],
                "seqlen": 4096 if dataset_key == "shape_change" else 1024,
                "heads": 1 if dataset_key == "shape_change" else 8,
            }
            for dataset_key in ("primary", "shape_change", "peer")
        },
        "dataset_seeds_distinct": dataset_seeds_distinct,
        "transition_protocol": {
            "same_stream": {
                "stream": int(current_stream.cuda_stream),
                "dataset_order": [
                    "primary",
                    "shape_change",
                    "primary",
                ],
                "shape_order": ["S1024H8", "S4096H1", "S1024H8"],
                "isolated_cache_key_change": "shape",
                "intermediate_device_sync": False,
                "intermediate_stream_sync": False,
                "completion": "active current stream synchronize after final leg",
            },
            "active_streams": {
                "stream_a": stream_a_handle,
                "stream_b": stream_b_handle,
                "stream_handles_distinct": stream_handles_distinct,
                "dataset_order": ["primary", "peer", "primary"],
                "stream_order": ["A", "B", "A"],
                "shape_order": ["S1024H8", "S1024H8", "S1024H8"],
                "fixed_shape": {"seqlen": 1024, "heads": 8},
                "isolated_cache_key_change": "owner_stream",
                "each_leg_active_stream_synchronized_before_next_cache_refresh": True,
                "intermediate_device_sync": False,
            },
            "device_wide_synchronize_calls": 0,
            "snapshot_timing": "immediately after each return on producing stream",
            "observation_before_permitted_stream_sync": False,
        },
        **evaluation,
    }


def timing_precheck(base, fused, extension, args):
    torch = base.torch
    data, raw_fn, uncached_fn, cached_fn = make_context(
        base, extension, args.seed, args.seqlen, args.heads
    )
    snapshots = capture_input_snapshot(data)
    reference = base.call_cute(data)
    torch.cuda.synchronize()
    cached = base.call_tk(cached_fn, data, args.seqlen)
    uncached = base.call_tk(uncached_fn, data, args.seqlen)
    raw = base.call_tk(raw_fn, data, args.seqlen)
    torch.cuda.synchronize()
    evaluation = evaluate_materialized_outputs(
        base,
        fused,
        args,
        data,
        snapshots,
        reference,
        raw,
        {"cached": cached, "uncached": uncached},
    )
    del reference, cached, uncached, raw
    return data, snapshots, raw_fn, uncached_fn, cached_fn, evaluation


def time_routes(base, fused, args, routes: dict) -> tuple[dict, list]:
    torch = base.torch
    orders = list(itertools.permutations(routes))
    for warmup_index in range(args.warmups):
        order = orders[(warmup_index + args.order_offset) % len(orders)]
        for name in order:
            torch.cuda.synchronize()
            result = routes[name]()
            torch.cuda.synchronize()
            del result
    values = {name: [] for name in routes}
    raw_samples = []
    for sample_index in range(args.samples):
        permutation_index = (sample_index + args.order_offset) % len(orders)
        order = list(orders[permutation_index])
        sample = {
            "sample_index": sample_index,
            "permutation_index": permutation_index,
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
            atomic_write_json(
                args.output,
                {
                    **args.manifest,
                    "samples_requested": args.samples,
                    "samples_completed": sample_index + 1,
                    "partial": {
                        name: fused.summarize(times)
                        for name, times in values.items()
                    },
                    "raw_samples": raw_samples,
                },
                replace=True,
            )
    return (
        {name: fused.summarize(times) for name, times in values.items()},
        raw_samples,
    )


def run_timing_ab(base, fused, extension, args) -> dict:
    data, snapshots, _, uncached_fn, cached_fn, evaluation = timing_precheck(
        base, fused, extension, args
    )
    if not evaluation["correctness_pass"]:
        return {
            **result_base(args, base),
            "gate_pass": False,
            "gate_scope": "correctness",
            **fused.contract_fields(False),
            "timing_skipped": True,
            "precheck": evaluation,
        }
    routes = {
        "cached_bf16": lambda: base.call_tk(cached_fn, data, args.seqlen),
        "uncached_bf16": lambda: base.call_tk(uncached_fn, data, args.seqlen),
    }
    summaries, raw_samples = time_routes(base, fused, args, routes)
    immutable = input_immutability_contract(base, data, snapshots)
    pair_contract = timing_pair_contract(args)
    correctness = (
        evaluation["correctness_pass"]
        and immutable["pass"]
        and pair_contract["pass"]
    )
    cached_over_uncached = (
        summaries["cached_bf16"]["median_us"]
        / summaries["uncached_bf16"]["median_us"]
    )
    retention = cached_over_uncached < 1.0
    gate_pass = correctness and retention
    fields = fused.contract_fields(correctness, retention)
    fields["overall_contract_scope"] = "correctness_and_cached_over_uncached"
    fields["overall_contract_pass"] = gate_pass
    return {
        **result_base(args, base),
        "gate_pass": gate_pass,
        "gate_scope": "correctness_and_cached_over_uncached",
        **fields,
        "gate_checks": {
            "correctness_precheck": evaluation["correctness_pass"],
            "post_timing_input_immutability": immutable["pass"],
            "paired_replicate_offset_contract": pair_contract["pass"],
            "cached_over_uncached": retention,
        },
        "precheck": evaluation,
        "post_timing_input_immutability": immutable,
        "timing_protocol": {
            "clock": "time.perf_counter_ns",
            "device_sync_immediately_before_route": True,
            "device_sync_immediately_after_route": True,
            "primary_routes": list(routes),
            "permutation_count": 2,
            "paired_replicate_offset_contract": pair_contract,
            "progress_io_inside_intervals": False,
        },
        "samples": args.samples,
        "warmups": args.warmups,
        "route_timings": summaries,
        "ratios": {"cached_over_uncached": cached_over_uncached},
        "raw_samples": raw_samples,
    }


def run_timing_canonical(base, fused, extension, args) -> dict:
    data, snapshots, raw_fn, _, cached_fn, evaluation = timing_precheck(
        base, fused, extension, args
    )
    if not evaluation["correctness_pass"]:
        return {
            **result_base(args, base),
            "gate_pass": False,
            "gate_scope": "correctness",
            **fused.contract_fields(False),
            "timing_skipped": True,
            "precheck": evaluation,
        }
    routes = {
        "cached_bf16": lambda: base.call_tk(cached_fn, data, args.seqlen),
        "raw_p": lambda: base.call_tk(raw_fn, data, args.seqlen),
        "cute": lambda: base.call_cute(data),
    }
    summaries, raw_samples = time_routes(base, fused, args, routes)
    immutable = input_immutability_contract(base, data, snapshots)
    pair_contract = timing_pair_contract(args)
    correctness = (
        evaluation["correctness_pass"]
        and immutable["pass"]
        and pair_contract["pass"]
    )
    cached_over_raw = (
        summaries["cached_bf16"]["median_us"]
        / summaries["raw_p"]["median_us"]
    )
    cached_over_cute = (
        summaries["cached_bf16"]["median_us"]
        / summaries["cute"]["median_us"]
    )
    objective = cached_over_cute < 1.0
    gate_pass = correctness and objective
    return {
        **result_base(args, base),
        "gate_pass": gate_pass,
        "gate_scope": "correctness_and_cute_objective",
        **fused.contract_fields(correctness, None, objective),
        "gate_checks": {
            "correctness_precheck": evaluation["correctness_pass"],
            "post_timing_input_immutability": immutable["pass"],
            "paired_replicate_offset_contract": pair_contract["pass"],
            "cached_over_cute": objective,
        },
        "precheck": evaluation,
        "post_timing_input_immutability": immutable,
        "timing_protocol": {
            "clock": "time.perf_counter_ns",
            "device_sync_immediately_before_route": True,
            "device_sync_immediately_after_route": True,
            "primary_routes": list(routes),
            "permutation_count": 6,
            "paired_replicate_offset_contract": pair_contract,
            "progress_io_inside_intervals": False,
        },
        "samples": args.samples,
        "warmups": args.warmups,
        "route_timings": summaries,
        "ratios": {
            "cached_over_raw_p": cached_over_raw,
            "cached_over_cute": cached_over_cute,
        },
        "raw_p_control": {
            "api_contract_valid": False,
            "gate_role": "diagnostic_only",
        },
        "raw_samples": raw_samples,
    }


def add_output_seed(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)


def add_shape(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seqlen", required=True, type=int)
    parser.add_argument("--heads", required=True, type=int)


def add_timing(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--samples", type=int, default=TIMING_SAMPLES)
    parser.add_argument("--warmups", type=int, default=TIMING_WARMUPS)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--replicate", required=True, type=int, choices=(1, 2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    onecall = subparsers.add_parser("onecall")
    add_output_seed(onecall)
    add_shape(onecall)

    stress = subparsers.add_parser("stress")
    add_output_seed(stress)
    add_shape(stress)
    stress.add_argument("--calls", required=True, type=int)
    stress.add_argument("--progress-every", type=int, default=100)

    transitions = subparsers.add_parser("transitions")
    add_output_seed(transitions)

    timing_ab = subparsers.add_parser("timing_ab")
    add_output_seed(timing_ab)
    add_shape(timing_ab)
    add_timing(timing_ab)

    timing_canonical = subparsers.add_parser("timing_canonical")
    add_output_seed(timing_canonical)
    add_shape(timing_canonical)
    add_timing(timing_canonical)

    args = parser.parse_args()
    if args.seed <= 0:
        parser.error("--seed must be positive")
    output = args.output.resolve()
    results_root = (REPO / "results").resolve()
    if output.suffix != ".json" or not output.is_relative_to(results_root):
        parser.error("--output must be a JSON path under results/")
    args.output = output
    if args.mode != "transitions" and (args.seqlen, args.heads) not in SUPPORTED_SHAPES:
        parser.error("shape must be S1024/H8 or S4096/H1")
    if args.mode == "stress":
        if args.calls not in STRESS_CALL_COUNTS:
            parser.error("stress requires exactly 50 or 2000 calls")
        if args.calls == 2000 and args.seed != STRESS_2000_SEED:
            parser.error(f"2000-call stress requires seed {STRESS_2000_SEED}")
        if args.progress_every <= 0:
            parser.error("--progress-every must be positive")
    if args.mode in ("timing_ab", "timing_canonical"):
        if args.samples != TIMING_SAMPLES:
            parser.error(f"timing requires exactly {TIMING_SAMPLES} samples")
        if args.warmups != TIMING_WARMUPS:
            parser.error(f"timing requires exactly {TIMING_WARMUPS} warmups")
        if args.progress_every <= 0:
            parser.error("--progress-every must be positive")
        paired_offsets = (0, 1) if args.mode == "timing_ab" else (0, 3)
        args.order_offset = paired_offsets[args.replicate - 1]
        if not timing_pair_contract(args)["pass"]:
            parser.error("timing replicate did not resolve to its fixed paired offset")
    return args


def main() -> None:
    args = parse_args()
    bootstrap = fallback_manifest(args)
    atomic_write_json(args.output, bootstrap, replace=False)
    manifest = bootstrap
    try:
        manifest = initial_manifest(args)
        atomic_write_json(args.output, manifest, replace=True)
        fused, dependency_gate_at_fused_import = load_audited_fused_harness()
        fused_dependency_contract = verify_fused_dependency_contract(fused)
        dependency_gate_before_base_loader = gate_audited_dependency_files()
        manifest = {
            **manifest,
            "audited_dependency_hash_gate_at_fused_import": (
                dependency_gate_at_fused_import
            ),
            "fused_exposed_dependency_contract": fused_dependency_contract,
            "audited_dependency_hash_gate_before_base_loader": (
                dependency_gate_before_base_loader
            ),
        }
        atomic_write_json(args.output, manifest, replace=True)
        base = fused.load_base_harness()
        base.torch.cuda.set_device(0)
        extension, loaded_extension = load_cached_extension(base)
        args.manifest = {**manifest, "loaded_extension": loaded_extension}
        atomic_write_json(args.output, args.manifest, replace=True)
        runner = {
            "onecall": run_onecall,
            "stress": run_stress,
            "transitions": run_transitions,
            "timing_ab": run_timing_ab,
            "timing_canonical": run_timing_canonical,
        }[args.mode]
        result = runner(base, fused, extension, args)
        atomic_write_json(args.output, result, replace=True)
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
    except SystemExit:
        raise
    except BaseException as error:
        failure = {
            **manifest,
            "complete": False,
            "state": "error",
            "gate_pass": False,
            "gate_scope": "runtime_error",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        atomic_write_json(args.output, failure, replace=True)
        raise


if __name__ == "__main__":
    main()
