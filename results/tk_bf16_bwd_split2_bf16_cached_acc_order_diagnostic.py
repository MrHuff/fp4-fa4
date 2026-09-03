from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
AUDITED_HARNESS = (
    REPO / "results" / "tk_bf16_bwd_split2_bf16_cached_acc_harness.py"
)
EXPECTED_HARNESS_SHA256 = (
    "70d76933b0effa3a902646fda773ac3df4af106534d9a8962f2ea432a2fe4450"
)
EXTENSION = (
    REPO
    / "results"
    / ".artifacts"
    / "tk_bf16_bwd_split2_bf16_cached_acc"
    / "_C.cpython-312-aarch64-linux-gnu.so"
)
EXPECTED_EXTENSION_SHA256 = (
    "264b4f8bc8cd314946506c68169bb1753df140a120582e752edd5670a2976d78"
)
SCENARIOS = (
    "uncached_alone",
    "cached_device_sync_uncached",
    "cached_current_stream_sync_uncached",
    "cached_uncached_no_sync",
    "cached_uncached_raw_no_sync",
    "cached_uncached_cached_no_sync",
    "cached_uncached_device_sync_raw",
    "cached_uncached_current_stream_sync_raw",
    "uncached_raw_no_sync",
    "uncached_cached_no_sync",
    "original_four_no_sync",
)
PREAMBLES = (
    "stable_device_sync",
    "clone_device_sync",
    "clone_current_stream_sync",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict, *, replace: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
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
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument(
        "--preamble", choices=PREAMBLES, default="stable_device_sync"
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.seed <= 0:
        parser.error("--seed must be positive")
    if args.output.suffix != ".json" or not args.output.is_relative_to(
        (REPO / "results").resolve()
    ):
        parser.error("--output must be a JSON path under results/")
    return args


def load_audited_harness():
    actual_sha = sha256_file(AUDITED_HARNESS)
    if actual_sha != EXPECTED_HARNESS_SHA256:
        raise RuntimeError(
            f"audited harness SHA mismatch: expected {EXPECTED_HARNESS_SHA256}, "
            f"got {actual_sha}"
        )
    spec = importlib.util.spec_from_file_location(
        "tk_bf16_bwd_cached_acc_order_diagnostic_base",
        AUDITED_HARNESS,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import audited harness: {AUDITED_HARNESS}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tensor_storage(tensor) -> dict:
    storage = tensor.untyped_storage()
    return {
        "storage_data_ptr": int(storage.data_ptr()),
        "storage_nbytes": int(storage.nbytes()),
        "storage_device": str(storage.device),
        "tensor_data_ptr": int(tensor.data_ptr()),
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "contiguous": bool(tensor.is_contiguous()),
    }


def output_storage(outputs) -> dict:
    return {
        name: tensor_storage(tensor)
        for name, tensor in zip(("dq", "dk", "dv"), outputs)
    }


def output_immutability(torch, outputs: dict, snapshots: dict) -> dict:
    rows = []
    for route_name, route_outputs in outputs.items():
        route_snapshot = snapshots[route_name]
        for index, output_name in enumerate(("dq", "dk", "dv")):
            equal = bool(torch.equal(route_outputs[index], route_snapshot[index]))
            rows.append(
                {
                    "route": route_name,
                    "output": output_name,
                    "equal": equal,
                }
            )
    return {
        "rows": rows,
        "pass": len(rows) == 3 * len(outputs)
        and all(row["equal"] for row in rows),
    }


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 1")
    tk_environment = {
        name: value
        for name, value in os.environ.items()
        if name.startswith("TK_FA4_")
    }
    if tk_environment:
        raise RuntimeError(
            "TK_FA4 probe/timing variables must be absent: "
            + ", ".join(sorted(tk_environment))
        )
    initial = {
        "schema_version": 1,
        "complete": False,
        "state": "initializing",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "scenario": args.scenario,
        "preamble": args.preamble,
        "seed": args.seed,
        "cuda_visible_devices": "1",
        "audited_harness": str(AUDITED_HARNESS),
        "expected_audited_harness_sha256": EXPECTED_HARNESS_SHA256,
        "extension": str(EXTENSION),
        "expected_extension_sha256": EXPECTED_EXTENSION_SHA256,
    }
    write_json(args.output, initial, replace=False)
    manifest = initial
    try:
        harness_sha = sha256_file(AUDITED_HARNESS)
        extension_sha = sha256_file(EXTENSION)
        if harness_sha != EXPECTED_HARNESS_SHA256:
            raise RuntimeError("audited harness changed after initial manifest")
        if extension_sha != EXPECTED_EXTENSION_SHA256:
            raise RuntimeError(
                f"extension SHA mismatch: expected {EXPECTED_EXTENSION_SHA256}, "
                f"got {extension_sha}"
            )
        harness = load_audited_harness()
        fused, dependency_gate = harness.load_audited_fused_harness()
        fused_contract = harness.verify_fused_dependency_contract(fused)
        dependency_before_base = harness.gate_audited_dependency_files()
        base = fused.load_base_harness()
        base.torch.cuda.set_device(0)
        extension, extension_record = harness.load_cached_extension(base)
        torch = base.torch
        data, raw_fn, uncached_fn, cached_fn = harness.make_context(
            base, extension, args.seed, 1024, 8
        )
        input_snapshot = harness.capture_input_snapshot(data)
        reference = base.call_cute(data)
        reference_snapshot = None
        if args.preamble != "stable_device_sync":
            reference_snapshot = tuple(tensor.clone() for tensor in reference)
        if args.preamble == "clone_current_stream_sync":
            torch.cuda.current_stream().synchronize()
        else:
            torch.cuda.synchronize()

        outputs = {}
        call_order = []
        sync_points = []

        def call(label, function):
            result = base.call_tk(function, data, 1024)
            outputs[label] = result
            call_order.append(label)

        if args.scenario == "uncached_alone":
            call("uncached", uncached_fn)
        elif args.scenario == "cached_device_sync_uncached":
            call("cached", cached_fn)
            torch.cuda.synchronize()
            sync_points.append("device_after_cached")
            call("uncached", uncached_fn)
        elif args.scenario == "cached_current_stream_sync_uncached":
            call("cached", cached_fn)
            torch.cuda.current_stream().synchronize()
            sync_points.append("current_stream_after_cached")
            call("uncached", uncached_fn)
        elif args.scenario == "cached_uncached_no_sync":
            call("cached", cached_fn)
            call("uncached", uncached_fn)
        elif args.scenario == "cached_uncached_raw_no_sync":
            call("cached", cached_fn)
            call("uncached", uncached_fn)
            call("raw_in_sequence", raw_fn)
        elif args.scenario == "cached_uncached_cached_no_sync":
            call("cold_cached", cached_fn)
            call("uncached", uncached_fn)
            call("warm_cached", cached_fn)
        elif args.scenario == "cached_uncached_device_sync_raw":
            call("cached", cached_fn)
            call("uncached", uncached_fn)
            torch.cuda.synchronize()
            sync_points.append("device_after_uncached")
            call("raw_in_sequence", raw_fn)
        elif args.scenario == "cached_uncached_current_stream_sync_raw":
            call("cached", cached_fn)
            call("uncached", uncached_fn)
            torch.cuda.current_stream().synchronize()
            sync_points.append("current_stream_after_uncached")
            call("raw_in_sequence", raw_fn)
        elif args.scenario == "uncached_raw_no_sync":
            call("uncached", uncached_fn)
            call("raw_in_sequence", raw_fn)
        elif args.scenario == "uncached_cached_no_sync":
            call("uncached", uncached_fn)
            call("cached", cached_fn)
        elif args.scenario == "original_four_no_sync":
            call("cold_cached", cached_fn)
            call("uncached", uncached_fn)
            call("raw_in_sequence", raw_fn)
            call("warm_cached", cached_fn)
        else:
            raise AssertionError(f"unhandled scenario: {args.scenario}")

        torch.cuda.synchronize()
        sync_points.append("device_after_sequence")
        if reference_snapshot is None:
            reference_snapshot = tuple(tensor.clone() for tensor in reference)
        sequence_snapshots = {
            name: tuple(tensor.clone() for tensor in route_outputs)
            for name, route_outputs in outputs.items()
        }
        torch.cuda.current_stream().synchronize()
        sync_points.append("current_stream_after_sequence_snapshot")
        immutable_after_sequence = output_immutability(
            torch, outputs, sequence_snapshots
        )
        contracts = {
            name: fused.output_contract(
                base,
                result,
                data,
                child=name != "raw_in_sequence",
            )
            for name, result in outputs.items()
        }
        metrics_before_raw = {
            name: base.metrics(result, reference)
            for name, result in outputs.items()
        }

        raw_control = base.call_tk(raw_fn, data, 1024)
        raw_control_snapshot = tuple(tensor.clone() for tensor in raw_control)
        raw_cast = fused.materialize_parent(base, raw_control)
        raw_cast_snapshot = tuple(tensor.clone() for tensor in raw_cast)
        torch.cuda.synchronize()
        sync_points.append("device_after_raw_control")
        immutable_after_raw = output_immutability(
            torch, outputs, sequence_snapshots
        )
        reference_immutable = all(
            bool(torch.equal(tensor, snapshot))
            for tensor, snapshot in zip(reference, reference_snapshot)
        )
        raw_immutable = all(
            bool(torch.equal(tensor, snapshot))
            for tensor, snapshot in zip(raw_control, raw_control_snapshot)
        )
        raw_cast_immutable = all(
            bool(torch.equal(tensor, snapshot))
            for tensor, snapshot in zip(raw_cast, raw_cast_snapshot)
        )
        comparisons = {}
        for name, result in outputs.items():
            comparisons[name] = {
                "dq_max_abs_vs_raw": float(
                    (result[0].float() - raw_control[0].float()).abs().max()
                ),
                "dk_bitwise_raw_cast": bool(torch.equal(result[1], raw_cast[1])),
                "dv_bitwise_raw_cast": bool(torch.equal(result[2], raw_cast[2])),
            }
        input_immutable = harness.input_immutability_contract(
            base, data, input_snapshot
        )
        input_reference = fused.input_reference_contract(
            base, data, reference, harness.shape_args(1024, 8)
        )
        result = {
            **manifest,
            "complete": True,
            "state": "finished",
            "audited_harness_sha256": harness_sha,
            "extension_sha256": extension_sha,
            "extension_record": extension_record,
            "dependency_gate_at_fused_import": dependency_gate,
            "dependency_gate_before_base": dependency_before_base,
            "fused_dependency_contract": fused_contract,
            "call_order": call_order,
            "sync_points": sync_points,
            "inter_call_snapshot_or_observation": False,
            "sequence_snapshot_after_device_sync": True,
            "outputs_preserved_until_final_comparison": True,
            "output_contracts": contracts,
            "metrics_vs_cute_before_raw_control": metrics_before_raw,
            "comparisons_vs_raw_cast": comparisons,
            "output_immutability_after_sequence": immutable_after_sequence,
            "output_immutability_after_raw_control": immutable_after_raw,
            "reference_immutable": reference_immutable,
            "raw_control_immutable": raw_immutable,
            "raw_cast_immutable": raw_cast_immutable,
            "input_immutability": input_immutable,
            "input_reference_contract": input_reference,
            "output_storage": {
                name: output_storage(route_outputs)
                for name, route_outputs in outputs.items()
            },
            "raw_control_storage": output_storage(raw_control),
            "raw_cast_storage": output_storage(raw_cast),
        }
        write_json(args.output, result, replace=True)
        print(
            json.dumps(
                {
                    "scenario": args.scenario,
                    "comparisons": comparisons,
                    "metrics": metrics_before_raw,
                    "immutability": immutable_after_raw["pass"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    except BaseException as error:
        failure = {
            **manifest,
            "complete": False,
            "state": "error",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
        write_json(args.output, failure, replace=True)
        raise


if __name__ == "__main__":
    main()
