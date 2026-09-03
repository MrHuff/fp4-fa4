#!/usr/bin/env python3
"""Measure MX/FP8 aggregate backward with one fixed model and one runner.

The saturated route benchmark intentionally uses one process per training
trajectory.  That is appropriate for numerical comparisons, but a sub-ms
backward attribution can be confounded by process, allocation, and parameter
trajectory differences.  This harness instead fixes the model, inputs, and
backward implementation/storage ownership:

* one Llama-1.2B model allocation and one immutable parameter state;
* one compiled low-precision FA4 backward runner and its workspaces;
* one fixed token/target batch; and
* balanced adjacent MX/FP8 pairs in ABBA/BAAB superblocks.

Only the construction-bound forward runtime is crossed over.  No optimizer is
constructed and no parameter update occurs.  Dynamic backward operands remain
route-natural because MX and FP8 forward outputs differ.  This is a timing-
parity diagnostic, not a claim of byte-identical dynamic state, convergence,
or training throughput.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from tk_fa4 import interface as tk_interface
from tk_fa4.lowp_fa4_bwd import backward_contract as backward_contract_module
from tk_fa4.lowp_fa4_bwd import (
    benchmark_llama12b_e2e as runtime_module,
)
from tk_fa4.lowp_fa4_bwd import (
    benchmark_llama12b_saturated as saturated_helpers_module,
)
from tk_fa4.lowp_fa4_bwd.backward_contract import (
    require_matching_backward_contracts,
    require_shared_backward_physical_identity,
)
from tk_fa4.lowp_fa4_bwd.benchmark_llama12b_e2e import (
    Llama12B,
    LowpAttentionRuntime,
    _load_forward,
    _make_llama3_rope,
    activate_model_forward_route,
    config_from_model_preset,
)
from tk_fa4.lowp_fa4_bwd.benchmark_llama12b_saturated import (
    DEFAULT_CONTROL,
    DEFAULT_FORWARDS,
    DEFAULT_PROJECTION,
    FORWARD_MODULES,
    PINNED_ARTIFACTS,
    _batch,
    _file_identity,
    _hardware_identity,
    _hidden_and_weight,
    _load_initial_checkpoint,
    _loss,
    _projection_expected_identity,
    _source_identity,
)


MX_ROUTE = "mx"
FP8_ROUTE = "fp8"
ROUTES = (MX_ROUTE, FP8_ROUTE)
MINIMUM_SUPERBLOCKS = 24
DEFAULT_RELATIVE_TOLERANCE = 0.01
LOSS_INVARIANCE_ATOL = 1.0e-6


def _balanced_abba_order(superblock_index: int) -> tuple[str, ...]:
    """Return an eight-call order with four drift-local MX/FP8 pairs."""
    if superblock_index < 0:
        raise ValueError("superblock index must be nonnegative")
    primary = (
        MX_ROUTE,
        FP8_ROUTE,
        FP8_ROUTE,
        MX_ROUTE,
        FP8_ROUTE,
        MX_ROUTE,
        MX_ROUTE,
        FP8_ROUTE,
    )
    complement = tuple(
        FP8_ROUTE if route == MX_ROUTE else MX_ROUTE for route in primary
    )
    complemented = (superblock_index % 2) ^ (
        (superblock_index // 2) % 2
    )
    return complement if complemented else primary


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("percentile fraction must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _sample_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("sample summary requires at least one value")
    converted = [float(value) for value in values]
    return {
        "samples": len(converted),
        "mean": statistics.fmean(converted),
        "p05": _percentile(converted, 0.05),
        "p50": statistics.median(converted),
        "p95": _percentile(converted, 0.95),
        "minimum": min(converted),
        "maximum": max(converted),
        "stdev": statistics.stdev(converted) if len(converted) > 1 else 0.0,
    }


def _adjacent_pair_deltas(
    records: list[dict[str, Any]],
    metric: str,
) -> list[float]:
    """Return oriented MX-minus-FP8 deltas for adjacent measured calls."""
    if len(records) % 8:
        raise RuntimeError("records must contain complete eight-call superblocks")
    deltas: list[float] = []
    for block_start in range(0, len(records), 8):
        block = records[block_start : block_start + 8]
        expected_block = block_start // 8
        expected_order = _balanced_abba_order(expected_block)
        actual_order = tuple(str(record["route"]) for record in block)
        if actual_order != expected_order:
            raise RuntimeError(
                f"superblock {expected_block} order mismatch: "
                f"{actual_order} != {expected_order}"
            )
        for position, record in enumerate(block):
            expected_global_index = block_start + position
            metadata = (
                int(record["superblock"]),
                int(record["position"]),
                int(record["global_call_index"]),
            )
            expected_metadata = (
                expected_block,
                position,
                expected_global_index,
            )
            if metadata != expected_metadata:
                raise RuntimeError(
                    f"record {expected_global_index} metadata mismatch: "
                    f"{metadata} != {expected_metadata}"
                )
        for pair_start in range(0, 8, 2):
            first, second = block[pair_start : pair_start + 2]
            pair = {str(first["route"]): first, str(second["route"]): second}
            if set(pair) != set(ROUTES):
                raise RuntimeError(
                    "each adjacent timing pair must contain one MX and one FP8 "
                    "record"
                )
            deltas.append(
                float(pair[MX_ROUTE][metric])
                - float(pair[FP8_ROUTE][metric])
            )
    return deltas


def _superblock_mean_deltas(
    records: list[dict[str, Any]],
    metric: str,
) -> list[float]:
    """Collapse four drift-local pairs into one independent block mean."""
    pair_deltas = _adjacent_pair_deltas(records, metric)
    return [
        statistics.fmean(pair_deltas[start : start + 4])
        for start in range(0, len(pair_deltas), 4)
    ]


def _superblock_symmetric_relative_effects(
    records: list[dict[str, Any]],
    metric: str,
) -> list[float]:
    """Return one symmetric MX-minus-FP8 relative effect per block."""
    if len(records) % 8:
        raise RuntimeError("records must contain complete eight-call superblocks")
    # Reuse the strict order and metadata validation before collapsing the
    # complete record stream.
    _adjacent_pair_deltas(records, metric)
    effects: list[float] = []
    for block_start in range(0, len(records), 8):
        block = records[block_start : block_start + 8]
        # Route means absorb the four local clock/thermal phases without
        # treating those correlated pairs as independent.
        route_means = {
            route: statistics.fmean(
                float(record[metric])
                for record in block
                if record["route"] == route
            )
            for route in ROUTES
        }
        denominator = 0.5 * (
            route_means[MX_ROUTE] + route_means[FP8_ROUTE]
        )
        if not math.isfinite(denominator) or denominator <= 0.0:
            raise RuntimeError(
                f"{metric} superblock mean must be finite and positive"
            )
        effects.append(
            (route_means[MX_ROUTE] - route_means[FP8_ROUTE])
            / denominator
        )
    return effects


def _paired_bootstrap_interval(
    deltas: list[float],
    *,
    draws: int,
    seed: int,
) -> tuple[float, float]:
    """Return a deterministic percentile interval over superblock means."""
    if len(deltas) < 2:
        raise ValueError("paired bootstrap requires at least two deltas")
    if draws < 1_000:
        raise ValueError("paired bootstrap requires at least 1000 draws")
    generator = random.Random(seed)
    count = len(deltas)
    means = [
        statistics.fmean(deltas[generator.randrange(count)] for _ in range(count))
        for _ in range(draws)
    ]
    return _percentile(means, 0.025), _percentile(means, 0.975)


def _memory_snapshot(stage: str) -> dict[str, float | str]:
    snapshot: dict[str, float | str] = {
        "stage": stage,
        "allocated_gib": torch.cuda.memory_allocated() / 2.0**30,
        "reserved_gib": torch.cuda.memory_reserved() / 2.0**30,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2.0**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2.0**30,
    }
    return snapshot


def _require_hbm_budget(
    stage: str,
    max_hbm_gib: float,
) -> dict[str, float | str]:
    snapshot = _memory_snapshot(stage)
    if float(snapshot["reserved_gib"]) > max_hbm_gib or float(
        snapshot["peak_reserved_gib"]
    ) > max_hbm_gib:
        raise RuntimeError(
            f"{stage} reserved HBM exceeds {max_hbm_gib:.3f} GiB: "
            f"current={float(snapshot['reserved_gib']):.3f} GiB, "
            f"peak={float(snapshot['peak_reserved_gib']):.3f} GiB"
        )
    return snapshot


def _parse_gpu_process_report(report: str) -> list[int]:
    """Parse torch's NVML report and reject every unknown format."""
    if not isinstance(report, str):
        raise RuntimeError("GPU process report must be text")
    lines = [line.strip() for line in report.splitlines() if line.strip()]
    if len(lines) < 2 or re.fullmatch(r"GPU:\d+", lines[0]) is None:
        raise RuntimeError(f"malformed GPU process report: {report!r}")
    entries = lines[1:]
    if entries == ["no processes are running"]:
        return []
    process_ids: list[int] = []
    for entry in entries:
        match = re.fullmatch(
            r"process\s+(\d+)\s+uses\s+"
            r"(?:\d+(?:\.\d+)?)\s+MB\s+GPU\s+memory",
            entry,
        )
        if match is None:
            raise RuntimeError(f"malformed GPU process report: {report!r}")
        process_ids.append(int(match.group(1)))
    if len(process_ids) != len(set(process_ids)):
        raise RuntimeError("GPU process report contains duplicate PIDs")
    return process_ids


def _require_exclusive_visible_gpu() -> dict[str, Any]:
    report = torch.cuda.list_gpu_processes(0)
    process_ids = _parse_gpu_process_report(report)
    own_pid = os.getpid()
    foreign = [pid for pid in process_ids if pid != own_pid]
    if foreign:
        raise RuntimeError(
            "backward-neutrality gate requires an exclusive visible GPU; "
            f"foreign compute PIDs: {foreign}"
        )
    return {
        "report": report,
        "observed_process_ids": process_ids,
        "own_pid": own_pid,
        "foreign_process_ids": foreign,
    }


def _canonical_contract_sha256(contract: dict[str, Any]) -> str:
    encoded = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _benchmark_source_identities() -> dict[str, Any]:
    return {
        "harness": _source_identity(Path(__file__)),
        "runtime": _source_identity(
            Path(__file__).with_name("benchmark_llama12b_e2e.py")
        ),
        "backward_contract": _source_identity(
            Path(__file__).with_name("backward_contract.py")
        ),
        "saturated_helpers": _source_identity(
            Path(__file__).with_name("benchmark_llama12b_saturated.py")
        ),
        "interface": _source_identity(
            Path(__file__).resolve().parents[1] / "interface.py"
        ),
    }


def _loaded_python_module_identity(
    module_name: str,
    module: Any,
    expected_path: Path,
) -> dict[str, Any]:
    """Authenticate the actual imported module, not an assumed worktree."""
    loaded = getattr(module, "__file__", None)
    if loaded is None:
        raise RuntimeError(f"loaded Python module {module_name} has no __file__")
    loaded_path = Path(loaded).resolve()
    resolved_expected = expected_path.resolve()
    if loaded_path != resolved_expected:
        raise RuntimeError(
            f"loaded Python module {module_name} is shadowed: "
            f"{loaded_path} != {resolved_expected}"
        )
    identity = _source_identity(loaded_path)
    identity["module"] = module_name
    identity["loaded_path"] = str(loaded_path)
    return identity


def _loaded_python_module_identities() -> dict[str, Any]:
    source_directory = Path(__file__).resolve().parent
    return {
        "interface": _loaded_python_module_identity(
            "tk_fa4.interface",
            tk_interface,
            source_directory.parent / "interface.py",
        ),
        "runtime": _loaded_python_module_identity(
            "tk_fa4.lowp_fa4_bwd.benchmark_llama12b_e2e",
            runtime_module,
            source_directory / "benchmark_llama12b_e2e.py",
        ),
        "backward_contract": _loaded_python_module_identity(
            "tk_fa4.lowp_fa4_bwd.backward_contract",
            backward_contract_module,
            source_directory / "backward_contract.py",
        ),
        "saturated_helpers": _loaded_python_module_identity(
            "tk_fa4.lowp_fa4_bwd.benchmark_llama12b_saturated",
            saturated_helpers_module,
            source_directory / "benchmark_llama12b_saturated.py",
        ),
    }


def _require_loaded_python_matches_sources(
    loaded_modules: dict[str, Any],
    source_files: dict[str, Any],
) -> None:
    for name, loaded in loaded_modules.items():
        expected = source_files.get(name)
        if expected is None:
            raise RuntimeError(
                f"source manifest omitted loaded Python module {name}"
            )
        comparable_loaded = {
            key: loaded[key] for key in ("path", "sha256", "bytes")
        }
        if comparable_loaded != expected:
            raise RuntimeError(
                f"loaded Python module {name} does not match its source "
                "manifest"
            )


def _workspace_owner_pointer_map(contract: dict[str, Any]) -> list[Any]:
    """Return the route-independent owner allocation identity per layer."""
    layers = contract.get("layers")
    if not isinstance(layers, list) or not layers:
        raise RuntimeError("workspace contract omitted its layer records")
    pointer_map: list[Any] = []
    for expected_layer, layer in enumerate(layers):
        if not isinstance(layer, dict) or layer.get("layer") != expected_layer:
            raise RuntimeError("workspace contract has invalid layer ordering")
        owners = layer.get("owners")
        if not isinstance(owners, dict) or not owners:
            raise RuntimeError(
                f"workspace layer {expected_layer} omitted owner records"
            )
        pointer_map.append(
            {
                str(name): {
                    "data_ptr": int(record["data_ptr"]),
                    "allocation_data_ptr": int(
                        record["allocation_data_ptr"]
                    ),
                    "bytes": int(record["bytes"]),
                    "shape": list(record["shape"]),
                    "dtype": str(record["dtype"]),
                }
                for name, record in sorted(owners.items())
            }
        )
    return pointer_map


def _make_runtime(
    route: str,
    *,
    config: Any,
    rope: tuple[torch.Tensor, torch.Tensor],
    forward_path: Path,
    backward_control: Path,
    shared_backward_runtime: LowpAttentionRuntime | None = None,
) -> tuple[LowpAttentionRuntime, dict[str, Any]]:
    if route not in ROUTES:
        raise ValueError(f"unsupported route {route!r}")
    extension, topology = _load_forward(
        forward_path,
        FORWARD_MODULES[route],
        config,
    )
    runtime = LowpAttentionRuntime(
        config,
        rope,
        forward_extension=extension,
        forward_topology=topology,
        loss_scale=2.0**18,
        gradient_global_scale=2.0**-8,
        projection_dgrad="bf16",
        qkv_projection_format="nvfp4",
        experimental_native_nvfp4_projection_out=True,
        experimental_fused_attention_rmsnorm_nvfp4=True,
        backward_exp2_degree=1,
        backward_exp2_period=2,
        backward_fp8_ds_lift=16,
        backward_reuse_quantized_p=False,
        backward_control_source=backward_control,
        backward_control_sha256=PINNED_ARTIFACTS["control"][0],
        backward_control_bytes=PINNED_ARTIFACTS["control"][1],
        backward_forward_mx_probability_replay=False,
        backward_forward_mx_probability_scale_handoff=False,
        backward_match_forward_operands=True,
        per_block_qk_scales=True,
        experimental_split_v_backward=(route == MX_ROUTE),
        backward_probability_correction=1.0,
        q_quant_scale=2.25,
        k_quant_scale=2.0,
        projection_weight_scale_2d=True,
        v_mxfp4_scale_2d=False,
        adaptive_qk_weight_scales=False,
        shared_backward_runtime=shared_backward_runtime,
    )
    return runtime, topology


def _bind_runtime(
    model: Llama12B,
    runtime: LowpAttentionRuntime,
) -> None:
    bound_layers = model.bind_lowp_attention_runtime(runtime)
    if bound_layers != model.config.layers:
        raise RuntimeError(
            f"bound {bound_layers} attention layers, expected "
            f"{model.config.layers}"
        )
    activate_model_forward_route(model)


def _fixed_state_step(
    model: Llama12B,
    tokens: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float | bool]:
    """Time one forward/backward without retaining or mutating optimizer state."""
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
    wall_start = time.perf_counter()
    events[0].record()
    hidden, weight = _hidden_and_weight(model, tokens)
    events[1].record()
    loss = _loss(hidden, weight, targets)
    events[2].record()
    loss.backward()
    events[3].record()
    events[3].synchronize()
    loss_value = float(loss.detach())
    record: dict[str, float | bool] = {
        "loss": loss_value,
        "finite": math.isfinite(loss_value),
        "decoder_forward_ms": float(events[0].elapsed_time(events[1])),
        "ce_forward_ms": float(events[1].elapsed_time(events[2])),
        "backward_ms": float(events[2].elapsed_time(events[3])),
        "forward_and_backward_ms": float(
            events[0].elapsed_time(events[3])
        ),
        "wall_ms": (time.perf_counter() - wall_start) * 1000.0,
    }
    del hidden, loss
    return record


def _require_projection_source(path: Path) -> None:
    configured = os.environ.get("TK_FA4_LOWP_BWD_EXTENSION_SOURCE")
    if configured is None:
        raise RuntimeError(
            "set TK_FA4_LOWP_BWD_EXTENSION_SOURCE to the authenticated "
            "projection extension before starting Python"
        )
    if Path(configured).resolve() != path.resolve():
        raise RuntimeError(
            "TK_FA4_LOWP_BWD_EXTENSION_SOURCE does not match "
            "--projection-extension"
        )


def _loaded_projection_identity(expected_path: Path) -> dict[str, Any]:
    extension = getattr(tk_interface, "_C_b300_lowp_bwd", None)
    loaded = getattr(extension, "__file__", None)
    if loaded is None:
        raise RuntimeError("the low-precision projection extension was not loaded")
    loaded_path = Path(loaded).resolve()
    if loaded_path != expected_path.resolve():
        raise RuntimeError(
            f"loaded projection {loaded_path} does not match "
            f"{expected_path.resolve()}"
        )
    return _source_identity(loaded_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mx-forward-extension",
        type=Path,
        default=DEFAULT_FORWARDS[MX_ROUTE],
    )
    parser.add_argument(
        "--fp8-forward-extension",
        type=Path,
        default=DEFAULT_FORWARDS[FP8_ROUTE],
    )
    parser.add_argument(
        "--backward-control",
        type=Path,
        default=DEFAULT_CONTROL,
    )
    parser.add_argument(
        "--projection-extension",
        type=Path,
        default=DEFAULT_PROJECTION,
    )
    parser.add_argument("--projection-sha256")
    parser.add_argument("--projection-bytes", type=int)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--warmups-per-route", type=int, default=3)
    parser.add_argument("--superblocks", type=int, default=24)
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    parser.add_argument(
        "--neutrality-relative-tolerance",
        type=float,
        default=DEFAULT_RELATIVE_TOLERANCE,
        help="Symmetric relative equivalence margin (default: 0.01).",
    )
    parser.add_argument(
        "--neutrality-tolerance-ms",
        type=float,
        help="Optional additional absolute equivalence margin in ms.",
    )
    parser.add_argument("--max-hbm-gib", type=float, default=180.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise RuntimeError("expose exactly one GPU to the benchmark")
    if args.warmups_per_route < 2:
        raise ValueError("--warmups-per-route must be at least 2")
    if args.superblocks < MINIMUM_SUPERBLOCKS:
        raise ValueError(
            f"--superblocks must be at least {MINIMUM_SUPERBLOCKS}"
        )
    if args.bootstrap_draws < 1_000:
        raise ValueError("--bootstrap-draws must be at least 1000")
    if not math.isfinite(args.neutrality_relative_tolerance) or (
        args.neutrality_relative_tolerance <= 0.0
    ):
        raise ValueError(
            "--neutrality-relative-tolerance must be finite and positive"
        )
    if args.neutrality_tolerance_ms is not None and (
        not math.isfinite(args.neutrality_tolerance_ms)
        or args.neutrality_tolerance_ms <= 0.0
    ):
        raise ValueError(
            "--neutrality-tolerance-ms must be finite and positive"
        )
    if not math.isfinite(args.max_hbm_gib) or args.max_hbm_gib <= 0.0:
        raise ValueError("--max-hbm-gib must be finite and positive")
    if os.path.lexists(args.output):
        raise FileExistsError(f"refusing to overwrite {args.output}")

    loaded_python_before = _loaded_python_module_identities()
    torch.cuda.set_device(0)
    gpu_processes_before = _require_exclusive_visible_gpu()
    hardware_before = _hardware_identity()
    if hardware_before["compute_capability"] != [10, 0]:
        raise RuntimeError("the backward-neutrality gate requires SM100")

    projection_expected, projection_authentication = (
        _projection_expected_identity(
            args.projection_extension,
            args.projection_sha256,
            args.projection_bytes,
        )
    )
    _require_projection_source(args.projection_extension)

    def capture_artifacts() -> dict[str, Any]:
        captured = {
            "mx_forward": _file_identity(
                args.mx_forward_extension,
                PINNED_ARTIFACTS["forward"][MX_ROUTE],
            ),
            "fp8_forward": _file_identity(
                args.fp8_forward_extension,
                PINNED_ARTIFACTS["forward"][FP8_ROUTE],
            ),
            "backward_control": _file_identity(
                args.backward_control,
                PINNED_ARTIFACTS["control"],
            ),
            "projection": _file_identity(
                args.projection_extension,
                projection_expected,
            ),
        }
        captured["projection"]["authentication"] = (
            projection_authentication
        )
        return captured

    artifacts = capture_artifacts()
    source_files_before = _benchmark_source_identities()
    _require_loaded_python_matches_sources(
        loaded_python_before,
        source_files_before,
    )

    config = config_from_model_preset(batch=16, sequence=4096, layers=16)
    rope = _make_llama3_rope(config)
    mx_runtime, mx_topology = _make_runtime(
        MX_ROUTE,
        config=config,
        rope=rope,
        forward_path=args.mx_forward_extension,
        backward_control=args.backward_control,
    )
    fp8_runtime, fp8_topology = _make_runtime(
        FP8_ROUTE,
        config=config,
        rope=rope,
        forward_path=args.fp8_forward_extension,
        backward_control=args.backward_control,
        shared_backward_runtime=mx_runtime,
    )
    artifacts["loaded_projection"] = _loaded_projection_identity(
        args.projection_extension
    )
    contracts = {
        MX_ROUTE: mx_runtime.backward_contract(),
        FP8_ROUTE: fp8_runtime.backward_contract(),
    }
    require_matching_backward_contracts(contracts)
    shared_before = require_shared_backward_physical_identity(
        mx_runtime,
        fp8_runtime,
    )
    contract_sha256 = _canonical_contract_sha256(contracts[MX_ROUTE])
    memory_checkpoints = [
        _require_hbm_budget(
            "after_runtime_construction",
            args.max_hbm_gib,
        )
    ]

    torch.manual_seed(args.seed)
    model = Llama12B(config, rope, mx_runtime)
    checkpoint = None
    if args.initial_checkpoint is not None:
        checkpoint = _load_initial_checkpoint(model, args.initial_checkpoint)
    parameter_versions_before = {
        name: int(parameter._version)
        for name, parameter in model.named_parameters()
    }
    tokens, targets = _batch(
        args.seed,
        0,
        config.batch,
        config.sequence,
        config.vocab,
    )
    memory_checkpoints.append(
        _require_hbm_budget(
            "after_model_and_batch_construction",
            args.max_hbm_gib,
        )
    )

    runtimes = {MX_ROUTE: mx_runtime, FP8_ROUTE: fp8_runtime}
    # First use validates both compact projection ABIs against every persistent
    # layer workspace.  Compile and allocator effects remain outside timing.
    torch.cuda.reset_peak_memory_stats()
    for warmup_index in range(args.warmups_per_route):
        for route in ROUTES:
            _bind_runtime(model, runtimes[route])
            warmup = _fixed_state_step(model, tokens, targets)
            if not warmup["finite"]:
                raise RuntimeError(f"non-finite {route} warmup")
            memory_checkpoints.append(
                _require_hbm_budget(
                    f"warmup_{warmup_index}_{route}",
                    args.max_hbm_gib,
                )
            )

    forward_dispatch_contracts: dict[str, Any] = {}
    model_workspace_contracts: dict[str, Any] = {}
    for route in ROUTES:
        runtime = runtimes[route]
        _bind_runtime(model, runtime)
        dispatch = runtime.forward_dispatch_contract()
        workspace = model.lowp_forward_workspace_contract()
        projection = dispatch["qkv_projection"]
        if (
            projection["first_call_full_abi_validation_complete"] is not True
            or projection["preallocated_forward_workspace_abi_validated"]
            is not True
            or projection["validated_forward_workspace_count"]
            != config.layers
            or projection["timed_forward_publication_allocation_fallback"]
            is not False
            or workspace["layer_count"] != config.layers
            or not workspace["owner_pointers_stable_since_allocation"]
            or workspace["supports_both_retained_routes"] is not True
        ):
            raise RuntimeError(
                f"{route} forward dispatch/workspace was not fully "
                "authenticated after warmup"
            )
        forward_dispatch_contracts[route] = dispatch
        model_workspace_contracts[route] = workspace
    workspace_owner_pointer_maps = {
        route: _workspace_owner_pointer_map(contract)
        for route, contract in model_workspace_contracts.items()
    }
    if (
        workspace_owner_pointer_maps[MX_ROUTE]
        != workspace_owner_pointer_maps[FP8_ROUTE]
    ):
        raise RuntimeError(
            "MX and FP8 bindings do not expose the same model-owned "
            "forward workspace allocation map"
        )
    workspace_owner_pointer_map_sha256 = _canonical_contract_sha256(
        {"layers": workspace_owner_pointer_maps[MX_ROUTE]}
    )

    torch.cuda.reset_peak_memory_stats()
    records: list[dict[str, Any]] = []
    periodic_gpu_exclusivity: list[dict[str, Any]] = []
    for superblock_index in range(args.superblocks):
        for position, route in enumerate(
            _balanced_abba_order(superblock_index)
        ):
            _bind_runtime(model, runtimes[route])
            record = _fixed_state_step(model, tokens, targets)
            record.update(
                {
                    "route": route,
                    "superblock": superblock_index,
                    "position": position,
                    "global_call_index": len(records),
                }
            )
            if not record["finite"]:
                raise RuntimeError(
                    f"non-finite measured record at call {len(records)}"
                )
            records.append(record)
            print(
                f"superblock={superblock_index} position={position} "
                f"route={route} backward={record['backward_ms']:.3f}ms",
                flush=True,
            )
        memory_checkpoints.append(
            _require_hbm_budget(
                f"measured_superblock_{superblock_index}",
                args.max_hbm_gib,
            )
        )
        periodic_gpu_exclusivity.append(_require_exclusive_visible_gpu())

    torch.cuda.synchronize()
    parameter_versions_after = {
        name: int(parameter._version)
        for name, parameter in model.named_parameters()
    }
    if parameter_versions_after != parameter_versions_before:
        raise RuntimeError("fixed-state benchmark mutated a model parameter")
    shared_after = require_shared_backward_physical_identity(
        mx_runtime,
        fp8_runtime,
    )
    contracts_after = {
        MX_ROUTE: mx_runtime.backward_contract(),
        FP8_ROUTE: fp8_runtime.backward_contract(),
    }
    require_matching_backward_contracts(contracts_after)
    if contracts_after != contracts:
        raise RuntimeError("backward contract changed during fixed-state timing")

    timing_fields = (
        "decoder_forward_ms",
        "ce_forward_ms",
        "backward_ms",
        "forward_and_backward_ms",
        "wall_ms",
    )
    route_summaries = {
        route: {
            field: _sample_summary(
                [
                    float(record[field])
                    for record in records
                    if record["route"] == route
                ]
            )
            for field in timing_fields
        }
        for route in ROUTES
    }
    loss_invariance = {}
    for route in ROUTES:
        losses = [
            float(record["loss"])
            for record in records
            if record["route"] == route
        ]
        spread = max(losses) - min(losses)
        loss_invariance[route] = {
            "minimum": min(losses),
            "maximum": max(losses),
            "spread": spread,
            "absolute_tolerance": LOSS_INVARIANCE_ATOL,
            "passed": spread <= LOSS_INVARIANCE_ATOL,
        }
    fixed_state_gate_passed = all(
        bool(entry["passed"]) for entry in loss_invariance.values()
    )
    paired: dict[str, Any] = {}
    for field in timing_fields:
        deltas = _adjacent_pair_deltas(records, field)
        superblock_means = _superblock_mean_deltas(records, field)
        interval = _paired_bootstrap_interval(
            superblock_means,
            draws=args.bootstrap_draws,
            seed=args.seed + sum(field.encode()),
        )
        paired[field] = {
            "orientation": "mx_minus_fp8",
            "adjacent_pair_deltas": deltas,
            "adjacent_pair_summary": _sample_summary(deltas),
            "superblock_mean_deltas": superblock_means,
            "superblock_mean_summary": _sample_summary(superblock_means),
            "clustered_mean_bootstrap_95_percent": list(interval),
        }
    backward_interval_ms = paired["backward_ms"][
        "clustered_mean_bootstrap_95_percent"
    ]
    backward_relative_effects = _superblock_symmetric_relative_effects(
        records,
        "backward_ms",
    )
    backward_relative_interval = _paired_bootstrap_interval(
        backward_relative_effects,
        draws=args.bootstrap_draws,
        seed=args.seed + 104729,
    )
    relative_equivalence_passed = (
        float(backward_relative_interval[0])
        >= -args.neutrality_relative_tolerance
        and float(backward_relative_interval[1])
        <= args.neutrality_relative_tolerance
    )
    absolute_equivalence_passed = (
        True
        if args.neutrality_tolerance_ms is None
        else (
            float(backward_interval_ms[0])
            >= -args.neutrality_tolerance_ms
            and float(backward_interval_ms[1])
            <= args.neutrality_tolerance_ms
        )
    )

    measurement_memory = _require_hbm_budget(
        "after_measurement",
        args.max_hbm_gib,
    )
    memory_checkpoints.append(measurement_memory)
    gpu_processes_after = _require_exclusive_visible_gpu()
    artifacts_after = capture_artifacts()
    artifacts_after["loaded_projection"] = _loaded_projection_identity(
        args.projection_extension
    )
    if artifacts_after != artifacts:
        raise RuntimeError("authenticated binary artifact changed during timing")
    source_files_after = _benchmark_source_identities()
    if source_files_after != source_files_before:
        raise RuntimeError("benchmark source artifact changed during timing")
    loaded_python_after = _loaded_python_module_identities()
    if loaded_python_after != loaded_python_before:
        raise RuntimeError("loaded Python module identity changed during timing")
    _require_loaded_python_matches_sources(
        loaded_python_after,
        source_files_after,
    )

    result = {
        "schema": "llama12b_fixed_state_backward_neutrality_v2",
        "purpose": (
            "same_process_fixed_parameter_shared_backward_implementation_"
            "mx_fp8_crossover"
        ),
        "interpretation": (
            "timing parity with route-natural dynamic operands; the physical "
            "runner and workspaces are shared but operand values need not be"
        ),
        "configuration": {
            **config.__dict__,
            "seed": args.seed,
            "warmups_per_route": args.warmups_per_route,
            "superblocks": args.superblocks,
            "measured_calls": len(records),
            "adjacent_pairs": len(records) // 2,
            "optimizer_constructed": False,
            "optimizer_updates": 0,
            "fixed_tokens_and_targets": True,
            "parameter_versions_unchanged": True,
            "qkv_projection_format": "nvfp4",
            "fused_attention_rmsnorm_nvfp4": True,
            "neutrality_relative_tolerance": (
                args.neutrality_relative_tolerance
            ),
            "optional_neutrality_tolerance_ms": (
                args.neutrality_tolerance_ms
            ),
            "bootstrap_draws": args.bootstrap_draws,
            "bootstrap_unit": "eight-call_abba_baab_superblock_mean",
        },
        "checkpoint": checkpoint,
        "artifacts": artifacts,
        "artifacts_after": artifacts_after,
        "artifact_identities_unchanged_across_timing": True,
        "source_files": source_files_before,
        "source_files_after": source_files_after,
        "source_identities_unchanged_across_timing": True,
        "loaded_python_modules": loaded_python_before,
        "loaded_python_modules_after": loaded_python_after,
        "loaded_python_identities_unchanged_across_timing": True,
        "shared_forward_workspace_owner_identity": {
            "maps_equal": True,
            "common_pointer_map_sha256": (
                workspace_owner_pointer_map_sha256
            ),
            "layer_count": len(
                workspace_owner_pointer_maps[MX_ROUTE]
            ),
        },
        "hardware_before": hardware_before,
        "hardware_after": _hardware_identity(),
        "forward_topologies": {
            MX_ROUTE: mx_topology,
            FP8_ROUTE: fp8_topology,
        },
        "forward_dispatch_contracts_after_warmup": (
            forward_dispatch_contracts
        ),
        "model_workspace_contracts_after_warmup": (
            model_workspace_contracts
        ),
        "backward_contracts_before": contracts,
        "backward_contracts_after": contracts_after,
        "common_backward_contract_sha256": contract_sha256,
        "shared_backward_physical_identity_before": shared_before,
        "shared_backward_physical_identity_after": shared_after,
        "records": records,
        "fixed_state_loss_invariance": loss_invariance,
        "route_summaries_ms": route_summaries,
        "paired_differences_ms": paired,
        "neutrality_gate": {
            "method": (
                "clustered_superblock_symmetric_relative_bootstrap_"
                "equivalence_interval"
            ),
            "orientation": "mx_minus_fp8",
            "relative_tolerance": args.neutrality_relative_tolerance,
            "backward_relative_superblock_effects": (
                backward_relative_effects
            ),
            "backward_relative_superblock_summary": _sample_summary(
                backward_relative_effects
            ),
            "backward_relative_clustered_mean_bootstrap_95_percent": (
                backward_relative_interval
            ),
            "optional_absolute_tolerance_ms": args.neutrality_tolerance_ms,
            "backward_absolute_clustered_mean_bootstrap_95_percent_ms": (
                backward_interval_ms
            ),
            "relative_equivalence_passed": relative_equivalence_passed,
            "optional_absolute_equivalence_passed": (
                absolute_equivalence_passed
            ),
            "fixed_state_loss_invariance_passed": fixed_state_gate_passed,
            "passed": (
                relative_equivalence_passed
                and absolute_equivalence_passed
                and fixed_state_gate_passed
            ),
        },
        "memory": {
            "measurement": measurement_memory,
            "checkpoints": memory_checkpoints,
            "gate_gib": args.max_hbm_gib,
        },
        "gpu_exclusivity": {
            "before": gpu_processes_before,
            "periodic_superblock_checks": periodic_gpu_exclusivity,
            "periodic_check_count": len(periodic_gpu_exclusivity),
            "after": gpu_processes_after,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    descriptor = os.open(
        args.output,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    with os.fdopen(descriptor, "w") as output_file:
        output_file.write(rendered)
        output_file.flush()
        os.fsync(output_file.fileno())
    print(json.dumps(result["neutrality_gate"], indent=2, sort_keys=True))
    if not result["neutrality_gate"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
