#!/usr/bin/env python3
"""Compare BF16, MXFP4-PV, and FP8-PV full Llama training steps.

The three routes start from identical weights, see identical batches, and use
a rotating execution order.  The initial-state audit is performed before any
optimizer update.  The subsequent repeated multi-batch run is a short
optimization/stability proxy, not a replacement for corpus-scale convergence.
The default remains the D64 Llama-3.2-1B-like shape; ``llama3.1-8b`` selects
the native D128 NVFP4 projection and non-interleaved causal forward routes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

import tk_fa4.interface as tk_interface
from tk_fa4.lowp_fa4_bwd.benchmark_llama12b_e2e import (
    Config,
    DEFAULT_MODEL_PRESET,
    Llama12B,
    LowpAttentionRuntime,
    MODEL_PRESETS,
    _cosine,
    _load_forward,
    _make_llama3_rope,
    _sample_gradients,
    _useful_flops,
    activate_model_forward_route as activate_bound_model_forward_route,
    config_from_model_preset,
)
from tk_fa4.lowp_fa4_bwd.backward_contract import (
    require_matching_backward_contracts,
)
from tk_fa4.lowp_fa4_bwd.profile_gqa_d128_chain import (
    _make_rope as _make_legacy_rope,
)
ROUTE_NAMES = (
    "bf16_cute",
    "nvfp4_qk_mxfp4_pv",
    "nvfp4_qk_fp8_pv_exact",
)

LOWP_ROUTE_NAMES = ROUTE_NAMES[1:]
MIN_CAUSAL_MACROBLOCKS = 20
MIN_BLOCKED_PHASE_CYCLES = 20
COMPLEMENT_PAIRS_PER_PHASE_CYCLE = 2
MACROBLOCKS_PER_COMPLEMENT_PAIR = 2
MACROBLOCKS_PER_PHASE_CYCLE = (
    COMPLEMENT_PAIRS_PER_PHASE_CYCLE * MACROBLOCKS_PER_COMPLEMENT_PAIR
)
MIN_BLOCKED_COMPLEMENT_PAIRS = (
    MIN_BLOCKED_PHASE_CYCLES * COMPLEMENT_PAIRS_PER_PHASE_CYCLE
)
MIN_BLOCKED_MACROBLOCKS = (
    MIN_BLOCKED_PHASE_CYCLES * MACROBLOCKS_PER_PHASE_CYCLE
)

DESCRIPTOR_CACHE_COUNTER_NAMES = (
    "hits",
    "misses",
    "evictions",
    "clears",
)

FORWARD_MX_PROBABILITY_REPLAY_PATCH_ENV = (
    "TK_GQA_FORWARD_MX_PROBABILITY_REPLAY_PATCH"
)

E4M3_PAIRED_PROJECTION_SYMBOL = (
    "project_qkv_gqa_d64_paired_unified_fp8_e4m3_rope_packed_"
    "interleaved_causal"
)

D128_NVFP4_PROJECTION_SYMBOL = (
    "project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered"
)


def _require_memory_safe_matched_replicas(
    config: Config,
    route_names: tuple[str, ...],
    *,
    operation: str,
) -> None:
    """Reject the known-unsafe full-depth, three-replica 8B layout early."""
    unique_routes = tuple(dict.fromkeys(route_names))
    if not (
        config.model_preset == "llama3.1-8b"
        and config.layers == config.full_model_layers
        and len(unique_routes) >= 3
    ):
        return
    # Even the optimistic persistent-state floor (BF16 parameter, gradient,
    # and two BF16 Adam moments) leaves no room for activations, route
    # workspaces, allocator fragmentation, or CUDA libraries on one GB200.
    persistent_floor_gib = (
        config.parameter_count * len(unique_routes) * 8 / 2.0**30
    )
    raise ValueError(
        f"refusing unsafe {operation}: {len(unique_routes)} full-depth "
        f"llama3.1-8b replicas have a {persistent_floor_gib:.1f} GiB "
        "optimistic persistent training-state floor before activations and "
        "FA4 workspaces. Use --layers 8 (or lower) for an in-process matched "
        "comparison, reduce --routes in the real-token trainer, or run each "
        "full-depth route in a separate process."
    )


def _effective_attention_provenance(config: Config) -> dict[str, Any]:
    """Describe the attention and RoPE tables actually passed to the model."""
    if config.head_dim == 128:
        rope = {
            "builder": "_make_llama3_rope",
            "theta": config.rope_theta,
            "scaling": {
                "rope_type": "llama3",
                "factor": config.rope_factor,
                "low_freq_factor": config.rope_low_frequency_factor,
                "high_freq_factor": config.rope_high_frequency_factor,
                "original_max_position_embeddings": (
                    config.rope_original_context
                ),
            },
        }
        model_preset_scope = "architecture_and_rope"
        rope_factor: float | None = config.rope_factor
        rope_low_frequency_factor: float | None = (
            config.rope_low_frequency_factor
        )
        rope_high_frequency_factor: float | None = (
            config.rope_high_frequency_factor
        )
        rope_original_context: int | None = config.rope_original_context
    else:
        # Preserve the established D64 comparator numerics, but do not label
        # its legacy, unscaled theta-10k tables as Llama-3.2 scaled RoPE.
        rope = {
            "builder": "_make_legacy_rope",
            "theta": 10_000.0,
            "scaling": None,
        }
        model_preset_scope = "architecture_shape_only_legacy_rope"
        rope_factor = None
        rope_low_frequency_factor = None
        rope_high_frequency_factor = None
        rope_original_context = None
    return {
        "model_preset_scope": model_preset_scope,
        # These keys intentionally override Config's preset metadata in the
        # rendered result so the top-level values are effective, not nominal.
        "rope_theta": rope["theta"],
        "rope_factor": rope_factor,
        "rope_low_frequency_factor": rope_low_frequency_factor,
        "rope_high_frequency_factor": rope_high_frequency_factor,
        "rope_original_context": rope_original_context,
        "effective_attention": {
            "causal": True,
            "grouped_query_attention": config.q_heads != config.kv_heads,
            "q_heads": config.q_heads,
            "kv_heads": config.kv_heads,
            "head_dim": config.head_dim,
            "rope": rope,
        },
    }


def _process_cpu_affinity() -> list[int]:
    """Return the exact sorted CPU set available to this process."""
    get_affinity = getattr(os, "sched_getaffinity", None)
    if get_affinity is None:
        raise RuntimeError(
            "process CPU affinity is unavailable on this platform"
        )
    affinity = sorted(int(cpu) for cpu in get_affinity(0))
    if not affinity:
        raise RuntimeError("process CPU affinity must not be empty")
    return affinity


def _require_singleton_cpu_affinity(
    cpu_affinity: list[int],
    *,
    required: bool,
) -> None:
    """Reject claim-grade timing when host scheduling is not controlled."""
    if required and len(cpu_affinity) != 1:
        raise RuntimeError(
            "--require-causal-speed-gate requires singleton process CPU "
            f"affinity, observed {cpu_affinity}; pin one CPU with taskset -c"
        )


def _classify_causal_speed_gate(
    causal_gate: dict[str, Any],
) -> dict[str, Any]:
    """Attach a validity-aware, tri-state conclusion to the causal gate."""
    validity_fields = (
        "forward_sufficient_macroblocks",
        "full_step_sufficient_macroblocks",
        "forward_stationary_within_tolerance",
        "full_step_stationary_within_tolerance",
    )
    performance_fields = (
        "mx_forward_faster",
        "mx_step_faster",
        "backward_equal_within_tolerance",
    )
    required_fields = validity_fields + performance_fields
    missing = [field for field in required_fields if field not in causal_gate]
    if missing:
        raise ValueError(
            "causal speed gate is missing required fields: "
            + ", ".join(missing)
        )
    measurement_valid = all(
        bool(causal_gate[field]) for field in validity_fields
    )
    performance_passed = all(
        bool(causal_gate[field]) for field in performance_fields
    )
    if not measurement_valid:
        conclusion = "inconclusive_nonstationary"
    elif performance_passed:
        conclusion = "pass"
    else:
        conclusion = "valid_performance_fail"
    return {
        **causal_gate,
        "measurement_valid": measurement_valid,
        "conclusion": conclusion,
        "passed": conclusion == "pass",
    }


def _require_causal_speed_gate_pass(
    causal_gate: dict[str, Any],
) -> None:
    """Raise a conclusion-specific error for a required non-passing gate."""
    conclusion = causal_gate.get("conclusion")
    if conclusion == "pass":
        return
    rendered_gate = json.dumps(causal_gate, sort_keys=True)
    if conclusion == "inconclusive_nonstationary":
        raise RuntimeError(
            "same-model causal speed gate was inconclusive_nonstationary; "
            "timing was nonstationary or insufficient: " + rendered_gate
        )
    if conclusion == "valid_performance_fail":
        raise RuntimeError(
            "same-model causal speed gate had a valid_performance_fail: "
            + rendered_gate
        )
    raise RuntimeError(
        f"same-model causal speed gate has invalid conclusion {conclusion!r}: "
        + rendered_gate
    )


def _descriptor_cache_counter_snapshots(
    runtimes: dict[str, LowpAttentionRuntime],
) -> dict[str, dict[str, int]]:
    """Read route-local descriptor-cache counters on the calling thread."""
    if tuple(runtimes) != LOWP_ROUTE_NAMES:
        raise ValueError(
            "descriptor-cache runtimes must follow LOWP_ROUTE_NAMES"
        )
    snapshots: dict[str, dict[str, int]] = {}
    for route_name, runtime in runtimes.items():
        read_topology = getattr(
            runtime.forward_extension,
            "read_hao_direct_topology",
            None,
        )
        if not callable(read_topology):
            raise RuntimeError(
                f"{route_name} forward extension does not expose "
                "read_hao_direct_topology"
            )
        topology = dict(read_topology())
        if (
            topology.get("tma_descriptor_cache_counter_scope")
            != "calling_host_thread"
        ):
            raise RuntimeError(
                f"{route_name} descriptor-cache counters do not have "
                "calling_host_thread scope"
            )
        counters: dict[str, int] = {}
        for counter_name in DESCRIPTOR_CACHE_COUNTER_NAMES:
            field = f"tma_descriptor_cache_{counter_name}"
            value = topology.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise RuntimeError(
                    f"{route_name} {field} must be a nonnegative integer, "
                    f"got {value!r}"
                )
            counters[counter_name] = value
        snapshots[route_name] = counters
    return snapshots


def _descriptor_cache_counter_interval(
    before: dict[str, dict[str, int]],
    after: dict[str, dict[str, int]],
) -> dict[str, Any]:
    """Summarize a monotonic interval of cumulative cache counters."""
    expected_routes = tuple(LOWP_ROUTE_NAMES)
    if tuple(before) != expected_routes or tuple(after) != expected_routes:
        raise ValueError(
            "descriptor-cache snapshots must follow LOWP_ROUTE_NAMES"
        )
    routes: dict[str, dict[str, Any]] = {}
    for route_name in expected_routes:
        before_route = before[route_name]
        after_route = after[route_name]
        if (
            tuple(before_route) != DESCRIPTOR_CACHE_COUNTER_NAMES
            or tuple(after_route) != DESCRIPTOR_CACHE_COUNTER_NAMES
        ):
            raise ValueError(
                f"{route_name} descriptor-cache snapshot has an invalid "
                "counter schema"
            )
        delta: dict[str, int] = {}
        for counter_name in DESCRIPTOR_CACHE_COUNTER_NAMES:
            before_value = before_route[counter_name]
            after_value = after_route[counter_name]
            for boundary, value in (
                ("before", before_value),
                ("after", after_value),
            ):
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                ):
                    raise ValueError(
                        f"{route_name} {boundary} {counter_name} must be a "
                        f"nonnegative integer, got {value!r}"
                    )
            if after_value < before_value:
                raise RuntimeError(
                    f"{route_name} descriptor-cache {counter_name} counter "
                    f"decreased from {before_value} to {after_value}"
                )
            delta[counter_name] = after_value - before_value
        lookups = delta["hits"] + delta["misses"]
        routes[route_name] = {
            "before": dict(before_route),
            "after": dict(after_route),
            "delta": delta,
            "descriptor_lookups": lookups,
            "hit_rate": delta["hits"] / lookups if lookups else None,
            "miss_rate": delta["misses"] / lookups if lookups else None,
            "no_clears_during_interval": delta["clears"] == 0,
            "counters_monotonic": True,
        }
    return {
        "schema": "tma_descriptor_cache_counter_interval_v1",
        "counter_scope": "calling_host_thread",
        "counter_semantics": "cumulative_per_forward_extension",
        "routes": routes,
    }


def _artifact_identity(path: Path) -> dict[str, Any]:
    """Fingerprint a consumed artifact without recording file contents."""
    resolved = path.resolve()
    if not resolved.is_file():
        raise RuntimeError(f"artifact does not exist: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _required_projection_symbols(
    args: argparse.Namespace,
    routes: tuple[str, ...] = ("mx", "fp8"),
    *,
    head_dim: int = 64,
) -> tuple[str, ...]:
    """Resolve the projection entrypoints implied by the selected routes."""
    if head_dim not in (64, 128):
        raise ValueError("projection symbol audit requires D64 or D128")
    if head_dim == 128:
        for route in routes:
            if route not in ("mx", "fp8"):
                raise ValueError(f"unsupported lowp route slot: {route!r}")
            projection_format = getattr(
                args,
                f"{route}_qkv_projection_format",
            )
            if projection_format != "nvfp4":
                raise ValueError(
                    "D128 projection authentication requires both routes "
                    "to use the native NVFP4 publisher"
                )
        return (D128_NVFP4_PROJECTION_SYMBOL,) if routes else ()

    symbols: list[str] = []
    for route in routes:
        if route not in ("mx", "fp8"):
            raise ValueError(f"unsupported lowp route slot: {route!r}")
        if getattr(args, f"{route}_qkv_projection_format") != "e4m3":
            continue
        represented = bool(
            getattr(args, f"{route}_backward_match_forward_operands")
            or getattr(args, f"{route}_per_block_qk_scales")
        )
        symbol = E4M3_PAIRED_PROJECTION_SYMBOL
        if represented:
            symbol += "_represented_backward"
        if getattr(args, f"{route}_per_block_qk_scales"):
            symbol += "_perblock_qk"
        if route == "mx" and args.mx_experimental_split_v_backward:
            symbol += "_split_v_backward"
        if symbol not in symbols:
            symbols.append(symbol)
        route_suffix = (
            "_mx_forward_out"
            if route == "mx"
            else "_fp8_forward_out"
        )
        forward_out_symbol = symbol + route_suffix
        unchecked_symbol = forward_out_symbol + "_unchecked"
        for required_symbol in (forward_out_symbol, unchecked_symbol):
            if required_symbol not in symbols:
                symbols.append(required_symbol)
    return tuple(symbols)


def _require_forward_route_slot(
    route_slot: str,
    topology: dict[str, Any],
    *,
    head_dim: int = 64,
) -> None:
    """Prevent swapped/duplicate artifacts from being mislabeled in output."""
    if head_dim not in (64, 128):
        raise ValueError("forward route audit requires D64 or D128")
    common_expected = {
        "fixed_route_fastpath": True,
        "route_env_guard_per_launch": False,
        "kernel_attribute_init": "once_per_host_thread_and_cuda_device",
        "tma_descriptor_cache": "bounded_thread_local_gl_descriptors",
        "tma_descriptor_cache_capacity": 256,
        "tma_descriptor_cache_lookup": (
            "splitmix64_device_pointer_four_way_set_associative"
        ),
        "tma_descriptor_cache_set_hash": "splitmix64_device_pointer_v1",
        "tma_descriptor_cache_sets": 64,
        "tma_descriptor_cache_ways": 4,
        "tma_descriptor_cache_capacity_scope": "per_compile_time_gl_slot",
        "tma_descriptor_cache_key": (
            "cuda_device_data_ptr_and_compile_time_gl_slot"
        ),
        "tma_descriptor_cache_owns_tensors": False,
        "tma_descriptor_cache_counter_scope": "calling_host_thread",
    }
    if route_slot == "mx":
        if head_dim == 128:
            expected = {
                **common_expected,
                "tma_descriptor_cache_gl_slots": 10,
                "tma_descriptor_cache_total_entry_ceiling": 2560,
                "pv_format": "mxfp4_e8m0_block32",
                "causal_interleaved_kv": False,
                "causal_diagonal_mask": True,
                "causal_stage0_tail_skip": True,
                "mx_mode23_native_density": 3,
                "mx_mode23_native_quarter_mask": 3,
                "mx_mode23_native_density3_quarter_mask": 1,
                "mx_mode23_native_density3_stage_mask": 3,
                "mx_stage0_affine_mask": 0,
                "mx_stage1_affine_mask": 0,
                "mx_global_anchor32": False,
                "mx_global_anchor128": False,
                "mx_global_anchor_margin_log2": 0,
                "mx_stored_scale_shift_log2": 16,
                "mx_anchor_affine_hoist": False,
                "mx_local_denom_pipeline": 2,
                "mx_defer_denom_finalize": 2,
                "fixed_p_ceiling": False,
                "score_pack_ceiling": False,
            }
        else:
            expected = {
                **common_expected,
                "tma_descriptor_cache_gl_slots": 10,
                "tma_descriptor_cache_total_entry_ceiling": 2560,
                "pv_format": "mxfp4_e8m0_block32",
                "causal_interleaved_kv": True,
                "mx_mode23_native_density": 4,
                "mx_mode23_native_quarter_mask": 3,
                "mx_stage0_affine_mask": 0,
                "mx_stage1_affine_mask": 0,
                "mx_global_anchor32": True,
                "mx_global_anchor128": False,
                "mx_global_anchor_margin_log2": 64,
                "mx_stored_scale_shift_log2": 16,
                "mx_anchor_affine_hoist": True,
                "fixed_p_ceiling": False,
                "score_pack_ceiling": False,
            }
    elif route_slot == "fp8":
        expected = {
            **common_expected,
            "tma_descriptor_cache_gl_slots": 9,
            "tma_descriptor_cache_total_entry_ceiling": 2304,
            "pv_format": "e4m3_fp8",
            "shiftless_fp8_mode": 0,
            "fixed_p_ceiling": False,
            "score_pack_ceiling": False,
        }
        if bool(topology.get("causal_interleaved_kv", False)):
            raise ValueError(
                "the FP8 route slot requires ordinary causal K/V order"
            )
    else:
        raise ValueError(f"unsupported lowp route slot: {route_slot!r}")
    for field, value in expected.items():
        if topology.get(field) != value:
            raise ValueError(
                f"{route_slot.upper()} route slot requires {field}={value!r}, "
                f"got {topology.get(field)!r}"
            )


def _projection_extension_identity(
    required_symbols: tuple[str, ...],
    expected_path: Path | None,
) -> dict[str, Any]:
    """Authenticate the loaded projection extension and required bindings."""
    extension = tk_interface._C_b300_lowp_bwd
    if extension is None or getattr(extension, "__file__", None) is None:
        raise RuntimeError(
            "the low-precision projection extension is not loaded from a "
            "filesystem path"
        )
    loaded_path = Path(extension.__file__).resolve()
    if expected_path is not None and loaded_path != expected_path.resolve():
        raise RuntimeError(
            f"loaded projection extension {loaded_path} does not match "
            f"expected extension {expected_path.resolve()}"
        )
    capabilities = {
        symbol: hasattr(extension, symbol) for symbol in required_symbols
    }
    missing = [
        symbol for symbol, available in capabilities.items() if not available
    ]
    if missing:
        raise RuntimeError(
            "the loaded projection extension lacks required matched-route "
            "bindings: " + ", ".join(missing)
        )
    return {
        "module": str(getattr(extension, "__name__", "<unknown>")),
        **_artifact_identity(loaded_path),
        "required_symbols": list(required_symbols),
        "capabilities": capabilities,
    }


def _matched_backward_contracts(
    mx_runtime: LowpAttentionRuntime,
    fp8_runtime: LowpAttentionRuntime,
) -> dict[str, dict[str, Any]]:
    """Return provenance only after the effective backward work matches."""
    contracts = {
        "nvfp4_qk_mxfp4_pv": mx_runtime.backward_contract(),
        "nvfp4_qk_fp8_pv_exact": fp8_runtime.backward_contract(),
    }
    require_matching_backward_contracts(contracts)
    return contracts


def _timed_forward_dispatch_contracts(
    mx_runtime: LowpAttentionRuntime | None,
    fp8_runtime: LowpAttentionRuntime | None,
    mx_model: Llama12B | None,
    fp8_model: Llama12B | None,
) -> dict[str, dict[str, Any]]:
    """Authenticate every selected construction-bound target before timing."""
    candidates = {
        "nvfp4_qk_mxfp4_pv": (mx_runtime, mx_model),
        "nvfp4_qk_fp8_pv_exact": (fp8_runtime, fp8_model),
    }
    mismatched_pairs = [
        route_name
        for route_name, (runtime, model) in candidates.items()
        if (runtime is None) != (model is None)
    ]
    if mismatched_pairs:
        raise RuntimeError(
            "timed forward runtime/model selection mismatch: "
            f"{mismatched_pairs}"
        )
    runtimes = {
        route_name: runtime
        for route_name, (runtime, _) in candidates.items()
        if runtime is not None
    }
    models = {
        route_name: model
        for route_name, (_, model) in candidates.items()
        if model is not None
    }
    if not runtimes:
        raise RuntimeError("timed forward preflight requires a lowp route")
    expected_attention_symbols = {
        "nvfp4_qk_mxfp4_pv": (
            "forward_hao_direct_fp4pv_with_p_scales"
            if mx_runtime is not None
            and mx_runtime.backward_forward_mx_probability_scale_handoff
            else "forward_hao_direct_fp4pv"
        ),
        "nvfp4_qk_fp8_pv_exact": "forward_hao_direct_fp8pv",
    }
    head_dims = {runtime.config.head_dim for runtime in runtimes.values()}
    if len(head_dims) != 1 or head_dims.pop() not in (64, 128):
        raise RuntimeError(
            "timed forward routes must share one supported head dimension"
        )
    contracts: dict[str, dict[str, Any]] = {}
    for route_name, runtime in runtimes.items():
        contract = runtime.forward_dispatch_contract()
        projection = contract["qkv_projection"]
        attention = contract["attention"]
        if runtime.config.head_dim == 128:
            expected_attention = expected_attention_symbols[route_name]
            expected_launcher = (
                "_launch_forward_mx"
                if route_name == "nvfp4_qk_mxfp4_pv"
                else "_launch_forward_fp8"
            )
            if not (
                projection["format"] == "nvfp4"
                and projection["dispatch"] == "public_api_per_invocation"
                and projection["symbol"] is None
                and projection["abi_validation_symbol"] is None
                and projection["checked_symbol"] is None
                and projection["unchecked_symbol"] is None
                and projection["shape_bound_at_construction"] is False
                and projection["first_call_full_abi_validation_complete"]
                is None
                and projection["subsequent_call_path"] == "public_api"
                and projection["preallocated_forward_workspace_required"]
                is False
                and projection[
                    "preallocated_forward_workspace_abi_validated"
                ]
                is None
                and projection["validated_forward_workspace_count"] is None
                and projection[
                    "timed_forward_publication_allocation_fallback"
                ]
                is True
                and projection["preallocated_forward_workspace_ownership"]
                == "allocated_publication_return_owned_by_autograd"
                and projection["runtime_crossover_reallocation"] is False
                and runtime.qkv_projection is None
                and callable(
                    getattr(
                        tk_interface,
                        "b300_project_qkv_gqa_d128_unified_lowp_nvfp4",
                        None,
                    )
                )
            ):
                raise RuntimeError(
                    f"{route_name} did not validate the native D128 NVFP4 "
                    "projection publication contract before timing"
                )
            if not (
                attention["dispatch"]
                == "construction_bound_route_specific_entrypoint"
                and attention["symbol"] == expected_attention
                and attention["launcher"] == expected_launcher
                and attention["entrypoint_bound_at_construction"] is True
                and attention["launcher_bound_to_runtime"] is True
            ):
                raise RuntimeError(
                    f"{route_name} did not bind the required D128 "
                    f"route-specific attention entrypoint "
                    f"{expected_attention!r} before timing"
                )
            workspace_stream = (
                models[route_name].require_lowp_forward_workspace_stream()
            )
            contracts[route_name] = {
                **contract,
                "d128_projection_publication": {
                    "schema": "d128_nvfp4_public_projection_preflight_v1",
                    "python_entrypoint": (
                        "tk_fa4.interface."
                        "b300_project_qkv_gqa_d128_unified_lowp_nvfp4"
                    ),
                    "extension_entrypoint": D128_NVFP4_PROJECTION_SYMBOL,
                    "projection_output_preallocated": False,
                    "timed_projection_output_allocations": True,
                    "allocations_shared_by_mx_and_fp8_routes": (
                        len(runtimes) == 2
                    ),
                    "native_feature_major_fp8_v_required": True,
                    "unfused_fp8_v_layout_materialization_allowed": False,
                    "native_publication_validated_by_interface": True,
                    "completed_compile_forward_before_preflight": True,
                    "inactive_superset_workspace_stream": workspace_stream,
                },
                "validated_after_compile_before_timing": True,
            }
            continue
        projection_symbol = projection["symbol"]
        abi_validation_symbol = projection["abi_validation_symbol"]
        checked_symbol = projection["checked_symbol"]
        unchecked_symbol = projection["unchecked_symbol"]
        route_suffix = (
            "_mx_forward_out"
            if route_name == "nvfp4_qk_mxfp4_pv"
            else "_fp8_forward_out"
        )
        expected_checked_symbol = abi_validation_symbol + route_suffix
        expected_unchecked_symbol = expected_checked_symbol + "_unchecked"
        expected_slots = [4, 6, 8, 9, 10, 11, 12, 13, 20, 21, 22, 23]
        if not (
            projection["format"] == "e4m3"
            and projection["dispatch"]
            == "construction_bound_exact_pybind_symbol"
            and projection["shape_bound_at_construction"] is True
            and projection["first_call_full_abi_validation_complete"] is True
            and projection["subsequent_call_path"]
            == "bound_exact_pybind_symbol_with_preallocated_forward_workspace"
            and projection["preallocated_forward_workspace_required"] is True
            and projection["preallocated_forward_publication_slots"]
            == expected_slots
            and projection["preallocated_forward_workspace_abi_validated"]
            is True
            and projection["validated_forward_workspace_count"]
            == runtime.config.layers
            and projection["timed_forward_publication_allocation_fallback"]
            is False
            and projection["preallocated_forward_workspace_ownership"]
            == "private_nonpersistent_layer_route_neutral_superset"
            and projection["qk_payload_typed_alias_materialization"]
            == "construction_time"
            and projection["runtime_crossover_reallocation"] is False
            and checked_symbol == expected_checked_symbol
            and unchecked_symbol == expected_unchecked_symbol
            and projection_symbol == expected_unchecked_symbol
        ):
            raise RuntimeError(
                f"{route_name} did not validate its construction-bound exact "
                "E4M3 QKV projection before timing"
            )
        workspace_stream = (
            models[route_name].require_lowp_forward_workspace_stream()
        )
        workspace = models[route_name].lowp_forward_workspace_contract()
        workspace["single_stream_preflight_cuda_stream"] = workspace_stream
        c = runtime.config
        expected_owners = {
            "q_payload": (4, [1, c.q_heads, c.sequence, c.head_dim // 2], "torch.uint8"),
            "k_payload": (6, [1, c.kv_heads, c.sequence, c.head_dim // 2], "torch.uint8"),
            "q_scale_pages": (8, [1, c.sequence // 128, c.q_heads, 512], "torch.float8_e4m3fn"),
            "q_global_scale": (9, [1, c.q_heads], "torch.float32"),
            "k_scale_pages": (10, [1, c.sequence // 64, c.kv_heads, 512], "torch.float8_e4m3fn"),
            "k_global_scale": (11, [1, c.kv_heads], "torch.float32"),
            "v_mxfp4_payload": (12, [1, c.kv_heads, c.head_dim, c.sequence // 2], "torch.float4_e2m1fn_x2"),
            "v_mxfp4_scale_pages": (13, [1, c.sequence // 128, c.kv_heads, 512], "torch.float8_e4m3fn"),
            "v_backward_fp8": (20, [1, c.sequence, c.kv_heads, c.head_dim], "torch.float8_e4m3fn"),
            "q_backward_fp8": (21, [1, c.sequence, c.q_heads, c.head_dim], "torch.float8_e4m3fn"),
            "k_backward_fp8": (22, [1, c.sequence, c.kv_heads, c.head_dim], "torch.float8_e4m3fn"),
            "v_fp8_payload": (23, [1, c.kv_heads, c.head_dim, c.sequence], "torch.float8_e4m3fn"),
        }
        common_fields = {
            "q_payload",
            "k_payload",
            "q_scale_pages",
            "q_global_scale",
            "k_scale_pages",
            "k_global_scale",
            "v_backward_fp8",
            "q_backward_fp8",
            "k_backward_fp8",
        }
        expected_active_fields = common_fields | (
            {"v_mxfp4_payload", "v_mxfp4_scale_pages"}
            if runtime.pv_format == "mxfp4_e8m0_block32"
            else {"v_fp8_payload"}
        )
        expected_device = str(runtime.qk_scales.device)
        workspace_layers = workspace["layers"]
        if not (
            workspace["schema"] == "lowp_model_forward_workspaces_v2"
            and
            workspace["layer_count"] == runtime.config.layers
            and workspace["owner_count"]
            == len(expected_owners) * runtime.config.layers
            and workspace["owner_pointers_globally_unique"] is True
            and workspace["owner_pointers_unique_across_layers"] is True
            and workspace["owner_pointers_stable_since_allocation"] is True
            and workspace["typed_aliases_match_owners"] is True
            and workspace["all_outputs_private_nonpersistent"] is True
            and workspace["supports_both_retained_routes"] is True
            and len(workspace_layers) == runtime.config.layers
            and all(
                layer["schema"] == "lowp_layer_forward_workspace_v2"
                and layer["publication_slots"] == expected_slots
                and layer["owner_count"] == len(expected_owners)
                and set(layer["owners"]) == set(expected_owners)
                and all(
                    owner["slot"] == expected_owners[name][0]
                    and owner["shape"] == expected_owners[name][1]
                    and owner["dtype"] == expected_owners[name][2]
                    and owner["device"] == expected_device
                    and owner["bytes"]
                    == math.prod(expected_owners[name][1])
                    * (1 if expected_owners[name][2] != "torch.float32" else 4)
                    and owner["pointer_stable_since_allocation"] is True
                    and owner["listed_in_named_buffers"] is False
                    and owner["listed_in_named_parameters"] is False
                    and owner["optimizer_visible_parameter"] is False
                    for name, owner in layer["owners"].items()
                )
                and layer["owner_pointers_unique_within_layer"] is True
                and layer["owner_pointers_stable_since_allocation"] is True
                and layer["typed_aliases_match_owners"] is True
                and layer["all_outputs_private_nonpersistent"] is True
                and layer["supports_both_retained_routes"] is True
                and layer["active_route"] == runtime.pv_format
                and set(layer["active_owner_fields"])
                == expected_active_fields
                and layer["single_stream_cuda_stream"] == workspace_stream
                and layer["bound_projection_symbol"] == projection_symbol
                and layer["bound_projection_checked_symbol"] == checked_symbol
                and layer["requires_forward_workspace"] is True
                and layer["forward_workspace_abi_validated"] is True
                and layer["validated_forward_workspace_count"]
                == runtime.config.layers
                for layer in workspace_layers
            )
        ):
            raise RuntimeError(
                f"{route_name} did not retain unique, stable, layer-owned "
                "forward-publication workspaces for its exact bound projection"
            )
        expected_attention = expected_attention_symbols[route_name]
        if not (
            attention["dispatch"]
            == "construction_bound_route_specific_entrypoint"
            and attention["symbol"] == expected_attention
            and attention["entrypoint_bound_at_construction"] is True
            and attention["launcher_bound_to_runtime"] is True
        ):
            raise RuntimeError(
                f"{route_name} did not bind the required route-specific "
                f"attention entrypoint {expected_attention!r} before timing"
            )
        contracts[route_name] = {
            **contract,
            "layer_forward_workspaces": workspace,
            "validated_after_compile_before_timing": True,
        }
    return contracts


def _share_matched_backward_runner(
    mx_runtime: LowpAttentionRuntime,
    fp8_runtime: LowpAttentionRuntime,
) -> dict[str, Any]:
    """Share every route-invariant object used by projection/backward."""
    _matched_backward_contracts(mx_runtime, fp8_runtime)
    fp8_runtime.backward = mx_runtime.backward
    fp8_runtime.control = mx_runtime.control
    fp8_runtime.paired_rope = mx_runtime.paired_rope
    fp8_runtime.gradient_global_scale = mx_runtime.gradient_global_scale
    _matched_backward_contracts(mx_runtime, fp8_runtime)
    return {
        "shared_runner_object": (
            mx_runtime.backward is fp8_runtime.backward
        ),
        "shared_workspace_data_ptr": (
            mx_runtime.backward.workspace_torch.data_ptr()
            == fp8_runtime.backward.workspace_torch.data_ptr()
        ),
        "shared_control_module_object": (
            mx_runtime.control is fp8_runtime.control
        ),
        "shared_packed_rope_data_ptr": (
            mx_runtime.paired_rope.data_ptr()
            == fp8_runtime.paired_rope.data_ptr()
        ),
        "shared_gradient_scale_data_ptr": (
            mx_runtime.gradient_global_scale.data_ptr()
            == fp8_runtime.gradient_global_scale.data_ptr()
        ),
    }


def _activate_model_forward_route(
    route_name: str,
    model: torch.nn.Module,
    forward_routes: dict[str, str],
) -> None:
    """Validate provenance and activate the runtime bound to ``model``."""
    expected_route = forward_routes.get(route_name)
    runtime = getattr(model, "lowp_attention_runtime", None)
    actual_route = (
        None
        if runtime is None
        else str(runtime.forward_topology["route"])
    )
    if actual_route != expected_route:
        raise RuntimeError(
            f"{route_name} route provenance {expected_route!r} does not "
            f"match the bound model runtime {actual_route!r}"
        )
    activate_bound_model_forward_route(model)


def _mx_probability_replay_provenance(
    runtime: LowpAttentionRuntime | None,
) -> dict[str, Any] | None:
    """Describe the source patch and generated control used by MX replay."""
    if runtime is None or not runtime.backward_forward_mx_probability_replay:
        return None
    control_source = runtime.backward_control_provenance
    control_path = Path(runtime.control.__file__)
    provenance = {
        "forward_probability_replay": True,
        "forward_probability_scale_handoff": (
            runtime.backward_forward_mx_probability_scale_handoff
        ),
        "control_source": control_source,
        "generated_control": {
            "module": runtime.control.__name__,
            **_artifact_identity(control_path),
        },
    }
    if control_source is not None:
        provenance["control_mode"] = "precomposed"
        provenance["patch_environment_override"] = False
        provenance["patch"] = None
        return provenance
    default_patch = Path(__file__).with_name(
        "d64_gqa_forward_mx_probability_replay.patch"
    )
    patch_path = Path(
        os.environ.get(
            FORWARD_MX_PROBABILITY_REPLAY_PATCH_ENV,
            str(default_patch),
        )
    )
    provenance["control_mode"] = "composed_current_patch_chain"
    provenance["patch_environment_override"] = (
        FORWARD_MX_PROBABILITY_REPLAY_PATCH_ENV in os.environ
    )
    provenance["patch"] = _artifact_identity(patch_path)
    return provenance


def _optimizer(
    model: torch.nn.Module, learning_rate: float
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1.0e-8,
        weight_decay=0.0,
        fused=True,
    )


def _relative_l2(reference: torch.Tensor, actual: torch.Tensor) -> float:
    return float(
        (actual - reference).norm()
        / reference.norm().clamp_min(1.0e-20)
    )


def _quality(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    reference = reference.reshape(-1)
    actual = actual.reshape(-1)
    reference_norm = reference.norm().clamp_min(1.0e-20)
    return {
        "cosine": _cosine(reference, actual),
        "relative_l2": _relative_l2(reference, actual),
        "norm_ratio": float(actual.norm() / reference_norm),
    }


def _initial_state_audit(
    models: dict[str, torch.nn.Module],
    tokens: torch.Tensor,
    targets: torch.Tensor,
    config: Config,
    forward_routes: dict[str, str],
) -> dict[str, Any]:
    samples: dict[str, dict[str, Any]] = {}
    for name in ROUTE_NAMES:
        model = models[name]
        model.zero_grad(set_to_none=True)
        _activate_model_forward_route(name, model, forward_routes)
        logits = model(tokens)
        loss = F.cross_entropy(
            logits.reshape(-1, config.vocab),
            targets.reshape(-1),
            reduction="mean",
        )
        loss.backward()
        samples[name] = {
            "loss": float(loss.detach()),
            "logits": logits.detach()[0, :16, :1024].float().cpu(),
            "gradients": _sample_gradients(model),
        }
        print(
            f"initial-audit route={name} loss={samples[name]['loss']:.6f}",
            flush=True,
        )
        model.zero_grad(set_to_none=True)
        del logits, loss

    reference = samples["bf16_cute"]
    result: dict[str, Any] = {
        "sampled_logits_shape": list(reference["logits"].shape),
        "sampled_gradient_elements_per_parameter": 8192,
        "losses": {
            name: samples[name]["loss"] for name in ROUTE_NAMES
        },
        "vs_bf16": {},
    }
    for name in ROUTE_NAMES[1:]:
        gradients = samples[name]["gradients"]
        result["vs_bf16"][name] = {
            "logits": _quality(reference["logits"], samples[name]["logits"]),
            "gradients": {
                key: _quality(reference_gradient, gradients[key])
                for key, reference_gradient in reference["gradients"].items()
                if key in gradients
            },
        }
    return result


def _step(
    name: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    config: Config,
    round_index: int,
    batch_index: int,
    execution_position: int,
    forward_routes: dict[str, str],
    *,
    warmup: bool,
) -> dict[str, Any]:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    forward_done = torch.cuda.Event(enable_timing=True)
    backward_done = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    optimizer.zero_grad(set_to_none=True)
    _activate_model_forward_route(name, model, forward_routes)
    wall_start = time.perf_counter()
    start.record()
    logits = model(tokens)
    loss = F.cross_entropy(
        logits.reshape(-1, config.vocab),
        targets.reshape(-1),
        reduction="mean",
    )
    forward_done.record()
    loss.backward()
    backward_done.record()
    optimizer.step()
    end.record()
    end.synchronize()
    loss_value = float(loss.detach())
    record = {
        "route": name,
        "round": round_index,
        "batch": batch_index,
        "execution_position": execution_position,
        "warmup": warmup,
        "loss": loss_value,
        "finite": math.isfinite(loss_value),
        "forward_ms": float(start.elapsed_time(forward_done)),
        "backward_ms": float(forward_done.elapsed_time(backward_done)),
        "optimizer_ms": float(backward_done.elapsed_time(end)),
        "step_ms": float(start.elapsed_time(end)),
        "wall_ms": (time.perf_counter() - wall_start) * 1000.0,
    }
    print(
        f"round={round_index} batch={batch_index} pos={execution_position} "
        f"route={name} warmup={warmup} loss={loss_value:.6f} "
        f"step={record['step_ms']:.3f} ms",
        flush=True,
    )
    del logits, loss
    return record


def _acclimate_block_iteration(
    name: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    config: Config,
    forward_routes: dict[str, str],
) -> None:
    """Run one full training iteration without creating timing events."""
    torch.cuda.synchronize()
    optimizer.zero_grad(set_to_none=True)
    _activate_model_forward_route(name, model, forward_routes)
    logits = model(tokens)
    loss = F.cross_entropy(
        logits.reshape(-1, config.vocab),
        targets.reshape(-1),
        reduction="mean",
    )
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    if not bool(torch.isfinite(loss.detach()).item()):
        raise RuntimeError(
            f"non-finite loss while acclimating blocked route {name}"
        )
    del logits, loss


def _compile_without_updating(
    name: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    config: Config,
    execution_position: int,
    forward_routes: dict[str, str],
) -> None:
    """Compile and allocate AdamW state without changing training state."""
    learning_rates = [group["lr"] for group in optimizer.param_groups]
    try:
        for group in optimizer.param_groups:
            group["lr"] = 0.0
        _step(
            name,
            model,
            optimizer,
            tokens,
            targets,
            config,
            -1,
            0,
            execution_position,
            forward_routes,
            warmup=True,
        )
    finally:
        for group, learning_rate in zip(
            optimizer.param_groups, learning_rates, strict=True
        ):
            group["lr"] = learning_rate

    # AdamW accumulated moments and advanced its step counter even though a
    # zero learning rate left the parameters untouched.  Restore the true
    # pre-training optimizer state while retaining the allocated buffers and
    # compiled fused code path.
    _reset_optimizer_state(optimizer)


def _reset_optimizer_state(optimizer: torch.optim.Optimizer) -> None:
    """Restore an already allocated optimizer to its zero-step state."""
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                value.zero_()
            elif key == "step" and isinstance(value, (int, float)):
                state[key] = type(value)(0)
    optimizer.zero_grad(set_to_none=True)


def _assign_model_lowp_runtime(
    model: torch.nn.Module,
    runtime: LowpAttentionRuntime,
) -> int:
    """Switch every attention layer while retaining one physical model.

    Separate model replicas are appropriate for independent loss trajectories,
    but their unrelated parameter/workspace addresses can move common GEMM
    timings by more than the MX-vs-FP8 attention delta.  The causal performance
    comparison therefore crosses both runtimes over one model allocation.
    """
    binder = getattr(model, "bind_lowp_attention_runtime", None)
    if binder is None:
        raise TypeError(
            "runtime crossover requires the decoder runtime-binding API"
        )
    return int(binder(runtime))


def _balanced_route_order(superblock_index: int) -> tuple[str, ...]:
    """Return a complement pair with alternating 16-call phase.

    Each adjacent pair of superblocks contains one ``ABBA+BAAB`` pattern and
    its route complement. Alternating which pattern comes first in successive
    macroblocks prevents a repeatable period-16 disturbance from remaining
    aliased with a route.
    """
    mx, fp8 = LOWP_ROUTE_NAMES
    complemented = (superblock_index % 2) ^ (
        (superblock_index // 2) % 2
    )
    if not complemented:
        return (mx, fp8, fp8, mx, fp8, mx, mx, fp8)
    return (fp8, mx, mx, fp8, mx, fp8, fp8, mx)


def _blocked_route_order(macroblock_index: int) -> tuple[str, ...]:
    """Return one half of an orientation-balanced ABBA/BAAB pair.

    Adjacent macroblocks are route complements.  Successive complement pairs
    reverse which pattern comes first, so a repeatable eight-block phase cannot
    remain assigned to one route across the complete measurement.
    """
    if (
        isinstance(macroblock_index, bool)
        or not isinstance(macroblock_index, int)
        or macroblock_index < 0
    ):
        raise ValueError("blocked crossover macroblock must be nonnegative")
    mx, fp8 = LOWP_ROUTE_NAMES
    pair_index, complement_index = divmod(macroblock_index, 2)
    complemented = complement_index ^ (pair_index % 2)
    if not complemented:
        return (mx, fp8, fp8, mx)
    return (fp8, mx, mx, fp8)


def _trimmed_mean(values: list[float]) -> float:
    ordered = sorted(values)
    trim = len(ordered) // 10
    retained = ordered[trim:-trim] if trim else ordered
    return statistics.mean(retained)


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot take a percentile of no values")
    location = probability * (len(ordered) - 1)
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return ordered[lower]
    fraction = location - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _cluster_bootstrap_trimmed_mean_interval(
    values: list[float],
    confidence: float,
    *,
    seed: int,
    resamples: int = 10_000,
) -> tuple[float, float]:
    """Return a deterministic percentile interval over balanced blocks."""
    if not values:
        raise ValueError("bootstrap requires at least one block")
    if not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap confidence must lie between zero and one")
    generator = random.Random(seed)
    estimates = []
    for _ in range(resamples):
        sample = [values[generator.randrange(len(values))] for _ in values]
        estimates.append(_trimmed_mean(sample))
    tail = (1.0 - confidence) * 0.5
    return _percentile(estimates, tail), _percentile(
        estimates, 1.0 - tail
    )


def _bootstrap_common_mode_drift_interval(
    early: list[float],
    late: list[float],
    *,
    seed: int,
    resamples: int = 10_000,
) -> tuple[float, float]:
    """Return a central 90% interval for early-to-late relative drift."""
    if not early or not late:
        raise ValueError("stationarity bootstrap requires two nonempty windows")
    generator = random.Random(seed)
    estimates = []
    for _ in range(resamples):
        early_sample = [
            early[generator.randrange(len(early))] for _ in early
        ]
        late_sample = [late[generator.randrange(len(late))] for _ in late]
        early_median = statistics.median(early_sample)
        late_median = statistics.median(late_sample)
        midpoint = 0.5 * (early_median + late_median)
        estimates.append(
            (late_median - early_median) / midpoint * 100.0
            if midpoint
            else float("inf")
        )
    return _percentile(estimates, 0.05), _percentile(estimates, 0.95)


def _balanced_block_metric_summary(
    mx_records: list[dict[str, Any]],
    fp8_records: list[dict[str, Any]],
    metric: str,
) -> dict[str, Any]:
    """Summarize drift-cancelled FP8-minus-MX timing by ABBA block.

    Each superblock contains four observations from each route at symmetric
    execution positions. Averaging within the block removes constant, linear,
    and quadratic call-position drift that an adjacent A/B pair aliases with
    the route. Pairing ``ABBA+BAAB`` with its complement cancels cubic drift;
    alternating the pair's phase across macroblocks cancels period-16 bias.
    """
    records_by_route_and_block: dict[
        str, dict[int, list[dict[str, Any]]]
    ] = {"mx": {}, "fp8": {}}

    def integer_metadata(record: dict[str, Any], field: str) -> int:
        value = record.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise RuntimeError(
                f"crossover {field} must be an exact integer, got {value!r}"
            )
        return value

    for route, records in (("mx", mx_records), ("fp8", fp8_records)):
        for record in records:
            block = integer_metadata(record, "crossover_superblock")
            position = integer_metadata(record, "execution_position")
            macroblock = integer_metadata(record, "crossover_macroblock")
            complement = integer_metadata(record, "complement_index")
            global_call = integer_metadata(record, "global_call_index")
            if block < 0 or position < 0:
                raise RuntimeError(
                    "crossover block and position must be nonnegative"
                )
            if macroblock != block // 2 or complement != block % 2:
                raise RuntimeError(
                    f"crossover superblock {block} has inconsistent "
                    "macroblock/complement metadata"
                )
            if global_call != block * 8 + position:
                raise RuntimeError(
                    f"crossover superblock {block} position {position} has "
                    f"invalid global call index {global_call}"
                )
            value = float(record[metric])
            if not math.isfinite(value) or value <= 0.0:
                raise RuntimeError(
                    f"crossover metric {metric} must be finite and positive, "
                    f"got {value!r}"
                )
            records_by_route_and_block[route].setdefault(block, []).append(
                record
            )
    mx_by_block = records_by_route_and_block["mx"]
    fp8_by_block = records_by_route_and_block["fp8"]
    if mx_by_block.keys() != fp8_by_block.keys() or not mx_by_block:
        raise RuntimeError("crossover records are not paired by ABBA block")

    superblock_rows: list[dict[str, Any]] = []
    for block in sorted(mx_by_block):
        mx_block = mx_by_block[block]
        fp8_block = fp8_by_block[block]
        if len(mx_block) != 4 or len(fp8_block) != 4:
            raise RuntimeError(
                f"crossover superblock {block} must contain four samples "
                "per route"
            )
        positions = sorted(
            integer_metadata(record, "execution_position")
            for record in (*mx_block, *fp8_block)
        )
        if positions != list(range(8)):
            raise RuntimeError(
                f"crossover block {block} has invalid positions {positions}"
            )
        mx_positions = {
            integer_metadata(record, "execution_position")
            for record in mx_block
        }
        fp8_positions = {
            integer_metadata(record, "execution_position")
            for record in fp8_block
        }
        complemented = (block % 2) ^ ((block // 2) % 2)
        expected_mx_positions = (
            {1, 2, 4, 7} if complemented else {0, 3, 5, 6}
        )
        if (
            mx_positions != expected_mx_positions
            or fp8_positions != set(range(8)) - expected_mx_positions
        ):
            raise RuntimeError(
                f"crossover superblock {block} does not match its "
                "predeclared route complement"
            )
        mx_mean = statistics.mean(float(record[metric]) for record in mx_block)
        fp8_mean = statistics.mean(
            float(record[metric]) for record in fp8_block
        )
        superblock_rows.append(
            {
                "superblock": block,
                "mx_mean_ms": mx_mean,
                "fp8_mean_ms": fp8_mean,
                "fp8_minus_mx_ms": fp8_mean - mx_mean,
            }
        )
    if [row["superblock"] for row in superblock_rows] != list(
        range(len(superblock_rows))
    ) or len(superblock_rows) % 2:
        raise RuntimeError(
            "crossover superblocks must be consecutive complement pairs"
        )
    macroblock_rows = []
    for macroblock in range(len(superblock_rows) // 2):
        first = superblock_rows[2 * macroblock]
        second = superblock_rows[2 * macroblock + 1]
        mx_mean = 0.5 * (first["mx_mean_ms"] + second["mx_mean_ms"])
        fp8_mean = 0.5 * (
            first["fp8_mean_ms"] + second["fp8_mean_ms"]
        )
        macroblock_rows.append(
            {
                "macroblock": macroblock,
                "superblocks": [
                    first["superblock"],
                    second["superblock"],
                ],
                "mx_mean_ms": mx_mean,
                "fp8_mean_ms": fp8_mean,
                "fp8_minus_mx_ms": fp8_mean - mx_mean,
            }
        )
    deltas = [row["fp8_minus_mx_ms"] for row in macroblock_rows]
    mx_values = [float(record[metric]) for record in mx_records]
    fp8_values = [float(record[metric]) for record in fp8_records]
    mx_macroblock_values = [row["mx_mean_ms"] for row in macroblock_rows]
    fp8_macroblock_values = [row["fp8_mean_ms"] for row in macroblock_rows]
    paired_mean = statistics.mean(deltas)
    paired_trimmed_mean = _trimmed_mean(deltas)
    paired_stdev = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    paired_standard_error = paired_stdev / math.sqrt(len(deltas))
    bootstrap_seed = sum(ord(character) for character in metric) + len(deltas)
    bootstrap_ci95 = _cluster_bootstrap_trimmed_mean_interval(
        deltas,
        0.95,
        seed=bootstrap_seed,
    )
    bootstrap_ci90 = _cluster_bootstrap_trimmed_mean_interval(
        deltas,
        0.90,
        seed=bootstrap_seed + 1,
    )
    common_mode = [
        0.5 * (row["mx_mean_ms"] + row["fp8_mean_ms"])
        for row in macroblock_rows
    ]
    stationarity_window = max(1, len(common_mode) // 4)
    early_window = common_mode[:stationarity_window]
    late_window = common_mode[-stationarity_window:]
    early_common = statistics.median(early_window)
    late_common = statistics.median(late_window)
    common_midpoint = 0.5 * (early_common + late_common)
    common_drift_percent = (
        (late_common - early_common) / common_midpoint * 100.0
        if common_midpoint
        else float("inf")
    )
    drift_ci90 = _bootstrap_common_mode_drift_interval(
        early_window,
        late_window,
        seed=bootstrap_seed + 2,
    )
    quartile_medians = []
    for quartile in range(4):
        start = quartile * len(common_mode) // 4
        stop = (quartile + 1) * len(common_mode) // 4
        values = common_mode[start:stop]
        if values:
            quartile_medians.append(statistics.median(values))
    common_reference = statistics.median(common_mode)
    quartile_range_percent = (
        (max(quartile_medians) - min(quartile_medians))
        / common_reference
        * 100.0
        if common_reference
        else float("inf")
    )
    slopes = [
        (common_mode[j] - common_mode[i]) / (j - i)
        for i in range(len(common_mode))
        for j in range(i + 1, len(common_mode))
    ]
    projected_robust_slope_percent = (
        statistics.median(slopes)
        * max(0, len(common_mode) - 1)
        / common_reference
        * 100.0
        if slopes and common_reference
        else 0.0
    )
    position_means = {
        route: {
            str(position): statistics.mean(
                float(record[metric])
                for record in records
                if integer_metadata(record, "execution_position") == position
            )
            for position in range(8)
            if any(
                integer_metadata(record, "execution_position") == position
                for record in records
            )
        }
        for route, records in (("mx", mx_records), ("fp8", fp8_records))
    }
    return {
        "unit": "milliseconds",
        "positive_means_mx_faster": True,
        "estimator": (
            "complement_paired_abba_baab_macroblock_trimmed_mean"
        ),
        "mx_median_ms": statistics.median(mx_values),
        "fp8_median_ms": statistics.median(fp8_values),
        "mx_macroblock_median_ms": statistics.median(mx_macroblock_values),
        "fp8_macroblock_median_ms": statistics.median(
            fp8_macroblock_values
        ),
        "fp8_minus_mx_block_median_ms": statistics.median(deltas),
        "fp8_minus_mx_block_mean_ms": paired_mean,
        "fp8_minus_mx_block_trimmed_mean_ms": paired_trimmed_mean,
        "fp8_minus_mx_block_stdev_ms": paired_stdev,
        "fp8_minus_mx_block_standard_error_ms": paired_standard_error,
        "fp8_minus_mx_block_trimmed_mean_bootstrap_ci95_lower_ms": (
            bootstrap_ci95[0]
        ),
        "fp8_minus_mx_block_trimmed_mean_bootstrap_ci95_upper_ms": (
            bootstrap_ci95[1]
        ),
        "fp8_minus_mx_block_trimmed_mean_bootstrap_ci90_lower_ms": (
            bootstrap_ci90[0]
        ),
        "fp8_minus_mx_block_trimmed_mean_bootstrap_ci90_upper_ms": (
            bootstrap_ci90[1]
        ),
        "mx_faster_blocks": sum(delta > 0.0 for delta in deltas),
        "ties": sum(delta == 0.0 for delta in deltas),
        "blocks": len(deltas),
        "superblocks": len(superblock_rows),
        "macroblocks": len(macroblock_rows),
        "minimum_macroblocks_for_gate": MIN_CAUSAL_MACROBLOCKS,
        "sufficient_samples": (
            len(macroblock_rows) >= MIN_CAUSAL_MACROBLOCKS
        ),
        "samples_per_route": len(mx_values),
        "common_mode_early_median_ms": early_common,
        "common_mode_late_median_ms": late_common,
        "common_mode_drift_percent": common_drift_percent,
        "common_mode_drift_bootstrap_ci90_lower_percent": drift_ci90[0],
        "common_mode_drift_bootstrap_ci90_upper_percent": drift_ci90[1],
        "common_mode_quartile_medians_ms": quartile_medians,
        "common_mode_quartile_range_percent": quartile_range_percent,
        "common_mode_projected_theil_sen_slope_percent": (
            projected_robust_slope_percent
        ),
        "execution_position_mean_ms": position_means,
        "superblock_records": superblock_rows,
        "macroblock_records": macroblock_rows,
    }


def _blocked_metric_summary(
    mx_records: list[dict[str, Any]],
    fp8_records: list[dict[str, Any]],
    metric: str,
    *,
    macroblocks: int,
    measured_steps_per_block: int,
) -> dict[str, Any]:
    """Summarize covariance-preserving four-macroblock phase cycles."""
    if macroblocks < MIN_BLOCKED_MACROBLOCKS or macroblocks % 4:
        raise ValueError(
            "blocked crossover requires a multiple of four macroblocks at "
            f"least {MIN_BLOCKED_MACROBLOCKS}"
        )
    if measured_steps_per_block <= 0:
        raise ValueError("measured steps per route block must be positive")

    def integer_metadata(record: dict[str, Any], field: str) -> int:
        value = record.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise RuntimeError(
                f"blocked crossover {field} must be an exact integer, "
                f"got {value!r}"
            )
        return value

    route_names = {
        "mx": LOWP_ROUTE_NAMES[0],
        "fp8": LOWP_ROUTE_NAMES[1],
    }
    records_by_block: dict[int, list[dict[str, Any]]] = {}
    for route, records in (("mx", mx_records), ("fp8", fp8_records)):
        for record in records:
            macroblock = integer_metadata(record, "blocked_macroblock")
            block_position = integer_metadata(
                record, "blocked_block_position"
            )
            step_in_block = integer_metadata(record, "blocked_step_in_block")
            global_block = integer_metadata(record, "global_block_index")
            global_call = integer_metadata(record, "global_call_index")
            execution_position = integer_metadata(
                record, "execution_position"
            )
            if not 0 <= macroblock < macroblocks:
                raise RuntimeError(
                    f"blocked crossover macroblock {macroblock} is out of range"
                )
            if not 0 <= block_position < 4:
                raise RuntimeError(
                    "blocked crossover block position must be in [0, 4)"
                )
            if not 0 <= step_in_block < measured_steps_per_block:
                raise RuntimeError(
                    "blocked crossover step-in-block is out of range"
                )
            if execution_position != block_position:
                raise RuntimeError(
                    "blocked crossover execution position does not match "
                    "its whole-route block position"
                )
            expected_global_block = macroblock * 4 + block_position
            if global_block != expected_global_block:
                raise RuntimeError(
                    f"blocked crossover block {global_block} does not match "
                    f"macroblock {macroblock} position {block_position}"
                )
            expected_global_call = (
                global_block * measured_steps_per_block + step_in_block
            )
            if global_call != expected_global_call:
                raise RuntimeError(
                    f"blocked crossover global call {global_call} does not "
                    f"match expected call {expected_global_call}"
                )
            expected_route = _blocked_route_order(macroblock)[block_position]
            if expected_route != route_names[route]:
                raise RuntimeError(
                    f"blocked crossover macroblock {macroblock} position "
                    f"{block_position} is assigned to {expected_route}, not "
                    f"{route_names[route]}"
                )
            if record.get("route") != route_names[route]:
                raise RuntimeError(
                    "blocked crossover record route does not match its "
                    "route collection"
                )
            value = float(record[metric])
            if not math.isfinite(value) or value <= 0.0:
                raise RuntimeError(
                    f"blocked crossover metric {metric} must be finite and "
                    f"positive, got {value!r}"
                )
            records_by_block.setdefault(global_block, []).append(record)

    expected_blocks = macroblocks * 4
    if sorted(records_by_block) != list(range(expected_blocks)):
        raise RuntimeError(
            "blocked crossover route blocks must be consecutive and complete"
        )
    route_block_rows: list[dict[str, Any]] = []
    for global_block in range(expected_blocks):
        block_records = records_by_block[global_block]
        if len(block_records) != measured_steps_per_block:
            raise RuntimeError(
                f"blocked crossover block {global_block} must contain "
                f"{measured_steps_per_block} measured steps"
            )
        block_records.sort(
            key=lambda record: integer_metadata(
                record, "blocked_step_in_block"
            )
        )
        if [
            integer_metadata(record, "blocked_step_in_block")
            for record in block_records
        ] != list(range(measured_steps_per_block)):
            raise RuntimeError(
                f"blocked crossover block {global_block} has duplicate or "
                "missing measured steps"
            )
        macroblock = global_block // 4
        block_position = global_block % 4
        route = _blocked_route_order(macroblock)[block_position]
        route_block_rows.append(
            {
                "global_block": global_block,
                "macroblock": macroblock,
                "block_position": block_position,
                "route": route,
                "mean_ms": statistics.mean(
                    float(record[metric]) for record in block_records
                ),
            }
        )

    macroblock_rows: list[dict[str, Any]] = []
    for macroblock in range(macroblocks):
        block_rows = route_block_rows[macroblock * 4 : (macroblock + 1) * 4]
        order = _blocked_route_order(macroblock)
        if tuple(row["route"] for row in block_rows) != order:
            raise RuntimeError(
                f"blocked crossover macroblock {macroblock} has invalid order"
            )
        mx_values = [
            float(record[metric])
            for record in mx_records
            if integer_metadata(record, "blocked_macroblock") == macroblock
        ]
        fp8_values = [
            float(record[metric])
            for record in fp8_records
            if integer_metadata(record, "blocked_macroblock") == macroblock
        ]
        expected_route_samples = 2 * measured_steps_per_block
        if (
            len(mx_values) != expected_route_samples
            or len(fp8_values) != expected_route_samples
        ):
            raise RuntimeError(
                f"blocked crossover macroblock {macroblock} must contain two "
                "complete blocks from each route"
            )
        mx_mean = statistics.mean(mx_values)
        fp8_mean = statistics.mean(fp8_values)
        macroblock_rows.append(
            {
                "macroblock": macroblock,
                "block_order": list(order),
                "global_blocks": [
                    int(row["global_block"]) for row in block_rows
                ],
                "mx_mean_ms": mx_mean,
                "fp8_mean_ms": fp8_mean,
                "fp8_minus_mx_ms": fp8_mean - mx_mean,
            }
        )

    complement_pair_rows: list[dict[str, Any]] = []
    matched_position_rows: list[dict[str, Any]] = []
    for pair_index in range(macroblocks // 2):
        first_macroblock = 2 * pair_index
        second_macroblock = first_macroblock + 1
        first_order = _blocked_route_order(first_macroblock)
        second_order = _blocked_route_order(second_macroblock)
        if any(
            first_route == second_route
            for first_route, second_route in zip(
                first_order, second_order, strict=True
            )
        ):
            raise RuntimeError(
                f"blocked crossover pair {pair_index} is not route "
                "complementary by block position"
            )
        pair_blocks = route_block_rows[
            first_macroblock * 4 : (second_macroblock + 1) * 4
        ]
        if len(pair_blocks) != 8:
            raise RuntimeError(
                f"blocked crossover pair {pair_index} must contain eight "
                "complete route blocks"
            )
        position_rows = []
        for block_position in range(4):
            first_block = pair_blocks[block_position]
            second_block = pair_blocks[4 + block_position]
            if first_block["route"] == second_block["route"]:
                raise RuntimeError(
                    f"blocked crossover pair {pair_index} position "
                    f"{block_position} is not route complementary"
                )
            by_route = {
                str(first_block["route"]): float(first_block["mean_ms"]),
                str(second_block["route"]): float(second_block["mean_ms"]),
            }
            mx_value = by_route[route_names["mx"]]
            fp8_value = by_route[route_names["fp8"]]
            position_row = {
                "complement_pair": pair_index,
                "pair_orientation": pair_index % 2,
                "phase_cycle": pair_index // 2,
                "pair_position_in_phase_cycle": pair_index % 2,
                "block_position": block_position,
                "mx_ms": mx_value,
                "fp8_ms": fp8_value,
                "fp8_minus_mx_ms": fp8_value - mx_value,
            }
            position_rows.append(position_row)
            matched_position_rows.append(position_row)
        mx_mean = statistics.mean(row["mx_ms"] for row in position_rows)
        fp8_mean = statistics.mean(row["fp8_ms"] for row in position_rows)
        complement_pair_rows.append(
            {
                "complement_pair": pair_index,
                "pair_orientation": pair_index % 2,
                "phase_cycle": pair_index // 2,
                "pair_position_in_phase_cycle": pair_index % 2,
                "macroblocks": [first_macroblock, second_macroblock],
                "block_orders": [
                    list(first_order),
                    list(second_order),
                ],
                "global_blocks": [
                    int(row["global_block"]) for row in pair_blocks
                ],
                "mx_mean_ms": mx_mean,
                "fp8_mean_ms": fp8_mean,
                "fp8_minus_mx_ms": fp8_mean - mx_mean,
            }
        )

    phase_cycle_rows: list[dict[str, Any]] = []
    if len(complement_pair_rows) % COMPLEMENT_PAIRS_PER_PHASE_CYCLE:
        raise RuntimeError(
            "blocked crossover complement pairs must form complete phase "
            "cycles"
        )
    for phase_cycle in range(
        len(complement_pair_rows) // COMPLEMENT_PAIRS_PER_PHASE_CYCLE
    ):
        cycle_pairs = complement_pair_rows[
            phase_cycle * COMPLEMENT_PAIRS_PER_PHASE_CYCLE :
            (phase_cycle + 1) * COMPLEMENT_PAIRS_PER_PHASE_CYCLE
        ]
        if [row["pair_orientation"] for row in cycle_pairs] != [0, 1]:
            raise RuntimeError(
                f"blocked crossover phase cycle {phase_cycle} does not "
                "contain adjacent orientation-zero and orientation-one "
                "complement pairs"
            )
        if [row["phase_cycle"] for row in cycle_pairs] != [
            phase_cycle,
            phase_cycle,
        ] or [
            row["pair_position_in_phase_cycle"] for row in cycle_pairs
        ] != [0, 1]:
            raise RuntimeError(
                f"blocked crossover phase cycle {phase_cycle} has invalid "
                "pair provenance"
            )
        mx_mean = statistics.mean(row["mx_mean_ms"] for row in cycle_pairs)
        fp8_mean = statistics.mean(
            row["fp8_mean_ms"] for row in cycle_pairs
        )
        phase_cycle_rows.append(
            {
                "phase_cycle": phase_cycle,
                "complement_pairs": [
                    int(row["complement_pair"]) for row in cycle_pairs
                ],
                "pair_orientations": [
                    int(row["pair_orientation"]) for row in cycle_pairs
                ],
                "pair_provenance": [
                    {
                        "complement_pair": int(row["complement_pair"]),
                        "pair_position_in_phase_cycle": int(
                            row["pair_position_in_phase_cycle"]
                        ),
                        "pair_orientation": int(row["pair_orientation"]),
                        "macroblocks": list(row["macroblocks"]),
                        "global_blocks": list(row["global_blocks"]),
                    }
                    for row in cycle_pairs
                ],
                "macroblocks": [
                    int(macroblock)
                    for row in cycle_pairs
                    for macroblock in row["macroblocks"]
                ],
                "global_blocks": [
                    int(global_block)
                    for row in cycle_pairs
                    for global_block in row["global_blocks"]
                ],
                "mx_mean_ms": mx_mean,
                "fp8_mean_ms": fp8_mean,
                "fp8_minus_mx_ms": fp8_mean - mx_mean,
            }
        )

    deltas = [row["fp8_minus_mx_ms"] for row in phase_cycle_rows]
    mx_values = [float(record[metric]) for record in mx_records]
    fp8_values = [float(record[metric]) for record in fp8_records]
    mx_macroblock_values = [row["mx_mean_ms"] for row in macroblock_rows]
    fp8_macroblock_values = [row["fp8_mean_ms"] for row in macroblock_rows]
    mx_complement_pair_values = [
        row["mx_mean_ms"] for row in complement_pair_rows
    ]
    fp8_complement_pair_values = [
        row["fp8_mean_ms"] for row in complement_pair_rows
    ]
    mx_phase_cycle_values = [row["mx_mean_ms"] for row in phase_cycle_rows]
    fp8_phase_cycle_values = [
        row["fp8_mean_ms"] for row in phase_cycle_rows
    ]
    complement_pair_deltas = [
        row["fp8_minus_mx_ms"] for row in complement_pair_rows
    ]
    paired_mean = statistics.mean(deltas)
    paired_trimmed_mean = _trimmed_mean(deltas)
    paired_stdev = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    paired_standard_error = paired_stdev / math.sqrt(len(deltas))
    bootstrap_seed = (
        10_000 + sum(ord(character) for character in metric) + len(deltas)
    )
    bootstrap_ci95 = _cluster_bootstrap_trimmed_mean_interval(
        deltas,
        0.95,
        seed=bootstrap_seed,
    )
    bootstrap_ci90 = _cluster_bootstrap_trimmed_mean_interval(
        deltas,
        0.90,
        seed=bootstrap_seed + 1,
    )

    common_mode = [
        0.5 * (row["mx_mean_ms"] + row["fp8_mean_ms"])
        for row in phase_cycle_rows
    ]
    stationarity_window = max(1, len(common_mode) // 4)
    early_window = common_mode[:stationarity_window]
    late_window = common_mode[-stationarity_window:]
    early_common = statistics.median(early_window)
    late_common = statistics.median(late_window)
    common_midpoint = 0.5 * (early_common + late_common)
    common_drift_percent = (
        (late_common - early_common) / common_midpoint * 100.0
        if common_midpoint
        else float("inf")
    )
    drift_ci90 = _bootstrap_common_mode_drift_interval(
        early_window,
        late_window,
        seed=bootstrap_seed + 2,
    )
    quartile_medians = []
    for quartile in range(4):
        start = quartile * len(common_mode) // 4
        stop = (quartile + 1) * len(common_mode) // 4
        values = common_mode[start:stop]
        if values:
            quartile_medians.append(statistics.median(values))
    common_reference = statistics.median(common_mode)
    quartile_range_percent = (
        (max(quartile_medians) - min(quartile_medians))
        / common_reference
        * 100.0
        if common_reference
        else float("inf")
    )
    slopes = [
        (common_mode[j] - common_mode[i]) / (j - i)
        for i in range(len(common_mode))
        for j in range(i + 1, len(common_mode))
    ]
    projected_robust_slope_percent = (
        statistics.median(slopes)
        * max(0, len(common_mode) - 1)
        / common_reference
        * 100.0
        if slopes and common_reference
        else 0.0
    )
    block_position_means = {
        route: {
            str(position): statistics.mean(
                float(record[metric])
                for record in records
                if integer_metadata(
                    record, "blocked_block_position"
                ) == position
            )
            for position in range(4)
            if any(
                integer_metadata(
                    record, "blocked_block_position"
                ) == position
                for record in records
            )
        }
        for route, records in (("mx", mx_records), ("fp8", fp8_records))
    }
    return {
        "unit": "milliseconds",
        "positive_means_mx_faster": True,
        "estimator": (
            "full_phase_cycle_abba_baab_baab_abba_trimmed_mean"
        ),
        "estimator_unit": (
            "two_adjacent_complement_pairs_four_raw_macroblocks"
        ),
        "bootstrap_cluster_unit": "full_phase_cycle",
        "bootstrap_preserves_pair_orientation": True,
        "bootstrap_preserves_adjacent_pair_covariance": True,
        "stationarity_unit": "full_phase_cycle_common_mode",
        "stationarity_window_phase_cycles": stationarity_window,
        "mx_median_ms": statistics.median(mx_values),
        "fp8_median_ms": statistics.median(fp8_values),
        "mx_macroblock_median_ms": statistics.median(mx_macroblock_values),
        "fp8_macroblock_median_ms": statistics.median(
            fp8_macroblock_values
        ),
        "mx_complement_pair_median_ms": statistics.median(
            mx_complement_pair_values
        ),
        "fp8_complement_pair_median_ms": statistics.median(
            fp8_complement_pair_values
        ),
        "mx_phase_cycle_median_ms": statistics.median(
            mx_phase_cycle_values
        ),
        "fp8_phase_cycle_median_ms": statistics.median(
            fp8_phase_cycle_values
        ),
        "fp8_minus_mx_phase_cycle_median_ms": statistics.median(deltas),
        "fp8_minus_mx_phase_cycle_mean_ms": paired_mean,
        "fp8_minus_mx_phase_cycle_trimmed_mean_ms": paired_trimmed_mean,
        "fp8_minus_mx_phase_cycle_stdev_ms": paired_stdev,
        "fp8_minus_mx_phase_cycle_standard_error_ms": (
            paired_standard_error
        ),
        "fp8_minus_mx_phase_cycle_trimmed_mean_bootstrap_ci95_lower_ms": (
            bootstrap_ci95[0]
        ),
        "fp8_minus_mx_phase_cycle_trimmed_mean_bootstrap_ci95_upper_ms": (
            bootstrap_ci95[1]
        ),
        "fp8_minus_mx_phase_cycle_trimmed_mean_bootstrap_ci90_lower_ms": (
            bootstrap_ci90[0]
        ),
        "fp8_minus_mx_phase_cycle_trimmed_mean_bootstrap_ci90_upper_ms": (
            bootstrap_ci90[1]
        ),
        # Compatibility aliases retain the previous output field names while
        # the estimator/schema fields above make the true resampling unit
        # explicit.  All aliases below now summarize full phase cycles.
        "fp8_minus_mx_block_median_ms": statistics.median(deltas),
        "fp8_minus_mx_block_mean_ms": paired_mean,
        "fp8_minus_mx_block_trimmed_mean_ms": paired_trimmed_mean,
        "fp8_minus_mx_block_stdev_ms": paired_stdev,
        "fp8_minus_mx_block_standard_error_ms": paired_standard_error,
        "fp8_minus_mx_block_trimmed_mean_bootstrap_ci95_lower_ms": (
            bootstrap_ci95[0]
        ),
        "fp8_minus_mx_block_trimmed_mean_bootstrap_ci95_upper_ms": (
            bootstrap_ci95[1]
        ),
        "fp8_minus_mx_block_trimmed_mean_bootstrap_ci90_lower_ms": (
            bootstrap_ci90[0]
        ),
        "fp8_minus_mx_block_trimmed_mean_bootstrap_ci90_upper_ms": (
            bootstrap_ci90[1]
        ),
        "mx_faster_blocks": sum(delta > 0.0 for delta in deltas),
        "mx_faster_phase_cycles": sum(delta > 0.0 for delta in deltas),
        "mx_faster_complement_pairs": sum(
            delta > 0.0 for delta in complement_pair_deltas
        ),
        "ties": sum(delta == 0.0 for delta in deltas),
        "blocks": len(deltas),
        "route_blocks": len(route_block_rows),
        "macroblocks": len(macroblock_rows),
        "complement_pairs": len(complement_pair_rows),
        "complement_pairs_by_orientation": {
            str(orientation): sum(
                row["pair_orientation"] == orientation
                for row in complement_pair_rows
            )
            for orientation in range(2)
        },
        "phase_cycles": len(phase_cycle_rows),
        "minimum_phase_cycles_for_gate": MIN_BLOCKED_PHASE_CYCLES,
        "minimum_complement_pairs_for_gate": (
            MIN_BLOCKED_COMPLEMENT_PAIRS
        ),
        "minimum_macroblocks_for_gate": MIN_BLOCKED_MACROBLOCKS,
        "sufficient_samples": (
            len(phase_cycle_rows) >= MIN_BLOCKED_PHASE_CYCLES
        ),
        "measured_steps_per_block": measured_steps_per_block,
        "samples_per_route": len(mx_values),
        "common_mode_early_median_ms": early_common,
        "common_mode_late_median_ms": late_common,
        "common_mode_drift_percent": common_drift_percent,
        "common_mode_drift_bootstrap_ci90_lower_percent": drift_ci90[0],
        "common_mode_drift_bootstrap_ci90_upper_percent": drift_ci90[1],
        "common_mode_quartile_medians_ms": quartile_medians,
        "common_mode_quartile_range_percent": quartile_range_percent,
        "common_mode_projected_theil_sen_slope_percent": (
            projected_robust_slope_percent
        ),
        "block_position_mean_ms": block_position_means,
        "route_block_records": route_block_rows,
        "macroblock_records": macroblock_rows,
        "matched_position_records": matched_position_rows,
        "complement_pair_records": complement_pair_rows,
        "phase_cycle_records": phase_cycle_rows,
    }


def _forward_only_step(
    name: str,
    model: torch.nn.Module,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    config: Config,
    sample_index: int,
    execution_position: int,
    forward_routes: dict[str, str],
    *,
    warmup: bool,
) -> dict[str, Any]:
    """Measure a training-mode forward without backward/Adam contamination."""
    _activate_model_forward_route(name, model, forward_routes)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record()
    logits = model(tokens)
    loss = F.cross_entropy(
        logits.reshape(-1, config.vocab),
        targets.reshape(-1),
    )
    end.record()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    record = {
        "route": name,
        "round": sample_index,
        "batch": 0,
        "execution_position": execution_position,
        "warmup": warmup,
        "loss": float(loss.detach()),
        "finite": bool(torch.isfinite(loss.detach()).item()),
        "forward_ms": start.elapsed_time(end),
        "wall_ms": wall_ms,
    }
    del loss, logits
    return record


def _same_model_forward_crossover(
    model: torch.nn.Module,
    runtimes: dict[str, LowpAttentionRuntime],
    tokens: torch.Tensor,
    targets: torch.Tensor,
    config: Config,
    forward_routes: dict[str, str],
    *,
    rounds: int,
    warmups: int,
    stationarity_tolerance_percent: float,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Isolate the causal route in a full training-mode model forward."""
    if tuple(runtimes) != LOWP_ROUTE_NAMES:
        raise ValueError("forward crossover runtimes must follow LOWP_ROUTE_NAMES")
    if rounds <= 0 or rounds % 8 or warmups <= 0 or warmups % 8:
        raise ValueError(
            "forward crossover samples and warmups must be positive "
            "multiples of eight per route"
        )
    if (
        not math.isfinite(stationarity_tolerance_percent)
        or stationarity_tolerance_percent <= 0.0
    ):
        raise ValueError("forward stationarity tolerance must be positive")
    original_runtime = getattr(model, "lowp_attention_runtime", None)
    if original_runtime is None:
        raise TypeError("forward crossover model must already be low precision")
    records = {name: [] for name in LOWP_ROUTE_NAMES}
    cache_counters_before = _descriptor_cache_counter_snapshots(runtimes)
    try:
        for warmup_block in range(warmups // 4):
            for execution_position, name in enumerate(
                _balanced_route_order(warmup_block)
            ):
                _assign_model_lowp_runtime(model, runtimes[name])
                _forward_only_step(
                    name,
                    model,
                    tokens,
                    targets,
                    config,
                    -2 - warmup_block,
                    execution_position,
                    forward_routes,
                    warmup=True,
                )

        cache_counters_after_warmups = (
            _descriptor_cache_counter_snapshots(runtimes)
        )
        sample_index_by_route = {name: 0 for name in LOWP_ROUTE_NAMES}
        for block_index in range(rounds // 4):
            for execution_position, name in enumerate(
                _balanced_route_order(block_index)
            ):
                _assign_model_lowp_runtime(model, runtimes[name])
                sample_index = sample_index_by_route[name]
                record = _forward_only_step(
                    name,
                    model,
                    tokens,
                    targets,
                    config,
                    sample_index,
                    execution_position,
                    forward_routes,
                    warmup=False,
                )
                record["measurement_phase"] = (
                    "same_model_forward_only_crossover"
                )
                record["crossover_superblock"] = block_index
                record["crossover_macroblock"] = block_index // 2
                record["complement_index"] = block_index % 2
                record["global_call_index"] = (
                    block_index * 8 + execution_position
                )
                record["route_sample_index"] = sample_index
                records[name].append(record)
                sample_index_by_route[name] += 1
    finally:
        _assign_model_lowp_runtime(model, original_runtime)
        activate_bound_model_forward_route(model)
    cache_counters_after = _descriptor_cache_counter_snapshots(runtimes)

    metrics = {
        metric: _balanced_block_metric_summary(
            records[LOWP_ROUTE_NAMES[0]],
            records[LOWP_ROUTE_NAMES[1]],
            metric,
        )
        for metric in ("forward_ms", "wall_ms")
    }
    forward = metrics["forward_ms"]
    sufficient_samples = bool(forward["sufficient_samples"])
    stationary = (
        sufficient_samples
        and forward["common_mode_drift_bootstrap_ci90_lower_percent"]
        >= -stationarity_tolerance_percent
        and forward["common_mode_drift_bootstrap_ci90_upper_percent"]
        <= stationarity_tolerance_percent
        and forward["common_mode_quartile_range_percent"]
        <= stationarity_tolerance_percent
    )
    gate = {
        "mx_forward_faster": (
            sufficient_samples
            and
            forward[
                "fp8_minus_mx_block_trimmed_mean_bootstrap_ci90_lower_ms"
            ] > 0.0
        ),
        "sufficient_macroblocks": sufficient_samples,
        "stationary_within_tolerance": stationary,
        "stationarity_tolerance_percent": stationarity_tolerance_percent,
    }
    gate["passed"] = all(bool(value) for key, value in gate.items() if key != "stationarity_tolerance_percent")
    return (
        {
            "schema": "same_model_forward_only_crossover_v1",
            "design": (
                "complement_paired_abba_baab_with_alternating_"
                "macroblock_phase"
            ),
            "causal_variable": "low_precision_attention_runtime",
            "same_model_and_parameter_addresses": True,
            "training_autograd_enabled": True,
            "backward_or_optimizer_between_samples": False,
            "weights_updated": False,
            "samples_per_route": rounds,
            "superblocks": rounds // 4,
            "macroblocks": rounds // 8,
            "warmups_per_route": warmups,
            "descriptor_cache_telemetry": {
                "schema": "forward_only_tma_descriptor_cache_telemetry_v1",
                "counter_scope": "calling_host_thread",
                "warmup": _descriptor_cache_counter_interval(
                    cache_counters_before,
                    cache_counters_after_warmups,
                ),
                "measured_forward_only": (
                    _descriptor_cache_counter_interval(
                        cache_counters_after_warmups,
                        cache_counters_after,
                    )
                ),
                "combined_warmup_and_measurement": (
                    _descriptor_cache_counter_interval(
                        cache_counters_before,
                        cache_counters_after,
                    )
                ),
            },
            "metrics": metrics,
            "gate": gate,
        },
        records,
    )


def _same_model_runtime_crossover(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    runtimes: dict[str, LowpAttentionRuntime],
    tokens: torch.Tensor,
    targets: torch.Tensor,
    config: Config,
    forward_routes: dict[str, str],
    *,
    rounds: int,
    warmups: int,
    backward_equality_tolerance_percent: float,
    stationarity_tolerance_percent: float,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Time both routes on one model without changing its parameters."""
    if tuple(runtimes) != LOWP_ROUTE_NAMES:
        raise ValueError("crossover runtimes must follow LOWP_ROUTE_NAMES")
    if rounds <= 0 or rounds % 8:
        raise ValueError(
            "crossover rounds must be a positive multiple-of-eight sample "
            "count per route"
        )
    if warmups <= 0 or warmups % 8:
        raise ValueError(
            "crossover warmups must be a positive multiple-of-eight count "
            "per route"
        )
    if (
        not math.isfinite(backward_equality_tolerance_percent)
        or backward_equality_tolerance_percent <= 0.0
    ):
        raise ValueError("backward equality tolerance must be positive")
    if (
        not math.isfinite(stationarity_tolerance_percent)
        or stationarity_tolerance_percent <= 0.0
    ):
        raise ValueError("stationarity tolerance must be positive")

    original_runtime = getattr(model, "lowp_attention_runtime", None)
    if original_runtime is None:
        raise TypeError("crossover model must already be low precision")
    learning_rates = [group["lr"] for group in optimizer.param_groups]
    records = {name: [] for name in LOWP_ROUTE_NAMES}
    try:
        # lr=0 retains the complete fused-AdamW launch while holding the same
        # physical weights fixed across both routes and every measured round.
        for group in optimizer.param_groups:
            group["lr"] = 0.0
        _reset_optimizer_state(optimizer)
        torch.cuda.synchronize()
        for warmup_block in range(warmups // 4):
            order = _balanced_route_order(warmup_block)
            for execution_position, name in enumerate(order):
                _assign_model_lowp_runtime(model, runtimes[name])
                _step(
                    name,
                    model,
                    optimizer,
                    tokens,
                    targets,
                    config,
                    -2 - warmup_block,
                    0,
                    execution_position,
                    forward_routes,
                    warmup=True,
                )
        sample_index_by_route = {name: 0 for name in LOWP_ROUTE_NAMES}
        for block_index in range(rounds // 4):
            order = _balanced_route_order(block_index)
            for execution_position, name in enumerate(order):
                _assign_model_lowp_runtime(model, runtimes[name])
                sample_index = sample_index_by_route[name]
                record = _step(
                    name,
                    model,
                    optimizer,
                    tokens,
                    targets,
                    config,
                    sample_index,
                    0,
                    execution_position,
                    forward_routes,
                    warmup=False,
                )
                record["measurement_phase"] = (
                    "same_model_runtime_crossover"
                )
                record["crossover_superblock"] = block_index
                record["crossover_macroblock"] = block_index // 2
                record["complement_index"] = block_index % 2
                record["global_call_index"] = (
                    block_index * 8 + execution_position
                )
                record["route_sample_index"] = sample_index
                records[name].append(record)
                sample_index_by_route[name] += 1
    finally:
        _assign_model_lowp_runtime(model, original_runtime)
        activate_bound_model_forward_route(model)
        for group, learning_rate in zip(
            optimizer.param_groups, learning_rates, strict=True
        ):
            group["lr"] = learning_rate
        _reset_optimizer_state(optimizer)

    metrics = {
        metric: _balanced_block_metric_summary(
            records[LOWP_ROUTE_NAMES[0]],
            records[LOWP_ROUTE_NAMES[1]],
            metric,
        )
        for metric in (
            "forward_ms",
            "backward_ms",
            "optimizer_ms",
            "step_ms",
            "wall_ms",
        )
    }
    backward = metrics["backward_ms"]
    backward_midpoint = 0.5 * (
        backward["mx_median_ms"] + backward["fp8_median_ms"]
    )
    backward_delta_percent = (
        abs(backward["fp8_minus_mx_block_trimmed_mean_ms"])
        / backward_midpoint
        * 100.0
    )
    backward_ci90_percent = (
        backward[
            "fp8_minus_mx_block_trimmed_mean_bootstrap_ci90_lower_ms"
        ]
        / backward_midpoint
        * 100.0,
        backward[
            "fp8_minus_mx_block_trimmed_mean_bootstrap_ci90_upper_ms"
        ]
        / backward_midpoint
        * 100.0,
    )
    forward = metrics["forward_ms"]
    step = metrics["step_ms"]
    sufficient_samples = all(
        bool(metrics[metric]["sufficient_samples"])
        for metric in ("forward_ms", "backward_ms", "step_ms")
    )
    stationary = sufficient_samples and all(
        metrics[metric][
            "common_mode_drift_bootstrap_ci90_lower_percent"
        ] >= -stationarity_tolerance_percent
        and metrics[metric][
            "common_mode_drift_bootstrap_ci90_upper_percent"
        ] <= stationarity_tolerance_percent
        and metrics[metric]["common_mode_quartile_range_percent"]
        <= stationarity_tolerance_percent
        for metric in ("forward_ms", "backward_ms", "step_ms")
    )
    gate = {
        "mx_forward_faster_diagnostic": (
            forward[
                "fp8_minus_mx_block_trimmed_mean_bootstrap_ci90_lower_ms"
            ] > 0.0
        ),
        "mx_step_faster": (
            sufficient_samples
            and
            step[
                "fp8_minus_mx_block_trimmed_mean_bootstrap_ci90_lower_ms"
            ] > 0.0
        ),
        "backward_equal_within_tolerance": (
            sufficient_samples
            and
            backward_ci90_percent[0]
            >= -backward_equality_tolerance_percent
            and backward_ci90_percent[1]
            <= backward_equality_tolerance_percent
        ),
        "backward_paired_delta_percent": backward_delta_percent,
        "backward_equality_tolerance_percent": (
            backward_equality_tolerance_percent
        ),
        "backward_bootstrap_ci90_percent": list(backward_ci90_percent),
        "sufficient_macroblocks": sufficient_samples,
        "stationary_within_tolerance": stationary,
        "stationarity_tolerance_percent": stationarity_tolerance_percent,
    }
    gate["passed"] = all(
        bool(gate[key])
        for key in (
            "mx_step_faster",
            "backward_equal_within_tolerance",
            "sufficient_macroblocks",
            "stationary_within_tolerance",
        )
    )
    summary = {
        "schema": "same_model_runtime_crossover_v2",
        "design": (
            "complement_paired_abba_baab_with_alternating_macroblock_phase"
        ),
        "causal_variable": "low_precision_attention_runtime",
        "same_model_and_parameter_addresses": True,
        "route_runtime_workspaces_retained": True,
        "weights_updated": False,
        "optimizer_work_included": True,
        "optimizer_state_policy": (
            "continuous_shared_zero_lr_after_one_pre_warmup_reset"
        ),
        "optimizer_state_reset_before_every_sample": False,
        "samples_per_route": rounds,
        "superblocks": rounds // 4,
        "macroblocks": rounds // 8,
        "warmups_per_route": warmups,
        "metrics": metrics,
        "gate": gate,
    }
    return summary, records


def _same_model_blocked_runtime_crossover(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    runtimes: dict[str, LowpAttentionRuntime],
    tokens: torch.Tensor,
    targets: torch.Tensor,
    config: Config,
    forward_routes: dict[str, str],
    *,
    macroblocks: int,
    preconditioning_macroblocks: int,
    acclimation_steps_per_block: int,
    measured_steps_per_block: int,
    backward_equality_tolerance_percent: float,
    stationarity_tolerance_percent: float,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Measure steady fixed-route blocks on one frozen model allocation."""
    if tuple(runtimes) != LOWP_ROUTE_NAMES:
        raise ValueError(
            "blocked crossover runtimes must follow LOWP_ROUTE_NAMES"
        )
    if macroblocks < MIN_BLOCKED_MACROBLOCKS or macroblocks % 4:
        raise ValueError(
            "blocked crossover macroblocks must be a multiple of four at "
            f"least {MIN_BLOCKED_MACROBLOCKS}"
        )
    if preconditioning_macroblocks < 0 or preconditioning_macroblocks % 4:
        raise ValueError(
            "blocked crossover preconditioning macroblocks must be a "
            "nonnegative multiple of four"
        )
    if acclimation_steps_per_block <= 0:
        raise ValueError(
            "blocked crossover acclimation steps per block must be positive"
        )
    if measured_steps_per_block <= 0:
        raise ValueError(
            "blocked crossover measured steps per block must be positive"
        )
    if (
        not math.isfinite(backward_equality_tolerance_percent)
        or backward_equality_tolerance_percent <= 0.0
    ):
        raise ValueError("backward equality tolerance must be positive")
    if (
        not math.isfinite(stationarity_tolerance_percent)
        or stationarity_tolerance_percent <= 0.0
    ):
        raise ValueError("stationarity tolerance must be positive")

    original_runtime = getattr(model, "lowp_attention_runtime", None)
    if original_runtime is None:
        raise TypeError("blocked crossover model must already be low precision")
    learning_rates = [group["lr"] for group in optimizer.param_groups]
    records = {name: [] for name in LOWP_ROUTE_NAMES}
    sample_index_by_route = {name: 0 for name in LOWP_ROUTE_NAMES}
    try:
        # This is a timing crossover, not an optimization trajectory.  Zero
        # LR preserves one exact parameter allocation and value while the real
        # fused AdamW, set_to_none gradient lifecycle, and backward all run.
        for group in optimizer.param_groups:
            group["lr"] = 0.0
        _reset_optimizer_state(optimizer)
        torch.cuda.synchronize()
        cache_counters_before_preconditioning = (
            _descriptor_cache_counter_snapshots(runtimes)
        )
        # Bring clocks, the caching allocator, fused AdamW state, and both
        # retained forward routes to the same sustained-training regime before
        # collecting any samples.  Each even ABBA/BAAB preconditioning pair
        # gives both routes the same number and positions of real full steps.
        preconditioning_steps_per_block = (
            acclimation_steps_per_block + measured_steps_per_block
        )
        for preconditioning_macroblock in range(
            preconditioning_macroblocks
        ):
            order = _blocked_route_order(preconditioning_macroblock)
            for name in order:
                _assign_model_lowp_runtime(model, runtimes[name])
                for _ in range(preconditioning_steps_per_block):
                    _acclimate_block_iteration(
                        name,
                        model,
                        optimizer,
                        tokens,
                        targets,
                        config,
                        forward_routes,
                    )
        cache_counters_after_preconditioning = (
            _descriptor_cache_counter_snapshots(runtimes)
        )
        for macroblock in range(macroblocks):
            order = _blocked_route_order(macroblock)
            for block_position, name in enumerate(order):
                global_block = macroblock * 4 + block_position
                _assign_model_lowp_runtime(model, runtimes[name])
                for _ in range(acclimation_steps_per_block):
                    _acclimate_block_iteration(
                        name,
                        model,
                        optimizer,
                        tokens,
                        targets,
                        config,
                        forward_routes,
                    )
                for step_in_block in range(measured_steps_per_block):
                    sample_index = sample_index_by_route[name]
                    record = _step(
                        name,
                        model,
                        optimizer,
                        tokens,
                        targets,
                        config,
                        sample_index,
                        0,
                        block_position,
                        forward_routes,
                        warmup=False,
                    )
                    record["measurement_phase"] = (
                        "same_model_blocked_runtime_crossover"
                    )
                    record["blocked_macroblock"] = macroblock
                    record["blocked_complement_pair"] = macroblock // 2
                    record["blocked_phase_cycle"] = macroblock // 4
                    record["blocked_pair_position_in_phase_cycle"] = (
                        (macroblock // 2) % 2
                    )
                    record["blocked_pair_orientation"] = (
                        (macroblock // 2) % 2
                    )
                    record["blocked_complement_index"] = macroblock % 2
                    record["blocked_block_position"] = block_position
                    record["blocked_step_in_block"] = step_in_block
                    record["global_block_index"] = global_block
                    record["global_call_index"] = (
                        global_block * measured_steps_per_block
                        + step_in_block
                    )
                    record["route_sample_index"] = sample_index
                    record["acclimation_steps_before_block"] = (
                        acclimation_steps_per_block
                    )
                    if not bool(record["finite"]):
                        raise RuntimeError(
                            f"non-finite loss in blocked route {name}"
                        )
                    records[name].append(record)
                    sample_index_by_route[name] += 1
        cache_counters_after_measurement = (
            _descriptor_cache_counter_snapshots(runtimes)
        )
    finally:
        _assign_model_lowp_runtime(model, original_runtime)
        activate_bound_model_forward_route(model)
        for group, learning_rate in zip(
            optimizer.param_groups, learning_rates, strict=True
        ):
            group["lr"] = learning_rate
        _reset_optimizer_state(optimizer)

    metrics = {
        metric: _blocked_metric_summary(
            records[LOWP_ROUTE_NAMES[0]],
            records[LOWP_ROUTE_NAMES[1]],
            metric,
            macroblocks=macroblocks,
            measured_steps_per_block=measured_steps_per_block,
        )
        for metric in ("forward_ms", "backward_ms", "step_ms")
    }
    forward = metrics["forward_ms"]
    backward = metrics["backward_ms"]
    step = metrics["step_ms"]
    backward_midpoint = 0.5 * (
        backward["mx_phase_cycle_median_ms"]
        + backward["fp8_phase_cycle_median_ms"]
    )
    backward_delta_percent = (
        abs(backward["fp8_minus_mx_phase_cycle_trimmed_mean_ms"])
        / backward_midpoint
        * 100.0
    )
    backward_ci90_percent = (
        backward[
            "fp8_minus_mx_phase_cycle_trimmed_mean_bootstrap_ci90_lower_ms"
        ]
        / backward_midpoint
        * 100.0,
        backward[
            "fp8_minus_mx_phase_cycle_trimmed_mean_bootstrap_ci90_upper_ms"
        ]
        / backward_midpoint
        * 100.0,
    )
    sufficient_samples = all(
        bool(metrics[metric]["sufficient_samples"])
        for metric in ("forward_ms", "backward_ms", "step_ms")
    )
    stationary = sufficient_samples and all(
        metrics[metric][
            "common_mode_drift_bootstrap_ci90_lower_percent"
        ] >= -stationarity_tolerance_percent
        and metrics[metric][
            "common_mode_drift_bootstrap_ci90_upper_percent"
        ] <= stationarity_tolerance_percent
        and metrics[metric]["common_mode_quartile_range_percent"]
        <= stationarity_tolerance_percent
        for metric in ("forward_ms", "backward_ms", "step_ms")
    )
    gate = {
        "mx_forward_faster_diagnostic": (
            forward[
                "fp8_minus_mx_phase_cycle_trimmed_mean_bootstrap_ci90_lower_ms"
            ] > 0.0
        ),
        "mx_step_faster": (
            sufficient_samples
            and step[
                "fp8_minus_mx_phase_cycle_trimmed_mean_bootstrap_ci90_lower_ms"
            ] > 0.0
        ),
        "backward_equal_within_tolerance": (
            sufficient_samples
            and backward_ci90_percent[0]
            >= -backward_equality_tolerance_percent
            and backward_ci90_percent[1]
            <= backward_equality_tolerance_percent
        ),
        "backward_paired_delta_percent": backward_delta_percent,
        "backward_equality_tolerance_percent": (
            backward_equality_tolerance_percent
        ),
        "backward_bootstrap_ci90_percent": list(backward_ci90_percent),
        "sufficient_phase_cycles": sufficient_samples,
        # Compatibility alias for consumers of the v2 schema.
        "sufficient_macroblocks": sufficient_samples,
        "stationary_within_tolerance": stationary,
        "stationarity_tolerance_percent": stationarity_tolerance_percent,
    }
    gate["passed"] = all(
        bool(gate[key])
        for key in (
            "mx_step_faster",
            "backward_equal_within_tolerance",
            "sufficient_phase_cycles",
            "stationary_within_tolerance",
        )
    )
    summary = {
        "schema": "same_model_blocked_runtime_crossover_v3",
        "design": (
            "covariance_preserving_full_phase_cycle_whole_route_blocks_"
            "abba_baab_baab_abba"
        ),
        "estimator_unit": (
            "two_adjacent_complement_pairs_four_raw_macroblocks"
        ),
        "causal_variable": "low_precision_attention_runtime",
        "same_model_and_parameter_addresses": True,
        "route_runtime_workspaces_retained": True,
        "route_bound_once_per_block": True,
        "production_gradient_zeroing": "set_to_none_true_before_every_step",
        "per_step_measurement": "real_step_with_forward_backward_fused_adamw",
        "weights_updated": False,
        "optimizer_work_included": True,
        "optimizer_state_policy": (
            "continuous_shared_zero_lr_after_one_pre_block_reset"
        ),
        "optimizer_state_reset_before_every_sample": False,
        "macroblocks": macroblocks,
        "complement_pairs": macroblocks // MACROBLOCKS_PER_COMPLEMENT_PAIR,
        "phase_cycles": macroblocks // MACROBLOCKS_PER_PHASE_CYCLE,
        "minimum_phase_cycles_for_gate": MIN_BLOCKED_PHASE_CYCLES,
        "route_blocks": macroblocks * 4,
        "preconditioning_macroblocks": preconditioning_macroblocks,
        "preconditioning_steps_per_block": (
            acclimation_steps_per_block + measured_steps_per_block
        ),
        "preconditioning_full_iterations": (
            preconditioning_macroblocks
            * 4
            * (acclimation_steps_per_block + measured_steps_per_block)
        ),
        "preconditioning_records_retained": False,
        "acclimation_steps_per_block": acclimation_steps_per_block,
        "measured_steps_per_block": measured_steps_per_block,
        "samples_per_route": macroblocks * 2 * measured_steps_per_block,
        "acclimation_records_retained": False,
        "descriptor_cache_telemetry": {
            "schema": "blocked_tma_descriptor_cache_telemetry_v1",
            "counter_scope": "calling_host_thread",
            "measured_interval_includes_block_acclimation": True,
            "preconditioning": _descriptor_cache_counter_interval(
                cache_counters_before_preconditioning,
                cache_counters_after_preconditioning,
            ),
            "measured_blocked_crossover": (
                _descriptor_cache_counter_interval(
                    cache_counters_after_preconditioning,
                    cache_counters_after_measurement,
                )
            ),
            "combined_preconditioning_and_measurement": (
                _descriptor_cache_counter_interval(
                    cache_counters_before_preconditioning,
                    cache_counters_after_measurement,
                )
            ),
        },
        "metrics": metrics,
        "gate": gate,
    }
    return summary, records


def _route_summary(
    records: list[dict[str, Any]],
    config: Config,
    training_batches: int,
) -> dict[str, Any]:
    medians = {
        key: statistics.median(float(record[key]) for record in records)
        for key in (
            "forward_ms",
            "backward_ms",
            "optimizer_ms",
            "step_ms",
            "wall_ms",
        )
    }
    step_seconds = medians["step_ms"] / 1000.0
    useful_flops = _useful_flops(config)
    final_window = records[-min(len(records), training_batches) :]
    per_batch: dict[str, Any] = {}
    for batch_index in range(training_batches):
        batch_records = [
            record for record in records if record["batch"] == batch_index
        ]
        if not batch_records:
            continue
        first_loss = float(batch_records[0]["loss"])
        final_loss = float(batch_records[-1]["loss"])
        per_batch[str(batch_index)] = {
            "observations": len(batch_records),
            "first_loss": first_loss,
            "final_loss": final_loss,
            "loss_reduction_percent": (
                (1.0 - final_loss / first_loss) * 100.0
                if first_loss != 0.0
                else float("nan")
            ),
        }
    return {
        "timing": {
            **medians,
            "tokens_per_second": config.sequence / step_seconds,
            "useful_tflops": useful_flops / step_seconds / 1.0e12,
            "useful_mfu_at_2250_tflops": (
                useful_flops / step_seconds / 2.25e15
            ),
        },
        "optimization_proxy": {
            "losses": [float(record["loss"]) for record in records],
            "first_loss": float(records[0]["loss"]),
            "last_loss": float(records[-1]["loss"]),
            "last_cycle_median_loss": statistics.median(
                float(record["loss"]) for record in final_window
            ),
            "all_steps_finite": all(bool(record["finite"]) for record in records),
            "per_batch": per_batch,
        },
    }


def _make_runtime(
    config: Config,
    rope: tuple[torch.Tensor, torch.Tensor],
    extension_path: Path,
    module_name: str,
    *,
    route_slot: str,
    backward_probability_correction: float | None,
    q_quant_scale: float,
    k_quant_scale: float,
    projection_weight_scale_2d: bool,
    v_mxfp4_scale_2d: bool,
    backward_q_gain: float | None,
    backward_k_gain: float | None,
    backward_v_gain: float | None,
    backward_v_weight_gain: float | None,
    backward_exp2_degree: int = 2,
    backward_exp2_period: int | None = None,
    backward_reuse_quantized_p: bool = False,
    backward_control_source: Path | str | None = None,
    backward_control_sha256: str | None = None,
    backward_control_bytes: int | None = None,
    backward_match_forward_operands: bool = False,
    per_block_qk_scales: bool = False,
    experimental_split_v_backward: bool = False,
    backward_forward_mx_probability_replay: bool = False,
    backward_forward_mx_probability_scale_handoff: bool | None = None,
    qkv_projection_format: str = "nvfp4",
    shared_backward_runtime: LowpAttentionRuntime | None = None,
) -> tuple[LowpAttentionRuntime, dict[str, Any]]:
    extension, topology = _load_forward(extension_path, module_name, config)
    _require_forward_route_slot(
        route_slot,
        topology,
        head_dim=config.head_dim,
    )
    runtime = LowpAttentionRuntime(
        config,
        rope,
        forward_extension=extension,
        forward_topology=topology,
        loss_scale=2.0**16,
        gradient_global_scale=2.0**-8,
        projection_dgrad=(
            "nvfp4" if config.head_dim == 128 else "bf16"
        ),
        qkv_projection_format=qkv_projection_format,
        backward_probability_correction=backward_probability_correction,
        q_quant_scale=q_quant_scale,
        k_quant_scale=k_quant_scale,
        projection_weight_scale_2d=projection_weight_scale_2d,
        v_mxfp4_scale_2d=v_mxfp4_scale_2d,
        backward_exp2_degree=backward_exp2_degree,
        backward_exp2_period=backward_exp2_period,
        backward_reuse_quantized_p=backward_reuse_quantized_p,
        backward_control_source=backward_control_source,
        backward_control_sha256=backward_control_sha256,
        backward_control_bytes=backward_control_bytes,
        backward_match_forward_operands=backward_match_forward_operands,
        per_block_qk_scales=per_block_qk_scales,
        experimental_split_v_backward=experimental_split_v_backward,
        backward_forward_mx_probability_replay=(
            backward_forward_mx_probability_replay
        ),
        backward_forward_mx_probability_scale_handoff=(
            backward_forward_mx_probability_scale_handoff
        ),
        shared_backward_runtime=shared_backward_runtime,
    )
    for field, value in (
        ("q", backward_q_gain),
        ("k", backward_k_gain),
        ("v", backward_v_gain),
    ):
        if value is None:
            continue
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"backward {field.upper()} gain must be positive")
        setattr(runtime, f"backward_{field}_gain", float(value))
    if backward_v_weight_gain is None:
        runtime.backward_v_weight_gain = runtime.backward_v_gain
    else:
        if (
            not math.isfinite(backward_v_weight_gain)
            or backward_v_weight_gain <= 0.0
        ):
            raise ValueError("backward V weight gain must be positive")
        runtime.backward_v_weight_gain = float(backward_v_weight_gain)
    return runtime, topology


def _argument_was_provided(argv: list[str], option: str) -> bool:
    """Return whether an argparse option was explicitly supplied."""
    return any(
        argument == option or argument.startswith(option + "=")
        for argument in argv
    )


def _resolve_model_preset_options(
    args: argparse.Namespace,
    config: Config,
    argv: list[str],
) -> None:
    """Select the D128 contract without changing the established D64 CLI."""
    if config.head_dim != 128:
        return
    d128_contract = (
        ("backward_exp2_degree", "--backward-exp2-degree", 1),
        ("backward_exp2_period", "--backward-exp2-period", 0),
        (
            "mx_backward_reuse_quantized_p",
            "--mx-backward-reuse-quantized-p",
            True,
        ),
        (
            "fp8_backward_reuse_quantized_p",
            "--fp8-backward-reuse-quantized-p",
            True,
        ),
        ("mx_qkv_projection_format", "--mx-qkv-projection-format", "nvfp4"),
        (
            "fp8_qkv_projection_format",
            "--fp8-qkv-projection-format",
            "nvfp4",
        ),
        (
            "mx_backward_match_forward_operands",
            "--mx-backward-match-forward-operands",
            False,
        ),
        (
            "fp8_backward_match_forward_operands",
            "--fp8-backward-match-forward-operands",
            False,
        ),
        ("mx_per_block_qk_scales", "--mx-per-block-qk-scales", False),
        ("fp8_per_block_qk_scales", "--fp8-per-block-qk-scales", False),
        (
            "mx_experimental_split_v_backward",
            "--mx-experimental-split-v-backward",
            False,
        ),
        (
            "mx_backward_forward_probability_replay",
            "--mx-backward-forward-probability-replay",
            False,
        ),
        (
            "mx_backward_forward_probability_scale_handoff",
            "--mx-backward-forward-probability-scale-handoff",
            False,
        ),
        ("v_mxfp4_scaling", "--v-mxfp4-scaling", "1d"),
    )
    for field, option, required in d128_contract:
        negative_option = "--no-" + option.removeprefix("--")
        explicit = _argument_was_provided(
            argv,
            option,
        ) or _argument_was_provided(argv, negative_option)
        actual = getattr(args, field)
        if explicit and actual != required:
            raise ValueError(
                f"{option}={actual!r} is incompatible with the D128 "
                f"comparison contract; expected {required!r}"
            )
        setattr(args, field, required)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-preset",
        choices=MODEL_PRESETS,
        default=DEFAULT_MODEL_PRESET,
        help=(
            "model architecture; llama3.1-8b selects H4096/D128 and the "
            "native NVFP4 projection contract"
        ),
    )
    parser.add_argument(
        "--layers",
        type=int,
        help="override preset depth only for integration smoke tests",
    )
    parser.add_argument("--sequence", type=int, default=4096)
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--training-batches", type=int, default=4)
    parser.add_argument(
        "--crossover-rounds",
        type=int,
        default=160,
        help=(
            "diagnostic per-sample alternating training samples per route; "
            "must be divisible by eight for complement-paired macroblocks"
        ),
    )
    parser.add_argument(
        "--crossover-warmups",
        type=int,
        default=160,
        help=(
            "warmup samples per route before the diagnostic crossover; "
            "must be divisible by eight"
        ),
    )
    parser.add_argument(
        "--blocked-crossover-macroblocks",
        type=int,
        default=80,
        help=(
            "raw ABBA/BAAB whole-route macroblocks for the production-"
            "faithful full-step gate; every four macroblocks form one "
            "covariance-preserving phase cycle, so the count must be a "
            f"multiple of four and at least {MIN_BLOCKED_MACROBLOCKS}"
        ),
    )
    parser.add_argument(
        "--blocked-crossover-preconditioning-macroblocks",
        type=int,
        default=16,
        help=(
            "untimed orientation-balanced ABBA/BAAB full-step macroblocks "
            "used to stabilize clocks and allocator state before the blocked "
            "speed gate; must be a multiple of four"
        ),
    )
    parser.add_argument(
        "--blocked-crossover-acclimation-steps",
        type=int,
        default=4,
        help="untimed fixed-route acclimation steps before each measured block",
    )
    parser.add_argument(
        "--blocked-crossover-measured-steps",
        type=int,
        default=4,
        help="consecutive real full steps measured in each fixed-route block",
    )
    parser.add_argument(
        "--backward-equality-tolerance-percent",
        type=float,
        default=0.5,
        help="maximum paired backward-time delta accepted as equal",
    )
    parser.add_argument(
        "--crossover-stationarity-tolerance-percent",
        type=float,
        default=0.5,
        help=(
            "maximum early-to-late route-blind common-mode drift accepted "
            "for each timed phase"
        ),
    )
    parser.add_argument(
        "--require-causal-speed-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "exit nonzero unless same-model MX forward/step are faster and "
            "backward is equal"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument(
        "--mx-extension",
        type=Path,
        default=Path("/tmp/_C_causal_scale_reuse_policy_d64.so"),
    )
    parser.add_argument(
        "--mx-module", default="_C_causal_scale_reuse_policy_d64"
    )
    parser.add_argument(
        "--fp8-extension",
        type=Path,
        default=Path(
            "/tmp/_C_causal_gqa_nvfp4_fp8pv_exact_d64_sm100_eval_20260818.so"
        ),
    )
    parser.add_argument(
        "--fp8-module",
        default="_C_causal_gqa_nvfp4_fp8pv_exact_d64_sm100_eval_20260818",
    )
    parser.add_argument(
        "--backward-control-source",
        type=Path,
        help="precomposed generated CuTe backward control Python source",
    )
    parser.add_argument(
        "--backward-control-sha256",
        help="required SHA256 for --backward-control-source",
    )
    parser.add_argument(
        "--backward-control-bytes",
        type=int,
        help="required byte size for --backward-control-source",
    )
    parser.add_argument(
        "--mx-backward-probability-correction", type=float, default=1.0
    )
    parser.add_argument(
        "--mx-backward-forward-probability-replay",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "replay the MX forward probability representation in backward; "
            "requires a compatible D4ALL forward topology"
        ),
    )
    parser.add_argument(
        "--mx-backward-forward-probability-scale-handoff",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "publish MX forward probability scales to backward; disabled in "
            "the production matched-backward comparison"
        ),
    )
    parser.add_argument(
        "--fp8-backward-probability-correction", type=float, default=1.0
    )
    for route in ("mx", "fp8"):
        for field in ("q", "k", "v"):
            parser.add_argument(
                f"--{route}-backward-{field}-gain",
                type=float,
                default=1.0,
            )
        parser.add_argument(
            f"--{route}-backward-v-weight-gain",
            type=float,
            default=1.0,
            help=(
                "diagnostic V projection-weight gain; V-to-dX keeps the "
                "ordinary V gain"
            ),
        )
    parser.add_argument("--q-quant-scale", type=float, default=2.25)
    parser.add_argument("--k-quant-scale", type=float, default=2.0)
    parser.add_argument(
        "--backward-exp2-degree",
        type=int,
        choices=(1, 2),
        default=2,
        help="polynomial degree used when --backward-exp2-period is explicit",
    )
    parser.add_argument(
        "--backward-exp2-period",
        type=int,
        choices=range(17),
        default=None,
        help=(
            "selective EX2 period; omission uses the measured-shape D64 "
            "dispatch (including degree-1/period-2 for S4096 H=32/8), "
            "while 0 "
            "explicitly forces native EX2"
        ),
    )
    for route in ("mx", "fp8"):
        parser.add_argument(
            f"--{route}-backward-reuse-quantized-p",
            action=argparse.BooleanOptionalAction,
            default=False,
            help=(
                f"reuse the {route.upper()} backward kernel's E4M3-rounded "
                "probability between dV and dS; this does not replay the "
                "forward probability"
            ),
        )
        parser.add_argument(
            f"--{route}-backward-match-forward-operands",
            action=argparse.BooleanOptionalAction,
            default=True,
            help=(
                f"publish {route.upper()} backward Q/K and, where applicable, "
                "V from the exact codes used by forward"
            ),
        )
        parser.add_argument(
            f"--{route}-per-block-qk-scales",
            action=argparse.BooleanOptionalAction,
            default=True,
            help=(
                f"publish {route.upper()} Q/K with one E4M3 scale per "
                "logical row x K16; enabled for the production matched "
                "comparison"
            ),
        )
    parser.add_argument(
        "--mx-experimental-split-v-backward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "retain represented per-block MX Q/K while publishing backward "
            "E4M3 V directly from the projection accumulator"
        ),
    )
    parser.add_argument(
        "--projection-weight-scaling",
        choices=("1d", "2d"),
        default="2d",
    )
    parser.add_argument(
        "--v-mxfp4-scaling",
        choices=("1d", "2d"),
        default="1d",
    )
    parser.add_argument(
        "--fp8-qkv-projection-format",
        choices=("nvfp4", "e4m3"),
        default="e4m3",
        help=(
            "projection operand format for the exact FP8-PV route; E4M3 "
            "requires a compatible projection-native backward extension"
        ),
    )
    parser.add_argument(
        "--mx-qkv-projection-format",
        choices=("nvfp4", "e4m3"),
        default="e4m3",
        help="projection operand format for the MXFP4-PV route",
    )
    parser.add_argument(
        "--expected-projection-extension",
        type=Path,
        help=(
            "optional exact path required for the already-loaded QKV "
            "projection extension"
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    process_cpu_affinity = _process_cpu_affinity()
    _require_singleton_cpu_affinity(
        process_cpu_affinity,
        required=args.require_causal_speed_gate,
    )
    try:
        config = config_from_model_preset(
            args.model_preset,
            sequence=args.sequence,
            layers=args.layers,
        )
        _resolve_model_preset_options(args, config, sys.argv[1:])
        _require_memory_safe_matched_replicas(
            config,
            ROUTE_NAMES,
            operation="comparator",
        )
    except ValueError as error:
        parser.error(str(error))

    backward_control_identity = (
        args.backward_control_source,
        args.backward_control_sha256,
        args.backward_control_bytes,
    )
    if any(value is not None for value in backward_control_identity) and not all(
        value is not None for value in backward_control_identity
    ):
        parser.error(
            "--backward-control-source, --backward-control-sha256, and "
            "--backward-control-bytes must be supplied together"
        )
    if config.head_dim == 64:
        for route in ("mx", "fp8"):
            if getattr(args, f"{route}_qkv_projection_format") != "e4m3":
                parser.error(
                    f"the production {route.upper()} route requires "
                    f"--{route}-qkv-projection-format=e4m3 to avoid an "
                    "unfused V-layout materialization"
                )
            if (
                getattr(args, f"{route}_per_block_qk_scales")
                and getattr(args, f"{route}_qkv_projection_format") != "e4m3"
            ):
                parser.error(
                    f"--{route}-per-block-qk-scales requires "
                    f"--{route}-qkv-projection-format=e4m3"
                )
        if args.mx_experimental_split_v_backward and not (
            args.mx_qkv_projection_format == "e4m3"
            and args.mx_backward_match_forward_operands
            and args.mx_per_block_qk_scales
        ):
            parser.error(
                "--mx-experimental-split-v-backward requires the MX E4M3 "
                "projection, matched forward operands, and per-block Q/K "
                "scales"
            )
    else:
        if any(value is not None for value in backward_control_identity):
            parser.error(
                "D128 requires the generated shared-P backward control; "
                "do not supply a D64 precomposed control artifact"
            )
    required_projection_symbols = _required_projection_symbols(
        args,
        head_dim=config.head_dim,
    )
    projection_extension = _projection_extension_identity(
        required_projection_symbols,
        args.expected_projection_extension,
    )

    if torch.cuda.device_count() != 1:
        raise RuntimeError("expose exactly one GPU to the benchmark")
    if args.rounds < 3 or args.rounds % 3 != 0:
        raise ValueError("--rounds must be a positive multiple of three")
    if args.crossover_rounds <= 0 or args.crossover_rounds % 8:
        raise ValueError(
            "--crossover-rounds must be a positive multiple of eight"
        )
    if args.crossover_warmups <= 0 or args.crossover_warmups % 8:
        raise ValueError(
            "--crossover-warmups must be a positive multiple of eight"
        )
    if (
        args.blocked_crossover_macroblocks < MIN_BLOCKED_MACROBLOCKS
        or args.blocked_crossover_macroblocks % 4
    ):
        raise ValueError(
            "--blocked-crossover-macroblocks must be a multiple of four and "
            f"at least {MIN_BLOCKED_MACROBLOCKS}"
        )
    if (
        args.blocked_crossover_preconditioning_macroblocks < 0
        or args.blocked_crossover_preconditioning_macroblocks % 4
    ):
        raise ValueError(
            "--blocked-crossover-preconditioning-macroblocks must be a "
            "nonnegative multiple of four"
        )
    if args.blocked_crossover_acclimation_steps <= 0:
        raise ValueError(
            "--blocked-crossover-acclimation-steps must be positive"
        )
    if args.blocked_crossover_measured_steps <= 0:
        raise ValueError(
            "--blocked-crossover-measured-steps must be positive"
        )
    if (
        not math.isfinite(args.backward_equality_tolerance_percent)
        or args.backward_equality_tolerance_percent <= 0.0
    ):
        raise ValueError(
            "--backward-equality-tolerance-percent must be positive"
        )
    if (
        not math.isfinite(args.crossover_stationarity_tolerance_percent)
        or args.crossover_stationarity_tolerance_percent <= 0.0
    ):
        raise ValueError(
            "--crossover-stationarity-tolerance-percent must be positive"
        )
    if args.training_batches < 2:
        raise ValueError("use at least two training batches")
    torch.cuda.set_device(0)

    rope = (
        _make_llama3_rope(config)
        if config.head_dim == 128
        else _make_legacy_rope(config.sequence, config.head_dim)
    )
    effective_attention_provenance = _effective_attention_provenance(config)
    mx_runtime, mx_topology = _make_runtime(
        config,
        rope,
        args.mx_extension,
        args.mx_module,
        route_slot="mx",
        backward_probability_correction=(
            args.mx_backward_probability_correction
        ),
        q_quant_scale=args.q_quant_scale,
        k_quant_scale=args.k_quant_scale,
        projection_weight_scale_2d=(
            args.projection_weight_scaling == "2d"
        ),
        v_mxfp4_scale_2d=(args.v_mxfp4_scaling == "2d"),
        backward_q_gain=args.mx_backward_q_gain,
        backward_k_gain=args.mx_backward_k_gain,
        backward_v_gain=args.mx_backward_v_gain,
        backward_v_weight_gain=args.mx_backward_v_weight_gain,
        backward_exp2_degree=args.backward_exp2_degree,
        backward_exp2_period=args.backward_exp2_period,
        backward_control_source=args.backward_control_source,
        backward_control_sha256=args.backward_control_sha256,
        backward_control_bytes=args.backward_control_bytes,
        backward_reuse_quantized_p=(
            args.mx_backward_reuse_quantized_p
        ),
        backward_match_forward_operands=(
            args.mx_backward_match_forward_operands
        ),
        per_block_qk_scales=args.mx_per_block_qk_scales,
        experimental_split_v_backward=(
            args.mx_experimental_split_v_backward
        ),
        backward_forward_mx_probability_replay=(
            args.mx_backward_forward_probability_replay
        ),
        backward_forward_mx_probability_scale_handoff=(
            args.mx_backward_forward_probability_scale_handoff
        ),
        qkv_projection_format=args.mx_qkv_projection_format,
    )
    fp8_runtime, fp8_topology = _make_runtime(
        config,
        rope,
        args.fp8_extension,
        args.fp8_module,
        route_slot="fp8",
        backward_probability_correction=(
            args.fp8_backward_probability_correction
        ),
        q_quant_scale=args.q_quant_scale,
        k_quant_scale=args.k_quant_scale,
        projection_weight_scale_2d=(
            args.projection_weight_scaling == "2d"
        ),
        v_mxfp4_scale_2d=(args.v_mxfp4_scaling == "2d"),
        backward_q_gain=args.fp8_backward_q_gain,
        backward_k_gain=args.fp8_backward_k_gain,
        backward_v_gain=args.fp8_backward_v_gain,
        backward_v_weight_gain=args.fp8_backward_v_weight_gain,
        backward_exp2_degree=args.backward_exp2_degree,
        backward_exp2_period=args.backward_exp2_period,
        backward_control_source=args.backward_control_source,
        backward_control_sha256=args.backward_control_sha256,
        backward_control_bytes=args.backward_control_bytes,
        backward_reuse_quantized_p=(
            args.fp8_backward_reuse_quantized_p
        ),
        backward_match_forward_operands=(
            args.fp8_backward_match_forward_operands
        ),
        per_block_qk_scales=args.fp8_per_block_qk_scales,
        qkv_projection_format=args.fp8_qkv_projection_format,
        shared_backward_runtime=mx_runtime,
    )
    backward_route_contracts = _matched_backward_contracts(
        mx_runtime,
        fp8_runtime,
    )
    shared_backward_runner = _share_matched_backward_runner(
        mx_runtime,
        fp8_runtime,
    )
    if not all(shared_backward_runner.values()):
        raise RuntimeError("low-precision routes did not share one backward")
    mx_probability_replay_provenance = _mx_probability_replay_provenance(
        mx_runtime
    )
    forward_routes = {
        "nvfp4_qk_mxfp4_pv": str(mx_topology["route"]),
        "nvfp4_qk_fp8_pv_exact": str(fp8_topology["route"]),
    }

    torch.manual_seed(args.seed)
    bf16_model = Llama12B(config, rope, None)
    torch.manual_seed(args.seed)
    mx_model = Llama12B(config, rope, mx_runtime)
    torch.manual_seed(args.seed)
    fp8_model = Llama12B(config, rope, fp8_runtime)
    models = {
        "bf16_cute": bf16_model,
        "nvfp4_qk_mxfp4_pv": mx_model,
        "nvfp4_qk_fp8_pv_exact": fp8_model,
    }
    optimizers = {
        name: _optimizer(model, args.learning_rate)
        for name, model in models.items()
    }
    parameter_counts = {
        name: sum(parameter.numel() for parameter in model.parameters())
        for name, model in models.items()
    }

    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed + 101)
    token_batches = torch.randint(
        config.vocab,
        (args.training_batches, config.sequence),
        generator=generator,
        device="cuda",
    )
    target_batches = torch.roll(token_batches, shifts=-1, dims=1)

    initial_audit = _initial_state_audit(
        models,
        token_batches[0:1],
        target_batches[0:1],
        config,
        forward_routes,
    )

    # Compile every route and initialize its fused AdamW buffers before timing
    # without performing a hidden training update.
    for execution_position, name in enumerate(ROUTE_NAMES):
        _compile_without_updating(
            name,
            models[name],
            optimizers[name],
            token_batches[0:1],
            target_batches[0:1],
            config,
            execution_position,
            forward_routes,
        )

    timed_forward_dispatch_contracts = _timed_forward_dispatch_contracts(
        mx_runtime,
        fp8_runtime,
        mx_model,
        fp8_model,
    )
    crossover_runtimes = {
        "nvfp4_qk_mxfp4_pv": mx_runtime,
        "nvfp4_qk_fp8_pv_exact": fp8_runtime,
    }
    (
        forward_crossover_summary,
        forward_crossover_records,
    ) = _same_model_forward_crossover(
        mx_model,
        crossover_runtimes,
        token_batches[0:1],
        target_batches[0:1],
        config,
        forward_routes,
        rounds=args.crossover_rounds,
        warmups=args.crossover_warmups,
        stationarity_tolerance_percent=(
            args.crossover_stationarity_tolerance_percent
        ),
    )
    (
        blocked_crossover_summary,
        blocked_crossover_records,
    ) = _same_model_blocked_runtime_crossover(
        mx_model,
        optimizers["nvfp4_qk_mxfp4_pv"],
        crossover_runtimes,
        token_batches[0:1],
        target_batches[0:1],
        config,
        forward_routes,
        macroblocks=args.blocked_crossover_macroblocks,
        preconditioning_macroblocks=(
            args.blocked_crossover_preconditioning_macroblocks
        ),
        acclimation_steps_per_block=(
            args.blocked_crossover_acclimation_steps
        ),
        measured_steps_per_block=args.blocked_crossover_measured_steps,
        backward_equality_tolerance_percent=(
            args.backward_equality_tolerance_percent
        ),
        stationarity_tolerance_percent=(
            args.crossover_stationarity_tolerance_percent
        ),
    )
    # Retain the per-sample alternating crossover as a launch/state diagnostic.
    # The blocked crossover above is the production full-step speed gate.
    crossover_summary, crossover_records = _same_model_runtime_crossover(
        mx_model,
        optimizers["nvfp4_qk_mxfp4_pv"],
        crossover_runtimes,
        token_batches[0:1],
        target_batches[0:1],
        config,
        forward_routes,
        rounds=args.crossover_rounds,
        warmups=args.crossover_warmups,
        backward_equality_tolerance_percent=(
            args.backward_equality_tolerance_percent
        ),
        stationarity_tolerance_percent=(
            args.crossover_stationarity_tolerance_percent
        ),
    )

    records: dict[str, list[dict[str, Any]]] = {
        name: [] for name in ROUTE_NAMES
    }
    torch.cuda.reset_peak_memory_stats()
    for round_index in range(args.rounds):
        batch_index = round_index % args.training_batches
        offset = round_index % len(ROUTE_NAMES)
        order = ROUTE_NAMES[offset:] + ROUTE_NAMES[:offset]
        for execution_position, name in enumerate(order):
            record = _step(
                name,
                models[name],
                optimizers[name],
                token_batches[batch_index : batch_index + 1],
                target_batches[batch_index : batch_index + 1],
                config,
                round_index,
                batch_index,
                execution_position,
                forward_routes,
                warmup=False,
            )
            records[name].append(record)
            if not bool(record["finite"]):
                raise RuntimeError(f"non-finite loss in {name}")

    summaries = {
        name: _route_summary(route_records, config, args.training_batches)
        for name, route_records in records.items()
    }
    bf16_timing = summaries["bf16_cute"]["timing"]
    bf16_final_loss = summaries["bf16_cute"]["optimization_proxy"][
        "last_cycle_median_loss"
    ]
    comparisons: dict[str, Any] = {}
    for name in ROUTE_NAMES[1:]:
        timing = summaries[name]["timing"]
        final_loss = summaries[name]["optimization_proxy"][
            "last_cycle_median_loss"
        ]
        comparisons[name] = {
            "speedup_over_bf16": bf16_timing["step_ms"] / timing["step_ms"],
            "step_time_reduction_percent": (
                1.0 - timing["step_ms"] / bf16_timing["step_ms"]
            )
            * 100.0,
            "forward_speedup_over_bf16": (
                bf16_timing["forward_ms"] / timing["forward_ms"]
            ),
            "backward_speedup_over_bf16": (
                bf16_timing["backward_ms"] / timing["backward_ms"]
            ),
            "last_cycle_loss_ratio_to_bf16": final_loss / bf16_final_loss,
        }
    crossover_metrics = blocked_crossover_summary["metrics"]
    causal_forward = forward_crossover_summary["metrics"]["forward_ms"]
    causal_backward = crossover_metrics["backward_ms"]
    causal_step = crossover_metrics["step_ms"]
    causal_gate = {
        "mx_forward_faster": bool(
            forward_crossover_summary["gate"]["mx_forward_faster"]
        ),
        "mx_step_faster": bool(
            blocked_crossover_summary["gate"]["mx_step_faster"]
        ),
        "backward_equal_within_tolerance": bool(
            blocked_crossover_summary["gate"][
                "backward_equal_within_tolerance"
            ]
        ),
        "forward_sufficient_macroblocks": bool(
            forward_crossover_summary["gate"]["sufficient_macroblocks"]
        ),
        "full_step_sufficient_macroblocks": bool(
            blocked_crossover_summary["gate"]["sufficient_macroblocks"]
        ),
        "forward_stationary_within_tolerance": bool(
            forward_crossover_summary["gate"][
                "stationary_within_tolerance"
            ]
        ),
        "full_step_stationary_within_tolerance": bool(
            blocked_crossover_summary["gate"][
                "stationary_within_tolerance"
            ]
        ),
    }
    causal_gate = _classify_causal_speed_gate(causal_gate)
    comparisons["fp8_pv_vs_mxfp4_pv"] = {
        "measurement": (
            "same-model forward-only plus production-faithful blocked "
            "full-step runtime crossover"
        ),
        "mx_forward_speedup_over_fp8": (
            1.0
            + causal_forward["fp8_minus_mx_block_trimmed_mean_ms"]
            / causal_forward["mx_macroblock_median_ms"]
        ),
        "mx_step_speedup_over_fp8": (
            1.0
            + causal_step["fp8_minus_mx_phase_cycle_trimmed_mean_ms"]
            / causal_step["mx_phase_cycle_median_ms"]
        ),
        "fp8_step_time_delta_percent": (
            causal_step["fp8_minus_mx_phase_cycle_trimmed_mean_ms"]
            / causal_step["mx_phase_cycle_median_ms"]
        )
        * 100.0,
        "fp8_forward_time_delta_percent": (
            causal_forward["fp8_minus_mx_block_trimmed_mean_ms"]
            / causal_forward["mx_macroblock_median_ms"]
        )
        * 100.0,
        "fp8_backward_time_delta_percent": (
            causal_backward["fp8_minus_mx_phase_cycle_trimmed_mean_ms"]
            / causal_backward["mx_phase_cycle_median_ms"]
        )
        * 100.0,
        "gate": causal_gate,
    }
    mx_timing = summaries["nvfp4_qk_mxfp4_pv"]["timing"]
    fp8_timing = summaries["nvfp4_qk_fp8_pv_exact"]["timing"]
    comparisons["separate_replica_training_proxy_fp8_vs_mx"] = {
        "causal_speed_claim_valid": False,
        "reason": (
            "common-kernel allocation/address effects are aliased with route"
        ),
        "mx_step_ms": mx_timing["step_ms"],
        "fp8_step_ms": fp8_timing["step_ms"],
        "mx_forward_ms": mx_timing["forward_ms"],
        "fp8_forward_ms": fp8_timing["forward_ms"],
        "mx_backward_ms": mx_timing["backward_ms"],
        "fp8_backward_ms": fp8_timing["backward_ms"],
    }

    result = {
        "configuration": {
            **config.__dict__,
            **effective_attention_provenance,
            "batch": 1,
            "rounds": args.rounds,
            "training_batches": args.training_batches,
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "optimizer": "fused AdamW",
            "process_cpu_affinity": process_cpu_affinity,
            "process_cpu_affinity_singleton": (
                len(process_cpu_affinity) == 1
            ),
            "measurement": {
                "causal_speed": (
                    "same-model balanced forward-only plus blocked fixed-route "
                    "full-step crossover"
                ),
                "optimization_proxy": "three-replica Latin rotation",
            },
            "crossover_rounds": args.crossover_rounds,
            "crossover_warmups": args.crossover_warmups,
            "blocked_crossover_macroblocks": (
                args.blocked_crossover_macroblocks
            ),
            "blocked_crossover_preconditioning_macroblocks": (
                args.blocked_crossover_preconditioning_macroblocks
            ),
            "blocked_crossover_acclimation_steps": (
                args.blocked_crossover_acclimation_steps
            ),
            "blocked_crossover_measured_steps": (
                args.blocked_crossover_measured_steps
            ),
            "crossover_stationarity_tolerance_percent": (
                args.crossover_stationarity_tolerance_percent
            ),
            "require_causal_speed_gate": args.require_causal_speed_gate,
            "forward_interval_excludes": [
                "optimizer_zero_grad",
                "host_forward_route_activation",
            ],
            "warmup_updates_model": False,
            "convergence_scope": (
                "short repeated-multi-batch optimization/stability proxy"
            ),
            "parameter_counts": parameter_counts,
            "projection_weight_scaling": (
                "2d" if mx_runtime.projection_weight_scale_2d else "1d"
            ),
            "v_mxfp4_scaling": (
                "2d" if mx_runtime.v_mxfp4_scale_2d else "1d"
            ),
            "mx_qkv_projection_format": args.mx_qkv_projection_format,
            "fp8_qkv_projection_format": args.fp8_qkv_projection_format,
            "projection_extension": projection_extension,
            "forward_routes": forward_routes,
            "timed_forward_dispatch_contracts": (
                timed_forward_dispatch_contracts
            ),
            "q_quant_scale": float(mx_runtime.qk_scales[0, 0, 0]),
            "k_quant_scale": float(mx_runtime.qk_scales[0, 0, 1]),
            "backward_exp2_degree": mx_runtime.backward_exp2_degree,
            "backward_exp2_period": mx_runtime.backward_exp2_period,
            "backward_exp2_requested_degree": args.backward_exp2_degree,
            "backward_exp2_requested_period": args.backward_exp2_period,
            "backward_control_provenance": (
                mx_runtime.backward_control_provenance
            ),
            "backward_control_route_provenance": {
                "nvfp4_qk_mxfp4_pv": (
                    mx_runtime.backward_control_provenance
                ),
                "nvfp4_qk_fp8_pv_exact": (
                    fp8_runtime.backward_control_provenance
                ),
            },
            "backward_route_contracts": backward_route_contracts,
            "matched_lowp_backward_contract": True,
            "shared_lowp_backward_runner": shared_backward_runner,
            "backward_detached_fp8_p_tmem": (
                mx_runtime.backward_detached_fp8_p_tmem
            ),
            "backward_probability_tmem_policy": (
                mx_runtime.backward_probability_tmem_policy
            ),
            "backward_head_fast_raster": (
                mx_runtime.backward_head_fast_raster
            ),
            "backward_raster_policy": mx_runtime.backward_raster_policy,
            "backward_exp2_policies": {
                "nvfp4_qk_mxfp4_pv": mx_runtime.backward_exp2_policy,
                "nvfp4_qk_fp8_pv_exact": fp8_runtime.backward_exp2_policy,
            },
            "backward_detached_fp8_p_tmem_routes": {
                "nvfp4_qk_mxfp4_pv": (
                    mx_runtime.backward_detached_fp8_p_tmem
                ),
                "nvfp4_qk_fp8_pv_exact": (
                    fp8_runtime.backward_detached_fp8_p_tmem
                ),
            },
            "backward_probability_tmem_policies": {
                "nvfp4_qk_mxfp4_pv": (
                    mx_runtime.backward_probability_tmem_policy
                ),
                "nvfp4_qk_fp8_pv_exact": (
                    fp8_runtime.backward_probability_tmem_policy
                ),
            },
            "backward_head_fast_rasters": {
                "nvfp4_qk_mxfp4_pv": (
                    mx_runtime.backward_head_fast_raster
                ),
                "nvfp4_qk_fp8_pv_exact": (
                    fp8_runtime.backward_head_fast_raster
                ),
            },
            "backward_raster_policies": {
                "nvfp4_qk_mxfp4_pv": mx_runtime.backward_raster_policy,
                "nvfp4_qk_fp8_pv_exact": fp8_runtime.backward_raster_policy,
            },
            "mx_backward_reuse_quantized_p": (
                args.mx_backward_reuse_quantized_p
            ),
            "fp8_backward_reuse_quantized_p": (
                args.fp8_backward_reuse_quantized_p
            ),
            "mx_backward_match_forward_operands": (
                args.mx_backward_match_forward_operands
            ),
            "mx_per_block_qk_scales": args.mx_per_block_qk_scales,
            "mx_experimental_split_v_backward": (
                args.mx_experimental_split_v_backward
            ),
            "mx_backward_forward_probability_replay_requested": (
                args.mx_backward_forward_probability_replay
            ),
            "mx_backward_forward_probability_scale_handoff_requested": (
                args.mx_backward_forward_probability_scale_handoff
            ),
            "mx_backward_forward_probability_replay": (
                mx_runtime.backward_forward_mx_probability_replay
            ),
            "mx_backward_forward_probability_scale_handoff": (
                mx_runtime.backward_forward_mx_probability_scale_handoff
            ),
            "mx_probability_replay_provenance": (
                mx_probability_replay_provenance
            ),
            "fp8_backward_match_forward_operands": (
                args.fp8_backward_match_forward_operands
            ),
            "fp8_per_block_qk_scales": args.fp8_per_block_qk_scales,
            "mx_backward_attention_branch_gain": (
                mx_runtime.backward_probability_correction
            ),
            "mx_backward_field_gains": {
                "q": mx_runtime.backward_q_gain,
                "k": mx_runtime.backward_k_gain,
                "v": mx_runtime.backward_v_gain,
                "v_weight": mx_runtime.backward_v_weight_gain,
            },
            "fp8_backward_attention_branch_gain": (
                fp8_runtime.backward_probability_correction
            ),
            "fp8_backward_field_gains": {
                "q": fp8_runtime.backward_q_gain,
                "k": fp8_runtime.backward_k_gain,
                "v": fp8_runtime.backward_v_gain,
                "v_weight": fp8_runtime.backward_v_weight_gain,
            },
            "mx_forward_topology": mx_topology,
            "fp8_forward_topology": fp8_topology,
        },
        "initial_state_accuracy": initial_audit,
        "causal_speed_gate": causal_gate,
        "causal_forward_only_crossover": {
            **forward_crossover_summary,
            "records": forward_crossover_records,
        },
        "causal_performance_crossover": {
            **crossover_summary,
            "claim_role": "diagnostic_only",
            "records": crossover_records,
        },
        "causal_blocked_performance_crossover": {
            **blocked_crossover_summary,
            "claim_role": "primary_full_step_speed_gate",
            "records": blocked_crossover_records,
        },
        "records": records,
        "routes": summaries,
        "comparisons": comparisons,
        "memory": {
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2.0**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2.0**30,
        },
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    if args.require_causal_speed_gate:
        _require_causal_speed_gate_pass(causal_gate)


if __name__ == "__main__":
    main()
