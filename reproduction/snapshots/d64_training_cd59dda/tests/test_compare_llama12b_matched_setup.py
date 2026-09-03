from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tk_fa4.lowp_fa4_bwd.backward_contract import (
    require_matching_backward_contracts,
)


ROOT = Path(__file__).resolve().parents[1]
COMPARE = ROOT / "tk_fa4" / "lowp_fa4_bwd" / "compare_llama12b_mx_fp8pv.py"
E4M3_BASE = (
    "project_qkv_gqa_d64_paired_unified_fp8_e4m3_rope_packed_"
    "interleaved_causal"
)
D128_NVFP4 = (
    "project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered"
)


def _function(name: str) -> ast.FunctionDef:
    module = ast.parse(COMPARE.read_text())
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing {name}")


def _execute_functions(
    names: tuple[str, ...],
    namespace: dict[str, Any],
) -> dict[str, Any]:
    module = ast.Module(
        body=[_function(name) for name in names],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(COMPARE), "exec"), namespace)
    return namespace


def _argument_call(main: ast.FunctionDef, option: str) -> ast.Call:
    for node in ast.walk(main):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument":
            continue
        rendered = ast.unparse(node.args[0])
        if rendered == option or rendered == repr(option):
            return node
    raise AssertionError(f"missing parser option {option}")


def _keyword_constant(call: ast.Call, name: str) -> object:
    for keyword in call.keywords:
        if keyword.arg == name and isinstance(keyword.value, ast.Constant):
            return keyword.value.value
    raise AssertionError(f"missing constant keyword {name}")


def test_production_matched_cli_defaults_are_explicit() -> None:
    main = _function("main")
    expected = {
        "--crossover-warmups": 160,
        "--blocked-crossover-macroblocks": 80,
        "--blocked-crossover-preconditioning-macroblocks": 16,
        "--blocked-crossover-acclimation-steps": 4,
        "--blocked-crossover-measured-steps": 4,
        "--mx-backward-forward-probability-replay": False,
        "--mx-backward-forward-probability-scale-handoff": False,
        "--mx-backward-probability-correction": 1.0,
        "--fp8-backward-probability-correction": 1.0,
        "--mx-experimental-split-v-backward": True,
        "--mx-qkv-projection-format": "e4m3",
        "--fp8-qkv-projection-format": "e4m3",
        "f'--{route}-backward-match-forward-operands'": True,
        "f'--{route}-per-block-qk-scales'": True,
        "f'--{route}-backward-{field}-gain'": 1.0,
        "f'--{route}-backward-v-weight-gain'": 1.0,
    }
    for option, default in expected.items():
        assert _keyword_constant(_argument_call(main, option), "default") == default


def test_claim_grade_cpu_affinity_is_sorted_and_singleton() -> None:
    observed_pids: list[int] = []
    fake_os = SimpleNamespace(
        sched_getaffinity=lambda pid: observed_pids.append(pid) or {11, 3, 7}
    )
    namespace = _execute_functions(
        ("_process_cpu_affinity", "_require_singleton_cpu_affinity"),
        {"os": fake_os},
    )
    affinity = namespace["_process_cpu_affinity"]()
    assert observed_pids == [0]
    assert affinity == [3, 7, 11]
    namespace["_require_singleton_cpu_affinity"](
        affinity,
        required=False,
    )
    namespace["_require_singleton_cpu_affinity"]([7], required=True)
    with pytest.raises(RuntimeError, match="requires singleton process CPU"):
        namespace["_require_singleton_cpu_affinity"](
            affinity,
            required=True,
        )


def test_required_gate_checks_affinity_before_cuda_or_model_allocation() -> None:
    body = ast.unparse(_function("main"))
    read_affinity = body.index(
        "process_cpu_affinity = _process_cpu_affinity()"
    )
    require_affinity = body.index(
        "_require_singleton_cpu_affinity(",
        read_affinity,
    )
    inspect_cuda = body.index("torch.cuda.device_count()")
    select_cuda = body.index("torch.cuda.set_device(0)")
    configure_model = body.index("config = config_from_model_preset(")
    allocate_model = body.index("bf16_model = Llama12B(")
    assert read_affinity < require_affinity < configure_model < inspect_cuda
    assert inspect_cuda < select_cuda < allocate_model
    source = COMPARE.read_text()
    assert '"process_cpu_affinity": process_cpu_affinity' in source
    assert '"process_cpu_affinity_singleton": (' in source


def test_full_8b_three_replica_preflight_fails_before_cuda_allocation() -> None:
    namespace = _execute_functions(
        ("_require_memory_safe_matched_replicas",),
        {"Config": object},
    )
    require_safe = namespace["_require_memory_safe_matched_replicas"]
    full = SimpleNamespace(
        model_preset="llama3.1-8b",
        layers=32,
        full_model_layers=32,
        parameter_count=8_030_261_248,
    )
    routes = (
        "bf16_cute",
        "nvfp4_qk_mxfp4_pv",
        "nvfp4_qk_fp8_pv_exact",
    )
    with pytest.raises(ValueError, match=r"179\.5 GiB.*--layers 8"):
        require_safe(full, routes, operation="comparator")

    require_safe(
        SimpleNamespace(**{**full.__dict__, "layers": 8}),
        routes,
        operation="comparator",
    )
    require_safe(full, routes[:2], operation="real-token trainer")

    body = ast.unparse(_function("main"))
    preflight = body.index("_require_memory_safe_matched_replicas(")
    inspect_cuda = body.index("torch.cuda.device_count()")
    allocate_model = body.index("bf16_model = Llama12B(")
    assert preflight < inspect_cuda < allocate_model


def test_comparator_reports_the_effective_d64_and_d128_rope() -> None:
    namespace = _execute_functions(
        ("_effective_attention_provenance",),
        {"Any": Any, "Config": object},
    )
    common = {
        "q_heads": 32,
        "kv_heads": 8,
        "rope_theta": 500_000.0,
        "rope_factor": 32.0,
        "rope_low_frequency_factor": 1.0,
        "rope_high_frequency_factor": 4.0,
        "rope_original_context": 8192,
    }
    d64 = namespace["_effective_attention_provenance"](
        SimpleNamespace(**common, head_dim=64)
    )
    assert d64["model_preset_scope"] == (
        "architecture_shape_only_legacy_rope"
    )
    assert d64["rope_theta"] == 10_000.0
    assert d64["rope_factor"] is None
    assert d64["effective_attention"]["rope"] == {
        "builder": "_make_legacy_rope",
        "theta": 10_000.0,
        "scaling": None,
    }

    d128 = namespace["_effective_attention_provenance"](
        SimpleNamespace(**{**common, "rope_factor": 8.0}, head_dim=128)
    )
    assert d128["model_preset_scope"] == "architecture_and_rope"
    assert d128["rope_theta"] == 500_000.0
    assert d128["rope_factor"] == 8.0
    assert d128["effective_attention"]["rope"]["scaling"]["factor"] == 8.0

    source = COMPARE.read_text()
    configuration = source.split('"configuration": {', 1)[1]
    assert configuration.index("**config.__dict__") < configuration.index(
        "**effective_attention_provenance"
    )


def test_causal_gate_classifies_invalid_and_valid_failures() -> None:
    namespace = _execute_functions(
        (
            "_classify_causal_speed_gate",
            "_require_causal_speed_gate_pass",
        ),
        {"Any": Any, "json": json},
    )
    classify = namespace["_classify_causal_speed_gate"]
    require_pass = namespace["_require_causal_speed_gate_pass"]
    passing = {
        "mx_forward_faster": True,
        "mx_step_faster": True,
        "backward_equal_within_tolerance": True,
        "forward_sufficient_macroblocks": True,
        "full_step_sufficient_macroblocks": True,
        "forward_stationary_within_tolerance": True,
        "full_step_stationary_within_tolerance": True,
    }
    passed = classify(passing)
    assert set(passing).issubset(passed)
    assert passed["measurement_valid"] is True
    assert passed["conclusion"] == "pass"
    assert passed["passed"] is True
    require_pass(passed)

    valid_failure = classify({**passing, "mx_step_faster": False})
    assert valid_failure["measurement_valid"] is True
    assert valid_failure["conclusion"] == "valid_performance_fail"
    assert valid_failure["passed"] is False
    with pytest.raises(RuntimeError, match="valid_performance_fail"):
        require_pass(valid_failure)

    inconclusive = classify(
        {
            **passing,
            "mx_step_faster": False,
            "forward_stationary_within_tolerance": False,
        }
    )
    assert inconclusive["measurement_valid"] is False
    assert inconclusive["conclusion"] == "inconclusive_nonstationary"
    assert inconclusive["passed"] is False
    with pytest.raises(RuntimeError, match="inconclusive_nonstationary"):
        require_pass(inconclusive)


def test_runtime_calls_publish_matched_per_block_qk_and_split_only_mx() -> None:
    main = _function("main")
    calls = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_make_runtime"
    ]
    assert len(calls) == 2
    keyword_sources = [
        {keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords}
        for call in calls
    ]
    assert keyword_sources[0]["per_block_qk_scales"] == (
        "args.mx_per_block_qk_scales"
    )
    assert keyword_sources[0]["experimental_split_v_backward"] == (
        "args.mx_experimental_split_v_backward"
    )
    assert keyword_sources[1]["per_block_qk_scales"] == (
        "args.fp8_per_block_qk_scales"
    )
    assert "experimental_split_v_backward" not in keyword_sources[1]


def test_required_projection_symbols_cover_both_production_entrypoints() -> None:
    namespace = _execute_functions(
        ("_required_projection_symbols",),
        {
            "argparse": argparse,
            "E4M3_PAIRED_PROJECTION_SYMBOL": E4M3_BASE,
        },
    )
    args = SimpleNamespace(
        mx_qkv_projection_format="e4m3",
        fp8_qkv_projection_format="e4m3",
        mx_backward_match_forward_operands=True,
        fp8_backward_match_forward_operands=True,
        mx_per_block_qk_scales=True,
        fp8_per_block_qk_scales=True,
        mx_experimental_split_v_backward=True,
    )
    mx_legacy = (
        E4M3_BASE
        + "_represented_backward_perblock_qk_split_v_backward"
    )
    fp8_legacy = E4M3_BASE + "_represented_backward_perblock_qk"
    assert namespace["_required_projection_symbols"](args) == (
        mx_legacy,
        mx_legacy + "_mx_forward_out",
        mx_legacy + "_mx_forward_out_unchecked",
        fp8_legacy,
        fp8_legacy + "_fp8_forward_out",
        fp8_legacy + "_fp8_forward_out_unchecked",
    )
    assert namespace["_required_projection_symbols"](args, ("fp8",)) == (
        fp8_legacy,
        fp8_legacy + "_fp8_forward_out",
        fp8_legacy + "_fp8_forward_out_unchecked",
    )


def test_d128_projection_symbols_authenticate_the_native_publisher() -> None:
    namespace = _execute_functions(
        ("_required_projection_symbols",),
        {
            "argparse": argparse,
            "E4M3_PAIRED_PROJECTION_SYMBOL": E4M3_BASE,
            "D128_NVFP4_PROJECTION_SYMBOL": D128_NVFP4,
        },
    )
    args = SimpleNamespace(
        mx_qkv_projection_format="nvfp4",
        fp8_qkv_projection_format="nvfp4",
    )
    assert namespace["_required_projection_symbols"](
        args,
        head_dim=128,
    ) == (D128_NVFP4,)
    args.fp8_qkv_projection_format = "e4m3"
    with pytest.raises(ValueError, match="native NVFP4 publisher"):
        namespace["_required_projection_symbols"](
            args,
            head_dim=128,
        )


def test_d128_preset_resolves_without_changing_d64_defaults() -> None:
    namespace = _execute_functions(
        ("_argument_was_provided", "_resolve_model_preset_options"),
        {"argparse": argparse, "Config": object},
    )

    def route_args() -> SimpleNamespace:
        return SimpleNamespace(
            backward_exp2_degree=2,
            backward_exp2_period=None,
            mx_backward_reuse_quantized_p=False,
            fp8_backward_reuse_quantized_p=False,
            mx_qkv_projection_format="e4m3",
            fp8_qkv_projection_format="e4m3",
            mx_backward_match_forward_operands=True,
            fp8_backward_match_forward_operands=True,
            mx_per_block_qk_scales=True,
            fp8_per_block_qk_scales=True,
            mx_experimental_split_v_backward=True,
            mx_backward_forward_probability_replay=False,
            mx_backward_forward_probability_scale_handoff=False,
            v_mxfp4_scaling="1d",
        )

    d64 = route_args()
    namespace["_resolve_model_preset_options"](
        d64,
        SimpleNamespace(head_dim=64),
        [],
    )
    assert d64.mx_qkv_projection_format == "e4m3"
    assert d64.mx_backward_match_forward_operands is True

    d128 = route_args()
    namespace["_resolve_model_preset_options"](
        d128,
        SimpleNamespace(head_dim=128),
        [],
    )
    assert d128.backward_exp2_degree == 1
    assert d128.backward_exp2_period == 0
    assert d128.mx_backward_reuse_quantized_p is True
    assert d128.fp8_backward_reuse_quantized_p is True
    assert d128.mx_qkv_projection_format == "nvfp4"
    assert d128.fp8_qkv_projection_format == "nvfp4"
    assert d128.mx_backward_match_forward_operands is False
    assert d128.fp8_backward_match_forward_operands is False
    assert d128.mx_per_block_qk_scales is False
    assert d128.fp8_per_block_qk_scales is False
    assert d128.mx_experimental_split_v_backward is False

    stale = route_args()
    with pytest.raises(ValueError, match="incompatible with the D128"):
        namespace["_resolve_model_preset_options"](
            stale,
            SimpleNamespace(head_dim=128),
            ["--mx-qkv-projection-format=e4m3"],
        )


def test_forward_route_slots_reject_swapped_or_duplicate_artifacts() -> None:
    namespace = _execute_functions(
        ("_require_forward_route_slot",),
        {"Any": Any},
    )
    require_slot = namespace["_require_forward_route_slot"]
    require_slot(
        "mx",
        {
            "fixed_route_fastpath": True,
            "route_env_guard_per_launch": False,
            "kernel_attribute_init": (
                "once_per_host_thread_and_cuda_device"
            ),
            "tma_descriptor_cache": (
                "bounded_thread_local_gl_descriptors"
            ),
            "tma_descriptor_cache_capacity": 256,
            "tma_descriptor_cache_lookup": (
                "splitmix64_device_pointer_four_way_set_associative"
            ),
            "tma_descriptor_cache_set_hash": (
                "splitmix64_device_pointer_v1"
            ),
            "tma_descriptor_cache_sets": 64,
            "tma_descriptor_cache_ways": 4,
            "tma_descriptor_cache_capacity_scope": (
                "per_compile_time_gl_slot"
            ),
            "tma_descriptor_cache_gl_slots": 10,
            "tma_descriptor_cache_total_entry_ceiling": 2560,
            "tma_descriptor_cache_key": (
                "cuda_device_data_ptr_and_compile_time_gl_slot"
            ),
            "tma_descriptor_cache_owns_tensors": False,
            "tma_descriptor_cache_counter_scope": "calling_host_thread",
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
        },
    )
    require_slot(
        "fp8",
        {
            "fixed_route_fastpath": True,
            "route_env_guard_per_launch": False,
            "kernel_attribute_init": (
                "once_per_host_thread_and_cuda_device"
            ),
            "tma_descriptor_cache": (
                "bounded_thread_local_gl_descriptors"
            ),
            "tma_descriptor_cache_capacity": 256,
            "tma_descriptor_cache_lookup": (
                "splitmix64_device_pointer_four_way_set_associative"
            ),
            "tma_descriptor_cache_set_hash": (
                "splitmix64_device_pointer_v1"
            ),
            "tma_descriptor_cache_sets": 64,
            "tma_descriptor_cache_ways": 4,
            "tma_descriptor_cache_capacity_scope": (
                "per_compile_time_gl_slot"
            ),
            "tma_descriptor_cache_gl_slots": 9,
            "tma_descriptor_cache_total_entry_ceiling": 2304,
            "tma_descriptor_cache_key": (
                "cuda_device_data_ptr_and_compile_time_gl_slot"
            ),
            "tma_descriptor_cache_owns_tensors": False,
            "tma_descriptor_cache_counter_scope": "calling_host_thread",
            "pv_format": "e4m3_fp8",
            "shiftless_fp8_mode": 0,
            "fixed_p_ceiling": False,
            "score_pack_ceiling": False,
        },
    )
    with pytest.raises(ValueError, match="MX route slot"):
        require_slot(
            "mx",
            {"pv_format": "e4m3_fp8", "shiftless_fp8_mode": 0},
        )
    with pytest.raises(ValueError, match="FP8 route slot"):
        require_slot(
            "fp8",
            {
                "pv_format": "mxfp4_e8m0_block32",
                "causal_interleaved_kv": True,
            },
        )


def test_d128_forward_route_slot_requires_the_retained_noninterleaved_policy() -> None:
    namespace = _execute_functions(
        ("_require_forward_route_slot",),
        {"Any": Any},
    )
    require_slot = namespace["_require_forward_route_slot"]
    common = {
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
    mx = {
        **common,
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
    fp8 = {
        **common,
        "tma_descriptor_cache_gl_slots": 9,
        "tma_descriptor_cache_total_entry_ceiling": 2304,
        "pv_format": "e4m3_fp8",
        "shiftless_fp8_mode": 0,
        "fixed_p_ceiling": False,
        "score_pack_ceiling": False,
    }
    require_slot("mx", mx, head_dim=128)
    require_slot("fp8", fp8, head_dim=128)
    with pytest.raises(ValueError, match="causal_interleaved_kv=False"):
        require_slot(
            "mx",
            {**mx, "causal_interleaved_kv": True},
            head_dim=128,
        )


def test_descriptor_cache_telemetry_snapshots_and_deltas_are_explicit() -> None:
    counter_names = ("hits", "misses", "evictions", "clears")
    namespace = _execute_functions(
        (
            "_descriptor_cache_counter_snapshots",
            "_descriptor_cache_counter_interval",
        ),
        {
            "Any": Any,
            "LowpAttentionRuntime": Any,
            "LOWP_ROUTE_NAMES": ("mx", "fp8"),
            "DESCRIPTOR_CACHE_COUNTER_NAMES": counter_names,
        },
    )

    def runtime(counters: tuple[int, int, int, int]) -> SimpleNamespace:
        topology = {
            "tma_descriptor_cache_counter_scope": "calling_host_thread",
            **{
                f"tma_descriptor_cache_{name}": value
                for name, value in zip(
                    counter_names,
                    counters,
                    strict=True,
                )
            },
        }
        return SimpleNamespace(
            forward_extension=SimpleNamespace(
                read_hao_direct_topology=lambda: dict(topology)
            )
        )

    snapshot = namespace["_descriptor_cache_counter_snapshots"](
        {
            "mx": runtime((10, 4, 1, 0)),
            "fp8": runtime((20, 5, 2, 0)),
        }
    )
    assert snapshot == {
        "mx": {"hits": 10, "misses": 4, "evictions": 1, "clears": 0},
        "fp8": {"hits": 20, "misses": 5, "evictions": 2, "clears": 0},
    }

    after = {
        "mx": {"hits": 19, "misses": 5, "evictions": 1, "clears": 0},
        "fp8": {"hits": 26, "misses": 7, "evictions": 3, "clears": 0},
    }
    interval = namespace["_descriptor_cache_counter_interval"](
        snapshot,
        after,
    )
    assert interval["schema"] == (
        "tma_descriptor_cache_counter_interval_v1"
    )
    assert interval["counter_scope"] == "calling_host_thread"
    assert interval["routes"]["mx"] == {
        "before": snapshot["mx"],
        "after": after["mx"],
        "delta": {"hits": 9, "misses": 1, "evictions": 0, "clears": 0},
        "descriptor_lookups": 10,
        "hit_rate": 0.9,
        "miss_rate": 0.1,
        "no_clears_during_interval": True,
        "counters_monotonic": True,
    }
    assert interval["routes"]["fp8"]["descriptor_lookups"] == 8
    assert interval["routes"]["fp8"]["delta"]["evictions"] == 1

    regressed = {name: dict(values) for name, values in after.items()}
    regressed["mx"]["hits"] = 9
    with pytest.raises(RuntimeError, match="hits counter decreased"):
        namespace["_descriptor_cache_counter_interval"](
            snapshot,
            regressed,
        )


def test_descriptor_cache_snapshot_rejects_invalid_tls_provenance() -> None:
    namespace = _execute_functions(
        ("_descriptor_cache_counter_snapshots",),
        {
            "LowpAttentionRuntime": Any,
            "LOWP_ROUTE_NAMES": ("mx", "fp8"),
            "DESCRIPTOR_CACHE_COUNTER_NAMES": (
                "hits",
                "misses",
                "evictions",
                "clears",
            ),
        },
    )
    invalid_extension = SimpleNamespace(
        read_hao_direct_topology=lambda: {
            "tma_descriptor_cache_counter_scope": "process",
            "tma_descriptor_cache_hits": 0,
            "tma_descriptor_cache_misses": 0,
            "tma_descriptor_cache_evictions": 0,
            "tma_descriptor_cache_clears": 0,
        }
    )
    runtimes = {
        name: SimpleNamespace(forward_extension=invalid_extension)
        for name in ("mx", "fp8")
    }
    with pytest.raises(RuntimeError, match="calling_host_thread scope"):
        namespace["_descriptor_cache_counter_snapshots"](runtimes)


def test_projection_extension_is_hashed_and_missing_symbol_fails_preflight(
    tmp_path: Path,
) -> None:
    extension_path = tmp_path / "projection.so"
    extension_path.write_bytes(b"projection-extension-test")
    required = (E4M3_BASE + "_represented_backward_perblock_qk",)
    extension = SimpleNamespace(
        __file__=str(extension_path),
        __name__="projection_test",
        **{required[0]: object()},
    )
    namespace = _execute_functions(
        ("_artifact_identity", "_projection_extension_identity"),
        {
            "Any": Any,
            "Path": Path,
            "hashlib": hashlib,
            "tk_interface": SimpleNamespace(_C_b300_lowp_bwd=extension),
        },
    )
    identity = namespace["_projection_extension_identity"](
        required,
        extension_path,
    )
    assert identity["path"] == str(extension_path.resolve())
    assert identity["sha256"] == hashlib.sha256(
        extension_path.read_bytes()
    ).hexdigest()
    assert identity["capabilities"] == {required[0]: True}

    delattr(extension, required[0])
    with pytest.raises(RuntimeError, match="lacks required matched-route"):
        namespace["_projection_extension_identity"](required, None)


def test_compare_helper_rejects_effective_backward_contract_mismatch() -> None:
    namespace = _execute_functions(
        ("_matched_backward_contracts",),
        {
            "Any": Any,
            "LowpAttentionRuntime": object,
            "require_matching_backward_contracts": (
                require_matching_backward_contracts
            ),
        },
    )

    class Runtime:
        def __init__(self, source: str) -> None:
            self.source = source

        def backward_contract(self) -> dict[str, object]:
            return {"projection": {"v_backward_source": self.source}}

    helper = namespace["_matched_backward_contracts"]
    contracts = helper(Runtime("e4m3"), Runtime("e4m3"))
    assert set(contracts) == {
        "nvfp4_qk_mxfp4_pv",
        "nvfp4_qk_fp8_pv_exact",
    }
    with pytest.raises(RuntimeError, match="projection.v_backward_source"):
        helper(Runtime("represented_mxfp4"), Runtime("e4m3"))


def test_timed_forward_contracts_require_exact_bound_route_symbols() -> None:
    namespace = _execute_functions(
        ("_timed_forward_dispatch_contracts",),
        {
            "Any": Any,
            "LowpAttentionRuntime": object,
            "Llama12B": object,
            "math": math,
        },
    )

    class Runtime:
        def __init__(
            self,
            attention_symbol: str,
            *,
            abi_validated: bool = True,
            scale_handoff: bool = False,
        ) -> None:
            self.attention_symbol = attention_symbol
            self.abi_validated = abi_validated
            self.backward_forward_mx_probability_scale_handoff = scale_handoff
            self.pv_format = (
                "mxfp4_e8m0_block32"
                if attention_symbol.startswith("forward_hao_direct_fp4pv")
                else "e4m3_fp8"
            )
            self.config = SimpleNamespace(
                sequence=4096,
                q_heads=32,
                kv_heads=8,
                head_dim=64,
                layers=2,
            )
            self.qk_scales = SimpleNamespace(device="cuda:0")

        def forward_dispatch_contract(self) -> dict[str, object]:
            is_mx = self.pv_format == "mxfp4_e8m0_block32"
            abi_symbol = (
                E4M3_BASE
                + "_represented_backward_perblock_qk_split_v_backward"
                if is_mx
                else E4M3_BASE + "_represented_backward_perblock_qk"
            )
            checked_symbol = abi_symbol + (
                "_mx_forward_out" if is_mx else "_fp8_forward_out"
            )
            unchecked_symbol = checked_symbol + "_unchecked"
            return {
                "schema": "lowp_forward_dispatch_contract_v2",
                "pv_format": self.pv_format,
                "qkv_projection": {
                    "format": "e4m3",
                    "dispatch": "construction_bound_exact_pybind_symbol",
                    "symbol": unchecked_symbol,
                    "abi_validation_symbol": abi_symbol,
                    "checked_symbol": checked_symbol,
                    "unchecked_symbol": unchecked_symbol,
                    "shape_bound_at_construction": True,
                    "first_call_full_abi_validation_complete": (
                        self.abi_validated
                    ),
                    "subsequent_call_path": (
                        "bound_exact_pybind_symbol_with_preallocated_"
                        "forward_workspace"
                    ),
                    "preallocated_forward_workspace_required": True,
                    "preallocated_forward_publication_slots": [
                        4, 6, 8, 9, 10, 11, 12, 13, 20, 21, 22, 23
                    ],
                    "preallocated_forward_workspace_abi_validated": (
                        self.abi_validated
                    ),
                    "validated_forward_workspace_count": (
                        self.config.layers if self.abi_validated else 0
                    ),
                    "timed_forward_publication_allocation_fallback": False,
                    "preallocated_forward_workspace_ownership": (
                        "private_nonpersistent_layer_route_neutral_superset"
                    ),
                    "qk_payload_typed_alias_materialization": (
                        "construction_time"
                    ),
                    "runtime_crossover_reallocation": False,
                },
                "attention": {
                    "dispatch": (
                        "construction_bound_route_specific_entrypoint"
                    ),
                    "symbol": self.attention_symbol,
                    "entrypoint_bound_at_construction": True,
                    "launcher_bound_to_runtime": True,
                },
            }

    class Model:
        def __init__(
            self,
            runtime: Runtime,
            *,
            workspace_valid: bool = True,
        ) -> None:
            self.runtime = runtime
            self.workspace_valid = workspace_valid

        def require_lowp_forward_workspace_stream(self) -> int:
            return 17

        def lowp_forward_workspace_contract(self) -> dict[str, object]:
            projection = self.runtime.forward_dispatch_contract()[
                "qkv_projection"
            ]
            config = self.runtime.config
            owner_specs = {
                "q_payload": (
                    4,
                    [1, config.q_heads, config.sequence, config.head_dim // 2],
                    "torch.uint8",
                ),
                "k_payload": (
                    6,
                    [1, config.kv_heads, config.sequence, config.head_dim // 2],
                    "torch.uint8",
                ),
                "q_scale_pages": (
                    8,
                    [1, config.sequence // 128, config.q_heads, 512],
                    "torch.float8_e4m3fn",
                ),
                "q_global_scale": (
                    9,
                    [1, config.q_heads],
                    "torch.float32",
                ),
                "k_scale_pages": (
                    10,
                    [1, config.sequence // 64, config.kv_heads, 512],
                    "torch.float8_e4m3fn",
                ),
                "k_global_scale": (
                    11,
                    [1, config.kv_heads],
                    "torch.float32",
                ),
                "v_mxfp4_payload": (
                    12,
                    [1, config.kv_heads, config.head_dim, config.sequence // 2],
                    "torch.float4_e2m1fn_x2",
                ),
                "v_mxfp4_scale_pages": (
                    13,
                    [1, config.sequence // 128, config.kv_heads, 512],
                    "torch.float8_e4m3fn",
                ),
                "v_backward_fp8": (
                    20,
                    [1, config.sequence, config.kv_heads, config.head_dim],
                    "torch.float8_e4m3fn",
                ),
                "q_backward_fp8": (
                    21,
                    [1, config.sequence, config.q_heads, config.head_dim],
                    "torch.float8_e4m3fn",
                ),
                "k_backward_fp8": (
                    22,
                    [1, config.sequence, config.kv_heads, config.head_dim],
                    "torch.float8_e4m3fn",
                ),
                "v_fp8_payload": (
                    23,
                    [1, config.kv_heads, config.head_dim, config.sequence],
                    "torch.float8_e4m3fn",
                ),
            }
            publication_slots = [
                spec[0] for spec in owner_specs.values()
            ]
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
            active_fields = common_fields | (
                {"v_mxfp4_payload", "v_mxfp4_scale_pages"}
                if self.runtime.pv_format == "mxfp4_e8m0_block32"
                else {"v_fp8_payload"}
            )
            layers = [
                {
                    "layer": index,
                    "schema": "lowp_layer_forward_workspace_v2",
                    "publication_slots": publication_slots,
                    "owners": {
                        name: {
                            "slot": slot,
                            "data_ptr": 10000 + index * 100 + slot,
                            "allocation_data_ptr": 10000 + index * 100 + slot,
                            "pointer_stable_since_allocation": True,
                            "shape": shape,
                            "dtype": dtype,
                            "device": "cuda:0",
                            "bytes": math.prod(shape)
                            * (4 if dtype == "torch.float32" else 1),
                            "listed_in_named_buffers": False,
                            "listed_in_named_parameters": False,
                            "optimizer_visible_parameter": False,
                        }
                        for name, (slot, shape, dtype) in owner_specs.items()
                    },
                    "aliases": {
                        "q_payload_fp4": {
                            "owner": "q_payload",
                            "pointer_matches_owner": True,
                        },
                        "k_payload_fp4": {
                            "owner": "k_payload",
                            "pointer_matches_owner": True,
                        },
                    },
                    "owner_count": len(owner_specs),
                    "owner_pointers_unique_within_layer": True,
                    "owner_pointers_stable_since_allocation": True,
                    "typed_aliases_match_owners": True,
                    "all_outputs_private_nonpersistent": True,
                    "supports_both_retained_routes": True,
                    "active_route": self.runtime.pv_format,
                    "active_owner_fields": sorted(active_fields),
                    "single_stream_cuda_stream": 17,
                    "bound_projection_symbol": projection["symbol"],
                    "bound_projection_checked_symbol": projection[
                        "checked_symbol"
                    ],
                    "requires_forward_workspace": True,
                    "forward_workspace_abi_validated": True,
                    "validated_forward_workspace_count": config.layers,
                }
                for index in range(config.layers)
            ]
            return {
                "schema": "lowp_model_forward_workspaces_v2",
                "layer_count": config.layers,
                "owner_count": len(owner_specs) * config.layers,
                "owner_pointers_globally_unique": self.workspace_valid,
                "owner_pointers_unique_across_layers": True,
                "owner_pointers_stable_since_allocation": True,
                "typed_aliases_match_owners": True,
                "all_outputs_private_nonpersistent": True,
                "supports_both_retained_routes": True,
                "layers": layers,
            }

    helper = namespace["_timed_forward_dispatch_contracts"]
    mx = Runtime("forward_hao_direct_fp4pv")
    fp8 = Runtime("forward_hao_direct_fp8pv")
    mx_model = Model(mx)
    fp8_model = Model(fp8)
    contracts = helper(mx, fp8, mx_model, fp8_model)
    assert all(
        contract["validated_after_compile_before_timing"] is True
        for contract in contracts.values()
    )

    with pytest.raises(RuntimeError, match="exact E4M3 QKV projection"):
        invalid_mx = Runtime(
            "forward_hao_direct_fp4pv", abi_validated=False
        )
        helper(invalid_mx, fp8, Model(invalid_mx), fp8_model)
    with pytest.raises(RuntimeError, match="route-specific attention"):
        invalid_fp8 = Runtime("forward_hao_direct_fp4pv")
        invalid_fp8.pv_format = "e4m3_fp8"
        helper(mx, invalid_fp8, mx_model, Model(invalid_fp8))
    with pytest.raises(RuntimeError, match="forward-publication workspaces"):
        helper(mx, fp8, Model(mx, workspace_valid=False), fp8_model)


def test_timed_d128_contract_authenticates_native_v_without_d64_workspace() -> None:
    namespace = _execute_functions(
        ("_timed_forward_dispatch_contracts",),
        {
            "Any": Any,
            "LowpAttentionRuntime": object,
            "Llama12B": object,
            "math": math,
            "D128_NVFP4_PROJECTION_SYMBOL": D128_NVFP4,
            "tk_interface": SimpleNamespace(
                b300_project_qkv_gqa_d128_unified_lowp_nvfp4=lambda: None
            ),
        },
    )

    class Runtime:
        def __init__(self, pv_format: str) -> None:
            self.pv_format = pv_format
            self.config = SimpleNamespace(head_dim=128)
            self.backward_forward_mx_probability_scale_handoff = False
            self.qkv_projection = None

        def forward_dispatch_contract(self) -> dict[str, object]:
            is_mx = self.pv_format == "mxfp4_e8m0_block32"
            return {
                "schema": "lowp_forward_dispatch_contract_v2",
                "pv_format": self.pv_format,
                "qkv_projection": {
                    "format": "nvfp4",
                    "dispatch": "public_api_per_invocation",
                    "symbol": None,
                    "abi_validation_symbol": None,
                    "checked_symbol": None,
                    "unchecked_symbol": None,
                    "shape_bound_at_construction": False,
                    "first_call_full_abi_validation_complete": None,
                    "subsequent_call_path": "public_api",
                    "preallocated_forward_workspace_required": False,
                    "preallocated_forward_workspace_abi_validated": None,
                    "validated_forward_workspace_count": None,
                    "timed_forward_publication_allocation_fallback": True,
                    "preallocated_forward_workspace_ownership": (
                        "allocated_publication_return_owned_by_autograd"
                    ),
                    "runtime_crossover_reallocation": False,
                },
                "attention": {
                    "dispatch": (
                        "construction_bound_route_specific_entrypoint"
                    ),
                    "symbol": (
                        "forward_hao_direct_fp4pv"
                        if is_mx
                        else "forward_hao_direct_fp8pv"
                    ),
                    "launcher": (
                        "_launch_forward_mx"
                        if is_mx
                        else "_launch_forward_fp8"
                    ),
                    "entrypoint_bound_at_construction": True,
                    "launcher_bound_to_runtime": True,
                },
            }

    class Model:
        def require_lowp_forward_workspace_stream(self) -> int:
            return 29

    helper = namespace["_timed_forward_dispatch_contracts"]
    mx = Runtime("mxfp4_e8m0_block32")
    fp8 = Runtime("e4m3_fp8")
    contracts = helper(mx, fp8, Model(), Model())
    for contract in contracts.values():
        publication = contract["d128_projection_publication"]
        assert publication["projection_output_preallocated"] is False
        assert publication["timed_projection_output_allocations"] is True
        assert publication["allocations_shared_by_mx_and_fp8_routes"] is True
        assert publication["native_feature_major_fp8_v_required"] is True
        assert (
            publication["unfused_fp8_v_layout_materialization_allowed"]
            is False
        )

    mx_only = helper(mx, None, Model(), None)
    assert set(mx_only) == {"nvfp4_qk_mxfp4_pv"}
    assert (
        mx_only["nvfp4_qk_mxfp4_pv"]["d128_projection_publication"]
        ["allocations_shared_by_mx_and_fp8_routes"]
        is False
    )
    fp8_only = helper(None, fp8, None, Model())
    assert set(fp8_only) == {"nvfp4_qk_fp8_pv_exact"}
    with pytest.raises(RuntimeError, match="runtime/model selection mismatch"):
        helper(mx, None, None, None)

    original = fp8.forward_dispatch_contract

    def invalid_contract() -> dict[str, object]:
        contract = original()
        contract["attention"]["launcher"] = "_launch_forward_mx"
        return contract

    fp8.forward_dispatch_contract = invalid_contract
    with pytest.raises(RuntimeError, match="route-specific attention"):
        helper(mx, fp8, Model(), Model())


def test_forward_dispatch_is_authenticated_between_compile_and_timing() -> None:
    main = ast.unparse(_function("main"))
    compile_loop = main.index(
        "for execution_position, name in enumerate(ROUTE_NAMES):"
    )
    dispatch_contract = main.index(
        "timed_forward_dispatch_contracts = "
        "_timed_forward_dispatch_contracts(",
        compile_loop,
    )
    first_crossover = main.index(
        "_same_model_forward_crossover(", dispatch_contract
    )
    assert compile_loop < dispatch_contract < first_crossover
    assert '"timed_forward_dispatch_contracts"' in COMPARE.read_text()


def test_matched_routes_share_physical_backward_and_common_operands() -> None:
    namespace = _execute_functions(
        ("_matched_backward_contracts", "_share_matched_backward_runner"),
        {
            "Any": Any,
            "LowpAttentionRuntime": object,
            "require_matching_backward_contracts": (
                require_matching_backward_contracts
            ),
        },
    )

    class Pointer:
        def __init__(self, value: int) -> None:
            self.value = value

        def data_ptr(self) -> int:
            return self.value

    class Runtime:
        def __init__(self, offset: int) -> None:
            self.backward = SimpleNamespace(workspace_torch=Pointer(offset))
            self.control = object()
            self.paired_rope = Pointer(offset + 1)
            self.gradient_global_scale = Pointer(offset + 2)

        def backward_contract(self) -> dict[str, object]:
            return {
                "schema": "lowp_backward_contract_v1",
                "projection": {"v_backward_source": "e4m3"},
            }

    mx = Runtime(10)
    fp8 = Runtime(20)
    provenance = namespace["_share_matched_backward_runner"](mx, fp8)
    assert all(provenance.values())
    assert fp8.backward is mx.backward
    assert fp8.control is mx.control
    assert fp8.paired_rope is mx.paired_rope
    assert fp8.gradient_global_scale is mx.gradient_global_scale


def test_comparison_activates_route_before_each_model_forward() -> None:
    source = COMPARE.read_text()
    assert source.count(
        "_activate_model_forward_route(name, model, forward_routes)"
    ) == 4
    assert "activate_bound_model_forward_route(model)" in source
    assert '"projection_extension": projection_extension' in source
    assert '"backward_route_contracts": backward_route_contracts' in source


def test_route_activation_and_zero_grad_are_outside_forward_interval() -> None:
    step = ast.unparse(_function("_step"))
    start = step.index("start.record()")
    assert step.index("optimizer.zero_grad(set_to_none=True)") < start
    assert step.index(
        "_activate_model_forward_route(name, model, forward_routes)"
    ) < start


def test_runtime_crossover_reuses_one_model_allocation() -> None:
    namespace = _execute_functions(
        ("_assign_model_lowp_runtime",),
        {
            "LowpAttentionRuntime": object,
        },
    )

    class Attention:
        runtime = "old"

    class Layer:
        def __init__(self) -> None:
            self.attention = Attention()

    model = SimpleNamespace(
        config=SimpleNamespace(layers=3),
        layers=[Layer(), Layer(), Layer()],
        lowp_attention_runtime="old",
    )

    def bind(runtime: object) -> int:
        for layer in model.layers:
            layer.attention.runtime = runtime
        model.lowp_attention_runtime = runtime
        return len(model.layers)

    model.bind_lowp_attention_runtime = bind
    runtime = object()
    switched = namespace["_assign_model_lowp_runtime"](model, runtime)
    assert switched == 3
    assert model.lowp_attention_runtime is runtime
    assert all(layer.attention.runtime is runtime for layer in model.layers)


def test_balanced_crossover_metric_uses_fp8_minus_mx() -> None:
    namespace = _execute_functions(
        (
            "_trimmed_mean",
            "_percentile",
            "_cluster_bootstrap_trimmed_mean_interval",
            "_bootstrap_common_mode_drift_interval",
            "_balanced_block_metric_summary",
        ),
        {
            "Any": Any,
            "math": math,
            "random": random,
            "statistics": statistics,
            "MIN_CAUSAL_MACROBLOCKS": 20,
        },
    )
    mx = []
    fp8 = []
    for block, mx_positions in (
        (0, (0, 3, 5, 6)),
        (1, (1, 2, 4, 7)),
    ):
        fp8_positions = tuple(set(range(8)) - set(mx_positions))
        mx.extend(
            {
                "crossover_superblock": block,
                "crossover_macroblock": block // 2,
                "complement_index": block % 2,
                "execution_position": position,
                "global_call_index": block * 8 + position,
                "forward_ms": 10.0,
            }
            for position in mx_positions
        )
        fp8.extend(
            {
                "crossover_superblock": block,
                "crossover_macroblock": block // 2,
                "complement_index": block % 2,
                "execution_position": position,
                "global_call_index": block * 8 + position,
                "forward_ms": 12.0,
            }
            for position in fp8_positions
        )
    summary = namespace["_balanced_block_metric_summary"](
        mx, fp8, "forward_ms"
    )
    assert summary["positive_means_mx_faster"] is True
    assert summary["fp8_minus_mx_block_trimmed_mean_ms"] == 2.0
    assert summary["mx_faster_blocks"] == 1
    assert summary["blocks"] == 1
    assert summary["superblocks"] == 2
    assert summary["macroblocks"] == 1
    assert summary["samples_per_route"] == 8
    assert summary["sufficient_samples"] is False


def test_balanced_crossover_rejects_fractional_or_inconsistent_metadata() -> None:
    namespace = _execute_functions(
        (
            "_trimmed_mean",
            "_percentile",
            "_cluster_bootstrap_trimmed_mean_interval",
            "_bootstrap_common_mode_drift_interval",
            "_balanced_block_metric_summary",
        ),
        {
            "Any": Any,
            "math": math,
            "random": random,
            "statistics": statistics,
            "MIN_CAUSAL_MACROBLOCKS": 20,
        },
    )

    def records() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        by_route: dict[str, list[dict[str, Any]]] = {"mx": [], "fp8": []}
        for block in range(2):
            mx_positions = (
                (0, 3, 5, 6) if block == 0 else (1, 2, 4, 7)
            )
            for position in range(8):
                route = "mx" if position in mx_positions else "fp8"
                by_route[route].append(
                    {
                        "crossover_superblock": block,
                        "crossover_macroblock": block // 2,
                        "complement_index": block % 2,
                        "execution_position": position,
                        "global_call_index": block * 8 + position,
                        "forward_ms": 10.0 if route == "mx" else 12.0,
                    }
                )
        return by_route["mx"], by_route["fp8"]

    summarize = namespace["_balanced_block_metric_summary"]
    mx, fp8 = records()
    mx[0]["execution_position"] = 0.5
    with pytest.raises(RuntimeError, match="exact integer"):
        summarize(mx, fp8, "forward_ms")

    mx, fp8 = records()
    mx[0]["global_call_index"] = 7
    with pytest.raises(RuntimeError, match="invalid global call index"):
        summarize(mx, fp8, "forward_ms")


def test_runtime_crossover_uses_continuous_zero_lr_optimizer_state() -> None:
    body = ast.unparse(_function("_same_model_runtime_crossover"))
    assert body.count("_reset_optimizer_state(optimizer)") == 2
    measured = body.split("for block_index in range(rounds // 4):", 1)[1]
    measured_loop = measured.split("finally:", 1)[0]
    assert "_reset_optimizer_state(optimizer)" not in measured_loop
    assert "record = _step(" in measured_loop
    assert "continuous_shared_zero_lr_after_one_pre_warmup_reset" in body
    assert "optimizer_state_reset_before_every_sample" in body


def test_crossover_order_balances_eight_call_superblocks() -> None:
    namespace = _execute_functions(
        ("_balanced_route_order",),
        {"LOWP_ROUTE_NAMES": ("mx", "fp8")},
    )
    order = namespace["_balanced_route_order"]
    assert order(0) == (
        "mx", "fp8", "fp8", "mx", "fp8", "mx", "mx", "fp8"
    )
    assert order(1) == (
        "fp8", "mx", "mx", "fp8", "mx", "fp8", "fp8", "mx"
    )
    assert order(2) == order(1)
    assert order(3) == order(0)
    assert order(4) == order(0)


def test_blocked_crossover_counterbalances_whole_route_blocks() -> None:
    namespace = _execute_functions(
        ("_blocked_route_order",),
        {"LOWP_ROUTE_NAMES": ("mx", "fp8")},
    )
    order = namespace["_blocked_route_order"]
    assert order(0) == ("mx", "fp8", "fp8", "mx")
    assert order(1) == ("fp8", "mx", "mx", "fp8")
    assert order(2) == order(1)
    assert order(3) == order(0)
    assert order(4) == order(0)
    for pair_index in range(4):
        first = order(2 * pair_index)
        second = order(2 * pair_index + 1)
        assert all(
            first_route != second_route
            for first_route, second_route in zip(first, second, strict=True)
        )
    with pytest.raises(ValueError, match="nonnegative"):
        order(-1)


def test_blocked_crossover_summary_preserves_twenty_full_phase_cycles() -> None:
    namespace = _execute_functions(
        (
            "_blocked_route_order",
            "_trimmed_mean",
            "_percentile",
            "_cluster_bootstrap_trimmed_mean_interval",
            "_bootstrap_common_mode_drift_interval",
            "_blocked_metric_summary",
        ),
        {
            "Any": Any,
            "LOWP_ROUTE_NAMES": ("mx", "fp8"),
            "MIN_CAUSAL_MACROBLOCKS": 20,
            "MIN_BLOCKED_PHASE_CYCLES": 20,
            "COMPLEMENT_PAIRS_PER_PHASE_CYCLE": 2,
            "MACROBLOCKS_PER_COMPLEMENT_PAIR": 2,
            "MACROBLOCKS_PER_PHASE_CYCLE": 4,
            "MIN_BLOCKED_COMPLEMENT_PAIRS": 40,
            "MIN_BLOCKED_MACROBLOCKS": 80,
            "math": math,
            "random": random,
            "statistics": statistics,
        },
    )
    records: dict[str, list[dict[str, Any]]] = {"mx": [], "fp8": []}
    measured_steps = 2
    for macroblock in range(80):
        for block_position, route in enumerate(
            namespace["_blocked_route_order"](macroblock)
        ):
            global_block = macroblock * 4 + block_position
            for step_in_block in range(measured_steps):
                records[route].append(
                    {
                        "route": route,
                        "blocked_macroblock": macroblock,
                        "blocked_block_position": block_position,
                        "blocked_step_in_block": step_in_block,
                        "global_block_index": global_block,
                        "global_call_index": (
                            global_block * measured_steps + step_in_block
                        ),
                        "execution_position": block_position,
                        "step_ms": 10.0 if route == "mx" else 12.0,
                    }
                )
    summarize = namespace["_blocked_metric_summary"]
    summary = summarize(
        records["mx"],
        records["fp8"],
        "step_ms",
        macroblocks=80,
        measured_steps_per_block=measured_steps,
    )
    assert summary["estimator"] == (
        "full_phase_cycle_abba_baab_baab_abba_trimmed_mean"
    )
    assert summary["estimator_unit"] == (
        "two_adjacent_complement_pairs_four_raw_macroblocks"
    )
    assert summary["bootstrap_cluster_unit"] == "full_phase_cycle"
    assert summary["bootstrap_preserves_pair_orientation"] is True
    assert summary["bootstrap_preserves_adjacent_pair_covariance"] is True
    assert summary["stationarity_unit"] == "full_phase_cycle_common_mode"
    assert summary["fp8_minus_mx_phase_cycle_trimmed_mean_ms"] == 2.0
    assert summary["fp8_minus_mx_block_trimmed_mean_ms"] == 2.0
    assert summary[
        "fp8_minus_mx_phase_cycle_trimmed_mean_bootstrap_ci90_lower_ms"
    ] == 2.0
    assert summary[
        "fp8_minus_mx_phase_cycle_trimmed_mean_bootstrap_ci90_upper_ms"
    ] == 2.0
    assert summary["mx_faster_blocks"] == 20
    assert summary["mx_faster_phase_cycles"] == 20
    assert summary["mx_faster_complement_pairs"] == 40
    assert summary["blocks"] == 20
    assert summary["phase_cycles"] == 20
    assert summary["complement_pairs"] == 40
    assert summary["complement_pairs_by_orientation"] == {"0": 20, "1": 20}
    assert summary["minimum_phase_cycles_for_gate"] == 20
    assert summary["minimum_complement_pairs_for_gate"] == 40
    assert summary["minimum_macroblocks_for_gate"] == 80
    assert summary["macroblocks"] == 80
    assert summary["route_blocks"] == 320
    assert summary["samples_per_route"] == 320
    assert summary["sufficient_samples"] is True
    assert summary["common_mode_quartile_range_percent"] == 0.0
    assert summary["stationarity_window_phase_cycles"] == 5
    first_cycle = summary["phase_cycle_records"][0]
    assert first_cycle["complement_pairs"] == [0, 1]
    assert first_cycle["pair_orientations"] == [0, 1]
    assert first_cycle["macroblocks"] == [0, 1, 2, 3]
    assert [
        row["pair_position_in_phase_cycle"]
        for row in first_cycle["pair_provenance"]
    ] == [0, 1]
    assert {
        (row["phase_cycle"], row["pair_position_in_phase_cycle"])
        for row in summary["matched_position_records"][:8]
    } == {(0, 0), (0, 1)}

    with pytest.raises(ValueError, match="multiple of four.*at least 80"):
        summarize(
            records["mx"],
            records["fp8"],
            "step_ms",
            macroblocks=40,
            measured_steps_per_block=measured_steps,
        )

    records["mx"][0]["blocked_step_in_block"] = 0.5
    with pytest.raises(RuntimeError, match="exact integer"):
        summarize(
            records["mx"],
            records["fp8"],
            "step_ms",
            macroblocks=80,
            measured_steps_per_block=measured_steps,
        )


def test_blocked_crossover_stratifies_repeatable_pair_phase_noise() -> None:
    namespace = _execute_functions(
        (
            "_blocked_route_order",
            "_trimmed_mean",
            "_percentile",
            "_cluster_bootstrap_trimmed_mean_interval",
            "_bootstrap_common_mode_drift_interval",
            "_blocked_metric_summary",
        ),
        {
            "Any": Any,
            "LOWP_ROUTE_NAMES": ("mx", "fp8"),
            "MIN_CAUSAL_MACROBLOCKS": 20,
            "MIN_BLOCKED_PHASE_CYCLES": 20,
            "COMPLEMENT_PAIRS_PER_PHASE_CYCLE": 2,
            "MACROBLOCKS_PER_COMPLEMENT_PAIR": 2,
            "MACROBLOCKS_PER_PHASE_CYCLE": 4,
            "MIN_BLOCKED_COMPLEMENT_PAIRS": 40,
            "MIN_BLOCKED_MACROBLOCKS": 80,
            "math": math,
            "random": random,
            "statistics": statistics,
        },
    )
    records: dict[str, list[dict[str, Any]]] = {"mx": [], "fp8": []}
    phase_penalty = (0.0, 1.0, 8.0, 27.0, 64.0, 125.0, 216.0, 343.0)
    for macroblock in range(80):
        for block_position, route in enumerate(
            namespace["_blocked_route_order"](macroblock)
        ):
            global_block = macroblock * 4 + block_position
            records[route].append(
                {
                    "route": route,
                    "blocked_macroblock": macroblock,
                    "blocked_block_position": block_position,
                    "blocked_step_in_block": 0,
                    "global_block_index": global_block,
                    "global_call_index": global_block,
                    "execution_position": block_position,
                    "step_ms": (
                        100.0
                        + phase_penalty[global_block % len(phase_penalty)]
                        + (0.25 if route == "fp8" else 0.0)
                    ),
                }
            )
    summary = namespace["_blocked_metric_summary"](
        records["mx"],
        records["fp8"],
        "step_ms",
        macroblocks=80,
        measured_steps_per_block=1,
    )
    phase_deltas = {
        orientation: {
            row["fp8_minus_mx_ms"]
            for row in summary["complement_pair_records"]
            if row["pair_orientation"] == orientation
        }
        for orientation in range(2)
    }
    assert len(phase_deltas[0]) == len(phase_deltas[1]) == 1
    assert next(iter(phase_deltas[0])) != next(iter(phase_deltas[1]))
    assert summary[
        "fp8_minus_mx_phase_cycle_trimmed_mean_ms"
    ] == pytest.approx(0.25)
    assert summary[
        "fp8_minus_mx_phase_cycle_trimmed_mean_bootstrap_ci90_lower_ms"
    ] == pytest.approx(0.25)
    assert summary[
        "fp8_minus_mx_phase_cycle_trimmed_mean_bootstrap_ci90_upper_ms"
    ] == pytest.approx(0.25)


def test_blocked_crossover_bootstraps_adjacent_pair_covariance() -> None:
    namespace = _execute_functions(
        (
            "_blocked_route_order",
            "_trimmed_mean",
            "_percentile",
            "_cluster_bootstrap_trimmed_mean_interval",
            "_bootstrap_common_mode_drift_interval",
            "_blocked_metric_summary",
        ),
        {
            "Any": Any,
            "LOWP_ROUTE_NAMES": ("mx", "fp8"),
            "MIN_CAUSAL_MACROBLOCKS": 20,
            "MIN_BLOCKED_PHASE_CYCLES": 20,
            "COMPLEMENT_PAIRS_PER_PHASE_CYCLE": 2,
            "MACROBLOCKS_PER_COMPLEMENT_PAIR": 2,
            "MACROBLOCKS_PER_PHASE_CYCLE": 4,
            "MIN_BLOCKED_COMPLEMENT_PAIRS": 40,
            "MIN_BLOCKED_MACROBLOCKS": 80,
            "math": math,
            "random": random,
            "statistics": statistics,
        },
    )
    cycle_deltas = [
        0.24,
        0.11,
        -0.06,
        0.05,
        0.08,
        0.03,
        0.17,
        0.10,
        0.12,
        -0.15,
    ] * 2
    records: dict[str, list[dict[str, Any]]] = {"mx": [], "fp8": []}
    for macroblock in range(80):
        phase_cycle = macroblock // 4
        for block_position, route in enumerate(
            namespace["_blocked_route_order"](macroblock)
        ):
            global_block = macroblock * 4 + block_position
            records[route].append(
                {
                    "route": route,
                    "blocked_macroblock": macroblock,
                    "blocked_block_position": block_position,
                    "blocked_step_in_block": 0,
                    "global_block_index": global_block,
                    "global_call_index": global_block,
                    "execution_position": block_position,
                    "step_ms": 100.0
                    + (cycle_deltas[phase_cycle] if route == "fp8" else 0.0),
                }
            )
    summary = namespace["_blocked_metric_summary"](
        records["mx"],
        records["fp8"],
        "step_ms",
        macroblocks=80,
        measured_steps_per_block=1,
    )
    observed_cycle_deltas = [
        row["fp8_minus_mx_ms"] for row in summary["phase_cycle_records"]
    ]
    assert observed_cycle_deltas == pytest.approx(cycle_deltas)
    bootstrap_seed = 10_000 + sum(ord(character) for character in "step_ms")
    bootstrap_seed += len(cycle_deltas)
    expected_ci90 = namespace["_cluster_bootstrap_trimmed_mean_interval"](
        observed_cycle_deltas,
        0.90,
        seed=bootstrap_seed + 1,
    )
    assert summary[
        "fp8_minus_mx_phase_cycle_trimmed_mean_bootstrap_ci90_lower_ms"
    ] == pytest.approx(expected_ci90[0])
    assert summary[
        "fp8_minus_mx_phase_cycle_trimmed_mean_bootstrap_ci90_upper_ms"
    ] == pytest.approx(expected_ci90[1])


def test_blocked_crossover_binds_once_then_acclimates_and_measures() -> None:
    body = ast.unparse(_function("_same_model_blocked_runtime_crossover"))
    block_loop = body.split(
        "for block_position, name in enumerate(order):", 1
    )[1].split("finally:", 1)[0]
    bind = block_loop.index(
        "_assign_model_lowp_runtime(model, runtimes[name])"
    )
    acclimate = block_loop.index(
        "for _ in range(acclimation_steps_per_block):"
    )
    measure = block_loop.index(
        "for step_in_block in range(measured_steps_per_block):"
    )
    assert bind < acclimate < measure
    assert block_loop.count(
        "_assign_model_lowp_runtime(model, runtimes[name])"
    ) == 1
    assert block_loop.count("_acclimate_block_iteration(") == 1
    assert block_loop.count("record = _step(") == 1
    assert "group['lr'] = 0.0" in body
    assert body.count("_reset_optimizer_state(optimizer)") == 2
    assert "same_model_blocked_runtime_crossover_v3" in body
    assert "sufficient_phase_cycles" in body
    assert "set_to_none_true_before_every_step" in body
    assert "for preconditioning_macroblock in range(" in body
    assert "preconditioning_records_retained" in body
    assert "preconditioning_full_iterations" in body
    assert body.count("_descriptor_cache_counter_snapshots(runtimes)") == 3
    assert "'preconditioning': _descriptor_cache_counter_interval(" in body
    assert "measured_blocked_crossover" in body
    assert "combined_preconditioning_and_measurement" in body


def test_forward_only_crossover_serializes_cache_counter_interval() -> None:
    body = ast.unparse(_function("_same_model_forward_crossover"))
    assert body.count("_descriptor_cache_counter_snapshots(runtimes)") == 3
    assert "forward_only_tma_descriptor_cache_telemetry_v1" in body
    assert "measured_forward_only" in body
    assert "combined_warmup_and_measurement" in body
    assert "_descriptor_cache_counter_interval(" in body


def test_primary_speed_claim_comes_from_same_model_crossover() -> None:
    source = COMPARE.read_text()
    crossover_call = source.index(
        ") = _same_model_blocked_runtime_crossover("
    )
    convergence_loop = source.index(
        "for round_index in range(args.rounds):",
        crossover_call,
    )
    assert crossover_call < convergence_loop
    assert '"causal_forward_only_crossover"' in source
    assert '"causal_performance_crossover"' in source
    assert '"causal_blocked_performance_crossover"' in source
    assert '"primary_full_step_speed_gate"' in source
    assert '"diagnostic_only"' in source
    assert '"same-model forward-only plus production-faithful blocked "' in source
    assert '"causal_speed_claim_valid": False' in source
    assert "causal_gate = _classify_causal_speed_gate(causal_gate)" in source
    assert "_require_causal_speed_gate_pass(causal_gate)" in source
