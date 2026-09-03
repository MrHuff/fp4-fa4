from __future__ import annotations

import ast
import random
import statistics
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
HARNESS = (
    ROOT
    / "tk_fa4"
    / "lowp_fa4_bwd"
    / "benchmark_llama12b_backward_neutrality.py"
)
RUNTIME = ROOT / "tk_fa4" / "lowp_fa4_bwd" / "benchmark_llama12b_e2e.py"


def _execute_functions(
    names: tuple[str, ...],
    namespace: dict[str, object] | None = None,
) -> dict[str, object]:
    tree = ast.parse(HARNESS.read_text())
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if len(selected) != len(names):
        found = {node.name for node in selected}
        raise AssertionError(f"missing functions: {set(names) - found}")
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            *selected,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    values: dict[str, object] = {
        "Any": Any,
        "FP8_ROUTE": "fp8",
        "MX_ROUTE": "mx",
        "ROUTES": ("mx", "fp8"),
        "math": __import__("math"),
        "random": random,
        "statistics": statistics,
    }
    if namespace is not None:
        values.update(namespace)
    exec(compile(module, str(HARNESS), "exec"), values)
    return values


def test_balanced_order_has_four_adjacent_mx_fp8_pairs() -> None:
    namespace = _execute_functions(("_balanced_abba_order",))
    order = namespace["_balanced_abba_order"]
    primary = order(0)
    complement = order(1)
    assert primary == (
        "mx",
        "fp8",
        "fp8",
        "mx",
        "fp8",
        "mx",
        "mx",
        "fp8",
    )
    assert complement == tuple(
        "fp8" if route == "mx" else "mx" for route in primary
    )
    assert order(2) == complement
    assert order(3) == primary
    for candidate in (primary, complement):
        assert candidate.count("mx") == candidate.count("fp8") == 4
        assert all(
            set(candidate[index : index + 2]) == {"mx", "fp8"}
            for index in range(0, 8, 2)
        )
    with pytest.raises(ValueError, match="nonnegative"):
        order(-1)


def test_adjacent_deltas_are_oriented_mx_minus_fp8() -> None:
    namespace = _execute_functions(
        ("_balanced_abba_order", "_adjacent_pair_deltas")
    )
    order = namespace["_balanced_abba_order"]
    records = []
    for superblock in range(2):
        for position, route in enumerate(order(superblock)):
            records.append(
                {
                    "route": route,
                    "backward_ms": 11.25 if route == "mx" else 10.0,
                    "superblock": superblock,
                    "position": position,
                    "global_call_index": len(records),
                }
            )
    deltas = namespace["_adjacent_pair_deltas"](
        records,
        "backward_ms",
    )
    assert deltas == [1.25] * 8

    records[0]["route"] = "fp8"
    with pytest.raises(RuntimeError, match="order mismatch"):
        namespace["_adjacent_pair_deltas"](records, "backward_ms")

    records[0]["route"] = order(0)[0]
    records[0]["global_call_index"] = 99
    with pytest.raises(RuntimeError, match="metadata mismatch"):
        namespace["_adjacent_pair_deltas"](records, "backward_ms")


def test_superblock_cluster_means_preserve_shared_block_shocks() -> None:
    namespace = _execute_functions(
        (
            "_balanced_abba_order",
            "_adjacent_pair_deltas",
            "_superblock_mean_deltas",
        )
    )
    order = namespace["_balanced_abba_order"]
    records = []
    expected_shocks = [-1.0, 1.0, -0.5, 0.5, -0.25, 0.25, -0.125, 0.125]
    for superblock, shock in enumerate(expected_shocks):
        for position, route in enumerate(order(superblock)):
            records.append(
                {
                    "route": route,
                    "backward_ms": 10.0 + (shock if route == "mx" else 0.0),
                    "superblock": superblock,
                    "position": position,
                    "global_call_index": len(records),
                }
            )
    means = namespace["_superblock_mean_deltas"](
        records,
        "backward_ms",
    )
    assert means == pytest.approx(expected_shocks)
    assert len(means) == 8


def test_superblock_relative_effect_is_symmetric_and_clustered() -> None:
    namespace = _execute_functions(
        (
            "_balanced_abba_order",
            "_adjacent_pair_deltas",
            "_superblock_symmetric_relative_effects",
        )
    )
    order = namespace["_balanced_abba_order"]
    records = []
    for superblock in range(24):
        for position, route in enumerate(order(superblock)):
            records.append(
                {
                    "route": route,
                    "backward_ms": 11.25 if route == "mx" else 10.0,
                    "superblock": superblock,
                    "position": position,
                    "global_call_index": len(records),
                }
            )
    effects = namespace["_superblock_symmetric_relative_effects"](
        records,
        "backward_ms",
    )
    assert effects == pytest.approx([2.5 / 21.25] * 24)


def test_paired_bootstrap_is_deterministic_and_preserves_constant_delta() -> None:
    namespace = _execute_functions(
        ("_percentile", "_paired_bootstrap_interval")
    )
    bootstrap = namespace["_paired_bootstrap_interval"]
    first = bootstrap([0.125] * 16, draws=1_000, seed=17)
    second = bootstrap([0.125] * 16, draws=1_000, seed=17)
    assert first == second == pytest.approx((0.125, 0.125))
    with pytest.raises(ValueError, match="at least 1000"):
        bootstrap([0.0, 0.0], draws=999, seed=17)


def test_harness_constructs_one_shared_physical_backward() -> None:
    source = HARNESS.read_text()
    make_runtime = source.split("def _make_runtime(", 1)[1].split(
        "def _bind_runtime(", 1
    )[0]
    main = source.split("def main()", 1)[1]
    assert "shared_backward_runtime: LowpAttentionRuntime | None = None" in (
        make_runtime
    )
    assert "shared_backward_runtime=shared_backward_runtime" in make_runtime
    assert "shared_backward_runtime=mx_runtime" in main
    assert 'dispatch["qkv_projection"]' in main
    assert 'workspace["supports_both_retained_routes"] is not True' in main
    assert "workspace_owner_pointer_maps[MX_ROUTE]" in main
    assert "workspace_owner_pointer_maps[FP8_ROUTE]" in main
    assert "same model-owned " in main
    assert "forward workspace allocation map" in main
    assert main.count("require_shared_backward_physical_identity(") == 2
    assert "require_matching_backward_contracts(contracts)" in main
    assert "require_matching_backward_contracts(contracts_after)" in main
    assert "contracts_after != contracts" in main
    assert '"common_backward_contract_sha256"' in main
    runtime_source = RUNTIME.read_text()
    assert "ctx.runtime = runtime.backward_execution_runtime" in runtime_source
    assert "shared_backward_runtime.backward_execution_runtime" in runtime_source
    native_gate = runtime_source.split(
        "def _require_experimental_native_batched_runtime_contract(", 1
    )[1].split("def _require_fused_attention_rmsnorm_nvfp4(", 1)[0]
    assert 'violations.append("one route-owned backward runtime")' not in (
        native_gate
    )


def test_timed_path_is_fixed_state_and_excludes_route_binding() -> None:
    tree = ast.parse(HARNESS.read_text())
    fixed_step = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_fixed_state_step"
    )
    body = ast.unparse(fixed_step)
    assert "model.zero_grad(set_to_none=True)" in body
    assert body.index("model.zero_grad(set_to_none=True)") < body.index(
        "events[0].record()"
    )
    assert "_bind_runtime" not in body
    assert not any(
        isinstance(node, ast.Name) and node.id == "optimizer"
        for node in ast.walk(fixed_step)
    )
    assert "loss.backward()" in body

    source = HARNESS.read_text()
    assert "parameter_versions_after != parameter_versions_before" in source
    assert '"optimizer_constructed": False' in source
    assert '"optimizer_updates": 0' in source


def test_harness_authenticates_all_binaries_and_refuses_overwrite() -> None:
    source = HARNESS.read_text()
    assert source.count("_file_identity(") >= 4
    assert "TK_FA4_LOWP_BWD_EXTENSION_SOURCE" in source
    assert "_loaded_projection_identity(" in source
    assert "artifacts_after != artifacts" in source
    assert "source_files_after != source_files_before" in source
    assert '"saturated_helpers": _source_identity(' in source
    assert '"interface": _source_identity(' in source
    assert "loaded_python_before = _loaded_python_module_identities()" in source
    assert "_require_loaded_python_matches_sources(" in source
    assert '"loaded_python_modules": loaded_python_before' in source
    assert '"periodic_superblock_checks": periodic_gpu_exclusivity' in source
    assert "periodic_gpu_exclusivity.append(" in source
    assert '"artifact_identities_unchanged_across_timing": True' in source
    assert '"source_identities_unchanged_across_timing": True' in source
    assert "if os.path.lexists(args.output):" in source
    assert "os.O_EXCL" in source
    assert "output_file.flush()" in source
    assert "os.fsync(output_file.fileno())" in source
    assert "raise FileExistsError" in source
    assert "torch.cuda.device_count() != 1" in source
    assert 'hardware_before["compute_capability"] != [10, 0]' in source
    assert "_require_exclusive_visible_gpu()" in source
    assert "_require_hbm_budget(" in source
    assert 'raise SystemExit(2)' in source


def test_loaded_python_module_identity_rejects_shadow_path() -> None:
    namespace = _execute_functions(
        ("_loaded_python_module_identity",),
        {
            "Path": Path,
            "_source_identity": lambda path: {"path": str(path)},
        },
    )
    identify = namespace["_loaded_python_module_identity"]
    shadow = type(
        "ShadowModule",
        (),
        {"__file__": str(HARNESS.with_name("shadow_runtime.py"))},
    )()
    with pytest.raises(RuntimeError, match="is shadowed"):
        identify("shadow.runtime", shadow, RUNTIME)


def test_gpu_process_parser_fails_closed_on_unknown_report() -> None:
    namespace = _execute_functions(
        ("_parse_gpu_process_report",),
        {"re": __import__("re")},
    )
    parse = namespace["_parse_gpu_process_report"]
    assert parse("GPU:3\nno processes are running") == []
    assert parse(
        "GPU:3\nprocess    123 uses   456.000 MB GPU memory"
    ) == [123]
    with pytest.raises(RuntimeError, match="malformed GPU process report"):
        parse("NVML unavailable")
    with pytest.raises(RuntimeError, match="malformed GPU process report"):
        parse("GPU:3\nunrecognized process line")


def test_workspace_owner_pointer_map_is_route_independent_and_strict() -> None:
    namespace = _execute_functions(("_workspace_owner_pointer_map",))
    pointer_map = namespace["_workspace_owner_pointer_map"]
    contract = {
        "layers": [
            {
                "layer": 0,
                "active_route": "mx",
                "owners": {
                    "q": {
                        "data_ptr": 100,
                        "allocation_data_ptr": 100,
                        "bytes": 16,
                        "shape": [2, 8],
                        "dtype": "torch.uint8",
                    }
                },
            }
        ]
    }
    expected = [
        {
            "q": {
                "data_ptr": 100,
                "allocation_data_ptr": 100,
                "bytes": 16,
                "shape": [2, 8],
                "dtype": "torch.uint8",
            }
        }
    ]
    assert pointer_map(contract) == expected
    contract["layers"][0]["active_route"] = "fp8"
    assert pointer_map(contract) == expected
    contract["layers"][0]["layer"] = 1
    with pytest.raises(RuntimeError, match="layer ordering"):
        pointer_map(contract)


def test_gate_clusters_bootstrap_and_checks_fixed_state_loss() -> None:
    source = HARNESS.read_text()
    assert "MINIMUM_SUPERBLOCKS = 24" in source
    assert 'parser.add_argument("--superblocks", type=int, default=24)' in source
    assert "DEFAULT_RELATIVE_TOLERANCE = 0.01" in source
    assert "superblock_means = _superblock_mean_deltas(records, field)" in source
    assert "_paired_bootstrap_interval(\n            superblock_means," in source
    assert "_superblock_symmetric_relative_effects(" in source
    assert "relative_equivalence_passed" in source
    assert '"bootstrap_unit": "eight-call_abba_baab_superblock_mean"' in source
    assert '"fixed_state_loss_invariance": loss_invariance' in source
    assert "spread <= LOSS_INVARIANCE_ATOL" in source
    assert "contracts_after != contracts" in source


def test_serialized_backward_contract_includes_fused_rmsnorm_policy() -> None:
    source = RUNTIME.read_text()
    contract = source.split("def backward_contract(self)", 1)[1].split(
        "def bind_backward_inputs(", 1
    )[0]
    assert '"autograd": {' in contract
    assert (
        '"experimental_fused_attention_rmsnorm_nvfp4": (' in contract
    )
    assert "self.experimental_fused_attention_rmsnorm_nvfp4" in contract
