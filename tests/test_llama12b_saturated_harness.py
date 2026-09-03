from __future__ import annotations

import ast
import hashlib
import math
import statistics
import string
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
HARNESS = (
    ROOT
    / "tk_fa4"
    / "lowp_fa4_bwd"
    / "benchmark_llama12b_saturated.py"
)
RUNTIME = ROOT / "tk_fa4" / "lowp_fa4_bwd" / "benchmark_llama12b_e2e.py"


def _namespace(*names: str) -> dict[str, Any]:
    tree = ast.parse(HARNESS.read_text())
    runtime_tree = ast.parse(RUNTIME.read_text())
    runtime_dependencies = {
        "D128_EXACT_FORWARD_TOPOLOGIES",
        "D128_MX_TOPOLOGY_KEY",
        "D128_MX_FORWARD_TOPOLOGY_VARIANTS",
        "D128_FORWARD_TOPOLOGY_VARIANTS",
        "_d128_forward_topology_recipe",
    }
    d128_nodes = []
    for node in runtime_tree.body:
        selected_name = getattr(node, "name", None)
        assigned_names = {
            target.id
            for target in getattr(node, "targets", ())
            if isinstance(target, ast.Name)
        }
        if (
            selected_name in runtime_dependencies
            or assigned_names.intersection(runtime_dependencies)
        ):
            d128_nodes.append(node)
    selected = []
    for node in tree.body:
        selected_name = getattr(node, "name", None)
        assigned_names = {
            target.id
            for target in getattr(node, "targets", ())
            if isinstance(target, ast.Name)
        }
        if selected_name in names or assigned_names.intersection(names):
            selected.append(node)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            *d128_nodes,
            *selected,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {
        "AUTHENTICATED_D128_EXACT_BATCHES": (2,),
        "Any": Any,
        "math": math,
        "statistics": statistics,
        "string": string,
        "Path": Path,
    }
    exec(compile(module, str(HARNESS), "exec"), namespace)
    return namespace


def test_custom_projection_identity_is_explicit_and_fail_closed(
    tmp_path: Path,
) -> None:
    namespace = _namespace(
        "DEFAULT_PROJECTION",
        "PINNED_ARTIFACTS",
        "_caller_declared_expected_identity",
        "_projection_expected_identity",
    )
    resolve = namespace["_projection_expected_identity"]
    expected, authentication = resolve(
        namespace["DEFAULT_PROJECTION"],
        None,
        None,
    )
    assert expected == namespace["PINNED_ARTIFACTS"]["projection"]
    assert authentication == "source_pinned"

    custom = tmp_path / "candidate.so"
    with pytest.raises(ValueError, match="requires both"):
        resolve(custom, "0" * 64, None)
    with pytest.raises(ValueError, match="explicitly declared"):
        resolve(custom, None, None)
    expected, authentication = resolve(custom, "AB" * 32, 123)
    assert expected == ("ab" * 32, 123)
    assert authentication == "caller_declared"
    with pytest.raises(ValueError, match="64 hexadecimal"):
        resolve(custom, "not-a-digest", 123)
    with pytest.raises(ValueError, match="positive"):
        resolve(custom, "0" * 64, 0)


def test_forward_identity_supports_pinned_and_caller_declared_artifacts(
    tmp_path: Path,
) -> None:
    namespace = _namespace(
        "DEFAULT_FORWARDS",
        "FORWARD_MODULES",
        "LOWP_ROUTES",
        "PINNED_ARTIFACTS",
        "_caller_declared_expected_identity",
        "_forward_expected_identity",
    )
    resolve = namespace["_forward_expected_identity"]
    expected, authentication, module = resolve(
        "fp8",
        namespace["DEFAULT_FORWARDS"]["fp8"],
        None,
        None,
        None,
    )
    assert expected == namespace["PINNED_ARTIFACTS"]["forward"]["fp8"]
    assert authentication == "source_pinned"
    assert module == namespace["FORWARD_MODULES"]["fp8"]

    custom = tmp_path / "d128-forward.so"
    expected, authentication, module = resolve(
        "mx",
        custom,
        "_C_d128_safe_mx",
        "AB" * 32,
        456,
    )
    assert expected == ("ab" * 32, 456)
    assert authentication == "caller_declared"
    assert module == "_C_d128_safe_mx"
    with pytest.raises(ValueError, match="requires --forward-module"):
        resolve("mx", custom, None, "ab" * 32, 456)
    with pytest.raises(ValueError, match="requires both"):
        resolve("mx", custom, "_C_d128_safe_mx", "ab" * 32, None)


def test_loaded_artifact_receipt_binds_requested_extension_bytes(
    tmp_path: Path,
) -> None:
    require = _namespace("_require_loaded_artifact_identity")[
        "_require_loaded_artifact_identity"
    ]
    extension_path = tmp_path / "candidate.so"
    payload = b"loaded extension image"
    extension_path.write_bytes(payload)
    expected = (hashlib.sha256(payload).hexdigest(), len(payload))
    identity = {
        "path": str(extension_path.resolve()),
        "sha256": expected[0],
        "bytes": expected[1],
        "device": extension_path.stat().st_dev,
        "inode": extension_path.stat().st_ino,
        "mtime_ns": extension_path.stat().st_mtime_ns,
    }
    extension = SimpleNamespace(
        _tk_fa4_loaded_artifact_identity=identity
    )
    assert require(
        "candidate", extension, extension_path, expected
    ) == identity

    extension._tk_fa4_loaded_artifact_identity = {
        **identity,
        "sha256": "0" * 64,
    }
    with pytest.raises(RuntimeError, match="identity mismatch"):
        require("candidate", extension, extension_path, expected)
    with pytest.raises(RuntimeError, match="no loaded-artifact identity"):
        require("candidate", SimpleNamespace(), extension_path, expected)


def test_saturated_shape_and_diagnostics_cover_d64_b16_and_d128_b1_b2() -> None:
    namespace = _namespace(
        "SAMPLED_PARAMETER_NAMES",
        "HIDDEN_SAMPLE_POSITIONS",
        "_require_saturated_shape",
        "_diagnostic_sample_layout",
    )
    namespace["DEFAULT_MODEL_PRESET"] = "llama3.2-1b"
    require_shape = namespace["_require_saturated_shape"]
    sample_layout = namespace["_diagnostic_sample_layout"]
    d64 = SimpleNamespace(
        model_preset="llama3.2-1b",
        batch=16,
        sequence=4096,
        layers=16,
        hidden=2048,
        head_dim=64,
    )
    d128 = SimpleNamespace(
        model_preset="llama3.1-8b",
        batch=1,
        sequence=4096,
        layers=32,
        hidden=4096,
        head_dim=128,
    )
    require_shape(d64)
    require_shape(d128)
    d128_b2 = SimpleNamespace(**{**vars(d128), "batch": 2})
    require_shape(d128_b2)
    d64_names, d64_batches, d64_positions = sample_layout(d64)
    d128_names, d128_batches, d128_positions = sample_layout(d128)
    assert d64_batches == (0, 8, 15)
    assert d128_batches == (0,)
    assert sample_layout(d128_b2)[1] == (0, 1)
    assert d64_names[-1] == "layers.15.attention.weights.q"
    assert d128_names[-1] == "layers.31.attention.weights.q"
    assert d64_positions == d128_positions == namespace["HIDDEN_SAMPLE_POSITIONS"]
    with pytest.raises(ValueError, match="requires exactly"):
        require_shape(SimpleNamespace(**{**vars(d128), "batch": 4}))


def _d128_runtime_populated_topology(route: str) -> SimpleNamespace:
    if route == "fp8":
        route_key = (
            "real_fwd_tk_hao_direct_causal_gqa_nvfp4_fp8pv",
            "e4m3_fp8",
        )
    elif route == "mx":
        route_key = (
            "real_fwd_tk_hao_direct_nvfp4_mxfp4pv",
            "mxfp4_e8m0_block32",
        )
    else:
        raise ValueError(f"unsupported test route: {route}")
    topology = {
        **_namespace()["D128_EXACT_FORWARD_TOPOLOGIES"][route_key],
        "valid": 1,
    }
    return SimpleNamespace(
        forward_topology_runtime_authenticated=True,
        forward_topology=topology,
    )


@pytest.mark.parametrize("route", ("fp8", "mx"))
def test_d128_runtime_populated_topology_accepts_final_routes(
    route: str,
) -> None:
    require = _namespace(
        "_require_d128_runtime_populated_forward_topology"
    )["_require_d128_runtime_populated_forward_topology"]
    config = SimpleNamespace(head_dim=128)

    require(route, config, _d128_runtime_populated_topology(route))

    # BF16 D128 and every D64 route remain outside this low-precision gate.
    require("bf16_packed", config, None)
    require(
        route,
        SimpleNamespace(head_dim=64),
        SimpleNamespace(
            forward_topology_runtime_authenticated=False,
            forward_topology={},
        ),
    )


@pytest.mark.parametrize(
    ("route", "field", "wrong"),
    (
        ("fp8", "valid", 0),
        ("fp8", "shiftless_fp8_mode", 1),
        ("fp8", "causal_interleaved_kv", True),
        ("fp8", "fixed_p_ceiling", True),
        ("fp8", "ex2_emu_mask", 0),
        ("fp8", "retain_q2_mode", 0),
        ("fp8", "affine_a", 1.5),
        ("mx", "valid", 0),
        ("mx", "mx_scale_select", 1),
        ("mx", "mx_log2_p_quant", False),
        ("mx", "mx_quantized_denom", False),
        ("mx", "mx_p_effective_max", 5),
        ("mx", "mx_pwl_exp2", False),
        ("mx", "mx_pwl_exp2_mode", 22),
        ("mx", "mx_mode23_native_density", 4),
        ("mx", "mx_mode23_native_density3_quarter_mask", 0),
        ("mx", "mx_affine_a", 1.6),
        ("mx", "mx_shiftless_softmax", False),
        ("mx", "mx_denom_decode_mode", 0),
        ("mx", "mx_ex2_q1_mask", 0),
        ("mx", "ex2_alu_degree", 2),
        ("mx", "mx_stored_scale_shift_log2", 16),
        ("mx", "mx_global_anchor32", False),
        ("mx", "mx_global_anchor128", True),
        ("mx", "mx_global_anchor_margin_log2", 32),
        ("mx", "mx_anchor_affine_hoist", True),
        ("mx", "nv_qk_folded_k64_scales", True),
        ("mx", "nv_qk_folded_k64_scale_mask", 3),
        ("mx", "nv_qk_compact_folded_scales", True),
        ("mx", "nv_qk_preload_page_mask", 0),
        ("mx", "rowmax_pack_ceiling", True),
    ),
)
def test_d128_runtime_populated_topology_rejects_unsafe_mutations(
    route: str,
    field: str,
    wrong: object,
) -> None:
    require = _namespace(
        "_require_d128_runtime_populated_forward_topology"
    )["_require_d128_runtime_populated_forward_topology"]
    runtime = _d128_runtime_populated_topology(route)
    runtime.forward_topology[field] = wrong

    with pytest.raises(RuntimeError, match=field):
        require(route, SimpleNamespace(head_dim=128), runtime)


def test_d128_runtime_populated_topology_requires_completed_authentication(
) -> None:
    require = _namespace(
        "_require_d128_runtime_populated_forward_topology"
    )["_require_d128_runtime_populated_forward_topology"]
    runtime = _d128_runtime_populated_topology("mx")
    runtime.forward_topology_runtime_authenticated = False

    with pytest.raises(RuntimeError, match="not runtime-authenticated"):
        require("mx", SimpleNamespace(head_dim=128), runtime)


def test_native_projection_selection_is_explicit_and_provenance_bound() -> None:
    namespace = _namespace(
        "EXPERIMENTAL_NATIVE_NVFP4_ROUTES",
        "_require_saturated_projection_selection",
    )
    require = namespace["_require_saturated_projection_selection"]

    # The production E4M3 default retains its existing pinned or custom auth.
    require("fp8", "e4m3", False, False, False, "source_pinned")
    require("mx", "e4m3", False, False, False, "caller_declared")

    for route in ("fp8", "mx"):
        require(route, "nvfp4", True, False, False, "caller_declared")
        require(route, "nvfp4", True, True, False, "caller_declared")
    require("mx", "nvfp4", True, True, True, "caller_declared")
    with pytest.raises(ValueError, match="output-shared-split-v"):
        require("fp8", "nvfp4", True, True, True, "caller_declared")
    with pytest.raises(ValueError, match="requires --experimental"):
        require("fp8", "nvfp4", False, False, False, "caller_declared")
    with pytest.raises(ValueError, match="requires --qkv-projection-format"):
        require("fp8", "e4m3", True, False, False, "caller_declared")
    with pytest.raises(ValueError, match="exact fp8 or mx"):
        require(
            "mx_unanchored", "nvfp4", True, False, False, "caller_declared"
        )
    with pytest.raises(ValueError, match="caller-declared SHA256"):
        require("mx", "nvfp4", True, False, False, "source_pinned")
    with pytest.raises(ValueError, match="fused-attention-rmsnorm"):
        require("bf16", "nvfp4", True, True, False, "caller_declared")
    with pytest.raises(ValueError, match="fused-attention-rmsnorm"):
        require("fp8", "e4m3", False, True, False, "caller_declared")


def test_native_projection_contract_requires_new_publication_semantics() -> None:
    contract = _namespace("_qkv_projection_contract")[
        "_qkv_projection_contract"
    ]
    retained_checked = (
        "project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_"
        "interleaved_causal_represented_backward_perblock_qk_"
        "split_v_backward_mx_forward_out"
    )
    retained_unchecked = retained_checked + "_unchecked"
    publication = {
        "qkv_projection_format": "nvfp4",
        "forward_pv_format": "mxfp4_e8m0_block32",
        "represented_backward": True,
        "per_block_qk_scales": True,
        "qk_backward_source": "represented_nvfp4_codes_per_row_k16",
        "v_backward_source": "projection_accumulator_e4m3",
        "experimental_split_v_backward": True,
        "experimental_output_shared_split_v": False,
        "experimental_output_shared_split_v_requested": False,
        "experimental_output_shared_split_v_resolved": False,
        "output_shared_split_v_path": "retained_split_v",
        "output_shared_split_v_checked_symbol": retained_checked,
        "experimental_native_nvfp4_projection_out": True,
        "experimental_fused_attention_rmsnorm_nvfp4": False,
    }
    dispatch = {
        "qkv_projection": {
            "format": "nvfp4",
            "experimental_native_nvfp4_caller_owned": True,
            "experimental_fused_attention_rmsnorm_nvfp4": False,
            "backward_publication_semantics": (
                "represented_nvfp4_qk_per_row_k16_with_"
                "projection_accumulator_e4m3_v"
            ),
            "shape_bound_at_construction": True,
            "preallocated_forward_workspace_required": True,
            "output_shared_split_v_requested": False,
            "output_shared_split_v_resolved": False,
            "output_shared_split_v_path": "retained_split_v",
            "symbol": retained_unchecked,
            "checked_symbol": retained_checked,
            "unchecked_symbol": retained_unchecked,
        }
    }
    runtime = SimpleNamespace(
        projection_publication_topology=publication,
        forward_dispatch_contract=lambda: dispatch,
        experimental_native_nvfp4_projection_out=True,
        experimental_fused_attention_rmsnorm_nvfp4=False,
        experimental_output_shared_split_v=False,
        experimental_output_shared_split_v_requested=False,
        experimental_output_shared_split_v_resolved=False,
        output_shared_split_v_path="retained_split_v",
        qkv_projection_symbol=retained_unchecked,
        qkv_projection=SimpleNamespace(
            checked_symbol=retained_checked,
            unchecked_symbol=retained_unchecked,
        ),
        pv_format="mxfp4_e8m0_block32",
        v_mxfp4_scale_2d=False,
        config=SimpleNamespace(
            batch=16,
            sequence=4096,
            hidden=2048,
            q_heads=32,
            kv_heads=8,
            head_dim=64,
        ),
        projection_weight_scale_2d=True,
        qkv_projection_format="nvfp4",
    )
    artifact = {"authentication": "caller_declared", "sha256": "a" * 64}

    result = contract(runtime, artifact)

    assert result["operand_preparation"]["input"]["scale_layout"] == (
        "row_by_k16"
    )
    assert result["operand_preparation"]["input"]["function"] == (
        "b300_prepare_nvfp4_projection_operand"
    )
    assert not result["experimental_fused_attention_rmsnorm_nvfp4"]
    assert result["operand_preparation"]["learned_weight"][
        "scale_layout"
    ] == "true_16x16"
    publication["experimental_fused_attention_rmsnorm_nvfp4"] = True
    dispatch["qkv_projection"][
        "experimental_fused_attention_rmsnorm_nvfp4"
    ] = True
    runtime.experimental_fused_attention_rmsnorm_nvfp4 = True
    fused_result = contract(runtime, artifact)
    assert fused_result["operand_preparation"]["input"]["function"] == (
        "b300_prepare_nvfp4_projection_operand_rmsnorm"
    )
    assert fused_result["operand_preparation"]["input"][
        "fuses_attention_rmsnorm"
    ]
    publication["qk_backward_source"] = "projection_accumulator_e4m3"
    with pytest.raises(RuntimeError, match="projection contract mismatch"):
        contract(runtime, artifact)


@pytest.mark.parametrize(
    ("pv_format", "requested", "resolved", "path", "symbol_tail"),
    (
        (
            "mxfp4_e8m0_block32",
            None,
            True,
            "output_shared_split_v",
            "interleaved_causal_represented_backward_perblock_qk_"
            "output_shared_split_v_mx_forward_out",
        ),
        (
            "mxfp4_e8m0_block32",
            True,
            True,
            "output_shared_split_v",
            "interleaved_causal_represented_backward_perblock_qk_"
            "output_shared_split_v_mx_forward_out",
        ),
        (
            "e4m3_fp8",
            None,
            False,
            "fp8",
            "represented_backward_perblock_qk_fp8_forward_out",
        ),
    ),
)
def test_native_projection_contract_derives_candidate_and_fp8_selection(
    pv_format: str,
    requested: bool | None,
    resolved: bool,
    path: str,
    symbol_tail: str,
) -> None:
    contract = _namespace("_qkv_projection_contract")[
        "_qkv_projection_contract"
    ]
    checked = (
        "project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_"
        + symbol_tail
    )
    unchecked = checked + "_unchecked"
    split_v = pv_format == "mxfp4_e8m0_block32"
    publication = {
        "qkv_projection_format": "nvfp4",
        "represented_backward": True,
        "per_block_qk_scales": True,
        "qk_backward_source": "represented_nvfp4_codes_per_row_k16",
        "v_backward_source": "projection_accumulator_e4m3",
        "experimental_split_v_backward": split_v,
        "experimental_output_shared_split_v": resolved,
        "experimental_output_shared_split_v_requested": requested,
        "experimental_output_shared_split_v_resolved": resolved,
        "output_shared_split_v_path": path,
        "output_shared_split_v_checked_symbol": checked,
        "experimental_native_nvfp4_projection_out": True,
        "experimental_fused_attention_rmsnorm_nvfp4": False,
    }
    projection_dispatch = {
        "format": "nvfp4",
        "experimental_native_nvfp4_caller_owned": True,
        "experimental_fused_attention_rmsnorm_nvfp4": False,
        "backward_publication_semantics": (
            "represented_nvfp4_qk_per_row_k16_with_"
            "projection_accumulator_e4m3_v"
        ),
        "shape_bound_at_construction": True,
        "preallocated_forward_workspace_required": True,
        "output_shared_split_v_requested": requested,
        "output_shared_split_v_resolved": resolved,
        "output_shared_split_v_path": path,
        "symbol": unchecked,
        "checked_symbol": checked,
        "unchecked_symbol": unchecked,
    }
    runtime = SimpleNamespace(
        projection_publication_topology=publication,
        forward_dispatch_contract=lambda: {
            "qkv_projection": projection_dispatch
        },
        experimental_native_nvfp4_projection_out=True,
        experimental_fused_attention_rmsnorm_nvfp4=False,
        experimental_output_shared_split_v=resolved,
        experimental_output_shared_split_v_requested=requested,
        experimental_output_shared_split_v_resolved=resolved,
        output_shared_split_v_path=path,
        qkv_projection_symbol=unchecked,
        qkv_projection=SimpleNamespace(
            checked_symbol=checked,
            unchecked_symbol=unchecked,
        ),
        pv_format=pv_format,
        v_mxfp4_scale_2d=False,
        config=SimpleNamespace(
            batch=16,
            sequence=4096,
            hidden=2048,
            q_heads=32,
            kv_heads=8,
            head_dim=64,
        ),
        projection_weight_scale_2d=True,
        qkv_projection_format="nvfp4",
    )
    result = contract(
        runtime,
        {"authentication": "caller_declared", "sha256": "a" * 64},
    )
    assert result["experimental_native_nvfp4_projection_out"] is True


@pytest.mark.parametrize(
    (
        "batch",
        "pv_format",
        "requested",
        "resolved",
        "suffix",
        "path",
        "active_v",
    ),
    (
        (
            1,
            "mxfp4_e8m0_block32",
            False,
            False,
            "_mx_forward_out",
            "retained_dual_v",
            "mxfp4",
        ),
        (
            1,
            "mxfp4_e8m0_block32",
            None,
            True,
            "_output_shared_dual_v_mx_forward_out",
            "output_shared_dual_v",
            "mxfp4",
        ),
        (
            1,
            "e4m3_fp8",
            None,
            False,
            "_fp8_forward_out",
            "fp8",
            "e4m3_fp8",
        ),
        (
            2,
            "mxfp4_e8m0_block32",
            None,
            True,
            "_output_shared_dual_v_mx_forward_out",
            "output_shared_dual_v",
            "mxfp4",
        ),
        (
            2,
            "mxfp4_e8m0_block32",
            False,
            False,
            "_mx_forward_out",
            "retained_dual_v",
            "mxfp4",
        ),
        (
            2,
            "e4m3_fp8",
            None,
            False,
            "_fp8_forward_out",
            "fp8",
            "e4m3_fp8",
        ),
    ),
)
def test_d128_projection_contract_requires_route_selective_shared_backward(
    batch: int,
    pv_format: str,
    requested: bool | None,
    resolved: bool,
    suffix: str,
    path: str,
    active_v: str,
) -> None:
    namespace = _namespace(
        "D128_PROJECTION_ABI_SYMBOL",
        "_d128_qkv_projection_contract",
    )
    contract = namespace["_d128_qkv_projection_contract"]
    abi = namespace["D128_PROJECTION_ABI_SYMBOL"]
    checked = abi + suffix
    unchecked = checked + "_unchecked"
    publication = {
        "qkv_projection_format": "nvfp4",
        "forward_pv_format": pv_format,
        "represented_backward": False,
        "per_block_qk_scales": True,
        "qk_backward_source": "projection_accumulator_e4m3",
        "v_backward_source": "projection_accumulator_e4m3",
        "experimental_split_v_backward": False,
        "experimental_d128_mxfp4_v_backward": False,
        "d128_mxfp4_v_scale_policy": None,
        "experimental_output_shared_split_v": resolved,
        "experimental_output_shared_split_v_requested": requested,
        "experimental_output_shared_split_v_resolved": resolved,
        "output_shared_split_v_path": path,
        "projection_forward_publication_path": (
            "caller_owned_output_shared_dual_v_d128"
            if resolved
            else "caller_owned_route_selective_d128"
        ),
        # The runtime compatibility field is deliberately sanitized in the
        # saturated result and must not be presented as output-shared.
        "output_shared_split_v_checked_symbol": checked,
        "experimental_native_nvfp4_projection_out": True,
        "experimental_fused_attention_rmsnorm_nvfp4": False,
    }
    projection_dispatch = {
        "format": "nvfp4",
        "experimental_native_nvfp4_caller_owned": True,
        "experimental_fused_attention_rmsnorm_nvfp4": False,
        "experimental_d128_mxfp4_v_backward": False,
        "output_shared_split_v_requested": requested,
        "output_shared_split_v_resolved": resolved,
        "output_shared_split_v_path": path,
        "projection_forward_publication_path": (
            "caller_owned_output_shared_dual_v_d128"
            if resolved
            else "caller_owned_route_selective_d128"
        ),
        "backward_publication_semantics": (
            "projection_accumulator_e4m3_qkv_shared_across_pv_routes"
        ),
        "dispatch": "construction_bound_exact_pybind_symbol",
        "symbol": unchecked,
        "abi_validation_symbol": abi,
        "checked_symbol": checked,
        "unchecked_symbol": unchecked,
        "shape_bound_at_construction": True,
        "preallocated_forward_workspace_required": True,
        "timed_forward_publication_allocation_fallback": False,
    }
    runtime = SimpleNamespace(
        config=SimpleNamespace(
            batch=batch,
            sequence=4096,
            hidden=4096,
            q_heads=32,
            kv_heads=8,
            head_dim=128,
        ),
        pv_format=pv_format,
        projection_publication_topology=publication,
        forward_dispatch_contract=lambda: {
            "qkv_projection": projection_dispatch
        },
        qkv_projection=SimpleNamespace(
            abi_validation_symbol=abi,
            checked_symbol=checked,
            unchecked_symbol=unchecked,
            projection_forward_publication_path=(
                "caller_owned_output_shared_dual_v_d128"
                if resolved
                else "caller_owned_route_selective_d128"
            ),
            backward_publication_semantics=(
                "projection_accumulator_e4m3_qkv_shared_across_pv_routes"
            ),
            per_block_qk_scales=True,
            requires_forward_workspace=True,
            experimental_mx_backward_v=False,
            experimental_shared_tile_mx_backward_v=False,
            v_backward_mxfp4_scale_policy=None,
            experimental_output_shared_split_v_requested=requested,
            experimental_output_shared_split_v_resolved=resolved,
            output_shared_split_v_path=path,
        ),
        qkv_projection_symbol=unchecked,
        projection_weight_scale_2d=True,
        v_mxfp4_scale_2d=False,
        experimental_output_shared_split_v=resolved,
        experimental_d128_mxfp4_v_backward=False,
        d128_mxfp4_v_scale_policy=None,
        experimental_output_shared_split_v_requested=requested,
        experimental_output_shared_split_v_resolved=resolved,
        output_shared_split_v_path=path,
        projection_dgrad="nvfp4",
        backward_match_forward_operands=False,
        per_block_qk_scales=True,
        experimental_split_v_backward=False,
        backward_reuse_quantized_p=True,
        backward_exp2_degree=1,
        backward_exp2_period=0,
        native_tk_d128_backward=False,
    )
    result = contract(
        runtime,
        {"authentication": "caller_declared", "sha256": "a" * 64},
    )
    assert result["schema"] == "saturated_qkv_projection_contract_v4"
    route_contract = result["d128_route_selective_publication"]
    assert route_contract["active_forward_v"] == active_v
    assert route_contract["inactive_forward_v_omitted"] is True
    assert route_contract["qk_scale_geometry"] == "row_by_k16"
    assert route_contract["shared_backward_qkv"] == (
        "e4m3_projection_accumulator"
    )
    assert route_contract["output_shared_split_v"] is resolved
    assert route_contract["output_shared_candidate_eligible"] is (
        batch in (1, 2) and pv_format == "mxfp4_e8m0_block32"
    )
    assert route_contract["publication_path"] == path
    assert result["publication"]["output_shared_split_v_checked_symbol"] == (
        checked if resolved else None
    )
    learned_weight = result["operand_preparation"]["learned_weight"]
    assert learned_weight["function"] == (
        "b300_prepare_gqa_d128_qkv_projection_weight_dual_out"
    )
    assert learned_weight["source"] == "canonical_split_qkv_parameters"
    assert learned_weight["forward_operand"] == {
        "format": "nvfp4",
        "physical_layout": "pair_interleaved_qk_then_canonical_v",
        "scale_layout": "true_16x16",
    }
    assert learned_weight["backward_operand"] == {
        "format": "nvfp4",
        "physical_layout": (
            "transpose_of_pair_interleaved_qk_then_canonical_v"
        ),
        "scale_layout": "true_16x16",
    }
    assert learned_weight["caller_owned"] is True
    assert learned_weight["shared_global_scale"] is True
    assert learned_weight["refresh"] == "every_forward"
    assert learned_weight["first_use_authentication"] == {
        "comparison": "bitwise_all_published_bytes",
        "reference": (
            "pair_interleave_concat_then_independent_true_2d_quantization"
        ),
        "checked_symbol": (
            "quantize_gqa_d128_qkv_projection_weight_dual_out"
        ),
        "hot_path_symbol": (
            "quantize_gqa_d128_qkv_projection_weight_dual_out_unchecked"
        ),
    }

    runtime.qkv_projection.per_block_qk_scales = False
    with pytest.raises(RuntimeError, match="bound_projection"):
        contract(
            runtime,
            {"authentication": "caller_declared", "sha256": "a" * 64},
        )
    runtime.qkv_projection.per_block_qk_scales = True

    projection_dispatch["backward_publication_semantics"] = "represented_mxfp4"
    with pytest.raises(RuntimeError, match="D128 route-selective"):
        contract(
            runtime,
            {"authentication": "caller_declared", "sha256": "a" * 64},
        )


def test_d128_output_shared_candidate_rejects_b3_even_when_requested() -> None:
    contract = _namespace(
        "D128_PROJECTION_ABI_SYMBOL",
        "_d128_qkv_projection_contract",
    )["_d128_qkv_projection_contract"]
    runtime = SimpleNamespace(
        config=SimpleNamespace(
            batch=3,
            sequence=4096,
            hidden=4096,
            q_heads=32,
            kv_heads=8,
            head_dim=128,
        ),
        pv_format="mxfp4_e8m0_block32",
        experimental_d128_mxfp4_v_backward=False,
        d128_mxfp4_v_scale_policy=None,
        experimental_output_shared_split_v_requested=True,
    )

    with pytest.raises(RuntimeError, match="outside the authenticated B1/B2"):
        contract(runtime, {"authentication": "caller_declared"})


def _d128_direct_dual_workspace_contract(
    layer_count: int = 2,
) -> dict[str, Any]:
    owner_names = (
        "qkv_weight_forward_packed",
        "qkv_weight_forward_scales",
        "qkv_weight_backward_packed",
        "qkv_weight_backward_scales",
        "qkv_weight_global_scale",
    )
    output_owner_names = (
        "output_weight_forward_packed",
        "output_weight_forward_scales",
        "output_weight_backward_packed",
        "output_weight_backward_scales",
        "output_weight_global_scale",
    )
    layers = []
    for layer_index in range(layer_count):
        owners = {}
        for owner_index, owner_name in enumerate(owner_names):
            owners[owner_name] = {
                "data_ptr": 100_000 + 10 * layer_index + owner_index,
                "pointer_stable_since_allocation": True,
                "bytes": 64 + owner_index,
                "listed_in_named_buffers": False,
                "listed_in_named_parameters": False,
                "optimizer_visible_parameter": False,
            }
        output_owners = {}
        for owner_index, owner_name in enumerate(output_owner_names):
            output_owners[owner_name] = {
                "data_ptr": 200_000 + 10 * layer_index + owner_index,
                "pointer_stable_since_allocation": True,
                "bytes": 96 + owner_index,
                "listed_in_named_buffers": False,
                "listed_in_named_parameters": False,
                "optimizer_visible_parameter": False,
            }
        layers.append(
            {
                "layer": layer_index,
                "publication_lifecycle": {
                    "generation_guard_enforced": True,
                    "same_stream_enforced": True,
                    "one_forward_in_flight_per_layer": True,
                    "current_generation": 0,
                    "in_flight": False,
                },
                "d128_dual_qkv_weight": {
                    "eligible": True,
                    "authenticated": True,
                    "schedule": "synchronous_same_stream",
                    "one_forward_in_flight_per_layer": True,
                    "generation_guard_enforced": True,
                    "same_stream_enforced": True,
                    "abi_identity_bound": True,
                    "abi_identity_tensor_count": 8,
                    "abi_identity_excludes_tensor_version": True,
                    "checked_symbol": (
                        "quantize_gqa_d128_qkv_projection_weight_dual_out"
                    ),
                    "unchecked_symbol": (
                        "quantize_gqa_d128_qkv_projection_weight_dual_out_"
                        "unchecked"
                    ),
                    "owners": owners,
                    "all_pointers_stable_since_allocation": True,
                    "all_pointers_unique": True,
                    "total_bytes": sum(
                        owner["bytes"] for owner in owners.values()
                    ),
                },
                "dual_output_weight": {
                    "eligible": True,
                    "authenticated": True,
                    "schedule": "synchronous_same_stream",
                    "one_forward_in_flight_per_layer": True,
                    "generation_guard_enforced": True,
                    "same_stream_enforced": True,
                    "abi_identity_bound": True,
                    "abi_identity_tensor_count": 6,
                    "abi_identity_excludes_tensor_version": True,
                    "checked_symbol": (
                        "quantize_nvfp4_projection_weight_dual_out"
                    ),
                    "unchecked_symbol": (
                        "quantize_nvfp4_projection_weight_dual_out_unchecked"
                    ),
                    "owners": output_owners,
                    "all_pointers_stable_since_allocation": True,
                    "all_pointers_unique": True,
                    "total_bytes": sum(
                        owner["bytes"] for owner in output_owners.values()
                    ),
                },
            }
        )
    return {
        "schema": "lowp_model_forward_workspaces_v2",
        "layer_count": layer_count,
        "layers": layers,
    }


def test_d128_direct_dual_weight_receipt_is_post_auth_and_compact() -> None:
    receipt_function = _namespace(
        "_d128_dual_qkv_weight_preparation_receipt"
    )["_d128_dual_qkv_weight_preparation_receipt"]
    contract = _d128_direct_dual_workspace_contract()
    model = SimpleNamespace(
        lowp_forward_workspace_contract=lambda: contract
    )
    config = SimpleNamespace(head_dim=128, layers=2)

    receipt = receipt_function(model, config, SimpleNamespace())

    assert receipt == {
        "schema": "d128_direct_dual_qkv_weight_preparation_v1",
        "observed_after": "initial_diagnostic_forward_backward",
        "source_contract_schema": "lowp_model_forward_workspaces_v2",
        "expected_layer_count": 2,
        "eligible_layer_count": 2,
        "authenticated_layer_count": 2,
        "abi_identity_bound_layer_count": 2,
        "generation_guard_enforced_layer_count": 2,
        "same_stream_enforced_layer_count": 2,
        "in_flight_layer_count": 0,
        "schedule": "synchronous_same_stream",
        "source": "canonical_split_qkv_parameters",
        "caller_owned": True,
        "refresh": "every_forward",
        "abi_identity_excludes_tensor_version": True,
        "first_use_authentication": (
            "bitwise_against_pair_interleave_concat_then_independent_"
            "true_2d_quantization"
        ),
        "checked_symbol": (
            "quantize_gqa_d128_qkv_projection_weight_dual_out"
        ),
        "unchecked_symbol": (
            "quantize_gqa_d128_qkv_projection_weight_dual_out_unchecked"
        ),
        "owner_fields": [
            "qkv_weight_forward_packed",
            "qkv_weight_forward_scales",
            "qkv_weight_backward_packed",
            "qkv_weight_backward_scales",
            "qkv_weight_global_scale",
        ],
        "owner_count_per_layer": 5,
        "all_owner_tensors_private_nonpersistent": True,
        "all_pointers_stable_since_allocation": True,
        "owner_pointers_globally_unique": True,
        "total_bytes": 660,
    }
    assert "layers" not in receipt
    assert "owners" not in receipt
    assert "data_ptr" not in receipt

    unused_model = SimpleNamespace(
        lowp_forward_workspace_contract=lambda: pytest.fail(
            "non-D128 routes must not inspect low-precision workspaces"
        )
    )
    assert receipt_function(
        unused_model,
        SimpleNamespace(head_dim=64, layers=2),
        SimpleNamespace(),
    ) is None
    assert receipt_function(
        unused_model,
        config,
        None,
    ) is None


def test_d128_direct_dual_weight_receipt_fails_closed() -> None:
    receipt_function = _namespace(
        "_d128_dual_qkv_weight_preparation_receipt"
    )["_d128_dual_qkv_weight_preparation_receipt"]
    config = SimpleNamespace(head_dim=128, layers=2)

    unauthenticated = _d128_direct_dual_workspace_contract()
    unauthenticated["layers"][1]["d128_dual_qkv_weight"][
        "authenticated"
    ] = False
    with pytest.raises(RuntimeError, match="contract mismatch at layer 1"):
        receipt_function(
            SimpleNamespace(
                lowp_forward_workspace_contract=lambda: unauthenticated
            ),
            config,
            SimpleNamespace(),
        )

    non_private = _d128_direct_dual_workspace_contract()
    non_private["layers"][0]["d128_dual_qkv_weight"]["owners"][
        "qkv_weight_forward_packed"
    ]["optimizer_visible_parameter"] = True
    with pytest.raises(RuntimeError, match="not private scratch"):
        receipt_function(
            SimpleNamespace(
                lowp_forward_workspace_contract=lambda: non_private
            ),
            config,
            SimpleNamespace(),
        )

    reused_pointer = _d128_direct_dual_workspace_contract()
    first_owner = reused_pointer["layers"][0]["d128_dual_qkv_weight"][
        "owners"
    ]["qkv_weight_forward_packed"]
    second_owner = reused_pointer["layers"][1]["d128_dual_qkv_weight"][
        "owners"
    ]["qkv_weight_forward_packed"]
    second_owner["data_ptr"] = first_owner["data_ptr"]
    with pytest.raises(RuntimeError, match="not globally unique"):
        receipt_function(
            SimpleNamespace(
                lowp_forward_workspace_contract=lambda: reused_pointer
            ),
            config,
            SimpleNamespace(),
        )


@pytest.mark.parametrize("head_dim", (64, 128))
def test_dual_output_weight_receipt_proves_every_layer(
    head_dim: int,
) -> None:
    receipt_function = _namespace(
        "_dual_output_weight_preparation_receipt"
    )["_dual_output_weight_preparation_receipt"]
    contract = _d128_direct_dual_workspace_contract()
    receipt = receipt_function(
        SimpleNamespace(lowp_forward_workspace_contract=lambda: contract),
        SimpleNamespace(head_dim=head_dim, layers=2),
        SimpleNamespace(projection_weight_scale_2d=True),
    )

    assert receipt["schema"] == "direct_dual_output_weight_preparation_v1"
    assert receipt["observed_head_dim"] == head_dim
    assert receipt["eligible_head_dims"] == [64, 128]
    assert receipt["expected_layer_count"] == 2
    assert receipt["eligible_layer_count"] == 2
    assert receipt["authenticated_layer_count"] == 2
    assert receipt["abi_identity_bound_layer_count"] == 2
    assert receipt["generation_guard_enforced_layer_count"] == 2
    assert receipt["same_stream_enforced_layer_count"] == 2
    assert receipt["in_flight_layer_count"] == 0
    assert receipt["function"] == (
        "b300_prepare_nvfp4_projection_weight_dual_out"
    )
    assert receipt["checked_symbol"] == (
        "quantize_nvfp4_projection_weight_dual_out"
    )
    assert receipt["unchecked_symbol"] == (
        "quantize_nvfp4_projection_weight_dual_out_unchecked"
    )
    assert receipt["caller_owned"] is True
    assert receipt["shared_global_scale"] is True
    assert receipt["refresh"] == "every_forward"
    assert receipt["abi_identity_excludes_tensor_version"] is True
    assert receipt["all_pointers_stable_since_allocation"] is True
    assert receipt["owner_pointers_globally_unique"] is True
    assert receipt["total_bytes"] == 980
    assert "layers" not in receipt
    assert "owners" not in receipt
    assert "data_ptr" not in receipt


def test_dual_output_weight_receipt_rejects_missing_guards_or_auth() -> None:
    receipt_function = _namespace(
        "_dual_output_weight_preparation_receipt"
    )["_dual_output_weight_preparation_receipt"]
    config = SimpleNamespace(head_dim=128, layers=2)
    runtime = SimpleNamespace(projection_weight_scale_2d=True)

    unbound = _d128_direct_dual_workspace_contract()
    unbound["layers"][0]["dual_output_weight"][
        "abi_identity_bound"
    ] = False
    with pytest.raises(RuntimeError, match="preparation contract mismatch"):
        receipt_function(
            SimpleNamespace(lowp_forward_workspace_contract=lambda: unbound),
            config,
            runtime,
        )

    in_flight = _d128_direct_dual_workspace_contract()
    in_flight["layers"][1]["publication_lifecycle"]["in_flight"] = True
    with pytest.raises(RuntimeError, match="lifecycle contract mismatch"):
        receipt_function(
            SimpleNamespace(
                lowp_forward_workspace_contract=lambda: in_flight
            ),
            config,
            runtime,
        )


def test_d128_direct_dual_receipt_runs_after_auth_before_timing() -> None:
    source = HARNESS.read_text()
    diagnostic = source.index("initial_diagnostic = _diagnostic_pass(")
    topology_gate = source.index(
        "_require_d128_runtime_populated_forward_topology(", diagnostic
    )
    receipt = source.index(
        "d128_dual_qkv_weight_preparation = (", topology_gate
    )
    output_receipt = source.index(
        "dual_output_weight_preparation = (", receipt
    )
    timing = source.index("torch.cuda.reset_peak_memory_stats()", output_receipt)

    assert diagnostic < topology_gate < receipt < output_receipt < timing
    assert '"d128_dual_qkv_weight_preparation": (' in source
    assert '"dual_output_weight_preparation": ' in source


@pytest.mark.parametrize(
    ("section", "field", "wrong"),
    (
        ("publication", "experimental_output_shared_split_v", True),
        (
            "publication",
            "experimental_output_shared_split_v_resolved",
            True,
        ),
        (
            "publication",
            "output_shared_split_v_path",
            "output_shared_split_v",
        ),
        (
            "publication",
            "output_shared_split_v_checked_symbol",
            "wrong_checked",
        ),
        ("dispatch", "output_shared_split_v_resolved", True),
        ("dispatch", "output_shared_split_v_path", "output_shared_split_v"),
        ("dispatch", "symbol", "wrong_unchecked"),
        ("dispatch", "checked_symbol", "wrong_checked"),
        ("dispatch", "unchecked_symbol", "wrong_unchecked"),
        ("runtime", "experimental_output_shared_split_v_requested", None),
        ("runtime", "experimental_output_shared_split_v_resolved", True),
        ("runtime", "experimental_output_shared_split_v", True),
        ("runtime", "output_shared_split_v_path", "output_shared_split_v"),
        ("runtime", "qkv_projection_symbol", "wrong_unchecked"),
        ("projection", "checked_symbol", "wrong_checked"),
        ("projection", "unchecked_symbol", "wrong_unchecked"),
    ),
)
def test_native_projection_contract_rejects_derived_field_or_symbol_mutation(
    section: str,
    field: str,
    wrong: object,
) -> None:
    contract = _namespace("_qkv_projection_contract")[
        "_qkv_projection_contract"
    ]
    retained_checked = (
        "project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_"
        "interleaved_causal_represented_backward_perblock_qk_"
        "split_v_backward_mx_forward_out"
    )
    retained_unchecked = retained_checked + "_unchecked"
    publication = {
        "qkv_projection_format": "nvfp4",
        "represented_backward": True,
        "per_block_qk_scales": True,
        "qk_backward_source": "represented_nvfp4_codes_per_row_k16",
        "v_backward_source": "projection_accumulator_e4m3",
        "experimental_split_v_backward": True,
        "experimental_output_shared_split_v": False,
        "experimental_output_shared_split_v_requested": False,
        "experimental_output_shared_split_v_resolved": False,
        "output_shared_split_v_path": "retained_split_v",
        "output_shared_split_v_checked_symbol": retained_checked,
        "experimental_native_nvfp4_projection_out": True,
        "experimental_fused_attention_rmsnorm_nvfp4": False,
    }
    projection_dispatch = {
        "format": "nvfp4",
        "experimental_native_nvfp4_caller_owned": True,
        "experimental_fused_attention_rmsnorm_nvfp4": False,
        "backward_publication_semantics": (
            "represented_nvfp4_qk_per_row_k16_with_"
            "projection_accumulator_e4m3_v"
        ),
        "shape_bound_at_construction": True,
        "preallocated_forward_workspace_required": True,
        "output_shared_split_v_requested": False,
        "output_shared_split_v_resolved": False,
        "output_shared_split_v_path": "retained_split_v",
        "symbol": retained_unchecked,
        "checked_symbol": retained_checked,
        "unchecked_symbol": retained_unchecked,
    }
    runtime = SimpleNamespace(
        projection_publication_topology=publication,
        forward_dispatch_contract=lambda: {
            "qkv_projection": projection_dispatch
        },
        experimental_native_nvfp4_projection_out=True,
        experimental_fused_attention_rmsnorm_nvfp4=False,
        experimental_output_shared_split_v=False,
        experimental_output_shared_split_v_requested=False,
        experimental_output_shared_split_v_resolved=False,
        output_shared_split_v_path="retained_split_v",
        qkv_projection_symbol=retained_unchecked,
        qkv_projection=SimpleNamespace(
            checked_symbol=retained_checked,
            unchecked_symbol=retained_unchecked,
        ),
        pv_format="mxfp4_e8m0_block32",
        v_mxfp4_scale_2d=False,
        config=SimpleNamespace(
            batch=16,
            sequence=4096,
            hidden=2048,
            q_heads=32,
            kv_heads=8,
            head_dim=64,
        ),
        projection_weight_scale_2d=True,
        qkv_projection_format="nvfp4",
    )
    if section == "publication":
        publication[field] = wrong
    elif section == "dispatch":
        projection_dispatch[field] = wrong
    elif section == "projection":
        setattr(runtime.qkv_projection, field, wrong)
    else:
        setattr(runtime, field, wrong)
    with pytest.raises(RuntimeError, match="projection contract mismatch"):
        contract(
            runtime,
            {"authentication": "caller_declared", "sha256": "a" * 64},
        )


def test_saturated_harness_wires_native_runtime_and_records_contract() -> None:
    source = HARNESS.read_text()
    runtime_source = RUNTIME.read_text()

    assert '"--qkv-projection-format"' in source
    assert 'default="e4m3"' in source
    assert '"--experimental-native-nvfp4-projection-out"' in source
    assert '"--experimental-fused-attention-rmsnorm-nvfp4"' in source
    assert '"--experimental-output-shared-split-v"' in source
    output_shared_cli = source.split(
        '"--experimental-output-shared-split-v"', 1
    )[1].split("parser.add_argument", 1)[0]
    assert "action=argparse.BooleanOptionalAction" in output_shared_cli
    assert "default=None" in output_shared_cli
    runtime_builder = source.split("def _runtime(", 1)[1].split(
        "def _timed_update(", 1
    )[0]
    assert "qkv_projection_format=qkv_projection_format" in runtime_builder
    assert "experimental_native_nvfp4_projection_out=(" in runtime_builder
    assert "experimental_fused_attention_rmsnorm_nvfp4=(" in runtime_builder
    assert "experimental_output_shared_split_v=(" in runtime_builder
    assert "backward_match_forward_operands=not is_d128" in runtime_builder
    assert "per_block_qk_scales=True" in runtime_builder
    assert (
        "(route in MX_ROUTES) if not is_d128 else False"
        in runtime_builder
    )
    assert "projection_weight_scale_2d=True" in runtime_builder
    assert '"qkv_projection_contract": (' in source
    assert '"experimental_native_nvfp4_projection_out": bool(' in source
    assert '"experimental_fused_attention_rmsnorm_nvfp4": bool(' in source
    assert '"experimental_output_shared_split_v": bool(' in source
    assert '"experimental_output_shared_split_v_requested": (' in source
    assert '"experimental_output_shared_split_v_resolved": bool(' in source
    assert '"output_shared_split_v_path": (' in source
    result_configuration = source.rsplit("result = {", 1)[1].split(
        '"data": data_receipt', 1
    )[0]
    generic_checked_symbol = result_configuration.split(
        '"output_shared_split_v_checked_symbol": (', 1
    )[1].split('"d128_route_selective_checked_symbol": (', 1)[0]
    assert "config.head_dim == 128" in generic_checked_symbol
    assert (
        "runtime.experimental_output_shared_split_v"
        in generic_checked_symbol
    )
    assert '"b300_prepare_nvfp4_projection_operand_rmsnorm"' in source
    d128_main_gate = source.split(
        "if is_d128 and args.route in LOWP_ROUTES:", 1
    )[1].split("elif not is_d128", 1)[0]
    d128_default_resolution = source.split(
        "output_shared_cli_request = args.experimental_output_shared_split_v",
        1,
    )[1].split("if is_d128 and args.route in LOWP_ROUTES:", 1)[0]
    assert "if is_d128 and output_shared_cli_request is None:" in (
        d128_default_resolution
    )
    assert "args.experimental_output_shared_split_v = False" in (
        d128_default_resolution
    )
    assert "args.experimental_output_shared_split_v = False" not in (
        d128_main_gate
    )
    assert "D128 uses route-selective publication" not in d128_main_gate

    d128_projection_bind = runtime_source.split(
        "b300_bind_qkv_gqa_d128_unified_lowp_nvfp4_projection(", 1
    )[1].split(")\n                )", 1)[0]
    assert "experimental_output_shared_dual_v=(" in d128_projection_bind
    assert "experimental_output_shared_split_v" in d128_projection_bind

    projection_forward = runtime_source.split(
        "class _LowpAttentionFunction", 1
    )[1].split("class LowpAttention", 1)[0]
    assert "b300_prepare_nvfp4_projection_operand(rows)" in projection_forward
    assert "b300_prepare_nvfp4_projection_weight" in projection_forward


def test_b1_native_mx_enables_split_v_without_changing_legacy_e4m3() -> None:
    source = RUNTIME.read_text()
    split_policy = source.split(
        "batched_mx_split_v_backward = (", 1
    )[1].split("rope = _make_llama3_rope(config)", 1)[0]

    assert 'topology.get("pv_format") == "mxfp4_e8m0_block32"' in split_policy
    assert "config.batch != 1" in split_policy
    assert "or args.experimental_native_nvfp4_projection_out" in split_policy
    assert (
        "experimental_split_v_backward=batched_mx_split_v_backward" in source
    )


def test_harness_requires_sustained_measurements_and_shared_state() -> None:
    source = HARNESS.read_text()

    assert "MINIMUM_MEASURED_UPDATES = 20" in source
    assert "add_mutually_exclusive_group(required=True)" in source
    assert 'checkpoint_group.add_argument("--initial-checkpoint"' in source
    assert 'checkpoint_group.add_argument("--save-initial-checkpoint"' in source
    assert "model.load_state_dict(state, strict=True)" in source
    assert "refusing to overwrite an existing benchmark output" in source
    assert 'parser.add_argument("--save-final-checkpoint", type=Path)' in source
    assert 'kind="post_trajectory_model_state"' in source


def test_harness_exposes_runtime_loss_scale_and_experimental_mx_arm() -> None:
    source = HARNESS.read_text()

    assert 'parser.add_argument("--loss-scale", type=float' in source
    assert '"loss_scale": args.loss_scale if runtime is not None else None' in source
    assert 'LOWP_ROUTES = ("fp8", "mx", "mx_unanchored")' in source
    assert 'variant="unanchored-splitmix-v6"' in source
    assert "load_authenticated_mx_extension(" in source
    assert "require_mx_variant_topology(" in source


def test_packed_bf16_control_uses_canonical_split_comparison_schema() -> None:
    source = HARNESS.read_text()

    assert 'BF16_ROUTES = ("bf16", "bf16_packed")' in source
    assert 'if args.route == "bf16_packed"' in source
    assert '"packed_qkv_single_linear"' in source
    assert "pack_qkv_state_dict(" in source
    assert "unpack_qkv_state_dict(" in source
    assert "canonical_split_qkv_tensors(" in source
    assert "_canonical_parameter_tensors(model)" in source
    assert "_canonical_gradient_tensors(model)" in source
    assert 'reference_route == "bf16_packed"' in source
    assert "initial checkpoint must use the canonical split-QKV schema" in source
    assert 'reference["route"] not in BF16_ROUTES' in source
    assert '"attention_route": model.attention_route' in source
    assert '"physical_optimizer_parameter_tensors"' in source
    assert '"reference_sample_artifact": reference_samples_identity' in source
    assert "source_files_before = _benchmark_source_identities()" in source
    assert "source_files_after = _benchmark_source_identities()" in source
    assert "if source_files_after != source_files_before:" in source
    assert 'source_files.pop("packed_bf16_qkv")' in source
    assert "args.route not in BF16_ROUTES" in source
    assert source.index("peak_allocated = torch.cuda.max_memory_allocated()") < source.index(
        'kind="post_trajectory_model_state"'
    )


def test_harness_exposes_only_authenticated_full_depth_d128_batches() -> None:
    source = HARNESS.read_text()

    assert '"--model-preset"' in source
    assert "choices=MODEL_PRESETS" in source
    assert 'choices=(1, 2, 16)' in source
    assert "_require_saturated_shape(config)" in source
    assert '"llama3.1-8b": {' in source
    assert "AUTHENTICATED_D128_EXACT_BATCHES" in source
    assert '"layers": 32' in source
    assert 'parser.add_argument("--forward-module")' in source
    assert 'parser.add_argument("--forward-sha256")' in source
    assert 'parser.add_argument("--forward-bytes", type=int)' in source
    assert "_forward_expected_identity(" in source
    assert 'projection_dgrad="nvfp4" if is_d128 else "bf16"' in source
    assert 'backward_exp2_period=0 if is_d128 else 2' in source
    assert "is_d128 and not native_tk_d128_backward" in source
    assert 'backward_control_sha256=control_sha256' in source
    assert 'backward_match_forward_operands=not is_d128' in source
    assert 'per_block_qk_scales=True' in source
    assert '"d128_route_selective_publication"' in source
    assert '"mxfp4_v_plus_e4m3_qk"' in source
    assert '"e4m3_projection_accumulator"' in source
    assert '"d128_route_selective_checked_symbol"' in source


def test_harness_fp8_lse_control_is_explicitly_diagnostic_and_bound() -> None:
    source = HARNESS.read_text()

    for flag in (
        "--diagnostic-fp8-lse-extension",
        "--diagnostic-fp8-lse-module",
        "--diagnostic-fp8-lse-sha256",
        "--diagnostic-fp8-lse-bytes",
        "--diagnostic-fp8-lse-substitution-mode",
    ):
        assert flag in source
    assert "diagnostic FP8-LSE control requires extension" in source
    assert "supported only by the D128 MX route" in source
    assert 'artifacts["diagnostic_fp8_lse_control"] = diagnostic_artifact' in source
    assert "runtime.install_diagnostic_fp8_lse_control(" in source
    assert 'default="all_rows"' in source
    assert "choices=DIAGNOSTIC_FP8_LSE_SUBSTITUTION_MODES" in source
    assert "mx_nonfinite_only retains every" in source
    assert '"substitution_counts": dict(' in source
    assert '"substitution_count_scope": (' in source
    assert "where_mx_lse_is_nonfinite_in_shared_backward" in source
    assert '"production_route": False' in source
    assert '"production_timing_valid": not bool(' in source
    assert '"first_launch_receipt"' in source
    assert '"backward_contract_unchanged": runtime.backward_contract()' in source


def test_harness_reports_torch_compile_ce_without_claiming_cut_cce() -> None:
    source = HARNESS.read_text()

    assert '"implementation": "torch_compile"' in source
    assert '"logical_full_logits": True' in source
    assert '"source_operation": "logits = e @ c.T"' in source
    assert '"materialized_full_logits": False' not in source
    assert "it is not\nthe Triton Cut Cross Entropy implementation" in source


def test_harness_samples_long_context_and_update_vectors() -> None:
    source = HARNESS.read_text()

    assert "HIDDEN_SAMPLE_BATCHES = (0, 8, 15)" in source
    assert (
        "HIDDEN_SAMPLE_POSITIONS = "
        "(0, 15, 63, 255, 1023, 2047, 3071, 4095)"
    ) in source
    assert "* (flattened.numel() - 1)" in source
    assert '"parameter_updates": _subtract_samples(' in source
    assert '"initial_sampled_logits": _compare_logits(' in source
    assert '"final_sampled_logits": _compare_logits(' in source


def test_timing_statistics_include_distribution_and_variability() -> None:
    namespace = _namespace("_percentile", "_timing_statistics")
    records = [{"step_ms": value} for value in (1.0, 2.0, 3.0, 4.0, 5.0)]

    result = namespace["_timing_statistics"](records, ("step_ms",))["step_ms"]

    assert result["p10"] == 1.4
    assert result["p50"] == 3.0
    assert result["p90"] == 4.6
    assert result["mean"] == 3.0
    assert result["minimum"] == 1.0
    assert result["maximum"] == 5.0
    assert result["coefficient_of_variation"] > 0.0


def test_reference_validation_is_fail_closed() -> None:
    source = HARNESS.read_text()

    assert "if set(reference) != expected_keys:" in source
    assert (
        'candidate["comparison_identity"] != '
        'reference["comparison_identity"]'
    ) in source
    assert "reference checkpoint {key} mismatch" in source
    assert "sample key mismatch" in source
    assert "sample shape mismatch" in source


def test_result_labels_recipe_scope_and_bf16_equivalent_mfu() -> None:
    source = HARNESS.read_text()

    assert 'artifacts: dict[str, Any] = {}' in source
    assert "if args.route in LOWP_ROUTES:" in source
    assert '"comparison_scope": _comparison_scope(' in source
    assert '"p50_bf16_equivalent_useful_mfu_at_2250_tflops"' in source
    assert '"hardware_utilization_requires_external_profiler": True' in source


def test_profile_step_exposes_nested_nvtx_phases() -> None:
    source = HARNESS.read_text()

    for label in (
        "profile_step",
        "decoder_forward",
        "ce_forward",
        "backward_total",
        "gradient_clip",
        "optimizer",
    ):
        assert f'torch.cuda.nvtx.range("{label}")' in source
    assert 'parser.add_argument(\n        "--profile-update"' in source
