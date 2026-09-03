from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "tk_fa4/lowp_fa4_bwd/benchmark_llama12b_e2e.py"


def _tree() -> ast.Module:
    return ast.parse(RUNTIME.read_text())


def _top_level_function(name: str) -> ast.FunctionDef:
    return next(
        node
        for node in _tree().body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _class_method(class_name: str, method_name: str) -> ast.FunctionDef:
    class_node = next(
        node
        for node in _tree().body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )


def _execute_function(name: str) -> Any:
    module = ast.Module(
        body=[
            ast.parse("from __future__ import annotations").body[0],
            _top_level_function(name),
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {"Any": Any}
    exec(compile(module, str(RUNTIME), "exec"), namespace)
    return namespace[name]


def _config(**overrides: int) -> SimpleNamespace:
    values = {
        "batch": 1,
        "sequence": 4096,
        "hidden": 4096,
        "q_heads": 32,
        "kv_heads": 8,
        "head_dim": 128,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _gate_kwargs(projection_format: str, **overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "qkv_projection_format": projection_format,
        "output_projection_format": projection_format,
        "experimental_native_nvfp4_projection_out": projection_format == "nvfp4",
        "native_tk_d128_native_score_backward": True,
        "backward_match_forward_operands": False,
        "per_block_qk_scales": True,
        "experimental_split_v_backward": False,
        "experimental_output_shared_split_v": False,
        "experimental_d128_mxfp4_v_backward": False,
        "v_mxfp4_scale_2d": False,
        "shared_runtime": None,
    }
    values.update(overrides)
    return values


@pytest.mark.parametrize(
    ("batch", "projection_format", "pv_format"),
    [
        (batch, projection_format, pv_format)
        for batch in (1, 2, 4)
        for projection_format, pv_format in (
            ("e4m3", "e4m3_fp8"),
            ("e4m3", "mxfp4_e8m0_block32"),
            ("nvfp4", "e4m3_fp8"),
            ("nvfp4", "mxfp4_e8m0_block32"),
        )
    ],
)
def test_v509_gate_admits_all_projection_pv_arms(
    batch: int,
    projection_format: str,
    pv_format: str,
) -> None:
    require = _execute_function(
        "_require_native_tk_d128_v509_e5m2_dout_runtime_contract"
    )
    require(
        _config(batch=batch),
        {"pv_format": pv_format},
        **_gate_kwargs(projection_format),
    )


@pytest.mark.parametrize(
    ("config_overrides", "projection_format", "kwarg_overrides", "message"),
    [
        ({"batch": 0}, "e4m3", {}, "B1/B2/B4"),
        ({"batch": 3}, "e4m3", {}, "B1/B2/B4"),
        ({"batch": 8}, "e4m3", {}, "B1/B2/B4"),
        ({"sequence": 2048}, "e4m3", {}, "S4096"),
        ({}, "e4m3", {"output_projection_format": "nvfp4"}, "learned projections"),
        ({}, "nvfp4", {"experimental_native_nvfp4_projection_out": False}, "caller-owned"),
        ({}, "e4m3", {"experimental_native_nvfp4_projection_out": True}, "without the NVFP4"),
        ({}, "e4m3", {"native_tk_d128_native_score_backward": False}, "native NVFP4 score"),
        ({}, "e4m3", {"backward_match_forward_operands": True}, "projection-accumulator"),
        ({}, "e4m3", {"per_block_qk_scales": False}, "row-by-K16"),
        ({}, "e4m3", {"experimental_split_v_backward": True}, "retained"),
        ({}, "e4m3", {"experimental_output_shared_split_v": True}, "disable"),
        ({}, "e4m3", {"experimental_d128_mxfp4_v_backward": True}, "no MXFP4"),
        ({}, "e4m3", {"v_mxfp4_scale_2d": True}, "no MXFP4"),
    ],
)
def test_v509_gate_rejects_competing_abis(
    config_overrides: dict[str, int],
    projection_format: str,
    kwarg_overrides: dict[str, object],
    message: str,
) -> None:
    require = _execute_function(
        "_require_native_tk_d128_v509_e5m2_dout_runtime_contract"
    )
    with pytest.raises(ValueError, match=message):
        require(
            _config(**config_overrides),
            {"pv_format": "e4m3_fp8"},
            **_gate_kwargs(projection_format, **kwarg_overrides),
        )


def test_v509_selector_is_explicit_default_off_and_separate() -> None:
    init = _class_method("LowpAttentionRuntime", "__init__")
    defaults = {
        argument.arg: default
        for argument, default in zip(
            init.args.kwonlyargs,
            init.args.kw_defaults,
            strict=True,
        )
    }
    selector = defaults["native_tk_d128_v509_e5m2_dout_backward"]
    assert isinstance(selector, ast.Constant)
    assert selector.value is False
    source = RUNTIME.read_text()
    assert "NativeTkD128NVFP4ScoreE4M3QKVE5M2DoutBackward" in source
    assert "NATIVE_TK_D128_V509_E5M2_DOUT_BACKEND" in source
    assert "b300_require_v509_e5m2_dout_route(" in source
    assert '!= "v509_only_fail_closed"' not in source  # no weak string dispatch


def test_v509_shape_gate_is_self_contained_for_cpu_contract_extraction() -> None:
    require = _top_level_function(
        "_require_native_tk_d128_v509_e5m2_dout_runtime_contract"
    )
    loaded_names = {
        node.id
        for node in ast.walk(require)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }

    assert "AUTHENTICATED_D128_EXACT_BATCHES" not in loaded_names
    assert "SUPPORTED_LOWP_BATCHES" not in loaded_names
    assert "shape[0] not in (1, 2, 4)" in ast.unparse(require)


def test_v509_autograd_has_exact_publisher_and_no_fallback() -> None:
    backward = _class_method("_LowpAttentionFunction", "backward")
    selector_if = next(
        node
        for node in ast.walk(backward)
        if isinstance(node, ast.If)
        and "native_tk_d128_v509_e5m2_dout_backward" in ast.unparse(node.test)
        and "b300_project_dout_unified_lowp_nvfp4_v509_e5m2" in ast.unparse(node)
    )
    selected = "\n".join(ast.unparse(node) for node in selector_if.body)
    fallback = "\n".join(ast.unparse(node) for node in selector_if.orelse)
    assert "b300_project_dout_unified_lowp_nvfp4_v509_e5m2" in selected
    assert "b300_project_dout_unified_lowp_nvfp4(" not in selected
    assert "stats_workspace=runtime.backward.workspace_torch" in selected
    assert "dq_clear=runtime.backward.dq" in selected
    assert "dout_bundle.dout_backward_e5m2" in selected
    assert "b300_project_dout_unified_lowp_nvfp4(" in fallback
    source = ast.unparse(backward)
    assert "if runtime.native_tk_d128_native_score_backward else None" in source


def test_v509_contract_and_cli_are_truthful() -> None:
    source = RUNTIME.read_text()
    for expected in (
        '"native_tk_d128_v509_e5m2_dout_backward"',
        '"v509_e5m2_dout_route"',
        '"dout_backward_format": "e5m2"',
        '"projection_accumulator_e5m2_x4"',
        '"b300_project_dout_unified_lowp_nvfp4_v509_e5m2"',
    ):
        assert expected in source
    backward_contract = ast.unparse(
        _class_method("LowpAttentionRuntime", "backward_contract")
    )
    assert "self.backward.contract(fused_publisher_precleared_dq=True)" in (
        backward_contract
    )
    assert "else self.backward.contract()" in backward_contract
    main = ast.unparse(_top_level_function("main"))
    assert "'forward_topology': dict(runtime.forward_topology)" in main
    assert "'forward_topology': topology" not in main
    assert "--native-tk-d128-v509-e5m2-dout-backward" in main
    assert "action='store_true'" in main
    assert "exact-batch v509" in main
    assert "args.native_tk_d128_v509_e5m2_dout_backward" in main
    assert "args.native_tk_d128_native_score_backward" in main
