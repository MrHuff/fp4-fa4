from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = (
    ROOT / "tk_fa4" / "lowp_fa4_bwd" / "benchmark_llama12b_e2e.py"
)


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


def _gate_kwargs(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "qkv_projection_format": "e4m3",
        "output_projection_format": "e4m3",
        "native_tk_d128_native_score_backward": False,
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
    ("projection_format", "pv_format"),
    [
        ("e4m3", "e4m3_fp8"),
        ("e4m3", "mxfp4_e8m0_block32"),
        ("nvfp4", "e4m3_fp8"),
        ("nvfp4", "mxfp4_e8m0_block32"),
    ],
)
def test_v510_gate_admits_all_four_learned_projection_pv_arms(
    projection_format: str,
    pv_format: str,
) -> None:
    require = _execute_function(
        "_require_native_tk_d128_v510_e5m2_dout_runtime_contract"
    )
    require(
        _config(),
        {"pv_format": pv_format},
        **_gate_kwargs(
            qkv_projection_format=projection_format,
            output_projection_format=projection_format,
        ),
    )
    require(
        _config(),
        {"pv_format": pv_format},
        **_gate_kwargs(
            qkv_projection_format=projection_format,
            output_projection_format=projection_format,
            shared_runtime=SimpleNamespace(
                native_tk_d128_v510_e5m2_dout_backward=True
            ),
        ),
    )


@pytest.mark.parametrize(
    ("config_overrides", "kwarg_overrides", "message"),
    [
        ({"batch": 2}, {}, "B1"),
        ({"sequence": 2048}, {}, "S4096"),
        ({}, {"output_projection_format": "nvfp4"}, "learned projections"),
        ({}, {"native_tk_d128_native_score_backward": True}, "dense E4M3"),
        ({}, {"backward_match_forward_operands": True}, "E4M3 Q/K"),
        ({}, {"per_block_qk_scales": False}, "row-by-K16"),
        ({}, {"experimental_split_v_backward": True}, "retained"),
        ({}, {"experimental_output_shared_split_v": True}, "disable"),
        ({}, {"experimental_output_shared_split_v": None}, "disable"),
        ({}, {"experimental_d128_mxfp4_v_backward": True}, "no MXFP4"),
        ({}, {"v_mxfp4_scale_2d": True}, "no MXFP4"),
        (
            {},
            {
                "shared_runtime": SimpleNamespace(
                    native_tk_d128_v510_e5m2_dout_backward=False
                )
            },
            "shared v510",
        ),
    ],
)
def test_v510_gate_rejects_every_competing_operand_abi(
    config_overrides: dict[str, int],
    kwarg_overrides: dict[str, object],
    message: str,
) -> None:
    require = _execute_function(
        "_require_native_tk_d128_v510_e5m2_dout_runtime_contract"
    )
    with pytest.raises(ValueError, match=message):
        require(
            _config(**config_overrides),
            {"pv_format": "e4m3_fp8"},
            **_gate_kwargs(**kwarg_overrides),
        )


def test_v510_selector_is_separate_default_false_and_authenticates_pair() -> None:
    init = _class_method("LowpAttentionRuntime", "__init__")
    keyword_defaults = {
        argument.arg: default
        for argument, default in zip(
            init.args.kwonlyargs,
            init.args.kw_defaults,
            strict=True,
        )
    }
    selector_default = keyword_defaults[
        "native_tk_d128_v510_e5m2_dout_backward"
    ]
    assert isinstance(selector_default, ast.Constant)
    assert selector_default.value is False

    source = RUNTIME.read_text()
    assert "NativeTkD128DenseE4M3ScoreQKVE5M2DoutBackward" in source
    assert "NATIVE_TK_D128_V510_E5M2_DOUT_BACKEND" in source
    assert "b300_require_v510_e5m2_dout_route(" in source
    assert "self.backward.extension_metadata" in source
    assert '!= "v510_only_fail_closed"' in source
    assert "v508 native-score and v510 dense-score" in source


def test_v510_autograd_uses_only_e5_publisher_in_selected_branch() -> None:
    backward = _class_method("_LowpAttentionFunction", "backward")
    selector_if = next(
        node
        for node in ast.walk(backward)
        if isinstance(node, ast.If)
        and "native_tk_d128_v510_e5m2_dout_backward"
        in ast.unparse(node.test)
        and "b300_project_dout_unified_lowp_nvfp4_v510_e5m2"
        in ast.unparse(node)
    )
    selected = "\n".join(ast.unparse(node) for node in selector_if.body)
    ordinary = "\n".join(ast.unparse(node) for node in selector_if.orelse)
    assert "b300_project_dout_unified_lowp_nvfp4_v510_e5m2" in selected
    assert "b300_project_dout_unified_lowp_nvfp4(" not in selected
    assert "stats_workspace=runtime.backward.workspace_torch" in selected
    assert "dq_clear=runtime.backward.dq" in selected
    assert "dout_bundle.dout_backward_e5m2" in selected
    assert "b300_project_dout_unified_lowp_nvfp4(" in ordinary
    assert "dout_bundle.dout_backward_fp8" in ordinary

    backward_source = ast.unparse(backward)
    assert (
        "runtime.bind_backward_inputs(q_fp8, k_fp8, v_backward, "
        "dout_backward"
    ) in backward_source
    assert (
        "if runtime.native_tk_d128_native_score_backward else None"
        in backward_source
    )


def test_v510_topology_and_contract_report_split_precision_truthfully() -> None:
    source = RUNTIME.read_text()
    for expected in (
        '"backward_attention_dout_format": "e5m2"',
        '"backward_attention_dout_source": (\n'
        '                        "projection_accumulator_e5m2_x4"',
        '"dout_backward_format": "e5m2"',
        '"dout_backward_source": (',
        '"dout_backward_kernel": (',
        '"native_tk_d128_v510_e5m2_dout_backward": True',
        '"v510_e5m2_dout_route": dict(',
    ):
        assert expected in source
    assert (
        '"b300_project_dout_unified_lowp_nvfp4_v510_e5m2"'
        in source
    )
    assert (
        '"projection_accumulator_e4m3"'
        in source
    )


def test_v510_cli_is_explicit_fail_closed_and_persisted() -> None:
    main = ast.unparse(_top_level_function("main"))
    assert "--native-tk-d128-v510-e5m2-dout-backward" in main
    assert "action='store_true'" in main
    assert (
        "args.native_tk_d128_v510_e5m2_dout_backward and (not "
        "native_tk_d128_backward)"
    ) in main
    assert (
        "args.native_tk_d128_native_score_backward and "
        "args.native_tk_d128_v510_e5m2_dout_backward"
    ) in main
    assert (
        "native_tk_d128_v510_e5m2_dout_backward="
        "args.native_tk_d128_v510_e5m2_dout_backward"
    ) in main
    assert "'v510_e5m2_dout_route': runtime.v510_e5m2_dout_route" in main
