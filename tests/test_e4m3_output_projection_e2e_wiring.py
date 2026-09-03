from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "tk_fa4" / "lowp_fa4_bwd" / "benchmark_llama12b_e2e.py"
SOURCE = RUNTIME.read_text()
TREE = ast.parse(SOURCE)


def _top_level_node(name: str) -> ast.stmt:
    for node in TREE.body:
        if getattr(node, "name", None) == name:
            return node
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return node
    raise AssertionError(f"missing top-level node {name}")


def _namespace(*names: str) -> dict[str, Any]:
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            *(_top_level_node(name) for name in names),
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {}
    exec(compile(module, str(RUNTIME), "exec"), namespace)
    return namespace


def _d128_config(**overrides: object) -> SimpleNamespace:
    values = {
        "batch": 2,
        "sequence": 4096,
        "hidden": 4096,
        "q_heads": 32,
        "kv_heads": 8,
        "head_dim": 128,
        "q_width": 4096,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_e4m3_output_contract_is_exact_d128_and_asymmetric() -> None:
    require = _namespace(
        "AUTHENTICATED_D128_EXACT_BATCHES",
        "AUTHENTICATED_D64_EXACT_BATCHES",
        "SUPPORTED_LOWP_BATCHES",
        "_require_output_projection_contract",
    )["_require_output_projection_contract"]

    for batch in (1, 2):
        require(
            _d128_config(batch=batch),
            qkv_projection_format="e4m3",
            output_projection_format="e4m3",
            projection_dgrad="nvfp4",
            projection_weight_scale_2d=True,
        )
    require(
        SimpleNamespace(
            batch=16,
            sequence=4096,
            hidden=2048,
            q_heads=32,
            kv_heads=8,
            head_dim=64,
            q_width=2048,
        ),
        qkv_projection_format="e4m3",
        output_projection_format="e4m3",
        projection_dgrad="bf16",
        projection_weight_scale_2d=True,
    )

    with pytest.raises(ValueError, match="requires an E4M3 forward"):
        require(
            _d128_config(),
            qkv_projection_format="e4m3",
            output_projection_format="nvfp4",
            projection_dgrad="nvfp4",
            projection_weight_scale_2d=True,
        )

    invalid_cases = (
        {"config": _d128_config(head_dim=64)},
        {"qkv_projection_format": "nvfp4"},
        {"projection_dgrad": "bf16"},
        {"projection_weight_scale_2d": False},
    )
    for overrides in invalid_cases:
        arguments = {
            "config": _d128_config(),
            "qkv_projection_format": "e4m3",
            "output_projection_format": "e4m3",
            "projection_dgrad": "nvfp4",
            "projection_weight_scale_2d": True,
            **overrides,
        }
        config = arguments.pop("config")
        with pytest.raises(ValueError, match="correctness canary"):
            require(config, **arguments)


def test_dual_nvfp4_output_workspace_is_not_allocated_for_e4m3() -> None:
    select = _namespace("_uses_direct_dual_output_weight_prep")[
        "_uses_direct_dual_output_weight_prep"
    ]

    assert select(
        SimpleNamespace(
            output_projection_format="nvfp4",
            projection_weight_scale_2d=True,
        )
    )
    assert not select(
        SimpleNamespace(
            output_projection_format="e4m3",
            projection_weight_scale_2d=True,
        )
    )


def test_autograd_forward_publishes_only_required_o_operands() -> None:
    function = next(
        node
        for node in TREE.body
        if isinstance(node, ast.ClassDef)
        and node.name == "_LowpAttentionFunction"
    )
    forward = next(
        node
        for node in function.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    source = ast.unparse(forward)

    assert "runtime.output_projection_format == 'e4m3'" in source
    assert "b300_prepare_e4m3_projection_operand(output_matrix)" in source
    assert "b300_prepare_e4m3_projection_weight(out_weight)" in source
    assert "b300_project_e4m3(output_operand, out_weight_operand)" in source
    assert (
        "b300_prepare_nvfp4_projection_weight(out_weight.T.contiguous())"
        in source
    )
    assert "No unused NVFP4 forward-O" in SOURCE


def test_runtime_contract_names_nonfinal_forward_and_retained_nvfp4_backward() -> None:
    runtime = next(
        node
        for node in TREE.body
        if isinstance(node, ast.ClassDef) and node.name == "LowpAttentionRuntime"
    )
    source = ast.unparse(runtime)

    for marker in (
        "allocating_generic_correctness_canary_nonfinal_speed",
        "b300_project_e4m3",
        "b300_project_dout_unified_lowp_nvfp4",
        "e4m3_backward_learned_projection_gemms",
        "asymmetric_forward_input_gradient",
        "unused_nvfp4_forward_weight_publication",
    ):
        assert marker in source


def test_benchmark_captures_projection_authentication_before_workspace_gc() -> None:
    benchmark = _top_level_node("_benchmark_route")
    assert isinstance(benchmark, ast.FunctionDef)

    capture_line = next(
        node.lineno
        for node in ast.walk(benchmark)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "forward_dispatch"
            for target in node.targets
        )
        and isinstance(node.value, ast.Dict)
    )
    release_line = next(
        node.lineno
        for node in ast.walk(benchmark)
        if isinstance(node, ast.Delete)
        and any(
            isinstance(target, ast.Name) and target.id == "model"
            for target in node.targets
        )
    )

    assert capture_line < release_line
    source = ast.unparse(benchmark)
    assert "captured_before_model_workspace_release" in source
