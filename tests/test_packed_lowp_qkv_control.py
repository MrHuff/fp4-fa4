from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
E2E = ROOT / "tk_fa4" / "lowp_fa4_bwd" / "benchmark_llama12b_e2e.py"
SATURATED = (
    ROOT / "tk_fa4" / "lowp_fa4_bwd" / "benchmark_llama12b_saturated.py"
)


def _function(path: Path, name: str, namespace: dict[str, Any]) -> Any:
    tree = ast.parse(path.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            function,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


def test_lowp_master_parameter_route_is_packed_only_for_d64() -> None:
    namespace = {
        "PACKED_D64_LOWP_QKV_LAYOUT": "packed",
        "SPLIT_D128_LOWP_QKV_LAYOUT": "split",
    }
    select = _function(
        E2E,
        "lowp_qkv_master_parameter_layout",
        namespace,
    )

    assert select(SimpleNamespace(head_dim=64)) == "packed"
    assert select(SimpleNamespace(head_dim=128)) == "split"
    with pytest.raises(ValueError, match="head_dim 64 or 128"):
        select(SimpleNamespace(head_dim=96))


def test_lowp_autograd_uses_packed_d64_and_preserves_d128_contract() -> None:
    source = E2E.read_text()
    function = source.split("class _LowpAttentionFunction", 1)[1].split(
        "class LowpAttention", 1
    )[0]
    lowp_module = source.split("class LowpAttention(nn.Module):", 1)[1].split(
        "class MLP", 1
    )[0]

    assert "packed_qkv_weight: torch.Tensor" in function
    assert function.count("qkv_weight = packed_qkv_weight") == 2
    assert function.count("_stack_lowp_qkv_weights(") == 2
    assert "dpacked_qkv_weight = qkv_weight_gradient" in function
    assert "dpacked_qkv_weight = None" in function
    assert "_deinterleave_d128_weight_gradient(" in function
    assert "qkv_weight = torch.cat" not in function
    assert "PackedQKVAttentionWeights(" in lowp_module
    assert "packed_qkv_weight = self.weights.qkv" in lowp_module
    assert "packed_qkv_weight = empty_weight" in lowp_module
    assert "self.weights.qkv," not in source.split(
        "class BF16Attention", 1
    )[1].split("class PackedQKVBF16Attention", 1)[0]

    tree = ast.parse(source)
    custom_function = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "_LowpAttentionFunction"
    )
    forward = next(
        node
        for node in custom_function.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    assert [argument.arg for argument in forward.args.args] == [
        "ctx",
        "x",
        "attention_norm_weight",
        "packed_qkv_weight",
        "q_weight",
        "k_weight",
        "v_weight",
        "out_weight",
        "qk_scales",
        "forward_workspace",
        "runtime",
    ]
    backward = next(
        node
        for node in custom_function.body
        if isinstance(node, ast.FunctionDef) and node.name == "backward"
    )
    result_assignment = next(
        node
        for node in ast.walk(backward)
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "result"
            and isinstance(node.value, ast.Tuple)
        )
    )
    assert [
        element.id if isinstance(element, ast.Name) else None
        for element in result_assignment.value.elts
    ] == [
        "dx",
        "dattention_norm_weight",
        "dpacked_qkv_weight",
        "dq_weight",
        "dk_weight",
        "dv_weight",
        "dout_weight",
        None,
        None,
        None,
    ]
    returned = next(
        node
        for node in ast.walk(backward)
        if isinstance(node, ast.Return)
    )
    assert isinstance(returned.value, ast.Name)
    assert returned.value.id == "result"


def test_saturated_schema_detection_covers_lowp_packed_routes() -> None:
    class PackedWeights:
        pass

    namespace: dict[str, Any] = {
        "PackedQKVAttentionWeights": PackedWeights,
    }
    uses_packed = _function(SATURATED, "_uses_packed_qkv", namespace)

    def model(*weights: object) -> SimpleNamespace:
        return SimpleNamespace(
            layers=[
                SimpleNamespace(
                    attention=SimpleNamespace(weights=value)
                )
                for value in weights
            ]
        )

    assert uses_packed(model(PackedWeights(), PackedWeights())) is True
    assert uses_packed(model(object(), object())) is False
    with pytest.raises(RuntimeError, match="mix packed and split"):
        uses_packed(model(PackedWeights(), object()))
    with pytest.raises(RuntimeError, match="requires decoder layers"):
        uses_packed(model())

    source = SATURATED.read_text()
    assert "_uses_packed_bf16_qkv" not in source
    assert source.count("if _uses_packed_qkv(model):") >= 5
    assert '"runtime_state_layout": (' in source
    assert '"qkv_parameter_layout": (' in source
    assert 'source_files.pop("packed_bf16_qkv")' in source
