from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import tk_fa4
import tk_fa4.interface as interface


ROOT = Path(__file__).resolve().parents[1]
E2E = ROOT / "tk_fa4" / "lowp_fa4_bwd" / "benchmark_llama12b_e2e.py"
CUDA = ROOT / "tk_fa4" / "lowp_fa4_bwd" / "lowp_fa4_bwd.cu"
SOURCE = E2E.read_text()
TREE = ast.parse(SOURCE)
OUTPUT_FIELDS = (
    "output_weight_forward_packed",
    "output_weight_forward_scales",
    "output_weight_backward_packed",
    "output_weight_backward_scales",
    "output_weight_global_scale",
)


def _arguments() -> tuple[torch.Tensor, ...]:
    weight = torch.zeros((128, 256), dtype=torch.bfloat16)
    return (
        weight,
        torch.zeros((128, 128), dtype=torch.uint8),
        torch.zeros((1, 4, 512), dtype=torch.uint8),
        torch.zeros((256, 64), dtype=torch.uint8),
        torch.zeros((2, 2, 512), dtype=torch.uint8),
        torch.zeros((1,), dtype=torch.float32),
    )


class _RecordingExtension:
    __file__ = "/tmp/recording-output-dual-extension.so"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def quantize_nvfp4_projection_weight_dual_out(
        self,
        *_arguments: torch.Tensor,
    ) -> None:
        self.calls.append("checked")

    def quantize_nvfp4_projection_weight_dual_out_unchecked(
        self,
        *_arguments: torch.Tensor,
    ) -> None:
        self.calls.append("unchecked")


def _function(name: str) -> ast.FunctionDef:
    for node in ast.walk(TREE):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function {name}")


def _function_text(name: str) -> str:
    return ast.unparse(_function(name))


def _execute_functions(
    names: tuple[str, ...],
    namespace: dict[str, Any],
) -> dict[str, Any]:
    future = ast.parse("from __future__ import annotations").body[0]
    module = ast.Module(
        body=[future, *(_function(name) for name in names)],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(E2E), "exec"), namespace)
    return namespace


def test_generic_dual_out_api_is_public_and_selects_both_symbols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert (
        tk_fa4.b300_prepare_nvfp4_projection_weight_dual_out
        is interface.b300_prepare_nvfp4_projection_weight_dual_out
    )
    extension = _RecordingExtension()
    monkeypatch.setattr(interface, "_C_b300_lowp_bwd", extension)
    arguments = _arguments()

    forward, backward = (
        interface.b300_prepare_nvfp4_projection_weight_dual_out(
            *arguments,
            checked=True,
        )
    )
    assert extension.calls == ["checked"]
    assert forward == (arguments[1], arguments[2], arguments[5])
    assert backward == (arguments[3], arguments[4], arguments[5])

    interface.b300_prepare_nvfp4_projection_weight_dual_out(
        *arguments,
        checked=False,
    )
    assert extension.calls == ["checked", "unchecked"]


def test_generic_dual_out_checked_contract_fails_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extension = _RecordingExtension()
    monkeypatch.setattr(interface, "_C_b300_lowp_bwd", extension)
    arguments = list(_arguments())
    arguments[3] = arguments[1]

    with pytest.raises(ValueError, match="must use disjoint storage"):
        interface.b300_prepare_nvfp4_projection_weight_dual_out(
            *arguments,
            checked=True,
        )
    assert extension.calls == []

    with pytest.raises(
        ValueError,
        match="bitwise authentication requires the checked path",
    ):
        interface.b300_prepare_nvfp4_projection_weight_dual_out(
            *_arguments(),
            checked=False,
            authenticate=True,
        )


def test_generic_dual_out_authenticates_w_and_physical_transpose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extension = _RecordingExtension()
    monkeypatch.setattr(interface, "_C_b300_lowp_bwd", extension)
    arguments = _arguments()
    references: list[torch.Tensor] = []

    def prepare(
        tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        references.append(tensor)
        if len(references) == 1:
            return arguments[1], arguments[2], arguments[5]
        return arguments[3], arguments[4], arguments[5]

    monkeypatch.setattr(
        interface,
        "b300_prepare_nvfp4_projection_weight",
        prepare,
    )
    interface.b300_prepare_nvfp4_projection_weight_dual_out(
        *arguments,
        checked=True,
        authenticate=True,
    )

    assert extension.calls == ["checked"]
    assert references[0] is arguments[0]
    assert torch.equal(references[1], arguments[0].T.contiguous())


def test_output_workspace_uses_checked_once_then_unchecked() -> None:
    calls: list[dict[str, object]] = []

    def direct_prepare(*arguments: object, **keywords: object):
        calls.append({"arguments": arguments, **keywords})
        return (
            (arguments[1], arguments[2], arguments[5]),
            (arguments[3], arguments[4], arguments[5]),
        )

    namespace = _execute_functions(
        (
            "_dual_output_weight_tensors",
            "_refresh_dual_weight_prep_authentication",
            "_prepare_direct_dual_output_weight",
        ),
        {
            "_DUAL_OUTPUT_WEIGHT_FIELDS": OUTPUT_FIELDS,
            "_tensor_abi_identity": id,
            "b300_prepare_nvfp4_projection_weight_dual_out": direct_prepare,
        },
    )
    destinations = [object() for _ in OUTPUT_FIELDS]
    workspace = SimpleNamespace(
        **dict(zip(OUTPUT_FIELDS, destinations, strict=True)),
        output_dual_weight_authenticated=False,
        output_dual_weight_abi_identity=None,
    )
    weight = object()

    forward, backward = namespace["_prepare_direct_dual_output_weight"](
        workspace,
        weight,
    )
    assert calls[0] == {
        "arguments": (weight, *destinations),
        "checked": True,
        "authenticate": True,
    }
    assert forward == (destinations[0], destinations[1], destinations[4])
    assert backward == (destinations[2], destinations[3], destinations[4])
    assert workspace.output_dual_weight_authenticated is True

    namespace["_prepare_direct_dual_output_weight"](workspace, weight)
    assert calls[1]["checked"] is False
    assert calls[1]["authenticate"] is False

    replacement_weight = object()
    namespace["_prepare_direct_dual_output_weight"](
        workspace,
        replacement_weight,
    )
    assert calls[2]["checked"] is True
    assert calls[2]["authenticate"] is True
    assert workspace.output_dual_weight_abi_identity[0] == id(
        replacement_weight
    )

    replacement_destination = object()
    workspace.output_weight_forward_packed = replacement_destination
    namespace["_prepare_direct_dual_output_weight"](
        workspace,
        replacement_weight,
    )
    assert calls[3]["checked"] is True
    assert calls[3]["authenticate"] is True
    assert workspace.output_dual_weight_abi_identity[1] == id(
        replacement_destination
    )


def test_failed_output_reauthentication_retains_prior_abi_identity() -> None:
    calls: list[dict[str, object]] = []

    def direct_prepare(*arguments: object, **keywords: object):
        calls.append(keywords)
        if len(calls) > 1:
            raise RuntimeError("re-authentication failed")
        return (
            (arguments[1], arguments[2], arguments[5]),
            (arguments[3], arguments[4], arguments[5]),
        )

    namespace = _execute_functions(
        (
            "_dual_output_weight_tensors",
            "_refresh_dual_weight_prep_authentication",
            "_prepare_direct_dual_output_weight",
        ),
        {
            "_DUAL_OUTPUT_WEIGHT_FIELDS": OUTPUT_FIELDS,
            "_tensor_abi_identity": id,
            "b300_prepare_nvfp4_projection_weight_dual_out": direct_prepare,
        },
    )
    workspace = SimpleNamespace(
        **dict(
            zip(OUTPUT_FIELDS, (object() for _ in OUTPUT_FIELDS), strict=True)
        ),
        output_dual_weight_authenticated=False,
        output_dual_weight_abi_identity=None,
    )
    prepare = namespace["_prepare_direct_dual_output_weight"]
    weight = object()
    prepare(workspace, weight)
    authenticated_identity = workspace.output_dual_weight_abi_identity

    replacement_weight = object()
    with pytest.raises(RuntimeError, match="re-authentication failed"):
        prepare(workspace, replacement_weight)
    assert workspace.output_dual_weight_authenticated is True
    assert workspace.output_dual_weight_abi_identity == authenticated_identity
    with pytest.raises(RuntimeError, match="re-authentication failed"):
        prepare(workspace, replacement_weight)
    assert calls[-1] == {"checked": True, "authenticate": True}


def test_true_2d_output_route_is_enabled_for_d64_and_d128() -> None:
    namespace = _execute_functions(
        ("_uses_direct_dual_output_weight_prep",),
        {},
    )
    eligible = namespace["_uses_direct_dual_output_weight_prep"]
    for head_dim in (64, 128):
        runtime = SimpleNamespace(
            config=SimpleNamespace(head_dim=head_dim),
            output_projection_format="nvfp4",
            projection_weight_scale_2d=True,
        )
        assert eligible(runtime) is True
    assert eligible(
        SimpleNamespace(
            output_projection_format="nvfp4",
            projection_weight_scale_2d=False,
        )
    ) is False


def test_runtime_removes_hot_output_pack_and_bf16_transpose_repack() -> None:
    forward = next(
        ast.unparse(node)
        for node in ast.walk(TREE)
        if isinstance(node, ast.ClassDef)
        and node.name == "_LowpAttentionFunction"
        for node in node.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    backward = next(
        ast.unparse(node)
        for node in ast.walk(TREE)
        if isinstance(node, ast.ClassDef)
        and node.name == "_LowpAttentionFunction"
        for node in node.body
        if isinstance(node, ast.FunctionDef) and node.name == "backward"
    )
    allocation = _function_text("_allocate_forward_workspace")

    assert "_prepare_direct_dual_output_weight(" in forward
    assert (
        "ctx.output_weight_backward_operand = "
        "out_weight_backward_operand"
    ) in forward
    assert "b300_prepare_nvfp4_projection_weight_dual(out_weight)" not in forward
    assert "*out_weight_backward_operand" not in forward
    assert (
        "out_weight_backward_operand = ctx.output_weight_backward_operand"
        in backward
    )
    assert "if out_weight_backward_operand is None" in backward
    assert backward.count("out_weight.T.contiguous()") == 1
    assert "if _uses_direct_dual_output_weight_prep(self.runtime)" in allocation
    assert (
        "output_weight_forward_packed = torch.empty(config.hidden, "
        "config.q_width // 2"
    ) in allocation
    assert (
        "output_weight_backward_packed = torch.empty(config.q_width, "
        "config.hidden // 2"
    ) in allocation


def test_native_source_exposes_generic_caller_owned_dual_contract() -> None:
    source = CUDA.read_text()
    implementation = source.split(
        "void quantize_nvfp4_projection_weight_dual_out_impl(",
        1,
    )[1].split(
        "void quantize_gqa_d128_qkv_projection_weight_dual_out_impl(",
        1,
    )[0]
    assert "check_nvfp4_dual_weight_outputs(" in implementation
    assert "launch_nvfp4_dual_weight_quantization(" in implementation
    assert "static_cast<int>(input.size(0) / 128)" in implementation
    assert "false" in implementation
    for symbol in (
        "quantize_nvfp4_projection_weight_dual_out",
        "quantize_nvfp4_projection_weight_dual_out_unchecked",
    ):
        assert source.count(f'"{symbol}"') == 1
