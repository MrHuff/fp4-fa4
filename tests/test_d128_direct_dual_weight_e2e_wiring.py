from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
E2E = ROOT / "tk_fa4" / "lowp_fa4_bwd" / "benchmark_llama12b_e2e.py"
SOURCE = E2E.read_text()
TREE = ast.parse(SOURCE)
DUAL_FIELDS = (
    "qkv_weight_forward_packed",
    "qkv_weight_forward_scales",
    "qkv_weight_backward_packed",
    "qkv_weight_backward_scales",
    "qkv_weight_global_scale",
)


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


def test_direct_dual_eligibility_is_fail_closed_to_d128_nvfp4_2d_dgrad() -> None:
    namespace = _execute_functions(
        ("_uses_direct_d128_dual_qkv_weight_prep",),
        {},
    )
    eligible = namespace["_uses_direct_d128_dual_qkv_weight_prep"]

    def runtime(**overrides: object) -> SimpleNamespace:
        values = {
            "config": SimpleNamespace(head_dim=128),
            "qkv_projection_format": "nvfp4",
            "projection_weight_scale_2d": True,
            "projection_dgrad": "nvfp4",
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    assert eligible(runtime()) is True
    assert eligible(runtime(config=SimpleNamespace(head_dim=64))) is False
    assert eligible(runtime(qkv_projection_format="e4m3")) is False
    assert eligible(runtime(projection_weight_scale_2d=False)) is False
    assert eligible(runtime(projection_dgrad="bf16")) is False


def test_synchronous_workspace_uses_checked_once_then_unchecked() -> None:
    calls: list[dict[str, object]] = []

    def direct_prepare(*arguments: object, **keywords: object):
        calls.append({"arguments": arguments, **keywords})
        return (
            (arguments[3], arguments[4], arguments[7]),
            (arguments[5], arguments[6], arguments[7]),
        )

    namespace = _execute_functions(
        (
            "_d128_dual_qkv_weight_tensors",
            "_refresh_dual_weight_prep_authentication",
            "_prepare_direct_d128_dual_qkv_weight",
        ),
        {
            "_D128_DUAL_QKV_WEIGHT_FIELDS": DUAL_FIELDS,
            "_tensor_abi_identity": id,
            "b300_prepare_gqa_d128_qkv_projection_weight_dual_out": (
                direct_prepare
            ),
        },
    )
    prepare = namespace["_prepare_direct_d128_dual_qkv_weight"]
    destinations = [object() for _ in DUAL_FIELDS]
    workspace = SimpleNamespace(
        **dict(zip(DUAL_FIELDS, destinations, strict=True)),
        d128_dual_qkv_weight_authenticated=False,
        d128_dual_qkv_weight_abi_identity=None,
    )
    q_weight, k_weight, v_weight = object(), object(), object()

    forward, backward = prepare(
        workspace,
        q_weight,
        k_weight,
        v_weight,
    )
    assert calls[0]["checked"] is True
    assert calls[0]["authenticate"] is True
    assert calls[0]["arguments"][:3] == (
        q_weight,
        k_weight,
        v_weight,
    )
    assert calls[0]["arguments"][3:] == tuple(destinations)
    assert forward == (destinations[0], destinations[1], destinations[4])
    assert backward == (destinations[2], destinations[3], destinations[4])
    assert workspace.d128_dual_qkv_weight_authenticated is True

    prepare(workspace, q_weight, k_weight, v_weight)
    assert calls[1]["checked"] is False
    assert calls[1]["authenticate"] is False
    assert calls[1]["arguments"][3:] == tuple(destinations)

    replacement_q_weight = object()
    prepare(workspace, replacement_q_weight, k_weight, v_weight)
    assert calls[2]["checked"] is True
    assert calls[2]["authenticate"] is True
    assert workspace.d128_dual_qkv_weight_abi_identity[0] == id(
        replacement_q_weight
    )

    replacement_destination = object()
    workspace.qkv_weight_forward_packed = replacement_destination
    prepare(workspace, replacement_q_weight, k_weight, v_weight)
    assert calls[3]["checked"] is True
    assert calls[3]["authenticate"] is True
    assert workspace.d128_dual_qkv_weight_abi_identity[3] == id(
        replacement_destination
    )


def test_failed_first_use_does_not_authenticate_workspace() -> None:
    def fail(*arguments: object, **keywords: object) -> None:
        raise RuntimeError("native preflight failed")

    namespace = _execute_functions(
        (
            "_d128_dual_qkv_weight_tensors",
            "_refresh_dual_weight_prep_authentication",
            "_prepare_direct_d128_dual_qkv_weight",
        ),
        {
            "_D128_DUAL_QKV_WEIGHT_FIELDS": DUAL_FIELDS,
            "_tensor_abi_identity": id,
            "b300_prepare_gqa_d128_qkv_projection_weight_dual_out": fail,
        },
    )
    workspace = SimpleNamespace(
        **dict(
            zip(DUAL_FIELDS, (object() for _ in DUAL_FIELDS), strict=True)
        ),
        d128_dual_qkv_weight_authenticated=False,
        d128_dual_qkv_weight_abi_identity=None,
    )
    with pytest.raises(RuntimeError, match="native preflight failed"):
        namespace["_prepare_direct_d128_dual_qkv_weight"](
            workspace,
            object(),
            object(),
            object(),
        )
    assert workspace.d128_dual_qkv_weight_authenticated is False


def test_failed_qkv_reauthentication_retains_prior_abi_identity() -> None:
    calls: list[dict[str, object]] = []

    def direct_prepare(*arguments: object, **keywords: object):
        calls.append(keywords)
        if len(calls) > 1:
            raise RuntimeError("re-authentication failed")
        return (
            (arguments[3], arguments[4], arguments[7]),
            (arguments[5], arguments[6], arguments[7]),
        )

    namespace = _execute_functions(
        (
            "_d128_dual_qkv_weight_tensors",
            "_refresh_dual_weight_prep_authentication",
            "_prepare_direct_d128_dual_qkv_weight",
        ),
        {
            "_D128_DUAL_QKV_WEIGHT_FIELDS": DUAL_FIELDS,
            "_tensor_abi_identity": id,
            "b300_prepare_gqa_d128_qkv_projection_weight_dual_out": (
                direct_prepare
            ),
        },
    )
    workspace = SimpleNamespace(
        **dict(
            zip(DUAL_FIELDS, (object() for _ in DUAL_FIELDS), strict=True)
        ),
        d128_dual_qkv_weight_authenticated=False,
        d128_dual_qkv_weight_abi_identity=None,
    )
    prepare = namespace["_prepare_direct_d128_dual_qkv_weight"]
    q_weight, k_weight, v_weight = object(), object(), object()
    prepare(workspace, q_weight, k_weight, v_weight)
    authenticated_identity = workspace.d128_dual_qkv_weight_abi_identity

    replacement_q = object()
    with pytest.raises(RuntimeError, match="re-authentication failed"):
        prepare(workspace, replacement_q, k_weight, v_weight)
    assert workspace.d128_dual_qkv_weight_authenticated is True
    assert workspace.d128_dual_qkv_weight_abi_identity == authenticated_identity
    with pytest.raises(RuntimeError, match="re-authentication failed"):
        prepare(workspace, replacement_q, k_weight, v_weight)
    assert calls[-1] == {"checked": True, "authenticate": True}


def test_partial_direct_workspace_fails_closed() -> None:
    namespace = _execute_functions(
        ("_d128_dual_qkv_weight_tensors",),
        {"_D128_DUAL_QKV_WEIGHT_FIELDS": DUAL_FIELDS},
    )
    workspace = SimpleNamespace(
        **{
            name: (object() if index == 0 else None)
            for index, name in enumerate(DUAL_FIELDS)
        }
    )
    with pytest.raises(RuntimeError, match="fully allocated or absent"):
        namespace["_d128_dual_qkv_weight_tensors"](workspace)


def test_private_workspace_allocates_exact_forward_and_transpose_geometry() -> None:
    allocation = _function_text("_allocate_forward_workspace")
    assert "if _uses_direct_d128_dual_qkv_weight_prep(self.runtime)" in allocation
    assert "qkv_rows = config.q_width + 2 * config.kv_width" in allocation
    assert (
        "qkv_weight_forward_packed = torch.empty(qkv_rows, "
        "config.hidden // 2"
    ) in allocation
    assert (
        "qkv_weight_backward_packed = torch.empty(config.hidden, "
        "qkv_rows // 2"
    ) in allocation
    assert (
        "qkv_weight_forward_scales = torch.empty(qkv_rows // 128, "
        "config.hidden // 64, 512"
    ) in allocation
    assert (
        "qkv_weight_backward_scales = torch.empty(config.hidden // 128, "
        "qkv_rows // 64, 512"
    ) in allocation
    assert "dtype=torch.float4_e2m1fn_x2" in allocation
    assert "dtype=torch.float8_e4m3fn" in allocation
    assert "qkv_weight_global_scale = torch.empty(1" in allocation
    assert "qkv_weight_forward_packed = None" in allocation


def test_forward_consumes_direct_operand_and_retains_matching_transpose() -> None:
    # Select the custom-autograd forward rather than the module wrapper.
    custom_forward = next(
        ast.unparse(node)
        for node in ast.walk(TREE)
        if isinstance(node, ast.ClassDef)
        and node.name == "_LowpAttentionFunction"
        for node in node.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    assert (
        "use_direct_d128_dual_qkv_weight = "
        "_uses_direct_d128_dual_qkv_weight_prep(runtime)"
    ) in custom_forward
    assert (
        "elif use_direct_d128_dual_qkv_weight:\n"
        "        qkv_weight = forward_workspace.outputs.empty_bf16"
    ) in custom_forward
    assert "_prepare_direct_d128_dual_qkv_weight(" in custom_forward
    assert (
        "qkv = runtime.qkv_projection(rows_operand, qkv_weight_operand"
    ) in custom_forward
    assert (
        "ctx.qkv_weight_backward_operand = qkv_weight_backward_operand"
    ) in custom_forward


def test_backward_consumes_saved_transpose_before_composition_fallback() -> None:
    backward = next(
        ast.unparse(node)
        for node in ast.walk(TREE)
        if isinstance(node, ast.ClassDef)
        and node.name == "_LowpAttentionFunction"
        for node in node.body
        if isinstance(node, ast.FunctionDef) and node.name == "backward"
    )
    saved = backward.index(
        "projection_weight_operand = ctx.qkv_weight_backward_operand"
    )
    guarded_concat = backward.index("elif projection_weight_operand is None")
    fallback_pack = backward.index("prepare_weight(qkv_weight.T.contiguous())")
    consume = backward.index(
        "dx_scaled = project_dgrad(",
        fallback_pack,
    )
    assert saved < guarded_concat < fallback_pack < consume
    assert "qkv_weight = None" in backward
    assert "assert qkv_weight is not None" in backward


def test_workspace_contract_reports_stable_private_synchronous_owners() -> None:
    contract = _function_text("forward_workspace_contract")
    assert "'d128_dual_qkv_weight'" in contract
    assert "weight_pack_schedule" in contract
    assert "'rolling_private_stream'" in contract
    assert "'synchronous_same_stream'" in contract
    assert "'rolling_controller_attached'" in contract
    assert "'composite_weight_prep_authenticated'" in contract
    assert "'one_forward_in_flight_per_layer': True" in contract
    assert (
        "'authenticated': workspace.d128_dual_qkv_weight_authenticated"
        in contract
    )
    assert "'all_pointers_stable_since_allocation'" in contract
    assert "'all_pointers_unique'" in contract
    assert "'quantize_gqa_d128_qkv_projection_weight_dual_out'" in contract
    assert (
        "'quantize_gqa_d128_qkv_projection_weight_dual_out_unchecked'"
        in contract
    )


def test_runtime_wires_direct_api_through_optional_rolling_controller() -> None:
    assert "b300_prepare_gqa_d128_qkv_projection_weight_dual_out," in SOURCE
    assert "class _DualWeightPackLayerController:" in SOURCE
    assert "dual_weight_pack_controller.consume_qkv(" in SOURCE
    assert "dual_weight_pack_controller.consume_output(" in SOURCE
    assert "dual_weight_pack_controller.enqueue_backward_consumed(" in SOURCE
    # The source-owned controller is optional. Standalone/local callers retain
    # the authenticated synchronous preparation path.
    assert "_prepare_direct_d128_dual_qkv_weight(" in SOURCE
