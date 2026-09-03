from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = (
    ROOT / "tk_fa4" / "lowp_fa4_bwd" / "benchmark_llama12b_e2e.py"
)
SOURCE = RUNTIME.read_text()
TREE = ast.parse(SOURCE)


def _top_level_class(name: str) -> ast.ClassDef:
    for node in TREE.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"missing top-level class {name}")


def _top_level_function(name: str) -> ast.FunctionDef:
    for node in TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing top-level function {name}")


def _execute_classes(*names: str) -> dict[str, Any]:
    module = ast.Module(
        body=[
            ast.parse("from __future__ import annotations").body[0],
            *(_top_level_class(name) for name in names),
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {
        "Any": Any,
        "dataclass": dataclass,
    }
    exec(compile(module, str(RUNTIME), "exec"), namespace)
    return namespace


def _method_text(class_name: str, method_name: str) -> str:
    class_node = _top_level_class(class_name)
    method = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    return ast.unparse(method)


def test_composite_authentication_tracks_both_direct_producers() -> None:
    function = _top_level_function(
        "_refresh_dual_weight_prep_authentication"
    )
    module = ast.Module(
        body=[
            ast.parse("from __future__ import annotations").body[0],
            function,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {}
    exec(compile(module, str(RUNTIME), "exec"), namespace)
    refresh = namespace["_refresh_dual_weight_prep_authentication"]
    workspace = SimpleNamespace(
        d128_dual_qkv_weight_authenticated=False,
        output_dual_weight_authenticated=False,
        weight_prep_authenticated=False,
    )

    assert refresh(workspace) is False
    workspace.d128_dual_qkv_weight_authenticated = True
    assert refresh(workspace) is False
    workspace.output_dual_weight_authenticated = True
    assert refresh(workspace) is True
    assert workspace.weight_prep_authenticated is True

    qkv_prepare = ast.unparse(
        _top_level_function("_prepare_direct_d128_dual_qkv_weight")
    )
    output_prepare = ast.unparse(
        _top_level_function("_prepare_direct_dual_output_weight")
    )
    assert "_refresh_dual_weight_prep_authentication(workspace)" in (
        qkv_prepare
    )
    assert "_refresh_dual_weight_prep_authentication(workspace)" in (
        output_prepare
    )


def test_publication_state_accepts_training_and_no_grad_lifecycles() -> None:
    state_type = _execute_classes("_DualWeightPackPublicationState")[
        "_DualWeightPackPublicationState"
    ]
    state = state_type()
    versions0 = (3, 5, 7, 11)
    state.begin(0, versions0)
    state.publish_qkv(0, versions0)
    state.publish_output(0, versions0)
    state.consume_qkv(0, versions0)
    state.consume_output(0, versions0)
    state.enqueue_backward(0, versions0)

    versions1 = (4, 6, 8, 12)
    state.begin(1, versions1)
    state.publish_qkv(1, versions1)
    state.publish_output(1, versions1)
    state.consume_qkv(1, versions1)
    state.consume_output(1, versions1)
    state.release_without_backward(1, versions1)
    assert state.backward_enqueued is True


@pytest.mark.parametrize(
    ("operation", "match"),
    (
        ("overwrite", "before its backward consumer"),
        ("stale_generation", "stale dual-weight generation"),
        ("stale_version", "stale dual-weight parameter versions"),
        ("output_before_qkv", "before QKV"),
        ("consume_before_publish", "unpublished"),
    ),
)
def test_publication_state_fails_closed(
    operation: str,
    match: str,
) -> None:
    state_type = _execute_classes("_DualWeightPackPublicationState")[
        "_DualWeightPackPublicationState"
    ]
    state = state_type()
    versions = (13, 17, 19, 23)
    state.begin(0, versions)

    with pytest.raises(RuntimeError, match=match):
        if operation == "overwrite":
            state.begin(1, (14, 18, 20, 24))
        elif operation == "stale_generation":
            state.publish_qkv(1, versions)
        elif operation == "stale_version":
            state.publish_qkv(0, (13, 17, 19, 24))
        elif operation == "output_before_qkv":
            state.publish_output(0, versions)
        else:
            state.consume_qkv(0, versions)


def test_controller_no_grad_release_records_reuse_fence() -> None:
    namespace = _execute_classes(
        "_DualWeightPackPublicationState",
        "_DualWeightPackLayerController",
    )
    state = namespace["_DualWeightPackPublicationState"]()
    versions = (29, 31, 37, 41)
    state.begin(3, versions)
    state.publish_qkv(3, versions)
    state.publish_output(3, versions)
    state.consume_qkv(3, versions)
    state.consume_output(3, versions)

    consumer_stream = object()

    class Event:
        def __init__(self) -> None:
            self.recorded_streams: list[object] = []

        def record(self, stream: object) -> None:
            self.recorded_streams.append(stream)

    event = Event()
    controller_type = namespace["_DualWeightPackLayerController"]
    controller = controller_type.__new__(controller_type)
    controller.state = state
    controller.backward_consumed_event = event
    controller._require_consumer_stream = lambda: consumer_stream

    controller.release_without_backward(
        generation=3,
        weight_versions=versions,
    )

    assert event.recorded_streams == [consumer_stream]
    assert state.backward_enqueued is True


def test_controller_composite_authentication_is_fail_closed() -> None:
    controller_type = _execute_classes(
        "_DualWeightPackPublicationState",
        "_DualWeightPackLayerController",
    )["_DualWeightPackLayerController"]
    authenticate = _method_text(
        "_DualWeightPackLayerController",
        "authenticate",
    )
    clear = authenticate.index(
        "self.workspace.weight_prep_authenticated = False"
    )
    clear_qkv = authenticate.index(
        "self.workspace.d128_dual_qkv_weight_authenticated = False"
    )
    clear_output = authenticate.index(
        "self.workspace.output_dual_weight_authenticated = False"
    )
    qkv = authenticate.index("_prepare_direct_d128_dual_qkv_weight(")
    output = authenticate.index("_prepare_direct_dual_output_weight(")
    validate = authenticate.index("self._authenticated_abi_matches(weights)")
    publish = authenticate.index(
        "self.workspace.weight_prep_authenticated = authenticated"
    )
    assert clear < clear_qkv < clear_output < qkv < output < validate < publish
    abi_match = _method_text(
        "_DualWeightPackLayerController",
        "_authenticated_abi_matches",
    )
    for guard in (
        "self.workspace.d128_dual_qkv_weight_authenticated",
        "self.workspace.output_dual_weight_authenticated",
        "self.workspace.d128_dual_qkv_weight_abi_identity == qkv_identity",
        "self.workspace.output_dual_weight_abi_identity == output_identity",
    ):
        assert guard in abi_match

    controller_methods = {
        node.name
        for node in _top_level_class(
            "_DualWeightPackLayerController"
        ).body
        if isinstance(node, ast.FunctionDef)
    }
    assert {
        "authenticate",
        "begin",
        "enqueue",
        "consume_qkv",
        "consume_output",
        "enqueue_backward_consumed",
        "release_without_backward",
        "require_can_begin",
        "require_bound_consumer_stream",
    } <= controller_methods
    assert controller_type is not None


def test_failed_output_authentication_cannot_leak_prior_composite() -> None:
    namespace = _execute_classes(
        "_DualWeightPackPublicationState",
        "_DualWeightPackLayerController",
    )
    controller_type = namespace["_DualWeightPackLayerController"]
    workspace = SimpleNamespace(
        weight_prep_authenticated=True,
        d128_dual_qkv_weight_authenticated=True,
        output_dual_weight_authenticated=True,
    )
    controller = controller_type.__new__(controller_type)
    controller.workspace = workspace
    controller.state = namespace["_DualWeightPackPublicationState"]()
    controller._require_consumer_stream = lambda: object()
    controller._require_bound_objects = lambda _weights: None

    def qkv_prepare(*_arguments: object) -> None:
        assert workspace.output_dual_weight_authenticated is False
        workspace.d128_dual_qkv_weight_authenticated = True
        workspace.weight_prep_authenticated = bool(
            workspace.d128_dual_qkv_weight_authenticated
            and workspace.output_dual_weight_authenticated
        )

    def output_prepare(*_arguments: object) -> None:
        raise RuntimeError("output checked authentication failed")

    namespace["_prepare_direct_d128_dual_qkv_weight"] = qkv_prepare
    namespace["_prepare_direct_dual_output_weight"] = output_prepare

    with pytest.raises(RuntimeError, match="output checked authentication"):
        controller.authenticate((object(), object(), object(), object()))

    assert workspace.d128_dual_qkv_weight_authenticated is True
    assert workspace.output_dual_weight_authenticated is False
    assert workspace.weight_prep_authenticated is False


def test_controller_preserves_split_events_versions_and_abi_guards() -> None:
    enqueue = _method_text("_DualWeightPackLayerController", "enqueue")
    consume_qkv = _method_text(
        "_DualWeightPackLayerController",
        "consume_qkv",
    )
    consume_output = _method_text(
        "_DualWeightPackLayerController",
        "consume_output",
    )
    backward = _method_text(
        "_DualWeightPackLayerController",
        "enqueue_backward_consumed",
    )

    assert "self._authenticated_abi_matches(weights)" not in enqueue
    assert "self._require_bound_objects(weights)" in enqueue
    assert "self.workspace.d128_dual_qkv_weight_authenticated" in enqueue
    assert "self.workspace.output_dual_weight_authenticated" in enqueue
    assert "_attention_weight_versions(weights)" in enqueue
    assert "checked=False" in enqueue
    assert "authenticate=False" in enqueue
    assert "self.qkv_ready_event.record(self.producer_stream)" in enqueue
    assert "self.output_ready_event.record(self.producer_stream)" in enqueue
    assert "self.producer_stream.wait_event(self.backward_consumed_event)" in (
        enqueue
    )
    assert "consumer_stream.wait_event(self.qkv_ready_event)" in consume_qkv
    assert (
        "consumer_stream.wait_event(self.output_ready_event)"
        in consume_output
    )
    assert "self.backward_consumed_event.record(consumer_stream)" in backward


def test_custom_autograd_consumes_rolling_operands_and_fences_backward() -> None:
    forward = _method_text("_LowpAttentionFunction", "forward")
    backward = _method_text("_LowpAttentionFunction", "backward")

    qkv_consume = forward.index("dual_weight_pack_controller.consume_qkv(")
    qkv_projection = forward.index("qkv = runtime.qkv_projection(")
    output_consume = forward.index(
        "dual_weight_pack_controller.consume_output("
    )
    output_projection = forward.index("projected = b300_project_nvfp4(")
    save_provenance = forward.index(
        "ctx.dual_weight_pack_controller = dual_weight_pack_controller"
    )
    assert qkv_consume < qkv_projection < output_consume < output_projection
    assert output_projection < save_provenance

    output_weight_gradient = backward.index(
        "with _stage('lowp/bwd/output_weight_gradient')"
    )
    rolling_fence = backward.index(
        "dual_weight_pack_controller.enqueue_backward_consumed("
    )
    workspace_release = backward.index(
        "forward_workspace.publication_state.finish_backward("
    )
    assert output_weight_gradient < rolling_fence < workspace_release


def test_workspace_exposes_lbt_composite_and_controller_slots() -> None:
    workspace = _top_level_class("_LowpAttentionForwardWorkspace")
    annotations = {
        target.target.id
        for target in workspace.body
        if isinstance(target, ast.AnnAssign)
        and isinstance(target.target, ast.Name)
    }
    assert "weight_prep_authenticated" in annotations
    assert "dual_weight_pack_controller" in annotations

    contract = _method_text("LowpAttention", "forward_workspace_contract")
    assert "rolling_private_stream" in contract
    assert "synchronous_same_stream" in contract
    assert "rolling_controller_attached" in contract
    assert "composite_weight_prep_authenticated" in contract
