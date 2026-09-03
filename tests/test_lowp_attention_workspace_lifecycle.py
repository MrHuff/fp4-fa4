from __future__ import annotations

import ast
import gc
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from weakref import WeakValueDictionary

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = (
    ROOT / "tk_fa4" / "lowp_fa4_bwd" / "benchmark_llama12b_e2e.py"
)
INTERFACE = ROOT / "tk_fa4" / "interface.py"


def _top_level_node(path: Path, name: str) -> ast.AST:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if getattr(node, "name", None) == name:
            return node
    raise AssertionError(f"missing top-level definition {name!r} in {path}")


def _execute_runtime(*names: str, namespace: dict[str, Any] | None = None):
    module = ast.Module(
        body=[
            ast.parse("from __future__ import annotations").body[0],
            *(_top_level_node(RUNTIME, name) for name in names),
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    result = {
        "Any": Any,
        "dataclass": dataclass,
        "field": field,
        "torch": torch,
    }
    if namespace is not None:
        result.update(namespace)
    exec(compile(module, str(RUNTIME), "exec"), result)
    return result


def test_generation_rejects_second_forward_until_matching_backward() -> None:
    state_type = _execute_runtime("_WorkspacePublicationState")[
        "_WorkspacePublicationState"
    ]
    state = state_type()

    first = state.begin_forward(requires_backward=True)
    assert first == 0
    assert state.in_flight_generation == first
    with pytest.raises(RuntimeError, match="still awaiting backward"):
        state.begin_forward(requires_backward=True)
    with pytest.raises(RuntimeError, match="still awaiting backward"):
        state.begin_forward(requires_backward=False)

    state.require_backward(first)
    state.finish_backward(first)
    second = state.begin_forward(requires_backward=True)
    assert second == 1
    assert state.in_flight_generation == second


def test_generation_rejects_stale_repeated_backward_and_abort() -> None:
    state_type = _execute_runtime("_WorkspacePublicationState")[
        "_WorkspacePublicationState"
    ]
    state = state_type()
    first = state.begin_forward(requires_backward=True)
    state.finish_backward(first)
    with pytest.raises(RuntimeError, match="does not own"):
        state.require_backward(first)
    with pytest.raises(RuntimeError, match="does not own"):
        state.finish_backward(first)

    no_grad = state.begin_forward(requires_backward=False)
    assert no_grad == 1
    assert state.in_flight_generation is None
    state.abort_forward(no_grad)
    active = state.begin_forward(requires_backward=True)
    with pytest.raises(RuntimeError, match="stale"):
        state.abort_forward(no_grad)
    assert state.in_flight_generation == active
    state.abort_forward(active)
    assert state.in_flight_generation is None
    with pytest.raises(TypeError, match="exactly bool"):
        state.begin_forward(requires_backward=1)


def test_tensor_abi_identity_excludes_version_but_covers_metadata() -> None:
    identity = _execute_runtime("_tensor_abi_identity")[
        "_tensor_abi_identity"
    ]
    tensor = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    before = identity(tensor)
    version = tensor._version

    tensor.add_(1)
    assert tensor._version > version
    assert identity(tensor) == before

    replacement = tensor.clone()
    assert identity(replacement) != before
    transposed = tensor.T
    assert transposed.data_ptr() == tensor.data_ptr()
    assert tuple(transposed.shape) == tuple(tensor.shape)
    assert identity(transposed) != before
    offset = tensor.reshape(-1)[1:]
    assert offset.storage_offset() != tensor.storage_offset()
    assert identity(offset) != before


def test_same_stream_guard_is_fail_closed_without_cuda() -> None:
    cuda_device = SimpleNamespace(type="cuda")
    cpu_device = SimpleNamespace(type="cpu")
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            current_stream=lambda _device: SimpleNamespace(cuda_stream=17)
        )
    )
    guard = _execute_runtime(
        "_require_forward_workspace_same_stream",
        namespace={"torch": fake_torch},
    )["_require_forward_workspace_same_stream"]

    cuda_workspace = SimpleNamespace(
        outputs=SimpleNamespace(
            q_payload=SimpleNamespace(device=cuda_device)
        ),
        cuda_stream=17,
    )
    guard(
        cuda_workspace,
        SimpleNamespace(device=cuda_device),
        phase="forward",
    )
    cuda_workspace.cuda_stream = 19
    with pytest.raises(RuntimeError, match="stream mismatch during backward"):
        guard(
            cuda_workspace,
            SimpleNamespace(device=cuda_device),
            phase="backward",
        )
    with pytest.raises(RuntimeError, match="device mismatch"):
        guard(
            cuda_workspace,
            SimpleNamespace(device=cpu_device),
            phase="forward",
        )

    cpu_workspace = SimpleNamespace(
        outputs=SimpleNamespace(q_payload=SimpleNamespace(device=cpu_device)),
        cuda_stream=None,
    )
    guard(
        cpu_workspace,
        SimpleNamespace(device=cpu_device),
        phase="forward",
    )
    cpu_workspace.cuda_stream = 17
    with pytest.raises(RuntimeError, match="unexpectedly records"):
        guard(
            cpu_workspace,
            SimpleNamespace(device=cpu_device),
            phase="forward",
        )


def test_module_and_custom_autograd_enforce_generation_and_stream_order() -> None:
    source = RUNTIME.read_text()
    tree = ast.parse(source)
    attention = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LowpAttention"
    )
    module_forward = next(
        node
        for node in attention.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    module_apply = next(
        node
        for node in attention.body
        if isinstance(node, ast.FunctionDef) and node.name == "_apply"
    )
    custom = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "_LowpAttentionFunction"
    )
    custom_forward = next(
        node
        for node in custom.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    custom_backward = next(
        node
        for node in custom.body
        if isinstance(node, ast.FunctionDef) and node.name == "backward"
    )
    module_text = ast.unparse(module_forward)
    apply_text = ast.unparse(module_apply)
    forward_text = ast.unparse(custom_forward)
    backward_text = ast.unparse(custom_backward)

    stream = module_text.index("_require_forward_workspace_same_stream")
    grad_mode = module_text.index("torch.is_grad_enabled()")
    begin = module_text.index("publication_state.begin_forward")
    apply = module_text.index("_LowpAttentionFunction.apply")
    assert stream < grad_mode < begin < apply
    assert "publication_state.abort_forward(generation)" in module_text
    assert "publications are awaiting backward" in apply_text
    assert "ctx.forward_workspace = forward_workspace" in forward_text
    assert "ctx.publication_generation = publication_generation" in forward_text

    require = backward_text.index("publication_state.require_backward")
    backward_stream = backward_text.index(
        "_require_forward_workspace_same_stream"
    )
    saved = backward_text.index("saved_tensors = ctx.saved_tensors")
    last_consumer = backward_text.index("lowp/bwd/output_weight_gradient")
    finish = backward_text.index("publication_state.finish_backward")
    result_return = backward_text.rindex("return result")
    assert require < backward_stream < saved < last_consumer < finish
    assert finish < result_return


def test_projection_workspace_validation_caches_are_weak() -> None:
    interface_source = INTERFACE.read_text()
    assert "@dataclass(frozen=True, slots=True, weakref_slot=True)" in (
        interface_source
    )
    assert interface_source.count("WeakValueDictionary()") == 4
    assert "_validated_forward_workspaces: dict[" not in interface_source

    workspace_node = _top_level_node(
        INTERFACE,
        "B300E4M3QKVForwardWorkspace",
    )
    module = ast.Module(
        body=[
            ast.parse("from __future__ import annotations").body[0],
            workspace_node,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {"dataclass": dataclass, "torch": torch}
    exec(compile(module, str(INTERFACE), "exec"), namespace)
    workspace_type = namespace["B300E4M3QKVForwardWorkspace"]
    assert "__weakref__" in workspace_type.__slots__

    workspace = object.__new__(workspace_type)
    reference = weakref.ref(workspace)
    cache: WeakValueDictionary[int, object] = WeakValueDictionary()
    cache[id(workspace)] = workspace
    assert len(cache) == 1
    del workspace
    gc.collect()
    assert reference() is None
    assert len(cache) == 0
