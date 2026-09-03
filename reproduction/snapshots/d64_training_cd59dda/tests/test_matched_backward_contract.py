from __future__ import annotations

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tk_fa4.lowp_fa4_bwd.backward_contract import (
    require_matching_backward_contracts,
)
from tk_fa4.lowp_fa4_bwd import forward_route
from tk_fa4 import interface


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "tk_fa4" / "lowp_fa4_bwd" / "benchmark_llama12b_e2e.py"
TRAINER = ROOT / "tk_fa4" / "lowp_fa4_bwd" / "train_llama12b_real_tokens.py"


def _fake_forward_workspace() -> interface.B300E4M3QKVForwardWorkspace:
    """Create an identity-only compact workspace without allocating CUDA."""
    workspace = object.__new__(interface.B300E4M3QKVForwardWorkspace)
    for field in interface.B300E4M3QKVForwardWorkspace.__dataclass_fields__:
        object.__setattr__(workspace, field, object())
    return workspace


def _contract() -> dict[str, object]:
    return {
        "schema": "lowp_backward_contract_v1",
        "probability": {
            "forward_mx_probability_replay": False,
            "forward_mx_probability_scale_handoff": False,
        },
        "projection": {
            "qk_backward_source": "represented_nvfp4_codes_per_row_k16",
            "v_backward_source": "projection_accumulator_e4m3",
        },
    }


def test_identical_backward_contracts_pass() -> None:
    contract = _contract()
    require_matching_backward_contracts(
        {"mx": contract, "fp8": _contract()}
    )


def test_probability_replay_or_operand_mismatch_fails_with_fields() -> None:
    mx = _contract()
    probability = dict(mx["probability"])
    probability["forward_mx_probability_replay"] = True
    mx["probability"] = probability
    projection = dict(mx["projection"])
    projection["v_backward_source"] = "represented_mxfp4_codes"
    mx["projection"] = projection

    with pytest.raises(RuntimeError) as error:
        require_matching_backward_contracts({"fp8": _contract(), "mx": mx})
    message = str(error.value)
    assert "probability.forward_mx_probability_replay" in message
    assert "projection.v_backward_source" in message


def test_single_lowp_route_does_not_require_a_match() -> None:
    require_matching_backward_contracts({"mx": _contract()})


def test_runtime_contract_covers_replay_schedule_and_operand_sources() -> None:
    source = RUNTIME.read_text()
    assert "def backward_contract(self)" in source
    for field in (
        '"forward_mx_probability_replay"',
        '"forward_mx_probability_scale_handoff"',
        '"reuse_quantized_p"',
        '"fp8_ds_lift"',
        '"head_fast_raster"',
        '"direct_tma_dkdv"',
        '"generated_source"',
        '"qk_backward_source"',
        '"v_backward_source"',
        '"projection_dgrad"',
        '"v_weight_gain"',
    ):
        assert field in source


def test_forward_dispatch_provenance_is_read_only_and_separate() -> None:
    source = RUNTIME.read_text()
    forward_contract = source.split(
        "def forward_dispatch_contract(self)", 1
    )[1].split("def backward_contract(self)", 1)[0]
    backward_contract = source.split("def backward_contract(self)", 1)[1]
    assert '"schema": "lowp_forward_dispatch_contract_v2"' in forward_contract
    assert '"construction_bound_exact_pybind_symbol"' in forward_contract
    assert '"construction_bound_route_specific_entrypoint"' in forward_contract
    assert '"first_call_full_abi_validation_complete"' in forward_contract
    assert '"abi_validation_symbol"' in forward_contract
    assert '"checked_symbol"' in forward_contract
    assert '"unchecked_symbol"' in forward_contract
    assert '"preallocated_forward_workspace_required"' in forward_contract
    assert '"preallocated_forward_publication_slots"' in forward_contract
    assert '"preallocated_forward_workspace_abi_validated"' in forward_contract
    assert '"private_nonpersistent_layer_route_neutral_superset"' in (
        forward_contract
    )
    assert '"runtime_crossover_reallocation": False' in forward_contract
    assert "torch." not in forward_contract
    assert "self.backward" not in forward_contract
    assert "lowp_forward_dispatch_contract_v2" not in backward_contract
    assert '"forward_dispatch": forward_dispatch' in source


def test_runtime_requires_projection_specialization_before_workspace() -> None:
    source = RUNTIME.read_text()
    runtime = source.split("class LowpAttentionRuntime:", 1)[1]
    capability = runtime.index(
        "b300_bind_qkv_gqa_d64_paired_unified_lowp_e4m3_projection("
    )
    workspace = runtime.index("q = torch.empty(", capability)
    backward = runtime.index("self.backward = CompiledGqaBackward(", workspace)
    assert capability < workspace < backward


def test_projection_capability_error_names_extension_and_symbol() -> None:
    interface = (ROOT / "tk_fa4" / "interface.py").read_text()
    helper = interface.split(
        "def b300_require_qkv_gqa_d64_paired_unified_lowp_e4m3_projection(",
        1,
    )[1].split(
        "def b300_project_qkv_gqa_d64_paired_unified_lowp_e4m3(", 1
    )[0]
    assert 'getattr(_C_b300_lowp_bwd, "__file__", "<unknown>")' in helper
    assert "does not provide required projection" in helper
    assert "experimental_split_v_backward" in helper


def test_projection_capability_resolves_requested_split_v_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    symbol = (
        "project_qkv_gqa_d64_paired_unified_fp8_e4m3_rope_packed_"
        "interleaved_causal_represented_backward_perblock_qk_"
        "split_v_backward"
    )
    extension = SimpleNamespace(__file__="/tmp/capable.so")
    setattr(extension, symbol, object())
    monkeypatch.setattr(interface, "_C_b300_lowp_bwd", extension)
    require_projection = getattr(
        interface,
        "b300_require_qkv_gqa_d64_paired_unified_lowp_e4m3_projection",
    )

    assert (
        require_projection(
            publish_mxfp4_v=True,
            represented_backward=True,
            per_block_qk_scales=True,
            experimental_split_v_backward=True,
        )
        == symbol
    )


def test_projection_capability_rejects_selected_extension_early(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        interface,
        "_C_b300_lowp_bwd",
        SimpleNamespace(__file__="/tmp/pinned-6480.so"),
    )
    require_projection = getattr(
        interface,
        "b300_require_qkv_gqa_d64_paired_unified_lowp_e4m3_projection",
    )
    with pytest.raises(RuntimeError) as error:
        require_projection(
            publish_mxfp4_v=True,
            represented_backward=True,
            per_block_qk_scales=True,
            experimental_split_v_backward=True,
        )
    message = str(error.value)
    assert "/tmp/pinned-6480.so" in message
    assert "split_v_backward" in message


def test_trainer_fails_before_model_allocation() -> None:
    source = TRAINER.read_text()
    projection_check = source.index(
        "projection_extension = _projection_extension_identity("
    )
    check = source.index(
        "require_matching_backward_contracts(backward_route_contracts)"
    )
    allocation = source.index("torch.manual_seed(args.seed)", check)
    token_allocation = source.index("train_tokens, train_targets")
    assert projection_check < token_allocation
    assert check < allocation
    assert '"projection_extension": projection_extension' in source
    assert '"backward_route_contracts": backward_route_contracts' in source


def test_trainer_defaults_to_production_matched_lowp_routes() -> None:
    module = ast.parse(TRAINER.read_text())
    defaults: dict[str, object] = {}
    for node in ast.walk(module):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument":
            continue
        option = ast.unparse(node.args[0])
        for keyword in node.keywords:
            if keyword.arg == "default" and isinstance(
                keyword.value, ast.Constant
            ):
                defaults[option] = keyword.value.value
    assert defaults["f'--{route}-backward-match-forward-operands'"] is True
    assert defaults["f'--{route}-per-block-qk-scales'"] is True
    assert defaults["'--mx-experimental-split-v-backward'"] is True
    assert defaults[
        "'--mx-backward-forward-probability-replay'"
    ] is False
    assert defaults[
        "'--mx-backward-forward-probability-scale-handoff'"
    ] is False
    assert defaults["'--mx-qkv-projection-format'"] == "e4m3"
    assert defaults["'--fp8-qkv-projection-format'"] == "e4m3"


def test_trainer_preflights_exact_forward_dispatch_after_compile() -> None:
    source = TRAINER.read_text()
    main = source.split("def main() -> None:", 1)[1]
    compile_loop = main.index(
        "for execution_position, name in enumerate(route_names):"
    )
    preflight = main.index(
        "timed_forward_dispatch_contracts = (", compile_loop
    )
    validation = main.index("validation_history = [", preflight)
    assert compile_loop < preflight < validation
    preflight_source = main[compile_loop:validation]
    assert "if mx_runtime is not None or fp8_runtime is not None:" in (
        preflight_source
    )
    assert "_timed_forward_dispatch_contracts(" in preflight_source
    assert 'models.get("nvfp4_qk_mxfp4_pv")' in preflight_source
    assert 'models.get("nvfp4_qk_fp8_pv_exact")' in preflight_source
    assert (
        '"timed_forward_dispatch_contracts": ('
        in main
    )


def test_trainer_constructs_the_same_fixed_routes_as_comparator() -> None:
    source = TRAINER.read_text()
    main = source.split("def main() -> None:", 1)[1]
    mx_call = main.split(
        'if "nvfp4_qk_mxfp4_pv" in route_names:', 1
    )[1].split('if "nvfp4_qk_fp8_pv_exact" in route_names:', 1)[0]
    fp8_call = main.split(
        'if "nvfp4_qk_fp8_pv_exact" in route_names:', 1
    )[1].split("backward_route_contracts = {", 1)[0]
    assert 'route_slot="mx"' in mx_call
    assert 'route_slot="fp8"' in fp8_call
    assert "shared_backward_runtime=mx_runtime" in fp8_call
    assert "experimental_split_v_backward" not in fp8_call


def test_attention_dispatch_has_no_per_layer_route_or_environment_check() -> None:
    source = RUNTIME.read_text()
    dispatch = source.split("def _run_lowp_forward_attention(", 1)[1].split(
        "class _LowpAttentionFunction", 1
    )[0]
    assert "require_active_forward_route(" not in dispatch
    assert "activate_forward_route(" not in dispatch
    assert "os.environ" not in dispatch
    assert ".permute(" not in dispatch
    assert "runtime.launch_forward_attention(" in dispatch
    fp8_binding = source.split("def _launch_forward_fp8(", 1)[1].split(
        "def _launch_forward_mx(", 1
    )[0]
    assert ".permute(" not in fp8_binding
    assert "refusing an unfused permute/contiguous fallback" in fp8_binding


@pytest.mark.parametrize(
    ("publish_mxfp4_v", "experimental_split_v_backward", "route_suffix"),
    (
        (False, False, "_fp8_forward_out"),
        (True, True, "_mx_forward_out"),
    ),
)
def test_bound_projection_authenticates_each_workspace_then_uses_unchecked(
    monkeypatch: pytest.MonkeyPatch,
    publish_mxfp4_v: bool,
    experimental_split_v_backward: bool,
    route_suffix: str,
) -> None:
    calls: list[tuple[str, object]] = []
    base_symbol = "bound_projection"
    checked_symbol = base_symbol + route_suffix
    unchecked_symbol = checked_symbol + "_unchecked"
    def compact_checked(*args: object) -> object:
        calls.append(("compact_checked", args))
        return args[-3:]

    def compact_unchecked(*args: object) -> object:
        calls.append(("compact_unchecked", args))
        return args[-3:]

    extension = SimpleNamespace(
        **{
            checked_symbol: compact_checked,
            unchecked_symbol: compact_unchecked,
        }
    )
    monkeypatch.setattr(interface, "_C_b300_lowp_bwd", extension)
    monkeypatch.setattr(
        interface,
        "b300_require_qkv_gqa_d64_paired_unified_lowp_e4m3_projection",
        lambda **_kwargs: base_symbol,
    )
    legacy_bundle = SimpleNamespace(
        backward=SimpleNamespace(
            score_q_fp4=object(),
            score_k_fp4=object(),
        ),
        q_forward_scales=object(),
        q_forward_global_scale=object(),
        k_forward_scales=object(),
        k_forward_global_scale=object(),
        v_forward_fp4=object(),
        v_forward_scales=object(),
        v_forward_fp8=object(),
        v_backward_fp8=object(),
        q_backward_fp8=object(),
        k_backward_fp8=object(),
    )

    def legacy(*args: object, **kwargs: object) -> object:
        calls.append(("legacy", (args, kwargs)))
        return legacy_bundle

    monkeypatch.setattr(
        interface,
        "b300_project_qkv_gqa_d64_paired_unified_lowp_e4m3",
        legacy,
    )
    monkeypatch.setattr(
        interface,
        "_b300_require_bitwise_equal",
        lambda *_, **__: None,
    )
    returned_bundles: list[object] = []

    def compact_bundle(*args: object, **kwargs: object) -> object:
        result = object()
        returned_bundles.append(result)
        calls.append(("bundle", (args, kwargs)))
        return result

    monkeypatch.setattr(
        interface,
        "_b300_compact_e4m3_qkv_bundle",
        compact_bundle,
    )
    bound = interface.B300BoundE4M3QKVProjection(
        batch=1,
        seqlen=4096,
        q_heads=32,
        kv_heads=8,
        publish_mxfp4_v=publish_mxfp4_v,
        v_mxfp4_scale_2d=False,
        represented_backward=True,
        per_block_qk_scales=True,
        experimental_split_v_backward=experimental_split_v_backward,
    )
    operands = ((object(), object()), (object(), object()), object(), object())
    workspace = _fake_forward_workspace()
    second_workspace = _fake_forward_workspace()
    with pytest.raises(TypeError, match="requires a B300E4M3QKVForwardWorkspace"):
        bound(*operands, forward_workspace=object())
    assert bound(*operands, forward_workspace=workspace) is returned_bundles[-1]
    assert bound.abi_validated is True
    assert bound.forward_workspace_abi_validated is True
    assert bound.validated_forward_workspace_count == 1
    assert bound(*operands, forward_workspace=workspace) is returned_bundles[-1]
    assert bound.validated_forward_workspace_count == 1
    assert (
        bound(*operands, forward_workspace=second_workspace)
        is returned_bundles[-1]
    )
    assert bound.validated_forward_workspace_count == 2
    assert bound.abi_validation_symbol == base_symbol
    assert bound.checked_symbol == checked_symbol
    assert bound.unchecked_symbol == unchecked_symbol
    assert bound.symbol == unchecked_symbol
    assert bound.requires_forward_workspace is True
    assert bound.requires_v_mxfp4_scales_out is False
    assert [kind for kind, _payload in calls] == [
        "legacy",
        "compact_checked",
        "bundle",
        "compact_unchecked",
        "bundle",
        "legacy",
        "compact_checked",
        "bundle",
    ]
    compact_calls = [
        payload
        for kind, payload in calls
        if kind in ("compact_checked", "compact_unchecked")
    ]
    assert all(isinstance(payload, tuple) for payload in compact_calls)
    assert all(len(payload) == 23 for payload in compact_calls)
    assert compact_calls[0][-12:] == workspace.compact_outputs()
    assert compact_calls[1][-12:] == workspace.compact_outputs()
    assert compact_calls[2][-12:] == second_workspace.compact_outputs()
    bundle_calls = [
        payload for kind, payload in calls if kind == "bundle"
    ]
    assert bundle_calls[0][0][1] == workspace.compact_outputs()[-3:]
    assert bundle_calls[1][0][1] == workspace.compact_outputs()[-3:]
    assert bundle_calls[2][0][1] == second_workspace.compact_outputs()[-3:]


def test_mx_v_scale_authentication_ignores_only_unwritten_d64_padding() -> None:
    reference = torch.zeros((1, 512), dtype=torch.uint8)
    padding_only_difference = reference.clone()
    padding_only_difference[0, 8:16] = 7
    valid_indices = tuple(
        depth_lane * 16 + depth_group * 4 + sequence_quarter
        for depth_lane in range(32)
        for depth_group in range(2)
        for sequence_quarter in range(4)
    )
    interface._b300_require_bitwise_equal(
        "MXFP4 V scale pages",
        reference,
        padding_only_difference,
        valid_last_dim_indices=valid_indices,
    )
    padding_only_difference[0, 0] = 1
    with pytest.raises(RuntimeError, match="not bitwise identical"):
        interface._b300_require_bitwise_equal(
            "MXFP4 V scale pages",
            reference,
            padding_only_difference,
            valid_last_dim_indices=valid_indices,
        )


def test_layer_forward_workspace_is_route_neutral_nonpersistent_scratch() -> None:
    source = RUNTIME.read_text()
    assert "class _LowpAttentionForwardWorkspace:" in source
    assert "outputs: B300E4M3QKVForwardWorkspace" in source
    for field in (
        '"q_payload": 4',
        '"k_payload": 6',
        '"q_scale_pages": 8',
        '"q_global_scale": 9',
        '"k_scale_pages": 10',
        '"k_global_scale": 11',
        '"v_mxfp4_payload": 12',
        '"v_mxfp4_scale_pages": 13',
        '"v_backward_fp8": 20',
        '"q_backward_fp8": 21',
        '"k_backward_fp8": 22',
        '"v_fp8_payload": 23',
    ):
        assert field in source
    assert "self._forward_workspace = self._allocate_forward_workspace()" in source
    assert "dtype=torch.float8_e4m3fn" in source
    assert 'register_buffer(\n            "q_payload' not in source
    assert "def _apply(self, fn: Any, recurse: bool = True)" in source
    assert "def require_lowp_forward_workspace_stream(self)" in source
    lowp_forward = source.split(
        "class LowpAttention(nn.Module):", 1
    )[1].split("class MLP", 1)[0].split("def forward(", 1)[1]
    assert "torch.cuda.current_stream" not in lowp_forward
    function = source.split("class _LowpAttentionFunction", 1)[1].split(
        "class LowpAttention", 1
    )[0]
    assert "forward_workspace: _LowpAttentionForwardWorkspace" in function
    assert "forward_workspace=forward_workspace.outputs" in function
    assert "ctx.save_for_backward(" in function
    saved = function.split("ctx.save_for_backward(", 1)[1].split(")", 1)[0]
    assert "forward_workspace" not in saved


def test_opaque_workspace_is_invisible_to_module_and_optimizer_state() -> None:
    class Owner(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            owners = tuple(torch.empty(8) for _ in range(12))
            self._forward_workspace = SimpleNamespace(
                owners=owners,
                cuda_stream=None,
            )

    owner = Owner()
    outputs = owner._forward_workspace.owners
    assert len(outputs) == 12
    assert all(
        all(buffer is not output for _name, buffer in owner.named_buffers())
        for output in outputs
    )
    assert all(
        all(
            parameter is not output
            for _name, parameter in owner.named_parameters()
        )
        for output in outputs
    )
    assert set(owner.state_dict()) == {"weight"}
    optimizer = torch.optim.SGD(owner.parameters(), lr=0.1)
    assert all(
        all(parameter is not output for output in outputs)
        for group in optimizer.param_groups
        for parameter in group["params"]
    )


def test_second_matched_runtime_can_reuse_backward_at_construction() -> None:
    source = RUNTIME.read_text()
    constructor = source.split("class LowpAttentionRuntime:", 1)[1].split(
        "def backward_contract", 1
    )[0]
    assert "shared_backward_runtime: LowpAttentionRuntime | None" in constructor
    shared_branch = constructor.split(
        "if shared_backward_runtime is not None:", 2
    )[-1]
    assert "self.backward = shared.backward" in shared_branch
    assert shared_branch.index("self.backward = shared.backward") < (
        shared_branch.index("q = torch.empty(")
    )


def test_trainer_activates_route_and_zeros_grad_before_forward_event() -> None:
    source = TRAINER.read_text()
    step = source.split("def _step(", 1)[1].split("def _compile_without_update", 1)[0]
    route = step.index("activate_model_forward_route(model)")
    zero_grad = step.index("optimizer.zero_grad(set_to_none=True)")
    start = step.index("start.record()")
    model_forward = step.index("logits = model(tokens)")
    assert route < start
    assert zero_grad < start
    assert start < model_forward


def test_trainer_does_not_read_loss_on_host_between_forward_and_backward() -> None:
    source = TRAINER.read_text()
    step = source.split("def _step(", 1)[1].split(
        "def _compile_without_update", 1
    )[0]
    timed_gap = step.split("forward_done.record()", 1)[1].split(
        "loss.backward()", 1
    )[0]
    assert "float(loss.detach())" not in timed_gap.split(
        "if diagnostics is not None:", 1
    )[0]
    ordinary_read = step.index("if diagnostics is None:")
    end_sync = step.index("end.synchronize()")
    assert end_sync < ordinary_read


def test_trainer_activates_each_model_route_before_validation_batches() -> None:
    source = TRAINER.read_text()
    evaluate = source.split("def _evaluate(", 1)[1].split(
        "def _timing_summary", 1
    )[0]
    model_loop = evaluate.index("for name, model in models.items():")
    activate = evaluate.index("activate_model_forward_route(model)", model_loop)
    batch_loop = evaluate.index("for batch_index in range(", model_loop)
    model_forward = evaluate.index("logits = model(", batch_loop)
    assert model_loop < activate < batch_loop < model_forward


def test_llama_model_requires_prebound_route_once_at_forward_boundary() -> None:
    source = RUNTIME.read_text()
    model = source.split("class Llama12B", 1)[1].split(
        "def _useful_flops", 1
    )[0]
    assert "self.lowp_attention_runtime = runtime" in model
    forward = model.split("def forward(", 1)[1]
    assert forward.index("require_active_forward_route(") < forward.index(
        "F.embedding("
    )
    assert "activate_model_forward_route(self)" not in forward
    assert "for layer_index" not in forward.split("F.embedding(", 1)[0]
    assert "def bind_lowp_attention_runtime(" in model
    assert "attention.runtime is not runtime" in model


def test_forward_route_activation_writes_once_then_switches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(forward_route.FORWARD_ROUTE_ENV, raising=False)
    monkeypatch.setattr(forward_route, "_active_forward_route", None)

    assert forward_route.activate_forward_route("mx")
    assert os.environ[forward_route.FORWARD_ROUTE_ENV] == "mx"
    assert not forward_route.activate_forward_route("mx")
    assert forward_route.activate_forward_route("fp8")
    assert os.environ[forward_route.FORWARD_ROUTE_ENV] == "fp8"
    assert not forward_route.activate_forward_route("fp8")


def test_forward_route_cache_recovers_from_external_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(forward_route, "_active_forward_route", "mx")
    monkeypatch.setenv(forward_route.FORWARD_ROUTE_ENV, "fp8")
    assert forward_route.activate_forward_route("mx")
    assert os.environ[forward_route.FORWARD_ROUTE_ENV] == "mx"


@pytest.mark.parametrize("route", ("", None, 7))
def test_forward_route_rejects_invalid_tokens(route: object) -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        forward_route.activate_forward_route(route)  # type: ignore[arg-type]
