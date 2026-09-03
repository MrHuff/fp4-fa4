from __future__ import annotations

import ast
import hashlib
import os
import runpy
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = (
    ROOT / "tk_fa4" / "lowp_fa4_bwd" / "benchmark_llama12b_e2e.py"
)
VALIDATOR = (
    ROOT
    / "tk_fa4"
    / "lowp_fa4_bwd"
    / "validate_llama12b_batched_exact_runtime.py"
)


def _nodes(*names: str) -> list[ast.stmt]:
    selected = []
    for node in ast.parse(RUNTIME.read_text()).body:
        name = getattr(node, "name", None)
        targets = (
            [target.id for target in node.targets if isinstance(target, ast.Name)]
            if isinstance(node, ast.Assign)
            else []
        )
        if name in names or any(target in names for target in targets):
            selected.append(node)
    return selected


def _namespace(*names: str) -> dict[str, Any]:
    selected_names = tuple(
        dict.fromkeys(
            (
                "AUTHENTICATED_D64_EXACT_BATCHES",
                "AUTHENTICATED_D128_EXACT_BATCHES",
                "D128_EXACT_FORWARD_TOPOLOGIES",
                "D128_MX_TOPOLOGY_KEY",
                "D128_MX_FORWARD_TOPOLOGY_VARIANTS",
                "D128_FORWARD_TOPOLOGY_VARIANTS",
                "DIAGNOSTIC_FP8_LSE_SUBSTITUTION_MODES",
                "_d128_forward_topology_recipe",
                "SUPPORTED_LOWP_BATCHES",
                *names,
            )
        )
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            *_nodes(*selected_names),
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {
        "Any": Any,
        "Path": Path,
        "dataclass": dataclass,
        "hashlib": hashlib,
        "statistics": statistics,
        "torch": torch,
    }
    exec(compile(module, str(RUNTIME), "exec"), namespace)
    return namespace


def _validator_namespace() -> dict[str, Any]:
    return runpy.run_path(str(VALIDATOR), run_name="validator_cpu_test")


def _exact_topology(*, batch: int = 2) -> dict[str, object]:
    return {
        "batch": batch,
        "seqlen": 4096,
        "heads": 32,
        "kv_heads": 8,
        "dqk": 64,
        "dvo": 64,
        "causal": True,
        "qk_format": "nvfp4_e4m3_block16",
        "pv_format": "e4m3_fp8",
        "shiftless_fp8_mode": 0,
        "causal_interleaved_kv": False,
        "route": "real_fwd_tk_hao_direct_causal_gqa_nvfp4_fp8pv",
        "schema": "tk_hao_direct_pipeline_v1",
        "fixed_route_fastpath": True,
        "fixed_p_ceiling": False,
        "score_pack_ceiling": False,
        "valid": 1,
    }


def _mx_topology(*, batch: int = 2) -> dict[str, object]:
    return {
        "batch": batch,
        "seqlen": 4096,
        "heads": 32,
        "kv_heads": 8,
        "dqk": 64,
        "dvo": 64,
        "causal": True,
        "qk_format": "nvfp4_e4m3_block16",
        "pv_format": "mxfp4_e8m0_block32",
        "causal_interleaved_kv": True,
        "route": "real_fwd_tk_hao_direct_nvfp4_mxfp4pv",
        "schema": "tk_hao_direct_pipeline_v1",
        "fixed_route_fastpath": True,
        "fixed_p_ceiling": False,
        "score_pack_ceiling": False,
        "d64_detached_p": True,
        "mx_mode23_native_density": 4,
        "mx_mode23_native_quarter_mask": 3,
        "mx_global_anchor32": True,
        "mx_anchor_affine_hoist": True,
        "mx_global_anchor_margin_log2": 64,
        "mx_stored_scale_shift_log2": 16,
        "valid": 1,
    }


def _d128_fp8_topology(*, batch: int = 2) -> dict[str, object]:
    recipe = _namespace()["D128_EXACT_FORWARD_TOPOLOGIES"][
        (
            "real_fwd_tk_hao_direct_causal_gqa_nvfp4_fp8pv",
            "e4m3_fp8",
        )
    ]
    return {
        "batch": batch,
        "seqlen": 4096,
        "heads": 32,
        "kv_heads": 8,
        "dqk": 128,
        "dvo": 128,
        "causal": True,
        "qk_format": "nvfp4_e4m3_block16",
        **recipe,
        "valid": 1,
    }


def _d128_mx_topology(*, batch: int = 2) -> dict[str, object]:
    recipe = _namespace()["D128_EXACT_FORWARD_TOPOLOGIES"][
        (
            "real_fwd_tk_hao_direct_nvfp4_mxfp4pv",
            "mxfp4_e8m0_block32",
        )
    ]
    return {
        "batch": batch,
        "seqlen": 4096,
        "heads": 32,
        "kv_heads": 8,
        "dqk": 128,
        "dvo": 128,
        "causal": True,
        "qk_format": "nvfp4_e4m3_block16",
        **recipe,
        "valid": 1,
    }


def _d64_batched_config(batch: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        batch=batch,
        sequence=4096,
        hidden=2048,
        q_heads=32,
        kv_heads=8,
        head_dim=64,
    )


def _d128_batched_config(
    batch: int = 2,
    *,
    topology_variant: str = "production",
) -> SimpleNamespace:
    return SimpleNamespace(
        batch=batch,
        sequence=4096,
        hidden=4096,
        q_heads=32,
        kv_heads=8,
        head_dim=128,
        d128_forward_topology_variant=topology_variant,
    )


def _exact_runtime_kwargs() -> dict[str, object]:
    return {
        "projection_dgrad": "bf16",
        "qkv_projection_format": "e4m3",
        "backward_exp2_degree": 1,
        "backward_exp2_period": 2,
        "backward_fp8_ds_lift": 16,
        "backward_reuse_quantized_p": False,
        "backward_forward_mx_probability_replay": False,
        "backward_forward_mx_probability_scale_handoff": False,
        "backward_match_forward_operands": True,
        "per_block_qk_scales": True,
        "experimental_split_v_backward": False,
        "backward_probability_correction": 1.0,
        "q_quant_scale": 2.25,
        "k_quant_scale": 2.0,
        "projection_weight_scale_2d": True,
        "v_mxfp4_scale_2d": False,
        "adaptive_qk_weight_scales": False,
        "shared_runtime": None,
    }


def _mx_runtime_kwargs() -> dict[str, object]:
    kwargs = _exact_runtime_kwargs()
    kwargs["experimental_split_v_backward"] = True
    return kwargs


def _native_runtime_kwargs(*, publish_mxfp4_v: bool = False) -> dict[str, object]:
    kwargs = _exact_runtime_kwargs()
    kwargs["qkv_projection_format"] = "nvfp4"
    kwargs["experimental_split_v_backward"] = publish_mxfp4_v
    return kwargs


def _d128_native_runtime_kwargs() -> dict[str, object]:
    kwargs = _native_runtime_kwargs()
    kwargs.update(
        {
            "projection_dgrad": "nvfp4",
            "backward_exp2_period": 0,
            "backward_reuse_quantized_p": True,
            "backward_match_forward_operands": False,
            "experimental_split_v_backward": False,
        }
    )
    return kwargs


def test_requested_backward_policy_uses_shared_p_only_for_d128() -> None:
    policy = _namespace("_requested_backward_approximation_policy")[
        "_requested_backward_approximation_policy"
    ]
    assert policy(_d128_batched_config(2)) == (1, 0, True)
    assert policy(_d128_batched_config(1)) == (1, 0, True)
    assert policy(_d64_batched_config(16)) == (1, 2, False)
    assert policy(_d64_batched_config(1)) == (2, None, False)


def test_qk_scale_summary_represents_both_d128_b2_batches() -> None:
    namespace = _namespace("_qk_scale_summary")

    class FakeLowpAttention:
        def __init__(self, scales: torch.Tensor) -> None:
            self.qk_scales = scales

    namespace["LowpAttention"] = FakeLowpAttention
    scales = torch.zeros((2, 4, 7), dtype=torch.float32)
    scales[0, :, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    scales[1, :, 0] = torch.tensor([10.0, 20.0, 30.0, 40.0])
    scales[0, :2, 1] = torch.tensor([5.0, 6.0])
    scales[1, :2, 1] = torch.tensor([50.0, 60.0])
    model = SimpleNamespace(
        layers=[SimpleNamespace(attention=FakeLowpAttention(scales))]
    )
    config = SimpleNamespace(batch=2, head_dim=128, kv_heads=2)

    summary = namespace["_qk_scale_summary"](model, config)

    assert summary["q"]["represented_batches"] == [0, 1]
    assert summary["q"]["per_batch"] == {
        "0": [[1.0, 2.0, 3.0, 4.0]],
        "1": [[10.0, 20.0, 30.0, 40.0]],
    }
    assert summary["q"]["per_layer"] == summary["q"]["per_batch"]["0"]
    assert summary["q"]["minimum"] == 1.0
    assert summary["q"]["maximum"] == 40.0
    assert summary["k"]["represented_batches"] == [0, 1]
    assert summary["k"]["per_batch"] == {
        "0": [[5.0, 6.0]],
        "1": [[50.0, 60.0]],
    }
    assert summary["k"]["maximum"] == 60.0


def test_output_shared_v_eligibility_retains_d64_and_adds_d128_b1_b2() -> None:
    eligible = _namespace("_native_output_shared_v_eligible")[
        "_native_output_shared_v_eligible"
    ]
    common = {
        "experimental_native_nvfp4_projection_out": True,
        "qkv_projection_format": "nvfp4",
        "publish_mxfp4_v": True,
        "per_block_qk_scales": True,
        "v_mxfp4_scale_2d": False,
    }
    assert eligible(
        _d64_batched_config(16),
        **common,
        experimental_split_v_backward=True,
        backward_match_forward_operands=True,
    )
    assert eligible(
        _d128_batched_config(1),
        **common,
        experimental_split_v_backward=False,
        backward_match_forward_operands=False,
    )
    assert eligible(
        _d128_batched_config(2),
        **common,
        experimental_split_v_backward=False,
        backward_match_forward_operands=False,
    )
    assert not eligible(
        _d128_batched_config(3),
        **common,
        experimental_split_v_backward=False,
        backward_match_forward_operands=False,
    )


@pytest.mark.parametrize(
    ("field", "wrong"),
    (
        ("experimental_native_nvfp4_projection_out", False),
        ("qkv_projection_format", "e4m3"),
        ("publish_mxfp4_v", False),
        ("experimental_split_v_backward", True),
        ("backward_match_forward_operands", True),
        ("per_block_qk_scales", False),
        ("v_mxfp4_scale_2d", True),
    ),
)
def test_d128_output_shared_v_eligibility_fails_closed(
    field: str,
    wrong: object,
) -> None:
    eligible = _namespace("_native_output_shared_v_eligible")[
        "_native_output_shared_v_eligible"
    ]
    policy = {
        "experimental_native_nvfp4_projection_out": True,
        "qkv_projection_format": "nvfp4",
        "publish_mxfp4_v": True,
        "experimental_split_v_backward": False,
        "backward_match_forward_operands": False,
        "per_block_qk_scales": True,
        "v_mxfp4_scale_2d": False,
    }
    policy[field] = wrong
    assert not eligible(_d128_batched_config(1), **policy)


def test_config_retains_b1_and_admits_only_authenticated_batches() -> None:
    namespace = _namespace(
        "Config",
        "DEFAULT_MODEL_PRESET",
        "MODEL_PRESETS",
        "config_from_model_preset",
    )
    make_config = namespace["config_from_model_preset"]
    assert make_config().batch == 1
    for batch in (2, 8, 16):
        assert make_config(batch=batch).batch == batch
        assert make_config(batch=batch).parameter_count == 1_235_814_400
    with pytest.raises(ValueError, match="D64 model preset batch"):
        make_config(batch=4)
    with pytest.raises(ValueError, match="batch must be one of"):
        make_config(batch=3)
    assert make_config("llama3.1-8b", batch=2).batch == 2
    assert make_config("llama3.1-8b", batch=4).batch == 4
    for batch in (8, 16):
        with pytest.raises(ValueError, match="D128 model preset batch"):
            make_config("llama3.1-8b", batch=batch)


def test_benchmark_releases_full_logits_before_loss_backward() -> None:
    source = RUNTIME.read_text()
    sampled = source.index("sampled_logits = logits.detach()")
    release = source.index("del logits")
    backward = source.index("loss.backward()")
    assert sampled < release < backward


@pytest.mark.parametrize("batch", (2, 8, 16))
def test_forward_topology_authenticates_batch_with_the_full_shape(
    batch: int,
) -> None:
    require = _namespace("_require_forward_topology")[
        "_require_forward_topology"
    ]
    config = _d64_batched_config(batch)
    require(config, _exact_topology(batch=batch))
    for key, value in (
        ("batch", 1),
        ("seqlen", 2048),
        ("heads", 16),
        ("kv_heads", 4),
        ("dqk", 128),
        ("dvo", 128),
        ("causal", False),
        ("qk_format", "bf16"),
    ):
        topology = _exact_topology(batch=batch)
        topology[key] = value
        with pytest.raises(ValueError, match=f"forward topology {key}"):
            require(config, topology)


@pytest.mark.parametrize("batch", (2, 8, 16))
def test_forward_topology_authenticates_reviewed_mx_batch(
    batch: int,
) -> None:
    require = _namespace("_require_forward_topology")[
        "_require_forward_topology"
    ]
    require(_d64_batched_config(batch), _mx_topology(batch=batch))


@pytest.mark.parametrize("topology_factory", (_d128_fp8_topology, _d128_mx_topology))
@pytest.mark.parametrize("batch", (1, 2))
def test_forward_topology_authenticates_d128_b1_b2_before_and_after_launch(
    topology_factory: Any,
    batch: int,
) -> None:
    require = _namespace("_require_forward_topology")[
        "_require_forward_topology"
    ]
    topology = topology_factory(batch=batch)
    config = _d128_batched_config(batch)
    require(config, topology)
    require(config, {**topology, "valid": 1}, runtime_populated=True)
    broken = dict(topology)
    broken["fixed_route_fastpath"] = False
    with pytest.raises(ValueError, match=f"D128 B{batch} exact forward topology"):
        require(config, broken)


@pytest.mark.parametrize(
    "variant",
    (
        "anchor128-m64-stable-represented-logsum",
        "anchor128-m64-stable-full-approx-lse",
    ),
)
def test_d128_mx_stable_lse_topology_is_explicit_and_type_exact(
    variant: str,
) -> None:
    namespace = _namespace("_require_forward_topology")
    require = namespace["_require_forward_topology"]
    topology = _d128_mx_topology(batch=2)
    topology.update(namespace["D128_MX_FORWARD_TOPOLOGY_VARIANTS"][variant])
    config = _d128_batched_config(2, topology_variant=variant)

    require(config, topology)
    with pytest.raises(
        ValueError,
        match="mx_(dual_lse_denom|stable_lse_logsum)",
    ):
        require(_d128_batched_config(2), topology)

    wrong_type = dict(topology)
    wrong_type["mx_stable_lse_logsum"] = 1
    with pytest.raises(ValueError, match="mx_stable_lse_logsum"):
        require(config, wrong_type)

    # The authenticated FP8-LSE diagnostic is not an MX policy and must not
    # inherit the candidate's MX-only semantic flags.
    require(config, _d128_fp8_topology(batch=2))


@pytest.mark.parametrize(
    ("field", "wrong"),
    (
        ("mx_log2_p_quant", False),
        ("mx_quantized_denom", False),
        ("mx_p_effective_max", 5),
        ("mx_pwl_exp2", False),
        ("mx_pwl_exp2_mode", 22),
        ("mx_mode23_native_density", 4),
        ("mx_mode23_native_density3_quarter_mask", 0),
        ("mx_affine_a", 1.6),
        ("mx_shiftless_softmax", False),
        ("mx_denom_decode_mode", 0),
        ("mx_ex2_q1_mask", 0),
        ("ex2_alu_degree", 2),
    ),
)
@pytest.mark.parametrize("batch", (1, 2))
def test_d128_mx_topology_rejects_unsafe_probability_numerics(
    batch: int,
    field: str,
    wrong: object,
) -> None:
    require = _namespace("_require_forward_topology")[
        "_require_forward_topology"
    ]
    topology = _d128_mx_topology(batch=batch)
    topology[field] = wrong
    with pytest.raises(
        ValueError,
        match=rf"D128 B{batch} exact forward topology {field}=",
    ):
        require(_d128_batched_config(batch), topology)


@pytest.mark.parametrize("batch", (1, 2))
def test_d128_forward_topology_requires_valid_after_launch(batch: int) -> None:
    require = _namespace("_require_forward_topology")[
        "_require_forward_topology"
    ]
    topology = _d128_mx_topology(batch=batch)
    topology["valid"] = 0
    require(_d128_batched_config(batch), topology)
    with pytest.raises(ValueError, match="valid=0"):
        require(
            _d128_batched_config(batch),
            topology,
            runtime_populated=True,
        )
    topology["valid"] = 1
    require(
        _d128_batched_config(batch),
        topology,
        runtime_populated=True,
    )


def test_d128_runtime_forces_a_postlaunch_topology_reread() -> None:
    source = RUNTIME.read_text()
    runtime_constructor = source.split(
        "class LowpAttentionRuntime:", 1
    )[1].split("@property", 1)[0]
    initialization = source.split(
        "initial_topology_populated =", 1
    )[1].split("pv_format =", 1)[0]
    launch = source.split(
        "def _run_lowp_forward_attention(", 1
    )[1].split("@dataclass", 1)[0]
    assert runtime_constructor.index(
        "_require_forward_topology(config, forward_topology)"
    ) < runtime_constructor.index("self.forward_extension = forward_extension")
    assert "initial_topology_populated and not exact_d128_forward" in (
        initialization
    )
    assert "if not runtime.forward_topology_runtime_authenticated:" in launch
    assert "runtime.forward_extension.read_hao_direct_topology()" in launch
    assert "runtime_populated=True" in launch


def test_d128_mx_diagnostic_fp8_lse_control_is_fail_closed() -> None:
    namespace = _namespace(
        "_require_forward_topology",
        "_tensor_abi_identity",
        "LowpAttentionRuntime",
    )
    runtime_type = namespace["LowpAttentionRuntime"]
    config = _d128_batched_config(2)
    runtime = SimpleNamespace(
        is_d128=True,
        pv_format="mxfp4_e8m0_block32",
        config=config,
        diagnostic_fp8_lse_entrypoint=None,
    )
    populated_topology = _d128_fp8_topology(batch=2)

    def launch(*arguments: object) -> None:
        output = arguments[-5]
        lse = arguments[-4]
        assert isinstance(output, torch.Tensor)
        assert isinstance(lse, torch.Tensor)
        output.fill_(99.0)
        lse.fill_(7.0)

    extension = SimpleNamespace(
        forward_hao_direct_fp8pv=launch,
        read_hao_direct_topology=lambda: populated_topology,
    )
    prelaunch_topology = {**populated_topology, "valid": 0}
    identity = {"path": "/tmp/fp8.so", "sha256": "a" * 64, "bytes": 1}
    runtime_type.install_diagnostic_fp8_lse_control(
        runtime,
        extension,
        prelaunch_topology,
        identity,
    )
    assert runtime.diagnostic_fp8_lse_loaded_artifact_identity == identity
    assert runtime.diagnostic_fp8_lse_substitution_mode == "all_rows"

    diagnostic_config = SimpleNamespace(
        batch=1,
        sequence=4,
        q_heads=4,
        kv_heads=2,
        head_dim=8,
    )
    qk_operands = (torch.ones(1), torch.ones(1))
    diagnostic_runtime = SimpleNamespace(
        config=diagnostic_config,
        diagnostic_fp8_lse_entrypoint=launch,
        diagnostic_fp8_lse_first_launch_receipt=None,
        diagnostic_fp8_lse_runtime_authenticated=True,
        diagnostic_fp8_lse_substitution_mode="all_rows",
        diagnostic_fp8_lse_substitution_counts={
            "control_launches": 0,
            "control_lse_entries_computed": 0,
            "mx_finite_entries_seen": 0,
            "mx_nonfinite_entries_seen": 0,
            "mx_nan_entries_seen": 0,
            "mx_posinf_entries_seen": 0,
            "mx_neginf_entries_seen": 0,
            "fp8_entries_substituted": 0,
            "mx_entries_retained": 0,
        },
        diagnostic_mx_qk_abi_identity=tuple(
            namespace["_tensor_abi_identity"](tensor)
            for tensor in qk_operands
        ),
    )
    qkv = SimpleNamespace(
        v_forward_fp8=None,
        v_backward_fp8=torch.ones(
            1,
            4,
            2,
            8,
            dtype=torch.float8_e4m3fn,
        ),
        qk_forward_operands=lambda: qk_operands,
    )
    lse = runtime_type.diagnostic_fp8_lse(
        diagnostic_runtime,
        qkv,
        torch.zeros(1, 4, 4, 8, dtype=torch.bfloat16),
        torch.zeros(1, 4, 1, 4, dtype=torch.float32),
    )
    assert torch.equal(lse, torch.full((1, 4, 1, 4), 7.0))
    receipt = diagnostic_runtime.diagnostic_fp8_lse_first_launch_receipt
    assert receipt["same_qk_operand_storage_as_mx_launch"] is True
    assert receipt["mx_output_bitwise_unchanged"] is True
    assert receipt["synthesized_v"]["shape"] == [1, 2, 8, 4]
    assert receipt["mx_vs_fp8_lse"]["mx_all_finite"] is True
    assert receipt["mx_vs_fp8_lse"]["fp8_all_finite"] is True
    assert receipt["mx_vs_fp8_lse"]["relative_l2_over_fp8"] == 1.0
    assert receipt["substitution"] == {
        "mode": "all_rows",
        "semantics": "substitute_authenticated_fp8_control_lse_for_all_rows",
        "selection_policy_allowlisted_at_install": True,
        "control_launch_computes_all_rows": True,
        "total_entries": 16,
        "mx_finite_entries": 16,
        "mx_nonfinite_entries": 0,
        "fp8_entries_substituted": 16,
        "mx_entries_retained": 0,
        "selected_lse_all_finite": True,
    }
    assert diagnostic_runtime.diagnostic_fp8_lse_substitution_counts == {
        "control_launches": 1,
        "control_lse_entries_computed": 16,
        "mx_finite_entries_seen": 16,
        "mx_nonfinite_entries_seen": 0,
        "mx_nan_entries_seen": 0,
        "mx_posinf_entries_seen": 0,
        "mx_neginf_entries_seen": 0,
        "fp8_entries_substituted": 16,
        "mx_entries_retained": 0,
    }
    assert receipt["mx_vs_fp8_lse"][
        "least_squares_affine_mx_to_fp8"
    ]["intercept"] == 7.0
    assert "data_ptr" not in receipt["qk_operands"][0]

    wrong = SimpleNamespace(
        is_d128=True,
        pv_format="e4m3_fp8",
        config=config,
        diagnostic_fp8_lse_entrypoint=None,
    )
    with pytest.raises(ValueError, match="D128 MXFP4-PV"):
        runtime_type.install_diagnostic_fp8_lse_control(
            wrong,
            extension,
            prelaunch_topology,
            identity,
        )


def test_d128_mx_diagnostic_fp8_lse_can_replace_only_nonfinite_rows() -> None:
    namespace = _namespace(
        "_require_forward_topology",
        "_tensor_abi_identity",
        "LowpAttentionRuntime",
    )
    runtime_type = namespace["LowpAttentionRuntime"]

    def launch(*arguments: object) -> None:
        output = arguments[-5]
        lse = arguments[-4]
        assert isinstance(output, torch.Tensor)
        assert isinstance(lse, torch.Tensor)
        output.fill_(99.0)
        lse.fill_(7.0)

    config = SimpleNamespace(
        batch=1,
        sequence=4,
        q_heads=1,
        kv_heads=1,
        head_dim=8,
    )
    qk_operands = (torch.ones(1), torch.ones(1))
    runtime = SimpleNamespace(
        config=config,
        diagnostic_fp8_lse_entrypoint=launch,
        diagnostic_fp8_lse_first_launch_receipt=None,
        diagnostic_fp8_lse_runtime_authenticated=True,
        diagnostic_fp8_lse_substitution_mode="mx_nonfinite_only",
        diagnostic_fp8_lse_substitution_counts={
            "control_launches": 0,
            "control_lse_entries_computed": 0,
            "mx_finite_entries_seen": 0,
            "mx_nonfinite_entries_seen": 0,
            "mx_nan_entries_seen": 0,
            "mx_posinf_entries_seen": 0,
            "mx_neginf_entries_seen": 0,
            "fp8_entries_substituted": 0,
            "mx_entries_retained": 0,
        },
        diagnostic_mx_qk_abi_identity=tuple(
            namespace["_tensor_abi_identity"](tensor)
            for tensor in qk_operands
        ),
    )
    qkv = SimpleNamespace(
        v_forward_fp8=None,
        v_backward_fp8=torch.ones(
            1,
            4,
            1,
            8,
            dtype=torch.float8_e4m3fn,
        ),
        qk_forward_operands=lambda: qk_operands,
    )
    mx_lse = torch.tensor(
        [[[[1.0, float("nan"), float("inf"), float("-inf")]]]],
        dtype=torch.float32,
    )
    selected = runtime_type.diagnostic_fp8_lse(
        runtime,
        qkv,
        torch.zeros(1, 4, 1, 8, dtype=torch.bfloat16),
        mx_lse,
    )

    assert torch.equal(selected, torch.tensor([[[[1.0, 7.0, 7.0, 7.0]]]]))
    assert runtime.diagnostic_fp8_lse_substitution_counts == {
        "control_launches": 1,
        "control_lse_entries_computed": 4,
        "mx_finite_entries_seen": 1,
        "mx_nonfinite_entries_seen": 3,
        "mx_nan_entries_seen": 1,
        "mx_posinf_entries_seen": 1,
        "mx_neginf_entries_seen": 1,
        "fp8_entries_substituted": 3,
        "mx_entries_retained": 1,
    }
    receipt = runtime.diagnostic_fp8_lse_first_launch_receipt
    assert receipt["substitution"]["mode"] == "mx_nonfinite_only"
    assert receipt["substitution"]["mx_nonfinite_entries"] == 3
    assert receipt["substitution"]["fp8_entries_substituted"] == 3
    assert receipt["substitution"]["mx_entries_retained"] == 1


def test_d128_mx_diagnostic_fp8_lse_rejects_unknown_selection_mode() -> None:
    namespace = _namespace(
        "_require_forward_topology",
        "LowpAttentionRuntime",
    )
    runtime_type = namespace["LowpAttentionRuntime"]
    config = _d128_batched_config(2)
    runtime = SimpleNamespace(
        is_d128=True,
        pv_format="mxfp4_e8m0_block32",
        config=config,
        diagnostic_fp8_lse_entrypoint=None,
    )
    extension = SimpleNamespace(forward_hao_direct_fp8pv=lambda *args: None)
    with pytest.raises(ValueError, match="unsupported diagnostic FP8-LSE"):
        runtime_type.install_diagnostic_fp8_lse_control(
            runtime,
            extension,
            _d128_fp8_topology(batch=2),
            {"path": "/tmp/fp8.so", "sha256": "a" * 64, "bytes": 1},
            substitution_mode="finite_only",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("route", "wrong_route"),
        ("schema", "wrong_schema"),
        ("pv_format", "mxfp4_e8m0_block32"),
        ("shiftless_fp8_mode", 5),
        ("fixed_route_fastpath", False),
        ("fixed_p_ceiling", True),
        ("score_pack_ceiling", True),
    ),
)
@pytest.mark.parametrize("batch", (2, 8, 16))
def test_batched_forward_topology_rejects_every_unsafe_fixed_route_field(
    batch: int,
    field: str,
    value: object,
) -> None:
    require = _namespace("_require_forward_topology")[
        "_require_forward_topology"
    ]
    topology = _exact_topology(batch=batch)
    topology[field] = value
    with pytest.raises(ValueError, match="batched exact forward topology"):
        require(_d64_batched_config(batch), topology)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("route", "wrong_route"),
        ("schema", "wrong_schema"),
        ("pv_format", "e4m3_fp8"),
        ("causal_interleaved_kv", False),
        ("fixed_route_fastpath", False),
        ("fixed_p_ceiling", True),
        ("score_pack_ceiling", True),
        ("d64_detached_p", False),
        ("mx_mode23_native_density", 3),
        ("mx_mode23_native_quarter_mask", 15),
        ("mx_global_anchor32", False),
        ("mx_anchor_affine_hoist", False),
        ("mx_global_anchor_margin_log2", 0),
        ("mx_stored_scale_shift_log2", 8),
    ),
)
@pytest.mark.parametrize("batch", (2, 8, 16))
def test_batched_mx_topology_rejects_every_unreviewed_policy_field(
    batch: int,
    field: str,
    value: object,
) -> None:
    require = _namespace("_require_forward_topology")[
        "_require_forward_topology"
    ]
    topology = _mx_topology(batch=batch)
    topology[field] = value
    with pytest.raises(ValueError, match="batched exact forward topology"):
        require(_d64_batched_config(batch), topology)


@pytest.mark.parametrize("batch", (2, 8, 16))
def test_batched_forward_topology_requires_runtime_valid_after_launch(
    batch: int,
) -> None:
    require = _namespace("_require_forward_topology")[
        "_require_forward_topology"
    ]
    topology = _exact_topology(batch=batch)
    topology["valid"] = 0
    require(_d64_batched_config(batch), topology)
    with pytest.raises(ValueError, match="valid=0"):
        require(
            _d64_batched_config(batch), topology, runtime_populated=True
        )
    topology["valid"] = 1
    require(_d64_batched_config(batch), topology, runtime_populated=True)


@pytest.mark.parametrize("batch", (2, 8, 16))
def test_batched_requires_complete_authenticated_precomposed_control(
    tmp_path: Path,
    batch: int,
) -> None:
    namespace = _namespace(
        "Config",
        "_source_content_identity",
        "_require_precomposed_backward_control",
    )
    config = namespace["Config"](batch=batch)
    require = namespace["_require_precomposed_backward_control"]
    control = tmp_path / "control.py"
    control.write_text("control = 'exact'\n")
    payload = control.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    with pytest.raises(ValueError, match="requires an authenticated"):
        require(config, None, None, None)
    with pytest.raises(ValueError, match="requires source"):
        require(config, control, None, None)
    with pytest.raises(ValueError, match="identity mismatch"):
        require(config, control, "0" * 64, len(payload))
    assert require(config, control, digest, len(payload)) == {
        "bytes": len(payload),
        "sha256": digest,
    }


def test_b1_preserves_implicit_control_but_rejects_partial_identity() -> None:
    namespace = _namespace(
        "Config",
        "_source_content_identity",
        "_require_precomposed_backward_control",
    )
    config = namespace["Config"](batch=1)
    require = namespace["_require_precomposed_backward_control"]
    assert require(config, None, None, None) is None
    with pytest.raises(ValueError, match="requires source"):
        require(config, Path("unused"), None, None)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("projection_dgrad", "nvfp4"),
        ("qkv_projection_format", "nvfp4"),
        ("backward_exp2_degree", 2),
        ("backward_exp2_period", None),
        ("backward_exp2_period", 0),
        ("backward_fp8_ds_lift", 256),
        ("backward_reuse_quantized_p", True),
        ("backward_forward_mx_probability_replay", True),
        ("backward_forward_mx_probability_scale_handoff", True),
        ("backward_match_forward_operands", False),
        ("per_block_qk_scales", False),
        ("experimental_split_v_backward", True),
        ("backward_probability_correction", 0.5),
        ("q_quant_scale", 2.0),
        ("k_quant_scale", 1.5),
        ("projection_weight_scale_2d", False),
        ("v_mxfp4_scale_2d", True),
        ("adaptive_qk_weight_scales", True),
        ("shared_runtime", object()),
    ),
)
@pytest.mark.parametrize("batch", (2, 8, 16))
def test_batched_runtime_rejects_every_unverified_policy(
    batch: int,
    field: str,
    value: object,
) -> None:
    require = _namespace("_require_batched_exact_runtime_contract")[
        "_require_batched_exact_runtime_contract"
    ]
    kwargs = _exact_runtime_kwargs()
    kwargs[field] = value
    with pytest.raises(ValueError, match="authenticated D64 route"):
        require(
            _d64_batched_config(batch),
            _exact_topology(batch=batch),
            **kwargs,
        )


@pytest.mark.parametrize("batch", (2, 8, 16))
def test_batched_runtime_accepts_only_exact_fp8_pv_topology(
    batch: int,
) -> None:
    require = _namespace("_require_batched_exact_runtime_contract")[
        "_require_batched_exact_runtime_contract"
    ]
    require(
        _d64_batched_config(batch),
        _exact_topology(batch=batch),
        **_exact_runtime_kwargs(),
    )
    for key, value in (
        ("pv_format", "mxfp4_e8m0_block32"),
        ("shiftless_fp8_mode", 5),
        ("causal_interleaved_kv", True),
    ):
        topology = _exact_topology(batch=batch)
        topology[key] = value
        with pytest.raises(ValueError, match="authenticated D64 route"):
            require(
                _d64_batched_config(batch),
                topology,
                **_exact_runtime_kwargs(),
            )


@pytest.mark.parametrize("batch", (2, 8, 16))
def test_batched_runtime_accepts_mx_forward_with_identical_e4m3_v_backward(
    batch: int,
) -> None:
    require = _namespace("_require_batched_exact_runtime_contract")[
        "_require_batched_exact_runtime_contract"
    ]
    require(
        _d64_batched_config(batch),
        _mx_topology(batch=batch),
        **_mx_runtime_kwargs(),
    )
    kwargs = _mx_runtime_kwargs()
    kwargs["experimental_split_v_backward"] = False
    with pytest.raises(ValueError, match="authenticated D64 route"):
        require(
            _d64_batched_config(batch),
            _mx_topology(batch=batch),
            **kwargs,
        )


def test_experimental_native_runtime_accepts_only_b16_exact_routes() -> None:
    require = _namespace(
        "_require_experimental_native_batched_runtime_contract"
    )["_require_experimental_native_batched_runtime_contract"]
    require(
        _d64_batched_config(16),
        _exact_topology(batch=16),
        **_native_runtime_kwargs(),
    )
    require(
        _d64_batched_config(16),
        _mx_topology(batch=16),
        **_native_runtime_kwargs(publish_mxfp4_v=True),
    )
    with pytest.raises(ValueError, match="native NVFP4 B16.*B16"):
        require(
            _d64_batched_config(8),
            _exact_topology(batch=8),
            **_native_runtime_kwargs(),
        )


@pytest.mark.parametrize("topology", (_d128_fp8_topology(), _d128_mx_topology()))
def test_experimental_native_runtime_accepts_d128_b2(
    topology: dict[str, object],
) -> None:
    require = _namespace(
        "_require_experimental_native_batched_runtime_contract"
    )["_require_experimental_native_batched_runtime_contract"]
    require(
        _d128_batched_config(),
        topology,
        **_d128_native_runtime_kwargs(),
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("projection_dgrad", "bf16"),
        ("backward_exp2_period", 2),
        ("backward_reuse_quantized_p", False),
        ("per_block_qk_scales", False),
        ("experimental_split_v_backward", True),
    ),
)
def test_experimental_native_d128_b2_rejects_unverified_policy(
    field: str,
    value: object,
) -> None:
    require = _namespace(
        "_require_experimental_native_batched_runtime_contract"
    )["_require_experimental_native_batched_runtime_contract"]
    kwargs = _d128_native_runtime_kwargs()
    kwargs[field] = value
    with pytest.raises(ValueError, match="native NVFP4 D128 B2"):
        require(_d128_batched_config(), _d128_fp8_topology(), **kwargs)


def test_experimental_native_d128_b2_represents_qk_for_fp8_pv_only() -> None:
    require = _namespace(
        "_require_experimental_native_batched_runtime_contract"
    )["_require_experimental_native_batched_runtime_contract"]
    kwargs = _d128_native_runtime_kwargs()
    kwargs["backward_match_forward_operands"] = True

    require(_d128_batched_config(), _d128_fp8_topology(), **kwargs)
    with pytest.raises(ValueError, match="native NVFP4 D128 B2"):
        require(_d128_batched_config(), _d128_mx_topology(), **kwargs)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("projection_dgrad", "nvfp4"),
        ("qkv_projection_format", "e4m3"),
        ("backward_exp2_degree", 2),
        ("backward_exp2_period", None),
        ("backward_fp8_ds_lift", 256),
        ("backward_reuse_quantized_p", True),
        ("backward_forward_mx_probability_replay", True),
        ("backward_forward_mx_probability_scale_handoff", True),
        ("backward_match_forward_operands", False),
        ("per_block_qk_scales", False),
        ("experimental_split_v_backward", True),
        ("backward_probability_correction", 0.5),
        ("q_quant_scale", 2.0),
        ("k_quant_scale", 1.5),
        ("projection_weight_scale_2d", False),
        ("v_mxfp4_scale_2d", True),
        ("adaptive_qk_weight_scales", True),
    ),
)
def test_experimental_native_b16_rejects_every_unverified_policy(
    field: str,
    value: object,
) -> None:
    require = _namespace(
        "_require_experimental_native_batched_runtime_contract"
    )["_require_experimental_native_batched_runtime_contract"]
    kwargs = _native_runtime_kwargs()
    kwargs[field] = value
    with pytest.raises(ValueError, match="experimental native NVFP4 B16"):
        require(
            _d64_batched_config(16),
            _exact_topology(batch=16),
            **kwargs,
        )


def test_experimental_native_b16_coarse_gate_allows_shared_backward() -> None:
    require = _namespace(
        "_require_experimental_native_batched_runtime_contract"
    )["_require_experimental_native_batched_runtime_contract"]
    kwargs = _native_runtime_kwargs()
    # The runtime constructor subsequently requires complete logical-contract
    # equality and exact physical runner/storage identity.  This coarse shape
    # and policy gate must no longer reject sharing before those checks run.
    kwargs["shared_runtime"] = object()
    require(
        _d64_batched_config(16),
        _exact_topology(batch=16),
        **kwargs,
    )


def test_controller_flattens_bs_once_and_launches_one_batched_attention() -> None:
    source = RUNTIME.read_text()
    module = ast.parse(source)
    function_class = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef)
        and node.name == "_LowpAttentionFunction"
    )
    forward = next(
        node
        for node in function_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    attention_launches = [
        node
        for node in ast.walk(forward)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_run_lowp_forward_attention"
    ]
    assert len(attention_launches) == 1
    body = ast.get_source_segment(source, function_class)
    assert body is not None
    assert "token_rows = c.batch * c.sequence" in body
    assert body.count("batch=c.batch") >= 3
    assert "output.reshape(token_rows, c.q_width)" in body
    assert "dx_matrix.reshape(c.batch, c.sequence, c.hidden)" in body
    assert "batch=1" not in body


def test_runtime_and_workspace_allocate_every_publication_for_config_batch() -> None:
    source = RUNTIME.read_text()
    runtime = source.split("class LowpAttentionRuntime:", 1)[1].split(
        "class _LowpAttentionFunction", 1
    )[0]
    layer = source.split("class LowpAttention(nn.Module):", 1)[1].split(
        "class MLP", 1
    )[0]
    assert "batch=config.batch" in runtime
    assert runtime.count("config.batch,") >= 5
    allocator = layer.split("def _allocate_forward_workspace(", 1)[1].split(
        "def _apply(", 1
    )[0]
    assert allocator.count("config.batch,") == 14
    assert '"batch": self.config.batch' in runtime


def test_cli_plumbs_authenticated_control_and_exact_batched_policy() -> None:
    source = RUNTIME.read_text()
    main = source.split("def main() -> None:", 1)[1]
    for flag in (
        "--backward-control-source",
        "--backward-control-sha256",
        "--backward-control-bytes",
    ):
        assert flag in main
    assert "authenticated_backward_control = (" not in main
    assert "_require_precomposed_backward_control(" in main
    assert "backward_control_source=args.backward_control_source" in main
    assert "backward_control_sha256=args.backward_control_sha256" in main
    assert "backward_control_bytes=args.backward_control_bytes" in main
    assert "_requested_backward_approximation_policy(config)" in main
    assert "backward_exp2_degree=requested_exp2_degree" in main
    assert "backward_exp2_period=requested_exp2_period" in main
    assert "backward_reuse_quantized_p=(" in main
    assert (
        "requested_reuse_quantized_p and not native_tk_d128_backward" in main
    )
    assert "batched_mx_split_v_backward = (" in main
    assert (
        "experimental_split_v_backward=batched_mx_split_v_backward" in main
    )


def test_validator_pins_b2_b8_b16_and_targets_b16() -> None:
    source = VALIDATOR.read_text()
    expected = {
        2: (
            "_C_cfwd_fp8exact0_b2_s4096h32kv8d64_sm100_20260825.so",
            "4e4c4c9b1afd7a751c3bae9d734f617a04b0b95778370deba9be3f131f5e05d1",
        ),
        8: (
            "_C_cfwd_fp8exact0_b8_s4096h32kv8d64_sm100_20260825.so",
            "34114089ab4631093dc2b4dbd38e01a597a6608c9cfb748bd927f8038271db9d",
        ),
        16: (
            "_C_cfwd_fp8exact0_b16_s4096h32kv8d64_sm100_topofix_b200_20260825.so",
            "88d81d3783e5aa80f0e9cf259a2ea7c935da4c2a5dc3ba1868e63f802a2c6208",
        ),
    }
    for path, digest in expected.values():
        assert path in source
        assert digest in source
    assert "choices=(2, 8, 16)" in source
    assert "default=16" in source
    assert "configured_batch != 1" in source


def test_validator_authenticates_projection_before_lazy_runtime_imports() -> None:
    source = VALIDATOR.read_text()
    tree = ast.parse(source)
    top_level_imports = {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
    } | {
        alias.name
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "torch" not in top_level_imports
    assert "torch.nn.functional" not in top_level_imports
    assert "tk_fa4" not in top_level_imports
    assert not any(
        module is not None and module.startswith("tk_fa4")
        for module in top_level_imports
    )
    main = source.split("def main() -> None:", 1)[1]
    assert main.index("_authenticate_projection_environment()") < main.index(
        "_load_runtime_imports()"
    )
    assert main.index("_load_runtime_imports()") < main.index(
        "torch.cuda.set_device(0)"
    )


def test_validator_help_needs_no_projection_or_runtime_import() -> None:
    environment = dict(os.environ)
    environment.pop("TK_FA4_LOWP_BWD_EXTENSION_SOURCE", None)
    completed = subprocess.run(
        [sys.executable, str(VALIDATOR), "--help"],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--batch {2,8,16}" in completed.stdout


def test_validator_projection_preflight_is_stdlib_only_and_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _validator_namespace()
    authenticate = namespace["_authenticate_projection_environment"]
    module_globals = authenticate.__globals__
    assert "torch" not in module_globals
    assert "tk_interface" not in module_globals
    assert not module_globals["_RUNTIME_IMPORTS_LOADED"]

    projection = tmp_path / "projection.so"
    payload = b"canonical-projection-payload\n"
    projection.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    module_globals["PROJECTION"] = projection
    module_globals["COMMON_ARTIFACTS"]["projection"] = (
        projection,
        digest,
        len(payload),
    )
    monkeypatch.setenv(
        "TK_FA4_LOWP_BWD_EXTENSION_SOURCE",
        str(projection),
    )
    identity = authenticate()
    assert identity == {
        "path": str(projection.resolve()),
        "sha256": digest,
        "bytes": len(payload),
    }
    assert "torch" not in module_globals
    assert "tk_interface" not in module_globals

    projection.write_bytes(b"tampered-projection-payload\n")
    with pytest.raises(RuntimeError, match="byte-count mismatch|SHA-256 mismatch"):
        authenticate()


def test_validator_projection_preflight_rejects_missing_wrong_and_symlink_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _validator_namespace()
    authenticate = namespace["_authenticate_projection_environment"]
    module_globals = authenticate.__globals__
    projection = tmp_path / "projection.so"
    projection.write_bytes(b"payload")
    digest = hashlib.sha256(b"payload").hexdigest()
    module_globals["PROJECTION"] = projection
    module_globals["COMMON_ARTIFACTS"]["projection"] = (
        projection,
        digest,
        len(b"payload"),
    )

    monkeypatch.delenv("TK_FA4_LOWP_BWD_EXTENSION_SOURCE", raising=False)
    with pytest.raises(RuntimeError, match="must name the pinned projection"):
        authenticate()

    wrong = tmp_path / "wrong.so"
    wrong.write_bytes(b"payload")
    monkeypatch.setenv("TK_FA4_LOWP_BWD_EXTENSION_SOURCE", str(wrong))
    with pytest.raises(RuntimeError, match="must name the canonical"):
        authenticate()

    link = tmp_path / "projection-link.so"
    link.symlink_to(projection)
    monkeypatch.setenv("TK_FA4_LOWP_BWD_EXTENSION_SOURCE", str(link))
    with pytest.raises(RuntimeError, match="non-symlink"):
        authenticate()


def test_validator_v4_authenticates_stages_before_output_projection() -> None:
    source = VALIDATOR.read_text()
    assert "llama12b_d64_exact_batched_controller_gate_v4" in source
    assert '"rmsnorm": normalized' in source
    for publication in (
        "qk_policy_scales",
        "backward_qk_scales",
        "q_payload",
        "q_forward_scales",
        "q_forward_global_scale",
        "k_payload",
        "k_forward_scales",
        "k_forward_global_scale",
        "v_forward_fp8",
        "q_backward_fp8",
        "k_backward_fp8",
        "v_backward_fp8",
        "raw_attention",
        "lse",
    ):
        assert f'"{publication}"' in source
    assert '.to(device="cpu", copy=True).contiguous()' in source
    assert '"qkv_publications_byte_equal"' in source
    assert '"raw_attention_byte_equal"' in source
    assert '"lse_byte_equal"' in source


def test_validator_v4_compares_both_output_packs_to_one_bf16_reference() -> None:
    source = VALIDATOR.read_text()
    assert "bf16_output_projection = F.linear(" in source
    assert '"batched_vs_bf16"' in source
    assert '"sequential_b1_vs_bf16"' in source
    assert "OUTPUT_PROJECTION_RELATIVE_L2_TOLERANCE = 0.002" in source
    assert "OUTPUT_PROJECTION_COSINE_TOLERANCE = 0.001" in source
    assert (
        '"batched_output_projection_relative_l2_no_worse_than_b1_plus_0.002"'
        in source
    )
    assert (
        '"batched_output_projection_cosine_no_worse_than_b1_minus_0.001"'
        in source
    )
    assert '"output_relative_l2_at_most_0.01"' not in source
    assert '"input_gradient_relative_l2_at_most_0.10"' not in source


def test_validator_v4_reports_reserved_hbm_and_matched_bf16_timing() -> None:
    source = VALIDATOR.read_text()
    assert source.count("torch.cuda.max_memory_reserved()") == 3
    assert source.count('"peak_reserved_bytes"') == 3
    assert 'bf16_route = f"bf16_b{batch}"' in source
    assert "bf16_layer.load_state_dict(owner.state_dict(), strict=True)" in source
    assert '"exact_batched_speedup_over_bf16_batched"' in source
    assert '"exact_batched_to_bf16_batched_ratio"' in source
    assert (
        "for key in (\"forward_ms\", \"backward_ms\", "
        "\"optimizer_ms\", \"step_ms\")"
        in source
    )


def test_validator_v4_has_absolute_projection_and_full_step_bf16_gates() -> None:
    source = VALIDATOR.read_text()
    assert "OUTPUT_PROJECTION_ABSOLUTE_RELATIVE_L2_CEILING = 0.16" in source
    assert "OUTPUT_PROJECTION_ABSOLUTE_COSINE_FLOOR = 0.985" in source
    assert '"batched_output_projection_relative_l2_at_most_0.16"' in source
    assert '"batched_output_projection_cosine_at_least_0.985"' in source
    assert '"exact_vs_bf16_numerics"' in source
    assert '"sequential_b1_vs_bf16_numerics"' in source
    for gate in (
        "exact_vs_bf16_output_relative_l2_at_most_0.25",
        "exact_vs_bf16_output_cosine_at_least_0.97",
        "exact_vs_bf16_input_gradient_relative_l2_at_most_0.70",
        "exact_vs_bf16_input_gradient_cosine_at_least_0.80",
        "exact_vs_bf16_parameter_gradient_relative_l2_at_most_0.55",
        "exact_vs_bf16_parameter_gradient_cosine_at_least_0.85",
        "exact_vs_bf16_parameter_gradient_worst_relative_l2_at_most_0.80",
        "exact_vs_bf16_post_update_relative_l2_at_most_0.005",
        "exact_vs_bf16_post_update_cosine_at_least_0.99999",
        "exact_vs_bf16_post_update_worst_relative_l2_at_most_0.006",
    ):
        assert f'"{gate}"' in source
    assert '"gate_policy"' in source
    assert '"rationale"' in source


def test_validator_v4_gates_batch_nonregression_against_same_bf16() -> None:
    source = VALIDATOR.read_text()
    for gate in (
        "batched_output_relative_l2_no_worse_than_b1_plus_0.02",
        "batched_output_cosine_no_worse_than_b1_minus_0.01",
        "batched_input_gradient_relative_l2_no_worse_than_b1_plus_0.10",
        "batched_input_gradient_cosine_no_worse_than_b1_minus_0.05",
        "batched_parameter_gradient_relative_l2_no_worse_than_b1_plus_0.10",
        "batched_parameter_gradient_cosine_no_worse_than_b1_minus_0.05",
        "batched_post_update_relative_l2_no_worse_than_b1_plus_0.002",
    ):
        assert f'"{gate}"' in source
    string_literals = {
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert not any(value.startswith("b16_") for value in string_literals)


def test_validator_v4_rotates_timing_order_and_requires_real_bf16_win() -> None:
    source = VALIDATOR.read_text()
    assert "MATCHED_BF16_SPEEDUP_FLOOR = 1.02" in source
    assert (
        '"exact_batched_speedup_over_bf16_batched_at_least_1.02"' in source
    )
    assert "sample_orders = tuple(" in source
    assert '"execution_order": list(order)' in source
    assert '"position": position' in source
    assert '"order_control": "three cyclic route-order shifts"' in source


def test_lowp_projection_dgrad_rescale_is_in_place_bf16() -> None:
    source = RUNTIME.read_text()
    assert "dx_matrix = dx_scaled.mul_(scale)" in source
    assert "(dx_scaled.float() * scale).bfloat16()" not in source

    scale = 2.0**-16
    values = torch.tensor(
        [0.0, 1.0, -1.0, 0.00390625, -448.0, 65504.0],
        dtype=torch.bfloat16,
    )
    expected = (values.float() * scale).bfloat16()
    storage_pointer = values.data_ptr()
    actual = values.mul_(scale)
    assert actual.data_ptr() == storage_pointer
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "scale",
    (2.0**-16, 2.0**-8, 0.75, 1.3, 2.0**7),
)
def test_lowp_projection_dgrad_rescale_matches_fp32_rounding_on_cuda(
    scale: float,
) -> None:
    bit_patterns = torch.arange(65536, dtype=torch.int32).to(torch.uint16)
    source = bit_patterns.view(torch.bfloat16).cuda()
    expected = (source.float() * scale).bfloat16()
    actual = source.clone()
    storage_pointer = actual.data_ptr()

    actual.mul_(scale)

    assert actual.data_ptr() == storage_pointer
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
