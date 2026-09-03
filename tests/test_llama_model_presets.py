from __future__ import annotations

import argparse
import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = (
    ROOT / "tk_fa4" / "lowp_fa4_bwd" / "benchmark_llama12b_e2e.py"
)
TRAINER = (
    ROOT / "tk_fa4" / "lowp_fa4_bwd" / "train_llama12b_real_tokens.py"
)


def _selected_definitions(
    path: Path,
    names: tuple[str, ...],
) -> list[ast.stmt]:
    selected = []
    for node in ast.parse(path.read_text()).body:
        name = getattr(node, "name", None)
        targets = (
            [target.id for target in node.targets if isinstance(target, ast.Name)]
            if isinstance(node, ast.Assign)
            else []
        )
        if name in names or any(target in names for target in targets):
            selected.append(node)
    return selected


def _model_config_namespace() -> dict[str, Any]:
    names = (
        "Config",
        "DEFAULT_MODEL_PRESET",
        "MODEL_PRESETS",
        "config_from_model_preset",
    )
    module = ast.Module(
        body=_selected_definitions(BENCHMARK, names),
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {"dataclass": dataclass}
    exec(compile(module, str(BENCHMARK), "exec"), namespace)
    return namespace


def _trainer_resolution_namespace() -> dict[str, Any]:
    names = ("_argument_was_provided", "_resolve_model_preset_options")
    module = ast.Module(
        body=_selected_definitions(TRAINER, names),
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {
        "argparse": argparse,
        "Config": object,
    }
    exec(compile(module, str(TRAINER), "exec"), namespace)
    return namespace


def _trainer_comparison_namespace() -> dict[str, Any]:
    definitions = _selected_definitions(
        TRAINER,
        ("_comparisons_against_bf16",),
    )
    module = ast.Module(body=definitions, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {"Any": Any}
    exec(compile(module, str(TRAINER), "exec"), namespace)
    return namespace


def _route_arguments() -> SimpleNamespace:
    return SimpleNamespace(
        rope_theta=500_000.0,
        rope_factor=32.0,
        backward_exp2_degree=2,
        backward_exp2_period=None,
        mx_backward_reuse_quantized_p=False,
        fp8_backward_reuse_quantized_p=False,
        v_mxfp4_scaling="2d",
        mx_qkv_projection_format="e4m3",
        fp8_qkv_projection_format="e4m3",
        mx_backward_match_forward_operands=True,
        fp8_backward_match_forward_operands=True,
        projection_weight_scaling="1d",
        mx_per_block_qk_scales=True,
        fp8_per_block_qk_scales=True,
        mx_experimental_split_v_backward=True,
        mx_backward_forward_probability_replay=False,
        mx_backward_forward_probability_scale_handoff=False,
    )


def test_default_preset_preserves_the_existing_12b_shape() -> None:
    namespace = _model_config_namespace()
    config = namespace["config_from_model_preset"]()
    assert config.model_preset == "llama3.2-1b"
    assert (
        config.layers,
        config.hidden,
        config.intermediate,
        config.q_heads,
        config.kv_heads,
        config.head_dim,
        config.vocab,
    ) == (16, 2048, 8192, 32, 8, 64, 128256)
    assert config.tie_word_embeddings is True
    assert config.rope_theta == 500_000.0
    assert config.rope_factor == 32.0
    assert config.parameter_count == 1_235_814_400


def test_llama31_8b_preset_matches_the_verified_local_config() -> None:
    namespace = _model_config_namespace()
    config = namespace["config_from_model_preset"]("llama3.1-8b")
    assert (
        config.layers,
        config.full_model_layers,
        config.hidden,
        config.intermediate,
        config.q_heads,
        config.kv_heads,
        config.head_dim,
        config.vocab,
    ) == (32, 32, 4096, 14336, 32, 8, 128, 128256)
    assert config.tie_word_embeddings is False
    assert config.max_position_embeddings == 131072
    assert config.rope_theta == 500_000.0
    assert config.rope_factor == 8.0
    assert config.rope_low_frequency_factor == 1.0
    assert config.rope_high_frequency_factor == 4.0
    assert config.rope_original_context == 8192
    assert config.parameter_count == 8_030_261_248


def test_preset_depth_override_is_smoke_only_and_bounded() -> None:
    namespace = _model_config_namespace()
    make_config = namespace["config_from_model_preset"]
    assert make_config("llama3.1-8b", layers=2).layers == 2
    with pytest.raises(ValueError, match="layers must be in"):
        make_config("llama3.1-8b", layers=33)
    with pytest.raises(ValueError, match="unknown model preset"):
        make_config("llama-unknown")


def test_d128_trainer_defaults_to_the_supported_projection_contract() -> None:
    namespace = _trainer_resolution_namespace()
    args = _route_arguments()
    args.mx_per_block_qk_scales = False
    args.fp8_per_block_qk_scales = False
    config = SimpleNamespace(
        head_dim=128,
        rope_theta=500_000.0,
        rope_factor=8.0,
    )
    namespace["_resolve_model_preset_options"](args, config, [])
    assert args.rope_factor == 8.0
    assert args.backward_exp2_degree == 1
    assert args.backward_exp2_period == 0
    assert args.mx_backward_reuse_quantized_p is True
    assert args.fp8_backward_reuse_quantized_p is True
    assert args.v_mxfp4_scaling == "1d"
    assert args.mx_qkv_projection_format == "nvfp4"
    assert args.fp8_qkv_projection_format == "nvfp4"
    assert args.mx_backward_match_forward_operands is False
    assert args.fp8_backward_match_forward_operands is False
    assert args.projection_weight_scaling == "2d"
    assert args.mx_per_block_qk_scales is True
    assert args.fp8_per_block_qk_scales is True
    assert args.mx_experimental_split_v_backward is False

    perblock = _route_arguments()
    perblock.mx_per_block_qk_scales = True
    perblock.fp8_per_block_qk_scales = True
    namespace["_resolve_model_preset_options"](
        perblock,
        config,
        ["--mx-per-block-qk-scales", "--fp8-per-block-qk-scales"],
    )
    assert perblock.mx_per_block_qk_scales is True
    assert perblock.fp8_per_block_qk_scales is True


@pytest.mark.parametrize("route", ("mx", "fp8"))
def test_d128_trainer_rejects_disabling_per_block_qk_scales(route: str) -> None:
    namespace = _trainer_resolution_namespace()
    args = _route_arguments()
    setattr(args, f"{route}_per_block_qk_scales", False)
    config = SimpleNamespace(
        head_dim=128,
        rope_theta=500_000.0,
        rope_factor=8.0,
    )
    with pytest.raises(ValueError, match="incompatible with the D128"):
        namespace["_resolve_model_preset_options"](
            args,
            config,
            [f"--no-{route}-per-block-qk-scales"],
        )


def test_d64_trainer_leaves_per_block_qk_scales_unchanged() -> None:
    namespace = _trainer_resolution_namespace()
    args = _route_arguments()
    args.mx_per_block_qk_scales = False
    config = SimpleNamespace(
        head_dim=64,
        rope_theta=500_000.0,
        rope_factor=32.0,
    )
    namespace["_resolve_model_preset_options"](
        args,
        config,
        ["--no-mx-per-block-qk-scales"],
    )
    assert args.mx_per_block_qk_scales is False
    assert args.fp8_per_block_qk_scales is True


def test_d128_trainer_rejects_an_explicit_d64_only_projection_choice() -> None:
    namespace = _trainer_resolution_namespace()
    args = _route_arguments()
    config = SimpleNamespace(
        head_dim=128,
        rope_theta=500_000.0,
        rope_factor=8.0,
    )
    with pytest.raises(ValueError, match="incompatible with the D128"):
        namespace["_resolve_model_preset_options"](
            args,
            config,
            ["--mx-qkv-projection-format=e4m3"],
        )


def test_d128_trainer_rejects_an_explicit_stale_backward_policy() -> None:
    namespace = _trainer_resolution_namespace()
    args = _route_arguments()
    config = SimpleNamespace(
        head_dim=128,
        rope_theta=500_000.0,
        rope_factor=8.0,
    )
    with pytest.raises(ValueError, match="incompatible with the D128"):
        namespace["_resolve_model_preset_options"](
            args,
            config,
            ["--backward-exp2-period=2"],
        )


def test_d128_trainer_rejects_explicit_2d_v_scaling() -> None:
    namespace = _trainer_resolution_namespace()
    args = _route_arguments()
    config = SimpleNamespace(
        head_dim=128,
        rope_theta=500_000.0,
        rope_factor=8.0,
    )
    with pytest.raises(ValueError, match="incompatible with the D128"):
        namespace["_resolve_model_preset_options"](
            args,
            config,
            ["--v-mxfp4-scaling=2d"],
        )


def test_d128_trainer_rejects_explicit_1d_projection_weights() -> None:
    namespace = _trainer_resolution_namespace()
    args = _route_arguments()
    config = SimpleNamespace(
        head_dim=128,
        rope_theta=500_000.0,
        rope_factor=8.0,
    )
    with pytest.raises(ValueError, match="incompatible with the D128"):
        namespace["_resolve_model_preset_options"](
            args,
            config,
            ["--projection-weight-scaling=1d"],
        )


def test_model_uses_an_independent_head_only_for_untied_presets() -> None:
    source = BENCHMARK.read_text()
    model = source.split("class Llama12B", 1)[1].split(
        "def activate_model_forward_route", 1
    )[0]
    assert "if not config.tie_word_embeddings:" in model
    assert "self.lm_head = _new_weight(config.vocab, config.hidden)" in model
    assert "self.embedding if self.lm_head is None else self.lm_head" in model


def test_benchmark_and_trainer_expose_the_model_preset() -> None:
    benchmark_source = BENCHMARK.read_text()
    trainer_source = TRAINER.read_text()
    assert '"--model-preset"' in benchmark_source
    assert '"--model-preset"' in trainer_source
    assert "config = config_from_model_preset(" in benchmark_source
    assert "config = config_from_model_preset(" in trainer_source
    assert "rope = _make_llama3_rope(config)" in benchmark_source


def test_trainer_preflights_full_8b_routes_and_snapshots_resolved_depth() -> None:
    source = TRAINER.read_text()
    main = source.split("def main() -> None:", 1)[1]
    preflight = main.index("_require_memory_safe_matched_replicas(")
    inspect_cuda = main.index("torch.cuda.device_count()")
    allocate_model = main.index("bf16_model = Llama12B(")
    assert preflight < inspect_cuda < allocate_model

    progress = main.split("def write_progress(", 1)[1]
    progress = progress.split('write_progress("initialized"', 1)[0]
    assert '"model_preset": config.model_preset' in progress
    assert '"layers": config.layers' in progress
    assert '"layers": args.layers' not in progress


def test_route_isolated_trainer_defers_cross_process_comparisons() -> None:
    compare = _trainer_comparison_namespace()["_comparisons_against_bf16"]
    lowp_only = {
        "nvfp4_qk_mxfp4_pv": {
            "timing": {"step_ms": 80.0},
            "validation": {"final_loss": 7.0},
        }
    }
    assert compare(lowp_only) == (None, {})

    routes = {
        "bf16_cute": {
            "timing": {"step_ms": 100.0},
            "validation": {"final_loss": 6.0},
        },
        **lowp_only,
    }
    reference, comparisons = compare(routes)
    assert reference == "bf16_cute"
    assert comparisons["nvfp4_qk_mxfp4_pv"] == {
        "speedup_over_bf16": 1.25,
        "final_validation_loss_delta": 1.0,
        "final_validation_loss_ratio": 7.0 / 6.0,
    }

    source = TRAINER.read_text()
    assert 'raise ValueError("--routes must include bf16_cute")' not in source
    assert "strict clone of BF16 state_dict" in source


def test_initial_state_probe_is_sparse_deterministic_and_value_sensitive() -> None:
    definitions = _selected_definitions(TRAINER, ("_state_dict_probe",))
    module = ast.Module(body=definitions, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {
        "Any": Any,
        "hashlib": hashlib,
        "json": json,
        "torch": torch,
    }
    exec(compile(module, str(TRAINER), "exec"), namespace)
    probe = namespace["_state_dict_probe"]

    first = {
        "weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "scale": torch.tensor([2.0], dtype=torch.bfloat16),
    }
    reordered = {"scale": first["scale"], "weight": first["weight"]}
    assert probe(first) == probe(reordered)
    assert probe(first)["sampled_values"] == 5

    changed = {name: value.clone() for name, value in first.items()}
    changed["weight"].reshape(-1)[-1] += 1
    assert probe(changed)["sha256"] != probe(first)["sha256"]


def test_cuda_hardware_identity_records_the_physical_training_device() -> None:
    definitions = _selected_definitions(TRAINER, ("_cuda_hardware_identity",))
    module = ast.Module(body=definitions, type_ignores=[])
    ast.fix_missing_locations(module)
    properties = SimpleNamespace(
        name="NVIDIA GB200",
        uuid="GPU-test",
        major=10,
        minor=0,
        total_memory=197897617408,
        multi_processor_count=152,
        L2_cache_size=135266304,
        pci_domain_id=0,
        pci_bus_id=41,
        pci_device_id=0,
    )
    fake_torch = SimpleNamespace(
        __version__="torch-test",
        version=SimpleNamespace(cuda="13.0"),
        cuda=SimpleNamespace(
            get_device_properties=lambda index: properties,
            device_count=lambda: 1,
        ),
    )
    namespace: dict[str, Any] = {"Any": Any, "torch": fake_torch}
    exec(compile(module, str(TRAINER), "exec"), namespace)
    identity = namespace["_cuda_hardware_identity"]()
    assert identity["schema"] == "cuda_hardware_identity_v1"
    assert identity["uuid"] == "GPU-test"
    assert identity["compute_capability"] == [10, 0]
    assert identity["visible_device_count"] == 1
    assert identity["total_memory_bytes"] == 197897617408
