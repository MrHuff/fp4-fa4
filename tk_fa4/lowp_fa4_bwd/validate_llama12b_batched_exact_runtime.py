#!/usr/bin/env python3
"""Gate one exact batched D64 controller against sequential B1 steps.

The selected projection extension must be installed before importing this
module with ``TK_FA4_LOWP_BWD_EXTENSION_SOURCE``.  This validator deliberately
pins every binary/control identity and exercises one complete decoder layer:
forward, backward, and a fused AdamW update.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import stat
import statistics
from pathlib import Path
from typing import Any, Iterable

PROJECTION = Path(
    "/tmp/fa4-dolma3-d64-assets.QZwFvk/assets/"
    "_C_b300_lowp_bwd.cpython-312-aarch64-linux-gnu.so"
)
CONTROL = Path(
    "/tmp/fa4-dolma3-d64-assets.QZwFvk/assets/"
    "fmha_bwd_d64_gqa_aug19_exact.py"
)
FORWARD_ARTIFACTS = {
    1: (
        Path("/tmp/_C_cfwd_fp8exact0_b1_s4096h32kv8d64_sm100_20260825.so"),
        "e7bb8e69625adf0e545c80d01b194c13af0ea9e12db8765150d2762267716c35",
        1_817_192,
    ),
    2: (
        Path("/tmp/_C_cfwd_fp8exact0_b2_s4096h32kv8d64_sm100_20260825.so"),
        "4e4c4c9b1afd7a751c3bae9d734f617a04b0b95778370deba9be3f131f5e05d1",
        1_817_192,
    ),
    8: (
        Path("/tmp/_C_cfwd_fp8exact0_b8_s4096h32kv8d64_sm100_20260825.so"),
        "34114089ab4631093dc2b4dbd38e01a597a6608c9cfb748bd927f8038271db9d",
        1_817_088,
    ),
    16: (
        Path("/tmp/_C_cfwd_fp8exact0_b16_s4096h32kv8d64_sm100_topofix_b200_20260825.so"),
        "88d81d3783e5aa80f0e9cf259a2ea7c935da4c2a5dc3ba1868e63f802a2c6208",
        1_817_256,
    ),
}
COMMON_ARTIFACTS = {
    "projection": (
        PROJECTION,
        "bfdec1e43a0a19acec5afbac3fa837e2f4d1b25be80ae7fb5ff3b5bc5e9e25ce",
        17_504_688,
    ),
    "control": (
        CONTROL,
        "cd57e3360082abe4bad7560c51a7793a4e9bfd4d16efc1259b92ce20238b99e1",
        220_876,
    ),
}

OUTPUT_PROJECTION_RELATIVE_L2_TOLERANCE = 0.002
OUTPUT_PROJECTION_COSINE_TOLERANCE = 0.001
OUTPUT_PROJECTION_ABSOLUTE_RELATIVE_L2_CEILING = 0.16
OUTPUT_PROJECTION_ABSOLUTE_COSINE_FLOOR = 0.985
FULL_STEP_OUTPUT_RELATIVE_L2_CEILING = 0.25
FULL_STEP_OUTPUT_COSINE_FLOOR = 0.97
FULL_STEP_INPUT_GRADIENT_RELATIVE_L2_CEILING = 0.70
FULL_STEP_INPUT_GRADIENT_COSINE_FLOOR = 0.80
FULL_STEP_PARAMETER_GRADIENT_RELATIVE_L2_CEILING = 0.55
FULL_STEP_PARAMETER_GRADIENT_COSINE_FLOOR = 0.85
FULL_STEP_PARAMETER_GRADIENT_WORST_TENSOR_RELATIVE_L2_CEILING = 0.80
FULL_STEP_POST_UPDATE_RELATIVE_L2_CEILING = 0.005
FULL_STEP_POST_UPDATE_COSINE_FLOOR = 0.99999
FULL_STEP_POST_UPDATE_WORST_TENSOR_RELATIVE_L2_CEILING = 0.006
B1_NONREGRESSION_RELATIVE_L2_MARGINS = {
    "output": 0.02,
    "input_gradient": 0.10,
    "parameter_gradient": 0.10,
    "post_optimizer_parameter": 0.002,
}
B1_NONREGRESSION_COSINE_MARGINS = {
    "output": 0.01,
    "input_gradient": 0.05,
    "parameter_gradient": 0.05,
}
MATCHED_BF16_SPEEDUP_FLOOR = 1.02
QKV_PUBLICATION_NAMES = (
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
)
_RUNTIME_IMPORTS_LOADED = False


def _authenticate_regular_artifact(
    path: Path,
    expected_sha256: str,
    expected_bytes: int,
    *,
    label: str,
) -> dict[str, Any]:
    """Authenticate one regular non-symlink file through a single open fd."""
    requested = Path(path)
    requested_stat = requested.lstat()
    if not stat.S_ISREG(requested_stat.st_mode):
        raise RuntimeError(
            f"{label} must be a regular non-symlink file: {requested}"
        )
    if requested_stat.st_size != expected_bytes:
        raise RuntimeError(
            f"{label} byte-count mismatch: expected {expected_bytes}, "
            f"found {requested_stat.st_size}"
        )
    resolved = requested.resolve(strict=True)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(resolved, flags)
    digest = hashlib.sha256()
    with os.fdopen(descriptor, "rb") as stream:
        opened_stat = os.fstat(stream.fileno())
        if not stat.S_ISREG(opened_stat.st_mode):
            raise RuntimeError(f"{label} stopped being a regular file")
        if opened_stat.st_size != expected_bytes:
            raise RuntimeError(f"{label} changed size while authenticating")
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    actual_sha256 = digest.hexdigest()
    if not hmac.compare_digest(actual_sha256, expected_sha256):
        raise RuntimeError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, "
            f"found {actual_sha256}"
        )
    return {
        "path": str(resolved),
        "sha256": actual_sha256,
        "bytes": opened_stat.st_size,
    }


def _authenticate_projection_environment() -> dict[str, Any]:
    """Authenticate the canonical projection before importing any runtime."""
    selected = os.environ.get("TK_FA4_LOWP_BWD_EXTENSION_SOURCE")
    if selected is None:
        raise RuntimeError(
            "TK_FA4_LOWP_BWD_EXTENSION_SOURCE must name the pinned projection"
        )
    selected_path = Path(selected)
    try:
        selected_stat = selected_path.lstat()
    except OSError as error:
        raise RuntimeError(
            f"unable to stat selected projection extension: {selected_path}"
        ) from error
    if not stat.S_ISREG(selected_stat.st_mode):
        raise RuntimeError(
            "TK_FA4_LOWP_BWD_EXTENSION_SOURCE must name a regular, "
            f"non-symlink file: {selected_path}"
        )
    selected_resolved = selected_path.resolve(strict=True)
    canonical_resolved = PROJECTION.resolve(strict=True)
    if selected_resolved != canonical_resolved:
        raise RuntimeError(
            "TK_FA4_LOWP_BWD_EXTENSION_SOURCE must name the canonical "
            f"projection {canonical_resolved}, got {selected_resolved}"
        )
    _path, expected_sha256, expected_bytes = COMMON_ARTIFACTS["projection"]
    return _authenticate_regular_artifact(
        selected_path,
        expected_sha256,
        expected_bytes,
        label="projection",
    )


def _load_runtime_imports() -> None:
    """Import Torch and tk_fa4 only after projection preauthentication."""
    global _RUNTIME_IMPORTS_LOADED
    global DecoderLayer, F, LowpAttentionRuntime, tk_interface, torch
    global _load_forward, _make_llama3_rope, config_from_model_preset
    if _RUNTIME_IMPORTS_LOADED:
        return

    import torch as torch_module
    import torch.nn.functional as functional
    from tk_fa4 import interface as interface_module
    from tk_fa4.lowp_fa4_bwd.benchmark_llama12b_e2e import (
        DecoderLayer as decoder_layer,
        LowpAttentionRuntime as lowp_attention_runtime,
        _load_forward as load_forward,
        _make_llama3_rope as make_llama3_rope,
        config_from_model_preset as make_config,
    )

    torch = torch_module
    F = functional
    tk_interface = interface_module
    DecoderLayer = decoder_layer
    LowpAttentionRuntime = lowp_attention_runtime
    _load_forward = load_forward
    _make_llama3_rope = make_llama3_rope
    config_from_model_preset = make_config
    _RUNTIME_IMPORTS_LOADED = True


def _authenticate_artifacts(
    batch: int,
    projection_preflight: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    selected_artifacts = {
        "b1_forward": FORWARD_ARTIFACTS[1],
        f"b{batch}_forward": FORWARD_ARTIFACTS[batch],
        "control": COMMON_ARTIFACTS["control"],
    }
    observed = {"projection": projection_preflight}
    for name, (path, expected_sha256, expected_bytes) in (
        selected_artifacts.items()
    ):
        observed[name] = _authenticate_regular_artifact(
            path,
            expected_sha256,
            expected_bytes,
            label=name,
        )
    loaded_projection = Path(
        str(tk_interface._C_b300_lowp_bwd.__file__)
    ).resolve()
    preflight_path = Path(str(projection_preflight["path"])).resolve()
    if loaded_projection != preflight_path:
        raise RuntimeError(
            "TK_FA4_LOWP_BWD_EXTENSION_SOURCE did not load the authenticated "
            f"projection: loaded {loaded_projection}"
        )
    return observed


def _runtime(batch: int) -> LowpAttentionRuntime:
    config = config_from_model_preset(batch=batch, layers=1)
    forward_path = FORWARD_ARTIFACTS[batch][0]
    module = forward_path.stem
    extension, topology = _load_forward(forward_path, module, config)
    return LowpAttentionRuntime(
        config,
        _make_llama3_rope(config),
        forward_extension=extension,
        forward_topology=topology,
        loss_scale=65_536.0,
        gradient_global_scale=2.0**-8,
        projection_dgrad="bf16",
        qkv_projection_format="e4m3",
        backward_exp2_degree=1,
        backward_exp2_period=2,
        backward_fp8_ds_lift=16,
        backward_reuse_quantized_p=False,
        backward_control_source=CONTROL,
        backward_control_sha256=COMMON_ARTIFACTS["control"][1],
        backward_control_bytes=COMMON_ARTIFACTS["control"][2],
        backward_forward_mx_probability_replay=False,
        backward_forward_mx_probability_scale_handoff=False,
        backward_match_forward_operands=True,
        per_block_qk_scales=True,
        experimental_split_v_backward=False,
        backward_probability_correction=1.0,
        q_quant_scale=2.25,
        k_quant_scale=2.0,
        projection_weight_scale_2d=True,
        v_mxfp4_scale_2d=False,
        adaptive_qk_weight_scales=False,
    )


def _share_b1_parameters(owner: DecoderLayer, follower: DecoderLayer) -> None:
    # The second B1 layer retains its own publication workspace, while both
    # calls accumulate gradients into one authentic shared parameter set.
    follower.attention_norm = owner.attention_norm
    follower.ffn_norm = owner.ffn_norm
    follower.mlp = owner.mlp
    follower.attention.weights = owner.attention.weights
    owner_parameters = dict(owner.named_parameters())
    follower_parameters = dict(follower.named_parameters())
    if owner_parameters.keys() != follower_parameters.keys() or any(
        owner_parameters[name] is not follower_parameters[name]
        for name in owner_parameters
    ):
        raise RuntimeError("the two B1 calls do not share every parameter")
    if not torch.equal(
        owner.attention.qk_scales,
        follower.attention.qk_scales,
    ):
        raise RuntimeError("the two B1 calls do not share one Q/K scale policy")


def _metrics(
    candidate: Iterable[tuple[str, torch.Tensor]],
    reference: Iterable[tuple[str, torch.Tensor]],
) -> dict[str, Any]:
    candidate_tensors = dict(candidate)
    reference_tensors = dict(reference)
    if candidate_tensors.keys() != reference_tensors.keys():
        raise RuntimeError("metric tensor names differ")
    dot = 0.0
    candidate_square = 0.0
    reference_square = 0.0
    difference_square = 0.0
    max_abs = 0.0
    worst_name = ""
    worst_relative_l2 = -1.0
    finite = True
    for name in candidate_tensors:
        lhs = candidate_tensors[name].detach().float()
        rhs = reference_tensors[name].detach().float()
        difference = lhs - rhs
        lhs_square = float(lhs.square().sum())
        rhs_square = float(rhs.square().sum())
        diff_square = float(difference.square().sum())
        relative_l2 = diff_square**0.5 / max(rhs_square**0.5, 1.0e-30)
        if relative_l2 > worst_relative_l2:
            worst_name = name
            worst_relative_l2 = relative_l2
        dot += float((lhs * rhs).sum())
        candidate_square += lhs_square
        reference_square += rhs_square
        difference_square += diff_square
        max_abs = max(max_abs, float(difference.abs().max()))
        finite = finite and bool(torch.isfinite(lhs).all())
        finite = finite and bool(torch.isfinite(rhs).all())
    denominator = max(
        (candidate_square * reference_square) ** 0.5,
        1.0e-30,
    )
    return {
        "finite": finite,
        "cosine": dot / denominator,
        "relative_l2": difference_square**0.5
        / max(reference_square**0.5, 1.0e-30),
        "max_abs": max_abs,
        "worst_tensor": worst_name,
        "worst_tensor_relative_l2": worst_relative_l2,
    }


def _tensor_comparison(
    candidate: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, Any]:
    """Report both semantic error and exact storage-byte equality."""
    if candidate.shape != reference.shape or candidate.dtype != reference.dtype:
        raise RuntimeError(
            "stage tensor metadata differ: "
            f"{tuple(candidate.shape)} {candidate.dtype} versus "
            f"{tuple(reference.shape)} {reference.dtype}"
        )
    candidate_bytes = candidate.contiguous().view(torch.uint8).reshape(-1)
    reference_bytes = reference.contiguous().view(torch.uint8).reshape(-1)
    byte_mismatches = int((candidate_bytes != reference_bytes).sum())
    result = _metrics(
        (("tensor", candidate),),
        (("tensor", reference),),
    )
    result.update(
        {
            "shape": list(candidate.shape),
            "dtype": str(candidate.dtype),
            "bytes": candidate_bytes.numel(),
            "byte_equal": byte_mismatches == 0,
            "byte_mismatch_count": byte_mismatches,
        }
    )
    return result


def _clone_stage_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Snapshot one mutable publication before its workspace is reused."""
    return tensor.detach().to(device="cpu", copy=True).contiguous()


def _capture_forward_stages(
    layer: DecoderLayer,
    normalized: torch.Tensor,
    diagnostic: dict[str, torch.Tensor | None],
) -> dict[str, torch.Tensor]:
    workspace = layer.attention._forward_workspace.outputs
    tensors: dict[str, torch.Tensor] = {
        "rmsnorm": normalized,
        "qk_policy_scales": diagnostic["qk_policy_scales"],
        "backward_qk_scales": diagnostic["backward_qk_scales"],
        "q_payload": workspace.q_payload,
        "q_forward_scales": diagnostic["q_forward_scales"],
        "q_forward_global_scale": diagnostic["q_forward_global_scale"],
        "k_payload": workspace.k_payload,
        "k_forward_scales": diagnostic["k_forward_scales"],
        "k_forward_global_scale": diagnostic["k_forward_global_scale"],
        "v_forward_fp8": workspace.v_fp8_payload,
        "q_backward_fp8": diagnostic["q_backward_fp8"],
        "k_backward_fp8": diagnostic["k_backward_fp8"],
        "v_backward_fp8": diagnostic["v_backward_fp8"],
        "raw_attention": diagnostic["attention_output"],
        "lse": diagnostic["lse"],
    }
    if any(tensor is None for tensor in tensors.values()):
        missing = [name for name, tensor in tensors.items() if tensor is None]
        raise RuntimeError(f"exact FP8-PV diagnostic omitted {missing}")
    return {
        name: _clone_stage_tensor(tensor)
        for name, tensor in tensors.items()
        if tensor is not None
    }


def _compare_batched_stages(
    batched: dict[str, torch.Tensor],
    reference_samples: tuple[dict[str, torch.Tensor], ...],
) -> dict[str, dict[str, Any]]:
    if not reference_samples:
        raise RuntimeError("stage comparison requires at least one B1 sample")
    if any(sample.keys() != batched.keys() for sample in reference_samples):
        raise RuntimeError("batched and B1 stage snapshots have different fields")
    result = {}
    for name, candidate in batched.items():
        reference = torch.cat(
            tuple(sample[name] for sample in reference_samples),
            dim=0,
        )
        result[name] = _tensor_comparison(candidate, reference)
    return result


def _optimizer(parameters: Iterable[torch.nn.Parameter]) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        parameters,
        lr=1.0e-4,
        betas=(0.9, 0.95),
        eps=1.0e-8,
        weight_decay=0.0,
        fused=True,
    )


def _timed_step(
    route: str,
    layers: tuple[DecoderLayer, ...],
    optimizer: torch.optim.Optimizer,
    source: torch.Tensor,
    upstream: torch.Tensor,
) -> dict[str, float]:
    optimizer.zero_grad(set_to_none=True)
    attention = layers[0].attention
    runtime = getattr(attention, "runtime", None)
    configured_batch = (
        runtime.config.batch if runtime is not None else attention.config.batch
    )
    if len(layers) == 1 and configured_batch != 1:
        inputs = (source.detach().clone().requires_grad_(True),)
        gradients = (upstream,)
    else:
        inputs = tuple(
            source[index : index + 1].detach().clone().requires_grad_(True)
            for index in range(len(layers))
        )
        gradients = tuple(
            upstream[index : index + 1] for index in range(len(layers))
        )
    begin = torch.cuda.Event(enable_timing=True)
    after_forward = torch.cuda.Event(enable_timing=True)
    after_backward = torch.cuda.Event(enable_timing=True)
    after_optimizer = torch.cuda.Event(enable_timing=True)
    begin.record()
    outputs = tuple(layer(value) for layer, value in zip(layers, inputs))
    after_forward.record()
    torch.autograd.backward(outputs, gradients)
    after_backward.record()
    optimizer.step()
    after_optimizer.record()
    after_optimizer.synchronize()
    result = {
        "route": route,
        "forward_ms": begin.elapsed_time(after_forward),
        "backward_ms": after_forward.elapsed_time(after_backward),
        "optimizer_ms": after_backward.elapsed_time(after_optimizer),
        "step_ms": begin.elapsed_time(after_optimizer),
    }
    del inputs, outputs
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch",
        type=int,
        choices=(2, 8, 16),
        default=16,
        help="authenticated target batch; B16 is primary and B8 fallback",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    batch = args.batch
    reference_route = f"{batch}x_b1"
    batched_route = f"b{batch}"
    bf16_route = f"bf16_b{batch}"
    output_path = args.output or Path(
        "results/"
        f"llama12b_d64_exact_b{batch}_controller_gate_final_20260825.json"
    )

    projection_preflight = _authenticate_projection_environment()
    _load_runtime_imports()
    torch.cuda.set_device(0)
    torch.manual_seed(20260825)
    artifacts = _authenticate_artifacts(batch, projection_preflight)
    runtime_b1 = _runtime(1)
    runtime_batched = _runtime(batch)
    config_b1 = runtime_b1.config
    config_batched = runtime_batched.config

    owner = DecoderLayer(
        config_b1, _make_llama3_rope(config_b1), runtime_b1
    )
    reference_layers = [owner]
    for _ in range(1, batch):
        follower = DecoderLayer(
            config_b1, _make_llama3_rope(config_b1), runtime_b1
        )
        _share_b1_parameters(owner, follower)
        reference_layers.append(follower)
    reference_layers_tuple = tuple(reference_layers)
    batched_layer = DecoderLayer(
        config_batched,
        _make_llama3_rope(config_batched),
        runtime_batched,
    )
    batched_layer.load_state_dict(owner.state_dict(), strict=True)
    bf16_layer = DecoderLayer(
        config_batched,
        _make_llama3_rope(config_batched),
        None,
    )
    bf16_layer.load_state_dict(owner.state_dict(), strict=True)

    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260825 + 1)
    source = torch.randn(
        config_batched.batch,
        config_batched.sequence,
        config_batched.hidden,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.02
    upstream = torch.randn(
        source.shape,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    ) * 1.0e-4

    # Prove the sequential reference consumes one live parameter set. Every
    # caller retains only its own publication workspace; quantized weight
    # operands are rebuilt from the shared Parameters on every forward.
    # Snapshot every mutable publication immediately after its producer. The
    # B1 runtime is reused across samples, so retaining workspace aliases here
    # would make the final sample overwrite earlier diagnostic evidence.
    with torch.no_grad():
        duplicate_input = source[:1]
        duplicate_owner = reference_layers[0](duplicate_input)
        duplicate_follower = reference_layers[1](duplicate_input)
        runtime_b1.forward_diagnostic_sink = []
        reference_projected_samples = []
        reference_stage_samples = []
        for index, layer in enumerate(reference_layers_tuple):
            normalized = layer.attention_norm(source[index : index + 1])
            projected = layer.attention(normalized)
            diagnostic = runtime_b1.forward_diagnostic_sink[-1]
            reference_projected_samples.append(projected)
            reference_stage_samples.append(
                _capture_forward_stages(layer, normalized, diagnostic)
            )
        reference_projected = torch.cat(
            tuple(reference_projected_samples), dim=0
        )
        runtime_b1.forward_diagnostic_sink = None

        runtime_batched.forward_diagnostic_sink = []
        batched_normalized = batched_layer.attention_norm(source)
        batched_projected = batched_layer.attention(batched_normalized)
        batched_diagnostic = runtime_batched.forward_diagnostic_sink[-1]
        raw_attention = batched_diagnostic["attention_output"].detach().clone()
        batched_stages = _capture_forward_stages(
            batched_layer,
            batched_normalized,
            batched_diagnostic,
        )
        runtime_batched.forward_diagnostic_sink = None

        stage_diagnostics = _compare_batched_stages(
            batched_stages,
            tuple(reference_stage_samples),
        )
        bf16_output_projection = F.linear(
            raw_attention.reshape(
                config_batched.batch * config_batched.sequence,
                config_batched.q_width,
            ),
            owner.attention.weights.o,
        ).reshape_as(batched_projected)
        output_projection_accuracy = {
            "batched_vs_bf16": _metrics(
                (("output_projection", batched_projected),),
                (("output_projection", bf16_output_projection),),
            ),
            "sequential_b1_vs_bf16": _metrics(
                (("output_projection", reference_projected),),
                (("output_projection", bf16_output_projection),),
            ),
            "batched_vs_sequential_b1": _metrics(
                (("output_projection", batched_projected),),
                (("output_projection", reference_projected),),
            ),
            "tolerances": {
                "relative_l2_additive": (
                    OUTPUT_PROJECTION_RELATIVE_L2_TOLERANCE
                ),
                "cosine_subtractive": OUTPUT_PROJECTION_COSINE_TOLERANCE,
            },
        }
    prestep_probes = {
        "duplicate_b1_full_layer": _metrics(
            (("output", duplicate_follower),),
            (("output", duplicate_owner),),
        ),
        "stage_diagnostics": stage_diagnostics,
        "output_projection_accuracy": output_projection_accuracy,
    }
    owner_parameters = dict(owner.named_parameters())
    workspace_ids = {
        id(layer.attention._forward_workspace)
        for layer in reference_layers_tuple
    }
    harness_audit = {
        "reference_batch": batch,
        "shared_parameter_objects": all(
            dict(layer.named_parameters())[name] is parameter
            for layer in reference_layers_tuple
            for name, parameter in owner_parameters.items()
        ),
        "shared_parameter_count": len(owner_parameters),
        "separate_attention_publication_workspaces": (
            len(workspace_ids) == batch
        ),
        "identical_qk_scale_policy": all(
            torch.equal(
                owner.attention.qk_scales,
                layer.attention.qk_scales,
            )
            for layer in reference_layers_tuple
        ),
        "projection_operands_prepared_from_live_parameters_per_forward": True,
        "bf16_batched_initial_state_matches_exact": all(
            torch.equal(value, bf16_layer.state_dict()[name])
            for name, value in batched_layer.state_dict().items()
        ),
    }
    del (
        duplicate_owner,
        duplicate_follower,
        reference_projected_samples,
        reference_projected,
        reference_stage_samples,
        batched_normalized,
        batched_projected,
        batched_diagnostic,
        raw_attention,
        batched_stages,
        bf16_output_projection,
    )
    torch.cuda.empty_cache()

    optimizer_reference = _optimizer(owner.parameters())
    optimizer_batched = _optimizer(batched_layer.parameters())
    optimizer_bf16 = _optimizer(bf16_layer.parameters())
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    reference_baseline = torch.cuda.memory_allocated()
    reference_reserved_baseline = torch.cuda.memory_reserved()
    reference_inputs = tuple(
        source[index : index + 1].detach().clone().requires_grad_(True)
        for index in range(batch)
    )
    reference_outputs = tuple(
        layer(value)
        for layer, value in zip(
            reference_layers_tuple, reference_inputs, strict=True
        )
    )
    torch.autograd.backward(
        reference_outputs,
        tuple(upstream[index : index + 1] for index in range(batch)),
    )
    optimizer_reference.step()
    torch.cuda.synchronize()
    reference_peak = torch.cuda.max_memory_allocated()
    reference_reserved_peak = torch.cuda.max_memory_reserved()

    torch.cuda.reset_peak_memory_stats()
    batched_baseline = torch.cuda.memory_allocated()
    batched_reserved_baseline = torch.cuda.memory_reserved()
    batched_input = source.detach().clone().requires_grad_(True)
    batched_output = batched_layer(batched_input)
    batched_output.backward(upstream)
    optimizer_batched.step()
    torch.cuda.synchronize()
    batched_peak = torch.cuda.max_memory_allocated()
    batched_reserved_peak = torch.cuda.max_memory_reserved()

    torch.cuda.reset_peak_memory_stats()
    bf16_baseline = torch.cuda.memory_allocated()
    bf16_reserved_baseline = torch.cuda.memory_reserved()
    bf16_input = source.detach().clone().requires_grad_(True)
    bf16_output = bf16_layer(bf16_input)
    bf16_output.backward(upstream)
    optimizer_bf16.step()
    torch.cuda.synchronize()
    bf16_peak = torch.cuda.max_memory_allocated()
    bf16_reserved_peak = torch.cuda.max_memory_reserved()
    bf16_full_step_sanity = {
        "output_finite": bool(torch.isfinite(bf16_output).all()),
        "input_gradient_finite": bool(torch.isfinite(bf16_input.grad).all()),
        "parameter_gradients_finite": all(
            parameter.grad is not None
            and bool(torch.isfinite(parameter.grad).all())
            for parameter in bf16_layer.parameters()
        ),
    }

    numerics = {
        "output": _metrics(
            (("output", batched_output),),
            (("output", torch.cat(reference_outputs, dim=0)),),
        ),
        "input_gradient": _metrics(
            (("input_gradient", batched_input.grad),),
            (
                (
                    "input_gradient",
                    torch.cat(
                        tuple(value.grad for value in reference_inputs),
                        dim=0,
                    ),
                ),
            ),
        ),
        "parameter_gradient": _metrics(
            (
                (name, parameter.grad)
                for name, parameter in batched_layer.named_parameters()
            ),
            (
                (name, parameter.grad)
                for name, parameter in owner.named_parameters()
            ),
        ),
        "post_optimizer_parameter": _metrics(
            batched_layer.named_parameters(), owner.named_parameters()
        ),
    }
    exact_vs_bf16_numerics = {
        "output": _metrics(
            (("output", batched_output),),
            (("output", bf16_output),),
        ),
        "input_gradient": _metrics(
            (("input_gradient", batched_input.grad),),
            (("input_gradient", bf16_input.grad),),
        ),
        "parameter_gradient": _metrics(
            (
                (name, parameter.grad)
                for name, parameter in batched_layer.named_parameters()
            ),
            (
                (name, parameter.grad)
                for name, parameter in bf16_layer.named_parameters()
            ),
        ),
        "post_optimizer_parameter": _metrics(
            batched_layer.named_parameters(), bf16_layer.named_parameters()
        ),
    }
    sequential_b1_vs_bf16_numerics = {
        "output": _metrics(
            (("output", torch.cat(reference_outputs, dim=0)),),
            (("output", bf16_output),),
        ),
        "input_gradient": _metrics(
            (
                (
                    "input_gradient",
                    torch.cat(
                        tuple(value.grad for value in reference_inputs),
                        dim=0,
                    ),
                ),
            ),
            (("input_gradient", bf16_input.grad),),
        ),
        "parameter_gradient": _metrics(
            (
                (name, parameter.grad)
                for name, parameter in owner.named_parameters()
            ),
            (
                (name, parameter.grad)
                for name, parameter in bf16_layer.named_parameters()
            ),
        ),
        "post_optimizer_parameter": _metrics(
            owner.named_parameters(), bf16_layer.named_parameters()
        ),
    }
    peak_memory = {
        reference_route: {
            "baseline_bytes": reference_baseline,
            "peak_bytes": reference_peak,
            "incremental_peak_bytes": reference_peak - reference_baseline,
            "baseline_allocated_bytes": reference_baseline,
            "peak_allocated_bytes": reference_peak,
            "incremental_peak_allocated_bytes": (
                reference_peak - reference_baseline
            ),
            "baseline_reserved_bytes": reference_reserved_baseline,
            "peak_reserved_bytes": reference_reserved_peak,
            "incremental_peak_reserved_bytes": (
                reference_reserved_peak - reference_reserved_baseline
            ),
        },
        batched_route: {
            "baseline_bytes": batched_baseline,
            "peak_bytes": batched_peak,
            "incremental_peak_bytes": batched_peak - batched_baseline,
            "baseline_allocated_bytes": batched_baseline,
            "peak_allocated_bytes": batched_peak,
            "incremental_peak_allocated_bytes": (
                batched_peak - batched_baseline
            ),
            "baseline_reserved_bytes": batched_reserved_baseline,
            "peak_reserved_bytes": batched_reserved_peak,
            "incremental_peak_reserved_bytes": (
                batched_reserved_peak - batched_reserved_baseline
            ),
        },
        bf16_route: {
            "baseline_bytes": bf16_baseline,
            "peak_bytes": bf16_peak,
            "incremental_peak_bytes": bf16_peak - bf16_baseline,
            "baseline_allocated_bytes": bf16_baseline,
            "peak_allocated_bytes": bf16_peak,
            "incremental_peak_allocated_bytes": bf16_peak - bf16_baseline,
            "baseline_reserved_bytes": bf16_reserved_baseline,
            "peak_reserved_bytes": bf16_reserved_peak,
            "incremental_peak_reserved_bytes": (
                bf16_reserved_peak - bf16_reserved_baseline
            ),
        },
    }
    print(
        json.dumps(
            {
                "checkpoint": "numerical_full_steps_complete",
                "batch": batch,
                "harness_audit": harness_audit,
                "prestep_probes": prestep_probes,
                "numerics": numerics,
                "exact_vs_bf16_numerics": exact_vs_bf16_numerics,
                "sequential_b1_vs_bf16_numerics": (
                    sequential_b1_vs_bf16_numerics
                ),
                "bf16_full_step_sanity": bf16_full_step_sanity,
                "peak_memory": peak_memory,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    del (
        reference_inputs,
        reference_outputs,
        batched_input,
        batched_output,
        bf16_input,
        bf16_output,
    )
    optimizer_reference.zero_grad(set_to_none=True)
    optimizer_batched.zero_grad(set_to_none=True)
    optimizer_bf16.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()

    # First-use publication authentication happened in the probes/full step.
    # One warmup plus three samples therefore measures only steady state.
    records = {reference_route: [], batched_route: [], bf16_route: []}
    timing_routes = {
        reference_route: (reference_layers_tuple, optimizer_reference),
        batched_route: ((batched_layer,), optimizer_batched),
        bf16_route: ((bf16_layer,), optimizer_bf16),
    }
    warmup_order = tuple(timing_routes)
    for route in warmup_order:
        layers, optimizer = timing_routes[route]
        _timed_step(route, layers, optimizer, source, upstream)
    sample_orders = tuple(
        tuple(
            warmup_order[(position + offset) % len(warmup_order)]
            for position in range(len(warmup_order))
        )
        for offset in range(3)
    )
    for round_index, order in enumerate(sample_orders):
        for position, route in enumerate(order):
            layers, optimizer = timing_routes[route]
            record = _timed_step(
                route,
                layers,
                optimizer,
                source,
                upstream,
            )
            record.update(
                {
                    "round": round_index,
                    "position": position,
                    "execution_order": list(order),
                }
            )
            records[route].append(record)
    timing = {
        "protocol": {
            "warmup_order": list(warmup_order),
            "sample_orders": [list(order) for order in sample_orders],
            "samples_per_route": 3,
            "aggregation": "component-wise median of three samples",
            "order_control": "three cyclic route-order shifts",
        }
    }
    for route, route_records in records.items():
        timing[route] = {
            key: statistics.median(record[key] for record in route_records)
            for key in (
                "forward_ms",
                "backward_ms",
                "optimizer_ms",
                "step_ms",
            )
        }
        timing[route]["samples"] = route_records
    timing["batched_speedup_over_sequential_b1"] = (
        timing[reference_route]["step_ms"]
        / timing[batched_route]["step_ms"]
    )
    timing["exact_batched_speedup_over_bf16_batched"] = (
        timing[bf16_route]["step_ms"] / timing[batched_route]["step_ms"]
    )
    timing["exact_batched_to_bf16_batched_ratio"] = {
        key: timing[batched_route][key] / timing[bf16_route][key]
        for key in ("forward_ms", "backward_ms", "optimizer_ms", "step_ms")
    }

    gate_policy = {
        "output_projection_absolute": {
            "relative_l2_ceiling": (
                OUTPUT_PROJECTION_ABSOLUTE_RELATIVE_L2_CEILING
            ),
            "cosine_floor": OUTPUT_PROJECTION_ABSOLUTE_COSINE_FLOOR,
            "rationale": (
                "Parity with B1 cannot excuse an inaccurate projection; "
                "the grouped batched projection must independently remain "
                "close "
                "to one BF16 F.linear reference."
            ),
        },
        "full_step_absolute": {
            "output_relative_l2_ceiling": (
                FULL_STEP_OUTPUT_RELATIVE_L2_CEILING
            ),
            "output_cosine_floor": FULL_STEP_OUTPUT_COSINE_FLOOR,
            "input_gradient_relative_l2_ceiling": (
                FULL_STEP_INPUT_GRADIENT_RELATIVE_L2_CEILING
            ),
            "input_gradient_cosine_floor": (
                FULL_STEP_INPUT_GRADIENT_COSINE_FLOOR
            ),
            "parameter_gradient_relative_l2_ceiling": (
                FULL_STEP_PARAMETER_GRADIENT_RELATIVE_L2_CEILING
            ),
            "parameter_gradient_cosine_floor": (
                FULL_STEP_PARAMETER_GRADIENT_COSINE_FLOOR
            ),
            "parameter_gradient_worst_tensor_relative_l2_ceiling": (
                FULL_STEP_PARAMETER_GRADIENT_WORST_TENSOR_RELATIVE_L2_CEILING
            ),
            "post_update_relative_l2_ceiling": (
                FULL_STEP_POST_UPDATE_RELATIVE_L2_CEILING
            ),
            "post_update_cosine_floor": FULL_STEP_POST_UPDATE_COSINE_FLOOR,
            "post_update_worst_tensor_relative_l2_ceiling": (
                FULL_STEP_POST_UPDATE_WORST_TENSOR_RELATIVE_L2_CEILING
            ),
            "rationale": (
                "Round absolute ceilings require finite, directionally "
                "aligned learning signals and a tightly matched one-step "
                "parameter state; they are one-step sanity bounds, not a "
                "claim of BF16 equivalence."
            ),
        },
        "b1_nonregression": {
            "relative_l2_additive_margins": (
                B1_NONREGRESSION_RELATIVE_L2_MARGINS
            ),
            "cosine_subtractive_margins": B1_NONREGRESSION_COSINE_MARGINS,
            "rationale": (
                "Sequential B1 is the established low-precision route. "
                "Comparing the batched route and B1 to the same BF16 full "
                "step separates "
                "known low-precision error from additional batching error."
            ),
        },
        "matched_bf16_speed": {
            "speedup_floor": MATCHED_BF16_SPEEDUP_FLOOR,
            "protocol": (
                "one warmup per route, three cyclically ordered samples, "
                "component medians"
            ),
            "rationale": (
                "A 1.02x floor rejects timing-noise parity and requires the "
                "batched exact route to improve the actual matched BF16 "
                "training step."
            ),
        },
    }

    gates = {
        "b1_reference_consumes_identical_weights": (
            harness_audit["shared_parameter_objects"]
            and harness_audit["identical_qk_scale_policy"]
            and prestep_probes["duplicate_b1_full_layer"]["relative_l2"]
            == 0.0
        ),
        "bf16_batched_initial_state_matches_exact": harness_audit[
            "bf16_batched_initial_state_matches_exact"
        ],
        "rmsnorm_byte_equal": stage_diagnostics["rmsnorm"][
            "byte_equal"
        ],
        "qkv_publications_byte_equal": all(
            stage_diagnostics[name]["byte_equal"]
            for name in QKV_PUBLICATION_NAMES
        ),
        "raw_attention_byte_equal": stage_diagnostics["raw_attention"][
            "byte_equal"
        ],
        "lse_byte_equal": stage_diagnostics["lse"]["byte_equal"],
        "output_projection_comparisons_finite": all(
            output_projection_accuracy[name]["finite"]
            for name in (
                "batched_vs_bf16",
                "sequential_b1_vs_bf16",
                "batched_vs_sequential_b1",
            )
        ),
        "batched_output_projection_relative_l2_at_most_0.16": (
            output_projection_accuracy["batched_vs_bf16"]["relative_l2"]
            <= OUTPUT_PROJECTION_ABSOLUTE_RELATIVE_L2_CEILING
        ),
        "batched_output_projection_cosine_at_least_0.985": (
            output_projection_accuracy["batched_vs_bf16"]["cosine"]
            >= OUTPUT_PROJECTION_ABSOLUTE_COSINE_FLOOR
        ),
        "batched_output_projection_relative_l2_no_worse_than_b1_plus_0.002": (
            output_projection_accuracy["batched_vs_bf16"]["relative_l2"]
            <= output_projection_accuracy[
                "sequential_b1_vs_bf16"
            ]["relative_l2"]
            + OUTPUT_PROJECTION_RELATIVE_L2_TOLERANCE
        ),
        "batched_output_projection_cosine_no_worse_than_b1_minus_0.001": (
            output_projection_accuracy["batched_vs_bf16"]["cosine"]
            >= output_projection_accuracy[
                "sequential_b1_vs_bf16"
            ]["cosine"]
            - OUTPUT_PROJECTION_COSINE_TOLERANCE
        ),
        "all_full_step_tensors_finite": (
            all(value["finite"] for value in numerics.values())
            and all(
                value["finite"] for value in exact_vs_bf16_numerics.values()
            )
            and all(
                value["finite"]
                for value in sequential_b1_vs_bf16_numerics.values()
            )
            and all(bf16_full_step_sanity.values())
        ),
        "exact_vs_bf16_output_relative_l2_at_most_0.25": (
            exact_vs_bf16_numerics["output"]["relative_l2"]
            <= FULL_STEP_OUTPUT_RELATIVE_L2_CEILING
        ),
        "exact_vs_bf16_output_cosine_at_least_0.97": (
            exact_vs_bf16_numerics["output"]["cosine"]
            >= FULL_STEP_OUTPUT_COSINE_FLOOR
        ),
        "exact_vs_bf16_input_gradient_relative_l2_at_most_0.70": (
            exact_vs_bf16_numerics["input_gradient"]["relative_l2"]
            <= FULL_STEP_INPUT_GRADIENT_RELATIVE_L2_CEILING
        ),
        "exact_vs_bf16_input_gradient_cosine_at_least_0.80": (
            exact_vs_bf16_numerics["input_gradient"]["cosine"]
            >= FULL_STEP_INPUT_GRADIENT_COSINE_FLOOR
        ),
        "exact_vs_bf16_parameter_gradient_relative_l2_at_most_0.55": (
            exact_vs_bf16_numerics["parameter_gradient"]["relative_l2"]
            <= FULL_STEP_PARAMETER_GRADIENT_RELATIVE_L2_CEILING
        ),
        "exact_vs_bf16_parameter_gradient_cosine_at_least_0.85": (
            exact_vs_bf16_numerics["parameter_gradient"]["cosine"]
            >= FULL_STEP_PARAMETER_GRADIENT_COSINE_FLOOR
        ),
        "exact_vs_bf16_parameter_gradient_worst_relative_l2_at_most_0.80": (
            exact_vs_bf16_numerics["parameter_gradient"][
                "worst_tensor_relative_l2"
            ]
            <= FULL_STEP_PARAMETER_GRADIENT_WORST_TENSOR_RELATIVE_L2_CEILING
        ),
        "exact_vs_bf16_post_update_relative_l2_at_most_0.005": (
            exact_vs_bf16_numerics["post_optimizer_parameter"]["relative_l2"]
            <= FULL_STEP_POST_UPDATE_RELATIVE_L2_CEILING
        ),
        "exact_vs_bf16_post_update_cosine_at_least_0.99999": (
            exact_vs_bf16_numerics["post_optimizer_parameter"]["cosine"]
            >= FULL_STEP_POST_UPDATE_COSINE_FLOOR
        ),
        "exact_vs_bf16_post_update_worst_relative_l2_at_most_0.006": (
            exact_vs_bf16_numerics["post_optimizer_parameter"][
                "worst_tensor_relative_l2"
            ]
            <= FULL_STEP_POST_UPDATE_WORST_TENSOR_RELATIVE_L2_CEILING
        ),
        "batched_output_relative_l2_no_worse_than_b1_plus_0.02": (
            exact_vs_bf16_numerics["output"]["relative_l2"]
            <= sequential_b1_vs_bf16_numerics["output"]["relative_l2"]
            + B1_NONREGRESSION_RELATIVE_L2_MARGINS["output"]
        ),
        "batched_output_cosine_no_worse_than_b1_minus_0.01": (
            exact_vs_bf16_numerics["output"]["cosine"]
            >= sequential_b1_vs_bf16_numerics["output"]["cosine"]
            - B1_NONREGRESSION_COSINE_MARGINS["output"]
        ),
        "batched_input_gradient_relative_l2_no_worse_than_b1_plus_0.10": (
            exact_vs_bf16_numerics["input_gradient"]["relative_l2"]
            <= sequential_b1_vs_bf16_numerics[
                "input_gradient"
            ]["relative_l2"]
            + B1_NONREGRESSION_RELATIVE_L2_MARGINS["input_gradient"]
        ),
        "batched_input_gradient_cosine_no_worse_than_b1_minus_0.05": (
            exact_vs_bf16_numerics["input_gradient"]["cosine"]
            >= sequential_b1_vs_bf16_numerics["input_gradient"]["cosine"]
            - B1_NONREGRESSION_COSINE_MARGINS["input_gradient"]
        ),
        "batched_parameter_gradient_relative_l2_no_worse_than_b1_plus_0.10": (
            exact_vs_bf16_numerics["parameter_gradient"]["relative_l2"]
            <= sequential_b1_vs_bf16_numerics[
                "parameter_gradient"
            ]["relative_l2"]
            + B1_NONREGRESSION_RELATIVE_L2_MARGINS["parameter_gradient"]
        ),
        "batched_parameter_gradient_cosine_no_worse_than_b1_minus_0.05": (
            exact_vs_bf16_numerics["parameter_gradient"]["cosine"]
            >= sequential_b1_vs_bf16_numerics[
                "parameter_gradient"
            ]["cosine"]
            - B1_NONREGRESSION_COSINE_MARGINS["parameter_gradient"]
        ),
        "batched_post_update_relative_l2_no_worse_than_b1_plus_0.002": (
            exact_vs_bf16_numerics["post_optimizer_parameter"]["relative_l2"]
            <= sequential_b1_vs_bf16_numerics[
                "post_optimizer_parameter"
            ]["relative_l2"]
            + B1_NONREGRESSION_RELATIVE_L2_MARGINS[
                "post_optimizer_parameter"
            ]
        ),
        "batched_faster_than_sequential_b1": (
            timing[batched_route]["step_ms"]
            < timing[reference_route]["step_ms"]
        ),
        "exact_batched_speedup_over_bf16_batched_at_least_1.02": (
            timing["exact_batched_speedup_over_bf16_batched"]
            >= MATCHED_BF16_SPEEDUP_FLOOR
        ),
    }
    result = {
        "schema": "llama12b_d64_exact_batched_controller_gate_v4",
        "batch": batch,
        "device": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "artifacts": artifacts,
        "b1_backward_contract": runtime_b1.backward_contract(),
        "batched_backward_contract": runtime_batched.backward_contract(),
        "harness_audit": harness_audit,
        "prestep_probes": prestep_probes,
        "numerics": numerics,
        "exact_vs_bf16_numerics": exact_vs_bf16_numerics,
        "sequential_b1_vs_bf16_numerics": (
            sequential_b1_vs_bf16_numerics
        ),
        "bf16_full_step_sanity": bf16_full_step_sanity,
        "peak_memory": peak_memory,
        "timing": timing,
        "gate_policy": gate_policy,
        "gates": gates,
        "passed": all(gates.values()),
    }
    encoded = json.dumps(result, indent=2, sort_keys=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(encoded + "\n")
    print(encoded)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
