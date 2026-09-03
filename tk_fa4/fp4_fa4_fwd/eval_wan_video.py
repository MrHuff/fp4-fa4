#!/usr/bin/env python3
"""Paired Wan2.1 evaluation for the retained FP4 attention kernel.

This harness replaces only Wan's noncausal video self-attention. Text
cross-attention and every other model operation remain in BF16. It runs a
BF16 reference and one or more FP4 providers from the same checkpoint,
prompt, seed, and diffusion schedule, then compares the final latents.

The current TK extension is batch-1 and shape-specialized. Wan2.1 executes
conditional and unconditional guidance as separate batch-1 transformer
calls, so no model batching workaround is required.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

try:
    from .eval_regular_attention import (
        RegularAttentionRunner,
        authenticate_asset_manifest,
        global_anchor_indices,
        load_extension,
        parse_layer_indices,
        portable_asset_identity,
        portable_file_identity,
        tensor_finite_stats,
        tensor_metrics,
    )
except ImportError:  # direct script execution
    from eval_regular_attention import (
        RegularAttentionRunner,
        authenticate_asset_manifest,
        global_anchor_indices,
        load_extension,
        parse_layer_indices,
        portable_asset_identity,
        portable_file_identity,
        tensor_finite_stats,
        tensor_metrics,
    )


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_MODEL = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
DEFAULT_OUTPUT = Path(
    "results/fp4_fa4_wan_20260805/wan21_1p3b_s7680.json"
)
DEFAULT_PROMPT = (
    "A red vintage car drives along a winding coastal road at sunset, "
    "with ocean waves below and natural camera motion."
)
DEFAULT_HAO_ROOT = REPO_ROOT / "third_party/hao_flash_attention_fp4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--model-path",
        type=Path,
        help=(
            "Authenticated local model snapshot. --model remains the immutable "
            "logical identifier checked by a policy manifest."
        ),
    )
    parser.add_argument(
        "--asset-manifest",
        type=Path,
        help="fa4_external_assets_v1 manifest rechecked before model use.",
    )
    parser.add_argument("--model-asset", help="Model key in --asset-manifest.")
    parser.add_argument(
        "--providers",
        default="bf16,tk",
        help=(
            "Comma-separated providers. Supported: bf16, hao-bf16, tk, "
            "hao-native, hao-fp8. Low-precision HAO providers require an "
            "NV/NV topology extension."
        ),
    )
    parser.add_argument(
        "--hao-root",
        type=Path,
        default=DEFAULT_HAO_ROOT,
        help="HAO flash-attention-fp4 checkout used by hao-bf16.",
    )
    parser.add_argument("--extension", type=Path)
    parser.add_argument(
        "--extension-module",
        default="_C_tk_hao_direct_fp4pv",
    )
    parser.add_argument(
        "--layer-extension",
        action="append",
        default=[],
        metavar="LAYERS=PATH:MODULE",
        help=(
            "Use a shape-compatible kernel policy for selected attention "
            "layers, for example 27-29=/tmp/anchor.so:_C_candidate."
        ),
    )
    parser.add_argument(
        "--policy-manifest",
        type=Path,
        help=(
            "Load the base and layer-routed extensions from a manifest "
            "created by build_wan_nv_mx_bundle.py."
        ),
    )
    parser.add_argument(
        "--policy",
        choices=("fast", "accurate"),
        default="fast",
        help="Policy selected from --policy-manifest.",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num-frames", type=int, default=17)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument(
        "--output-type",
        choices=("latent", "np"),
        default="latent",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--save-tensors",
        action="store_true",
        help="Save each provider's final latent tensor beside the JSON.",
    )
    parser.add_argument(
        "--save-media",
        action="store_true",
        help="For decoded output, save an MP4 and a three-frame PNG strip.",
    )
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument(
        "--key-centering",
        choices=("none", "sequence-mean"),
        default="none",
        help=(
            "Apply a softmax-invariant K transformation before FP4 QK. "
            "Sequence-mean centering is a numerical experiment until it "
            "is fused into K preparation."
        ),
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def authenticate_policy_extension(
    manifest_path: Path,
    record: object,
    label: str,
) -> tuple[Path, dict[str, str | int]]:
    if not isinstance(record, dict):
        raise ValueError(f"{label} extension record must be an object")
    required = {"path", "module", "bytes", "sha256"}
    if not required <= set(record):
        raise ValueError(f"{label} extension record is incomplete")
    path_text = record["path"]
    module = record["module"]
    if not isinstance(path_text, str) or not path_text:
        raise ValueError(f"{label} extension path is invalid")
    relative = Path(path_text)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} extension path must be bundle-relative")
    if not isinstance(module, str) or not module:
        raise ValueError(f"{label} extension module is invalid")
    expected_bytes = record["bytes"]
    expected_sha256 = record["sha256"]
    if type(expected_bytes) is not int or expected_bytes <= 0:
        raise ValueError(f"{label} extension byte identity is invalid")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        raise ValueError(f"{label} extension SHA256 is invalid")
    path = (manifest_path.parent / relative).resolve()
    try:
        path.relative_to(manifest_path.parent)
    except ValueError as error:
        raise ValueError(f"{label} extension escapes its policy bundle") from error
    identity = portable_file_identity(path)
    if identity["bytes"] != expected_bytes:
        raise ValueError(f"{label} extension byte identity mismatch")
    if identity["sha256"] != expected_sha256:
        raise ValueError(f"{label} extension SHA256 mismatch")
    return path, {**identity, "module": module}


def apply_policy_manifest(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.policy_manifest is None:
        return None
    if args.layer_extension:
        raise ValueError(
            "--policy-manifest cannot be combined with --layer-extension"
        )

    manifest_path = args.policy_manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "tk_wan_nv_mx_policy_bundle_v2":
        raise ValueError(f"unsupported policy manifest: {manifest_path}")
    if manifest.get("model") != args.model:
        raise ValueError(
            f"manifest model {manifest.get('model')} does not match {args.model}"
        )
    try:
        selected = manifest["policies"][args.policy]
        base = selected["base"]
    except KeyError as error:
        raise ValueError(
            f"policy {args.policy!r} is absent from {manifest_path}"
        ) from error

    base_path, base_identity = authenticate_policy_extension(
        manifest_path,
        base,
        "base",
    )
    args.extension = base_path
    args.extension_module = base["module"]
    layer_identities: list[dict[str, Any]] = []
    for layer_extension in selected.get("layer_extensions", []):
        path, identity = authenticate_policy_extension(
            manifest_path,
            layer_extension,
            "layer",
        )
        args.layer_extension.append(
            f"{layer_extension['layers']}={path}:{layer_extension['module']}"
        )
        layer_identities.append(
            {
                "layers": layer_extension["layers"],
                "purpose": layer_extension.get("purpose"),
                **identity,
            }
        )
    return {
        "manifest": portable_file_identity(manifest_path),
        "policy": args.policy,
        "base": base_identity,
        "layer_extensions": layer_identities,
        "guard_layers": manifest.get("guard_layers"),
        "qk_guard": manifest.get("qk_guard"),
        "fast_affine_code_map": manifest.get("fast_affine_code_map"),
        "formats": manifest.get("formats"),
        "softmax": manifest.get("softmax"),
    }


def apply_rotary_emb(
    hidden_states: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> torch.Tensor:
    """Match diffusers.models.transformers.transformer_wan exactly."""
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    output = torch.empty_like(hidden_states)
    output[..., 0::2] = x1 * cos - x2 * sin
    output[..., 1::2] = x1 * sin + x2 * cos
    return output.type_as(hidden_states)


def qkv_projections(
    attention: Any,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if attention.fused_projections:
        return attention.to_qkv(hidden_states).chunk(3, dim=-1)
    return (
        attention.to_q(hidden_states),
        attention.to_k(hidden_states),
        attention.to_v(hidden_states),
    )


class WanLowPrecisionSelfAttentionProcessor:
    """Wan self-attention processor backed by RegularAttentionRunner."""

    _attention_backend = None
    _parallel_config = None

    def __init__(
        self,
        runner: RegularAttentionRunner,
        layer_index: int,
        shadow_bf16: bool = False,
        lowp_layers: set[int] | None = None,
    ) -> None:
        self.runner = runner
        self.layer_index = layer_index
        self.shadow_bf16 = shadow_bf16
        self.lowp_layers = lowp_layers
        self.call_count = 0
        self.observed_shapes: set[tuple[int, ...]] = set()
        self.shadow_records: list[dict[str, Any]] = []

    def __call__(
        self,
        attention: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        **_: Any,
    ) -> torch.Tensor:
        if encoder_hidden_states is not None:
            raise ValueError("the TK Wan processor only supports self-attention")
        if attention_mask is not None:
            raise ValueError("the retained TK kernel is unmasked and noncausal")

        query, key, value = qkv_projections(attention, hidden_states)
        query = attention.norm_q(query)
        key = attention.norm_k(key)

        query = query.unflatten(2, (attention.heads, -1))
        key = key.unflatten(2, (attention.heads, -1))
        value = value.unflatten(2, (attention.heads, -1))
        if rotary_emb is not None:
            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        self.observed_shapes.add(tuple(query.shape))
        batch, seqlen, heads, dim = query.shape
        expected = (
            1,
            self.runner.target_seqlen,
            self.runner.target_heads,
            self.runner.target_dim,
        )
        if (batch, seqlen, heads, dim) != expected:
            raise ValueError(
                f"Wan attention shape {(batch, seqlen, heads, dim)} does "
                f"not match extension shape {expected}"
            )

        query_bhsd = query.permute(0, 2, 1, 3).contiguous()
        original_key_bhsd = key.permute(0, 2, 1, 3).contiguous()
        key_bhsd = self.runner.center_key(original_key_bhsd, None)
        value_bhsd = value.permute(0, 2, 1, 3).contiguous()
        use_lowp = self.lowp_layers is None or self.layer_index in self.lowp_layers
        reference = None
        lowp_error = None
        if use_lowp:
            try:
                context = self.runner(
                    query_bhsd,
                    key_bhsd,
                    value_bhsd,
                    layer_index=self.layer_index,
                    invocation_index=self.call_count,
                )
            except FloatingPointError as error:
                if not self.shadow_bf16:
                    raise
                lowp_error = str(error)
                context = (
                    self.runner._output[:, :seqlen, :heads, :dim]
                    .permute(0, 2, 1, 3)
                    .contiguous()
                )
        else:
            reference = F.scaled_dot_product_attention(
                query_bhsd,
                original_key_bhsd,
                value_bhsd,
                dropout_p=0.0,
                is_causal=False,
            )
            context = reference
        if self.shadow_bf16 and use_lowp:
            if reference is None:
                reference = F.scaled_dot_product_attention(
                    query_bhsd,
                    original_key_bhsd,
                    value_bhsd,
                    dropout_p=0.0,
                    is_causal=False,
                )
            sample_count = min(32, seqlen)
            sample_rows = (
                torch.linspace(
                    0,
                    seqlen - 1,
                    sample_count,
                    device=query.device,
                    dtype=torch.float32,
                )
                .round()
                .to(torch.long)
            )
            sampled_scores = torch.matmul(
                query_bhsd.index_select(-2, sample_rows).float(),
                key_bhsd.float().transpose(-1, -2),
            ) / math.sqrt(dim)
            sampled_row_max = sampled_scores.amax(dim=-1)
            record: dict[str, Any] = {
                "call": self.call_count,
                "sampled_score_min": float(sampled_scores.min().item()),
                "sampled_score_max": float(sampled_scores.max().item()),
                "sampled_row_max_min": float(
                    sampled_row_max.min().item()
                ),
                "sampled_row_max_max": float(
                    sampled_row_max.max().item()
                ),
                "query_rms": float(
                    query_bhsd.float().square().mean().sqrt().item()
                ),
                "key_rms": float(
                    key_bhsd.float().square().mean().sqrt().item()
                ),
                "value_rms": float(
                    value_bhsd.float().square().mean().sqrt().item()
                ),
                "lowp_finite": bool(context.isfinite().all().item()),
            }
            for anchor_count in (32, 128):
                anchor_columns = global_anchor_indices(
                    seqlen,
                    anchor_count,
                    query.device,
                )
                anchor_max = sampled_scores.index_select(
                    -1, anchor_columns
                ).amax(dim=-1)
                anchor_gap = sampled_row_max - anchor_max
                record[f"anchor{anchor_count}_gap_max"] = float(
                    anchor_gap.max().item()
                )
                record[f"anchor{anchor_count}_gap_mean"] = float(
                    anchor_gap.mean().item()
                )
            if record["lowp_finite"]:
                record["lowp_vs_bf16"] = tensor_metrics(context, reference)
            if lowp_error is not None:
                record["lowp_error"] = lowp_error
            self.shadow_records.append(record)
            context = reference
        output = context.permute(0, 2, 1, 3).flatten(2, 3)
        output = output.type_as(query)
        output = attention.to_out[0](output)
        output = attention.to_out[1](output)
        self.call_count += 1
        return output


class WanCuteBF16SelfAttentionProcessor:
    """Wan self-attention processor backed by HAO's CuTe-DSL BF16 FA4."""

    _attention_backend = None
    _parallel_config = None

    def __init__(self, attention_fn: Any) -> None:
        self.attention_fn = attention_fn
        self.call_count = 0
        self.observed_shapes: set[tuple[int, ...]] = set()

    def __call__(
        self,
        attention: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        **_: Any,
    ) -> torch.Tensor:
        if encoder_hidden_states is not None:
            raise ValueError("the HAO BF16 Wan processor only supports self-attention")
        if attention_mask is not None:
            raise ValueError("the HAO BF16 Wan processor is unmasked and noncausal")

        query, key, value = qkv_projections(attention, hidden_states)
        query = attention.norm_q(query)
        key = attention.norm_k(key)
        query = query.unflatten(2, (attention.heads, -1))
        key = key.unflatten(2, (attention.heads, -1))
        value = value.unflatten(2, (attention.heads, -1))
        if rotary_emb is not None:
            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        self.observed_shapes.add(tuple(query.shape))
        attention_result = self.attention_fn(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            causal=False,
        )
        context = (
            attention_result[0]
            if isinstance(attention_result, tuple)
            else attention_result
        )
        output = context.flatten(2, 3).type_as(query)
        output = attention.to_out[0](output)
        output = attention.to_out[1](output)
        self.call_count += 1
        return output


def load_hao_bf16_attention(hao_root: Path) -> Any:
    resolved_root = hao_root.resolve()
    root = str(resolved_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    flash_attn = sys.modules.get("flash_attn")
    hao_package = str(resolved_root / "flash_attn")
    if flash_attn is not None and hasattr(flash_attn, "__path__"):
        if hao_package not in flash_attn.__path__:
            flash_attn.__path__.insert(0, hao_package)
    from flash_attn.cute.interface import flash_attn_func

    return flash_attn_func


def make_runner(
    extension: Any,
    attention_backend: str,
    key_centering: str,
    layer_extensions: dict[int, Any] | None = None,
) -> RegularAttentionRunner:
    topology = dict(extension.read_hao_direct_topology())
    global_anchor_samples = (
        128
        if bool(topology.get("mx_global_anchor128", False))
        else (
            64
            if bool(topology.get("nv_global_anchor64", False))
            else (
                32
                if bool(topology.get("mx_global_anchor32", False))
                or bool(topology.get("nv_global_anchor32", False))
                else 0
            )
        )
    )
    return RegularAttentionRunner(
        extension,
        layer_extensions=layer_extensions,
        attention_backend=attention_backend,
        mask_value=20.0,
        scale_factors=[],
        scale_sweep_samples=0,
        finite_diagnostics=False,
        interleave_quarters=False,
        global_anchor=bool(global_anchor_samples),
        global_anchor_samples=global_anchor_samples or 32,
        mx_q_quant_mode=0,
        mx_k_quant_mode=0,
        mx_v_quant_mode=0,
        key_centering=key_centering,
        nv_qk_fold_k64_scales="auto",
        nv_qk_fold_scale_select="mse",
        nv_qk_fold_scale_multiplier=1.0,
        collect_layer_metrics=False,
    )


def install_lowp_processors(
    transformer: Any,
    runner: RegularAttentionRunner,
    shadow_bf16: bool = False,
    lowp_layers: set[int] | None = None,
) -> list[WanLowPrecisionSelfAttentionProcessor]:
    processors = []
    for layer_index, block in enumerate(transformer.blocks):
        processor = WanLowPrecisionSelfAttentionProcessor(
            runner,
            layer_index,
            shadow_bf16=shadow_bf16,
            lowp_layers=lowp_layers,
        )
        block.attn1.set_processor(processor)
        processors.append(processor)
    return processors


def install_hao_bf16_processors(
    transformer: Any,
    attention_fn: Any,
) -> list[WanCuteBF16SelfAttentionProcessor]:
    processors = []
    for block in transformer.blocks:
        processor = WanCuteBF16SelfAttentionProcessor(attention_fn)
        block.attn1.set_processor(processor)
        processors.append(processor)
    return processors


def restore_processors(transformer: Any, processors: list[Any]) -> None:
    for block, processor in zip(transformer.blocks, processors, strict=True):
        block.attn1.set_processor(processor)


def latent_token_count(
    pipeline: Any,
    height: int,
    width: int,
    num_frames: int,
) -> int:
    latent_frames = (num_frames - 1) // pipeline.vae_scale_factor_temporal + 1
    latent_height = height // pipeline.vae_scale_factor_spatial
    latent_width = width // pipeline.vae_scale_factor_spatial
    patch_t, patch_h, patch_w = pipeline.transformer.config.patch_size
    return (
        math.ceil(latent_frames / patch_t)
        * math.ceil(latent_height / patch_h)
        * math.ceil(latent_width / patch_w)
    )


def run_provider(
    pipeline: Any,
    provider: str,
    runner: RegularAttentionRunner | None,
    original_processors: list[Any],
    args: argparse.Namespace,
    lowp_layers: set[int] | None = None,
    hao_bf16_attention: Any | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    restore_processors(pipeline.transformer, original_processors)
    lowp_processors: list[WanLowPrecisionSelfAttentionProcessor] = []
    hao_bf16_processors: list[WanCuteBF16SelfAttentionProcessor] = []
    if provider == "hao-bf16":
        if hao_bf16_attention is None:
            raise ValueError("hao-bf16 provider requires the CuTe-DSL function")
        hao_bf16_processors = install_hao_bf16_processors(
            pipeline.transformer,
            hao_bf16_attention,
        )
    elif provider != "bf16":
        if runner is None:
            raise ValueError(f"{provider} provider requires an extension")
        runner.begin_sample(0)
        lowp_processors = install_lowp_processors(
            pipeline.transformer,
            runner,
            shadow_bf16=provider == "tk-shadow",
            lowp_layers=lowp_layers,
        )

    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    torch.cuda.synchronize(args.device)
    started = time.perf_counter()
    with torch.inference_mode():
        result = pipeline(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            output_type=args.output_type,
        ).frames
    torch.cuda.synchronize(args.device)
    elapsed = time.perf_counter() - started

    if not isinstance(result, torch.Tensor):
        import numpy as np

        if isinstance(result, list):
            result = np.stack(result)
        result = torch.from_numpy(result)
    result_cpu = result.detach().float().cpu()
    record: dict[str, Any] = {
        "elapsed_seconds": elapsed,
        "output": tensor_finite_stats(result_cpu),
    }
    if lowp_processors:
        record.update(
            {
                "self_attention_calls": sum(
                    processor.call_count for processor in lowp_processors
                ),
                "calls_per_layer": [
                    processor.call_count for processor in lowp_processors
                ],
                "lowp_layers": (
                    "all" if lowp_layers is None else sorted(lowp_layers)
                ),
                "observed_shapes": sorted(
                    {
                        str(shape)
                        for processor in lowp_processors
                        for shape in processor.observed_shapes
                    }
                ),
                "runner": runner.summary(),
            }
        )
        if provider == "tk-shadow":
            record["shadow_layers"] = [
                {
                    "layer": processor.layer_index,
                    "calls": processor.shadow_records,
                }
                for processor in lowp_processors
            ]
    elif hao_bf16_processors:
        record.update(
            {
                "self_attention_calls": sum(
                    processor.call_count for processor in hao_bf16_processors
                ),
                "calls_per_layer": [
                    processor.call_count for processor in hao_bf16_processors
                ],
                "observed_shapes": sorted(
                    {
                        str(shape)
                        for processor in hao_bf16_processors
                        for shape in processor.observed_shapes
                    }
                ),
                "attention_backend": "hao-cute-dsl-bf16",
            }
        )
    return result_cpu, record


def main() -> None:
    args = parse_args()
    policy_manifest = apply_policy_manifest(args)
    asset_manifest_identity = None
    if args.model_path is not None:
        if args.asset_manifest is None or args.model_asset is None:
            raise ValueError(
                "--model-path requires --asset-manifest and --model-asset"
            )
        model_root = args.model_path.resolve()
        model_identity, asset_manifest_identity = authenticate_asset_manifest(
            args.asset_manifest,
            args.model_asset,
            "model",
            model_root,
        )
        if model_identity["identifier"] != args.model:
            raise ValueError(
                f"authenticated model identifier {model_identity['identifier']!r} "
                f"does not match {args.model!r}"
            )
        model_source = str(model_root)
    else:
        model_identity = portable_asset_identity(
            args.model,
            identifier=None,
            revision=None,
            tree_sha256=None,
        )
        model_source = args.model
    providers = [item.strip() for item in args.providers.split(",") if item.strip()]
    if not providers or len(set(providers)) != len(providers):
        raise ValueError("--providers must contain distinct provider names")
    unknown = set(providers) - {
        "bf16",
        "hao-bf16",
        "tk",
        "tk-shadow",
        "hao-native",
        "hao-fp8",
    }
    if unknown:
        raise ValueError(f"unsupported providers: {sorted(unknown)}")

    hao_bf16_attention = (
        load_hao_bf16_attention(args.hao_root)
        if "hao-bf16" in providers
        else None
    )

    from diffusers import WanPipeline

    extension = None
    layer_extensions: dict[int, Any] = {}
    runners: dict[str, RegularAttentionRunner] = {}
    extension_provider_names = {"tk", "tk-shadow", "hao-native", "hao-fp8"}
    if any(provider in extension_provider_names for provider in providers):
        if args.extension is None:
            raise ValueError(
                "an explicit --extension or --policy-manifest is required for "
                "a low-precision provider"
            )
        extension = load_extension(args.extension.resolve(), args.extension_module)
        for specification in args.layer_extension:
            try:
                layers_text, extension_text = specification.split("=", 1)
                path_text, module_name = extension_text.rsplit(":", 1)
            except ValueError as error:
                raise ValueError(
                    "--layer-extension must use LAYERS=PATH:MODULE"
                ) from error
            layer_extension = load_extension(
                Path(path_text).resolve(), module_name
            )
            for layer in parse_layer_indices(layers_text):
                if layer in layer_extensions:
                    raise ValueError(
                        f"multiple extensions specified for layer {layer}"
                    )
                layer_extensions[layer] = layer_extension
        for provider in providers:
            if provider in extension_provider_names:
                runners[provider] = make_runner(
                    extension,
                    "tk" if provider == "tk-shadow" else provider,
                    args.key_centering,
                    layer_extensions=layer_extensions,
                )

    pipeline = WanPipeline.from_pretrained(
        model_source,
        torch_dtype=torch.bfloat16,
        local_files_only=args.local_files_only,
    )
    pipeline.to(args.device)
    pipeline.set_progress_bar_config(disable=False)

    original_processors = [
        block.attn1.processor for block in pipeline.transformer.blocks
    ]
    sequence_length = latent_token_count(
        pipeline, args.height, args.width, args.num_frames
    )
    model_shape = {
        "layers": len(pipeline.transformer.blocks),
        "heads": int(pipeline.transformer.config.num_attention_heads),
        "head_dim": int(pipeline.transformer.config.attention_head_dim),
        "sequence_length": sequence_length,
    }
    if runners:
        runner = next(iter(runners.values()))
        expected = (
            runner.target_seqlen,
            runner.target_heads,
            runner.target_dim,
        )
        actual = (
            model_shape["sequence_length"],
            model_shape["heads"],
            model_shape["head_dim"],
        )
        if actual != expected:
            raise ValueError(
                f"model attention shape {actual} does not match extension {expected}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, torch.Tensor] = {}
    records: dict[str, Any] = {}
    for provider in providers:
        try:
            output, record = run_provider(
                pipeline,
                provider,
                runners.get(provider),
                original_processors,
                args,
                hao_bf16_attention=hao_bf16_attention,
            )
        except Exception as error:
            records[provider] = {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            }
            continue
        outputs[provider] = output
        record["status"] = "complete"
        records[provider] = record
        if args.save_tensors:
            tensor_path = args.output.with_name(
                f"{args.output.stem}_{provider}.pt"
            )
            torch.save(output, tensor_path)
            record["tensor"] = portable_file_identity(tensor_path)
        if args.save_media and args.output_type == "np":
            from diffusers.utils import export_to_video
            from PIL import Image

            video = output[0].numpy()
            video_path = args.output.with_name(
                f"{args.output.stem}_{provider}.mp4"
            )
            export_to_video(video, video_path, fps=args.fps)
            selected = [0, len(video) // 2, len(video) - 1]
            frames = [
                Image.fromarray(
                    (video[index].clip(0.0, 1.0) * 255.0).round().astype(
                        "uint8"
                    )
                )
                for index in selected
            ]
            strip = Image.new(
                "RGB",
                (sum(frame.width for frame in frames), frames[0].height),
            )
            left = 0
            for frame in frames:
                strip.paste(frame, (left, 0))
                left += frame.width
            preview_path = args.output.with_name(
                f"{args.output.stem}_{provider}_preview.png"
            )
            strip.save(preview_path)
            record["video"] = portable_file_identity(video_path)
            record["preview"] = portable_file_identity(preview_path)

    comparisons: dict[str, Any] = {}
    reference_provider = (
        "hao-bf16" if "hao-bf16" in outputs else "bf16" if "bf16" in outputs else None
    )
    if reference_provider is not None:
        for provider, output in outputs.items():
            if provider != reference_provider:
                comparisons[f"{provider}_vs_{reference_provider}"] = tensor_metrics(
                    output, outputs[reference_provider]
                )

    payload = {
        "schema": "tk_fp4_fa4_wan_paired_v2",
        "model": model_identity,
        "asset_manifest": asset_manifest_identity,
        "scope": (
            "Wan video self-attention only; text cross-attention and all "
            "other operations remain BF16"
        ),
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "generation": {
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "seed": args.seed,
            "output_type": args.output_type,
            "key_centering": args.key_centering,
        },
        "model_attention_shape": model_shape,
        "policy_manifest": policy_manifest,
        "reference_provider": reference_provider,
        "providers": records,
        "comparisons": comparisons,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
