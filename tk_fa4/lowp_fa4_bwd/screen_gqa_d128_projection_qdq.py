#!/usr/bin/env python3
"""Screen transpose-consistent D128 QKV projection quantization policies.

The screen is intentionally a dense-QDQ oracle.  It answers whether a format
is numerically worth integrating before CUDA fusion work, and separates input
quantization, weight quantization, and their interaction.  It does not claim
kernel speed: finalists must still match this oracle in the fused projection
and then pass isolated and end-to-end timing gates.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from tk_fa4.lowp_fa4_bwd.benchmark_llama12b_e2e import (
    Llama12B,
    _make_llama3_rope,
    _stack_lowp_qkv_weights,
    config_from_model_preset,
)
from tk_fa4.lowp_fa4_bwd.projection_quantization_reference import (
    FakeQuantResult,
    fake_quantize_mxfp4,
    fake_quantize_nvfp4,
    tensor_error_metrics,
)


DEFAULT_CORPUS = (
    Path(os.environ["FA4_CORPUS_PATH"]).expanduser()
    if os.environ.get("FA4_CORPUS_PATH")
    else None
)
DEFAULT_TOKENIZER = (
    Path(os.environ["FA4_TOKENIZER_PATH"]).expanduser()
    if os.environ.get("FA4_TOKENIZER_PATH")
    else None
)
LLAMA_BOS = 128000
LLAMA_EOS = 128001


@dataclass(frozen=True)
class Policy:
    name: str
    format: str
    activation_mode: str
    weight_mode: str
    scale_target: float | None = None


POLICIES = (
    Policy("nv_static6_h448", "nvfp4", "static6", "static6", 448.0),
    Policy("nv_static6_h384", "nvfp4", "static6", "static6", 384.0),
    Policy(
        "nv_static6_h512_saturation_stress",
        "nvfp4",
        "static6",
        "static6",
        512.0,
    ),
    Policy(
        "nv_4to6_mae_activation_h448",
        "nvfp4",
        "adaptive_mae",
        "static6",
        448.0,
    ),
    Policy(
        "nv_4to6_mse_activation_h448",
        "nvfp4",
        "adaptive_mse",
        "static6",
        448.0,
    ),
    Policy(
        "nv_4to6_mae_activation_weight_h448",
        "nvfp4",
        "adaptive_mae",
        "adaptive_mae",
        448.0,
    ),
    Policy("mx_ceil_2d", "mxfp4", "ceil", "ceil"),
    Policy("mx_rte_activation_dense_2d_weight", "mxfp4", "rte", "dense"),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=int, default=4096)
    parser.add_argument(
        "--capture-layers",
        type=int,
        nargs="+",
        default=[0],
        help="BF16 initialization layers whose normalized inputs are screened",
    )
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS,
        required=DEFAULT_CORPUS is None,
        help="input JSONL (or set FA4_CORPUS_PATH)",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=DEFAULT_TOKENIZER,
        required=DEFAULT_TOKENIZER is None,
        help="tokenizer.json (or set FA4_TOKENIZER_PATH)",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=[policy.name for policy in POLICIES],
        default=[policy.name for policy in POLICIES],
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_first_dolma_tokens(
    corpus: Path,
    tokenizer_path: Path,
    sequence: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not corpus.is_file():
        raise RuntimeError(f"corpus does not exist: {corpus}")
    if not tokenizer_path.is_file():
        raise RuntimeError(f"tokenizer does not exist: {tokenizer_path}")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    stream: list[int] = []
    documents = 0
    with corpus.open() as source:
        for line_number, line in enumerate(source, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"invalid JSON on {corpus}:{line_number}"
                ) from error
            text = str(row.get("text", ""))
            if not text.strip():
                continue
            stream.append(LLAMA_BOS)
            stream.extend(
                int(token)
                for token in tokenizer.encode(
                    text,
                    add_special_tokens=False,
                ).ids
            )
            stream.append(LLAMA_EOS)
            documents += 1
            if len(stream) >= sequence:
                break
    if len(stream) < sequence:
        raise RuntimeError(
            f"corpus supplied only {len(stream)} tokens; need {sequence}"
        )
    token_cpu = torch.tensor(stream[:sequence], dtype=torch.int64)
    return token_cpu.view(1, sequence).cuda(), {
        "corpus": str(corpus.resolve()),
        "tokenizer": str(tokenizer_path.resolve()),
        "documents_consumed": documents,
        "token_sha256": hashlib.sha256(token_cpu.numpy().tobytes()).hexdigest(),
    }


def _quantize_policy(
    policy: Policy,
    activation: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[FakeQuantResult, FakeQuantResult]:
    if policy.format == "nvfp4":
        assert policy.scale_target is not None
        activation_result = fake_quantize_nvfp4(
            activation,
            block_shape=(1, 16),
            selector=policy.activation_mode,
            scale_target=policy.scale_target,
        )
        # This 16x16 block is the non-negotiable projection-weight contract.
        weight_result = fake_quantize_nvfp4(
            weight,
            block_shape=(16, 16),
            selector=policy.weight_mode,
            scale_target=policy.scale_target,
        )
    elif policy.format == "mxfp4":
        activation_result = fake_quantize_mxfp4(
            activation,
            block_shape=(1, 32),
            scale_mode=policy.activation_mode,
        )
        # A shared 32x32 E8M0 scale and payload are reused after transpose.
        weight_result = fake_quantize_mxfp4(
            weight,
            block_shape=(32, 32),
            scale_mode=policy.weight_mode,
        )
    else:
        raise ValueError(f"unknown policy format {policy.format!r}")
    return activation_result, weight_result


def _component_slices(
    q_width: int,
    kv_width: int,
) -> dict[str, slice]:
    return {
        "q": slice(0, q_width),
        "k": slice(q_width, q_width + kv_width),
        "v": slice(q_width + kv_width, q_width + 2 * kv_width),
    }


def _per_head_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    heads: int,
    head_dim: int,
) -> dict[str, Any]:
    expected = reference.float().reshape(reference.shape[0], heads, head_dim)
    actual = candidate.float().reshape(candidate.shape[0], heads, head_dim)
    expected = expected.permute(1, 0, 2).reshape(heads, -1)
    actual = actual.permute(1, 0, 2).reshape(heads, -1)
    difference = actual - expected
    expected_norm = torch.linalg.vector_norm(expected, dim=1)
    actual_norm = torch.linalg.vector_norm(actual, dim=1)
    tiny = torch.finfo(torch.float32).tiny
    cosine = (expected * actual).sum(dim=1) / (
        expected_norm * actual_norm
    ).clamp_min(tiny)
    relative_l2 = torch.linalg.vector_norm(difference, dim=1) / (
        expected_norm.clamp_min(tiny)
    )
    worst_cosine = int(torch.argmin(cosine))
    worst_l2 = int(torch.argmax(relative_l2))
    return {
        "cosine_min": float(cosine[worst_cosine]),
        "cosine_median": float(cosine.median()),
        "cosine_max": float(cosine.max()),
        "worst_cosine_head": worst_cosine,
        "relative_l2_min": float(relative_l2.min()),
        "relative_l2_median": float(relative_l2.median()),
        "relative_l2_max": float(relative_l2[worst_l2]),
        "worst_relative_l2_head": worst_l2,
    }


def _projection_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    q_width: int,
    kv_width: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "all": tensor_error_metrics(reference, candidate),
        "components": {},
    }
    for name, component_slice in _component_slices(q_width, kv_width).items():
        expected = reference[:, component_slice]
        actual = candidate[:, component_slice]
        heads = q_heads if name == "q" else kv_heads
        result["components"][name] = {
            **tensor_error_metrics(expected, actual),
            "per_head": _per_head_metrics(
                expected,
                actual,
                heads,
                head_dim,
            ),
        }
    return result


@torch.no_grad()
def _screen_capture(
    activation: torch.Tensor,
    model: Llama12B,
    layer_index: int,
    policies: Iterable[Policy],
) -> dict[str, Any]:
    config = model.config
    weights = model.layers[layer_index].attention.weights
    qkv_weight = _stack_lowp_qkv_weights(
        config,
        weights.q,
        weights.k,
        weights.v,
    )
    reference = F.linear(activation, qkv_weight)
    capture: dict[str, Any] = {
        "layer": layer_index,
        "activation_shape": list(activation.shape),
        "weight_shape": list(qkv_weight.shape),
        "activation_rms": float(activation.float().square().mean().sqrt()),
        "weight_rms": float(qkv_weight.float().square().mean().sqrt()),
        "policies": {},
    }
    projection_arguments = {
        "q_width": config.q_width,
        "kv_width": config.kv_width,
        "q_heads": config.q_heads,
        "kv_heads": config.kv_heads,
        "head_dim": config.head_dim,
    }

    for policy in policies:
        activation_qdq, weight_qdq = _quantize_policy(
            policy,
            activation,
            qkv_weight,
        )
        activation_only = F.linear(activation_qdq.values, qkv_weight)
        weight_only = F.linear(activation, weight_qdq.values)
        both = F.linear(activation_qdq.values, weight_qdq.values)
        component_weight_metrics: dict[str, Any] = {}
        for name, component_slice in _component_slices(
            config.q_width,
            config.kv_width,
        ).items():
            component_weight_metrics[name] = tensor_error_metrics(
                qkv_weight[component_slice],
                weight_qdq.values[component_slice],
            )
        capture["policies"][policy.name] = {
            "policy": asdict(policy),
            "activation_qdq": {
                "error": tensor_error_metrics(
                    activation,
                    activation_qdq.values,
                ),
                "format": activation_qdq.diagnostics,
            },
            "weight_qdq": {
                "error": tensor_error_metrics(
                    qkv_weight,
                    weight_qdq.values,
                ),
                "components": component_weight_metrics,
                "format": weight_qdq.diagnostics,
                "transpose_consistent_2d": True,
            },
            "projection": {
                "activation_only": _projection_metrics(
                    reference,
                    activation_only,
                    **projection_arguments,
                ),
                "weight_only": _projection_metrics(
                    reference,
                    weight_only,
                    **projection_arguments,
                ),
                "activation_and_weight": _projection_metrics(
                    reference,
                    both,
                    **projection_arguments,
                ),
            },
        }
        del activation_qdq, weight_qdq
        del activation_only, weight_only, both
        gc.collect()
    del reference, qkv_weight
    return capture


@torch.no_grad()
def main() -> None:
    arguments = _parse_args()
    capture_layers = sorted(set(arguments.capture_layers))
    if capture_layers[0] < 0 or capture_layers[-1] >= 32:
        raise ValueError("capture layers must be in [0, 31]")
    if arguments.sequence <= 0 or arguments.sequence % 128:
        raise ValueError("sequence must be positive and divisible by 128")

    selected_names = set(arguments.policies)
    policies = [policy for policy in POLICIES if policy.name in selected_names]
    torch.manual_seed(arguments.seed)
    torch.cuda.manual_seed_all(arguments.seed)
    config = config_from_model_preset(
        "llama3.1-8b",
        sequence=arguments.sequence,
        layers=capture_layers[-1] + 1,
    )
    tokens, token_metadata = _load_first_dolma_tokens(
        arguments.corpus,
        arguments.tokenizer,
        arguments.sequence,
    )
    rope = _make_llama3_rope(config)
    model = Llama12B(config, rope, None)
    hidden = F.embedding(tokens, model.embedding)

    result: dict[str, Any] = {
        "schema": "gqa_d128_projection_qdq_screen_v1",
        "scope": (
            "Dense QDQ accuracy oracle at random Llama-3.1-8B initialization; "
            "not fused-kernel timing and not a trained checkpoint"
        ),
        "configuration": {
            "model_preset": config.model_preset,
            "sequence": config.sequence,
            "hidden": config.hidden,
            "q_heads": config.q_heads,
            "kv_heads": config.kv_heads,
            "head_dim": config.head_dim,
            "instantiated_layers": config.layers,
            "capture_layers": capture_layers,
            "seed": arguments.seed,
            "policies": [asdict(policy) for policy in policies],
        },
        "data": token_metadata,
        "captures": [],
    }

    for layer_index, layer in enumerate(model.layers):
        normalized_attention = layer.attention_norm(hidden).reshape(
            config.sequence,
            config.hidden,
        )
        if layer_index in capture_layers:
            result["captures"].append(
                _screen_capture(
                    normalized_attention,
                    model,
                    layer_index,
                    policies,
                )
            )
        hidden = hidden + layer.attention(normalized_attention.reshape_as(hidden))
        normalized_ffn = layer.ffn_norm(hidden)
        hidden = hidden + layer.mlp(normalized_ffn)

    result["memory"] = {
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
