#
# Copyright (c) 2025-2026 Graphcore Ltd. All rights reserved.
#
"""Llama model geometries and optimizer wiring used by the FA4 paper."""

from dataclasses import replace

from torchtitan.models.llama3 import get_train_spec as get_llama3_train_spec
from torchtitan.models.llama3.model.args import RoPEScalingArgs, TransformerModelArgs
from torchtitan.protocols.train_spec import register_train_spec

from .data import build_fa4_text_dataloader, register_slimpajama
from .optimizer import build_optimizers
from .validator import build_fa4_validator


def _rope_scaling(factor: float) -> RoPEScalingArgs:
    return RoPEScalingArgs(
        scaling_factor=factor,
        low_freq_factor=1.0,
        high_freq_factor=4.0,
        original_max_position_embeddings=8192,
    )


fa4_llama_args = {
    # Llama-3.2-like 1.2B topology used by the D64 experiments.  The exact
    # converter ties input/output embeddings and authenticates the resulting
    # 1,235,814,400 unique parameters.
    "1B": TransformerModelArgs(
        dim=2048,
        n_layers=16,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=8192 / 4 / 2048 * 3 / 2,
        multiple_of=256,
        rope_theta=500000,
        rope_scaling_args=_rope_scaling(32.0),
        max_seq_len=4096,
    ),
    # Llama-3.1-style 8B topology used by the D128 experiments.
    "8B_llama3_blog": TransformerModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
        rope_scaling_args=_rope_scaling(8.0),
        max_seq_len=4096,
    ),
    # Accepted alias in the recovered adapter.
    "8B": TransformerModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
        rope_scaling_args=_rope_scaling(8.0),
        max_seq_len=4096,
    ),
}


def register_fa4_train_spec() -> None:
    register_slimpajama()
    stock = get_llama3_train_spec()
    register_train_spec(
        "llama3_gc",
        replace(
            stock,
            model_args=fa4_llama_args,
            build_optimizers_fn=build_optimizers,
            build_dataloader_fn=build_fa4_text_dataloader,
            build_validator_fn=build_fa4_validator,
        ),
    )


__all__ = ["fa4_llama_args", "register_fa4_train_spec"]
