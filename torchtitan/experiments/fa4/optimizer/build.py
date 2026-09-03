#
# Copyright (c) 2025-2026 Graphcore Ltd. All rights reserved.
#
"""Portable optimizer factory for FA4 TorchTitan runs."""

import torch
import torch.nn as nn

from torchtitan.components.ft import FTManager
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.optimizer import (
    build_optimizers as build_torchtitan_optimizers,
)
from torchtitan.config import Optimizer as OptimizerConfig
from torchtitan.distributed import ParallelDims


_ADAMW_BF16_SR = "AdamWBF16SR"


def _build_adamw_bf16_sr(
    model_parts: list[nn.Module],
    optimizer_config: OptimizerConfig,
    ft_manager: FTManager | None,
) -> OptimizersContainer:
    if optimizer_config.implementation != "fused":
        raise ValueError("AdamWBF16SR requires optimizer.implementation='fused'")
    if optimizer_config.early_step_in_backward:
        raise ValueError("AdamWBF16SR does not support early_step_in_backward")
    if ft_manager is not None and ft_manager.enabled:
        raise ValueError("AdamWBF16SR is not authenticated with TorchFT")
    if len(model_parts) != 1:
        raise ValueError(
            "AdamWBF16SR requires exactly one model part; pipeline parallelism "
            "does not preserve its single stochastic phase"
        )

    invalid = [
        (f"model_parts.{part_index}.{name}", parameter.dtype)
        for part_index, model_part in enumerate(model_parts)
        for name, parameter in model_part.named_parameters()
        if parameter.requires_grad and parameter.dtype != torch.bfloat16
    ]
    if invalid:
        examples = ", ".join(f"{name}={dtype}" for name, dtype in invalid[:8])
        if len(invalid) > 8:
            examples += f", ... ({len(invalid)} total)"
        raise ValueError(
            "AdamWBF16SR requires every trainable parameter in BF16; found " + examples
        )

    from .optimizer_sr_state import AdamWBF16SR, configured_sr_seed

    kwargs = {
        "lr": optimizer_config.lr,
        "betas": (optimizer_config.beta1, optimizer_config.beta2),
        "eps": optimizer_config.eps,
        "weight_decay": optimizer_config.weight_decay,
        "bf16_stochastic_round": True,
        "sr_seed": configured_sr_seed(),
    }
    return OptimizersContainer(model_parts, AdamWBF16SR, kwargs)


def build_optimizers(
    model_parts: list[nn.Module],
    optimizer_config: OptimizerConfig,
    parallel_dims: ParallelDims,
    ft_manager: FTManager | None = None,
) -> OptimizersContainer:
    if optimizer_config.name == _ADAMW_BF16_SR:
        return _build_adamw_bf16_sr(model_parts, optimizer_config, ft_manager)
    return build_torchtitan_optimizers(
        model_parts, optimizer_config, parallel_dims, ft_manager
    )


__all__ = ["build_optimizers"]
