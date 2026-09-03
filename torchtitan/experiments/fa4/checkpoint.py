#
# Copyright (c) 2025-2026 Graphcore Ltd. All rights reserved.
#
"""Checkpoint hooks required by the saturated FA4 training recipe.

The local-batch-four 8B model leaves insufficient device memory for a second
world-size NCCL communicator during synchronous distributed-checkpoint
planning.  When explicitly requested, metadata collectives use a dedicated
Gloo group.  The same hook also installs the checkpointed stochastic phase for
``AdamWBF16SR`` before the trainer performs its first load.
"""

from __future__ import annotations

import os
import types
from typing import Any

import torch.distributed as dist
import torch.distributed.checkpoint as dcp

from torchtitan.components.checkpoint import AsyncMode, CheckpointManager, MODEL
from torchtitan.tools.logging import logger

from .optimizer.optimizer_sr_state import (
    AdamWBF16SR,
    register_with_checkpointer as register_optimizer_sr_with_checkpointer,
)


def install_sync_dcp_cpu_process_group(checkpointer: CheckpointManager) -> None:
    """Move synchronous DCP planning collectives to an explicit Gloo group."""
    if os.environ.get("LBT_DCP_SYNC_CPU_PROCESS_GROUP", "0") != "1":
        return
    if not dist.is_initialized():
        raise RuntimeError(
            "LBT_DCP_SYNC_CPU_PROCESS_GROUP=1 requires initialized distributed"
        )
    if getattr(checkpointer, "async_mode", None) != AsyncMode.DISABLED:
        raise RuntimeError(
            "LBT_DCP_SYNC_CPU_PROCESS_GROUP=1 requires checkpoint.async_mode="
            "'disabled'"
        )
    if getattr(checkpointer, "ft_manager", None) is not None or getattr(
        checkpointer, "enable_ft_dataloader_checkpoints", False
    ):
        raise RuntimeError(
            "the FA4 synchronous Gloo checkpoint route is incompatible with "
            "TorchFT replica checkpoints"
        )
    if getattr(checkpointer, "initial_load_in_hf", False) or getattr(
        checkpointer, "last_save_in_hf", False
    ):
        raise RuntimeError(
            "the FA4 synchronous Gloo checkpoint route supports ordinary DCP only"
        )
    if hasattr(checkpointer, "_lbt_dcp_process_group"):
        raise RuntimeError("the FA4 DCP process group is already installed")

    process_group = dist.new_group(backend="gloo")
    if dist.get_backend(process_group) != "gloo":
        raise RuntimeError("the FA4 DCP process group backend is not Gloo")
    if dist.get_world_size(process_group) != dist.get_world_size():
        raise RuntimeError("the FA4 DCP process group has the wrong world size")
    checkpointer._lbt_dcp_process_group = process_group

    def dcp_load_with_cpu_group(
        this: CheckpointManager,
        state_dict: dict[str, Any],
        checkpoint_id: str,
        from_hf: bool,
        from_quantized: bool,
    ) -> None:
        if from_hf or from_quantized:
            raise RuntimeError(
                "the FA4 Gloo checkpoint route supports ordinary DCP load only"
            )
        dcp.load(
            state_dict,
            checkpoint_id=checkpoint_id,
            process_group=process_group,
        )
        # TorchTitan flattens model state into the DCP state dictionary, so its
        # stock ordinary-load path explicitly materializes the model wrapper
        # after DCP has filled the tensors. Preserve that behavior here.
        if MODEL in this.states:
            this.states[MODEL].load_state_dict(state_dict)

    def dcp_save_with_cpu_group(
        this: CheckpointManager,
        state_dict: dict[str, Any],
        checkpoint_id: str,
        async_mode: AsyncMode,
        enable_garbage_collection: bool = False,
        to_hf: bool = False,
    ):
        del this
        if async_mode != AsyncMode.DISABLED:
            raise RuntimeError("the FA4 Gloo checkpoint route is synchronous only")
        if to_hf:
            raise RuntimeError("the FA4 Gloo checkpoint route does not export HF")
        result = dcp.save(
            state_dict,
            checkpoint_id=checkpoint_id,
            process_group=process_group,
        )
        if enable_garbage_collection:
            # Preserve the stock method's post-save cleanup without importing
            # private scheduler/checkpoint wrappers.
            from torchtitan.tools.utils import GarbageCollection

            GarbageCollection.collect("GC collection invoked by checkpointer.")
        return result

    checkpointer.dcp_save = types.MethodType(dcp_save_with_cpu_group, checkpointer)
    checkpointer.dcp_load = types.MethodType(dcp_load_with_cpu_group, checkpointer)
    logger.info(
        "LBT_SYNC_DCP_CPU_PROCESS_GROUP_READY backend=gloo world_size=%d",
        dist.get_world_size(),
    )


_ORIGINAL_CHECKPOINT_MANAGER_INIT = CheckpointManager.__init__


def _checkpoint_manager_init_with_fa4_hooks(self, *args, **kwargs) -> None:
    _ORIGINAL_CHECKPOINT_MANAGER_INIT(self, *args, **kwargs)
    if not getattr(self, "enable", False):
        return

    install_sync_dcp_cpu_process_group(self)

    optimizer_container = kwargs.get("optimizers")
    if optimizer_container is None and len(args) >= 3:
        optimizer_container = args[2]
    optimizers = getattr(optimizer_container, "optimizers", None)
    if (
        isinstance(optimizers, list)
        and optimizers
        and all(isinstance(optimizer, AdamWBF16SR) for optimizer in optimizers)
    ):
        register_optimizer_sr_with_checkpointer(self, optimizer_container, logger)


def install_checkpoint_hooks() -> None:
    """Install hooks once; called by the explicit FA4 custom import."""
    current = CheckpointManager.__init__
    if getattr(current, "_fa4_reproduction_hook", False):
        return
    _checkpoint_manager_init_with_fa4_hooks._fa4_reproduction_hook = True
    CheckpointManager.__init__ = _checkpoint_manager_init_with_fa4_hooks


__all__ = ["install_checkpoint_hooks", "install_sync_dcp_cpu_process_group"]
