#
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
#
"""Checkpoint alignment and opt-in fingerprints for prefetched training batches."""

from __future__ import annotations

import copy
import hashlib
import os
import pickle
import sys
from dataclasses import dataclass
from typing import Any, Iterator

import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from torchtitan.components.dataloader import ParallelAwareDataloader


_DATALOADER_STATE_KEY = "dataloader"


@dataclass(frozen=True)
class _DeferredParallelAwareState:
    """Raw TorchData snapshot awaiting the loader's rank-aware serialization."""

    raw_state: dict[str, Any]
    rank_id: str
    world_size: int


def _capture_state_before_prefetch(dataloader: Any) -> Any:
    """Capture state cheaply when the loader's public wrapper only serializes it."""
    if (
        isinstance(dataloader, ParallelAwareDataloader)
        and type(dataloader).state_dict is ParallelAwareDataloader.state_dict
        and dataloader.num_workers > 0
    ):
        # Multi-process StatefulDataLoader replaces `_snapshot` after every
        # yielded batch. Retaining that raw snapshot is therefore safe while
        # avoiding ParallelAwareDataloader's pickle.dumps() on every prefetch.
        return _DeferredParallelAwareState(
            raw_state=StatefulDataLoader.state_dict(dataloader),
            rank_id=dataloader._rank_id,
            world_size=dataloader.dp_world_size,
        )
    return dataloader.state_dict()


def _materialize_captured_state(state: Any) -> dict[str, Any]:
    if isinstance(state, _DeferredParallelAwareState):
        # Match ParallelAwareDataloader.state_dict()'s on-disk schema exactly.
        return {
            state.rank_id: pickle.dumps(copy.deepcopy(state.raw_state)),
            "world_size": state.world_size,
        }
    return copy.deepcopy(state)


class CheckpointAlignedDataloader:
    """Expose the state immediately before a trainer-owned lookahead batch.

    ``StatefulDataLoader.state_dict()`` describes batches returned by ``next()``.
    The LBT CUDA batch generator keeps one such batch in a device-copy lookahead.
    While that batch is pending, checkpointing the underlying dataloader directly
    would resume *after* it. This adapter retains the state captured immediately
    before the lookahead ``next()`` and exposes that state to the checkpointer.

    The retained state is copied only when a checkpoint requests ``state_dict``.
    TorchData replaces its per-step snapshot instead of mutating the prior one, so
    the per-microbatch hot path only stores a reference to the prior snapshot.
    """

    def __init__(self, dataloader: Any):
        self.dataloader = dataloader
        self._state_before_prefetch: Any | None = None
        self._has_prefetched_batch = False

    def next_for_prefetch(self, iterator: Iterator[Any]) -> Any:
        """Return the next batch while retaining its pre-consumption state."""
        if self._has_prefetched_batch:
            raise RuntimeError(
                "Cannot prefetch another batch before promoting the pending batch"
            )

        state_before_prefetch = _capture_state_before_prefetch(self.dataloader)
        try:
            batch = next(iterator)
        except BaseException:
            self._state_before_prefetch = None
            self._has_prefetched_batch = False
            raise

        self._state_before_prefetch = state_before_prefetch
        self._has_prefetched_batch = True
        return batch

    def mark_prefetched_batch_current(self) -> None:
        """Mark the pending batch as the batch about to be yielded to training."""
        if not self._has_prefetched_batch:
            raise RuntimeError("No prefetched batch is available to promote")
        self._state_before_prefetch = None
        self._has_prefetched_batch = False

    def state_dict(self) -> dict[str, Any]:
        if self._has_prefetched_batch:
            assert self._state_before_prefetch is not None
            return _materialize_captured_state(self._state_before_prefetch)
        return self.dataloader.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._state_before_prefetch = None
        self._has_prefetched_batch = False
        self.dataloader.load_state_dict(state_dict)


def install_checkpoint_aligned_dataloader(
    checkpointer: Any, dataloader: Any
) -> CheckpointAlignedDataloader:
    """Install one adapter in persistent and FT dataloader checkpoint state."""
    adapter = CheckpointAlignedDataloader(dataloader)

    states = getattr(checkpointer, "states", None)
    if (
        not isinstance(states, dict)
        or states.get(_DATALOADER_STATE_KEY) is not dataloader
    ):
        raise RuntimeError("Checkpointer does not reference the trainer dataloader")
    states[_DATALOADER_STATE_KEY] = adapter

    ft_states = getattr(checkpointer, "ft_states", None)
    if (
        isinstance(ft_states, dict)
        and ft_states.get(_DATALOADER_STATE_KEY) is dataloader
    ):
        ft_states[_DATALOADER_STATE_KEY] = adapter

    return adapter


@dataclass(frozen=True)
class BatchFingerprint:
    input_sha256: str
    labels_sha256: str
    combined_sha256: str
    input_numel: int
    labels_numel: int


def batch_fingerprints_enabled() -> bool:
    """Whether to fingerprint yielded batches; false has no tensor work."""
    return os.environ.get("USE_LBT_DEBUG_BATCH_HASHES", "0") == "1"


def _update_tree_hash(digest: Any, value: Any) -> int:
    if torch.is_tensor(value):
        cpu_tensor = value.detach().contiguous().cpu().reshape(-1)
        tensor_bytes = cpu_tensor.view(torch.uint8).numpy().tobytes()
        digest.update(b"tensor\0")
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(b"\0")
        digest.update(repr(tuple(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor_bytes)
        return value.numel()

    if isinstance(value, dict):
        digest.update(b"dict\0")
        numel = 0
        for key in sorted(value, key=lambda item: repr(item)):
            digest.update(repr(key).encode("utf-8"))
            digest.update(b"\0")
            numel += _update_tree_hash(digest, value[key])
        return numel

    if isinstance(value, tuple):
        digest.update(b"tuple\0")
        return sum(_update_tree_hash(digest, item) for item in value)

    if isinstance(value, list):
        digest.update(b"list\0")
        return sum(_update_tree_hash(digest, item) for item in value)

    digest.update(type(value).__name__.encode("utf-8"))
    digest.update(b"\0")
    digest.update(repr(value).encode("utf-8"))
    digest.update(b"\0")
    return 0


def _fingerprint_tree(value: Any) -> tuple[str, int]:
    digest = hashlib.sha256()
    numel = _update_tree_hash(digest, value)
    return digest.hexdigest(), numel


def fingerprint_batch(input_dict: Any, labels: Any) -> BatchFingerprint:
    """Hash batch contents without exposing token values."""
    input_sha256, input_numel = _fingerprint_tree(input_dict)
    labels_sha256, labels_numel = _fingerprint_tree(labels)
    combined = hashlib.sha256()
    combined.update(bytes.fromhex(input_sha256))
    combined.update(bytes.fromhex(labels_sha256))
    return BatchFingerprint(
        input_sha256=input_sha256,
        labels_sha256=labels_sha256,
        combined_sha256=combined.hexdigest(),
        input_numel=input_numel,
        labels_numel=labels_numel,
    )


def emit_batch_fingerprint(
    fingerprint: BatchFingerprint, *, step: int, microbatch: int
) -> None:
    """Emit one stable, grep-friendly fingerprint line on every rank."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = int(os.environ.get("RANK", "0"))

    print(
        "[LBT BATCH HASH] "
        f"rank={rank} step={step} microbatch={microbatch} "
        f"input_sha256={fingerprint.input_sha256} "
        f"labels_sha256={fingerprint.labels_sha256} "
        f"combined_sha256={fingerprint.combined_sha256} "
        f"input_numel={fingerprint.input_numel} "
        f"labels_numel={fingerprint.labels_numel}",
        file=sys.stderr,
        flush=True,
    )
