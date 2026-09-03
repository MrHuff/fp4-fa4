# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
"""Opt-in TorchTitan training-loop support for the FA4 experiments.

The stock :mod:`torchtitan.train` implementation remains unchanged.  Launch
``python -m torchtitan.experiments.fa4.train`` to select this trainer, then use
the ``fa4`` configuration fields to enable individual facilities.

The CUDA input path owns one device-copy lookahead.  Its checkpoint adapter
therefore saves the dataloader state immediately *after* that lookahead plus
the pending raw CPU batch.  A resumed job replays the pending batch exactly
once before advancing the underlying loader.  Capturing this envelope only at
checkpoint time keeps dataloader state traversal and serialization out of the
per-microbatch hot path.  Expensive gradient inspection is deliberately
separate from the inexpensive finite loss/gradient-norm guard.
"""

from __future__ import annotations

import json
import os
import pickle
import time
from typing import Any, Iterable, Iterator

import torch

from torchtitan.components.checkpoint import DATALOADER
from torchtitan.components.dataloader import (
    DataloaderExhaustedError,
    ParallelAwareDataloader,
)
from torchtitan.distributed import utils as dist_utils
from torchtitan.tools.logging import logger
from torchtitan.train import Trainer as TorchTitanTrainer


_CHECKPOINT_ENVELOPE_FORMAT = "torchtitan.experiments.fa4.CheckpointAlignedDataloader"
_CHECKPOINT_ENVELOPE_VERSION = 1
_CHECKPOINT_ENVELOPE_KEY = "__fa4_checkpoint_aligned_dataloader__"
_NO_PENDING_BATCH = object()


def _encode_checkpoint_envelope(
    dataloader: Any, loader_state: dict[str, Any], pending_batch: Any
) -> dict[str, Any]:
    """Return a versioned, opaque checkpoint without changing rank-key shape.

    ``ParallelAwareDataloader`` has used ``{dp_rank_N, world_size}`` as its
    persistent DCP schema for years.  Reusing its rank key for the opaque bytes
    lets a new adapter load both new envelopes and legacy raw loader states.
    Generic stateful loaders use a reserved top-level key.
    """

    payload = {
        "format": _CHECKPOINT_ENVELOPE_FORMAT,
        "version": _CHECKPOINT_ENVELOPE_VERSION,
        "loader_state": loader_state,
        "has_pending_batch": pending_batch is not _NO_PENDING_BATCH,
        "pending_batch": (
            None if pending_batch is _NO_PENDING_BATCH else pending_batch
        ),
    }
    opaque_payload = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)

    rank_id = getattr(dataloader, "_rank_id", None)
    if (
        isinstance(dataloader, ParallelAwareDataloader)
        and isinstance(rank_id, str)
        and rank_id in loader_state
        and "world_size" in loader_state
    ):
        return {
            rank_id: opaque_payload,
            "world_size": loader_state["world_size"],
        }
    return {_CHECKPOINT_ENVELOPE_KEY: opaque_payload}


def _decode_checkpoint_envelope(
    dataloader: Any, state_dict: dict[str, Any]
) -> tuple[dict[str, Any], Any] | None:
    """Decode a new envelope, returning ``None`` for a legacy raw state."""

    reserved_key = _CHECKPOINT_ENVELOPE_KEY in state_dict
    opaque_payload = state_dict.get(_CHECKPOINT_ENVELOPE_KEY)
    if opaque_payload is None:
        rank_id = getattr(dataloader, "_rank_id", None)
        if isinstance(rank_id, str):
            opaque_payload = state_dict.get(rank_id)

    if not isinstance(opaque_payload, (bytes, bytearray)):
        if reserved_key:
            raise RuntimeError("invalid FA4 dataloader checkpoint envelope")
        return None
    try:
        payload = pickle.loads(opaque_payload)
    except (EOFError, pickle.PickleError):
        if reserved_key:
            raise RuntimeError("invalid FA4 dataloader checkpoint envelope") from None
        return None
    if not isinstance(payload, dict) or payload.get("format") != (
        _CHECKPOINT_ENVELOPE_FORMAT
    ):
        return None
    if payload.get("version") != _CHECKPOINT_ENVELOPE_VERSION:
        raise RuntimeError(
            "unsupported FA4 dataloader checkpoint envelope version: "
            f"{payload.get('version')!r}"
        )
    loader_state = payload.get("loader_state")
    if not isinstance(loader_state, dict):
        raise RuntimeError("FA4 dataloader checkpoint has no loader state")
    if payload.get("has_pending_batch"):
        if "pending_batch" not in payload:
            raise RuntimeError("FA4 dataloader checkpoint has no pending batch")
        pending_batch = payload["pending_batch"]
    else:
        pending_batch = _NO_PENDING_BATCH
    return loader_state, pending_batch


class CheckpointAlignedDataloader:
    """Checkpoint a loader's post-lookahead state and its pending CPU batch."""

    def __init__(self, dataloader: Any):
        self.dataloader = dataloader
        self._pending_batch: Any = _NO_PENDING_BATCH
        self._replay_pending_batch = False
        self._has_prefetched_batch = False

    def next_for_prefetch(self, iterator: Iterator[Any]) -> Any:
        if self._has_prefetched_batch:
            raise RuntimeError(
                "cannot prefetch another batch before promoting the pending batch"
            )
        if self._replay_pending_batch:
            self._replay_pending_batch = False
            self._has_prefetched_batch = True
            return self._pending_batch
        try:
            batch = next(iterator)
        except BaseException:
            self._pending_batch = _NO_PENDING_BATCH
            self._replay_pending_batch = False
            self._has_prefetched_batch = False
            raise
        self._pending_batch = batch
        self._has_prefetched_batch = True
        return batch

    def mark_prefetched_batch_current(self) -> None:
        if not self._has_prefetched_batch:
            raise RuntimeError("no prefetched batch is available to promote")
        self._pending_batch = _NO_PENDING_BATCH
        self._replay_pending_batch = False
        self._has_prefetched_batch = False

    def state_dict(self) -> dict[str, Any]:
        loader_state = self.dataloader.state_dict()
        return _encode_checkpoint_envelope(
            self.dataloader, loader_state, self._pending_batch
        )

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._pending_batch = _NO_PENDING_BATCH
        self._replay_pending_batch = False
        self._has_prefetched_batch = False
        decoded = _decode_checkpoint_envelope(self.dataloader, state_dict)
        if decoded is None:
            # Legacy checkpoints captured the raw loader state immediately
            # before lookahead.  Loading that state directly preserves their
            # original replay semantics.
            self.dataloader.load_state_dict(state_dict)
            return
        loader_state, pending_batch = decoded
        self.dataloader.load_state_dict(loader_state)
        if pending_batch is not _NO_PENDING_BATCH:
            self._pending_batch = pending_batch
            self._replay_pending_batch = True


def install_checkpoint_aligned_dataloader(
    checkpointer: Any, dataloader: Any
) -> CheckpointAlignedDataloader:
    """Replace matching persistent and TorchFT dataloader checkpoint state."""

    adapter = CheckpointAlignedDataloader(dataloader)
    states = getattr(checkpointer, "states", None)
    if not isinstance(states, dict) or states.get(DATALOADER) is not dataloader:
        raise RuntimeError("checkpointer does not reference the trainer dataloader")
    states[DATALOADER] = adapter

    ft_states = getattr(checkpointer, "ft_states", None)
    if isinstance(ft_states, dict) and ft_states.get(DATALOADER) is dataloader:
        ft_states[DATALOADER] = adapter
    return adapter


def _pin_memory_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.device.type == "cpu" and not value.is_pinned():
            return value.pin_memory()
        return value
    if isinstance(value, dict):
        return {key: _pin_memory_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_pin_memory_tree(item) for item in value)
    if isinstance(value, list):
        return [_pin_memory_tree(item) for item in value]
    return value


def _to_device_tree(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _to_device_tree(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_device_tree(item, device) for item in value)
    if isinstance(value, list):
        return [_to_device_tree(item, device) for item in value]
    return value


def _record_stream_tree(value: Any, stream: torch.cuda.Stream) -> None:
    if torch.is_tensor(value):
        if value.device.type == "cuda":
            value.record_stream(stream)
        return
    if isinstance(value, dict):
        for item in value.values():
            _record_stream_tree(item, stream)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _record_stream_tree(item, stream)


def _local_tensor(value: torch.Tensor) -> torch.Tensor:
    """Return the local shard for diagnostics without gathering parameters."""

    value = value.detach()
    to_local = getattr(value, "to_local", None)
    if callable(to_local):
        value = to_local()
    return value


def require_finite_metric(value: torch.Tensor | float, label: str) -> None:
    """Fail on a non-finite scalar without a CUDA-to-host synchronization."""

    scalar = value.detach() if torch.is_tensor(value) else torch.as_tensor(value)
    finite = torch.isfinite(scalar).all()
    message = f"non-finite {label} detected by the FA4 training guard"
    if finite.device.type == "cuda":
        torch._assert_async(finite, message)
    elif not bool(finite.item()):
        raise FloatingPointError(message)


def find_nonfinite_gradients(
    model_parts: Iterable[torch.nn.Module], *, limit: int = 8
) -> list[dict[str, Any]]:
    """Return bounded, local-shard diagnostics for non-finite gradients."""

    if limit < 1:
        raise ValueError("non-finite gradient diagnostic limit must be positive")
    findings: list[dict[str, Any]] = []
    for model_index, module in enumerate(model_parts):
        for name, parameter in module.named_parameters():
            if parameter.grad is None or not torch.is_tensor(parameter.grad):
                continue
            gradient = _local_tensor(parameter.grad)
            if bool(torch.isfinite(gradient).all().item()):
                continue
            findings.append(
                {
                    "model_part": model_index,
                    "parameter": name,
                    "shape": list(gradient.shape),
                    "nan": int(torch.isnan(gradient).sum().item()),
                    "positive_infinity": int(torch.isposinf(gradient).sum().item()),
                    "negative_infinity": int(torch.isneginf(gradient).sum().item()),
                }
            )
            if len(findings) == limit:
                return findings
    return findings


def require_finite_gradients(model_parts: Iterable[torch.nn.Module]) -> None:
    findings = find_nonfinite_gradients(model_parts)
    if findings:
        raise RuntimeError(
            "non-finite parameter gradients detected before clipping: "
            + json.dumps(findings, sort_keys=True, separators=(",", ":"))
        )


def gradient_tensor_summaries(
    model_parts: Iterable[torch.nn.Module], *, topk: int
) -> list[dict[str, Any]]:
    """Summarize the largest local gradient tensors by root-mean-square value."""

    if topk < 1:
        raise ValueError("gradient diagnostic topk must be positive")
    rows: list[dict[str, Any]] = []
    for model_index, module in enumerate(model_parts):
        for name, parameter in module.named_parameters():
            if parameter.grad is None or not torch.is_tensor(parameter.grad):
                continue
            gradient = _local_tensor(parameter.grad)
            if gradient.numel() == 0:
                continue
            gradient = gradient.float()
            local_parameter = _local_tensor(parameter).float()
            grad_rms = float(torch.sqrt(torch.mean(gradient.square())).item())
            rows.append(
                {
                    "model_part": model_index,
                    "parameter": name,
                    "local_shape": list(gradient.shape),
                    "grad_rms": grad_rms,
                    "grad_max_abs": float(gradient.abs().max().item()),
                    "grad_norm": float(torch.linalg.vector_norm(gradient).item()),
                    "parameter_rms": float(
                        torch.sqrt(torch.mean(local_parameter.square())).item()
                    ),
                    "parameter_max_abs": float(local_parameter.abs().max().item()),
                }
            )
    rows.sort(key=lambda row: row["grad_rms"], reverse=True)
    return rows[:topk]


class FA4Trainer(TorchTitanTrainer):
    """TorchTitan trainer with explicitly selected FA4 runtime facilities."""

    def __init__(self, job_config):
        super().__init__(job_config)
        config = job_config.fa4
        self._fa4_cuda_data_prefetch = bool(config.cuda_data_prefetch)
        self._fa4_fail_on_nonfinite_metrics = bool(config.fail_on_nonfinite_metrics)
        self._fa4_scan_nonfinite_gradients = bool(config.scan_nonfinite_gradients)
        self._fa4_gradient_diagnostics_topk = int(config.gradient_diagnostics_topk)
        if self._fa4_gradient_diagnostics_topk < 0:
            raise ValueError("fa4.gradient_diagnostics_topk must be non-negative")
        if (
            self._fa4_fail_on_nonfinite_metrics
            or self._fa4_scan_nonfinite_gradients
            or self._fa4_gradient_diagnostics_topk
        ) and os.environ.get("TORCHTITAN_SKIP_GRAD_CLIP", "0") == "1":
            raise RuntimeError(
                "FA4 gradient diagnostics require TorchTitan gradient clipping"
            )

        self._fa4_checkpoint_dataloader: CheckpointAlignedDataloader | None = None
        if self._fa4_cuda_data_prefetch:
            self._fa4_checkpoint_dataloader = install_checkpoint_aligned_dataloader(
                self.checkpointer, self.dataloader
            )
            logger.info("FA4_CUDA_DATA_PREFETCH_READY checkpoint_aligned=true depth=1")

    def batch_generator(
        self,
        data_iterable: Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]],
    ) -> Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        if not self._fa4_cuda_data_prefetch:
            yield from super().batch_generator(data_iterable)
            return

        adapter = self._fa4_checkpoint_dataloader
        if adapter is None:
            raise RuntimeError("FA4 CUDA prefetch has no checkpoint adapter")
        if data_iterable is not adapter.dataloader:
            raise RuntimeError("FA4 CUDA prefetch requires the trainer dataloader")

        device = self.device
        if device.type != "cuda":
            raise RuntimeError("fa4.cuda_data_prefetch requires a CUDA trainer device")
        data_iterator = iter(data_iterable)

        def load_cpu_batch():
            data_load_start = time.perf_counter()
            try:
                input_dict, labels = adapter.next_for_prefetch(data_iterator)
            except StopIteration as error:
                raise DataloaderExhaustedError() from error
            data_loading_time = time.perf_counter() - data_load_start
            return (
                _pin_memory_tree(input_dict),
                _pin_memory_tree(labels),
                data_loading_time,
            )

        def record_actual_yield(labels: torch.Tensor, data_loading_time: float) -> None:
            ntokens_batch = labels.numel()
            self.ntokens_seen += ntokens_batch
            self.metrics_processor.ntokens_since_last_log += ntokens_batch
            self.metrics_processor.data_loading_times.append(data_loading_time)

        copy_stream = torch.cuda.Stream(device=device)

        def prefetch_to_device(cpu_inputs, cpu_labels, data_loading_time):
            with torch.cuda.stream(copy_stream):
                device_inputs = _to_device_tree(cpu_inputs, device)
                device_labels = _to_device_tree(cpu_labels, device)
                ready_event = torch.cuda.Event()
                ready_event.record(copy_stream)
            return device_inputs, device_labels, ready_event, data_loading_time

        next_inputs, next_labels, next_ready_event, next_loading_time = (
            prefetch_to_device(*load_cpu_batch())
        )
        has_next_batch = True
        while has_next_batch:
            adapter.mark_prefetched_batch_current()
            current_stream = torch.cuda.current_stream(device=device)
            current_stream.wait_event(next_ready_event)
            inputs = next_inputs
            labels = next_labels
            loading_time = next_loading_time
            _record_stream_tree(inputs, current_stream)
            _record_stream_tree(labels, current_stream)

            try:
                (
                    next_inputs,
                    next_labels,
                    next_ready_event,
                    next_loading_time,
                ) = prefetch_to_device(*load_cpu_batch())
            except DataloaderExhaustedError:
                has_next_batch = False

            record_actual_yield(labels, loading_time)
            yield inputs, labels

    def forward_backward_step(self, input_dict, labels):
        loss = super().forward_backward_step(input_dict, labels)
        if self._fa4_fail_on_nonfinite_metrics:
            require_finite_metric(loss, "loss")
        return loss

    def train_step(self, data_iterator):
        inspect_gradients = (
            self._fa4_fail_on_nonfinite_metrics
            or self._fa4_scan_nonfinite_gradients
            or self._fa4_gradient_diagnostics_topk > 0
        )
        if not inspect_gradients:
            return super().train_step(data_iterator)

        original_clip_grad_norm = dist_utils.clip_grad_norm_

        def guarded_clip_grad_norm(parameters, max_norm, *args, **kwargs):
            if self._fa4_scan_nonfinite_gradients:
                require_finite_gradients(self.model_parts)
            if self._fa4_gradient_diagnostics_topk:
                summaries = gradient_tensor_summaries(
                    self.model_parts,
                    topk=self._fa4_gradient_diagnostics_topk,
                )
                logger.warning(
                    "FA4_GRADIENT_DIAGNOSTICS %s",
                    json.dumps(summaries, sort_keys=True, separators=(",", ":")),
                )
            grad_norm = original_clip_grad_norm(parameters, max_norm, *args, **kwargs)
            if self._fa4_fail_on_nonfinite_metrics:
                require_finite_metric(grad_norm, "gradient norm")
            return grad_norm

        dist_utils.clip_grad_norm_ = guarded_clip_grad_norm
        try:
            return super().train_step(data_iterator)
        finally:
            dist_utils.clip_grad_norm_ = original_clip_grad_norm


__all__ = [
    "CheckpointAlignedDataloader",
    "FA4Trainer",
    "find_nonfinite_gradients",
    "gradient_tensor_summaries",
    "install_checkpoint_aligned_dataloader",
    "require_finite_gradients",
    "require_finite_metric",
]
