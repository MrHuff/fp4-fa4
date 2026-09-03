# Copyright (c) 2026 Graphcore Ltd. All rights reserved.

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.utils.data import Dataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.experiments.fa4 import trainer as trainer_module
from torchtitan.experiments.fa4.trainer import (
    CheckpointAlignedDataloader,
    FA4Trainer,
    find_nonfinite_gradients,
    gradient_tensor_summaries,
    install_checkpoint_aligned_dataloader,
    require_finite_gradients,
    require_finite_metric,
)
from torchtitan.train import Trainer as TorchTitanTrainer


class _CursorDataloader:
    def __init__(self, *, limit: int | None = None):
        self.cursor = 0
        self.limit = limit
        self.state_dict_calls = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.limit is not None and self.cursor >= self.limit:
            raise StopIteration
        value = self.cursor
        self.cursor += 1
        labels = torch.tensor([value + 1, value + 2])
        return {"input": labels - 1}, labels

    def state_dict(self):
        self.state_dict_calls += 1
        return {"cursor": self.cursor}

    def load_state_dict(self, state_dict):
        self.cursor = state_dict["cursor"]


class _NumberedDataset(Dataset):
    def __len__(self):
        return 32

    def __getitem__(self, index):
        value = torch.tensor(index)
        return {"input": value}, value + 1


class _FakeStream:
    def wait_event(self, event):
        del event


class _FakeEvent:
    def record(self, stream):
        del stream


def _bare_trainer(*, prefetch: bool) -> FA4Trainer:
    trainer = FA4Trainer.__new__(FA4Trainer)
    trainer._fa4_cuda_data_prefetch = prefetch
    trainer._fa4_fail_on_nonfinite_metrics = False
    trainer._fa4_scan_nonfinite_gradients = False
    trainer._fa4_gradient_diagnostics_topk = 0
    return trainer


def _batch_index(batch) -> int:
    return int(batch[0]["input"].reshape(-1)[0].item())


def test_checkpoint_envelope_replays_pending_then_uses_post_prefetch_state() -> None:
    loader = _CursorDataloader()
    adapter = CheckpointAlignedDataloader(loader)
    iterator = iter(loader)

    first = adapter.next_for_prefetch(iterator)
    adapter.mark_prefetched_batch_current()
    pending = adapter.next_for_prefetch(iterator)

    assert _batch_index(first) == 0
    assert _batch_index(pending) == 1
    assert loader.cursor == 2
    assert loader.state_dict_calls == 0
    state = adapter.state_dict()
    assert loader.state_dict_calls == 1

    resumed = _CursorDataloader()
    resumed_adapter = CheckpointAlignedDataloader(resumed)
    resumed_adapter.load_state_dict(state)
    resumed_iterator = iter(resumed)

    # The replay is supplied by the adapter, so the underlying post-prefetch
    # cursor does not advance until the following batch.
    replayed = resumed_adapter.next_for_prefetch(resumed_iterator)
    assert _batch_index(replayed) == _batch_index(pending)
    assert resumed.cursor == 2
    resumed_adapter.mark_prefetched_batch_current()
    following = resumed_adapter.next_for_prefetch(resumed_iterator)
    assert _batch_index(following) == 2
    assert resumed.cursor == 3


def test_prefetch_hot_path_never_calls_dataloader_state_dict() -> None:
    loader = _CursorDataloader()
    adapter = CheckpointAlignedDataloader(loader)
    iterator = iter(loader)

    for expected in range(8):
        assert _batch_index(adapter.next_for_prefetch(iterator)) == expected
        adapter.mark_prefetched_batch_current()

    assert loader.state_dict_calls == 0
    adapter.state_dict()
    assert loader.state_dict_calls == 1


@pytest.mark.parametrize("checkpoint_after_replay", (False, True))
def test_checkpointing_again_around_a_restored_pending_batch_is_exact(
    checkpoint_after_replay: bool,
) -> None:
    source = _CursorDataloader()
    source_adapter = CheckpointAlignedDataloader(source)
    source_iterator = iter(source)
    source_adapter.next_for_prefetch(source_iterator)
    source_adapter.mark_prefetched_batch_current()
    pending = source_adapter.next_for_prefetch(source_iterator)
    first_state = source_adapter.state_dict()

    intermediate = _CursorDataloader()
    intermediate_adapter = CheckpointAlignedDataloader(intermediate)
    intermediate_adapter.load_state_dict(first_state)
    intermediate_iterator = iter(intermediate)
    if checkpoint_after_replay:
        replayed = intermediate_adapter.next_for_prefetch(intermediate_iterator)
        assert _batch_index(replayed) == _batch_index(pending)
    second_state = intermediate_adapter.state_dict()

    resumed = _CursorDataloader()
    resumed_adapter = CheckpointAlignedDataloader(resumed)
    resumed_adapter.load_state_dict(second_state)
    resumed_iterator = iter(resumed)
    assert _batch_index(resumed_adapter.next_for_prefetch(resumed_iterator)) == 1
    assert resumed.cursor == 2
    resumed_adapter.mark_prefetched_batch_current()
    assert _batch_index(resumed_adapter.next_for_prefetch(resumed_iterator)) == 2


def test_checkpoint_without_pending_resumes_at_underlying_cursor() -> None:
    source = _CursorDataloader()
    adapter = CheckpointAlignedDataloader(source)
    batch = adapter.next_for_prefetch(iter(source))
    assert _batch_index(batch) == 0
    adapter.mark_prefetched_batch_current()
    state = adapter.state_dict()

    resumed = _CursorDataloader()
    resumed_adapter = CheckpointAlignedDataloader(resumed)
    resumed_adapter.load_state_dict(state)
    assert _batch_index(resumed_adapter.next_for_prefetch(iter(resumed))) == 1


def test_checkpoint_after_eof_stays_exhausted() -> None:
    source = _CursorDataloader(limit=2)
    adapter = CheckpointAlignedDataloader(source)
    iterator = iter(source)
    for expected in range(2):
        assert _batch_index(adapter.next_for_prefetch(iterator)) == expected
        adapter.mark_prefetched_batch_current()
    with pytest.raises(StopIteration):
        adapter.next_for_prefetch(iterator)
    state = adapter.state_dict()

    resumed = _CursorDataloader(limit=2)
    resumed_adapter = CheckpointAlignedDataloader(resumed)
    resumed_adapter.load_state_dict(state)
    with pytest.raises(StopIteration):
        resumed_adapter.next_for_prefetch(iter(resumed))


def test_legacy_raw_state_remains_loadable() -> None:
    resumed = _CursorDataloader()
    adapter = CheckpointAlignedDataloader(resumed)

    adapter.load_state_dict({"cursor": 7})

    assert _batch_index(adapter.next_for_prefetch(iter(resumed))) == 7


def test_checkpoint_adapter_replaces_persistent_and_ft_state() -> None:
    loader = _CursorDataloader()
    checkpointer = SimpleNamespace(
        states={"dataloader": loader, "train_state": object()},
        ft_states={"dataloader": loader},
    )

    adapter = install_checkpoint_aligned_dataloader(checkpointer, loader)

    assert checkpointer.states["dataloader"] is adapter
    assert checkpointer.ft_states["dataloader"] is adapter


@pytest.mark.parametrize("num_workers", (0, 1))
def test_parallel_loader_replays_pending_and_subsequent_batches_exactly(
    num_workers: int,
) -> None:
    worker_options = (
        {"prefetch_factor": 2, "persistent_workers": True} if num_workers else {}
    )
    source = ParallelAwareDataloader(
        _NumberedDataset(),
        dp_rank=0,
        dp_world_size=1,
        batch_size=1,
        num_workers=num_workers,
        **worker_options,
    )
    resumed = ParallelAwareDataloader(
        _NumberedDataset(),
        dp_rank=0,
        dp_world_size=1,
        batch_size=1,
        num_workers=num_workers,
        **worker_options,
    )
    try:
        adapter = CheckpointAlignedDataloader(source)
        iterator = iter(source)
        adapter.next_for_prefetch(iterator)
        adapter.mark_prefetched_batch_current()
        pending = adapter.next_for_prefetch(iterator)
        state = adapter.state_dict()

        expected = [_batch_index(pending)]
        adapter.mark_prefetched_batch_current()
        for _ in range(4):
            expected.append(_batch_index(adapter.next_for_prefetch(iterator)))
            adapter.mark_prefetched_batch_current()

        resumed_adapter = CheckpointAlignedDataloader(resumed)
        resumed_adapter.load_state_dict(state)
        resumed_iterator = iter(resumed)
        actual = []
        for _ in range(5):
            actual.append(
                _batch_index(resumed_adapter.next_for_prefetch(resumed_iterator))
            )
            resumed_adapter.mark_prefetched_batch_current()

        assert actual == expected
        assert set(state) == {"dp_rank_0", "world_size"}
    finally:
        for loader in (source, resumed):
            iterator = getattr(loader, "_iterator", None)
            shutdown_workers = getattr(iterator, "_shutdown_workers", None)
            if callable(shutdown_workers):
                shutdown_workers()


def test_parallel_loader_envelope_round_trips_through_dcp(tmp_path) -> None:
    source = ParallelAwareDataloader(
        _NumberedDataset(),
        dp_rank=0,
        dp_world_size=1,
        batch_size=1,
        num_workers=0,
    )
    source_adapter = CheckpointAlignedDataloader(source)
    source_iterator = iter(source)
    source_adapter.next_for_prefetch(source_iterator)
    source_adapter.mark_prefetched_batch_current()
    pending = source_adapter.next_for_prefetch(source_iterator)
    dcp.save({"dataloader": source_adapter}, checkpoint_id=tmp_path / "new")

    resumed = ParallelAwareDataloader(
        _NumberedDataset(),
        dp_rank=0,
        dp_world_size=1,
        batch_size=1,
        num_workers=0,
    )
    resumed_adapter = CheckpointAlignedDataloader(resumed)
    dcp.load({"dataloader": resumed_adapter}, checkpoint_id=tmp_path / "new")
    resumed_iterator = iter(resumed)

    assert _batch_index(resumed_adapter.next_for_prefetch(resumed_iterator)) == (
        _batch_index(pending)
    )
    resumed_adapter.mark_prefetched_batch_current()
    assert _batch_index(resumed_adapter.next_for_prefetch(resumed_iterator)) == 2


def test_parallel_loader_legacy_dcp_schema_remains_loadable(tmp_path) -> None:
    legacy = ParallelAwareDataloader(
        _NumberedDataset(),
        dp_rank=0,
        dp_world_size=1,
        batch_size=1,
        num_workers=0,
    )
    legacy_iterator = iter(legacy)
    assert _batch_index(next(legacy_iterator)) == 0
    dcp.save({"dataloader": legacy}, checkpoint_id=tmp_path / "legacy")

    resumed = ParallelAwareDataloader(
        _NumberedDataset(),
        dp_rank=0,
        dp_world_size=1,
        batch_size=1,
        num_workers=0,
    )
    resumed_adapter = CheckpointAlignedDataloader(resumed)
    dcp.load({"dataloader": resumed_adapter}, checkpoint_id=tmp_path / "legacy")

    assert _batch_index(resumed_adapter.next_for_prefetch(iter(resumed))) == 1


def test_cuda_lookahead_accounts_only_the_batch_actually_yielded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = _CursorDataloader()
    trainer = _bare_trainer(prefetch=True)
    trainer.device = torch.device("cuda")
    trainer.ntokens_seen = 0
    trainer.metrics_processor = SimpleNamespace(
        ntokens_since_last_log=0,
        data_loading_times=[],
    )
    trainer._fa4_checkpoint_dataloader = CheckpointAlignedDataloader(loader)

    stream = _FakeStream()
    monkeypatch.setattr(trainer_module, "_pin_memory_tree", lambda value: value)
    monkeypatch.setattr(trainer_module, "_to_device_tree", lambda value, device: value)
    monkeypatch.setattr(torch.cuda, "Stream", lambda **kwargs: stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda value: nullcontext())
    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda **kwargs: stream)

    batches = trainer.batch_generator(loader)
    inputs, labels = next(batches)

    assert inputs["input"].tolist() == [0, 1]
    assert labels.tolist() == [1, 2]
    assert loader.cursor == 2
    assert trainer.ntokens_seen == 2
    assert trainer.metrics_processor.ntokens_since_last_log == 2
    assert len(trainer.metrics_processor.data_loading_times) == 1
    assert loader.state_dict_calls == 0

    state = trainer._fa4_checkpoint_dataloader.state_dict()
    resumed = _CursorDataloader()
    resumed_adapter = CheckpointAlignedDataloader(resumed)
    resumed_adapter.load_state_dict(state)
    replayed = resumed_adapter.next_for_prefetch(iter(resumed))
    assert _batch_index(replayed) == 1


@pytest.mark.parametrize("value", (float("nan"), float("inf"), -float("inf")))
def test_finite_metric_guard_rejects_nonfinite_cpu_values(value: float) -> None:
    with pytest.raises(FloatingPointError, match="non-finite loss"):
        require_finite_metric(torch.tensor(value), "loss")


def test_nonfinite_gradient_scan_is_bounded_and_names_the_parameter() -> None:
    module = torch.nn.Linear(2, 2, bias=False)
    module.weight.grad = torch.tensor([[1.0, float("nan")], [float("inf"), 0.0]])

    findings = find_nonfinite_gradients([module])

    assert findings == [
        {
            "model_part": 0,
            "parameter": "weight",
            "shape": [2, 2],
            "nan": 1,
            "positive_infinity": 1,
            "negative_infinity": 0,
        }
    ]
    with pytest.raises(RuntimeError, match='"parameter":"weight"'):
        require_finite_gradients([module])


def test_gradient_summaries_rank_local_tensors_by_rms() -> None:
    module = torch.nn.Module()
    module.register_parameter("small", torch.nn.Parameter(torch.ones(2)))
    module.register_parameter("large", torch.nn.Parameter(torch.ones(2) * 3))
    module.small.grad = torch.tensor([1.0, 1.0])
    module.large.grad = torch.tensor([4.0, 4.0])

    summaries = gradient_tensor_summaries([module], topk=1)

    assert len(summaries) == 1
    assert summaries[0]["parameter"] == "large"
    assert summaries[0]["grad_rms"] == pytest.approx(4.0)
    assert summaries[0]["parameter_rms"] == pytest.approx(3.0)


def test_fa4_trainer_default_train_step_is_stock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _bare_trainer(prefetch=False)
    sentinel = object()

    def stock(self, iterator):
        del self, iterator
        return sentinel

    monkeypatch.setattr(TorchTitanTrainer, "train_step", stock)

    assert trainer.train_step(iter(())) is sentinel


def test_fa4_trainer_default_batch_generator_is_stock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _bare_trainer(prefetch=False)
    sentinel = ({"input": torch.tensor([3])}, torch.tensor([4]))

    def stock(self, iterable):
        del self, iterable
        yield sentinel

    monkeypatch.setattr(TorchTitanTrainer, "batch_generator", stock)

    assert list(trainer.batch_generator(object())) == [sentinel]


def test_fa4_trainer_scans_before_stock_gradient_clip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _bare_trainer(prefetch=False)
    trainer._fa4_scan_nonfinite_gradients = True
    module = torch.nn.Linear(1, 1, bias=False)
    module.weight.grad = torch.tensor([[float("nan")]])
    trainer.model_parts = [module]

    def stock_train_step(self, iterator):
        del iterator
        return trainer_module.dist_utils.clip_grad_norm_(
            list(self.model_parts[0].parameters()), 1.0
        )

    monkeypatch.setattr(TorchTitanTrainer, "train_step", stock_train_step)

    with pytest.raises(RuntimeError, match="before clipping"):
        trainer.train_step(iter(()))
