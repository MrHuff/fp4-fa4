# Copyright (c) 2026 Graphcore Ltd. All rights reserved.

"""CPU-only checks for the fail-closed FA4 dataloader contract."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from torchtitan.experiments.fa4 import data as fa4_data
from torchtitan.hf_datasets.text_datasets import build_text_dataloader
from torchtitan.models.llama3 import get_train_spec as get_llama3_train_spec
from torchtitan.protocols.train_spec import get_train_spec


_ENV = {
    "FA4_TRAIN_DATALOADER_NUM_WORKERS": "8",
    "FA4_VALIDATION_DATALOADER_NUM_WORKERS": "1",
    "FA4_DATALOADER_PREFETCH_FACTOR": "8",
    "FA4_DATALOADER_PIN_MEMORY": "0",
}


def _set_env(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> None:
    for name, value in (_ENV | overrides).items():
        monkeypatch.setenv(name, value)


def _job_config() -> SimpleNamespace:
    return SimpleNamespace(
        training=SimpleNamespace(
            dataset="slimpajama",
            dataset_path="training-path",
            local_batch_size=4,
            seq_len=4096,
        ),
        validation=SimpleNamespace(
            dataset="slimpajama_val",
            dataset_path="validation-path",
            local_batch_size=2,
            seq_len=4096,
        ),
    )


@pytest.mark.parametrize(
    ("builder_name", "workers", "dataset_name", "dataset_path", "batch_size"),
    (
        (
            "build_fa4_text_dataloader",
            8,
            "slimpajama",
            "training-path",
            4,
        ),
        (
            "build_fa4_text_validation_dataloader",
            1,
            "slimpajama_val",
            "validation-path",
            2,
        ),
    ),
)
def test_fa4_dataloader_passes_authenticated_worker_kwargs(
    monkeypatch: pytest.MonkeyPatch,
    builder_name: str,
    workers: int,
    dataset_name: str,
    dataset_path: str,
    batch_size: int,
) -> None:
    _set_env(monkeypatch)
    dataset = object()
    tokenizer = object()
    dataset_factory = Mock(return_value=dataset)
    loader_factory = Mock(return_value=object())
    monkeypatch.setattr(fa4_data, "HuggingFaceTextDataset", dataset_factory)
    monkeypatch.setattr(fa4_data, "ParallelAwareDataloader", loader_factory)

    builder = getattr(fa4_data, builder_name)
    builder(
        dp_world_size=16,
        dp_rank=3,
        tokenizer=tokenizer,
        job_config=_job_config(),
    )

    dataset_factory.assert_called_once_with(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        seq_len=4096,
        dp_rank=3,
        dp_world_size=16,
        infinite=builder_name == "build_fa4_text_dataloader",
    )
    loader_factory.assert_called_once_with(
        dataset=dataset,
        dp_rank=3,
        dp_world_size=16,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=False,
        prefetch_factor=8,
        persistent_workers=True,
    )


def test_zero_workers_disables_prefetch_and_persistence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_env(
        monkeypatch,
        FA4_TRAIN_DATALOADER_NUM_WORKERS="0",
        FA4_DATALOADER_PIN_MEMORY="1",
    )
    monkeypatch.setattr(fa4_data, "HuggingFaceTextDataset", Mock(return_value=object()))
    loader_factory = Mock(return_value=object())
    monkeypatch.setattr(fa4_data, "ParallelAwareDataloader", loader_factory)

    fa4_data.build_fa4_text_dataloader(
        dp_world_size=1,
        dp_rank=0,
        tokenizer=object(),
        job_config=_job_config(),
    )

    assert loader_factory.call_args.kwargs["num_workers"] == 0
    assert loader_factory.call_args.kwargs["prefetch_factor"] is None
    assert loader_factory.call_args.kwargs["persistent_workers"] is False
    assert loader_factory.call_args.kwargs["pin_memory"] is True


@pytest.mark.parametrize("missing", tuple(_ENV))
def test_fa4_dataloader_rejects_missing_required_environment(
    monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    _set_env(monkeypatch)
    monkeypatch.delenv(missing)

    with pytest.raises(RuntimeError, match=missing):
        fa4_data.load_fa4_dataloader_settings()


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("FA4_TRAIN_DATALOADER_NUM_WORKERS", "-1"),
        ("FA4_TRAIN_DATALOADER_NUM_WORKERS", "08"),
        ("FA4_VALIDATION_DATALOADER_NUM_WORKERS", "one"),
        ("FA4_DATALOADER_PREFETCH_FACTOR", "0"),
        ("FA4_DATALOADER_PREFETCH_FACTOR", " 8"),
        ("FA4_DATALOADER_PIN_MEMORY", "true"),
    ),
)
def test_fa4_dataloader_rejects_malformed_environment(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    _set_env(monkeypatch, **{name: value})

    with pytest.raises(RuntimeError, match=name):
        fa4_data.load_fa4_dataloader_settings()


def test_only_fa4_train_spec_uses_explicit_dataloader_builders() -> None:
    fa4 = get_train_spec("llama3_gc")
    stock = get_llama3_train_spec()

    assert fa4.build_dataloader_fn is fa4_data.build_fa4_text_dataloader
    assert fa4.build_validator_fn.__module__.endswith("fa4.validator")
    assert stock.build_dataloader_fn is build_text_dataloader
    assert stock.build_dataloader_fn is not fa4.build_dataloader_fn
