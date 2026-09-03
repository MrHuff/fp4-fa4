# Copyright (c) 2026 Graphcore Ltd. All rights reserved.

"""Credential-free SlimPajama registration and dataloaders for FA4 runs."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import JobConfig
from torchtitan.hf_datasets import DatasetConfig
from torchtitan.hf_datasets.text_datasets import DATASETS, HuggingFaceTextDataset


_COMMIT = re.compile(r"[0-9a-f]{40}")
_UNSIGNED_DECIMAL = re.compile(r"(?:0|[1-9][0-9]*)")

_TRAIN_WORKERS_ENV = "FA4_TRAIN_DATALOADER_NUM_WORKERS"
_VALIDATION_WORKERS_ENV = "FA4_VALIDATION_DATALOADER_NUM_WORKERS"
_PREFETCH_FACTOR_ENV = "FA4_DATALOADER_PREFETCH_FACTOR"
_PIN_MEMORY_ENV = "FA4_DATALOADER_PIN_MEMORY"


@dataclass(frozen=True)
class FA4DataloaderSettings:
    """Authenticated worker settings shared by FA4 training and validation."""

    train_num_workers: int
    validation_num_workers: int
    prefetch_factor: int
    pin_memory: bool


def _required_env(name: str) -> str:
    try:
        value = os.environ[name]
    except KeyError:
        raise RuntimeError(
            f"required FA4 dataloader environment variable {name} is unset"
        ) from None
    if not value or value != value.strip():
        raise RuntimeError(
            f"{name} must be present without leading or trailing whitespace"
        )
    return value


def _required_nonnegative_integer(name: str) -> int:
    value = _required_env(name)
    if _UNSIGNED_DECIMAL.fullmatch(value) is None:
        raise RuntimeError(f"{name} must be a canonical non-negative decimal integer")
    return int(value)


def _required_positive_integer(name: str) -> int:
    value = _required_nonnegative_integer(name)
    if value == 0:
        raise RuntimeError(f"{name} must be greater than zero")
    return value


def _required_boolean(name: str) -> bool:
    value = _required_env(name)
    if value not in {"0", "1"}:
        raise RuntimeError(f"{name} must be exactly '0' or '1'")
    return value == "1"


def load_fa4_dataloader_settings() -> FA4DataloaderSettings:
    """Read the complete fail-closed FA4 dataloader environment contract."""

    return FA4DataloaderSettings(
        train_num_workers=_required_nonnegative_integer(_TRAIN_WORKERS_ENV),
        validation_num_workers=_required_nonnegative_integer(_VALIDATION_WORKERS_ENV),
        prefetch_factor=_required_positive_integer(_PREFETCH_FACTOR_ENV),
        pin_memory=_required_boolean(_PIN_MEMORY_ENV),
    )


def _text(sample: dict[str, object]) -> str:
    value = sample.get("text")
    if not isinstance(value, str):
        raise TypeError("SlimPajama sample has no string 'text' field")
    return value


def _load(path: str, *, split: str):
    local = Path(path).expanduser()
    revision = os.environ.get("FA4_SLIMPAJAMA_REVISION", "").strip()
    # A rendered config can bind a public dataset revision without relying on
    # ambient state by encoding ``repository@commit`` in dataset_path.
    if not local.exists() and "@" in path:
        path, embedded_revision = path.rsplit("@", 1)
        if revision and revision != embedded_revision:
            raise RuntimeError(
                "SlimPajama revision in dataset_path conflicts with "
                "FA4_SLIMPAJAMA_REVISION"
            )
        revision = embedded_revision
    if not local.exists() and _COMMIT.fullmatch(revision) is None:
        raise RuntimeError(
            "remote SlimPajama loading requires an immutable lowercase 40-hex "
            "FA4_SLIMPAJAMA_REVISION; alternatively set "
            "training.dataset_path to a locally checksummed dataset snapshot"
        )
    kwargs = {
        "name": "default",
        "split": split,
        "streaming": True,
    }
    if revision:
        kwargs["revision"] = revision
    return load_dataset(path, **kwargs)


def register_slimpajama() -> None:
    DATASETS.setdefault(
        "slimpajama",
        DatasetConfig(
            path="cerebras/SlimPajama-627B",
            loader=lambda path: _load(path, split="train"),
            sample_processor=_text,
        ),
    )
    DATASETS.setdefault(
        "slimpajama_val",
        DatasetConfig(
            path="cerebras/SlimPajama-627B",
            loader=lambda path: _load(path, split="validation"),
            sample_processor=_text,
        ),
    )


def _build_fa4_text_dataloader(
    *,
    dataset_name: str,
    dataset_path: str | None,
    batch_size: int,
    seq_len: int,
    num_workers: int,
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    infinite: bool,
    settings: FA4DataloaderSettings,
) -> ParallelAwareDataloader:
    dataset = HuggingFaceTextDataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        seq_len=seq_len,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
    )

    # StatefulDataLoader forbids a prefetch factor and persistent workers when
    # loading in the main process. Keep that degenerate diagnostic mode valid
    # without changing the authenticated environment contract.
    workers_enabled = num_workers > 0
    return ParallelAwareDataloader(
        dataset=dataset,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=settings.pin_memory,
        prefetch_factor=settings.prefetch_factor if workers_enabled else None,
        persistent_workers=workers_enabled,
    )


def build_fa4_text_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    job_config: JobConfig,
    infinite: bool = True,
) -> ParallelAwareDataloader:
    """Build the training loader under the explicit FA4 worker contract."""

    settings = load_fa4_dataloader_settings()
    return _build_fa4_text_dataloader(
        dataset_name=job_config.training.dataset,
        dataset_path=job_config.training.dataset_path,
        batch_size=job_config.training.local_batch_size,
        seq_len=job_config.training.seq_len,
        num_workers=settings.train_num_workers,
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        tokenizer=tokenizer,
        infinite=infinite,
        settings=settings,
    )


def build_fa4_text_validation_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    job_config: JobConfig,
    infinite: bool = False,
) -> ParallelAwareDataloader:
    """Build the validation loader under the explicit FA4 worker contract."""

    settings = load_fa4_dataloader_settings()
    return _build_fa4_text_dataloader(
        dataset_name=job_config.validation.dataset,
        dataset_path=job_config.validation.dataset_path,
        batch_size=job_config.validation.local_batch_size,
        seq_len=job_config.validation.seq_len,
        num_workers=settings.validation_num_workers,
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        tokenizer=tokenizer,
        infinite=infinite,
        settings=settings,
    )


__all__ = [
    "FA4DataloaderSettings",
    "build_fa4_text_dataloader",
    "build_fa4_text_validation_dataloader",
    "load_fa4_dataloader_settings",
    "register_slimpajama",
]
