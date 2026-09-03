#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
from pathlib import Path
from typing import List, Dict, Any
import json
from torchtitan.components.dataloader import (
    BaseDataLoader,
    StatefulDataLoader,
)
from torchtitan.components.tokenizer import BaseTokenizer

from torchtitan.config import JobConfig
from torchtitan.tools.logging import logger

import torch
from torch.utils.data import IterableDataset
from torch.distributed.checkpoint.stateful import Stateful

from streaming import StreamingDataset, Stream
from streaming.base import distributed as dist
from streaming.base.dataset import EPOCH_DATA, EPOCH_SHAPE, _get_path

from uritools import urisplit
import os
import tempfile
from multiprocessing.shared_memory import SharedMemory


def make_dataset_streams(
    dataset: str,
    dataset_path: str,
    split: str | None,
    tmp_dir: str,
    stream_specs: list[dict[str, Any]] | None = None,
) -> List[Stream]:
    """Make MosaicML streams from dataset name and path.

    Note: at the moment supporting only a single stream. May support multiple
        in the future for dataset mixing.

    Args:
        dataset: Dataset name
        dataset_path: Path, local or s3.
        split: train, val, ...
        tmp_dir: Local temporary dir, for dataset cache.
    """
    assert dataset.startswith(("mosaic/", "lbt/")), (
        f"Not recognising a MosaicML streaming dataset: '{dataset}'."
    )
    assert dataset_path is not None, "Please provide a valid dataset path."

    if dataset.startswith("lbt/"):
        if dataset != "lbt/dolma3-longmino-pool":
            raise ValueError(f"Unsupported LBT Mosaic dataset: {dataset!r}")
        if not stream_specs:
            raise ValueError(
                "Dolma3 Longmino requires a non-empty load_dataset_kwargs.streams"
            )
        streams = []
        root = dataset_path.rstrip("/")
        cache_root = Path(tmp_dir)
        for index, spec in enumerate(stream_specs):
            if set(spec) != {"sub_path", "proportion"}:
                raise ValueError(
                    f"Dolma3 stream {index} must contain only sub_path/proportion"
                )
            sub_path = spec["sub_path"]
            proportion = spec["proportion"]
            if (
                not isinstance(sub_path, str)
                or not sub_path
            ):
                raise ValueError(f"Unsafe Dolma3 stream sub_path: {sub_path!r}")
            parsed = Path(sub_path)
            if parsed.is_absolute() or ".." in parsed.parts:
                raise ValueError(f"Unsafe Dolma3 stream sub_path: {sub_path!r}")
            if not isinstance(proportion, (int, float)) or proportion <= 0:
                raise ValueError(
                    f"Dolma3 stream proportion must be positive: {proportion!r}"
                )
            streams.append(
                Stream(
                    remote=f"{root}/{sub_path}",
                    local=str(cache_root / sub_path),
                    split=split,
                    proportion=proportion,
                )
            )
        return streams

    is_dataset_local = urisplit(dataset_path).scheme is None
    if is_dataset_local:
        return [Stream(remote=None, local=dataset_path, split=split)]
    # Remote dataset => use local dir. for caching.
    stream = Stream(remote=dataset_path, local=tmp_dir, split=split)
    return [stream]


class MosaicStreamingDataset(IterableDataset, Stateful):
    """MosaicML Streaming dataset, compatible with TorchTitan.

    Inspired by https://github.com/mosaicml/streaming/blob/main/streaming/text/c4.py
    """

    def __init__(
        self,
        job_config: JobConfig,
        tokenizer: BaseTokenizer,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
        data_config: Any | None = None,
    ) -> None:
        self._samples_to_skip_on_reload = 0
        data_config = data_config or job_config.training
        mosaic_config = data_config.mosaicml_dataset
        # Extra dataset args to forward to MosaicML streaming.
        dataset_extra_kwargs = json.loads(data_config.load_dataset_kwargs)
        stream_specs = dataset_extra_kwargs.pop("streams", None)
        # Removing the arguments we don't to pass to MosaicML dataset __init__.
        dataset_extra_kwargs.pop("num_workers", None)
        dataset_extra_kwargs.pop("pin_memory", None)
        dataset_extra_kwargs.pop("prefetch_factor", None)
        split = dataset_extra_kwargs.pop("split", mosaic_config.split)
        shuffle = dataset_extra_kwargs.pop("shuffle", mosaic_config.shuffle)
        batching_method = dataset_extra_kwargs.pop(
            "batching_method", mosaic_config.batching_method
        )

        # Make sure num_canonical_nodes >= number accelerator nodes.
        num_canonical_nodes = dataset_extra_kwargs.pop(
            "num_canonical_nodes", mosaic_config.num_canonical_nodes
        )
        if num_canonical_nodes is not None:
            num_canonical_nodes = max(
                dp_world_size // dist.get_local_world_size(), int(num_canonical_nodes)
            )

        # Local temporary dir needs to be common between all local ranks.
        # Default value to /tmp/RUN_NAME/mosaicml
        self.tmp_dir = dataset_extra_kwargs.pop(
            "cache_dir", mosaic_config.cache_dir
        )
        if self.tmp_dir is None:
            assert len(job_config.wandb.name) > 0
            self.tmp_dir = os.path.join(
                tempfile.gettempdir(), job_config.wandb.name, "mosaicml"
            )
        logger.info(f"MosaicML streaming local cache directory: '{self.tmp_dir}'.")

        # Not clear yet how to setup "replicate" in MosaicML StreamingDataset.
        # Make sure there is consistency between MosaicML distributed and TorchTitan.
        assert (
            dist.get_world_size() == dp_world_size
        ), "Only supports data parallel (DDP, FSDP) Not supporting SP, TP or PP yet."
        assert (
            dist.get_rank() == dp_rank
        ), "MosaicML rank and TorchTitan DP rank must coincide."

        # TODO: how to pass DP rank and world size?
        # MosaicML streaming using `replicate` and env. RANK, ... variables.
        # https://docs.mosaicml.com/projects/streaming/en/latest/dataset_configuration/replication_and_sampling.html#replication
        logger.info(
            f"MosaicML streaming dataset additional arguments: {dataset_extra_kwargs}."
        )
        self._mosaic_ds = StreamingDataset(
            streams=make_dataset_streams(
                data_config.dataset,
                data_config.dataset_path,
                split,
                self.tmp_dir,
                stream_specs,
            ),
            # num_canonical_nodes allows restarting with same data ordering on a smaller cluster.
            # (i.e. with any #nodes <= num_canonical_nodes).
            num_canonical_nodes=num_canonical_nodes,
            batch_size=data_config.local_batch_size,
            shuffle=shuffle,
            batching_method=batching_method,
            # Forwarding dataset kwargs.
            **dataset_extra_kwargs,
            # TODO: shuffle_seed from job config?
            # shuffle_seed=9176,
            # TODO: Default for shuffle block size?
            # shuffle_block_size=job_config.training.local_batch_size,
        )
        # NOTE: no explicit splitting required, Mosaic StreamingDataset doing it under the hood
        # using "RANK" and "WORLD_SIZE" env. variables.

        self._tokenizer = tokenizer
        self._seq_len = data_config.seq_len
        self._infinite = infinite

        # Additional variables for checkpointing
        self._sample_idx = 0
        self._token_buffer: list[int] = []
        self._dp_rank = dp_rank
        self._dp_world_size = dp_world_size

    def __iter__(self):
        max_buffer_token_len = 1 + self._seq_len

        # Build on the same pattern as Torchtitan HF streaming dataset:
        # load + concat a collection  of samples from dataset in a `token_buffer`,
        # then slice from the buffer individual inputs of `seq_len` size.
        # NOTE: the `token_buffer` needs to be saved during checkpointing to
        # ensure perfect resuming of dataset streaming.
        while True:
            for sample in iter(self._mosaic_ds):
                # Stratified resumption restarts at the last complete global
                # batch. Skip the already-consumed remainder restored below.
                if self._samples_to_skip_on_reload:
                    self._samples_to_skip_on_reload -= 1
                    continue
                # Use the dataset-specific text processor
                sample_text = sample["text"]
                sample_tokens = self._tokenizer.encode(
                    sample_text, add_bos=True, add_eos=True
                )
                # Buffer of tokens to sample from.
                self._token_buffer.extend(sample_tokens)
                self._sample_idx += 1

                while len(self._token_buffer) >= max_buffer_token_len:
                    x = torch.LongTensor(self._token_buffer[:max_buffer_token_len])
                    # update tokens to the remaining tokens
                    self._token_buffer = self._token_buffer[max_buffer_token_len:]
                    input = x[:-1]
                    label = x[1:]
                    yield {"input": input}, label

            if not self._infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        # Closing the dataset => exiting the internal iterator if existing.
        # Necessary to close properly when underlying local data is in a temporary directory
        # getting deleted.
        # Otherwise MosaicML may throw an error from internal prefetching loop thread.
        if hasattr(self, "_mosaic_ds") and hasattr(self._mosaic_ds, "_iterator"):
            self._mosaic_ds._iterator.non_blocking_exit()

    def load_state_dict(self, state_dict: Dict[str, Any]):
        # Keep the live iterator independent from the checkpoint object.
        self._token_buffer = list(state_dict["token_buffer"])
        self._sample_idx = state_dict["sample_idx"]
        assert "data" in state_dict
        if (
            self._mosaic_ds.batching_method == "stratified"
            and self._mosaic_ds.batch_size is not None
        ):
            self._samples_to_skip_on_reload = (
                state_dict["data"]["sample_in_epoch"]
                % self._mosaic_ds.batch_size
            )
        self._mosaic_ds.load_state_dict(state_dict["data"])
        # Mosaic can otherwise advance to a different shuffled epoch while
        # reconstructing the iterator from a worker snapshot.
        self._mosaic_ds._next_epoch.set(state_dict["data"]["epoch"])
        # Different DP rank or DP world size => discard token buffer.
        if (
            self._dp_rank != state_dict["dp_rank"]
            or self._dp_world_size != state_dict["dp_world_size"]
        ):
            logger.warning(
                "DP rank and world size are different. Discarding Mosaic dataset token buffer."
            )
            self._token_buffer = []

    def state_dict(self) -> Dict[str, Any]:
        # MosaicML Streaming dataloader expecting global DP number of samples. See:
        # https://github.com/mosaicml/streaming/blob/main/streaming/base/dataloader.py
        dp_num_samples = self._sample_idx * self._dp_world_size
        _state_dict = {
            # TorchData's incremental worker snapshots may be serialized by a
            # multiprocessing queue after this method returns.  Never expose
            # the live list: the next fetch can extend it before serialization.
            "token_buffer": list(self._token_buffer),
            # Sample idx is just the local sample idx the dataset worker.
            "sample_idx": self._sample_idx,
            # Keep DP world size + rank to check when re-loading on a different topology.
            "dp_rank": self._dp_rank,
            "dp_world_size": self._dp_world_size,
            # Passing world number of samples to MosaicML state dict.
            "data": self._mosaic_ds.state_dict(dp_num_samples, from_beginning=True),
        }
        return _state_dict


def get_dataloader_worker_states(state_dict: dict[str, Any]) -> List[dict[str, Any]]:
    """Get dataloader worker states from the dataloader state dict."""
    worker_snapshots = state_dict["_snapshot"]["_worker_snapshots"]
    worker_states = list(worker_snapshots.values())
    return worker_states


class MosaicDataloader(StatefulDataLoader, BaseDataLoader):
    """Mosaic streaming dataloader.

    Batching individual inputs from the dataset.
    """

    def __init__(
        self,
        dataset: IterableDataset,
        dp_rank: int,
        dp_world_size: int,
        batch_size: int,  # local batch size
        num_workers: int | None = None,
        pin_memory: bool = False,
        prefetch_factor: int = 4,
    ):
        # Default value if None: 8 workers per GPU. Minimal number from experiments to keep up on 1B model.
        if num_workers is None:
            num_workers = 8
        logger.info(
            f"MosaicML streaming dataloader with batch size {batch_size}, "
            f"{num_workers} workers, prefetch_factor={prefetch_factor}, "
            f"pin_memory={pin_memory}."
        )
        super().__init__(
            dataset,
            batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
        )
        # TODO: useful to save these?
        self.dp_world_size = dp_world_size
        self.dp_rank = dp_rank
        self._rank_id = f"dp_rank_{dp_rank}"

    def __del__(self) -> None:
        self.close()

    def state_dict(self) -> dict[str, Any]:
        # TODO: should we use the same DP rank trick as TorchTitan HF dataloader to avoid duplication?
        state_dict = super().state_dict()

        # We need to correct the `sample_in_epoch` from MosaicML state, to incorporate all dataset workers.
        worker_states = get_dataloader_worker_states(state_dict)
        local_num_samples = sum([w["dataset_state"]["sample_idx"] for w in worker_states])
        # Update MosaicML `sample_in_epoch` to a value accounting for all workers progress.
        # Otherwise, we would restart "back in time".
        for w in worker_states:
            w["dataset_state"]["data"]["sample_in_epoch"] = (
                local_num_samples * self.dp_world_size
            )
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # State being empty is valid (e.g. restarting from a checkpoint with a different dataset).
        if not state_dict:
            logger.warning("Empty MosaicML streaming dataloader state dict.")
            return

        # Checking the num of workers for restoring the dataloader state.
        # NOTE: Torch data doing a similar check, but we are providing a better error message here!
        state_num_workers = len(get_dataloader_worker_states(state_dict))
        assert (
            state_num_workers == self.num_workers
        ), f"The dataloader must be restored with the same number of workers: {state_num_workers} instead of {self.num_workers}."

        super().load_state_dict(state_dict)

        # StatefulDataLoader creates temporary workers while restoring its
        # snapshot. Mosaic's worker barrier/shared-memory state can outlive
        # those workers and deadlock the replacement workers on first use.
        clean_up_stale_workers(self.dataset._mosaic_ds)

    def close(self) -> None:
        # Shutdown iterator workers, following MosaicML reference dataloader.
        # https://github.com/mosaicml/streaming/blob/main/streaming/base/dataloader.py
        if self._iterator is not None:
            self._iterator._shutdown_workers()

        if hasattr(self.dataset, "close"):
            self.dataset.close()


def clean_up_stale_workers(streaming_dataset: StreamingDataset) -> None:
    """Remove stale Mosaic epoch state and reset its shared worker barrier.

    This must run after ``StatefulDataLoader.load_state_dict`` has stopped its
    temporary restore workers and before replacement workers start iterating.
    Shared-memory cleanup is best effort, but resetting the barrier is required
    so new workers cannot inherit an incomplete enter/exit cycle.
    """
    shm_prefix_int = streaming_dataset._shm_prefix_int
    for memory_object in (EPOCH_SHAPE, EPOCH_DATA):
        shared_memory = None
        try:
            shared_memory = SharedMemory(
                name=_get_path(shm_prefix_int, memory_object), create=False
            )
            shared_memory.unlink()
        except FileNotFoundError:
            pass
        except Exception as error:
            logger.warning(
                "Failed to clean stale Mosaic shared memory %s: %s",
                memory_object,
                error,
            )
        finally:
            if shared_memory is not None:
                shared_memory.close()

    barrier = streaming_dataset._shared_barrier
    barrier.num_enter = 0
    barrier.num_exit = -1
    barrier.flag = True


def build_mosaic_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    job_config: JobConfig,
    infinite: bool = True,
) -> BaseDataLoader:
    """Build MosaicML streaming dataloader."""
    dataset_extra_kwargs = json.loads(job_config.training.load_dataset_kwargs)
    local_batch_size = job_config.training.local_batch_size
    mosaic_config = job_config.training.mosaicml_dataset
    num_workers = dataset_extra_kwargs.get("num_workers", mosaic_config.num_workers)
    pin_memory = bool(dataset_extra_kwargs.get("pin_memory", False))
    prefetch_factor = int(
        dataset_extra_kwargs.get("prefetch_factor", mosaic_config.prefetch_factor)
    )

    logger.info(f"Building Mosaic Streaming dataset '{job_config.training.dataset}'.")
    dataset = MosaicStreamingDataset(
        job_config=job_config,
        tokenizer=tokenizer,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
    )
    return MosaicDataloader(
        dataset=dataset,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=local_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )


def build_mosaic_validation_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    job_config: JobConfig,
    infinite: bool = False,
) -> BaseDataLoader:
    """Build a Mosaic Streaming validation dataloader from validation config."""
    dataset_extra_kwargs = json.loads(job_config.validation.load_dataset_kwargs)
    mosaic_config = job_config.validation.mosaicml_dataset
    num_workers = dataset_extra_kwargs.get("num_workers", mosaic_config.num_workers)
    pin_memory = bool(dataset_extra_kwargs.get("pin_memory", False))
    prefetch_factor = int(
        dataset_extra_kwargs.get("prefetch_factor", mosaic_config.prefetch_factor)
    )
    logger.info(
        f"Building Mosaic validation dataset '{job_config.validation.dataset}'."
    )
    dataset = MosaicStreamingDataset(
        job_config=job_config,
        data_config=job_config.validation,
        tokenizer=tokenizer,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
    )
    return MosaicDataloader(
        dataset=dataset,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=job_config.validation.local_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )
