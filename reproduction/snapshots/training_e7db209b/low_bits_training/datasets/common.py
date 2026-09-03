#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
import torchtitan.components.validate
from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.components.tokenizer import BaseTokenizer

from torchtitan.config import JobConfig

from .hf_datasets import build_hf_dataloader
from .packed_binary import build_packed_binary_dataloader


def build_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    job_config: JobConfig,
    infinite: bool = True,
) -> BaseDataLoader:
    """Build a dataloader. By default HuggingFace dataloader, Mosaic if dataset name is starting with `mosaic/`.

    Args:
        dp_world_size: Data parallel world size.
        dp_rank: Data parallel rank.
        tokenizer: Tokenizer to use.
        job_config: Job config.
    """
    dataset_name = job_config.training.dataset
    if dataset_name in {"packed-bin", "packed_binary", "token-packed"}:
        return build_packed_binary_dataloader(
            dp_world_size, dp_rank, tokenizer, job_config, infinite
        )
    # Using dataset name prefix to select proper builder method.
    if dataset_name.startswith(("mosaic/", "lbt/")):
        from .mosaic_datasets import build_mosaic_dataloader

        return build_mosaic_dataloader(
            dp_world_size, dp_rank, tokenizer, job_config, infinite
        )
    # Default to HF dataloader
    return build_hf_dataloader(dp_world_size, dp_rank, tokenizer, job_config, infinite)


def build_text_validation_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    job_config: JobConfig,
    infinite: bool = False,
) -> BaseDataLoader:
    """Dispatch validation through the LBT Mosaic loader when requested."""
    dataset_name = job_config.validation.dataset
    if dataset_name.startswith(("mosaic/", "lbt/")):
        from .mosaic_datasets import build_mosaic_validation_dataloader

        return build_mosaic_validation_dataloader(
            dp_world_size, dp_rank, tokenizer, job_config, infinite
        )

    from torchtitan.hf_datasets.text_datasets import (
        build_text_validation_dataloader as build_hf_validation_dataloader,
    )

    return build_hf_validation_dataloader(
        dp_world_size, dp_rank, tokenizer, job_config, infinite
    )


# TorchTitan imports this symbol into the validation component at module load.
# Replace that module-global dispatch point before Trainer constructs Validator.
torchtitan.components.validate.build_text_validation_dataloader = (
    build_text_validation_dataloader
)
