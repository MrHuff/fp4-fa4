#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
from typing import Union

from torchtitan.config import JobConfig
from torchtitan.components.tokenizer import build_hf_tokenizer, HuggingFaceTokenizer

from .tiktoken_tokenizer import TikTokenizer


def build_hf_or_tiktoken_tokenizer(
    job_config: JobConfig,
) -> Union[HuggingFaceTokenizer, TikTokenizer]:
    """
    Builds a HuggingFaceTokenizer or TikTokenizer from the specified path.
    Backward compatible function, supporting older TorchTitan Tiktoken version.

    Args:
        JobConfig: A JobConfig object containing the path to the tokenizer directory.
    """
    path = job_config.model.tokenizer_path
    # Backward compatibility with Tiktoken models.
    if path is not None and path.endswith(".model"):
        return TikTokenizer(job_config.model.tokenizer_path)
    return build_hf_tokenizer(job_config)
