"""Portable registration entry point for the FP4 FA4 experiments.

Use this module as both ``job.custom_config_module`` and
``experimental.custom_import``.  Importing it registers the measured Llama
geometries, BF16 storage conversion, BF16 FA4 comparator, exact low-precision
adapter, custom optimizer, and checkpoint-state hooks.
"""

from . import converters as _converters  # noqa: F401
from . import exact_lowp_attention as _exact_lowp_attention  # noqa: F401
from . import fa4_attention as _fa4_attention  # noqa: F401
from .checkpoint import install_checkpoint_hooks
from .job_config import JobConfig
from .train_spec import register_fa4_train_spec


register_fa4_train_spec()
install_checkpoint_hooks()


__all__ = ["JobConfig"]
