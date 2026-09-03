#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
"""Small converter subset required by the reproduced FA4 training routes.

This is the portable subset of ``low_bits_training.converters`` at
``e7db209b``.  It deliberately excludes unrelated quantization and CCE
converters.  The BF16 conversion and safe initializer are part of the measured
training contract: the latter avoids low-precision inverse-CDF endpoint atoms
when initializing large tensors.  It also carries the structural MLP and FP32
master-parameter converters required by the recovered SFU-B1 and D64 chains.
"""

from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
from threading import RLock
from types import MethodType
from typing import Any, Callable, Generator, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.distributed import ParallelDims
from torchtitan.models.llama3 import Transformer
from torchtitan.models.llama3.model.model import FeedForward
from torchtitan.protocols.model_converter import (
    ModelConverter,
    register_model_converter,
)
from torchtitan.tools.logging import logger

from .job_config import JobConfig


_ORIGINAL_TRUNC_NORMAL = nn.init.trunc_normal_
_SAFE_TRUNC_NORMAL_LOCK = RLock()


def _safe_trunc_normal_(
    tensor: torch.Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Initialize BF16/FP16/FP32 without inverse-CDF endpoint atoms."""
    safe_dtypes = (torch.bfloat16, torch.float16, torch.float32)
    if tensor.dtype not in safe_dtypes:
        return _ORIGINAL_TRUNC_NORMAL(
            tensor, mean=mean, std=std, a=a, b=b, generator=generator
        )

    far_bounds = std > 0.0 and (mean - a) / std >= 8.0 and (b - mean) / std >= 8.0
    if tensor.dtype == torch.float32 and not far_bounds:
        return _ORIGINAL_TRUNC_NORMAL(
            tensor, mean=mean, std=std, a=a, b=b, generator=generator
        )

    with torch.no_grad():
        temporary = (
            tensor
            if tensor.dtype == torch.float32
            else torch.empty_like(tensor, dtype=torch.float32)
        )
        if far_bounds:
            temporary.normal_(mean=mean, std=std, generator=generator)
        else:
            _ORIGINAL_TRUNC_NORMAL(
                temporary, mean=mean, std=std, a=a, b=b, generator=generator
            )
        if temporary is not tensor:
            tensor.copy_(temporary)
    return tensor


@contextmanager
def _safe_low_precision_trunc_normal() -> Generator[None, None, None]:
    with _SAFE_TRUNC_NORMAL_LOCK:
        previous = nn.init.trunc_normal_
        nn.init.trunc_normal_ = _safe_trunc_normal_
        try:
            yield
        finally:
            nn.init.trunc_normal_ = previous


def _safe_bfloat16_init_weights(
    root: nn.Module,
    *args: Any,
    **kwargs: Any,
) -> Any:
    with _safe_low_precision_trunc_normal():
        return root._fa4_original_init_weights(*args, **kwargs)


def _install_safe_bfloat16_init(model: nn.Module) -> None:
    if getattr(model, "_fa4_safe_bfloat16_init", False):
        return
    original_init_weights = getattr(model, "init_weights", None)
    if not callable(original_init_weights):
        raise TypeError("bfloat16 conversion requires model.init_weights")
    model._fa4_original_init_weights = original_init_weights
    model.init_weights = MethodType(_safe_bfloat16_init_weights, model)
    model._fa4_safe_bfloat16_init = True


class Bfloat16Converter(ModelConverter):
    """Store Llama trainable parameters in BF16 while preserving RoPE."""

    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        del parallel_dims
        if job_config.model.name != "llama3_gc":
            raise ValueError(
                "the reproduced bfloat16 converter supports model.name="
                "'llama3_gc' only"
            )

    def convert(self, model: nn.Module) -> None:
        if not isinstance(model, Transformer):
            raise TypeError("the FA4 bfloat16 converter requires TorchTitan Llama")
        # Casting the root would destroy the complex component of the RoPE
        # buffer. Cast only trainable submodules, as in the measured code.
        model.layers.to(dtype=torch.bfloat16)
        model.tok_embeddings.to(dtype=torch.bfloat16)
        model.norm.to(dtype=torch.bfloat16)
        model.output.to(dtype=torch.bfloat16)
        _install_safe_bfloat16_init(model)

    def post_optimizer_hook(self, model: Union[nn.Module, List[nn.Module]]) -> None:
        del model


register_model_converter(Bfloat16Converter, "bfloat16")


class FeedForwardWithFusedLinear(nn.Module):
    """SwiGLU MLP with a single concatenated gate/up projection."""

    @classmethod
    @torch.no_grad()
    def from_unfused(cls, module: FeedForward) -> "FeedForwardWithFusedLinear":
        module = deepcopy(module)
        module.__class__ = cls
        dim, hidden_dim = module.w2.weight.shape
        device = module.w1.weight.device
        dtype = module.w1.weight.dtype

        module.w_in = nn.Linear(
            dim,
            2 * hidden_dim,
            bias=False,
            device=device,
            dtype=dtype,
        )
        module.w_out = nn.Linear(
            hidden_dim,
            dim,
            bias=False,
            device=device,
            dtype=dtype,
        )
        if device != torch.device("meta"):
            module.w_in.weight.copy_(
                torch.cat((module.w1.weight, module.w3.weight), dim=0)
            )
            module.w_out.weight.copy_(module.w2.weight)
        del module.w1, module.w2, module.w3
        return module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activation_input, gate = torch.chunk(self.w_in(x), chunks=2, dim=-1)
        return self.w_out(F.silu(activation_input) * gate)

    def init_weights(self, init_std: float) -> None:
        hidden_dim = self.w_out.weight.shape[-1]
        nn.init.trunc_normal_(self.w_in.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.w_in.weight[hidden_dim:].mul_(init_std / 0.02)
        nn.init.trunc_normal_(self.w_out.weight, mean=0.0, std=init_std)
        if hasattr(self.w_in, "norm_weight"):
            self.w_in.norm_weight.data.fill_(1.0)


def _replace_modules(
    module: nn.Module,
    source_type: type[nn.Module],
    replacement: Callable[[nn.Module], nn.Module],
) -> None:
    for name, child in module.named_children():
        if isinstance(child, source_type):
            setattr(module, name, replacement(child))
        else:
            _replace_modules(child, source_type, replacement)


class FusedMLPLinearConverter(ModelConverter):
    """Fuse only SwiGLU gate/up projections, leaving attention untouched."""

    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        del job_config, parallel_dims

    def convert(self, model: nn.Module) -> None:
        logger.info("Converting MLPs to use fused gate/up linear layers")
        _replace_modules(
            model,
            FeedForward,
            FeedForwardWithFusedLinear.from_unfused,
        )

    def post_optimizer_hook(self, model: Union[nn.Module, List[nn.Module]]) -> None:
        del model


register_model_converter(FusedMLPLinearConverter, "fuse_mlp_linear")


class FeedForwardWithPatchedActivation(nn.Module):
    """Unfused SwiGLU MLP whose activation is selected by configuration."""

    activation_impl_name: str
    activation_fn: Callable[[torch.Tensor], torch.Tensor]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.activation_fn(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: float) -> None:
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


class FeedForwardFusedWithPatchedActivation(nn.Module):
    """Fused gate/up MLP whose activation is selected by configuration."""

    activation_impl_name: str
    activation_fn: Callable[[torch.Tensor], torch.Tensor]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activation_input, gate = torch.chunk(self.w_in(x), chunks=2, dim=-1)
        return self.w_out(self.activation_fn(activation_input) * gate)

    def init_weights(self, init_std: float) -> None:
        hidden_dim = self.w_out.weight.shape[-1]
        nn.init.trunc_normal_(self.w_in.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.w_in.weight[hidden_dim:].mul_(init_std / 0.02)
        nn.init.trunc_normal_(self.w_out.weight, mean=0.0, std=init_std)
        if hasattr(self.w_in, "norm_weight"):
            self.w_in.norm_weight.data.fill_(1.0)


def _resolve_mlp_activation_impl(
    name: str,
) -> tuple[str, Callable[[torch.Tensor], torch.Tensor]]:
    if name == "native_silu":
        return name, F.silu
    if name == "native_gelu":
        return name, F.gelu
    raise ValueError(
        f"unsupported public MLP activation_impl {name!r}; expected "
        "'native_silu' or 'native_gelu'. The external experimental spline "
        "extension is not in the authenticated release source closure."
    )


def _patch_mlp_modules(
    module: nn.Module,
    activation_impl_name: str,
    activation_fn: Callable[[torch.Tensor], torch.Tensor],
) -> int:
    patched = 0
    for child in module.modules():
        if all(hasattr(child, name) for name in ("w1", "w2", "w3")):
            child.activation_impl_name = activation_impl_name
            child.activation_fn = activation_fn
            child.__class__ = FeedForwardWithPatchedActivation
            patched += 1
        elif all(hasattr(child, name) for name in ("w_in", "w_out")):
            child.activation_impl_name = activation_impl_name
            child.activation_fn = activation_fn
            child.__class__ = FeedForwardFusedWithPatchedActivation
            patched += 1
    return patched


class SplineMLPConverter(ModelConverter):
    """Reproduce the SFU-B1 activation-selection stage.

    The measured recipe selected ``native_silu``.  The legacy converter name
    remains part of the authenticated ordered chain, but this public version
    fails closed for the unrecovered external spline extension.
    """

    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        del parallel_dims
        self.activation_impl_name, self.activation_fn = _resolve_mlp_activation_impl(
            job_config.spline_mlp.activation_impl
        )

    def convert(self, model: nn.Module) -> None:
        patched = _patch_mlp_modules(
            model,
            self.activation_impl_name,
            self.activation_fn,
        )
        logger.info(
            "Patched %d MLP modules to activation_impl=%s",
            patched,
            self.activation_impl_name,
        )

    def post_optimizer_hook(self, model: Union[nn.Module, List[nn.Module]]) -> None:
        del model


register_model_converter(SplineMLPConverter, "spline_mlp")


class Float32MasterParamsConverter(ModelConverter):
    """Promote optimizer-owned floating parameters while preserving aliases."""

    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        del parallel_dims
        self.compute_dtype = job_config.training.mixed_precision_param

    def convert(self, model: nn.Module) -> None:
        promoted = 0
        promoted_numel = 0
        for parameter in model.parameters():
            if parameter.is_floating_point() and parameter.dtype != torch.float32:
                # This runs before FSDP and optimizer construction. Replacing
                # ``data`` preserves the Parameter object and weight ties.
                parameter.data = parameter.data.to(dtype=torch.float32)
                promoted += 1
                promoted_numel += parameter.numel()
        logger.info(
            "Promoted %d parameters (%d elements) to FP32 master storage; "
            "mixed-precision compute dtype=%s",
            promoted,
            promoted_numel,
            self.compute_dtype,
        )

    def post_optimizer_hook(self, model: Union[nn.Module, List[nn.Module]]) -> None:
        del model


register_model_converter(Float32MasterParamsConverter, "fp32_master")


__all__ = [
    "Bfloat16Converter",
    "FeedForwardFusedWithPatchedActivation",
    "FeedForwardWithFusedLinear",
    "FeedForwardWithPatchedActivation",
    "Float32MasterParamsConverter",
    "FusedMLPLinearConverter",
    "SplineMLPConverter",
    "_safe_trunc_normal_",
]
