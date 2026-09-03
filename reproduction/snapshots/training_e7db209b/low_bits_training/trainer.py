#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
import torch
import torch.nn as nn
import logging
import math
import os
import sys
import time
import types
from contextlib import contextmanager
from typing import Any, Dict, List, Tuple, Union
from unittest.mock import patch
import importlib

# Disable CUDA graphs for torch.compile - Quartet custom ops are not CUDA graph compatible
# This must be done BEFORE any torch.compile calls
import torch._inductor.config
torch._inductor.config.triton.cudagraphs = False
# torch._inductor.config.triton.cudagraph_trees = False # Not exposed in wrapper
torch._inductor.config.force_disable_caches = True

# Titan Components
from .components.remote_synchronizer import (
    RemoteSynchronizer,
    is_remote_synchronizer_active,
)
from .ema_checkpoint import CheckpointManagerPatching
from .ema_checkpoint import CheckpointManagerWithEMAWeight
from .ema_checkpoint import install_sync_dcp_cpu_process_group
from .config import JobConfig
from .converters import ensure_fp32_master_params
from .cce_config import cce_path_handles_loss
from .signal_handler import SignalHandler, distributed_exit_requested
from .utils import NanLossDetectingHandler
from . import compat
from . import experiments
from .batch_resume import (
    BatchFingerprint,
    batch_fingerprints_enabled,
    emit_batch_fingerprint,
    fingerprint_batch,
    install_checkpoint_aligned_dataloader,
)

from torchtitan.tools.logging import logger
from torchtitan.train import Trainer as BaseTrainer
import torchtitan.train as titan_train_module
from torchtitan.components.dataloader import DataloaderExhaustedError
try:
    from torchtitan.train import _lbt_step_trace
except ImportError:
    def _lbt_step_trace(step: int, event: str) -> None:
        return None
import torchtitan.protocols.train_spec as spec_module
from torchtitan.distributed import ParallelDims, utils as dist_utils
# torch.set_float32_matmul_precision('medium')

# Keep both CCE implementations off the dense standard-cross-entropy import
# path.  Importing low_bits_training.cce executes its backend module, which in
# turn probes cut_cross_entropy and initializes a large amount of optional CCE
# machinery.  The job configuration is not available while this module is
# imported, so defer both imports until an explicitly CCE-enabled Trainer is
# constructed (or the legacy manual CCE helper is called).
linear_cross_entropy = None
cut_cross_entropy_cce = None
cut_cross_entropy_torch_compile = None
cut_cross_entropy_utils = None
_cut_cross_entropy_import_attempted = False


def _load_cce_backend(job_config: JobConfig):
    """Load the internal-loss backend only for an explicit CCE configuration."""
    if not cce_path_handles_loss(job_config):
        return None
    return importlib.import_module(".cce", package=__package__)


def _load_cut_cross_entropy():
    """Lazily load the legacy Cut Cross Entropy implementation."""
    global linear_cross_entropy
    global cut_cross_entropy_cce
    global cut_cross_entropy_torch_compile
    global cut_cross_entropy_utils
    global _cut_cross_entropy_import_attempted

    if _cut_cross_entropy_import_attempted:
        return linear_cross_entropy
    _cut_cross_entropy_import_attempted = True

    try:
        package = importlib.import_module("cut_cross_entropy")
        cut_cross_entropy_cce = importlib.import_module("cut_cross_entropy.cce")
        cut_cross_entropy_torch_compile = importlib.import_module(
            "cut_cross_entropy.torch_compile"
        )
        cut_cross_entropy_utils = importlib.import_module("cut_cross_entropy.utils")
        linear_cross_entropy = torch._dynamo.disable(package.linear_cross_entropy)
    except ImportError:
        linear_cross_entropy = None
        cut_cross_entropy_cce = None
        cut_cross_entropy_torch_compile = None
        cut_cross_entropy_utils = None
        return None

    _patch_cut_cross_entropy_dense_label_fastpath()
    return linear_cross_entropy


_HSDP_REDUCE_SCATTER_ACCUMULATION_ENV = (
    "TORCHTITAN_HSDP_ACCUMULATE_WITH_REDUCE_SCATTER"
)
_FSDP_NO_SYNC_ACCUMULATION_ENV = "TORCHTITAN_FSDP_ACCUMULATE_WITHOUT_SYNC"

_REMOTE_SYNC_ENV_SPECS = {
    "LBT_REMOTE_SYNC_COMMAND_TIMEOUT_SECS": (
        "sync_command_timeout_secs",
        float,
        False,
    ),
    "LBT_REMOTE_SYNC_FINAL_TIMEOUT_SECS": (
        "final_sync_timeout_secs",
        float,
        False,
    ),
    "LBT_REMOTE_SYNC_FINAL_ATTEMPTS": ("final_sync_attempts", int, False),
    "LBT_REMOTE_SYNC_FINAL_RETRY_SECS": (
        "final_sync_retry_secs",
        float,
        True,
    ),
}


def _remote_synchronizer_kwargs_from_env() -> dict[str, int | float]:
    """Parse optional bounded-shutdown settings without changing defaults."""
    kwargs: dict[str, int | float] = {}
    for env_name, (argument_name, value_type, allow_zero) in (
        _REMOTE_SYNC_ENV_SPECS.items()
    ):
        raw_value = os.environ.get(env_name)
        if raw_value is None:
            continue
        if not raw_value or raw_value != raw_value.strip():
            raise ValueError(f"{env_name} must be a finite numeric value")

        if value_type is int:
            if any(character not in "0123456789" for character in raw_value):
                raise ValueError(f"{env_name} must be a positive integer")
            value: int | float = int(raw_value)
        else:
            try:
                value = float(raw_value)
            except ValueError as error:
                raise ValueError(
                    f"{env_name} must be a finite numeric value"
                ) from error
            if not math.isfinite(value):
                raise ValueError(f"{env_name} must be a finite numeric value")

        if value < 0 or (value == 0 and not allow_zero):
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{env_name} must be {qualifier}")
        kwargs[argument_name] = value
    return kwargs


@contextmanager
def _hsdp_reduce_scatter_accumulation(enabled: bool):
    """Use FSDP's accumulation loop while suppressing only HSDP all-reduce."""
    if not enabled:
        yield
        return

    if os.environ.get(_FSDP_NO_SYNC_ACCUMULATION_ENV, "0") == "1":
        raise RuntimeError(
            f"{_HSDP_REDUCE_SCATTER_ACCUMULATION_ENV} and "
            f"{_FSDP_NO_SYNC_ACCUMULATION_ENV} are mutually exclusive"
        )

    from torch.distributed.fsdp import FSDPModule

    set_requires_all_reduce = getattr(
        FSDPModule, "set_requires_all_reduce", None
    )
    if set_requires_all_reduce is None:
        raise RuntimeError(
            f"{_HSDP_REDUCE_SCATTER_ACCUMULATION_ENV}=1 requires a "
            "PyTorch FSDPModule with set_requires_all_reduce()"
        )

    previous_no_sync = os.environ.get(_FSDP_NO_SYNC_ACCUMULATION_ENV)
    os.environ[_FSDP_NO_SYNC_ACCUMULATION_ENV] = "1"
    try:
        # TorchTitan already toggles this API around each microbatch. For HSDP,
        # keep shard-local reduce-scatter and defer only replica all-reduce.
        with patch.object(
            FSDPModule,
            "set_requires_gradient_sync",
            set_requires_all_reduce,
        ):
            yield
    finally:
        if previous_no_sync is None:
            os.environ.pop(_FSDP_NO_SYNC_ACCUMULATION_ENV, None)
        else:
            os.environ[_FSDP_NO_SYNC_ACCUMULATION_ENV] = previous_no_sync


def _use_lbt_cce_assume_dense_labels() -> bool:
    return os.environ.get("USE_LBT_CCE_ASSUME_DENSE_LABELS", "1") == "1"


def _pin_memory_tree(obj: Any) -> Any:
    if torch.is_tensor(obj):
        if obj.device.type == "cpu" and not obj.is_pinned():
            return obj.pin_memory()
        return obj
    if isinstance(obj, dict):
        return {k: _pin_memory_tree(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(_pin_memory_tree(v) for v in obj)
    if isinstance(obj, list):
        return [_pin_memory_tree(v) for v in obj]
    return obj


def _to_device_tree(obj: Any, device: torch.device | str) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _to_device_tree(v, device) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(_to_device_tree(v, device) for v in obj)
    if isinstance(obj, list):
        return [_to_device_tree(v, device) for v in obj]
    return obj


def _record_stream_tree(obj: Any, stream: torch.cuda.Stream) -> None:
    if torch.is_tensor(obj):
        if obj.device.type == "cuda":
            obj.record_stream(stream)
        return
    if isinstance(obj, dict):
        for value in obj.values():
            _record_stream_tree(value, stream)
        return
    if isinstance(obj, (tuple, list)):
        for value in obj:
            _record_stream_tree(value, stream)


def _patch_cut_cross_entropy_dense_label_fastpath() -> None:
    if (
        linear_cross_entropy is None
        or cut_cross_entropy_cce is None
        or cut_cross_entropy_torch_compile is None
        or cut_cross_entropy_utils is None
    ):
        return

    if getattr(cut_cross_entropy_utils, "_lbt_dense_valids_patch_installed", False):
        return

    original_build_flat_valids = cut_cross_entropy_utils._build_flat_valids

    def _build_flat_valids_fast(
        targets: torch.Tensor,
        ignore_index: int,
        shift: int,
    ) -> torch.Tensor | None:
        if _use_lbt_cce_assume_dense_labels() and int(shift) == 0:
            return None
        return original_build_flat_valids(targets, ignore_index, shift)

    cut_cross_entropy_utils._build_flat_valids = _build_flat_valids_fast
    cut_cross_entropy_cce._build_flat_valids = _build_flat_valids_fast
    cut_cross_entropy_torch_compile._build_flat_valids = _build_flat_valids_fast
    cut_cross_entropy_utils._lbt_dense_valids_patch_installed = True


# ------------------------------------------------------------------------------
# CLASS: FSDP-SAFE CCE WRAPPER
# ------------------------------------------------------------------------------
class TitanCCELayer(nn.Module):
    def __init__(self, original_linear):
        super().__init__()
        # Preserve the original Parameter object, not merely its storage.  The
        # authenticated D64 model ties output.weight to tok_embeddings.weight;
        # manufacturing a second Parameter here silently breaks that topology
        # and makes optimizers/FSDP observe a duplicate leaf.
        self.weight = original_linear.weight
        self.bias = original_linear.bias

    @torch._dynamo.disable
    def forward(self, x, labels=None):
        if labels is not None:
            # TRAINING: Fused CCE
            legacy_cce = _load_cut_cross_entropy()
            if legacy_cce is None:
                raise ImportError("cut_cross_entropy is not installed.")
            step = os.environ.get("LBT_TRACE_ACTIVE_STEP", "").strip() or "?"
            if _use_lbt_cce_timing():
                print(
                    f"[CCE TRACE] step={step} input shape={tuple(x.shape)} stride={tuple(x.stride())} "
                    f"contiguous={x.is_contiguous()} dtype={x.dtype} weight_dtype={self.weight.dtype}",
                    file=sys.stderr,
                    flush=True,
                )
            cce_fwd_start = _lbt_cce_begin(step, "cce_forward")
            loss = legacy_cce(
                x, 
                self.weight, 
                labels, 
                shift=False, 
                reduction='mean'
            )
            _lbt_cce_end(step, "cce_forward", cce_fwd_start)

            if _use_lbt_cce_timing() and loss.requires_grad and x.requires_grad:
                state = {"start": None}

                def _loss_hook(grad):
                    if state["start"] is None:
                        state["start"] = _lbt_cce_begin(step, "cce_backward")
                    return grad

                def _input_hook(grad):
                    _lbt_cce_end(step, "cce_backward", state["start"])
                    return grad

                loss.register_hook(_loss_hook)
                x.register_hook(_input_hook)
            return loss
        else:
            # INFERENCE: Standard Linear
            return torch.nn.functional.linear(x, self.weight, self.bias)

# ------------------------------------------------------------------------------
# MANUAL CCE SURGERY FUNCTION
# ------------------------------------------------------------------------------
def apply_titan_manual_cce(model):
    if _load_cut_cross_entropy() is None:
        raise ImportError("cut_cross_entropy is not installed.")

    # 1. Validation 
    check_model = model
    if hasattr(check_model, 'module'): check_model = check_model.module
    
    if not hasattr(check_model, 'output'):
        logger.warning("⚠️ Could not find 'output' on top level model. Proceeding with surgery anyway...")
    
    logger.info("🔪 CCE SURGERY: Replacing 'model.output' with FSDP-Safe 'TitanCCELayer'...")

    # 2. REPLACE THE LAYER
    check_model.output = TitanCCELayer(check_model.output)
    
    # 3. PATCH THE MODEL FORWARD
    def forward_cce(self, tokens: torch.Tensor, start_pos: int = 0, labels: torch.Tensor = None):
        # --- HELPER: DEEP PEELING ---
        def get_raw_model(obj):
            max_depth = 20
            depth = 0
            while depth < max_depth:
                if hasattr(obj, '_fsdp_wrapped_module'):
                    obj = obj._fsdp_wrapped_module
                elif hasattr(obj, 'module'):
                    obj = obj.module
                elif hasattr(obj, '_orig_mod'):
                    obj = obj._orig_mod
                else:
                    return obj
                depth += 1
            return obj

        raw_model = get_raw_model(self)

        tok_embeddings = getattr(raw_model, 'tok_embeddings', None)
        layers = getattr(raw_model, 'layers', None)
        norm = getattr(raw_model, 'norm', None)
        output = getattr(raw_model, 'output', None)
        
        if tok_embeddings is None or layers is None:
            raise AttributeError(f"Could not find layers/embeddings in raw model: {type(raw_model)}")

        freqs_cis = getattr(raw_model, 'freqs_cis', None)
        mask = getattr(raw_model, 'mask', None)

        # Do not generate tensor mask if None (Model expects BlockMask or None)
        
        if freqs_cis is not None:
             seqlen = tokens.shape[1]
             freqs_cis = freqs_cis.to(tokens.device)
             current_freqs = freqs_cis[start_pos : start_pos + seqlen]
        else:
             current_freqs = None

        step = _lbt_active_step()

        embed_start = _lbt_cce_begin(step, "model_embed")
        h = tok_embeddings(tokens)
        _lbt_cce_end(step, "model_embed", embed_start)
        
        layer_iter = layers.values() if isinstance(layers, (nn.ModuleDict, dict)) else layers
        layer_stack_start = _lbt_cce_begin(step, "model_layers")
        for layer in layer_iter:
            h = layer(h, current_freqs, mask)
        _lbt_cce_end(step, "model_layers", layer_stack_start)
            
        norm_start = _lbt_cce_begin(step, "model_norm")
        h = norm(h)
        _lbt_cce_end(step, "model_norm", norm_start)
        
        if labels is not None:
            output_start = _lbt_cce_begin(step, "model_output")
            result = output(h, labels=labels)
            _lbt_cce_end(step, "model_output", output_start)
            return result
        else:
            return output(h)

    model.forward = types.MethodType(forward_cce, model)
    return model


def _use_debug_nonfinite_grad_scan() -> bool:
    return os.environ.get("USE_LBT_DEBUG_NONFINITE_GRADS", "0") == "1"


def _fail_on_nonfinite_metric(value: torch.Tensor | float, label: str) -> None:
    """Fail exact jobs on-device without a per-microbatch host sync."""
    if os.environ.get("LBT_FAIL_ON_NONFINITE_METRICS", "0") != "1":
        return
    scalar = value.detach() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    finite = torch.isfinite(scalar).all()
    message = f"non-finite {label} detected by exact training guard"
    if finite.device.type == "cuda":
        torch._assert_async(finite, message)
    elif not bool(finite.item()):
        raise FloatingPointError(message)


def _use_lbt_cce_timing() -> bool:
    return os.environ.get("USE_LBT_CCE_TIMING", "0") == "1"


def _lbt_active_step() -> str:
    return os.environ.get("LBT_TRACE_ACTIVE_STEP", "").strip() or "?"


def _lbt_cce_begin(step: int | str | None, name: str) -> float | None:
    if not _use_lbt_cce_timing():
        return None
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    print(
        f"[CCE TRACE] t={start:.6f} step={step} {name} start",
        file=sys.stderr,
        flush=True,
    )
    return start


def _lbt_cce_end(step: int | str | None, name: str, start: float | None) -> None:
    if start is None:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()
    print(
        f"[CCE TRACE] t={end:.6f} step={step} {name} end elapsed_ms={(end - start) * 1000.0:.3f}",
        file=sys.stderr,
        flush=True,
    )


def _scan_nonfinite_grads(model_parts) -> None:
    bad: list[str] = []
    for model_idx, module in enumerate(model_parts):
        for name, param in module.named_parameters():
            grad = param.grad
            if grad is None:
                continue
            if torch.is_tensor(grad) and not bool(torch.isfinite(grad).all().item()):
                grad_detached = grad.detach()
                nan_count = int(torch.isnan(grad_detached).sum().item())
                posinf_count = int(torch.isposinf(grad_detached).sum().item())
                neginf_count = int(torch.isneginf(grad_detached).sum().item())
                bad.append(
                    f"model_parts[{model_idx}].{name} "
                    f"shape={tuple(grad_detached.shape)} "
                    f"nan={nan_count} +inf={posinf_count} -inf={neginf_count}"
                )
                if len(bad) >= 8:
                    break
        if bad:
            break

    if bad:
        details = "\n".join(bad)
        raise RuntimeError(f"Non-finite parameter gradients detected before clip_grad_norm_:\n{details}")


def _use_debug_top_param_grads() -> bool:
    return os.environ.get("USE_TK_DEBUG_TOP_PARAM_GRADS", "0") == "1"


def _debug_top_param_grads_topk() -> int:
    value = os.environ.get("USE_TK_DEBUG_TOP_PARAM_GRADS_TOPK", "32").strip()
    try:
        return max(1, int(value))
    except Exception:
        return 32


def _dump_top_param_grads(model_parts) -> None:
    rows = []
    for model_idx, module in enumerate(model_parts):
        for name, param in module.named_parameters():
            grad = param.grad
            if grad is None or not torch.is_tensor(grad):
                continue
            g = grad.detach()
            if g.numel() == 0:
                continue
            if not g.is_floating_point():
                g = g.to(torch.float32)
            else:
                g = g.float()
            p = param.detach()
            if not p.is_floating_point():
                p = p.to(torch.float32)
            else:
                p = p.float()
            rms = float(torch.sqrt((g * g).mean()).item())
            max_abs = float(g.abs().max().item())
            norm = float(torch.linalg.vector_norm(g).item())
            p_rms = float(torch.sqrt((p * p).mean()).item())
            p_max_abs = float(p.abs().max().item())
            rows.append((rms, max_abs, norm, p_rms, p_max_abs, model_idx, name, tuple(grad.shape)))

    rows.sort(key=lambda x: x[0], reverse=True)
    topk = _debug_top_param_grads_topk()
    print(
        f"[LBT TOP PARAM GRADS] count={len(rows)} topk={topk}",
        file=sys.stderr,
        flush=True,
    )
    for rms, max_abs, norm, p_rms, p_max_abs, model_idx, name, shape in rows[:topk]:
        print(
            f"[LBT TOP PARAM GRADS] model_parts[{model_idx}].{name} "
            f"shape={shape} rms={rms:.6e} max_abs={max_abs:.6e} norm={norm:.6e} "
            f"param_rms={p_rms:.6e} param_max_abs={p_max_abs:.6e}",
            file=sys.stderr,
            flush=True,
        )

# ------------------------------------------------------------------------------
# TRAINER CLASS
# ------------------------------------------------------------------------------
class Trainer(BaseTrainer):
    def __init__(self, job_config: JobConfig):
        for mname in job_config.job.experimental_modules:
            logger.info(f"Loading experimental module '{mname}'")
            experiments.import_experimental_module(mname)
        
        compat.enable_compat_mode(job_config)
        ensure_fp32_master_params(job_config)

        self.checkpoint_patcher = None
        if job_config.ema_checkpoint.enable_checkpoint:
            self.checkpoint_patcher = CheckpointManagerPatching(
                CheckpointManagerWithEMAWeight
            )
            self.checkpoint_patcher.__enter__()

        torch._logging.set_logs(output_code=job_config.profiling.capture_compiled_kernels)

        # ----------------------------------------------------------------------
        # STRATEGY: Patch the 'TrainSpec' Registry for CCE
        # ----------------------------------------------------------------------
        model_name = job_config.model.name
        
        registry = None
        registry_name = None
        for key in dir(spec_module):
            val = getattr(spec_module, key)
            if isinstance(val, dict) and model_name in val:
                registry = val
                registry_name = key
                break
        
        def _cce_factory_wrapper(original_cls, *args, **kwargs):
            cce_backend = _load_cce_backend(job_config)
            use_cce = cce_backend is not None
            if use_cce:
                logger.info(
                    "⚡ Intercepted '%s' creation for CCE patching...",
                    model_name,
                )
            else:
                logger.info(
                    "Building '%s' with the ordinary output head and external loss",
                    model_name,
                )
            model = original_cls(*args, **kwargs)
            if use_cce:
                try:
                    model = cce_backend.apply_cce_backend_patch(model, job_config)
                    logger.info("Applied internal-loss CCE backend patch.")
                except Exception as e:
                    logger.error(f"❌ Failed to apply CCE patch: {e}")
                    raise e
            return model

        if registry is not None:
            logger.info(f"🔍 Found spec registry: 'torchtitan.protocols.train_spec.{registry_name}'")
            target_spec = registry[model_name]
            original_model_cls = target_spec.model_cls
            
            def factory_wrapper(*args, **kwargs):
                return _cce_factory_wrapper(original_model_cls, *args, **kwargs)
            
            target_spec.model_cls = factory_wrapper
            try:
                super().__init__(job_config)
            finally:
                target_spec.model_cls = original_model_cls
        else:
            super().__init__(job_config)

        # ----------------------------------------------------------------------

        # Local-B4 8B training leaves too little HBM for DCP to lazily create a
        # second world-size NCCL gather communicator at the first checkpoint.
        # The authenticated production recipe opts into a dedicated CPU/Gloo
        # group for synchronous checkpoint planning and resume metadata.
        install_sync_dcp_cpu_process_group(self.checkpointer, logger)

        if job_config.optimizer.name == "AdamWBF16SR":
            from .optimizer_sr_state import (
                register_with_checkpointer as register_optimizer_sr_with_checkpointer,
            )

            register_optimizer_sr_with_checkpointer(
                self.checkpointer,
                self.optimizers,
                logger,
            )

        # localCTA-v4 gradient SR must not depend on the extension's legacy
        # process-global invocation atomic.  Install stable per-layer/op CUDA
        # states before checkpoint loading (which happens in BaseTrainer.train)
        # and before optional torch.compile wraps the converted modules.
        from .quantization.localcta_sr_state import (
            build_localcta_sr_state_for_trainer,
            register_with_checkpointer,
        )

        self.localcta_sr_state = build_localcta_sr_state_for_trainer(
            self.model_parts,
            device=self.device,
            training_steps=job_config.training.steps,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
        )
        if self.localcta_sr_state is not None:
            register_with_checkpointer(
                self.checkpointer, self.localcta_sr_state, logger
            )
            logger.info(
                "Installed checkpointed localCTA-v4 SR state for %d logical "
                "backward producers (%d reservations/producer headroom).",
                len(self.localcta_sr_state.logical_keys),
                self.localcta_sr_state.reservations_per_slot,
            )
        # The CUDA batch generator owns a one-batch device-copy lookahead. Point
        # checkpointing at an adapter that resumes before that pending batch.
        self._prefetch_checkpoint_dataloader = install_checkpoint_aligned_dataloader(
            self.checkpointer, self.dataloader
        )
        # The fused output-head row-SR kernel historically advanced a
        # process-global CUDA counter, which cannot survive a checkpoint
        # restart.  Install the opt-in rank-namespaced state before checkpoint
        # loading (BaseTrainer.train) and before optional torch.compile.
        self.output_head_sr_state = None
        if cce_path_handles_loss(job_config):
            from .cce.head_sr_state import (
                build_output_head_sr_state_for_trainer,
                register_with_checkpointer as register_output_head_sr_state,
            )

            self.output_head_sr_state = build_output_head_sr_state_for_trainer(
                device=self.device,
                training_steps=job_config.training.steps,
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                step_getter=lambda: self.step,
            )
        if self.output_head_sr_state is not None:
            register_output_head_sr_state(
                self.checkpointer, self.output_head_sr_state, logger
            )
            logger.info(
                "Installed checkpointed output-head SR ABI v1 for rank %d/%d "
                "(%d reservations of headroom).",
                self.output_head_sr_state.rank,
                self.output_head_sr_state.world_size,
                self.output_head_sr_state.reservations_per_rank,
            )
        
        # Disable external loss if CCE is active
        if cce_path_handles_loss(job_config):
            logger.info("🛑 Disabling external loss function (Model handles it internally)")
            self.loss_fn = lambda x, y: x 

        # Manual Torch.Compile
        should_compile = getattr(job_config.training, "compile", False)
        if should_compile:
            logger.info("🔥 Compiling model parts with torch.compile (Post-FSDP)...")
            try:
                if hasattr(self, 'model_parts'):
                    self.model_parts = [torch.compile(m) for m in self.model_parts]
                    logger.info("✅ Model parts compiled successfully.")
                elif hasattr(self, 'model'):
                    self.model = torch.compile(self.model)
                    logger.info("✅ Model compiled successfully.")
            except Exception as e:
                logger.warning(f"⚠️ torch.compile failed: {e}")

        # (Quartet prefetch chain removed — TE handles AllGather internally)

        compat.decorate_trainer_methods(self)
        self.metrics_processor.lr_schedulers = self.lr_schedulers

        self.signal_handler = SignalHandler(logger=logger)
        if not torch.distributed.is_initialized():
            raise RuntimeError("Distributed is not initialized")
        self.signal_handler.register_handlers()
        self._signal_consensus_tensor = None
        if os.environ.get("LBT_DISTRIBUTED_SIGNAL_CONSENSUS") == "1":
            self._signal_consensus_tensor = torch.zeros(
                (), dtype=torch.int32, device=self.device
            )

        self.nan_loss_detector = NanLossDetectingHandler()
        self.nan_loss_detector.setLevel(logging.INFO)
        logger.addHandler(self.nan_loss_detector)

        assert self.tokenizer.get_vocab_size() <= self.model_args.vocab_size, (
            "Tokenizer vocab size must be <= model args vocab size"
        )

        self.cutoff_step = job_config.job.steps

        if is_remote_synchronizer_active(self.job_config):
            self.remote_synchronizer = RemoteSynchronizer(
                self.job_config.job.dump_folder,
                self.job_config.job.remote_folder,
                **_remote_synchronizer_kwargs_from_env(),
            )
            self.remote_synchronizer.start()

        if self.job_config.checkpoint.save_aligned_checkpoint:
            logger.info("💾 Saving Aligned Checkpoint (Step 0) for Parity Debugging...")
            self.save_checkpoint(0)

    def end_training_early(self):
        self.job_config.training.steps = self.step
        self.job_config.checkpoint.last_save_in_hf = False
        self.job_config.checkpoint.last_save_model_only = False
        self.checkpointer.last_save_in_hf = False
        self.checkpointer.last_save_model_only = False

    def _end_training_after_signal(self):
        # BaseTrainer runs scheduled validation after its per-step checkpoint.
        # A forced stop must preserve the bounded grace budget for final DCP and
        # mandatory remote synchronization. Cutoff canaries still validate.
        self.job_config.validation.enable = False
        self.end_training_early()

    def forward_backward_step(self, input_dict, labels):
        """
        Custom override:
        If an internal-loss CCE backend is enabled, we unpack data our way and call the model directly.
        If CCE is disabled, we let standard Titan handle it.
        """
        # 1. Fallback for standard training
        if not cce_path_handles_loss(self.job_config):
            loss = super().forward_backward_step(input_dict, labels)
            _fail_on_nonfinite_metric(loss, "loss")
            return loss
        

        # -------------------------------------------------------
        # CCE CUSTOM LOGIC
        # -------------------------------------------------------
        
        # 2. Aggressive Unpacking (Your custom logic)
        # Note: Titan passes 'input_dict' and 'labels' from the data loader
        input_ids = input_dict.get('input_ids') or input_dict.get('tokens') or input_dict.get('input')
        
        # Deep extraction for nested dicts (if necessary)
        if isinstance(input_ids, dict):
            for key in ['input_ids', 'tokens', 'text', 'content']:
                if key in input_ids:
                    input_ids = input_ids[key]
                    break
        
        # Final sanity check
        if not isinstance(input_ids, torch.Tensor):
            # If input_dict['input'] is the tensor (Standard Titan format)
            input_ids = input_dict.get('input') 

        # 3. Get Model (Titan stores it in model_parts list)
        model = self.model_parts[0]

        _lbt_step_trace(self.step, "cce_model_forward_start")
        loss = model(input_ids, labels=labels)
        _fail_on_nonfinite_metric(loss, "loss")
        _lbt_step_trace(self.step, "cce_model_forward_done")
        # the loss so gradients don't explode during accumulation.
        scaled_loss = loss / self.gradient_accumulation_steps
        _lbt_step_trace(self.step, "cce_backward_start")
        scaled_loss.backward()
        _lbt_step_trace(self.step, "cce_backward_done")
        if _use_debug_top_param_grads():
            _dump_top_param_grads(self.model_parts)

        # 6. Return Scaled Loss
        # Titan's train_step expects the scaled loss to track metrics correctly.
        return scaled_loss.detach()

    def train_step(self, *args, **kwargs):
        prev_trace_step = os.environ.get("LBT_TRACE_ACTIVE_STEP")
        os.environ["LBT_TRACE_ACTIVE_STEP"] = str(self.step)
        try:
            hsdp_reduce_scatter_accumulation = (
                os.environ.get(_HSDP_REDUCE_SCATTER_ACCUMULATION_ENV, "0")
                == "1"
                and self.gradient_accumulation_steps > 1
            )
            if (
                hsdp_reduce_scatter_accumulation
                and not self.parallel_dims.dp_replicate_enabled
            ):
                raise RuntimeError(
                    f"{_HSDP_REDUCE_SCATTER_ACCUMULATION_ENV}=1 requires "
                    "HSDP with data-parallel replication enabled"
                )

            with _hsdp_reduce_scatter_accumulation(
                hsdp_reduce_scatter_accumulation
            ):
                if (
                    _use_debug_nonfinite_grad_scan()
                    or _use_debug_top_param_grads()
                    or os.environ.get("LBT_FAIL_ON_NONFINITE_METRICS", "0") == "1"
                ):
                    orig_clip_grad_norm = dist_utils.clip_grad_norm_
                    titan_dist_utils = getattr(
                        titan_train_module, "dist_utils", None
                    )
                    orig_titan_clip_grad_norm = (
                        getattr(titan_dist_utils, "clip_grad_norm_", None)
                        if titan_dist_utils is not None
                        else None
                    )

                    def _wrapped_clip_grad_norm(
                        params, max_norm, *clip_args, **clip_kwargs
                    ):
                        if _use_debug_nonfinite_grad_scan():
                            _scan_nonfinite_grads(self.model_parts)
                        if _use_debug_top_param_grads():
                            _dump_top_param_grads(self.model_parts)
                        grad_norm = orig_clip_grad_norm(
                            params, max_norm, *clip_args, **clip_kwargs
                        )
                        _fail_on_nonfinite_metric(grad_norm, "gradient norm")
                        return grad_norm

                    dist_utils.clip_grad_norm_ = _wrapped_clip_grad_norm
                    if titan_dist_utils is not None:
                        titan_dist_utils.clip_grad_norm_ = (
                            _wrapped_clip_grad_norm
                        )
                    try:
                        super().train_step(*args, **kwargs)
                    finally:
                        dist_utils.clip_grad_norm_ = orig_clip_grad_norm
                        if (
                            titan_dist_utils is not None
                            and orig_titan_clip_grad_norm is not None
                        ):
                            titan_dist_utils.clip_grad_norm_ = (
                                orig_titan_clip_grad_norm
                            )
                else:
                    super().train_step(*args, **kwargs)
        finally:
            if prev_trace_step is None:
                os.environ.pop("LBT_TRACE_ACTIVE_STEP", None)
            else:
                os.environ["LBT_TRACE_ACTIVE_STEP"] = prev_trace_step

        if distributed_exit_requested(
            self.signal_handler, self._signal_consensus_tensor
        ):
            logger.info("Exiting due to signal.")
            self._end_training_after_signal()

        if self.nan_loss_detector.nonfinite_seen():
            logger.info("Exiting because non-finite loss/gradient metric detected.")
            self.end_training_early()

        if self.cutoff_step != -1 and self.step >= self.cutoff_step:
            logger.info("Reached cutoff step, exiting.")
            self.end_training_early()

    def batch_generator(self, data_iterable):
        device_type = dist_utils.device_type
        data_iterator = iter(data_iterable)
        device = (
            self.device
            if isinstance(self.device, torch.device)
            else torch.device(device_type)
        )
        hash_batches = batch_fingerprints_enabled()
        last_yield_step: int | None = None
        microbatch = -1

        def _load_cpu_batch(*, checkpoint_lookahead: bool = False):
            data_load_start = time.perf_counter()
            try:
                if checkpoint_lookahead:
                    batch = self._prefetch_checkpoint_dataloader.next_for_prefetch(
                        data_iterator
                    )
                else:
                    batch = next(data_iterator)
            except StopIteration as ex:
                raise DataloaderExhaustedError() from ex

            input_dict, labels = batch
            data_loading_time = time.perf_counter() - data_load_start
            fingerprint = (
                fingerprint_batch(input_dict, labels) if hash_batches else None
            )
            return (
                _pin_memory_tree(input_dict),
                _pin_memory_tree(labels),
                data_loading_time,
                fingerprint,
            )

        def _record_actual_yield(
            labels: torch.Tensor,
            data_loading_time: float,
            fingerprint: BatchFingerprint | None,
        ) -> None:
            nonlocal last_yield_step, microbatch

            ntokens_batch = labels.numel()
            self.ntokens_seen += ntokens_batch
            self.metrics_processor.ntokens_since_last_log += ntokens_batch
            self.metrics_processor.data_loading_times.append(data_loading_time)

            if fingerprint is None:
                return
            yield_step = int(self.step)
            if last_yield_step != yield_step:
                last_yield_step = yield_step
                microbatch = 0
            else:
                microbatch += 1
            emit_batch_fingerprint(fingerprint, step=yield_step, microbatch=microbatch)

        if device.type != "cuda":
            while True:
                input_dict, labels, data_loading_time, fingerprint = _load_cpu_batch()
                input_dict = _to_device_tree(input_dict, device)
                labels = _to_device_tree(labels, device)
                _record_actual_yield(labels, data_loading_time, fingerprint)
                yield input_dict, labels

        copy_stream = torch.cuda.Stream(device=device)

        def _prefetch_to_device(
            cpu_input_dict,
            cpu_labels,
            data_loading_time,
            fingerprint,
        ):
            with torch.cuda.stream(copy_stream):
                gpu_input_dict = _to_device_tree(cpu_input_dict, device)
                gpu_labels = _to_device_tree(cpu_labels, device)
                ready_event = torch.cuda.Event()
                ready_event.record(copy_stream)
            return (
                gpu_input_dict,
                gpu_labels,
                ready_event,
                data_loading_time,
                fingerprint,
            )

        (
            next_input_dict,
            next_labels,
            next_ready_event,
            next_data_loading_time,
            next_fingerprint,
        ) = _prefetch_to_device(*_load_cpu_batch(checkpoint_lookahead=True))
        has_next_batch = True

        while has_next_batch:
            self._prefetch_checkpoint_dataloader.mark_prefetched_batch_current()
            current_stream = torch.cuda.current_stream(device=device)
            current_stream.wait_event(next_ready_event)
            input_dict = next_input_dict
            labels = next_labels
            data_loading_time = next_data_loading_time
            fingerprint = next_fingerprint
            _record_stream_tree(input_dict, current_stream)
            _record_stream_tree(labels, current_stream)

            try:
                (
                    next_input_dict,
                    next_labels,
                    next_ready_event,
                    next_data_loading_time,
                    next_fingerprint,
                ) = _prefetch_to_device(
                    *_load_cpu_batch(checkpoint_lookahead=True)
                )
            except DataloaderExhaustedError:
                has_next_batch = False

            _record_actual_yield(labels, data_loading_time, fingerprint)
            yield input_dict, labels
            
    def close(self):
        cleanup_errors: list[Exception] = []
        if self.checkpoint_patcher is not None:
            try:
                self.checkpoint_patcher.__exit__(None, None, None)
            except Exception as error:
                cleanup_errors.append(error)
            finally:
                self.checkpoint_patcher = None
        try:
            super().close()
        except Exception as error:
            cleanup_errors.append(error)
        # The final remote checkpoint/log sync is mandatory.  Always attempt
        # it even when an earlier cleanup phase failed; train.py may use
        # os._exit after this method returns and would otherwise kill the sync
        # worker before it can publish its terminal status.
        if hasattr(self, "remote_synchronizer"):
            try:
                self.remote_synchronizer.stop()
            except Exception as error:
                cleanup_errors.append(error)
        if len(cleanup_errors) == 1:
            raise cleanup_errors[0]
        if cleanup_errors:
            raise ExceptionGroup("trainer cleanup failed", cleanup_errors)

    def state_dict(self) -> dict[str, Any]:
        sd = super().state_dict()
        compat_str = self.job_config.checkpoint.compatibility
        if compat_str and int(compat_str.strip("lbt-v")) <= 1:
            _ = sd.pop("ntokens_seen")
        return sd

    def load_state_dict(self, state_dict: dict[str, Any]):
        compat_str = self.job_config.checkpoint.compatibility
        if compat_str and int(compat_str.strip("lbt-v")) <= 1:
            state_dict["ntokens_seen"] = 0
        super().load_state_dict(state_dict)
