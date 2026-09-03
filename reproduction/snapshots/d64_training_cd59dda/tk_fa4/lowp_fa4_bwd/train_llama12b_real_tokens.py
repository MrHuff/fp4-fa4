#!/usr/bin/env python3
"""Train BF16, MXFP4-PV, and FP8-PV models on real text.

This is a local, single-GPU next-token training probe.  Unlike the synthetic
random-token benchmark, it uses disjoint corpus train/validation splits and
reports FP32 cross-entropy on held-out token sequences throughout training.
Selected routes start from identical weights and consume identical batches.
Multi-route runs use a rotating execution order; a single selected route lets
full-depth models run in memory-safe, matched independent processes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import pyarrow as pa
import pyarrow.ipc as pa_ipc
from tokenizers import Tokenizer

from tk_fa4.lowp_fa4_bwd.benchmark_llama12b_e2e import (
    Config,
    DEFAULT_MODEL_PRESET,
    Llama12B,
    MODEL_PRESETS,
    activate_model_forward_route,
    config_from_model_preset,
    _useful_flops,
)
from tk_fa4.lowp_fa4_bwd.backward_policy import (
    resolve_backward_exp2_policy,
)
from tk_fa4.lowp_fa4_bwd.backward_contract import (
    require_matching_backward_contracts,
)
from tk_fa4.lowp_fa4_bwd.compare_llama12b_mx_fp8pv import (
    ROUTE_NAMES,
    _make_runtime,
    _mx_probability_replay_provenance,
    _optimizer,
    _projection_extension_identity,
    _require_memory_safe_matched_replicas,
    _required_projection_symbols,
    _share_matched_backward_runner,
    _timed_forward_dispatch_contracts,
)
from tk_fa4.lowp_fa4_bwd.training_drift_gate import RouteLossDriftGate
from tk_fa4.lowp_fa4_bwd.training_telemetry import (
    forward_diagnostic_tensor_statistics,
    mark_matched_round_timing_eligibility,
    select_timing_records,
)


DEFAULT_CORPUS = Path(
    "/workspace/codebases/cce_fp4/low-bits-training/torchtitan_submodule/"
    "tests/assets/c4_test/data.json"
)
DEFAULT_TOKENIZER = Path(
    "/workspace/codebases/poly_stuff/low-precision-functions/"
    "low-bits-training/assets/hf/Meta-Llama-3.1-8B/tokenizer.json"
)
LLAMA_BOS = 128000
LLAMA_EOS = 128001
LLAMA3_ROPE_THETA = 500_000.0
LLAMA32_ROPE_FACTOR = 32.0
LLAMA3_LOW_FREQ_FACTOR = 1.0
LLAMA3_HIGH_FREQ_FACTOR = 4.0
LLAMA3_ORIGINAL_CONTEXT = 8192
BACKWARD_EXTENSION_MODULE = "tk_fa4._C_b300_lowp_bwd"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    """Return a durable identity for a file consumed by this process."""
    resolved = path.resolve()
    if not resolved.is_file():
        raise RuntimeError(f"file does not exist: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _file_sha256(resolved),
    }


def _extension_identity(path: Path, module: str) -> dict[str, Any]:
    """Return a durable identity for a runtime extension selected by CLI."""
    return {
        "module": module,
        **_file_identity(path),
    }


def _git_identity(repo_root: Path) -> dict[str, Any]:
    """Record the commit and tracked diff that define the Python runtime."""

    def run(*arguments: str, text: bool = True) -> str | bytes:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *arguments],
            check=False,
            capture_output=True,
            text=text,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"git command failed: {' '.join(arguments)}")
        return completed.stdout

    try:
        head = str(run("rev-parse", "HEAD")).strip()
        branch = str(run("rev-parse", "--abbrev-ref", "HEAD")).strip()
        diff = run(
            "diff",
            "--binary",
            "--no-ext-diff",
            "HEAD",
            "--",
            text=False,
        )
        assert isinstance(diff, bytes)
    except (OSError, RuntimeError) as error:
        return {
            "available": False,
            "error_type": type(error).__name__,
        }
    return {
        "available": True,
        "repo_root": str(repo_root.resolve()),
        "head": head,
        "branch": branch,
        "tracked_dirty": bool(diff),
        "tracked_diff_bytes": len(diff),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def _load_corpus(path: Path) -> tuple[list[str], dict[str, Any]]:
    documents: list[str] = []
    source_rows = 0
    empty_rows = 0
    duplicate_rows = 0
    seen: set[bytes] = set()

    def append(text: str) -> None:
        nonlocal empty_rows, duplicate_rows
        if not text.strip():
            empty_rows += 1
            return
        fingerprint = hashlib.sha256(text.encode()).digest()
        if fingerprint in seen:
            duplicate_rows += 1
            return
        seen.add(fingerprint)
        documents.append(text)

    if path.suffix == ".arrow":
        with pa.memory_map(str(path), "r") as source:
            reader = pa_ipc.open_stream(source)
            if "text" not in reader.schema.names:
                raise RuntimeError(f"{path} has no text column")
            text_index = reader.schema.get_field_index("text")
            for batch in reader:
                source_rows += batch.num_rows
                for value in batch.column(text_index).to_pylist():
                    append("" if value is None else str(value))
        corpus_format = "Arrow IPC stream"
    else:
        with path.open() as stream:
            for line_number, line in enumerate(stream, start=1):
                source_rows += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RuntimeError(
                        f"invalid JSON on {path}:{line_number}"
                    ) from error
                append(str(row.get("text", "")))
        corpus_format = "JSON Lines"
    if len(documents) < 2:
        raise RuntimeError(f"{path} contains fewer than two nonempty documents")
    return documents, {
        "format": corpus_format,
        "source_rows": source_rows,
        "empty_rows_removed": empty_rows,
        "exact_duplicate_rows_removed": duplicate_rows,
        "unique_documents": len(documents),
    }


def _make_llama3_rope(
    sequence: int,
    depth: int,
    theta: float = LLAMA3_ROPE_THETA,
    factor: float = LLAMA32_ROPE_FACTOR,
) -> tuple[torch.Tensor, torch.Tensor]:
    if depth not in (64, 128):
        raise ValueError("RoPE depth must be 64 or 128")
    pair_count = depth // 2
    positions = torch.arange(sequence, device="cuda", dtype=torch.float32)
    frequencies = 1.0 / (
        theta
        ** (
            torch.arange(pair_count, device="cuda", dtype=torch.float32)
            / pair_count
        )
    )
    wavelengths = 2.0 * math.pi / frequencies
    low_frequency_wavelength = (
        LLAMA3_ORIGINAL_CONTEXT / LLAMA3_LOW_FREQ_FACTOR
    )
    high_frequency_wavelength = (
        LLAMA3_ORIGINAL_CONTEXT / LLAMA3_HIGH_FREQ_FACTOR
    )
    scaled_frequencies = torch.where(
        wavelengths > low_frequency_wavelength,
        frequencies / factor,
        frequencies,
    )
    smooth = (
        LLAMA3_ORIGINAL_CONTEXT / wavelengths - LLAMA3_LOW_FREQ_FACTOR
    ) / (LLAMA3_HIGH_FREQ_FACTOR - LLAMA3_LOW_FREQ_FACTOR)
    smoothed_frequencies = (
        (1.0 - smooth) * scaled_frequencies / factor
        + smooth * scaled_frequencies
    )
    medium = ~(
        (wavelengths < high_frequency_wavelength)
        | (wavelengths > low_frequency_wavelength)
    )
    frequencies = torch.where(
        medium, smoothed_frequencies, scaled_frequencies
    )
    angles = positions[:, None] * frequencies[None, :]
    return angles.cos()[None].bfloat16(), angles.sin()[None].bfloat16()


def _backward_extension_identity(
    expected_path: Path | None,
) -> dict[str, str]:
    module = sys.modules.get(BACKWARD_EXTENSION_MODULE)
    if module is None or getattr(module, "__file__", None) is None:
        raise RuntimeError(
            f"{BACKWARD_EXTENSION_MODULE} is not loaded from a filesystem path"
        )
    actual_path = Path(module.__file__).resolve()
    if expected_path is not None and actual_path != expected_path.resolve():
        raise RuntimeError(
            f"loaded backward extension {actual_path}, expected "
            f"{expected_path.resolve()}"
        )
    return {
        "module": BACKWARD_EXTENSION_MODULE,
        "path": str(actual_path),
        "sha256": _file_sha256(actual_path),
    }


def _comparisons_against_bf16(
    routes: dict[str, Any],
) -> tuple[str | None, dict[str, dict[str, float]]]:
    """Compare selected low-precision routes when BF16 is in this process."""
    reference_name = "bf16_cute"
    if reference_name not in routes:
        return None, {}
    reference_timing = routes[reference_name]["timing"]
    reference_validation = routes[reference_name]["validation"]["final_loss"]
    comparisons: dict[str, dict[str, float]] = {}
    for name, route in routes.items():
        if name == reference_name:
            continue
        final_validation = route["validation"]["final_loss"]
        comparisons[name] = {
            "speedup_over_bf16": (
                reference_timing["step_ms"] / route["timing"]["step_ms"]
            ),
            "final_validation_loss_delta": (
                final_validation - reference_validation
            ),
            "final_validation_loss_ratio": (
                final_validation / reference_validation
            ),
        }
    return reference_name, comparisons


@torch.no_grad()
def _state_dict_probe(state: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Fingerprint sparse values plus full tensor metadata without a CPU clone."""
    digest = hashlib.sha256()
    sampled_values = 0
    for name, tensor in sorted(state.items()):
        value = tensor.detach()
        metadata = {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "numel": value.numel(),
        }
        digest.update(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
        )
        flat = value.reshape(-1)
        if not flat.numel():
            continue
        positions = sorted(
            {
                0,
                flat.numel() // 3,
                (2 * flat.numel()) // 3,
                flat.numel() - 1,
            }
        )
        indices = torch.tensor(positions, device=flat.device)
        sample = flat.index_select(0, indices).contiguous().view(torch.uint8)
        digest.update(sample.cpu().numpy().tobytes())
        sampled_values += len(positions)
    return {
        "schema": "state_dict_sparse_probe_v1",
        "tensor_count": len(state),
        "sampled_values": sampled_values,
        "sha256": digest.hexdigest(),
    }


def _cuda_hardware_identity() -> dict[str, Any]:
    """Record the physical CUDA device used by an isolated training arm."""
    properties = torch.cuda.get_device_properties(0)
    return {
        "schema": "cuda_hardware_identity_v1",
        "visible_device_count": torch.cuda.device_count(),
        "logical_device_index": 0,
        "name": properties.name,
        "uuid": str(properties.uuid),
        "compute_capability": [properties.major, properties.minor],
        "total_memory_bytes": properties.total_memory,
        "multiprocessor_count": properties.multi_processor_count,
        "l2_cache_bytes": properties.L2_cache_size,
        "pci_domain_id": properties.pci_domain_id,
        "pci_bus_id": properties.pci_bus_id,
        "pci_device_id": properties.pci_device_id,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }


def _token_batches(
    documents: list[str],
    document_indices: list[int],
    tokenizer: Tokenizer,
    *,
    batch_count: int,
    sequence: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Build deterministic non-overlapping next-token batches from documents."""
    required = batch_count * (sequence + 1)
    stream: list[int] = []
    documents_consumed = 0
    for index in document_indices:
        encoded = tokenizer.encode(
            documents[index], add_special_tokens=False
        ).ids
        stream.append(LLAMA_BOS)
        stream.extend(int(token) for token in encoded)
        stream.append(LLAMA_EOS)
        documents_consumed += 1
        if len(stream) >= required:
            break
    if len(stream) < required:
        raise RuntimeError(
            f"split supplied {len(stream)} tokens ({len(stream) // (sequence + 1)} "
            f"complete batches); need {required}"
        )

    chunks = torch.tensor(stream[:required], dtype=torch.int64).reshape(
        batch_count, sequence + 1
    )
    token_hash = hashlib.sha256(chunks.numpy().tobytes()).hexdigest()
    tokens = chunks[:, :-1].contiguous().cuda()
    targets = chunks[:, 1:].contiguous().cuda()
    return tokens, targets, {
        "batches": batch_count,
        "sequence": sequence,
        "tokens_with_boundary": required,
        "packed_tokens_before_truncation": len(stream),
        "documents_consumed": documents_consumed,
        "document_order_sha256": hashlib.sha256(
            json.dumps(document_indices, separators=(",", ":")).encode()
        ).hexdigest(),
        "sha256": token_hash,
    }


def _cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # BF16 cross entropy quantizes losses near log(vocab) in increments of
    # roughly 0.0625.  Cast logits so the training signal and reported loss are
    # both FP32 for every route.
    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="mean",
    )


@torch.no_grad()
def _named_tensor_health(
    tensors: list[tuple[str, torch.Tensor]],
    *,
    top_k: int = 8,
) -> dict[str, Any]:
    nonfinite: list[dict[str, Any]] = []
    maxima: list[tuple[float, str]] = []
    for name, tensor in tensors:
        value = tensor.detach()
        finite = torch.isfinite(value)
        if not bool(finite.all()):
            nonfinite.append(
                {
                    "name": name,
                    "nan": int(torch.isnan(value).sum()),
                    "positive_inf": int(torch.isposinf(value).sum()),
                    "negative_inf": int(torch.isneginf(value).sum()),
                    "elements": value.numel(),
                }
            )
            continue
        maximum = float(value.abs().max()) if value.numel() else 0.0
        maxima.append((maximum, name))
    maxima.sort(reverse=True)
    return {
        "all_finite": not nonfinite,
        "nonfinite": nonfinite,
        "largest_max_abs": [
            {"name": name, "max_abs": maximum}
            for maximum, name in maxima[:top_k]
        ],
        "tensor_count": len(tensors),
    }


@torch.no_grad()
def _lowp_forward_diagnostics(
    captures: list[dict[str, torch.Tensor | None]],
) -> dict[str, Any]:
    """Summarize projection scales and attention values for every layer."""
    return {
        "layers": [
            {
                "layer": layer_index,
                "tensors": {
                    name: forward_diagnostic_tensor_statistics(name, tensor)
                    for name, tensor in capture.items()
                },
            }
            for layer_index, capture in enumerate(captures)
        ],
        "layer_count": len(captures),
    }


def _begin_lowp_forward_capture(
    model: torch.nn.Module,
) -> tuple[Any | None, list[dict[str, torch.Tensor | None]]]:
    """Install a temporary capture sink on a model's shared lowp runtime."""
    runtimes = {
        id(runtime): runtime
        for layer in getattr(model, "layers", ())
        if (runtime := getattr(getattr(layer, "attention", None), "runtime", None))
        is not None
    }
    if not runtimes:
        return None, []
    if len(runtimes) != 1:
        raise RuntimeError("expected every lowp layer to share one runtime")
    runtime = next(iter(runtimes.values()))
    if runtime.forward_diagnostic_sink is not None:
        raise RuntimeError("lowp forward diagnostic capture is already active")
    captures: list[dict[str, torch.Tensor | None]] = []
    runtime.forward_diagnostic_sink = captures
    return runtime, captures


def _atomic_write_json(path: Path, value: Any) -> None:
    """Atomically replace a bounded progress snapshot."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _model_forward_trace(
    model: torch.nn.Module,
) -> tuple[list[Any], list[dict[str, Any]]]:
    trace: list[dict[str, Any]] = []
    handles = []
    wanted = re.compile(
        r"^layers\.\d+(?:\.(?:attention_norm|attention|ffn_norm|mlp))?$"
    )

    def make_hook(name: str) -> Any:
        def hook(_: Any, __: Any, output: Any) -> None:
            if not torch.is_tensor(output):
                return
            health = _named_tensor_health([(name, output)], top_k=1)
            trace.append(
                {
                    "kind": "forward_output",
                    "name": name,
                    "all_finite": health["all_finite"],
                    "nonfinite": health["nonfinite"],
                    "max_abs": (
                        health["largest_max_abs"][0]["max_abs"]
                        if health["largest_max_abs"]
                        else None
                    ),
                }
            )
            if output.requires_grad:
                def gradient_hook(gradient: torch.Tensor) -> None:
                    gradient_health = _named_tensor_health(
                        [(name, gradient)], top_k=1
                    )
                    entry: dict[str, Any] = {
                        "kind": "backward_output_gradient",
                        "name": name,
                        "all_finite": gradient_health["all_finite"],
                        "nonfinite": gradient_health["nonfinite"],
                        "max_abs": (
                            gradient_health["largest_max_abs"][0]["max_abs"]
                            if gradient_health["largest_max_abs"]
                            else None
                        ),
                    }
                    if name.endswith(".attention"):
                        # dO is loss-scaled by 2^16 and published as fixed
                        # E4M3 x4. Values above this unscaled threshold clip.
                        threshold = 448.0 / (4.0 * 2.0**16)
                        entry["fixed_fp8_clip_threshold"] = threshold
                        entry["values_above_threshold"] = int(
                            (gradient.detach().abs() > threshold).sum()
                        )
                    trace.append(entry)

                output.register_hook(gradient_hook)

        return hook

    for name, module in model.named_modules():
        if wanted.match(name) or name == "final_norm":
            handles.append(module.register_forward_hook(make_hook(name)))
    return handles, trace


def _optimizer_tensor_health(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    tensors: list[tuple[str, torch.Tensor]] = []
    for parameter, state in optimizer.state.items():
        parameter_name = names.get(id(parameter), "<unknown>")
        for state_name, value in state.items():
            if torch.is_tensor(value):
                tensors.append((f"{parameter_name}.{state_name}", value))
    return _named_tensor_health(tensors)


@torch.no_grad()
def _refresh_model_qk_scales(model: torch.nn.Module) -> int:
    """Refresh each low-precision layer's paired-head Q/K encode policy."""
    refreshed = 0
    for layer in getattr(model, "layers", ()):
        attention = getattr(layer, "attention", None)
        refresh = getattr(attention, "refresh_qk_quant_scales", None)
        if refresh is not None:
            refresh()
            refreshed += 1
    return refreshed


def _step(
    name: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    *,
    round_index: int,
    batch_index: int,
    execution_position: int,
    warmup: bool,
    diagnose: bool = False,
    refresh_qk_scales: bool = False,
    gradient_clip_norm: float | None = None,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] | None = None
    handles: list[Any] = []
    forward_trace: list[dict[str, Any]] = []
    diagnostic_runtime = None
    lowp_forward_captures: list[
        dict[str, torch.Tensor | None]
    ] = []
    if diagnose:
        diagnostics = {
            "parameters_before": _named_tensor_health(
                list(model.named_parameters())
            )
        }
        handles, forward_trace = _model_forward_trace(model)
        diagnostic_runtime, lowp_forward_captures = (
            _begin_lowp_forward_capture(model)
        )
    # These are host-side setup operations. Keep them ahead of the first CUDA
    # event so the reported forward interval cannot include a route-switch or
    # zero-grad launch gap.
    activate_model_forward_route(model)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    forward_done = torch.cuda.Event(enable_timing=True)
    backward_done = torch.cuda.Event(enable_timing=True)
    gradient_clip_done = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record()
    try:
        logits = model(tokens)
        if diagnostics is not None and diagnostic_runtime is not None:
            diagnostics["lowp_forward"] = _lowp_forward_diagnostics(
                lowp_forward_captures
            )
    finally:
        if diagnostic_runtime is not None:
            diagnostic_runtime.forward_diagnostic_sink = None
        for handle in handles:
            handle.remove()
    loss = _cross_entropy(logits, targets)
    forward_done.record()
    failure_stage: str | None = None
    loss_finite = True
    if diagnostics is not None:
        # Diagnostic steps are excluded from throughput summaries and may
        # synchronize here to stop before propagating a non-finite loss.
        value = float(loss.detach())
        loss_finite = math.isfinite(value)
        diagnostics["forward_trace"] = forward_trace
        diagnostics["logits"] = _named_tensor_health(
            [("logits", logits)], top_k=1
        )
    if loss_finite:
        loss.backward()
    else:
        failure_stage = "forward_loss"
    backward_done.record()
    if diagnostics is not None and failure_stage is None:
        gradient_health = _named_tensor_health(
            [
                (name, parameter.grad)
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
            ]
        )
        diagnostics["gradients"] = gradient_health
        if not gradient_health["all_finite"]:
            failure_stage = "gradients"
    preclip_gradient_norm: torch.Tensor | None = None
    gradient_clip_error: str | None = None
    if failure_stage is None and gradient_clip_norm is not None:
        try:
            preclip_gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=gradient_clip_norm,
                norm_type=2.0,
                error_if_nonfinite=True,
                foreach=True,
            )
        except RuntimeError as error:
            if "non-finite" not in str(error):
                raise
            failure_stage = "gradients"
            gradient_clip_error = str(error)
    gradient_clip_done.record()
    if failure_stage is None:
        optimizer.step()
        if refresh_qk_scales:
            _refresh_model_qk_scales(model)
    if diagnostics is not None and failure_stage is None:
        parameter_health = _named_tensor_health(
            list(model.named_parameters())
        )
        optimizer_health = _optimizer_tensor_health(model, optimizer)
        diagnostics["parameters_after"] = parameter_health
        diagnostics["optimizer_after"] = optimizer_health
        if not parameter_health["all_finite"]:
            failure_stage = "parameters_after_optimizer"
        elif not optimizer_health["all_finite"]:
            failure_stage = "optimizer_state"
    end.record()
    end.synchronize()
    if diagnostics is None:
        # Keep the ordinary training queue continuous from forward through
        # backward and AdamW. Tensor-to-host loss inspection here is free of
        # the timed CUDA interval and cannot inflate backward launch latency.
        value = float(loss.detach())
        loss_finite = math.isfinite(value)
        if not loss_finite and failure_stage is None:
            failure_stage = "forward_loss_detected_after_step"
    preclip_gradient_norm_value = (
        float(preclip_gradient_norm)
        if preclip_gradient_norm is not None
        else None
    )
    gradient_was_clipped = (
        preclip_gradient_norm_value is not None
        and gradient_clip_norm is not None
        and preclip_gradient_norm_value > gradient_clip_norm
    )
    result = {
        "route": name,
        "round": round_index,
        "batch": batch_index,
        "execution_position": execution_position,
        "warmup": warmup,
        "diagnostic": diagnostics is not None,
        "loss": value,
        "finite": loss_finite and failure_stage is None,
        "failure_stage": failure_stage,
        "gradient_clip_norm": gradient_clip_norm,
        "gradient_preclip_total_norm": preclip_gradient_norm_value,
        "gradient_was_clipped": gradient_was_clipped,
        "gradient_clip_error": gradient_clip_error,
        "forward_ms": float(start.elapsed_time(forward_done)),
        "backward_ms": float(forward_done.elapsed_time(backward_done)),
        "gradient_clip_ms": float(
            backward_done.elapsed_time(gradient_clip_done)
        ),
        "optimizer_ms": float(gradient_clip_done.elapsed_time(end)),
        "step_ms": float(start.elapsed_time(end)),
        "wall_ms": (time.perf_counter() - wall_start) * 1000.0,
    }
    gradient_suffix = (
        f" preclip_norm={preclip_gradient_norm_value:.6f}"
        if preclip_gradient_norm_value is not None
        else ""
    )
    print(
        f"train round={round_index} batch={batch_index} "
        f"pos={execution_position} route={name} loss={value:.6f} "
        f"step={result['step_ms']:.3f} ms{gradient_suffix}",
        flush=True,
    )
    if diagnostics is not None:
        diagnostics["failure_stage"] = failure_stage
        result["diagnostics"] = diagnostics
        print(
            "numerical-diagnostic "
            + json.dumps(
                {
                    "route": name,
                    "round": round_index,
                    **diagnostics,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    del logits, loss
    return result


def _compile_without_update(
    name: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    execution_position: int,
    gradient_clip_norm: float | None,
) -> None:
    learning_rates = [group["lr"] for group in optimizer.param_groups]
    try:
        for group in optimizer.param_groups:
            group["lr"] = 0.0
        _step(
            name,
            model,
            optimizer,
            tokens,
            targets,
            round_index=-1,
            batch_index=0,
            execution_position=execution_position,
            warmup=True,
            gradient_clip_norm=gradient_clip_norm,
        )
    finally:
        for group, learning_rate in zip(
            optimizer.param_groups, learning_rates, strict=True
        ):
            group["lr"] = learning_rate
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                value.zero_()
    optimizer.zero_grad(set_to_none=True)


@torch.no_grad()
def _evaluate(
    models: dict[str, torch.nn.Module],
    tokens: torch.Tensor,
    targets: torch.Tensor,
    *,
    round_index: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {"round": round_index, "routes": {}}
    for name, model in models.items():
        losses: list[float] = []
        # Validation rotates across independently bound low-precision models.
        # Activate each model's fixed route once before any of its batches;
        # otherwise the route left active by the preceding training step can
        # reject the first validation forward or benchmark the wrong binary.
        activate_model_forward_route(model)
        model.eval()
        for batch_index in range(tokens.shape[0]):
            logits = model(tokens[batch_index : batch_index + 1])
            loss = float(
                _cross_entropy(
                    logits, targets[batch_index : batch_index + 1]
                )
            )
            if not math.isfinite(loss):
                raise RuntimeError(f"non-finite validation loss in {name}")
            losses.append(loss)
            del logits
        model.train()
        mean_loss = statistics.fmean(losses)
        result["routes"][name] = {
            "losses": losses,
            "mean_loss": mean_loss,
            "perplexity": math.exp(min(mean_loss, 80.0)),
        }
        print(
            f"validation round={round_index} route={name} "
            f"loss={mean_loss:.6f}",
            flush=True,
        )
    return result


def _timing_summary(
    records: list[dict[str, Any]], config: Config
) -> dict[str, float | int]:
    # A diagnostic route can perturb later routes in the same rotated round.
    # The caller therefore marks the entire matched round ineligible rather
    # than excluding only the instrumented route's record.
    timing_records = select_timing_records(records)
    timing_fallback_used = not timing_records
    if not timing_records:
        timing_records = records
    medians = {
        key: statistics.median(
            float(record[key]) for record in timing_records
        )
        for key in (
            "forward_ms",
            "backward_ms",
            "gradient_clip_ms",
            "optimizer_ms",
            "step_ms",
            "wall_ms",
        )
    }
    step_seconds = medians["step_ms"] / 1000.0
    useful_flops = _useful_flops(config)
    return {
        **medians,
        "timed_records": len(timing_records),
        "matched_round_records_ineligible": sum(
            not bool(record.get("timing_eligible", True))
            for record in records
        ),
        "route_diagnostic_records": sum(
            bool(record.get("diagnostic")) for record in records
        ),
        "timing_fallback_used": timing_fallback_used,
        "tokens_per_second": config.sequence / step_seconds,
        "useful_tflops": useful_flops / step_seconds / 1.0e12,
        "useful_mfu_at_2250_tflops": useful_flops / step_seconds / 2.25e15,
    }


def _argument_was_provided(argv: list[str], option: str) -> bool:
    """Return whether an argparse option was explicitly supplied."""
    return any(
        argument == option or argument.startswith(option + "=")
        for argument in argv
    )


def _resolve_model_preset_options(
    args: argparse.Namespace,
    config: Config,
    argv: list[str],
) -> None:
    """Apply architecture-specific defaults without hiding contradictions."""
    for field, option, default in (
        ("rope_theta", "--rope-theta", config.rope_theta),
        ("rope_factor", "--rope-factor", config.rope_factor),
    ):
        if not _argument_was_provided(argv, option):
            setattr(args, field, default)
    if config.head_dim != 128:
        return

    d128_contract = (
        ("backward_exp2_degree", "--backward-exp2-degree", 1),
        ("backward_exp2_period", "--backward-exp2-period", 0),
        (
            "mx_backward_reuse_quantized_p",
            "--mx-backward-reuse-quantized-p",
            True,
        ),
        (
            "fp8_backward_reuse_quantized_p",
            "--fp8-backward-reuse-quantized-p",
            True,
        ),
        ("mx_qkv_projection_format", "--mx-qkv-projection-format", "nvfp4"),
        (
            "fp8_qkv_projection_format",
            "--fp8-qkv-projection-format",
            "nvfp4",
        ),
        (
            "mx_backward_match_forward_operands",
            "--mx-backward-match-forward-operands",
            False,
        ),
        (
            "fp8_backward_match_forward_operands",
            "--fp8-backward-match-forward-operands",
            False,
        ),
        ("mx_per_block_qk_scales", "--mx-per-block-qk-scales", False),
        ("fp8_per_block_qk_scales", "--fp8-per-block-qk-scales", False),
        (
            "mx_experimental_split_v_backward",
            "--mx-experimental-split-v-backward",
            False,
        ),
        (
            "mx_backward_forward_probability_replay",
            "--mx-backward-forward-probability-replay",
            False,
        ),
        (
            "mx_backward_forward_probability_scale_handoff",
            "--mx-backward-forward-probability-scale-handoff",
            False,
        ),
        ("v_mxfp4_scaling", "--v-mxfp4-scaling", "1d"),
    )
    for field, option, required in d128_contract:
        negative_option = "--no-" + option.removeprefix("--")
        explicit = _argument_was_provided(
            argv, option
        ) or _argument_was_provided(argv, negative_option)
        actual = getattr(args, field)
        if explicit and actual != required:
            raise ValueError(
                f"{option}={actual!r} is incompatible with the D128 "
                f"projection contract; expected {required!r}"
            )
        setattr(args, field, required)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-preset",
        choices=MODEL_PRESETS,
        default=DEFAULT_MODEL_PRESET,
        help=(
            "model architecture; llama3.1-8b selects L32/H4096/D128, "
            "Llama-3.1 RoPE, and an untied language-model head"
        ),
    )
    parser.add_argument(
        "--layers",
        type=int,
        help="override the preset depth only for integration smoke tests",
    )
    parser.add_argument("--rounds", type=int, default=96)
    parser.add_argument("--training-batches", type=int, default=96)
    parser.add_argument("--validation-batches", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=12)
    parser.add_argument(
        "--diagnostic-start",
        type=int,
        help="one-based update at which to begin finite-value tracing",
    )
    parser.add_argument(
        "--diagnostic-every",
        type=int,
        default=1,
        help="trace every N updates at or after --diagnostic-start",
    )
    parser.add_argument(
        "--diagnostic-routes",
        nargs="+",
        choices=ROUTE_NAMES,
        default=[
            "nvfp4_qk_mxfp4_pv",
            "nvfp4_qk_fp8_pv_exact",
        ],
        help="routes eligible for scheduled numerical diagnostics",
    )
    parser.add_argument(
        "--progress-output",
        type=Path,
        help="atomic JSON progress snapshot retained if training stops early",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=32,
        help="replace --progress-output every N complete rounds",
    )
    parser.add_argument(
        "--mx-loss-drift-window",
        type=int,
        default=32,
        help="matched-batch rolling window for the optional MX loss gate",
    )
    parser.add_argument(
        "--mx-loss-drift-warning-threshold",
        type=float,
        help="latch MX diagnostics when its rolling loss gap exceeds this",
    )
    parser.add_argument(
        "--mx-loss-drift-failure-threshold",
        type=float,
        help="stop early when MX exceeds both controls by this rolling gap",
    )
    parser.add_argument(
        "--mx-loss-drift-failure-patience",
        type=int,
        default=3,
        help="consecutive over-threshold windows required to stop",
    )
    parser.add_argument(
        "--mx-loss-drift-minimum-updates",
        type=int,
        default=256,
        help="do not evaluate the optional MX drift gate before this update",
    )
    parser.add_argument(
        "--diagnostic-on-drift-warning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="trace every subsequent MX step after the drift warning latches",
    )
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        help=(
            "clip every route's global gradient norm before AdamW; the "
            "pre-clip norm and clipping cost are recorded"
        ),
    )
    parser.add_argument(
        "--routes",
        nargs="+",
        choices=ROUTE_NAMES,
        default=list(ROUTE_NAMES),
        help=(
            "routes to retain and train in this process; selecting one route "
            "supports memory-safe full-depth independent-process runs while "
            "preserving seeded BF16-reference initialization"
        ),
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--rope-theta", type=float, default=LLAMA3_ROPE_THETA)
    parser.add_argument("--rope-factor", type=float, default=LLAMA32_ROPE_FACTOR)
    parser.add_argument("--expected-backward-extension", type=Path)
    parser.add_argument(
        "--backward-control-source",
        type=Path,
        help="precomposed generated CuTe backward control Python source",
    )
    parser.add_argument(
        "--backward-control-sha256",
        help="required SHA256 for --backward-control-source",
    )
    parser.add_argument(
        "--backward-control-bytes",
        type=int,
        help="required byte size for --backward-control-source",
    )
    parser.add_argument(
        "--mx-extension",
        type=Path,
        default=Path("/tmp/_C_causal_scale_reuse_policy_d64.so"),
    )
    parser.add_argument(
        "--mx-module", default="_C_causal_scale_reuse_policy_d64"
    )
    parser.add_argument(
        "--fp8-extension",
        type=Path,
        default=Path(
            "/tmp/_C_causal_gqa_nvfp4_fp8pv_exact_d64_sm100_eval_20260818.so"
        ),
    )
    parser.add_argument(
        "--fp8-module",
        default="_C_causal_gqa_nvfp4_fp8pv_exact_d64_sm100_eval_20260818",
    )
    parser.add_argument("--backward-gain", type=float, default=1.0)
    parser.add_argument(
        "--mx-backward-gain",
        type=float,
        help="MX-route backward gain; defaults to --backward-gain",
    )
    parser.add_argument(
        "--fp8-backward-gain",
        type=float,
        help="FP8-route backward gain; defaults to --backward-gain",
    )
    for route in ("mx", "fp8"):
        for field in ("q", "k", "v", "v-weight"):
            parser.add_argument(
                f"--{route}-backward-{field}-gain",
                type=float,
                help=(
                    f"optional {route.upper()} {field.upper()}-gradient "
                    "calibration; defaults to the route-wide backward gain"
                ),
            )
    parser.add_argument("--q-quant-scale", type=float, default=2.25)
    parser.add_argument("--k-quant-scale", type=float, default=2.0)
    parser.add_argument(
        "--qk-scale-refresh-every",
        type=int,
        default=0,
        help=(
            "refresh paired-head Q/K encode scales from the current projection "
            "weights every N optimizer updates; 0 keeps fixed scales"
        ),
    )
    parser.add_argument(
        "--backward-exp2-degree",
        type=int,
        choices=(1, 2),
        default=2,
        help="polynomial degree used when --backward-exp2-period is explicit",
    )
    parser.add_argument(
        "--backward-exp2-period",
        type=int,
        choices=range(17),
        default=None,
        help=(
            "selective EX2 period; omission uses the measured-shape D64 "
            "dispatch (including degree-1/period-2 for S4096 H=32/8), "
            "while 0 "
            "explicitly forces native EX2"
        ),
    )
    for route in ("mx", "fp8"):
        parser.add_argument(
            f"--{route}-backward-reuse-quantized-p",
            action=argparse.BooleanOptionalAction,
            default=False,
            help=(
                f"reuse the {route.upper()} backward kernel's E4M3-rounded "
                "probability between dV and dS; this does not replay the "
                "forward probability"
            ),
        )
        parser.add_argument(
            f"--{route}-backward-match-forward-operands",
            action=argparse.BooleanOptionalAction,
            default=True,
            help=(
                f"publish {route.upper()} backward Q/K and, where applicable, "
                "V from the exact codes used by forward"
            ),
        )
        parser.add_argument(
            f"--{route}-per-block-qk-scales",
            action=argparse.BooleanOptionalAction,
            default=True,
            help=(
                f"publish {route.upper()} Q/K with one E4M3 scale per "
                "logical row x K16; implies represented backward operands "
                "and requires the E4M3 projection route"
            ),
        )
    parser.add_argument(
        "--mx-experimental-split-v-backward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "retain represented per-block MX Q/K while publishing backward "
            "E4M3 V directly from the projection accumulator; this is the "
            "production matched-backward path"
        ),
    )
    parser.add_argument(
        "--mx-backward-forward-probability-replay",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "replay the MX forward probability representation in backward; "
            "requires a compatible D4ALL forward topology"
        ),
    )
    parser.add_argument(
        "--mx-backward-forward-probability-scale-handoff",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "publish MX forward probability scales to backward; disabled in "
            "the production matched-backward path"
        ),
    )
    parser.add_argument(
        "--projection-weight-scaling",
        choices=("1d", "2d"),
        default="2d",
    )
    parser.add_argument(
        "--v-mxfp4-scaling", choices=("1d", "2d"), default="1d"
    )
    parser.add_argument(
        "--fp8-qkv-projection-format",
        choices=("nvfp4", "e4m3"),
        default="e4m3",
        help=(
            "projection operand format for the exact FP8-PV route; E4M3 "
            "requires a compatible projection-native backward extension"
        ),
    )
    parser.add_argument(
        "--mx-qkv-projection-format",
        choices=("nvfp4", "e4m3"),
        default="e4m3",
        help="projection operand format for the MXFP4-PV route",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    backward_control_identity = (
        args.backward_control_source,
        args.backward_control_sha256,
        args.backward_control_bytes,
    )
    if any(value is not None for value in backward_control_identity) and not all(
        value is not None for value in backward_control_identity
    ):
        parser.error(
            "--backward-control-source, --backward-control-sha256, and "
            "--backward-control-bytes must be supplied together"
        )
    repo_root = Path(__file__).resolve().parents[2]
    source_identity = {
        "trainer": _file_identity(Path(__file__)),
        "git": _git_identity(repo_root),
    }
    try:
        config = config_from_model_preset(
            args.model_preset,
            layers=args.layers,
        )
        _resolve_model_preset_options(args, config, sys.argv[1:])
        route_names = tuple(dict.fromkeys(args.routes))
        _require_memory_safe_matched_replicas(
            config,
            route_names,
            operation="real-token trainer",
        )
    except ValueError as error:
        parser.error(str(error))
    if config.head_dim == 128 and any(
        value is not None for value in backward_control_identity
    ):
        parser.error(
            "D128 requires the generated shared-P backward control; "
            "do not supply a D64 precomposed control artifact"
        )

    if torch.cuda.device_count() != 1:
        raise RuntimeError("expose exactly one GPU")
    if args.rounds < 3:
        raise ValueError("--rounds must be at least three")
    if args.training_batches < 2 or args.validation_batches < 1:
        raise ValueError("need at least two train and one validation batch")
    if args.eval_every < 1:
        raise ValueError("--eval-every must be positive")
    if args.diagnostic_start is not None and args.diagnostic_start < 1:
        raise ValueError("--diagnostic-start must be positive")
    if args.diagnostic_every < 1:
        raise ValueError("--diagnostic-every must be positive")
    if args.progress_output is not None and args.progress_every < 1:
        raise ValueError("--progress-every must be positive")
    if (
        args.progress_output is not None
        and args.output is not None
        and args.progress_output.resolve() == args.output.resolve()
    ):
        parser.error("--progress-output and --output must be different paths")
    if (
        args.mx_loss_drift_warning_threshold is not None
        and args.mx_loss_drift_failure_threshold is None
    ):
        parser.error(
            "--mx-loss-drift-warning-threshold requires "
            "--mx-loss-drift-failure-threshold"
        )
    if args.diagnostic_on_drift_warning and (
        args.mx_loss_drift_warning_threshold is None
    ):
        parser.error(
            "--diagnostic-on-drift-warning requires "
            "--mx-loss-drift-warning-threshold"
        )
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("--train-fraction must be in (0,1)")
    if not math.isfinite(args.rope_theta) or args.rope_theta <= 0.0:
        raise ValueError("--rope-theta must be finite and positive")
    if not math.isfinite(args.rope_factor) or args.rope_factor <= 0.0:
        raise ValueError("--rope-factor must be finite and positive")
    if not math.isfinite(args.backward_gain) or args.backward_gain <= 0.0:
        raise ValueError("--backward-gain must be finite and positive")
    if args.gradient_clip_norm is not None and (
        not math.isfinite(args.gradient_clip_norm)
        or args.gradient_clip_norm <= 0.0
    ):
        raise ValueError("--gradient-clip-norm must be finite and positive")
    for name, gain in (
        ("--mx-backward-gain", args.mx_backward_gain),
        ("--fp8-backward-gain", args.fp8_backward_gain),
        ("--mx-backward-q-gain", args.mx_backward_q_gain),
        ("--mx-backward-k-gain", args.mx_backward_k_gain),
        ("--mx-backward-v-gain", args.mx_backward_v_gain),
        ("--mx-backward-v-weight-gain", args.mx_backward_v_weight_gain),
        ("--fp8-backward-q-gain", args.fp8_backward_q_gain),
        ("--fp8-backward-k-gain", args.fp8_backward_k_gain),
        ("--fp8-backward-v-gain", args.fp8_backward_v_gain),
        ("--fp8-backward-v-weight-gain", args.fp8_backward_v_weight_gain),
    ):
        if gain is not None and (not math.isfinite(gain) or gain <= 0.0):
            raise ValueError(f"{name} must be finite and positive")
    if args.qk_scale_refresh_every < 0:
        raise ValueError("--qk-scale-refresh-every must be non-negative")
    for route in ("mx", "fp8"):
        if (
            getattr(args, f"{route}_per_block_qk_scales")
            and getattr(args, f"{route}_qkv_projection_format") != "e4m3"
        ):
            parser.error(
                f"--{route}-per-block-qk-scales requires "
                f"--{route}-qkv-projection-format=e4m3"
            )
    diagnostic_routes = tuple(dict.fromkeys(args.diagnostic_routes))
    selected_lowp_slots = tuple(
        slot
        for slot, route_name in (
            ("mx", "nvfp4_qk_mxfp4_pv"),
            ("fp8", "nvfp4_qk_fp8_pv_exact"),
        )
        if route_name in route_names
    )
    production_projection_format = (
        "e4m3" if config.head_dim == 64 else "nvfp4"
    )
    for slot in selected_lowp_slots:
        if (
            getattr(args, f"{slot}_qkv_projection_format")
            != production_projection_format
        ):
            parser.error(
                f"the production {slot.upper()} route requires "
                f"--{slot}-qkv-projection-format="
                f"{production_projection_format} for D{config.head_dim}"
            )
    required_projection_symbols = _required_projection_symbols(
        args,
        selected_lowp_slots,
        head_dim=config.head_dim,
    )
    projection_extension = _projection_extension_identity(
        required_projection_symbols,
        args.expected_backward_extension,
    )
    backward_extension = _backward_extension_identity(
        args.expected_backward_extension
    )
    unavailable_diagnostic_routes = sorted(
        set(diagnostic_routes).difference(route_names)
    )
    if args.diagnostic_start is not None and unavailable_diagnostic_routes:
        parser.error(
            "scheduled diagnostic routes were not selected: "
            f"{unavailable_diagnostic_routes}"
        )
    drift_gate = None
    if args.mx_loss_drift_failure_threshold is not None:
        drift_routes = {
            "bf16_cute",
            "nvfp4_qk_mxfp4_pv",
            "nvfp4_qk_fp8_pv_exact",
        }
        missing_drift_routes = sorted(drift_routes.difference(route_names))
        if missing_drift_routes:
            parser.error(
                "the MX drift gate requires matched BF16, MX, and FP8 "
                f"routes; missing {missing_drift_routes}"
            )
        drift_gate = RouteLossDriftGate(
            subject_route="nvfp4_qk_mxfp4_pv",
            reference_routes=(
                "bf16_cute",
                "nvfp4_qk_fp8_pv_exact",
            ),
            window=args.mx_loss_drift_window,
            warning_threshold=args.mx_loss_drift_warning_threshold,
            failure_threshold=args.mx_loss_drift_failure_threshold,
            failure_patience=args.mx_loss_drift_failure_patience,
            minimum_updates=args.mx_loss_drift_minimum_updates,
        )
    if (
        "nvfp4_qk_mxfp4_pv" in route_names
        and args.mx_experimental_split_v_backward
        and not (
            args.mx_qkv_projection_format == "e4m3"
            and args.mx_per_block_qk_scales
        )
    ):
        parser.error(
            "--mx-experimental-split-v-backward requires "
            "--mx-qkv-projection-format=e4m3 and "
            "--mx-per-block-qk-scales when the MX route is selected"
        )
    torch.cuda.set_device(0)
    hardware_identity = _cuda_hardware_identity()

    backward_exp2_policy = resolve_backward_exp2_policy(
        sequence=config.sequence,
        head_dim=config.head_dim,
        q_heads=config.q_heads,
        kv_heads=config.kv_heads,
        lowp=True,
        exp2_degree=args.backward_exp2_degree,
        exp2_period=args.backward_exp2_period,
    )
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    tokenizer_vocab = tokenizer.get_vocab_size(with_added_tokens=True)
    if tokenizer_vocab != config.vocab:
        raise RuntimeError(
            f"tokenizer vocabulary {tokenizer_vocab} != model {config.vocab}"
        )
    documents, corpus_metadata = _load_corpus(args.corpus)
    document_indices = list(range(len(documents)))
    split_seed = args.seed + 303
    random.Random(split_seed).shuffle(document_indices)
    split_index = int(args.train_fraction * len(document_indices))
    train_indices = document_indices[:split_index]
    validation_indices = document_indices[split_index:]
    train_tokens, train_targets, train_metadata = _token_batches(
        documents,
        train_indices,
        tokenizer,
        batch_count=args.training_batches,
        sequence=config.sequence,
    )
    validation_tokens, validation_targets, validation_metadata = _token_batches(
        documents,
        validation_indices,
        tokenizer,
        batch_count=args.validation_batches,
        sequence=config.sequence,
    )

    rope = _make_llama3_rope(
        config.sequence,
        config.head_dim,
        args.rope_theta,
        args.rope_factor,
    )
    common_runtime = {
        "q_quant_scale": args.q_quant_scale,
        "k_quant_scale": args.k_quant_scale,
        "projection_weight_scale_2d": (
            args.projection_weight_scaling == "2d"
        ),
        "v_mxfp4_scale_2d": args.v_mxfp4_scaling == "2d",
        "backward_exp2_degree": args.backward_exp2_degree,
        "backward_exp2_period": args.backward_exp2_period,
        "backward_control_source": args.backward_control_source,
        "backward_control_sha256": args.backward_control_sha256,
        "backward_control_bytes": args.backward_control_bytes,
    }
    mx_runtime = None
    mx_topology = None
    mx_forward_extension = None
    if "nvfp4_qk_mxfp4_pv" in route_names:
        mx_runtime, mx_topology = _make_runtime(
            config,
            rope,
            args.mx_extension,
            args.mx_module,
            route_slot="mx",
            qkv_projection_format=args.mx_qkv_projection_format,
            backward_probability_correction=(
                args.mx_backward_gain
                if args.mx_backward_gain is not None
                else args.backward_gain
            ),
            backward_q_gain=args.mx_backward_q_gain,
            backward_k_gain=args.mx_backward_k_gain,
            backward_v_gain=args.mx_backward_v_gain,
            backward_v_weight_gain=args.mx_backward_v_weight_gain,
            backward_reuse_quantized_p=(
                args.mx_backward_reuse_quantized_p
            ),
            backward_match_forward_operands=(
                args.mx_backward_match_forward_operands
            ),
            per_block_qk_scales=args.mx_per_block_qk_scales,
            experimental_split_v_backward=(
                args.mx_experimental_split_v_backward
            ),
            backward_forward_mx_probability_replay=(
                args.mx_backward_forward_probability_replay
            ),
            backward_forward_mx_probability_scale_handoff=(
                args.mx_backward_forward_probability_scale_handoff
            ),
            **common_runtime,
        )
        mx_forward_extension = _extension_identity(
            args.mx_extension, args.mx_module
        )
    mx_probability_replay_provenance = _mx_probability_replay_provenance(
        mx_runtime
    )
    fp8_runtime = None
    fp8_topology = None
    fp8_forward_extension = None
    if "nvfp4_qk_fp8_pv_exact" in route_names:
        fp8_runtime, fp8_topology = _make_runtime(
            config,
            rope,
            args.fp8_extension,
            args.fp8_module,
            route_slot="fp8",
            qkv_projection_format=args.fp8_qkv_projection_format,
            backward_probability_correction=(
                args.fp8_backward_gain
                if args.fp8_backward_gain is not None
                else args.backward_gain
            ),
            backward_q_gain=args.fp8_backward_q_gain,
            backward_k_gain=args.fp8_backward_k_gain,
            backward_v_gain=args.fp8_backward_v_gain,
            backward_v_weight_gain=args.fp8_backward_v_weight_gain,
            backward_reuse_quantized_p=(
                args.fp8_backward_reuse_quantized_p
            ),
            backward_match_forward_operands=(
                args.fp8_backward_match_forward_operands
            ),
            per_block_qk_scales=args.fp8_per_block_qk_scales,
            shared_backward_runtime=mx_runtime,
            **common_runtime,
        )
        fp8_forward_extension = _extension_identity(
            args.fp8_extension, args.fp8_module
        )
    backward_route_contracts = {
        name: runtime.backward_contract()
        for name, runtime in (
            ("nvfp4_qk_mxfp4_pv", mx_runtime),
            ("nvfp4_qk_fp8_pv_exact", fp8_runtime),
        )
        if runtime is not None
    }
    require_matching_backward_contracts(backward_route_contracts)
    shared_backward_runner = None
    if mx_runtime is not None and fp8_runtime is not None:
        shared_backward_runner = _share_matched_backward_runner(
            mx_runtime,
            fp8_runtime,
        )
        if not all(shared_backward_runner.values()):
            raise RuntimeError(
                "low-precision training routes did not share one backward"
            )
    backward_policy_runtime = (
        mx_runtime if mx_runtime is not None else fp8_runtime
    )

    torch.manual_seed(args.seed)
    bf16_model = Llama12B(config, rope, None)
    initial_state = bf16_model.state_dict()
    initial_state_probe = _state_dict_probe(initial_state)
    parameter_count = sum(
        parameter.numel() for parameter in bf16_model.parameters()
    )
    if parameter_count != config.parameter_count:
        raise RuntimeError(
            f"constructed {parameter_count} parameters, expected "
            f"{config.parameter_count} for {config.model_preset}"
        )
    models = {"bf16_cute": bf16_model}
    if mx_runtime is not None:
        mx_model = Llama12B(config, rope, mx_runtime)
        mx_model.load_state_dict(initial_state, strict=True)
        models["nvfp4_qk_mxfp4_pv"] = mx_model
    if fp8_runtime is not None:
        fp8_model = Llama12B(config, rope, fp8_runtime)
        fp8_model.load_state_dict(initial_state, strict=True)
        models["nvfp4_qk_fp8_pv_exact"] = fp8_model
    models = {name: models[name] for name in route_names}
    per_block_qk_by_route = {
        "bf16_cute": False,
        "nvfp4_qk_mxfp4_pv": bool(args.mx_per_block_qk_scales),
        "nvfp4_qk_fp8_pv_exact": bool(args.fp8_per_block_qk_scales),
    }
    effective_qk_scale_refresh_every = {
        name: (
            0
            if name == "bf16_cute" or per_block_qk_by_route[name]
            else args.qk_scale_refresh_every
        )
        for name in route_names
    }
    if args.qk_scale_refresh_every:
        for name, model in models.items():
            if effective_qk_scale_refresh_every[name]:
                _refresh_model_qk_scales(model)
    del initial_state, bf16_model
    optimizers = {
        name: _optimizer(model, args.learning_rate)
        for name, model in models.items()
    }

    for execution_position, name in enumerate(route_names):
        _compile_without_update(
            name,
            models[name],
            optimizers[name],
            train_tokens[0:1],
            train_targets[0:1],
            execution_position,
            args.gradient_clip_norm,
        )

    timed_forward_dispatch_contracts = None
    if mx_runtime is not None or fp8_runtime is not None:
        # Apply the comparator's exact post-compile preflight to the retained
        # training models too.  This proves every layer workspace completed
        # first-use ABI authentication and that steady-state forward dispatch
        # is bound to the route-specific unchecked projection/attention
        # symbols before any real-token training update is accepted.
        timed_forward_dispatch_contracts = (
            _timed_forward_dispatch_contracts(
                mx_runtime,
                fp8_runtime,
                models.get("nvfp4_qk_mxfp4_pv"),
                models.get("nvfp4_qk_fp8_pv_exact"),
            )
        )

    validation_history = [
        _evaluate(
            models,
            validation_tokens,
            validation_targets,
            round_index=-1,
        )
    ]
    records: dict[str, list[dict[str, Any]]] = {
        name: [] for name in route_names
    }

    def write_progress(
        state: str,
        *,
        last_complete_round: int,
        active_round: int | None = None,
    ) -> None:
        if args.progress_output is None:
            return
        _atomic_write_json(
            args.progress_output,
            {
                "schema": "llama12b_real_tokens_training_progress_v1",
                "state": state,
                "updated_unix_seconds": time.time(),
                "command": [sys.executable, *sys.argv],
                "source": source_identity,
                "configuration": {
                    "model_preset": config.model_preset,
                    "layers": config.layers,
                    "rounds": args.rounds,
                    "training_batches": args.training_batches,
                    "validation_batches": args.validation_batches,
                    "eval_every": args.eval_every,
                    "seed": args.seed,
                    "routes": list(route_names),
                    "diagnostic_start": args.diagnostic_start,
                    "diagnostic_every": args.diagnostic_every,
                    "diagnostic_routes": list(diagnostic_routes),
                    "diagnostic_on_drift_warning": (
                        args.diagnostic_on_drift_warning
                    ),
                    "progress_every": args.progress_every,
                },
                "last_complete_round": last_complete_round,
                "active_round": active_round,
                "route_record_counts": {
                    name: len(route_records)
                    for name, route_records in records.items()
                },
                "records": records,
                "validation_history": validation_history,
                "drift_gate": (
                    drift_gate.as_dict() if drift_gate is not None else None
                ),
            },
        )
        print(
            f"progress-checkpoint state={state} "
            f"last_complete_round={last_complete_round} "
            f"path={args.progress_output}",
            flush=True,
        )

    write_progress("initialized", last_complete_round=-1)
    torch.cuda.reset_peak_memory_stats()
    for round_index in range(args.rounds):
        batch_index = round_index % args.training_batches
        offset = round_index % len(route_names)
        order = route_names[offset:] + route_names[:offset]
        for execution_position, name in enumerate(order):
            scheduled_diagnostic = bool(
                args.diagnostic_start is not None
                and round_index + 1 >= args.diagnostic_start
                and (
                    round_index + 1 - args.diagnostic_start
                )
                % args.diagnostic_every
                == 0
                and name in diagnostic_routes
            )
            drift_diagnostic = bool(
                drift_gate is not None
                and drift_gate.warning_active
                and args.diagnostic_on_drift_warning
                and name == "nvfp4_qk_mxfp4_pv"
            )
            diagnostic_reason = (
                "scheduled+drift_warning"
                if scheduled_diagnostic and drift_diagnostic
                else "scheduled"
                if scheduled_diagnostic
                else "drift_warning"
                if drift_diagnostic
                else None
            )
            record = _step(
                name,
                models[name],
                optimizers[name],
                train_tokens[batch_index : batch_index + 1],
                train_targets[batch_index : batch_index + 1],
                round_index=round_index,
                batch_index=batch_index,
                execution_position=execution_position,
                warmup=False,
                diagnose=diagnostic_reason is not None,
                refresh_qk_scales=(
                    effective_qk_scale_refresh_every[name] > 0
                    and (round_index + 1)
                    % effective_qk_scale_refresh_every[name]
                    == 0
                ),
                gradient_clip_norm=args.gradient_clip_norm,
            )
            record["diagnostic_reason"] = diagnostic_reason
            records[name].append(record)
            if not record["finite"]:
                write_progress(
                    "nonfinite_training_state",
                    last_complete_round=round_index - 1,
                    active_round=round_index,
                )
                raise RuntimeError(
                    "non-finite training state in "
                    f"{name}: {record['failure_stage']}"
                )
        mark_matched_round_timing_eligibility(
            {name: records[name][-1] for name in route_names}
        )
        gate_report = None
        if drift_gate is not None:
            gate_report = drift_gate.observe(
                round_index,
                {
                    name: float(records[name][-1]["loss"])
                    for name in route_names
                },
            )
            if gate_report["warning_exceeded"] or gate_report["failed"]:
                print(
                    "loss-drift-gate "
                    + json.dumps(gate_report, sort_keys=True),
                    flush=True,
                )
            if gate_report["failed"]:
                write_progress(
                    "loss_drift_gate_failed",
                    last_complete_round=round_index,
                )
                raise RuntimeError(
                    "MX rolling loss drift exceeded the configured "
                    "matched-route gate"
                )
        if (round_index + 1) % args.eval_every == 0 or (
            round_index + 1 == args.rounds
        ):
            validation_history.append(
                _evaluate(
                    models,
                    validation_tokens,
                    validation_targets,
                    round_index=round_index,
                )
            )
        if (
            args.progress_output is not None
            and (round_index + 1) % args.progress_every == 0
        ):
            write_progress(
                "running",
                last_complete_round=round_index,
            )

    routes: dict[str, Any] = {}
    for name in route_names:
        losses = [float(record["loss"]) for record in records[name]]
        preclip_gradient_norms = [
            float(record["gradient_preclip_total_norm"])
            for record in records[name]
            if record["gradient_preclip_total_norm"] is not None
        ]
        clipped_steps = sum(
            bool(record["gradient_was_clipped"])
            for record in records[name]
        )
        routes[name] = {
            "timing": _timing_summary(records[name], config),
            "training": {
                "losses": losses,
                "first_loss": losses[0],
                "last_loss": losses[-1],
                "last_eight_mean": statistics.fmean(losses[-8:]),
                "minimum_loss": min(losses),
                "all_steps_finite": all(
                    bool(record["finite"]) for record in records[name]
                ),
                "gradient_clipping": {
                    "clipped_steps": clipped_steps,
                    "clipped_fraction": clipped_steps / len(records[name]),
                    "preclip_total_norm_median": statistics.median(
                        preclip_gradient_norms
                    ),
                    "preclip_total_norm_maximum": max(
                        preclip_gradient_norms
                    ),
                }
                if preclip_gradient_norms
                else None,
            },
            "validation": {
                "initial_loss": validation_history[0]["routes"][name][
                    "mean_loss"
                ],
                "final_loss": validation_history[-1]["routes"][name][
                    "mean_loss"
                ],
            },
        }

    comparison_reference_route, comparisons = _comparisons_against_bf16(
        routes
    )

    result = {
        "schema": "llama12b_real_tokens_training_v3",
        "command": [sys.executable, *sys.argv],
        "source": source_identity,
        "configuration": {
            **config.__dict__,
            "batch": 1,
            "python_executable": sys.executable,
            "python_executable_resolved": str(Path(sys.executable).resolve()),
            "hardware_identity": hardware_identity,
            "rounds": args.rounds,
            "routes": list(route_names),
            "training_batches": args.training_batches,
            "validation_batches": args.validation_batches,
            "eval_every": args.eval_every,
            "diagnostic_start": args.diagnostic_start,
            "diagnostic_every": args.diagnostic_every,
            "diagnostic_routes": list(diagnostic_routes),
            "diagnostic_on_drift_warning": (
                args.diagnostic_on_drift_warning
            ),
            "progress_output": (
                str(args.progress_output)
                if args.progress_output is not None
                else None
            ),
            "progress_every": (
                args.progress_every
                if args.progress_output is not None
                else None
            ),
            "mx_loss_drift_gate": (
                drift_gate.as_dict()["configuration"]
                if drift_gate is not None
                else None
            ),
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "gradient_clip_norm": args.gradient_clip_norm,
            "gradient_clip_norm_type": (
                2.0 if args.gradient_clip_norm is not None else None
            ),
            "gradient_clip_foreach": (
                True if args.gradient_clip_norm is not None else None
            ),
            "gradient_error_if_nonfinite": (
                True if args.gradient_clip_norm is not None else None
            ),
            "optimizer": "fused AdamW",
            "loss_dtype": "fp32",
            "warmup_updates_model": False,
            "corpus": str(args.corpus),
            "corpus_sha256": _file_sha256(args.corpus),
            "corpus_documents": len(documents),
            "corpus_metadata": corpus_metadata,
            "train_fraction": args.train_fraction,
            "split_seed": split_seed,
            "train_documents": len(train_indices),
            "validation_documents": len(validation_indices),
            "tokenizer": str(args.tokenizer),
            "tokenizer_sha256": _file_sha256(args.tokenizer),
            "tokenizer_vocab": tokenizer_vocab,
            "bos_token_id": LLAMA_BOS,
            "eos_token_id": LLAMA_EOS,
            "rope_theta": args.rope_theta,
            "rope_scaling": {
                "rope_type": "llama3",
                "factor": args.rope_factor,
                "low_freq_factor": LLAMA3_LOW_FREQ_FACTOR,
                "high_freq_factor": LLAMA3_HIGH_FREQ_FACTOR,
                "original_max_position_embeddings": LLAMA3_ORIGINAL_CONTEXT,
            },
            "backward_extension": backward_extension,
            "projection_extension": projection_extension,
            "mx_forward_extension": mx_forward_extension,
            "fp8_forward_extension": fp8_forward_extension,
            "initialization": "strict clone of BF16 state_dict",
            "initial_state_probe": initial_state_probe,
            "targets": "next token from S+1 chunks; no sequence wraparound",
            "train_tokens": train_metadata,
            "validation_tokens": validation_metadata,
            "projection_weight_scaling": args.projection_weight_scaling,
            "v_mxfp4_scaling": args.v_mxfp4_scaling,
            "fp8_qkv_projection_format": args.fp8_qkv_projection_format,
            "mx_qkv_projection_format": args.mx_qkv_projection_format,
            "q_quant_scale": args.q_quant_scale,
            "k_quant_scale": args.k_quant_scale,
            "qk_scale_refresh_every": args.qk_scale_refresh_every,
            "effective_qk_scale_refresh_every": (
                effective_qk_scale_refresh_every
            ),
            "mx_per_block_qk_scales": args.mx_per_block_qk_scales,
            "fp8_per_block_qk_scales": args.fp8_per_block_qk_scales,
            "mx_experimental_split_v_backward": (
                args.mx_experimental_split_v_backward
            ),
            "mx_backward_forward_probability_replay_requested": (
                args.mx_backward_forward_probability_replay
            ),
            "mx_backward_forward_probability_scale_handoff_requested": (
                args.mx_backward_forward_probability_scale_handoff
            ),
            "mx_backward_forward_probability_replay": (
                mx_runtime.backward_forward_mx_probability_replay
                if mx_runtime is not None
                else None
            ),
            "mx_backward_forward_probability_scale_handoff": (
                mx_runtime.backward_forward_mx_probability_scale_handoff
                if mx_runtime is not None
                else None
            ),
            "mx_probability_replay_provenance": (
                mx_probability_replay_provenance
            ),
            "mx_projection_publication_topology": (
                mx_runtime.projection_publication_topology
                if mx_runtime is not None
                else None
            ),
            "fp8_projection_publication_topology": (
                fp8_runtime.projection_publication_topology
                if fp8_runtime is not None
                else None
            ),
            "backward_gain": args.backward_gain,
            "mx_backward_gain": (
                args.mx_backward_gain
                if args.mx_backward_gain is not None
                else args.backward_gain
            ),
            "fp8_backward_gain": (
                args.fp8_backward_gain
                if args.fp8_backward_gain is not None
                else args.backward_gain
            ),
            "mx_backward_component_gains": {
                "q": mx_runtime.backward_q_gain if mx_runtime else None,
                "k": mx_runtime.backward_k_gain if mx_runtime else None,
                "v": mx_runtime.backward_v_gain if mx_runtime else None,
                "v_weight": (
                    mx_runtime.backward_v_weight_gain if mx_runtime else None
                ),
            },
            "fp8_backward_component_gains": {
                "q": fp8_runtime.backward_q_gain if fp8_runtime else None,
                "k": fp8_runtime.backward_k_gain if fp8_runtime else None,
                "v": fp8_runtime.backward_v_gain if fp8_runtime else None,
                "v_weight": (
                    fp8_runtime.backward_v_weight_gain if fp8_runtime else None
                ),
            },
            "backward_exp2_degree": (
                backward_exp2_policy.effective_degree
            ),
            "backward_exp2_period": (
                backward_exp2_policy.effective_period
            ),
            "backward_exp2_requested_degree": args.backward_exp2_degree,
            "backward_exp2_requested_period": args.backward_exp2_period,
            "backward_control_provenance": (
                backward_policy_runtime.backward_control_provenance
                if backward_policy_runtime is not None
                else None
            ),
            "backward_control_route_provenance": {
                "nvfp4_qk_mxfp4_pv": (
                    mx_runtime.backward_control_provenance
                    if mx_runtime is not None
                    else None
                ),
                "nvfp4_qk_fp8_pv_exact": (
                    fp8_runtime.backward_control_provenance
                    if fp8_runtime is not None
                    else None
                ),
            },
            "backward_route_contracts": backward_route_contracts,
            "matched_lowp_backward_contract": (
                len(backward_route_contracts) == 2
            ),
            "shared_lowp_backward_runner": shared_backward_runner,
            "timed_forward_dispatch_contracts": (
                timed_forward_dispatch_contracts
            ),
            "backward_exp2_policy": backward_exp2_policy.as_dict(),
            "backward_detached_fp8_p_tmem": (
                backward_policy_runtime.backward_detached_fp8_p_tmem
                if backward_policy_runtime is not None
                else None
            ),
            "backward_probability_tmem_policy": (
                backward_policy_runtime.backward_probability_tmem_policy
                if backward_policy_runtime is not None
                else None
            ),
            "backward_head_fast_raster": (
                backward_policy_runtime.backward_head_fast_raster
                if backward_policy_runtime is not None
                else None
            ),
            "backward_raster_policy": (
                backward_policy_runtime.backward_raster_policy
                if backward_policy_runtime is not None
                else None
            ),
            "backward_exp2_route_policies": {
                "nvfp4_qk_mxfp4_pv": (
                    mx_runtime.backward_exp2_policy
                    if mx_runtime is not None
                    else None
                ),
                "nvfp4_qk_fp8_pv_exact": (
                    fp8_runtime.backward_exp2_policy
                    if fp8_runtime is not None
                    else None
                ),
            },
            "backward_detached_fp8_p_tmem_routes": {
                "nvfp4_qk_mxfp4_pv": (
                    mx_runtime.backward_detached_fp8_p_tmem
                    if mx_runtime is not None
                    else None
                ),
                "nvfp4_qk_fp8_pv_exact": (
                    fp8_runtime.backward_detached_fp8_p_tmem
                    if fp8_runtime is not None
                    else None
                ),
            },
            "backward_probability_tmem_route_policies": {
                "nvfp4_qk_mxfp4_pv": (
                    mx_runtime.backward_probability_tmem_policy
                    if mx_runtime is not None
                    else None
                ),
                "nvfp4_qk_fp8_pv_exact": (
                    fp8_runtime.backward_probability_tmem_policy
                    if fp8_runtime is not None
                    else None
                ),
            },
            "backward_head_fast_rasters": {
                "nvfp4_qk_mxfp4_pv": (
                    mx_runtime.backward_head_fast_raster
                    if mx_runtime is not None
                    else None
                ),
                "nvfp4_qk_fp8_pv_exact": (
                    fp8_runtime.backward_head_fast_raster
                    if fp8_runtime is not None
                    else None
                ),
            },
            "backward_raster_route_policies": {
                "nvfp4_qk_mxfp4_pv": (
                    mx_runtime.backward_raster_policy
                    if mx_runtime is not None
                    else None
                ),
                "nvfp4_qk_fp8_pv_exact": (
                    fp8_runtime.backward_raster_policy
                    if fp8_runtime is not None
                    else None
                ),
            },
            "mx_backward_reuse_quantized_p": (
                args.mx_backward_reuse_quantized_p
            ),
            "fp8_backward_reuse_quantized_p": (
                args.fp8_backward_reuse_quantized_p
            ),
            "mx_backward_match_forward_operands": (
                args.mx_backward_match_forward_operands
            ),
            "fp8_backward_match_forward_operands": (
                args.fp8_backward_match_forward_operands
            ),
            "mx_effective_backward_match_forward_operands": (
                mx_runtime.backward_match_forward_operands
                if mx_runtime is not None
                else None
            ),
            "fp8_effective_backward_match_forward_operands": (
                fp8_runtime.backward_match_forward_operands
                if fp8_runtime is not None
                else None
            ),
            "parameter_count": parameter_count,
            "mx_forward_topology": mx_topology,
            "fp8_forward_topology": fp8_topology,
        },
        "validation_history": validation_history,
        "records": records,
        "drift_gate": (
            drift_gate.as_dict() if drift_gate is not None else None
        ),
        "routes": routes,
        "comparison_reference_route": comparison_reference_route,
        "comparisons": comparisons,
        "memory": {
            "peak_allocated_gib": (
                torch.cuda.max_memory_allocated() / 2.0**30
            ),
            "peak_reserved_gib": (
                torch.cuda.max_memory_reserved() / 2.0**30
            ),
        },
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    write_progress("complete", last_complete_round=args.rounds - 1)


if __name__ == "__main__":
    main()
