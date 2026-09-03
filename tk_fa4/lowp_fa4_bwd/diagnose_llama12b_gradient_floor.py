#!/usr/bin/env python3
"""Localize a late weak-gradient cliff in the batched low-precision route.

This is a diagnostic, not a performance benchmark.  It loads one fixed final
model state and one fixed heldout Dolma batch, then repeats backward at several
runtime loss scales.  The first reverse attention invocation is decoder layer
15; synchronously sampling its raw dQ/dK/dV before the shared workspace is
reused distinguishes a probability/dS collapse from a stitch or weight-GEMM
failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
from pathlib import Path
from typing import Any

import torch

from tk_fa4 import interface as tk_interface

BENCHMARK_SCHEMA = "llama12b_saturated_route_benchmark_v2"
SAMPLE_SCHEMA = "llama12b_saturated_route_samples_v2"
DIAGNOSTIC_SCHEMA = "llama12b_lowp_gradient_floor_diagnostic_v2"
STATISTICS_CHUNK_ELEMENTS = 1 << 20
STRIDED_SAMPLE_ELEMENTS = 512
LOSS_ABSOLUTE_TOLERANCE = 1.0e-6
LOSS_RELATIVE_TOLERANCE = 1.0e-7
REPEAT_MINIMUM_COSINE = 0.99
REPEAT_MAXIMUM_RELATIVE_L2 = 0.10


def _load_runtime_dependencies() -> None:
    """Import the CUDA model stack only when the diagnostic is executed."""
    from tk_fa4.lowp_fa4_bwd import benchmark_llama12b_e2e as e2e
    from tk_fa4.lowp_fa4_bwd import benchmark_llama12b_saturated as saturated

    globals().update(
        {
            "Llama12B": e2e.Llama12B,
            "_make_llama3_rope": e2e._make_llama3_rope,
            "activate_model_forward_route": e2e.activate_model_forward_route,
            "config_from_model_preset": e2e.config_from_model_preset,
            "DEFAULT_CONTROL": saturated.DEFAULT_CONTROL,
            "DEFAULT_CORPUS": saturated.DEFAULT_CORPUS,
            "DEFAULT_FORWARDS": saturated.DEFAULT_FORWARDS,
            "DEFAULT_PROJECTION": saturated.DEFAULT_PROJECTION,
            "DEFAULT_TOKENIZER": saturated.DEFAULT_TOKENIZER,
            "PINNED_ARTIFACTS": saturated.PINNED_ARTIFACTS,
            "_dolma_batches": saturated._dolma_batches,
            "_hidden_and_weight": saturated._hidden_and_weight,
            "_loss": saturated._loss,
            "_runtime": saturated._runtime,
            "_source_identity": saturated._source_identity,
        }
    )


def _authenticate_regular_file(
    path: Path,
    *,
    label: str,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
) -> dict[str, Any]:
    """Hash one stable regular non-symlink file through a single descriptor."""
    requested = Path(path)
    try:
        requested_stat = requested.lstat()
    except OSError as error:
        raise RuntimeError(f"unable to stat {label}: {requested}") from error
    if not stat.S_ISREG(requested_stat.st_mode):
        raise RuntimeError(
            f"{label} must be a regular non-symlink file: {requested}"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(requested, flags)
    digest = hashlib.sha256()
    with os.fdopen(descriptor, "rb") as stream:
        opened_stat = os.fstat(stream.fileno())
        if not stat.S_ISREG(opened_stat.st_mode):
            raise RuntimeError(f"{label} stopped being a regular file")
        if (
            opened_stat.st_dev != requested_stat.st_dev
            or opened_stat.st_ino != requested_stat.st_ino
        ):
            raise RuntimeError(f"{label} changed while it was being opened")
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
        final_stat = os.fstat(stream.fileno())
    if (
        final_stat.st_size != opened_stat.st_size
        or final_stat.st_mtime_ns != opened_stat.st_mtime_ns
    ):
        raise RuntimeError(f"{label} changed while it was being authenticated")
    observed_sha256 = digest.hexdigest()
    if expected_bytes is not None and opened_stat.st_size != expected_bytes:
        raise RuntimeError(
            f"{label} byte-count mismatch: expected {expected_bytes}, "
            f"found {opened_stat.st_size}"
        )
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise RuntimeError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, "
            f"found {observed_sha256}"
        )
    return {
        "path": str(requested.resolve(strict=True)),
        "sha256": observed_sha256,
        "bytes": opened_stat.st_size,
    }


def _read_authenticated_json(
    path: Path,
    *,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = _authenticate_regular_file(path, label=label)
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"unable to read {label} JSON: {path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must contain one JSON object")
    # Close the small read/authenticate race by proving the bytes did not move.
    repeated = _authenticate_regular_file(
        path,
        label=label,
        expected_sha256=str(identity["sha256"]),
        expected_bytes=int(identity["bytes"]),
    )
    if repeated != identity:
        raise RuntimeError(f"{label} identity changed while loading JSON")
    return payload, identity


def _require_identity_receipt(
    receipt: Any,
    observed: dict[str, Any],
    *,
    label: str,
) -> None:
    if not isinstance(receipt, dict):
        raise RuntimeError(f"{label} receipt must be an object")
    for key in ("path", "sha256", "bytes"):
        if receipt.get(key) != observed[key]:
            raise RuntimeError(
                f"{label} receipt {key}={receipt.get(key)!r}, "
                f"observed {observed[key]!r}"
            )


def _receipt_file_fields(
    receipt: Any,
    *,
    label: str,
) -> tuple[Path, str, int]:
    if not isinstance(receipt, dict):
        raise RuntimeError(f"{label} receipt must be an object")
    path = receipt.get("path")
    sha256 = receipt.get("sha256")
    byte_count = receipt.get("bytes")
    if not isinstance(path, str) or not path:
        raise RuntimeError(f"{label} receipt is missing its path")
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
    ):
        raise RuntimeError(f"{label} receipt has an invalid SHA-256")
    if (
        not isinstance(byte_count, int)
        or isinstance(byte_count, bool)
        or byte_count < 0
    ):
        raise RuntimeError(f"{label} receipt has an invalid byte count")
    return Path(path), sha256, byte_count


def _deterministic_strided_sample(
    flattened: torch.Tensor,
    *,
    elements: int,
) -> dict[str, Any]:
    count = min(elements, flattened.numel())
    if count == 0:
        return {
            "scheme": "floor(i * tensor_elements / sample_elements)",
            "elements": 0,
            "indices_sha256": hashlib.sha256(b"").hexdigest(),
            "values_sha256": hashlib.sha256(b"").hexdigest(),
            "values": [],
        }
    indices = (
        torch.arange(count, device=flattened.device, dtype=torch.int64)
        * flattened.numel()
        // count
    )
    sampled = flattened.index_select(0, indices).detach().cpu().contiguous()
    cpu_indices = indices.cpu().contiguous()
    index_bytes = cpu_indices.numpy().tobytes()
    value_bytes = sampled.view(torch.uint8).numpy().tobytes()
    result = {
        "scheme": "floor(i * tensor_elements / sample_elements)",
        "elements": count,
        "indices_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "values_sha256": hashlib.sha256(value_bytes).hexdigest(),
        "values": sampled.float().tolist(),
    }
    del indices, sampled, cpu_indices
    return result


def _tensor_statistics(
    tensor: torch.Tensor,
    *,
    chunk_elements: int = STATISTICS_CHUNK_ELEMENTS,
    sample_elements: int = STRIDED_SAMPLE_ELEMENTS,
) -> dict[str, Any]:
    """Return full statistics while bounding each temporary FP32 conversion."""
    if chunk_elements < 1 or sample_elements < 1:
        raise ValueError("chunk and sample element counts must be positive")
    detached = tensor.detach()
    if not detached.is_contiguous():
        raise RuntimeError(
            "diagnostic gradient tensors must be contiguous; refusing an "
            "implicit full-tensor copy"
        )
    flattened = detached.view(-1)
    elements = flattened.numel()
    nonzero = 0
    finite = True
    maximum = 0.0
    absolute_sum = 0.0
    squared_sum = 0.0
    for start in range(0, elements, chunk_elements):
        source_chunk = flattened[start : start + chunk_elements]
        value_chunk = source_chunk.float()
        nonzero += int(torch.count_nonzero(source_chunk))
        finite = finite and bool(torch.isfinite(source_chunk).all())
        if value_chunk.numel():
            maximum = max(maximum, float(value_chunk.abs().max()))
            absolute_sum += float(
                value_chunk.abs().sum(dtype=torch.float64)
            )
            squared_sum += float(
                value_chunk.square().sum(dtype=torch.float64)
            )
        del source_chunk, value_chunk
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "elements": elements,
        "nonzero": nonzero,
        "nonzero_fraction": nonzero / elements if elements else 0.0,
        "l2": math.sqrt(squared_sum),
        "max_abs": maximum,
        "mean_abs": absolute_sum / elements if elements else 0.0,
        "finite": finite,
        "full_statistics_chunk_elements": chunk_elements,
        "strided_sample": _deterministic_strided_sample(
            flattened,
            elements=sample_elements,
        ),
    }


def _is_power_of_two(value: float) -> bool:
    if value <= 0.0 or not math.isfinite(value):
        return False
    significand, _exponent = math.frexp(value)
    return significand == 0.5


def _validate_loss_scales(scales: list[float] | tuple[float, ...]) -> None:
    if len(scales) < 2:
        raise ValueError("require at least two loss scales")
    if len(set(scales)) != len(scales):
        raise ValueError("loss scales must be distinct")
    if any(not _is_power_of_two(scale) for scale in scales):
        raise ValueError("loss scales must be finite positive powers of two")


def _authenticate_runtime_artifacts(
    *,
    route: str,
    forward: Path,
    projection: Path,
    control: Path,
) -> dict[str, dict[str, Any]]:
    selected_projection = os.environ.get("TK_FA4_LOWP_BWD_EXTENSION_SOURCE")
    if selected_projection is None:
        raise RuntimeError(
            "TK_FA4_LOWP_BWD_EXTENSION_SOURCE must select "
            "--projection-extension"
        )
    if Path(selected_projection).resolve() != projection.resolve():
        raise RuntimeError(
            "TK_FA4_LOWP_BWD_EXTENSION_SOURCE must select "
            "--projection-extension"
        )
    identities = {
        "forward": _authenticate_regular_file(
            forward,
            label="forward extension",
            expected_sha256=PINNED_ARTIFACTS["forward"][route][0],
            expected_bytes=PINNED_ARTIFACTS["forward"][route][1],
        ),
        "projection": _authenticate_regular_file(
            projection,
            label="projection extension",
            expected_sha256=PINNED_ARTIFACTS["projection"][0],
            expected_bytes=PINNED_ARTIFACTS["projection"][1],
        ),
        "control": _authenticate_regular_file(
            control,
            label="backward control",
            expected_sha256=PINNED_ARTIFACTS["control"][0],
            expected_bytes=PINNED_ARTIFACTS["control"][1],
        ),
    }
    loaded_projection = getattr(tk_interface, "_C_b300_lowp_bwd", None)
    loaded_path = getattr(loaded_projection, "__file__", None)
    if loaded_path is None or Path(loaded_path).resolve() != projection.resolve():
        raise RuntimeError(
            "the authenticated projection extension is not the extension "
            "loaded by tk_fa4.interface"
        )
    return identities


def _load_authenticated_torch_payload(
    receipt: Any,
    *,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path, expected_sha256, expected_bytes = _receipt_file_fields(
        receipt,
        label=label,
    )
    observed = _authenticate_regular_file(
        path,
        label=label,
        expected_sha256=expected_sha256,
        expected_bytes=expected_bytes,
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must contain one mapping")
    repeated = _authenticate_regular_file(
        path,
        label=label,
        expected_sha256=str(observed["sha256"]),
        expected_bytes=int(observed["bytes"]),
    )
    if repeated != observed:
        raise RuntimeError(f"{label} changed while loading")
    return payload, observed


def _require_model_configuration(
    configuration: Any,
    expected: dict[str, Any],
    *,
    label: str,
) -> None:
    if not isinstance(configuration, dict):
        raise RuntimeError(f"{label} configuration must be an object")
    mismatches = {
        key: (configuration.get(key), value)
        for key, value in expected.items()
        if configuration.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"{label} model configuration mismatch: {mismatches}")


def _validate_benchmark_payload(
    payload: dict[str, Any],
    sample: dict[str, Any],
    *,
    route: str,
    seed: int,
    model_configuration: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
    checkpoint: dict[str, Any],
    sources: dict[str, dict[str, Any]],
    sample_identity: dict[str, Any],
) -> None:
    """Fail closed on every benchmark field that selects the probe state."""
    if payload.get("schema") != BENCHMARK_SCHEMA:
        raise RuntimeError("benchmark result schema mismatch")
    if payload.get("route") != route:
        raise RuntimeError("benchmark result route mismatch")
    configuration = payload.get("configuration")
    _require_model_configuration(
        configuration,
        model_configuration,
        label="benchmark",
    )
    assert isinstance(configuration, dict)
    benchmark_loss_scale = configuration.get("loss_scale")
    if not isinstance(benchmark_loss_scale, (int, float)) or not _is_power_of_two(
        float(benchmark_loss_scale)
    ):
        raise RuntimeError("benchmark loss scale is not a positive power of two")
    if configuration.get("token_source") != "dolma":
        raise RuntimeError("gradient-floor diagnostic requires a Dolma benchmark")
    if not isinstance(payload.get("forward_topology"), dict):
        raise RuntimeError("benchmark result is missing its forward topology")
    if not isinstance(payload.get("backward_contract"), dict):
        raise RuntimeError("benchmark result is missing its backward contract")
    recorded_artifacts = payload.get("artifacts")
    recorded_sources = payload.get("source_files")
    if not isinstance(recorded_artifacts, dict):
        raise RuntimeError("benchmark artifact receipts are missing")
    if not isinstance(recorded_sources, dict):
        raise RuntimeError("benchmark source receipts are missing")
    for name, observed in artifacts.items():
        _require_identity_receipt(
            recorded_artifacts.get(name),
            observed,
            label=f"benchmark {name}",
        )
    for name, observed in sources.items():
        _require_identity_receipt(
            recorded_sources.get(name),
            observed,
            label=f"benchmark {name} source",
        )
    final_receipt = payload.get("final_checkpoint")
    _require_identity_receipt(
        final_receipt,
        checkpoint,
        label="benchmark final checkpoint",
    )
    assert isinstance(final_receipt, dict)
    if final_receipt.get("kind") != "post_trajectory_model_state":
        raise RuntimeError("benchmark final checkpoint kind mismatch")
    if final_receipt.get("serialized_state_layout") != "canonical_split_qkv":
        raise RuntimeError("benchmark final checkpoint state layout mismatch")
    if final_receipt.get("runtime_state_layout") != "split_qkv":
        raise RuntimeError("low-precision final checkpoint must use split QKV")
    _require_identity_receipt(
        payload.get("sample_artifact"),
        sample_identity,
        label="benchmark sample artifact",
    )

    if sample.get("schema") != SAMPLE_SCHEMA:
        raise RuntimeError("benchmark sample schema mismatch")
    if sample.get("route") != route:
        raise RuntimeError("benchmark sample route mismatch")
    comparison = sample.get("comparison_identity")
    if not isinstance(comparison, dict):
        raise RuntimeError("benchmark sample comparison identity is missing")
    if comparison.get("seed") != seed:
        raise RuntimeError("benchmark sample seed mismatch")
    if comparison.get("configuration") != model_configuration:
        raise RuntimeError("benchmark sample model configuration mismatch")
    if comparison.get("data") != payload.get("data"):
        raise RuntimeError("benchmark result/sample data receipts differ")
    initial_checkpoint = payload.get("checkpoint")
    sample_checkpoint = sample.get("checkpoint")
    if not isinstance(initial_checkpoint, dict) or not isinstance(
        sample_checkpoint, dict
    ):
        raise RuntimeError("benchmark initial checkpoint receipt is missing")
    for key in ("sha256", "bytes"):
        if sample_checkpoint.get(key) != initial_checkpoint.get(key):
            raise RuntimeError(f"benchmark initial checkpoint {key} mismatch")
        if comparison.get(f"checkpoint_{key}") != initial_checkpoint.get(key):
            raise RuntimeError(
                f"benchmark comparison checkpoint {key} mismatch"
            )
    warmups = configuration.get("warmups")
    measured = configuration.get("measured_updates")
    if (
        not isinstance(warmups, int)
        or not isinstance(measured, int)
        or warmups < 1
        or measured < 1
    ):
        raise RuntimeError("benchmark update-count receipt is invalid")
    losses = sample.get("losses")
    if not isinstance(losses, list) or len(losses) != warmups + measured:
        raise RuntimeError("benchmark sample loss trajectory length mismatch")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise RuntimeError("benchmark data receipt is missing")
    if data.get("updates_including_probe") != warmups + measured + 1:
        raise RuntimeError("benchmark heldout/update data count mismatch")
    if data.get("batch") != model_configuration["batch"]:
        raise RuntimeError("benchmark heldout batch mismatch")
    if data.get("sequence") != model_configuration["sequence"]:
        raise RuntimeError("benchmark heldout sequence mismatch")
    heldout = payload.get("heldout_loss")
    initial_diagnostic = sample.get("initial_diagnostic")
    final_diagnostic = sample.get("final_diagnostic")
    if not all(
        isinstance(value, dict)
        for value in (heldout, initial_diagnostic, final_diagnostic)
    ):
        raise RuntimeError("benchmark heldout diagnostics are missing")
    assert isinstance(heldout, dict)
    assert isinstance(initial_diagnostic, dict)
    assert isinstance(final_diagnostic, dict)
    if heldout.get("initial") != initial_diagnostic.get("loss"):
        raise RuntimeError("benchmark initial heldout loss receipt mismatch")
    if heldout.get("final") != final_diagnostic.get("loss"):
        raise RuntimeError("benchmark final heldout loss receipt mismatch")


def _authenticate_benchmark(
    path: Path,
    *,
    route: str,
    seed: int,
    model_configuration: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
    checkpoint_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    payload, benchmark_identity = _read_authenticated_json(
        path,
        label="benchmark result",
    )
    if payload.get("schema") != BENCHMARK_SCHEMA:
        raise RuntimeError("benchmark result schema mismatch")
    if payload.get("route") != route:
        raise RuntimeError("benchmark result route mismatch")
    final_receipt = payload.get("final_checkpoint")
    if not isinstance(final_receipt, dict):
        raise RuntimeError("benchmark did not save a final checkpoint")
    receipt_path, expected_sha256, expected_bytes = _receipt_file_fields(
        final_receipt,
        label="benchmark final checkpoint",
    )
    if receipt_path.resolve() != checkpoint_path.resolve():
        raise RuntimeError("--checkpoint is not the benchmark final checkpoint")
    checkpoint = _authenticate_regular_file(
        checkpoint_path,
        label="final checkpoint",
        expected_sha256=expected_sha256,
        expected_bytes=expected_bytes,
    )
    sample, sample_identity = _load_authenticated_torch_payload(
        payload.get("sample_artifact"),
        label="benchmark sample artifact",
    )
    source_paths = {
        "harness": Path(__file__).with_name("benchmark_llama12b_saturated.py"),
        "runtime": Path(__file__).with_name("benchmark_llama12b_e2e.py"),
    }
    sources = {
        name: _authenticate_regular_file(source, label=f"{name} source")
        for name, source in source_paths.items()
    }
    _validate_benchmark_payload(
        payload,
        sample,
        route=route,
        seed=seed,
        model_configuration=model_configuration,
        artifacts=artifacts,
        checkpoint=checkpoint,
        sources=sources,
        sample_identity=sample_identity,
    )
    return payload, benchmark_identity, checkpoint


def _load_model_state(
    model: Llama12B,
    path: Path,
    expected_identity: dict[str, Any],
) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise RuntimeError("final checkpoint must contain one state mapping")
    model.load_state_dict(state, strict=True)
    del state
    repeated = _authenticate_regular_file(
        path,
        label="final checkpoint",
        expected_sha256=str(expected_identity["sha256"]),
        expected_bytes=int(expected_identity["bytes"]),
    )
    if repeated != expected_identity:
        raise RuntimeError("final checkpoint changed while loading")
    return dict(repeated)


def _run_probe(
    model: Llama12B,
    runtime: Any,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    *,
    loss_scale: float,
    repeat: int,
) -> dict[str, Any]:
    runtime.loss_scale = loss_scale
    activate_model_forward_route(model)
    raw_first_reverse: dict[str, Any] = {}
    reverse_calls = 0
    original_run = runtime.backward.run

    def run_and_capture(*args: Any, **kwargs: Any) -> Any:
        nonlocal reverse_calls
        result = original_run(*args, **kwargs)
        if reverse_calls == 0:
            raw_first_reverse.update(
                {
                    "decoder_layer": model.config.layers - 1,
                    "dq": _tensor_statistics(runtime.backward.dq),
                    "dk": _tensor_statistics(runtime.backward.dk),
                    "dv": _tensor_statistics(runtime.backward.dv),
                }
            )
        reverse_calls += 1
        return result

    runtime.backward.run = run_and_capture
    hooked_gradients: dict[str, Any] = {}
    handles = []
    for layer_index in (0, model.config.layers - 1):
        attention = model.layers[layer_index].attention
        for projection in ("q", "k", "v"):
            parameter = getattr(attention.weights, projection)
            name = f"layers.{layer_index}.attention.weights.{projection}"

            def capture(
                gradient: torch.Tensor,
                *,
                gradient_name: str = name,
            ) -> None:
                hooked_gradients[gradient_name] = _tensor_statistics(gradient)

            handles.append(parameter.register_hook(capture))
    try:
        model.zero_grad(set_to_none=True)
        hidden, weight = _hidden_and_weight(model, tokens)
        loss = _loss(hidden, weight, targets)
        loss.backward()
        torch.cuda.synchronize()
        if reverse_calls != model.config.layers:
            raise RuntimeError(
                f"captured {reverse_calls} reverse attention calls, expected "
                f"{model.config.layers}"
            )
        required_hooks = 2 * 3
        if len(hooked_gradients) != required_hooks:
            raise RuntimeError(
                f"captured {len(hooked_gradients)} projection gradients, "
                f"expected {required_hooks}"
            )
        return {
            "loss_scale": loss_scale,
            "repeat": repeat,
            "loss": float(loss.detach()),
            "reverse_attention_calls": reverse_calls,
            "raw_first_reverse": raw_first_reverse,
            "projection_weight_gradients": hooked_gradients,
            "backward_contract": runtime.backward_contract(),
        }
    finally:
        model.zero_grad(set_to_none=True)
        runtime.backward.run = original_run
        for handle in handles:
            handle.remove()


def _contract_without_loss_scale(contract: dict[str, Any]) -> dict[str, Any]:
    # JSON normalization is a compact deep copy of the receipt-only mapping.
    normalized = json.loads(json.dumps(contract, sort_keys=True))
    scaling = normalized.get("scaling")
    if not isinstance(scaling, dict) or "loss_scale" not in scaling:
        raise RuntimeError("backward contract is missing scaling.loss_scale")
    del scaling["loss_scale"]
    return normalized


def _probe_gradient_hashes(probe: dict[str, Any]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, statistics in probe["raw_first_reverse"].items():
        if name == "decoder_layer":
            continue
        hashes[f"raw.{name}"] = statistics["strided_sample"]["values_sha256"]
    for name, statistics in probe["projection_weight_gradients"].items():
        hashes[f"weight.{name}"] = statistics["strided_sample"][
            "values_sha256"
        ]
    return hashes


def _probe_gradient_statistics(
    probe: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    statistics: dict[str, dict[str, Any]] = {}
    for name, value in probe["raw_first_reverse"].items():
        if name != "decoder_layer":
            statistics[f"raw.{name}"] = value
    for name, value in probe["projection_weight_gradients"].items():
        statistics[f"weight.{name}"] = value
    return statistics


def _compare_strided_statistics(
    candidate: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    candidate_values = torch.tensor(
        candidate["strided_sample"]["values"], dtype=torch.float64
    )
    reference_values = torch.tensor(
        reference["strided_sample"]["values"], dtype=torch.float64
    )
    if candidate_values.shape != reference_values.shape:
        raise RuntimeError("gradient strided sample shape changed")
    candidate_norm = float(candidate_values.norm())
    reference_norm = float(reference_values.norm())
    if candidate_norm == 0.0 and reference_norm == 0.0:
        cosine = 1.0
        relative_l2 = 0.0
        norm_ratio = 1.0
    else:
        denominator = max(reference_norm, 1.0e-30)
        cosine_denominator = max(candidate_norm * reference_norm, 1.0e-30)
        cosine = float(
            torch.dot(candidate_values, reference_values)
            / cosine_denominator
        )
        relative_l2 = float(
            (candidate_values - reference_values).norm() / denominator
        )
        norm_ratio = candidate_norm / denominator
    return {
        "cosine": cosine,
        "relative_l2": relative_l2,
        "norm_ratio": norm_ratio,
        "max_abs": float((candidate_values - reference_values).abs().max()),
        "candidate_nonzero": int(candidate.get("nonzero", 0)),
        "reference_nonzero": int(reference.get("nonzero", 0)),
        "revived_from_zero": bool(
            reference.get("nonzero", 0) == 0
            and candidate.get("nonzero", 0) != 0
        ),
    }


def _compare_probe_gradients(
    candidate: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    candidate_statistics = _probe_gradient_statistics(candidate)
    reference_statistics = _probe_gradient_statistics(reference)
    if set(candidate_statistics) != set(reference_statistics):
        raise RuntimeError("probe gradient sample keys changed")
    return {
        name: _compare_strided_statistics(
            candidate_statistics[name], reference_statistics[name]
        )
        for name in sorted(candidate_statistics)
    }


def _validate_probe_grid(
    probes: list[dict[str, Any]],
    *,
    loss_scales: list[float] | tuple[float, ...],
    repeats: int,
    benchmark_contract: dict[str, Any],
) -> dict[str, Any]:
    expected_coordinates = [
        (scale, repeat)
        for scale in loss_scales
        for repeat in range(repeats)
    ]
    actual_coordinates = [
        (probe.get("loss_scale"), probe.get("repeat")) for probe in probes
    ]
    if actual_coordinates != expected_coordinates:
        raise RuntimeError("probe scale/repeat grid is incomplete or reordered")
    losses = [float(probe["loss"]) for probe in probes]
    if not losses or any(not math.isfinite(loss) for loss in losses):
        raise RuntimeError("probe losses must all be finite")
    reference_loss = losses[0]
    if any(
        not math.isclose(
            loss,
            reference_loss,
            rel_tol=LOSS_RELATIVE_TOLERANCE,
            abs_tol=LOSS_ABSOLUTE_TOLERANCE,
        )
        for loss in losses[1:]
    ):
        raise RuntimeError("forward loss changed across backward-only probes")
    benchmark_normalized = _contract_without_loss_scale(benchmark_contract)
    first_probe_by_scale: dict[float, dict[str, Any]] = {}
    repeat_comparisons: dict[str, list[dict[str, Any]]] = {}
    exact_repeat_hashes = True
    for probe in probes:
        scale = float(probe["loss_scale"])
        contract = probe.get("backward_contract")
        if not isinstance(contract, dict):
            raise RuntimeError("probe is missing its backward contract")
        if contract.get("scaling", {}).get("loss_scale") != scale:
            raise RuntimeError("probe backward contract loss scale mismatch")
        if _contract_without_loss_scale(contract) != benchmark_normalized:
            raise RuntimeError(
                "probe backward contract differs from benchmark beyond "
                "scaling.loss_scale"
            )
        if scale not in first_probe_by_scale:
            first_probe_by_scale[scale] = probe
            continue
        reference_probe = first_probe_by_scale[scale]
        exact_repeat_hashes &= (
            _probe_gradient_hashes(probe)
            == _probe_gradient_hashes(reference_probe)
        )
        repeat_comparisons.setdefault(f"{scale:g}", []).append(
            {
                "repeat": probe["repeat"],
                "samples": _compare_probe_gradients(
                    probe, reference_probe
                ),
            }
        )
    repeat_stable = all(
        sample["cosine"] >= REPEAT_MINIMUM_COSINE
        and sample["relative_l2"] <= REPEAT_MAXIMUM_RELATIVE_L2
        for comparisons in repeat_comparisons.values()
        for comparison in comparisons
        for sample in comparison["samples"].values()
    )
    if not repeat_stable:
        raise RuntimeError(
            "gradient samples are not directionally stable across repeats"
        )
    base_scale = float(loss_scales[0])
    scale_comparisons = {
        f"{scale:g}": _compare_probe_gradients(
            first_probe_by_scale[scale], first_probe_by_scale[base_scale]
        )
        for scale in map(float, loss_scales[1:])
    }
    return {
        "loss_reference": reference_loss,
        "loss_absolute_tolerance": LOSS_ABSOLUTE_TOLERANCE,
        "loss_relative_tolerance": LOSS_RELATIVE_TOLERANCE,
        "loss_invariant": True,
        "gradient_sample_hashes_exactly_repeatable": exact_repeat_hashes,
        "gradient_samples_directionally_stable": repeat_stable,
        "repeat_minimum_cosine": REPEAT_MINIMUM_COSINE,
        "repeat_maximum_relative_l2": REPEAT_MAXIMUM_RELATIVE_L2,
        "repeat_comparisons": repeat_comparisons,
        "scale_comparisons_vs_lowest": scale_comparisons,
        "probe_grid_complete": True,
        "contracts_match_benchmark_except_loss_scale": True,
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Publish a complete JSON inode without an overwrite race."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise RuntimeError(f"temporary diagnostic output exists: {temporary}")
    try:
        with temporary.open("x") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise RuntimeError(
                f"refusing to overwrite diagnostic output: {path}"
            ) from error
        directory_descriptor = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def _tensor_payload_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    payload = value.view(torch.uint8).numpy().tobytes()
    del value
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    _load_runtime_dependencies()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", choices=("fp8", "mx"), required=True)
    parser.add_argument("--benchmark-result", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--loss-scales",
        type=float,
        nargs="+",
        default=(65_536.0, 131_072.0, 262_144.0),
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--forward-extension", type=Path)
    parser.add_argument(
        "--projection-extension", type=Path, default=DEFAULT_PROJECTION
    )
    parser.add_argument("--backward-control", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.repeats < 2:
        raise ValueError("require at least two repeats")
    _validate_loss_scales(args.loss_scales)
    if args.output.exists():
        raise RuntimeError("refusing to overwrite diagnostic output")
    if args.forward_extension is None:
        args.forward_extension = DEFAULT_FORWARDS[args.route]
    config = config_from_model_preset(batch=16, layers=16)
    artifacts = _authenticate_runtime_artifacts(
        route=args.route,
        forward=args.forward_extension,
        projection=args.projection_extension,
        control=args.backward_control,
    )
    benchmark, benchmark_identity, checkpoint_identity = (
        _authenticate_benchmark(
            args.benchmark_result,
            route=args.route,
            seed=args.seed,
            model_configuration=config.__dict__,
            artifacts=artifacts,
            checkpoint_path=args.checkpoint,
        )
    )
    benchmark_loss_scale = float(benchmark["configuration"]["loss_scale"])
    if args.loss_scales[0] != benchmark_loss_scale:
        raise RuntimeError(
            "the first diagnostic loss scale must reproduce the benchmark "
            f"loss scale {benchmark_loss_scale:g}"
        )
    if torch.cuda.device_count() != 1:
        raise RuntimeError("expose exactly one GPU to this diagnostic")

    torch.cuda.set_device(0)
    runtime, _topology = _runtime(
        args.route,
        config,
        args.forward_extension,
        args.backward_control,
        args.loss_scales[0],
    )
    benchmark_contract = benchmark["backward_contract"]
    if runtime.backward_contract() != benchmark_contract:
        raise RuntimeError(
            "diagnostic runtime backward contract does not reproduce "
            "the benchmark contract"
        )
    torch.manual_seed(args.seed)
    model = Llama12B(config, _make_llama3_rope(config), runtime)
    activate_model_forward_route(model)
    checkpoint = _load_model_state(
        model,
        args.checkpoint,
        checkpoint_identity,
    )
    data_count = int(benchmark["data"]["updates_including_probe"])
    tokens, targets, data_receipt = _dolma_batches(
        args.corpus,
        args.tokenizer,
        seed=args.seed,
        count=data_count,
        batch=config.batch,
        sequence=config.sequence,
    )
    if data_receipt != benchmark["data"]:
        raise RuntimeError(
            "regenerated Dolma trajectory/heldout receipt does not exactly "
            "match the benchmark"
        )
    probe_tokens = tokens[0].clone()
    probe_targets = targets[0].clone()
    heldout_receipt = {
        "selection": "packed_batch_index_0_before_all_training_updates",
        "seed": args.seed,
        "tokens_sha256": _tensor_payload_sha256(probe_tokens),
        "targets_sha256": _tensor_payload_sha256(probe_targets),
        "shape": list(probe_tokens.shape),
        "dtype": str(probe_tokens.dtype),
        "full_trajectory_data_receipt": data_receipt,
    }
    del tokens, targets

    # Compile the CE and extension paths before collecting the repeated scale
    # grid.  This warmup has no optimizer and leaves parameters unchanged.
    model.zero_grad(set_to_none=True)
    warmup_hidden, warmup_weight = _hidden_and_weight(model, probe_tokens)
    warmup_loss = _loss(warmup_hidden, warmup_weight, probe_targets)
    warmup_loss.backward()
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    warmup_loss_value = float(warmup_loss.detach())
    del warmup_hidden, warmup_weight, warmup_loss
    if dict(runtime.forward_topology) != benchmark["forward_topology"]:
        raise RuntimeError(
            "diagnostic runtime forward topology does not reproduce the "
            "post-launch benchmark topology"
        )

    probes = [
        _run_probe(
            model,
            runtime,
            probe_tokens,
            probe_targets,
            loss_scale=loss_scale,
            repeat=repeat,
        )
        for loss_scale in args.loss_scales
        for repeat in range(args.repeats)
    ]
    probe_gates = _validate_probe_grid(
        probes,
        loss_scales=args.loss_scales,
        repeats=args.repeats,
        benchmark_contract=benchmark_contract,
    )
    result = {
        "schema": DIAGNOSTIC_SCHEMA,
        "route": args.route,
        "configuration": config.__dict__,
        "seed": args.seed,
        "benchmark_result": benchmark_identity,
        "benchmark_trial_label": benchmark.get("trial_label"),
        "benchmark_source_files": benchmark["source_files"],
        "benchmark_sample_artifact": benchmark["sample_artifact"],
        "checkpoint": {
            **checkpoint,
            "kind": "post_trajectory_model_state",
            "serialized_state_layout": "canonical_split_qkv",
            "runtime_state_layout": "split_qkv",
            "source_benchmark_route": args.route,
        },
        "heldout": heldout_receipt,
        "loss_scales": args.loss_scales,
        "repeats": args.repeats,
        "artifacts": artifacts,
        "source_files": {
            "diagnostic": _source_identity(Path(__file__)),
            "runtime": _source_identity(
                Path(__file__).with_name("benchmark_llama12b_e2e.py")
            ),
        },
        "forward_topology": dict(runtime.forward_topology),
        "benchmark_backward_contract": benchmark_contract,
        "warmup_loss": warmup_loss_value,
        "probe_gates": probe_gates,
        "probes": probes,
    }
    del probe_tokens, probe_targets
    _atomic_write_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "probe_gates": probe_gates,
                "route": args.route,
                "schema": DIAGNOSTIC_SCHEMA,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
