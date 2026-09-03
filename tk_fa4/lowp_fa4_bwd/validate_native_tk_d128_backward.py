#!/usr/bin/env python3
"""Validate and benchmark a native TK D128 causal-GQA backward artifact.

The validator deliberately defines a narrow production boundary.  A candidate
must consume contiguous BSHD E4M3 publications plus the existing lstat/dstat
pages and write caller-owned contiguous BSHD BF16 destinations.  The named
``backward_*_out`` entry point semantically resets those destinations
internally, so every timed sample includes exactly its own reset-and-compute
boundary.  A route may implement that reset with physical clears or with a
complete unique-writer overwrite.  An
authenticated candidate is always checked at S128 against a deterministic
PyTorch reference before a larger shape is timed.

An optional CuTe comparator may be supplied as ``module:callable``.  The
callable must implement the same direct-output signature as the native entry
point; it is intentionally an adapter hook rather than an implicit dependency
on one generated CuTe artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import stat
import statistics
import string
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import torch


Q_HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128
HEAD_RATIO = Q_HEADS // KV_HEADS
INPUT_ENCODING_SCALE = 4.0
OUTPUT_ENCODING_SCALE = 4.0
PROBABILITY_LIFT_LOG2 = 8.0
SOFTMAX_SCALE = HEAD_DIM**-0.5
DIRECT_OUTPUT_ENTRYPOINT = "backward_e4m3_bshd_precomputed_out"
METADATA_ENTRYPOINT = "native_tk_d128_backward_metadata"
METADATA_SCHEMA = "tkfa4.native_tk_d128_backward.v1"

EXPECTED_SEMANTIC_METADATA: dict[str, Any] = {
    "schema": METADATA_SCHEMA,
    "backend": "thunderkittens_sm100a",
    "production_data_abi_compatible": True,
    "batch_values": (1, 2),
    "sequence": "dynamic_positive_multiple_of_128",
    "query_heads": Q_HEADS,
    "kv_heads": KV_HEADS,
    "head_ratio": HEAD_RATIO,
    "head_dim": HEAD_DIM,
    "threads": 512,
    "key_tile": 128,
    "query_tile": 128,
    "causal": True,
    "operand_dtype": "float8_e4m3fn",
    "operand_layout": "BSHD_contiguous",
    "encoding_scale": INPUT_ENCODING_SCALE,
    "lstat_abi": "8-LSE*log2(e)",
    "dstat_abi": "-16*sum(O*dO)",
    "stats_layout": "B,Hq,1,S_fp32_contiguous",
    "public_softmax_scale": "natural",
    "internal_beta_divisor": 16.0,
    "gradient_epilogue_scale": 1.0 / 256.0,
    "output_dtype": "bfloat16_additive",
    "output_layout": "BSHD_contiguous",
    "output_encoding_scale": OUTPUT_ENCODING_SCALE,
    "caller_owned_output_api": True,
    "caller_zeros_outputs_for_main": True,
    "backward_out_clears_outputs": True,
    "direct_output_entrypoint": DIRECT_OUTPUT_ENTRYPOINT,
}


@dataclass(frozen=True)
class Shape:
    """The only production geometry accepted by this validator."""

    batch: int
    sequence: int
    q_heads: int = Q_HEADS
    kv_heads: int = KV_HEADS
    head_dim: int = HEAD_DIM

    def __post_init__(self) -> None:
        if type(self.batch) is not int or self.batch not in (1, 2):
            raise ValueError("batch must be exactly 1 or 2")
        if (
            type(self.sequence) is not int
            or self.sequence <= 0
            or self.sequence % 128
        ):
            raise ValueError(
                "sequence must be a positive integer multiple of 128"
            )
        if (
            self.q_heads != Q_HEADS
            or self.kv_heads != KV_HEADS
            or self.head_dim != HEAD_DIM
        ):
            raise ValueError("shape must be Hq32/Hkv8/D128")

    @property
    def q_shape(self) -> tuple[int, int, int, int]:
        return (self.batch, self.sequence, self.q_heads, self.head_dim)

    @property
    def kv_shape(self) -> tuple[int, int, int, int]:
        return (self.batch, self.sequence, self.kv_heads, self.head_dim)

    @property
    def stats_shape(self) -> tuple[int, int, int, int]:
        return (self.batch, self.q_heads, 1, self.sequence)

    def as_dict(self) -> dict[str, int]:
        return {
            "batch": self.batch,
            "sequence": self.sequence,
            "q_heads": self.q_heads,
            "kv_heads": self.kv_heads,
            "head_dim": self.head_dim,
        }


@dataclass
class DirectOutputs:
    dq: torch.Tensor
    dk: torch.Tensor
    dv: torch.Tensor

    @classmethod
    def allocate(
        cls,
        shape: Shape,
        *,
        device: torch.device | str,
    ) -> DirectOutputs:
        return cls(
            dq=torch.zeros(shape.q_shape, device=device, dtype=torch.bfloat16),
            dk=torch.zeros(shape.kv_shape, device=device, dtype=torch.bfloat16),
            dv=torch.zeros(shape.kv_shape, device=device, dtype=torch.bfloat16),
        )

    def zero_(self) -> None:
        self.dq.zero_()
        self.dk.zero_()
        self.dv.zero_()

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.dq, self.dk, self.dv


@dataclass
class RepresentedState:
    shape: Shape
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    dout: torch.Tensor
    lstat: torch.Tensor
    dstat: torch.Tensor


@dataclass
class ReferenceGradients:
    dq: torch.Tensor
    dk: torch.Tensor
    dv: torch.Tensor

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.dq, self.dk, self.dv


def _validate_sha256(value: str, *, label: str) -> str:
    normalized = value.lower()
    if (
        len(normalized) != 64
        or any(character not in string.hexdigits for character in normalized)
    ):
        raise ValueError(f"{label} must contain 64 hexadecimal digits")
    return normalized


def _stable_file_identity(
    path: Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
    label: str,
) -> dict[str, int | str]:
    """Authenticate one unchanged regular, non-symlink file image."""
    expected_digest = _validate_sha256(expected_sha256, label="SHA-256")
    if type(expected_bytes) is not int or expected_bytes <= 0:
        raise ValueError(f"{label} expected byte count must be positive")
    try:
        before = path.lstat()
    except OSError as error:
        raise RuntimeError(f"cannot stat {label}: {path}") from error
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError(
            f"{label} must be a regular non-symlink file: {path}"
        )
    resolved = path.resolve(strict=True)
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    after = path.lstat()
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(
        getattr(before, field) != getattr(after, field)
        for field in stable_fields
    ):
        raise RuntimeError(f"{label} changed while hashing: {path}")
    actual_digest = digest.hexdigest()
    if after.st_size != expected_bytes or actual_digest != expected_digest:
        raise RuntimeError(
            f"{label} identity mismatch: bytes={after.st_size}, "
            f"sha256={actual_digest}"
        )
    return {
        "path": str(resolved),
        "sha256": actual_digest,
        "bytes": after.st_size,
        "device": after.st_dev,
        "inode": after.st_ino,
        "mtime_ns": after.st_mtime_ns,
    }


def _looks_like_path(value: str) -> bool:
    return (
        Path(value).is_absolute()
        or "/" in value
        or "\\" in value
        or value.endswith(".so")
    )


def load_authenticated_extension(
    extension: str,
    *,
    expected_sha256: str,
    expected_bytes: int,
    module_name: str | None = None,
) -> tuple[ModuleType, dict[str, int | str]]:
    """Load an exact native extension by absolute path or importable module."""
    if _looks_like_path(extension):
        requested = Path(extension)
        if not requested.is_absolute():
            raise ValueError("an extension path must be absolute")
        identity_before = _stable_file_identity(
            requested,
            expected_sha256=expected_sha256,
            expected_bytes=expected_bytes,
            label="native extension",
        )
        resolved = Path(str(identity_before["path"]))
        selected_name = module_name or resolved.name.split(".", 1)[0]
        if not selected_name or selected_name.rsplit(".", 1)[-1] == "":
            raise ValueError("module name must be nonempty")
        module_leaf = selected_name.rsplit(".", 1)[-1]
        if not (
            resolved.name == f"{module_leaf}.so"
            or resolved.name.startswith(f"{module_leaf}.")
        ):
            raise RuntimeError(
                "extension basename does not bind the requested module name: "
                f"{resolved.name!r} vs {selected_name!r}"
            )
        existing = sys.modules.get(selected_name)
        if existing is not None:
            existing_file = getattr(existing, "__file__", None)
            if (
                existing_file is None
                or Path(existing_file).resolve(strict=True) != resolved
            ):
                raise RuntimeError(
                    f"module {selected_name!r} is already loaded from "
                    f"{existing_file!r}"
                )
            loaded = existing
        else:
            spec = importlib.util.spec_from_file_location(
                selected_name,
                resolved,
            )
            if spec is None or spec.loader is None:
                raise RuntimeError(f"cannot load extension spec from {resolved}")
            loaded = importlib.util.module_from_spec(spec)
            sys.modules[selected_name] = loaded
            try:
                spec.loader.exec_module(loaded)
            except BaseException:
                sys.modules.pop(selected_name, None)
                raise
    else:
        if module_name is not None:
            raise ValueError(
                "--module-name is only valid when --extension is a path"
            )
        spec = importlib.util.find_spec(extension)
        if spec is None or spec.origin is None:
            raise RuntimeError(f"cannot resolve extension module {extension!r}")
        origin = Path(spec.origin)
        identity_before = _stable_file_identity(
            origin,
            expected_sha256=expected_sha256,
            expected_bytes=expected_bytes,
            label="native extension",
        )
        resolved = Path(str(identity_before["path"]))
        loaded = importlib.import_module(extension)
        loaded_file = getattr(loaded, "__file__", None)
        if loaded_file is None or Path(loaded_file).resolve(strict=True) != resolved:
            raise RuntimeError(
                f"loaded module origin mismatch: {loaded_file!r} != {resolved}"
            )
    identity_after = _stable_file_identity(
        resolved,
        expected_sha256=expected_sha256,
        expected_bytes=expected_bytes,
        label="native extension",
    )
    if identity_after != identity_before:
        raise RuntimeError(f"native extension changed while loading: {resolved}")
    loaded._tk_fa4_loaded_artifact_identity = dict(identity_before)
    return loaded, identity_before


def _resolve_declared_source(
    source_file: str,
    *,
    source_root: Path,
) -> Path:
    declared = Path(source_file)
    if declared.is_absolute():
        return declared
    return source_root.resolve(strict=True) / declared


def require_extension_metadata(
    extension: Any,
    *,
    expected_source_identity: str,
    expected_source_sha256: str,
    expected_source_bytes: int,
    source_root: Path,
) -> tuple[dict[str, Any], dict[str, int | str]]:
    """Require exact semantic metadata and authenticate its source file."""
    if (
        not isinstance(expected_source_identity, str)
        or not expected_source_identity
        or expected_source_identity.strip() != expected_source_identity
    ):
        raise ValueError("expected source identity must be a nonempty string")
    metadata_fn = getattr(extension, METADATA_ENTRYPOINT, None)
    if not callable(metadata_fn):
        raise RuntimeError(
            f"native extension lacks {METADATA_ENTRYPOINT}()"
        )
    raw_metadata = metadata_fn()
    if not isinstance(raw_metadata, Mapping):
        raise RuntimeError("native D128 metadata must be a mapping")
    metadata = dict(raw_metadata)
    expected = {
        **EXPECTED_SEMANTIC_METADATA,
        "source_identity": expected_source_identity,
    }
    missing = {*expected, "source_file", "topology"} - set(metadata)
    if missing:
        raise RuntimeError(
            f"native D128 metadata is missing fields: {sorted(missing)}"
        )
    mismatches = {
        field: {"actual": metadata[field], "expected": expected_value}
        for field, expected_value in expected.items()
        if (
            metadata[field] != expected_value
            or type(metadata[field]) is not type(expected_value)
        )
    }
    source_file = metadata["source_file"]
    if not isinstance(source_file, str) or not source_file:
        mismatches["source_file"] = {
            "actual": source_file,
            "expected": "nonempty path string",
        }
    topology = metadata["topology"]
    if not isinstance(topology, str) or not topology:
        mismatches["topology"] = {
            "actual": topology,
            "expected": "nonempty topology string",
        }
    if mismatches:
        raise RuntimeError(
            "native D128 metadata does not match the direct-output ABI: "
            f"{mismatches}"
        )
    source_path = _resolve_declared_source(
        source_file,
        source_root=source_root,
    )
    source_identity = _stable_file_identity(
        source_path,
        expected_sha256=expected_source_sha256,
        expected_bytes=expected_source_bytes,
        label="declared native source",
    )
    return metadata, source_identity


def require_direct_output_entrypoint(
    extension: Any,
    metadata: Mapping[str, Any],
) -> Callable[..., Any]:
    name = metadata.get("direct_output_entrypoint")
    if name != DIRECT_OUTPUT_ENTRYPOINT or type(name) is not str:
        raise RuntimeError(
            "native D128 metadata does not name the required direct-output ABI"
        )
    entrypoint = getattr(extension, name, None)
    if not callable(entrypoint):
        raise RuntimeError(f"native extension lacks callable {name}")
    return entrypoint


def _represented_e4m3(
    shape: tuple[int, ...],
    *,
    standard_deviation: float,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    source = torch.randn(
        shape,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    source.mul_(standard_deviation)
    source = source.bfloat16()
    return source.float().mul_(INPUT_ENCODING_SCALE).to(torch.float8_e4m3fn)


def _decoded_heads(
    tensor: torch.Tensor,
    *,
    expand_gqa: bool,
) -> torch.Tensor:
    values = tensor.float().div_(INPUT_ENCODING_SCALE).permute(0, 2, 1, 3)
    if expand_gqa:
        values = values.repeat_interleave(HEAD_RATIO, dim=1)
    return values


def _production_stats(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    *,
    query_chunk: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build production lstat/dstat without materializing an S-by-S graph."""
    batch, sequence, q_heads, depth = q.shape
    qh = _decoded_heads(q, expand_gqa=False)
    kh = _decoded_heads(k, expand_gqa=True)
    vh = _decoded_heads(v, expand_gqa=True)
    doh = _decoded_heads(dout, expand_gqa=False)
    lstat = torch.empty(
        (batch, q_heads, 1, sequence),
        device=q.device,
        dtype=torch.float32,
    )
    dstat = torch.empty_like(lstat)
    key_positions = torch.arange(sequence, device=q.device)
    for start in range(0, sequence, query_chunk):
        stop = min(start + query_chunk, sequence)
        scores = torch.matmul(
            qh[:, :, start:stop],
            kh.transpose(-1, -2),
        ).mul_(depth**-0.5)
        query_positions = torch.arange(start, stop, device=q.device)
        scores.masked_fill_(
            key_positions.view(1, 1, 1, sequence)
            > query_positions.view(1, 1, stop - start, 1),
            float("-inf"),
        )
        lse = torch.logsumexp(scores, dim=-1)
        probability = torch.softmax(scores, dim=-1)
        output = torch.matmul(probability, vh)
        lstat[:, :, 0, start:stop] = (
            PROBABILITY_LIFT_LOG2 - lse * math.log2(math.e)
        )
        dstat[:, :, 0, start:stop] = -16.0 * (
            output * doh[:, :, start:stop]
        ).sum(dim=-1)
    return lstat.contiguous(), dstat.contiguous()


def make_represented_state(
    shape: Shape,
    *,
    device: torch.device | str,
    seed: int,
) -> RepresentedState:
    selected_device = torch.device(device)
    generator = torch.Generator(device=selected_device)
    generator.manual_seed(seed)
    q = _represented_e4m3(
        shape.q_shape,
        standard_deviation=0.25,
        device=selected_device,
        generator=generator,
    )
    k = _represented_e4m3(
        shape.kv_shape,
        standard_deviation=0.25,
        device=selected_device,
        generator=generator,
    )
    v = _represented_e4m3(
        shape.kv_shape,
        standard_deviation=0.25,
        device=selected_device,
        generator=generator,
    )
    dout = _represented_e4m3(
        shape.q_shape,
        standard_deviation=0.10,
        device=selected_device,
        generator=generator,
    )
    with torch.inference_mode():
        lstat, dstat = _production_stats(q, k, v, dout)
    return RepresentedState(
        shape=shape,
        q=q,
        k=k,
        v=v,
        dout=dout,
        lstat=lstat,
        dstat=dstat,
    )


def represented_e4m3_causal_reference(
    state: RepresentedState,
) -> ReferenceGradients:
    """Return deterministic FP32 gradients of the represented S128 problem."""
    if state.shape.sequence != 128:
        raise ValueError("the PyTorch reference is intentionally limited to S128")
    qh = _decoded_heads(state.q, expand_gqa=False)
    kh_owner = _decoded_heads(state.k, expand_gqa=False)
    vh_owner = _decoded_heads(state.v, expand_gqa=False)
    kh = kh_owner.repeat_interleave(HEAD_RATIO, dim=1)
    vh = vh_owner.repeat_interleave(HEAD_RATIO, dim=1)
    doh = _decoded_heads(state.dout, expand_gqa=False)
    scores = torch.matmul(qh, kh.transpose(-1, -2)).mul_(SOFTMAX_SCALE)
    causal_mask = torch.ones(
        (state.shape.sequence, state.shape.sequence),
        device=state.q.device,
        dtype=torch.bool,
    ).triu_(1)
    scores.masked_fill_(causal_mask, float("-inf"))
    probability = torch.softmax(scores, dim=-1)
    dp = torch.matmul(doh, vh.transpose(-1, -2))
    ds = probability * (
        dp - (dp * probability).sum(dim=-1, keepdim=True)
    )
    dq = torch.matmul(ds, kh).mul_(SOFTMAX_SCALE)
    dk_per_q_head = torch.matmul(ds.transpose(-1, -2), qh).mul_(
        SOFTMAX_SCALE
    )
    dv_per_q_head = torch.matmul(probability.transpose(-1, -2), doh)
    owner_shape = (
        state.shape.batch,
        state.shape.kv_heads,
        HEAD_RATIO,
        state.shape.sequence,
        state.shape.head_dim,
    )
    dk = dk_per_q_head.reshape(owner_shape).sum(dim=2)
    dv = dv_per_q_head.reshape(owner_shape).sum(dim=2)
    return ReferenceGradients(
        dq=dq.permute(0, 2, 1, 3).contiguous(),
        dk=dk.permute(0, 2, 1, 3).contiguous(),
        dv=dv.permute(0, 2, 1, 3).contiguous(),
    )


def _require_state_contract(state: RepresentedState) -> None:
    expected = (
        ("q", state.q, state.shape.q_shape, torch.float8_e4m3fn),
        ("k", state.k, state.shape.kv_shape, torch.float8_e4m3fn),
        ("v", state.v, state.shape.kv_shape, torch.float8_e4m3fn),
        ("dout", state.dout, state.shape.q_shape, torch.float8_e4m3fn),
        ("lstat", state.lstat, state.shape.stats_shape, torch.float32),
        ("dstat", state.dstat, state.shape.stats_shape, torch.float32),
    )
    device = state.q.device
    for name, tensor, tensor_shape, dtype in expected:
        if (
            tuple(tensor.shape) != tensor_shape
            or tensor.dtype != dtype
            or tensor.device != device
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"{name} violates production tensor ABI: shape="
                f"{tuple(tensor.shape)}, dtype={tensor.dtype}, "
                f"device={tensor.device}, contiguous={tensor.is_contiguous()}"
            )


def launch_reset_inclusive(
    entrypoint: Callable[..., Any],
    state: RepresentedState,
    outputs: DirectOutputs,
) -> None:
    """Invoke the direct ABI, whose boundary semantically resets outputs."""
    _require_state_contract(state)
    entrypoint(
        state.q,
        state.k,
        state.v,
        state.dout,
        state.lstat,
        state.dstat,
        outputs.dq,
        outputs.dk,
        outputs.dv,
        SOFTMAX_SCALE,
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, Any]:
    reference_f = reference.float().reshape(-1)
    actual_f = actual.float().reshape(-1)
    difference = actual_f - reference_f
    reference_norm = reference_f.norm().clamp_min(1.0e-30)
    actual_norm = actual_f.norm().clamp_min(1.0e-30)
    return {
        "reference_finite": bool(torch.isfinite(reference_f).all()),
        "actual_finite": bool(torch.isfinite(actual_f).all()),
        "cosine": float(
            torch.dot(reference_f, actual_f) / (reference_norm * actual_norm)
        ),
        "relative_l2": float(difference.norm() / reference_norm),
        "norm_ratio": float(actual_norm / reference_norm),
        "max_abs": float(difference.abs().max()),
    }


def _gradient_metrics(
    reference: ReferenceGradients,
    outputs: DirectOutputs,
) -> dict[str, dict[str, Any]]:
    decoded = tuple(
        tensor.float().div_(OUTPUT_ENCODING_SCALE)
        for tensor in outputs.tensors()
    )
    return {
        name: _metrics(reference_tensor, actual_tensor)
        for name, reference_tensor, actual_tensor in zip(
            ("dq", "dk", "dv"),
            reference.tensors(),
            decoded,
            strict=True,
        )
    }


def _finite_and_nontrivial(outputs: DirectOutputs) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, tensor in zip(("dq", "dk", "dv"), outputs.tensors(), strict=True):
        result[name] = {
            "finite": bool(torch.isfinite(tensor.float()).all()),
            "nonzero": int(torch.count_nonzero(tensor)),
        }
    result["passed"] = all(
        item["finite"] and item["nonzero"] > 0
        for name, item in result.items()
        if name != "passed"
    )
    return result


def check_exact_zero_dout(
    entrypoint: Callable[..., Any],
    state: RepresentedState,
) -> dict[str, Any]:
    zero_state = RepresentedState(
        shape=state.shape,
        q=state.q,
        k=state.k,
        v=state.v,
        dout=torch.zeros_like(state.dout),
        lstat=state.lstat,
        dstat=torch.zeros_like(state.dstat),
    )
    outputs = DirectOutputs.allocate(state.shape, device=state.q.device)
    for tensor in outputs.tensors():
        tensor.fill_(1.0)
    launch_reset_inclusive(entrypoint, zero_state, outputs)
    _synchronize(state.q.device)
    nonzero = {
        name: int(torch.count_nonzero(tensor))
        for name, tensor in zip(("dq", "dk", "dv"), outputs.tensors(), strict=True)
    }
    finite = {
        name: bool(torch.isfinite(tensor.float()).all())
        for name, tensor in zip(("dq", "dk", "dv"), outputs.tensors(), strict=True)
    }
    return {
        "exact_nonzero_counts": nonzero,
        "finite": finite,
        "passed": all(value == 0 for value in nonzero.values())
        and all(finite.values()),
    }


def validate_small_reference(
    entrypoint: Callable[..., Any],
    *,
    batch: int,
    device: torch.device | str,
    seed: int,
    minimum_cosine: float,
    maximum_relative_l2: float,
    minimum_norm_ratio: float,
    maximum_norm_ratio: float,
) -> tuple[dict[str, Any], RepresentedState]:
    state = make_represented_state(
        Shape(batch=batch, sequence=128),
        device=device,
        seed=seed,
    )
    with torch.inference_mode():
        reference = represented_e4m3_causal_reference(state)
    outputs = DirectOutputs.allocate(state.shape, device=state.q.device)
    launch_reset_inclusive(entrypoint, state, outputs)
    _synchronize(state.q.device)
    metrics = _gradient_metrics(reference, outputs)
    passed = all(
        values["reference_finite"]
        and values["actual_finite"]
        and values["cosine"] >= minimum_cosine
        and values["relative_l2"] <= maximum_relative_l2
        and minimum_norm_ratio <= values["norm_ratio"] <= maximum_norm_ratio
        for values in metrics.values()
    )
    return {
        "shape": state.shape.as_dict(),
        "metrics": metrics,
        "thresholds": {
            "minimum_cosine": minimum_cosine,
            "maximum_relative_l2": maximum_relative_l2,
            "minimum_norm_ratio": minimum_norm_ratio,
            "maximum_norm_ratio": maximum_norm_ratio,
        },
        "passed": passed,
    }, state


def time_reset_inclusive(
    runners: Mapping[str, Callable[[], None]],
    *,
    device: torch.device,
    warmups: int,
    samples: int,
) -> dict[str, dict[str, Any]]:
    """Time rotated clear-plus-launch boundaries on one device stream."""
    if warmups < 0 or samples <= 0:
        raise ValueError("warmups must be nonnegative and samples positive")
    if not runners:
        raise ValueError("at least one timing runner is required")
    names = tuple(runners)
    for iteration in range(warmups):
        for offset in range(len(names)):
            runners[names[(iteration + offset) % len(names)]]()
    _synchronize(device)
    observed: dict[str, list[float]] = {name: [] for name in names}
    for iteration in range(samples):
        for offset in range(len(names)):
            name = names[(iteration + offset) % len(names)]
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                runners[name]()
                end.record()
                end.synchronize()
                elapsed_us = float(start.elapsed_time(end) * 1000.0)
            else:
                started = time.perf_counter_ns()
                runners[name]()
                elapsed_us = (time.perf_counter_ns() - started) / 1000.0
            observed[name].append(elapsed_us)
    return {
        name: {
            "median_us": statistics.median(values),
            "minimum_us": min(values),
            "maximum_us": max(values),
            "samples_us": values,
        }
        for name, values in observed.items()
    }


def load_optional_comparator(
    specification: str,
) -> tuple[Callable[..., Any], dict[str, Any]]:
    module_name, separator, attribute = specification.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("CuTe comparator must be specified as module:callable")
    module = importlib.import_module(module_name)
    comparator = getattr(module, attribute, None)
    if not callable(comparator):
        raise RuntimeError(f"CuTe comparator {specification!r} is not callable")
    module_file = getattr(module, "__file__", None)
    identity: dict[str, Any] = {
        "module": module_name,
        "callable": attribute,
        "file": module_file,
    }
    if module_file is not None:
        path = Path(module_file).resolve(strict=True)
        payload = path.read_bytes()
        identity.update(
            {
                "file": str(path),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
        )
    return comparator, identity


def run_validation(
    entrypoint: Callable[..., Any],
    *,
    shape: Shape,
    device: torch.device,
    seed: int,
    warmups: int,
    samples: int,
    minimum_cosine: float,
    maximum_relative_l2: float,
    minimum_norm_ratio: float,
    maximum_norm_ratio: float,
    cute_comparator: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    small, small_state = validate_small_reference(
        entrypoint,
        batch=shape.batch,
        device=device,
        seed=seed,
        minimum_cosine=minimum_cosine,
        maximum_relative_l2=maximum_relative_l2,
        minimum_norm_ratio=minimum_norm_ratio,
        maximum_norm_ratio=maximum_norm_ratio,
    )
    state = (
        small_state
        if shape.sequence == 128
        else make_represented_state(shape, device=device, seed=seed + 1)
    )
    outputs = DirectOutputs.allocate(shape, device=device)
    launch_reset_inclusive(entrypoint, state, outputs)
    _synchronize(device)
    target_quality = _finite_and_nontrivial(outputs)
    zero_dout = check_exact_zero_dout(entrypoint, state)

    timing_runners: dict[str, Callable[[], None]] = {
        "native_tk_reset_inclusive": lambda: launch_reset_inclusive(
            entrypoint,
            state,
            outputs,
        )
    }
    comparator_report: dict[str, Any] | None = None
    if cute_comparator is not None:
        if shape.sequence != 4096:
            raise ValueError("the optional CuTe comparator is restricted to S4096")
        comparator_outputs = DirectOutputs.allocate(shape, device=device)
        launch_reset_inclusive(cute_comparator, state, comparator_outputs)
        _synchronize(device)
        comparator_report = {
            "finite_and_nontrivial": _finite_and_nontrivial(comparator_outputs),
            "native_vs_cute": {
                name: _metrics(
                    cute.float().div(OUTPUT_ENCODING_SCALE),
                    native.float().div(OUTPUT_ENCODING_SCALE),
                )
                for name, cute, native in zip(
                    ("dq", "dk", "dv"),
                    comparator_outputs.tensors(),
                    outputs.tensors(),
                    strict=True,
                )
            },
        }
        timing_runners["cute_reset_inclusive"] = lambda: launch_reset_inclusive(
            cute_comparator,
            state,
            comparator_outputs,
        )
    timing = time_reset_inclusive(
        timing_runners,
        device=device,
        warmups=warmups,
        samples=samples,
    )
    if comparator_report is not None:
        comparator_report["native_speedup_vs_cute"] = (
            timing["cute_reset_inclusive"]["median_us"]
            / timing["native_tk_reset_inclusive"]["median_us"]
        )
    passed = small["passed"] and target_quality["passed"] and zero_dout["passed"]
    return {
        "status": "passed" if passed else "failed",
        "shape": shape.as_dict(),
        "small_s128_reference": small,
        "target_finite_and_nontrivial": target_quality,
        "exact_zero_dout": zero_dout,
        "timing": {
            "protocol": (
                "rotated CUDA-event timing; each direct-output launch "
                "internally resets dq/dk/dv before computing"
            ),
            "measurements": timing,
        },
        "cute_comparator": comparator_report,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extension",
        required=True,
        help="absolute .so path or importable native module",
    )
    parser.add_argument("--module-name")
    parser.add_argument("--artifact-sha256", required=True)
    parser.add_argument("--artifact-bytes", required=True, type=int)
    parser.add_argument(
        "--source-identity",
        required=True,
        help="exact expected metadata source_identity for this candidate",
    )
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--source-bytes", required=True, type=int)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path.cwd(),
        help="base directory for a relative metadata source_file",
    )
    parser.add_argument("--batch", type=int, choices=(1, 2), default=1)
    parser.add_argument("--sequence", type=int, default=4096)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--minimum-cosine", type=float, default=0.9999)
    parser.add_argument("--maximum-relative-l2", type=float, default=0.02)
    parser.add_argument("--minimum-norm-ratio", type=float, default=0.995)
    parser.add_argument("--maximum-norm-ratio", type=float, default=1.005)
    parser.add_argument(
        "--cute-comparator",
        help="optional S4096 direct-output adapter as module:callable",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    shape = Shape(batch=args.batch, sequence=args.sequence)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("the native D128 validator requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    extension, artifact_identity = load_authenticated_extension(
        args.extension,
        expected_sha256=args.artifact_sha256,
        expected_bytes=args.artifact_bytes,
        module_name=args.module_name,
    )
    metadata, source_identity = require_extension_metadata(
        extension,
        expected_source_identity=args.source_identity,
        expected_source_sha256=args.source_sha256,
        expected_source_bytes=args.source_bytes,
        source_root=args.source_root,
    )
    entrypoint = require_direct_output_entrypoint(extension, metadata)
    cute_comparator = None
    cute_identity = None
    if args.cute_comparator is not None:
        cute_comparator, cute_identity = load_optional_comparator(
            args.cute_comparator
        )
    result = run_validation(
        entrypoint,
        shape=shape,
        device=device,
        seed=args.seed,
        warmups=args.warmups,
        samples=args.samples,
        minimum_cosine=args.minimum_cosine,
        maximum_relative_l2=args.maximum_relative_l2,
        minimum_norm_ratio=args.minimum_norm_ratio,
        maximum_norm_ratio=args.maximum_norm_ratio,
        cute_comparator=cute_comparator,
    )
    document = {
        "schema": "tkfa4.validate_native_tk_d128_backward.v1",
        **result,
        "artifact": artifact_identity,
        "source": source_identity,
        "extension_metadata": metadata,
        "policy": {
            "input": "represented_e4m3_q_k_v_dout_x4",
            "statistics": "production_lstat_dstat",
            "output": "caller_owned_backward_out_reset_bf16_x4",
            "direct_output_entrypoint": DIRECT_OUTPUT_ENTRYPOINT,
            "mandatory_small_reference_sequence": 128,
        },
        "cute_comparator_identity": cute_identity,
        "device": {
            "name": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
        },
    }
    rendered = json.dumps(document, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    if document["status"] != "passed":
        raise RuntimeError("native TK D128 backward validation failed")


if __name__ == "__main__":
    main()
