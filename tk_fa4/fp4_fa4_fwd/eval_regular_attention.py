#!/usr/bin/env python3
"""Evaluate the retained FP4 attention kernel in a regular-attention model.

The default target is a CIFAR-10 ViT-B/16 checkpoint. ViT uses 197 tokens,
12 heads, and D64 at its native 224x224 resolution. The adapter:

* pads heads and head dimension with zeros;
* scales Q by sqrt(128 / 64) to preserve the original attention scale;
* pads the token axis to the extension specialization and uses one
  otherwise-unused dimension to make padded keys receive a large negative
  score.

Higher image resolutions use ViT positional interpolation and exercise
long-sequence kernel specializations with real model activations.

This is an accuracy harness.  Runtime includes dynamic Q/K/V quantization
and is not an end-to-end performance measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import types
from collections import defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_OUTPUT = Path(
    "results/fp4_fa4_downstream_20260728/vit_cifar10.json"
)
DEFAULT_HAO_ROOT = REPO_ROOT / "third_party/hao_flash_attention_fp4"
E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
SIGNED_E2M1_LEVELS = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)
E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
E4M3_MIN_SUBNORMAL = 2.0**-9
E4M3_MIN_NORMAL = 2.0**-6
E4M3_MAX = 448.0
LOG2_E = math.log2(math.e)
LOG2_FP4_MAX = math.log2(6.0)
MX_E8M0_RTE_SHIFT = 1.0 - math.log2(1.5)
MX_QUANT_MODES = {
    "rte": 0,
    "ceil": 1,
    "floor": 2,
    "up-1-8": 3,
    "up-1-4": 4,
    "up-3-8": 5,
    "l2-oracle": 6,
}


def add_asset_identity_arguments(
    parser: argparse.ArgumentParser,
    roles: tuple[str, ...] = ("model", "dataset"),
) -> None:
    for role in roles:
        parser.add_argument(f"--{role}-identifier")
        parser.add_argument(f"--{role}-revision")
        parser.add_argument(f"--{role}-tree-sha256")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def authenticate_asset_record(
    assets: object,
    name: str,
    role: str,
    expected_root: Path,
) -> dict[str, str]:
    """Recheck one external-asset record immediately before model use."""
    if not isinstance(assets, dict):
        raise ValueError("asset manifest assets must be a JSON object")
    record = assets.get(name)
    if not isinstance(record, dict):
        raise ValueError(f"asset manifest does not contain {name!r}")
    expected_fields = {
        "kind",
        "identifier",
        "revision",
        "root",
        "files",
        "tree_sha256",
    }
    if set(record) != expected_fields:
        raise ValueError(
            f"asset manifest entry {name!r} fields differ from "
            f"{sorted(expected_fields)}"
        )

    root_text = record["root"]
    if not isinstance(root_text, str) or not Path(root_text).is_absolute():
        raise ValueError(f"asset manifest entry {name!r} root must be absolute")
    root = Path(root_text).resolve()
    expected_root = expected_root.resolve()
    if root != expected_root:
        raise ValueError(
            f"{role} root does not match asset manifest entry {name!r}"
        )
    if not root.is_dir():
        raise FileNotFoundError(f"{role} root is not a directory: {root}")

    required_text = ("kind", "identifier", "revision", "tree_sha256")
    for field in required_text:
        if not isinstance(record[field], str) or not record[field]:
            raise ValueError(f"asset manifest entry {name!r} has no {field}")
    revision = record["revision"]
    if len(revision) not in (40, 64) or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ValueError(
            f"asset manifest entry {name!r} revision is not immutable"
        )
    tree_sha256 = record["tree_sha256"]
    if len(tree_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in tree_sha256
    ):
        raise ValueError(
            f"asset manifest entry {name!r} tree_sha256 is not lowercase SHA256"
        )

    files = record["files"]
    if not isinstance(files, list) or not files:
        raise ValueError(f"asset manifest entry {name!r} has no authenticated files")
    declared: dict[str, tuple[int, str]] = {}
    for index, value in enumerate(files):
        if not isinstance(value, dict) or set(value) != {"path", "bytes", "sha256"}:
            raise ValueError(
                f"asset manifest entry {name!r} file {index} has invalid fields"
            )
        relative_text = value["path"]
        if not isinstance(relative_text, str) or not relative_text:
            raise ValueError(
                f"asset manifest entry {name!r} file {index} has no path"
            )
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(
                f"asset manifest entry {name!r} file path is not safe: "
                f"{relative_text!r}"
            )
        canonical_name = relative.as_posix()
        if canonical_name in declared:
            raise ValueError(
                f"asset manifest entry {name!r} repeats {canonical_name!r}"
            )
        size = value["bytes"]
        digest = value["sha256"]
        if type(size) is not int or size <= 0:
            raise ValueError(
                f"asset manifest entry {name!r} file {canonical_name!r} "
                "has invalid byte count"
            )
        if not isinstance(digest, str) or len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(
                f"asset manifest entry {name!r} file {canonical_name!r} "
                "has invalid SHA256"
            )
        declared[canonical_name] = (size, digest)

    observed: set[str] = set()
    symlink_directories: list[str] = []
    for directory, directories, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        directories.sort()
        filenames.sort()
        for child in directories:
            child_path = directory_path / child
            if child_path.is_symlink():
                symlink_directories.append(child_path.relative_to(root).as_posix())
        observed.update(
            (directory_path / filename).relative_to(root).as_posix()
            for filename in filenames
        )
    if symlink_directories:
        raise ValueError(
            f"asset manifest entry {name!r} contains symlink directories: "
            f"{symlink_directories}"
        )
    if observed != set(declared):
        raise ValueError(
            f"asset manifest entry {name!r} file list is not exhaustive: "
            f"missing={sorted(observed - set(declared))}, "
            f"absent={sorted(set(declared) - observed)}"
        )

    tree = hashlib.sha256()
    for relative_name in sorted(declared):
        path = root / relative_name
        if path.is_symlink():
            raise ValueError(
                f"asset manifest entry {name!r} contains a symlink file: "
                f"{relative_name!r}"
            )
        resolved = path.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"asset manifest entry {name!r} file escapes its root: "
                f"{relative_name!r}"
            ) from error
        if not resolved.is_file():
            raise FileNotFoundError(f"authenticated asset file is missing: {resolved}")
        expected_size, expected_digest = declared[relative_name]
        actual_size = resolved.stat().st_size
        if actual_size != expected_size:
            raise ValueError(
                f"asset file byte identity mismatch for {relative_name!r}: "
                f"{actual_size} != {expected_size}"
            )
        actual_digest = _sha256_file(resolved)
        if actual_digest != expected_digest:
            raise ValueError(
                f"asset file SHA256 mismatch for {relative_name!r}: "
                f"{actual_digest} != {expected_digest}"
            )
        tree.update(relative_name.encode("utf-8"))
        tree.update(b"\0")
        tree.update(str(expected_size).encode("ascii"))
        tree.update(b"\0")
        tree.update(expected_digest.encode("ascii"))
        tree.update(b"\n")
    observed_tree = tree.hexdigest()
    if observed_tree != tree_sha256:
        raise ValueError(
            f"asset manifest entry {name!r} tree_sha256 mismatch: "
            f"{observed_tree} != {tree_sha256}"
        )
    return {
        "name": name,
        **{field: record[field] for field in required_text},
        "source": "authenticated_local_snapshot",
    }


def authenticate_asset_manifest(
    manifest_path: Path,
    name: str,
    role: str,
    expected_root: Path,
) -> tuple[dict[str, str], dict[str, str | int]]:
    resolved_manifest = manifest_path.resolve()
    if not resolved_manifest.is_file():
        raise FileNotFoundError(f"asset manifest is missing: {resolved_manifest}")
    raw = json.loads(resolved_manifest.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"schema", "assets"}:
        raise ValueError("asset manifest fields are invalid")
    if raw["schema"] != "fa4_external_assets_v1":
        raise ValueError("asset manifest must use fa4_external_assets_v1")
    identity = authenticate_asset_record(
        raw["assets"],
        name,
        role,
        expected_root,
    )
    return identity, portable_file_identity(resolved_manifest)


def portable_asset_identity(
    load_target: str,
    *,
    identifier: str | None,
    revision: str | None,
    tree_sha256: str | None,
) -> dict[str, str | None]:
    metadata = (identifier, revision, tree_sha256)
    if any(value is not None for value in metadata) and not all(metadata):
        raise ValueError(
            "asset identity requires identifier, immutable revision, and "
            "tree SHA256 together"
        )
    is_local = Path(load_target).is_absolute()
    if is_local and not all(metadata):
        raise ValueError(
            "an absolute local asset requires authenticated identity metadata"
        )
    if revision is not None and (
        len(revision) not in (40, 64)
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise ValueError("asset revision must be an immutable lowercase hex id")
    if tree_sha256 is not None and (
        len(tree_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in tree_sha256
        )
    ):
        raise ValueError("asset tree SHA256 must be lowercase hex")
    return {
        "identifier": identifier or load_target,
        "revision": revision,
        "tree_sha256": tree_sha256,
        "source": "authenticated_local_snapshot" if is_local else "logical_identifier",
    }


def portable_file_identity(path: Path) -> dict[str, str | int]:
    resolved = path.resolve()
    return {
        "file": resolved.name,
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="nateraw/vit-base-patch16-224-cifar10",
    )
    parser.add_argument("--dataset", default="uoft-cs/cifar10")
    add_asset_identity_arguments(parser)
    parser.add_argument("--split", default="test")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--scale-sweep-samples", type=int, default=4)
    parser.add_argument(
        "--scale-factors",
        default="1,2,4,8,12,16,24,32,320,448,1440,1536,2688",
        help="Comma-separated global P multipliers for the offline sweep",
    )
    parser.add_argument("--mask-value", type=float, default=20.0)
    parser.add_argument(
        "--image-size",
        type=int,
        default=0,
        help=(
            "Override the processor's square image size. Non-native sizes "
            "enable ViT positional interpolation."
        ),
    )
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument(
        "--extension-module",
        default="_C_tk_hao_direct_fp4pv",
    )
    parser.add_argument(
        "--attention-backend",
        choices=("tk", "hao-native", "hao-fp8"),
        default="tk",
        help=(
            "Execute the prepared tensors with the TK extension, HAO's "
            "native NVFP4/NVFP4 implementation, or HAO's preferred "
            "NVFP4/FP8 implementation."
        ),
    )
    parser.add_argument("--hao-root", type=Path, default=DEFAULT_HAO_ROOT)
    parser.add_argument(
        "--layer-extension",
        action="append",
        default=[],
        metavar="LAYERS=PATH:MODULE",
        help=(
            "Override the extension for selected attention layers. "
            "LAYERS is a comma-separated list of indices or inclusive "
            "ranges, for example 0-2,8=/tmp/candidate.so:_C_candidate."
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument(
        "--finite-diagnostics",
        action="store_true",
        help="Record per-layer tensor ranges when debugging non-finite output.",
    )
    parser.add_argument(
        "--stop-on-nonfinite",
        action="store_true",
        help="Stop after the first sample that produces a non-finite row.",
    )
    parser.add_argument(
        "--interleave-kv-quarters",
        action="store_true",
        help=(
            "Apply the same stride-4 permutation to K and V within each "
            "128-token tile so score quarter 0 samples the full tile."
        ),
    )
    parser.add_argument(
        "--global-anchor-kv",
        action="store_true",
        help=(
            "Place 32 keys sampled across the logical sequence in physical "
            "Q0 of the first N128 tile, with the same permutation for V."
        ),
    )
    parser.add_argument(
        "--global-anchor-samples",
        type=int,
        choices=(32, 64, 128),
        default=32,
        help="Number of globally distributed keys placed first.",
    )
    parser.add_argument(
        "--mx-q-quant-mode",
        choices=tuple(MX_QUANT_MODES),
        default="rte",
    )
    parser.add_argument(
        "--mx-k-quant-mode",
        choices=tuple(MX_QUANT_MODES),
        default="rte",
    )
    parser.add_argument(
        "--mx-v-quant-mode",
        choices=tuple(MX_QUANT_MODES),
        default="rte",
    )
    parser.add_argument(
        "--nv-qk-fold-k64-scales",
        choices=("auto", "none", "q", "k", "both"),
        default="auto",
        help=(
            "Fold matching NVFP4 K64 scale groups. 'auto' follows the "
            "extension topology."
        ),
    )
    parser.add_argument(
        "--nv-qk-fold-scale-select",
        choices=("max", "mse"),
        default="mse",
    )
    parser.add_argument(
        "--nv-qk-fold-scale-multiplier",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--qk-equalization",
        type=float,
        default=1.0,
        help=(
            "Quantize alpha*Q and K/alpha. Their exact dot product is "
            "unchanged, but MXFP4 exponent-boundary placement changes."
        ),
    )
    parser.add_argument(
        "--qk-channel-equalization",
        choices=("none", "rms", "amax"),
        default="none",
        help=(
            "Reciprocally equalize Q/K channels before quantization while "
            "preserving their exact dot product."
        ),
    )
    parser.add_argument(
        "--qk-channel-equalization-strength",
        type=float,
        default=1.0,
        help="Exponent applied to the reciprocal Q/K channel equalizer.",
    )
    parser.add_argument(
        "--qk-channel-permutation",
        choices=("none", "active-spread", "rms-balanced"),
        default="none",
        help=(
            "Apply the same dot-product-preserving channel permutation to "
            "Q and K before MXFP4 quantization."
        ),
    )
    parser.add_argument(
        "--qk-orthogonal-transform",
        choices=(
            "none",
            "hadamard",
            "signed-hadamard",
            "signal-dct",
        ),
        default="none",
        help=(
            "Apply the same normalized orthogonal transform to Q and K. "
            "This preserves QK exactly before quantization."
        ),
    )
    parser.add_argument(
        "--key-centering",
        choices=("none", "projection-bias", "sequence-mean"),
        default="none",
        help=(
            "Subtract a key vector shared by every token. This leaves exact "
            "softmax unchanged while controlling shiftless score offsets."
        ),
    )
    parser.add_argument(
        "--score-shift",
        type=float,
        default=0.0,
        help=(
            "Subtract this nonnegative constant from every normalized QK "
            "score through an otherwise-unused padded Q/K dimension."
        ),
    )
    parser.add_argument(
        "--score-shift-predictor",
        choices=(
            "fixed",
            "q-rms",
            "qk-rms",
            "sample32-rowmax",
            "sample-rowmax",
            "exact-rowmax",
        ),
        default="fixed",
        help="Statistic multiplied by --score-shift for each query row.",
    )
    parser.add_argument(
        "--score-shift-bias",
        type=float,
        default=0.0,
        help="Additive normalized-score margin after the row-shift predictor.",
    )
    parser.add_argument(
        "--p-replay-diagnostics",
        action="store_true",
        help=(
            "Replay MXFP4 P transforms offline on real score rows and "
            "decompose anchor, encoder, scale, and denominator error."
        ),
    )
    parser.add_argument(
        "--p-replay-samples",
        type=int,
        default=1,
        help="Number of model samples included in P-transform replay.",
    )
    parser.add_argument(
        "--p-replay-layers",
        default="0",
        help="Comma-separated attention-layer indices included in replay.",
    )
    parser.add_argument(
        "--p-replay-query-chunk",
        type=int,
        default=64,
        help="Number of query rows replayed at once.",
    )
    parser.add_argument(
        "--p-replay-affine-search",
        action="store_true",
        help=(
            "Sweep same-cost MXFP4 affine encoders on sampled real score "
            "rows during P replay."
        ),
    )
    parser.add_argument(
        "--p-replay-nv-scale-search",
        action="store_true",
        help=(
            "Sweep compile-time NVFP4 P exponent rebases without the "
            "larger affine grid."
        ),
    )
    return parser.parse_args()


def load_extension(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import extension {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_layer_indices(specification: str) -> set[int]:
    layers: set[int] = set()
    for item in specification.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            first_text, last_text = item.split("-", 1)
            first = int(first_text)
            last = int(last_text)
            if first < 0 or last < first:
                raise ValueError(f"invalid layer range: {item}")
            layers.update(range(first, last + 1))
        else:
            layer = int(item)
            if layer < 0:
                raise ValueError(f"invalid layer index: {item}")
            layers.add(layer)
    if not layers:
        raise ValueError("layer extension requires at least one layer")
    return layers


def tensor_metrics(actual: Any, reference: Any) -> dict[str, float]:
    import torch

    actual32 = actual.float()
    reference32 = reference.float()
    delta = actual32 - reference32
    rmse = delta.square().mean().sqrt()
    reference_rms = reference32.square().mean().sqrt()
    denominator = reference_rms.clamp_min(torch.finfo(torch.float32).tiny)
    cosine_dtype = (
        torch.float64 if actual32.numel() >= 1_000_000 else torch.float32
    )
    actual_cosine = actual32.flatten().to(cosine_dtype)
    reference_cosine = reference32.flatten().to(cosine_dtype)
    cosine = (
        torch.dot(actual_cosine, reference_cosine)
        / (
            torch.linalg.vector_norm(actual_cosine)
            * torch.linalg.vector_norm(reference_cosine)
        ).clamp_min(torch.finfo(cosine_dtype).tiny)
    ).clamp(-1.0, 1.0)
    return {
        "cosine": float(cosine.item()),
        "rmse": float(rmse.item()),
        "reference_rms": float(reference_rms.item()),
        "relative_l2": float((rmse / denominator).item()),
        "max_abs": float(delta.abs().max().item()),
    }


def tensor_finite_stats(tensor: Any) -> dict[str, float | int | bool]:
    import torch

    values = tensor.float()
    finite = torch.isfinite(values)
    finite_count = int(finite.sum().item())
    total = values.numel()
    result: dict[str, float | int | bool] = {
        "all_finite": finite_count == total,
        "finite_count": finite_count,
        "nonfinite_count": total - finite_count,
    }
    if finite_count:
        finite_values = values[finite]
        result.update(
            {
                "minimum": float(finite_values.min().item()),
                "maximum": float(finite_values.max().item()),
                "max_abs": float(finite_values.abs().max().item()),
            }
        )
    return result


def mean_records(records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        return {}
    return {
        key: sum(record[key] for record in records) / len(records)
        for key in records[0]
    }


def summarize_scale_records(
    records: list[dict[str, float]],
) -> dict[str, float]:
    if not records:
        return {}
    fraction_keys = (
        "fraction_below_e4m3_subnormal",
        "fraction_below_e4m3_normal",
        "fraction_above_e4m3_max",
    )
    result = {
        key: sum(record[key] for record in records) / len(records)
        for key in fraction_keys
    }
    result["minimum"] = min(record["minimum"] for record in records)
    result["maximum"] = max(record["maximum"] for record in records)
    return result


def distribution_quantiles(values: Any) -> dict[str, float]:
    import torch

    finite = values.float().flatten()
    finite = finite[torch.isfinite(finite)]
    if not finite.numel():
        return {}
    max_quantile_values = 4 * 1024 * 1024
    if finite.numel() > max_quantile_values:
        stride = math.ceil(finite.numel() / max_quantile_values)
        finite = finite[::stride]
    probabilities = torch.tensor(
        (0.0, 0.5, 0.9, 0.99, 0.999, 1.0),
        device=finite.device,
        dtype=torch.float32,
    )
    quantiles = torch.quantile(finite, probabilities)
    return {
        "min": float(quantiles[0].item()),
        "p50": float(quantiles[1].item()),
        "p90": float(quantiles[2].item()),
        "p99": float(quantiles[3].item()),
        "p999": float(quantiles[4].item()),
        "max": float(quantiles[5].item()),
        "mean": float(finite.mean().item()),
    }


def mxfp4_replay_context(
    scores: Any,
    value: Any,
    *,
    anchor: Any,
    encoder: str,
    denominator: str,
    scale_select_bias: float,
    query_chunk: int,
    affine_a: float = 1.62330034,
    affine_b: float = 0.92083546,
) -> Any:
    import torch
    import torch.nn.functional as functional

    if encoder not in ("exact", "fast"):
        raise ValueError(f"unsupported MX replay encoder: {encoder}")
    if denominator not in (
        "represented",
        "sampled",
        "native8-sampled",
        "native8-prior6",
        "native8-replace-max6",
        "native4-max-cv",
        "native8-max-cv",
        "word8-max-cv",
    ):
        raise ValueError(
            f"unsupported MX replay denominator: {denominator}"
        )

    batch, heads, queries, keys = scores.shape
    padded_keys = math.ceil(keys / 32) * 32
    levels = torch.tensor(
        E2M1_LEVELS,
        device=scores.device,
        dtype=torch.float32,
    )
    midpoints = torch.tensor(
        E2M1_MIDPOINTS,
        device=scores.device,
        dtype=torch.float32,
    )
    output = torch.empty(
        (batch, heads, queries, value.shape[-1]),
        device=scores.device,
        dtype=torch.float32,
    )

    pair = torch.arange(16, device=scores.device)
    quarter = torch.arange(
        padded_keys // 32,
        device=scores.device,
    ).remainder(4)
    sample_pair = quarter[:, None]
    native_pair = (
        (pair[None, :] == sample_pair)
        | (pair[None, :] == sample_pair + 8)
        | (pair[None, :] == 4)
        | (pair[None, :] == 12)
    )
    native_mask = (
        native_pair[:, :, None]
        .expand(-1, -1, 2)
        .reshape(1, 1, 1, padded_keys // 32, 32)
    )
    sampled_mask = (
        (
            (pair[None, :] == sample_pair)
            | (pair[None, :] == sample_pair + 8)
        )[:, :, None]
        .expand(-1, -1, 2)
        .reshape(1, 1, 1, padded_keys // 32, 32)
    )
    word = quarter.remainder(4)
    position = torch.arange(32, device=scores.device)
    word8_mask = (
        (position[None, :] >= word[:, None] * 8)
        & (position[None, :] < (word[:, None] + 1) * 8)
    ).reshape(1, 1, 1, padded_keys // 32, 32)

    for query_start in range(0, queries, query_chunk):
        query_end = min(query_start + query_chunk, queries)
        score_chunk = scores[:, :, query_start:query_end].float()
        anchor_chunk = anchor[:, :, query_start:query_end].float()
        log_weight = (score_chunk - anchor_chunk) * LOG2_E
        if padded_keys != keys:
            log_weight = functional.pad(
                log_weight,
                (0, padded_keys - keys),
                value=-math.inf,
            )
        grouped_log_weight = log_weight.reshape(
            batch,
            heads,
            query_end - query_start,
            padded_keys // 32,
            32,
        )
        group_max = grouped_log_weight.amax(dim=-1)
        scale_exponent = torch.floor(
            group_max + MX_E8M0_RTE_SHIFT + scale_select_bias
        ).clamp(-126.0, 127.0)
        local_log = (
            grouped_log_weight
            + LOG2_FP4_MAX
            - scale_exponent.unsqueeze(-1)
        )

        native = torch.exp2(local_log)
        if encoder == "exact":
            encoded_value = native
        else:
            affine = affine_a * local_log + affine_b
            encoded_value = torch.where(native_mask, native, affine)
        code = (
            encoded_value.unsqueeze(-1) > midpoints
        ).sum(dim=-1)
        represented_level = levels[code]
        group_scale = torch.exp2(scale_exponent) / 6.0
        represented = represented_level * group_scale.unsqueeze(-1)

        if denominator == "represented":
            normalizer = represented.sum(
                dim=(-2, -1),
                keepdim=False,
            )
        elif denominator == "sampled":
            sampled_sum = (
                encoded_value.masked_fill(~sampled_mask, 0.0)
                .sum(dim=-1)
                * 8.0
            )
            normalizer = (
                sampled_sum * group_scale
            ).sum(dim=-1)
        elif denominator in (
            "native8-sampled",
            "native8-prior6",
            "native8-replace-max6",
        ):
            native_sample = encoded_value.masked_fill(
                ~native_mask,
                0.0,
            )
            native_sum = native_sample.sum(dim=-1)
            if denominator == "native8-sampled":
                estimated_sum = native_sum * 4.0
            elif denominator == "native8-prior6":
                estimated_sum = native_sum * (31.0 / 8.0) + 6.0
            else:
                native_max = native_sample.amax(dim=-1)
                estimated_sum = (
                    native_sum - native_max
                ) * (31.0 / 7.0) + 6.0
            normalizer = (
                estimated_sum * group_scale
            ).sum(dim=-1)
        else:
            if denominator == "native4-max-cv":
                estimator_mask = sampled_mask
                sample_count = 4.0
            elif denominator == "native8-max-cv":
                estimator_mask = native_mask
                sample_count = 8.0
            else:
                estimator_mask = word8_mask
                sample_count = 8.0
            represented_max = represented_level.amax(
                dim=-1,
                keepdim=True,
            )
            sampled_level = represented_level.masked_fill(
                ~estimator_mask,
                0.0,
            )
            sampled_sum = sampled_level.sum(dim=-1)
            sampled_has_max = (
                (sampled_level == represented_max)
                & estimator_mask
                & (represented_max > 0.0)
            ).any(dim=-1)
            hit = sampled_has_max.float()
            residual_sum = (
                sampled_sum - hit * represented_max.squeeze(-1)
            )
            estimated_level_sum = (
                represented_max.squeeze(-1)
                + residual_sum
                * (31.0 / (sample_count - hit))
            )
            normalizer = (
                estimated_level_sum * group_scale
            ).sum(dim=-1)
        normalizer = normalizer.clamp_min(
            torch.finfo(torch.float32).tiny
        )
        represented = represented.reshape(
            batch,
            heads,
            query_end - query_start,
            padded_keys,
        )[..., :keys]
        probability = represented / normalizer.unsqueeze(-1)
        output[:, :, query_start:query_end] = torch.matmul(
            probability,
            value.float(),
        )
    return output


def nvfp4_replay_context(
    scores: Any,
    value: Any,
    *,
    anchor: Any,
    query_chunk: int,
    affine_a: float,
    affine_b: float,
    p_global_log2: float = 0.0,
    exact_exp2: bool = False,
) -> tuple[Any, dict[str, float]]:
    """Replay the fast quarter-scale NVFP4 P encoder and exact denominator."""
    import torch
    import torch.nn.functional as functional

    batch, heads, queries, keys = scores.shape
    padded_keys = math.ceil(keys / 32) * 32
    levels = torch.tensor(
        E2M1_LEVELS,
        device=scores.device,
        dtype=torch.float32,
    )
    midpoints = torch.tensor(
        E2M1_MIDPOINTS,
        device=scores.device,
        dtype=torch.float32,
    )
    output = torch.empty(
        (batch, heads, queries, value.shape[-1]),
        device=scores.device,
        dtype=torch.float32,
    )

    pair = torch.arange(16, device=scores.device)
    quarter = torch.arange(
        padded_keys // 32,
        device=scores.device,
    ).remainder(4)
    sample_pair = quarter[:, None]
    native_pair = (
        (pair[None, :] == sample_pair)
        | (pair[None, :] == sample_pair + 8)
        | (pair[None, :] == 4)
        | (pair[None, :] == 12)
    )
    native_mask = (
        native_pair[:, :, None]
        .expand(-1, -1, 2)
        .reshape(1, 1, 1, padded_keys // 32, 32)
    )

    scale_count = 0
    scale_below = 0
    scale_above = 0
    raw_scale_min = math.inf
    raw_scale_max = -math.inf
    for query_start in range(0, queries, query_chunk):
        query_end = min(query_start + query_chunk, queries)
        score_chunk = scores[:, :, query_start:query_end].float()
        anchor_chunk = anchor[:, :, query_start:query_end].float()
        if padded_keys != keys:
            score_chunk = functional.pad(
                score_chunk,
                (0, padded_keys - keys),
                value=-math.inf,
            )
        grouped_score = score_chunk.reshape(
            batch,
            heads,
            query_end - query_start,
            padded_keys // 32,
            32,
        )
        raw_scale_log2 = (
            (grouped_score.amax(dim=-1) - anchor_chunk) * LOG2_E
            - LOG2_FP4_MAX
            + p_global_log2
        )
        finite_scale = torch.isfinite(raw_scale_log2)
        scale_count += int(finite_scale.sum().item())
        scale_below += int(
            (finite_scale & (raw_scale_log2 < -6.0)).sum().item()
        )
        scale_above += int(
            (finite_scale & (raw_scale_log2 > 8.75)).sum().item()
        )
        if finite_scale.any():
            finite_values = raw_scale_log2[finite_scale]
            raw_scale_min = min(
                raw_scale_min, float(finite_values.min().item())
            )
            raw_scale_max = max(
                raw_scale_max, float(finite_values.max().item())
            )

        scale_code = torch.round(
            raw_scale_log2 * 8.0 + 56.0
        ).clamp(8.0, 126.0).to(torch.int32)
        encoded_log2 = (scale_code.float() - 56.0) * 0.125
        scale_bits = (scale_code << 20) + (120 << 23)
        represented_scale = scale_bits.view(torch.float32)
        local_log = (
            grouped_score * LOG2_E
            - anchor_chunk.unsqueeze(-1) * LOG2_E
            - encoded_log2.unsqueeze(-1)
            + p_global_log2
        )
        native = torch.exp2(local_log)
        affine = affine_a * local_log + affine_b
        encoded = native if exact_exp2 else torch.where(
            native_mask, native, affine
        )
        code = (encoded.unsqueeze(-1) > midpoints).sum(dim=-1)
        represented_level = levels[code]
        represented = (
            represented_level * represented_scale.unsqueeze(-1)
        )
        normalizer = represented.sum(dim=(-2, -1)).clamp_min(
            torch.finfo(torch.float32).tiny
        )
        represented = represented.reshape(
            batch,
            heads,
            query_end - query_start,
            padded_keys,
        )[..., :keys]
        probability = represented / normalizer.unsqueeze(-1)
        output[:, :, query_start:query_end] = torch.matmul(
            probability,
            value.float(),
        )

    denominator = max(scale_count, 1)
    return output, {
        "scale_log2_min": raw_scale_min,
        "scale_log2_max": raw_scale_max,
        "scale_fraction_below_e4m3": scale_below / denominator,
        "scale_fraction_above_e4m3": scale_above / denominator,
    }


def decode_mxfp4_payload(payload: Any) -> Any:
    import torch

    packed = payload.contiguous().view(torch.uint8)
    levels = torch.tensor(
        SIGNED_E2M1_LEVELS,
        device=payload.device,
        dtype=torch.float32,
    )
    low = levels[(packed & 0x0F).to(torch.long)]
    high = levels[(packed >> 4).to(torch.long)]
    return torch.stack((low, high), dim=-1).flatten(-2)


def e8m0_decode_scale(encoded: Any) -> Any:
    import torch

    scale = torch.ldexp(
        torch.ones_like(encoded, dtype=torch.float32),
        encoded.to(torch.int32) - 127,
    ) * (1.0 / 6.0)
    return torch.where(encoded != 0, scale, torch.zeros_like(scale))


def dequantize_prepared_mxfp4_qk(
    payload: Any,
    prepared_scale: Any,
) -> Any:
    import torch

    batch, heads, rows, packed_columns = payload.shape
    columns = packed_columns * 2
    if rows % 128 or columns != 128:
        raise ValueError("MXFP4 QK replay expects M128/K128 tiling")
    row_tiles = rows // 128
    scale_bytes = prepared_scale.contiguous().view(torch.uint8)
    if scale_bytes.shape[1] == row_tiles * 2:
        scale_bytes = scale_bytes[:, ::2]
    scales = (
        scale_bytes.reshape(batch, row_tiles, heads, 32, 16)
        .permute(0, 2, 1, 3, 4)
        .contiguous()
    )

    row = torch.arange(rows, device=payload.device)
    block = torch.arange(columns // 32, device=payload.device)
    tile_index = (row // 128)[:, None]
    row_lane = (row % 32)[:, None]
    scale_slot = (
        ((row % 128) // 32)[:, None] * 4 + block[None, :]
    )
    decoded = decode_mxfp4_payload(payload).reshape(
        batch, heads, rows, columns // 32, 32
    )
    for batch_index in range(batch):
        for head_index in range(heads):
            encoded = scales[
                batch_index,
                head_index,
                tile_index,
                row_lane,
                scale_slot,
            ]
            decoded[batch_index, head_index].mul_(
                e8m0_decode_scale(encoded)[..., None]
            )
    return decoded.reshape(batch, heads, rows, columns)


def dequantize_prepared_mxfp4_v(
    payload: Any,
    prepared_scale: Any,
) -> Any:
    import torch

    batch, heads, rows, packed_columns = payload.shape
    columns = packed_columns * 2
    if rows != 128 or columns % 128:
        raise ValueError("MXFP4 V replay expects M128/K128 tiling")
    column_tiles = columns // 128
    scales = (
        prepared_scale.contiguous()
        .view(torch.uint8)
        .reshape(batch, column_tiles, heads, 32, 16)
        .permute(0, 2, 1, 3, 4)
        .contiguous()
    )

    row = torch.arange(rows, device=payload.device)
    block = torch.arange(columns // 32, device=payload.device)
    tile_index = (block // 4)[None, :]
    row_lane = (row % 32)[:, None]
    scale_slot = (
        ((row % 128) // 32)[:, None] * 4
        + (block % 4)[None, :]
    )
    decoded = decode_mxfp4_payload(payload).reshape(
        batch, heads, rows, columns // 32, 32
    )
    for batch_index in range(batch):
        for head_index in range(heads):
            encoded = scales[
                batch_index,
                head_index,
                tile_index,
                row_lane,
                scale_slot,
            ]
            decoded[batch_index, head_index].mul_(
                e8m0_decode_scale(encoded)[..., None]
            )
    return decoded.reshape(
        batch, heads, rows, columns
    ).transpose(-2, -1).contiguous()


def nvfp4_roundtrip_linear(matrix: Any) -> Any:
    """Quantize/dequantize with the same NVFP4 values in linear SF layout."""
    import torch
    from flashinfer.quantization import (
        SfLayout,
        nvfp4_kv_dequantize,
        nvfp4_quantize,
    )

    source = matrix.to(torch.bfloat16).contiguous()
    one = torch.ones(1, device=matrix.device, dtype=torch.float32)
    payload, scale = nvfp4_quantize(
        source,
        one,
        sfLayout=SfLayout.layout_linear,
        do_shuffle=False,
    )
    return nvfp4_kv_dequantize(
        payload.view(torch.uint8).contiguous(),
        scale.view(torch.uint8).contiguous(),
        one,
    ).float()


def quantize_nvfp4_qk(ref: Any) -> tuple[Any, Any]:
    import torch
    from flashinfer.quantization import SfLayout, nvfp4_quantize

    batch, seqlen, heads, dim = ref.shape
    if seqlen % 128 or dim % 128:
        raise ValueError("NVFP4 Q/K adapter requires S and D divisible by 128")
    tensor_2d = ref.to(torch.bfloat16).reshape(
        batch * seqlen,
        heads * dim,
    )
    one = torch.ones(1, device=ref.device, dtype=torch.float32)
    payload, scale_data = nvfp4_quantize(
        tensor_2d,
        one,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
    )
    payload = (
        payload.reshape(batch, seqlen, heads, dim // 2)
        .view(torch.uint8)
        .view(torch.float4_e2m1fn_x2)
    )
    rest_m = seqlen // 128
    rest_k = (dim // 16) // 4
    total_m = batch * rest_m
    total_k = (heads * (dim // 16)) // 4
    scales = scale_data.reshape(total_m, total_k, 32, 4, 4)
    scales = scales.reshape(
        batch,
        rest_m,
        heads,
        rest_k,
        32,
        4,
        4,
    )
    scales = (
        scales.permute(0, 2, 1, 3, 4, 5, 6)
        .contiguous()
        .permute(4, 5, 2, 6, 3, 1, 0)
    )
    return payload, scales


def quantize_nvfp4_v(ref: Any) -> tuple[Any, Any]:
    import torch
    from flashinfer.quantization import SfLayout, nvfp4_quantize

    batch, seqlen, heads, dim = ref.shape
    if seqlen % 128 or dim % 128:
        raise ValueError("NVFP4 V adapter requires S and D divisible by 128")
    k_major = (
        ref.to(torch.bfloat16)
        .permute(0, 2, 3, 1)
        .contiguous()
        .reshape(batch * heads * dim, seqlen)
    )
    one = torch.ones(1, device=ref.device, dtype=torch.float32)
    payload_data, scale_data = nvfp4_quantize(
        k_major,
        one,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
    )
    payload = (
        payload_data.reshape(batch, heads, dim, seqlen // 2)
        .view(torch.uint8)
        .view(torch.float4_e2m1fn_x2)
    )
    rest_m = dim // 128
    rest_k = seqlen // 64
    scales = scale_data.reshape(
        batch * heads * rest_m,
        rest_k,
        32,
        4,
        4,
    )
    scales = scales.reshape(
        batch,
        heads,
        rest_m,
        rest_k,
        32,
        4,
        4,
    ).permute(4, 5, 2, 6, 3, 1, 0)
    return payload, scales


def interleave_kv_quarters(ref: Any) -> Any:
    """Distribute each contiguous 32-token quarter across its 128-token tile."""
    batch, seqlen, heads, dim = ref.shape
    if seqlen % 128:
        raise ValueError("K/V quarter interleave requires S divisible by 128")
    return (
        ref.reshape(batch, seqlen // 128, 32, 4, heads, dim)
        .transpose(2, 3)
        .contiguous()
        .reshape(batch, seqlen, heads, dim)
    )


def global_anchor_indices(
    valid_tokens: int,
    samples: int,
    device: Any,
) -> Any:
    """Build an evenly distributed global score probe."""
    import torch

    if samples not in (32, 64, 128):
        raise ValueError("global K/V anchor supports 32, 64, or 128 samples")
    if valid_tokens < samples:
        raise ValueError(
            "global K/V anchor requires samples <= valid tokens"
        )
    return (
        torch.linspace(
            0,
            valid_tokens - 1,
            samples,
            device=device,
            dtype=torch.float32,
        )
        .round()
        .to(torch.long)
    )


def global_anchor_kv(
    ref: Any,
    valid_tokens: int,
    samples: int = 32,
) -> Any:
    """Place globally distributed keys at the start of the physical sequence."""
    import torch

    _, seqlen, _, _ = ref.shape
    if seqlen % 128:
        raise ValueError("global K/V anchor requires S divisible by 128")
    if valid_tokens > seqlen:
        raise ValueError("global K/V anchor requires valid tokens <= S")
    anchor = global_anchor_indices(valid_tokens, samples, ref.device)
    selected = torch.zeros(seqlen, device=ref.device, dtype=torch.bool)
    selected[anchor] = True
    remainder = torch.arange(seqlen, device=ref.device)[~selected]
    order = torch.cat((anchor, remainder))
    return ref.index_select(1, order).contiguous()


def prepare_kernel_inputs(
    query: Any,
    key: Any,
    value: Any,
    *,
    qk_format: str,
    pv_format: str,
    target_seqlen: int,
    target_heads: int,
    target_dim: int,
    mask_value: float,
    key_valid_mask: Any | None = None,
    interleave_quarters: bool = False,
    global_anchor: bool = False,
    global_anchor_samples: int = 32,
    mx_q_quant_mode: int = 0,
    mx_k_quant_mode: int = 0,
    mx_v_quant_mode: int = 0,
    nv_qk_fold_k64_scales: str = "none",
    nv_qk_fold_scale_select: str = "mse",
    nv_qk_fold_scale_multiplier: float = 1.0,
    compact_folded_qk_scales: bool = False,
    qk_equalization: float = 1.0,
    qk_channel_equalization: str = "none",
    qk_channel_equalization_strength: float = 1.0,
    qk_channel_permutation: str = "none",
    qk_orthogonal_transform: str = "none",
    score_shift: float = 0.0,
    score_shift_predictor: str = "fixed",
    score_shift_bias: float = 0.0,
    retain_replay_tensors: bool = False,
) -> tuple[Any, int, int, int]:
    import torch

    from hao_direct_fp4pv_benchmark import (
        prepare_native_inputs,
        quantize_mxfp4_qk,
        quantize_mxfp4_v,
        quantize_nvfp4_qk_folded_k64_scales,
    )

    batch, heads, seqlen, dim = query.shape
    if batch != 1:
        raise ValueError("the retained downstream extension is batch-1")
    if key.shape != query.shape or value.shape != query.shape:
        raise ValueError("the ViT adapter requires matching Q/K/V shapes")
    if key_valid_mask is not None:
        if tuple(key_valid_mask.shape) != (batch, seqlen):
            raise ValueError(
                "key_valid_mask must have shape "
                f"{(batch, seqlen)}, got {tuple(key_valid_mask.shape)}"
            )
        key_valid_mask = key_valid_mask.to(
            device=query.device,
            dtype=torch.bool,
        )
    needs_key_mask = seqlen < target_seqlen or key_valid_mask is not None
    required_dim = (
        dim + int(needs_key_mask) + int(score_shift > 0.0)
    )
    if (
        seqlen > target_seqlen
        or heads > target_heads
        or required_dim > target_dim
    ):
        raise ValueError(
            f"cannot pad {(seqlen, heads, dim)} into "
            f"{(target_seqlen, target_heads, target_dim)}"
        )

    q_ref = torch.zeros(
        (batch, target_seqlen, target_heads, target_dim),
        device=query.device,
        dtype=torch.float32,
    )
    k_ref = torch.zeros_like(q_ref)
    v_ref = torch.zeros_like(q_ref)
    q_bshd = query.permute(0, 2, 1, 3).float()
    k_bshd = key.permute(0, 2, 1, 3).float()
    v_bshd = value.permute(0, 2, 1, 3).float()

    # The kernel scales QK by 1/sqrt(target_dim).  This restores ViT's
    # original 1/sqrt(dim) scale after zero-padding the head dimension.
    q_ref[:, :seqlen, :heads, :dim] = q_bshd * math.sqrt(
        target_dim / dim
    )
    k_ref[:, :seqlen, :heads, :dim] = k_bshd
    v_ref[:, :seqlen, :heads, :dim] = v_bshd

    # Give adapter- and model-padded keys a large negative score while
    # leaving valid QK scores unchanged. Exact-shape D128 calls need no
    # synthetic mask dimension.
    if needs_key_mask:
        mask_dim = dim
        q_ref[:, :seqlen, :heads, mask_dim] = mask_value
        k_ref[:, seqlen:, :heads, mask_dim] = -mask_value
        if key_valid_mask is not None:
            invalid_key = ~key_valid_mask
            k_ref[:, :seqlen, :heads, mask_dim] = torch.where(
                invalid_key[:, :, None],
                torch.full(
                    (batch, seqlen, 1),
                    -mask_value,
                    device=query.device,
                    dtype=torch.float32,
                ),
                torch.zeros(
                    (batch, seqlen, 1),
                    device=query.device,
                    dtype=torch.float32,
                ),
            )

    if score_shift > 0.0:
        shift_dim = dim + int(needs_key_mask)
        if score_shift_predictor == "fixed":
            row_shift = torch.full(
                (batch, seqlen, heads),
                score_shift,
                device=query.device,
                dtype=torch.float32,
            )
            key_amplitude = math.sqrt(score_shift * math.sqrt(target_dim))
        else:
            query_rms = q_bshd.square().mean(dim=-1).sqrt()
            if score_shift_predictor == "q-rms":
                row_shift = score_shift * query_rms
            elif score_shift_predictor == "qk-rms":
                key_rms = (
                    k_bshd.square()
                    .mean(dim=(1, 3), keepdim=False)
                    .sqrt()
                )
                row_shift = score_shift * query_rms * key_rms[:, None, :]
            elif score_shift_predictor in (
                "sample32-rowmax",
                "sample-rowmax",
            ):
                sample_count = (
                    32
                    if score_shift_predictor == "sample32-rowmax"
                    else global_anchor_samples
                )
                sample_indices = (
                    torch.linspace(
                        0,
                        seqlen - 1,
                        sample_count,
                        device=query.device,
                        dtype=torch.float32,
                    )
                    .round()
                    .to(torch.long)
                )
                sampled_key = k_bshd.index_select(1, sample_indices)
                sample_max = (
                    torch.matmul(
                        q_bshd.permute(0, 2, 1, 3),
                        sampled_key.permute(0, 2, 1, 3).transpose(-1, -2),
                    )
                    / math.sqrt(dim)
                ).amax(dim=-1)
                row_shift = score_shift * sample_max.permute(0, 2, 1)
            elif score_shift_predictor == "exact-rowmax":
                score_max = (
                    torch.matmul(
                        q_bshd.permute(0, 2, 1, 3),
                        k_bshd.permute(0, 2, 1, 3).transpose(-1, -2),
                    )
                    / math.sqrt(dim)
                ).amax(dim=-1)
                row_shift = score_shift * score_max.permute(0, 2, 1)
            else:
                raise ValueError(
                    f"unsupported score-shift predictor: "
                    f"{score_shift_predictor}"
                )
            row_shift = row_shift + score_shift_bias
            key_amplitude = math.sqrt(math.sqrt(target_dim))
        q_ref[:, :seqlen, :heads, shift_dim] = (
            row_shift * math.sqrt(target_dim) / key_amplitude
        )
        k_ref[:, :, :heads, shift_dim] = -key_amplitude

    if interleave_quarters and global_anchor:
        raise ValueError("select only one K/V permutation")
    if global_anchor:
        valid_tokens = seqlen
        if key_valid_mask is not None:
            valid_tokens = int(key_valid_mask[0].sum().item())
            expected_prefix = torch.arange(
                seqlen,
                device=query.device,
            ) < valid_tokens
            if not torch.equal(key_valid_mask[0], expected_prefix):
                raise ValueError(
                    "global K/V anchoring requires right-padded keys"
                )
            valid_tokens = max(valid_tokens, global_anchor_samples)
        k_ref = global_anchor_kv(
            k_ref,
            valid_tokens,
            global_anchor_samples,
        )
        v_ref = global_anchor_kv(
            v_ref,
            valid_tokens,
            global_anchor_samples,
        )
    elif interleave_quarters:
        k_ref = interleave_kv_quarters(k_ref)
        v_ref = interleave_kv_quarters(v_ref)

    if qk_equalization != 1.0:
        q_ref.mul_(qk_equalization)
        k_ref.mul_(1.0 / qk_equalization)

    if qk_channel_equalization != "none":
        q_active = q_ref[:, :seqlen, :heads, :dim]
        k_active = k_ref[:, :seqlen, :heads, :dim]
        if qk_channel_equalization == "rms":
            q_stat = q_active.square().mean(dim=(0, 1)).sqrt()
            k_stat = k_active.square().mean(dim=(0, 1)).sqrt()
        elif qk_channel_equalization == "amax":
            q_stat = q_active.abs().amax(dim=(0, 1))
            k_stat = k_active.abs().amax(dim=(0, 1))
        else:
            raise ValueError(
                f"unsupported QK channel equalization: "
                f"{qk_channel_equalization}"
            )
        equalizer = (
            k_stat.clamp_min(1e-12)
            / q_stat.clamp_min(1e-12)
        ).pow(0.5 * qk_channel_equalization_strength)
        equalizer.clamp_(0.25, 4.0)
        q_ref[:, :, :heads, :dim].mul_(
            equalizer[None, None]
        )
        k_ref[:, :, :heads, :dim].mul_(
            equalizer.reciprocal()[None, None]
        )

    if qk_channel_permutation != "none":
        block_size = 32
        block_count = target_dim // block_size
        signal_block_count = block_count - 1
        if (
            target_dim % block_size
            or signal_block_count < 1
            or dim > signal_block_count * block_size
        ):
            raise ValueError(
                "QK channel spreading requires one spare 32-channel block"
            )
        active = torch.arange(dim, device=query.device)
        special_count = 1 + int(score_shift > 0.0)
        special = torch.arange(
            dim, dim + special_count, device=query.device
        )
        inactive = torch.arange(
            dim + special_count, target_dim, device=query.device
        )

        def make_permutation(active_order: Any) -> Any:
            bins = [
                active_order[index::signal_block_count].tolist()
                for index in range(signal_block_count)
            ]
            inactive_values = inactive.tolist()
            cursor = 0
            for channel_bin in bins:
                needed = block_size - len(channel_bin)
                channel_bin.extend(
                    inactive_values[cursor : cursor + needed]
                )
                cursor += needed
            final_bin = special.tolist() + inactive_values[cursor:]
            if len(final_bin) != block_size:
                raise RuntimeError("invalid QK channel-spread permutation")
            return torch.tensor(
                [
                    channel
                    for channel_bin in (*bins, final_bin)
                    for channel in channel_bin
                ],
                device=query.device,
                dtype=torch.long,
            )

        if qk_channel_permutation == "active-spread":
            permutation = make_permutation(active)
            q_ref = q_ref.index_select(-1, permutation)
            k_ref = k_ref.index_select(-1, permutation)
        elif qk_channel_permutation == "rms-balanced":
            for head in range(heads):
                q_rms = q_ref[:, :seqlen, head, :dim].square().mean(
                    dim=(0, 1)
                ).sqrt()
                k_rms = k_ref[:, :seqlen, head, :dim].square().mean(
                    dim=(0, 1)
                ).sqrt()
                importance = (
                    q_rms / q_rms.mean().clamp_min(1e-12)
                    + k_rms / k_rms.mean().clamp_min(1e-12)
                )
                active_order = torch.argsort(
                    importance, descending=True
                )
                permutation = make_permutation(active_order)
                q_ref[:, :, head] = q_ref[
                    :, :, head
                ].index_select(-1, permutation)
                k_ref[:, :, head] = k_ref[
                    :, :, head
                ].index_select(-1, permutation)
        else:
            raise ValueError(
                f"unsupported QK channel permutation: "
                f"{qk_channel_permutation}"
            )

    if qk_orthogonal_transform == "signal-dct":
        special_count = 1 + int(score_shift > 0.0)
        signal_width = target_dim - 32
        if dim > signal_width:
            raise ValueError(
                "signal DCT requires one spare 32-channel block"
            )
        special = torch.arange(
            dim, dim + special_count, device=query.device
        )
        inactive = torch.arange(
            dim + special_count, target_dim, device=query.device
        )
        signal_padding = signal_width - dim
        permutation = torch.cat(
            (
                torch.arange(dim, device=query.device),
                inactive[:signal_padding],
                special,
                inactive[signal_padding:],
            )
        )
        q_ref = q_ref.index_select(-1, permutation)
        k_ref = k_ref.index_select(-1, permutation)

        row = torch.arange(
            signal_width,
            device=query.device,
            dtype=torch.float32,
        )[:, None]
        column = torch.arange(
            signal_width,
            device=query.device,
            dtype=torch.float32,
        )[None, :]
        dct = torch.cos(
            math.pi / signal_width * (row + 0.5) * column
        )
        dct[:, 0].mul_(1.0 / math.sqrt(signal_width))
        dct[:, 1:].mul_(math.sqrt(2.0 / signal_width))
        q_ref[..., :signal_width] = torch.matmul(
            q_ref[..., :signal_width], dct
        )
        k_ref[..., :signal_width] = torch.matmul(
            k_ref[..., :signal_width], dct
        )
    elif qk_orthogonal_transform != "none":
        if target_dim < 1 or target_dim & (target_dim - 1):
            raise ValueError("Hadamard QK transform requires power-of-two D")
        hadamard = torch.ones(
            (1, 1),
            device=query.device,
            dtype=torch.float32,
        )
        while hadamard.shape[0] < target_dim:
            hadamard = torch.cat(
                (
                    torch.cat((hadamard, hadamard), dim=1),
                    torch.cat((hadamard, -hadamard), dim=1),
                ),
                dim=0,
            )
        hadamard.mul_(1.0 / math.sqrt(target_dim))
        if qk_orthogonal_transform == "signed-hadamard":
            head_index = torch.arange(
                target_heads,
                device=query.device,
                dtype=torch.int64,
            )[:, None]
            channel_index = torch.arange(
                target_dim,
                device=query.device,
                dtype=torch.int64,
            )[None, :]
            sign_bits = (
                head_index * 1103515245
                + channel_index * 2654435761
                + 12345
            ) & 1
            signs = sign_bits.to(torch.float32).mul_(2.0).sub_(1.0)
            q_ref.mul_(signs[None, None])
            k_ref.mul_(signs[None, None])
        elif qk_orthogonal_transform != "hadamard":
            raise ValueError(
                f"unsupported QK orthogonal transform: "
                f"{qk_orthogonal_transform}"
            )
        q_ref = torch.matmul(q_ref, hadamard)
        k_ref = torch.matmul(k_ref, hadamard)

    if qk_format == "nvfp4":
        fold_q = nv_qk_fold_k64_scales in ("q", "both")
        fold_k = nv_qk_fold_k64_scales in ("k", "both")
        if fold_q:
            q_fp4, q_scale = quantize_nvfp4_qk_folded_k64_scales(
                q_ref,
                1.0,
                nv_qk_fold_scale_select,
                nv_qk_fold_scale_multiplier,
            )
        else:
            q_fp4, q_scale = quantize_nvfp4_qk(q_ref)
        if fold_k:
            k_fp4, k_scale = quantize_nvfp4_qk_folded_k64_scales(
                k_ref,
                1.0,
                nv_qk_fold_scale_select,
                nv_qk_fold_scale_multiplier,
            )
        else:
            k_fp4, k_scale = quantize_nvfp4_qk(k_ref)
    elif qk_format == "mxfp4":
        q_fp4, q_scale = quantize_mxfp4_qk(
            q_ref, mx_q_quant_mode
        )
        k_fp4, k_scale = quantize_mxfp4_qk(
            k_ref, mx_k_quant_mode
        )
    else:
        raise ValueError(f"unsupported QK format: {qk_format}")

    if pv_format == "nvfp4":
        v_fp4, v_scale = quantize_nvfp4_v(v_ref)
    elif pv_format == "mxfp4":
        v_fp4, v_scale = quantize_mxfp4_v(
            v_ref, mx_v_quant_mode
        )
    else:
        raise ValueError(f"unsupported PV format: {pv_format}")

    prepared = prepare_native_inputs(
        q_fp4,
        k_fp4,
        v_fp4,
        q_scale,
        k_scale,
        v_scale,
        qk_format,
        pv_format,
        batch,
        target_seqlen,
        target_heads,
        target_dim,
        target_dim,
        compact_folded_qk_scales=compact_folded_qk_scales,
    )
    prepared.hao_q_fp4 = q_fp4
    prepared.hao_k_fp4 = k_fp4
    prepared.hao_v_fp4 = v_fp4
    prepared.hao_v_fp8 = v_ref.to(torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    prepared.hao_q_scale = q_scale
    prepared.hao_k_scale = k_scale
    prepared.hao_v_scale = v_scale
    if retain_replay_tensors:
        prepared.replay_q_ref = q_ref
        prepared.replay_k_ref = k_ref
        prepared.replay_v_ref = v_ref
    return prepared, seqlen, heads, dim


class RegularAttentionRunner:
    def __init__(
        self,
        extension: Any,
        *,
        layer_extensions: dict[int, Any] | None = None,
        attention_backend: str = "tk",
        hao_root: Path = DEFAULT_HAO_ROOT,
        mask_value: float,
        scale_factors: list[float],
        scale_sweep_samples: int,
        finite_diagnostics: bool = False,
        interleave_quarters: bool = False,
        global_anchor: bool = False,
        global_anchor_samples: int = 32,
        mx_q_quant_mode: int = 0,
        mx_k_quant_mode: int = 0,
        mx_v_quant_mode: int = 0,
        nv_qk_fold_k64_scales: str = "auto",
        nv_qk_fold_scale_select: str = "mse",
        nv_qk_fold_scale_multiplier: float = 1.0,
        qk_equalization: float = 1.0,
        qk_channel_equalization: str = "none",
        qk_channel_equalization_strength: float = 1.0,
        qk_channel_permutation: str = "none",
        qk_orthogonal_transform: str = "none",
        key_centering: str = "none",
        score_shift: float = 0.0,
        score_shift_predictor: str = "fixed",
        score_shift_bias: float = 0.0,
        p_replay_diagnostics: bool = False,
        p_replay_samples: int = 1,
        p_replay_layers: set[int] | None = None,
        p_replay_query_chunk: int = 64,
        p_replay_affine_search: bool = False,
        p_replay_nv_scale_search: bool = False,
        collect_layer_metrics: bool = True,
    ) -> None:
        import torch

        topology = dict(extension.read_hao_direct_topology())
        expected = {
            "batch": 1,
            "dqk": 128,
            "dvo": 128,
        }
        for key, value in expected.items():
            if int(topology[key]) != value:
                raise ValueError(
                    f"extension topology {key}={topology[key]}, expected {value}"
                )
        self.target_seqlen = int(topology["seqlen"])
        self.target_heads = int(topology["heads"])
        self.target_dim = int(topology["dqk"])
        self.extension = extension
        self.layer_extensions = dict(layer_extensions or {})
        self.layer_topologies: dict[int, dict[str, Any]] = {}
        compatible_keys = (
            "batch",
            "seqlen",
            "heads",
            "dqk",
            "dvo",
            "route",
            "qk_format",
            "pv_format",
        )
        for layer, layer_extension in self.layer_extensions.items():
            layer_topology = dict(
                layer_extension.read_hao_direct_topology()
            )
            for key in compatible_keys:
                if layer_topology.get(key) != topology.get(key):
                    raise ValueError(
                        f"layer {layer} extension topology {key}="
                        f"{layer_topology.get(key)}, expected "
                        f"{topology.get(key)}"
                    )
            self.layer_topologies[layer] = layer_topology
        self.topology = topology
        self.route = str(topology["route"])
        self.qk_format = (
            "mxfp4"
            if str(topology["qk_format"]).startswith("mxfp4")
            else "nvfp4"
        )
        self.pv_format = (
            "mxfp4"
            if str(topology["pv_format"]).startswith("mxfp4")
            else "nvfp4"
        )
        if attention_backend not in ("tk", "hao-native", "hao-fp8"):
            raise ValueError(
                f"unsupported attention backend: {attention_backend}"
            )
        if attention_backend == "hao-native" and (
            self.qk_format != "nvfp4" or self.pv_format != "nvfp4"
        ):
            raise ValueError(
                "HAO native downstream execution requires NVFP4 QK and PV"
            )
        if attention_backend == "hao-fp8" and self.qk_format != "nvfp4":
            raise ValueError("HAO FP8-PV execution requires NVFP4 QK")
        self.attention_backend = attention_backend
        self.hao_interface = None
        if attention_backend in ("hao-native", "hao-fp8"):
            import flash_attn

            hao_package = str((hao_root / "flash_attn").resolve())
            if hao_package not in flash_attn.__path__:
                flash_attn.__path__.insert(0, hao_package)
            from flash_attn.cute import interface

            self.hao_interface = interface
        fold_mask = int(topology.get("nv_qk_folded_k64_scale_mask", 0))
        topology_fold = {0: "none", 1: "q", 2: "k", 3: "both"}[
            fold_mask
        ]
        if nv_qk_fold_k64_scales == "auto":
            nv_qk_fold_k64_scales = topology_fold
        elif fold_mask and nv_qk_fold_k64_scales != topology_fold:
            raise ValueError(
                "extension topology requires --nv-qk-fold-k64-scales "
                f"{topology_fold}, got {nv_qk_fold_k64_scales}"
            )
        topology_anchor_samples = (
            128
            if bool(topology.get("mx_global_anchor128", False))
            else (
                64
                if bool(topology.get("nv_global_anchor64", False))
                else (
                    32
                    if (
                        bool(topology.get("nv_global_anchor32", False))
                        or bool(
                            topology.get("mx_global_anchor32", False)
                        )
                    )
                    else 0
                )
            )
        )
        if bool(topology_anchor_samples) != global_anchor:
            required = (
                f"--global-anchor-kv --global-anchor-samples "
                f"{topology_anchor_samples}"
                if topology_anchor_samples
                else "no global-anchor permutation"
            )
            raise ValueError(
                f"extension topology requires {required}"
            )
        if global_anchor and global_anchor_samples != topology_anchor_samples:
            raise ValueError(
                "extension topology requires "
                f"{topology_anchor_samples} global-anchor samples, got "
                f"{global_anchor_samples}"
            )
        self.mask_value = mask_value
        self.scale_factors = scale_factors
        self.scale_sweep_samples = scale_sweep_samples
        self.finite_diagnostics = finite_diagnostics
        self.interleave_quarters = interleave_quarters
        self.global_anchor = global_anchor
        self.global_anchor_samples = global_anchor_samples
        self.mx_q_quant_mode = mx_q_quant_mode
        self.mx_k_quant_mode = mx_k_quant_mode
        self.mx_v_quant_mode = mx_v_quant_mode
        self.nv_qk_fold_k64_scales = nv_qk_fold_k64_scales
        self.nv_qk_fold_scale_select = nv_qk_fold_scale_select
        self.nv_qk_fold_scale_multiplier = nv_qk_fold_scale_multiplier
        self.compact_folded_qk_scales = bool(
            topology.get("nv_qk_compact_folded_scales", False)
        )
        self.qk_equalization = qk_equalization
        self.qk_channel_equalization = qk_channel_equalization
        self.qk_channel_equalization_strength = (
            qk_channel_equalization_strength
        )
        self.qk_channel_permutation = qk_channel_permutation
        self.qk_orthogonal_transform = qk_orthogonal_transform
        self.key_centering = key_centering
        self.score_shift = score_shift
        self.score_shift_predictor = score_shift_predictor
        self.score_shift_bias = score_shift_bias
        self.p_replay_diagnostics = p_replay_diagnostics
        self.p_replay_samples = p_replay_samples
        self.p_replay_layers = p_replay_layers or {0}
        self.p_replay_query_chunk = p_replay_query_chunk
        self.p_replay_affine_search = p_replay_affine_search
        self.p_replay_nv_scale_search = p_replay_nv_scale_search
        self.collect_layer_metrics = collect_layer_metrics
        self.enabled = False
        self.sample_index = -1
        self.nonfinite_output_count = 0
        self.layer_metrics: dict[int, list[dict[str, float]]] = defaultdict(list)
        self.layer_finite_stats: dict[int, list[dict[str, Any]]] = defaultdict(
            list
        )
        self.scale_sweep: dict[
            float, list[dict[str, float]]
        ] = defaultdict(list)
        self.scale_stats: dict[str, list[dict[str, float]]] = defaultdict(list)
        self.p_replay_ranges: dict[
            int, list[dict[str, Any]]
        ] = defaultdict(list)
        self.p_replay_metrics: dict[
            str, list[dict[str, float]]
        ] = defaultdict(list)
        self._output = torch.empty(
            (
                1,
                self.target_seqlen,
                self.target_heads,
                self.target_dim,
            ),
            device="cuda",
            dtype=torch.bfloat16,
        )
        self._lse = torch.empty(
            (1, self.target_heads, 1, self.target_seqlen),
            device="cuda",
            dtype=torch.float32,
        )

    def begin_sample(self, sample_index: int) -> None:
        self.sample_index = sample_index

    def center_key(self, key: Any, projection_bias: Any | None) -> Any:
        if self.key_centering == "none":
            return key
        key32 = key.float()
        if self.key_centering == "sequence-mean":
            return key32 - key32.mean(dim=-2, keepdim=True)
        if projection_bias is None:
            raise ValueError(
                "projection-bias key centering requires a K projection bias"
            )
        return key32 - projection_bias.float().reshape(
            1,
            key.shape[1],
            1,
            key.shape[3],
        )

    @staticmethod
    def exact_attention(
        query: Any,
        key: Any,
        value: Any,
        key_valid_mask: Any | None = None,
    ) -> tuple[Any, Any]:
        import torch

        scores = torch.matmul(
            query.float(),
            key.float().transpose(-1, -2),
        ) / math.sqrt(query.shape[-1])
        if key_valid_mask is not None:
            scores = scores.masked_fill(
                ~key_valid_mask[:, None, None, :].to(torch.bool),
                -math.inf,
            )
        probability = torch.softmax(scores, dim=-1)
        context = torch.matmul(probability, value.float())
        return context, scores

    @staticmethod
    def scale_distribution(scales: Any) -> dict[str, float]:
        finite = scales.isfinite()
        if not bool(finite.any().item()):
            return {
                "minimum": math.nan,
                "maximum": math.nan,
                "fraction_below_e4m3_subnormal": math.nan,
                "fraction_below_e4m3_normal": math.nan,
                "fraction_above_e4m3_max": math.nan,
            }
        return {
            "minimum": float(scales[finite].min().item()),
            "maximum": float(scales[finite].max().item()),
            "fraction_below_e4m3_subnormal": float(
                (scales < E4M3_MIN_SUBNORMAL).float().mean().item()
            ),
            "fraction_below_e4m3_normal": float(
                (scales < E4M3_MIN_NORMAL).float().mean().item()
            ),
            "fraction_above_e4m3_max": float(
                (scales > E4M3_MAX).float().mean().item()
            ),
        }

    def analyze_p_scales(
        self,
        scores: Any,
        value: Any,
        exact_context: Any,
    ) -> None:
        import torch
        import torch.nn.functional as functional

        stabilized_p = torch.exp(
            scores - scores.amax(dim=-1, keepdim=True)
        )
        padded_length = math.ceil(stabilized_p.shape[-1] / 32) * 32
        stabilized_p_padded = functional.pad(
            stabilized_p,
            (0, padded_length - stabilized_p.shape[-1]),
        )
        blocks = stabilized_p_padded.reshape(
            *stabilized_p.shape[:-1],
            padded_length // 32,
            32,
        )
        stable_scales = blocks.amax(dim=-1) / 6.0
        shiftless = torch.exp(scores)
        shiftless_padded = functional.pad(
            shiftless,
            (0, padded_length - shiftless.shape[-1]),
        )
        shiftless_scales = shiftless_padded.reshape(
            *shiftless.shape[:-1],
            padded_length // 32,
            32,
        ).amax(dim=-1) / 6.0
        self.scale_stats["stable"].append(
            self.scale_distribution(stable_scales)
        )
        self.scale_stats["shiftless"].append(
            self.scale_distribution(shiftless_scales)
        )

        levels = torch.tensor(
            E2M1_LEVELS,
            device=blocks.device,
            dtype=torch.float32,
        )
        midpoints = torch.tensor(
            E2M1_MIDPOINTS,
            device=blocks.device,
            dtype=torch.float32,
        )
        for factor in self.scale_factors:
            scaled = stable_scales * factor
            underflow = (scaled < E4M3_MIN_SUBNORMAL).float().mean()
            subnormal = (scaled < E4M3_MIN_NORMAL).float().mean()
            overflow = (scaled > E4M3_MAX).float().mean()
            encoded = (
                scaled.clamp(max=E4M3_MAX)
                .to(torch.float8_e4m3fn)
                .float()
            )
            effective_scale = (encoded / factor).clamp_min(
                torch.finfo(torch.float32).tiny
            )
            normalized = blocks / effective_scale.unsqueeze(-1)
            code = (
                normalized.unsqueeze(-1) > midpoints
            ).sum(dim=-1)
            quantized = levels[code] * effective_scale.unsqueeze(-1)
            quantized = quantized.reshape(
                *stabilized_p.shape[:-1],
                padded_length,
            )[..., : stabilized_p.shape[-1]]
            denominator = stabilized_p.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(
                torch.finfo(torch.float32).tiny
            )
            quantized_context = torch.matmul(
                quantized / denominator,
                value.float(),
            )
            record = tensor_metrics(quantized_context, exact_context)
            record.update(
                {
                    "scale_underflow_fraction": float(underflow.item()),
                    "scale_subnormal_fraction": float(subnormal.item()),
                    "scale_overflow_fraction": float(overflow.item()),
                }
            )
            self.scale_sweep[factor].append(record)

    def replay_p_transforms(
        self,
        scores: Any,
        value: Any,
        exact_context: Any,
        prepared: Any,
        *,
        layer_index: int,
    ) -> None:
        import torch

        keys = scores.shape[-1]
        sample_indices = {}
        anchors = {
            "exact": scores.amax(dim=-1, keepdim=True),
            "zero": torch.zeros_like(scores[..., :1]),
        }
        for samples in (32, 64):
            indices = (
                torch.linspace(
                    0,
                    keys - 1,
                    samples,
                    device=scores.device,
                    dtype=torch.float32,
                )
                .round()
                .to(torch.long)
            )
            sample_indices[samples] = indices
            anchors[f"sample{samples}"] = (
                scores.index_select(-1, indices)
                .amax(dim=-1, keepdim=True)
            )

        exact_max = anchors["exact"]
        anchor_range = {}
        for name, anchor in anchors.items():
            miss = exact_max - anchor
            log2_mass = torch.logsumexp(
                scores - anchor,
                dim=-1,
            ) * LOG2_E
            anchor_range[name] = {
                "anchor": distribution_quantiles(anchor),
                "max_miss": distribution_quantiles(miss),
                "log2_mass": distribution_quantiles(log2_mass),
                "fraction_log2_mass_above_120": float(
                    (log2_mass > 120.0).float().mean().item()
                ),
                "fraction_log2_mass_above_127": float(
                    (log2_mass > 127.0).float().mean().item()
                ),
            }
        self.p_replay_ranges[layer_index].append(
            {
                "sample": self.sample_index,
                "score": distribution_quantiles(scores),
                "anchors": anchor_range,
            }
        )

        candidates = [
            ("mx-exact-anchor-exact-represented", "exact", "exact",
             "represented", 0.0),
            ("mx-sample32-exact-represented", "sample32", "exact",
             "represented", 0.0),
            ("mx-sample64-exact-represented", "sample64", "exact",
             "represented", 0.0),
            ("mx-sample32-fast-represented", "sample32", "fast",
             "represented", 0.0),
            ("mx-sample32-fast-sampled", "sample32", "fast",
             "sampled", 0.0),
            ("mx-sample32-fast-native8-sampled", "sample32", "fast",
             "native8-sampled", 0.0),
            ("mx-sample32-fast-native8-prior6", "sample32", "fast",
             "native8-prior6", 0.0),
            ("mx-sample32-fast-native8-replace-max6", "sample32", "fast",
             "native8-replace-max6", 0.0),
            ("mx-sample32-fast-native4-max-cv", "sample32", "fast",
             "native4-max-cv", 0.0),
            ("mx-sample32-fast-native8-max-cv", "sample32", "fast",
             "native8-max-cv", 0.0),
            ("mx-sample32-fast-word8-max-cv", "sample32", "fast",
             "word8-max-cv", 0.0),
        ]
        for bias in (-0.5, -0.25, 0.25, 0.5):
            candidates.append(
                (
                    f"mx-sample32-fast-represented-bias{bias:+.2f}",
                    "sample32",
                    "fast",
                    "represented",
                    bias,
                )
            )
        for (
            name,
            anchor_name,
            encoder,
            denominator,
            scale_bias,
        ) in candidates:
            replay_context = mxfp4_replay_context(
                scores,
                value,
                anchor=anchors[anchor_name],
                encoder=encoder,
                denominator=denominator,
                scale_select_bias=scale_bias,
                query_chunk=self.p_replay_query_chunk,
            )
            record = tensor_metrics(replay_context, exact_context)
            record.update(
                {
                    "sample": float(self.sample_index),
                    "layer": float(layer_index),
                }
            )
            self.p_replay_metrics[name].append(record)

        if self.p_replay_affine_search or self.p_replay_nv_scale_search:
            query_count = min(32, scores.shape[-2])
            query_indices = (
                torch.linspace(
                    0,
                    scores.shape[-2] - 1,
                    query_count,
                    device=scores.device,
                    dtype=torch.float32,
                )
                .round()
                .to(torch.long)
            )
            fit_context = exact_context.index_select(-2, query_indices)
            model_heads = scores.shape[1]
            model_dim = value.shape[-1]
            q_ref = prepared.replay_q_ref
            k_ref = prepared.replay_k_ref
            v_ref = prepared.replay_v_ref
            batch, target_keys, target_heads, target_dim = q_ref.shape

            if self.qk_format == "mxfp4":
                quantized_q = dequantize_prepared_mxfp4_qk(
                    prepared.q_fp4_bhsd,
                    prepared.q_scale_prepared,
                )
                quantized_k = dequantize_prepared_mxfp4_qk(
                    prepared.k_fp4_bhsd,
                    prepared.k_scale_prepared,
                )
            else:
                quantized_q = nvfp4_roundtrip_linear(
                    q_ref.reshape(batch * target_keys, target_heads * target_dim)
                ).reshape(
                    batch, target_keys, target_heads, target_dim
                ).permute(0, 2, 1, 3)
                quantized_k = nvfp4_roundtrip_linear(
                    k_ref.reshape(batch * target_keys, target_heads * target_dim)
                ).reshape(
                    batch, target_keys, target_heads, target_dim
                ).permute(0, 2, 1, 3)

            if self.pv_format == "mxfp4":
                quantized_v = dequantize_prepared_mxfp4_v(
                    prepared.v_fp4_bhds,
                    prepared.v_scale_prepared,
                )
            else:
                v_k_major = (
                    v_ref.permute(0, 2, 3, 1)
                    .contiguous()
                    .reshape(batch * target_heads * target_dim, target_keys)
                )
                quantized_v = nvfp4_roundtrip_linear(v_k_major).reshape(
                    batch, target_heads, target_dim, target_keys
                ).transpose(-2, -1)

            fit_scores = torch.matmul(
                quantized_q[:, :model_heads].index_select(-2, query_indices),
                quantized_k[:, :model_heads].transpose(-1, -2),
            ) / math.sqrt(self.target_dim)
            fit_value = quantized_v[:, :model_heads, :, :model_dim]
            exact_fit_scores = scores.index_select(-2, query_indices)
            exact_fit_value = value
            valid_keys = exact_fit_scores.shape[-1]
            if valid_keys < target_keys:
                exact_fit_scores = torch.nn.functional.pad(
                    exact_fit_scores,
                    (0, target_keys - valid_keys),
                    value=-math.inf,
                )
                exact_fit_value = torch.nn.functional.pad(
                    exact_fit_value,
                    (0, 0, 0, target_keys - valid_keys),
                )
            if self.global_anchor:
                anchor_indices = (
                    torch.linspace(
                        0, valid_keys - 1,
                        self.global_anchor_samples,
                        device=scores.device,
                        dtype=torch.float32,
                    )
                    .round()
                    .to(torch.long)
                )
                selected = torch.zeros(
                    target_keys,
                    device=scores.device,
                    dtype=torch.bool,
                )
                selected[anchor_indices] = True
                remainder = torch.arange(
                    target_keys,
                    device=scores.device,
                )[~selected]
                order = torch.cat((anchor_indices, remainder))
                exact_fit_scores = exact_fit_scores.index_select(-1, order)
                exact_fit_value = exact_fit_value.index_select(-2, order)
                fit_anchor = fit_scores[..., : self.global_anchor_samples].amax(
                    dim=-1, keepdim=True
                )
            elif (
                self.pv_format == "mxfp4"
                and bool(self.topology.get("mx_shiftless_softmax", False))
            ):
                fit_anchor = torch.zeros_like(fit_scores[..., :1])
            else:
                fit_anchor = fit_scores.index_select(
                    -1, sample_indices[32]
                ).amax(dim=-1, keepdim=True)

            exact_probability = torch.softmax(
                exact_fit_scores.float(), dim=-1
            )
            quantized_probability = torch.softmax(
                fit_scores.float(), dim=-1
            )
            route_prefix = f"{self.qk_format}-{self.pv_format}"
            component_contexts = {
                f"{route_prefix}-component-qk-only": torch.matmul(
                    quantized_probability, exact_fit_value.float()
                ),
                f"{route_prefix}-component-v-only": torch.matmul(
                    exact_probability, fit_value.float()
                ),
                f"{route_prefix}-component-qkv": torch.matmul(
                    quantized_probability, fit_value.float()
                ),
            }
            if self.pv_format == "mxfp4":
                component_contexts[
                    f"{route_prefix}-component-qkv-p-exact-pack"
                ] = mxfp4_replay_context(
                    fit_scores,
                    fit_value,
                    anchor=fit_anchor,
                    encoder="exact",
                    denominator="represented",
                    scale_select_bias=0.0,
                    query_chunk=self.p_replay_query_chunk,
                )
            else:
                nv_exact_pack, nv_scale_stats = nvfp4_replay_context(
                    fit_scores,
                    fit_value,
                    anchor=fit_anchor,
                    query_chunk=self.p_replay_query_chunk,
                    affine_a=float(self.topology["nv_affine_a"]),
                    affine_b=float(self.topology["nv_affine_b"]),
                    p_global_log2=float(
                        self.topology["nv_p_global_log2"]
                    ),
                    exact_exp2=True,
                )
                component_contexts[
                    f"{route_prefix}-component-qkv-p-exact-pack"
                ] = nv_exact_pack
            for name, component_context in component_contexts.items():
                record = tensor_metrics(component_context, fit_context)
                record.update(
                    {
                        "sample": float(self.sample_index),
                        "layer": float(layer_index),
                        "query_rows": float(query_count),
                    }
                )
                if name.endswith("qkv-p-exact-pack") and (
                    self.pv_format == "nvfp4"
                ):
                    record.update(nv_scale_stats)
                self.p_replay_metrics[name].append(record)

            affine_a_values = (
                0.75,
                0.90,
                1.00,
                1.05,
                1.10,
                1.20,
                1.30,
                1.40,
                1.45,
                1.50,
                1.55,
                1.60,
                1.62330034,
                1.65,
                1.70,
                1.75,
                1.80,
                1.90,
                2.00,
                2.10,
                2.20,
                2.40,
                2.60,
                2.80,
                3.00,
            ) if self.p_replay_affine_search else ()
            affine_b_values = (
                -1.00,
                -0.80,
                -0.60,
                -0.40,
                -0.20,
                0.00,
                0.10,
                0.20,
                0.30,
                0.40,
                0.60,
                0.70,
                0.80,
                0.90,
                1.00,
                1.20,
                1.40,
                1.52631878,
                1.60,
                1.80,
                2.00,
            ) if self.p_replay_affine_search else ()
            for affine_a in affine_a_values:
                for affine_b in affine_b_values:
                    if self.pv_format == "mxfp4":
                        replay_context = mxfp4_replay_context(
                            fit_scores,
                            fit_value,
                            anchor=fit_anchor,
                            encoder="fast",
                            denominator="represented",
                            scale_select_bias=0.0,
                            query_chunk=self.p_replay_query_chunk,
                            affine_a=affine_a,
                            affine_b=affine_b,
                        )
                        scale_stats = {}
                    else:
                        replay_context, scale_stats = nvfp4_replay_context(
                            fit_scores,
                            fit_value,
                            anchor=fit_anchor,
                            query_chunk=self.p_replay_query_chunk,
                            affine_a=affine_a,
                            affine_b=affine_b,
                            p_global_log2=float(
                                self.topology["nv_p_global_log2"]
                            ),
                        )
                    name = (
                        f"{self.pv_format}-affine-search-a{affine_a:.8f}"
                        f"-b{affine_b:.8f}"
                    )
                    record = tensor_metrics(
                        replay_context, fit_context
                    )
                    record.update(scale_stats)
                    record.update(
                        {
                            "sample": float(self.sample_index),
                            "layer": float(layer_index),
                            "affine_a": affine_a,
                            "affine_b": affine_b,
                            "query_rows": float(query_count),
                        }
                    )
                    self.p_replay_metrics[name].append(record)

            if self.pv_format == "nvfp4":
                affine_a = float(self.topology["nv_affine_a"])
                affine_b = float(self.topology["nv_affine_b"])
                for p_global_log2 in (
                    -6.0,
                    -4.0,
                    -3.0,
                    -2.0,
                    -1.0,
                    0.0,
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    6.0,
                ):
                    replay_context, scale_stats = nvfp4_replay_context(
                        fit_scores,
                        fit_value,
                        anchor=fit_anchor,
                        query_chunk=self.p_replay_query_chunk,
                        affine_a=affine_a,
                        affine_b=affine_b,
                        p_global_log2=p_global_log2,
                    )
                    name = f"nvfp4-global-log2-{p_global_log2:+.1f}"
                    record = tensor_metrics(replay_context, fit_context)
                    record.update(scale_stats)
                    record.update(
                        {
                            "sample": float(self.sample_index),
                            "layer": float(layer_index),
                            "p_global_log2": p_global_log2,
                            "query_rows": float(query_count),
                        }
                    )
                    self.p_replay_metrics[name].append(record)

    def __call__(
        self,
        query: Any,
        key: Any,
        value: Any,
        *,
        layer_index: int,
        invocation_index: int | None = None,
        key_valid_mask: Any | None = None,
    ) -> Any:
        import torch

        layer_topology = self.layer_topologies.get(
            layer_index, self.topology
        )
        layer_anchor_samples = (
            128
            if bool(layer_topology.get("mx_global_anchor128", False))
            else (
                64
                if bool(layer_topology.get("nv_global_anchor64", False))
                else (
                    32
                    if bool(
                        layer_topology.get("nv_global_anchor32", False)
                    )
                    or bool(
                        layer_topology.get("mx_global_anchor32", False)
                    )
                    else 0
                )
            )
        )
        layer_fold_mask = int(
            layer_topology.get("nv_qk_folded_k64_scale_mask", 0)
        )
        layer_fold_mode = {0: "none", 1: "q", 2: "k", 3: "both"}[
            layer_fold_mask
        ]
        layer_compact_folded_scales = bool(
            layer_topology.get("nv_qk_compact_folded_scales", False)
        )
        prepared, seqlen, heads, dim = prepare_kernel_inputs(
            query,
            key,
            value,
            qk_format=self.qk_format,
            pv_format=self.pv_format,
            target_seqlen=self.target_seqlen,
            target_heads=self.target_heads,
            target_dim=self.target_dim,
            mask_value=self.mask_value,
            key_valid_mask=key_valid_mask,
            interleave_quarters=self.interleave_quarters,
            global_anchor=bool(layer_anchor_samples),
            global_anchor_samples=layer_anchor_samples or 32,
            mx_q_quant_mode=self.mx_q_quant_mode,
            mx_k_quant_mode=self.mx_k_quant_mode,
            mx_v_quant_mode=self.mx_v_quant_mode,
            nv_qk_fold_k64_scales=layer_fold_mode,
            nv_qk_fold_scale_select=self.nv_qk_fold_scale_select,
            nv_qk_fold_scale_multiplier=(
                self.nv_qk_fold_scale_multiplier
            ),
            compact_folded_qk_scales=layer_compact_folded_scales,
            qk_equalization=self.qk_equalization,
            qk_channel_equalization=self.qk_channel_equalization,
            qk_channel_equalization_strength=(
                self.qk_channel_equalization_strength
            ),
            qk_channel_permutation=self.qk_channel_permutation,
            qk_orthogonal_transform=self.qk_orthogonal_transform,
            score_shift=self.score_shift,
            score_shift_predictor=self.score_shift_predictor,
            score_shift_bias=self.score_shift_bias,
            retain_replay_tensors=(
                self.p_replay_diagnostics
                and self.sample_index < self.p_replay_samples
                and layer_index in self.p_replay_layers
            ),
        )
        if self.attention_backend in ("hao-native", "hao-fp8"):
            hao_fp8_pv = self.attention_backend == "hao-fp8"
            self.hao_interface._flash_attn_fwd(
                prepared.hao_q_fp4,
                prepared.hao_k_fp4,
                prepared.hao_v_fp8 if hao_fp8_pv else prepared.hao_v_fp4,
                mSFQ=prepared.hao_q_scale,
                mSFK=prepared.hao_k_scale,
                mSFV=None if hao_fp8_pv else prepared.hao_v_scale,
                out=self._output,
                causal=False,
                return_lse=False,
                num_splits=1,
                pack_gqa=False,
                _compute_capability=10,
            )
        else:
            previous_route = os.environ.get("TK_FA4_FP4PV_FWD_CONFIG")
            os.environ["TK_FA4_FP4PV_FWD_CONFIG"] = self.route
            extension = self.layer_extensions.get(
                layer_index, self.extension
            )
            try:
                extension.forward_hao_direct_fp4pv(
                    prepared.q_fp4_bhsd,
                    prepared.q_scale_prepared,
                    prepared.q_global_scale,
                    prepared.k_fp4_bhsd,
                    prepared.k_scale_prepared,
                    prepared.k_global_scale,
                    prepared.v_fp4_bhds,
                    prepared.v_scale_prepared,
                    self._output,
                    self._lse,
                    0,
                    True,
                    self.finite_diagnostics,
                )
            finally:
                if previous_route is None:
                    os.environ.pop("TK_FA4_FP4PV_FWD_CONFIG", None)
                else:
                    os.environ[
                        "TK_FA4_FP4PV_FWD_CONFIG"
                    ] = previous_route
        context = (
            self._output[:, :seqlen, :heads, :dim]
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        needs_reference = (
            self.collect_layer_metrics
            or self.finite_diagnostics
            or (
                self.p_replay_diagnostics
                and self.sample_index < self.p_replay_samples
                and layer_index in self.p_replay_layers
            )
            or (
                self.pv_format == "nvfp4"
                and self.sample_index < self.scale_sweep_samples
                and layer_index == 0
            )
        )
        exact_context = None
        scores = None
        if needs_reference:
            exact_context, scores = self.exact_attention(
                query,
                key,
                value,
                key_valid_mask,
            )
        if (
            self.p_replay_diagnostics
            and self.sample_index < self.p_replay_samples
            and layer_index in self.p_replay_layers
        ):
            assert scores is not None and exact_context is not None
            self.replay_p_transforms(
                scores,
                value,
                exact_context,
                prepared,
                layer_index=layer_index,
            )
        finite_output_rows = context.isfinite().all(dim=-1)
        bad_output_rows = finite_output_rows.logical_not().nonzero()
        bad_output_count = int(bad_output_rows.shape[0])
        self.nonfinite_output_count += bad_output_count
        if bad_output_count and not self.finite_diagnostics:
            first = bad_output_rows[0]
            batch_index, head_index, row_index = (
                int(item.item()) for item in first
            )
            if scores is None:
                row_scores = torch.matmul(
                    query[batch_index, head_index, row_index].float(),
                    key[batch_index, head_index].float().transpose(0, 1),
                ) / math.sqrt(query.shape[-1])
                if key_valid_mask is not None:
                    row_scores = row_scores.masked_fill(
                        ~key_valid_mask[batch_index].to(torch.bool),
                        -math.inf,
                    )
            else:
                row_scores = scores[
                    batch_index, head_index, row_index
                ]
            finite_row_scores = row_scores[torch.isfinite(row_scores)]
            score_detail = "no finite exact scores"
            if finite_row_scores.numel():
                query_row = query[
                    batch_index, head_index, row_index
                ].float()
                key_head = key[batch_index, head_index].float()
                value_head = value[batch_index, head_index].float()
                anchor_detail = ""
                if layer_anchor_samples:
                    exact_max = row_scores.amax()
                    anchor_misses = []
                    for sample_count in (32, 64, 128):
                        anchor_indices = global_anchor_indices(
                            row_scores.shape[-1],
                            sample_count,
                            row_scores.device,
                        )
                        sampled_max = row_scores.index_select(
                            0, anchor_indices
                        ).amax()
                        anchor_misses.append(
                            f"anchor{sample_count}_max="
                            f"{float(sampled_max.item()):.6f}, "
                            f"anchor{sample_count}_miss="
                            f"{float((exact_max - sampled_max).item()):.6f}"
                        )
                    anchor_detail = ", " + ", ".join(anchor_misses)
                score_detail = (
                    "exact_score_min="
                    f"{float(finite_row_scores.min().item()):.6f}, "
                    "exact_score_max="
                    f"{float(finite_row_scores.max().item()):.6f}, "
                    "exact_logsumexp="
                    f"{float(torch.logsumexp(finite_row_scores, dim=0).item()):.6f}, "
                    "query_row_rms="
                    f"{float(query_row.square().mean().sqrt().item()):.6f}, "
                    "query_row_max_abs="
                    f"{float(query_row.abs().max().item()):.6f}, "
                    "key_head_rms="
                    f"{float(key_head.square().mean().sqrt().item()):.6f}, "
                    "key_head_max_abs="
                    f"{float(key_head.abs().max().item()):.6f}, "
                    "value_head_rms="
                    f"{float(value_head.square().mean().sqrt().item()):.6f}, "
                    "value_head_max_abs="
                    f"{float(value_head.abs().max().item()):.6f}"
                    f"{anchor_detail}"
                )
            raise FloatingPointError(
                "FP4 attention produced a non-finite row at "
                f"sample={self.sample_index}, layer={layer_index}, "
                f"invocation={invocation_index}, "
                f"batch={batch_index}, head={head_index}, row={row_index}; "
                f"{score_detail}"
            )
        if self.finite_diagnostics:
            assert scores is not None
            bad_row_records = []
            for index in bad_output_rows[:32]:
                batch_index, head_index, row_index = (
                    int(item.item()) for item in index
                )
                row_scores = scores[
                    batch_index, head_index, row_index
                ]
                record = {
                    "batch": batch_index,
                    "head": head_index,
                    "row": row_index,
                    "score_minimum": float(row_scores.min().item()),
                    "score_maximum": float(row_scores.max().item()),
                    "kernel_lse_payload": float(
                        self._lse[
                            batch_index,
                            head_index,
                            0,
                            row_index,
                        ].item()
                    ),
                }
                if self.global_anchor:
                    anchor_candidates = {}
                    for sample_count in (32, 64, 128):
                        anchor_indices = (
                            torch.linspace(
                                0,
                                row_scores.shape[-1] - 1,
                                sample_count,
                                device=row_scores.device,
                                dtype=torch.float32,
                            )
                            .round()
                            .to(torch.long)
                        )
                        sampled_anchor = row_scores.index_select(
                            0, anchor_indices
                        ).amax()
                        anchor_candidates[str(sample_count)] = {
                            "sampled_anchor": float(
                                sampled_anchor.item()
                            ),
                            "max_miss": float(
                                (
                                    row_scores.amax() -
                                    sampled_anchor
                                ).item()
                            ),
                            "log2_mass": float(
                                (
                                    torch.logsumexp(
                                        row_scores - sampled_anchor,
                                        dim=0,
                                    )
                                    * LOG2_E
                                ).item()
                            ),
                        }
                    selected_anchor = anchor_candidates[
                        str(self.global_anchor_samples)
                    ]
                    record.update(
                        {
                            "sampled_anchor": selected_anchor[
                                "sampled_anchor"
                            ],
                            "anchor_max_miss": selected_anchor[
                                "max_miss"
                            ],
                            "anchor_log2_mass": selected_anchor[
                                "log2_mass"
                            ],
                            "anchor_candidates": anchor_candidates,
                        }
                    )
                bad_row_records.append(record)
            self.layer_finite_stats[layer_index].append(
                {
                    "sample": self.sample_index,
                    "query": tensor_finite_stats(query),
                    "key": tensor_finite_stats(key),
                    "value": tensor_finite_stats(value),
                    "scores": tensor_finite_stats(scores),
                    "q_scale": tensor_finite_stats(
                        prepared.q_scale_prepared
                    ),
                    "k_scale": tensor_finite_stats(
                        prepared.k_scale_prepared
                    ),
                    "v_scale": tensor_finite_stats(
                        prepared.v_scale_prepared
                    ),
                    "output": tensor_finite_stats(context),
                    "bad_output_rows": bad_row_records,
                    "lse": tensor_finite_stats(
                        self._lse[:, :heads, :, :seqlen]
                    ),
                }
            )
        if self.collect_layer_metrics:
            assert exact_context is not None
            self.layer_metrics[layer_index].append(
                tensor_metrics(context, exact_context)
            )
        if (
            self.pv_format == "nvfp4"
            and self.sample_index < self.scale_sweep_samples
            and layer_index == 0
        ):
            assert scores is not None and exact_context is not None
            self.analyze_p_scales(scores, value, exact_context)
        return context

    def summary(self) -> dict[str, Any]:
        return {
            "attention_backend": self.attention_backend,
            "topology": self.topology,
            "layer_extension_topologies": {
                str(layer): topology
                for layer, topology in sorted(
                    self.layer_topologies.items()
                )
            },
            "formats": {
                "qk": self.qk_format,
                "pv": self.pv_format,
            },
            "interleave_kv_quarters": self.interleave_quarters,
            "global_anchor_kv": self.global_anchor,
            "global_anchor_samples": self.global_anchor_samples,
            "mx_quant_modes": {
                "q": self.mx_q_quant_mode,
                "k": self.mx_k_quant_mode,
                "v": self.mx_v_quant_mode,
            },
            "nv_qk_fold_k64_scales": self.nv_qk_fold_k64_scales,
            "nv_qk_fold_scale_select": self.nv_qk_fold_scale_select,
            "nv_qk_fold_scale_multiplier": (
                self.nv_qk_fold_scale_multiplier
            ),
            "qk_equalization": self.qk_equalization,
            "qk_channel_equalization": self.qk_channel_equalization,
            "qk_channel_equalization_strength": (
                self.qk_channel_equalization_strength
            ),
            "qk_channel_permutation": self.qk_channel_permutation,
            "qk_orthogonal_transform": self.qk_orthogonal_transform,
            "key_centering": self.key_centering,
            "score_shift": self.score_shift,
            "score_shift_predictor": self.score_shift_predictor,
            "score_shift_bias": self.score_shift_bias,
            "collect_layer_metrics": self.collect_layer_metrics,
            "all_outputs_finite": self.nonfinite_output_count == 0,
            "nonfinite_output_rows": self.nonfinite_output_count,
            "layer_output_error": {
                str(layer): mean_records(records)
                for layer, records in sorted(self.layer_metrics.items())
            },
            "layer_finite_stats": {
                str(layer): records
                for layer, records in sorted(
                    self.layer_finite_stats.items()
                )
            },
            "p_scale_sweep": {
                str(factor): mean_records(records)
                for factor, records in self.scale_sweep.items()
            },
            "p_scale_sweep_applicable": self.pv_format == "nvfp4",
            "p_scale_distribution": {
                name: summarize_scale_records(records)
                for name, records in self.scale_stats.items()
            },
            "p_replay": {
                "enabled": self.p_replay_diagnostics,
                "samples": self.p_replay_samples,
                "layers": sorted(self.p_replay_layers),
                "ranges": {
                    str(layer): records
                    for layer, records in sorted(
                        self.p_replay_ranges.items()
                    )
                },
                "candidate_metrics": {
                    name: mean_records(records)
                    for name, records in self.p_replay_metrics.items()
                },
                "candidate_metrics_by_layer": {
                    str(layer): {
                        name: mean_records(
                            [
                                record
                                for record in records
                                if int(record["layer"]) == layer
                            ]
                        )
                        for name, records in self.p_replay_metrics.items()
                        if any(
                            int(record["layer"]) == layer
                            for record in records
                        )
                    }
                    for layer in sorted(self.p_replay_layers)
                },
            },
        }


def install_vit_attention(model: Any, runner: RegularAttentionRunner) -> None:
    for layer_index, layer in enumerate(model.vit.encoder.layer):
        attention = layer.attention.attention
        original_forward = attention.forward

        def patched_forward(
            this: Any,
            hidden_states: Any,
            head_mask: Any = None,
            output_attentions: bool = False,
            *,
            _layer_index: int = layer_index,
            _original_forward: Any = original_forward,
        ) -> tuple[Any, ...]:
            if not runner.enabled:
                return _original_forward(
                    hidden_states,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                )
            if head_mask is not None:
                raise ValueError("head masks are unsupported by the FP4 adapter")
            if output_attentions:
                raise ValueError(
                    "attention-probability output is unsupported by the FP4 adapter"
                )
            mixed_query = this.query(hidden_states)
            query = this.transpose_for_scores(mixed_query)
            key = this.transpose_for_scores(this.key(hidden_states))
            key = runner.center_key(key, this.key.bias)
            value = this.transpose_for_scores(this.value(hidden_states))
            context = runner(
                query,
                key,
                value,
                layer_index=_layer_index,
            )
            context = context.permute(0, 2, 1, 3).contiguous()
            context = context.view(
                *context.shape[:-2],
                this.all_head_size,
            )
            return (context,)

        attention.forward = types.MethodType(patched_forward, attention)


def classification_metrics(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    import torch

    count = len(records)
    baseline_correct = sum(
        record["baseline_prediction"] == record["label"]
        for record in records
    )
    fp4_correct = sum(
        record["fp4_prediction"] == record["label"]
        for record in records
    )
    agreement = sum(
        record["baseline_prediction"] == record["fp4_prediction"]
        for record in records
    )
    baseline_logits = torch.tensor(
        [record["baseline_logits"] for record in records],
        dtype=torch.float32,
    )
    fp4_logits = torch.tensor(
        [record["fp4_logits"] for record in records],
        dtype=torch.float32,
    )
    baseline_log_probability = baseline_logits.log_softmax(dim=-1)
    fp4_log_probability = fp4_logits.log_softmax(dim=-1)
    kl = torch.nn.functional.kl_div(
        fp4_log_probability,
        baseline_log_probability.exp(),
        reduction="batchmean",
    )
    return {
        "samples": count,
        "baseline_accuracy": baseline_correct / count,
        "fp4_accuracy": fp4_correct / count,
        "top1_agreement": agreement / count,
        "logit_error": tensor_metrics(fp4_logits, baseline_logits),
        "mean_kl_fp4_vs_baseline": float(kl.item()),
    }


def main() -> None:
    import torch
    from datasets import load_dataset
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    args = parse_args()
    model_identity = portable_asset_identity(
        args.model,
        identifier=args.model_identifier,
        revision=args.model_revision,
        tree_sha256=args.model_tree_sha256,
    )
    dataset_identity = portable_asset_identity(
        args.dataset,
        identifier=args.dataset_identifier,
        revision=args.dataset_revision,
        tree_sha256=args.dataset_tree_sha256,
    )
    extension_identity = portable_file_identity(args.extension)
    if args.samples < 1:
        raise ValueError("--samples must be positive")
    if args.image_size < 0:
        raise ValueError("--image-size cannot be negative")
    if args.scale_sweep_samples < 0:
        raise ValueError("--scale-sweep-samples cannot be negative")
    if args.progress_every < 1:
        raise ValueError("--progress-every must be positive")
    if not math.isfinite(args.score_shift) or args.score_shift < 0.0:
        raise ValueError("--score-shift must be finite and nonnegative")
    if not math.isfinite(args.score_shift_bias):
        raise ValueError("--score-shift-bias must be finite")
    if (
        not math.isfinite(args.qk_equalization)
        or args.qk_equalization <= 0.0
    ):
        raise ValueError("--qk-equalization must be finite and positive")
    if (
        not math.isfinite(args.qk_channel_equalization_strength)
        or args.qk_channel_equalization_strength < 0.0
    ):
        raise ValueError(
            "--qk-channel-equalization-strength must be finite and "
            "nonnegative"
        )
    if (
        not math.isfinite(args.nv_qk_fold_scale_multiplier)
        or args.nv_qk_fold_scale_multiplier <= 0.0
    ):
        raise ValueError(
            "--nv-qk-fold-scale-multiplier must be finite and positive"
        )
    scale_factors = [
        float(item) for item in args.scale_factors.split(",") if item
    ]
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.set_device(0)

    extension = load_extension(
        args.extension.resolve(),
        args.extension_module,
    )
    layer_extensions: dict[int, Any] = {}
    for specification in args.layer_extension:
        try:
            layers_text, extension_text = specification.split("=", 1)
            path_text, module_name = extension_text.rsplit(":", 1)
        except ValueError as error:
            raise ValueError(
                "--layer-extension must use LAYERS=PATH:MODULE"
            ) from error
        layer_extension = load_extension(
            Path(path_text).resolve(),
            module_name,
        )
        for layer in parse_layer_indices(layers_text):
            if layer in layer_extensions:
                raise ValueError(
                    f"multiple extensions specified for layer {layer}"
                )
            layer_extensions[layer] = layer_extension
    runner = RegularAttentionRunner(
        extension,
        layer_extensions=layer_extensions,
        attention_backend=args.attention_backend,
        hao_root=args.hao_root,
        mask_value=args.mask_value,
        scale_factors=scale_factors,
        scale_sweep_samples=args.scale_sweep_samples,
        finite_diagnostics=args.finite_diagnostics,
        interleave_quarters=args.interleave_kv_quarters,
        global_anchor=args.global_anchor_kv,
        global_anchor_samples=args.global_anchor_samples,
        mx_q_quant_mode=MX_QUANT_MODES[args.mx_q_quant_mode],
        mx_k_quant_mode=MX_QUANT_MODES[args.mx_k_quant_mode],
        mx_v_quant_mode=MX_QUANT_MODES[args.mx_v_quant_mode],
        nv_qk_fold_k64_scales=args.nv_qk_fold_k64_scales,
        nv_qk_fold_scale_select=args.nv_qk_fold_scale_select,
        nv_qk_fold_scale_multiplier=args.nv_qk_fold_scale_multiplier,
        qk_equalization=args.qk_equalization,
        qk_channel_equalization=args.qk_channel_equalization,
        qk_channel_equalization_strength=(
            args.qk_channel_equalization_strength
        ),
        qk_channel_permutation=args.qk_channel_permutation,
        qk_orthogonal_transform=args.qk_orthogonal_transform,
        key_centering=args.key_centering,
        score_shift=args.score_shift,
        score_shift_predictor=args.score_shift_predictor,
        score_shift_bias=args.score_shift_bias,
        p_replay_diagnostics=args.p_replay_diagnostics,
        p_replay_samples=args.p_replay_samples,
        p_replay_layers={
            int(value)
            for value in args.p_replay_layers.split(",")
            if value.strip()
        },
        p_replay_query_chunk=args.p_replay_query_chunk,
        p_replay_affine_search=args.p_replay_affine_search,
        p_replay_nv_scale_search=args.p_replay_nv_scale_search,
    )
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForImageClassification.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
    ).eval().cuda()
    install_vit_attention(model, runner)
    dataset = load_dataset(args.dataset, split=args.split)
    if args.samples > len(dataset):
        raise ValueError(
            f"requested {args.samples} samples from {len(dataset)}-item split"
        )

    records = []
    processed_image_size: tuple[int, int] | None = None
    valid_tokens: int | None = None
    model_image_size = model.config.image_size
    if isinstance(model_image_size, (tuple, list)):
        model_image_size = int(model_image_size[0])
    else:
        model_image_size = int(model_image_size)
    patch_size = model.config.patch_size
    if isinstance(patch_size, (tuple, list)):
        patch_height = int(patch_size[0])
        patch_width = int(patch_size[1])
    else:
        patch_height = patch_width = int(patch_size)
    interpolate_pos_encoding = (
        args.image_size > 0 and args.image_size != model_image_size
    )
    with torch.inference_mode():
        for index in range(args.samples):
            item = dataset[index]
            processor_args: dict[str, Any] = {
                "images": item["img"],
                "return_tensors": "pt",
            }
            if args.image_size:
                processor_args["size"] = {
                    "height": args.image_size,
                    "width": args.image_size,
                }
            pixel_values = processor(**processor_args).pixel_values.to(
                device="cuda",
                dtype=torch.bfloat16,
            )
            current_image_size = (
                int(pixel_values.shape[-2]),
                int(pixel_values.shape[-1]),
            )
            current_valid_tokens = (
                (current_image_size[0] // patch_height) *
                (current_image_size[1] // patch_width) +
                1
            )
            if processed_image_size is None:
                processed_image_size = current_image_size
                valid_tokens = current_valid_tokens
            elif (
                processed_image_size != current_image_size
                or valid_tokens != current_valid_tokens
            ):
                raise ValueError("processor emitted inconsistent image shapes")
            model_args = {
                "pixel_values": pixel_values,
                "interpolate_pos_encoding": interpolate_pos_encoding,
            }
            runner.enabled = False
            baseline_logits = model(**model_args).logits.float()
            runner.begin_sample(index)
            runner.enabled = True
            fp4_logits = model(**model_args).logits.float()
            record = {
                "index": index,
                "label": int(item["label"]),
                "baseline_prediction": int(
                    baseline_logits.argmax(dim=-1).item()
                ),
                "fp4_prediction": int(fp4_logits.argmax(dim=-1).item()),
                "baseline_logits": baseline_logits[0].cpu().tolist(),
                "fp4_logits": fp4_logits[0].cpu().tolist(),
            }
            records.append(record)
            if args.stop_on_nonfinite and runner.nonfinite_output_count:
                print(
                    f"stopping after sample {index}: "
                    f"{runner.nonfinite_output_count} non-finite rows",
                    flush=True,
                )
                break
            if (
                (index + 1) % args.progress_every == 0
                or index + 1 == args.samples
            ):
                print(
                    f"[{index + 1}/{args.samples}] "
                    f"label={record['label']} "
                    f"bf16={record['baseline_prediction']} "
                    f"fp4={record['fp4_prediction']}",
                    flush=True,
                )

    result = {
        "schema": "tk_fp4_regular_attention_eval_v1",
        "model": model_identity,
        "dataset": dataset_identity,
        "split": args.split,
        "seed": args.seed,
        "extension": extension_identity,
        "adapter": {
            "model_shape": {
                "image_height": processed_image_size[0],
                "image_width": processed_image_size[1],
                "seqlen": valid_tokens,
                "heads": model.config.num_attention_heads,
                "dim": (
                    model.config.hidden_size //
                    model.config.num_attention_heads
                ),
            },
            "kernel_shape": {
                "seqlen": runner.target_seqlen,
                "heads": runner.target_heads,
                "dim": runner.target_dim,
            },
            "q_scale_for_head_padding": math.sqrt(
                runner.target_dim /
                (
                    model.config.hidden_size /
                    model.config.num_attention_heads
                )
            ),
            "interpolate_pos_encoding": interpolate_pos_encoding,
            "padding_mask_value": args.mask_value,
            "padding_score": -(
                args.mask_value * args.mask_value / math.sqrt(128)
            ),
            "interleave_kv_quarters": args.interleave_kv_quarters,
            "global_anchor_kv": args.global_anchor_kv,
            "global_anchor_samples": args.global_anchor_samples,
            "nv_qk_fold_k64_scales": runner.nv_qk_fold_k64_scales,
            "nv_qk_fold_scale_select": runner.nv_qk_fold_scale_select,
            "nv_qk_fold_scale_multiplier": (
                runner.nv_qk_fold_scale_multiplier
            ),
            "key_centering": args.key_centering,
            "score_shift": args.score_shift,
            "score_shift_predictor": args.score_shift_predictor,
            "score_shift_bias": args.score_shift_bias,
            "timing_scope": (
                "accuracy only; dynamic Q/K/V quantization is not timed"
            ),
            "requested_samples": args.samples,
            "stop_on_nonfinite": args.stop_on_nonfinite,
        },
        "classification": classification_metrics(records),
        "attention": runner.summary(),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result["classification"], indent=2), flush=True)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
