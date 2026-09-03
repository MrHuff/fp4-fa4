#!/usr/bin/env python3
"""Authenticate one rendered FA4 training config before launching TorchTitan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tomllib
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.render_fa4_training_config import (
    REQUIRED_ENVIRONMENT,
    _HISTORICAL_TOKENIZER,
)
from tools.fa4_dataset_manifest import verify_dataset_manifest
from torchtitan.experiments.fa4.artifacts import load_artifact_manifest


SCHEMA = "fa4_training_config_receipt_v2"
TRAINER_MODULE = "torchtitan.experiments.fa4.train"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_HISTORICAL_TOKENIZER_TREE_SHA256 = (
    "ba9162eb542cf6445c6a1c9cf997dc176458b1dcfa127aad434b563ec5d94718"
)
_RECEIPT_KEYS = {
    "schema",
    "config",
    "config_relative_to_receipt",
    "config_bytes",
    "config_sha256",
    "artifact_manifest",
    "artifact_manifest_sha256",
    "route",
    "shape",
    "world_size",
    "global_batch_size",
    "gradient_accumulation_steps",
    "trainer_module",
    "training_integration",
    "dataset",
    "tokenizer",
    "required_environment",
}


class VerificationError(RuntimeError):
    """A rendered-config launch invariant was not satisfied."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VerificationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise VerificationError(f"{label} must contain a JSON object")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise VerificationError(f"{label} must be a non-empty string")
    return value


def _positive_integer(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise VerificationError(f"{label} must be a positive integer")
    return value


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise VerificationError(f"{label} must be a JSON object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise VerificationError(
            f"{label} keys differ: missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _sha256(value: object, label: str) -> str:
    digest = _string(value, label)
    if _SHA256.fullmatch(digest) is None:
        raise VerificationError(f"{label} must be a lowercase SHA256 digest")
    return digest


def _absolute_path(value: object, label: str) -> Path:
    path = Path(_string(value, label)).expanduser()
    if not path.is_absolute():
        raise VerificationError(f"{label} must be absolute: {path}")
    return path.resolve()


def _authenticate_file(
    path: Path,
    *,
    expected_bytes: int | None,
    expected_sha256: str,
    label: str,
) -> bytes:
    if not path.is_file():
        raise VerificationError(f"{label} does not exist: {path}")
    contents = path.read_bytes()
    if expected_bytes is not None and len(contents) != expected_bytes:
        raise VerificationError(
            f"{label} byte identity mismatch: {len(contents)} != {expected_bytes}"
        )
    observed_sha256 = _sha256_bytes(contents)
    if observed_sha256 != expected_sha256:
        raise VerificationError(
            f"{label} SHA256 mismatch: {observed_sha256} != {expected_sha256}"
        )
    return contents


def _parse_config(config_contents: bytes) -> Mapping[str, Any]:
    try:
        config = tomllib.loads(config_contents.decode("utf-8"))
    except (UnicodeError, tomllib.TOMLDecodeError) as error:
        raise VerificationError(
            f"rendered training config is invalid TOML: {error}"
        ) from error
    if not isinstance(config, Mapping):
        raise VerificationError("rendered training config must contain a TOML table")
    return config


def _config_dataset_path(config: Mapping[str, Any]) -> str:
    paths: list[str] = []
    for section_name in ("training", "validation"):
        section = _mapping(config.get(section_name), f"config.{section_name}")
        value = _string(
            section.get("dataset_path"),
            f"config.{section_name}.dataset_path",
        )
        paths.append(value)
    if paths[0] != paths[1]:
        raise VerificationError("training and validation dataset paths differ")
    return paths[0]


def _verify_dataset(value: object, config: Mapping[str, Any]) -> dict[str, object]:
    raw = _mapping(value, "dataset")
    kind = _string(raw.get("kind"), "dataset.kind")
    configured_path = _config_dataset_path(config)
    if kind == "local_snapshot":
        _exact_keys(
            raw,
            {
                "kind",
                "path",
                "manifest",
                "manifest_sha256",
                "tree_sha256",
                "file_count",
            },
            "dataset",
        )
        root = _absolute_path(raw["path"], "dataset.path")
        if configured_path != str(root):
            raise VerificationError(
                f"config dataset path {configured_path!r} != receipt path {str(root)!r}"
            )
        manifest_path = _absolute_path(raw["manifest"], "dataset.manifest")
        manifest_sha256 = _sha256(raw["manifest_sha256"], "dataset.manifest_sha256")
        _authenticate_file(
            manifest_path,
            expected_bytes=None,
            expected_sha256=manifest_sha256,
            label="dataset manifest",
        )
        identity = verify_dataset_manifest(manifest_path, expected_root=root)
        tree_sha256 = _sha256(raw["tree_sha256"], "dataset.tree_sha256")
        file_count = _positive_integer(raw["file_count"], "dataset.file_count")
        if identity.tree_sha256 != tree_sha256 or len(identity.files) != file_count:
            raise VerificationError(
                "dataset receipt disagrees with the authenticated dataset manifest"
            )
        return {
            "kind": kind,
            "path": str(root),
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "tree_sha256": tree_sha256,
            "file_count": file_count,
        }
    if kind == "huggingface_revision":
        _exact_keys(
            raw,
            {"kind", "identifier", "revision", "resolved"},
            "dataset",
        )
        identifier = _string(raw["identifier"], "dataset.identifier")
        revision = _string(raw["revision"], "dataset.revision")
        if _COMMIT.fullmatch(revision) is None:
            raise VerificationError(
                "dataset.revision must be a lowercase 40-hex commit"
            )
        resolved = _string(raw["resolved"], "dataset.resolved")
        if resolved != f"{identifier}@{revision}" or configured_path != resolved:
            raise VerificationError(
                "remote dataset receipt does not match the rendered config"
            )
        return {
            "kind": kind,
            "identifier": identifier,
            "revision": revision,
        }
    raise VerificationError(f"unsupported dataset kind {kind!r}")


def _tokenizer_file_records(
    value: object,
    *,
    root: Path,
) -> list[dict[str, object]]:
    if not isinstance(value, list) or not value:
        raise VerificationError("tokenizer.files must be a non-empty JSON array")

    records: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        label = f"tokenizer.files[{index}]"
        raw = _mapping(item, label)
        _exact_keys(raw, {"path", "bytes", "sha256"}, label)
        name = _string(raw["path"], f"{label}.path")
        relative = PurePosixPath(name)
        if (
            name == "."
            or relative.is_absolute()
            or relative.as_posix() != name
            or any(part in {".", ".."} for part in relative.parts)
            or "\\" in name
        ):
            raise VerificationError(
                f"{label}.path must be a normalized relative POSIX path: {name!r}"
            )
        if name in seen:
            raise VerificationError(f"duplicate tokenizer file path: {name}")
        seen.add(name)

        expected_bytes = _positive_integer(raw["bytes"], f"{label}.bytes")
        expected_sha256 = _sha256(raw["sha256"], f"{label}.sha256")
        path = (root / Path(*relative.parts)).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise VerificationError(
                f"{label}.path escapes tokenizer root: {name!r}"
            ) from error
        _authenticate_file(
            path,
            expected_bytes=expected_bytes,
            expected_sha256=expected_sha256,
            label=f"tokenizer file {name}",
        )
        records.append(
            {
                "path": name,
                "bytes": expected_bytes,
                "sha256": expected_sha256,
            }
        )
    return records


def _verify_tokenizer(value: object, config: Mapping[str, Any]) -> dict[str, object]:
    raw = _mapping(value, "tokenizer")
    historical = raw.get("historical_four_file_identity")
    if type(historical) is not bool:
        raise VerificationError(
            "tokenizer.historical_four_file_identity must be a boolean"
        )
    expected_keys = {
        "root",
        "files",
        "historical_four_file_identity",
        ("historical_tree_sha256" if historical else "tree_sha256"),
    }
    _exact_keys(raw, expected_keys, "tokenizer")

    root = _absolute_path(raw["root"], "tokenizer.root")
    if not root.is_dir():
        raise VerificationError(f"tokenizer root is not a directory: {root}")
    model = _mapping(config.get("model"), "config.model")
    configured_root = _absolute_path(
        model.get("hf_assets_path"), "config.model.hf_assets_path"
    )
    if configured_root != root:
        raise VerificationError(
            f"config tokenizer root {configured_root} != receipt root {root}"
        )

    records = _tokenizer_file_records(raw["files"], root=root)
    names = [str(record["path"]) for record in records]
    if historical:
        expected_names = list(_HISTORICAL_TOKENIZER)
        if names != expected_names:
            raise VerificationError(
                "historical tokenizer file inventory differs: "
                f"{names!r} != {expected_names!r}"
            )
        for record in records:
            expected_bytes, expected_sha256 = _HISTORICAL_TOKENIZER[str(record["path"])]
            if record["bytes"] != expected_bytes or record["sha256"] != expected_sha256:
                raise VerificationError(
                    f"historical tokenizer identity mismatch for {record['path']}"
                )
        tree_sha256 = _sha256(
            raw["historical_tree_sha256"],
            "tokenizer.historical_tree_sha256",
        )
        if tree_sha256 != _HISTORICAL_TOKENIZER_TREE_SHA256:
            raise VerificationError("historical tokenizer tree identity mismatch")
        return {
            "root": str(root),
            "files": records,
            "historical_four_file_identity": True,
            "historical_tree_sha256": tree_sha256,
        }

    allowed_inventories = {
        ("tokenizer.json", "tokenizer_config.json"),
        ("vocab.json", "tokenizer_config.json"),
        ("vocab.json", "merges.txt", "tokenizer_config.json"),
        ("vocab.txt", "tokenizer_config.json"),
        ("vocab.txt", "merges.txt", "tokenizer_config.json"),
    }
    if tuple(names) not in allowed_inventories:
        raise VerificationError(
            f"unsupported nonhistorical tokenizer file inventory: {names!r}"
        )
    digest = hashlib.sha256()
    for record in records:
        digest.update(str(record["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(record["sha256"]).encode("ascii"))
        digest.update(b"\0")
    tree_sha256 = _sha256(raw["tree_sha256"], "tokenizer.tree_sha256")
    if digest.hexdigest() != tree_sha256:
        raise VerificationError("tokenizer tree identity mismatch")
    return {
        "root": str(root),
        "files": records,
        "historical_four_file_identity": False,
        "tree_sha256": tree_sha256,
    }


def verify_training_config(
    config_path: str | Path,
    receipt_path: str | Path | None = None,
    *,
    environment: Mapping[str, str] | None = None,
    expected_world_size: int | None = None,
) -> dict[str, object]:
    """Verify the receipt, config, artifacts, and launch environment."""

    config = Path(config_path).expanduser().resolve()
    receipt = (
        Path(receipt_path).expanduser().resolve()
        if receipt_path is not None
        else Path(str(config) + ".receipt.json")
    )
    if not receipt.is_file():
        raise VerificationError(f"training config receipt does not exist: {receipt}")
    raw = _strict_json(receipt, "training config receipt")
    if set(raw) != _RECEIPT_KEYS:
        raise VerificationError(
            "training config receipt keys differ: "
            f"missing={sorted(_RECEIPT_KEYS - set(raw))}, "
            f"extra={sorted(set(raw) - _RECEIPT_KEYS)}"
        )
    if raw["schema"] != SCHEMA:
        raise VerificationError(f"unsupported training config schema {raw['schema']!r}")

    recorded_absolute = _absolute_path(raw["config"], "config")
    relative = Path(
        _string(raw["config_relative_to_receipt"], "config_relative_to_receipt")
    )
    if relative.is_absolute():
        raise VerificationError("config_relative_to_receipt must be relative")
    recorded_relative = (receipt.parent / relative).resolve()
    if config not in {recorded_absolute, recorded_relative}:
        raise VerificationError(
            "supplied config path matches neither receipt identity: "
            f"{config} not in ({recorded_absolute}, {recorded_relative})"
        )
    config_bytes = _positive_integer(raw["config_bytes"], "config_bytes")
    config_sha256 = _sha256(raw["config_sha256"], "config_sha256")
    config_contents = _authenticate_file(
        config,
        expected_bytes=config_bytes,
        expected_sha256=config_sha256,
        label="rendered training config",
    )
    parsed_config = _parse_config(config_contents)

    required = raw["required_environment"]
    if not isinstance(required, Mapping):
        raise VerificationError("required_environment must be a JSON object")
    if dict(required) != REQUIRED_ENVIRONMENT:
        raise VerificationError(
            "required_environment does not match the supported FA4 launch contract"
        )
    observed_environment = os.environ if environment is None else environment
    for name, expected in REQUIRED_ENVIRONMENT.items():
        observed = observed_environment.get(name)
        if observed != expected:
            raise VerificationError(
                f"required environment mismatch for {name}: "
                f"expected {expected!r}, got {observed!r}"
            )

    manifest_path = _absolute_path(raw["artifact_manifest"], "artifact_manifest")
    manifest_sha256 = _sha256(
        raw["artifact_manifest_sha256"], "artifact_manifest_sha256"
    )
    _authenticate_file(
        manifest_path,
        expected_bytes=None,
        expected_sha256=manifest_sha256,
        label="artifact manifest",
    )
    # Parse strictly before the normal artifact loader performs its complete
    # schema and binary/source authentication.
    _strict_json(manifest_path, "artifact manifest")
    manifest = load_artifact_manifest(manifest_path, require_training=True)

    if raw["trainer_module"] != TRAINER_MODULE:
        raise VerificationError(
            f"trainer_module must be {TRAINER_MODULE!r}, got {raw['trainer_module']!r}"
        )
    if raw["route"] != manifest.route:
        raise VerificationError(
            f"receipt route {raw['route']!r} != manifest route {manifest.route!r}"
        )
    expected_shape = {
        "batch": manifest.batch,
        "sequence": manifest.sequence,
        "q_heads": manifest.q_heads,
        "kv_heads": manifest.kv_heads,
        "head_dim": manifest.head_dim,
    }
    if raw["shape"] != expected_shape:
        raise VerificationError(
            f"receipt shape {raw['shape']!r} != manifest shape {expected_shape!r}"
        )
    world_size = _positive_integer(raw["world_size"], "world_size")
    if expected_world_size is not None:
        expected_world_size = _positive_integer(
            expected_world_size, "expected_world_size"
        )
        if world_size != expected_world_size:
            raise VerificationError(
                f"launcher world size {expected_world_size} != receipt world size "
                f"{world_size}"
            )
    global_batch = _positive_integer(raw["global_batch_size"], "global_batch_size")
    accumulation = _positive_integer(
        raw["gradient_accumulation_steps"], "gradient_accumulation_steps"
    )
    denominator = manifest.batch * world_size
    if global_batch % denominator or global_batch // denominator != accumulation:
        raise VerificationError(
            "gradient accumulation does not match local batch, world size, and "
            "global batch"
        )

    # Rehashing a local dataset can be the most expensive CPU-only check, so
    # perform it only after every cheap launch invariant has passed.
    tokenizer = _verify_tokenizer(raw["tokenizer"], parsed_config)
    dataset = _verify_dataset(raw["dataset"], parsed_config)

    return {
        "config": str(config),
        "config_sha256": config_sha256,
        "artifact_manifest": str(manifest_path),
        "artifact_manifest_sha256": manifest_sha256,
        "route": manifest.route,
        "world_size": world_size,
        "global_batch_size": global_batch,
        "gradient_accumulation_steps": accumulation,
        "trainer_module": TRAINER_MODULE,
        "tokenizer": tokenizer,
        "dataset": dataset,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--receipt",
        type=Path,
        help="defaults to <config>.receipt.json",
    )
    parser.add_argument("--world-size", required=True, type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        result = verify_training_config(
            args.config,
            args.receipt,
            expected_world_size=args.world_size,
        )
    except (
        VerificationError,
        FileNotFoundError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        raise SystemExit(f"FA4 training config verification failed: {error}") from error
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
