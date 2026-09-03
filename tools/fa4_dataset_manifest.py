#!/usr/bin/env python3
"""Create or verify an exhaustive manifest for one local FA4 dataset snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "fa4_local_dataset_manifest_v1"
ALGORITHM = "sha256-path-size-content-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")


class DatasetManifestError(RuntimeError):
    """A local dataset manifest or snapshot invariant was not satisfied."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DatasetManifestError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DatasetManifestError(
            f"cannot read dataset manifest {path}: {error}"
        ) from error
    if not isinstance(value, Mapping):
        raise DatasetManifestError("dataset manifest must contain a JSON object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise DatasetManifestError(
            f"{label} keys differ: missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise DatasetManifestError(f"{label} must be a non-empty string")
    return value


def _nonnegative_integer(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise DatasetManifestError(f"{label} must be a non-negative integer")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(files: Sequence["DatasetFileIdentity"]) -> str:
    digest = hashlib.sha256()
    for record in files:
        digest.update(record.path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record.bytes).encode("ascii"))
        digest.update(b"\0")
        digest.update(record.sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _paths(root: Path) -> tuple[Path, ...]:
    files: list[Path] = []
    for directory, directories, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        directories.sort()
        filenames.sort()
        for name in directories:
            path = directory_path / name
            if path.is_symlink():
                raise DatasetManifestError(
                    f"dataset snapshot contains a symlink directory: {path}"
                )
        for name in filenames:
            path = directory_path / name
            if path.is_symlink():
                raise DatasetManifestError(
                    f"dataset snapshot contains a symlink file: {path}"
                )
            if not path.is_file():
                raise DatasetManifestError(
                    f"dataset snapshot contains a non-regular file: {path}"
                )
            files.append(path)
    return tuple(sorted(files, key=lambda path: path.relative_to(root).as_posix()))


@dataclass(frozen=True)
class DatasetFileIdentity:
    path: str
    bytes: int
    sha256: str

    @classmethod
    def from_path(cls, path: Path, root: Path) -> "DatasetFileIdentity":
        return cls(
            path=path.relative_to(root).as_posix(),
            bytes=path.stat().st_size,
            sha256=_sha256_file(path),
        )

    @classmethod
    def parse(cls, value: object, label: str) -> "DatasetFileIdentity":
        if not isinstance(value, Mapping):
            raise DatasetManifestError(f"{label} must be a JSON object")
        _exact_keys(value, {"path", "bytes", "sha256"}, label)
        relative = Path(_string(value["path"], f"{label}.path"))
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise DatasetManifestError(f"{label}.path must be a safe relative path")
        size = _nonnegative_integer(value["bytes"], f"{label}.bytes")
        sha256 = _string(value["sha256"], f"{label}.sha256")
        if _SHA256.fullmatch(sha256) is None:
            raise DatasetManifestError(f"{label}.sha256 must be lowercase SHA256")
        return cls(path=relative.as_posix(), bytes=size, sha256=sha256)


@dataclass(frozen=True)
class DatasetManifest:
    path: Path
    root: Path
    tree_sha256: str
    files: tuple[DatasetFileIdentity, ...]


def _root_from_manifest(
    raw: Mapping[str, Any],
    manifest_path: Path,
    expected_root: str | Path | None,
) -> Path:
    absolute = Path(_string(raw["root"], "root")).expanduser()
    if not absolute.is_absolute():
        raise DatasetManifestError("root must be absolute")
    absolute = absolute.resolve()
    relative = Path(
        _string(raw["root_relative_to_manifest"], "root_relative_to_manifest")
    )
    if relative.is_absolute():
        raise DatasetManifestError("root_relative_to_manifest must be relative")
    relative_root = (manifest_path.parent / relative).resolve()
    if expected_root is not None:
        root = Path(expected_root).expanduser().resolve()
        if root not in {absolute, relative_root}:
            raise DatasetManifestError(
                "dataset root matches neither manifest identity: "
                f"{root} not in ({absolute}, {relative_root})"
            )
        return root
    candidates = {path for path in (absolute, relative_root) if path.is_dir()}
    if len(candidates) != 1:
        raise DatasetManifestError(
            "dataset root identity is missing or ambiguous; pass --root explicitly"
        )
    return candidates.pop()


def verify_dataset_manifest(
    manifest_path: str | Path,
    *,
    expected_root: str | Path | None = None,
) -> DatasetManifest:
    """Strictly verify manifest structure, exhaustiveness, and every file hash."""

    manifest = Path(manifest_path).expanduser().resolve()
    if not manifest.is_file():
        raise DatasetManifestError(f"dataset manifest does not exist: {manifest}")
    raw = _strict_json(manifest)
    _exact_keys(
        raw,
        {
            "schema",
            "algorithm",
            "root",
            "root_relative_to_manifest",
            "file_count",
            "tree_sha256",
            "files",
        },
        "dataset manifest",
    )
    if raw["schema"] != SCHEMA:
        raise DatasetManifestError(f"unsupported dataset schema {raw['schema']!r}")
    if raw["algorithm"] != ALGORITHM:
        raise DatasetManifestError(
            f"unsupported dataset digest algorithm {raw['algorithm']!r}"
        )
    root = _root_from_manifest(raw, manifest, expected_root)
    if not root.is_dir():
        raise DatasetManifestError(f"dataset root does not exist: {root}")
    try:
        manifest.relative_to(root)
    except ValueError:
        pass
    else:
        raise DatasetManifestError(
            "dataset manifest must live outside the dataset root"
        )

    raw_files = raw["files"]
    if not isinstance(raw_files, list):
        raise DatasetManifestError("files must be a JSON array")
    files = tuple(
        DatasetFileIdentity.parse(value, f"files[{index}]")
        for index, value in enumerate(raw_files)
    )
    file_count = _nonnegative_integer(raw["file_count"], "file_count")
    if not files or len(files) != file_count:
        raise DatasetManifestError(
            "dataset manifest must contain a non-empty file list matching file_count"
        )
    relative_paths = tuple(record.path for record in files)
    if relative_paths != tuple(sorted(relative_paths)) or len(
        set(relative_paths)
    ) != len(relative_paths):
        raise DatasetManifestError("dataset file paths must be unique and sorted")
    expected_tree = _string(raw["tree_sha256"], "tree_sha256")
    if _SHA256.fullmatch(expected_tree) is None:
        raise DatasetManifestError("tree_sha256 must be lowercase SHA256")
    recorded_tree = _tree_sha256(files)
    if recorded_tree != expected_tree:
        raise DatasetManifestError(
            f"dataset manifest tree digest mismatch: {recorded_tree} != {expected_tree}"
        )

    observed_paths = tuple(path.relative_to(root).as_posix() for path in _paths(root))
    if observed_paths != relative_paths:
        raise DatasetManifestError(
            "dataset manifest is not exhaustive: "
            f"added={sorted(set(observed_paths) - set(relative_paths))}, "
            f"missing={sorted(set(relative_paths) - set(observed_paths))}"
        )
    observed_files: list[DatasetFileIdentity] = []
    for record in files:
        path = root / record.path
        observed_size = path.stat().st_size
        if observed_size != record.bytes:
            raise DatasetManifestError(
                f"dataset file byte identity mismatch for {record.path}: "
                f"{observed_size} != {record.bytes}"
            )
        observed_sha256 = _sha256_file(path)
        if observed_sha256 != record.sha256:
            raise DatasetManifestError(
                f"dataset file SHA256 mismatch for {record.path}: "
                f"{observed_sha256} != {record.sha256}"
            )
        observed_files.append(
            DatasetFileIdentity(record.path, observed_size, observed_sha256)
        )
    observed_tree = _tree_sha256(observed_files)
    if observed_tree != expected_tree:
        raise DatasetManifestError(
            f"dataset tree SHA256 mismatch: {observed_tree} != {expected_tree}"
        )
    return DatasetManifest(
        path=manifest,
        root=root,
        tree_sha256=expected_tree,
        files=files,
    )


def create_dataset_manifest(root: str | Path, output: str | Path) -> DatasetManifest:
    """Create a new manifest, refusing ambiguous placement or overwrite."""

    dataset_root = Path(root).expanduser().resolve()
    manifest = Path(output).expanduser().resolve()
    if not dataset_root.is_dir():
        raise DatasetManifestError(f"dataset root does not exist: {dataset_root}")
    if manifest.exists():
        raise DatasetManifestError(f"refusing to overwrite {manifest}")
    try:
        manifest.relative_to(dataset_root)
    except ValueError:
        pass
    else:
        raise DatasetManifestError(
            "dataset manifest must live outside the dataset root"
        )
    files = tuple(
        DatasetFileIdentity.from_path(path, dataset_root)
        for path in _paths(dataset_root)
    )
    if not files:
        raise DatasetManifestError("dataset snapshot must contain at least one file")
    value = {
        "schema": SCHEMA,
        "algorithm": ALGORITHM,
        "root": str(dataset_root),
        "root_relative_to_manifest": os.path.relpath(dataset_root, manifest.parent),
        "file_count": len(files),
        "tree_sha256": _tree_sha256(files),
        "files": [
            {"path": item.path, "bytes": item.bytes, "sha256": item.sha256}
            for item in files
        ],
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return verify_dataset_manifest(manifest, expected_root=dataset_root)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create", help="create a new exhaustive manifest")
    create.add_argument("--root", required=True, type=Path)
    create.add_argument("--output", required=True, type=Path)
    check = subparsers.add_parser(
        "check", help="rehash and verify an existing manifest"
    )
    check.add_argument("--manifest", required=True, type=Path)
    check.add_argument("--root", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        if args.command == "create":
            identity = create_dataset_manifest(args.root, args.output)
        else:
            identity = verify_dataset_manifest(args.manifest, expected_root=args.root)
    except DatasetManifestError as error:
        raise SystemExit(f"FA4 dataset manifest failed: {error}") from error
    print(
        json.dumps(
            {
                "manifest": str(identity.path),
                "root": str(identity.root),
                "file_count": len(identity.files),
                "tree_sha256": identity.tree_sha256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
