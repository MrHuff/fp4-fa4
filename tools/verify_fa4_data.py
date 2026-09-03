#!/usr/bin/env python3
"""Verify external data and tokenizer bytes used by the FA4 experiments.

This tool is CPU-only, uses the Python standard library, does not download
anything, and never needs credentials.  It authenticates user-supplied assets
against release/data_manifest.json.  A successful tokenizer check also
reconstructs the historical deterministic tar digest without writing a tar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "release" / "data_manifest.json"


class VerificationError(RuntimeError):
    """A supplied asset does not satisfy its recorded identity."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VerificationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot read data manifest {path}: {exc}") from exc
    _require(isinstance(manifest, dict), "data manifest root must be an object")
    _require(
        manifest.get("schema") == "fa4_data_provenance_v1",
        "unsupported data-manifest schema",
    )
    return manifest


def _stream_identity(path: Path, *, count_lines: bool) -> dict[str, Any]:
    _require(path.is_file(), f"input is not a file: {path}")
    digest = hashlib.sha256()
    size = 0
    lines = 0
    final_byte = b""
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
            if count_lines:
                lines += chunk.count(b"\n")
                final_byte = chunk[-1:]
    result: dict[str, Any] = {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "bytes": size,
    }
    if count_lines:
        result["rows"] = lines
        result["terminal_newline"] = final_byte == b"\n"
    return result


def _verify_file_target(
    manifest: dict[str, Any], target_name: str, path: Path
) -> dict[str, Any]:
    targets = manifest.get("verifier_targets")
    _require(isinstance(targets, dict), "verifier_targets must be an object")
    expected = targets.get(target_name)
    _require(isinstance(expected, dict), f"missing verifier target: {target_name}")
    actual = _stream_identity(path, count_lines=expected.get("kind") == "jsonl")
    for field in ("sha256", "bytes"):
        _require(
            actual[field] == expected.get(field),
            f"{target_name} {field} mismatch: expected {expected.get(field)!r}, "
            f"found {actual[field]!r}",
        )
    if expected.get("kind") == "jsonl":
        for field in ("rows", "terminal_newline"):
            _require(
                actual[field] == expected.get(field),
                f"{target_name} {field} mismatch: expected {expected.get(field)!r}, "
                f"found {actual[field]!r}",
            )
    return {"target": target_name, "ok": True, **actual}


def _tokenizer_payload_root(supplied: Path, archive_prefix: str) -> Path:
    supplied = supplied.expanduser().resolve()
    if supplied.name == archive_prefix:
        return supplied
    nested = supplied / archive_prefix
    if nested.is_dir():
        return nested
    return supplied


def _safe_tokenizer_relative(full_path: str, archive_prefix: str) -> Path:
    pure = PurePosixPath(full_path)
    _require(not pure.is_absolute(), f"absolute tokenizer manifest path: {full_path}")
    _require(".." not in pure.parts, f"unsafe tokenizer manifest path: {full_path}")
    _require(
        pure.parts and pure.parts[0] == archive_prefix,
        f"tokenizer path is outside archive prefix: {full_path}",
    )
    relative = PurePosixPath(*pure.parts[1:])
    _require(relative.parts, f"tokenizer manifest path names no file: {full_path}")
    return Path(*relative.parts)


class _DigestWriter:
    """Minimal write-only file object that hashes a tar stream."""

    def __init__(self) -> None:
        self.digest = hashlib.sha256()
        self.position = 0

    def write(self, data: bytes) -> int:
        self.digest.update(data)
        self.position += len(data)
        return len(data)

    def tell(self) -> int:
        return self.position

    def flush(self) -> None:
        return None


def _canonical_tar_identity(
    payload_root: Path, entries: list[dict[str, Any]]
) -> dict[str, Any]:
    writer = _DigestWriter()
    with tarfile.open(
        fileobj=writer,
        mode="w",
        format=tarfile.PAX_FORMAT,
    ) as archive:
        for entry in entries:
            source = payload_root / entry["relative_path"]
            information = tarfile.TarInfo(entry["path"])
            information.type = tarfile.REGTYPE
            information.size = entry["bytes"]
            information.mode = int(entry["mode"], 8)
            information.mtime = 0
            information.uid = 0
            information.gid = 0
            information.uname = ""
            information.gname = ""
            information.pax_headers = {}
            with source.open("rb") as contents:
                archive.addfile(information, contents)
    return {"sha256": writer.digest.hexdigest(), "bytes": writer.position}


def _verify_tokenizer(
    manifest: dict[str, Any], supplied: Path, *, allow_extra_files: bool
) -> dict[str, Any]:
    tokenizer = manifest.get("tokenizer")
    _require(isinstance(tokenizer, dict), "tokenizer record must be an object")
    tree = tokenizer.get("tree")
    archive_record = tokenizer.get("canonical_archive")
    _require(isinstance(tree, dict), "tokenizer tree record must be an object")
    _require(
        isinstance(archive_record, dict),
        "tokenizer canonical_archive record must be an object",
    )
    archive_prefix = tree.get("archive_prefix")
    entries = tree.get("entries")
    _require(isinstance(archive_prefix, str), "invalid tokenizer archive prefix")
    _require(isinstance(entries, list) and entries, "tokenizer entries are missing")
    payload_root = _tokenizer_payload_root(supplied, archive_prefix)
    _require(
        payload_root.is_dir(), f"tokenizer directory does not exist: {payload_root}"
    )

    expected_relatives: set[str] = set()
    verified_entries: list[dict[str, Any]] = []
    tree_digest = hashlib.sha256()
    total_bytes = 0
    for raw_entry in sorted(entries, key=lambda value: value.get("path", "")):
        _require(isinstance(raw_entry, dict), "tokenizer entry must be an object")
        full_path = raw_entry.get("path")
        _require(isinstance(full_path, str), "tokenizer entry path must be a string")
        relative = _safe_tokenizer_relative(full_path, archive_prefix)
        relative_posix = relative.as_posix()
        _require(
            relative_posix not in expected_relatives, f"duplicate path: {full_path}"
        )
        expected_relatives.add(relative_posix)
        source = payload_root / relative
        actual = _stream_identity(source, count_lines=False)
        for field in ("sha256", "bytes"):
            _require(
                actual[field] == raw_entry.get(field),
                f"tokenizer {full_path} {field} mismatch: expected "
                f"{raw_entry.get(field)!r}, found {actual[field]!r}",
            )
        mode = raw_entry.get("mode")
        _require(
            isinstance(mode, str) and len(mode) == 4 and mode.isdigit(),
            f"invalid canonical mode for {full_path}",
        )
        tree_digest.update(
            (f"{actual['sha256']} {actual['bytes']} {mode} " f"{full_path}\n").encode(
                "utf-8"
            )
        )
        total_bytes += actual["bytes"]
        verified_entries.append(
            {
                "path": full_path,
                "relative_path": relative_posix,
                "sha256": actual["sha256"],
                "bytes": actual["bytes"],
                "mode": mode,
            }
        )

    observed_relatives = {
        path.relative_to(payload_root).as_posix()
        for path in payload_root.rglob("*")
        if path.is_file()
    }
    extras = sorted(observed_relatives - expected_relatives)
    if extras and not allow_extra_files:
        preview = ", ".join(extras[:5])
        suffix = " ..." if len(extras) > 5 else ""
        raise VerificationError(
            "tokenizer directory contains files outside the four-file production "
            f"tree: {preview}{suffix}; pass --allow-extra-tokenizer-files to "
            "authenticate only the selected payload"
        )

    actual_tree = {
        "sha256": tree_digest.hexdigest(),
        "files": len(verified_entries),
        "bytes": total_bytes,
    }
    for field in ("sha256", "files", "bytes"):
        _require(
            actual_tree[field] == tree.get(field),
            f"tokenizer tree {field} mismatch: expected {tree.get(field)!r}, "
            f"found {actual_tree[field]!r}",
        )

    actual_archive = _canonical_tar_identity(payload_root, verified_entries)
    for field in ("sha256", "bytes"):
        _require(
            actual_archive[field] == archive_record.get(field),
            f"tokenizer archive {field} mismatch: expected "
            f"{archive_record.get(field)!r}, found {actual_archive[field]!r}",
        )
    return {
        "target": tokenizer.get("id"),
        "ok": True,
        "payload_root": str(payload_root),
        "selected_files": len(verified_entries),
        "ignored_extra_files": extras if allow_extra_files else [],
        "tree": actual_tree,
        "canonical_archive": actual_archive,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--tokenizer-dir", type=Path)
    parser.add_argument(
        "--allow-extra-tokenizer-files",
        action="store_true",
        help="verify the four selected tokenizer files inside a larger snapshot",
    )
    parser.add_argument("--dolma3-prefix-jsonl", type=Path)
    parser.add_argument("--dclm-a319-prefix-jsonl", type=Path)
    parser.add_argument("--canonical-dclm-arrow", type=Path)
    args = parser.parse_args()
    if not any(
        (
            args.tokenizer_dir,
            args.dolma3_prefix_jsonl,
            args.dclm_a319_prefix_jsonl,
            args.canonical_dclm_arrow,
        )
    ):
        parser.error("provide at least one asset to verify")
    if args.allow_extra_tokenizer_files and args.tokenizer_dir is None:
        parser.error("--allow-extra-tokenizer-files requires --tokenizer-dir")
    return args


def main() -> int:
    args = _parse_args()
    try:
        manifest = _load_manifest(args.manifest)
        checks: list[dict[str, Any]] = []
        if args.tokenizer_dir is not None:
            checks.append(
                _verify_tokenizer(
                    manifest,
                    args.tokenizer_dir,
                    allow_extra_files=args.allow_extra_tokenizer_files,
                )
            )
        if args.dolma3_prefix_jsonl is not None:
            checks.append(
                _verify_file_target(
                    manifest,
                    "dolma3_longmino_len_8_16k_first512_jsonl",
                    args.dolma3_prefix_jsonl,
                )
            )
        if args.dclm_a319_prefix_jsonl is not None:
            checks.append(
                _verify_file_target(
                    manifest,
                    "dclm_a319_first20000_jsonl",
                    args.dclm_a319_prefix_jsonl,
                )
            )
        if args.canonical_dclm_arrow is not None:
            checks.append(
                _verify_file_target(
                    manifest,
                    "canonical_dclm_arrow_subshard",
                    args.canonical_dclm_arrow,
                )
            )
    except VerificationError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        return 1
    print(json.dumps({"ok": True, "checks": checks}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
