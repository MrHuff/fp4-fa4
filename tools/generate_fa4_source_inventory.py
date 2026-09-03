#!/usr/bin/env python3
"""Generate the deterministic file inventory checked by the release verifier."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "release" / "source_files.sha256"


def _payload(path: Path) -> bytes:
    if path.is_symlink():
        return path.readlink().as_posix().encode()
    return path.read_bytes()


def _run_git_bytes(*args: str) -> bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout


def _worktree_path(relative_name: str, *, tracked: bool) -> Path:
    relative = Path(relative_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"unsafe Git worktree path: {relative_name}")
    path = ROOT / relative
    if not path.exists() and not path.is_symlink():
        kind = "tracked" if tracked else "untracked"
        raise RuntimeError(
            f"{kind} path is absent from the worktree (sparse checkout or "
            f"unstaged deletion): {relative_name}"
        )
    return path


def _git_visible_files() -> list[Path]:
    """Return every index file and non-ignored new file in the worktree."""

    paths: list[Path] = []
    index = _run_git_bytes("ls-files", "--stage", "-z")
    for encoded_record in index.split(b"\0"):
        if not encoded_record:
            continue
        try:
            encoded_metadata, encoded_name = encoded_record.split(b"\t", 1)
            mode, _object_name, stage = encoded_metadata.decode("ascii").split()
        except ValueError as error:
            raise RuntimeError("malformed git ls-files --stage record") from error
        relative_name = encoded_name.decode("utf-8", errors="strict")
        if stage != "0":
            raise RuntimeError(f"unmerged index entry: {relative_name}")
        if mode == "160000":
            # Gitlinks are authenticated by commit and checked-out HEAD in the
            # release verifier; their worktree directories are not root files.
            continue
        if mode not in {"100644", "100755", "120000"}:
            raise RuntimeError(f"unsupported index mode {mode}: {relative_name}")
        paths.append(_worktree_path(relative_name, tracked=True))

    untracked = _run_git_bytes("ls-files", "--others", "--exclude-standard", "-z")
    for encoded_name in untracked.split(b"\0"):
        if not encoded_name:
            continue
        relative_name = encoded_name.decode("utf-8", errors="strict")
        path = _worktree_path(relative_name, tracked=False)
        if path.is_file() or path.is_symlink():
            paths.append(path)

    if len(paths) != len(set(paths)):
        raise RuntimeError("duplicate path in Git-visible source inventory")
    paths.sort(key=lambda item: item.relative_to(ROOT).as_posix())
    return paths


def inventory_payload() -> tuple[str, int]:
    # Cover the complete release checkout, including the inherited TorchTitan
    # base and repository metadata.  Gitlinks are directories in an initialized
    # checkout and are authenticated separately by the release verifier.
    paths = {
        path
        for path in _git_visible_files()
        if path.is_symlink() or path.is_file()
    }

    records: list[str] = []
    for path in sorted(paths, key=lambda item: item.relative_to(ROOT).as_posix()):
        if path == OUTPUT or path.name == ".git":
            continue
        relative_name = path.relative_to(ROOT).as_posix()
        digest = hashlib.sha256(_payload(path)).hexdigest()
        records.append(f"{digest}  {relative_name}\n")
    return "".join(records), len(records)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed inventory without modifying it",
    )
    args = parser.parse_args(argv)
    payload, count = inventory_payload()
    if args.check:
        if not OUTPUT.is_file():
            print(f"missing {OUTPUT.relative_to(ROOT)}")
            return 1
        if OUTPUT.read_text(encoding="utf-8") != payload:
            print(f"stale {OUTPUT.relative_to(ROOT)}")
            return 1
        print(f"verified {count} records in {OUTPUT.relative_to(ROOT)}")
        return 0
    OUTPUT.write_text(payload, encoding="utf-8")
    print(f"wrote {count} records to {OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
