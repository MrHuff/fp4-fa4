from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.fa4_dataset_manifest import (
    DatasetManifestError,
    create_dataset_manifest,
    verify_dataset_manifest,
)


def _dataset(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    (root / "nested").mkdir(parents=True)
    (root / "part-00000.jsonl").write_text('{"text":"first"}\n')
    (root / "nested/part-00001.jsonl").write_text('{"text":"second"}\n')
    return root


def test_dataset_manifest_is_exhaustive_and_detects_one_byte_mutation(
    tmp_path: Path,
) -> None:
    root = _dataset(tmp_path)
    manifest = tmp_path / "dataset.manifest.json"

    created = create_dataset_manifest(root, manifest)
    checked = verify_dataset_manifest(manifest, expected_root=root)

    assert created.tree_sha256 == checked.tree_sha256
    assert [record.path for record in checked.files] == [
        "nested/part-00001.jsonl",
        "part-00000.jsonl",
    ]
    target = root / "part-00000.jsonl"
    original = target.read_bytes()
    target.write_bytes(original[:-2] + b"!\n")
    assert target.stat().st_size == len(original)
    with pytest.raises(DatasetManifestError, match="SHA256 mismatch"):
        verify_dataset_manifest(manifest, expected_root=root)


def test_dataset_manifest_detects_added_file_and_duplicate_json_key(
    tmp_path: Path,
) -> None:
    root = _dataset(tmp_path)
    manifest = tmp_path / "dataset.manifest.json"
    create_dataset_manifest(root, manifest)
    (root / "unrecorded.jsonl").write_text("{}\n")

    with pytest.raises(DatasetManifestError, match="not exhaustive"):
        verify_dataset_manifest(manifest, expected_root=root)

    (root / "unrecorded.jsonl").unlink()
    contents = manifest.read_text()
    manifest.write_text(
        contents.replace(
            '"schema": "fa4_local_dataset_manifest_v1",',
            '"schema": "fa4_local_dataset_manifest_v1",\n'
            '  "schema": "fa4_local_dataset_manifest_v1",',
            1,
        )
    )
    with pytest.raises(DatasetManifestError, match="duplicate JSON key"):
        verify_dataset_manifest(manifest, expected_root=root)


def test_dataset_manifest_cli_create_and_check(tmp_path: Path) -> None:
    root = _dataset(tmp_path)
    manifest = tmp_path / "dataset.manifest.json"
    tool = Path(__file__).parents[1] / "tools/fa4_dataset_manifest.py"

    created = subprocess.run(
        [
            sys.executable,
            str(tool),
            "create",
            "--root",
            str(root),
            "--output",
            str(manifest),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    checked = subprocess.run(
        [
            sys.executable,
            str(tool),
            "check",
            "--manifest",
            str(manifest),
            "--root",
            str(root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert (
        json.loads(created.stdout)["tree_sha256"]
        == json.loads(checked.stdout)["tree_sha256"]
    )
