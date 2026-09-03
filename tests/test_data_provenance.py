from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "reproduction" / "snapshots" / "training_e7db209b"
DATA_MANIFEST = ROOT / "release" / "data_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_git_mode(path: Path) -> str:
    """Return the portable regular-file mode represented by Git."""

    executable = path.stat().st_mode & stat.S_IXUSR
    return "0755" if executable else "0644"


def test_historical_training_snapshot_checksums() -> None:
    checksum_lines = (SNAPSHOT / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    tree_digest = hashlib.sha256()
    tree_bytes = 0
    for line in checksum_lines:
        expected, relative = line.split("  ", 1)
        path = SNAPSHOT / relative
        assert path.is_file(), relative
        assert _sha256(path) == expected, relative
        size = path.stat().st_size
        # Git stores only the executable bit for regular files.  Ignore ambient
        # umask/group-write differences so this identity is stable in a fresh
        # clone while still detecting executable-bit changes.
        mode = _canonical_git_mode(path)
        tree_digest.update(f"{expected} {size} {mode} {relative}\n".encode())
        tree_bytes += size

    manifest = json.loads(DATA_MANIFEST.read_text(encoding="utf-8"))
    snapshot = manifest["experiments"]["llama8b_slimpajama_matched_b4"][
        "source_identity"
    ]["materialized_data_code_snapshot"]
    assert snapshot["tree_hash_schema"] == "sha256_size_canonical_git_mode_path_v2"
    assert len(checksum_lines) == snapshot["tree_files"]
    assert tree_bytes == snapshot["tree_bytes"]
    assert tree_digest.hexdigest() == snapshot["tree_sha256"]
    assert _sha256(SNAPSHOT / "SHA256SUMS") == snapshot["sha256sums_sha256"]


def test_historical_training_template_is_sanitized() -> None:
    template = (
        SNAPSHOT / "train_configs" / "slimpajama_fa4_exact_50b" / "llama3_8b_40b.toml"
    ).read_text(encoding="utf-8")
    assert template.count("__HISTORICAL_SLIMPAJAMA_MDS_ROOT__") == 2
    assert "s3://" not in template
    assert "aws_secret_access_key" not in template.lower()
    assert "aws_session_token" not in template.lower()


def test_data_manifest_binds_materialized_training_sources() -> None:
    manifest = json.loads(DATA_MANIFEST.read_text(encoding="utf-8"))
    experiment = manifest["experiments"]["llama8b_slimpajama_matched_b4"]
    records = list(experiment["historical_data"]["loader_source"])
    records.extend(
        record
        for record in experiment["source_identity"]["files"]
        if record["path"].startswith("reproduction/snapshots/training_e7db209b/")
    )
    assert len(records) == 8
    for record in records:
        path = ROOT / record["path"]
        assert path.is_file(), record["path"]
        assert _sha256(path) == record["sha256"], record["path"]


def test_excluded_scheduler_evidence_is_not_presented_as_release_files() -> None:
    manifest = json.loads(DATA_MANIFEST.read_text(encoding="utf-8"))
    experiment = manifest["experiments"]["llama_d64_dolma3_prefix256"]
    records = [
        *experiment["evidence"],
        *experiment["additional_8b_gate"]["evidence"],
    ]

    excluded = [record for record in records if "historical_path" in record]
    assert len(excluded) == 3
    for record in excluded:
        assert "path" not in record
        assert record["redistribution"]["included"] is False
        assert not (ROOT / record["historical_path"]).exists()
