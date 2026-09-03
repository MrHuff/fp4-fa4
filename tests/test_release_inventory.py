from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess

import pytest

from tools import generate_fa4_source_inventory as inventory
from tools import verify_fa4_release as verifier


def _git(tmp_path: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _commit(tmp_path: Path, message: str) -> str:
    marker = tmp_path / "marker.txt"
    with marker.open("a", encoding="utf-8") as stream:
        stream.write(f"{message}\n")
    _git(tmp_path, "add", "marker.txt")
    _git(tmp_path, "commit", "-q", "-m", message)
    return _git(tmp_path, "rev-parse", "HEAD")


def _history_manifest(
    *,
    visibility: str,
    base: str,
    audited: str,
    public_root: str | None = None,
    audit_scope: str | None = None,
) -> dict:
    manifest = {
        "project": {
            "visibility": visibility,
            "offline_clone_audit": {"audited_commit": audited},
        },
        "source_pins": {"torchtitan": {"commit": base}},
    }
    if public_root is not None:
        manifest["project"]["public_history"] = {
            "root_commit": public_root,
            "policy": "parentless_root_with_ordinary_descendants",
        }
    if audit_scope is not None:
        manifest["project"]["offline_clone_audit"]["scope"] = audit_scope
    return manifest


def _init_history_repo(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    _git(tmp_path, "config", "user.name", "Release Test")
    _git(tmp_path, "config", "user.email", "release-test@example.invalid")


def test_git_visible_files_excludes_ignored_build_products(
    monkeypatch, tmp_path: Path
) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("*.log\n", encoding="utf-8")
    (tmp_path / "tracked.py").write_text("pass\n", encoding="utf-8")
    (tmp_path / "new.py").write_text("pass\n", encoding="utf-8")
    (tmp_path / "build.log").write_text("local build\n", encoding="utf-8")
    subprocess.run(["git", "add", ".gitignore", "tracked.py"], cwd=tmp_path, check=True)
    monkeypatch.setattr(inventory, "ROOT", tmp_path)

    assert inventory._git_visible_files() == [
        tmp_path / ".gitignore",
        tmp_path / "new.py",
        tmp_path / "tracked.py",
    ]


def test_public_history_accepts_parentless_root_and_ordinary_descendant(
    monkeypatch, tmp_path: Path
) -> None:
    _init_history_repo(tmp_path)
    public_root = _commit(tmp_path, "public root")
    monkeypatch.setattr(verifier, "ROOT", tmp_path)
    monkeypatch.setattr(verifier, "EXPECTED_PUBLIC_ROOT", public_root)
    manifest = _history_manifest(
        visibility="public",
        base="1" * 40,
        audited="2" * 40,
        public_root=public_root,
    )

    assert verifier._verify_history_boundary(manifest) == "public_export"

    _commit(tmp_path, "ordinary public update")
    assert verifier._verify_history_boundary(manifest) == "public_export"


def test_public_history_accepts_audited_public_ancestor(
    monkeypatch, tmp_path: Path
) -> None:
    _init_history_repo(tmp_path)
    public_root = _commit(tmp_path, "public root")
    audited = _commit(tmp_path, "audited public source")
    _commit(tmp_path, "audit receipt child")
    monkeypatch.setattr(verifier, "ROOT", tmp_path)
    monkeypatch.setattr(verifier, "EXPECTED_PUBLIC_ROOT", public_root)
    manifest = _history_manifest(
        visibility="public",
        base="1" * 40,
        audited=audited,
        public_root=public_root,
        audit_scope="public_source_closure_and_offline_reproduction",
    )

    assert verifier._verify_history_boundary(manifest) == "public_export"


def test_public_history_rejects_commit_with_private_parent(
    monkeypatch, tmp_path: Path
) -> None:
    _init_history_repo(tmp_path)
    base = _commit(tmp_path, "private base")
    _commit(tmp_path, "ordinary squash with parent")
    monkeypatch.setattr(verifier, "ROOT", tmp_path)
    monkeypatch.setattr(verifier, "EXPECTED_PUBLIC_ROOT", base)
    manifest = _history_manifest(
        visibility="public", base=base, audited=base, public_root=base
    )

    with pytest.raises(verifier.VerificationError, match="private project ancestry"):
        verifier._verify_history_boundary(manifest)


def test_public_history_rejects_second_reachable_root(
    monkeypatch, tmp_path: Path
) -> None:
    _init_history_repo(tmp_path)
    public_root = _commit(tmp_path, "public root")
    _commit(tmp_path, "ordinary public update")
    unrelated = _git(
        tmp_path,
        "commit-tree",
        _git(tmp_path, "rev-parse", f"{public_root}^{{tree}}"),
        "-m",
        "unrelated root",
    )
    _git(tmp_path, "update-ref", "refs/heads/unrelated", unrelated)
    monkeypatch.setattr(verifier, "ROOT", tmp_path)
    monkeypatch.setattr(verifier, "EXPECTED_PUBLIC_ROOT", public_root)
    manifest = _history_manifest(
        visibility="public",
        base="1" * 40,
        audited="2" * 40,
        public_root=public_root,
    )

    with pytest.raises(verifier.VerificationError, match="exactly the pinned"):
        verifier._verify_history_boundary(manifest)


def test_private_history_requires_and_accepts_declared_ancestry(
    monkeypatch, tmp_path: Path
) -> None:
    _init_history_repo(tmp_path)
    base = _commit(tmp_path, "TorchTitan base")
    audited = _commit(tmp_path, "audited candidate")
    _commit(tmp_path, "current private candidate")
    monkeypatch.setattr(verifier, "ROOT", tmp_path)

    manifest = _history_manifest(visibility="private", base=base, audited=audited)
    assert verifier._verify_history_boundary(manifest) == "private_history"

    missing_base = _history_manifest(
        visibility="private", base="3" * 40, audited=audited
    )
    with pytest.raises(verifier.VerificationError, match="TorchTitan base"):
        verifier._verify_history_boundary(missing_base)


def test_public_hosting_surface_rejects_workflows_and_codeowners(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(verifier, "ROOT", tmp_path)
    manifest = {"project": {"visibility": "public"}}

    verifier._verify_hosting_surface(manifest)

    workflow = tmp_path / ".github" / "workflows" / "publish.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text("on: push\n", encoding="utf-8")
    with pytest.raises(verifier.VerificationError, match="Actions workflows"):
        verifier._verify_hosting_surface(manifest)

    workflow.unlink()
    codeowners = tmp_path / "CODEOWNERS"
    codeowners.write_text("* @upstream-maintainer\n", encoding="utf-8")
    with pytest.raises(verifier.VerificationError, match="CODEOWNERS"):
        verifier._verify_hosting_surface(manifest)


def test_git_visible_files_includes_tracked_cache_and_symlink(
    monkeypatch, tmp_path: Path
) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("__pycache__/\n*.pyc\n", encoding="utf-8")
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    bytecode = cache / "tracked.pyc"
    bytecode.write_bytes(b"tracked cache payload")
    target = tmp_path / "target.txt"
    target.write_text("target\n", encoding="utf-8")
    link = tmp_path / "link.txt"
    link.symlink_to("target.txt")
    subprocess.run(
        ["git", "add", ".gitignore", "target.txt", "link.txt"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "add", "-f", str(bytecode.relative_to(tmp_path))],
        cwd=tmp_path,
        check=True,
    )
    monkeypatch.setattr(inventory, "ROOT", tmp_path)

    assert inventory._git_visible_files() == [
        tmp_path / ".gitignore",
        bytecode,
        link,
        target,
    ]


def test_git_visible_files_rejects_missing_tracked_path(
    monkeypatch, tmp_path: Path
) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    tracked = tmp_path / "tracked.py"
    tracked.write_text("pass\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.py"], cwd=tmp_path, check=True)
    tracked.unlink()
    monkeypatch.setattr(inventory, "ROOT", tmp_path)

    with pytest.raises(RuntimeError, match="absent from the worktree"):
        inventory._git_visible_files()


def test_help_does_not_rewrite_inventory(monkeypatch, tmp_path: Path) -> None:
    output = tmp_path / "source_files.sha256"
    output.write_text("sentinel\n", encoding="utf-8")
    monkeypatch.setattr(inventory, "OUTPUT", output)

    try:
        inventory.main(["--help"])
    except SystemExit as error:
        assert error.code == 0

    assert output.read_text(encoding="utf-8") == "sentinel\n"


def test_inventory_covers_every_git_visible_regular_file(
    monkeypatch, tmp_path: Path
) -> None:
    inherited = tmp_path / "torchtitan" / "base.py"
    inherited.parent.mkdir()
    inherited.write_text("pass\n", encoding="utf-8")
    release = tmp_path / "release" / "manifest.json"
    release.parent.mkdir()
    release.write_text("{}\n", encoding="utf-8")
    gitlink = tmp_path / "flash-attention"
    gitlink.mkdir()
    output = tmp_path / "release" / "source_files.sha256"
    output.write_text("self\n", encoding="utf-8")
    monkeypatch.setattr(inventory, "ROOT", tmp_path)
    monkeypatch.setattr(inventory, "OUTPUT", output)
    monkeypatch.setattr(
        inventory,
        "_git_visible_files",
        lambda: [inherited, release, gitlink, output],
    )

    payload, count = inventory.inventory_payload()

    assert count == 2
    assert "torchtitan/base.py" in payload
    assert "release/manifest.json" in payload
    assert "flash-attention" not in payload
    assert "release/source_files.sha256" not in payload


def test_release_scan_rejects_object_store_locators() -> None:
    pattern = verifier.SECRET_PATTERNS["object-store URI"]
    locator = b"source = 's3:" + b"//example-bucket/private-key'"
    assert pattern.search(locator)


def test_release_scan_rejects_modern_token_and_url_shapes() -> None:
    payloads = {
        "fine-grained GitHub token": b"github_" + b"pat_" + b"a" * 24,
        "Hugging Face token": b"h" + b"f_" + b"a" * 24,
        "JSON Web Token": (b"eyJ" + b"a" * 10 + b"." + b"b" * 10 + b"." + b"c" * 10),
        "assigned bearer or access token": (b"access_token='" + b"a" * 24 + b"'"),
        "credential-bearing HTTPS URL": (
            b"https:" + b"//user:secret@example.test/path"
        ),
        "AWS presigned HTTPS parameter": (
            b"https:" + b"//example.test/?X-Amz-" + b"Signature=" + b"a" * 64
        ),
    }

    for label, payload in payloads.items():
        assert verifier.SECRET_PATTERNS[label].search(payload), label


def test_manifest_rejects_duplicate_route_and_export_ids() -> None:
    manifest = json.loads(verifier.MANIFEST_PATH.read_text(encoding="utf-8"))
    duplicate_route = copy.deepcopy(manifest)
    duplicate_route["route_matrix"].append(
        copy.deepcopy(duplicate_route["route_matrix"][0])
    )
    with pytest.raises(verifier.VerificationError, match="duplicate route"):
        verifier._verify_manifest(duplicate_route)

    duplicate_export = copy.deepcopy(manifest)
    duplicate_export["source_exports"].append(
        copy.deepcopy(duplicate_export["source_exports"][0])
    )
    with pytest.raises(verifier.VerificationError, match="duplicate or malformed"):
        verifier._verify_manifest(duplicate_export)


def test_manifest_rejects_mismatched_historical_audit_metadata() -> None:
    manifest = json.loads(verifier.MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["project"]["offline_clone_audit"]["audited_commit"] = "0" * 40

    with pytest.raises(verifier.VerificationError, match="audit declaration"):
        verifier._verify_manifest(manifest)


def test_development_route_catalog_is_complete_and_resolves_paths() -> None:
    verifier._verify_route_catalog()


def test_v510_snapshot_inventory_authenticates_materialized_overlay() -> None:
    verifier._verify_v510_snapshot()


def test_cute_overlay_authenticates_pinned_submodule() -> None:
    verifier._verify_cute_overlay()
