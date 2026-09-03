from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tk_fa4.fp4_fa4_fwd import downstream_provider_suite as suite
from tk_fa4.fp4_fa4_fwd import build_wan_nv_mx_bundle as wan_builder
from tk_fa4.fp4_fa4_fwd import eval_regular_attention as evaluator
from tk_fa4.fp4_fa4_fwd import eval_wan_video as wan_evaluator


def _asset_record(root: Path, payload: bytes = b"weights") -> dict[str, object]:
    file = root / "weights.bin"
    file.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    tree = hashlib.sha256(
        f"weights.bin\0{len(payload)}\0{digest}\n".encode()
    ).hexdigest()
    return {
        "kind": "huggingface_snapshot",
        "identifier": "example/model",
        "revision": "1" * 40,
        "root": str(root.resolve()),
        "files": [
            {
                "path": "weights.bin",
                "bytes": len(payload),
                "sha256": digest,
            }
        ],
        "tree_sha256": tree,
    }


def test_provider_extension_is_rooted_only_in_explicit_build_directory() -> None:
    root = Path("/explicit/noncausal-build")

    extension, arguments = suite.provider_extension("nvmx-accurate", "s4096-h24", root)

    assert extension == root / "b1_s4096_h24_d128_nvmx-accurate.so"
    assert arguments == ["--global-anchor-kv", "--global-anchor-samples", "32"]


def test_no_shape_embeds_a_historical_temporary_root() -> None:
    assert all(
        set(configuration) == {"prefix"} for configuration in suite.SHAPES.values()
    )


def test_asset_record_requires_immutable_authenticated_metadata(
    tmp_path: Path,
) -> None:
    root = tmp_path / "model"
    root.mkdir()
    assets = {"model": _asset_record(root)}

    record = suite.authenticate_asset_record(
        assets,
        "model",
        "model",
        root.resolve(),
    )

    assert record["identifier"] == "example/model"
    assert "root" not in record
    assets["model"]["revision"] = "main"
    with pytest.raises(ValueError, match="not immutable"):
        suite.authenticate_asset_record(assets, "model", "model", root.resolve())


def test_asset_record_rechecks_file_bytes_at_execution(tmp_path: Path) -> None:
    root = tmp_path / "model"
    root.mkdir()
    assets = {"model": _asset_record(root)}

    (root / "weights.bin").write_bytes(b"changed")

    with pytest.raises(ValueError, match="identity mismatch|SHA256 mismatch"):
        suite.authenticate_asset_record(assets, "model", "model", root.resolve())


def test_extension_identities_are_content_addressed(tmp_path: Path) -> None:
    extension = tmp_path / "b1_s256_h16_d128_nvmx-fast.so"
    extension.write_bytes(b"extension")

    identities = suite.extension_identities(
        ["nvmx-fast"],
        "vit-s256",
        tmp_path,
    )

    assert identities["nvmx-fast"] == {
        "file": extension.name,
        "bytes": len(b"extension"),
        "sha256": hashlib.sha256(b"extension").hexdigest(),
    }


def test_asset_identity_arguments_never_include_local_roots() -> None:
    selected = {
        role: {
            "name": role,
            "kind": "snapshot",
            "identifier": f"example/{role}",
            "revision": "1" * 40,
            "tree_sha256": "2" * 64,
        }
        for role in ("model", "dataset")
    }

    arguments = suite.asset_identity_arguments(selected)

    assert not any(Path(argument).is_absolute() for argument in arguments)
    assert arguments == [
        "--model-identifier",
        "example/model",
        "--model-revision",
        "1" * 40,
        "--model-tree-sha256",
        "2" * 64,
        "--dataset-identifier",
        "example/dataset",
        "--dataset-revision",
        "1" * 40,
        "--dataset-tree-sha256",
        "2" * 64,
    ]


def test_portable_asset_identity_hides_authenticated_local_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "asset"
    root.mkdir()

    record = evaluator.portable_asset_identity(
        str(root.resolve()),
        identifier="example/asset",
        revision="1" * 40,
        tree_sha256="2" * 64,
    )

    assert record == {
        "identifier": "example/asset",
        "revision": "1" * 40,
        "tree_sha256": "2" * 64,
        "source": "authenticated_local_snapshot",
    }
    assert str(root.resolve()) not in record.values()


def test_local_asset_without_complete_identity_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "asset"
    root.mkdir()

    with pytest.raises(ValueError, match="authenticated identity"):
        evaluator.portable_asset_identity(
            str(root.resolve()),
            identifier=None,
            revision=None,
            tree_sha256=None,
        )


def test_downstream_reproduction_is_retained_per_task(tmp_path: Path) -> None:
    for task in ("vit-s256", "bert-mlm-s256"):
        suite.write_task_reproduction(
            tmp_path,
            task,
            {
                "asset_manifest_sha256": "1" * 64,
                "assets": {},
                "extensions": {},
            },
        )

    summary = suite.write_summary(tmp_path)

    assert set(summary["reproduction_by_task"]) == {
        "vit-s256",
        "bert-mlm-s256",
    }


def test_downstream_reproduction_merges_provider_shards(tmp_path: Path) -> None:
    common = {
        "asset_manifest_sha256": "1" * 64,
        "assets": {"model": {}, "dataset": {}},
    }
    suite.write_task_reproduction(
        tmp_path,
        "vit-s256",
        {**common, "extensions": {"nvmx-fast": {"sha256": "2" * 64}}},
    )
    suite.write_task_reproduction(
        tmp_path,
        "vit-s256",
        {**common, "extensions": {"hao-nvnv": {"sha256": "3" * 64}}},
    )

    reproduction = suite.load_task_reproductions(tmp_path)["vit-s256"]

    assert set(reproduction["extensions"]) == {"nvmx-fast", "hao-nvnv"}


def test_reused_downstream_result_must_match_current_identities(
    tmp_path: Path,
) -> None:
    selected = {
        role: {
            "identifier": f"example/{role}",
            "revision": "1" * 40,
            "tree_sha256": "2" * 64,
        }
        for role in ("model", "dataset")
    }
    extension = {"file": "kernel.so", "bytes": 6, "sha256": "3" * 64}
    result = {
        role: {
            **selected[role],
            "source": "authenticated_local_snapshot",
        }
        for role in ("model", "dataset")
    }
    result["extension"] = extension
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result))

    suite.assert_reused_result_identity(path, selected, extension)
    result["extension"] = {**extension, "sha256": "4" * 64}
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="extension identity"):
        suite.assert_reused_result_identity(path, selected, extension)


def test_wan_policy_extensions_are_relative_and_authenticated(tmp_path: Path) -> None:
    extension = tmp_path / "kernel.so"
    extension.write_bytes(b"kernel")
    record = wan_builder.extension_record(extension, "_C_kernel", tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")

    path, identity = wan_evaluator.authenticate_policy_extension(
        manifest,
        record,
        "base",
    )

    assert record["path"] == "kernel.so"
    assert path == extension.resolve()
    assert identity["file"] == "kernel.so"
    assert not Path(str(identity["file"])).is_absolute()
    extension.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="identity mismatch|SHA256 mismatch"):
        wan_evaluator.authenticate_policy_extension(manifest, record, "base")


def test_wan_policy_receipt_contains_identities_not_local_paths(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base.so"
    layer = tmp_path / "layer.so"
    base.write_bytes(b"base")
    layer.write_bytes(b"layer")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "tk_wan_nv_mx_policy_bundle_v2",
                "model": "example/model",
                "policies": {
                    "fast": {
                        "base": wan_builder.extension_record(
                            base, "_C_base", tmp_path
                        ),
                        "layer_extensions": [
                            {
                                "layers": "1",
                                "purpose": "test",
                                **wan_builder.extension_record(
                                    layer, "_C_layer", tmp_path
                                ),
                            }
                        ],
                    }
                },
            }
        )
    )
    args = SimpleNamespace(
        policy_manifest=manifest,
        layer_extension=[],
        model="example/model",
        policy="fast",
    )

    receipt = wan_evaluator.apply_policy_manifest(args)

    assert receipt is not None
    serialized = json.dumps(receipt)
    assert str(tmp_path.resolve()) not in serialized
    assert receipt["base"]["file"] == "base.so"
    assert receipt["layer_extensions"][0]["file"] == "layer.so"
