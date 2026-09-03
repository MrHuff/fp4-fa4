from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import plan_fa4_measurements as planner
from tools import render_fa4_training_config as renderer


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOTS = {
    "d64_training_cd59dda": {
        "commit": "cd59dda37ebf22e0d77b9c9d6851ec164b86e3af",
        "parent": "1f3cae064fd3bd5c72c713f7dcea53c4b073952d",
        "tree": "046066c1fd54f1f79fe363fbc4a38a37a495060c",
        "file_count": 7,
        "source_tree_sha256": (
            "7ccb006810b753ab689801efc84e5705a9b614c771d70d5fbcc304971f9e9447"
        ),
    },
    "d64_v416_713819d": {
        "commit": "713819d730369ad9e73ded1aedbc301c261f1130",
        "parent": "abd3f33104ac885434f1d6136ab5100361de51ee",
        "tree": "4133134eb97b45910962f513fc5d3f71b6f0d1cd",
        "file_count": 33,
        "source_tree_sha256": (
            "69d31f3ab7e7586374fbea3a73dd357bca922e1fb696504761301102399c2118"
        ),
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode()
    return hashlib.sha1(header + payload).hexdigest()


def _identity(path: Path, *, module: str | None = None) -> SimpleNamespace:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(path.name.encode())
    return SimpleNamespace(
        path=path.resolve(),
        module=module,
        bytes=path.stat().st_size,
        sha256=_sha256(path),
    )


def _d64_manifest(tmp_path: Path, route: str) -> SimpleNamespace:
    runtime_path = ROOT / "tk_fa4/lowp_fa4_bwd/benchmark_llama12b_e2e.py"
    flash_path = ROOT / "flash-attention/flash_attn/cute/interface.py"
    runtime = _identity(runtime_path)
    flash = _identity(flash_path)
    manifest_path = tmp_path / f"{route}.json"
    manifest_path.write_text("{}\n")
    if route == "bf16_fa4":
        forward = publisher = v416 = None
        pv_format = projection_format = None
    else:
        forward_module = (
            "_C_tk_causal_gqa_nvfp4_fp8pv_exact_b16s4096h32kv8d64"
            if route.endswith("fp8_pv")
            else "_C_cfwd_mx_d4q01_b16s4096h32kv8d64"
        )
        forward = _identity(
            tmp_path / f"{route}-forward.so",
            module=forward_module,
        )
        publisher = _identity(
            tmp_path / "publisher.so",
            module="_C_b300_lowp_bwd",
        )
        v416 = _identity(
            tmp_path / "v416.so",
            module="_C_sm100_gqa_tk_v416_d64_e4m3_production_bshd_dq_first",
        )
        pv_format = "e4m3_fp8" if route.endswith("fp8_pv") else "mxfp4_e8m0_block32"
        projection_format = "e4m3"
    return SimpleNamespace(
        path=manifest_path.resolve(),
        purpose="training",
        profile="llama1p2b-d64-b16",
        route=route,
        batch=16,
        sequence=4096,
        q_heads=32,
        kv_heads=8,
        head_dim=64,
        pv_format=pv_format,
        learned_projection_format=projection_format,
        is_low_precision=route != "bf16_fa4",
        forward=forward,
        projection_publisher=publisher,
        native_backward=v416,
        v416_backward=v416,
        v509_backward=None,
        runtime_source=runtime,
        flash_interface=flash,
        cutlass_dsl=SimpleNamespace(
            root=(tmp_path / "cutlass-dsl").resolve(),
            version="4.5.2",
            native=_identity(tmp_path / "cutlass-native.so"),
        ),
    )


def _render_args(tmp_path: Path, manifest: SimpleNamespace) -> argparse.Namespace:
    hf_assets = tmp_path / "hf-assets"
    hf_assets.mkdir()
    return argparse.Namespace(
        profile="llama1p2b-d64-50b",
        artifact_manifest=manifest.path,
        output=tmp_path / "run.toml",
        dump_folder=str(tmp_path / "output"),
        hf_assets_path=str(hf_assets),
        allow_nonhistorical_tokenizer=True,
        dataset_path="fixture/slimpajama",
        dataset_revision="1" * 40,
        dataset_manifest=None,
        world_size=16,
        global_batch_size=256,
        steps=47_684,
        warmup_steps=954,
        checkpoint_interval=954,
        validation_frequency=262,
        validation_steps=16,
        no_validation=False,
        seed=42,
    )


def _planner_context(tmp_path: Path) -> planner.Context:
    artifacts = {
        (route, 16): _d64_manifest(tmp_path, route) for route in planner.D64_ROUTES
    }
    return planner.Context(
        python=Path("/usr/bin/python3"),
        output_root=(tmp_path / "measurements").resolve(),
        noncausal_build_root=(tmp_path / "noncausal").resolve(),
        cuda_home=None,
        cutlass_dsl_root=None,
        gpu="0",
        artifacts=artifacts,
        artifact_errors=(),
        assets={},
        asset_manifest_path=None,
        launcher=None,
        launcher_error=None,
    )


@pytest.mark.parametrize("name,expected", tuple(SNAPSHOTS.items()))
def test_historical_d64_source_snapshot_is_hash_and_blob_bound(
    name: str,
    expected: dict[str, object],
) -> None:
    root = ROOT / "reproduction/snapshots" / name
    manifest = json.loads((root / "manifest.json").read_text())

    assert manifest["schema"] == "fa4_historical_source_snapshot_v1"
    assert manifest["source"]["commit"] == expected["commit"]
    assert manifest["source"]["parent"] == expected["parent"]
    assert manifest["source"]["commit_tree"] == expected["tree"]
    assert manifest["file_count"] == expected["file_count"]
    assert manifest["source_tree_sha256"] == expected["source_tree_sha256"]
    assert manifest["scope"]["exclusions"] == ["results/**", "**/*.yaml", "**/*.yml"]

    patch = root / manifest["source_patch"]["path"]
    assert patch.stat().st_size == manifest["source_patch"]["bytes"]
    assert _sha256(patch) == manifest["source_patch"]["sha256"]
    patch_bytes = patch.read_bytes()
    assert b"diff --git a/results/" not in patch_bytes
    assert b".yaml" not in patch_bytes
    assert b".yml" not in patch_bytes

    for record in manifest["files"]:
        relative = Path(record["path"])
        assert "results" not in relative.parts
        assert relative.suffix not in {".yaml", ".yml"}
        payload = (root / relative).read_bytes()
        assert len(payload) == record["bytes"]
        assert hashlib.sha256(payload).hexdigest() == record["sha256"]
        assert _git_blob_sha1(payload) == record["git_blob"]

    sums = {
        path: digest
        for digest, path in (
            line.split("  ", 1)
            for line in (root / "SHA256SUMS").read_text().splitlines()
        )
    }
    expected_paths = {
        "README.md",
        "manifest.json",
        manifest["source_patch"]["path"],
        *(record["path"] for record in manifest["files"]),
    }
    assert set(sums) == expected_paths
    for relative, digest in sums.items():
        assert _sha256(root / relative) == digest


def test_route_catalog_exposes_profile_specific_d64_routes() -> None:
    catalog = json.loads((ROOT / "release/routes.json").read_text())
    routes = {route["id"]: route for route in catalog["routes"]}

    backward = routes["v416_d64_represented_e4m3_backward"]
    assert backward["status"] == "release_candidate"
    assert backward["selection"]["portable_build"] is True
    assert backward["selection"]["torchtitan_dispatch"] == "release_candidate"

    fp8 = routes["causal_d64_e4m3_projection_fp8_pv_v416"]
    mx = routes["causal_d64_e4m3_projection_mxfp4_pv_v416"]
    assert fp8["status"] == "release_candidate"
    assert mx["status"] == "diagnostic"
    for route in (fp8, mx):
        assert route["phase"] == "forward_backward"
        assert route["selection"]["portable_build"] is True
        assert route["shape_profiles"] == [
            {
                "label": "1.2B matched release profile",
                "constraint": (
                    "The schema-v3 manifest and TorchTitan adapter admit exactly "
                    "B16/S4096/Hq32/Hkv8/D64 under profile llama1p2b-d64-b16."
                ),
                "batch": [16],
                "sequence": [4096],
                "query_heads": [32],
                "key_value_heads": [8],
                "head_dimension": [64],
            }
        ]
        assert "tests/test_d64_release_recipe.py" in route["test_paths"]
        assert "tools/build_fa4.py" in route["build_paths"]


def test_data_catalog_distinguishes_portable_and_historical_1p2b_recipes() -> None:
    catalog = json.loads((ROOT / "release/data_manifest.json").read_text())
    record = catalog["experiments"]["llama1p2b_long_horizon_templates"]
    profile = record["portable_profile"]

    assert record["status"] == "recipe_only"
    assert len(record["historical_templates"]) == 2
    assert profile["artifact_manifest_schema"] == "fa4_artifact_manifest_v3"
    assert profile["artifact_profile"] == "llama1p2b-d64-b16"
    assert profile["training"] == {
        "world_size": 16,
        "local_batch": 16,
        "global_batch": 256,
        "gradient_accumulation": 1,
        "sequence_length": 4096,
        "updates": 47_684,
        "target_tokens": 50_000_297_984,
        "learning_rate": 0.00048828125,
        "warmup_updates": 954,
        "checkpoint_interval_updates": 954,
        "validation_frequency_updates": 262,
        "validation_steps": 16,
        "validation_local_batch": 16,
        "optimizer": (
            "fused BF16 stochastic-rounding AdamW; historical 1.2B templates "
            "used ordinary AdamW"
        ),
        "cross_entropy": (
            "ordinary dense BF16 cross entropy compiled with TorchInductor"
        ),
        "cut_cross_entropy": False,
    }
    assert profile["dataset"]["historical_mds_identity_complete"] is False
    assert "have not been run" in profile["completion"]


def test_d64_training_profile_binds_the_paper_recipe() -> None:
    profile = renderer.TRAINING_PROFILES["llama1p2b-d64-50b"]

    assert profile.artifact_profile == "llama1p2b-d64-b16"
    assert profile.model_flavor == "1B"
    assert profile.tied_embeddings is True
    assert (
        profile.local_batch,
        profile.sequence,
        profile.query_heads,
        profile.key_value_heads,
        profile.head_dimension,
    ) == (16, 4096, 32, 8, 64)
    assert (profile.world_size, profile.global_batch_size) == (16, 256)
    assert (profile.steps, profile.warmup_steps) == (47_684, 954)
    assert profile.checkpoint_interval == 954
    assert profile.validation_frequency == 262
    assert profile.validation_local_batch == 16
    assert profile.learning_rate == 0.00048828125
    assert (
        profile.steps * profile.global_batch_size * profile.sequence
        == profile.target_tokens
        == 50_000_297_984
    )


def test_d64_renderer_emits_only_the_v416_profile_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _d64_manifest(
        tmp_path,
        "e4m3_proj_nvfp4_qk_fp8_pv",
    )
    monkeypatch.setattr(renderer, "load_artifact_manifest", lambda *_a, **_k: manifest)
    monkeypatch.setattr(
        renderer,
        "_tokenizer_identity",
        lambda *_a, **_k: {"fixture": True},
    )
    monkeypatch.setattr(
        renderer,
        "_dataset",
        lambda _args: ("fixture/slimpajama@" + "1" * 40, {"fixture": True}),
    )

    config, receipt = renderer._render(_render_args(tmp_path, manifest))

    assert 'flavor = "1B"' in config
    assert "scaling_factor = 32.0" in config
    assert "local_batch_size = 16" in config
    assert "global_batch_size = 256" in config
    assert "steps = 47684" in config
    assert "warmup_steps = 954" in config
    assert "interval = 954" in config
    assert "freq = 262" in config
    assert 'components = ["loss"]' in config
    assert "enable_cce = false" in config
    assert 'exact_artifact_profile = "llama1p2b-d64-b16"' in config
    assert "exact_native_tk_d64_backward_extension" in config
    assert "exact_native_tk_d64_backward_module" in config
    assert "exact_native_tk_d64_backward_sha256" in config
    assert "exact_native_tk_d64_backward_bytes" in config
    assert "exact_native_tk_d128_backward" not in config
    assert "exact_d128_" not in config
    assert "exact_backward_control_" not in config
    assert receipt["world_size"] == 16
    assert receipt["global_batch_size"] == 256
    assert receipt["gradient_accumulation_steps"] == 1


def test_d64_renderer_rejects_cross_profile_or_cross_shape_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _d64_manifest(
        tmp_path,
        "e4m3_proj_nvfp4_qk_fp8_pv",
    )
    args = _render_args(tmp_path, manifest)
    monkeypatch.setattr(renderer, "load_artifact_manifest", lambda *_a, **_k: manifest)

    manifest.profile = "llama8b-d128-b4"
    with pytest.raises(ValueError, match="requires artifact profile"):
        renderer._render(args)

    manifest.profile = "llama1p2b-d64-b16"
    manifest.head_dim = 128
    with pytest.raises(ValueError, match="requires attention shape"):
        renderer._render(args)

    manifest.head_dim = 64
    args.world_size = 8
    with pytest.raises(ValueError, match="fixes world size 16"):
        renderer._render(args)


def test_d64_measurement_plan_names_every_evidence_boundary(tmp_path: Path) -> None:
    steps = planner._d64_steps(_planner_context(tmp_path))
    by_name = {step.name: step for step in steps}

    isolated = by_name["d64.isolated-forward"]
    assert isolated.runnable
    assert isolated.command is not None
    assert isolated.command[2:4] == (
        "-m",
        "tk_fa4.lowp_fa4_bwd.benchmark_b16_forward_factorial",
    )
    assert isolated.command[isolated.command.index("--samples") + 1] == "40"

    backward = by_name["d64.isolated-v416-backward"]
    assert not backward.runnable
    assert any("driver was not retained" in blocker for blocker in backward.blockers)
    assert any("cd57" in blocker for blocker in backward.blockers)

    combined = by_name["d64.combined-forward-backward"]
    assert not combined.runnable
    assert any("attention-only F+B" in blocker for blocker in combined.blockers)

    saturated = {
        name: step
        for name, step in by_name.items()
        if name.startswith("d64.saturated.")
    }
    assert set(saturated) == {f"d64.saturated.{route}" for route in planner.D64_ROUTES}
    assert all(step.command is not None for step in saturated.values())
    assert all("--batch" in step.command for step in saturated.values())
    assert all(
        step.command[step.command.index("--batch") + 1] == "16"
        for step in saturated.values()
        if step.command is not None
    )

    for route in planner.D64_ROUTES:
        rendered = by_name[f"d64.distributed.render.{route}"]
        assert rendered.command is not None
        command = rendered.command
        assert command[command.index("--profile") + 1] == "llama1p2b-d64-50b"
        assert command[command.index("--world-size") + 1] == "16"
        assert command[command.index("--global-batch-size") + 1] == "256"
        assert command[command.index("--steps") + 1] == "47684"
        assert command[command.index("--warmup-steps") + 1] == "954"
        assert command[command.index("--checkpoint-interval") + 1] == "954"
        assert command[command.index("--validation-frequency") + 1] == "262"

    assert by_name["d64.ddp16-smoke.launch.fresh-resume"].dependencies == (
        "d64.ddp16-smoke.verify.fresh-resume",
    )
    assert by_name["d64.ddp16-smoke.render.fresh-resume"].dependencies == (
        "d64.ddp16-smoke.launch.save",
    )


def test_d64_measurement_plan_rejects_cross_profile_manifest(tmp_path: Path) -> None:
    context = _planner_context(tmp_path)
    manifest = context.artifacts[("e4m3_proj_nvfp4_qk_fp8_pv", 16)]
    manifest.profile = "llama8b-d128-b4"

    by_name = {step.name: step for step in planner._d64_steps(context)}

    assert planner.ARTIFACT_SCHEMA == "fa4_artifact_manifest_v3"
    for name in (
        "d64.isolated-forward",
        "d64.saturated.e4m3_proj_nvfp4_qk_fp8_pv",
        "d64.distributed.render.e4m3_proj_nvfp4_qk_fp8_pv",
    ):
        assert not by_name[name].runnable
        assert any("D64 artifact profile" in item for item in by_name[name].blockers)
