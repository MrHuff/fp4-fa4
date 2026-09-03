from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import plan_fa4_measurements as planner


def _identity(path: Path, module: str) -> SimpleNamespace:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(path.name.encode())
    return SimpleNamespace(
        path=path.resolve(),
        module=module,
        bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def _manifest(tmp_path: Path, route: str, batch: int, *, training: bool = True):
    if route == "bf16_fa4":
        forward = publisher = backward = None
        pv = None
        projection_format = None
    else:
        forward = _identity(
            tmp_path / f"{route}-b{batch}-forward.so", f"_C_{route}_b{batch}"
        )
        publisher = _identity(tmp_path / "publisher.so", "_C_b300_lowp_bwd")
        backward = _identity(
            tmp_path / f"v509-b{batch}.so",
            "_C_sm100_gqa_tk_v509_d128_nvfp4_score_e4m3_qkv_"
            f"e5m2_dout_b{batch}_s4096",
        )
        pv = "e4m3_fp8" if route.endswith("fp8_pv") else "mxfp4_e8m0_block32"
        projection_format = "e4m3" if route.startswith("e4m3_proj_") else "nvfp4"
    runtime_path = planner.ROOT / "tk_fa4/lowp_fa4_bwd/benchmark_llama12b_e2e.py"
    flash_path = planner.ROOT / "flash-attention/flash_attn/cute/interface.py"
    runtime = SimpleNamespace(
        path=runtime_path,
        bytes=runtime_path.stat().st_size,
        sha256=hashlib.sha256(runtime_path.read_bytes()).hexdigest(),
    )
    flash = SimpleNamespace(
        path=flash_path,
        bytes=flash_path.stat().st_size,
        sha256=hashlib.sha256(flash_path.read_bytes()).hexdigest(),
    )
    return SimpleNamespace(
        path=(tmp_path / f"{route}-b{batch}.json").resolve(),
        purpose="training" if training else "operator_only",
        route=route,
        batch=batch,
        pv_format=pv,
        learned_projection_format=projection_format,
        is_low_precision=route != "bf16_fa4",
        forward=forward,
        projection_publisher=publisher,
        v509_backward=backward,
        runtime_source=runtime,
        flash_interface=flash,
        cutlass_dsl=SimpleNamespace(root=(tmp_path / "cutlass-dsl").resolve()),
    )


def _context(tmp_path: Path) -> planner.Context:
    artifacts = {}
    for route in planner.ROUTES:
        for batch in planner.BATCHES:
            artifacts[(route, batch)] = _manifest(
                tmp_path / "artifacts", route, batch, training=batch != 2
            )
    return planner.Context(
        python=Path("/usr/bin/python"),
        output_root=(tmp_path / "output").resolve(),
        noncausal_build_root=(tmp_path / "noncausal-build").resolve(),
        cuda_home=(tmp_path / "cuda").resolve(),
        cutlass_dsl_root=(tmp_path / "cutlass-dsl").resolve(),
        gpu="0",
        artifacts=artifacts,
        artifact_errors=(),
        assets={},
        asset_manifest_path=None,
        launcher=None,
        launcher_error=None,
    )


def _external_asset(tmp_path: Path, name: str) -> planner.ExternalAsset:
    kind, identifier = planner.EXPECTED_ASSETS[name]
    root = tmp_path / "external-assets" / name
    root.mkdir(parents=True, exist_ok=True)
    return planner.ExternalAsset(
        name=name,
        kind=kind,
        identifier=identifier or name,
        revision="1" * 40,
        root=root.resolve(),
        files=(),
        tree_sha256="2" * 64,
    )


def test_registry_covers_every_experiment_matrix_family(tmp_path: Path) -> None:
    steps = planner.build_steps(_context(tmp_path))
    covered = {family for step in steps for family in step.families}

    assert covered == set(planner.FAMILIES)
    assert any(step.name == "causal.v509-report-boundaries" for step in steps)
    assert any(step.name == "b300.aggregate.from-summary" for step in steps)
    assert any(step.name == "wan.affine-route-table" for step in steps)


def test_causal_boundary_uses_one_authenticated_shared_capture(tmp_path: Path) -> None:
    step = planner._boundary_steps(_context(tmp_path))[0]

    assert step.families == ("causal-backward", "projection-fwd-bwd")
    assert step.runnable
    assert step.command is not None
    assert step.command[2:4] == (
        "-m",
        "tk_fa4.lowp_fa4_bwd.benchmark_v509_report_boundaries",
    )
    assert "--projection-sha256" in step.command
    assert "--forward-sha256" in step.command
    assert "--backward-sha256" in step.command
    assert step.environment["TK_FA4_LOWP_BWD_EXTENSION_SOURCE"].endswith("publisher.so")


def test_e2e_graph_has_exact_batches_routes_and_fails_closed_on_numa(
    tmp_path: Path,
) -> None:
    steps = planner._e2e_steps(_context(tmp_path))

    assert len(steps) == 12
    assert all(not step.runnable for step in steps)
    assert all(any("NUMA0" in blocker for blocker in step.blockers) for step in steps)
    assert {step.name.rsplit(".", 1)[-1] for step in steps} == {"b1", "b2", "b4"}
    for step in steps:
        assert step.command is not None
        assert step.command[2:4] == (
            "-m",
            "tk_fa4.lowp_fa4_bwd.benchmark_llama12b_e2e",
        )
        assert "--compile-loss" in step.command
        if "e4m3_proj_" in step.name:
            assert "--experimental-native-nvfp4-projection-out" not in step.command
            assert (
                step.command[step.command.index("--qkv-projection-format") + 1]
                == "e4m3"
            )
            assert (
                step.command[step.command.index("--output-projection-format") + 1]
                == "e4m3"
            )
        else:
            assert "--experimental-native-nvfp4-projection-out" in step.command
            assert (
                step.command[step.command.index("--qkv-projection-format") + 1]
                == "nvfp4"
            )
            assert (
                step.command[step.command.index("--output-projection-format") + 1]
                == "nvfp4"
            )
        assert "--native-tk-d128-v509-e5m2-dout-backward" in step.command
        assert not any("*" in argument or "?" in argument for argument in step.command)


def test_dependency_blocker_prevents_dependent_command_emission(tmp_path: Path) -> None:
    context = _context(tmp_path)
    selected = planner._selected_steps(planner.build_steps(context), ("vit-mae",))
    unified = next(step for step in selected if step.name == "noncausal.unified")
    assert unified.runnable
    mae = next(step for step in selected if step.name == "vit-mae.tk.nvmx-fast")
    assert not mae.runnable
    assert "missing authenticated external asset 'vit_mae_model'" in mae.blockers
    shell = planner._render_shell(selected)
    assert "# blocked command:" in shell


def test_downstream_replacement_uses_authenticated_assets_and_portable_build_root(
    tmp_path: Path,
) -> None:
    asset_names = {name for pair in planner.DOWNSTREAM_ASSETS.values() for name in pair}
    manifest = tmp_path / "assets.json"
    manifest.write_text("{}")
    context = replace(
        _context(tmp_path),
        assets={name: _external_asset(tmp_path, name) for name in asset_names},
        asset_manifest_path=manifest.resolve(),
    )

    selected = planner._selected_steps(planner.build_steps(context), ("downstream",))
    downstream = [step for step in selected if step.name.startswith("downstream.")]

    assert len(downstream) == len(planner.DOWNSTREAM_TASKS)
    assert all(step.runnable for step in downstream)
    assert all(step.dependencies == ("noncausal.unified",) for step in downstream)
    for step in downstream:
        assert step.command is not None
        assert "--extension-root" in step.command
        assert str(context.noncausal_build_root / "unified") in step.command
        assert "--model-root" in step.command
        assert "--dataset-root" in step.command
        assert "--asset-manifest" in step.command
        assert not any("tk_hao_unified_g" in argument for argument in step.command)


def test_wan_replay_separates_manifest_identifier_from_local_model_path(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "assets.json"
    manifest.write_text("{}")
    context = replace(
        _context(tmp_path),
        assets={
            name: _external_asset(tmp_path, name)
            for name in ("wan_1_3b_model", "wan_14b_model")
        },
        asset_manifest_path=manifest.resolve(),
    )

    replays = [
        step
        for step in planner._wan_steps(context)
        if step.name.startswith("wan.replay.")
    ]

    assert replays
    assert all(step.runnable for step in replays)
    for step in replays:
        assert step.command is not None
        model_index = step.command.index("--model")
        path_index = step.command.index("--model-path")
        assert step.command[model_index + 1].startswith("Wan-AI/")
        assert Path(step.command[path_index + 1]).is_absolute()
        assert step.command[step.command.index("--asset-manifest") + 1] == str(
            manifest.resolve()
        )
        assert step.command[step.command.index("--model-asset") + 1].startswith("wan_")


def test_mae_replay_passes_runtime_asset_authentication_contract(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "assets.json"
    manifest.write_text("{}")
    context = replace(
        _context(tmp_path),
        assets={
            name: _external_asset(tmp_path, name)
            for name in ("vit_mae_model", "coco_val_100")
        },
        asset_manifest_path=manifest.resolve(),
    )

    steps = planner._mae_steps(context)

    assert steps
    for step in steps:
        assert step.command is not None
        assert step.command[step.command.index("--asset-manifest") + 1] == str(
            manifest.resolve()
        )
        assert step.command[step.command.index("--model-asset") + 1] == (
            "vit_mae_model"
        )
        assert step.command[step.command.index("--image-asset") + 1] == ("coco_val_100")


def test_external_asset_manifest_authenticates_explicit_file_list(
    tmp_path: Path,
) -> None:
    root = tmp_path / "asset"
    file = root / "weights.bin"
    file.parent.mkdir()
    file.write_bytes(b"weights")
    identity = {
        "path": "weights.bin",
        "bytes": len(b"weights"),
        "sha256": hashlib.sha256(b"weights").hexdigest(),
    }
    tree = hashlib.sha256(
        (
            "weights.bin\0" + str(len(b"weights")) + "\0" + identity["sha256"] + "\n"
        ).encode()
    ).hexdigest()
    manifest = tmp_path / "assets.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": planner.ASSET_SCHEMA,
                "assets": {
                    "vit_mae_model": {
                        "kind": "huggingface_snapshot",
                        "identifier": "facebook/vit-mae-base",
                        "revision": "1" * 40,
                        "root": str(root.resolve()),
                        "files": [identity],
                        "tree_sha256": tree,
                    }
                },
            }
        )
    )

    assets, observed_path = planner.load_external_assets(manifest.resolve())
    assert observed_path == manifest.resolve()
    assert assets["vit_mae_model"].revision == "1" * 40

    file.write_bytes(b"tampered")
    with pytest.raises(planner.PlanError, match="byte identity mismatch"):
        planner.load_external_assets(manifest.resolve())


def test_relative_user_paths_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(planner.PlanError, match="absolute path"):
        planner._absolute("relative/output", "output root")


def test_python_path_preserves_selected_virtualenv_symlink(tmp_path: Path) -> None:
    interpreter = tmp_path / "venv/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(Path(sys.executable).resolve())
    args = planner._parser().parse_args(
        [
            "check",
            "--family",
            "b300-aggregate",
            "--python",
            str(interpreter),
            "--output-root",
            str((tmp_path / "output").resolve()),
        ]
    )

    context = planner._context(args, ("b300-aggregate",))
    assert context.python == interpreter.absolute()


def test_existing_measurement_output_is_rejected(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "stale.json").write_text("{}")
    args = planner._parser().parse_args(
        [
            "check",
            "--family",
            "b300-aggregate",
            "--python",
            str(Path(sys.executable).resolve()),
            "--output-root",
            str(output.resolve()),
        ]
    )

    with pytest.raises(planner.PlanError, match="absent or empty"):
        planner._context(args, ("b300-aggregate",))


def test_source_building_family_requires_measured_cutlass_dsl_version(
    tmp_path: Path,
) -> None:
    cuda_home = tmp_path / "cuda"
    (cuda_home / "bin").mkdir(parents=True)
    (cuda_home / "bin/nvcc").write_text("nvcc")
    cutlass_root = tmp_path / "cutlass-dsl"
    (cutlass_root / "cutlass").mkdir(parents=True)
    cutlass_init = cutlass_root / "cutlass/__init__.py"
    cutlass_init.write_text('__version__ = "4.5.1"\n')
    args = planner._parser().parse_args(
        [
            "check",
            "--family",
            "noncausal-forward",
            "--python",
            str(Path(sys.executable).resolve()),
            "--output-root",
            str((tmp_path / "output").resolve()),
            "--noncausal-build-root",
            str((tmp_path / "build").resolve()),
            "--cuda-home",
            str(cuda_home.resolve()),
            "--cutlass-dsl-root",
            str(cutlass_root.resolve()),
        ]
    )

    with pytest.raises(planner.PlanError, match="4.5.1.*4.5.2"):
        planner._context(args, ("noncausal-forward",))

    cutlass_init.write_text(f'__version__ = "{planner.EXPECTED_CUTLASS_DSL}"\n')
    context = planner._context(args, ("noncausal-forward",))
    assert context.cutlass_dsl_root == cutlass_root.resolve()


def test_launcher_cannot_override_audited_training_environment(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "torchrun"
    executable.write_bytes(b"#!/bin/sh\n")
    executable.chmod(0o700)
    manifest = tmp_path / "launcher.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": planner.LAUNCHER_SCHEMA,
                "source_revision": "1" * 40,
                "world_size": 64,
                "executable": {
                    "path": str(executable.resolve()),
                    "bytes": executable.stat().st_size,
                    "sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
                },
                "argv_prefix": ["--nnodes=16", "--nproc-per-node=4"],
                "environment": {"LBT_ADAMW_BF16_SR_SEED": "changed"},
            }
        )
    )

    launcher, error = planner.load_launcher(manifest.resolve())
    assert launcher is None
    assert "may not override audited setting" in (error or "")


def test_distributed_recipe_selects_the_fa4_trainer_module(tmp_path: Path) -> None:
    executable = _identity(tmp_path / "torchrun", "")
    executable.path.chmod(0o700)
    launcher = planner.Launcher(
        executable=executable,
        source_revision="1" * 40,
        world_size=64,
        argv_prefix=("--nnodes=16", "--nproc-per-node=4"),
        environment={},
    )
    context = replace(_context(tmp_path), launcher=launcher)

    launch_steps = [
        step
        for step in planner._training_steps(context)
        if step.kind == "distributed-gpu-training"
    ]

    assert len(launch_steps) == 5
    for step in launch_steps:
        assert step.command is not None
        module_index = step.command.index("-m") + 1
        assert step.command[module_index] == "torchtitan.experiments.fa4.train"


def test_distributed_recipe_builds_and_binds_local_dataset_manifest(
    tmp_path: Path,
) -> None:
    context = replace(
        _context(tmp_path),
        assets={
            name: _external_asset(tmp_path, name)
            for name in ("llama3_8b_assets", "slimpajama_dataset")
        },
    )

    steps = planner._training_steps(context)
    dataset_step = next(
        step for step in steps if step.name == "distributed.dataset-manifest"
    )
    render_steps = [step for step in steps if step.kind == "config-render"]
    verify_steps = [step for step in steps if step.kind == "config-verify"]

    assert dataset_step.command is not None
    assert dataset_step.command[2].endswith("tools/fa4_dataset_manifest.py")
    assert dataset_step.command[3] == "create"
    assert len(render_steps) == 5
    for step in render_steps:
        assert step.command is not None
        assert step.dependencies == (dataset_step.name,)
        manifest_index = step.command.index("--dataset-manifest") + 1
        assert step.command[manifest_index] == str(dataset_step.outputs[0])
    assert len(verify_steps) == 5
    for step in verify_steps:
        assert step.command is not None
        assert step.command[2].endswith("tools/verify_fa4_training_config.py")
        assert step.dependencies[0].startswith("distributed.render.")
        assert step.environment["TORCHTITAN_FSDP_ACCUMULATE_WITHOUT_SYNC"] == "1"


def test_known_historical_gaps_are_not_silently_runnable(tmp_path: Path) -> None:
    steps = planner.build_steps(_context(tmp_path))
    names = {
        step.name: step
        for step in steps
        if step.name.startswith("downstream.")
        or step.name.startswith("wan.hao-controls")
        or step.name == "b300.aggregate.from-raw"
    }

    assert names
    assert all(not step.runnable for step in names.values())
    assert all(step.blockers for step in names.values())
