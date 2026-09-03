# Copyright (c) 2026 Graphcore Ltd. All rights reserved.

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import torchtitan.experiments.fa4  # noqa: F401 - performs explicit registration
from torchtitan.config import ConfigManager
from torchtitan.experiments.fa4 import JobConfig as FA4JobConfig
from torchtitan.experiments.fa4.artifacts import (
    SCHEMA,
    SOURCE_CLOSURE_ALGORITHM,
    SOURCE_CLOSURE_FILES,
    SOURCE_CLOSURE_TREES,
    SOURCE_IDENTITY_SCHEMA,
    load_artifact_manifest,
)
from torchtitan.experiments.fa4 import checkpoint as fa4_checkpoint
from torchtitan.experiments.fa4 import data as fa4_data
from torchtitan.experiments.fa4.converters import (
    FeedForwardFusedWithPatchedActivation,
    FeedForwardWithFusedLinear,
    Float32MasterParamsConverter,
    FusedMLPLinearConverter,
    SplineMLPConverter,
)
from torchtitan.experiments.fa4.exact_lowp_attention import (
    _D64_PROFILE,
    _D128_PROFILE,
    _EXACT_FP8_PV,
    _EXACT_MXFP4_PV,
    _EXACT_NVFP4_PROJECTIONS,
    _EXACT_RETAINED_SPLIT_V,
    _contract_log_line,
    _current_runtime_only_kwargs,
    _restore_root_embedding_alias,
    _validate_bf16_topology_converter_contract,
    _validate_converter_contract,
    _validate_root_parameter_contract,
    _validated_local_batch_size,
)
from torchtitan.experiments.fa4.optimizer.fused_adamw_bf16_sr import (
    CUDA_FLAGS,
    SOURCE_SHA256,
    provider_receipt,
)
from torchtitan.experiments.fa4.optimizer.optimizer_sr_state import (
    AdamWBF16SR,
)
from torchtitan.experiments.fa4.train_spec import fa4_llama_args
from torchtitan.models.llama3.model.model import FeedForward, Transformer
from torchtitan.distributed import ParallelDims
from torchtitan.protocols.model_converter import build_model_converters
from torchtitan.protocols.train_spec import get_train_spec
from tools.fa4_dataset_manifest import DatasetManifestError, create_dataset_manifest
from tools.verify_fa4_training_config import (
    VerificationError as TrainingConfigVerificationError,
    verify_training_config,
)


def _identity(path: Path, *, module: str | None = None) -> dict[str, object]:
    value: dict[str, object] = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    if module is not None:
        value["module"] = module
    return value


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _initialize_git_fixture(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "--quiet")
    _git(root, "add", "--all")
    _git(
        root,
        "-c",
        "user.name=FA4 test",
        "-c",
        "user.email=fa4-test@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "fixture",
    )


def _git_identity(root: Path) -> dict[str, object]:
    return {
        "head": _git(root, "rev-parse", "HEAD"),
        "head_tree": _git(root, "rev-parse", "HEAD^{tree}"),
        "dirty": bool(_git(root, "status", "--porcelain=v1", "--untracked-files=all")),
    }


def _release_source_identity(root: Path) -> dict[str, object]:
    for relative_tree in SOURCE_CLOSURE_TREES:
        (root / relative_tree).mkdir(parents=True, exist_ok=True)
    relative_paths = (
        "ThunderKittens/kernels/common.mk",
        "ThunderKittens/kernels/gemm/nvfp4_b200/nvfp4_quantize.cuh",
        "flash-attention/flash_attn/cute/interface.py",
        "scripts/fa4/run_torchrun.sh",
        "tk_fa4/lowp_fa4_bwd/benchmark_llama12b_e2e.py",
        "tk_fa4/lowp_fa4_bwd/native_tk_d64_backward.py",
        "tools/build_fa4.py",
        "tools/fa4_dataset_manifest.py",
        "tools/plan_fa4_measurements.py",
        "tools/render_fa4_training_config.py",
        "tools/verify_fa4_training_config.py",
        "torchtitan/experiments/fa4/artifacts.py",
        "torchtitan/experiments/fa4/exact_lowp_attention.py",
    )
    files = []
    digest = hashlib.sha256()
    for relative in relative_paths:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(f"source:{relative}\n".encode())
        record = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        files.append(record)
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(str(record["bytes"]).encode())
        digest.update(b"\0")
        digest.update(record["sha256"].encode())
        digest.update(b"\n")
    submodule_names = ("ThunderKittens", "cutlass", "flash-attention", "qutlass")
    for name in submodule_names:
        submodule = root / name
        submodule.mkdir(parents=True, exist_ok=True)
        placeholder = submodule / ".fixture"
        if not any(path.is_file() for path in submodule.rglob("*")):
            placeholder.write_text("fixture\n")
        _initialize_git_fixture(submodule)
    _initialize_git_fixture(root)
    # Keep the root deliberately dirty. A later mutation inside the source
    # scope therefore exercises the content closure rather than merely
    # toggling the recorded dirty boolean.
    (root / ".unrelated-test-scratch").write_text("untracked\n")
    git = _git_identity(root)
    return {
        "schema": SOURCE_IDENTITY_SCHEMA,
        "root": str(root.resolve()),
        "git": git,
        "submodules": {name: _git_identity(root / name) for name in submodule_names},
        "closure": {
            "algorithm": SOURCE_CLOSURE_ALGORITHM,
            "scope": {
                "trees": list(SOURCE_CLOSURE_TREES),
                "files": list(SOURCE_CLOSURE_FILES),
                "excluded_directories": ["__pycache__", "results"],
                "excluded_suffixes": [
                    ".a",
                    ".cubin",
                    ".ncu-rep",
                    ".nsys-rep",
                    ".o",
                    ".ptx",
                    ".pyc",
                    ".sass",
                    ".so",
                ],
            },
            "sha256": digest.hexdigest(),
            "file_count": len(files),
            "files": files,
        },
    }


def _content_closure(root: Path, paths: tuple[Path, ...]) -> dict[str, object]:
    files = []
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        record = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        files.append(record)
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(str(record["bytes"]).encode())
        digest.update(b"\0")
        digest.update(record["sha256"].encode())
        digest.update(b"\n")
    return {
        "algorithm": SOURCE_CLOSURE_ALGORITHM,
        "sha256": digest.hexdigest(),
        "file_count": len(files),
        "files": files,
    }


def _fake_manifest(
    tmp_path: Path,
    *,
    route: str = "nvfp4_qk_fp8_pv",
    batch: int = 4,
) -> Path:
    source_root = tmp_path / "source"
    runtime = source_root / "tk_fa4/lowp_fa4_bwd/benchmark_llama12b_e2e.py"
    flash = source_root / "flash-attention/flash_attn/cute/interface.py"
    cutlass_root = tmp_path / "cutlass-python"
    cutlass_init = cutlass_root / "cutlass/__init__.py"
    cutlass_helper = cutlass_root / "cutlass/runtime.py"
    cutlass_native = cutlass_root / "cutlass/_mlir/_mlir_libs/_cutlass_ir_test.so"
    artifacts = tmp_path / "artifacts"
    forward = artifacts / "forward.so"
    projection = artifacts / "projection.so"
    v509 = artifacts / "v509.so"
    for path, contents in (
        (runtime, b"runtime-source"),
        (flash, b"flash-interface"),
        (cutlass_init, b"__version__ = '4.5.2'\n"),
        (cutlass_helper, b"# runtime helper\n"),
        (cutlass_native, b"cutlass-native"),
        (forward, b"forward-extension"),
        (projection, b"projection-extension"),
        (v509, b"v509-extension"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)

    if route == "bf16_fa4":
        pv_format = None
        projection_format = None
        artifact_values = {
            "forward": None,
            "projection_publisher": None,
            "native_backward": None,
        }
    else:
        pv_format = "e4m3_fp8" if route.endswith("fp8_pv") else "mxfp4_e8m0_block32"
        projection_format = "e4m3" if route.startswith("e4m3_proj_") else "nvfp4"
        artifact_values = {
            "forward": _identity(
                forward,
                module=(
                    "_C_tk_causal_gqa_nvfp4_fp8pv_exact_" f"b{batch}s4096h32kv8d128"
                    if pv_format == "e4m3_fp8"
                    else (
                        "_C_d128_mx_maxsafe_anchor32_represented_"
                        f"b{batch}s4096h32kv8d128_b200_sm152"
                    )
                ),
            ),
            "projection_publisher": _identity(projection, module="_C_b300_lowp_bwd"),
            "native_backward": _identity(
                v509,
                module=(
                    "_C_sm100_gqa_tk_v509_d128_nvfp4_score_e4m3_qkv_"
                    f"e5m2_dout_b{batch}_s4096"
                ),
            ),
        }
    value = {
        "schema": SCHEMA,
        "purpose": "training",
        "profile": {
            "name": f"llama8b-d128-b{batch}",
            "model_preset": "llama3.1-8b",
            "runtime_contract": f"d128-b{batch}-native-v509",
            "fp8_pv_forward": "exact",
            "mxfp4_pv_forward": "maxsafe-anchor32-represented",
            "projection_publisher": "b300-lowp-bwd",
            "native_backward": "v509",
        },
        "route": {
            "name": route,
            "pv_format": pv_format,
            "learned_projection_format": projection_format,
        },
        "shape": {
            "batch": batch,
            "sequence": 4096,
            "q_heads": 32,
            "kv_heads": 8,
            "head_dim": 128,
        },
        "architecture": {
            "gpu": "NVIDIA GB200",
            "compute_capability": [10, 0],
            "cuda_arch": "sm_100a",
        },
        "artifacts": artifact_values,
        "sources": {
            "release": _release_source_identity(source_root),
            "runtime_source": _identity(runtime),
            "flash_interface": _identity(flash),
            "cutlass_dsl": {
                "root": str(cutlass_root.resolve()),
                "version": "4.5.2",
                "native": _identity(cutlass_native),
                "closure": _content_closure(
                    cutlass_root,
                    (cutlass_init, cutlass_helper, cutlass_native),
                ),
            },
        },
    }
    manifest = tmp_path / f"{route}-b{batch}.json"
    manifest.write_text(json.dumps(value))
    return manifest


def _local_dataset(tmp_path: Path) -> tuple[Path, Path]:
    dataset = tmp_path / "slimpajama-snapshot"
    dataset.mkdir()
    (dataset / "part-00000.jsonl").write_text('{"text":"fixture"}\n')
    manifest = tmp_path / "slimpajama-snapshot.manifest.json"
    create_dataset_manifest(dataset, manifest)
    return dataset, manifest


def _render_test_training_config(
    tmp_path: Path,
    *,
    world_size: int = 1,
) -> tuple[Path, Path, Path]:
    manifest = _fake_manifest(tmp_path)
    hf_assets = tmp_path / "hf-assets"
    hf_assets.mkdir()
    (hf_assets / "tokenizer.json").write_text('{"version": "1.0"}')
    (hf_assets / "tokenizer_config.json").write_text("{}")
    dataset, dataset_manifest = _local_dataset(tmp_path)
    output = tmp_path / "rendered.toml"
    renderer = Path(__file__).parents[2] / "tools/render_fa4_training_config.py"
    subprocess.run(
        [
            sys.executable,
            str(renderer),
            "--artifact-manifest",
            str(manifest),
            "--output",
            str(output),
            "--dump-folder",
            str(tmp_path / "run"),
            "--hf-assets-path",
            str(hf_assets),
            "--allow-nonhistorical-tokenizer",
            "--dataset-path",
            str(dataset),
            "--dataset-manifest",
            str(dataset_manifest),
            "--world-size",
            str(world_size),
            "--global-batch-size",
            str(4 * world_size),
        ],
        cwd=Path(__file__).parents[2],
        check=True,
        capture_output=True,
        text=True,
    )
    receipt = Path(str(output) + ".receipt.json")
    return manifest, output, receipt


def test_custom_config_merges_fa4_fields_into_torchtitan() -> None:
    manager = ConfigManager()
    merged = manager._merge_configs(manager.config_cls, FA4JobConfig)
    config = merged()

    assert hasattr(config, "fa4")
    assert config.fa4.exact_pv_format == _EXACT_FP8_PV
    assert config.fa4.cuda_data_prefetch is False
    assert config.fa4.fail_on_nonfinite_metrics is False
    assert config.fa4.scan_nonfinite_gradients is False
    assert config.fa4.gradient_diagnostics_topk == 0
    assert config.training.enable_cce is False
    assert config.training.enable_fp32_master_params is False
    assert config.spline_mlp.activation_impl == "native_silu"


def _single_rank_parallel_dims() -> ParallelDims:
    return ParallelDims(
        dp_replicate=1,
        dp_shard=1,
        cp=1,
        tp=1,
        pp=1,
        ep=1,
        etp=1,
        world_size=1,
    )


def test_sfu_b1_mlp_converter_chain_is_registered_and_preserves_silu() -> None:
    torch.manual_seed(11)
    original = FeedForward(
        dim=8,
        hidden_dim=16,
        multiple_of=4,
        ffn_dim_multiplier=None,
    )
    model = torch.nn.Sequential(original)
    sample = torch.randn(3, 5, 8)
    expected = model(sample)
    gate_weight = original.w1.weight.detach().clone()
    up_weight = original.w3.weight.detach().clone()
    down_weight = original.w2.weight.detach().clone()

    config = FA4JobConfig()
    config.model.converters = ["fuse_mlp_linear", "spline_mlp"]
    config.spline_mlp.activation_impl = "native_silu"
    converters = build_model_converters(config, _single_rank_parallel_dims())

    assert [type(converter) for converter in converters.converters] == [
        FusedMLPLinearConverter,
        SplineMLPConverter,
    ]
    converters.converters[0].convert(model)
    fused = model[0]
    assert type(fused) is FeedForwardWithFusedLinear
    torch.testing.assert_close(
        fused.w_in.weight,
        torch.cat((gate_weight, up_weight), dim=0),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(fused.w_out.weight, down_weight, rtol=0, atol=0)
    torch.testing.assert_close(model(sample), expected, rtol=1e-6, atol=1e-6)

    converters.converters[1].convert(model)
    fused = model[0]
    assert type(fused) is FeedForwardFusedWithPatchedActivation
    assert fused.activation_impl_name == "native_silu"
    torch.testing.assert_close(model(sample), expected, rtol=1e-6, atol=1e-6)


def test_spline_converter_rejects_unrecovered_external_extension() -> None:
    config = FA4JobConfig()
    config.model.converters = ["spline_mlp"]
    config.spline_mlp.activation_impl = "spline_silu"
    with pytest.raises(ValueError, match="not in the authenticated release"):
        build_model_converters(config, _single_rank_parallel_dims())


def test_fp32_master_converter_preserves_parameter_aliases_and_buffers() -> None:
    class AliasedParameters(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(4, 4, dtype=torch.bfloat16))
            self.tied_weight = self.weight
            self.register_buffer(
                "rope",
                torch.polar(torch.ones(4), torch.arange(4, dtype=torch.float32)),
            )
            self.register_buffer("floating_buffer", torch.ones(4, dtype=torch.bfloat16))

    model = AliasedParameters()
    parameter_identity = id(model.weight)
    rope = model.rope.clone()
    config = FA4JobConfig()
    config.model.converters = ["fp32_master"]
    converters = build_model_converters(config, _single_rank_parallel_dims())

    assert isinstance(converters.converters[0], Float32MasterParamsConverter)
    converters.convert(model)
    assert id(model.weight) == parameter_identity
    assert model.tied_weight is model.weight
    assert model.weight.dtype == torch.float32
    assert model.rope.dtype == torch.complex64
    assert model.floating_buffer.dtype == torch.bfloat16
    torch.testing.assert_close(model.rope, rope, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("flavor", "profile"),
    (("1B", _D64_PROFILE), ("8B_llama3_blog", _D128_PROFILE)),
)
def test_paper_model_geometry_matches_authenticated_parameter_count(
    flavor, profile
) -> None:
    with torch.device("meta"):
        model = Transformer(fa4_llama_args[flavor])
    _restore_root_embedding_alias(model, profile)
    _validate_root_parameter_contract(model, profile)


def test_registered_train_spec_uses_portable_optimizer_factory() -> None:
    spec = get_train_spec("llama3_gc")
    assert set(spec.model_args) == {"1B", "8B", "8B_llama3_blog"}
    assert spec.build_optimizers_fn.__module__.endswith("fa4.optimizer.build")


def test_d128_routes_accept_only_authenticated_local_batches() -> None:
    assert (
        _validated_local_batch_size(
            _D128_PROFILE, 4, label="test", allow_current_d128=True
        )
        == 4
    )
    with pytest.raises(ValueError, match="local batch"):
        _validated_local_batch_size(
            _D128_PROFILE, 3, label="test", allow_current_d128=True
        )


@pytest.mark.parametrize("pv_format", (_EXACT_FP8_PV, _EXACT_MXFP4_PV))
def test_measured_d128_routes_share_backward_publication(pv_format) -> None:
    kwargs = _current_runtime_only_kwargs(
        False,
        pv_format,
        _EXACT_RETAINED_SPLIT_V,
        current_d128_route=True,
        learned_projection_format=_EXACT_NVFP4_PROJECTIONS,
        native_score_backward=True,
        represented_qk_backward=False,
        e5m2_dout_backward=True,
    )
    assert kwargs == {
        "experimental_native_nvfp4_projection_out": True,
        "experimental_fused_attention_rmsnorm_nvfp4": False,
        "experimental_split_v_backward": False,
        "experimental_output_shared_split_v": False,
        "experimental_d128_mxfp4_v_backward": False,
        "v_mxfp4_scale_2d": False,
        "native_tk_d128_native_score_backward": True,
        "native_tk_d128_v509_e5m2_dout_backward": True,
    }


def test_adamw_provider_identity_matches_measured_source() -> None:
    receipt = provider_receipt()
    assert SOURCE_SHA256 == (
        "05e9133ac24ac286e059ebaaef4311921c5566f0b57e07367af30ac2f48f4dbd"
    )
    assert receipt["source_sha256"] == SOURCE_SHA256
    assert CUDA_FLAGS == ("-O3", "-DSR_ADAMW_HASH32=1")


def test_adamw_provider_fails_clearly_without_cuda() -> None:
    parameter = torch.nn.Parameter(torch.ones(8, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="requires one BF16 CUDA device"):
        AdamWBF16SR([parameter])


@pytest.mark.parametrize(
    "route",
    (
        "bf16_fa4",
        "nvfp4_qk_fp8_pv",
        "nvfp4_qk_mxfp4_pv",
        "e4m3_proj_nvfp4_qk_fp8_pv",
        "e4m3_proj_nvfp4_qk_mxfp4_pv",
    ),
)
def test_training_artifact_manifest_authenticates_every_route(
    tmp_path: Path, route: str
) -> None:
    manifest_path = _fake_manifest(tmp_path, route=route)
    manifest = load_artifact_manifest(manifest_path)

    assert manifest.route == route
    assert manifest.batch == 4
    assert manifest.compute_capability == (10, 0)
    assert manifest.is_low_precision is (route != "bf16_fa4")


def test_artifact_manifest_rejects_tampered_binary(tmp_path: Path) -> None:
    manifest_path = _fake_manifest(tmp_path)
    raw = json.loads(manifest_path.read_text())
    Path(raw["artifacts"]["forward"]["path"]).write_bytes(b"tampered")

    with pytest.raises(RuntimeError, match="identity mismatch|SHA256 mismatch"):
        load_artifact_manifest(manifest_path)


def test_artifact_manifest_rejects_post_build_source_mutation(tmp_path: Path) -> None:
    manifest_path = _fake_manifest(tmp_path)
    raw = json.loads(manifest_path.read_text())
    release_root = Path(raw["sources"]["release"]["root"])
    helper = release_root / "torchtitan/experiments/fa4/exact_lowp_attention.py"
    helper.write_bytes(b"post-build mutation")

    with pytest.raises(RuntimeError, match="identity mismatch|SHA256 mismatch"):
        load_artifact_manifest(manifest_path)


def test_artifact_manifest_rejects_post_build_source_addition(tmp_path: Path) -> None:
    manifest_path = _fake_manifest(tmp_path)
    raw = json.loads(manifest_path.read_text())
    release_root = Path(raw["sources"]["release"]["root"])
    (release_root / "tk_fa4/new_runtime_helper.py").write_text("VALUE = 1\n")

    with pytest.raises(RuntimeError, match="source inventory differs"):
        load_artifact_manifest(manifest_path)


def test_artifact_manifest_rejects_source_symlink_escape(tmp_path: Path) -> None:
    manifest_path = _fake_manifest(tmp_path)
    raw = json.loads(manifest_path.read_text())
    release_root = Path(raw["sources"]["release"]["root"])
    source = release_root / "torchtitan/experiments/fa4/exact_lowp_attention.py"
    outside = tmp_path / "outside.py"
    outside.write_bytes(source.read_bytes())
    source.unlink()
    source.symlink_to(outside)

    with pytest.raises(ValueError, match="escapes release source root"):
        load_artifact_manifest(manifest_path)


def test_artifact_manifest_rejects_cutlass_dsl_source_mutation(tmp_path: Path) -> None:
    manifest_path = _fake_manifest(tmp_path)
    raw = json.loads(manifest_path.read_text())
    cutlass_root = Path(raw["sources"]["cutlass_dsl"]["root"])
    (cutlass_root / "cutlass/runtime.py").write_bytes(b"post-build mutation")

    with pytest.raises(RuntimeError, match="identity mismatch|SHA256 mismatch"):
        load_artifact_manifest(manifest_path)


def test_artifact_manifest_rejects_release_git_identity_drift(tmp_path: Path) -> None:
    manifest_path = _fake_manifest(tmp_path)
    raw = json.loads(manifest_path.read_text())
    release_root = Path(raw["sources"]["release"]["root"])
    _git(release_root, "add", ".unrelated-test-scratch")
    _git(
        release_root,
        "-c",
        "user.name=FA4 test",
        "-c",
        "user.email=fa4-test@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "advance fixture",
    )

    with pytest.raises(RuntimeError, match="release source Git identity mismatch"):
        load_artifact_manifest(manifest_path)


def test_artifact_manifest_rejects_unclosed_submodule_drift(tmp_path: Path) -> None:
    manifest_path = _fake_manifest(tmp_path)
    raw = json.loads(manifest_path.read_text())
    release_root = Path(raw["sources"]["release"]["root"])
    (release_root / "qutlass/.fixture").write_text("mutated\n")

    with pytest.raises(RuntimeError, match="submodule qutlass Git identity mismatch"):
        load_artifact_manifest(manifest_path)


def test_contract_log_names_projection_qk_pv_and_backward_formats() -> None:
    line = _contract_log_line(
        _D128_PROFILE,
        _EXACT_MXFP4_PV,
        True,
        "e4m3",
    )

    assert "route=e4m3_proj_nvfp4_qk_mxfp4_pv_e5m2_dout_backward" in line
    assert "learned_projection_format=e4m3" in line
    assert "qk_format=nvfp4_e4m3_block16" in line
    assert "pv_format=mxfp4_e8m0_block32" in line
    assert "backward_format=nvfp4_score_e4m3_qkv_e5m2_dout" in line


def test_artifact_manifest_rejects_projection_route_mismatch(tmp_path: Path) -> None:
    manifest_path = _fake_manifest(tmp_path, route="e4m3_proj_nvfp4_qk_fp8_pv")
    raw = json.loads(manifest_path.read_text())
    raw["route"]["learned_projection_format"] = "nvfp4"
    manifest_path.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="learned_projection_format='e4m3'"):
        load_artifact_manifest(manifest_path)


def test_artifact_manifest_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    manifest_path = _fake_manifest(tmp_path)
    text = manifest_path.read_text()
    manifest_path.write_text(
        text.replace(
            f'"schema": "{SCHEMA}",',
            f'"schema": "{SCHEMA}",\n  "schema": "{SCHEMA}",',
            1,
        )
    )

    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_artifact_manifest(manifest_path)


@pytest.mark.parametrize(
    "route",
    (
        "bf16_fa4",
        "nvfp4_qk_fp8_pv",
        "nvfp4_qk_mxfp4_pv",
        "e4m3_proj_nvfp4_qk_fp8_pv",
        "e4m3_proj_nvfp4_qk_mxfp4_pv",
    ),
)
def test_training_config_renderer_is_manifest_driven_and_parseable(
    tmp_path: Path, route: str
) -> None:
    manifest_path = _fake_manifest(tmp_path, route=route)
    hf_assets = tmp_path / "hf-assets"
    hf_assets.mkdir()
    (hf_assets / "tokenizer.json").write_text('{"version": "1.0"}')
    (hf_assets / "tokenizer_config.json").write_text(
        '{"bos_token": "<s>", "eos_token": "</s>"}'
    )
    dataset, dataset_manifest = _local_dataset(tmp_path)
    output = tmp_path / "rendered.toml"
    renderer = Path(__file__).parents[2] / "tools/render_fa4_training_config.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(renderer),
            "--artifact-manifest",
            str(manifest_path),
            "--output",
            str(output),
            "--dump-folder",
            str(tmp_path / "run"),
            "--hf-assets-path",
            str(hf_assets),
            "--allow-nonhistorical-tokenizer",
            "--dataset-path",
            str(dataset),
            "--dataset-manifest",
            str(dataset_manifest),
        ],
        cwd=Path(__file__).parents[2],
        check=True,
        capture_output=True,
        text=True,
    )
    assert str(output) in completed.stdout
    rendered = output.read_text()
    assert "gradient accumulation: 4" in rendered.lower()

    config = ConfigManager().parse_args(["--job.config-file", str(output)])
    assert config.model.name == "llama3_gc"
    assert config.model.flavor == "8B_llama3_blog"
    assert isinstance(
        config.model.rope_scaling_args,
        type(FA4JobConfig().model.rope_scaling_args),
    )
    assert config.model.rope_scaling_args.scaling_factor == 8.0
    assert config.training.local_batch_size == 4
    assert config.training.global_batch_size == 1024
    assert config.training.enable_cce is False
    assert config.compile.components == ["loss"]
    assert config.checkpoint.load_step == -1
    assert config.comm.init_timeout_seconds == 1800
    assert config.comm.train_timeout_seconds == 3600
    assert config.fa4.cuda_data_prefetch is True
    assert config.fa4.fail_on_nonfinite_metrics is True
    assert config.fa4.scan_nonfinite_gradients is False
    assert config.fa4.gradient_diagnostics_topk == 0
    parallel_dims = ParallelDims(
        dp_replicate=64,
        dp_shard=1,
        cp=1,
        tp=1,
        pp=1,
        ep=1,
        etp=1,
        world_size=64,
    )
    artifact_manifest = load_artifact_manifest(manifest_path)
    if route == "bf16_fa4":
        assert "exact_forward_extension" not in rendered
        assert config.model.converters == [
            "bfloat16",
            "fa4_exact_bf16_topology",
            "fa4_attention",
        ]
        _validate_bf16_topology_converter_contract(config, parallel_dims)
    else:
        assert f'exact_pv_format = "{artifact_manifest.pv_format}"' in rendered
        assert (
            "exact_learned_projection_format = "
            f'"{artifact_manifest.learned_projection_format}"'
        ) in rendered
        assert config.model.converters == [
            "bfloat16",
            "fa4_exact_lowp_attention",
        ]
        assert config.fa4.exact_forward_sha256 == (artifact_manifest.forward.sha256)
        _validate_converter_contract(config, parallel_dims)

    receipt = json.loads(Path(str(output) + ".receipt.json").read_text())
    assert receipt["schema"] == "fa4_training_config_receipt_v2"
    assert Path(receipt["config"]) == output.resolve()
    assert receipt["config_relative_to_receipt"] == output.name
    assert receipt["config_bytes"] == len(output.read_bytes())
    assert receipt["config_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert Path(receipt["artifact_manifest"]) == manifest_path.resolve()
    assert (
        receipt["artifact_manifest_sha256"]
        == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    )
    assert receipt["route"] == route
    assert receipt["gradient_accumulation_steps"] == 4
    assert receipt["trainer_module"] == "torchtitan.experiments.fa4.train"
    assert receipt["training_integration"] == {
        "cuda_data_prefetch": True,
        "checkpoint_aligned_lookahead": True,
        "fail_on_nonfinite_metrics": True,
        "scan_nonfinite_gradients": False,
        "gradient_diagnostics_topk": 0,
    }
    assert receipt["tokenizer"]["historical_four_file_identity"] is False
    assert [item["path"] for item in receipt["tokenizer"]["files"]] == [
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    assert receipt["dataset"]["kind"] == "local_snapshot"
    assert receipt["dataset"]["path"] == str(dataset.resolve())
    assert receipt["dataset"]["manifest"] == str(dataset_manifest.resolve())
    assert (
        receipt["dataset"]["manifest_sha256"]
        == hashlib.sha256(dataset_manifest.read_bytes()).hexdigest()
    )
    assert receipt["dataset"]["file_count"] == 1
    assert receipt["required_environment"] == {
        "FA4_DATALOADER_PIN_MEMORY": "0",
        "FA4_DATALOADER_PREFETCH_FACTOR": "8",
        "FA4_TRAIN_DATALOADER_NUM_WORKERS": "8",
        "FA4_VALIDATION_DATALOADER_NUM_WORKERS": "1",
        "LBT_ADAMW_BF16_SR_CHECKPOINT_SCHEMA": "v2-fused-stateless",
        "LBT_ADAMW_BF16_SR_PROVIDER": "lbt_fused_stateless_adamw_bf16_sr",
        "LBT_ADAMW_BF16_SR_PROVIDER_VERSION": "1",
        "LBT_ADAMW_BF16_SR_SEED": "0",
        "LBT_ADAMW_BF16_SR_SOURCE_SHA256": SOURCE_SHA256,
        "LBT_DCP_SYNC_CPU_PROCESS_GROUP": "1",
        "TORCHTITAN_FSDP_ACCUMULATE_WITHOUT_SYNC": "1",
    }


def test_training_config_renderer_rejects_unverified_tokenizer_by_default(
    tmp_path: Path,
) -> None:
    manifest_path = _fake_manifest(tmp_path)
    hf_assets = tmp_path / "hf-assets"
    hf_assets.mkdir()
    (hf_assets / "tokenizer.json").write_text('{"version": "1.0"}')
    (hf_assets / "tokenizer_config.json").write_text("{}")
    dataset, dataset_manifest = _local_dataset(tmp_path)
    renderer = Path(__file__).parents[2] / "tools/render_fa4_training_config.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(renderer),
            "--artifact-manifest",
            str(manifest_path),
            "--output",
            str(tmp_path / "rendered.toml"),
            "--dump-folder",
            str(tmp_path / "run"),
            "--hf-assets-path",
            str(hf_assets),
            "--dataset-path",
            str(dataset),
            "--dataset-manifest",
            str(dataset_manifest),
        ],
        cwd=Path(__file__).parents[2],
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "historical tokenizer requires" in completed.stderr


def test_training_config_renderer_requires_manifest_for_local_dataset(
    tmp_path: Path,
) -> None:
    manifest_path = _fake_manifest(tmp_path)
    hf_assets = tmp_path / "hf-assets"
    hf_assets.mkdir()
    (hf_assets / "tokenizer.json").write_text('{"version": "1.0"}')
    (hf_assets / "tokenizer_config.json").write_text("{}")
    dataset = tmp_path / "slimpajama-snapshot"
    dataset.mkdir()
    (dataset / "part.jsonl").write_text("{}\n")
    renderer = Path(__file__).parents[2] / "tools/render_fa4_training_config.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(renderer),
            "--artifact-manifest",
            str(manifest_path),
            "--output",
            str(tmp_path / "rendered.toml"),
            "--dump-folder",
            str(tmp_path / "run"),
            "--hf-assets-path",
            str(hf_assets),
            "--allow-nonhistorical-tokenizer",
            "--dataset-path",
            str(dataset),
        ],
        cwd=Path(__file__).parents[2],
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "requires --dataset-manifest" in completed.stderr


def test_training_config_remote_dataset_uses_immutable_revision_without_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = _fake_manifest(tmp_path)
    hf_assets = tmp_path / "hf-assets"
    hf_assets.mkdir()
    (hf_assets / "tokenizer.json").write_text('{"version": "1.0"}')
    (hf_assets / "tokenizer_config.json").write_text("{}")
    output = tmp_path / "rendered.toml"
    revision = "1" * 40
    renderer = Path(__file__).parents[2] / "tools/render_fa4_training_config.py"
    subprocess.run(
        [
            sys.executable,
            str(renderer),
            "--artifact-manifest",
            str(manifest_path),
            "--output",
            str(output),
            "--dump-folder",
            str(tmp_path / "run"),
            "--hf-assets-path",
            str(hf_assets),
            "--allow-nonhistorical-tokenizer",
            "--dataset-path",
            "cerebras/SlimPajama-627B",
            "--dataset-revision",
            revision,
            "--world-size",
            "1",
            "--global-batch-size",
            "4",
        ],
        cwd=Path(__file__).parents[2],
        check=True,
        capture_output=True,
        text=True,
    )
    receipt_path = Path(str(output) + ".receipt.json")
    receipt = _install_receipt_environment(monkeypatch, receipt_path)

    verified = verify_training_config(output, receipt_path, expected_world_size=1)

    assert receipt["dataset"] == {
        "identifier": "cerebras/SlimPajama-627B",
        "kind": "huggingface_revision",
        "resolved": f"cerebras/SlimPajama-627B@{revision}",
        "revision": revision,
    }
    assert verified["dataset"] == {
        "identifier": "cerebras/SlimPajama-627B",
        "kind": "huggingface_revision",
        "revision": revision,
    }


def _install_receipt_environment(
    monkeypatch: pytest.MonkeyPatch,
    receipt_path: Path,
) -> dict[str, object]:
    receipt = json.loads(receipt_path.read_text())
    for name, value in receipt["required_environment"].items():
        monkeypatch.setenv(name, value)
    return receipt


def test_training_config_verifier_authenticates_complete_launch_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, config, receipt_path = _render_test_training_config(tmp_path)
    receipt = _install_receipt_environment(monkeypatch, receipt_path)

    verified = verify_training_config(
        config,
        receipt_path,
        expected_world_size=1,
    )

    assert verified["config_sha256"] == receipt["config_sha256"]
    assert verified["artifact_manifest"] == str(manifest.resolve())
    assert verified["route"] == "nvfp4_qk_fp8_pv"
    assert verified["trainer_module"] == "torchtitan.experiments.fa4.train"
    assert verified["tokenizer"] == receipt["tokenizer"]


def test_training_config_verifier_rejects_config_byte_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, config, receipt_path = _render_test_training_config(tmp_path)
    _install_receipt_environment(monkeypatch, receipt_path)
    config.write_bytes(config.read_bytes() + b"\n# drift\n")

    with pytest.raises(TrainingConfigVerificationError, match="byte identity"):
        verify_training_config(config, receipt_path, expected_world_size=1)


def test_training_config_verifier_rejects_duplicate_receipt_key(
    tmp_path: Path,
) -> None:
    _, config, receipt_path = _render_test_training_config(tmp_path)
    receipt = receipt_path.read_text()
    receipt_path.write_text(
        receipt.replace(
            '"schema": "fa4_training_config_receipt_v2",',
            '"schema": "fa4_training_config_receipt_v2",\n'
            '  "schema": "fa4_training_config_receipt_v2",',
            1,
        )
    )

    with pytest.raises(TrainingConfigVerificationError, match="duplicate JSON key"):
        verify_training_config(config, receipt_path, expected_world_size=1)


def test_training_config_verifier_rejects_duplicate_manifest_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, config, receipt_path = _render_test_training_config(tmp_path)
    _install_receipt_environment(monkeypatch, receipt_path)
    manifest.write_text('{"schema":"duplicate",' + manifest.read_text().lstrip()[1:])
    receipt = json.loads(receipt_path.read_text())
    receipt["artifact_manifest_sha256"] = hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest()
    receipt_path.write_text(json.dumps(receipt))

    with pytest.raises(TrainingConfigVerificationError, match="duplicate JSON key"):
        verify_training_config(config, receipt_path, expected_world_size=1)


def test_training_config_verifier_invokes_artifact_authentication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, config, receipt_path = _render_test_training_config(tmp_path)
    _install_receipt_environment(monkeypatch, receipt_path)
    manifest_value = json.loads(manifest.read_text())
    Path(manifest_value["artifacts"]["forward"]["path"]).write_bytes(b"tampered")

    with pytest.raises(RuntimeError, match="identity mismatch|SHA256 mismatch"):
        verify_training_config(config, receipt_path, expected_world_size=1)


def test_training_config_verifier_rehashes_local_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, config, receipt_path = _render_test_training_config(tmp_path)
    receipt = _install_receipt_environment(monkeypatch, receipt_path)
    dataset_file = Path(receipt["dataset"]["path"]) / "part-00000.jsonl"
    contents = dataset_file.read_bytes()
    dataset_file.write_bytes(contents.replace(b"fixture", b"changed"))
    assert dataset_file.stat().st_size == len(contents)

    with pytest.raises(DatasetManifestError, match="SHA256 mismatch"):
        verify_training_config(config, receipt_path, expected_world_size=1)


def test_training_config_verifier_rehashes_tokenizer_at_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, config, receipt_path = _render_test_training_config(tmp_path)
    receipt = _install_receipt_environment(monkeypatch, receipt_path)
    tokenizer_file = Path(receipt["tokenizer"]["root"]) / "tokenizer.json"
    contents = tokenizer_file.read_bytes()
    tokenizer_file.write_bytes(contents.replace(b"1.0", b"2.0"))
    assert tokenizer_file.stat().st_size == len(contents)

    with pytest.raises(TrainingConfigVerificationError, match="SHA256 mismatch"):
        verify_training_config(config, receipt_path, expected_world_size=1)


def test_training_config_verifier_matches_tokenizer_root_to_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, config, receipt_path = _render_test_training_config(tmp_path)
    receipt = _install_receipt_environment(monkeypatch, receipt_path)
    original = Path(receipt["tokenizer"]["root"])
    other = tmp_path / "other-hf-assets"
    other.mkdir()
    for record in receipt["tokenizer"]["files"]:
        name = record["path"]
        (other / name).write_bytes((original / name).read_bytes())
    receipt["tokenizer"]["root"] = str(other.resolve())
    receipt_path.write_text(json.dumps(receipt))

    with pytest.raises(TrainingConfigVerificationError, match="config tokenizer root"):
        verify_training_config(config, receipt_path, expected_world_size=1)


def test_training_config_verifier_rejects_environment_or_topology_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, config, receipt_path = _render_test_training_config(tmp_path)
    _install_receipt_environment(monkeypatch, receipt_path)
    monkeypatch.setenv("TORCHTITAN_FSDP_ACCUMULATE_WITHOUT_SYNC", "0")

    with pytest.raises(
        TrainingConfigVerificationError,
        match="TORCHTITAN_FSDP_ACCUMULATE_WITHOUT_SYNC",
    ):
        verify_training_config(config, receipt_path, expected_world_size=1)

    monkeypatch.setenv("TORCHTITAN_FSDP_ACCUMULATE_WITHOUT_SYNC", "1")
    with pytest.raises(TrainingConfigVerificationError, match="launcher world size"):
        verify_training_config(config, receipt_path, expected_world_size=2)


def test_fa4_torchrun_launcher_verifies_then_uses_experiment_module(
    tmp_path: Path,
) -> None:
    _, config, receipt_path = _render_test_training_config(tmp_path)
    receipt = json.loads(receipt_path.read_text())
    trace = tmp_path / "torchrun-arguments.txt"
    fake_torchrun = tmp_path / "torchrun"
    fake_torchrun.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$TRACE_FILE"\n')
    fake_torchrun.chmod(0o755)
    environment = {
        **os.environ,
        **receipt["required_environment"],
        "NNODES": "1",
        "NPROC_PER_NODE": "1",
        "NODE_RANK": "0",
        "CONFIG_FILE": str(config),
        "FA4_RUN_NCCL_PREFLIGHT": "0",
        "PYTHON_BIN": sys.executable,
        "TORCHRUN_BIN": str(fake_torchrun),
        "TRACE_FILE": str(trace),
    }
    launcher = Path(__file__).parents[2] / "scripts/fa4/run_torchrun.sh"

    completed = subprocess.run(
        ["bash", str(launcher)],
        cwd=Path(__file__).parents[2],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    arguments = trace.read_text().splitlines()
    assert '"trainer_module": "torchtitan.experiments.fa4.train"' in completed.stdout
    assert arguments[arguments.index("-m") + 1] == "torchtitan.experiments.fa4.train"
    assert arguments[arguments.index("--job.config-file") + 1] == str(config)

    trace.unlink()
    config.write_bytes(config.read_bytes() + b"\n# post-receipt drift\n")
    rejected = subprocess.run(
        ["bash", str(launcher)],
        cwd=Path(__file__).parents[2],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "byte identity mismatch" in rejected.stderr
    assert not trace.exists(), "torchrun must not execute after verification fails"


def test_cpu_checkpoint_group_is_strictly_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LBT_DCP_SYNC_CPU_PROCESS_GROUP", raising=False)
    new_group = Mock(side_effect=AssertionError("must remain disabled"))
    monkeypatch.setattr(fa4_checkpoint.dist, "new_group", new_group)

    fa4_checkpoint.install_sync_dcp_cpu_process_group(SimpleNamespace())

    new_group.assert_not_called()


def test_cpu_checkpoint_group_routes_only_synchronous_save(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LBT_DCP_SYNC_CPU_PROCESS_GROUP", "1")
    process_group = object()
    monkeypatch.setattr(fa4_checkpoint.dist, "is_initialized", Mock(return_value=True))
    monkeypatch.setattr(
        fa4_checkpoint.dist, "new_group", Mock(return_value=process_group)
    )
    monkeypatch.setattr(fa4_checkpoint.dist, "get_backend", Mock(return_value="gloo"))
    monkeypatch.setattr(
        fa4_checkpoint.dist,
        "get_world_size",
        Mock(side_effect=lambda group=None: 64),
    )
    save_result = object()
    save = Mock(return_value=save_result)
    monkeypatch.setattr(fa4_checkpoint.dcp, "save", save)
    checkpointer = SimpleNamespace(
        async_mode=fa4_checkpoint.AsyncMode.DISABLED,
        ft_manager=None,
        enable_ft_dataloader_checkpoints=False,
        initial_load_in_hf=False,
        last_save_in_hf=False,
    )
    state = {"model": object()}

    fa4_checkpoint.install_sync_dcp_cpu_process_group(checkpointer)
    result = checkpointer.dcp_save(
        state,
        "/checkpoints/step-239",
        fa4_checkpoint.AsyncMode.DISABLED,
    )

    assert result is save_result
    assert checkpointer._lbt_dcp_process_group is process_group
    save.assert_called_once_with(
        state,
        checkpoint_id="/checkpoints/step-239",
        process_group=process_group,
    )


def test_cpu_checkpoint_group_routes_ordinary_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LBT_DCP_SYNC_CPU_PROCESS_GROUP", "1")
    process_group = object()
    monkeypatch.setattr(fa4_checkpoint.dist, "is_initialized", Mock(return_value=True))
    monkeypatch.setattr(
        fa4_checkpoint.dist, "new_group", Mock(return_value=process_group)
    )
    monkeypatch.setattr(fa4_checkpoint.dist, "get_backend", Mock(return_value="gloo"))
    monkeypatch.setattr(
        fa4_checkpoint.dist,
        "get_world_size",
        Mock(side_effect=lambda group=None: 64),
    )
    load = Mock()
    monkeypatch.setattr(fa4_checkpoint.dcp, "load", load)
    model = Mock()
    state = {"model": object(), "optimizer": object()}
    checkpointer = SimpleNamespace(
        async_mode=fa4_checkpoint.AsyncMode.DISABLED,
        ft_manager=None,
        enable_ft_dataloader_checkpoints=False,
        initial_load_in_hf=False,
        last_save_in_hf=False,
        states={fa4_checkpoint.MODEL: model},
    )

    fa4_checkpoint.install_sync_dcp_cpu_process_group(checkpointer)
    result = checkpointer.dcp_load(
        state,
        "/checkpoints/step-239",
        from_hf=False,
        from_quantized=False,
    )

    assert result is None
    load.assert_called_once_with(
        state,
        checkpoint_id="/checkpoints/step-239",
        process_group=process_group,
    )
    model.load_state_dict.assert_called_once_with(state)


@pytest.mark.parametrize(
    ("from_hf", "from_quantized"),
    [(True, False), (False, True), (True, True)],
)
def test_cpu_checkpoint_group_rejects_nonordinary_load(
    monkeypatch: pytest.MonkeyPatch,
    from_hf: bool,
    from_quantized: bool,
) -> None:
    monkeypatch.setenv("LBT_DCP_SYNC_CPU_PROCESS_GROUP", "1")
    process_group = object()
    monkeypatch.setattr(fa4_checkpoint.dist, "is_initialized", Mock(return_value=True))
    monkeypatch.setattr(
        fa4_checkpoint.dist, "new_group", Mock(return_value=process_group)
    )
    monkeypatch.setattr(fa4_checkpoint.dist, "get_backend", Mock(return_value="gloo"))
    monkeypatch.setattr(
        fa4_checkpoint.dist,
        "get_world_size",
        Mock(side_effect=lambda group=None: 64),
    )
    load = Mock(side_effect=AssertionError("rejected load must not enter DCP"))
    monkeypatch.setattr(fa4_checkpoint.dcp, "load", load)
    checkpointer = SimpleNamespace(
        async_mode=fa4_checkpoint.AsyncMode.DISABLED,
        ft_manager=None,
        enable_ft_dataloader_checkpoints=False,
        initial_load_in_hf=False,
        last_save_in_hf=False,
        states={},
    )

    fa4_checkpoint.install_sync_dcp_cpu_process_group(checkpointer)
    with pytest.raises(RuntimeError, match="ordinary DCP load only"):
        checkpointer.dcp_load(
            {},
            "/checkpoints/step-239",
            from_hf=from_hf,
            from_quantized=from_quantized,
        )

    load.assert_not_called()


def test_slimpajama_loader_binds_remote_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision = "0123456789abcdef0123456789abcdef01234567"
    loaded = object()
    load_dataset = Mock(return_value=loaded)
    monkeypatch.setattr(fa4_data, "load_dataset", load_dataset)

    result = fa4_data._load(f"cerebras/SlimPajama-627B@{revision}", split="train")

    assert result is loaded
    load_dataset.assert_called_once_with(
        "cerebras/SlimPajama-627B",
        name="default",
        split="train",
        streaming=True,
        revision=revision,
    )


def test_slimpajama_loader_rejects_floating_remote_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FA4_SLIMPAJAMA_REVISION", raising=False)
    monkeypatch.setattr(
        fa4_data,
        "load_dataset",
        Mock(side_effect=AssertionError("invalid revision must fail first")),
    )

    with pytest.raises(RuntimeError, match="immutable lowercase 40-hex"):
        fa4_data._load("cerebras/SlimPajama-627B", split="train")
