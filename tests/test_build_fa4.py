from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from tools import build_fa4


def _artifact_schema_module():
    path = build_fa4.ROOT / "torchtitan" / "experiments" / "fa4" / "artifacts.py"
    spec = importlib.util.spec_from_file_location("_fa4_artifacts_schema", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _plan(tmp_path: Path, **overrides: object):
    arguments = {
        "build_root": (tmp_path / "build").resolve(),
        "cuda_home": Path("/opt/cuda-13.0"),
        "python": Path("/opt/fa4-python/bin/python3"),
        "extension_suffix": ".cpython-312-aarch64-linux-gnu.so",
    }
    arguments.update(overrides)
    return build_fa4.build_plan(**arguments)


def _option(command: tuple[str, ...], name: str) -> str:
    index = command.index(name)
    return command[index + 1]


def test_complete_plan_covers_all_kernel_families_and_batches(tmp_path: Path):
    layout, specs = _plan(tmp_path)

    assert len(specs) == 11
    assert {spec.group for spec in specs} == set(build_fa4.D128_TARGET_GROUPS)
    for group in ("fp8-forward", "mx-forward", "v509-backward"):
        assert {spec.batch for spec in specs if spec.group == group} == {1, 2, 4}
    assert all(spec.output.is_relative_to(layout.build_root) for spec in specs)
    assert all("/tmp/tkfa4" not in " ".join(spec.command) for spec in specs)
    assert all(
        "/workspace/mfu-analysis" not in " ".join(spec.command) for spec in specs
    )


@pytest.mark.parametrize("group", ("fp8-forward", "mx-forward"))
def test_forward_commands_pin_the_paper_shape(tmp_path: Path, group: str):
    _, specs = _plan(tmp_path, targets=(group,))

    assert len(specs) == 3
    for spec in specs:
        assert _option(spec.command, "--batch") == str(spec.batch)
        assert _option(spec.command, "--sequence") == "4096"
        assert _option(spec.command, "--q-heads") == "32"
        assert _option(spec.command, "--kv-heads") == "8"
        assert "--output" in spec.command
        assert Path(_option(spec.command, "--output")) == spec.output
        assert _option(spec.command, "--module") == spec.module
    if group == "fp8-forward":
        assert all(
            _option(spec.command, "--probability-policy") == "exact" for spec in specs
        )
    else:
        assert all(
            _option(spec.command, "--anchor-variant") == "anchor32" for spec in specs
        )
        assert all(
            _option(spec.command, "--saved-lse-denom") == "represented"
            for spec in specs
        )


def test_make_commands_override_all_historical_output_defaults(tmp_path: Path):
    layout, specs = _plan(
        tmp_path,
        targets=("mxfp4-quantizer", "projection-publisher", "v509-backward"),
    )
    by_name = {spec.name: spec for spec in specs}

    quantizer = by_name["mxfp4-quantizer"]
    assert f"OUT={layout.quantizer}" in quantizer.command
    assert "NVCC=/opt/cuda-13.0/bin/nvcc" in quantizer.command
    assert "PYTHON=/opt/fa4-python/bin/python3" in quantizer.command

    publisher = by_name["projection-publisher"]
    assert f"OUT={layout.publisher}" in publisher.command

    for batch in (1, 2, 4):
        backward = by_name[f"v509-backward-b{batch}"]
        expected_makefile = "Makefile.v509" if batch == 1 else f"Makefile.v509_b{batch}"
        assert expected_makefile in backward.command
        assert f"BUILD_DIR={backward.output.parent}" in backward.command
        assert backward.module.endswith(f"b{batch}_s4096")


def test_subset_plan_is_deterministic(tmp_path: Path):
    first = _plan(tmp_path, targets=("mx-forward",), batches=(4,))[1]
    second = _plan(tmp_path, targets=("mx-forward",), batches=(4,))[1]

    assert first == second
    assert len(first) == 1
    assert first[0].name == "mx-forward-b4"


def test_build_plan_preserves_virtual_environment_interpreter(tmp_path: Path):
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    interpreter = venv_bin / "python"
    interpreter.symlink_to(Path(sys.executable))

    _, specs = build_fa4.build_plan(
        build_root=(tmp_path / "build").absolute(),
        cuda_home=Path("/usr/local/cuda-13.0"),
        python=interpreter.absolute(),
        batches=(1,),
        targets=("mxfp4-quantizer",),
        extension_suffix=".so",
    )

    assert f"PYTHON={interpreter.absolute()}" in specs[0].command


def _write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def test_operator_only_manifests_use_exact_artifact_identities(tmp_path: Path):
    layout, _ = _plan(tmp_path, batches=(4,))
    _write(layout.publisher, b"publisher")
    backward_module, backward = layout.backward(4)
    _write(backward, b"backward")
    for route in ("nvfp4_qk_fp8_pv", "nvfp4_qk_mxfp4_pv"):
        _, forward = layout.forward(route, 4)
        _write(forward, route.encode())

    written = build_fa4.write_manifests(
        layout,
        batches=(4,),
        cutlass_dsl_root=None,
        operator_only=True,
    )

    assert len(written) == 5
    lowp_path = layout.manifest_root / "nvfp4_qk_fp8_pv_b4_s4096_sm100.json"
    manifest = json.loads(lowp_path.read_text())
    assert manifest["schema"] == build_fa4.ARTIFACT_MANIFEST_SCHEMA
    assert manifest["purpose"] == "operator_only"
    assert manifest["shape"] == {
        "batch": 4,
        "head_dim": 128,
        "kv_heads": 8,
        "q_heads": 32,
        "sequence": 4096,
    }
    assert manifest["profile"] == {
        "fp8_pv_forward": "exact",
        "model_preset": "llama3.1-8b",
        "mxfp4_pv_forward": "maxsafe-anchor32-represented",
        "name": "llama8b-d128-b4",
        "native_backward": "v509",
        "projection_publisher": "b300-lowp-bwd",
        "runtime_contract": "d128-b4-native-v509",
    }
    assert manifest["artifacts"]["native_backward"]["module"] == backward_module
    expected = hashlib.sha256(b"backward").hexdigest()
    assert manifest["artifacts"]["native_backward"]["sha256"] == expected
    assert manifest["sources"]["runtime_source"] is None
    assert manifest["sources"]["flash_interface"] is None
    assert manifest["sources"]["cutlass_dsl"] is None
    release = manifest["sources"]["release"]
    assert release["schema"] == build_fa4.SOURCE_IDENTITY_SCHEMA
    assert release["root"] == str(build_fa4.ROOT)
    assert release["git"]["head"] == build_fa4._run_output(["git", "rev-parse", "HEAD"])
    assert release["git"]["head_tree"] == build_fa4._run_output(
        ["git", "rev-parse", "HEAD^{tree}"]
    )
    assert type(release["git"]["dirty"]) is bool
    assert not {
        "fp4_matmul_commit",
        "training_integration_commit",
    } & set(manifest["sources"])
    assert release["closure"]["file_count"] == len(release["closure"]["files"])
    assert release["closure"]["file_count"] > 100
    closure_paths = {record["path"] for record in release["closure"]["files"]}
    assert "tools/build_fa4.py" in closure_paths
    assert "tk_fa4/native_gqa_tk_bwd/Makefile.v509_b4" in closure_paths
    assert "tk_fa4/lowp_fa4_bwd/benchmark_llama12b_e2e.py" in closure_paths
    assert "torchtitan/experiments/fa4/exact_lowp_attention.py" in closure_paths
    assert "ThunderKittens/kernels/common.mk" in closure_paths
    parsed = _artifact_schema_module().load_artifact_manifest(
        lowp_path, require_training=False
    )
    assert parsed.route == "nvfp4_qk_fp8_pv"
    assert parsed.batch == 4

    e4_path = layout.manifest_root / "e4m3_proj_nvfp4_qk_mxfp4_pv_b4_s4096_sm100.json"
    e4_manifest = json.loads(e4_path.read_text())
    assert e4_manifest["route"] == {
        "name": "e4m3_proj_nvfp4_qk_mxfp4_pv",
        "pv_format": "mxfp4_e8m0_block32",
        "learned_projection_format": "e4m3",
    }
    parsed_e4 = _artifact_schema_module().load_artifact_manifest(
        e4_path, require_training=False
    )
    assert parsed_e4.learned_projection_format == "e4m3"


def test_d64_b16_profile_builds_anchored_forwards_publisher_and_v416(
    tmp_path: Path,
):
    layout, specs = _plan(
        tmp_path,
        profile=build_fa4.D64_BUILD_PROFILE,
    )

    assert layout.profile == "llama1p2b-d64-b16"
    assert len(specs) == 4
    assert {spec.group for spec in specs} == set(build_fa4.D64_TARGET_GROUPS)
    assert {spec.batch for spec in specs if spec.batch is not None} == {16}
    by_group = {spec.group: spec for spec in specs}
    fp8 = by_group["fp8-forward"]
    assert _option(fp8.command, "--head-dim") == "64"
    assert fp8.module == ("_C_tk_causal_gqa_nvfp4_fp8pv_exact_b16s4096h32kv8d64")
    mx = by_group["mx-forward"]
    assert mx.command[1].endswith("build_causal_gqa_mxfp4pv_forward.py")
    assert _option(mx.command, "--mx-policy") == "d4q01"
    assert _option(mx.command, "--variant") == "anchored"
    assert mx.module == "_C_cfwd_mx_d4q01_b16s4096h32kv8d64"
    backward = by_group["v416-backward"]
    assert "Makefile.v416" in backward.command
    assert backward.module == ("_C_sm100_gqa_tk_v416_d64_e4m3_production_bshd_dq_first")


def test_build_profiles_reject_cross_shape_targets_and_batches(tmp_path: Path):
    with pytest.raises(build_fa4.BuildError, match="does not provide"):
        _plan(
            tmp_path,
            profile=build_fa4.D64_BUILD_PROFILE,
            targets=("v509-backward",),
        )
    with pytest.raises(build_fa4.BuildError, match=r"\{16\}"):
        _plan(
            tmp_path,
            profile=build_fa4.D64_BUILD_PROFILE,
            batches=(4,),
        )


def test_d64_manifest_authenticates_v416_and_rejects_d128_substitution(
    tmp_path: Path,
):
    layout, _ = _plan(
        tmp_path,
        profile=build_fa4.D64_BUILD_PROFILE,
    )
    _write(layout.publisher, b"publisher")
    module, backward = layout.native_backward(16)
    _write(backward, b"v416")
    for route in (
        "e4m3_proj_nvfp4_qk_fp8_pv",
        "e4m3_proj_nvfp4_qk_mxfp4_pv",
    ):
        _, forward = layout.forward(route, 16)
        _write(forward, route.encode())

    written = build_fa4.write_manifests(
        layout,
        batches=(16,),
        cutlass_dsl_root=None,
        operator_only=True,
    )
    assert len(written) == 3
    path = layout.manifest_root / "e4m3_proj_nvfp4_qk_fp8_pv_b16_s4096_sm100.json"
    raw = json.loads(path.read_text())
    assert raw["profile"] == {
        "fp8_pv_forward": "exact",
        "model_preset": "llama3.2-1b",
        "mxfp4_pv_forward": "d4q01-anchored",
        "name": "llama1p2b-d64-b16",
        "native_backward": "v416",
        "projection_publisher": "b300-lowp-bwd",
        "runtime_contract": "d64-b16-native-v416",
    }
    assert raw["artifacts"]["native_backward"]["module"] == module
    parsed = _artifact_schema_module().load_artifact_manifest(
        path,
        require_training=False,
    )
    assert parsed.profile == "llama1p2b-d64-b16"
    assert parsed.v416_backward == parsed.native_backward
    assert parsed.v509_backward is None

    d128_module = (
        "_C_sm100_gqa_tk_v509_d128_nvfp4_score_e4m3_qkv_" "e5m2_dout_b16_s4096"
    )
    raw["artifacts"]["native_backward"]["module"] = d128_module
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="v416 backward module"):
        _artifact_schema_module().load_artifact_manifest(
            path,
            require_training=False,
        )

    raw["artifacts"]["native_backward"]["module"] = module
    raw["artifacts"]["forward"][
        "module"
    ] = "_C_tk_causal_gqa_nvfp4_fp8pv_exact_b4s4096h32kv8d128"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="forward module does not match profile"):
        _artifact_schema_module().load_artifact_manifest(
            path,
            require_training=False,
        )


def test_paths_must_be_explicit_and_absolute(tmp_path: Path):
    with pytest.raises(build_fa4.BuildError, match="build root must be absolute"):
        build_fa4.build_plan(
            build_root=Path("relative-build"),
            cuda_home=Path("/opt/cuda-13.0"),
            python=Path("/opt/python/bin/python3"),
        )
    with pytest.raises(build_fa4.BuildError, match="CUDA root must be absolute"):
        build_fa4.build_plan(
            build_root=(tmp_path / "build").resolve(),
            cuda_home=Path("relative-cuda"),
            python=Path("/opt/python/bin/python3"),
        )


def test_clean_build_root_rejects_stale_files(tmp_path: Path):
    build_root = tmp_path / "build"
    build_root.mkdir()
    build_fa4.require_clean_build_root(build_root)
    (build_root / "stale.so").write_bytes(b"old")

    with pytest.raises(build_fa4.BuildError, match="new or empty"):
        build_fa4.require_clean_build_root(build_root)


def test_cutlass_dsl_closure_covers_python_and_native_runtime(tmp_path: Path):
    root = tmp_path / "cutlass-dsl"
    package = root / "cutlass"
    _write(package / "__init__.py", b"__version__ = '4.5.2'\n")
    helper = package / "runtime.py"
    _write(helper, b"VALUE = 1\n")
    _write(package / "_mlir/_mlir_libs/_cutlass_ir_test.so", b"native")

    first = build_fa4._cutlass_dsl_source_closure(root)
    helper.write_bytes(b"VALUE = 2\n")
    second = build_fa4._cutlass_dsl_source_closure(root)

    assert first["file_count"] == 3
    assert {record["path"] for record in first["files"]} == {
        "cutlass/__init__.py",
        "cutlass/runtime.py",
        "cutlass/_mlir/_mlir_libs/_cutlass_ir_test.so",
    }
    assert first["sha256"] != second["sha256"]


def test_source_closure_rejects_escaping_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "release"
    source = root / "source"
    source.mkdir(parents=True)
    outside = tmp_path / "outside.cuh"
    outside.write_text("not release source\n")
    (source / "escape.cuh").symlink_to(outside)
    monkeypatch.setattr(build_fa4, "SOURCE_CLOSURE_TREES", ("source",))
    monkeypatch.setattr(build_fa4, "SOURCE_CLOSURE_FILES", ())

    with pytest.raises(build_fa4.BuildError, match="escapes its root"):
        build_fa4._source_closure_paths(root.resolve())
