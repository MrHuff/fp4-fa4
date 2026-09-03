from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "tk_fa4"
    / "lowp_fa4_bwd"
    / "run_causal_forward_matrix.py"
)
WORKER = SCRIPT.with_name("benchmark_causal_forward_matrix.py")
SPEC = importlib.util.spec_from_file_location("run_causal_forward_matrix", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MATRIX = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MATRIX
SPEC.loader.exec_module(MATRIX)


def _arguments(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        run_tag="unit",
        artifact_dir=tmp_path / "artifacts",
        projection_extension=tmp_path / "projection.so",
        projection_module="_C_projection",
        gpu_arch="B200",
        num_sm=152,
        mx_quarter_mask=15,
        mx_route_name="d4all",
        hidden=2048,
        seed=7,
        warmups=2,
        samples=5,
        minimum_free_gpu_gib=16.0,
    )


def test_default_matrix_has_requested_d64_shapes() -> None:
    actual = [
        (shape.sequence, shape.q_heads, shape.kv_heads)
        for shape in MATRIX.DEFAULT_SHAPES
    ]
    assert actual == [
        (512, 32, 8),
        (1024, 32, 8),
        (2048, 32, 8),
        (4096, 32, 8),
        (8192, 32, 8),
        (4096, 16, 4),
        (4096, 64, 16),
    ]
    assert all(shape.head_dim == 64 for shape in MATRIX.DEFAULT_SHAPES)


@pytest.mark.parametrize("value", ("4096/32/8", "4096,32,8", "4096x32x8"))
def test_parse_shape_accepts_unambiguous_separators(value: str) -> None:
    assert MATRIX._parse_shape(value) == MATRIX.Shape(4096, 32, 8)


@pytest.mark.parametrize("value", ("4096/32", "4100/32/8", "4096/31/8"))
def test_parse_shape_rejects_invalid_contracts(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        MATRIX._parse_shape(value)


def test_commands_are_serial_and_pin_reconstructed_topologies(tmp_path: Path) -> None:
    args = _arguments(tmp_path)
    shape = MATRIX.Shape(4096, 32, 8)
    commands = MATRIX._commands(
        args,
        shape,
        Path(sys.executable),
        ".cpython-test.so",
        tmp_path / "result.json",
    )
    mx = commands["build_mx"]["argv"]
    fp8 = commands["build_fp8_exact"]["argv"]
    worker = commands["run_worker"]["argv"]

    assert "-j1" in mx
    assert "NVCC_THREADS=1" in mx
    assert "NVCC_SPLIT_COMPILE=1" in mx
    assert "HAO_FIXED_ROUTE_FASTPATH=1" in mx
    assert "HAO_CAUSAL_INTERLEAVED_KV=1" in mx
    assert "HAO_FP4PV_MX_POLICY=causal-accurate" in mx
    assert "HAO_FP4PV_MX_MODE23_NATIVE_DENSITY_OVERRIDE=4" in mx
    assert "HAO_FP4PV_MX_MODE23_NATIVE_QUARTER_MASK_OVERRIDE=15" in mx
    assert "HAO_FP4PV_MX_GLOBAL_ANCHOR32_OVERRIDE=1" in mx
    assert "HAO_FP4PV_MX_GLOBAL_ANCHOR128_OVERRIDE=0" in mx
    assert "HAO_FP4PV_MX_GLOBAL_ANCHOR_BIAS_X8_OVERRIDE=0" in mx
    assert "HAO_FP4PV_MX_GLOBAL_ANCHOR_MARGIN_LOG2_OVERRIDE=64" in mx
    assert "HAO_FP4PV_MX_STORED_SCALE_SHIFT_LOG2_OVERRIDE=16" in mx
    assert "HAO_FP4PV_MX_ANCHOR_AFFINE_HOIST_OVERRIDE=1" in mx
    assert commands["build_mx"]["mx_route_name"] == "d4all"
    assert "d4all" in commands["build_mx"]["module"]
    assert "d4all" in commands["build_mx"]["kernel_symbol_tag"]
    assert fp8[fp8.index("--jobs") + 1] == "1"
    assert fp8[fp8.index("--nvcc-threads") + 1] == "1"
    assert fp8[fp8.index("--nvcc-split-compile") + 1] == "1"
    assert fp8[fp8.index("--probability-policy") + 1] == "exact"
    assert worker[worker.index("--gpu") + 1] == "0"
    assert "--projection-extension" in worker


def test_worker_uses_the_deployed_e4m3_publication_contract() -> None:
    tree = ast.parse(WORKER.read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id
        == "b300_project_qkv_gqa_d64_paired_unified_lowp_e4m3"
    ]
    assert len(calls) == 2

    by_mx_publication = {}
    for call in calls:
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        publish_mxfp4_v = ast.literal_eval(keywords["publish_mxfp4_v"])
        by_mx_publication[publish_mxfp4_v] = keywords
        assert ast.literal_eval(keywords["represented_backward"]) is True
        assert ast.literal_eval(keywords["per_block_qk_scales"]) is True

    assert "experimental_split_v_backward" not in by_mx_publication[False]
    assert ast.literal_eval(
        by_mx_publication[True]["experimental_split_v_backward"]
    ) is True


def test_modules_and_kernel_tags_are_unique_per_shape() -> None:
    first = MATRIX._modules(MATRIX.Shape(4096, 16, 4), "run", 15)
    second = MATRIX._modules(MATRIX.Shape(4096, 64, 16), "run", 15)
    assert len(set(first)) == 3
    assert set(first).isdisjoint(second)
    assert all(value.isidentifier() for value in (*first, *second))


@pytest.mark.parametrize(
    ("mask", "route_name"),
    ((3, "d4q01"), (15, "d4all")),
)
def test_quarter_mask_names_are_in_module_and_command_identity(
    tmp_path: Path,
    mask: int,
    route_name: str,
) -> None:
    args = _arguments(tmp_path)
    args.mx_quarter_mask = mask
    args.mx_route_name = route_name
    modules = MATRIX._modules(MATRIX.Shape(512, 32, 8), "run", mask)
    commands = MATRIX._commands(
        args,
        MATRIX.Shape(512, 32, 8),
        Path(sys.executable),
        ".cpython-test.so",
        tmp_path / "result.json",
    )
    assert all(route_name in module for module in modules)
    assert (
        f"HAO_FP4PV_MX_MODE23_NATIVE_QUARTER_MASK_OVERRIDE={mask}"
        in commands["build_mx"]["argv"]
    )


def test_validate_topology_requires_density4_mask15_and_exact_mode0() -> None:
    shape = MATRIX.Shape(4096, 32, 8)
    shape_fields = {
        "batch": 1,
        "seqlen": 4096,
        "heads": 32,
        "kv_heads": 8,
        "dqk": 64,
        "dvo": 64,
    }
    result = {
        "topology": {
            "nvfp4_qk_mxfp4_pv": {
                **shape_fields,
                **MATRIX._expected_mx_topology(15),
            },
            "nvfp4_qk_fp8_pv_exact": {
                **shape_fields,
                **MATRIX.EXPECTED_FP8_TOPOLOGY,
            },
        }
    }
    MATRIX._validate_topology(result, shape, 15)
    result["topology"]["nvfp4_qk_mxfp4_pv"][
        "mx_mode23_native_density"
    ] = 3
    with pytest.raises(RuntimeError, match="mx_mode23_native_density"):
        MATRIX._validate_topology(result, shape, 15)


def test_validate_topology_distinguishes_d4q01_from_d4all() -> None:
    shape = MATRIX.Shape(512, 32, 8)
    shape_fields = {
        "batch": 1,
        "seqlen": 512,
        "heads": 32,
        "kv_heads": 8,
        "dqk": 64,
        "dvo": 64,
    }
    result = {
        "topology": {
            "nvfp4_qk_mxfp4_pv": {
                **shape_fields,
                **MATRIX._expected_mx_topology(3),
            },
            "nvfp4_qk_fp8_pv_exact": {
                **shape_fields,
                **MATRIX.EXPECTED_FP8_TOPOLOGY,
            },
        }
    }
    MATRIX._validate_topology(result, shape, 3)
    with pytest.raises(RuntimeError, match="native_quarter_mask"):
        MATRIX._validate_topology(result, shape, 15)


def test_summary_records_route_name_mask_and_observed_topology(tmp_path: Path) -> None:
    shape = MATRIX.Shape(512, 32, 8)
    shape_fields = {
        "batch": 1,
        "seqlen": 512,
        "heads": 32,
        "kv_heads": 8,
        "dqk": 64,
        "dvo": 64,
    }
    metric = {"output": {"cosine": 0.99}}
    result = {
        "timing": {
            "providers": {
                "bf16_cute": {"median_us": 10.0},
                "nvfp4_qk_mxfp4_pv": {"median_us": 8.0},
                "nvfp4_qk_fp8_pv_exact": {"median_us": 9.0},
            }
        },
        "speedup": {"mxfp4_pv_over_bf16": 1.25},
        "correctness": {
            "nvfp4_qk_mxfp4_pv_vs_bf16": metric,
            "nvfp4_qk_fp8_pv_exact_vs_bf16": metric,
            "mxfp4_pv_vs_exact_fp8_pv": metric,
        },
        "causal_leakage": {"all_passed": True},
        "topology": {
            "nvfp4_qk_mxfp4_pv": {
                **shape_fields,
                **MATRIX._expected_mx_topology(3),
            },
            "nvfp4_qk_fp8_pv_exact": {
                **shape_fields,
                **MATRIX.EXPECTED_FP8_TOPOLOGY,
            },
        },
    }
    row = MATRIX._summary_row(
        shape,
        result,
        tmp_path / "manifest.json",
        tmp_path / "result.json",
        3,
    )
    assert row["mx_route"] == {
        "name": "d4q01",
        "native_density": 4,
        "native_quarter_mask": 3,
    }
    assert row["topology"]["mx"]["mx_mode23_native_quarter_mask"] == 3


def test_dry_run_argument_defaults_do_not_create_output(tmp_path: Path) -> None:
    result_dir = tmp_path / "results"
    artifact_dir = tmp_path / "artifacts"
    args = MATRIX._parse_args(
        [
            "--run-tag",
            "unit-dry",
            "--result-dir",
            str(result_dir),
            "--artifact-dir",
            str(artifact_dir),
            "--dry-run",
        ]
    )
    assert args.dry_run is True
    assert args.shapes == MATRIX.DEFAULT_SHAPES
    assert args.mx_quarter_mask == 15
    assert args.mx_route_name == "d4all"
    assert not result_dir.exists()
    assert not artifact_dir.exists()


def test_parse_d4q01_route_name_without_changing_default_shapes() -> None:
    args = MATRIX._parse_args(["--dry-run", "--mx-quarter-mask", "3"])
    assert args.mx_quarter_mask == 3
    assert args.mx_route_name == "d4q01"
    assert args.shapes == MATRIX.DEFAULT_SHAPES


def test_dry_run_preserves_selected_venv_python_symlink(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    venv_python = tmp_path / "cute-venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(Path(sys.executable).resolve())
    result_dir = tmp_path / "results"
    artifact_dir = tmp_path / "artifacts"
    monkeypatch.setattr(
        MATRIX,
        "_python_record",
        lambda python: (
            {
                "executable": str(python),
                "ext_suffix": ".cpython-test.so",
                "cutlass_cute_imported": True,
            },
            {},
        ),
    )

    status = MATRIX.main(
        [
            "--dry-run",
            "--shape",
            "512/32/8",
            "--python",
            str(venv_python),
            "--result-dir",
            str(result_dir),
            "--artifact-dir",
            str(artifact_dir),
        ]
    )
    plan = json.loads(capsys.readouterr().out)
    expected = str(venv_python.absolute())
    commands = plan["shapes"][0]["commands"]

    assert status == 0
    assert commands["build_fp8_exact"]["argv"][0] == expected
    assert commands["run_worker"]["argv"][0] == expected
    assert not result_dir.exists()
    assert not artifact_dir.exists()
