from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "tk_fa4"
    / "lowp_fa4_bwd"
    / "benchmark_causal_forward_boundaries.py"
)
SPEC = importlib.util.spec_from_file_location(
    "benchmark_causal_forward_boundaries", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
BOUNDARIES = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BOUNDARIES
SPEC.loader.exec_module(BOUNDARIES)


def _required_arguments(tmp_path: Path) -> list[str]:
    return [
        "--mx-extension",
        str(tmp_path / "mx.so"),
        "--fp8-extension",
        str(tmp_path / "fp8.so"),
        "--projection-extension",
        str(tmp_path / "projection.so"),
    ]


def test_publication_modes_match_current_contract_symbols() -> None:
    modes = BOUNDARIES.PUBLICATION_MODES
    assert [
        (mode.represented_backward, mode.per_block_qk_scales)
        for mode in modes
    ] == [(False, False), (True, False), (True, True)]
    assert [mode.name for mode in modes] == [
        "independent_headwide",
        "represented_headwide",
        "represented_k16",
    ]
    assert modes[0].projection_symbol.endswith("interleaved_causal")
    assert modes[1].projection_symbol.endswith(
        "interleaved_causal_represented_backward"
    )
    assert modes[2].projection_symbol.endswith(
        "interleaved_causal_represented_backward_perblock_qk"
    )
    assert BOUNDARIES.SPLIT_V_MODE.projection_symbol.endswith(
        "interleaved_causal_represented_backward_perblock_qk_split_v_backward"
    )
    cases = BOUNDARIES._all_cases()
    assert len(cases) == 7
    assert cases[-1] == BOUNDARIES.Case(
        BOUNDARIES.MX_ROUTE, BOUNDARIES.SPLIT_V_MODE
    )


def test_parse_defaults_pin_deployed_scaling_and_balanced_samples(
    tmp_path: Path,
) -> None:
    args = BOUNDARIES._parse_args(_required_arguments(tmp_path))
    assert args.projection_weight_scaling == "2d"
    assert args.v_mxfp4_scaling == "1d"
    assert args.samples == 42
    assert args.samples % len(BOUNDARIES._all_cases()) == 0
    assert args.sequence == 4096
    assert args.q_heads == 32
    assert args.kv_heads == 8
    assert args.hidden == 2048


@pytest.mark.parametrize(
    "extra",
    (
        ["--samples", "25"],
        ["--sequence", "4100"],
        ["--q-heads", "31"],
        ["--projection-weight-scaling", "1d"],
        ["--v-mxfp4-scaling", "2d"],
    ),
)
def test_parse_rejects_unbalanced_or_non_deployed_contracts(
    tmp_path: Path,
    extra: list[str],
) -> None:
    with pytest.raises(SystemExit):
        BOUNDARIES._parse_args([*_required_arguments(tmp_path), *extra])


def test_no_split_v_uses_six_case_balance(tmp_path: Path) -> None:
    args = BOUNDARIES._parse_args(
        [
            *_required_arguments(tmp_path),
            "--no-experimental-split-v",
            "--samples",
            "24",
        ]
    )
    assert args.experimental_split_v is False
    assert len(BOUNDARIES._all_cases(include_split_v=False)) == 6


def test_parse_rejects_colliding_module_identities(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        BOUNDARIES._parse_args(
            [
                *_required_arguments(tmp_path),
                "--mx-module",
                "_C_same",
                "--fp8-module",
                "_C_same",
            ]
        )


def test_rotating_order_is_position_balanced() -> None:
    names = [case.key for case in BOUNDARIES._all_cases()]
    orders = BOUNDARIES._rotating_orders(names, 14)
    assert len(orders) == 14
    assert all(set(order) == set(names) for order in orders)
    for name in names:
        positions = [order.index(name) for order in orders]
        assert positions.count(0) == 2
        assert positions.count(6) == 2
        assert sorted(set(positions)) == list(range(7))


def test_timing_summary_has_stable_percentiles() -> None:
    summary = BOUNDARIES._timing_summary(
        [1.0, 2.0, 3.0, 4.0, 5.0], "microseconds"
    )
    assert summary["median_us"] == 3.0
    assert summary["mean_us"] == 3.0
    assert summary["p10_us"] == pytest.approx(1.4)
    assert summary["p90_us"] == pytest.approx(4.6)
    assert summary["samples_us"] == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_operand_identity_compares_storage_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(BOUNDARIES, "torch", torch)
    operand = torch.tensor([[1.0, -2.0]], dtype=torch.float32)
    assert BOUNDARIES._operands_bitwise_equal((operand,), (operand.clone(),))
    changed = operand.clone()
    changed[0, 1] = 3.0
    assert not BOUNDARIES._operands_bitwise_equal((operand,), (changed,))
    assert not BOUNDARIES._operands_bitwise_equal(
        (operand,), (operand.to(torch.float16),)
    )


def test_dry_run_preserves_venv_symlink_modules_and_paths(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    venv_python = tmp_path / "selected-venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(Path(sys.executable).resolve())
    output = tmp_path / "never-created" / "result.json"
    arguments = [
        *_required_arguments(tmp_path),
        "--mx-module",
        "_C_selected_mx",
        "--fp8-module",
        "_C_selected_fp8",
        "--projection-module",
        "_C_selected_projection",
        "--python",
        str(venv_python),
        "--output",
        str(output),
        "--dry-run",
    ]

    status = BOUNDARIES.main(arguments)
    plan = json.loads(capsys.readouterr().out)
    command = plan["worker_command"]

    assert status == 0
    assert plan["touches_cuda"] is False
    assert plan["creates_output"] is False
    assert plan["selected_python"] == str(venv_python.absolute())
    assert command[0] == str(venv_python.absolute())
    assert command[command.index("--mx-module") + 1] == "_C_selected_mx"
    assert command[command.index("--fp8-module") + 1] == "_C_selected_fp8"
    assert (
        command[command.index("--projection-module") + 1]
        == "_C_selected_projection"
    )
    assert command[command.index("--mx-extension") + 1] == str(
        (tmp_path / "mx.so").absolute()
    )
    assert not output.exists()
    assert not output.parent.exists()


def test_dry_run_declares_all_boundaries_and_scaling(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    status = BOUNDARIES.main([*_required_arguments(tmp_path), "--dry-run"])
    plan = json.loads(capsys.readouterr().out)

    assert status == 0
    assert plan["deployed_contract"] == {
        "qkv_projection": "e4m3",
        "qkv_weight_scaling": "one decode per output channel",
        "output_projection_weight_scaling": "2d",
        "mx_v_scaling": "1d",
        "mx_v_scale_2d_argument": False,
        "mx_native_density": 4,
        "mx_native_quarter_mask": 3,
    }
    assert {
        "qkv_rope_publication",
        "attention_store_lse_false",
        "attention_store_lse_true",
        "prepacked_publication_attention_store_lse_false",
        "prepacked_publication_attention_store_lse_true",
        "allocated_publication_attention_store_lse_true",
        "full_one_layer_attention_boundary_preallocated_store_lse_true",
        "full_one_layer_attention_boundary_allocated_store_lse_true",
    }.issubset(plan["timed_stages"])
    assert len(plan["cases"]) == 7
    assert plan["cases"][-1]["route"] == BOUNDARIES.MX_ROUTE
    assert plan["cases"][-1]["experimental_split_v_backward"] is True


def test_dry_run_succeeds_without_site_packages_or_extension_files(
    tmp_path: Path,
) -> None:
    output = tmp_path / "not-created" / "result.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-S",
            str(SCRIPT),
            *_required_arguments(tmp_path),
            "--python",
            sys.executable,
            "--output",
            str(output),
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    plan = json.loads(completed.stdout)
    assert plan["touches_cuda"] is False
    assert plan["creates_output"] is False
    assert not output.parent.exists()


def test_module_import_keeps_cuda_runtime_lazy() -> None:
    assert BOUNDARIES.torch is None
    assert BOUNDARIES.tk_interface is None


def test_topology_validation_distinguishes_mx_and_exact() -> None:
    args = argparse.Namespace(
        sequence=4096,
        q_heads=32,
        kv_heads=8,
        mx_native_quarter_mask=3,
    )
    common = {
        "batch": 1,
        "seqlen": 4096,
        "heads": 32,
        "kv_heads": 8,
        "dqk": 64,
        "dvo": 64,
        "causal": True,
        "qk_format": "nvfp4_e4m3_block16",
        "fixed_p_ceiling": False,
        "score_pack_ceiling": False,
    }
    mx = {
        **common,
        "pv_format": "mxfp4_e8m0_block32",
        "causal_interleaved_kv": True,
        "mx_mode23_native_density": 4,
        "mx_mode23_native_quarter_mask": 3,
    }
    exact = {
        **common,
        "pv_format": "e4m3_fp8",
        "causal_interleaved_kv": False,
        "shiftless_fp8_mode": 0,
    }
    BOUNDARIES._validate_topology("mx", mx, args)
    BOUNDARIES._validate_topology("fp8", exact, args)
    exact["causal_interleaved_kv"] = True
    with pytest.raises(ValueError, match="normal-order"):
        BOUNDARIES._validate_topology("fp8", exact, args)

    mx["mx_mode23_native_quarter_mask"] = 15
    with pytest.raises(ValueError, match="quarter mask"):
        BOUNDARIES._validate_topology("mx", mx, args)


class _FakeBundle:
    def __init__(self) -> None:
        self.v_forward_fp8 = "forward-v"
        self.v_backward_fp8 = None

    def forward_operands(self) -> tuple[str, ...]:
        return ("mx-q", "mx-k", "mx-v")

    def qk_forward_operands(self) -> tuple[str, ...]:
        return ("exact-q", "exact-k")


class _FakeExtension:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    def forward_hao_direct_fp4pv(self, *arguments: object) -> None:
        self.calls.append(("mx", arguments))

    def forward_hao_direct_fp8pv(self, *arguments: object) -> None:
        self.calls.append(("fp8", arguments))


def test_split_v_publication_is_mx_only_and_explicit() -> None:
    captured: dict[str, object] = {}

    def project(*arguments: object, **keywords: object) -> str:
        captured["arguments"] = arguments
        captured["keywords"] = keywords
        return "bundle"

    previous_interface = BOUNDARIES.tk_interface
    BOUNDARIES.tk_interface = SimpleNamespace(
        b300_project_qkv_gqa_d64_paired_unified_lowp_e4m3=project
    )
    try:
        result = BOUNDARIES._publish(
            argparse.Namespace(sequence=4096, q_heads=32, kv_heads=8),
            {"qk_scales": "scales", "paired_rope": "rope"},
            BOUNDARIES.Case(BOUNDARIES.MX_ROUTE, BOUNDARIES.SPLIT_V_MODE),
            input_operand=("input", "input-scale"),
            weight_operand=("weight", "weight-scale"),
        )
    finally:
        BOUNDARIES.tk_interface = previous_interface

    assert result == "bundle"
    keywords = captured["keywords"]
    assert isinstance(keywords, dict)
    assert keywords["publish_mxfp4_v"] is True
    assert keywords["interleave_causal_kv"] is True
    assert keywords["represented_backward"] is True
    assert keywords["per_block_qk_scales"] is True
    assert keywords["experimental_split_v_backward"] is True
    assert keywords["v_mxfp4_scale_2d"] is False


def test_attention_dispatch_exercises_both_store_lse_values() -> None:
    mx_extension = _FakeExtension()
    fp8_extension = _FakeExtension()
    extensions = {
        BOUNDARIES.MX_ROUTE: mx_extension,
        BOUNDARIES.FP8_ROUTE: fp8_extension,
    }
    topologies = {
        BOUNDARIES.MX_ROUTE: {"route": "mx-route"},
        BOUNDARIES.FP8_ROUTE: {"route": "fp8-route"},
    }
    mode = BOUNDARIES.PUBLICATION_MODES[2]
    previous = os.environ.get("TK_FA4_FP4PV_FWD_CONFIG")
    try:
        BOUNDARIES._run_attention(
            BOUNDARIES.Case(BOUNDARIES.MX_ROUTE, mode),
            extensions,
            topologies,
            _FakeBundle(),
            "output",
            "lse",
            store_lse=False,
        )
        BOUNDARIES._run_attention(
            BOUNDARIES.Case(BOUNDARIES.FP8_ROUTE, mode),
            extensions,
            topologies,
            _FakeBundle(),
            "output",
            "lse",
            store_lse=True,
        )
    finally:
        if previous is None:
            os.environ.pop("TK_FA4_FP4PV_FWD_CONFIG", None)
        else:
            os.environ["TK_FA4_FP4PV_FWD_CONFIG"] = previous

    assert mx_extension.calls[0][0] == "mx"
    assert mx_extension.calls[0][1][-1] is False
    assert fp8_extension.calls[0][0] == "fp8"
    assert fp8_extension.calls[0][1][-1] is True
    assert "forward-v" in fp8_extension.calls[0][1]


def test_worker_command_preserves_strict_mode_and_output(tmp_path: Path) -> None:
    args = BOUNDARIES._parse_args(
        [
            *_required_arguments(tmp_path),
            "--strict-modes",
            "--output",
            str(tmp_path / "out.json"),
        ]
    )
    selected = Path(sys.executable).absolute()
    command = BOUNDARIES._worker_command(args, selected)
    assert command[0] == str(selected)
    assert "--strict-modes" in command
    assert command[command.index("--output") + 1] == str(
        (tmp_path / "out.json").absolute()
    )


def test_atomic_result_publication_never_overwrites(tmp_path: Path) -> None:
    destination = tmp_path / "new" / "result.json"
    BOUNDARIES._write_new_atomic(destination, "first\n")
    assert destination.read_text() == "first\n"
    with pytest.raises(FileExistsError):
        BOUNDARIES._write_new_atomic(destination, "second\n")
    assert destination.read_text() == "first\n"
    assert list(destination.parent.glob("*.tmp")) == []
