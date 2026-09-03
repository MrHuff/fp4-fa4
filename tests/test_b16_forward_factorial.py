from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tk_fa4" / "lowp_fa4_bwd" / "benchmark_b16_forward_factorial.py"
SPEC = importlib.util.spec_from_file_location("benchmark_b16_forward_factorial", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
FACTORIAL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = FACTORIAL
SPEC.loader.exec_module(FACTORIAL)


def _required_arguments(tmp_path: Path) -> list[str]:
    return [
        "--mx-extension",
        str(tmp_path / "mx.so"),
        "--fp8-extension",
        str(tmp_path / "fp8.so"),
        "--projection-extension",
        str(tmp_path / "projection.so"),
        "--output",
        str(tmp_path / "result.json"),
    ]


def test_factorial_is_fixed_to_saturated_llama12b_shape() -> None:
    assert FACTORIAL.torch is None
    assert FACTORIAL.tk_interface is None
    assert (
        FACTORIAL.BATCH,
        FACTORIAL.SEQUENCE,
        FACTORIAL.HIDDEN,
        FACTORIAL.Q_HEADS,
        FACTORIAL.KV_HEADS,
        FACTORIAL.HEAD_DIM,
        FACTORIAL.QKV_WIDTH,
    ) == (16, 4096, 2048, 32, 8, 64, 3072)
    assert [case.key for case in FACTORIAL.CASES] == [
        "e4m3_qkv__fp8_pv",
        "e4m3_qkv__mx_pv",
        "nvfp4_qkv__fp8_pv",
        "nvfp4_qkv__mx_pv",
    ]
    assert FACTORIAL.STAGE_NAMES == (
        "operand_preparation",
        "projection_publication",
        "attention",
        "prepared_projection_attention",
        "full_combined",
    )


def test_parse_defaults_require_balanced_rotating_samples(tmp_path: Path) -> None:
    args = FACTORIAL._parse_args(_required_arguments(tmp_path))
    assert args.warmups == 4
    assert args.samples == 40
    assert args.samples % len(FACTORIAL.CASES) == 0
    assert args.minimum_free_gib == 24.0
    assert args.input_std == 1.0
    assert args.weight_std == 0.02
    assert args.q_quant_scale == 2.25
    assert args.k_quant_scale == 2.0
    assert args.minimum_bf16_output_cosine == 0.95
    assert args.maximum_bf16_output_relative_l2 == 0.35

    with pytest.raises(SystemExit):
        FACTORIAL._parse_args([*_required_arguments(tmp_path), "--samples", "6"])
    with pytest.raises(SystemExit):
        FACTORIAL._parse_args(
            [
                *_required_arguments(tmp_path),
                "--mx-module",
                "_C_same",
                "--fp8-module",
                "_C_same",
            ]
        )


def test_rotating_order_balances_every_case_position() -> None:
    names = [case.key for case in FACTORIAL.CASES]
    orders = FACTORIAL._rotating_orders(names, 8)
    assert len(orders) == 8
    assert all(set(order) == set(names) for order in orders)
    for name in names:
        positions = [order.index(name) for order in orders]
        assert positions.count(0) == 2
        assert positions.count(1) == 2
        assert positions.count(2) == 2
        assert positions.count(3) == 2


def test_timing_summaries_use_explicit_units() -> None:
    microseconds = FACTORIAL._timing_summary([1.0, 2.0, 3.0, 4.0], unit="microseconds")
    nanoseconds = FACTORIAL._timing_summary(
        [10.0, 20.0, 30.0, 40.0], unit="nanoseconds"
    )
    assert microseconds["median_us"] == 2.5
    assert microseconds["p10_us"] == pytest.approx(1.3)
    assert nanoseconds["median_ns"] == 25.0
    assert nanoseconds["p90_ns"] == pytest.approx(37.0)
    with pytest.raises(ValueError, match="unsupported timing unit"):
        FACTORIAL._timing_summary([1.0], unit="milliseconds")


def test_allocation_contract_is_caller_owned_and_trace_honest() -> None:
    contract = FACTORIAL._allocation_contract()
    stages = contract["stages"]
    assert contract["allocator_tracing_performed"] is False
    assert contract["transient_cuda_allocation_freedom_proven"] is False
    assert stages["operand_preparation"]["caller_owned_output_api"] is False
    assert stages["projection_publication"]["caller_owned_output_api"] is True
    assert stages["attention"]["caller_owned_output_api"] is True
    assert stages["prepared_projection_attention"]["caller_owned_output_api"] is True
    assert stages["full_combined"]["caller_owned_output_api"] is False
    assert "functional APIs allocate" in stages["operand_preparation"]["reason"]


def test_dry_run_is_cuda_lazy_and_preserves_selected_venv(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    venv_python = tmp_path / "selected-venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(Path(sys.executable).resolve())
    output = tmp_path / "never-created" / "result.json"
    args = [
        *_required_arguments(tmp_path)[:-1],
        str(output),
        "--python",
        str(venv_python),
        "--mx-module",
        "_C_selected_mx",
        "--fp8-module",
        "_C_selected_fp8",
        "--dry-run",
    ]

    status = FACTORIAL.main(args)
    plan = json.loads(capsys.readouterr().out)

    assert status == 0
    assert plan["touches_cuda"] is False
    assert plan["imports_torch"] is False
    assert plan["creates_output"] is False
    assert plan["selected_python"] == str(venv_python.absolute())
    assert plan["worker_command"][0] == str(venv_python.absolute())
    projection_module_index = plan["worker_command"].index("--projection-module")
    assert plan["worker_command"][projection_module_index + 1] == "_C_b300_lowp_bwd"
    assert plan["shape"]["batch"] == 16
    assert plan["shape"]["sequence"] == 4096
    assert len(plan["cases"]) == 4
    assert plan["timed_stages"] == list(FACTORIAL.STAGE_NAMES)
    assert plan["correctness_policy"] == {
        "reference": "untimed_bf16_projection_causal_sdpa",
        "minimum_bf16_output_cosine": 0.95,
        "maximum_bf16_output_relative_l2": 0.35,
        "fail_closed": True,
    }
    assert not output.exists()
    assert not output.parent.exists()


def test_direct_dry_run_succeeds_without_site_packages_or_extensions(
    tmp_path: Path,
) -> None:
    output = tmp_path / "not-created" / "result.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-S",
            str(SCRIPT),
            *_required_arguments(tmp_path)[:-1],
            str(output),
            "--python",
            sys.executable,
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    plan = json.loads(completed.stdout)
    assert plan["touches_cuda"] is False
    assert plan["imports_torch"] is False
    assert not output.exists()
    assert not output.parent.exists()


def test_source_uses_bound_out_abis_and_preallocated_attention_buffers() -> None:
    source = SCRIPT.read_text()
    assert "b300_bind_qkv_gqa_d64_paired_unified_lowp_e4m3_projection" in source
    assert "b300_bind_qkv_gqa_d64_paired_unified_lowp_nvfp4_projection" in source
    assert "forward_workspace=workspace" in source
    assert "projector.forward_workspace_abi_validated" in source
    assert "projector.validated_forward_workspace_count != 1" in source
    assert "first_use_authentication_excluded_from_timing" in source
    assert "output_shape = (BATCH, SEQUENCE, Q_HEADS, HEAD_DIM)" in source
    assert "lse_shape = (BATCH, Q_HEADS, 1, SEQUENCE)" in source
    assert "forward_hao_direct_fp4pv" in source
    assert "forward_hao_direct_fp8pv" in source
    assert "_run_bf16_causal_sdpa_reference(state)" in source
    assert "scaled_dot_product_attention(" in source
    assert '"projection_publishes_training_backward_operands": True' in source
    assert '"backward_kernel_executed": False' in source
    assert '"forward_only": True' not in source
    assert "loss.backward" not in source
    assert "optimizer.step" not in source


def test_bf16_verdict_is_fail_closed_on_each_threshold() -> None:
    passing = FACTORIAL._bf16_output_verdict(
        {"finite": True, "cosine": 0.96, "relative_l2": 0.20},
        minimum_cosine=0.95,
        maximum_relative_l2=0.35,
    )
    assert passing["passed"] is True

    nonfinite = FACTORIAL._bf16_output_verdict(
        {"finite": False, "cosine": 0.99, "relative_l2": 0.01},
        minimum_cosine=0.95,
        maximum_relative_l2=0.35,
    )
    low_cosine = FACTORIAL._bf16_output_verdict(
        {"finite": True, "cosine": 0.94, "relative_l2": 0.20},
        minimum_cosine=0.95,
        maximum_relative_l2=0.35,
    )
    high_l2 = FACTORIAL._bf16_output_verdict(
        {"finite": True, "cosine": 0.99, "relative_l2": 0.36},
        minimum_cosine=0.95,
        maximum_relative_l2=0.35,
    )
    assert nonfinite["passed"] is False
    assert low_cosine["passed"] is False
    assert high_l2["passed"] is False


def test_loaded_artifact_is_reauthenticated_after_import(tmp_path: Path) -> None:
    extension = tmp_path / "candidate.so"
    extension.write_bytes(b"authenticated bytes")
    provenance = FACTORIAL._regular_file_provenance(extension)
    module = SimpleNamespace(__file__=str(extension))

    receipt = FACTORIAL._authenticate_loaded_extension(
        "candidate",
        module,
        provenance,
    )
    assert receipt["path_and_sha256_match_preload_receipt"] is True
    assert receipt["post_load_sha256"] == provenance["sha256"]

    extension.write_bytes(b"different bytes")
    with pytest.raises(RuntimeError, match="bytes changed"):
        FACTORIAL._authenticate_loaded_extension(
            "candidate",
            module,
            provenance,
        )


def test_topology_gate_distinguishes_exact_fp8_and_mx() -> None:
    common = {
        "batch": 16,
        "seqlen": 4096,
        "heads": 32,
        "kv_heads": 8,
        "dqk": 64,
        "dvo": 64,
        "causal": True,
        "qk_format": "nvfp4_e4m3_block16",
        "fixed_route_fastpath": True,
        "route_env_guard_per_launch": False,
        "fixed_p_ceiling": False,
        "score_pack_ceiling": False,
        "valid": 1,
    }
    mx = {
        **common,
        "route": "real_fwd_tk_hao_direct_nvfp4_mxfp4pv",
        "pv_format": "mxfp4_e8m0_block32",
        "causal_interleaved_kv": True,
        "mx_mode23_native_density": 4,
        "mx_mode23_native_quarter_mask": 3,
    }
    fp8 = {
        **common,
        "route": "real_fwd_tk_hao_direct_causal_gqa_nvfp4_fp8pv",
        "pv_format": "e4m3_fp8",
        "causal_interleaved_kv": False,
        "shiftless_fp8_mode": 0,
    }
    FACTORIAL._validate_topology("mx", mx, require_runtime_valid=True)
    FACTORIAL._validate_topology("fp8", fp8, require_runtime_valid=True)

    invalid = dict(mx)
    invalid["batch"] = 1
    with pytest.raises(ValueError, match="expected 16"):
        FACTORIAL._validate_topology("mx", invalid, require_runtime_valid=True)
    invalid = dict(fp8)
    invalid["shiftless_fp8_mode"] = 1
    with pytest.raises(ValueError, match="shiftless_fp8_mode"):
        FACTORIAL._validate_topology("fp8", invalid, require_runtime_valid=True)
