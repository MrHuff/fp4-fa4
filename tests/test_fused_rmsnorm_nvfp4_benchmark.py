import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
HARNESS = (
    ROOT
    / "tk_fa4"
    / "lowp_fa4_bwd"
    / "benchmark_fused_rmsnorm_nvfp4.py"
)


def _load_harness():
    spec = importlib.util.spec_from_file_location(
        "test_benchmark_fused_rmsnorm_nvfp4", HARNESS
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixed_shape_and_fail_closed_defaults() -> None:
    module = _load_harness()
    assert (module.BATCH, module.SEQUENCE, module.HIDDEN) == (16, 4096, 2048)
    assert module.ROWS == 65536
    assert module.DEFAULT_SEEDS == (20260826, 20260827)
    assert module.DEFAULT_MIN_NORMALIZED_COSINE >= 0.999
    assert module.DEFAULT_MAX_NORMALIZED_RELATIVE_L2 <= 2.0e-4
    assert module.DEFAULT_MIN_DX_COSINE >= 0.999
    assert module.DEFAULT_MAX_DX_RELATIVE_L2 <= 2.0e-3
    assert module.DEFAULT_MIN_DGAMMA_COSINE >= 0.999
    assert module.DEFAULT_MAX_DGAMMA_RELATIVE_L2 <= 2.0e-2


def test_extension_authentication_requires_exact_regular_file(tmp_path: Path) -> None:
    module = _load_harness()
    extension = tmp_path / "extension.so"
    extension.write_bytes(b"authenticated test extension")
    expected_sha = hashlib.sha256(extension.read_bytes()).hexdigest()
    identity = module._authenticate_extension_candidate(
        extension,
        expected_sha.upper(),
        extension.stat().st_size,
    )
    assert identity["authenticated"] is True
    assert identity["sha256"] == expected_sha
    assert identity["resolved_path"] == str(extension.resolve())

    with pytest.raises(ValueError, match="identity mismatch"):
        module._authenticate_extension_candidate(
            extension, "0" * 64, extension.stat().st_size
        )
    with pytest.raises(ValueError, match="identity mismatch"):
        module._authenticate_extension_candidate(
            extension, expected_sha, extension.stat().st_size + 1
        )
    with pytest.raises(ValueError, match="64 hex digits"):
        module._authenticate_extension_candidate(
            extension, "not-a-digest", extension.stat().st_size
        )

    symlink = tmp_path / "extension-link.so"
    symlink.symlink_to(extension)
    with pytest.raises(ValueError, match="non-symlink"):
        module._authenticate_extension_candidate(
            symlink, expected_sha, extension.stat().st_size
        )


def test_post_load_authentication_checks_module_file_and_path(
    tmp_path: Path,
) -> None:
    module = _load_harness()
    selected = tmp_path / "selected.so"
    selected.write_bytes(b"selected")
    digest = hashlib.sha256(selected.read_bytes()).hexdigest()
    expected = module._authenticate_extension_candidate(
        selected, digest, selected.stat().st_size
    )
    loaded = module._authenticate_loaded_extension(
        SimpleNamespace(__file__=str(selected)), expected
    )
    assert loaded["post_load_authenticated"] is True
    assert loaded["module"] == module.EXTENSION_MODULE

    different = tmp_path / "different.so"
    different.write_bytes(selected.read_bytes())
    with pytest.raises(RuntimeError, match="path differs"):
        module._authenticate_loaded_extension(
            SimpleNamespace(__file__=str(different)), expected
        )


def test_create_only_json_writer_refuses_every_existing_target(
    tmp_path: Path,
) -> None:
    module = _load_harness()
    output = tmp_path / "result.json"
    document = {"passed": True, "value": 7}
    module._write_new_json(output, document)
    assert json.loads(output.read_text()) == document

    with pytest.raises(FileExistsError):
        module._output_path(str(output))
    with pytest.raises(FileExistsError):
        module._write_new_json(output, {"passed": False})
    assert json.loads(output.read_text()) == document

    dangling = tmp_path / "dangling.json"
    dangling.symlink_to(tmp_path / "absent")
    with pytest.raises(FileExistsError):
        module._output_path(str(dangling))


def test_rotating_orders_and_summary_are_deterministic() -> None:
    module = _load_harness()
    assert module._rotating_orders(("eager", "fused"), 5) == [
        ["eager", "fused"],
        ["fused", "eager"],
        ["eager", "fused"],
        ["fused", "eager"],
        ["eager", "fused"],
    ]
    summary = module._timing_summary((4.0, 1.0, 3.0, 2.0))
    assert summary["unit"] == "microseconds"
    assert summary["samples"] == 4
    assert summary["minimum_us"] == 1.0
    assert summary["p50_us"] == 2.5
    assert summary["mean_us"] == 2.5
    assert summary["maximum_us"] == 4.0
    assert summary["samples_us"] == [4.0, 1.0, 3.0, 2.0]


def test_eager_controls_match_independent_cpu_autograd() -> None:
    import torch

    module = _load_harness()
    tensor = torch.tensor(
        [[1.0, -2.0, 0.5, 3.0], [-0.25, 0.75, 2.0, -1.5]],
        dtype=torch.bfloat16,
    )
    gamma = torch.tensor([0.75, 1.25, 0.5, 1.5], dtype=torch.bfloat16)
    gradient = torch.tensor(
        [[0.5, -1.0, 2.0, 0.25], [1.5, 0.5, -0.75, 1.0]],
        dtype=torch.bfloat16,
    )
    epsilon = 1.0e-5
    packed_inputs = []

    def record_pack(normalized):
        packed_inputs.append(normalized.clone())
        return normalized.clone(), normalized.clone(), normalized[:1, :1].clone()

    eager_forward = module._eager_forward_preparation(
        torch, tensor, gamma, epsilon, record_pack
    )
    values = tensor.float()
    expected_inverse = torch.rsqrt(
        values.square().mean(dim=1, keepdim=True) + epsilon
    )
    expected_normalized = (
        values * expected_inverse * gamma.float().unsqueeze(0)
    ).bfloat16()
    assert torch.equal(eager_forward[3], expected_inverse.reshape(-1))
    assert torch.equal(eager_forward[4], expected_normalized)
    assert torch.equal(packed_inputs[0], expected_normalized)

    actual_dx, actual_dgamma = module._eager_rmsnorm_backward(
        torch, tensor, gamma, eager_forward[3], gradient
    )
    differentiable_input = tensor.float().requires_grad_(True)
    differentiable_gamma = gamma.float().requires_grad_(True)
    normalized = differentiable_input * torch.rsqrt(
        differentiable_input.square().mean(dim=1, keepdim=True) + epsilon
    )
    loss = (
        normalized * differentiable_gamma.unsqueeze(0) * gradient.float()
    ).sum()
    loss.backward()
    expected_dx = differentiable_input.grad.bfloat16()
    expected_dgamma = differentiable_gamma.grad.bfloat16()
    torch.testing.assert_close(actual_dx, expected_dx, rtol=0.0, atol=0.015625)
    torch.testing.assert_close(
        actual_dgamma, expected_dgamma, rtol=0.0, atol=0.015625
    )


def test_correctness_checks_are_fail_closed() -> None:
    module = _load_harness()
    metric = {
        "reference_finite": True,
        "actual_finite": True,
        "cosine": 1.0,
        "relative_l2": 0.0,
        "max_abs": 0.0,
    }
    correctness = {
        "forward": {
            "normalized_vs_eager": dict(metric),
            "inv_rms_vs_eager": dict(metric),
            "pack_on_fused_normalized": {
                name: {"byte_equal": True}
                for name in ("payload", "scales", "global_decode")
            },
        },
        "backward": {
            "dx_same_saved_inv_rms": dict(metric),
            "dgamma_same_saved_inv_rms": dict(metric),
        },
    }
    thresholds = {
        "min_normalized_cosine": 0.9999,
        "max_normalized_relative_l2": 2.0e-4,
        "max_inv_rms_abs": 2.0e-5,
        "min_dx_cosine": 0.9999,
        "max_dx_relative_l2": 2.0e-3,
        "min_dgamma_cosine": 0.999,
        "max_dgamma_relative_l2": 2.0e-2,
    }
    checks = module._comparison_checks(correctness, thresholds)
    assert checks
    assert all(check["passed"] for check in checks)

    correctness["backward"]["dx_same_saved_inv_rms"]["relative_l2"] = 0.1
    checks = module._comparison_checks(correctness, thresholds)
    failed = [check for check in checks if not check["passed"]]
    assert [check["name"] for check in failed] == ["backward_dx_relative_l2"]

    correctness["backward"]["dx_same_saved_inv_rms"]["relative_l2"] = None
    checks = module._comparison_checks(correctness, thresholds)
    failed = [check for check in checks if not check["passed"]]
    assert [check["name"] for check in failed] == ["backward_dx_relative_l2"]


def test_source_keeps_authentication_before_and_after_load_and_exact_controls() -> None:
    source = HARNESS.read_text()
    main = source.split("def main(", 1)[1]
    assert main.index("_authenticate_extension_candidate(") < main.index(
        "import torch"
    )
    assert "_authenticate_loaded_extension(module, candidate_identity)" in main
    assert main.count("_authenticate_loaded_extension(module, candidate_identity)") == 2
    assert "sys.modules.get(EXTENSION_MODULE) is not module" in main
    assert "capability != (10, 0)" in main
    assert "exact_shape_authenticated" in main
    assert "return 0 if passed else 2" in main

    eager_forward = source.split("def _eager_forward_preparation(", 1)[1].split(
        "def _eager_rmsnorm_backward(", 1
    )[0]
    assert "values.square().mean(dim=1, keepdim=True)" in eager_forward
    assert "gamma.float().unsqueeze(0)" in eager_forward
    assert "pack(normalized)" in eager_forward

    eager_backward = source.split("def _eager_rmsnorm_backward(", 1)[1].split(
        "def _comparison_checks(", 1
    )[0]
    assert "projection = (weighted_gradient * values).mean(" in eager_backward
    assert "correction = inverse.square() * projection" in eager_backward
    assert ").sum(dim=0).bfloat16().contiguous()" in eager_backward

    timing = source.split("def _time_interleaved(", 1)[1].split(
        "def _tensor_metrics(", 1
    )[0]
    assert "torch.cuda.Event(enable_timing=True)" in timing
    assert "_rotating_orders(names, warmups)" in timing
    assert "_rotating_orders(names, samples)" in timing


def test_cli_has_no_shape_escape_hatch_and_requires_artifact_identity() -> None:
    module = _load_harness()
    with pytest.raises(SystemExit):
        module._arguments([])
    arguments = module._arguments(
        [
            "--extension-source",
            "/tmp/extension.so",
            "--extension-sha256",
            "0" * 64,
            "--extension-bytes",
            "123",
            "--output",
            "/tmp/result.json",
        ]
    )
    assert arguments.seeds == [20260826, 20260827]
    assert not hasattr(arguments, "batch")
    assert not hasattr(arguments, "sequence")
    assert not hasattr(arguments, "hidden")
