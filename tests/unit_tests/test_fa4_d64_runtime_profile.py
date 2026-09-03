from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from torchtitan.experiments.fa4.exact_lowp_attention import (
    _D64_BUILD_PROFILE,
    _D64_PROFILE,
    _ExactRuntimeContext,
    _ExactSettings,
    _NATIVE_TK_D64_V416_MODULE,
    _NATIVE_TK_D64_V416_SOURCE,
    _exact_backward_policy,
    _load_native_tk_d64_backward,
)
from torchtitan.experiments.fa4.job_config import JobConfig


def _write(path: Path, contents: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)
    return path.resolve()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _d64_job_config(tmp_path: Path) -> tuple[JobConfig, dict[str, Path]]:
    source_root = tmp_path / "release"
    runtime = _write(
        source_root / "tk_fa4/lowp_fa4_bwd/benchmark_llama12b_e2e.py",
        b"class _LowpAttentionFunction:\n"
        b"    @staticmethod\n"
        b"    def forward(ctx, x, attention_norm_weight, packed_qkv_weight, "
        b"q_weight, k_weight, v_weight, out_weight, qk_scales, "
        b"forward_workspace, runtime):\n"
        b"        pass\n",
    )
    flash_root = source_root / "flash-attention"
    flash = _write(flash_root / "flash_attn/cute/interface.py", b"# interface\n")
    cutlass_root = tmp_path / "cutlass-dsl"
    _write(cutlass_root / "cutlass/__init__.py", b"__version__ = '4.5.2'\n")
    cutlass_native = _write(
        cutlass_root / "cutlass/_mlir/_mlir_libs/_cutlass_ir_test.so",
        b"cutlass-native",
    )
    forward = _write(tmp_path / "artifacts/forward.so", b"d64-forward")
    publisher = _write(tmp_path / "artifacts/publisher.so", b"publisher")
    v416 = _write(tmp_path / "artifacts/v416.so", b"native-v416")

    config = JobConfig()
    fa4 = config.fa4
    fa4.exact_source_root = str(source_root.resolve())
    fa4.exact_runtime_source_sha256 = _sha256(runtime)
    fa4.exact_flash_attn_root = str(flash_root.resolve())
    fa4.exact_flash_attn_source_sha256 = _sha256(flash)
    fa4.exact_cutlass_dsl_root = str(cutlass_root.resolve())
    fa4.exact_cutlass_dsl_native_sha256 = _sha256(cutlass_native)
    fa4.exact_artifact_profile = _D64_BUILD_PROFILE
    fa4.exact_forward_extension = str(forward)
    fa4.exact_forward_module = "_C_tk_causal_gqa_nvfp4_fp8pv_exact_b16s4096h32kv8d64"
    fa4.exact_forward_sha256 = _sha256(forward)
    fa4.exact_forward_batch_size = 16
    fa4.exact_pv_format = "e4m3_fp8"
    fa4.exact_learned_projection_format = "e4m3"
    fa4.exact_backward_extension = str(publisher)
    fa4.exact_backward_sha256 = _sha256(publisher)
    fa4.exact_native_tk_d64_backward_extension = str(v416)
    fa4.exact_native_tk_d64_backward_module = _NATIVE_TK_D64_V416_MODULE
    fa4.exact_native_tk_d64_backward_sha256 = _sha256(v416)
    fa4.exact_native_tk_d64_backward_bytes = v416.stat().st_size
    return config, {
        "source_root": source_root.resolve(),
        "runtime": runtime,
        "v416": v416,
    }


def test_d64_native_identity_is_all_or_none(tmp_path: Path) -> None:
    config, _ = _d64_job_config(tmp_path)
    config.fa4.exact_native_tk_d64_backward_sha256 = ""

    with pytest.raises(ValueError, match="requires extension, module, SHA256"):
        _ExactSettings.from_job_config(config)


def test_d64_b16_context_requires_matching_profile_and_v416(tmp_path: Path) -> None:
    config, paths = _d64_job_config(tmp_path)
    settings = _ExactSettings.from_job_config(config)
    context = _ExactRuntimeContext(settings, _D64_PROFILE, 16, 16)

    assert context.current_d64_route is True
    assert context.uses_packed_qkv is True
    assert settings.native_tk_d64_backward_extension == paths["v416"]
    assert _exact_backward_policy(_D64_PROFILE, native_d64_route=True) == {
        "backward_exp2_degree": 0,
        "backward_exp2_period": 0,
        "backward_fp8_ds_lift": None,
        "backward_reuse_quantized_p": False,
    }

    with pytest.raises(ValueError, match="requires fa4.exact_artifact_profile"):
        _ExactRuntimeContext(
            replace(settings, artifact_profile=""),
            _D64_PROFILE,
            16,
            16,
        )
    with pytest.raises(ValueError, match="D128 artifact profile"):
        _ExactRuntimeContext(
            replace(settings, artifact_profile="llama8b-d128-b4"),
            _D64_PROFILE,
            16,
            16,
        )


def test_d64_loader_uses_authenticated_runner_and_v416_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, paths = _d64_job_config(tmp_path)
    settings = _ExactSettings.from_job_config(config)
    v416 = paths["v416"]
    stat = v416.stat()
    extension = SimpleNamespace(
        __name__=_NATIVE_TK_D64_V416_MODULE,
        _tk_fa4_loaded_artifact_identity={
            "path": str(v416),
            "sha256": _sha256(v416),
            "bytes": stat.st_size,
            "device": stat.st_dev,
            "inode": stat.st_ino,
            "mtime_ns": stat.st_mtime_ns,
        },
    )
    exact_module = SimpleNamespace(_load_extension=lambda path, module: extension)
    runner_path = _write(
        paths["source_root"] / "tk_fa4/lowp_fa4_bwd/native_tk_d64_backward.py",
        b"# authenticated runner\n",
    )
    runner = SimpleNamespace(
        __file__=str(runner_path),
        _require_extension_metadata=lambda value: {
            "source_identity": _NATIVE_TK_D64_V416_SOURCE
        },
    )

    monkeypatch.setattr(
        "torchtitan.experiments.fa4.exact_lowp_attention.importlib.import_module",
        lambda name: runner,
    )

    assert _load_native_tk_d64_backward(exact_module, settings) is extension


def test_d64_settings_reject_d128_backward_module_substitution(
    tmp_path: Path,
) -> None:
    config, _ = _d64_job_config(tmp_path)
    config.fa4.exact_native_tk_d64_backward_module = (
        "_C_sm100_gqa_tk_v509_d128_nvfp4_score_e4m3_qkv_" "e5m2_dout_b4_s4096"
    )

    with pytest.raises(ValueError, match="authenticated v416 image"):
        _ExactSettings.from_job_config(config)


def test_d64_v416_is_not_aliased_to_cute_control(tmp_path: Path) -> None:
    config, _ = _d64_job_config(tmp_path)
    control = _write(tmp_path / "control.py", b"# distinct CuTe control\n")
    config.fa4.exact_backward_control_source = str(control)
    config.fa4.exact_backward_control_sha256 = _sha256(control)
    config.fa4.exact_backward_control_bytes = control.stat().st_size

    with pytest.raises(ValueError, match="distinct backends"):
        _ExactSettings.from_job_config(config)
