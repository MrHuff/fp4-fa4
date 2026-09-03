from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tk_fa4.lowp_fa4_bwd.tune_d64_gqa_cute import _load_control


ROOT = Path(__file__).resolve().parents[1]
INTERFACE = ROOT / "tk_fa4" / "interface.py"
RUNTIME = ROOT / "tk_fa4" / "lowp_fa4_bwd" / "benchmark_llama12b_e2e.py"
COMPARE = ROOT / "tk_fa4" / "lowp_fa4_bwd" / "compare_llama12b_mx_fp8pv.py"
TRAIN = ROOT / "tk_fa4" / "lowp_fa4_bwd" / "train_llama12b_real_tokens.py"


def _identity(path: Path) -> tuple[str, int]:
    payload = path.read_bytes()
    return hashlib.sha256(payload).hexdigest(), len(payload)


def test_precomposed_control_is_authenticated_before_import(
    tmp_path: Path,
) -> None:
    source = tmp_path / "verified_control.py"
    source.write_text("CONTROL_SENTINEL = 17\n")
    digest, size = _identity(source)

    module = _load_control(
        fp8_p_storage="tmem",
        direct_tma_dkdv=True,
        detached_fp8_p_tmem=False,
        precomposed_control_source=source,
        precomposed_control_sha256=digest,
        precomposed_control_bytes=size,
    )

    assert module.CONTROL_SENTINEL == 17
    assert module.TK_DIRECT_TMA_DKDV is True
    assert module.TK_FP8_P_STORAGE == "tmem"
    assert module.TK_DETACHED_FP8_P_TMEM is False
    assert Path(module.__file__).resolve() != source.resolve()
    assert Path(module.__file__).read_bytes() == source.read_bytes()
    assert module.TK_PRECOMPOSED_CONTROL_PROVENANCE == {
        "mode": "precomposed",
        "source": {
            "path": str(source.resolve()),
            "bytes": size,
            "sha256": digest,
        },
        "required_constants": {
            "TK_DIRECT_TMA_DKDV": True,
            "TK_FP8_P_STORAGE": "tmem",
            "TK_DETACHED_FP8_P_TMEM": False,
        },
        "required_runtime_policy": {
            "owner_fused_dq_scale": False,
        },
    }


def test_precomposed_control_mismatch_fails_before_execution(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "executed"
    source = tmp_path / "untrusted_control.py"
    source.write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed')\n"
    )
    _, size = _identity(source)

    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        _load_control(
            fp8_p_storage="tmem",
            direct_tma_dkdv=True,
            detached_fp8_p_tmem=False,
            precomposed_control_source=source,
            precomposed_control_sha256="0" * 64,
            precomposed_control_bytes=size,
        )
    assert not marker.exists()


@pytest.mark.parametrize(
    "kwargs",
    (
        {"precomposed_control_sha256": "0" * 64},
        {"precomposed_control_bytes": 1},
    ),
)
def test_precomposed_identity_without_source_is_rejected(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="require a control source"):
        _load_control(**kwargs)


def test_precomposed_control_requires_exact_runtime_constants(
    tmp_path: Path,
) -> None:
    source = tmp_path / "control.py"
    source.write_text("CONTROL_SENTINEL = 1\n")
    digest, size = _identity(source)
    with pytest.raises(ValueError, match="requires direct TMA"):
        _load_control(
            precomposed_control_source=source,
            precomposed_control_sha256=digest,
            precomposed_control_bytes=size,
        )

    with pytest.raises(ValueError, match="non-owner-fused"):
        _load_control(
            fp8_p_storage="tmem",
            direct_tma_dkdv=True,
            owner_fused_dq_scale=True,
            precomposed_control_source=source,
            precomposed_control_sha256=digest,
            precomposed_control_bytes=size,
        )


def test_precomposed_control_rejects_size_before_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "oversized.py"
    source.write_text("CONTROL_SENTINEL = 1\n")

    def fail_if_opened(*args: object, **kwargs: object) -> int:
        raise AssertionError("size mismatch must fail before opening source")

    monkeypatch.setattr(os, "open", fail_if_opened)
    with pytest.raises(RuntimeError, match="size mismatch"):
        _load_control(
            fp8_p_storage="tmem",
            direct_tma_dkdv=True,
            precomposed_control_source=source,
            precomposed_control_sha256="0" * 64,
            precomposed_control_bytes=1,
        )


def test_native_extension_override_rejects_relative_path_in_subprocess() -> None:
    environment = os.environ.copy()
    environment["TK_FA4_LOWP_BWD_EXTENSION_SOURCE"] = "relative.so"
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(ROOT), environment.get("PYTHONPATH", ""))
    )
    completed = subprocess.run(
        [sys.executable, "-c", "import tk_fa4.interface"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "must be an absolute path" in completed.stderr


def test_runtime_and_clis_thread_and_record_precomposed_control() -> None:
    interface_source = INTERFACE.read_text()
    runtime_source = RUNTIME.read_text()
    compare_source = COMPARE.read_text()
    train_source = TRAIN.read_text()
    assert 'LOWP_BWD_EXTENSION_MODULE = "tk_fa4._C_b300_lowp_bwd"' in (
        interface_source
    )
    assert "precomposed_control_source=backward_control_source" in runtime_source
    for source in (compare_source, train_source):
        assert '"--backward-control-source"' in source
        assert '"--backward-control-sha256"' in source
        assert '"--backward-control-bytes"' in source
        assert '"backward_control_provenance"' in source
