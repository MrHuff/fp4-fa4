from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DRIVER = (
    ROOT
    / "results/llama8b_nvfp4_qk_backward_reconstruction_20260831"
    / "validate_v508_native_score_hybrid_step5000_v3.py"
)


def test_v508_diagnostic_has_no_original_machine_path() -> None:
    source = DRIVER.read_text(encoding="utf-8")

    assert "/workspace/" not in source
    assert "--capture-root" in source
    assert "--v508-binary" in source
    assert "RELEASE_ROOT = Path(__file__).resolve().parents[2]" in source


def test_v508_diagnostic_help_exposes_explicit_external_inputs() -> None:
    completed = subprocess.run(
        [sys.executable, str(DRIVER), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--capture-root" in completed.stdout
    assert "--v508-binary" in completed.stdout
    assert "--repo-root" in completed.stdout
