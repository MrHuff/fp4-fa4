from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FORWARD_SOURCE = REPO_ROOT / "tk_fa4" / "fp4_fa4_fwd"
FORWARD_PATCH = (
    REPO_ROOT
    / "tk_fa4"
    / "lowp_fa4_bwd"
    / "causal_gqa_fp8pv_forward.patch"
)


def test_exact_fp8_patch_applies_to_retained_forward_source() -> None:
    subprocess.run(
        (
            "patch",
            "--batch",
            "--silent",
            "--dry-run",
            "--directory",
            str(FORWARD_SOURCE),
            "--input",
            str(FORWARD_PATCH),
        ),
        check=True,
    )


def test_exact_fp8_patch_declares_optional_interleaved_mapping() -> None:
    patch = FORWARD_PATCH.read_text()
    assert "#ifndef TK_HAO_DIRECT_CAUSAL_INTERLEAVED_KV" in patch
    assert "#define TK_HAO_DIRECT_CAUSAL_INTERLEAVED_KV 0" in patch
    assert (
        "HAO_DIRECT_CAUSAL && TK_HAO_DIRECT_CAUSAL_INTERLEAVED_KV" in patch
    )


def test_exact_fp8_host_explicitly_exports_contiguous_causal_mapping() -> None:
    host = (FORWARD_SOURCE / "hao_direct_host.inc").read_text()
    assert 'out["causal_interleaved_kv"] = false;' in host
