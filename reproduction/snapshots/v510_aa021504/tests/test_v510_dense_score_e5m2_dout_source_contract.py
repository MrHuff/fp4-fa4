from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "tk_fa4" / "native_gqa_tk_bwd"
V510_STEM = (
    "v510_d128_gqa_e4m3_score_qkv_e5m2_dout_b1_exact_s4096_"
    "experimental_bshd"
)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v510_preserves_v501_and_v509_sources_verbatim() -> None:
    assert _sha256(
        NATIVE / "v501_d128_gqa_e4m3_unified_best_route_production_bshd.cu"
    ) == "1083abf27af1f392ecb436750787498942bca4b6200dbd8fe0d909a78ca8caa2"
    assert _sha256(
        NATIVE / "v501_d128_gqa_e4m3_unified_best_route_production_bshd.cuh"
    ) == "3c940a227d5338445cdc114ab9e14d940f68d7fdb59cd13e38e2396dacc2e501"
    assert _sha256(
        NATIVE
        / (
            "v509_d128_gqa_nvfp4_score_e4m3_qkv_e5m2_dout_b1_exact_"
            "s4096_experimental_bshd.cu"
        )
    ) == "4b9b52113d29b55ab72986c0a19983b1dff972c903ecfa038c877e8a49d24295"
    assert _sha256(
        NATIVE
        / (
            "v509_d128_gqa_nvfp4_score_e4m3_qkv_e5m2_dout_b1_exact_"
            "s4096_experimental_bshd.cuh"
        )
    ) == "1243725218cdda7a461826e6e0200b23c012948e0fe32d6d991bd3d060414432"


def test_v510_keeps_dense_e4_score_and_changes_only_dout_products() -> None:
    header = _text(NATIVE / f"{V510_STEM}.cuh")

    assert "using dout_tile = st_fp8e5m2<kKeyTile, core::kDepth>;" in header
    assert "kittens::py::tensor_to_gl<globals::dout_gl>(dout)" in header
    assert header.count("core::issue_score_or_dp(") == 2
    assert "score_tmem,\n                storage.k," in header
    assert "mixed::operation::dp_abt" in header
    assert header.count("mixed::operation::dv_ab") == 2
    assert header.count("issue_dp_e4m3_e5m2(") == 3
    assert header.count("issue_dv_e4m3_e5m2_runtime_accumulate(") == 2

    # dK = dS^T @ Q and dQ = dS @ K retain the dense E4M3 route.
    assert header.count("exact::issue_gradient_ab_runtime_accumulate(") == 1
    assert "storage.ds,\n                    storage.q[stage]" in header
    assert header.count("core::issue_gradient_atb(") == 1
    assert "storage.ds,\n                    storage.k" in header


def test_v510_preserves_v488_scaling_and_is_shape_fail_closed() -> None:
    header = _text(NATIVE / f"{V510_STEM}.cuh")

    assert "const float beta = softmax_scale / 16.0f;" in header
    assert "beta * kLog2E" in header
    assert '"v510 is fail-closed to B1/S4096"' in header
    assert "fallback::launch(" not in header
    assert "namespace fallback" not in header
    assert "static_assert(sizeof(shared_storage) == 162 * 1024);" in header
    assert "e5m2_dout_mixed_mma_microgate_20260831.cuh" in header


def test_v510_extension_metadata_and_makefile_are_unique() -> None:
    source = _text(NATIVE / f"{V510_STEM}.cu")
    makefile = _text(NATIVE / "Makefile.v510")

    assert f'#include "{V510_STEM}.cuh"' in source
    assert "Float8_e5m2" in source
    assert 'check_bshd(\n        dout,\n        "dout_e5m2"' in source
    assert (
        "v510_dense_e4m3_score_qkv_e5m2_dout_b1_s4096_"
        "experimental_v1"
    ) in source
    assert 'result["production_dispatch_connected"] = false;' in source
    assert 'result["mixed_mma_b_format_mask"] = 1024;' in source
    assert 'result["score_internal_beta_divisor"] = 16.0;' in source
    assert 'result["dout_encode_scale"] = 4.0;' in source
    assert 'result["dout_decode_scale"] = 0.25;' in source
    assert (
        'result["dstat_physical_abi"] = "-4*sum(O*raw_E5M2_dO)";'
        in source
    )

    assert f"SRC := {V510_STEM}.cu" in makefile
    assert "_C_sm100_gqa_tk_v510_" in makefile
    assert "e5m2_dout_mixed_mma_microgate_20260831.cuh" in makefile
    assert "v509_d128" not in makefile


def test_v510_validator_is_authenticated_and_natural_capture_only() -> None:
    validator = _text(NATIVE / "validate_v510_dense_score_e5m2_dout.py")

    for field in (
        '"q_fp8"',
        '"k_fp8"',
        '"v_fp8"',
        '"dout_e5m2"',
        '"lstat"',
        '"dstat"',
        '"dq_reference"',
        '"dk_reference"',
        '"dv_reference"',
    ):
        assert field in validator
    assert "_authenticate_file(" in validator
    assert "_require_extension_metadata(extension)" in validator
    assert "actual.float().mul(0.25)" in validator
    assert "exact_zero_dout" in validator
    assert "torch.cuda" not in validator.split("def main", 1)[0]
