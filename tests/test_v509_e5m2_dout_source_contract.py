from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "tk_fa4" / "native_gqa_tk_bwd"
V508_STEM = (
    "v508_d128_gqa_nvfp4_score_e4m3_gradient_b1_exact_s4096_"
    "experimental_bshd"
)
V509_STEM = (
    "v509_d128_gqa_nvfp4_score_e4m3_qkv_e5m2_dout_b1_exact_s4096_"
    "experimental_bshd"
)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v509_preserves_the_authenticated_v508_sources_verbatim() -> None:
    assert _sha256(NATIVE / f"{V508_STEM}.cu") == (
        "809cad1db2f0cc053ba736372b854abcafa102fc65ed86512803c448feae3ba4"
    )
    assert _sha256(NATIVE / f"{V508_STEM}.cuh") == (
        "209e524ada3f45e0155bc36c7fcee0c60627a18f30c0d804250ff5b0056b64e6"
    )
    assert _sha256(NATIVE / "Makefile.v508") == (
        "2b6272791f14e7b97e36a16926a4a6793d3da004e95753209d070d1fd42f53bc"
    )


def test_v509_changes_only_the_dout_dependent_dense_products() -> None:
    header = _text(NATIVE / f"{V509_STEM}.cuh")

    assert "using dout_tile = st_fp8e5m2<kKeyTile, core::kDepth>;" in header
    assert "kittens::py::tensor_to_gl<globals::dout_gl>(dout)" in header
    assert "mixed::operation::dp_abt" in header
    assert header.count("mixed::operation::dv_ab") == 2
    assert "mixed::kE5m2BFormatMask" not in header
    assert "e5m2_dout_mixed_mma_microgate_20260831.cuh" in header

    # dP no longer dispatches the same-type helper.  The only remaining
    # runtime-accumulate same-type product is dK = dS^T @ Q; dQ also remains
    # on the existing E4M3 path.
    assert "core::issue_score_or_dp(" not in header
    assert header.count("exact::issue_gradient_ab_runtime_accumulate(") == 1
    assert "storage.ds,\n                    storage.q[stage]" in header
    assert header.count("core::issue_gradient_atb(") == 1
    assert "storage.ds,\n                    storage.k" in header


def test_v509_extension_and_makefiles_are_exact_batch_fail_closed() -> None:
    source = _text(NATIVE / f"{V509_STEM}.cu")
    header = _text(NATIVE / f"{V509_STEM}.cuh")
    makefiles = {
        batch: _text(
            NATIVE / ("Makefile.v509" if batch == 1 else f"Makefile.v509_b{batch}")
        )
        for batch in (1, 2, 4)
    }

    assert f'#include "{V509_STEM}.cuh"' in source
    assert "#define TKFA4_V509_EXACT_BATCH 1" in source
    assert "constexpr int kBatch = TKFA4_V509_EXACT_BATCH;" in source
    assert (
        "TKFA4_V509_EXACT_BATCH == 1 || TKFA4_V509_EXACT_BATCH == 2"
        in source
    )
    assert "TKFA4_V509_EXACT_BATCH == 4" in source
    assert "Float8_e5m2" in source
    assert 'check_bshd(dout, "dout_e5m2", kE5m2' in source
    for batch in (1, 2, 4):
        assert (
            "v509_native_nvfp4_score_e4m3_qkv_e5m2_dout_"
            f"b{batch}_s4096_experimental_v1"
        ) in source
        assert f'"fail_closed_B{batch}_S4096_only_no_fallback"' in source
    assert 'result["production_dispatch_connected"] = false;' in source
    assert 'result["batch"] = kBatch;' in source
    assert (
        'result["selected_kernel"] =\n'
        '        "v509::b1_native_nvfp4_score_e4m3_qkv_e5m2_dout_'
        'exact_s4096_kernel";'
    ) in source
    assert 'result["mixed_mma_b_format_mask"] = 1024;' in source
    assert 'result["dout_encode_scale"] = 4.0;' in source
    assert 'result["dout_decode_scale"] = 0.25;' in source
    assert (
        'result["dstat_physical_abi"] = "-4*sum(O*raw_E5M2_dO)";'
        in source
    )

    assert "#define TKFA4_V509_EXACT_BATCH 1" in header
    assert "q_native_scale,\n            q.size(0)," in header
    assert "k_native_scale,\n            q.size(0)," in header
    assert "q_global_scale, q.size(0), kQueryHeads" in header
    assert "k_global_scale, q.size(0), kKvHeads" in header
    assert "q.size(0) == TKFA4_V509_EXACT_BATCH" in header

    for batch, makefile in makefiles.items():
        assert f"SRC := {V509_STEM}.cu" in makefile
        assert (
            "MODULE := _C_sm100_gqa_tk_v509_d128_nvfp4_score_e4m3_qkv_"
            f"e5m2_dout_b{batch}_s4096"
        ) in makefile
        assert "e5m2_dout_mixed_mma_microgate_20260831.cuh" in makefile
        assert "v508_d128" not in makefile
        assert (
            "override EXTRA_NVCCFLAGS += "
            f"-DTKFA4_V509_EXACT_BATCH={batch}"
        ) in makefile
