from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKWARD = (
    ROOT / "tk_fa4/lowp_fa4_bwd/b300_bwd_cute16_kernel_candidate.cuh"
)
EPILOGUE = ROOT / "tk_fa4/lowp_fa4_bwd/projection_fp4_epilogue.cuh"
EXTENSION = ROOT / "tk_fa4/lowp_fa4_bwd/lowp_fa4_bwd.cu"


def test_native_mxfp4_dp_uses_its_own_reconstruction_correction() -> None:
    source = BACKWARD.read_text(encoding="utf-8")

    corrected_call = """cta2_role_split_make_native_x32_ds_stage_fp8<
                                ReuseFp8PForDs,
                                UseFp8Dp,
                                UseMxFp4Dp
                            >"""
    assert source.count(corrected_call) == 2
    assert """cta2_role_split_make_native_x32_ds_stage_fp8<
                                ReuseFp8PForDs,
                                UseFp8Dp || UseMxFp4Dp
                            >""" not in source
    assert "constexpr float kMxFp4Correction = 1.0f / 36.0f;" in source


def test_backward_v_and_dout_share_standard_width_six_mxfp4() -> None:
    epilogue = EPILOGUE.read_text(encoding="utf-8")
    extension = EXTENSION.read_text(encoding="utf-8")

    assert "const float row_multiplier = e8m0_encode_multiplier(row_e8m0);" in epilogue
    assert "e8m0_pow2_encode_multiplier(row_e8m0)" not in epilogue
    assert "bf16_amax_to_e8m0_1d_mse(row_amax_bits)" in epilogue
    assert "row_words[word] = quantize_four_bf16_pairs(" in epilogue
    assert """constexpr float dpsum_scale =
                                        PUBLISH_V_MXFP4 && !PUBLISH_V_FP8
                                        ? 1.0f
                                        : 16.0f;""" in epilogue
    assert """.v_scale_rows = 1,
        // dP's native MX path uses the standard width-six reconstruction""" in extension
    assert ".v_mxfp4_scale_2d = true," in extension


def test_d128_no_fp8_publication_keeps_sequence_major_forward_scales() -> None:
    source = EXTENSION.read_text(encoding="utf-8")

    # Every non-FP8 QKV launch explicitly selects sequence-major MX scale
    # pages for D128.  Omitting these trailing template arguments silently
    # reorders [B,S/128,H,512] as [B,H,S/128,512].
    selector = """ApplyRope, false, true, true,
                kCompactPublishesMxBackwardV, kQkDepth == 128,
                !kCompactForwardOut,
                kQkDepth == 128 && PackedRope,
                kQkDepth == 128 && SharedPackedRope,
                kQkDepth == 128 && CacheAdaptiveQkScale,
                false, false, kInterleaveCausalKv, kPublishForwardFp8,
                false,
                kCompactPublishesMxBackwardV && PerBlockQkScales,
                false, false,
                kExperimentalOutputSharedSplitV &&
                    kCompactPublishesMxBackwardV"""
    retained_selector = selector + "\n            >"
    experimental_selector = selector + ",\n                true\n            >"
    shared_tile_selector = (
        selector + ",\n                false,\n                true\n            >"
    )
    assert source.count(retained_selector) == 6
    assert source.count(experimental_selector) == 1
    assert source.count(shared_tile_selector) == 1
    assert source.count(selector) == 8


def test_compact_d128_can_publish_mx_backward_v_with_fp8_qk() -> None:
    source = EXTENSION.read_text(encoding="utf-8")

    assert "bool kCompactPublishesMxBackwardV = false" in source
    assert "kCompactPublishesMxBackwardV, kQkDepth == 128" in source
    assert source.count("mx_backward_v_mx_forward_out") >= 6
    assert "v_backward_mxfp4_scales_out" in source
    assert "true, VALIDATE, true, true" in source
