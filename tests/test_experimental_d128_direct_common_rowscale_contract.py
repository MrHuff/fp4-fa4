from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CUDA = ROOT / "tk_fa4/lowp_fa4_bwd/lowp_fa4_bwd.cu"
EPILOGUE = ROOT / "tk_fa4/lowp_fa4_bwd/projection_fp4_epilogue.cuh"
INTERFACE = ROOT / "tk_fa4/interface.py"
E2E = ROOT / "tk_fa4/lowp_fa4_bwd/benchmark_llama12b_e2e.py"
SATURATED_E2E = (
    ROOT / "tk_fa4/lowp_fa4_bwd/benchmark_llama12b_saturated.py"
)

SYMBOL_PREFIX = (
    "project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered_"
)
RETAINED_SYMBOL = SYMBOL_PREFIX + "mx_backward_v_mx_forward_out"
EXPERIMENTAL_SUFFIX = (
    "direct_common_rowscale_mx_backward_v_mx_forward_out"
)
EXPERIMENTAL_SYMBOL = SYMBOL_PREFIX + EXPERIMENTAL_SUFFIX


def _function_body(source: str, name: str, next_marker: str) -> str:
    assert source.count(name) == 1
    assert source.count(next_marker) == 1
    return source.split(name, 1)[1].split(next_marker, 1)[0]


def _macro_arguments(source: str, symbol: str) -> str:
    marker = (
        "TKFA4_DEFINE_D128_NVFP4_MX_BACKWARD_V_FORWARD_OUT(\n"
        f"    {symbol},"
    )
    assert source.count(marker) == 1
    arguments = source.split(marker, 1)[1].split(")", 1)[0]
    return "".join(arguments.split())


def test_experimental_direct_common_rowscale_is_fail_closed() -> None:
    cuda = CUDA.read_text(encoding="utf-8")
    epilogue = EPILOGUE.read_text(encoding="utf-8")

    assert "bool kExperimentalCommonRowscaleMxfp4V = false" in cuda
    assert "bool ExperimentalCommonRowscaleMxBackwardV = false" in cuda
    assert "bool EXPERIMENTAL_COMMON_ROWSCALE_MXFP4_V = false" in epilogue

    outer_contract = cuda.split(
        "static_assert(\n        !kExperimentalCommonRowscaleMxfp4V ||",
        1,
    )[1].split(");", 1)[0]
    for requirement in (
        "kExperimentalOutputSharedSplitV",
        "kCompactPublishesMxBackwardV && kCompactPublishesMxV",
        "kCompactForwardOut && kQkDepth == 128 && !kPairedD64",
        "!kPublishRepresentedBackwardFp8 && kPerBlockQkScales",
    ):
        assert requirement in outer_contract

    kernel_contract = epilogue.split(
        "static_assert(\n        !EXPERIMENTAL_COMMON_ROWSCALE_MXFP4_V ||",
        1,
    )[1].split(");", 1)[0]
    for requirement in (
        "EXPERIMENTAL_OUTPUT_SHARED_SPLIT_V",
        "PUBLISH_V_MXFP4 && PUBLISH_V_BACKWARD_MXFP4",
        "!PUBLISH_V_FP8 && !PUBLISH_REPRESENTED_BACKWARD_FP8",
        "!INTERLEAVE_CAUSAL_KV && !OUTPUT_IS_DOUT",
        "C::QK_DEPTH == 128 && G::D_tile::rows == 128",
        "G::D_tile::cols == 32",
    ):
        assert requirement in kernel_contract

    expected_symbols = (
        (RETAINED_SYMBOL, "true,false,false"),
        (f"{RETAINED_SYMBOL}_unchecked", "false,false,false"),
        (EXPERIMENTAL_SYMBOL, "true,true,false"),
        (f"{EXPERIMENTAL_SYMBOL}_unchecked", "false,true,false"),
    )
    for symbol, expected_arguments in expected_symbols:
        assert _macro_arguments(cuda, symbol) == expected_arguments
        assert cuda.count(f"&{symbol},") == 1

    for integrated_source in (INTERFACE, E2E, SATURATED_E2E):
        assert EXPERIMENTAL_SUFFIX not in integrated_source.read_text(
            encoding="utf-8"
        )


def test_experimental_direct_common_rowscale_publisher_is_separate() -> None:
    source = EPILOGUE.read_text(encoding="utf-8")
    publisher = _function_body(
        source,
        "publish_v_common_rowscale_mxfp4_from_output_ring(",
        "template <\n"
        "    typename C,\n"
        "    bool SEQUENCE_MAJOR_COLUMN_SCALES = false,\n"
        "    bool PUBLISH_BACKWARD_MXFP4 = true,",
    )

    assert "constexpr int kDepthBlocks = kDepth / 32;" in publisher
    assert publisher.count(
        "for (int k_block = 0; k_block < kDepthBlocks; ++k_block)"
    ) == 2
    assert "const uint16_t common_amax_bits = max(" in publisher
    assert "bf16_amax_to_e8m0_1d_mse(common_amax_bits)" in publisher
    assert (
        "const float common_multiplier = e8m0_encode_multiplier(common_code);"
        in publisher
    )
    assert "quantize_four_bf16_pairs(" in publisher
    assert "g.v_backward_mxfp4 + row_payload_base + k_block * 16" in publisher
    assert "static_cast<uint32_t>(common_code) * 0x01010101u" in publisher
    assert "g.v_mxfp4 + payload_base" not in publisher

    branch = source.split(
        "if constexpr (EXPERIMENTAL_OUTPUT_SHARED_SPLIT_V) {",
        1,
    )[1].split("output_rt registers;", 1)[0]
    assert branch.count("publish_v_mxfp4_from_output_shared<") == 1
    assert (
        "PUBLISH_V_BACKWARD_MXFP4 &&\n"
        "                                    "
        "!EXPERIMENTAL_COMMON_ROWSCALE_MXFP4_V"
        in branch
    )
    assert branch.count(
        "publish_v_common_rowscale_mxfp4_from_output_ring<"
    ) == 1
    assert "if ((local_col & 127) == 96)" in branch
    assert "epi - 3" in branch
    assert "local_col - 96" in branch
