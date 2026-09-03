from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "tk_fa4/native_gqa_tk_bwd"
V502_HEADER = NATIVE / (
    "v502_d128_gqa_mxfp4v_e4m3do_b2_s4096_owner4_"
    "experimental_bshd.cuh"
)
V507_STEM = (
    "v507_d128_gqa_mxfp4v_sharedtile_e4m3do_b2_s4096_owner4_"
    "experimental_bshd"
)
V507_HEADER = NATIVE / f"{V507_STEM}.cuh"
V507_BINDING = NATIVE / f"{V507_STEM}.cu"
V507_MAKEFILE = NATIVE / "Makefile.v507"


def _between(source: str, start: str, end: str) -> str:
    begin = source.index(start)
    return source[begin : source.index(end, begin)]


def test_v507_preserves_v502_exact_four_anchor_mma_contract() -> None:
    v502 = V502_HEADER.read_text(encoding="utf-8")
    v507 = V507_HEADER.read_text(encoding="utf-8")

    # The producer ABI changes provenance, not values: persistent restaging,
    # physical scale-page loading, and all four block-scale MMA chunks must
    # remain byte-for-byte identical to the authenticated v502 consumer.
    helper_start = (
        "__device__ __forceinline__ void "
        "stage_persistent_mxfp4_v_and_scales("
    )
    helper_end = "__global__ __launch_bounds__(kThreads, 1)"
    assert _between(v507, helper_start, helper_end) == _between(
        v502, helper_start, helper_end
    )

    helper = _between(v507, helper_start, helper_end)
    assert "for (int chunk = 0; chunk < 4; ++chunk)" in helper
    assert "kind::mxf8f6f4.block_scale" in helper
    assert "static_assert(kInstructionBase == 0x08A00280u);" in helper
    assert "(static_cast<uint32_t>(chunk) << 29)" in helper
    assert "(static_cast<uint32_t>(chunk) << 4)" in helper


def test_v507_issues_next_score_and_restages_scales_at_earliest_safe_gates(
) -> None:
    source = V507_HEADER.read_text(encoding="utf-8")
    issuer = _between(
        source,
        "} else if (physical_warp == kTensorIssueWarp && lane == 0) {",
        "} else if (\n        physical_warp >= kReduceWarpBase",
    )
    iteration = _between(
        issuer,
        "wait(probability_half_ready[0], phase);",
        "// The dK commit tracks both earlier dV halves",
    )

    ordered_tokens = (
        "issue_gradient_ab_runtime_accumulate_half<0>(",
        "wait(dp_ready, phase);",
        "core::issue_score_or_dp(",
        "wait(probability_half_ready[1], phase);",
        "core::issue_gradient_atb(",
        "prior::issue_gradient_ab_runtime_accumulate(",
        "wait(score_consumed, next_phase);",
        "stage_mixed_dp_scale_tmem(",
        "wait(dq_tmem_drained, phase);",
        "issue_mxfp4v_e4m3do_dp(",
    )
    positions = [iteration.index(token) for token in ordered_tokens]
    assert positions == sorted(positions)
    assert iteration.count("wait(dp_ready, phase);") == 1
    assert iteration.count("core::issue_score_or_dp(") == 1
    assert iteration.count("stage_mixed_dp_scale_tmem(") == 1
    assert iteration.count("wait(dq_tmem_drained, phase);") == 1

    # Head boundaries still need the same alias proof for the final dP, which
    # has no within-head successor to consume its dp_ready phase.
    head_setup = _between(
        issuer,
        "if (local_head > 0) {",
        "const int first_stage",
    )
    assert "wait(score_consumed, previous_phase);" in head_setup
    assert "wait(dp_ready, previous_phase);" in head_setup


def test_v507_metadata_is_fail_closed_to_shared_d32xs32_producer() -> None:
    binding = V507_BINDING.read_text(encoding="utf-8")
    makefile = V507_MAKEFILE.read_text(encoding="utf-8")

    required_metadata = (
        '"tkfa4.native_tk_d128_backward.experimental.v3"',
        '"experimental_shared_tile_mxfp4_v1_exact_D32xS32_forward_anchor_"',
        '"shared_tile_mx_backward_v_mx_forward_out"',
        '"B2_S4096_D128_GQA_shared_D32xS32_producer_only"',
        'result["producer_reuses_forward_quantization"] = true;',
        'result["producer_shared_tile_shape"] = "D32xS32";',
        'result["producer_v_mxfp4_scale_2d"] = true;',
        'result["consumer_anchor_count_per_d128_row"] = 4;',
        'result["consumer_requantization"] = "none";',
        'result["common_rowscale_abi_compatible"] = false;',
        'result["dp_instruction_descriptor"] = "0x08A00280";',
        'result["next_score_overwrite_gate"] = "current_dp_ready";',
        'result["next_scale_restage_gate"] = "next_score_consumed";',
        'result["next_dp_overwrite_gate"] = "current_dq_tmem_drained";',
        'result["expected_static_shared_bytes"] = 168208;',
        '"uncompiled_v507_source_layout_expectation"',
    )
    for token in required_metadata:
        assert token in binding

    assert f'#include "{V507_STEM}.cuh"' in binding
    assert f"SRC := {V507_STEM}.cu" in makefile
    assert f"\t{V507_STEM}.cuh \\" in makefile
    assert "_C_sm100_gqa_tk_v507_d128_mxfp4v_sharedtile" in makefile
    assert "v502_d128_gqa_mxfp4v_e4m3do" not in makefile
    assert "compiled_static_shared_bytes" not in binding
