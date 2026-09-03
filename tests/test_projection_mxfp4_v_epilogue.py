from __future__ import annotations

import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EPILOGUE = (
    ROOT / "tk_fa4" / "lowp_fa4_bwd" / "projection_fp4_epilogue.cuh"
)
CUDA = ROOT / "tk_fa4" / "lowp_fa4_bwd" / "lowp_fa4_bwd.cu"
SATURATED = (
    ROOT / "tk_fa4" / "lowp_fa4_bwd" / "benchmark_llama12b_saturated.py"
)


def _publisher_source() -> str:
    source = EPILOGUE.read_text()
    return source.split("void publish_v_mxfp4(", 1)[1].split(
        "\ntemplate <\n    typename C,\n    bool PUBLISH_FP4,",
        1,
    )[0]


def _old_gather(
    pairs: list[list[list[int]]],
    *,
    warp: int,
    lane: int,
    interleaved: bool,
) -> list[int]:
    pair_index = lane >> 1
    shift = (lane & 1) * 16
    values = []
    for row in range(32):
        source_warp = row >> 3 if interleaved else warp
        source_row = warp + (row & 7) * 4 if interleaved else row
        values.append((pairs[source_warp][source_row][pair_index] >> shift) & 0xFFFF)
    return values


def _paired_gather(
    pairs: list[list[list[int]]],
    *,
    warp: int,
    lane: int,
    interleaved: bool,
) -> list[int]:
    pair_index = lane >> 1
    shift = (lane & 1) * 16
    gathered = []
    for packed_pair in range(16):
        row0 = 2 * packed_pair
        row1 = row0 + 1
        source_warp0 = row0 >> 3 if interleaved else warp
        source_warp1 = row1 >> 3 if interleaved else warp
        source_row0 = warp + (row0 & 7) * 4 if interleaved else row0
        source_row1 = warp + (row1 & 7) * 4 if interleaved else row1
        value0 = (pairs[source_warp0][source_row0][pair_index] >> shift) & 0xFFFF
        value1 = (pairs[source_warp1][source_row1][pair_index] >> shift) & 0xFFFF
        gathered.append(value0 | (value1 << 16))
    return [
        value
        for pair in gathered
        for value in (pair & 0xFFFF, pair >> 16)
    ]


def _striped_amax(values: list[int]) -> int:
    accumulators = [0, 0, 0, 0]
    for packed_pair in range(16):
        absolute0 = values[2 * packed_pair] & 0x7FFF
        absolute1 = values[2 * packed_pair + 1] & 0x7FFF
        stripe = packed_pair & 3
        accumulators[stripe] = max(
            accumulators[stripe],
            absolute0,
            absolute1,
        )
    return max(accumulators)


def _e8m0_1d_mse(absolute_bits: int) -> int:
    exponent = (absolute_bits >> 7) & 0xFF
    if exponent == 0:
        return 0
    round_up = (absolute_bits & 0x7F) >= 0x1A
    return exponent + int(round_up and exponent < 0xFE)


def test_paired_gather_is_bit_exact_for_normal_and_causal_layouts() -> None:
    generator = random.Random(20260825)
    pairs = [
        [
            [generator.getrandbits(32) for _ in range(16)]
            for _ in range(32)
        ]
        for _ in range(4)
    ]
    for interleaved in (False, True):
        for warp in range(4):
            for lane in range(32):
                expected = _old_gather(
                    pairs,
                    warp=warp,
                    lane=lane,
                    interleaved=interleaved,
                )
                actual = _paired_gather(
                    pairs,
                    warp=warp,
                    lane=lane,
                    interleaved=interleaved,
                )
                assert actual == expected


def test_striped_amax_and_1x32_selector_are_bit_exact() -> None:
    generator = random.Random(7)
    cases = [
        [0] * 32,
        [0x8000] * 32,
        [0x0001, 0x007F, 0x0080, 0x7F80] * 8,
        [0x7FC1, 0xFFC1, 0x7F80, 0xFF80] * 8,
    ]
    cases.extend(
        [generator.getrandbits(16) for _ in range(32)]
        for _ in range(256)
    )
    for values in cases:
        serial_amax = max(value & 0x7FFF for value in values)
        striped_amax = _striped_amax(values)
        assert striped_amax == serial_amax
        assert _e8m0_1d_mse(striped_amax) == _e8m0_1d_mse(serial_amax)


def test_four_pair_ptx_pack_preserves_fp32_quantization_contract() -> None:
    source = EPILOGUE.read_text().split(
        "uint32_t quantize_four_bf16_pairs(", 1
    )[1].split("\n}\n", 1)[0]
    assert source.count("cvt.f32.bf16") == 8
    assert source.count("mul.rn.f32") == 8
    assert source.count("cvt.rn.satfinite.e2m1x2.f32") == 4
    assert '"mov.b32 %0, {byte0, byte1, byte2, byte3};\\n"' in source
    assert "e2m1x2.f16x2" not in source
    assert "cvt.rs" not in source


def test_forward_mx_publisher_removes_serial_chains_without_contract_drift() -> None:
    source = _publisher_source()
    assert "uint16_t values[32]" not in source
    assert "uint32_t gathered_pairs[16];" in source
    assert "uint16_t amax_bits[4] = {0, 0, 0, 0};" in source
    assert "amax_bits[packed_pair & 3]" in source
    assert "quantize_four_bf16_pairs(" in source
    assert "bf16_amax_to_e8m0_1d_mse(row_amax_bits)" in source
    assert "bf16_amax_to_e8m0_rte(" in source

    # Preserve the causal transpose, feature-major payload, and native scale
    # page formulas exactly; no synchronization is removed from represented
    # backward compatibility specializations.
    assert "INTERLEAVE_CAUSAL_KV ? row0 >> 3 : warp" in source
    assert "INTERLEAVE_CAUSAL_KV ? row1 >> 3 : warp" in source
    assert "warp + (row0 & 7) * 4" in source
    assert "warp + (row1 & 7) * 4" in source
    assert "output_head_depth +\n         depth) * (g.seq_len / 2)" in source
    assert "depth_lane * 16 + depth_group * 4 + sequence_quarter" in source
    assert source.count("kittens::warpgroup::sync(1);") == 3


def test_saturated_mx_route_keeps_1x32_scales_and_split_e4m3_backward() -> None:
    saturated = SATURATED.read_text()
    assert (
        "experimental_split_v_backward=(route in MX_ROUTES) if not is_d128 else False"
        in saturated
    )
    assert "experimental_d128_shared_tile_mxfp4_v: bool = False" in saturated
    assert (
        "v_mxfp4_scale_2d=experimental_d128_shared_tile_mxfp4_v" in saturated
    )

    cuda = CUDA.read_text()
    active_instantiation = cuda.split(
        "auto launch = [&]<\n        bool PublishMxfp4V,",
        1,
    )[-1].split(">(globals);", 1)[0]
    assert "false,          // PUBLISH_V_BACKWARD_MXFP4" in active_instantiation
    assert "ExperimentalSplitVBackward" in active_instantiation
