import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EPILOGUE = ROOT / "tk_fa4/lowp_fa4_bwd/projection_fp4_epilogue.cuh"
EXTENSION = ROOT / "tk_fa4/lowp_fa4_bwd/lowp_fa4_bwd.cu"


def _shuffle_xor(values: list[int], delta: int, width: int = 32) -> list[int]:
    assert len(values) == 32
    return [
        values[(lane // width) * width + ((lane % width) ^ delta)]
        for lane in range(32)
    ]


def _transpose_network(source: list[list[int]]) -> list[list[int]]:
    """Model the CUDA warp-shuffle network one lane at a time."""
    assert len(source) == 32
    assert all(len(lane_words) == 4 for lane_words in source)
    words = [lane_words.copy() for lane_words in source]

    stages = (
        (1, 0x0F0F0F0F, 0xF0F0F0F0, 4),
        (2, 0x00FF00FF, 0xFF00FF00, 8),
        (4, 0x0000FFFF, 0xFFFF0000, 16),
    )
    for quarter in range(4):
        values = [words[lane][quarter] for lane in range(32)]
        for delta, low_mask, high_mask, shift in stages:
            peers = _shuffle_xor(values, delta, width=8)
            values = [
                (value & high_mask) | ((peer & high_mask) >> shift)
                if lane & delta
                else (value & low_mask) | ((peer & low_mask) << shift)
                for lane, (value, peer) in enumerate(zip(values, peers))
            ]
        for lane in range(32):
            words[lane][quarter] = values[lane]

    # Evaluate both shuffle inputs before mutating either member of a pair,
    # exactly as the CUDA donor does.
    bit0_inputs_01 = [
        lane_words[0] if (lane >> 3) & 1 else lane_words[1]
        for lane, lane_words in enumerate(words)
    ]
    bit0_inputs_23 = [
        lane_words[2] if (lane >> 3) & 1 else lane_words[3]
        for lane, lane_words in enumerate(words)
    ]
    exchange_01 = _shuffle_xor(bit0_inputs_01, 8)
    exchange_23 = _shuffle_xor(bit0_inputs_23, 8)
    for lane in range(32):
        if (lane >> 3) & 1:
            words[lane][0] = exchange_01[lane]
            words[lane][2] = exchange_23[lane]
        else:
            words[lane][1] = exchange_01[lane]
            words[lane][3] = exchange_23[lane]

    bit1_inputs_02 = [
        lane_words[0] if (lane >> 3) & 2 else lane_words[2]
        for lane, lane_words in enumerate(words)
    ]
    bit1_inputs_13 = [
        lane_words[1] if (lane >> 3) & 2 else lane_words[3]
        for lane, lane_words in enumerate(words)
    ]
    exchange_02 = _shuffle_xor(bit1_inputs_02, 16)
    exchange_13 = _shuffle_xor(bit1_inputs_13, 16)
    for lane in range(32):
        if (lane >> 3) & 2:
            words[lane][0] = exchange_02[lane]
            words[lane][1] = exchange_13[lane]
        else:
            words[lane][2] = exchange_02[lane]
            words[lane][3] = exchange_13[lane]

    return words


def _pack_nibbles(values: list[int]) -> int:
    assert len(values) == 8
    assert all(0 <= value < 16 for value in values)
    return sum(value << (4 * index) for index, value in enumerate(values))


def _store_uint4(buffer: bytearray, offset: int, words: list[int]) -> None:
    assert offset % 16 == 0
    assert len(words) == 4
    packed = b"".join(word.to_bytes(4, "little") for word in words)
    buffer[offset : offset + 16] = packed


def test_register_network_exhaustively_transposes_every_nibble() -> None:
    # Code zero is included even though every placement is the same all-zero
    # matrix. Codes 1..15 exercise every bit pattern at all 1,024 positions.
    for depth in range(32):
        for sequence in range(32):
            for code in range(16):
                source = [[0] * 4 for _ in range(32)]
                source[depth][sequence // 8] = code << (
                    4 * (sequence & 7)
                )

                expected = [[0] * 4 for _ in range(32)]
                expected[sequence][depth // 8] = code << (4 * (depth & 7))
                assert _transpose_network(source) == expected


def test_register_network_is_an_involution_for_dense_tiles() -> None:
    generator = random.Random(20260830)
    for _ in range(32):
        source = [
            [
                _pack_nibbles([generator.randrange(16) for _ in range(8)])
                for _ in range(4)
            ]
            for _ in range(32)
        ]
        assert _transpose_network(_transpose_network(source)) == source


def test_payload_offsets_and_scale_swizzles_preserve_one_code_matrix() -> None:
    sequence_length = 256
    depth = 128
    logical = [
        [
            (13 * sequence + 7 * feature + (sequence ^ feature)) & 0xF
            for feature in range(depth)
        ]
        for sequence in range(sequence_length)
    ]

    forward = bytearray(depth * sequence_length // 2)
    backward = bytearray(sequence_length * depth // 2)
    forward_scales = bytearray((sequence_length // 128) * 512)
    backward_scales = bytearray((sequence_length // 128) * 512)

    for sequence_base in range(0, sequence_length, 32):
        for depth_base in range(0, depth, 32):
            source = [
                [
                    _pack_nibbles(
                        [
                            logical[sequence_base + 8 * word + nibble][
                                depth_base + depth_lane
                            ]
                            for nibble in range(8)
                        ]
                    )
                    for word in range(4)
                ]
                for depth_lane in range(32)
            ]
            transposed = _transpose_network(source)

            for depth_lane in range(32):
                forward_offset = (
                    (depth_base + depth_lane) * (sequence_length // 2)
                    + sequence_base // 2
                )
                _store_uint4(forward, forward_offset, source[depth_lane])

            for sequence_lane in range(32):
                backward_offset = (
                    (sequence_base + sequence_lane) * (depth // 2)
                    + depth_base // 2
                )
                _store_uint4(
                    backward,
                    backward_offset,
                    transposed[sequence_lane],
                )

            tile_code = 1 + (sequence_base // 32) * 4 + depth_base // 32
            sequence_page = (sequence_base // 128) * 512
            sequence_quarter = (sequence_base & 127) // 32
            depth_group = depth_base // 32
            for depth_lane in range(32):
                forward_scales[
                    sequence_page
                    + depth_lane * 16
                    + depth_group * 4
                    + sequence_quarter
                ] = tile_code
            for sequence_lane in range(32):
                sequence = sequence_base + sequence_lane
                backward_scales[
                    sequence_page
                    + (sequence & 31) * 16
                    + ((sequence >> 5) & 3) * 4
                    + depth_group
                ] = tile_code

    for sequence in range(sequence_length):
        for feature in range(depth):
            expected = logical[sequence][feature]
            forward_byte = forward[
                feature * (sequence_length // 2) + sequence // 2
            ]
            backward_byte = backward[sequence * (depth // 2) + feature // 2]
            assert (forward_byte >> (4 * (sequence & 1))) & 0xF == expected
            assert (backward_byte >> (4 * (feature & 1))) & 0xF == expected

            tile_code = 1 + (sequence // 32) * 4 + feature // 32
            sequence_page = (sequence // 128) * 512
            forward_scale_index = (
                sequence_page
                + (feature & 31) * 16
                + (feature // 32) * 4
                + ((sequence & 127) // 32)
            )
            backward_scale_index = (
                sequence_page
                + (sequence & 31) * 16
                + ((sequence >> 5) & 3) * 4
                + feature // 32
            )
            assert forward_scales[forward_scale_index] == tile_code
            assert backward_scales[backward_scale_index] == tile_code


def test_shared_tile_cuda_route_remains_explicit_and_fail_closed() -> None:
    epilogue = EPILOGUE.read_text(encoding="utf-8")
    extension = EXTENSION.read_text(encoding="utf-8")

    assert "bool SHARE_MXFP4_TILE_WITH_BACKWARD = false" in epilogue
    assert "bool EXPERIMENTAL_SHARED_TILE_MXFP4_V = false" in epilogue
    assert "!SHARE_MXFP4_TILE_WITH_BACKWARD || PUBLISH_BACKWARD_MXFP4" in epilogue
    assert "PUBLISH_BACKWARD_MXFP4 && !SHARE_MXFP4_TILE_WITH_BACKWARD" in epilogue
    assert "transpose_mxfp4_32x32_nibbles(packed_words);" in epilogue
    assert "V_SEQUENCE_MAJOR_SCALES && !INTERLEAVE_CAUSAL_KV" in epilogue
    assert "!EXPERIMENTAL_COMMON_ROWSCALE_MXFP4_V" in epilogue

    assert "bool kExperimentalSharedTileMxfp4V = false" in extension
    assert "bool ExperimentalSharedTileMxBackwardV = false" in extension
    assert "shared-tile D128 MX backward V requires D32xS32 scales" in extension
    assert "ExperimentalSharedTileMxBackwardV\n                         ? v_mxfp4_scale_2d" in extension
    assert extension.count("shared_tile_mx_backward_v_mx_forward_out") >= 6
