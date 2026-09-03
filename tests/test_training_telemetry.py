from __future__ import annotations

import pytest
import torch

from tk_fa4.lowp_fa4_bwd.training_telemetry import (
    forward_diagnostic_tensor_statistics,
    mark_matched_round_timing_eligibility,
    select_timing_records,
)


def test_forward_scale_statistics_respect_storage_encoding() -> None:
    q_scales = torch.tensor(
        [0.5, 1.0, 2.0], dtype=torch.float32
    ).to(torch.float8_e4m3fn)
    q_statistics = forward_diagnostic_tensor_statistics(
        "q_forward_scales", q_scales
    )
    assert q_statistics["encoding"] == "nvfp4_e4m3_block_scale"
    assert q_statistics["minimum"] == 0.5
    assert q_statistics["maximum"] == 2.0

    v_codes = torch.tensor([0, 126, 127, 130], dtype=torch.uint8)
    v_storage = v_codes.view(torch.float8_e4m3fn)
    v_statistics = forward_diagnostic_tensor_statistics(
        "v_forward_scales", v_storage
    )
    assert v_statistics["encoding"] == "mxfp4_e8m0_block_scale"
    assert v_statistics["raw_code_minimum"] == 0
    assert v_statistics["raw_code_maximum"] == 130
    assert v_statistics["decoded_exponent_minimum"] == -1
    assert v_statistics["decoded_exponent_maximum"] == 3
    assert v_statistics["zero_code_fraction"] == 0.25


def test_packed_probability_scales_and_empty_sentinel() -> None:
    probability_codes = torch.tensor(
        [0, 110, 111, 112, 120, 121, 122, 123], dtype=torch.uint8
    )
    packed_words = probability_codes.view(torch.int32)
    statistics = forward_diagnostic_tensor_statistics(
        "forward_mx_probability_scales", packed_words
    )
    assert statistics["packing"] == "four_codes_per_int32"
    assert statistics["elements"] == 2
    assert statistics["encoded_elements"] == 8
    assert statistics["decoded_exponent_minimum"] == -1
    assert statistics["decoded_exponent_maximum"] == 12

    empty = torch.empty(0, dtype=torch.float8_e4m3fn)
    empty_statistics = forward_diagnostic_tensor_statistics(
        "v_forward_scales", empty
    )
    assert not empty_statistics["present"]
    assert empty_statistics["elements"] == 0
    assert "raw_code_minimum" not in empty_statistics


def test_timing_eligibility_is_matched_across_every_route() -> None:
    clean_round = {
        route: {"round": 4, "diagnostic": False, "step_ms": value}
        for route, value in (("bf16", 3.0), ("mx", 2.0), ("fp8", 2.1))
    }
    assert mark_matched_round_timing_eligibility(clean_round)
    assert all(record["timing_eligible"] for record in clean_round.values())

    diagnostic_round = {
        "bf16": {"round": 5, "diagnostic": False, "step_ms": 3.1},
        "mx": {"round": 5, "diagnostic": True, "step_ms": 8.0},
        "fp8": {"round": 5, "diagnostic": False, "step_ms": 2.2},
    }
    assert not mark_matched_round_timing_eligibility(diagnostic_round)
    assert not any(
        record["timing_eligible"] for record in diagnostic_round.values()
    )

    for route in clean_round:
        selected = select_timing_records(
            [clean_round[route], diagnostic_round[route]]
        )
        assert selected == [clean_round[route]]


def test_matched_timing_rejects_cross_round_records() -> None:
    with pytest.raises(ValueError, match="multiple rounds"):
        mark_matched_round_timing_eligibility(
            {
                "mx": {"round": 1, "diagnostic": False},
                "fp8": {"round": 2, "diagnostic": False},
            }
        )
