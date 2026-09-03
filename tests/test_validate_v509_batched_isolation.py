from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tk_fa4.native_gqa_tk_bwd.validate_v509_batched_isolation import (
    batched_capture_orders,
    dq_store_add_order_ok,
    output_error,
    precleared_zero_dout_semantics,
    require_nontrivial_independent_outputs,
    require_pairwise_distinct_effective_inputs,
    resolve_distinct_capture_paths,
)


def test_capture_gate_rejects_two_paths_to_the_same_file(tmp_path: Path) -> None:
    capture = tmp_path / "capture.pt"
    capture.touch()
    alias = tmp_path / "capture-alias.pt"
    alias.symlink_to(capture)

    with pytest.raises(ValueError, match="exactly 2 distinct"):
        resolve_distinct_capture_paths([capture, alias], batch=2)


def test_capture_gate_accepts_two_distinct_resolved_files(tmp_path: Path) -> None:
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    first.touch()
    second.touch()

    assert resolve_distinct_capture_paths([first, second], batch=2) == [
        first.resolve(),
        second.resolve(),
    ]


@pytest.mark.parametrize(
    ("batch", "listed", "reversed_order"),
    (
        (2, (0, 1), (1, 0)),
        (4, (0, 1, 2, 3), (3, 2, 1, 0)),
    ),
)
def test_batched_gate_checks_listed_and_reversed_capture_orders(
    batch: int,
    listed: tuple[int, ...],
    reversed_order: tuple[int, ...],
) -> None:
    assert batched_capture_orders(batch, batch) == {
        "listed": listed,
        "reversed": reversed_order,
    }


def test_batched_gate_requires_two_source_captures() -> None:
    with pytest.raises(ValueError, match="must equal"):
        batched_capture_orders(2, 4)


def test_batched_gate_rejects_effective_input_duplicates() -> None:
    with pytest.raises(ValueError, match="pairwise-distinct"):
        require_pairwise_distinct_effective_inputs(
            ["a", "b", "a", "b"], batch=4
        )
    require_pairwise_distinct_effective_inputs(
        ["a", "b", "c", "d"], batch=4
    )


def test_batched_gate_requires_nontrivial_independent_b1_gradients() -> None:
    nonzero = torch.ones(4, dtype=torch.bfloat16)
    counts = require_nontrivial_independent_outputs(
        [{name: nonzero.clone() for name in ("dq", "dk", "dv")}]
    )
    assert counts == [{"dq": 4, "dk": 4, "dv": 4}]

    with pytest.raises(ValueError, match="must produce nonzero"):
        require_nontrivial_independent_outputs(
            [
                {
                    "dq": nonzero.clone(),
                    "dk": torch.zeros_like(nonzero),
                    "dv": nonzero.clone(),
                }
            ]
        )


def test_bf16_error_gate_measures_encoding_ulps() -> None:
    expected = torch.tensor([1.0, -1.0], dtype=torch.bfloat16)
    actual_bits = expected.view(torch.int16).clone()
    actual_bits[0] += 1
    actual_bits[1] -= 1
    error = output_error(actual_bits.view(torch.bfloat16), expected)
    assert error["max_bf16_ulp"] == 1
    assert error["over_one_bf16_ulp_elements"] == 0


def test_dq_store_add_gate_accepts_sparse_bf16_order_variance() -> None:
    expected = torch.ones(10_000, dtype=torch.bfloat16)
    actual = expected.clone()
    actual[17] = torch.tensor(1.0078125, dtype=torch.bfloat16)

    assert dq_store_add_order_ok(output_error(actual, expected))


def test_dq_store_add_gate_rejects_batch_alias_scale_error() -> None:
    expected = torch.ones(10_000, dtype=torch.bfloat16)
    actual = expected.clone()
    actual[:100] = 2.0

    assert not dq_store_add_order_ok(output_error(actual, expected))


def test_dq_store_add_gate_requires_exact_zero_for_zero_reference() -> None:
    expected = torch.zeros(64, dtype=torch.bfloat16)
    assert dq_store_add_order_ok(output_error(expected.clone(), expected))
    actual = expected.clone()
    actual[0] = 1.0
    assert not dq_store_add_order_ok(output_error(actual, expected))


def test_precleared_zero_dout_semantics_requires_preserved_dq_and_zero_dkdv(
) -> None:
    sentinel = torch.full((8,), 0.75, dtype=torch.bfloat16)
    outputs = {
        "dq": sentinel.clone(),
        "dk": torch.zeros(8, dtype=torch.bfloat16),
        "dv": torch.zeros(8, dtype=torch.bfloat16),
    }

    receipt = precleared_zero_dout_semantics(outputs, sentinel)

    assert receipt["passed"] is True
    assert receipt["dq_sentinel_preserved_bitwise"] is True
    assert receipt["dq_nonzero_count"] == 8
    assert receipt["dk_nonzero_count"] == 0
    assert receipt["dv_nonzero_count"] == 0

    outputs["dq"].zero_()
    assert precleared_zero_dout_semantics(outputs, sentinel)["passed"] is False
    outputs["dq"].copy_(sentinel)
    outputs["dk"][0] = 1
    assert precleared_zero_dout_semantics(outputs, sentinel)["passed"] is False
