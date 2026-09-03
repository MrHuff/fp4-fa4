from __future__ import annotations

import pytest

from tk_fa4.lowp_fa4_bwd.training_drift_gate import RouteLossDriftGate


def _gate() -> RouteLossDriftGate:
    return RouteLossDriftGate(
        subject_route="mx",
        reference_routes=("bf16", "fp8"),
        window=4,
        warning_threshold=0.05,
        failure_threshold=0.10,
        failure_patience=3,
        minimum_updates=4,
    )


def test_gate_requires_sustained_drift_against_every_reference() -> None:
    gate = _gate()
    reports = []
    for round_index, mx_loss in enumerate(
        (1.00, 1.00, 1.00, 1.24, 1.24, 1.24, 1.24, 1.24)
    ):
        reports.append(
            gate.observe(
                round_index,
                {"mx": mx_loss, "bf16": 1.00, "fp8": 1.02},
            )
        )

    assert reports[4]["warning_exceeded"]
    assert gate.warning_active
    assert not reports[5]["failed"]
    assert not reports[6]["failed"]
    assert reports[7]["failed"]
    assert gate.failure == reports[7]
    assert [item["kind"] for item in gate.as_dict()["transitions"]] == [
        "warning",
        "failure",
    ]


def test_gate_does_not_fail_when_only_one_reference_diverges() -> None:
    gate = _gate()
    for round_index in range(12):
        report = gate.observe(
            round_index,
            {"mx": 1.20, "bf16": 1.00, "fp8": 1.19},
        )
    assert not report["failure_exceeded"]
    assert not gate.failed


def test_gate_rejects_skipped_round_and_nonfinite_loss() -> None:
    gate = _gate()
    with pytest.raises(ValueError, match="expected round 0"):
        gate.observe(1, {"mx": 1.0, "bf16": 1.0, "fp8": 1.0})
    with pytest.raises(ValueError, match="non-finite"):
        gate.observe(0, {"mx": float("nan"), "bf16": 1.0, "fp8": 1.0})


def test_gate_uses_strict_thresholds_and_resets_failure_patience() -> None:
    gate = RouteLossDriftGate(
        subject_route="mx",
        reference_routes=("bf16", "fp8"),
        window=1,
        warning_threshold=0.125,
        failure_threshold=0.25,
        failure_patience=2,
        minimum_updates=1,
    )

    equal_warning = gate.observe(
        0, {"mx": 1.125, "bf16": 1.00, "fp8": 1.00}
    )
    assert not equal_warning["warning_exceeded"]

    equal_failure = gate.observe(
        1, {"mx": 1.25, "bf16": 1.00, "fp8": 1.00}
    )
    assert equal_failure["warning_exceeded"]
    assert not equal_failure["failure_exceeded"]

    assert not gate.observe(
        2, {"mx": 1.50, "bf16": 1.00, "fp8": 1.00}
    )["failed"]
    reset = gate.observe(
        3, {"mx": 1.00, "bf16": 1.00, "fp8": 1.00}
    )
    assert reset["failure_streak"] == 0
    assert not gate.observe(
        4, {"mx": 1.50, "bf16": 1.00, "fp8": 1.00}
    )["failed"]
    assert gate.observe(
        5, {"mx": 1.50, "bf16": 1.00, "fp8": 1.00}
    )["failed"]


def test_gate_honors_minimum_updates_after_window_is_full() -> None:
    gate = RouteLossDriftGate(
        subject_route="mx",
        reference_routes=("bf16", "fp8"),
        window=2,
        warning_threshold=0.05,
        failure_threshold=0.10,
        minimum_updates=4,
    )
    for round_index in range(3):
        report = gate.observe(
            round_index,
            {"mx": 1.25, "bf16": 1.00, "fp8": 1.00},
        )
        assert not report["ready"]
    ready = gate.observe(3, {"mx": 1.25, "bf16": 1.00, "fp8": 1.00})
    assert ready["ready"]
    assert ready["failed"]
