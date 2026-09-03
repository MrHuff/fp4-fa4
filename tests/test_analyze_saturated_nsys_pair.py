from __future__ import annotations

import pytest

from tk_fa4.lowp_fa4_bwd.analyze_saturated_nsys_pair import (
    _exact_partition_summary,
    _kernel_signature,
    _merge_intervals,
    _signature_digest,
)


def test_exact_partition_requires_disjoint_complete_union() -> None:
    reference = [(0, 1, 10), (0, 1, 11), (0, 1, 12)]
    exact = _exact_partition_summary(
        reference,
        [[reference[0]], [reference[1], reference[2]]],
    )
    assert exact["exact_partition"] is True
    assert exact["partitions_are_disjoint"] is True
    assert exact["partition_union_matches_reference"] is True

    overlapping_and_missing = _exact_partition_summary(
        reference,
        [[reference[0], reference[1]], [reference[1]]],
    )
    assert overlapping_and_missing["partition_entry_count"] == len(reference)
    assert overlapping_and_missing["partitions_are_disjoint"] is False
    assert (
        overlapping_and_missing["partition_union_matches_reference"] is False
    )
    assert overlapping_and_missing["exact_partition"] is False


def test_exact_partition_rejects_duplicate_reference_identities() -> None:
    with pytest.raises(ValueError, match="reference kernel identities"):
        _exact_partition_summary([(0, 1, 10), (0, 1, 10)], [])


def test_merge_intervals_merges_overlap_and_adjacency() -> None:
    assert _merge_intervals([(8, 10), (1, 4), (4, 7), (12, 13)]) == [
        (1, 7),
        (8, 10),
        (12, 13),
    ]


def test_merge_intervals_rejects_invalid_or_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        _merge_intervals([])
    with pytest.raises(ValueError, match="invalid interval"):
        _merge_intervals([(2, 1)])


def test_kernel_signature_excludes_timing_and_grid_id() -> None:
    kernel = {
        "demangled_name": "kernel<int>",
        "gridX": 2,
        "gridY": 3,
        "gridZ": 4,
        "blockX": 32,
        "blockY": 2,
        "blockZ": 1,
        "registersPerThread": 128,
        "staticSharedMemory": 400,
        "dynamicSharedMemory": 90112,
        "start": 10,
        "end": 20,
        "gridId": 99,
    }
    assert _kernel_signature(kernel) == (
        "kernel<int>",
        2,
        3,
        4,
        32,
        2,
        1,
        128,
        400,
        90112,
    )


def test_signature_digest_is_order_sensitive_and_deterministic() -> None:
    signatures = [("a", 1), ("b", 2)]
    assert _signature_digest(signatures) == _signature_digest(signatures)
    assert _signature_digest(signatures) != _signature_digest(
        list(reversed(signatures))
    )
