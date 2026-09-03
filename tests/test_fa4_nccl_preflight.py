from __future__ import annotations

import pytest

from tools.fa4_nccl_preflight import validate_rank_inventory


def _records() -> list[dict[str, object]]:
    return [
        {
            "rank": rank,
            "hostname": f"node-{rank // 2}",
            "local_rank": rank % 2,
            "device_uuid": f"gpu-{rank}",
        }
        for rank in range(4)
    ]


def test_validate_rank_inventory_accepts_complete_topology() -> None:
    validate_rank_inventory(
        _records(),
        expected_world_size=4,
        expected_nodes=2,
        expected_local_world_size=2,
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rows: rows.pop(), "rank records"),
        (lambda rows: rows[3].update(rank=2), "global rank inventory"),
        (lambda rows: rows[3].update(hostname="node-0"), "host"),
        (lambda rows: rows[3].update(local_rank=0), "local ranks"),
        (lambda rows: rows[3].update(device_uuid="gpu-2"), "UUID"),
    ],
)
def test_validate_rank_inventory_rejects_incomplete_topology(mutate, message) -> None:
    rows = _records()
    mutate(rows)
    with pytest.raises(RuntimeError, match=message):
        validate_rank_inventory(
            rows,
            expected_world_size=4,
            expected_nodes=2,
            expected_local_world_size=2,
        )
