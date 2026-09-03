#!/usr/bin/env python3
"""Fail-fast topology and NCCL collective check before distributed FA4.

Launch this module with the same torchrun topology as training. It performs no
file or service access and prints one record per rank plus a world summary.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
import os
import socket
import time

import torch
import torch.distributed as dist


PROBE_ELEMENTS = 16 * 1024 * 1024  # 64 MiB of float32 per rank.
WARMUP_ITERATIONS = 3
MEASURED_ITERATIONS = 5


def validate_rank_inventory(
    records: list[dict[str, object]],
    *,
    expected_world_size: int,
    expected_nodes: int,
    expected_local_world_size: int,
) -> None:
    """Validate a gathered rank/device inventory without requiring CUDA."""
    if len(records) != expected_world_size:
        raise RuntimeError(
            f"expected {expected_world_size} rank records, found {len(records)}"
        )
    if {int(record["rank"]) for record in records} != set(range(expected_world_size)):
        raise RuntimeError("global rank inventory is not contiguous")

    by_host: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        by_host[str(record["hostname"])].append(record)
    if len(by_host) != expected_nodes:
        raise RuntimeError(f"expected {expected_nodes} hosts, found {sorted(by_host)}")

    expected_local_ranks = set(range(expected_local_world_size))
    for hostname, host_records in by_host.items():
        local_ranks = {int(record["local_rank"]) for record in host_records}
        if local_ranks != expected_local_ranks:
            raise RuntimeError(
                f"host {hostname} local ranks mismatch: {sorted(local_ranks)}"
            )

    device_uuids = [str(record["device_uuid"]) for record in records]
    if len(set(device_uuids)) != expected_world_size:
        raise RuntimeError("global CUDA device UUID inventory is not unique")


def _positive_env(name: str) -> int:
    value = os.environ.get(name, "")
    if not value.isdigit() or int(value) <= 0:
        raise RuntimeError(f"{name} must be a positive integer")
    return int(value)


def main() -> int:
    expected_world_size = _positive_env("WORLD_SIZE")
    expected_nodes = _positive_env("NNODES")
    expected_local_world_size = _positive_env("LOCAL_WORLD_SIZE")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(seconds=180))
    try:
        if dist.get_world_size() != expected_world_size:
            raise RuntimeError(
                f"process group world size {dist.get_world_size()} != "
                f"{expected_world_size}"
            )

        properties = torch.cuda.get_device_properties(local_rank)
        record: dict[str, object] = {
            "rank": rank,
            "hostname": socket.gethostname(),
            "local_rank": local_rank,
            "device_uuid": str(properties.uuid),
            "device_name": properties.name,
        }
        gathered: list[dict[str, object] | None] = [
            None for _ in range(expected_world_size)
        ]
        dist.all_gather_object(gathered, record)
        records = [item for item in gathered if item is not None]
        validate_rank_inventory(
            records,
            expected_world_size=expected_world_size,
            expected_nodes=expected_nodes,
            expected_local_world_size=expected_local_world_size,
        )

        payload = torch.ones(PROBE_ELEMENTS, dtype=torch.float32, device=device)
        for _ in range(WARMUP_ITERATIONS):
            dist.all_reduce(payload)
        torch.cuda.synchronize(device)
        dist.barrier(device_ids=[local_rank])

        started = time.perf_counter()
        for _ in range(MEASURED_ITERATIONS):
            dist.all_reduce(payload)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        maximum = torch.tensor(elapsed, dtype=torch.float64, device=device)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)

        expected_value = float(
            expected_world_size ** (WARMUP_ITERATIONS + MEASURED_ITERATIONS)
        )
        if float(payload[0].item()) != expected_value:
            raise RuntimeError(
                f"all-reduce value mismatch: expected {expected_value}, "
                f"found {float(payload[0].item())}"
            )

        print(
            "FA4_NCCL_PREFLIGHT_RANK_OK "
            f"rank={rank} host={record['hostname']} local_rank={local_rank} "
            f"device_uuid={record['device_uuid']}",
            flush=True,
        )
        if rank == 0:
            print(
                "FA4_NCCL_PREFLIGHT_WORLD_OK "
                f"world={expected_world_size} nodes={expected_nodes} "
                f"local_world={expected_local_world_size} "
                f"payload_mib={PROBE_ELEMENTS * 4 / 2**20:.0f} "
                f"iterations={MEASURED_ITERATIONS} "
                f"max_elapsed_seconds={maximum.item():.6f}",
                flush=True,
            )
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
