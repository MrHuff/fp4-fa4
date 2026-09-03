#!/usr/bin/env python3
"""Compare two Nsight Systems saturated-route captures.

Kernel attribution uses the CUDA runtime call correlated with each kernel.
Comparing GPU timestamps directly with CPU-side NVTX timestamps is incorrect
for asynchronous launches and can silently omit most of backward.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import statistics
from pathlib import Path
from typing import Any, Iterable, Sequence


PROFILE_RANGE = "profile_step"
BACKWARD_RANGE = "backward_total"
HEADLINE_METRICS = (
    "GPC Clock Frequency [MHz]",
    "SYS Clock Frequency [MHz]",
    "GR Active [Throughput %]",
    "SMs Active [Throughput %]",
    "Tensor Active [Throughput %]",
    "DRAM Read Bandwidth [Throughput %]",
    "DRAM Write Bandwidth [Throughput %]",
)
TOP_LEVEL_RANGES = (
    "decoder_forward",
    "ce_forward",
    BACKWARD_RANGE,
    "gradient_clip",
    "optimizer",
)
METRIC_RANGES = ("decoder_forward", BACKWARD_RANGE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"expected a regular file: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _connect_read_only(path: Path) -> sqlite3.Connection:
    resolved = path.resolve(strict=True)
    connection = sqlite3.connect(f"file:{resolved}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    required = {
        "CUPTI_ACTIVITY_KIND_KERNEL",
        "CUPTI_ACTIVITY_KIND_RUNTIME",
        "GPU_METRICS",
        "NVTX_EVENTS",
        "StringIds",
        "TARGET_INFO_GPU_METRICS",
    }
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    missing = sorted(required - tables)
    if missing:
        raise ValueError(f"Nsight SQLite is missing required tables: {missing}")
    return connection


def _capture_scope(connection: sqlite3.Connection) -> dict[str, int]:
    kernel_scopes = connection.execute(
        """
        SELECT DISTINCT deviceId, contextId
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        ORDER BY deviceId, contextId
        """
    ).fetchall()
    if len(kernel_scopes) != 1:
        scopes = [tuple(int(value) for value in row) for row in kernel_scopes]
        raise ValueError(
            "analysis requires exactly one CUDA kernel device/context; "
            f"found {scopes}"
        )
    metric_scopes = connection.execute(
        """
        SELECT DISTINCT metric.typeId, metadata.sourceId
        FROM GPU_METRICS AS metric
        JOIN TARGET_INFO_GPU_METRICS AS metadata
          ON metadata.typeId = metric.typeId
         AND metadata.metricId = metric.metricId
        ORDER BY metric.typeId, metadata.sourceId
        """
    ).fetchall()
    if len(metric_scopes) != 1:
        scopes = [tuple(int(value) for value in row) for row in metric_scopes]
        raise ValueError(
            "analysis requires exactly one GPU metric type/source; "
            f"found {scopes}"
        )
    return {
        "kernel_device_id": int(kernel_scopes[0]["deviceId"]),
        "kernel_context_id": int(kernel_scopes[0]["contextId"]),
        "metric_type_id": int(metric_scopes[0]["typeId"]),
        "metric_source_id": int(metric_scopes[0]["sourceId"]),
    }


def _nvtx_ranges(
    connection: sqlite3.Connection,
    name: str,
) -> list[tuple[int, int]]:
    rows = connection.execute(
        """
        SELECT event.start, event.end
        FROM NVTX_EVENTS AS event
        LEFT JOIN StringIds AS strings ON strings.id = event.textId
        WHERE COALESCE(event.text, strings.value) = ?
          AND event.end IS NOT NULL
        ORDER BY event.start, event.end
        """,
        (name,),
    ).fetchall()
    if not rows:
        raise ValueError(f"capture does not contain a completed {name!r} range")
    return [(int(row["start"]), int(row["end"])) for row in rows]


def _range_names(connection: sqlite3.Connection) -> list[str]:
    rows = connection.execute(
        """
        SELECT DISTINCT COALESCE(event.text, strings.value) AS name
        FROM NVTX_EVENTS AS event
        LEFT JOIN StringIds AS strings ON strings.id = event.textId
        WHERE event.end IS NOT NULL
          AND (
              COALESCE(event.text, strings.value) LIKE 'lowp/%'
              OR COALESCE(event.text, strings.value) IN (
                  'decoder_forward', 'ce_forward', 'backward_total',
                  'gradient_clip', 'optimizer'
              )
          )
        ORDER BY name
        """
    ).fetchall()
    return [str(row["name"]) for row in rows]


def _kernels_for_ranges(
    connection: sqlite3.Connection,
    ranges: Iterable[tuple[int, int]],
) -> list[dict[str, Any]]:
    kernels: list[dict[str, Any]] = []
    query = """
        SELECT
            kernel.start,
            kernel.end,
            kernel.deviceId,
            kernel.contextId,
            kernel.gridId,
            kernel.streamId,
            strings.value AS demangled_name,
            kernel.gridX,
            kernel.gridY,
            kernel.gridZ,
            kernel.blockX,
            kernel.blockY,
            kernel.blockZ,
            kernel.registersPerThread,
            kernel.staticSharedMemory,
            kernel.dynamicSharedMemory,
            runtime.start AS launch_start,
            runtime.end AS launch_end
        FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernel
        JOIN StringIds AS strings ON strings.id = kernel.demangledName
        JOIN CUPTI_ACTIVITY_KIND_RUNTIME AS runtime
          ON runtime.correlationId = kernel.correlationId
        WHERE runtime.start >= ? AND runtime.end <= ?
        ORDER BY runtime.start, runtime.end, kernel.gridId
    """
    for start, end in ranges:
        for row in connection.execute(query, (start, end)):
            kernels.append({key: row[key] for key in row.keys()})
    kernel_ids = [_kernel_identity(kernel) for kernel in kernels]
    if len(set(kernel_ids)) != len(kernel_ids):
        raise ValueError(
            "runtime-correlation query returned duplicate CUDA kernel IDs"
        )
    kernels.sort(
        key=lambda kernel: (
            int(kernel["launch_start"]),
            int(kernel["launch_end"]),
            int(kernel["gridId"]),
        )
    )
    return kernels


def _kernel_identity(kernel: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(kernel["deviceId"]),
        int(kernel["contextId"]),
        int(kernel["gridId"]),
    )


def _exact_partition_summary(
    reference: Sequence[tuple[Any, ...]],
    partitions: Iterable[Sequence[tuple[Any, ...]]],
) -> dict[str, Any]:
    reference_set = set(reference)
    if len(reference_set) != len(reference):
        raise ValueError("reference kernel identities are not unique")
    flattened = [identity for partition in partitions for identity in partition]
    flattened_set = set(flattened)
    disjoint = len(flattened_set) == len(flattened)
    union_matches = flattened_set == reference_set
    return {
        "partition_entry_count": len(flattened),
        "partition_unique_count": len(flattened_set),
        "partitions_are_disjoint": disjoint,
        "partition_union_matches_reference": union_matches,
        "exact_partition": disjoint and union_matches,
    }


def _kernel_signature(kernel: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(kernel["demangled_name"]),
        int(kernel["gridX"]),
        int(kernel["gridY"]),
        int(kernel["gridZ"]),
        int(kernel["blockX"]),
        int(kernel["blockY"]),
        int(kernel["blockZ"]),
        int(kernel["registersPerThread"]),
        int(kernel["staticSharedMemory"]),
        int(kernel["dynamicSharedMemory"]),
    )


def _signature_digest(signatures: Sequence[tuple[Any, ...]]) -> str:
    payload = json.dumps(
        signatures,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _merge_intervals(
    intervals: Iterable[tuple[int, int]],
) -> list[tuple[int, int]]:
    ordered = sorted((int(start), int(end)) for start, end in intervals)
    if not ordered:
        raise ValueError("cannot merge an empty interval collection")
    merged: list[list[int]] = []
    for start, end in ordered:
        if end < start:
            raise ValueError(f"invalid interval [{start}, {end}]")
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _execution_summary(kernels: Sequence[dict[str, Any]]) -> dict[str, Any]:
    merged = _merge_intervals(
        (int(kernel["start"]), int(kernel["end"])) for kernel in kernels
    )
    first_start = merged[0][0]
    last_end = merged[-1][1]
    span_ns = last_end - first_start
    busy_ns = sum(end - start for start, end in merged)
    gaps_ns = [
        merged[index + 1][0] - merged[index][1]
        for index in range(len(merged) - 1)
    ]
    return {
        "first_kernel_start_ns": first_start,
        "last_kernel_end_ns": last_end,
        "kernel_span_ms": span_ns / 1.0e6,
        "kernel_union_busy_ms": busy_ns / 1.0e6,
        "kernel_union_busy_percent": 100.0 * busy_ns / span_ns,
        "maximum_internal_gap_ms": max(gaps_ns, default=0) / 1.0e6,
        "merged_interval_count": len(merged),
    }


def _metric_summary(
    connection: sqlite3.Connection,
    metric_name: str,
    start: int,
    end: int,
) -> dict[str, Any]:
    rows = connection.execute(
        """
        SELECT metric.value
        FROM GPU_METRICS AS metric
        JOIN TARGET_INFO_GPU_METRICS AS metadata
          ON metadata.typeId = metric.typeId
         AND metadata.metricId = metric.metricId
        WHERE metadata.metricName = ?
          AND metric.timestamp >= ?
          AND metric.timestamp <= ?
        ORDER BY metric.timestamp
        """,
        (metric_name, start, end),
    ).fetchall()
    if not rows:
        raise ValueError(f"capture has no samples for GPU metric {metric_name!r}")
    scale = 1.0e6 if "Clock Frequency" in metric_name else 1.0
    values = [float(row["value"]) / scale for row in rows]
    return {
        "samples": len(values),
        "mean": statistics.fmean(values),
        "p50": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "unit": "MHz" if scale != 1.0 else "percent",
    }


def _range_summary(kernels: Sequence[dict[str, Any]]) -> dict[str, Any]:
    signatures = [_kernel_signature(kernel) for kernel in kernels]
    return {
        "kernel_count": len(kernels),
        "kernel_sum_ms": sum(
            int(kernel["end"]) - int(kernel["start"]) for kernel in kernels
        )
        / 1.0e6,
        "ordered_signature_sha256": _signature_digest(signatures),
    }


def analyze_capture(path: Path) -> dict[str, Any]:
    connection = _connect_read_only(path)
    try:
        capture_scope = _capture_scope(connection)
        profile_ranges = _nvtx_ranges(connection, PROFILE_RANGE)
        if len(profile_ranges) != 1:
            raise ValueError(
                f"expected one {PROFILE_RANGE!r} range, found {len(profile_ranges)}"
            )
        profile_kernels = _kernels_for_ranges(connection, profile_ranges)
        table_kernel_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL"
            ).fetchone()[0]
        )
        execution = _execution_summary(profile_kernels)
        metrics = {
            name: _metric_summary(
                connection,
                name,
                int(execution["first_kernel_start_ns"]),
                int(execution["last_kernel_end_ns"]),
            )
            for name in HEADLINE_METRICS
        }
        ranges: dict[str, Any] = {}
        signatures: dict[str, list[tuple[Any, ...]]] = {}
        for name in _range_names(connection):
            kernels = _kernels_for_ranges(connection, _nvtx_ranges(connection, name))
            range_summary = {
                "nvtx_range_count": len(_nvtx_ranges(connection, name)),
                **_range_summary(kernels),
            }
            if name in METRIC_RANGES:
                range_execution = _execution_summary(kernels)
                range_summary["execution"] = range_execution
                range_summary["metrics_over_first_to_last_kernel"] = {
                    metric_name: _metric_summary(
                        connection,
                        metric_name,
                        int(range_execution["first_kernel_start_ns"]),
                        int(range_execution["last_kernel_end_ns"]),
                    )
                    for metric_name in HEADLINE_METRICS
                }
            ranges[name] = range_summary
            signatures[name] = [_kernel_signature(kernel) for kernel in kernels]
        missing_top_level = sorted(set(TOP_LEVEL_RANGES) - set(ranges))
        if missing_top_level:
            raise ValueError(
                "capture is missing required top-level NVTX ranges: "
                f"{missing_top_level}"
            )
        top_level_partition = _exact_partition_summary(
            [_kernel_identity(kernel) for kernel in profile_kernels],
            (
                [_kernel_identity(kernel) for kernel in _kernels_for_ranges(
                    connection,
                    _nvtx_ranges(connection, name),
                )]
                for name in TOP_LEVEL_RANGES
            ),
        )
        profile_signatures = [
            _kernel_signature(kernel) for kernel in profile_kernels
        ]
        return {
            "artifact": _identity(path),
            "capture_scope": capture_scope,
            "profile": {
                "nvtx_range": PROFILE_RANGE,
                "nvtx_range_count": len(profile_ranges),
                "kernel_count": len(profile_kernels),
                "sqlite_kernel_count": table_kernel_count,
                "all_captured_kernels_attributed_to_profile_step": (
                    len(profile_kernels) == table_kernel_count
                ),
                "top_level_partition": top_level_partition,
                "top_level_ranges_close_profile": top_level_partition[
                    "exact_partition"
                ],
                "kernel_sum_ms": sum(
                    int(kernel["end"]) - int(kernel["start"])
                    for kernel in profile_kernels
                )
                / 1.0e6,
                "ordered_signature_sha256": _signature_digest(
                    profile_signatures
                ),
                **execution,
            },
            "metrics_over_first_to_last_kernel": metrics,
            "ranges": ranges,
            "_signatures": {
                PROFILE_RANGE: profile_signatures,
                **signatures,
            },
        }
    finally:
        connection.close()


def compare_captures(
    fp8: dict[str, Any],
    mx: dict[str, Any],
) -> dict[str, Any]:
    fp8_signatures = fp8["_signatures"]
    mx_signatures = mx["_signatures"]
    fp8_profile = fp8_signatures[PROFILE_RANGE]
    mx_profile = mx_signatures[PROFILE_RANGE]
    if len(fp8_profile) != len(mx_profile):
        differing_indices: list[int] | None = None
    else:
        differing_indices = [
            index
            for index, (fp8_signature, mx_signature) in enumerate(
                zip(fp8_profile, mx_profile)
            )
            if fp8_signature != mx_signature
        ]
    fp8_backward = fp8_signatures[BACKWARD_RANGE]
    mx_backward = mx_signatures[BACKWARD_RANGE]

    stage_deltas_ms = {
        name: (
            mx["ranges"][name]["kernel_sum_ms"]
            - fp8["ranges"][name]["kernel_sum_ms"]
        )
        for name in sorted(set(fp8["ranges"]) & set(mx["ranges"]))
    }
    fp8_clock = fp8["metrics_over_first_to_last_kernel"][
        "GPC Clock Frequency [MHz]"
    ]["mean"]
    mx_clock = mx["metrics_over_first_to_last_kernel"][
        "GPC Clock Frequency [MHz]"
    ]["mean"]
    return {
        "profile_kernel_counts_equal": len(fp8_profile) == len(mx_profile),
        "profile_signature_difference_count": (
            None if differing_indices is None else len(differing_indices)
        ),
        "profile_signature_difference_indices": differing_indices,
        "backward_kernel_counts": {
            "fp8": len(fp8_backward),
            "mx": len(mx_backward),
        },
        "backward_ordered_signatures_equal": fp8_backward == mx_backward,
        "backward_ordered_signature_sha256": (
            _signature_digest(fp8_backward)
            if fp8_backward == mx_backward
            else None
        ),
        "stage_kernel_sum_delta_mx_minus_fp8_ms": stage_deltas_ms,
        "mean_gpc_clock_mhz": {"fp8": fp8_clock, "mx": mx_clock},
        "mean_gpc_clock_ratio_fp8_over_mx": fp8_clock / mx_clock,
    }


def _strip_private_signatures(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "_signatures"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp8-sqlite", type=Path, required=True)
    parser.add_argument("--mx-sqlite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fp8 = analyze_capture(args.fp8_sqlite)
    mx = analyze_capture(args.mx_sqlite)
    result = {
        "schema": "saturated_nsys_pair_analysis_v1",
        "attribution": (
            "CUDA kernels are attributed through runtime correlation IDs "
            "whose launch calls occur inside CPU-side NVTX ranges"
        ),
        "metric_window": "first captured kernel start to last kernel end",
        "fp8": _strip_private_signatures(fp8),
        "mx": _strip_private_signatures(mx),
        "comparison": compare_captures(fp8, mx),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
