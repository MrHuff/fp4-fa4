#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
PREFIX = "forward_issue_lane_overlap_bf16_matrix_20260711"
RETAINED_FP4 = {"stage2", "e16pc", "fp4_auto"}
ISOLATED_MANIFESTS = (
    f"{PREFIX}_isolated_bf16_short.json",
    f"{PREFIX}_isolated_bf16_long.json",
    f"{PREFIX}_isolated_fp4_short.json",
    f"{PREFIX}_isolated_fp4_s8192.json",
    f"{PREFIX}_isolated_fp4_long.json",
)


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def shape_key(shape: dict[str, Any]) -> tuple[int, int, int]:
    return int(shape["batch"]), int(shape["seqlen"]), int(shape["heads"])


def route_summary(record: dict[str, Any], source: str) -> dict[str, Any]:
    timing = record.get("timing_ms", {})
    return {
        "name": record.get("name"),
        "family": record.get("family"),
        "implementation": record.get("implementation"),
        "launch_mode": record.get("launch_mode"),
        "config": record.get("config"),
        "timing_ms": {
            key: timing.get(key)
            for key in ("count", "p50", "p25", "p75", "min", "max")
        },
        "finite": bool(record.get("finite", False)),
        "correct": bool(record.get("correct", False)),
        "deterministic": bool(record.get("deterministic", False)),
        "complete_timing": bool(record.get("complete_timing", False)),
        "eligible": bool(record.get("eligible", False)),
        "unsupported": bool(record.get("unsupported", False)),
        "error": record.get("error"),
        "reason": record.get("reason"),
        "source": source,
    }


def route_usable(route: dict[str, Any], *, strict: bool) -> bool:
    timing = route.get("timing_ms", {})
    base = (
        route.get("finite", False)
        and route.get("correct", False)
        and route.get("complete_timing", False)
        and timing.get("p50") is not None
    )
    return bool(base and (route.get("deterministic", False) if strict else True))


def comparison(
    routes: dict[str, dict[str, Any]],
    *,
    strict: bool,
) -> dict[str, Any] | None:
    fp4 = [
        route
        for route in routes.values()
        if route.get("family") == "fp4" and route_usable(route, strict=strict)
    ]
    bf16 = [
        route
        for route in routes.values()
        if route.get("family") == "bf16" and route_usable(route, strict=strict)
    ]
    if not fp4 or not bf16:
        return None
    fp4_best = min(fp4, key=lambda route: float(route["timing_ms"]["p50"]))
    bf16_best = min(bf16, key=lambda route: float(route["timing_ms"]["p50"]))
    fp4_ms = float(fp4_best["timing_ms"]["p50"])
    bf16_ms = float(bf16_best["timing_ms"]["p50"])
    speedup = bf16_ms / fp4_ms
    status = (
        "robust_win"
        if speedup > 1.02
        else "win"
        if speedup > 1.0
        else "parity"
        if speedup >= 0.98
        else "loss"
    )
    return {
        "fp4_route": fp4_best["name"],
        "fp4_launch_mode": fp4_best["launch_mode"],
        "fp4_p50_ms": fp4_ms,
        "bf16_route": bf16_best["name"],
        "bf16_launch_mode": bf16_best["launch_mode"],
        "bf16_p50_ms": bf16_ms,
        "speedup": speedup,
        "margin_us": (bf16_ms - fp4_ms) * 1000.0,
        "status": status,
    }


def add_isolated(
    cells: dict[tuple[int, int, int], dict[str, Any]],
    manifest_path: Path,
) -> None:
    manifest = load(manifest_path)
    for item in manifest.get("results", []):
        key = shape_key(item["shape"])
        cell = cells[key]
        route_name = item["route"]
        source = manifest_path.name
        payload = item.get("payload") or {}
        record = payload.get("records", {}).get(route_name)
        if record is not None:
            route = route_summary(record, source)
            route["isolation_status"] = item["status"]
            route["correctness_verified"] = bool(
                route["family"] == "bf16" or record.get("vs_bf16_reference")
            )
            if route["family"] == "fp4" and not route["correctness_verified"]:
                route["correct"] = False
                route["eligible"] = False
        else:
            family = "bf16" if "bf16" in route_name else "fp4"
            route = {
                "name": route_name,
                "family": family,
                "launch_mode": item.get("fp4_launch_mode", "auto"),
                "timing_ms": {"p50": None},
                "finite": False,
                "correct": False,
                "deterministic": False,
                "complete_timing": False,
                "eligible": False,
                "isolation_status": item["status"],
                "error": (
                    f"bounded process timeout after {item.get('timeout_s')}s"
                    if item["status"] == "timeout"
                    else f"isolated process {item['status']}"
                ),
                "source": source,
            }
        existing = cell["routes"].get(route_name)
        if existing is None or not route_usable(existing, strict=False):
            cell["routes"][route_name] = route
        cell["source_artifacts"].add(source)


def add_fullgrid_batch4(
    cells: dict[tuple[int, int, int], dict[str, Any]],
    path: Path,
) -> None:
    manifest = load(path)
    for item in manifest.get("results", []):
        if shape_key(item["shape"]) != (4, 4096, 4):
            continue
        payload = item.get("payload") or {}
        record = payload.get("records", {}).get(item["route"])
        if record is None:
            continue
        key = f"{item['route']}@fullgrid"
        route = route_summary(record, path.name)
        route["name"] = key
        route["correctness_verified"] = bool(record.get("vs_bf16_reference"))
        if not route["correctness_verified"]:
            route["correct"] = False
            route["eligible"] = False
        cells[(4, 4096, 4)]["routes"][key] = route
        cells[(4, 4096, 4)]["source_artifacts"].add(path.name)


def annotate_pair_checks(cells: dict[tuple[int, int, int], dict[str, Any]]) -> None:
    for path in HERE.glob(f"{PREFIX}_paircheck_*.json"):
        payload = load(path)
        key = shape_key(payload["shape"])
        for name, record in payload.get("records", {}).items():
            if record.get("family") != "fp4":
                continue
            check = {
                "finite": record.get("finite"),
                "correct": record.get("correct"),
                "deterministic": record.get("deterministic"),
                "vs_bf16_reference": record.get("vs_bf16_reference"),
                "source": path.name,
            }
            matched = False
            for route_name, route in cells[key]["routes"].items():
                if route_name.split("@", 1)[0] == name:
                    route["paired_correctness"] = check
                    route["correctness_verified"] = True
                    route["correct"] = bool(check["correct"])
                    route["deterministic"] = bool(
                        route.get("deterministic", False)
                        and check["deterministic"]
                    )
                    route["eligible"] = route_usable(route, strict=True)
                    matched = True
            if not matched:
                continue
            cells[key]["source_artifacts"].add(path.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        type=Path,
        default=HERE / f"{PREFIX}_summary.json",
    )
    args = parser.parse_args()
    matrix_path = HERE / f"{PREFIX}_matrix.json"
    matrix = load(matrix_path)
    cells: dict[tuple[int, int, int], dict[str, Any]] = {}
    for raw_cell in matrix["cells"]:
        key = shape_key(raw_cell["shape"])
        routes = {}
        for name, record in raw_cell.get("records", {}).items():
            if name in RETAINED_FP4 or record.get("family") == "bf16":
                routes[name] = route_summary(record, matrix_path.name)
        cells[key] = {
            "shape": {
                "batch": key[0],
                "seqlen": key[1],
                "heads": key[2],
                "dqk": 192,
                "dvo": 128,
                "causal": True,
            },
            "routes": routes,
            "original_worker_error": raw_cell.get("worker_error"),
            "source_artifacts": {matrix_path.name},
        }

    for name in ISOLATED_MANIFESTS:
        add_isolated(cells, HERE / name)
    add_fullgrid_batch4(cells, HERE / f"{PREFIX}_gap_highload_fullgrid.json")

    confirm_path = HERE / f"{PREFIX}_confirm60_b1_s1024_h32.json"
    confirm = load(confirm_path)
    confirm_key = shape_key(confirm["shape"])
    for name, record in confirm["records"].items():
        if name in RETAINED_FP4 or record.get("family") == "bf16":
            cells[confirm_key]["routes"][name] = route_summary(record, confirm_path.name)
    cells[confirm_key]["source_artifacts"].add(confirm_path.name)
    annotate_pair_checks(cells)

    output_cells = []
    lists: dict[str, list[str]] = {
        "robust_win": [],
        "win": [],
        "parity": [],
        "loss": [],
        "no_finite_fp4": [],
        "strict_unresolved": [],
    }
    heatmap: dict[str, dict[str, Any]] = {}
    for key in sorted(cells):
        cell = cells[key]
        measured = comparison(cell["routes"], strict=False)
        strict = comparison(cell["routes"], strict=True)
        label = f"b{key[0]}/s{key[1]}/h{key[2]}"
        if measured is None:
            status = "no_finite_fp4"
        else:
            status = measured["status"]
        lists[status].append(label)
        if strict is None:
            lists["strict_unresolved"].append(label)
        cell["measured_comparison"] = measured
        cell["strict_comparison"] = strict
        cell["status"] = status
        cell["source_artifacts"] = sorted(cell["source_artifacts"])
        output_cells.append(cell)
        if key[0] == 1 and key[1] <= 16384 and key[2] <= 32:
            heatmap.setdefault(str(key[1]), {})[str(key[2])] = {
                "status": status,
                "speedup": measured.get("speedup") if measured else None,
                "margin_us": measured.get("margin_us") if measured else None,
                "strict": strict is not None,
            }

    primary_labels = {
        f"b1/s{seqlen}/h{heads}"
        for seqlen in (128, 256, 512, 1024, 2048, 4096, 8192, 16384)
        for heads in (1, 2, 4, 8, 16, 32)
    }
    all_primary_strict_wins = True
    for cell in output_cells:
        label = (
            f"b{cell['shape']['batch']}/s{cell['shape']['seqlen']}/"
            f"h{cell['shape']['heads']}"
        )
        if label not in primary_labels:
            continue
        strict = cell["strict_comparison"]
        if strict is None or strict["speedup"] <= 1.0:
            all_primary_strict_wins = False
            break

    output = {
        "task": "issue-lane overlap and broad fastest-BF16 consolidated matrix",
        "contract": {
            "kernel_only": True,
            "quantization_timed": False,
            "preallocated_outputs": True,
            "dqk": 192,
            "dvo": 128,
            "causal": True,
            "strict_requires_finite_correct_deterministic_complete": True,
            "retained_fp4_routes": sorted(RETAINED_FP4),
        },
        "cell_count": len(output_cells),
        "primary_cell_count": len(primary_labels),
        "product_objective_met": all_primary_strict_wins,
        "counts": {name: len(values) for name, values in lists.items()},
        "lists": lists,
        "primary_heatmap": heatmap,
        "cells": output_cells,
    }
    args.json.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: output[key] for key in ("cell_count", "counts", "product_objective_met")}, indent=2))


if __name__ == "__main__":
    main()
