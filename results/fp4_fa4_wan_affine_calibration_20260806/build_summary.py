#!/usr/bin/env python3
"""Summarize Wan layer-wise affine E2M1 boundary calibration runs."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any


HERE = Path(__file__).resolve().parent
GRID_PATTERN = "wan14b_teacherforced_grid_1step_g*.json"
BASE = "a160b095"
GUARD_LAYERS = {33, 34, 38, 39}
MEANINGFUL_IMPROVEMENT = 1e-4
CANDIDATES = {
    "a1575b105": {"a": 1.575, "b": 1.05},
    "a159b095": {"a": 1.59, "b": 0.95},
    "a160b0925": {"a": 1.60, "b": 0.925},
    "a160b095": {"a": 1.60, "b": 0.95},
    "a160b0975": {"a": 1.60, "b": 0.975},
    "a160b100": {"a": 1.60, "b": 1.00},
    "a1605b095": {"a": 1.605, "b": 0.95},
    "a1625b095": {"a": 1.625, "b": 0.95},
}
PROMOTED_LAYERS = [
    1,
    3,
    6,
    *range(8, 13),
    *range(15, 18),
    *range(22, 28),
    30,
    31,
    35,
]
WAN14B_FAST_ROUTED_MS = 0.421104
WAN14B_TK_BF16_MS = 0.760160
WAN14B_HAO_FP8_MS = 0.751648


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def teacher_forced_grid() -> dict[int, dict[str, dict[str, float]]]:
    records: dict[int, dict[str, dict[str, float]]] = {}
    expression = re.compile(r"(a\d+b\d+)_l(\d+)_vs_bf16")
    for path in sorted(HERE.glob(GRID_PATTERN)):
        for name, metrics in load(path)["comparisons"].items():
            match = expression.fullmatch(name)
            if match is None:
                continue
            candidate, layer_text = match.groups()
            records.setdefault(int(layer_text), {})[candidate] = metrics
    expected = set(range(40)) - GUARD_LAYERS
    if set(records) != expected:
        raise ValueError(
            f"teacher-forced grid has layers {sorted(records)}, expected {sorted(expected)}"
        )
    for layer, candidates in records.items():
        if set(candidates) != set(CANDIDATES):
            raise ValueError(f"layer {layer} has incomplete candidate coverage")
    return records


def route_samples() -> dict[str, dict[str, list[dict[str, float]] | int]]:
    specifications = {
        "calibration_prompt": [
            (
                "wan14b_teacherforced_routes_train4.json",
                ["baseline"],
                ["conservative"],
            ),
            (
                "wan14b_affine_repeat_train4.json",
                ["baseline1", "baseline2"],
                ["tuned1", "tuned2"],
            ),
            (
                "wan14b_gridcal_routes_train4.json",
                ["baseline"],
                ["conservative"],
            ),
        ],
        "held_out_prompt": [
            (
                "wan14b_teacherforced_routes_holdout4.json",
                ["baseline"],
                ["conservative"],
            ),
            (
                "wan14b_affine_repeat_holdout4.json",
                ["baseline1", "baseline2"],
                ["tuned1", "tuned2"],
            ),
            (
                "wan14b_gridcal_routes_holdout4.json",
                ["baseline"],
                ["conservative"],
            ),
        ],
    }
    result = {}
    for split, entries in specifications.items():
        split_result: dict[str, Any] = {
            "global": [],
            "calibrated": [],
            "global_attempts": 0,
            "calibrated_attempts": 0,
        }
        for filename, global_names, calibrated_names in entries:
            payload = load(HERE / filename)
            for route, names in (
                ("global", global_names),
                ("calibrated", calibrated_names),
            ):
                for name in names:
                    split_result[f"{route}_attempts"] += 1
                    if payload["providers"][name]["status"] == "complete":
                        split_result[route].append(
                            payload["comparisons"][f"{name}_vs_bf16"]
                        )
        result[split] = split_result
    return result


def aggregate(
    records: list[dict[str, float]], attempts: int
) -> dict[str, float | int]:
    def stats(key: str) -> tuple[float, float]:
        values = [record[key] for record in records]
        return statistics.mean(values), statistics.pstdev(values)

    cosine_mean, cosine_std = stats("cosine")
    relative_l2_mean, relative_l2_std = stats("relative_l2")
    return {
        "finite": len(records),
        "attempts": attempts,
        "cosine_mean": cosine_mean,
        "cosine_std": cosine_std,
        "relative_l2_mean": relative_l2_mean,
        "relative_l2_std": relative_l2_std,
    }


def optional_long_runs() -> dict[str, Any]:
    result = {}
    for split in ("train", "holdout"):
        path = HERE / f"wan14b_affine_calibrated_{split}20.json"
        if not path.exists():
            continue
        payload = load(path)
        result[split] = {
            name: {
                "status": payload["providers"][name]["status"],
                "metrics": payload["comparisons"].get(f"{name}_vs_bf16"),
            }
            for name in ("global", "calibrated")
        }
    return result


def write_table(repeats: dict[str, Any]) -> None:
    labels = {
        "calibration_prompt": "Calibration prompt",
        "held_out_prompt": "Held-out prompt",
    }
    rows = []
    for split in ("calibration_prompt", "held_out_prompt"):
        for route, route_label in (
            ("global", "Global $A,B$"),
            ("calibrated", "Layer calibrated"),
        ):
            record = repeats[split][route]
            rows.append(
                f'{labels[split]} & {route_label} & '
                f'{record["finite"]}/{record["attempts"]} & '
                f'{record["cosine_mean"]:.6f} $\\pm$ {record["cosine_std"]:.6f} & '
                f'{record["relative_l2_mean"]:.6f} $\\pm$ '
                f'{record["relative_l2_std"]:.6f} & '
                f'{WAN14B_FAST_ROUTED_MS:.6f} & '
                f'{WAN14B_TK_BF16_MS / WAN14B_FAST_ROUTED_MS:.2f}$\\times$ & '
                f'{WAN14B_HAO_FP8_MS / WAN14B_FAST_ROUTED_MS:.2f}$\\times$ \\\\'
            )
    table_dir = HERE / "tables"
    table_dir.mkdir(exist_ok=True)
    (table_dir / "wan_affine_calibration_rows.tex").write_text(
        "\n".join(rows) + "\n"
    )


def main() -> None:
    grid = teacher_forced_grid()
    per_layer = {}
    winner_counts: Counter[str] = Counter()
    for layer, candidates in sorted(grid.items()):
        winner = min(candidates, key=lambda name: candidates[name]["relative_l2"])
        winner_counts[winner] += 1
        per_layer[str(layer)] = {
            "winner": winner,
            "base_relative_l2": candidates[BASE]["relative_l2"],
            "winner_relative_l2": candidates[winner]["relative_l2"],
            "relative_l2_improvement": (
                candidates[BASE]["relative_l2"]
                - candidates[winner]["relative_l2"]
            ),
            "winner_cosine_improvement": (
                candidates[winner]["cosine"] - candidates[BASE]["cosine"]
            ),
        }

    samples = route_samples()
    repeats = {
        split: {
            route: aggregate(
                records,
                int(samples[split][f"{route}_attempts"]),
            )
            for route, records in (
                ("global", samples[split]["global"]),
                ("calibrated", samples[split]["calibrated"]),
            )
        }
        for split in samples
    }
    payload = {
        "schema": "tk_fp4_fa4_wan_affine_calibration_v1",
        "interpretation": (
            "A and B calibrate scaled 2^x-to-E2M1 code boundaries; they do not "
            "replace or approximate the softmax normalization itself."
        ),
        "model": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        "shape": {"batch": 1, "sequence": 7680, "heads": 40, "dim": 128},
        "candidates": CANDIDATES,
        "guard_layers_excluded": sorted(GUARD_LAYERS),
        "teacher_forced": {
            "steps": 1,
            "layers_preferring_nonbase": sum(
                record["winner"] != BASE for record in per_layer.values()
            ),
            "layers_tested": len(per_layer),
            "meaningful_improvement_threshold": MEANINGFUL_IMPROVEMENT,
            "layers_with_meaningful_improvement": sum(
                record["relative_l2_improvement"] > MEANINGFUL_IMPROVEMENT
                for record in per_layer.values()
            ),
            "winner_counts": dict(sorted(winner_counts.items())),
            "per_layer": per_layer,
        },
        "promoted_two_code_route": {
            "base": {"label": BASE, **CANDIDATES[BASE]},
            "override": {"label": "a1575b105", **CANDIDATES["a1575b105"]},
            "override_layers": PROMOTED_LAYERS,
            "runtime_cost": (
                "No kernel instruction or resource increase; constants and "
                "static layer dispatch change only."
            ),
        },
        "kernel_speed": {
            "routed_ms": WAN14B_FAST_ROUTED_MS,
            "tk_bf16_ms": WAN14B_TK_BF16_MS,
            "hao_nvfp4_fp8_ms": WAN14B_HAO_FP8_MS,
            "speedup_vs_tk_bf16": WAN14B_TK_BF16_MS
            / WAN14B_FAST_ROUTED_MS,
            "speedup_vs_hao_nvfp4_fp8": WAN14B_HAO_FP8_MS
            / WAN14B_FAST_ROUTED_MS,
            "calibrated_route_timing": (
                "Same warmed routed time as the global fast route: only "
                "compile-time affine constants and static layer selection change."
            ),
        },
        "four_step_repeat_summary": repeats,
        "unrestricted_route_warning": (
            "Selecting a different winner independently at every layer overfits "
            "the one-step teacher-forced objective and is worse end to end."
        ),
        "twenty_step": optional_long_runs(),
    }
    (HERE / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    write_table(repeats)


if __name__ == "__main__":
    main()
