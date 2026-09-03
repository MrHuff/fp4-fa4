#!/usr/bin/env python3
"""Build the Wan CuTe-BF16 quality and speed tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
TABLES = HERE / "tables"
GUARD_FIX = HERE.parent / "fp4_fa4_wan_guard_fix_20260806"

MODELS = {
    "Wan2.1-1.3B": {
        "stem": "wan1p3b",
        "timing_ms": {
            "hao-bf16": 0.288768,
            "fast": 0.165309,
            "accurate": 0.196051,
            "hao-native": 0.346624,
            "hao-fp8": 0.288768,
        },
    },
    "Wan2.1-14B": {
        "stem": "wan14b",
        "timing_ms": {
            "hao-bf16": 0.888160,
            "fast": 0.424995,
            "accurate": 0.507158,
            "hao-native": 0.907264,
            "hao-fp8": 0.751648,
        },
    },
}

METHODS = (
    ("hao-bf16", "HAO CuTe BF16", "reference"),
    ("fast", "TK NV/MX fast", "fixed"),
    ("accurate", "TK NV/MX accurate", "tk"),
    ("hao-native", "HAO NV/NV", "hao"),
    ("hao-fp8", "HAO NV/FP8", "hao"),
)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def method_record(
    stem: str,
    steps: int,
    method: str,
    source: str,
) -> dict[str, Any]:
    if source == "reference":
        return {
            "finite": True,
            "cosine": 1.0,
            "relative_l2": 0.0,
        }
    if source == "fixed":
        payload = load(GUARD_FIX / f"{stem}_calibration_step{steps}.json")
        provider = "fast"
    else:
        payload = load(HERE / f"{stem}_{source}_step{steps}.json")
        provider = method
    status = payload["providers"][provider].get("status", "complete")
    comparison = payload["comparisons"].get(
        f"{provider}_vs_hao-bf16"
    )
    if status != "complete" or comparison is None:
        return {
            "finite": False,
            "error": payload["providers"][provider].get("error"),
        }
    return {
        "finite": True,
        "cosine": comparison["cosine"],
        "relative_l2": comparison["relative_l2"],
    }


def metric_cell(record: dict[str, Any]) -> str:
    if not record["finite"]:
        return r"\textit{non-finite}"
    return f'{record["cosine"]:.4f} / {record["relative_l2"]:.4f}'


def affine_record(path: Path, provider: str) -> dict[str, Any]:
    payload = load(path)
    status = payload["providers"][provider].get("status", "complete")
    comparison = payload["comparisons"].get(
        f"{provider}_vs_hao-bf16"
    )
    if status != "complete" or comparison is None:
        return {"finite": False}
    return {
        "finite": True,
        "cosine": comparison["cosine"],
        "relative_l2": comparison["relative_l2"],
    }


def main() -> None:
    TABLES.mkdir(exist_ok=True)
    quality_rows = []
    summary: dict[str, Any] = {
        "schema": "tk_fp4_fa4_wan_cute_bf16_tables_v1",
        "reference": "HAO CuTe-DSL BF16 FlashAttention-4",
        "quality_speed": [],
        "affine_control": [],
    }
    for model, specification in MODELS.items():
        stem = specification["stem"]
        timing = specification["timing_ms"]
        baseline_ms = timing["hao-bf16"]
        for method, label, source in METHODS:
            records = {
                steps: method_record(stem, steps, method, source)
                for steps in (1, 4, 20)
            }
            speedup = baseline_ms / timing[method]
            quality_rows.append(
                f"{model} & {label} & {timing[method]:.4f} & "
                f"{speedup:.2f}$\\times$ & {metric_cell(records[1])} & "
                f"{metric_cell(records[4])} & {metric_cell(records[20])} \\\\"
            )
            summary["quality_speed"].append(
                {
                    "model": model,
                    "method": method,
                    "label": label,
                    "timing_ms": timing[method],
                    "speedup_vs_hao_bf16": speedup,
                    "metrics": {str(step): record for step, record in records.items()},
                }
            )
        quality_rows.append(r"\addlinespace")
    quality_rows.pop()
    (TABLES / "wan_quality_speed_rows.tex").write_text(
        "\n".join(quality_rows) + "\n"
    )

    affine_sources = (
        (
            "Calibration",
            "Global $A,B$",
            HERE / "wan14b_affine_calibration_step4.json",
            "global",
        ),
        (
            "Calibration",
            "Layer calibrated",
            HERE / "wan14b_tk_step4.json",
            "calibrated",
        ),
        (
            "Held-out",
            "Global $A,B$",
            HERE / "wan14b_affine_holdout_global_step4.json",
            "global",
        ),
        (
            "Held-out",
            "Layer calibrated",
            HERE / "wan14b_affine_holdout_calibrated_step4.json",
            "calibrated",
        ),
    )
    baseline_ms = MODELS["Wan2.1-14B"]["timing_ms"]["hao-bf16"]
    route_ms = MODELS["Wan2.1-14B"]["timing_ms"]["fast"]
    affine_rows = []
    for split in ("Calibration", "Held-out"):
        affine_rows.append(
            f"{split} & HAO CuTe BF16 & {baseline_ms:.4f} & "
            r"1.00$\times$ & yes & 1.0000 & 0.0000 \\"
        )
        summary["affine_control"].append(
            {
                "split": split,
                "route": "hao-bf16",
                "timing_ms": baseline_ms,
                "speedup_vs_hao_bf16": 1.0,
                "finite": True,
                "cosine": 1.0,
                "relative_l2": 0.0,
            }
        )
        for source_split, label, path, provider in affine_sources:
            if source_split != split:
                continue
            record = affine_record(path, provider)
            finite = "yes" if record["finite"] else "no"
            cosine = f'{record["cosine"]:.4f}' if record["finite"] else "--"
            relative_l2 = (
                f'{record["relative_l2"]:.4f}' if record["finite"] else "--"
            )
            speedup = baseline_ms / route_ms
            affine_rows.append(
                f"{split} & {label} & {route_ms:.4f} & "
                f"{speedup:.2f}$\\times$ & {finite} & {cosine} & "
                f"{relative_l2} \\\\"
            )
            summary["affine_control"].append(
                {
                    "split": split,
                    "route": provider,
                    "label": label,
                    "timing_ms": route_ms,
                    "speedup_vs_hao_bf16": speedup,
                    **record,
                    "source": str(path.relative_to(HERE.parent)),
                }
            )
        if split == "Calibration":
            affine_rows.append(r"\addlinespace")
    (TABLES / "wan_affine_rows.tex").write_text(
        "\n".join(affine_rows) + "\n"
    )
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
