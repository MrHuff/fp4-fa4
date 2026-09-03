#!/usr/bin/env python3
"""Regenerate the Wan downstream table from recorded paired evaluations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
CALIBRATION = HERE.parent / "fp4_fa4_wan_affine_calibration_20260806"

ROWS = [
    {
        "model": "Wan2.1-1.3B",
        "steps": 1,
        "output": "latent",
        "calibrated": (CALIBRATION / "wan1p3b_final_step1.json", "tk"),
        "accurate": (HERE / "wan21_1p3b_s7680_accurate_a128_l27_29_step1.json", "tk"),
        "hao": (HERE / "wan21_1p3b_s7680_hao_step1.json", None),
    },
    {
        "model": "Wan2.1-1.3B",
        "steps": 4,
        "output": "latent",
        "calibrated": (CALIBRATION / "wan1p3b_l0_l11_train4.json", "tk"),
        "accurate": (HERE / "wan21_1p3b_s7680_accurate_a128_l27_29_step4.json", "tk"),
        "hao": (HERE / "wan21_1p3b_s7680_hao_step4.json", None),
    },
    {
        "model": "Wan2.1-1.3B",
        "steps": 20,
        "output": "decoded",
        "calibrated": (
            CALIBRATION / "wan1p3b_final_step20_decoded.json",
            "tk",
        ),
        "accurate": (
            HERE / "wan21_1p3b_s7680_accurate_a128_l27_29_step20_decoded.json",
            "tk",
        ),
        "hao": (HERE / "wan21_1p3b_s7680_hao_step20_exact.json", None),
    },
    {
        "model": "Wan2.1-14B",
        "steps": 1,
        "output": "latent",
        "calibrated": (CALIBRATION / "wan14b_final_step1.json", "tk"),
        "accurate": (HERE / "wan21_14b_s7680_accurate_qk_guard_step1.json", "tk"),
        "hao": (HERE / "wan21_14b_s7680_hao_step1.json", None),
    },
    {
        "model": "Wan2.1-14B",
        "steps": 4,
        "output": "latent",
        "calibrated": (CALIBRATION / "wan14b_final_step4.json", "tk"),
        "accurate": (
            HERE / "wan21_14b_s7680_accurate_anchor128_l33_34_38_39_step4.json",
            "tk",
        ),
        "hao": (HERE / "wan21_14b_s7680_hao_step4.json", None),
    },
    {
        "model": "Wan2.1-14B",
        "steps": 20,
        "output": "latent",
        "calibrated": (
            CALIBRATION / "wan14b_affine_calibrated_train20.json",
            "calibrated",
        ),
        "accurate": (HERE / "wan21_14b_s7680_accurate_qk_guard_step20.json", "tk"),
        "hao": (HERE / "wan21_14b_s7680_hao_step20.json", None),
    },
]


def comparison(path: Path, provider: str) -> dict[str, float]:
    payload = json.loads(path.read_text())
    record = payload["comparisons"][f"{provider}_vs_bf16"]
    if payload["providers"][provider].get("status", "complete") != "complete":
        raise ValueError(f"incomplete provider {provider} in {path}")
    return record


def cell(record: dict[str, float]) -> str:
    return f'{record["cosine"]:.5f} / {record["relative_l2"]:.5f}'


def main() -> None:
    tex_rows = []
    summary: dict[str, Any] = {
        "schema": "tk_fp4_fa4_wan_calibrated_downstream_v1",
        "calibrated_routes": {
            "Wan2.1-1.3B": {
                "base": {"a": 1.60, "b": 0.95},
                "overrides": {
                    "0": {"a": 1.625, "b": 0.95},
                    "11": {"a": 1.575, "b": 1.05},
                },
                "guard_layers": [27, 28, 29],
            },
            "Wan2.1-14B": {
                "base": {"a": 1.60, "b": 0.95},
                "override": {"a": 1.575, "b": 1.05},
                "override_layers": [
                    1, 3, 6, 8, 9, 10, 11, 12, 15, 16, 17, 22, 23, 24,
                    25, 26, 27, 30, 31, 35,
                ],
                "guard_layers": [33, 34, 38, 39],
            },
        },
        "rows": [],
    }
    for specification in ROWS:
        calibrated_path, calibrated_provider = specification["calibrated"]
        accurate_path, accurate_provider = specification["accurate"]
        hao_path, _ = specification["hao"]
        calibrated = comparison(calibrated_path, calibrated_provider)
        accurate = comparison(accurate_path, accurate_provider)
        hao_nvnv = comparison(hao_path, "hao-native")
        hao_fp8 = comparison(hao_path, "hao-fp8")
        tex_rows.append(
            f'{specification["model"]} & {specification["steps"]} & '
            f'{specification["output"]} & {cell(calibrated)} & '
            f'{cell(accurate)} & {cell(hao_nvnv)} & {cell(hao_fp8)} \\\\'
        )
        summary["rows"].append(
            {
                "model": specification["model"],
                "steps": specification["steps"],
                "output": specification["output"],
                "calibrated": calibrated,
                "accurate": accurate,
                "hao_nvfp4_nvfp4": hao_nvnv,
                "hao_nvfp4_fp8": hao_fp8,
                "sources": {
                    "calibrated": str(calibrated_path.relative_to(HERE.parent)),
                    "accurate": str(accurate_path.relative_to(HERE.parent)),
                    "hao": str(hao_path.relative_to(HERE.parent)),
                },
            }
        )
    (HERE / "wan_downstream_rows.tex").write_text("\n".join(tex_rows) + "\n")
    (CALIBRATION / "wan_calibrated_downstream_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
