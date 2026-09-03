#!/usr/bin/env python3
"""Build paired ViT-MAE reconstruction summaries and LaTeX rows."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics


HERE = Path(__file__).resolve().parent
UNIFIED = HERE.parent / "fp4_fa4_unified_20260801" / "summary.json"
PROVIDERS = {
    "HAO NV/FP8": "hao_nvfp8_100.json",
    "HAO NV/NV": "hao_nvnv_100.json",
    "TK NV/MX accurate": "nvmx_accurate_100.json",
    "TK NV/MX fast": "nvmx_fast_100.json",
}
UNIFIED_NAMES = {
    "HAO NV/FP8": "HAO NV/FP8",
    "HAO NV/NV": "HAO NV/NV",
    "TK NV/MX accurate": "TK NV/MX accurate",
    "TK NV/MX fast": "TK NV/MX fast",
}


def mean_layer_metric(result: dict, name: str) -> float:
    layers = result["summary"]["attention"]["layer_output_error"]
    return statistics.fmean(float(layer[name]) for layer in layers.values())


def confidence_interval(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    # Two-sided 95% t interval for n=100 (t_0.975,99 = 1.984).
    return 1.984 * statistics.stdev(values) / len(values) ** 0.5


def main() -> None:
    results = {
        name: json.loads((HERE / filename).read_text())
        for name, filename in PROVIDERS.items()
    }
    first = next(iter(results.values()))
    images = first["images"]
    baseline_records = first["records"]
    for name, result in results.items():
        if result["images"] != images:
            raise ValueError(f"image order differs for {name}")
        if not result["summary"]["attention"]["all_outputs_finite"]:
            raise ValueError(f"non-finite output in {name}")
        for reference, candidate in zip(baseline_records, result["records"]):
            if reference["baseline_quality"] != candidate["baseline_quality"]:
                raise ValueError(f"BF16 replay differs for {name}")

    unified = json.loads(UNIFIED.read_text())
    speedup = {
        row["provider"]: float(row["speedup_vs_bf16"])
        for row in unified["rows"]
        if row["shape"] == "b1_s256_h16_d128"
    }
    baseline_psnr = statistics.fmean(
        record["baseline_quality"]["masked_rgb_psnr_db"]
        for record in baseline_records
    )
    baseline_mse = statistics.fmean(
        record["baseline_quality"]["masked_normalized_mse"]
        for record in baseline_records
    )
    rows = [
        {
            "provider": "BF16",
            "speedup": 1.0,
            "masked_psnr_db": baseline_psnr,
            "psnr_delta_db": 0.0,
            "psnr_delta_ci95_db": 0.0,
            "masked_normalized_mse": baseline_mse,
            "masked_mse_delta_percent": 0.0,
            "reconstruction_cosine": 1.0,
            "reconstruction_relative_l2": 0.0,
            "mean_layer_cosine": 1.0,
            "mean_layer_relative_l2": 0.0,
            "all_outputs_finite": True,
        }
    ]
    for name, result in results.items():
        records = result["records"]
        psnr_deltas = [
            record["fp4_quality"]["masked_rgb_psnr_db"]
            - record["baseline_quality"]["masked_rgb_psnr_db"]
            for record in records
        ]
        candidate_mse = statistics.fmean(
            record["fp4_quality"]["masked_normalized_mse"]
            for record in records
        )
        summary = result["summary"]
        rows.append(
            {
                "provider": name,
                "speedup": speedup[UNIFIED_NAMES[name]],
                "masked_psnr_db": statistics.fmean(
                    record["fp4_quality"]["masked_rgb_psnr_db"]
                    for record in records
                ),
                "psnr_delta_db": statistics.fmean(psnr_deltas),
                "psnr_delta_ci95_db": confidence_interval(psnr_deltas),
                "masked_normalized_mse": candidate_mse,
                "masked_mse_delta_percent": 100.0
                * (candidate_mse / baseline_mse - 1.0),
                "reconstruction_cosine": summary[
                    "fp4_vs_bf16_reconstruction"
                ]["cosine"],
                "reconstruction_relative_l2": summary[
                    "fp4_vs_bf16_reconstruction"
                ]["relative_l2"],
                "mean_layer_cosine": mean_layer_metric(result, "cosine"),
                "mean_layer_relative_l2": mean_layer_metric(
                    result, "relative_l2"
                ),
                "all_outputs_finite": True,
            }
        )

    summary = {
        "schema": "tk_fp4_vit_mae_reconstruction_summary_v1",
        "model": first["model"],
        "images": len(images),
        "mask_ratio": first["mask_ratio"],
        "replaced_attention_layers": first["replaced_attention_layers"],
        "model_attention_shape": first["model_attention_shape"],
        "kernel_shape": first["kernel_shape"],
        "rows": rows,
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (HERE / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=rows[0].keys(),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    table_dir = HERE / "tables"
    table_dir.mkdir(exist_ok=True)
    labels = {
        "BF16": "BF16",
        "HAO NV/FP8": "HAO NV/FP8",
        "HAO NV/NV": "HAO NV/NV",
        "TK NV/MX accurate": "TK NV/MX \\code{accurate}",
        "TK NV/MX fast": "TK NV/MX \\code{fast}",
    }
    lines = []
    for row in rows:
        delta = (
            "--"
            if row["provider"] == "BF16"
            else f"{row['psnr_delta_db']:+.3f} $\\pm$ "
            f"{row['psnr_delta_ci95_db']:.3f}"
        )
        lines.append(
            f"{labels[row['provider']]} & {row['speedup']:.3f}$\\times$ & "
            f"{row['masked_psnr_db']:.3f} & {delta} & "
            f"{row['masked_mse_delta_percent']:+.2f} & "
            f"{row['reconstruction_cosine']:.5f} / "
            f"{row['reconstruction_relative_l2']:.4f} & "
            f"{row['mean_layer_cosine']:.4f} / "
            f"{row['mean_layer_relative_l2']:.4f} \\\\"
        )
    lines.append("\\bottomrule")
    (table_dir / "reconstruction_rows.tex").write_text(
        "\n".join(lines) + "\n"
    )


if __name__ == "__main__":
    main()
