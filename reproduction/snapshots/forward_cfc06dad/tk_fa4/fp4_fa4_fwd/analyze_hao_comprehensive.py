#!/usr/bin/env python3
"""Summarize and plot a hao_comprehensive_suite.py manifest."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any


PROVIDERS = (
    ("tk_fp4", "TK NVFP4/NVFP4", "#D81B60", "o"),
    ("hao_fp4", "HAO NVFP4/NVFP4", "#8E24AA", "s"),
    ("tk_fp8", "TK NVFP4/FP8", "#00897B", "^"),
    ("hao_fp8", "HAO NVFP4/FP8", "#1565C0", "D"),
    ("bf16", "HAO BF16", "#303030", "x"),
)
HEADLINE_SHAPES = {
    (1, 256, 16, 128),
    (1, 1024, 16, 128),
    (4, 4096, 16, 128),
    (1, 32768, 16, 128),
    (4, 4096, 32, 128),
    (1, 4096, 12, 128),
    (1, 32768, 12, 128),
    (1, 4096, 24, 128),
    (1, 32768, 24, 128),
    (1, 32768, 24, 64),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def global_accuracy(value: dict[str, Any]) -> dict[str, float]:
    if "global" in value:
        value = value["global"]
    return {
        key: float(value[key])
        for key in (
            "cosine",
            "max_abs",
            "mean_abs",
            "rmse",
            "reference_rms",
            "relative_l2",
        )
        if key in value
    }


def populate_row(row: dict[str, Any], result: dict[str, Any]) -> None:
    benchmark = result.get("benchmark")
    if not benchmark:
        return
    timings = benchmark["timing_ms"]
    throughputs = benchmark["tflops"]
    variant = result["variant"]
    if variant == "pure-fp4" and result["status"] == "complete":
        row["tk_fp4_ms"] = timings["tk_hao_direct_nvfp4_nvfp4pv"]
        row["tk_fp4_tflops"] = throughputs["tk_hao_direct_nvfp4_nvfp4pv"]
        row["hao_fp4_ms"] = timings["hao_native_nvfp4_nvfp4pv"]
        row["hao_fp4_tflops"] = throughputs["hao_native_nvfp4_nvfp4pv"]
        row.setdefault("bf16_samples_ms", []).append(timings["hao_native_bf16"])
        accuracy = global_accuracy(
            benchmark["correctness"]["tk_vs_bf16_output"]
        )
        for key, value in accuracy.items():
            row[f"tk_fp4_{key}"] = value
    elif variant == "fp8" and result["status"] == "complete":
        row["tk_fp8_ms"] = timings["tk_hao_direct_nvfp4_fp8pv"]
        row["tk_fp8_tflops"] = throughputs["tk_hao_direct_nvfp4_fp8pv"]
        row["hao_fp8_ms"] = timings["hao_native_nvfp4_fp8pv"]
        row["hao_fp8_tflops"] = throughputs["hao_native_nvfp4_fp8pv"]
        row.setdefault("bf16_samples_ms", []).append(timings["hao_native_bf16"])
        accuracy = global_accuracy(
            benchmark["correctness"]["tk_vs_bf16_output"]
        )
        for key, value in accuracy.items():
            row[f"tk_fp8_{key}"] = value
    elif variant == "fp8" and result["status"] == "reference-only":
        row["hao_fp8_ms"] = timings["hao_native_nvfp4_fp8pv"]
        row["hao_fp8_tflops"] = throughputs["hao_native_nvfp4_fp8pv"]
        row.setdefault("bf16_samples_ms", []).append(timings["hao_native_bf16"])


def aggregate(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for result in manifest["results"]:
        label = result["label"]
        shape = result["shape"]
        row = rows.setdefault(
            label,
            {
                "label": label,
                "batch": int(shape["batch"]),
                "seqlen": int(shape["seqlen"]),
                "heads": int(shape["heads"]),
                "dim": int(shape["dim"]),
            },
        )
        populate_row(row, result)

    for row in rows.values():
        reference_rms = next(
            (
                row[f"{provider}_reference_rms"]
                for provider in ("tk_fp4", "tk_fp8")
                if f"{provider}_reference_rms" in row
            ),
            None,
        )
        if reference_rms is not None:
            row["bf16_output_rms"] = reference_rms
            for provider in ("tk_fp4", "tk_fp8"):
                rmse_key = f"{provider}_rmse"
                relative_key = f"{provider}_relative_l2"
                if rmse_key in row and relative_key not in row:
                    row[relative_key] = row[rmse_key] / reference_rms

        samples = row.pop("bf16_samples_ms", [])
        if samples:
            row["bf16_ms"] = statistics.median(samples)
            flops = (
                row["batch"]
                * row["heads"]
                * 2
                * row["seqlen"]
                * row["seqlen"]
                * (row["dim"] + row["dim"])
            )
            row["bf16_tflops"] = flops / (row["bf16_ms"] * 1e-3) / 1e12
            for provider in ("tk_fp4", "hao_fp4", "tk_fp8", "hao_fp8"):
                if f"{provider}_ms" in row:
                    row[f"{provider}_speedup_bf16"] = (
                        row["bf16_ms"] / row[f"{provider}_ms"]
                    )
    return sorted(
        rows.values(),
        key=lambda row: (
            row["batch"],
            row["heads"],
            row["dim"],
            row["seqlen"],
        ),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "label",
        "batch",
        "seqlen",
        "heads",
        "dim",
        "bf16_output_rms",
        "tk_fp4_ms",
        "tk_fp4_tflops",
        "tk_fp4_speedup_bf16",
        "tk_fp4_cosine",
        "tk_fp4_relative_l2",
        "tk_fp4_rmse",
        "hao_fp4_ms",
        "hao_fp4_tflops",
        "hao_fp4_speedup_bf16",
        "tk_fp8_ms",
        "tk_fp8_tflops",
        "tk_fp8_speedup_bf16",
        "tk_fp8_cosine",
        "tk_fp8_relative_l2",
        "tk_fp8_rmse",
        "hao_fp8_ms",
        "hao_fp8_tflops",
        "hao_fp8_speedup_bf16",
        "bf16_ms",
        "bf16_tflops",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def setup_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "figure.dpi": 160,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def sweep_rows(rows: list[dict[str, Any]], heads: int) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["batch"] == 1
        and row["heads"] == heads
        and row["dim"] == 128
        and row["seqlen"] >= 1024
    ]


def plot_sweeps(rows: list[dict[str, Any]], output_dir: Path) -> None:
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), sharey=True)
    for axis, heads in zip(axes, (12, 24, 32)):
        subset = sweep_rows(rows, heads)
        for key, label, color, marker in PROVIDERS:
            points = [
                (row["seqlen"], row[f"{key}_tflops"])
                for row in subset
                if f"{key}_tflops" in row
            ]
            if points:
                axis.plot(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    color=color,
                    marker=marker,
                    linewidth=1.35,
                    markersize=3.6,
                    label=label,
                )
        axis.set_xscale("log", base=2)
        axis.set_title(f"H={heads}")
        axis.set_xlabel("Sequence length")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.5)
    axes[0].set_ylabel("Throughput (TFLOP/s)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.08),
    )
    fig.tight_layout()
    fig.savefig(output_dir / "throughput_sweeps.pdf")
    fig.savefig(output_dir / "throughput_sweeps.png")
    plt.close(fig)


def plot_speedups(rows: list[dict[str, Any]], output_dir: Path) -> None:
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), sharey=True)
    selected = [provider for provider in PROVIDERS if provider[0] != "bf16"]
    for axis, heads in zip(axes, (12, 24, 32)):
        subset = sweep_rows(rows, heads)
        for key, label, color, marker in selected:
            points = [
                (row["seqlen"], row[f"{key}_speedup_bf16"])
                for row in subset
                if f"{key}_speedup_bf16" in row
            ]
            if points:
                axis.plot(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    color=color,
                    marker=marker,
                    linewidth=1.35,
                    markersize=3.6,
                    label=label,
                )
        axis.axhline(1.0, color="#555555", linewidth=0.8, linestyle="--")
        axis.set_xscale("log", base=2)
        axis.set_title(f"H={heads}")
        axis.set_xlabel("Sequence length")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.5)
    axes[0].set_ylabel("Speedup over HAO BF16")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.08),
    )
    fig.tight_layout()
    fig.savefig(output_dir / "speedup_sweeps.pdf")
    fig.savefig(output_dir / "speedup_sweeps.png")
    plt.close(fig)


def plot_accuracy(rows: list[dict[str, Any]], output_dir: Path) -> None:
    plt = setup_matplotlib()
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(7.2, 4.0),
        sharex="col",
        sharey="row",
    )
    policies = (
        ("tk_fp4", "TK NVFP4/NVFP4", "#D81B60", "o"),
        ("tk_fp8", "TK NVFP4/FP8", "#00897B", "^"),
    )
    metrics = (
        ("cosine", "Output cosine vs BF16"),
        ("relative_l2", "Relative L2 error vs BF16"),
    )
    for column, heads in enumerate((12, 24, 32)):
        subset = sweep_rows(rows, heads)
        for row_index, (metric, ylabel) in enumerate(metrics):
            axis = axes[row_index, column]
            for key, label, color, marker in policies:
                points = [
                    (row["seqlen"], row[f"{key}_{metric}"])
                    for row in subset
                    if f"{key}_{metric}" in row
                ]
                if points:
                    axis.plot(
                        [point[0] for point in points],
                        [point[1] for point in points],
                        color=color,
                        marker=marker,
                        linewidth=1.35,
                        markersize=3.6,
                        label=label,
                    )
            axis.set_xscale("log", base=2)
            axis.grid(axis="y", color="#D9D9D9", linewidth=0.5)
            if column == 0:
                axis.set_ylabel(ylabel)
        axes[0, column].set_title(f"H={heads}")
        axes[1, column].set_xlabel("Sequence length")
    handles, labels = axes[0, -1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.08),
    )
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_sweeps.pdf")
    fig.savefig(output_dir / "accuracy_sweeps.png")
    plt.close(fig)


def plot_headline(rows: list[dict[str, Any]], output_dir: Path) -> None:
    plt = setup_matplotlib()
    headline = [
        row
        for row in rows
        if (
            row["batch"],
            row["seqlen"],
            row["heads"],
            row["dim"],
        )
        in HEADLINE_SHAPES
        and row["dim"] == 128
    ]
    headline.sort(key=lambda row: (row["seqlen"], row["batch"], row["heads"]))
    policies = [
        provider
        for provider in PROVIDERS
        if provider[0] in ("tk_fp4", "hao_fp4", "tk_fp8", "hao_fp8")
    ]
    width = 0.19
    x_values = list(range(len(headline)))
    fig, axis = plt.subplots(figsize=(7.2, 2.8))
    for index, (key, label, color, _) in enumerate(policies):
        offset = (index - 1.5) * width
        values = [
            row.get(f"{key}_speedup_bf16", math.nan)
            for row in headline
        ]
        axis.bar(
            [value + offset for value in x_values],
            values,
            width=width,
            label=label,
            color=color,
        )
    axis.axhline(1.0, color="#303030", linewidth=0.8, linestyle="--")
    axis.set_ylabel("Speedup over HAO BF16")
    axis.set_xticks(
        x_values,
        [
            f"B{row['batch']} S{row['seqlen']//1024 if row['seqlen'] >= 1024 else row['seqlen']}"
            f"{'K' if row['seqlen'] >= 1024 else ''} H{row['heads']}"
            for row in headline
        ],
        rotation=35,
        ha="right",
    )
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.5)
    axis.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "headline_speedups.pdf")
    fig.savefig(output_dir / "headline_speedups.png")
    plt.close(fig)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Comprehensive HAO comparison",
        "",
        "| Shape | TK FP4 TF | HAO FP4 TF | TK FP8 TF | HAO FP8 TF | BF16 TF | TK FP4 speedup | FP4 cosine | FP4 rel L2 | FP8 cosine | FP8 rel L2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        shape = (
            f"B{row['batch']}/S{row['seqlen']}/"
            f"H{row['heads']}/D{row['dim']}"
        )

        def value(key: str, digits: int = 1) -> str:
            item = row.get(key)
            return "-" if item is None else f"{item:.{digits}f}"

        lines.append(
            "| "
            + " | ".join(
                (
                    shape,
                    value("tk_fp4_tflops"),
                    value("hao_fp4_tflops"),
                    value("tk_fp8_tflops"),
                    value("hao_fp8_tflops"),
                    value("bf16_tflops"),
                    value("tk_fp4_speedup_bf16", 3),
                    value("tk_fp4_cosine", 5),
                    value("tk_fp4_relative_l2", 5),
                    value("tk_fp8_cosine", 5),
                    value("tk_fp8_relative_l2", 5),
                )
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n")


def geometric_mean(values: list[float]) -> float:
    if not values:
        return math.nan
    return math.exp(sum(math.log(value) for value in values) / len(values))


def summarize_statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    d128 = [row for row in rows if row["dim"] == 128]
    steady_d128 = [row for row in d128 if row["seqlen"] >= 4096]

    def provider_statistics(
        subset: list[dict[str, Any]], key: str, label: str
    ) -> dict[str, Any]:
        available = [
            row for row in subset if f"{key}_speedup_bf16" in row
        ]
        speedups = [
            row[f"{key}_speedup_bf16"] for row in available
        ]
        return {
            "label": label,
            "shape_count": len(available),
            "geomean_speedup_vs_bf16": geometric_mean(speedups),
            "wins_vs_bf16": sum(value > 1.0 for value in speedups),
            "best_speedup_vs_bf16": max(speedups, default=math.nan),
            "best_tflops": max(
                (row[f"{key}_tflops"] for row in available),
                default=math.nan,
            ),
        }

    statistics_by_provider: dict[str, Any] = {}
    steady_statistics_by_provider: dict[str, Any] = {}
    for key, label, _, _ in PROVIDERS:
        if key == "bf16":
            continue
        statistics_by_provider[key] = provider_statistics(d128, key, label)
        steady_statistics_by_provider[key] = provider_statistics(
            steady_d128, key, label
        )

    paired_fp4 = [
        row
        for row in d128
        if "tk_fp4_ms" in row and "hao_fp4_ms" in row
    ]
    paired_fp8 = [
        row
        for row in d128
        if "tk_fp8_ms" in row and "hao_fp8_ms" in row
    ]
    fp4_accuracy = [row for row in d128 if "tk_fp4_cosine" in row]
    fp8_accuracy = [row for row in d128 if "tk_fp8_cosine" in row]
    return {
        "d128_shape_count": len(d128),
        "providers": statistics_by_provider,
        "steady_state": {
            "definition": "D128 and sequence length >= 4096",
            "shape_count": len(steady_d128),
            "providers": steady_statistics_by_provider,
        },
        "tk_vs_hao_same_format": {
            "fp4_geomean_speedup": geometric_mean(
                [row["hao_fp4_ms"] / row["tk_fp4_ms"] for row in paired_fp4]
            ),
            "fp4_wins": sum(
                row["tk_fp4_ms"] < row["hao_fp4_ms"] for row in paired_fp4
            ),
            "fp4_shape_count": len(paired_fp4),
            "fp8_geomean_speedup": geometric_mean(
                [row["hao_fp8_ms"] / row["tk_fp8_ms"] for row in paired_fp8]
            ),
            "fp8_wins": sum(
                row["tk_fp8_ms"] < row["hao_fp8_ms"] for row in paired_fp8
            ),
            "fp8_shape_count": len(paired_fp8),
        },
        "accuracy": {
            "tk_fp4_cosine_min": min(
                (row["tk_fp4_cosine"] for row in fp4_accuracy),
                default=math.nan,
            ),
            "tk_fp4_cosine_median": statistics.median(
                row["tk_fp4_cosine"] for row in fp4_accuracy
            ),
            "tk_fp4_rmse_median": statistics.median(
                row["tk_fp4_rmse"] for row in fp4_accuracy
            ),
            "tk_fp4_relative_l2_median": statistics.median(
                row["tk_fp4_relative_l2"] for row in fp4_accuracy
            ),
            "tk_fp4_relative_l2_max": max(
                row["tk_fp4_relative_l2"] for row in fp4_accuracy
            ),
            "tk_fp8_cosine_min": min(
                (row["tk_fp8_cosine"] for row in fp8_accuracy),
                default=math.nan,
            ),
            "tk_fp8_cosine_median": statistics.median(
                row["tk_fp8_cosine"] for row in fp8_accuracy
            ),
            "tk_fp8_rmse_median": statistics.median(
                row["tk_fp8_rmse"] for row in fp8_accuracy
            ),
            "tk_fp8_relative_l2_median": statistics.median(
                row["tk_fp8_relative_l2"] for row in fp8_accuracy
            ),
            "tk_fp8_relative_l2_max": max(
                row["tk_fp8_relative_l2"] for row in fp8_accuracy
            ),
        },
    }


def write_statistics_markdown(path: Path, values: dict[str, Any]) -> None:
    lines = [
        "# Aggregate statistics",
        "",
        "| Provider | Shapes | Geomean vs BF16 | Wins vs BF16 | Best speedup | Peak TFLOP/s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for provider in values["providers"].values():
        lines.append(
            f"| {provider['label']} | {provider['shape_count']} | "
            f"{provider['geomean_speedup_vs_bf16']:.3f}x | "
            f"{provider['wins_vs_bf16']} | "
            f"{provider['best_speedup_vs_bf16']:.3f}x | "
            f"{provider['best_tflops']:.1f} |"
        )
    paired = values["tk_vs_hao_same_format"]
    accuracy = values["accuracy"]
    lines.extend(
        (
            "",
            f"- TK FP4 vs HAO FP4: {paired['fp4_geomean_speedup']:.3f}x "
            f"geomean, {paired['fp4_wins']}/{paired['fp4_shape_count']} wins.",
            f"- TK FP8 vs HAO FP8: {paired['fp8_geomean_speedup']:.3f}x "
            f"geomean, {paired['fp8_wins']}/{paired['fp8_shape_count']} wins.",
            f"- TK FP4 cosine: median {accuracy['tk_fp4_cosine_median']:.6f}, "
            f"minimum {accuracy['tk_fp4_cosine_min']:.6f}; relative L2: "
            f"median {accuracy['tk_fp4_relative_l2_median']:.6f} "
            f"({100 * accuracy['tk_fp4_relative_l2_median']:.2f}%), "
            f"maximum {accuracy['tk_fp4_relative_l2_max']:.6f} "
            f"({100 * accuracy['tk_fp4_relative_l2_max']:.2f}%).",
            f"- TK FP8 cosine: median {accuracy['tk_fp8_cosine_median']:.6f}, "
            f"minimum {accuracy['tk_fp8_cosine_min']:.6f}; relative L2: "
            f"median {accuracy['tk_fp8_relative_l2_median']:.6f} "
            f"({100 * accuracy['tk_fp8_relative_l2_median']:.2f}%), "
            f"maximum {accuracy['tk_fp8_relative_l2_max']:.6f} "
            f"({100 * accuracy['tk_fp8_relative_l2_max']:.2f}%).",
            "",
            "## Steady-state subset",
            "",
            "D128 rows with sequence length at least 4096:",
        )
    )
    for provider in values["steady_state"]["providers"].values():
        lines.append(
            f"- {provider['label']}: "
            f"{provider['geomean_speedup_vs_bf16']:.3f}x geomean vs BF16, "
            f"{provider['wins_vs_bf16']}/{provider['shape_count']} wins."
        )
    path.write_text("\n".join(lines) + "\n")


def write_statistics_latex(path: Path, values: dict[str, Any]) -> None:
    providers = values["providers"]
    steady = values["steady_state"]
    steady_providers = steady["providers"]
    paired = values["tk_vs_hao_same_format"]
    accuracy = values["accuracy"]
    macros = {
        "DShapeCount": str(values["d128_shape_count"]),
        "TKFPFourGeoBF": (
            f"{providers['tk_fp4']['geomean_speedup_vs_bf16']:.3f}"
        ),
        "TKFPFourWins": str(providers["tk_fp4"]["wins_vs_bf16"]),
        "TKFPFourCount": str(providers["tk_fp4"]["shape_count"]),
        "TKFPFourPeak": f"{providers['tk_fp4']['best_tflops']:.0f}",
        "TKFPFourGeoHAO": f"{paired['fp4_geomean_speedup']:.3f}",
        "TKFPFourHAOWins": str(paired["fp4_wins"]),
        "TKFPFourHAOCount": str(paired["fp4_shape_count"]),
        "TKFPEightGeoBF": (
            f"{providers['tk_fp8']['geomean_speedup_vs_bf16']:.3f}"
        ),
        "TKFPEightWins": str(providers["tk_fp8"]["wins_vs_bf16"]),
        "TKFPEightCount": str(providers["tk_fp8"]["shape_count"]),
        "TKFPEightPeak": f"{providers['tk_fp8']['best_tflops']:.0f}",
        "TKFPEightGeoHAO": f"{paired['fp8_geomean_speedup']:.3f}",
        "TKFPEightHAOWins": str(paired["fp8_wins"]),
        "TKFPEightHAOCount": str(paired["fp8_shape_count"]),
        "TKFPFourCosineMedian": (
            f"{accuracy['tk_fp4_cosine_median']:.6f}"
        ),
        "TKFPFourCosineMin": f"{accuracy['tk_fp4_cosine_min']:.6f}",
        "TKFPFourRMSEMedian": f"{accuracy['tk_fp4_rmse_median']:.6f}",
        "TKFPFourRelLTwoMedian": (
            f"{accuracy['tk_fp4_relative_l2_median']:.6f}"
        ),
        "TKFPFourRelLTwoMedianPercent": (
            f"{100 * accuracy['tk_fp4_relative_l2_median']:.2f}"
        ),
        "TKFPFourRelLTwoMax": (
            f"{accuracy['tk_fp4_relative_l2_max']:.6f}"
        ),
        "TKFPFourRelLTwoMaxPercent": (
            f"{100 * accuracy['tk_fp4_relative_l2_max']:.2f}"
        ),
        "TKFPEightCosineMedian": (
            f"{accuracy['tk_fp8_cosine_median']:.6f}"
        ),
        "TKFPEightCosineMin": f"{accuracy['tk_fp8_cosine_min']:.6f}",
        "TKFPEightRMSEMedian": f"{accuracy['tk_fp8_rmse_median']:.6f}",
        "TKFPEightRelLTwoMedian": (
            f"{accuracy['tk_fp8_relative_l2_median']:.6f}"
        ),
        "TKFPEightRelLTwoMedianPercent": (
            f"{100 * accuracy['tk_fp8_relative_l2_median']:.2f}"
        ),
        "TKFPEightRelLTwoMax": (
            f"{accuracy['tk_fp8_relative_l2_max']:.6f}"
        ),
        "TKFPEightRelLTwoMaxPercent": (
            f"{100 * accuracy['tk_fp8_relative_l2_max']:.2f}"
        ),
        "SteadyShapeCount": str(steady["shape_count"]),
        "TKFPFourSteadyGeoBF": (
            f"{steady_providers['tk_fp4']['geomean_speedup_vs_bf16']:.3f}"
        ),
        "TKFPFourSteadyWins": str(
            steady_providers["tk_fp4"]["wins_vs_bf16"]
        ),
        "TKFPEightSteadyGeoBF": (
            f"{steady_providers['tk_fp8']['geomean_speedup_vs_bf16']:.3f}"
        ),
        "TKFPEightSteadyWins": str(
            steady_providers["tk_fp8"]["wins_vs_bf16"]
        ),
    }
    path.write_text(
        "\n".join(
            rf"\newcommand{{\{name}}}{{{value}}}"
            for name, value in macros.items()
        )
        + "\n"
    )


def latex_value(row: dict[str, Any], key: str, digits: int = 0) -> str:
    value = row.get(key)
    if value is None:
        return "--"
    return f"{value:.{digits}f}"


def write_headline_latex(path: Path, rows: list[dict[str, Any]]) -> None:
    headline = [
        row
        for row in rows
        if (
            row["batch"],
            row["seqlen"],
            row["heads"],
            row["dim"],
        )
        in HEADLINE_SHAPES
    ]
    published_order = [
        (1, 256, 16, 128),
        (1, 1024, 16, 128),
        (4, 4096, 16, 128),
        (1, 32768, 16, 128),
        (4, 4096, 32, 128),
        (1, 4096, 12, 128),
        (1, 32768, 12, 128),
        (1, 4096, 24, 128),
        (1, 32768, 24, 128),
        (1, 32768, 24, 64),
    ]
    order = {shape: index for index, shape in enumerate(published_order)}
    headline.sort(
        key=lambda row: order[
            (row["batch"], row["seqlen"], row["heads"], row["dim"])
        ]
    )
    lines = [
        r"\begin{tabular}{rrrrrrrr}",
        r"\toprule",
        r"$B$ & $S$ & $H$ & $d$ & TK FP4 & HAO FP4 & TK FP8 & HAO FP8 / BF16 \\",
        r"\midrule",
    ]
    for row in headline:
        lines.append(
            f"{row['batch']} & {row['seqlen']} & {row['heads']} & "
            f"{row['dim']} & {latex_value(row, 'tk_fp4_tflops')} & "
            f"{latex_value(row, 'hao_fp4_tflops')} & "
            f"{latex_value(row, 'tk_fp8_tflops')} & "
            f"{latex_value(row, 'hao_fp8_tflops')} / "
            f"{latex_value(row, 'bf16_tflops')} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    path.write_text("\n".join(lines) + "\n")


def write_selected_sweep_latex(path: Path, rows: list[dict[str, Any]]) -> None:
    selected = [
        row
        for row in rows
        if row["batch"] == 1
        and row["dim"] == 128
        and row["heads"] in (12, 24, 32)
        and row["seqlen"] in (4096, 32768)
    ]
    selected.sort(key=lambda row: (row["heads"], row["seqlen"]))
    lines = [
        r"\begin{tabular}{rrrrrrrr}",
        r"\toprule",
        r"$H$ & $S$ & TK FP4 & HAO FP4 & TK FP8 & HAO FP8 & BF16 & FP4/BF16 \\",
        r"\midrule",
    ]
    for row in selected:
        lines.append(
            f"{row['heads']} & {row['seqlen']} & "
            f"{latex_value(row, 'tk_fp4_tflops')} & "
            f"{latex_value(row, 'hao_fp4_tflops')} & "
            f"{latex_value(row, 'tk_fp8_tflops')} & "
            f"{latex_value(row, 'hao_fp8_tflops')} & "
            f"{latex_value(row, 'bf16_tflops')} & "
            f"{latex_value(row, 'tk_fp4_speedup_bf16', 2)}x \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    if not manifest.get("complete"):
        raise RuntimeError("refusing to summarize an incomplete manifest")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = aggregate(manifest)
    write_csv(args.output_dir / "summary.csv", rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n"
    )
    write_markdown(args.output_dir / "summary.md", rows)
    aggregate_statistics = summarize_statistics(rows)
    (args.output_dir / "statistics.json").write_text(
        json.dumps(aggregate_statistics, indent=2, sort_keys=True) + "\n"
    )
    write_statistics_markdown(
        args.output_dir / "statistics.md", aggregate_statistics
    )
    write_statistics_latex(
        args.output_dir / "statistics_macros.tex", aggregate_statistics
    )
    write_headline_latex(args.output_dir / "headline_table.tex", rows)
    write_selected_sweep_latex(
        args.output_dir / "selected_sweep_table.tex", rows
    )
    plot_sweeps(rows, args.output_dir)
    plot_speedups(rows, args.output_dir)
    plot_accuracy(rows, args.output_dir)
    plot_headline(rows, args.output_dir)
    print(json.dumps({"rows": len(rows), "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
