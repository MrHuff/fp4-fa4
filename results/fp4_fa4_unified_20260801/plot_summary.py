#!/usr/bin/env python3
"""Render paper figures from the unified FP4 FA4 summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


STYLE = {
    "nvmx-fast": ("NV/MX fast", "#c83e4d", "o"),
    "nvmx-accurate": ("NV/MX accurate", "#28865d", "D"),
    "fp8": ("TK NV/FP8", "#2a6fbb", "^"),
    "hao-nv-nv": ("HAO NV/NV", "#7856a3", "v"),
    "hao-nv-fp8": ("HAO NV/FP8", "#3c4650", "P"),
    "bf16": ("HAO BF16", "#7c858d", "X"),
}
PDF_METADATA = {
    "Creator": "plot_summary.py",
    "CreationDate": None,
    "ModDate": None,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(__file__).resolve().parent / "summary.json",
    )
    return parser.parse_args()


def setup() -> None:
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "legend.fontsize": 7.5,
            "figure.dpi": 160,
            # Keep text as embedded outline fonts.  Matplotlib's PDF default
            # is Type 3, which does not meet arXiv's portable-font guidance.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
        }
    )


def headline_figure(summary: dict, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 3.25))
    for row in summary["headline"]:
        family = row["family"]
        if family not in STYLE or family in {"nv-nv", "mx-nv", "nv-mx", "mx-mx"}:
            continue
        label, color, marker = STYLE[family]
        x = row["relative_l2_bf16"]
        y = row["speedup_vs_bf16"]
        ax.scatter(x, y, s=52, color=color, marker=marker, zorder=3)
        offset = (5, 5)
        if family == "hao-nv-nv":
            offset = (-56, 5)
        ax.annotate(label, (x, y), xytext=offset, textcoords="offset points")
    ax.axhline(1.0, color="#7c858d", linestyle="--", linewidth=0.9)
    ax.set_xlabel(r"Relative $L_2$ error versus BF16 (lower is better)")
    ax.set_ylabel(r"Speedup over BF16 (higher is better)")
    ax.set_title("B1 / S4096 / H24 / D128")
    fig.tight_layout()
    fig.savefig(output, metadata=PDF_METADATA)
    plt.close(fig)


def cross_shape_figure(summary: dict, output: Path) -> None:
    rows = summary["cross_shape"]
    shape_order = []
    for row in rows:
        shape = row["shape"]
        if shape not in shape_order:
            shape_order.append(shape)
    labels = []
    for shape in shape_order:
        row = next(value for value in rows if value["shape"] == shape)
        labels.append(f'S{row["seqlen"]}\nH{row["heads"]}')

    fig, (speed_ax, error_ax) = plt.subplots(
        2,
        1,
        figsize=(7.25, 5.0),
        sharex=True,
        constrained_layout=True,
    )
    for family in STYLE:
        family_rows = {row["shape"]: row for row in rows if row["family"] == family}
        if not family_rows:
            continue
        label, color, marker = STYLE[family]
        speed = [family_rows[shape]["speedup_vs_bf16"] for shape in shape_order]
        error = [family_rows[shape]["relative_l2_bf16"] for shape in shape_order]
        kwargs = {
            "label": label,
            "color": color,
            "marker": marker,
            "markersize": 4.5,
            "linewidth": 1.35,
        }
        speed_ax.plot(range(len(shape_order)), speed, **kwargs)
        error_ax.plot(range(len(shape_order)), error, **kwargs)

    speed_ax.axhline(1.0, color="#7c858d", linestyle="--", linewidth=0.9)
    speed_ax.set_ylabel("Speedup / BF16")
    speed_ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.32))
    error_ax.set_ylabel(r"Relative $L_2$ / BF16")
    error_ax.set_xlabel("Sequence length and head count")
    error_ax.set_xticks(range(len(labels)), labels)
    fig.savefig(output, metadata=PDF_METADATA)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.read_text())
    output_dir = args.summary.parent / "figures"
    output_dir.mkdir(exist_ok=True)
    setup()
    headline_figure(summary, output_dir / "headline_pareto.pdf")
    cross_shape_figure(summary, output_dir / "cross_shape_speed_accuracy.pdf")


if __name__ == "__main__":
    main()
