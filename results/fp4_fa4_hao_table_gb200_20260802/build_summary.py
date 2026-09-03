#!/usr/bin/env python3
"""Merge the matched GB200 runs over HAO's published D128 shape grid."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "tables"
PUBLISHED_HAO = ROOT / "published_hao_results.json"
EXPECTED_SHAPES = {
    (1, 256, 16, 128),
    (1, 1024, 16, 128),
    (4, 4096, 16, 128),
    (1, 32768, 16, 128),
    (4, 4096, 32, 128),
    (1, 4096, 12, 128),
    (1, 32768, 12, 128),
    (1, 4096, 24, 128),
    (1, 32768, 24, 128),
}


def load_published_hao() -> tuple[dict[tuple[int, int, int, int], dict], dict]:
    published = json.loads(PUBLISHED_HAO.read_text())
    by_shape = {
        tuple(int(value) for value in row["shape"]): row
        for row in published["rows"]
    }
    expected = EXPECTED_SHAPES | {(1, 32768, 24, 64)}
    if set(by_shape) != expected:
        missing = sorted(expected - set(by_shape))
        extra = sorted(set(by_shape) - expected)
        raise RuntimeError(
            f"published HAO shape mismatch: missing={missing}, extra={extra}"
        )
    for shape, row in by_shape.items():
        for hardware in ("b200", "gb300"):
            for field in ("nvfp4_fp8", "bf16"):
                value = row[hardware][field]
                if not isinstance(value, (int, float)) or value <= 0:
                    raise RuntimeError(
                        f"invalid HAO {hardware}/{field} value at {shape}: {value}"
                    )
        cosine = row["nvfp4_fp8_precision"]["cosine"]
        if not 0.0 <= cosine <= 1.0:
            raise RuntimeError(f"invalid HAO cosine at {shape}: {cosine}")
    return by_shape, published


def shape_key(record: dict) -> tuple[int, int, int, int]:
    shape = record["shape"]
    return (
        int(shape["batch"]),
        int(shape["seqlen"]),
        int(shape["heads"]),
        int(shape["dim"]),
    )


def load_variant(prefix: str, variant: str) -> tuple[dict, dict]:
    manifests = sorted(ROOT.glob(f"{prefix}_shard*/manifest.json"))
    if len(manifests) != 4:
        raise RuntimeError(f"expected four {prefix} manifests, found {len(manifests)}")

    by_shape: dict[tuple[int, int, int, int], dict] = {}
    provenance = []
    for path in manifests:
        manifest = json.loads(path.read_text())
        if not manifest.get("complete") or manifest.get("failures"):
            raise RuntimeError(f"incomplete or failed manifest: {path}")
        provenance.append(
            {
                "path": str(path.relative_to(ROOT)),
                "created_utc": manifest["created_utc"],
                "completed_utc": manifest["completed_utc"],
                "gpu": manifest["gpu"],
                "protocol": manifest["protocol"],
                "sources": manifest["sources"],
                "software": manifest["software"],
            }
        )
        for record in manifest["results"]:
            if record.get("status") != "complete" or record.get("variant") != variant:
                continue
            key = shape_key(record)
            if key in by_shape:
                raise RuntimeError(f"duplicate {variant} result for {key}")
            by_shape[key] = record["benchmark"]

    if set(by_shape) != EXPECTED_SHAPES:
        missing = sorted(EXPECTED_SHAPES - set(by_shape))
        extra = sorted(set(by_shape) - EXPECTED_SHAPES)
        raise RuntimeError(f"{variant} shape mismatch: missing={missing}, extra={extra}")
    return by_shape, {"variant": variant, "manifests": provenance}


def metrics(values: dict | None) -> dict[str, float]:
    if values is None:
        return {"cosine": 1.0, "relative_l2": 0.0, "rmse": 0.0}
    return {
        "cosine": float(values["cosine"]),
        "relative_l2": float(values["relative_l2"]),
        "rmse": float(values["rmse"]),
    }


def provider_row(
    shape: tuple[int, int, int, int],
    provider: str,
    time_ms: float,
    tflops: float,
    bf16_ms: float,
    error: dict | None,
) -> dict:
    batch, seqlen, heads, dim = shape
    return {
        "shape": f"b{batch}_s{seqlen}_h{heads}_d{dim}",
        "batch": batch,
        "seqlen": seqlen,
        "heads": heads,
        "dim": dim,
        "provider": provider,
        "time_ms": float(time_ms),
        "tflops": float(tflops),
        "speedup_vs_bf16": float(bf16_ms / time_ms),
        **metrics(error),
    }


def geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def format_primary_tex_rows(
    rows: list[dict], published_by_shape: dict[tuple[int, int, int, int], dict]
) -> str:
    by_shape: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_shape.setdefault(row["shape"], {})[row["provider"]] = row

    rendered = []
    for providers in by_shape.values():
        fast = providers["TK NV/MX fast"]
        accurate = providers["TK NV/MX accurate"]
        published = published_by_shape[
            (fast["batch"], fast["seqlen"], fast["heads"], fast["dim"])
        ]
        shape = (
            f"B{fast['batch']}/S{fast['seqlen']}/H{fast['heads']}"
            f"/D{fast['dim']}"
        )

        def speed_cell(row: dict) -> str:
            return f"{row['time_ms']:.6f} / {row['tflops']:.0f}"

        def error_cell(row: dict) -> str:
            return f"{row['cosine']:.4f} / {row['relative_l2']:.4f}"

        rendered.append(
            f"{shape} & {speed_cell(fast)} & {error_cell(fast)} & "
            f"{speed_cell(accurate)} & {error_cell(accurate)} & "
            f"{published['b200']['nvfp4_fp8']:.0f} & "
            f"{published['gb300']['nvfp4_fp8']:.0f} & "
            f"{published['nvfp4_fp8_precision']['cosine']:.4f} & "
            f"{published['gb300']['bf16']:.0f} \\\\"
        )
    return "\n".join(rendered) + "\n\\bottomrule\n"


def main() -> None:
    published_by_shape, published = load_published_hao()
    fast, fast_provenance = load_variant("fast", "nvmx-fast")
    accurate, accurate_provenance = load_variant(
        "accurate", "nvmx-accurate"
    )

    rows = []
    compact = []
    for shape in sorted(EXPECTED_SHAPES, key=lambda item: (item[1], item[0], item[2])):
        fast_case = fast[shape]
        accurate_case = accurate[shape]
        fast_timing = fast_case["timing_ms"]
        fast_tflops = fast_case["tflops"]
        accurate_timing = accurate_case["timing_ms"]
        accurate_tflops = accurate_case["tflops"]
        bf16_ms = float(fast_timing["hao_native_bf16"])

        rows.extend(
            [
                provider_row(
                    shape,
                    "TK NV/MX fast",
                    fast_timing["tk_hao_direct_nvfp4_mxfp4pv"],
                    fast_tflops["tk_hao_direct_nvfp4_mxfp4pv"],
                    bf16_ms,
                    fast_case["correctness"]["tk_vs_bf16_output"],
                ),
                provider_row(
                    shape,
                    "TK NV/MX accurate",
                    accurate_timing["tk_hao_direct_nvfp4_mxfp4pv"],
                    accurate_tflops["tk_hao_direct_nvfp4_mxfp4pv"],
                    bf16_ms,
                    accurate_case["correctness"]["tk_vs_bf16_output"],
                ),
                provider_row(
                    shape,
                    "HAO NV/NV",
                    fast_timing["hao_native_nvfp4_nvfp4pv"],
                    fast_tflops["hao_native_nvfp4_nvfp4pv"],
                    bf16_ms,
                    fast_case["correctness"]["hao_vs_bf16_output"],
                ),
                provider_row(
                    shape,
                    "HAO BF16",
                    bf16_ms,
                    fast_tflops["hao_native_bf16"],
                    bf16_ms,
                    None,
                ),
            ]
        )
        compact.append(
            {
                "shape": rows[-4]["shape"],
                "fast_tflops": rows[-4]["tflops"],
                "accurate_tflops": rows[-3]["tflops"],
                "hao_nvnv_tflops": rows[-2]["tflops"],
                "bf16_tflops": rows[-1]["tflops"],
            }
        )

    by_provider = {}
    for provider in ("TK NV/MX fast", "TK NV/MX accurate", "HAO NV/NV"):
        selected = [row for row in rows if row["provider"] == provider]
        by_provider[provider] = {
            "geomean_speedup_vs_bf16": geometric_mean(
                [row["speedup_vs_bf16"] for row in selected]
            ),
            "min_speedup_vs_bf16": min(
                row["speedup_vs_bf16"] for row in selected
            ),
            "max_speedup_vs_bf16": max(
                row["speedup_vs_bf16"] for row in selected
            ),
            "peak_tflops": max(row["tflops"] for row in selected),
            "mean_cosine": sum(row["cosine"] for row in selected) / len(selected),
            "mean_relative_l2": sum(row["relative_l2"] for row in selected)
            / len(selected),
        }

    summary = {
        "schema": "fp4-fa4-hao-grid-gb200-v2",
        "hardware": fast_provenance["manifests"][0]["gpu"],
        "protocol": fast_provenance["manifests"][0]["protocol"],
        "unsupported": [
            {
                "shape": "b1_s32768_h24_d64",
                "reason": "The specialized TK scaled-FP4 PV path requires D128.",
            }
        ],
        "aggregate": by_provider,
        "rows": rows,
        "published_hao": published,
        "provenance": {
            "fast": fast_provenance,
            "accurate": accurate_provenance,
        },
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    with (ROOT / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    TABLES.mkdir(exist_ok=True)
    labels = {
        "TK NV/MX fast": r"TK NV/MX \code{fast}",
        "TK NV/MX accurate": r"TK NV/MX \code{accurate}",
        "HAO NV/NV": "HAO NV/NV",
        "HAO BF16": "HAO BF16",
    }
    full_lines = []
    previous_shape = None
    for row in rows:
        shape = (
            f"B{row['batch']}/S{row['seqlen']}/H{row['heads']}"
            if row["shape"] != previous_shape
            else ""
        )
        full_lines.append(
            f"{shape} & {labels[row['provider']]} & {row['time_ms']:.6f} & "
            f"{row['tflops']:.0f} & {row['speedup_vs_bf16']:.3f}$\\times$ & "
            f"{row['cosine']:.6f} & {row['relative_l2']:.6f} & "
            f"{row['rmse']:.6f} \\\\"
        )
        previous_shape = row["shape"]
    full_lines.append(r"B1/S32768/H24/D64 & TK full FP4 & \multicolumn{6}{c}{unsupported} \\")
    full_lines.append(r"\bottomrule")
    (TABLES / "hao_grid_rows.tex").write_text("\n".join(full_lines) + "\n")
    (TABLES / "hao_grid_primary_rows.tex").write_text(
        format_primary_tex_rows(rows, published_by_shape)
    )

    compact_lines = []
    for row in compact:
        shape = row["shape"].replace("b", "B", 1).replace("_s", "/S").replace("_h", "/H").replace("_d128", "")
        compact_lines.append(
            f"{shape} & {row['fast_tflops']:.0f} & "
            f"{row['accurate_tflops']:.0f} & {row['hao_nvnv_tflops']:.0f} & "
            f"{row['bf16_tflops']:.0f} \\\\"
        )
    compact_lines.append(r"B1/S32768/H24/D64 & -- & -- & -- & -- \\")
    compact_lines.append(r"\bottomrule")
    (TABLES / "hao_grid_speed_rows.tex").write_text(
        "\n".join(compact_lines) + "\n"
    )

    macros = ["% Generated by build_summary.py; do not edit by hand."]
    for prefix, provider in (
        ("HaoGridFast", "TK NV/MX fast"),
        ("HaoGridAccurate", "TK NV/MX accurate"),
    ):
        aggregate = by_provider[provider]
        macros.extend(
            [
                rf"\newcommand{{\{prefix}PeakTflops}}{{{aggregate['peak_tflops']:.0f}}}",
                rf"\newcommand{{\{prefix}GeoSpeedup}}{{{aggregate['geomean_speedup_vs_bf16']:.3f}}}",
                rf"\newcommand{{\{prefix}MeanCosine}}{{{aggregate['mean_cosine']:.6f}}}",
                rf"\newcommand{{\{prefix}MeanRelLTwo}}{{{aggregate['mean_relative_l2']:.6f}}}",
            ]
        )
    published_d128 = [
        published_by_shape[shape]
        for shape in sorted(EXPECTED_SHAPES)
    ]
    macros.extend(
        [
            rf"\newcommand{{\HaoPublishedBTwoHundredGeoSpeedup}}{{{geometric_mean([row['b200']['nvfp4_fp8'] / row['b200']['bf16'] for row in published_d128]):.3f}}}",
            rf"\newcommand{{\HaoPublishedGBThreeHundredGeoSpeedup}}{{{geometric_mean([row['gb300']['nvfp4_fp8'] / row['gb300']['bf16'] for row in published_d128]):.3f}}}",
            rf"\newcommand{{\HaoPublishedMeanCosine}}{{{sum(row['nvfp4_fp8_precision']['cosine'] for row in published_d128) / len(published_d128):.4f}}}",
        ]
    )
    (TABLES / "hao_grid_macros.tex").write_text("\n".join(macros) + "\n")


if __name__ == "__main__":
    main()
