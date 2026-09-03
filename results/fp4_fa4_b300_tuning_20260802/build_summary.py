#!/usr/bin/env python3
"""Merge downloaded B300 Volt results into report-ready artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
GB200_SUMMARY = ROOT.parent / "fp4_fa4_hao_table_gb200_20260802" / "summary.json"
PUBLISHED_HAO = (
    ROOT.parent
    / "fp4_fa4_hao_table_gb200_20260802"
    / "published_hao_results.json"
)
GB200_D64_ROOT = ROOT.parent / "fp4_fa4_d64_gb200_full_20260802"
UNIFIED_SUMMARY = ROOT.parent / "fp4_fa4_unified_20260801" / "summary.json"
GB200_HARDWARE = ROOT / "local_gb200" / "hardware.json"
TABLE_DIR = ROOT / "tables"


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def published_hao_by_shape() -> dict[tuple[int, int, int, int], dict[str, Any]]:
    published = load_json(PUBLISHED_HAO)
    return {
        tuple(int(value) for value in row["shape"]): row
        for row in published["rows"]
    }


def find_one(pattern: str) -> Path | None:
    matches = sorted(ROOT.glob(pattern))
    if len(matches) > 1:
        raise RuntimeError(f"expected at most one {pattern!r}, found {matches}")
    return matches[0] if matches else None


def canonical_row(
    *,
    label: str,
    hardware: str,
    grid_sms: int,
    density: int,
    record: dict[str, Any],
    source: str,
) -> dict[str, Any]:
    error = record["correctness_global"]
    return {
        "label": label,
        "hardware": hardware,
        "persistent_grid_sms": grid_sms,
        "native_ex2_density": density,
        "timing_ms": float(record["timing_ms"]),
        "tflops": float(record["tflops"]),
        "cosine": float(error["cosine"]),
        "relative_l2": float(error["relative_l2"]),
        "rmse": float(error["rmse"]),
        "source": source,
    }


def format_tex_rows(rows: list[dict[str, Any]]) -> str:
    rendered = []
    for row in rows:
        label = row["label"].replace("_", r"\_")
        rendered.append(
            f"{label} & {row['persistent_grid_sms']} & "
            f"{row['native_ex2_density']} & {row['timing_ms']:.6f} & "
            f"{row['tflops']:.0f} & {row['cosine']:.6f} & "
            f"{row['relative_l2']:.6f} & {row['rmse']:.6f} \\\\"
        )
    return "\n".join(rendered) + "\n\\bottomrule\n"


def optional_metric(value: float | None, digits: int = 6) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def suite_result_map(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    manifest = load_json(path)
    if not manifest["complete"] or manifest["failures"]:
        raise RuntimeError(f"incomplete suite manifest: {path}")
    result = {}
    for case in manifest["results"]:
        benchmark = case["benchmark"]
        shape = benchmark["shape"]
        key = (int(shape["heads"]), int(shape["seqlen"]))
        if key in result:
            raise RuntimeError(f"duplicate D64 shape {key} in {path}")
        result[key] = case
    return result


def tk_value(values: dict[str, float]) -> float:
    matches = [float(value) for key, value in values.items() if key.startswith("tk_")]
    if len(matches) != 1:
        raise RuntimeError(f"expected one TK metric, found {values}")
    return matches[0]


def finite_metric(value: Any) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def attention_tflops(
    *, batch: int, seqlen: int, heads: int, dqk: int, dvo: int, time_ms: float
) -> float:
    operation_count = batch * heads * 2 * seqlen * seqlen * (dqk + dvo)
    return operation_count / (time_ms * 1.0e9)


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0.0 for value in values):
        raise ValueError(f"geometric mean requires positive values: {values}")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def d128_three_k_rows() -> list[dict[str, Any]]:
    path = find_one(
        "artifacts/*/workers/0/fp4-fa4-b300-d128-3ktflops-confirm/summary.json"
    )
    if path is None:
        raise FileNotFoundError("B300 D128 3 PFLOP/s confirmation not found")

    confirmation = load_json(path)
    if not confirmation.get("all_above_target"):
        raise RuntimeError("B300 D128 confirmation did not clear 3 PFLOP/s")
    expected = {(8192, 64), (9472, 64)}
    rows = []
    for record in confirmation["rows"]:
        batch, seqlen, heads, dim = map(int, record["shape"])
        key = (seqlen, heads)
        correctness = record["correctness"]
        if (
            batch != 1
            or dim != 128
            or int(record["grid"]) != 148
            or record["variant"] != "kv6_native_q3"
            or key not in expected
            or len(correctness) != 2
        ):
            raise RuntimeError(f"unexpected B300 3 PFLOP/s row: {record}")
        metrics = [
            float(sample[name])
            for sample in correctness
            for name in ("cosine", "relative_l2", "rmse")
        ]
        if not all(math.isfinite(value) for value in metrics):
            raise RuntimeError(f"non-finite B300 confirmation row: {record}")
        median_ms = float(record["median_ms"])
        tflops = attention_tflops(
            batch=batch,
            seqlen=seqlen,
            heads=heads,
            dqk=dim,
            dvo=dim,
            time_ms=median_ms,
        )
        if not math.isclose(tflops, float(record["tflops"]), rel_tol=1.0e-9):
            raise RuntimeError(f"B300 confirmation TFLOP/s drift: {record}")
        if tflops <= float(confirmation["target_tflops"]):
            raise RuntimeError(f"B300 confirmation below target: {record}")
        rows.append(
            {
                "batch": batch,
                "seqlen": seqlen,
                "heads": heads,
                "dim": dim,
                "grid": int(record["grid"]),
                "variant": record["variant"],
                "timing_ms": [float(value) for value in record["timing_ms"]],
                "median_ms": median_ms,
                "minimum_ms": float(record["minimum_ms"]),
                "maximum_ms": float(record["maximum_ms"]),
                "relative_span": float(record["relative_span"]),
                "tflops": tflops,
                "cosine_min": min(float(sample["cosine"]) for sample in correctness),
                "cosine_max": max(float(sample["cosine"]) for sample in correctness),
                "relative_l2_min": min(
                    float(sample["relative_l2"]) for sample in correctness
                ),
                "relative_l2_max": max(
                    float(sample["relative_l2"]) for sample in correctness
                ),
                "rmse_min": min(float(sample["rmse"]) for sample in correctness),
                "rmse_max": max(float(sample["rmse"]) for sample in correctness),
                "correctness": correctness,
                "source": str(path.relative_to(ROOT)),
            }
        )
    if {(row["seqlen"], row["heads"]) for row in rows} != expected:
        raise RuntimeError("B300 D128 confirmation has the wrong shape set")
    return sorted(rows, key=lambda row: (row["seqlen"], row["heads"]))


def format_three_k_tex_rows(rows: list[dict[str, Any]]) -> str:
    rendered = []
    for row in rows:
        rendered.append(
            f"{row['heads']} & {row['seqlen']} & {row['grid']} & "
            f"{row['median_ms']:.6f} & {row['tflops']:.0f} & "
            f"{100.0 * row['relative_span']:.3f}\\% & "
            f"{row['cosine_min']:.6f}--{row['cosine_max']:.6f} & "
            f"{row['relative_l2_min']:.6f}--{row['relative_l2_max']:.6f} \\\\"
        )
    return "\n".join(rendered) + "\n\\bottomrule\n"


def d64_matrix() -> dict[str, Any]:
    nvmx_density_two_path = find_one(
        "artifacts/*/workers/0/fp4-fa4-b300-d64-nvmx-full-sweep/"
        "density2_default/manifest.json"
    )
    nvmx_density_one_path = find_one(
        "artifacts/*/workers/0/fp4-fa4-b300-d64-nvmx-full-sweep/"
        "density1_control/manifest.json"
    )
    nvnv_path = find_one(
        "artifacts/*/workers/0/fp4-fa4-b300-d64-nvnv-full-sweep/"
        "production/manifest.json"
    )
    if None in (nvmx_density_two_path, nvmx_density_one_path, nvnv_path):
        raise FileNotFoundError("downloaded B300 D64 matrix manifests not found")

    nvmx = suite_result_map(nvmx_density_two_path)
    density_one = suite_result_map(nvmx_density_one_path)
    nvnv = suite_result_map(nvnv_path)
    expected = {
        (heads, seqlen)
        for heads in (12, 24, 32, 64)
        for seqlen in (1024, 2048, 4096, 8192, 16384, 32768)
    }
    if set(nvmx) != expected or set(density_one) != expected or set(nvnv) != expected:
        raise RuntimeError("B300 D64 manifests do not contain the full 24-shape grid")

    gb200_nvmx: dict[tuple[int, int], dict[str, Any]] = {}
    gb200_nvnv: dict[tuple[int, int], dict[str, Any]] = {}
    for heads in (12, 24, 32, 64):
        gb200_nvmx.update(
            suite_result_map(GB200_D64_ROOT / f"nvmx_h{heads}_manifest.json")
        )
        gb200_nvnv.update(
            suite_result_map(GB200_D64_ROOT / f"nvnv_h{heads}_manifest.json")
        )
    if set(gb200_nvmx) != expected or set(gb200_nvnv) != expected:
        raise RuntimeError("GB200 D64 manifests do not contain the full 24-shape grid")

    grid_sources: dict[tuple[int, int], tuple[int, Path]] = {}
    for cap, key in ((136, (32, 2048)), (128, (64, 1024))):
        path = find_one(
            "artifacts/*/workers/0/fp4-fa4-b300-d64-nvmx-grid-recheck/"
            f"grid{cap}/manifest.json"
        )
        if path is None:
            raise FileNotFoundError(f"B300 D64 grid-{cap} manifest not found")
        nvmx[key] = suite_result_map(path)[key]
        grid_sources[key] = (cap, path)

    rows = []
    for heads, seqlen in sorted(expected):
        key = (heads, seqlen)
        mx_benchmark = nvmx[key]["benchmark"]
        nv_benchmark = nvnv[key]["benchmark"]
        mx_error = mx_benchmark["correctness"]["tk_vs_bf16_output"]
        nv_error = nv_benchmark["correctness"]["tk_vs_bf16_output"]
        mx_time = tk_value(mx_benchmark["timing_ms"])
        nv_time = tk_value(nv_benchmark["timing_ms"])
        bf16_time = float(mx_benchmark["timing_ms"]["hao_native_bf16"])
        gb_mx_benchmark = gb200_nvmx[key]["benchmark"]
        gb_nv_benchmark = gb200_nvnv[key]["benchmark"]
        gb_mx_error = gb_mx_benchmark["correctness"]["tk_vs_bf16_output"]
        gb_nv_error = gb_nv_benchmark["correctness"]["tk_vs_bf16_output"]
        gb_mx_time = tk_value(gb_mx_benchmark["timing_ms"])
        gb_nv_time = tk_value(gb_nv_benchmark["timing_ms"])
        grid_cap, source_path = grid_sources.get(
            key, (148, nvmx_density_two_path)
        )
        rows.append(
            {
                "heads": heads,
                "seqlen": seqlen,
                "physical_grid_ctas": grid_cap,
                "nvmx_time_ms": mx_time,
                "nvmx_tflops": tk_value(mx_benchmark["tflops"]),
                "nvmx_speedup_bf16": bf16_time / mx_time,
                "nvmx_cosine": float(mx_error["cosine"]),
                "nvmx_relative_l2": float(mx_error["relative_l2"]),
                "nvmx_rmse": float(mx_error["rmse"]),
                "nvnv_time_ms": nv_time,
                "nvnv_cosine": finite_metric(nv_error["cosine"]),
                "nvnv_relative_l2": finite_metric(nv_error["relative_l2"]),
                "nvnv_rmse": finite_metric(nv_error["rmse"]),
                "nvnv_finite": all(
                    finite_metric(nv_error[name]) is not None
                    for name in ("cosine", "relative_l2", "rmse")
                ),
                "gb200_nvmx_time_ms": gb_mx_time,
                "gb200_nvmx_tflops": tk_value(gb_mx_benchmark["tflops"]),
                "gb200_nvmx_cosine": float(gb_mx_error["cosine"]),
                "gb200_nvmx_relative_l2": float(gb_mx_error["relative_l2"]),
                "gb200_nvnv_time_ms": gb_nv_time,
                "gb200_nvnv_tflops": tk_value(gb_nv_benchmark["tflops"]),
                "gb200_nvnv_cosine": finite_metric(gb_nv_error["cosine"]),
                "gb200_nvnv_relative_l2": finite_metric(
                    gb_nv_error["relative_l2"]
                ),
                "gb200_nvnv_finite": all(
                    finite_metric(gb_nv_error[name]) is not None
                    for name in ("cosine", "relative_l2", "rmse")
                ),
                "bf16_time_ms": bf16_time,
                "source": str(source_path.relative_to(ROOT)),
            }
        )

    density_two_wins = sum(
        tk_value(nvmx_density_two["benchmark"]["timing_ms"])
        <= tk_value(density_one[key]["benchmark"]["timing_ms"])
        for key, nvmx_density_two in suite_result_map(
            nvmx_density_two_path
        ).items()
    )
    saturated = [row for row in rows if row["seqlen"] >= 4096]
    finite_nvnv = [row for row in rows if row["nvnv_finite"]]
    safe_generation_ratios = [
        row["gb200_nvmx_time_ms"] / row["nvmx_time_ms"] for row in rows
    ]
    safe_saturated_generation_ratios = [
        row["gb200_nvmx_time_ms"] / row["nvmx_time_ms"]
        for row in saturated
    ]

    bounded_path = find_one(
        "artifacts/*/workers/0/fp4-fa4-b300-d64-nvnv-bounded/"
        "bounded_mode4/manifest.json"
    )
    bounded_rows = []
    if bounded_path is not None:
        for (heads, seqlen), case in sorted(suite_result_map(bounded_path).items()):
            benchmark = case["benchmark"]
            error = benchmark["correctness"]["tk_vs_bf16_output"]
            bounded_rows.append(
                {
                    "heads": heads,
                    "seqlen": seqlen,
                    "time_ms": tk_value(benchmark["timing_ms"]),
                    "speedup_bf16": tk_value(
                        benchmark["speedup_vs_hao_bf16"]
                    ),
                    "cosine": float(error["cosine"]),
                    "relative_l2": float(error["relative_l2"]),
                    "actual_nonfinite": int(error["actual_nonfinite"]),
                    "source": str(bounded_path.relative_to(ROOT)),
                }
            )

    promoted_path = find_one(
        "artifacts/*/workers/0/fp4-fa4-b300-d64-promoted-policy/"
        "default/manifest.json"
    )
    promoted_rows = []
    if promoted_path is not None:
        expected_grids = {(32, 2048): 136, (64, 1024): 128, (24, 4096): 148}
        promoted_cases = suite_result_map(promoted_path)
        if set(promoted_cases) != set(expected_grids):
            raise RuntimeError("promoted D64 policy manifest has the wrong shapes")
        for key, case in sorted(promoted_cases.items()):
            benchmark = case["benchmark"]
            topology = benchmark["topology"]
            error = benchmark["correctness"]["tk_vs_bf16_output"]
            build = case["build"]
            row = {
                "heads": key[0],
                "seqlen": key[1],
                "time_ms": tk_value(benchmark["timing_ms"]),
                "physical_grid_ctas": int(topology["physical_grid_ctas"]),
                "native_ex2_density": int(topology["mx_mode23_native_density"]),
                "actual_nonfinite": int(error["actual_nonfinite"]),
                "registers": int(build["registers"]),
                "barriers": int(build["barriers"]),
                "spill_load_bytes": int(build["spill_load_bytes"]),
                "spill_store_bytes": int(build["spill_store_bytes"]),
                "source": str(promoted_path.relative_to(ROOT)),
            }
            if row["physical_grid_ctas"] != expected_grids[key]:
                raise RuntimeError(f"promoted D64 grid drift at {key}: {row}")
            if (
                row["native_ex2_density"] != 2
                or row["actual_nonfinite"] != 0
                or row["registers"] != 128
                or row["barriers"] != 1
                or row["spill_load_bytes"] != 0
                or row["spill_store_bytes"] != 0
            ):
                raise RuntimeError(f"promoted D64 policy drift at {key}: {row}")
            promoted_rows.append(row)

    return {
        "rows": rows,
        "density_two_wins": density_two_wins,
        "saturated_speedup_min": min(
            row["nvmx_speedup_bf16"] for row in saturated
        ),
        "saturated_speedup_max": max(
            row["nvmx_speedup_bf16"] for row in saturated
        ),
        "saturated_nvmx_vs_nvnv_min": min(
            row["nvnv_time_ms"] / row["nvmx_time_ms"] for row in saturated
        ),
        "saturated_nvmx_vs_nvnv_max": max(
            row["nvnv_time_ms"] / row["nvmx_time_ms"] for row in saturated
        ),
        "nvnv_finite_cases": len(finite_nvnv),
        "gb200_nvnv_finite_cases": sum(row["gb200_nvnv_finite"] for row in rows),
        "safe_cross_generation_geomean": geometric_mean(safe_generation_ratios),
        "safe_cross_generation_wins": sum(
            ratio > 1.0 for ratio in safe_generation_ratios
        ),
        "safe_saturated_cross_generation_geomean": geometric_mean(
            safe_saturated_generation_ratios
        ),
        "safe_saturated_cross_generation_min": min(
            safe_saturated_generation_ratios
        ),
        "safe_saturated_cross_generation_max": max(
            safe_saturated_generation_ratios
        ),
        "bounded_nvnv": bounded_rows,
        "promoted_policy": promoted_rows,
    }


def format_d64_tex_rows(matrix: dict[str, Any]) -> str:
    selected = {
        (12, 4096),
        (24, 4096),
        (24, 32768),
        (32, 2048),
        (32, 32768),
        (64, 1024),
        (64, 4096),
        (64, 32768),
    }
    rendered = []
    for row in matrix["rows"]:
        if (row["heads"], row["seqlen"]) not in selected:
            continue
        nv_status = "finite" if row["nvnv_finite"] else "non-finite"
        rendered.append(
            f"{row['heads']} & {row['seqlen']} & {row['nvmx_time_ms']:.6f} & "
            f"{row['nvmx_tflops']:.0f} & "
            f"{row['nvmx_cosine']:.6f} & {row['nvmx_relative_l2']:.6f} & "
            f"{row['nvnv_time_ms']:.6f} & "
            f"{optional_metric(row['nvnv_cosine'])} & "
            f"{optional_metric(row['nvnv_relative_l2'])} & {nv_status} \\\\"
        )
    return "\n".join(rendered) + "\n\\bottomrule\n"


def cross_generation_row(
    *,
    dim: int,
    heads: int,
    seqlen: int,
    gb200_route: str,
    gb200_time_ms: float,
    b300_route: str,
    b300_time_ms: float,
    gb200_source: str,
    b300_source: str,
    b300_error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "dim": dim,
        "heads": heads,
        "seqlen": seqlen,
        "gb200_route": gb200_route,
        "gb200_time_ms": gb200_time_ms,
        "gb200_tflops": attention_tflops(
            batch=1,
            seqlen=seqlen,
            heads=heads,
            dqk=dim,
            dvo=dim,
            time_ms=gb200_time_ms,
        ),
        "b300_route": b300_route,
        "b300_time_ms": b300_time_ms,
        "b300_tflops": attention_tflops(
            batch=1,
            seqlen=seqlen,
            heads=heads,
            dqk=dim,
            dvo=dim,
            time_ms=b300_time_ms,
        ),
        "b300_latency_delta_pct": 100.0 * (b300_time_ms / gb200_time_ms - 1.0),
        "gb200_source": gb200_source,
        "b300_source": b300_source,
    }
    if b300_error is None:
        row.update(
            {
                "b300_cosine_min": None,
                "b300_cosine_max": None,
                "b300_relative_l2_min": None,
                "b300_relative_l2_max": None,
            }
        )
    elif "cosine_min" in b300_error:
        row.update(
            {
                "b300_cosine_min": float(b300_error["cosine_min"]),
                "b300_cosine_max": float(b300_error["cosine_max"]),
                "b300_relative_l2_min": float(b300_error["relative_l2_min"]),
                "b300_relative_l2_max": float(b300_error["relative_l2_max"]),
            }
        )
    else:
        cosine = float(b300_error["cosine"])
        relative_l2 = float(b300_error["relative_l2"])
        row.update(
            {
                "b300_cosine_min": cosine,
                "b300_cosine_max": cosine,
                "b300_relative_l2_min": relative_l2,
                "b300_relative_l2_max": relative_l2,
            }
        )
    return row


def d64_cross_generation_rows(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in matrix["rows"]:
        if row["heads"] not in (24, 64):
            continue
        if row["gb200_nvmx_time_ms"] <= row["gb200_nvnv_time_ms"]:
            gb_route = "NV/MX"
            gb_time = row["gb200_nvmx_time_ms"]
        else:
            gb_route = "NV/NV"
            gb_time = row["gb200_nvnv_time_ms"]
        if row["nvmx_time_ms"] <= row["nvnv_time_ms"]:
            b300_route = "NV/MX"
            b300_time = row["nvmx_time_ms"]
        else:
            b300_route = "NV/NV"
            b300_time = row["nvnv_time_ms"]
        rows.append(
            cross_generation_row(
                dim=64,
                heads=row["heads"],
                seqlen=row["seqlen"],
                gb200_route=gb_route,
                gb200_time_ms=gb_time,
                b300_route=b300_route,
                b300_time_ms=b300_time,
                gb200_source="fp4_fa4_d64_gb200_full_20260802",
                b300_source=row["source"],
            )
        )
    return rows


def d128_cross_generation_rows(
    *,
    gb200: dict[str, Any],
    unified: dict[str, Any],
    headline: dict[str, Any],
    three_k: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    gb200_by_shape = {
        row["shape"]: row
        for row in gb200["rows"]
        if row["provider"] == "TK NV/MX fast"
    }
    unified_by_shape = {
        row["shape"]: row
        for row in unified["cross_shape"]
        if row["family"] == "nvmx-fast"
    }

    short_path = find_one(
        "artifacts/*/workers/0/fp4-fa4-b300-d128-order-recheck/"
        "order1_kv06_g148_s0_00_s1_15/cases/"
        "b1_s4096_h24_d128_nvmx-fast.json"
    )
    saturated_path = find_one(
        "artifacts/*/workers/0/fp4-fa4-b300-d128-saturated-p-balance/"
        "benchmark_combo_d1_n8_s08.json"
    )
    long_mid_path = find_one(
        "artifacts/*/workers/0/fp4-fa4-b300-d128-saturated-prefetch-sfu/"
        "summary.json"
    )
    if None in (short_path, saturated_path, long_mid_path):
        raise FileNotFoundError("final B300 D128 comparison records not found")

    short_benchmark = load_json(short_path)["benchmark"]
    saturated = load_json(saturated_path)
    long_mid = load_json(long_mid_path)
    headline_row = next(row for row in headline["rows"] if row["provider"] == "tk_nv_mx")
    standard_three_k = next(
        row for row in three_k if row["seqlen"] == 8192 and row["heads"] == 64
    )

    records = [
        (
            24,
            4096,
            float(gb200_by_shape["b1_s4096_h24_d128"]["time_ms"]),
            tk_value(short_benchmark["timing_ms"]),
            str(GB200_SUMMARY.relative_to(ROOT.parent)),
            str(short_path.relative_to(ROOT)),
            short_benchmark["correctness"]["tk_vs_bf16_output"],
        ),
        (
            64,
            4096,
            float(unified_by_shape["b1_s4096_h64_d128"]["time_ms"]),
            float(saturated["timing_ms"]),
            str(UNIFIED_SUMMARY.relative_to(ROOT.parent)),
            str(saturated_path.relative_to(ROOT)),
            saturated["correctness_global"],
        ),
        (
            64,
            6144,
            float(long_mid["matched_gb200_ms"]),
            float(long_mid["rows"]["kv6"]["median_ms"]),
            "matched GB200 record cited by saturated-prefetch-sfu/summary.json",
            str(long_mid_path.relative_to(ROOT)),
            long_mid["rows"]["kv6"]["correctness_global"],
        ),
        (
            64,
            8192,
            float(unified_by_shape["b1_s8192_h64_d128"]["time_ms"]),
            float(standard_three_k["median_ms"]),
            str(UNIFIED_SUMMARY.relative_to(ROOT.parent)),
            standard_three_k["source"],
            standard_three_k,
        ),
        (
            24,
            32768,
            float(gb200_by_shape["b1_s32768_h24_d128"]["time_ms"]),
            float(headline_row["timing_ms"]),
            str(GB200_SUMMARY.relative_to(ROOT.parent)),
            "B300 headline long-context record",
            headline_row["correctness_global"],
        ),
    ]
    return [
        cross_generation_row(
            dim=128,
            heads=heads,
            seqlen=seqlen,
            gb200_route="NV/MX",
            gb200_time_ms=gb_time,
            b300_route="NV/MX",
            b300_time_ms=b300_time,
            gb200_source=gb_source,
            b300_source=b300_source,
            b300_error=b300_error,
        )
        for (
            heads,
            seqlen,
            gb_time,
            b300_time,
            gb_source,
            b300_source,
            b300_error,
        ) in records
    ]


def format_cross_generation_tex_rows(rows: list[dict[str, Any]]) -> str:
    rendered = []
    for row in rows:
        gb_route = rf"\code{{{row['gb200_route']}}}"
        if row["gb200_route"] == "NV/NV":
            gb_route += r"\textsuperscript{\dagger}"
        b300_route = rf"\code{{{row['b300_route']}}}"
        rendered.append(
            f"{row['dim']} & {row['heads']} & {row['seqlen']} & "
            f"{gb_route} & {row['gb200_time_ms']:.6f} & "
            f"{row['gb200_tflops']:.0f} & {b300_route} & "
            f"{row['b300_time_ms']:.6f} & {row['b300_tflops']:.0f} & "
            f"{row['b300_latency_delta_pct']:+.2f}\\% \\\\"
        )
    return "\n".join(rendered) + "\n\\bottomrule\n"


def primary_generation_rows(
    *,
    matrix: dict[str, Any],
    d128_rows: list[dict[str, Any]],
    three_k: list[dict[str, Any]],
    published_hao: dict[tuple[int, int, int, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    retained_d128 = list(d128_rows)
    peak = next(
        row for row in three_k if row["heads"] == 64 and row["seqlen"] == 9472
    )
    retained_d128.append(
        {
            "dim": 128,
            "heads": 64,
            "seqlen": 9472,
            "gb200_route": None,
            "gb200_time_ms": None,
            "gb200_tflops": None,
            "b300_route": "NV/MX",
            "b300_time_ms": peak["median_ms"],
            "b300_tflops": peak["tflops"],
            "b300_latency_delta_pct": None,
            "gb200_source": None,
            "b300_source": peak["source"],
            "b300_cosine_min": peak["cosine_min"],
            "b300_cosine_max": peak["cosine_max"],
            "b300_relative_l2_min": peak["relative_l2_min"],
            "b300_relative_l2_max": peak["relative_l2_max"],
        }
    )
    retained_d128.sort(key=lambda row: (row["seqlen"], row["heads"]))

    selected_d64 = {(24, 4096), (64, 4096), (24, 32768), (64, 32768)}
    retained_d64 = []
    for record in matrix["rows"]:
        key = (record["heads"], record["seqlen"])
        if key not in selected_d64:
            continue
        retained_d64.append(
            cross_generation_row(
                dim=64,
                heads=record["heads"],
                seqlen=record["seqlen"],
                gb200_route="NV/MX",
                gb200_time_ms=record["gb200_nvmx_time_ms"],
                b300_route="NV/MX",
                b300_time_ms=record["nvmx_time_ms"],
                gb200_source="fp4_fa4_d64_gb200_full_20260802",
                b300_source=record["source"],
                b300_error={
                    "cosine": record["nvmx_cosine"],
                    "relative_l2": record["nvmx_relative_l2"],
                },
            )
        )
    retained_d64.sort(key=lambda row: (row["seqlen"], row["heads"]))
    rows = retained_d128 + retained_d64
    for row in rows:
        published = published_hao.get(
            (1, row["seqlen"], row["heads"], row["dim"])
        )
        row["hao_gb300_nvfp8_tflops"] = (
            None if published is None else published["gb300"]["nvfp4_fp8"]
        )
        row["hao_gb300_nvfp8_cosine"] = (
            None
            if published is None
            else published["nvfp4_fp8_precision"]["cosine"]
        )
    return rows


def format_metric_range(
    row: dict[str, Any], minimum_key: str, maximum_key: str
) -> str:
    minimum = row[minimum_key]
    maximum = row[maximum_key]
    if minimum is None or maximum is None:
        return "--"
    if math.isclose(float(minimum), float(maximum), rel_tol=0.0, abs_tol=5.0e-7):
        return f"{float(minimum):.4f}"
    return f"{float(minimum):.6f}--{float(maximum):.6f}"


def format_primary_generation_tex_rows(rows: list[dict[str, Any]]) -> str:
    rendered = []
    for row in rows:
        if row["gb200_time_ms"] is None:
            gb200 = "--"
            delta = "--"
        else:
            gb200 = f"{row['gb200_time_ms']:.6f} / {row['gb200_tflops']:.0f}"
            delta = f"{row['b300_latency_delta_pct']:+.2f}\\%"
        b300 = f"{row['b300_time_ms']:.6f} / {row['b300_tflops']:.0f}"
        if row["b300_tflops"] >= 3000.0:
            b300 = rf"\textbf{{{b300}}}"
        error = (
            format_metric_range(
                row, "b300_cosine_min", "b300_cosine_max"
            )
            + " / "
            + format_metric_range(
                row, "b300_relative_l2_min", "b300_relative_l2_max"
            )
        )
        hao = row["hao_gb300_nvfp8_tflops"]
        hao_cosine = row["hao_gb300_nvfp8_cosine"]
        hao_cell = (
            "--"
            if hao is None or hao_cosine is None
            else f"{hao:.0f} / {hao_cosine:.4f}"
        )
        rendered.append(
            f"D{row['dim']}/H{row['heads']}/S{row['seqlen']} & {gb200} & "
            f"{b300} & {hao_cell} & {delta} & {error} \\\\"
        )
    return "\n".join(rendered) + "\n\\bottomrule\n"


def format_primary_hao_b300_tex_rows(rows: list[dict[str, Any]]) -> str:
    rendered = []
    for dim in (128, 64):
        ours = next(
            row
            for row in rows
            if row["dim"] == dim and row["label"] == "TK NV/MX fast"
        )
        hao = next(
            row
            for row in rows
            if row["dim"] == dim and row["label"] == "HAO NV/FP8"
        )
        bf16 = next(
            row
            for row in rows
            if row["dim"] == dim and row["label"] == "HAO BF16"
        )
        rendered.append(
            f"B1/S32768/H24/D{dim} & "
            f"{ours['timing_ms']:.3f} / {ours['tflops']:.0f} & "
            f"{ours['cosine']:.4f} / {ours['relative_l2']:.4f} & "
            f"{hao['timing_ms']:.3f} / {hao['tflops']:.0f} & "
            f"{hao['cosine']:.4f} / -- & "
            f"{bf16['timing_ms']:.3f} / {bf16['tflops']:.0f} \\\\"
        )
    return "\n".join(rendered) + "\n\\bottomrule\n"


def format_headline_tex_rows(rows: list[dict[str, Any]]) -> str:
    rendered = []
    for row in rows:
        label = row["label"].replace("_", r"\_")
        source = row["provenance"].replace("_", r"\_")
        rendered.append(
            f"{row['dim']} & {label} & {source} & "
            f"{row['timing_ms']:.3f} & {row['tflops']:.0f} & "
            f"{row['ratio_to_hao_bf16']:.3f} & "
            f"{optional_metric(row['cosine'])} & "
            f"{optional_metric(row['relative_l2'])} \\\\"
        )
    return "\n".join(rendered) + "\n\\bottomrule\n"


def format_accuracy_matched_tex_rows(rows: list[dict[str, Any]]) -> str:
    selected = [row for row in rows if row["dim"] == 128]
    rendered = []
    for row in selected:
        label = row["label"]
        if row["provenance"] == "HAO published":
            label += r"\textsuperscript{p}"
        rendered.append(
            f"{label} & {row['timing_ms']:.3f} & {row['tflops']:.0f} & "
            f"{optional_metric(row['cosine'], 4)} & "
            f"{optional_metric(row['relative_l2'], 4)} \\\\"
        )
    return "\n".join(rendered) + "\n\\bottomrule\n"


def measured_headline_row(
    *,
    dim: int,
    label: str,
    provenance: str,
    record: dict[str, Any],
    hao_bf16_tflops: float,
    source: str,
) -> dict[str, Any]:
    error = record["correctness_global"]
    tflops = float(record["tflops"])
    return {
        "dim": dim,
        "label": label,
        "provenance": provenance,
        "timing_ms": float(record["timing_ms"]),
        "tflops": tflops,
        "ratio_to_hao_bf16": tflops / hao_bf16_tflops,
        "cosine": float(error["cosine"]),
        "relative_l2": float(error["relative_l2"]),
        "rmse": float(error["rmse"]),
        "source": source,
    }


def published_headline_row(
    *,
    dim: int,
    label: str,
    tflops: float,
    hao_bf16_tflops: float,
    cosine: float | None,
) -> dict[str, Any]:
    batch, seqlen, heads = 1, 32768, 24
    operation_count = batch * heads * 2 * seqlen * seqlen * (dim + dim)
    return {
        "dim": dim,
        "label": label,
        "provenance": "HAO published",
        "timing_ms": operation_count / (tflops * 1.0e9),
        "tflops": tflops,
        "ratio_to_hao_bf16": tflops / hao_bf16_tflops,
        "cosine": cosine,
        "relative_l2": None,
        "rmse": None,
        "source": "HAO flash-attention-fp4 README, GB300 tables",
    }


def format_macros(summary: dict[str, Any]) -> str:
    best = summary["best_b300"]
    quality = summary["density_two_b300"]
    compatibility = summary["rows"][0]
    gb200 = summary["gb200_reference"]
    long_rows = summary["long_context_rows"]
    long_nvmx = next(
        row
        for row in long_rows
        if row["dim"] == 128 and row["label"] == "TK NV/MX fast"
    )
    long_nvfp8 = next(
        row
        for row in long_rows
        if row["dim"] == 128
        and row["label"] == "TK NV/FP8 optimized"
    )
    d64_nvfp8 = next(
        row
        for row in long_rows
        if row["dim"] == 64
        and row["label"] == "TK NV/FP8 optimized"
    )
    d64 = summary["d64"]
    d64_long = next(
        row
        for row in d64["rows"]
        if row["heads"] == 24 and row["seqlen"] == 32768
    )
    d64_bounded_long = next(
        row
        for row in d64["bounded_nvnv"]
        if row["heads"] == 32 and row["seqlen"] == 32768
    )
    cross_generation = summary["cross_generation_rows"]
    d128_short = next(
        row
        for row in cross_generation
        if row["dim"] == 128 and row["heads"] == 24 and row["seqlen"] == 4096
    )
    d128_saturated = next(
        row
        for row in cross_generation
        if row["dim"] == 128 and row["heads"] == 64 and row["seqlen"] == 4096
    )
    d128_mid = next(
        row
        for row in cross_generation
        if row["dim"] == 128 and row["heads"] == 64 and row["seqlen"] == 6144
    )
    d128_three_k = summary["d128_three_k_rows"]
    d128_three_k_standard = next(
        row
        for row in d128_three_k
        if row["heads"] == 64 and row["seqlen"] == 8192
    )
    d128_three_k_peak = max(d128_three_k, key=lambda row: row["tflops"])
    d128_three_k_generation = next(
        row
        for row in cross_generation
        if row["dim"] == 128 and row["heads"] == 64 and row["seqlen"] == 8192
    )
    d128_long = next(
        row
        for row in cross_generation
        if row["dim"] == 128 and row["heads"] == 24 and row["seqlen"] == 32768
    )
    normalized = summary["generation_normalization"]
    return "\n".join(
        [
            "% Generated by fp4_fa4_b300_tuning_20260802/build_summary.py.",
            rf"\newcommand{{\BThreeBestTime}}{{{best['timing_ms']:.6f}}}",
            rf"\newcommand{{\BThreeBestTflops}}{{{best['tflops']:.0f}}}",
            rf"\newcommand{{\BThreeBestDensity}}{{{best['native_ex2_density']}}}",
            rf"\newcommand{{\BThreeBestCosine}}{{{best['cosine']:.6f}}}",
            rf"\newcommand{{\BThreeBestRelLTwo}}{{{best['relative_l2']:.6f}}}",
            rf"\newcommand{{\BThreeGridFixSpeedup}}{{{compatibility['timing_ms'] / best['timing_ms']:.3f}}}",
            rf"\newcommand{{\BThreeBestVsGBSlowdownPct}}{{{100.0 * (best['timing_ms'] / gb200['timing_ms'] - 1.0):.2f}}}",
            rf"\newcommand{{\BThreeDensityTwoTime}}{{{quality['timing_ms']:.6f}}}",
            rf"\newcommand{{\BThreeDensityTwoTflops}}{{{quality['tflops']:.0f}}}",
            rf"\newcommand{{\BThreeDensityTwoCosine}}{{{quality['cosine']:.6f}}}",
            rf"\newcommand{{\BThreeDensityTwoRelLTwo}}{{{quality['relative_l2']:.6f}}}",
            rf"\newcommand{{\BThreeLongNVMXTime}}{{{long_nvmx['timing_ms']:.6f}}}",
            rf"\newcommand{{\BThreeLongNVMXTflops}}{{{long_nvmx['tflops']:.0f}}}",
            rf"\newcommand{{\BThreeLongNVFPTime}}{{{long_nvfp8['timing_ms']:.6f}}}",
            rf"\newcommand{{\BThreeLongNVFPTflops}}{{{long_nvfp8['tflops']:.0f}}}",
            rf"\newcommand{{\BThreeDsixtyfourNVFPTime}}{{{d64_nvfp8['timing_ms']:.6f}}}",
            rf"\newcommand{{\BThreeDsixtyfourNVFPTflops}}{{{d64_nvfp8['tflops']:.0f}}}",
            rf"\newcommand{{\BThreeDsixtyfourNVMXTime}}{{{d64_long['nvmx_time_ms']:.6f}}}",
            rf"\newcommand{{\BThreeDsixtyfourNVMXTflops}}{{{d64_long['nvmx_tflops']:.0f}}}",
            rf"\newcommand{{\BThreeDsixtyfourNVMXSpeedup}}{{{d64_long['nvmx_speedup_bf16']:.3f}}}",
            rf"\newcommand{{\BThreeDsixtyfourNVMXCosine}}{{{d64_long['nvmx_cosine']:.6f}}}",
            rf"\newcommand{{\BThreeDsixtyfourNVMXRelLTwo}}{{{d64_long['nvmx_relative_l2']:.6f}}}",
            rf"\newcommand{{\BThreeDsixtyfourMinSpeedup}}{{{d64['saturated_speedup_min']:.3f}}}",
            rf"\newcommand{{\BThreeDsixtyfourMaxSpeedup}}{{{d64['saturated_speedup_max']:.3f}}}",
            rf"\newcommand{{\BThreeDsixtyfourDensityWins}}{{{d64['density_two_wins']}}}",
            rf"\newcommand{{\BThreeDsixtyfourNVNVFiniteCases}}{{{d64['nvnv_finite_cases']}}}",
            rf"\newcommand{{\BThreeDsixtyfourSafeGenerationGeo}}{{{d64['safe_cross_generation_geomean']:.3f}}}",
            rf"\newcommand{{\BThreeDsixtyfourSafeGenerationWins}}{{{d64['safe_cross_generation_wins']}}}",
            rf"\newcommand{{\BThreeDsixtyfourSafeSaturatedGenerationGeo}}{{{d64['safe_saturated_cross_generation_geomean']:.3f}}}",
            rf"\newcommand{{\BThreeDsixtyfourBoundedNVNVTime}}{{{d64_bounded_long['time_ms']:.6f}}}",
            rf"\newcommand{{\BThreeDsixtyfourBoundedNVNVSpeedup}}{{{d64_bounded_long['speedup_bf16']:.3f}}}",
            rf"\newcommand{{\BThreeDOneTwentyEightShortTime}}{{{d128_short['b300_time_ms']:.6f}}}",
            rf"\newcommand{{\BThreeDOneTwentyEightShortTflops}}{{{d128_short['b300_tflops']:.0f}}}",
            rf"\newcommand{{\BThreeDOneTwentyEightShortGainPct}}{{{-d128_short['b300_latency_delta_pct']:.2f}}}",
            rf"\newcommand{{\BThreeDOneTwentyEightSaturatedTime}}{{{d128_saturated['b300_time_ms']:.6f}}}",
            rf"\newcommand{{\BThreeDOneTwentyEightSaturatedTflops}}{{{d128_saturated['b300_tflops']:.0f}}}",
            rf"\newcommand{{\BThreeDOneTwentyEightSaturatedGainPct}}{{{-d128_saturated['b300_latency_delta_pct']:.2f}}}",
            rf"\newcommand{{\BThreeDOneTwentyEightMidTime}}{{{d128_mid['b300_time_ms']:.6f}}}",
            rf"\newcommand{{\BThreeDOneTwentyEightMidTflops}}{{{d128_mid['b300_tflops']:.0f}}}",
            rf"\newcommand{{\BThreeDOneTwentyEightMidGainPct}}{{{-d128_mid['b300_latency_delta_pct']:.2f}}}",
            rf"\newcommand{{\BThreeThreeKStandardTime}}{{{d128_three_k_standard['median_ms']:.6f}}}",
            rf"\newcommand{{\BThreeThreeKStandardTflops}}{{{d128_three_k_standard['tflops']:.0f}}}",
            rf"\newcommand{{\BThreeThreeKStandardSpanPct}}{{{100.0 * d128_three_k_standard['relative_span']:.3f}}}",
            rf"\newcommand{{\BThreeThreeKStandardGainPct}}{{{-d128_three_k_generation['b300_latency_delta_pct']:.2f}}}",
            rf"\newcommand{{\BThreeThreeKPeakSeq}}{{{d128_three_k_peak['seqlen']}}}",
            rf"\newcommand{{\BThreeThreeKPeakTime}}{{{d128_three_k_peak['median_ms']:.6f}}}",
            rf"\newcommand{{\BThreeThreeKPeakTflops}}{{{d128_three_k_peak['tflops']:.0f}}}",
            rf"\newcommand{{\BThreeThreeKPeakSpanPct}}{{{100.0 * d128_three_k_peak['relative_span']:.3f}}}",
            rf"\newcommand{{\BThreeDOneTwentyEightLongDeltaPct}}{{{d128_long['b300_latency_delta_pct']:.2f}}}",
            rf"\newcommand{{\BThreePerSMClockGainPct}}{{{100.0 * (normalized['b300_per_sm_clock_vs_gb200'] - 1.0):.2f}}}",
            rf"\newcommand{{\BThreeSMClockEnvelopeDeficitPct}}{{{100.0 * (1.0 - normalized['b300_sm_clock_envelope_ratio']):.2f}}}",
            rf"\newcommand{{\BThreeEnvelopeTflops}}{{{normalized['b300_rate_at_gb200_sm_clock_envelope_tflops']:.0f}}}",
            "",
        ]
    )


def write_outputs(summary: dict[str, Any], *, write_summary: bool) -> None:
    """Render report tables from a validated aggregate summary."""
    rows = summary["rows"]
    long_context_rows = summary["long_context_rows"]
    d64 = summary["d64"]
    d128_three_k = summary["d128_three_k_rows"]
    cross_generation_rows = summary["cross_generation_rows"]
    primary_rows = summary["primary_generation_rows"]

    TABLE_DIR.mkdir(exist_ok=True)
    if write_summary:
        (ROOT / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    (TABLE_DIR / "b300_tuning_rows.tex").write_text(
        format_tex_rows(rows), encoding="utf-8"
    )
    (TABLE_DIR / "b300_tuning_macros.tex").write_text(
        format_macros(summary), encoding="utf-8"
    )
    (TABLE_DIR / "b300_headline_rows.tex").write_text(
        format_headline_tex_rows(long_context_rows), encoding="utf-8"
    )
    (TABLE_DIR / "b300_d64_rows.tex").write_text(
        format_d64_tex_rows(d64), encoding="utf-8"
    )
    (TABLE_DIR / "b300_cross_generation_rows.tex").write_text(
        format_cross_generation_tex_rows(cross_generation_rows), encoding="utf-8"
    )
    (TABLE_DIR / "b300_3ktflops_rows.tex").write_text(
        format_three_k_tex_rows(d128_three_k), encoding="utf-8"
    )
    (TABLE_DIR / "primary_cross_generation_rows.tex").write_text(
        format_primary_generation_tex_rows(primary_rows), encoding="utf-8"
    )
    (TABLE_DIR / "primary_cross_generation_d128_rows.tex").write_text(
        format_primary_generation_tex_rows(
            [row for row in primary_rows if row["dim"] == 128]
        ),
        encoding="utf-8",
    )
    (TABLE_DIR / "primary_hao_b300_rows.tex").write_text(
        format_primary_hao_b300_tex_rows(long_context_rows), encoding="utf-8"
    )
    (TABLE_DIR / "accuracy_matched_rows.tex").write_text(
        format_accuracy_matched_tex_rows(long_context_rows), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-summary",
        action="store_true",
        help=(
            "render tables from the committed aggregate summary without "
            "requiring the private raw B300 capture archive"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.from_summary:
        summary = load_json(ROOT / "summary.json")
        if summary.get("schema") != "fp4-fa4-b300-tuning-v7":
            raise RuntimeError("unsupported committed B300 summary schema")
        write_outputs(summary, write_summary=False)
        return

    published_hao = published_hao_by_shape()
    smoke_path = find_one(
        "artifacts/*/workers/0/fp4-fa4-b300-smoke/"
        "b1_s4096_h24_d128_fast.json"
    )
    if smoke_path is None:
        raise FileNotFoundError("downloaded B300 compatibility JSON not found")

    smoke = load_json(smoke_path)
    rows = [
        canonical_row(
            label="B300 compatibility",
            hardware=smoke["hardware"]["name"],
            grid_sms=152,
            density=int(smoke["topology"]["mx_mode23_native_density"]),
            record=smoke,
            source=str(smoke_path.relative_to(ROOT)),
        )
    ]

    sweep_path = find_one(
        "artifacts/*/workers/0/fp4-fa4-b300-visible-sm-density/"
        "b1_s4096_h24_d128_visible_sm_density_sweep.json"
    )
    if sweep_path is not None:
        sweep = load_json(sweep_path)
        for record in sweep["rows"]:
            rows.append(
                canonical_row(
                    label=f"B300 density {record['native_ex2_density']}",
                    hardware=sweep["hardware"]["name"],
                    grid_sms=int(sweep["protocol"]["persistent_grid_sms"]),
                    density=int(record["native_ex2_density"]),
                    record=record,
                    source=str(sweep_path.relative_to(ROOT)),
                )
            )

    gb200 = load_json(GB200_SUMMARY)
    unified = load_json(UNIFIED_SUMMARY)
    gb200_row = next(
        row
        for row in gb200["rows"]
        if row["shape"] == "b1_s4096_h24_d128"
        and row["provider"] == "TK NV/MX fast"
    )
    gb200_reference = {
        "timing_ms": float(gb200_row["time_ms"]),
        "tflops": float(gb200_row["tflops"]),
        "cosine": float(gb200_row["cosine"]),
        "relative_l2": float(gb200_row["relative_l2"]),
        "rmse": float(gb200_row["rmse"]),
        "source": str(GB200_SUMMARY.relative_to(ROOT.parent)),
    }
    gb200_long_row = next(
        row
        for row in gb200["rows"]
        if row["shape"] == "b1_s32768_h24_d128"
        and row["provider"] == "TK NV/MX fast"
    )
    gb200_hardware = load_json(GB200_HARDWARE)

    best = min(rows, key=lambda row: row["timing_ms"])
    density_two = next(
        row
        for row in rows
        if row["persistent_grid_sms"] == 148
        and row["native_ex2_density"] == 2
    )
    if not all(math.isfinite(row["timing_ms"]) for row in rows):
        raise RuntimeError("non-finite B300 timing")

    d64 = d64_matrix()
    d128_three_k = d128_three_k_rows()

    headline_path = find_one(
        "artifacts/*/workers/0/fp4-fa4-b300-headline/"
        "b1_s32768_h24_d128_headline.json"
    )
    if headline_path is None:
        raise FileNotFoundError("downloaded B300 long-context JSON not found")
    headline = load_json(headline_path)
    headline_by_provider = {
        row["provider"]: row for row in headline["rows"]
    }
    optimized_path = find_one(
        "artifacts/*/workers/0/fp4-fa4-b300-nvfp8-d64/"
        "b1_s32768_h24_nvfp8_d128_d64.json"
    )
    optimized = load_json(optimized_path) if optimized_path is not None else None
    optimized_by_dim = (
        {int(row["shape"]["dim"]): row for row in optimized["rows"]}
        if optimized is not None
        else {}
    )
    hao_long = {
        dim: published_hao[(1, 32768, 24, dim)] for dim in (128, 64)
    }
    hao_bf16 = {
        dim: float(hao_long[dim]["gb300"]["bf16"]) for dim in (128, 64)
    }
    long_context_rows = [
        measured_headline_row(
            dim=128,
            label="TK NV/MX fast",
            provenance="measured",
            record=headline_by_provider["tk_nv_mx"],
            hao_bf16_tflops=hao_bf16[128],
            source=str(headline_path.relative_to(ROOT)),
        )
    ]
    if 128 in optimized_by_dim:
        long_context_rows.append(
            measured_headline_row(
                dim=128,
                label="TK NV/FP8 optimized",
                provenance="measured",
                record=optimized_by_dim[128],
                hao_bf16_tflops=hao_bf16[128],
                source=str(optimized_path.relative_to(ROOT)),
            )
        )
    long_context_rows.append(
        measured_headline_row(
            dim=128,
            label="TK NV/FP8 exact",
            provenance="measured",
            record=headline_by_provider["tk_nv_fp8"],
            hao_bf16_tflops=hao_bf16[128],
            source=str(headline_path.relative_to(ROOT)),
        )
    )
    long_context_rows.extend(
        [
            published_headline_row(
                dim=128,
                label="HAO NV/FP8",
                tflops=float(hao_long[128]["gb300"]["nvfp4_fp8"]),
                hao_bf16_tflops=hao_bf16[128],
                cosine=float(
                    hao_long[128]["nvfp4_fp8_precision"]["cosine"]
                ),
            ),
            published_headline_row(
                dim=128,
                label="HAO BF16",
                tflops=hao_bf16[128],
                hao_bf16_tflops=hao_bf16[128],
                cosine=1.0,
            ),
        ]
    )
    b300_nvmx = long_context_rows[0]
    b300_hardware = headline["hardware"]
    if optimized is not None:
        b300_hardware = optimized["hardware"]
    b300_sm_clock_rate = (
        b300_nvmx["tflops"]
        / float(b300_hardware["multiprocessor_count"])
        / float(b300_hardware["max_sm_clock_mhz"])
    )
    gb200_sm_clock_rate = (
        float(gb200_long_row["tflops"])
        / float(gb200_hardware["multiprocessor_count"])
        / float(gb200_hardware["max_sm_clock_mhz"])
    )
    generation_normalization = {
        "b300": {
            "tflops": b300_nvmx["tflops"],
            "multiprocessor_count": int(
                b300_hardware["multiprocessor_count"]
            ),
            "max_sm_clock_mhz": int(b300_hardware["max_sm_clock_mhz"]),
        },
        "gb200": {
            "tflops": float(gb200_long_row["tflops"]),
            "multiprocessor_count": int(
                gb200_hardware["multiprocessor_count"]
            ),
            "max_sm_clock_mhz": int(gb200_hardware["max_sm_clock_mhz"]),
        },
        "b300_aggregate_vs_gb200": (
            b300_nvmx["tflops"] / float(gb200_long_row["tflops"])
        ),
        "b300_sm_clock_envelope_ratio": (
            float(b300_hardware["multiprocessor_count"])
            * float(b300_hardware["max_sm_clock_mhz"])
            / (
                float(gb200_hardware["multiprocessor_count"])
                * float(gb200_hardware["max_sm_clock_mhz"])
            )
        ),
        "b300_per_sm_clock_vs_gb200": (
            b300_sm_clock_rate / gb200_sm_clock_rate
        ),
        "b300_rate_at_gb200_sm_clock_envelope_tflops": (
            b300_sm_clock_rate
            * float(gb200_hardware["multiprocessor_count"])
            * float(gb200_hardware["max_sm_clock_mhz"])
        ),
    }
    if 64 in optimized_by_dim:
        d64_long = next(
            row
            for row in d64["rows"]
            if row["heads"] == 24 and row["seqlen"] == 32768
        )
        long_context_rows.append(
            {
                "dim": 64,
                "label": "TK NV/MX fast",
                "provenance": "measured",
                "timing_ms": d64_long["nvmx_time_ms"],
                "tflops": d64_long["nvmx_tflops"],
                "ratio_to_hao_bf16": d64_long["nvmx_tflops"] / hao_bf16[64],
                "cosine": d64_long["nvmx_cosine"],
                "relative_l2": d64_long["nvmx_relative_l2"],
                "rmse": d64_long["nvmx_rmse"],
                "source": d64_long["source"],
            }
        )
        long_context_rows.append(
            measured_headline_row(
                dim=64,
                label="TK NV/FP8 optimized",
                provenance="measured",
                record=optimized_by_dim[64],
                hao_bf16_tflops=hao_bf16[64],
                source=str(optimized_path.relative_to(ROOT)),
            )
        )
    long_context_rows.extend(
        [
            published_headline_row(
                dim=64,
                label="HAO NV/FP8",
                tflops=float(hao_long[64]["gb300"]["nvfp4_fp8"]),
                hao_bf16_tflops=hao_bf16[64],
                cosine=float(
                    hao_long[64]["nvfp4_fp8_precision"]["cosine"]
                ),
            ),
            published_headline_row(
                dim=64,
                label="HAO BF16",
                tflops=hao_bf16[64],
                hao_bf16_tflops=hao_bf16[64],
                cosine=1.0,
            ),
        ]
    )
    cross_generation_rows = d64_cross_generation_rows(d64)
    cross_generation_rows.extend(
        d128_cross_generation_rows(
            gb200=gb200,
            unified=unified,
            headline=headline,
            three_k=d128_three_k,
        )
    )
    retained_d128_rows = [
        row for row in cross_generation_rows if row["dim"] == 128
    ]
    primary_rows = primary_generation_rows(
        matrix=d64,
        d128_rows=retained_d128_rows,
        three_k=d128_three_k,
        published_hao=published_hao,
    )
    summary = {
        "schema": "fp4-fa4-b300-tuning-v7",
        "shape": {"batch": 1, "seqlen": 4096, "heads": 24, "dim": 128},
        "gb200_reference": gb200_reference,
        "published_hao_source": str(PUBLISHED_HAO.relative_to(ROOT.parent)),
        "published_hao_b300_tflops": published_hao[
            (1, 4096, 24, 128)
        ]["gb300"],
        "rows": rows,
        "best_b300": best,
        "density_two_b300": density_two,
        "long_context_rows": long_context_rows,
        "d64": d64,
        "d128_three_k_rows": d128_three_k,
        "cross_generation_rows": cross_generation_rows,
        "primary_generation_rows": primary_rows,
        "generation_normalization": generation_normalization,
    }

    write_outputs(summary, write_summary=True)


if __name__ == "__main__":
    main()
