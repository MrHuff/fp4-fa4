#!/usr/bin/env python3
"""Merge unified FP4 FA4 shards into paper-ready speed/accuracy tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


VARIANT_LABELS = {
    "nv-nv": "TK NV/NV fixed schedule",
    "mx-nv": "TK MX/NV fixed schedule",
    "nv-mx": "TK NV/MX fixed schedule",
    "mx-mx": "TK MX/MX fixed schedule",
    "nvmx-fast": "TK NV/MX fast",
    "nvmx-balanced": "TK NV/MX balanced",
    "nvmx-accurate": "TK NV/MX accurate",
    "fp8": "TK NV/FP8",
}

VARIANT_ORDER = tuple(VARIANT_LABELS)
PUBLISHED_VARIANTS = tuple(
    variant for variant in VARIANT_ORDER if variant != "nvmx-balanced"
)
HEADLINE_LABEL = "b1_s4096_h24_d128"
SHORT_LABEL = "b1_s256_h16_d128"
CROSS_SHAPE_VARIANTS = (
    "nvmx-fast",
    "nvmx-accurate",
    "fp8",
    "hao-nv-nv",
    "hao-nv-fp8",
    "bf16",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    return parser.parse_args()


def metric(value: dict[str, Any], name: str) -> float | None:
    result = value.get(name)
    return float(result) if result is not None else None


def normalized_row(
    *,
    shape: dict[str, int],
    shape_label: str,
    provider: str,
    family: str,
    time_ms: float,
    bf16_ms: float,
    error: dict[str, Any],
) -> dict[str, Any]:
    return {
        "shape": shape_label,
        **shape,
        "provider": provider,
        "family": family,
        "time_ms": float(time_ms),
        "bf16_reference_ms": float(bf16_ms),
        "speedup_vs_bf16": float(bf16_ms / time_ms),
        "cosine_bf16": metric(error, "cosine"),
        "relative_l2_bf16": metric(error, "relative_l2"),
        "rmse_bf16": metric(error, "rmse"),
    }


def load_cases(root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    cases: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(root.glob("shard*/cases/*.json")):
        value = json.loads(path.read_text())
        if value.get("status") != "complete":
            continue
        key = (value["label"], value["variant"])
        if key in cases:
            raise RuntimeError(f"duplicate case {key}: {path}")
        value["source_file"] = str(path.relative_to(root))
        cases[key] = value
    expected_cases = 6 * len(VARIANT_ORDER)
    if len(cases) != expected_cases:
        raise RuntimeError(
            f"expected {expected_cases} complete cases, found {len(cases)}"
        )
    for key, value in cases.items():
        benchmark = value["benchmark"]
        if benchmark["shape"] != value["shape"]:
            raise RuntimeError(f"shape mismatch in case {key}")
        protocol = benchmark["protocol"]
        required = {
            "factory": "HAO create_nvfp4_attention_tensors",
            "warmup_ms": 300,
            "rep_ms": 3000,
            "timer": "triton.testing.do_bench median",
        }
        for field, expected in required.items():
            if protocol.get(field) != expected:
                raise RuntimeError(
                    f"case {key} has {field}={protocol.get(field)!r}, "
                    f"expected {expected!r}"
                )
    return cases


def collect_rows(
    cases: dict[tuple[str, str], dict[str, Any]],
    root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    shape_labels = sorted({label for label, _ in cases})
    rows: list[dict[str, Any]] = []
    reference_audit: dict[str, Any] = {}
    for label in shape_labels:
        shape_cases = {
            variant: case
            for (case_label, variant), case in cases.items()
            if case_label == label
        }
        missing = set(VARIANT_ORDER) - set(shape_cases)
        if missing:
            raise RuntimeError(f"{label} is missing variants: {sorted(missing)}")
        shape = shape_cases[VARIANT_ORDER[0]]["shape"]
        bf16_samples = {
            variant: float(case["benchmark"]["timing_ms"]["hao_native_bf16"])
            for variant, case in shape_cases.items()
        }
        reference_path = root / "references" / f"{label}.json"
        reference = json.loads(reference_path.read_text())
        if reference["shape"] != shape:
            raise RuntimeError(f"shape mismatch in {reference_path}")
        reference_protocol = reference["protocol"]
        required_reference_protocol = {
            "factory": "HAO create_nvfp4_attention_tensors",
            "warmup_ms": 50,
            "rep_ms": 500,
            "rounds": 6,
            "provider_order": "balanced six-permutation cycle",
            "timer": "triton.testing.do_bench median",
            "seed": 20260814,
        }
        for field, expected in required_reference_protocol.items():
            if reference_protocol.get(field) != expected:
                raise RuntimeError(
                    f"{reference_path} has {field}="
                    f"{reference_protocol.get(field)!r}, expected {expected!r}"
                )
        reference_samples = reference.get("timing_samples_ms", {})
        if any(len(samples) != 6 for samples in reference_samples.values()):
            raise RuntimeError(f"{reference_path} lacks six timing samples")
        if len(reference.get("timing_round_orders", [])) != 6:
            raise RuntimeError(f"{reference_path} lacks six provider orders")
        bf16_ms = float(reference["timing_ms"]["hao_native_bf16"])
        origin_shard = Path(shape_cases[VARIANT_ORDER[0]]["source_file"]).parts[0]
        origin_manifest = json.loads((root / origin_shard / "manifest.json").read_text())
        reference_audit[label] = {
            "dedicated_reference_file": str(reference_path.relative_to(root)),
            "origin_shard": origin_shard,
            "origin_gpu": origin_manifest["gpu"],
            "dedicated_bf16_ms": bf16_ms,
            "dedicated_protocol": reference["protocol"],
            "dedicated_timing_samples_ms": reference.get("timing_samples_ms"),
            "dedicated_timing_round_orders": reference.get(
                "timing_round_orders"
            ),
            "rejected_in_process_bf16_ms": bf16_samples,
            "rejected_min_ms": min(bf16_samples.values()),
            "rejected_max_ms": max(bf16_samples.values()),
            "rejected_max_over_min": max(bf16_samples.values())
            / min(bf16_samples.values()),
        }

        for variant in PUBLISHED_VARIANTS:
            case = shape_cases[variant]
            benchmark = case["benchmark"]
            tk_key = next(
                name
                for name in benchmark["timing_ms"]
                if name.startswith("tk_")
            )
            rows.append(
                normalized_row(
                    shape=shape,
                    shape_label=label,
                    provider=VARIANT_LABELS[variant],
                    family=variant,
                    time_ms=benchmark["timing_ms"][tk_key],
                    bf16_ms=bf16_ms,
                    error=benchmark["correctness"]["tk_vs_bf16_output"],
                )
            )

        reference_correctness = reference["correctness"]
        hao_nvfp4_error = reference_correctness.get(
            "hao_nvfp4_vs_bf16_output"
        )
        if hao_nvfp4_error is None:
            raise RuntimeError(f"{reference_path} lacks HAO NVFP4 output error")
        hao_fp8_error = reference_correctness.get(
            "hao_fp8_vs_bf16_output",
            reference_correctness.get("hao_vs_bf16_output"),
        )
        if hao_fp8_error is None:
            raise RuntimeError(f"{reference_path} lacks HAO FP8 output error")
        rows.append(
            normalized_row(
                shape=shape,
                shape_label=label,
                provider="HAO NV/NV",
                family="hao-nv-nv",
                time_ms=reference["timing_ms"]["hao_native_nvfp4_nvfp4pv"],
                bf16_ms=bf16_ms,
                error=hao_nvfp4_error,
            )
        )

        rows.append(
            normalized_row(
                shape=shape,
                shape_label=label,
                provider="HAO NV/FP8",
                family="hao-nv-fp8",
                time_ms=reference["timing_ms"]["hao_native_nvfp4_fp8pv"],
                bf16_ms=bf16_ms,
                error=hao_fp8_error,
            )
        )
        rows.append(
            normalized_row(
                shape=shape,
                shape_label=label,
                provider="HAO BF16",
                family="bf16",
                time_ms=bf16_ms,
                bf16_ms=bf16_ms,
                error={"cosine": 1.0, "relative_l2": 0.0, "rmse": 0.0},
            )
        )
    return rows, reference_audit


def fmt(value: float | None, digits: int = 6) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def latex_escape(value: str) -> str:
    return value.replace("/", "/").replace("_", r"\_")


def shape_name(row: dict[str, Any]) -> str:
    return f'B{row["batch"]}/S{row["seqlen"]}/H{row["heads"]}'


def write_latex_table(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    include_shape: bool = False,
) -> None:
    lines = []
    previous_shape = None
    for row in rows:
        fields = []
        if include_shape:
            current_shape = shape_name(row)
            fields.append(current_shape if current_shape != previous_shape else "")
            previous_shape = current_shape
        fields.extend(
            (
                latex_escape(row["provider"]),
                fmt(row["time_ms"]),
                f'{row["speedup_vs_bf16"]:.3f}$\\times$',
                fmt(row["cosine_bf16"]),
                fmt(row["relative_l2_bf16"]),
                fmt(row["rmse_bf16"]),
            )
        )
        lines.append(
            " & ".join(fields) + r" \\"
        )
    path.write_text("\n".join((*lines, r"\bottomrule")) + "\n")


def write_macros(path: Path, headline: list[dict[str, Any]]) -> None:
    by_family = {row["family"]: row for row in headline}
    names = {
        "nvmx-fast": "UnifiedFast",
        "nvmx-accurate": "UnifiedAccurate",
        "fp8": "UnifiedTkFpEight",
        "hao-nv-nv": "UnifiedHaoNvNv",
        "hao-nv-fp8": "UnifiedHaoFpEight",
        "bf16": "UnifiedBfSixteen",
    }
    lines = ["% Generated by build_summary.py; do not edit by hand."]
    for family, prefix in names.items():
        row = by_family[family]
        lines.extend(
            (
                rf"\newcommand{{\{prefix}Time}}{{{row['time_ms']:.6f}}}",
                rf"\newcommand{{\{prefix}Speedup}}{{{row['speedup_vs_bf16']:.3f}}}",
                rf"\newcommand{{\{prefix}Cosine}}{{{row['cosine_bf16']:.6f}}}",
                rf"\newcommand{{\{prefix}RelLTwo}}{{{row['relative_l2_bf16']:.6f}}}",
                rf"\newcommand{{\{prefix}Rmse}}{{{row['rmse_bf16']:.6f}}}",
            )
        )
    path.write_text("\n".join(lines) + "\n")


def write_downstream_tables(
    table_dir: Path,
    rows: list[dict[str, Any]],
    downstream: dict[str, Any],
) -> None:
    by_key = {(row["shape"], row["family"]): row for row in rows}
    provider_family = {
        "nvmx-fast": "nvmx-fast",
        "nvmx-accurate": "nvmx-accurate",
        "hao-nvnv": "hao-nv-nv",
        "tk-nvnv-control": "nv-nv",
    }
    task_shape = {
        "vit-s256": SHORT_LABEL,
        "vit-s1024": "b1_s1024_h24_d128",
        "vit-s4096": HEADLINE_LABEL,
        "bert-mlm-s256": SHORT_LABEL,
        "bert-mlm-s512": "b1_s1024_h24_d128",
        "bert-sst2-s256": SHORT_LABEL,
    }
    task_label = {
        "vit-s256": "ViT S256",
        "vit-s1024": "ViT S1024",
        "vit-s4096": "ViT S4096",
        "bert-mlm-s256": "BERT MLM S256",
        "bert-mlm-s512": "BERT MLM S512",
        "bert-sst2-s256": "BERT SST-2 S256",
    }
    provider_label = {
        "nvmx-fast": r"TK NV/MX \code{fast}",
        "nvmx-accurate": r"TK NV/MX \code{accurate}",
        "hao-nvnv": "HAO NV/NV",
        "tk-nvnv-control": "TK NV/NV fixed schedule",
    }
    provider_order = tuple(provider_family)
    downstream_rows = {
        (row["task"], row["provider"]): row
        for row in downstream["rows"]
    }

    def common_fields(task: str, provider: str) -> tuple[str, ...]:
        value = downstream_rows[(task, provider)]
        family = provider_family[provider]
        kernel = by_key[(task_shape[task], family)]
        metrics = value["task_metrics"]
        baseline_metrics = downstream_rows[
            (task, "nvmx-fast")
        ]["task_metrics"]
        finite = value["status"] == "complete"
        status = "finite" if finite else "non-finite @ sample 1"
        return (
            task_label[task],
            provider_label[provider],
            fmt(kernel["time_ms"]),
            f'{kernel["speedup_vs_bf16"]:.3f}$\\times$',
            (
                f'{100.0 * metrics["provider_accuracy"]:.3f}'
                if finite
                else "--"
            ),
            f'{100.0 * baseline_metrics["baseline_accuracy"]:.3f}',
            (
                f'{100.0 * metrics["top1_agreement"]:.3f}'
                if finite
                else "--"
            ),
            fmt(metrics["logit_cosine"] if finite else None),
            fmt(metrics["logit_relative_l2"] if finite else None),
            status,
        )

    classification_lines = []
    for task in ("vit-s256", "vit-s1024", "vit-s4096", "bert-sst2-s256"):
        for provider in provider_order:
            classification_lines.append(
                " & ".join(common_fields(task, provider)) + r" \\"
            )

    mlm_lines = []
    for task in ("bert-mlm-s256", "bert-mlm-s512"):
        for provider in provider_order:
            value = downstream_rows[(task, provider)]
            common = common_fields(task, provider)
            metrics = value["task_metrics"]
            finite = value["status"] == "complete"
            loss_delta = (
                metrics["provider_loss"] - metrics["baseline_loss"]
                if finite
                else None
            )
            mlm_lines.append(
                " & ".join((*common[:6], fmt(loss_delta), *common[6:]))
                + r" \\"
            )

    combined_lines = []
    main_lines = []
    task_order = (
        "vit-s256",
        "vit-s1024",
        "vit-s4096",
        "bert-sst2-s256",
        "bert-mlm-s256",
        "bert-mlm-s512",
    )
    for task in task_order:
        for provider in provider_order[:3]:
            value = downstream_rows[(task, provider)]
            family = provider_family[provider]
            kernel = by_key[(task_shape[task], family)]
            metrics = value["task_metrics"]
            finite = value["status"] == "complete"
            score = (
                f'{100.0 * metrics["provider_accuracy"]:.2f} / '
                f'{100.0 * metrics["baseline_accuracy"]:.2f}'
                if finite
                else "--"
            )
            agreement = (
                f'{100.0 * metrics["top1_agreement"]:.2f}'
                if finite
                else "--"
            )
            final_error = (
                f'{metrics["logit_cosine"]:.4f} / '
                f'{metrics["logit_relative_l2"]:.4f}'
                if finite
                else "--"
            )
            layer_error = (
                f'{value["mean_layer_cosine"]:.4f} / '
                f'{value["mean_layer_relative_l2"]:.4f}'
                if finite
                else "--"
            )
            loss_delta = "--"
            if finite and "provider_loss" in metrics:
                loss_delta = (
                    f'{metrics["provider_loss"] - metrics["baseline_loss"]:+.3f}'
                )
            combined_lines.append(
                " & ".join(
                    (
                        task_label[task],
                        provider_label[provider],
                        f'{kernel["speedup_vs_bf16"]:.3f}$\\times$',
                        score,
                        agreement,
                        final_error,
                        layer_error,
                        loss_delta,
                    )
                )
                + r" \\"
            )
            main_lines.append(
                " & ".join(
                    (
                        task_label[task],
                        provider_label[provider],
                        f'{kernel["speedup_vs_bf16"]:.3f}$\\times$',
                        score,
                        final_error,
                        loss_delta,
                    )
                )
                + r" \\"
            )

    failure_lines = []
    for task in task_label:
        value = downstream_rows[(task, "tk-nvnv-control")]
        shiftless = value["p_scale_distribution"]["shiftless"]
        stable = value["p_scale_distribution"]["stable"]
        failure_lines.append(
            " & ".join(
                (
                    task_label[task],
                    str(value["completed_samples"]),
                    f'{value["nonfinite_output_rows"]:,}',
                    f'{100.0 * shiftless["fraction_above_e4m3_max"]:.3f}',
                    f'{shiftless["maximum"]:.3e}',
                    f'{100.0 * stable["fraction_above_e4m3_max"]:.3f}',
                    f'{stable["maximum"]:.6f}',
                )
            )
            + r" \\"
        )

    outputs = {
        "downstream_classification_rows.tex": classification_lines,
        "downstream_mlm_rows.tex": mlm_lines,
        "downstream_combined_rows.tex": combined_lines,
        "downstream_main_rows.tex": main_lines,
        "downstream_nvnv_failure_rows.tex": failure_lines,
    }
    for name, lines in outputs.items():
        (table_dir / name).write_text(
            "\n".join((*lines, r"\bottomrule")) + "\n"
        )

    classification_tasks = (
        "vit-s256",
        "vit-s1024",
        "vit-s4096",
        "bert-sst2-s256",
    )
    margin_macros = ["% Generated by build_summary.py; do not edit by hand."]
    total_samples = sum(
        downstream_rows[(task, "nvmx-fast")]["completed_samples"]
        for task in classification_tasks
    )
    margin_macros.append(
        rf"\newcommand{{\DownstreamClassificationSamples}}{{{total_samples}}}"
    )
    for provider, prefix in (
        ("nvmx-fast", "DownstreamFast"),
        ("nvmx-accurate", "DownstreamAccurate"),
        ("hao-nvnv", "DownstreamHao"),
    ):
        analyses = [
            downstream_rows[(task, provider)]["task_metrics"]["margin_analysis"]
            for task in classification_tasks
        ]
        changes = sum(row["prediction_changes"] for row in analyses)
        low_changes = sum(
            row["lowest_margin_quartile_changes"] for row in analyses
        )
        regressions = sum(
            row["baseline_correct_provider_wrong"] for row in analyses
        )
        corrections = sum(
            row["baseline_wrong_provider_correct"] for row in analyses
        )
        margin_macros.extend(
            (
                rf"\newcommand{{\{prefix}PredictionChanges}}{{{changes}}}",
                rf"\newcommand{{\{prefix}LowMarginChanges}}{{{low_changes}}}",
                rf"\newcommand{{\{prefix}Regressions}}{{{regressions}}}",
                rf"\newcommand{{\{prefix}Corrections}}{{{corrections}}}",
            )
        )
    (table_dir / "downstream_margin_macros.tex").write_text(
        "\n".join(margin_macros) + "\n"
    )


def main() -> None:
    args = parse_args()
    cases = load_cases(args.root)
    rows, reference_audit = collect_rows(cases, args.root)
    rows.sort(
        key=lambda row: (
            row["batch"],
            row["seqlen"],
            row["heads"],
            (*VARIANT_ORDER, "hao-nv-nv", "hao-nv-fp8", "bf16").index(
                row["family"]
            ),
        )
    )

    headline = [row for row in rows if row["shape"] == HEADLINE_LABEL]
    cross_shape = [
        row for row in rows if row["family"] in CROSS_SHAPE_VARIANTS
    ]
    downstream_path = (
        args.root.parent
        / "fp4_fa4_downstream_matrix_20260801"
        / "summary.json"
    )
    downstream = json.loads(downstream_path.read_text())
    summary = {
        "schema": "tk_fp4_fa4_unified_v1",
        "protocol": {
            "seed": 20260814,
            "warmup_ms": 300,
            "rep_ms": 3000,
            "timer": "triton.testing.do_bench median",
            "factory": "HAO create_nvfp4_attention_tensors",
            "accuracy_reference": "BF16 output from the same deterministic input",
            "speed_reference": "dedicated same-GPU, balanced-order, direct-harness HAO BF16 reference for each shape",
            "reference_windows": "6 balanced-order windows; 50 ms warmup and 500 ms timing per provider per window",
        },
        "case_count": len(cases),
        "published_tk_case_count": 6 * len(PUBLISHED_VARIANTS),
        "retired_variants": ["nvmx-balanced"],
        "rows": rows,
        "headline": headline,
        "cross_shape": cross_shape,
        "reference_timing_audit": reference_audit,
        "downstream": {
            "source": str(downstream_path.relative_to(args.root.parent)),
            "scope": (
                "fixed model replays with identical BF16 inputs; kernel "
                "timing is joined by provider and adapter shape"
            ),
            "providers": downstream["providers"],
            "tasks": downstream["tasks"],
            "baseline_audit": downstream["baseline_audit"],
            "results": downstream["rows"],
        },
    }
    (args.root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    fields = (
        "shape",
        "provider",
        "family",
        "time_ms",
        "bf16_reference_ms",
        "speedup_vs_bf16",
        "cosine_bf16",
        "relative_l2_bf16",
        "rmse_bf16",
    )
    with (args.root / "unified.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    table_dir = args.root / "tables"
    table_dir.mkdir(exist_ok=True)
    write_latex_table(table_dir / "headline_rows.tex", headline)
    write_latex_table(
        table_dir / "format_rows.tex",
        [
            row
            for row in headline
            if row["family"]
            in ("nv-nv", "mx-nv", "nv-mx", "mx-mx", "hao-nv-nv", "bf16")
        ],
    )
    write_latex_table(
        table_dir / "full_format_rows.tex",
        [
            row
            for row in rows
            if row["family"]
            in ("nv-nv", "mx-nv", "nv-mx", "mx-mx", "hao-nv-nv", "bf16")
        ],
        include_shape=True,
    )
    write_latex_table(
        table_dir / "pareto_rows.tex",
        [
            row
            for row in headline
            if row["family"]
            in (
                "nvmx-fast",
                "nvmx-accurate",
                "fp8",
                "hao-nv-nv",
                "hao-nv-fp8",
                "bf16",
            )
        ],
    )
    write_latex_table(
        table_dir / "cross_shape_rows.tex",
        cross_shape,
        include_shape=True,
    )
    write_macros(table_dir / "unified_macros.tex", headline)
    write_downstream_tables(table_dir, rows, downstream)


if __name__ == "__main__":
    main()
