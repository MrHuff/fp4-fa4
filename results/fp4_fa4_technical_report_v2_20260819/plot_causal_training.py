#!/usr/bin/env python3
"""Render compact causal-training figures from committed report receipts."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPORT_DIR = Path(__file__).resolve().parent
DEFAULT_E2E_RECEIPT = (
    REPORT_DIR.parent
    / "tk_fa4_8b_batch_scaling_20260901"
    / "e2e_batch_scaling_summary.json"
)
DEFAULT_BOUNDARY_RECEIPT = (
    REPORT_DIR / "receipts" / "causal_d128_report_boundaries_20260901.json"
)
DEFAULT_TRAINING_RECEIPT = (
    REPORT_DIR / "receipts" / "llama8b_training_curves_20260901.json"
)
DEFAULT_MATCHED_B4_RECEIPT = (
    REPORT_DIR / "receipts" / "llama8b_b4_matched_snapshot_20260902T1358Z.json"
)
DEFAULT_E2E_OUTPUT = REPORT_DIR / "figures" / "llama8b_e2e_batch_scaling.pdf"
DEFAULT_ISOLATED_OUTPUT = REPORT_DIR / "figures" / "causal_isolated_backward.pdf"
DEFAULT_COMBINED_OUTPUT = (
    REPORT_DIR / "figures" / "causal_combined_forward_backward.pdf"
)
DEFAULT_TRAINING_OUTPUT = REPORT_DIR / "figures" / "llama8b_training_curves.pdf"
DEFAULT_DIVERGENCE_OUTPUT = REPORT_DIR / "figures" / "llama8b_mxfp4_divergence.pdf"
DEFAULT_MATCHED_B4_TRAINING_OUTPUT = (
    REPORT_DIR / "figures" / "llama8b_b4_matched_training_curves.pdf"
)
DEFAULT_MATCHED_B4_THROUGHPUT_OUTPUT = (
    REPORT_DIR / "figures" / "llama8b_b4_matched_throughput.pdf"
)
DEFAULT_MATCHED_B4_MX_FAILURE_OUTPUT = (
    REPORT_DIR / "figures" / "llama8b_b4_mxfp4_failure.pdf"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--e2e-receipt",
        "--receipt",
        dest="e2e_receipt",
        type=Path,
        default=DEFAULT_E2E_RECEIPT,
    )
    parser.add_argument(
        "--boundary-receipt",
        type=Path,
        default=DEFAULT_BOUNDARY_RECEIPT,
    )
    parser.add_argument(
        "--training-receipt",
        type=Path,
        default=DEFAULT_TRAINING_RECEIPT,
    )
    parser.add_argument(
        "--matched-b4-receipt",
        type=Path,
        default=DEFAULT_MATCHED_B4_RECEIPT,
    )
    parser.add_argument(
        "--e2e-output",
        "--output",
        dest="e2e_output",
        type=Path,
        default=DEFAULT_E2E_OUTPUT,
    )
    parser.add_argument(
        "--isolated-output",
        type=Path,
        default=DEFAULT_ISOLATED_OUTPUT,
    )
    parser.add_argument(
        "--combined-output",
        type=Path,
        default=DEFAULT_COMBINED_OUTPUT,
    )
    parser.add_argument(
        "--training-output",
        type=Path,
        default=DEFAULT_TRAINING_OUTPUT,
    )
    parser.add_argument(
        "--divergence-output",
        type=Path,
        default=DEFAULT_DIVERGENCE_OUTPUT,
    )
    parser.add_argument(
        "--matched-b4-training-output",
        type=Path,
        default=DEFAULT_MATCHED_B4_TRAINING_OUTPUT,
    )
    parser.add_argument(
        "--matched-b4-throughput-output",
        type=Path,
        default=DEFAULT_MATCHED_B4_THROUGHPUT_OUTPUT,
    )
    parser.add_argument(
        "--matched-b4-mx-failure-output",
        type=Path,
        default=DEFAULT_MATCHED_B4_MX_FAILURE_OUTPUT,
    )
    return parser.parse_args()


def validate_e2e_receipt(receipt: dict[str, Any]) -> None:
    if receipt.get("schema") != "tkfa4.llama8b.batch_scaling.v2":
        raise ValueError("unsupported receipt schema")
    expected_purpose = "single_gpu_full_step_performance_only_not_convergence"
    if receipt.get("purpose") != expected_purpose:
        raise ValueError("end-to-end claim boundary is missing")
    if receipt["measurement"].get("local_batches") != [1, 2, 4]:
        raise ValueError("expected the exact B1/B2/B4 batch matrix")
    if receipt["measurement"].get("statistic") != "median":
        raise ValueError("unexpected timing statistic")
    if not receipt["low_precision_contract"].get(
        "backward_shared_between_pv_routes_at_each_batch"
    ):
        raise ValueError("FP8/MX backward-sharing contract is missing")

    expected_routes = {
        (batch, pv) for batch in (1, 2, 4) for pv in ("e4m3_fp8", "mxfp4_e8m0_block32")
    }
    results = receipt.get("results", [])
    observed_routes = {(row["batch"], row["pv"]) for row in results}
    if observed_routes != expected_routes or len(results) != len(expected_routes):
        raise ValueError("incomplete or duplicate end-to-end batch matrix")

    sequence = receipt["model"]["sequence"]
    timing_metrics = (
        "step_ms",
        "forward_ms",
        "backward_ms",
        "optimizer_ms",
        "tokens_per_second_per_gpu",
        "mfu_at_2250_tflops",
        "peak_allocated_gib",
    )
    for row in results:
        for arm_name in ("bf16", "lowp"):
            arm = row[arm_name]
            if not all(math.isfinite(arm[metric]) for metric in timing_metrics):
                raise ValueError(
                    f"{row['batch']}/{row['pv']}/{arm_name}: nonfinite result"
                )
            expected_throughput = row["batch"] * sequence * 1000.0 / arm["step_ms"]
            if not math.isclose(
                arm["tokens_per_second_per_gpu"],
                expected_throughput,
                rel_tol=1.0e-12,
                abs_tol=0.0,
            ):
                raise ValueError(
                    f"{row['batch']}/{row['pv']}/{arm_name}: throughput mismatch"
                )
        expected_speedup = row["bf16"]["step_ms"] / row["lowp"]["step_ms"]
        if not math.isclose(row["speedup"], expected_speedup, rel_tol=1.0e-12):
            raise ValueError(f"{row['batch']}/{row['pv']}: speedup mismatch")


def validate_boundary_receipt(receipt: dict[str, Any]) -> None:
    expected_schema = "tkfa4.report.causal_d128_boundaries_receipt.v1"
    if receipt.get("schema") != expected_schema:
        raise ValueError("unsupported boundary receipt schema")
    expected_samples = receipt["configuration"]["samples"]
    for boundary_name, arms in receipt["timings"].items():
        for arm_name, summary in arms.items():
            values = summary["raw_ms"]
            if len(values) != expected_samples:
                raise ValueError(
                    f"{boundary_name}.{arm_name}: expected {expected_samples} "
                    f"samples, found {len(values)}"
                )
            ordered = sorted(values)
            recomputed = {
                "samples": len(values),
                "median_ms": statistics.median(ordered),
                "mean_ms": statistics.fmean(ordered),
                "p10_ms": ordered[int(0.10 * (len(ordered) - 1))],
                "p90_ms": ordered[int(0.90 * (len(ordered) - 1))],
                "minimum_ms": ordered[0],
                "maximum_ms": ordered[-1],
            }
            for statistic, value in recomputed.items():
                committed = summary[statistic]
                if isinstance(value, int):
                    matches = value == committed
                else:
                    matches = math.isclose(
                        value,
                        committed,
                        rel_tol=0.0,
                        abs_tol=1.0e-12,
                    )
                if not matches:
                    raise ValueError(
                        f"{boundary_name}.{arm_name}.{statistic}: "
                        f"{value} != {committed}"
                    )

    medians = {
        boundary: {arm: summary["median_ms"] for arm, summary in arms.items()}
        for boundary, arms in receipt["timings"].items()
    }
    comparisons = receipt["comparisons"]
    expected_comparisons = {
        "isolated_replay_core_speedup_vs_bf16": (
            medians["isolated_backward"]["bf16_cute_fa4"]
            / medians["isolated_backward"]["replay_core"]
        ),
        "isolated_publisher_plus_replay_speedup_vs_bf16": (
            medians["isolated_backward"]["bf16_cute_fa4"]
            / medians["isolated_backward"]["publisher_plus_replay"]
        ),
        "module_backward_speedup_vs_bf16": (
            medians["module_backward"]["bf16_fa4"]
            / medians["module_backward"]["forward_payload_replay"]
        ),
        "module_forward_backward_speedup_vs_bf16": (
            medians["module_forward_backward"]["bf16_fa4"]
            / medians["module_forward_backward"]["forward_payload_replay"]
        ),
    }
    for name, value in expected_comparisons.items():
        if not math.isclose(
            value,
            comparisons[name],
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(f"{name}: {value} != {comparisons[name]}")


def validate_training_receipt(receipt: dict[str, Any]) -> None:
    expected = "tkfa4.report.llama8b_training_curves.v1"
    if receipt.get("schema") != expected:
        raise ValueError("unsupported training-curve receipt schema")
    required_arms = {"e4_fp8", "e4_mx", "nv_fp8", "nv_mx"}
    if set(receipt["current_arms"]) != required_arms:
        raise ValueError("training receipt does not contain the four-arm control")
    smoothing = receipt["capture"]["smoothing"]
    if smoothing.get("half_life_tokens") != 1_000_000_000.0:
        raise ValueError("unexpected training-curve smoothing half-life")
    if not smoothing.get("current_arms_use_every_unique_token_coordinate"):
        raise ValueError("training-curve coordinate policy is missing")
    for arm_name, arm in receipt["current_arms"].items():
        series = arm["series"]
        if len(series) != arm["display_rows"] or not series:
            raise ValueError(f"{arm_name}: invalid display series length")
        tokens = [row["tokens"] for row in series]
        if any(left >= right for left, right in zip(tokens, tokens[1:])):
            raise ValueError(f"{arm_name}: token coordinates are not increasing")
        if any(token <= 0 or token % (64 * 4096) for token in tokens):
            raise ValueError(f"{arm_name}: invalid token coordinate")
        if series[-1]["tokens"] != arm["last_tokens"]:
            raise ValueError(f"{arm_name}: summary/series endpoint mismatch")
        if series[-1]["update"] != arm["last_update"]:
            raise ValueError(f"{arm_name}: update endpoint mismatch")
        for row in series:
            for field in (
                "loss",
                "smoothed_loss",
                "grad_norm",
                "grad_norm_bin_max",
            ):
                if not math.isfinite(row[field]):
                    raise ValueError(f"{arm_name}: nonfinite {field}")
    baseline = receipt["historical_bf16"]
    if baseline["comparison_class"] != "historical_convergence_sanity_reference":
        raise ValueError("BF16 curve claim boundary is missing")
    baseline_tokens = [row["tokens"] for row in baseline["series"]]
    if len(baseline_tokens) != baseline["display_rows"] or not baseline_tokens:
        raise ValueError("historical BF16 display series length is invalid")
    if any(left >= right for left, right in zip(baseline_tokens, baseline_tokens[1:])):
        raise ValueError("historical BF16 token coordinates are not increasing")
    stable_horizon = min(
        receipt["current_arms"][name]["last_tokens"] for name in ("e4_fp8", "nv_fp8")
    )
    if stable_horizon != receipt["plot_boundaries"]["common_stable_tokens"]:
        raise ValueError("stable plot horizon does not match arm intersection")
    four_arm_horizon = min(
        arm["last_tokens"] for arm in receipt["current_arms"].values()
    )
    if four_arm_horizon != receipt["plot_boundaries"]["common_four_arm_tokens"]:
        raise ValueError("four-arm plot horizon does not match arm intersection")


def _linear_percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    coordinate = (len(ordered) - 1) * probability
    lower = math.floor(coordinate)
    upper = math.ceil(coordinate)
    fraction = coordinate - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _validate_summary(values: list[float], summary: dict[str, Any], name: str) -> None:
    recomputed = {
        "samples": len(values),
        "minimum": min(values),
        "p10": _linear_percentile(values, 0.10),
        "median": statistics.median(values),
        "p90": _linear_percentile(values, 0.90),
        "maximum": max(values),
    }
    for statistic, expected in recomputed.items():
        observed = summary[statistic]
        if isinstance(expected, int):
            matches = expected == observed
        else:
            matches = math.isclose(expected, observed, rel_tol=0.0, abs_tol=1.0e-12)
        if not matches:
            raise ValueError(
                f"{name}.{statistic}: recomputed {expected} != receipt {observed}"
            )


def _validate_exact_fields(
    observed: dict[str, Any], expected: dict[str, Any], context: str
) -> None:
    for field, value in expected.items():
        if observed.get(field) != value:
            raise ValueError(
                f"{context}.{field}: expected {value!r}, found {observed.get(field)!r}"
            )


def validate_matched_b4_receipt(receipt: dict[str, Any]) -> None:
    expected = "tkfa4.report.llama8b_b4_matched_snapshot.v1"
    if receipt.get("schema") != expected:
        raise ValueError("unsupported matched-B4 receipt schema")
    if not receipt["capture"].get("credential_free"):
        raise ValueError("matched-B4 receipt is not marked credential-free")
    if not receipt["capture"].get("full_healthy_histories"):
        raise ValueError("matched-B4 receipt does not contain full healthy histories")
    _validate_exact_fields(
        receipt.get("identity", {}),
        {
            "gc_training_commit": "e7db209b0c7017c415fdd66e04e85f96ae24f276",
            "gc_training_branch": "codex/fa4-d128-numerics-no-cce-20260830",
            "source_archive_sha256": "20af94e0899a56fcd1eb6b8dae9a75217012631d72554f75612b35bcb84b4181",
            "runtime_bundle_sha256": "e0cef3469d9203169e0497152fa83dc9d6f12a5c1c1bbcbb9c0fd43edc58281c",
            "fp4_matmul_runtime_commit": "4590537f1479e1a7e847f2783e9ab7aa7f11b975",
            "native_backward": (
                "v509 B4/S4096/D128 native NVFP4-score replay with E4M3 QKV "
                "and E5M2 dO"
            ),
        },
        "matched-B4 identity",
    )
    _validate_exact_fields(
        receipt["capture"],
        {
            "launch_receipt_basename": "llama8b_b4_w64_launch_check_20260902.json",
            "launch_receipt_sha256": "f652ea07c34048e9180629737dc000933e481e88856e7c64ee87f148eea21063",
        },
        "matched-B4 capture",
    )
    recipe = receipt["shared_recipe"]
    if recipe.get("local_batch") != 4 or recipe.get("world_size") != 64:
        raise ValueError("matched-B4 receipt does not use B4/W64")
    tokens_per_update = recipe.get("tokens_per_update")
    if tokens_per_update != 4_194_304:
        raise ValueError("matched-B4 token coordinate is unexpected")

    arms = receipt["arms"]
    if set(arms) != {"bf16", "fp8", "mx"}:
        raise ValueError("matched-B4 receipt has an unexpected arm set")
    expected_arms = {
        "bf16": {
            "public_arm_label": "bf16_b4_control",
            "route": "BF16 FA4 baseline",
            "status_at_capture": "cancelled_for_normal_priority_resume",
            "wandb_state_at_capture": "crashed",
        },
        "fp8": {
            "public_arm_label": "nvfp4_projection_fp8_pv_b4",
            "route": (
                "NVFP4 learned QKV/O projections + NVFP4 attention QK + "
                "E4M3 FP8 attention PV + E5M2-dO v509 backward"
            ),
            "status_at_capture": "cancelled_for_normal_priority_resume",
            "wandb_state_at_capture": "finished",
        },
        "mx": {
            "public_arm_label": "e4m3_projection_mxfp4_pv_b4",
            "route": (
                "E4M3 learned QKV/O projections + NVFP4 attention QK + "
                "MXFP4/E8M0-block32 attention PV + E5M2-dO v509 backward"
            ),
            "status_at_capture": "cancelled_after_confirmed_divergence",
            "wandb_state_at_capture": "crashed",
        },
    }
    expected_sources = {
        "bf16": (
            "8b3c5ef94cd57a3718e15df3fb7b32ba66509fe7e3d39c53a46433d3705980dc",
            "07dda1dcd8a01a32a58a56642a82a22e6bd026c2a1901706da440016dc407e56",
        ),
        "fp8": (
            "7f5a781265b04418bf61dca99041e21b1cd13a46fba18681df5faef95a2264fc",
            "534ed6a5e5453181d59d0dba0072eed40873ddb1ac720f9526c7f807628c9133",
        ),
        "mx": (
            "a19a237c747c148196dc83c263886ead7910fda08e62565d016a740cec233de5",
            "50649cc015c14d72cabdd401643addbc9aa7bdf3b3d2288622b3d1be531e68f3",
        ),
    }
    capture_sources = receipt["capture"].get("sources", {})
    if set(capture_sources) != set(expected_arms):
        raise ValueError("matched-B4 receipt has an unexpected capture-source set")
    for arm_name, arm in arms.items():
        _validate_exact_fields(arm, expected_arms[arm_name], f"{arm_name} identity")
        source = capture_sources[arm_name]
        if source.get("public_arm_label") != expected_arms[arm_name][
            "public_arm_label"
        ]:
            raise ValueError(f"{arm_name}: capture-source arm label disagrees")
        history_sha, worker_sha = expected_sources[arm_name]
        if source["wandb_history_export"].get("sha256") != history_sha:
            raise ValueError(f"{arm_name}: W&B history source hash disagrees")
        if source["worker_log_crosscheck"].get("sha256") != worker_sha:
            raise ValueError(f"{arm_name}: worker-log source hash disagrees")
        history = arm["wandb_history_deduplication"]
        if history["train"].get("first_update") != 1:
            raise ValueError(f"{arm_name}: W&B training history does not start at 1")
        if history["validation"].get("first_update") != 1:
            raise ValueError(f"{arm_name}: W&B validation history does not start at 1")
        deduplication = arm["worker_log_four_rank_deduplication"]
        if deduplication.get("expected_local_rank_copies_per_update") != 4:
            raise ValueError(f"{arm_name}: local-rank multiplicity is not four")
        if deduplication.get("observed_train_multiplicities") != [4]:
            raise ValueError(f"{arm_name}: training rows were not exactly 4-way")
        if deduplication.get("observed_validation_multiplicities") != [4]:
            raise ValueError(f"{arm_name}: validation rows were not exactly 4-way")

    healthy = receipt["healthy_matched_comparison"]
    train = healthy["training"]
    train_series = train["series"]
    if len(train_series) != train["common_rows"] or not train_series:
        raise ValueError("matched-B4 training series length is invalid")
    updates = [row["update"] for row in train_series]
    if any(left >= right for left, right in zip(updates, updates[1:])):
        raise ValueError("matched-B4 training updates are not increasing")
    if (
        updates[0] != train["first_common_update"]
        or updates[-1] != train["last_common_update"]
    ):
        raise ValueError("matched-B4 training endpoints disagree")
    if train.get("endpoint") != train_series[-1]:
        raise ValueError("matched-B4 training endpoint row disagrees")
    for row in train_series:
        if row["tokens"] != row["update"] * tokens_per_update:
            raise ValueError("matched-B4 training token coordinate is invalid")
        for arm_key in ("bf16", "nvfp4_projection_fp8_pv"):
            arm_row = row[arm_key]
            if (arm_row["update"], arm_row["tokens"]) != (
                row["update"],
                row["tokens"],
            ):
                raise ValueError(f"matched-B4 {arm_key} coordinate disagrees")
            for field in (
                "loss",
                "preclip_grad_norm",
                "tokens_per_second_per_gpu",
                "mfu_percent",
                "smoothed_loss",
            ):
                if not math.isfinite(arm_row[field]):
                    raise ValueError(f"matched-B4 {arm_key}.{field} is nonfinite")
        expected_loss_difference = (
            row["nvfp4_projection_fp8_pv"]["loss"] - row["bf16"]["loss"]
        )
        if not math.isclose(
            row["loss_difference_fp8_minus_bf16"],
            expected_loss_difference,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("matched-B4 training loss difference is inconsistent")
        expected_ratio = (
            row["nvfp4_projection_fp8_pv"]["tokens_per_second_per_gpu"]
            / row["bf16"]["tokens_per_second_per_gpu"]
        )
        if not math.isclose(
            row["throughput_ratio_fp8_over_bf16"],
            expected_ratio,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("matched-B4 paired throughput ratio is inconsistent")

    validation = healthy["validation"]
    validation_series = validation["series"]
    if len(validation_series) != validation["common_rows"] or not validation_series:
        raise ValueError("matched-B4 validation series length is invalid")
    validation_updates = [row["update"] for row in validation_series]
    if any(
        left >= right for left, right in zip(validation_updates, validation_updates[1:])
    ):
        raise ValueError("matched-B4 validation updates are not increasing")
    if (
        validation_updates[0] != validation["first_common_update"]
        or validation_updates[-1] != validation["last_common_update"]
    ):
        raise ValueError("matched-B4 validation endpoints disagree")
    if validation.get("endpoint") != validation_series[-1]:
        raise ValueError("matched-B4 validation endpoint row disagrees")
    for row in validation_series:
        if row["tokens"] != row["update"] * tokens_per_update:
            raise ValueError("matched-B4 validation token coordinate is invalid")
        for field in ("bf16_loss", "fp8_loss", "loss_difference_fp8_minus_bf16"):
            if not math.isfinite(row[field]):
                raise ValueError(f"matched-B4 validation {field} is nonfinite")
        if not math.isclose(
            row["loss_difference_fp8_minus_bf16"],
            row["fp8_loss"] - row["bf16_loss"],
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("matched-B4 validation loss difference is inconsistent")

    throughput = healthy["throughput"]
    throughput_series = [
        row
        for row in train_series
        if throughput["first_update"] <= row["update"] <= throughput["last_update"]
    ]
    if len(throughput_series) != throughput["common_rows"]:
        raise ValueError("matched-B4 throughput window length is inconsistent")
    bf16_tps = [row["bf16"]["tokens_per_second_per_gpu"] for row in throughput_series]
    fp8_tps = [
        row["nvfp4_projection_fp8_pv"]["tokens_per_second_per_gpu"]
        for row in throughput_series
    ]
    ratios = [fp8 / bf16 for bf16, fp8 in zip(bf16_tps, fp8_tps)]
    _validate_summary(
        bf16_tps,
        throughput["bf16_tokens_per_second_per_gpu"],
        "bf16 throughput",
    )
    _validate_summary(
        fp8_tps,
        throughput["fp8_tokens_per_second_per_gpu"],
        "FP8 throughput",
    )
    _validate_summary(ratios, throughput["paired_throughput_ratio"], "paired ratio")
    ratio_of_medians = statistics.median(fp8_tps) / statistics.median(bf16_tps)
    if not math.isclose(
        ratio_of_medians,
        throughput["ratio_of_median_throughputs"],
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("matched-B4 ratio of median throughputs is inconsistent")

    mx = receipt["mxfp4_divergence"]
    if "excluded" not in mx.get("comparison_class", ""):
        raise ValueError("MXFP4 divergence is not excluded from throughput claims")
    if mx.get("observed_departure_update") != 325:
        raise ValueError("unexpected MXFP4 departure marker")
    if mx.get("observed_departure_tokens") != 325 * tokens_per_update:
        raise ValueError("unexpected MXFP4 departure token coordinate")
    mx_train = mx.get("full_training_series", [])
    if not mx_train or mx_train[-1].get("update") != 2550:
        raise ValueError("current MXFP4 diagnostic does not reach update 2550")
    if any(
        left["update"] >= right["update"] for left, right in zip(mx_train, mx_train[1:])
    ):
        raise ValueError("current MXFP4 training updates are not increasing")
    mx_validation = mx.get("full_validation_series", [])
    if not mx_validation or any(
        left["update"] >= right["update"]
        for left, right in zip(mx_validation, mx_validation[1:])
    ):
        raise ValueError("current MXFP4 validation series is invalid")
    for kind, series in (("training", mx_train), ("validation", mx_validation)):
        for row in series:
            if row["tokens"] != row["update"] * tokens_per_update:
                raise ValueError(f"current MXFP4 {kind} token coordinate is invalid")
            if not math.isfinite(row["loss"]):
                raise ValueError(f"current MXFP4 {kind} loss is nonfinite")
            if kind == "training" and not math.isfinite(row["preclip_grad_norm"]):
                raise ValueError("current MXFP4 gradient norm is nonfinite")
    if mx.get("last_training_row") != mx_train[-1]:
        raise ValueError("current MXFP4 last training row disagrees")
    if mx.get("last_validation_row") != mx_validation[-1]:
        raise ValueError("current MXFP4 last validation row disagrees")
    maximum = max(mx_train, key=lambda row: row["preclip_grad_norm"])
    recomputed_maximum = {
        "update": maximum["update"],
        "tokens": maximum["tokens"],
        "value": maximum["preclip_grad_norm"],
        "loss": maximum["loss"],
    }
    if mx.get("maximum_observed_preclip_grad_norm") != recomputed_maximum:
        raise ValueError("current MXFP4 maximum-gradient record disagrees")
    selected_updates = (300, 325, 350, 400)
    mx_by_update = {row["update"]: row for row in mx_train}
    if mx.get("selected_training_rows") != [
        mx_by_update[update] for update in selected_updates
    ]:
        raise ValueError("current MXFP4 selected diagnostic rows disagree")


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9.0,
            "axes.titlesize": 10.0,
            "legend.fontsize": 8.0,
            "figure.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "pdf.fonttype": 42,
        }
    )


def render_e2e_batch_scaling(receipt: dict[str, Any], output: Path) -> None:
    route_specs = (
        ("e4m3_fp8", "NVFP4 + FP8 P/V", "#625a9c"),
        ("mxfp4_e8m0_block32", "NVFP4 + MXFP4 P/V", "#e57a44"),
    )
    batches = receipt["measurement"]["local_batches"]
    x_positions = list(range(len(batches)))
    width = 0.34
    indexed = {(row["batch"], row["pv"]): row for row in receipt["results"]}

    fig, ax = plt.subplots(figsize=(5.35, 3.45), constrained_layout=True)
    for route_index, (route, label, color) in enumerate(route_specs):
        speedups = [indexed[(batch, route)]["speedup"] for batch in batches]
        positions = [x + (route_index - 0.5) * width for x in x_positions]
        bars = ax.bar(
            positions,
            [speedup - 1.0 for speedup in speedups],
            width=width,
            bottom=1.0,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            label=label,
            zorder=3,
        )
        for bar, speedup in zip(bars, speedups):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                speedup + 0.004,
                f"{speedup:.3f}×",
                ha="center",
                va="bottom",
                fontsize=7.8,
                fontweight="medium",
                color="#303030",
            )

    ax.axhline(1.0, color="#6f6f6f", linewidth=0.8, zorder=2)
    ax.set_xticks(x_positions, [f"B{batch}" for batch in batches])
    ax.set_xlabel("Local batch size")
    ax.set_ylabel("Complete-update speedup over paired BF16")
    ax.set_ylim(1.0, 1.17)
    ax.set_xlim(-0.55, len(batches) - 0.45)
    ax.yaxis.grid(True, color="#d7d7d7", linewidth=0.55, alpha=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(
        "Llama 3.1 8B speedup grows with batch saturation",
        loc="left",
        pad=9,
        fontweight="bold",
    )
    ax.text(
        0.0,
        1.01,
        "S4096 · one GB200 · 10 warmups + 21 timed steps · median",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.4,
        color="#555555",
    )
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.31),
        ncol=2,
        frameon=False,
        handlelength=1.4,
        columnspacing=1.4,
    )
    ax.text(
        0.5,
        -0.19,
        "Full low-precision route includes NVFP4 QKV/output projections; performance only.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.1,
        color="#555555",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        bbox_inches="tight",
        metadata={
            "Title": "Llama 3.1 8B end-to-end batch scaling",
            "Creator": "plot_causal_training.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)


def _error_bars(
    summaries: list[dict[str, Any]],
) -> tuple[list[float], list[float]]:
    medians = [summary["median_ms"] for summary in summaries]
    return (
        [median - summary["p10_ms"] for median, summary in zip(medians, summaries)],
        [summary["p90_ms"] - median for median, summary in zip(medians, summaries)],
    )


def render_isolated_backward(receipt: dict[str, Any], output: Path) -> None:
    arms = (
        ("bf16_cute_fa4", "BF16 FA4", "#6f6f6f"),
        ("replay_core", "Saved-Q/K core", "#625a9c"),
        ("publisher_plus_replay", "Core + publisher", "#e57a44"),
    )
    summaries = [receipt["timings"]["isolated_backward"][name] for name, _, _ in arms]
    medians = [summary["median_ms"] for summary in summaries]
    x_positions = list(range(len(arms)))

    fig, ax = plt.subplots(figsize=(5.35, 3.35), constrained_layout=True)
    bars = ax.bar(
        x_positions,
        medians,
        width=0.62,
        color=[color for _, _, color in arms],
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    ax.errorbar(
        x_positions,
        medians,
        yerr=_error_bars(summaries),
        fmt="none",
        ecolor="#252525",
        elinewidth=0.8,
        capsize=2.5,
        capthick=0.8,
        zorder=5,
    )
    baseline = medians[0]
    for index, (bar, median) in enumerate(zip(bars, medians)):
        label = f"{median * 1000:.1f} µs"
        if index:
            label += f"\n{baseline / median:.3f}×"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            median + 0.025,
            label,
            ha="center",
            va="bottom",
            fontsize=8.1,
            fontweight="medium",
        )

    ax.set_xticks(x_positions, [label for _, label, _ in arms])
    ax.set_ylabel("Median backward time (ms)")
    ax.set_ylim(0.0, max(medians) * 1.30)
    ax.yaxis.grid(True, color="#d7d7d7", linewidth=0.55, alpha=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(
        "Causal D128 isolated backward",
        loc="left",
        pad=9,
        fontweight="bold",
    )
    ax.text(
        0.0,
        1.01,
        "B1 · S4096 · GQA 32:8 · GB200 · 101 timed samples",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.4,
        color="#555555",
    )
    ax.text(
        0.5,
        -0.20,
        (
            "The saved Q/K payload supplies score reconstruction; the final "
            "bar also publishes E5M2 dO and statistics."
        ),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.2,
        color="#555555",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        bbox_inches="tight",
        metadata={
            "Title": "Causal D128 isolated backward timing",
            "Creator": "plot_causal_training.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)


def render_combined_forward_backward(receipt: dict[str, Any], output: Path) -> None:
    boundaries = (
        ("module_backward", "Backward only"),
        ("module_forward_backward", "Forward + backward"),
    )
    arms = (
        ("bf16_fa4", "BF16 FA4", "#6f6f6f"),
        ("forward_payload_replay", "Saved-Q/K path", "#625a9c"),
    )
    group_centers = (0.0, 1.0)
    width = 0.32

    fig, ax = plt.subplots(figsize=(5.35, 3.45), constrained_layout=True)
    for arm_index, (arm_name, arm_label, color) in enumerate(arms):
        summaries = [
            receipt["timings"][boundary_name][arm_name]
            for boundary_name, _ in boundaries
        ]
        medians = [summary["median_ms"] for summary in summaries]
        positions = [center + (arm_index - 0.5) * width for center in group_centers]
        bars = ax.bar(
            positions,
            medians,
            width=width,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            label=arm_label,
            zorder=3,
        )
        ax.errorbar(
            positions,
            medians,
            yerr=_error_bars(summaries),
            fmt="none",
            ecolor="#252525",
            elinewidth=0.8,
            capsize=2.5,
            capthick=0.8,
            zorder=5,
        )
        for bar, median in zip(bars, medians):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                median - 0.09,
                f"{median:.3f} ms",
                ha="center",
                va="top",
                fontsize=7.6,
                fontweight="medium",
                color="white",
            )

    comparisons = receipt["comparisons"]
    speedups = (
        comparisons["module_backward_speedup_vs_bf16"],
        comparisons["module_forward_backward_speedup_vs_bf16"],
    )
    for center, (boundary_name, _), speedup in zip(group_centers, boundaries, speedups):
        group_top = max(
            receipt["timings"][boundary_name][arm_name]["median_ms"]
            for arm_name, _, _ in arms
        )
        ax.text(
            center,
            group_top + 0.14,
            f"{speedup:.3f}×",
            ha="center",
            va="bottom",
            color="#1e6f50",
            fontsize=8.8,
            fontweight="bold",
        )

    ax.set_xticks(group_centers, [label for _, label in boundaries])
    ax.set_ylabel("Median projection-inclusive time (ms)")
    ax.set_ylim(0.0, 3.05)
    ax.set_xlim(-0.52, 1.52)
    ax.yaxis.grid(True, color="#d7d7d7", linewidth=0.55, alpha=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(
        "Causal D128 attention sublayer",
        loc="left",
        pad=9,
        fontweight="bold",
    )
    ax.text(
        0.0,
        1.01,
        "B1 · S4096 · projections, RoPE, attention, and publications included",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.4,
        color="#555555",
    )
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=2,
        frameon=False,
        handlelength=1.2,
        columnspacing=1.7,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        bbox_inches="tight",
        metadata={
            "Title": "Causal D128 projection-inclusive attention timing",
            "Creator": "plot_causal_training.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)


def _clipped_series(series: list[dict[str, Any]], horizon: int) -> list[dict[str, Any]]:
    clipped = [row for row in series if row["tokens"] <= horizon]
    if not clipped:
        raise ValueError("plot horizon excludes every history row")
    return clipped


def _plot_loss_curve(
    ax: Any,
    series: list[dict[str, Any]],
    *,
    color: str,
    label: str,
    linestyle: str = "-",
    raw_alpha: float = 0.12,
    zorder: int = 3,
) -> None:
    tokens = [row["tokens"] / 1.0e9 for row in series]
    ax.plot(
        tokens,
        [row["loss"] for row in series],
        color=color,
        linewidth=0.42,
        alpha=raw_alpha,
        zorder=zorder - 1,
    )
    ax.plot(
        tokens,
        [row["smoothed_loss"] for row in series],
        color=color,
        linewidth=1.55,
        linestyle=linestyle,
        label=label,
        zorder=zorder,
    )


def render_training_curves(receipt: dict[str, Any], output: Path) -> None:
    horizon = receipt["plot_boundaries"]["common_stable_tokens"]
    curves = (
        (
            receipt["historical_bf16"]["series"],
            "Historical BF16 FA4",
            "#6f6f6f",
            "-",
        ),
        (
            receipt["current_arms"]["e4_fp8"]["series"],
            "E4M3 projections + FP8 P/V (retained)",
            "#625a9c",
            "-",
        ),
        (
            receipt["current_arms"]["nv_fp8"]["series"],
            "NVFP4 projections + FP8 P/V",
            "#3c78a8",
            "--",
        ),
    )

    fig, ax = plt.subplots(figsize=(5.35, 3.75), constrained_layout=True)
    clipped_curves: list[tuple[list[dict[str, Any]], str, str, str]] = []
    for series, label, color, linestyle in curves:
        clipped = _clipped_series(series, horizon)
        clipped_curves.append((clipped, label, color, linestyle))
        _plot_loss_curve(
            ax,
            clipped,
            color=color,
            label=label,
            linestyle=linestyle,
        )

    horizon_billions = horizon / 1.0e9
    ax.set_xlim(0.0, horizon_billions)
    ax.set_ylim(2.25, 12.45)
    ax.set_xlabel("Processed tokens (billions)")
    ax.set_ylabel("Training loss")
    ax.yaxis.grid(True, color="#d7d7d7", linewidth=0.55, alpha=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(
        "Working FP8 P/V training routes", loc="left", pad=9, fontweight="bold"
    )
    ax.text(
        0.0,
        1.01,
        (
            f"Llama 3.1 8B · S4096 · common observed horizon "
            f"{horizon_billions:.1f}B tokens · 1B-token EWM"
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.4,
        color="#555555",
    )
    ax.legend(loc="upper right", frameon=False, handlelength=2.2)

    inset = ax.inset_axes([0.49, 0.39, 0.48, 0.34])
    inset_start = max(0.0, horizon_billions - 10.0)
    inset_values: list[float] = []
    for series, _, color, linestyle in clipped_curves:
        rows = [row for row in series if row["tokens"] / 1.0e9 >= inset_start]
        x = [row["tokens"] / 1.0e9 for row in rows]
        y = [row["smoothed_loss"] for row in rows]
        inset_values.extend(y)
        inset.plot(x, y, color=color, linestyle=linestyle, linewidth=1.15)
    if inset_values:
        lower = min(inset_values) - 0.04
        upper = max(inset_values) + 0.04
        inset.set_ylim(lower, upper)
    inset.set_xlim(inset_start, horizon_billions)
    inset.set_title("Final 10B tokens", fontsize=7.2, loc="left", pad=2)
    inset.tick_params(axis="both", labelsize=6.3, length=2)
    inset.yaxis.grid(True, color="#dedede", linewidth=0.4, alpha=0.7)
    for spine in inset.spines.values():
        spine.set_linewidth(0.55)
        spine.set_color("#777777")

    ax.text(
        0.5,
        -0.20,
        (
            "Thin lines are downsampled raw loss; thick lines are causal token-domain EWMs.\n"
            "The BF16 run is a historical, non-paired reference."
        ),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.1,
        color="#555555",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        bbox_inches="tight",
        metadata={
            "Title": "Llama 3.1 8B working FP8 P/V training curves",
            "Creator": "plot_causal_training.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)


def render_mxfp4_divergence(receipt: dict[str, Any], output: Path) -> None:
    horizon = receipt["plot_boundaries"]["common_four_arm_tokens"]
    curves = (
        ("e4_fp8", "E4M3 + FP8 P/V", "#9b96b8", "-", 0.68, 2),
        ("nv_fp8", "NVFP4 + FP8 P/V", "#86a5b8", "--", 0.68, 2),
        ("e4_mx", "E4M3 + MXFP4 P/V", "#c84c3a", "-", 1.0, 5),
        ("nv_mx", "NVFP4 + MXFP4 P/V", "#e0873d", "--", 1.0, 5),
    )
    fig, (loss_ax, grad_ax) = plt.subplots(
        2,
        1,
        figsize=(5.35, 5.05),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": (1.05, 0.95)},
    )
    plotted: dict[str, list[dict[str, Any]]] = {}
    for key, label, color, linestyle, alpha, zorder in curves:
        series = _clipped_series(receipt["current_arms"][key]["series"], horizon)
        plotted[key] = series
        _plot_loss_curve(
            loss_ax,
            series,
            color=color,
            label=label,
            linestyle=linestyle,
            raw_alpha=0.08 * alpha,
            zorder=zorder,
        )
        grad_ax.plot(
            [row["tokens"] / 1.0e9 for row in series],
            [max(row["grad_norm_bin_max"], 1.0e-12) for row in series],
            color=color,
            linestyle=linestyle,
            linewidth=1.05 if "mx" in key else 0.8,
            alpha=alpha,
            zorder=zorder,
        )

    horizon_billions = horizon / 1.0e9
    loss_ax.set_ylim(2.25, 13.5)
    loss_ax.set_ylabel("Training loss")
    loss_ax.set_title(
        "MXFP4 P/V divergence diagnostic",
        loc="left",
        pad=9,
        fontweight="bold",
    )
    loss_ax.text(
        0.0,
        1.01,
        (
            f"Four-arm projection × P/V control · common horizon "
            f"{horizon_billions:.1f}B tokens"
        ),
        transform=loss_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.4,
        color="#555555",
    )
    loss_ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.19),
        ncol=2,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.1,
    )
    loss_ax.yaxis.grid(True, color="#d7d7d7", linewidth=0.55, alpha=0.8)

    early = loss_ax.inset_axes([0.50, 0.08, 0.47, 0.34])
    early_limit = 0.55
    for key, _, color, linestyle, alpha, zorder in curves:
        rows = [row for row in plotted[key] if row["tokens"] / 1.0e9 <= early_limit]
        early.plot(
            [row["tokens"] / 1.0e9 for row in rows],
            [row["loss"] for row in rows],
            color=color,
            linestyle=linestyle,
            linewidth=1.0,
            alpha=alpha,
            zorder=zorder,
        )
    early.axvspan(0.08, 0.14, color="#c84c3a", alpha=0.10, linewidth=0)
    early.set_xlim(0.0, early_limit)
    early.set_ylim(4.5, 9.2)
    early.set_title("Early split", fontsize=7.2, loc="left", pad=2)
    early.tick_params(axis="both", labelsize=6.3, length=2)
    early.yaxis.grid(True, color="#dedede", linewidth=0.4, alpha=0.7)
    for spine in early.spines.values():
        spine.set_linewidth(0.55)
        spine.set_color("#777777")

    grad_ax.set_yscale("log")
    grad_ax.set_xlim(0.0, horizon_billions)
    grad_ax.set_ylim(1.0e-1, 1.0e8)
    grad_ax.set_xlabel("Processed tokens (billions)")
    grad_ax.set_ylabel("Pre-clipping gradient norm\n(maximum per 100 updates)")
    grad_ax.yaxis.grid(True, which="major", color="#d7d7d7", linewidth=0.55, alpha=0.8)
    grad_ax.text(
        0.5,
        -0.27,
        (
            "Both MXFP4 arms fail under both tested learned-projection formats;\n"
            "the FP8 control arms remain non-divergent."
        ),
        transform=grad_ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.1,
        color="#555555",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        bbox_inches="tight",
        metadata={
            "Title": "Llama 3.1 8B MXFP4 P/V divergence diagnostic",
            "Creator": "plot_causal_training.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)


def render_matched_b4_training(receipt: dict[str, Any], output: Path) -> None:
    healthy = receipt["healthy_matched_comparison"]
    train = healthy["training"]
    validation = healthy["validation"]
    train_series = train["series"]
    validation_series = validation["series"]
    colors = {"bf16": "#6f6f6f", "fp8": "#3c78a8"}
    labels = {
        "bf16": "BF16 FA4",
        "fp8": "NVFP4 projections + FP8 P/V",
    }

    fig, (train_ax, validation_ax) = plt.subplots(
        2,
        1,
        figsize=(5.35, 4.75),
        constrained_layout=True,
        gridspec_kw={"height_ratios": (1.25, 0.85)},
    )
    tokens = [row["tokens"] / 1.0e9 for row in train_series]
    arm_fields = (
        ("bf16", "bf16"),
        ("fp8", "nvfp4_projection_fp8_pv"),
    )
    for short_name, field in arm_fields:
        raw = [row[field]["loss"] for row in train_series]
        smooth = [row[field]["smoothed_loss"] for row in train_series]
        train_ax.plot(
            tokens,
            raw,
            color=colors[short_name],
            linewidth=0.55,
            alpha=0.22,
            zorder=2,
        )
        train_ax.plot(
            tokens,
            smooth,
            color=colors[short_name],
            linewidth=1.65,
            label=labels[short_name],
            zorder=3,
        )

    train_values = [
        row[field][metric]
        for row in train_series
        for _, field in arm_fields
        for metric in ("loss", "smoothed_loss")
    ]
    train_margin = max(0.035, (max(train_values) - min(train_values)) * 0.10)
    train_ax.set_ylim(
        min(train_values) - train_margin, max(train_values) + train_margin
    )
    train_ax.set_xlim(tokens[0], tokens[-1])
    zoom_start = max(tokens[0], 40.0)
    train_ax.axvspan(
        zoom_start,
        tokens[-1],
        color="#3c78a8",
        alpha=0.055,
        linewidth=0.0,
        zorder=0,
    )
    train_ax.text(
        (zoom_start + tokens[-1]) / 2.0,
        0.035,
        "late window enlarged below",
        transform=train_ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=6.8,
        color="#555555",
    )
    train_ax.set_xlabel("Processed tokens (billions)")
    train_ax.set_ylabel("Training loss")
    train_ax.yaxis.grid(True, color="#d7d7d7", linewidth=0.55, alpha=0.8)
    train_ax.set_axisbelow(True)
    train_ax.set_title(
        "Matched 8B training snapshot",
        loc="left",
        pad=9,
        fontweight="bold",
    )
    train_ax.text(
        0.0,
        1.01,
        ("B4 · W64 · S4096 · identical data coordinates · " "one trajectory per arm"),
        transform=train_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.4,
        color="#555555",
    )
    train_ax.legend(loc="upper right", frameon=False, handlelength=2.0)

    late_validation_series = [
        row for row in validation_series if row["tokens"] / 1.0e9 >= zoom_start
    ]
    validation_tokens = [
        row["tokens"] / 1.0e9 for row in late_validation_series
    ]
    for short_name, field in (
        ("bf16", "bf16_loss"),
        ("fp8", "fp8_loss"),
    ):
        validation_ax.plot(
            validation_tokens,
            [row[field] for row in late_validation_series],
            color=colors[short_name],
            linewidth=1.35,
            marker="o",
            markersize=3.2,
            markeredgecolor="white",
            markeredgewidth=0.45,
            zorder=3,
        )
    validation_values = [
        row[field]
        for row in late_validation_series
        for field in ("bf16_loss", "fp8_loss")
    ]
    validation_margin = max(
        0.025, (max(validation_values) - min(validation_values)) * 0.15
    )
    validation_ax.set_ylim(
        min(validation_values) - validation_margin,
        max(validation_values) + validation_margin,
    )
    validation_ax.set_xlim(zoom_start, tokens[-1])
    validation_ax.set_xlabel("Processed tokens (billions)")
    validation_ax.set_ylabel("Validation loss")
    validation_ax.yaxis.grid(True, color="#d7d7d7", linewidth=0.55, alpha=0.8)
    validation_ax.set_axisbelow(True)
    validation_ax.set_title(
        "Late-stage validation zoom (expanded y-axis)",
        loc="left",
        pad=6,
        fontsize=8.2,
        fontweight="bold",
    )

    endpoint = validation["endpoint"]
    validation_ax.annotate(
        f"same-update gap +{endpoint['loss_difference_fp8_minus_bf16']:.3f}",
        xy=(endpoint["tokens"] / 1.0e9, endpoint["fp8_loss"]),
        xytext=(-4, 9),
        textcoords="offset points",
        ha="right",
        va="bottom",
        color=colors["fp8"],
        fontsize=7.2,
        fontweight="medium",
    )
    validation_ax.text(
        0.5,
        -0.29,
        (
            "Thin lines: 25-update training reports; thick lines: 1B-token EWM.\n"
            "The lower panel enlarges same-update validation; "
            "this snapshot ends before the 100B-token target."
        ),
        transform=validation_ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.0,
        color="#555555",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        bbox_inches="tight",
        metadata={
            "Title": "Matched B4/W64 Llama 3.1 8B training snapshot",
            "Creator": "plot_causal_training.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)


def render_matched_b4_throughput(receipt: dict[str, Any], output: Path) -> None:
    healthy = receipt["healthy_matched_comparison"]
    throughput = healthy["throughput"]
    series = [
        row
        for row in healthy["training"]["series"]
        if throughput["first_update"] <= row["update"] <= throughput["last_update"]
    ]
    if len(series) != throughput["common_rows"]:
        raise ValueError("throughput plot window does not match receipt summary")
    specs = (
        (
            "bf16",
            "BF16 FA4",
            "#6f6f6f",
            throughput["bf16_tokens_per_second_per_gpu"]["median"],
        ),
        (
            "nvfp4_projection_fp8_pv",
            "NVFP4 projections + FP8 P/V",
            "#3c78a8",
            throughput["fp8_tokens_per_second_per_gpu"]["median"],
        ),
    )
    tokens = [row["tokens"] / 1.0e9 for row in series]

    fig, ax = plt.subplots(figsize=(5.35, 3.55), constrained_layout=True)
    for field, label, color, median_tps in specs:
        values = [row[field]["tokens_per_second_per_gpu"] / 1.0e3 for row in series]
        ax.plot(
            tokens,
            values,
            color=color,
            linewidth=0.75,
            alpha=0.55,
            marker="o",
            markersize=1.9,
            markeredgewidth=0.0,
            label=label,
            zorder=3,
        )
        ax.axhline(
            median_tps / 1.0e3,
            color=color,
            linewidth=1.2,
            linestyle="--",
            alpha=0.95,
            zorder=4,
        )

    bf16_median = throughput["bf16_tokens_per_second_per_gpu"]["median"]
    fp8_median = throughput["fp8_tokens_per_second_per_gpu"]["median"]
    speedup = throughput["ratio_of_median_throughputs"]
    ax.set_xlim(tokens[0], tokens[-1])
    ax.set_xlabel("Processed tokens (billions)")
    ax.set_ylabel("Tokens/s/GPU (thousands)")
    ax.yaxis.grid(True, color="#d7d7d7", linewidth=0.55, alpha=0.8)
    ax.set_axisbelow(True)
    ax.set_title(
        "Matched distributed throughput",
        loc="left",
        pad=9,
        fontweight="bold",
    )
    ax.text(
        0.0,
        1.01,
        (
            f"{throughput['common_rows']} aligned post-warmup reports · median "
            f"{bf16_median / 1.0e3:.2f}k → {fp8_median / 1.0e3:.2f}k "
            f"tokens/s/GPU · {speedup:.3f}×"
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.4,
        color="#555555",
    )
    ax.legend(loc="lower right", frameon=False, handlelength=2.1)
    ax.text(
        0.5,
        -0.22,
        (
            "Dashed lines are medians. Scheduled-save and input-stall windows "
            "are retained; no outliers are removed."
        ),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.1,
        color="#555555",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        bbox_inches="tight",
        metadata={
            "Title": "Matched B4/W64 Llama 3.1 8B distributed throughput",
            "Creator": "plot_causal_training.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)


def render_matched_b4_mx_failure(receipt: dict[str, Any], output: Path) -> None:
    diagnostic = receipt["mxfp4_divergence"]
    train = diagnostic["full_training_series"]
    validation = diagnostic["full_validation_series"]
    tokens = [row["tokens"] / 1.0e9 for row in train]
    color = "#c84c3a"
    validation_color = "#7f2f27"

    fig, (loss_ax, grad_ax) = plt.subplots(
        2,
        1,
        figsize=(5.35, 4.65),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": (1.0, 0.92)},
    )
    loss_ax.plot(
        tokens,
        [row["loss"] for row in train],
        color=color,
        linewidth=1.05,
        label="Training loss",
        zorder=3,
    )
    loss_ax.plot(
        [row["tokens"] / 1.0e9 for row in validation],
        [row["loss"] for row in validation],
        color=validation_color,
        linewidth=1.0,
        marker="o",
        markersize=3.1,
        markeredgecolor="white",
        markeredgewidth=0.45,
        label="Validation loss",
        zorder=4,
    )
    loss_ax.set_ylabel("Loss")
    loss_ax.yaxis.grid(True, color="#d7d7d7", linewidth=0.55, alpha=0.8)
    loss_ax.set_axisbelow(True)
    loss_ax.set_title(
        "Current B4 MXFP4 P/V failure",
        loc="left",
        pad=9,
        fontweight="bold",
    )
    loss_ax.text(
        0.0,
        1.01,
        "E4M3 learned projections · NVFP4 QK · MXFP4/E8M0-block32 P/V",
        transform=loss_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.4,
        color="#555555",
    )
    loss_ax.legend(loc="upper right", frameon=False, ncol=2, handlelength=1.7)

    grad_ax.plot(
        tokens,
        [max(row["preclip_grad_norm"], 1.0e-12) for row in train],
        color=color,
        linewidth=1.05,
        zorder=3,
    )
    grad_ax.set_yscale("log")
    grad_ax.set_xlim(tokens[0], tokens[-1])
    grad_ax.set_xlabel("Processed tokens (billions)")
    grad_ax.set_ylabel("Pre-clipping gradient norm")
    grad_ax.yaxis.grid(True, which="major", color="#d7d7d7", linewidth=0.55, alpha=0.8)
    grad_ax.set_axisbelow(True)

    maximum = diagnostic["maximum_observed_preclip_grad_norm"]
    grad_ax.annotate(
        f"{maximum['value'] / 1.0e6:.1f}M at update {maximum['update']:,}",
        xy=(maximum["tokens"] / 1.0e9, maximum["value"]),
        xytext=(7, -7),
        textcoords="offset points",
        ha="left",
        va="top",
        color=validation_color,
        fontsize=7.2,
        fontweight="medium",
    )
    grad_ax.text(
        0.5,
        -0.29,
        (
            "The run was cancelled after update 2,550. This panel is a "
            "numerical-failure diagnostic, not a throughput comparison."
        ),
        transform=grad_ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.1,
        color="#555555",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        bbox_inches="tight",
        metadata={
            "Title": "Current B4 MXFP4 P/V numerical-failure diagnostic",
            "Creator": "plot_causal_training.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    e2e_receipt = json.loads(args.e2e_receipt.read_text())
    boundary_receipt = json.loads(args.boundary_receipt.read_text())
    training_receipt = json.loads(args.training_receipt.read_text())
    matched_b4_receipt = json.loads(args.matched_b4_receipt.read_text())
    validate_e2e_receipt(e2e_receipt)
    validate_boundary_receipt(boundary_receipt)
    validate_training_receipt(training_receipt)
    validate_matched_b4_receipt(matched_b4_receipt)
    setup_style()
    render_e2e_batch_scaling(e2e_receipt, args.e2e_output)
    render_isolated_backward(boundary_receipt, args.isolated_output)
    render_combined_forward_backward(boundary_receipt, args.combined_output)
    render_training_curves(training_receipt, args.training_output)
    render_mxfp4_divergence(training_receipt, args.divergence_output)
    render_matched_b4_training(matched_b4_receipt, args.matched_b4_training_output)
    render_matched_b4_throughput(matched_b4_receipt, args.matched_b4_throughput_output)
    render_matched_b4_mx_failure(matched_b4_receipt, args.matched_b4_mx_failure_output)


if __name__ == "__main__":
    main()
