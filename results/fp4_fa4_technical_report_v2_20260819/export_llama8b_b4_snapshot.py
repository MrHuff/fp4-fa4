#!/usr/bin/env python3
"""Export a credential-free matched-B4 snapshot from scrubbed tracker histories.

The full, credential-scrubbed metric-history exports are the primary
healthy-series source.  Overlapping worker-0 logs provide an independent
check: each contains one Titan metric line from every one of the four local
ranks.  This exporter verifies that multiplicity and rank agreement before it
compares the two sources.  The complete MX worker log supplies its final row.
Neither raw logs nor authentication material are copied into the report tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


REPORT_DIR = Path(__file__).resolve().parent
DEFAULT_LAUNCH_RECEIPT = (
    REPORT_DIR / "receipts" / "llama8b_b4_w64_launch_check_20260902.json"
)
TOKENS_PER_UPDATE = 4_194_304
EXPECTED_LOCAL_RANKS = 4
WARMUP_UPDATES = 2_000
SMOOTHING_HALF_LIFE_TOKENS = 1_000_000_000.0
MX_DIVERGENCE_ONSET_UPDATE = 325

ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
TRAIN_LINE = re.compile(
    r"(?<!validate )step:\s*(\d+)\s+"
    r"loss:\s*(\S+)\s+"
    r"grad_norm:\s*(\S+)\s+"
    r"memory:.*?"
    r"tps:\s*([\d,]+)\s+"
    r"tflops:\s*(\S+)\s+"
    r"mfu:\s*(\S+)%"
)
VALIDATION_LINE = re.compile(
    r"validate step:\s*(\d+)\s+" r"loss:\s*(\S+)\s+" r"memory:.*?" r"tps:\s*([\d,]+)"
)

ARM_SPECS = {
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf16-history", type=Path, required=True)
    parser.add_argument("--fp8-history", type=Path, required=True)
    parser.add_argument("--mx-history", type=Path, required=True)
    parser.add_argument("--bf16-log", type=Path, required=True)
    parser.add_argument("--fp8-log", type=Path, required=True)
    parser.add_argument("--mx-log", type=Path, required=True)
    parser.add_argument("--launch-receipt", type=Path, default=DEFAULT_LAUNCH_RECEIPT)
    parser.add_argument(
        "--captured-utc",
        required=True,
        help="UTC timestamp assigned when the scrubbed histories were frozen",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing output (disabled by default)",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_finite(token: str, *, context: str) -> float:
    try:
        value = float(token.replace(",", ""))
    except ValueError as error:
        raise ValueError(f"{context}: invalid numeric token {token!r}") from error
    if not math.isfinite(value):
        raise ValueError(f"{context}: nonfinite numeric token {token!r}")
    return value


def percentile(values: list[float], probability: float) -> float:
    """Linear-interpolated sample percentile (NumPy's default convention)."""
    if not values:
        raise ValueError("cannot summarize an empty sample")
    ordered = sorted(values)
    coordinate = (len(ordered) - 1) * probability
    lower = math.floor(coordinate)
    upper = math.ceil(coordinate)
    fraction = coordinate - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize(values: list[float]) -> dict[str, float | int]:
    return {
        "samples": len(values),
        "minimum": min(values),
        "p10": percentile(values, 0.10),
        "median": statistics.median(values),
        "p90": percentile(values, 0.90),
        "maximum": max(values),
    }


def load_manifest(log_path: Path) -> tuple[dict[str, Any], Path]:
    manifest_path = log_path.parent.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entries = manifest.get("entries", [])
    matching = [
        entry
        for entry in entries
        if entry.get("workerIdx") == 0 and entry.get("status") == "success"
    ]
    if len(matching) != 1:
        raise ValueError(
            f"{manifest_path}: expected one successful worker-0 entry, "
            f"found {len(matching)}"
        )
    return manifest, manifest_path


def check_spread(values: list[float], *, tolerance: float, context: str) -> float:
    spread = max(values) - min(values)
    if spread > tolerance:
        raise ValueError(f"{context}: local-rank spread {spread} exceeds {tolerance}")
    return spread


def parse_worker_log(path: Path, *, arm_name: str) -> dict[str, Any]:
    train_groups: dict[int, list[dict[str, float]]] = defaultdict(list)
    validation_groups: dict[int, list[dict[str, float]]] = defaultdict(list)

    for line_number, raw_line in enumerate(
        path.read_text(errors="strict").splitlines(), start=1
    ):
        line = ANSI_ESCAPE.sub("", raw_line)
        validation_match = VALIDATION_LINE.search(line)
        if validation_match:
            update = int(validation_match.group(1))
            validation_groups[update].append(
                {
                    "loss": as_finite(
                        validation_match.group(2),
                        context=f"{arm_name}:{line_number}:validation loss",
                    ),
                    "tokens_per_second_per_gpu": as_finite(
                        validation_match.group(3),
                        context=f"{arm_name}:{line_number}:validation throughput",
                    ),
                }
            )
            continue

        train_match = TRAIN_LINE.search(line)
        if train_match:
            update = int(train_match.group(1))
            train_groups[update].append(
                {
                    "loss": as_finite(
                        train_match.group(2),
                        context=f"{arm_name}:{line_number}:training loss",
                    ),
                    "preclip_grad_norm": as_finite(
                        train_match.group(3),
                        context=f"{arm_name}:{line_number}:gradient norm",
                    ),
                    "tokens_per_second_per_gpu": as_finite(
                        train_match.group(4),
                        context=f"{arm_name}:{line_number}:training throughput",
                    ),
                    "tflops_per_gpu": as_finite(
                        train_match.group(5),
                        context=f"{arm_name}:{line_number}:TFLOP/s",
                    ),
                    "mfu_percent": as_finite(
                        train_match.group(6),
                        context=f"{arm_name}:{line_number}:MFU",
                    ),
                }
            )

    if not train_groups or not validation_groups:
        raise ValueError(f"{path}: missing training or validation metrics")

    for kind, groups in (("train", train_groups), ("validation", validation_groups)):
        bad = {
            update: len(rows)
            for update, rows in groups.items()
            if len(rows) != EXPECTED_LOCAL_RANKS
        }
        if bad:
            raise ValueError(
                f"{arm_name}:{kind}: expected {EXPECTED_LOCAL_RANKS} local-rank "
                f"copies at every update; mismatches={bad}"
            )

    spread_maxima = {
        "train_loss": 0.0,
        "train_preclip_grad_norm": 0.0,
        "train_tokens_per_second_per_gpu": 0.0,
        "train_tflops_per_gpu": 0.0,
        "train_mfu_percent": 0.0,
        "validation_loss": 0.0,
        "validation_tokens_per_second_per_gpu": 0.0,
    }
    train_tolerances = {
        "loss": 0.0,
        "preclip_grad_norm": 0.0,
        "tokens_per_second_per_gpu": 16.0,
        "tflops_per_gpu": 0.5,
        "mfu_percent": 0.02,
    }
    validation_tolerances = {
        "loss": 0.0,
        # Validation ranks time their local iterator windows independently.
        # This bound still catches gross disagreement while admitting the
        # measured 30--37 token/s spread in the complete terminal logs.
        "tokens_per_second_per_gpu": 64.0,
    }

    train_series: list[dict[str, Any]] = []
    for update in sorted(train_groups):
        copies = train_groups[update]
        row: dict[str, Any] = {
            "update": update,
            "tokens": update * TOKENS_PER_UPDATE,
        }
        for metric, tolerance in train_tolerances.items():
            values = [copy[metric] for copy in copies]
            spread = check_spread(
                values,
                tolerance=tolerance,
                context=f"{arm_name}:train:{update}:{metric}",
            )
            spread_maxima[f"train_{metric}"] = max(
                spread_maxima[f"train_{metric}"], spread
            )
            row[metric] = statistics.median(values)
        train_series.append(row)

    validation_series: list[dict[str, Any]] = []
    for update in sorted(validation_groups):
        copies = validation_groups[update]
        row = {"update": update, "tokens": update * TOKENS_PER_UPDATE}
        for metric, tolerance in validation_tolerances.items():
            values = [copy[metric] for copy in copies]
            spread = check_spread(
                values,
                tolerance=tolerance,
                context=f"{arm_name}:validation:{update}:{metric}",
            )
            spread_maxima[f"validation_{metric}"] = max(
                spread_maxima[f"validation_{metric}"], spread
            )
            row[metric] = statistics.median(values)
        validation_series.append(row)

    return {
        "train": train_series,
        "validation": validation_series,
        "deduplication": {
            "expected_local_rank_copies_per_update": EXPECTED_LOCAL_RANKS,
            "raw_train_rows": sum(len(rows) for rows in train_groups.values()),
            "unique_train_updates": len(train_series),
            "raw_validation_rows": sum(
                len(rows) for rows in validation_groups.values()
            ),
            "unique_validation_updates": len(validation_series),
            "observed_train_multiplicities": sorted(
                {len(rows) for rows in train_groups.values()}
            ),
            "observed_validation_multiplicities": sorted(
                {len(rows) for rows in validation_groups.values()}
            ),
            "maximum_local_rank_spread": spread_maxima,
            "reduction": "median across four local-rank copies",
        },
    }


def _exact_integer(value: Any, *, context: str) -> int:
    parsed = as_finite(str(value), context=context)
    integer = int(parsed)
    if parsed != integer:
        raise ValueError(f"{context}: expected an integer, found {value!r}")
    return integer


def _validate_complete_cadence(
    updates: list[int], *, cadence: int, context: str
) -> None:
    maximum = updates[-1]
    expected = [1, *range(cadence, maximum + 1, cadence)]
    if updates != expected:
        missing = sorted(set(expected) - set(updates))
        extra = sorted(set(updates) - set(expected))
        raise ValueError(
            f"{context}: incomplete cadence; missing={missing[:8]} extra={extra[:8]}"
        )


def parse_wandb_history(path: Path, *, arm_name: str) -> dict[str, Any]:
    """Parse a credential-scrubbed W&B scan_history export."""
    payload = json.loads(path.read_text())
    spec = ARM_SPECS[arm_name]
    expected_state = spec["wandb_state_at_capture"]
    if payload.get("run_state") != expected_state:
        raise ValueError(
            f"{path}: expected W&B state {expected_state!r}, "
            f"found {payload.get('run_state')!r}"
        )

    field_maps = {
        "train": {
            "loss": "loss_metrics/global_avg_loss",
            "preclip_grad_norm": "grad_norm",
            "tokens_per_second_per_gpu": "throughput(tps)",
            "mfu_percent": "mfu(%)",
        },
        "validation": {
            "loss": "validation_metrics/loss",
            "tokens_per_second_per_gpu": "validation_metrics/throughput(tps)",
        },
    }
    result: dict[str, Any] = {}
    history_diagnostics: dict[str, Any] = {}
    for kind, field_map in field_maps.items():
        raw_rows = payload.get(kind)
        if not isinstance(raw_rows, list) or not raw_rows:
            raise ValueError(f"{path}: missing {kind} history")
        unique: dict[int, dict[str, Any]] = {}
        duplicate_updates: list[int] = []
        for row_index, source_row in enumerate(raw_rows):
            update = _exact_integer(
                source_row.get("_step"),
                context=f"{arm_name}:{kind}:{row_index}:_step",
            )
            if update <= 0:
                raise ValueError(f"{arm_name}:{kind}: non-positive update")
            parsed_row: dict[str, Any] = {
                "update": update,
                "tokens": update * TOKENS_PER_UPDATE,
            }
            if kind == "train":
                observed_tokens = _exact_integer(
                    source_row.get("n_tokens_seen"),
                    context=f"{arm_name}:{kind}:{update}:n_tokens_seen",
                )
                if observed_tokens != parsed_row["tokens"]:
                    raise ValueError(
                        f"{arm_name}:{kind}:{update}: token coordinate "
                        f"{observed_tokens} != {parsed_row['tokens']}"
                    )
            for output_name, source_name in field_map.items():
                parsed_row[output_name] = as_finite(
                    str(source_row.get(source_name)),
                    context=f"{arm_name}:{kind}:{update}:{source_name}",
                )
            if update in unique:
                if unique[update] != parsed_row:
                    raise ValueError(
                        f"{arm_name}:{kind}:{update}: conflicting W&B duplicates"
                    )
                duplicate_updates.append(update)
            else:
                unique[update] = parsed_row

        updates = sorted(unique)
        _validate_complete_cadence(
            updates,
            cadence=25 if kind == "train" else 298,
            context=f"{arm_name}:{kind}",
        )
        result[kind] = [unique[update] for update in updates]
        history_diagnostics[kind] = {
            "raw_rows": len(raw_rows),
            "unique_updates": len(updates),
            "identical_duplicate_updates_removed": duplicate_updates,
            "first_update": updates[0],
            "last_update": updates[-1],
            "cadence_after_update_1": 25 if kind == "train" else 298,
        }

    result["history_deduplication"] = history_diagnostics
    return result


def compare_history_with_worker_log(
    history: dict[str, Any], worker: dict[str, Any], *, arm_name: str
) -> dict[str, Any]:
    """Validate the rounded four-rank log against full-precision rank-0 history."""
    tolerances = {
        "train": {
            "loss": 5.1e-5,
            "preclip_grad_norm": 5.1e-5,
            "tokens_per_second_per_gpu": 16.0,
            # Worker logs round MFU to two decimals after local-rank timing;
            # the terminal capture's largest W&B-to-median delta is 0.024.
            "mfu_percent": 0.03,
        },
        "validation": {
            "loss": 5.1e-5,
            "tokens_per_second_per_gpu": 64.0,
        },
    }
    result: dict[str, Any] = {}
    for kind in ("train", "validation"):
        history_rows = indexed(history[kind])
        worker_rows = indexed(worker[kind])
        overlap = sorted(set(history_rows) & set(worker_rows))
        if not overlap:
            raise ValueError(f"{arm_name}:{kind}: sources have no overlap")
        fields = (
            ("loss", "tokens_per_second_per_gpu")
            if kind == "validation"
            else (
                "loss",
                "preclip_grad_norm",
                "tokens_per_second_per_gpu",
                "mfu_percent",
            )
        )
        maximum_differences = {}
        for field in fields:
            differences = [
                abs(history_rows[update][field] - worker_rows[update][field])
                for update in overlap
            ]
            maximum = max(differences)
            tolerance = tolerances[kind][field]
            if maximum > tolerance:
                raise ValueError(
                    f"{arm_name}:{kind}:{field}: source difference {maximum} "
                    f"exceeds {tolerance}"
                )
            maximum_differences[field] = maximum
        result[kind] = {
            "overlapping_updates": len(overlap),
            "first_overlap_update": overlap[0],
            "last_overlap_update": overlap[-1],
            "maximum_absolute_difference": maximum_differences,
            "tolerances": {field: tolerances[kind][field] for field in fields},
        }
    return result


def merge_mx_history_with_log_tail(
    history: dict[str, Any], worker: dict[str, Any]
) -> dict[str, Any]:
    history_train = indexed(history["train"])
    worker_train = indexed(worker["train"])
    history_last = max(history_train)
    suffix = sorted(update for update in worker_train if update > history_last)
    if suffix != [history_last + 25]:
        raise ValueError(f"MX worker-log suffix is unexpected: {suffix}")
    merged_train = [*history["train"], *(worker_train[update] for update in suffix)]
    return {
        "train": merged_train,
        "validation": history["validation"],
        "history_deduplication": history["history_deduplication"],
        "worker_log_tail_updates": suffix,
    }


def causal_ewm(losses: list[float], tokens: list[int]) -> list[float]:
    if not losses or len(losses) != len(tokens):
        raise ValueError("invalid EWM inputs")
    smoothed = [losses[0]]
    for index in range(1, len(losses)):
        delta_tokens = tokens[index] - tokens[index - 1]
        if delta_tokens <= 0:
            raise ValueError("EWM token coordinates are not increasing")
        old_weight = math.exp(
            -math.log(2.0) * delta_tokens / SMOOTHING_HALF_LIFE_TOKENS
        )
        smoothed.append(old_weight * smoothed[-1] + (1.0 - old_weight) * losses[index])
    return smoothed


def indexed(series: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result = {row["update"]: row for row in series}
    if len(result) != len(series):
        raise ValueError("series contains duplicate update coordinates")
    return result


def aligned_healthy_comparison(
    bf16: dict[str, Any], fp8: dict[str, Any]
) -> dict[str, Any]:
    bf16_train = indexed(bf16["train"])
    fp8_train = indexed(fp8["train"])
    train_updates = sorted(set(bf16_train) & set(fp8_train))
    if not train_updates:
        raise ValueError("BF16/FP8 histories have no aligned training rows")

    bf16_losses = [bf16_train[update]["loss"] for update in train_updates]
    fp8_losses = [fp8_train[update]["loss"] for update in train_updates]
    token_coordinates = [update * TOKENS_PER_UPDATE for update in train_updates]
    bf16_smoothed = causal_ewm(bf16_losses, token_coordinates)
    fp8_smoothed = causal_ewm(fp8_losses, token_coordinates)

    train_series = []
    for index, update in enumerate(train_updates):
        baseline = dict(bf16_train[update])
        low_precision = dict(fp8_train[update])
        baseline["smoothed_loss"] = bf16_smoothed[index]
        low_precision["smoothed_loss"] = fp8_smoothed[index]
        train_series.append(
            {
                "update": update,
                "tokens": update * TOKENS_PER_UPDATE,
                "bf16": baseline,
                "nvfp4_projection_fp8_pv": low_precision,
                "loss_difference_fp8_minus_bf16": (
                    low_precision["loss"] - baseline["loss"]
                ),
                "throughput_ratio_fp8_over_bf16": (
                    low_precision["tokens_per_second_per_gpu"]
                    / baseline["tokens_per_second_per_gpu"]
                ),
            }
        )

    bf16_validation = indexed(bf16["validation"])
    fp8_validation = indexed(fp8["validation"])
    validation_updates = sorted(set(bf16_validation) & set(fp8_validation))
    if not validation_updates:
        raise ValueError("BF16/FP8 logs have no aligned validation rows")
    validation_series = []
    for update in validation_updates:
        baseline = bf16_validation[update]
        low_precision = fp8_validation[update]
        validation_series.append(
            {
                "update": update,
                "tokens": update * TOKENS_PER_UPDATE,
                "bf16_loss": baseline["loss"],
                "fp8_loss": low_precision["loss"],
                "loss_difference_fp8_minus_bf16": (
                    low_precision["loss"] - baseline["loss"]
                ),
            }
        )

    throughput_series = [row for row in train_series if row["update"] >= WARMUP_UPDATES]
    if not throughput_series:
        raise ValueError("BF16/FP8 histories have no aligned post-warmup rows")
    bf16_tps = [row["bf16"]["tokens_per_second_per_gpu"] for row in throughput_series]
    fp8_tps = [
        row["nvfp4_projection_fp8_pv"]["tokens_per_second_per_gpu"]
        for row in throughput_series
    ]
    paired_ratios = [row["throughput_ratio_fp8_over_bf16"] for row in throughput_series]
    bf16_mfu = [row["bf16"]["mfu_percent"] for row in throughput_series]
    fp8_mfu = [
        row["nvfp4_projection_fp8_pv"]["mfu_percent"] for row in throughput_series
    ]
    throughput = {
        "selection": (
            "all common logged training coordinates at or after warmup; no "
            "checkpoint, validation, input-stall, or outlier exclusions"
        ),
        "percentile_method": "linear interpolation at (n-1)*p",
        "first_update": throughput_series[0]["update"],
        "last_update": throughput_series[-1]["update"],
        "first_tokens": throughput_series[0]["tokens"],
        "last_tokens": throughput_series[-1]["tokens"],
        "common_rows": len(throughput_series),
        "bf16_tokens_per_second_per_gpu": summarize(bf16_tps),
        "fp8_tokens_per_second_per_gpu": summarize(fp8_tps),
        "bf16_mfu_percent": summarize(bf16_mfu),
        "fp8_mfu_percent": summarize(fp8_mfu),
        "ratio_of_median_throughputs": (
            statistics.median(fp8_tps) / statistics.median(bf16_tps)
        ),
        "paired_throughput_ratio": summarize(paired_ratios),
        "interpretation": (
            "The median is the headline aggregate because scheduled saves and "
            "input stalls create a small number of unmatched reporting-window dips."
        ),
    }

    return {
        "training": {
            "first_common_update": train_updates[0],
            "last_common_update": train_updates[-1],
            "first_common_tokens": token_coordinates[0],
            "last_common_tokens": token_coordinates[-1],
            "common_rows": len(train_series),
            "series": train_series,
            "endpoint": train_series[-1],
            "smoothing": {
                "method": "causal token-domain exponential moving average",
                "half_life_tokens": SMOOTHING_HALF_LIFE_TOKENS,
                "initialization": "each arm starts at common update 1",
            },
        },
        "validation": {
            "first_common_update": validation_updates[0],
            "last_common_update": validation_updates[-1],
            "first_common_tokens": validation_updates[0] * TOKENS_PER_UPDATE,
            "last_common_tokens": validation_updates[-1] * TOKENS_PER_UPDATE,
            "common_rows": len(validation_series),
            "series": validation_series,
            "endpoint": validation_series[-1],
        },
        "throughput": throughput,
    }


def build_mx_diagnostic(mx: dict[str, Any]) -> dict[str, Any]:
    train = indexed(mx["train"])
    validation = mx["validation"]
    required = (300, MX_DIVERGENCE_ONSET_UPDATE, 350, 400)
    missing = [update for update in required if update not in train]
    if missing:
        raise ValueError(f"MX log misses diagnostic updates {missing}")
    maximum = max(mx["train"], key=lambda row: row["preclip_grad_norm"])
    return {
        "comparison_class": (
            "numerical divergence diagnostic; excluded from healthy-route "
            "throughput comparisons"
        ),
        "observed_departure_update": MX_DIVERGENCE_ONSET_UPDATE,
        "observed_departure_tokens": MX_DIVERGENCE_ONSET_UPDATE * TOKENS_PER_UPDATE,
        "selected_training_rows": [train[update] for update in required],
        "maximum_observed_preclip_grad_norm": {
            "update": maximum["update"],
            "tokens": maximum["tokens"],
            "value": maximum["preclip_grad_norm"],
            "loss": maximum["loss"],
        },
        "last_training_row": mx["train"][-1],
        "last_validation_row": validation[-1],
        "full_training_series": mx["train"],
        "full_validation_series": validation,
    }


def validate_launch_receipt(receipt: dict[str, Any]) -> None:
    if receipt.get("schema") != "tkfa4.report.llama8b_b4_w64_launch_check.v1":
        raise ValueError("unsupported launch receipt schema")
    if receipt["shared_recipe"].get("tokens_per_update") != TOKENS_PER_UPDATE:
        raise ValueError("launch receipt uses a different token coordinate")
    if receipt["shared_recipe"].get("local_batch") != 4:
        raise ValueError("launch receipt is not the B4 recipe")
    working = receipt.get("working_arms", {})
    if working.get("bf16_fa4", {}).get("public_arm_label") != ARM_SPECS["bf16"][
        "public_arm_label"
    ]:
        raise ValueError("BF16 public arm label disagrees with launch receipt")
    if working.get("nvfp4_projection_fp8_pv", {}).get(
        "public_arm_label"
    ) != ARM_SPECS["fp8"]["public_arm_label"]:
        raise ValueError("FP8 public arm label disagrees with launch receipt")


def main() -> None:
    args = parse_args()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", args.captured_utc):
        raise ValueError("--captured-utc must use YYYY-MM-DDTHH:MM:SSZ")
    if args.output.exists() and not args.force:
        raise FileExistsError(
            f"refusing to overwrite {args.output}; pass --force explicitly"
        )

    launch_receipt = json.loads(args.launch_receipt.read_text())
    validate_launch_receipt(launch_receipt)

    log_paths = {"bf16": args.bf16_log, "fp8": args.fp8_log, "mx": args.mx_log}
    history_paths = {
        "bf16": args.bf16_history,
        "fp8": args.fp8_history,
        "mx": args.mx_history,
    }
    histories: dict[str, dict[str, Any]] = {}
    worker_logs: dict[str, dict[str, Any]] = {}
    crosschecks: dict[str, dict[str, Any]] = {}
    capture_sources: dict[str, dict[str, Any]] = {}
    for arm_name, path in log_paths.items():
        spec = ARM_SPECS[arm_name]
        manifest, manifest_path = load_manifest(path)
        worker_logs[arm_name] = parse_worker_log(path, arm_name=arm_name)
        history_path = history_paths[arm_name]
        histories[arm_name] = parse_wandb_history(history_path, arm_name=arm_name)
        crosschecks[arm_name] = compare_history_with_worker_log(
            histories[arm_name], worker_logs[arm_name], arm_name=arm_name
        )
        capture_sources[arm_name] = {
            "public_arm_label": spec["public_arm_label"],
            "wandb_history_export": {
                "basename": history_path.name,
                "bytes": history_path.stat().st_size,
                "sha256": sha256_file(history_path),
                "source_api": (
                    "read-only metric-history export; credentials and service "
                    "identifiers excluded"
                ),
            },
            "worker_log_crosscheck": {
                "worker_index": 0,
                "basename": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "download_manifest_sha256": sha256_file(manifest_path),
                "downloaded_at_utc": manifest["generatedAt"],
                "role": (
                    "four-local-rank multiplicity and rounded-metric cross-check; "
                    "MX also contributes the final update-2550 row"
                    if arm_name == "mx"
                    else "four-local-rank multiplicity and rounded-metric cross-check"
                ),
            },
        }

    parsed = {
        "bf16": histories["bf16"],
        "fp8": histories["fp8"],
        "mx": merge_mx_history_with_log_tail(histories["mx"], worker_logs["mx"]),
    }
    healthy = aligned_healthy_comparison(parsed["bf16"], parsed["fp8"])
    arms = {}
    for arm_name in ("bf16", "fp8", "mx"):
        arms[arm_name] = {
            **ARM_SPECS[arm_name],
            "observed_train_range": {
                "first_update": parsed[arm_name]["train"][0]["update"],
                "last_update": parsed[arm_name]["train"][-1]["update"],
                "rows": len(parsed[arm_name]["train"]),
            },
            "observed_validation_range": {
                "first_update": parsed[arm_name]["validation"][0]["update"],
                "last_update": parsed[arm_name]["validation"][-1]["update"],
                "rows": len(parsed[arm_name]["validation"]),
            },
            "wandb_history_deduplication": histories[arm_name]["history_deduplication"],
            "worker_log_four_rank_deduplication": worker_logs[arm_name][
                "deduplication"
            ],
            "wandb_worker_log_crosscheck": crosschecks[arm_name],
        }
        if arm_name == "mx":
            arms[arm_name]["worker_log_tail_updates"] = parsed[arm_name][
                "worker_log_tail_updates"
            ]

    receipt = {
        "schema": "tkfa4.report.llama8b_b4_matched_snapshot.v1",
        "capture": {
            "captured_utc": args.captured_utc,
            "credential_free": True,
            "source_kind": (
                "full read-only metric-history exports with worker-0 log "
                "cross-checks"
            ),
            "full_healthy_histories": True,
            "raw_source_artifacts_committed": False,
            "sources": capture_sources,
            "launch_receipt_basename": args.launch_receipt.name,
            "launch_receipt_sha256": sha256_file(args.launch_receipt),
        },
        "identity": launch_receipt["identity"],
        "shared_recipe": launch_receipt["shared_recipe"],
        "arms": arms,
        "healthy_matched_comparison": healthy,
        "mxfp4_divergence": build_mx_diagnostic(parsed["mx"]),
        "claim_boundary": [
            "BF16 and NVFP4-projection plus FP8-PV are compared only at common update coordinates from the same B4/W64 recipe.",
            "This is an interim snapshot of one trajectory per arm, not a completed 100B-token convergence result or a repeated-run uncertainty estimate.",
            "Healthy curves use complete credential-scrubbed metric histories through the frozen common endpoint; worker-log tails are cross-checks, not the curve source.",
            "MXFP4-PV uses E4M3 learned projections and is retained only as a divergence diagnostic; it is excluded from the healthy throughput comparison.",
            "The original BF16 and FP8 allocations were cancelled after this snapshot and resumed from complete checkpoints at normal Volt priority; resumed observations are not included here.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
