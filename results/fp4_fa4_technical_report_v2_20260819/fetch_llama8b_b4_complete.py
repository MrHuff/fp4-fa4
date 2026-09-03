#!/usr/bin/env python3
"""Freeze the checkpoint-selected 100B-token matched-B4 trajectories.

Each logical trajectory spans several metric-tracker runs because allocations
moved between clusters.  Resume checkpoints define the branch that continued
to the paper endpoint.  This exporter keeps rows before or after those
checkpoints as specified below, verifies that the stitched coordinates are
complete, and writes a credential-free receipt.  Private tracker locators are
supplied in an uncommitted source map and are never copied into the output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import wandb

from export_llama8b_b4_snapshot import (
    aligned_healthy_comparison,
    compare_history_with_worker_log,
    parse_worker_log,
    sha256_file,
)


REPORT_DIR = Path(__file__).resolve().parent
BASE_RECEIPT = (
    REPORT_DIR / "receipts" / "llama8b_b4_matched_snapshot_20260902T1358Z.json"
)
TOKENS_PER_UPDATE = 4_194_304
TARGET_UPDATE = 23_842
LAST_TRAIN_REPORT = 23_825
LAST_VALIDATION_REPORT = 23_840
TRAIN_CADENCE = 25
VALIDATION_CADENCE = 298

HISTORY_FIELDS = {
    "train": {
        "loss_metrics/global_avg_loss": "loss",
        "grad_norm": "preclip_grad_norm",
        "throughput(tps)": "tokens_per_second_per_gpu",
        "mfu(%)": "mfu_percent",
    },
    "validation": {
        "validation_metrics/loss": "loss",
        "validation_metrics/throughput(tps)": "tokens_per_second_per_gpu",
    },
}

# Bounds are applied to both training and validation coordinates.  A row at a
# resume step belongs to the checkpoint that was loaded; the continued run
# contributes only later observations.  The abandoned BF16 resume is
# intentionally excluded because the final run restarted from the earlier
# 16,252 checkpoint.
LINEAGES: dict[str, dict[str, Any]] = {
    "bf16": {
        "public_arm_label": "bf16_b4_control",
        "route": "BF16 FA4 baseline",
        "segments": [
            {
                "public_segment_label": "bf16_initial_through_16252",
                "expected_state": "crashed",
                "lower_exclusive": None,
                "upper_inclusive": 16_252,
                "role": "initial trajectory through the selected checkpoint",
            },
            {
                "public_segment_label": "bf16_completion_after_16252",
                "expected_state": "finished",
                "lower_exclusive": 16_252,
                "upper_inclusive": None,
                "role": "selected continuation to the 100B-token target",
            },
        ],
    },
    "fp8": {
        "public_arm_label": "nvfp4_projection_fp8_pv_b4",
        "route": (
            "NVFP4 learned QKV/O projections + NVFP4 attention QK + "
            "E4M3 FP8 attention PV + E5M2-dO v509 backward"
        ),
        "segments": [
            {
                "public_segment_label": "fp8_initial_through_17925",
                "expected_state": "finished",
                "lower_exclusive": None,
                "upper_inclusive": 17_925,
                "role": "initial trajectory through the selected checkpoint",
            },
            {
                "public_segment_label": "fp8_bridge_17925_to_18881",
                "expected_state": "crashed",
                "lower_exclusive": 17_925,
                "upper_inclusive": 18_881,
                "role": "selected first continuation",
            },
            {
                "public_segment_label": "fp8_completion_after_18881",
                "expected_state": "finished",
                "lower_exclusive": 18_881,
                "upper_inclusive": None,
                "role": "selected continuation to the 100B-token target",
            },
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-map",
        type=Path,
        required=True,
        help=(
            "uncommitted JSON containing tracker paths and expected run names; "
            "service identifiers are never copied into the output"
        ),
    )
    parser.add_argument("--bf16-final-log", type=Path, required=True)
    parser.add_argument("--fp8-final-log", type=Path, required=True)
    parser.add_argument(
        "--captured-utc",
        help=(
            "explicit YYYY-MM-DDTHH:MM:SSZ receipt timestamp; defaults to the "
            "current UTC time"
        ),
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="permit a running final arm for a private preview receipt",
    )
    return parser.parse_args()


def load_source_map(path: Path) -> dict[str, Any]:
    """Validate private tracker locators without returning them in a receipt."""
    payload = json.loads(path.read_text())
    if payload.get("schema") != "tkfa4.metric_lineage_sources.v1":
        raise ValueError("unsupported --source-map schema")
    arms = payload.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(LINEAGES):
        raise ValueError("--source-map must contain exactly the BF16 and FP8 arms")
    for arm_name, spec in LINEAGES.items():
        private_arm = arms.get(arm_name)
        if not isinstance(private_arm, dict):
            raise ValueError(f"--source-map: {arm_name} must be an object")
        expected_run_name = private_arm.get("expected_run_name")
        if not isinstance(expected_run_name, str) or not expected_run_name:
            raise ValueError(f"--source-map: {arm_name} run name is missing")
        private_segments = private_arm.get("segments")
        expected_labels = {
            segment["public_segment_label"] for segment in spec["segments"]
        }
        if (
            not isinstance(private_segments, dict)
            or set(private_segments) != expected_labels
        ):
            raise ValueError(
                f"--source-map: {arm_name} must define exactly {sorted(expected_labels)}"
            )
        for label, locator in private_segments.items():
            if not isinstance(locator, str) or locator.count("/") != 2:
                raise ValueError(
                    f"--source-map: {arm_name}:{label} must be entity/project/run"
                )
    return arms


def finite_float(value: Any, *, context: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context}: invalid numeric value {value!r}") from error
    if not math.isfinite(parsed):
        raise ValueError(f"{context}: nonfinite numeric value")
    return parsed


def exact_int(value: Any, *, context: str) -> int:
    parsed = finite_float(value, context=context)
    integer = int(parsed)
    if parsed != integer:
        raise ValueError(f"{context}: expected an integer, found {value!r}")
    return integer


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def selected(update: int, segment: dict[str, Any]) -> bool:
    lower = segment["lower_exclusive"]
    upper = segment["upper_inclusive"]
    return (lower is None or update > lower) and (upper is None or update <= upper)


def fetch_series(
    run: Any,
    *,
    arm_name: str,
    segment: dict[str, Any],
    kind: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    segment_label = segment["public_segment_label"]
    field_map = HISTORY_FIELDS[kind]
    keys = ["_step", *field_map]
    if kind == "train":
        keys.insert(1, "n_tokens_seen")

    unique: dict[int, dict[str, Any]] = {}
    raw_rows = 0
    duplicate_updates: list[int] = []
    for row_index, source in enumerate(run.scan_history(keys=keys, page_size=5_000)):
        if source.get("_step") is None:
            continue
        first_metric = next(iter(field_map))
        if source.get(first_metric) is None:
            continue
        update = exact_int(
            source["_step"],
            context=f"{arm_name}:{segment_label}:{kind}:{row_index}:_step",
        )
        if update <= 0:
            raise ValueError(f"{arm_name}:{segment_label}:{kind}: bad update")
        parsed: dict[str, Any] = {
            "update": update,
            "tokens": update * TOKENS_PER_UPDATE,
        }
        if kind == "train":
            observed_tokens = exact_int(
                source.get("n_tokens_seen"),
                context=f"{arm_name}:{segment_label}:{update}:tokens",
            )
            if observed_tokens != parsed["tokens"]:
                raise ValueError(f"{arm_name}:{segment_label}:{update}: token mismatch")
        for source_name, output_name in field_map.items():
            parsed[output_name] = finite_float(
                source.get(source_name),
                context=(f"{arm_name}:{segment_label}:{kind}:{update}:{source_name}"),
            )
        if update in unique:
            if unique[update] != parsed:
                raise ValueError(
                    f"{arm_name}:{segment_label}:{kind}:{update}: "
                    "conflicting duplicate"
                )
            duplicate_updates.append(update)
        else:
            unique[update] = parsed
        raw_rows += 1

    if not unique:
        raise ValueError(f"{arm_name}:{segment_label}:{kind}: no history")
    retained = [
        unique[update] for update in sorted(unique) if selected(update, segment)
    ]
    if not retained:
        raise ValueError(
            f"{arm_name}:{segment_label}:{kind}: selection retained no rows"
        )
    diagnostics = {
        "source_raw_rows": raw_rows,
        "source_unique_rows": len(unique),
        "source_first_update": min(unique),
        "source_last_update": max(unique),
        "identical_duplicate_updates_removed": duplicate_updates,
        "selected_rows": len(retained),
        "selected_first_update": retained[0]["update"],
        "selected_last_update": retained[-1]["update"],
        "selected_series_sha256": canonical_sha256(retained),
    }
    return retained, diagnostics


def expected_coordinates(last_update: int, *, kind: str) -> list[int]:
    cadence = TRAIN_CADENCE if kind == "train" else VALIDATION_CADENCE
    return [1, *range(cadence, last_update + 1, cadence)]


def merge_arm(
    api: wandb.Api,
    arm_name: str,
    spec: dict[str, Any],
    private_source: dict[str, Any],
    *,
    allow_incomplete: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    merged = {"train": {}, "validation": {}}
    segment_receipts: list[dict[str, Any]] = []
    for segment_index, segment in enumerate(spec["segments"]):
        segment_label = segment["public_segment_label"]
        run = api.run(private_source["segments"][segment_label])
        expected_run_name = private_source["expected_run_name"]
        if run.name != expected_run_name:
            raise ValueError(
                f"{arm_name}:{segment_label}: run name {run.name!r} "
                f"!= {expected_run_name!r}"
            )
        expected_state = segment["expected_state"]
        accepted_states = {expected_state}
        is_final = segment_index == len(spec["segments"]) - 1
        if allow_incomplete and is_final:
            accepted_states.add("running")
        if run.state not in accepted_states:
            raise ValueError(
                f"{arm_name}:{segment_label}: state {run.state!r} not in "
                f"{sorted(accepted_states)!r}"
            )

        kinds: dict[str, Any] = {}
        for kind in ("train", "validation"):
            rows, diagnostics = fetch_series(
                run,
                arm_name=arm_name,
                segment=segment,
                kind=kind,
            )
            for row in rows:
                update = row["update"]
                if update in merged[kind]:
                    raise ValueError(
                        f"{arm_name}:{kind}:{update}: selected lineages overlap"
                    )
                merged[kind][update] = row
            kinds[kind] = diagnostics
        segment_receipts.append(
            {
                **segment,
                "run_state_at_capture": run.state,
                "history": kinds,
            }
        )

    result: dict[str, Any] = {}
    for kind in ("train", "validation"):
        updates = sorted(merged[kind])
        expected = expected_coordinates(updates[-1], kind=kind)
        if updates != expected:
            missing = sorted(set(expected) - set(updates))
            extra = sorted(set(updates) - set(expected))
            raise ValueError(
                f"{arm_name}:{kind}: incomplete stitched cadence; "
                f"missing={missing[:8]} extra={extra[:8]}"
            )
        result[kind] = [merged[kind][update] for update in updates]

    if not allow_incomplete:
        if result["train"][-1]["update"] != LAST_TRAIN_REPORT:
            raise ValueError(f"{arm_name}: final training report is incomplete")
        if result["validation"][-1]["update"] != LAST_VALIDATION_REPORT:
            raise ValueError(f"{arm_name}: final validation report is incomplete")

    arm_receipt = {
        "public_arm_label": spec["public_arm_label"],
        "route": spec["route"],
        "status_at_capture": (
            "completed_100b_target"
            if result["train"][-1]["update"] == LAST_TRAIN_REPORT
            and result["validation"][-1]["update"] == LAST_VALIDATION_REPORT
            and segment_receipts[-1]["run_state_at_capture"] == "finished"
            else "incomplete_private_preview"
        ),
        "observed_train_range": {
            "first_update": result["train"][0]["update"],
            "last_update": result["train"][-1]["update"],
            "rows": len(result["train"]),
        },
        "observed_validation_range": {
            "first_update": result["validation"][0]["update"],
            "last_update": result["validation"][-1]["update"],
            "rows": len(result["validation"]),
        },
    }
    return result, {
        "summary": arm_receipt,
        "segments": segment_receipts,
    }


def check_final_log(
    path: Path,
    *,
    arm_name: str,
    history: dict[str, Any],
    allow_incomplete: bool,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    worker = parse_worker_log(path, arm_name=arm_name)
    crosscheck = compare_history_with_worker_log(
        history,
        worker,
        arm_name=arm_name,
    )
    text = path.read_text(errors="strict")
    evidence = {
        "final_checkpoint_step_23842": (
            "Saving a full checkpoint at last step, step 23842." in text
        ),
        "training_completed": text.count("Training completed") >= 4,
        "remote_sync_stop_complete": "event=remote_sync_stop_complete" in text,
        "node_success_at_step_23842": (
            "FA4 NODE SUCCESS" in text and "last_step=23842" in text
        ),
    }
    if not allow_incomplete and not all(evidence.values()):
        raise ValueError(f"{arm_name}: final worker log lacks completion evidence")
    return {
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "role": "worker-0 four-local-rank metric and terminal-completion cross-check",
        "wandb_crosscheck": crosscheck,
        "completion_evidence": evidence,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY must be supplied in the environment")
    if args.captured_utc is None:
        captured_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        captured_utc = args.captured_utc
        try:
            datetime.strptime(captured_utc, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError as error:
            raise ValueError("--captured-utc must use YYYY-MM-DDTHH:MM:SSZ") from error

    base = json.loads(BASE_RECEIPT.read_text())
    if base.get("schema") != "tkfa4.report.llama8b_b4_matched_snapshot.v1":
        raise ValueError("unexpected base matched-B4 receipt schema")
    if base["shared_recipe"].get("tokens_per_update") != TOKENS_PER_UPDATE:
        raise ValueError("base receipt token coordinate changed")
    if base["shared_recipe"].get("target_tokens") != TARGET_UPDATE * TOKENS_PER_UPDATE:
        raise ValueError("base receipt target coordinate changed")

    private_sources = load_source_map(args.source_map)
    api = wandb.Api(timeout=180)
    parsed: dict[str, dict[str, Any]] = {}
    arm_captures: dict[str, dict[str, Any]] = {}
    for arm_name, spec in LINEAGES.items():
        parsed[arm_name], arm_captures[arm_name] = merge_arm(
            api,
            arm_name,
            spec,
            private_sources[arm_name],
            allow_incomplete=args.allow_incomplete,
        )
    healthy = aligned_healthy_comparison(parsed["bf16"], parsed["fp8"])

    log_paths = {
        "bf16": args.bf16_final_log,
        "fp8": args.fp8_final_log,
    }
    final_logs = {
        arm_name: check_final_log(
            path,
            arm_name=arm_name,
            history=parsed[arm_name],
            allow_incomplete=args.allow_incomplete,
        )
        for arm_name, path in log_paths.items()
    }

    target_complete = all(
        arm_captures[name]["summary"]["status_at_capture"] == "completed_100b_target"
        and all(final_logs[name]["completion_evidence"].values())
        for name in ("bf16", "fp8")
    )
    if not args.allow_incomplete and not target_complete:
        raise ValueError("both matched trajectories must complete the target")
    arms = {
        name: {
            **arm_captures[name]["summary"],
            "completion": {
                "target_update": TARGET_UPDATE,
                "target_tokens": TARGET_UPDATE * TOKENS_PER_UPDATE,
                "last_scheduled_train_report": LAST_TRAIN_REPORT,
                "last_scheduled_validation_report": LAST_VALIDATION_REPORT,
                **final_logs[name]["completion_evidence"],
            },
        }
        for name in ("bf16", "fp8")
    }

    receipt = {
        "schema": "tkfa4.report.llama8b_b4_completed.v2",
        "capture": {
            "captured_utc": captured_utc,
            "credential_free": True,
            "source_kind": (
                "read-only metric history stitched at authenticated checkpoint "
                "resume coordinates; service identifiers redacted"
            ),
            "source_map_schema": "tkfa4.metric_lineage_sources.v1",
            "service_identifiers_redacted": True,
            "full_healthy_histories": True,
            "raw_source_artifacts_committed": False,
            "target_complete": target_complete,
            "target_update": TARGET_UPDATE,
            "last_scheduled_train_report": LAST_TRAIN_REPORT,
            "last_scheduled_validation_report": LAST_VALIDATION_REPORT,
            "selection_policy": (
                "ordered checkpoint lineage; keep update > lower_exclusive and "
                "update <= upper_inclusive; never resolve forks by timestamp"
            ),
            "base_receipt_basename": BASE_RECEIPT.name,
            "base_negative_control_payload_sha256": canonical_sha256(
                base["mxfp4_divergence"]
            ),
            "sources": {
                name: {
                    "segments": arm_captures[name]["segments"],
                    "final_worker_log": final_logs[name],
                }
                for name in ("bf16", "fp8")
            },
            "excluded_segments": [
                {
                    "arm": "bf16",
                    "public_segment_label": "bf16_abandoned_resume_after_16252",
                    "reason": (
                        "the final continuation restarted from checkpoint 16252; "
                        "including both branches would duplicate one trajectory"
                    ),
                }
            ],
            "superseded_suffixes_excluded": [
                {
                    "arm": "bf16",
                    "public_segment_label": "bf16_initial_through_16252",
                    "updates": "greater than 16252",
                },
                {
                    "arm": "fp8",
                    "public_segment_label": "fp8_initial_through_17925",
                    "updates": "greater than 17925",
                },
                {
                    "arm": "fp8",
                    "public_segment_label": "fp8_bridge_17925_to_18881",
                    "updates": "greater than 18881",
                },
            ],
        },
        "identity": base["identity"],
        "shared_recipe": base["shared_recipe"],
        "arms": arms,
        "healthy_matched_comparison": healthy,
        "negative_control_reference": {
            "role": "separate MXFP4-PV divergence diagnostic",
            "receipt_basename": BASE_RECEIPT.name,
            "payload_sha256": canonical_sha256(base["mxfp4_divergence"]),
            "plot_data_key": "mxfp4_divergence",
        },
        "claim_boundary": [
            "BF16 and NVFP4-projection plus FP8-PV are compared only at common update coordinates from the same B4/W64 recipe.",
            "Resume checkpoints select one continuous logical trajectory per arm; abandoned or superseded observations after a selected checkpoint are excluded.",
            "Both retained trajectories reached the 100,000,595,968-token target; the last scheduled training and validation reports occur just before the terminal update.",
            "This is one trajectory per arm and does not estimate run-to-run uncertainty.",
            "MXFP4-PV remains the separately inherited divergence diagnostic and is excluded from healthy-route throughput comparisons.",
            "Throughput aggregates every common post-warmup report, including scheduled-save and input-stall windows.",
        ],
    }
    serialized = json.dumps(receipt, sort_keys=True)
    private_values = [
        private_sources[arm_name]["expected_run_name"]
        for arm_name in sorted(private_sources)
    ]
    private_values.extend(
        locator
        for arm_name in sorted(private_sources)
        for locator in private_sources[arm_name]["segments"].values()
    )
    leaked = [value for value in private_values if value in serialized]
    if leaked:
        raise RuntimeError("private tracker identifiers leaked into the receipt")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": sha256_file(args.output),
                "captured_utc": captured_utc,
                "target_complete": target_complete,
                "training_endpoint": healthy["training"]["last_common_update"],
                "validation_endpoint": healthy["validation"]["last_common_update"],
                "throughput_ratio": healthy["throughput"][
                    "ratio_of_median_throughputs"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
