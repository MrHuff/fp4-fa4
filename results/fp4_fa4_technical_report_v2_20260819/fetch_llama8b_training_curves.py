#!/usr/bin/env python3
"""Freeze redacted Llama-8B training curves from a metric tracker.

The script performs read-only API calls. It requires ``WANDB_API_KEY`` in the
environment and an operator-supplied source map, but never writes credentials,
service-side run identifiers, run names, configuration, or environment into
the receipt. Every unique token coordinate returned for a current arm
contributes to its token-domain smoother; only one display row per 100 updates
is retained to keep the artifact compact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import wandb


REPORT_DIR = Path(__file__).resolve().parent
INITIAL_CUTOFF_RECEIPT = (
    REPORT_DIR / "receipts" / "v509_four_arm_cutoff_20260831T2209Z.json"
)
TOKENS_PER_UPDATE = 64 * 4096
HALF_LIFE_TOKENS = 1_000_000_000.0
CURRENT_STRIDE_UPDATES = 100
BASELINE_MAX_ROWS = 3_000
HISTORY_KEYS = (
    "_step",
    "n_tokens_seen",
    "loss_metrics/global_avg_loss",
    "grad_norm",
)
ARMS = {
    "e4_fp8": {
        "public_arm_label": "e4m3_projection_fp8_pv_b1",
        "learned_projection": "E4M3",
        "forward_pv": "FP8",
        "status_class": "working_at_snapshot",
    },
    "e4_mx": {
        "public_arm_label": "e4m3_projection_mxfp4_pv_b1",
        "learned_projection": "E4M3",
        "forward_pv": "MXFP4",
        "status_class": "diverged",
    },
    "nv_fp8": {
        "public_arm_label": "nvfp4_projection_fp8_pv_b1",
        "learned_projection": "NVFP4",
        "forward_pv": "FP8",
        "status_class": "working_at_snapshot",
    },
    "nv_mx": {
        "public_arm_label": "nvfp4_projection_mxfp4_pv_b1",
        "learned_projection": "NVFP4",
        "forward_pv": "MXFP4",
        "status_class": "diverged",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new timestamped receipt path; an existing file is never overwritten",
    )
    parser.add_argument(
        "--source-map",
        type=Path,
        required=True,
        help=(
            "uncommitted JSON mapping each public arm key to its tracker run "
            "path; see this script's load_source_map() contract"
        ),
    )
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument(
        "--baseline-provenance",
        type=Path,
        required=True,
    )
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def update_ewm(
    previous: float | None,
    previous_tokens: int | None,
    loss: float,
    tokens: int,
) -> float:
    if previous is None or previous_tokens is None:
        return loss
    delta_tokens = max(0, tokens - previous_tokens)
    alpha = 1.0 - math.exp2(-delta_tokens / HALF_LIFE_TOKENS)
    return alpha * loss + (1.0 - alpha) * previous


def load_source_map(path: Path) -> dict[str, str]:
    """Load private tracker locators without copying them into public output.

    The input schema is ``{"schema": "tkfa4.metric_sources.v1", "runs":
    {"e4_fp8": "entity/project/run", ...}}``. The file is an operator input and
    must not be committed when it contains service-side identifiers.
    """
    payload = json.loads(path.read_text())
    if payload.get("schema") != "tkfa4.metric_sources.v1":
        raise ValueError("unsupported --source-map schema")
    runs = payload.get("runs")
    if not isinstance(runs, dict) or set(runs) != set(ARMS):
        raise ValueError("--source-map must define exactly the four public arm keys")
    if not all(isinstance(value, str) and value for value in runs.values()):
        raise ValueError("--source-map run paths must be non-empty strings")
    return runs


def fetch_arm(
    key: str, spec: dict[str, Any], source_locator: str
) -> tuple[str, dict[str, Any]]:
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            api = wandb.Api(timeout=180)
            run = api.run(source_locator)

            source_rows = 0
            rows_by_tokens: dict[int, tuple[float, float]] = {}
            for row in run.scan_history(keys=list(HISTORY_KEYS), page_size=5_000):
                tokens = int(row["n_tokens_seen"])
                loss = float(row["loss_metrics/global_avg_loss"])
                grad_norm = float(row["grad_norm"])
                if tokens <= 0 or tokens % TOKENS_PER_UPDATE:
                    raise RuntimeError(f"{key}: invalid token coordinate {tokens}")
                if not all(math.isfinite(value) for value in (loss, grad_norm)):
                    raise RuntimeError(f"{key}: nonfinite history row at {tokens}")
                # A resumed W&B stream can replay token coordinates and reset
                # its private _step counter.  Token count is the training
                # coordinate; preserve the final observation returned by
                # scan_history for each coordinate.
                rows_by_tokens[tokens] = (loss, grad_norm)
                source_rows += 1
            if not rows_by_tokens:
                raise RuntimeError(f"{key}: no finite history rows")

            previous_update = 0
            previous_tokens: int | None = None
            smoothed_loss: float | None = None
            maximum_grad_norm = 0.0
            series: list[dict[str, Any]] = []
            bin_id: int | None = None
            bin_grad_max = 0.0
            bin_last: dict[str, Any] | None = None

            def flush_bin() -> None:
                nonlocal bin_last, bin_grad_max
                if bin_last is None:
                    return
                series.append(
                    {
                        **bin_last,
                        "grad_norm_bin_max": bin_grad_max,
                    }
                )
                bin_last = None
                bin_grad_max = 0.0

            for tokens, (loss, grad_norm) in sorted(rows_by_tokens.items()):
                update = tokens // TOKENS_PER_UPDATE
                if update <= previous_update:
                    raise RuntimeError(f"{key}: non-increasing update {update}")

                smoothed_loss = update_ewm(
                    smoothed_loss,
                    previous_tokens,
                    loss,
                    tokens,
                )
                current_bin = (update - 1) // CURRENT_STRIDE_UPDATES
                if bin_id is not None and current_bin != bin_id:
                    flush_bin()
                bin_id = current_bin
                bin_grad_max = max(bin_grad_max, grad_norm)
                maximum_grad_norm = max(maximum_grad_norm, grad_norm)
                bin_last = {
                    "update": update,
                    "tokens": tokens,
                    "loss": loss,
                    "smoothed_loss": smoothed_loss,
                    "grad_norm": grad_norm,
                }
                previous_update = update
                previous_tokens = tokens

            flush_bin()
            unique_rows = len(rows_by_tokens)
            missing_updates = previous_update - unique_rows
            if not series or missing_updates < 0:
                raise RuntimeError(f"{key}: invalid deduplicated history")
            return key, {
                "public_arm_label": spec["public_arm_label"],
                "source_state_at_capture": run.state,
                "learned_projection": spec["learned_projection"],
                "forward_pv": spec["forward_pv"],
                "status_class": spec["status_class"],
                "source_history_rows": source_rows,
                "full_history_rows": unique_rows,
                "replayed_rows_removed": source_rows - unique_rows,
                "missing_update_coordinates": missing_updates,
                "display_stride_updates": CURRENT_STRIDE_UPDATES,
                "display_rows": len(series),
                "last_update": previous_update,
                "last_tokens": previous_tokens,
                "last_loss": series[-1]["loss"],
                "last_smoothed_loss": series[-1]["smoothed_loss"],
                "last_grad_norm": series[-1]["grad_norm"],
                "maximum_grad_norm": maximum_grad_norm,
                "series": series,
            }
        except Exception as error:  # W&B occasionally returns transient 5xx.
            last_error = error
            if attempt < 3:
                time.sleep(2**attempt)
    assert last_error is not None
    raise last_error


def load_historical_bf16(path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    previous_tokens: int | None = None
    smoothed: float | None = None
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["case"] != "B1" or row["role"] != "baseline":
                continue
            tokens = int(float(row["tokens_seen"]))
            loss = float(row["loss"])
            committed_smoothed = float(row["smoothed_loss"])
            if previous_tokens is not None and tokens <= previous_tokens:
                raise RuntimeError("historical BF16 token coordinates are not increasing")
            smoothed = update_ewm(smoothed, previous_tokens, loss, tokens)
            if not math.isclose(
                smoothed,
                committed_smoothed,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise RuntimeError(
                    f"historical BF16 EWM mismatch at {tokens}: "
                    f"{smoothed} != {committed_smoothed}"
                )
            rows.append(
                {
                    "tokens": tokens,
                    "loss": loss,
                    "smoothed_loss": smoothed,
                }
            )
            previous_tokens = tokens
    if not rows:
        raise RuntimeError("historical BF16 source contains no B1 baseline rows")
    stride = max(1, math.ceil(len(rows) / BASELINE_MAX_ROWS))
    display = rows[::stride]
    if display[-1] is not rows[-1]:
        display.append(rows[-1])
    return {
        "identity": "SFU B1 native-SiLU BF16 causal-FA4 reference",
        "comparison_class": "historical_convergence_sanity_reference",
        "topology": "32 GB200; FSDP32; local B1; global B32; S4096",
        "full_history_rows": len(rows),
        "display_stride_rows": stride,
        "display_rows": len(display),
        "last_tokens": rows[-1]["tokens"],
        "last_loss": rows[-1]["loss"],
        "last_smoothed_loss": rows[-1]["smoothed_loss"],
        "series": display,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(
            f"refusing to overwrite frozen training receipt: {args.output}"
        )
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY must be supplied in the environment")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    for path in (
        args.source_map,
        args.baseline_csv,
        args.baseline_provenance,
        INITIAL_CUTOFF_RECEIPT,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    source_map = load_source_map(args.source_map)
    with ThreadPoolExecutor(max_workers=min(args.workers, len(ARMS))) as executor:
        fetched = dict(
            executor.map(
                lambda item: fetch_arm(item[0], item[1], source_map[item[0]]),
                ARMS.items(),
            )
        )
    historical_bf16 = load_historical_bf16(args.baseline_csv)
    initial_cutoff = json.loads(INITIAL_CUTOFF_RECEIPT.read_text())

    stable_keys = ("e4_fp8", "nv_fp8")
    common_stable_tokens = min(fetched[key]["last_tokens"] for key in stable_keys)
    common_four_arm_tokens = min(arm["last_tokens"] for arm in fetched.values())
    payload = {
        "schema": "tkfa4.report.llama8b_training_curves.v1",
        "snapshot_utc": datetime.now(timezone.utc).isoformat(),
        "capture": {
            "mode": "read-only metric history plus materialized historical CSV",
            "credential_free_receipt": True,
            "source_map_sha256": sha256_file(args.source_map),
            "source_map_contents_committed": False,
            "history_keys": list(HISTORY_KEYS),
            "smoothing": {
                "kind": "causal-token-ewm",
                "half_life_tokens": HALF_LIFE_TOKENS,
                "formula": "alpha = 1 - 2**(-delta_tokens / half_life_tokens)",
                "current_arms_use_every_unique_token_coordinate": True,
                "duplicate_resolution": (
                    "final observation returned by W&B scan_history for each "
                    "token coordinate"
                ),
            },
            "current_display_stride_updates": CURRENT_STRIDE_UPDATES,
        },
        "source_identity": initial_cutoff["source_identity"],
        "shared_recipe": initial_cutoff["shared_recipe"],
        "sources": {
            "initial_four_arm_cutoff": {
                "path": str(INITIAL_CUTOFF_RECEIPT.relative_to(REPORT_DIR)),
                "sha256": sha256_file(INITIAL_CUTOFF_RECEIPT),
            },
            "historical_bf16_csv": {
                "path": str(args.baseline_csv),
                "sha256": sha256_file(args.baseline_csv),
            },
            "historical_bf16_provenance": {
                "path": str(args.baseline_provenance),
                "sha256": sha256_file(args.baseline_provenance),
            },
        },
        "historical_bf16": historical_bf16,
        "current_arms": fetched,
        "plot_boundaries": {
            "common_stable_tokens": common_stable_tokens,
            "common_four_arm_tokens": common_four_arm_tokens,
            "first_retained_bad_observations": initial_cutoff[
                "first_retained_bad_observations"
            ],
        },
        "claim_boundary": [
            "The historical BF16 curve is a convergence sanity reference, not a paired control.",
            "The historical run used FSDP32/global B32; current arms use DDP64/global B64.",
            "Exact historical tokenizer bytes and sample order were not preserved.",
            "Both FP8-PV arms are working through the snapshot; this is not completed convergence.",
            "Both MXFP4-PV arms diverged; the crossed design identifies P/V format or induced model state as the common separator, not a single causal instruction.",
            "Instantaneous distributed throughput is not a matched BF16 speedup measurement.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "sha256": sha256_file(args.output),
        "snapshot_utc": payload["snapshot_utc"],
        "last_tokens": {key: arm["last_tokens"] for key, arm in fetched.items()},
        "last_smoothed_loss": {
            key: arm["last_smoothed_loss"] for key, arm in fetched.items()
        },
        "maximum_grad_norm": {
            key: arm["maximum_grad_norm"] for key, arm in fetched.items()
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
