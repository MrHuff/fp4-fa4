#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parent
PREFIX = "forward_large_shape_high_head_adaptation_20260711"
WORKER = RESULTS / "forward_issue_lane_overlap_bf16_matrix_20260711_driver.py"
ROUTES = (
    "stage2,e16pc,fp4_auto,split_full,split_k64,"
    "tk_bf16_fullgrid,cute_bf16"
)
CELLS = (
    (1, 2048, 16), (1, 2048, 32), (1, 2048, 64),
    (1, 4096, 8), (1, 4096, 16), (1, 4096, 32), (1, 4096, 64),
    (1, 8192, 4), (1, 8192, 8), (1, 8192, 16), (1, 8192, 32),
    (1, 8192, 64),
    (1, 16384, 1), (1, 16384, 2), (1, 16384, 4), (1, 16384, 8),
    (1, 16384, 16),
    (1, 32768, 1), (1, 32768, 4), (1, 32768, 16),
    (2, 4096, 8), (4, 4096, 4),
)


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    aggregate: dict[str, object] = {
        "task": "large-shape high-head required matrix",
        "created_unix": time.time(),
        "device": args.device,
        "warmup": args.warmup,
        "samples": args.samples,
        "routes": ROUTES.split(","),
        "launch_mode": "fullgrid",
        "cells": [],
    }
    if args.resume and args.output.exists():
        aggregate = json.loads(args.output.read_text())
    completed = {
        (int(row["shape"]["batch"]), int(row["shape"]["seqlen"]),
         int(row["shape"]["heads"]))
        for row in aggregate.get("cells", [])
        if isinstance(row, dict) and "shape" in row and "worker_error" not in row
    }

    for index, (batch, seqlen, heads) in enumerate(CELLS):
        if (batch, seqlen, heads) in completed:
            continue
        cell_path = RESULTS / (
            f"{PREFIX}_matrix_cell_b{batch}_s{seqlen}_h{heads}.json"
        )
        command = [
            sys.executable, str(WORKER), "--worker",
            "--batch", str(batch), "--seqlen", str(seqlen),
            "--heads", str(heads), "--seed", str(94601 + index),
            "--device", args.device, "--warmup", str(args.warmup),
            "--samples", str(args.samples), "--order-offset", str(index),
            "--routes", ROUTES, "--fp4-launch-mode", "fullgrid",
            "--timeout-ms", "30000", "--json", str(cell_path),
        ]
        timeout_s = 900 if seqlen >= 16384 else 420
        try:
            completed_run = subprocess.run(
                command,
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
            if completed_run.returncode == 0 and cell_path.exists():
                row = json.loads(cell_path.read_text())
            else:
                row = {
                    "shape": {"batch": batch, "seqlen": seqlen, "heads": heads},
                    "worker_error": f"exit {completed_run.returncode}",
                    "stdout_tail": completed_run.stdout[-4000:],
                    "stderr_tail": completed_run.stderr[-4000:],
                }
        except subprocess.TimeoutExpired as exc:
            row = {
                "shape": {"batch": batch, "seqlen": seqlen, "heads": heads},
                "worker_error": f"timeout after {timeout_s}s",
                "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
                "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
            }
        aggregate.setdefault("cells", []).append(row)
        aggregate["updated_unix"] = time.time()
        write_json(args.output, aggregate)
        comparison = row.get("measured_comparison") if isinstance(row, dict) else None
        print(json.dumps({"shape": [batch, seqlen, heads], "comparison": comparison,
                          "error": row.get("worker_error")}), flush=True)

    aggregate["completed_unix"] = time.time()
    write_json(args.output, aggregate)


if __name__ == "__main__":
    main()
