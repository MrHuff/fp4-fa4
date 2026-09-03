#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DRIVER = Path(__file__).with_name("forward_stage2_pchain_driver.py")
DEFAULT_ROUTES = ("stage2", "e16pc")


def percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "p25": None, "median": None, "p75": None}
    return {
        "count": len(values),
        "p25": percentile(values, 0.25),
        "median": float(statistics.median(values)),
        "p75": percentile(values, 0.75),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqlen", type=int, required=True)
    parser.add_argument("--heads", type=int, required=True)
    parser.add_argument("--target-idx", type=int, required=True)
    parser.add_argument("--valid-runs", type=int, default=11)
    parser.add_argument("--max-attempts", type=int, default=24)
    parser.add_argument("--process-timeout-s", type=float, default=60.0)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--routes", default=",".join(DEFAULT_ROUTES))
    parser.add_argument("--temp-dir", type=Path, default=Path("/dev/shm"))
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()

    routes = tuple(route.strip() for route in args.routes.split(",") if route.strip())
    if not routes:
        raise ValueError("at least one route is required")
    collected: dict[str, list[dict[str, Any]]] = {route: [] for route in routes}
    attempts: list[dict[str, Any]] = []
    args.temp_dir.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, args.max_attempts + 1):
        if all(len(collected[route]) >= args.valid_runs for route in routes):
            break
        with tempfile.NamedTemporaryFile(
            prefix="e16pc_pchain_",
            suffix=".json",
            dir=args.temp_dir,
            delete=False,
        ) as temp_file:
            child_json = Path(temp_file.name)
        command = [
            sys.executable,
            str(DRIVER),
            "--seqlen",
            str(args.seqlen),
            "--heads",
            str(args.heads),
            "--device",
            args.device,
            "--configs",
            ",".join(routes),
            "--warmup",
            "0",
            "--rounds",
            "0",
            "--samples-per-round",
            "1",
            "--stamp-runs",
            "1",
            "--stamp-target-idx",
            str(args.target_idx),
            "--timeout-ms",
            str(max(1000.0, args.process_timeout_s * 500.0)),
            "--json",
            str(child_json),
        ]
        attempt_record: dict[str, Any] = {"attempt": attempt, "status": "unknown"}
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
                timeout=args.process_timeout_s,
            )
            attempt_record["returncode"] = completed.returncode
            if completed.returncode != 0 or not child_json.exists():
                attempt_record["status"] = "child_error"
                attempt_record["stderr_tail"] = completed.stderr[-1000:]
            else:
                child = json.loads(child_json.read_text())
                attempt_record["status"] = "completed"
                attempt_record["routes"] = {}
                for route in routes:
                    stamp_record = child["records"][route].get("pchain_stamps", {})
                    valid_runs = stamp_record.get("valid_runs", [])
                    rejected_runs = stamp_record.get("rejected_runs", [])
                    attempt_record["routes"][route] = {
                        "valid": bool(valid_runs),
                        "rejected": rejected_runs,
                    }
                    if valid_runs and len(collected[route]) < args.valid_runs:
                        collected[route].append(
                            {
                                "attempt": attempt,
                                "raw": stamp_record["raw_runs"][0],
                                **valid_runs[0],
                            }
                        )
        except subprocess.TimeoutExpired:
            attempt_record["status"] = "timeout"
        finally:
            child_json.unlink(missing_ok=True)
        attempts.append(attempt_record)

    route_summaries: dict[str, Any] = {}
    for route, records in collected.items():
        interval_names = sorted(
            {
                name
                for record in records
                for name in record["intervals_cycles"]
            }
        )
        route_summaries[route] = {
            "valid_count": len(records),
            "records": records,
            "intervals_cycles": {
                name: summarize(
                    [record["intervals_cycles"][name] for record in records]
                )
                for name in interval_names
            },
        }

    complete = all(
        len(collected[route]) >= args.valid_runs for route in routes
    )
    result = {
        "shape": f"h{args.heads}_s{args.seqlen}",
        "target_idx": args.target_idx,
        "device": args.device,
        "requested_valid_runs": args.valid_runs,
        "max_attempts": args.max_attempts,
        "process_timeout_s": args.process_timeout_s,
        "complete": complete,
        "attempts": attempts,
        "routes": route_summaries,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "shape": result["shape"],
                "target_idx": args.target_idx,
                "complete": complete,
                "attempts": len(attempts),
                "valid_counts": {
                    route: len(records) for route, records in collected.items()
                },
            },
            indent=2,
        )
    )
    if not complete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
