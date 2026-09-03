#!/usr/bin/env python3
"""Build and benchmark the direct HAO topology across HAO's D128 suite."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIGS = (
    (1, 256, 16),
    (1, 1024, 16),
    (4, 4096, 16),
    (1, 32768, 16),
    (4, 4096, 32),
    (1, 4096, 12),
    (1, 32768, 12),
    (1, 4096, 24),
    (1, 32768, 24),
)
MODULES = {
    "nvfp4": "_C_tk_hao_direct",
    "mxfp4": "_C_tk_hao_direct_mxqk_fp8pv",
}


def parse_config(value: str) -> tuple[int, int, int]:
    try:
        batch, seqlen, heads = (int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "config must be batch,seqlen,heads"
        ) from exc
    if batch < 1 or heads < 1 or seqlen % 256:
        raise argparse.ArgumentTypeError(
            "batch/heads must be positive and seqlen divisible by 256"
        )
    return batch, seqlen, heads


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qk-format",
        choices=("nvfp4", "mxfp4", "both"),
        default="both",
    )
    parser.add_argument(
        "--config",
        action="append",
        type=parse_config,
        help="Repeat batch,seqlen,heads; defaults to HAO's nine D128 shapes",
    )
    parser.add_argument("--warmup-ms", type=int, default=10)
    parser.add_argument("--rep-ms", type=int, default=100)
    parser.add_argument("--cooldown-seconds", type=float, default=0.8)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--keep-builds", action="store_true")
    return parser.parse_args()


def extract_result(stdout: str) -> dict[str, Any]:
    start = stdout.rfind("\n{")
    if start < 0:
        start = stdout.find("{")
    else:
        start += 1
    if start < 0:
        raise RuntimeError("benchmark produced no JSON result")
    return json.loads(stdout[start:])


def run_checked(
    command: list[str],
    *,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=HERE,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )


def main() -> None:
    args = parse_args()
    configs = tuple(args.config or DEFAULT_CONFIGS)
    formats = (
        ("nvfp4", "mxfp4")
        if args.qk_format == "both"
        else (args.qk_format,)
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or (
        HERE.parents[1]
        / "results"
        / f"hao_direct_suite_{timestamp}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    report: dict[str, Any] = {
        "schema": "tk_hao_direct_hao_suite_v1",
        "created_utc": timestamp,
        "source_suite": "HAO bench_fp4 default D128 matrix",
        "pv_format": "fp8_e4m3",
        "results": [],
    }
    for qk_format in formats:
        module = MODULES[qk_format]
        qk_mode = 0 if qk_format == "nvfp4" else 1
        for batch, seqlen, heads in configs:
            label = f"{qk_format}_b{batch}s{seqlen}h{heads}d128"
            extension = Path("/tmp") / f"{module}_{label}.so"
            flags = " ".join(
                (
                    f"-DTK_HAO_DIRECT_QK_SCALE_MODE={qk_mode}",
                    f"-DTK_HAO_DIRECT_BATCH={batch}",
                    f"-DTK_HAO_DIRECT_SEQ_LEN={seqlen}",
                    f"-DTK_HAO_DIRECT_HEADS={heads}",
                )
            )
            print(f"[build] {label}", flush=True)
            build = run_checked(
                [
                    "make",
                    "-B",
                    "-f",
                    "Makefile.hao_direct",
                    "-j1",
                    f"OUT={extension}",
                    f"MODULE={module}",
                    f"EXTRA_NVCCFLAGS={flags}",
                ],
                env=env,
            )
            if args.cooldown_seconds > 0:
                time.sleep(args.cooldown_seconds)
            print(f"[bench] {label}", flush=True)
            try:
                bench = run_checked(
                    [
                        sys.executable,
                        "hao_direct_benchmark.py",
                        "--extension",
                        str(extension),
                        "--qk-format",
                        qk_format,
                        "--batch",
                        str(batch),
                        "--seqlen",
                        str(seqlen),
                        "--heads",
                        str(heads),
                        "--warmup-ms",
                        str(args.warmup_ms),
                        "--rep-ms",
                        str(args.rep_ms),
                    ],
                    env=env,
                )
                result = extract_result(bench.stdout)
                result["label"] = label
                build_log = build.stdout + build.stderr
                spill_warning = re.search(
                    r"Registers are spilled to local memory.*?"
                    r"(\d+) bytes spill stores, "
                    r"(\d+) bytes spill loads",
                    build_log,
                    re.DOTALL,
                )
                result["build_zero_spill"] = spill_warning is None
                if spill_warning is not None:
                    result["build_spill_store_bytes"] = int(
                        spill_warning.group(1)
                    )
                    result["build_spill_load_bytes"] = int(
                        spill_warning.group(2)
                    )
                report["results"].append(result)
                output.write_text(
                    json.dumps(report, indent=2, sort_keys=True) + "\n"
                )
                tk_key = f"tk_hao_direct_{qk_format}_fp8pv"
                print(
                    f"[done] {label}: "
                    f"{result['timing_ms'][tk_key]:.6f} ms, "
                    f"{result['tflops'][tk_key]:.1f} TFLOPS",
                    flush=True,
                )
            except subprocess.CalledProcessError as exc:
                failure = {
                    "label": label,
                    "returncode": exc.returncode,
                    "stdout": exc.stdout[-12000:],
                    "stderr": exc.stderr[-12000:],
                }
                report.setdefault("failures", []).append(failure)
                output.write_text(
                    json.dumps(report, indent=2, sort_keys=True) + "\n"
                )
                raise
            finally:
                if not args.keep_builds:
                    extension.unlink(missing_ok=True)

    print(f"[report] {output}", flush=True)


if __name__ == "__main__":
    main()
