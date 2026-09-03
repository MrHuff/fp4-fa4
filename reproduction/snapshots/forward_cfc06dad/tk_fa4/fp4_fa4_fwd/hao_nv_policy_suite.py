#!/usr/bin/env python3
"""Benchmark the named NVFP4 P-construction accuracy policies."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
HAO_ROOT = Path("/workspace/codebases/flash-attention-fp4")
POLICY_IDS = {
    "fast": 1,
    "balanced": 2,
    "accurate": 3,
    "exact": 4,
    "hao-cosine": 5,
    "hao-l2": 6,
    "universal": 7,
}
DEFAULT_POLICIES = tuple(POLICY_IDS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seqlen", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument(
        "--policy",
        action="append",
        choices=tuple(POLICY_IDS),
        dest="policies",
        help="Policy to run; repeat to select multiple policies.",
    )
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        dest="seeds",
        help="Input seed; repeat to select multiple seeds.",
    )
    parser.add_argument("--warmup-ms", type=int, default=20)
    parser.add_argument("--rep-ms", type=int, default=100)
    parser.add_argument(
        "--build-root",
        type=Path,
        default=Path("/tmp/tk_hao_nv_policy"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument(
        "--reuse-results",
        action="store_true",
        help="Reuse existing per-policy seed JSON while refreshing summary.",
    )
    parser.add_argument("--skip-hao", action="store_true")
    return parser.parse_args()


def environment() -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    entries = [str(HAO_ROOT)]
    if pythonpath:
        entries.append(pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(entries)
    return env


def run(
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


def parse_json_output(text: str) -> dict[str, Any]:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.strip() != "{":
            continue
        candidate = "\n".join(lines[index:])
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError("command output did not end in a JSON object")


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_policy(
    *,
    args: argparse.Namespace,
    policy: str,
    env: dict[str, str],
) -> tuple[Path, str, str]:
    slug = policy.replace("-", "_")
    module = (
        f"_C_tk_nv_policy_{slug}_"
        f"b{args.batch}s{args.seqlen}h{args.heads}"
    )
    extension = args.build_root / f"{module}.so"
    if extension.exists() and not args.rebuild:
        return extension, module, "reused existing extension\n"

    args.build_root.mkdir(parents=True, exist_ok=True)
    command = [
        "make",
        "-B",
        "-f",
        "Makefile.hao_direct_fp4pv",
        "-j1",
        f"OUT={extension}",
        f"MODULE={module}",
        f"HAO_BATCH={args.batch}",
        f"HAO_SEQ_LEN={args.seqlen}",
        f"HAO_HEADS={args.heads}",
        "HAO_QK_SCALE_MODE=0",
        "HAO_PV_SCALE_MODE=0",
        f"HAO_FP4PV_NV_POLICY={policy}",
        "NVCC_SPLIT_COMPILE=1",
    ]
    completed = run(command, env=env)
    return extension, module, completed.stdout + completed.stderr


def benchmark(
    *,
    args: argparse.Namespace,
    extension: Path,
    module: str,
    seed: int,
    env: dict[str, str],
    tk_only: bool,
) -> tuple[dict[str, Any], str]:
    command = [
        sys.executable,
        "hao_direct_fp4pv_benchmark.py",
        "--extension",
        str(extension),
        "--extension-module",
        module,
        "--qk-format",
        "nvfp4",
        "--pv-format",
        "nvfp4",
        "--warmup-ms",
        str(args.warmup_ms),
        "--rep-ms",
        str(args.rep_ms),
        "--seed",
        str(seed),
        "--summary-only",
    ]
    if tk_only:
        command.append("--tk-only")
    completed = run(command, env=env)
    combined = completed.stdout + completed.stderr
    return parse_json_output(completed.stdout), combined


def mean_metric(records: list[dict[str, Any]], name: str) -> float:
    return statistics.fmean(
        record["correctness_global"][name] for record in records
    )


def summarize_policy(
    policy: str,
    records: list[dict[str, Any]],
    hao_reference: dict[str, Any] | None,
) -> dict[str, Any]:
    timings = [record["timing_ms"] for record in records]
    result = {
        "policy": policy,
        "policy_id": POLICY_IDS[policy],
        "seeds": [record["protocol"]["seed"] for record in records],
        "timing_ms": {
            "mean": statistics.fmean(timings),
            "min": min(timings),
            "max": max(timings),
        },
        "vs_bf16": {
            name: mean_metric(records, name)
            for name in (
                "cosine",
                "relative_l2",
                "rmse",
                "mean_abs",
                "max_abs",
            )
        },
        "topology": records[0]["topology"],
    }
    if hao_reference is None:
        return result

    hao_timing = hao_reference["timing_ms"][
        "hao_native_nvfp4_nvfp4pv"
    ]
    bf16_timing = hao_reference["timing_ms"]["hao_native_bf16"]
    hao_error = hao_reference["vs_bf16"]
    result["speedup"] = {
        "vs_hao_native_nvfp4": hao_timing / result["timing_ms"]["mean"],
        "vs_hao_native_bf16": bf16_timing / result["timing_ms"]["mean"],
    }
    result["error_delta_vs_hao"] = {
        "cosine": result["vs_bf16"]["cosine"] - hao_error["cosine"],
        "relative_l2": (
            result["vs_bf16"]["relative_l2"]
            - hao_error["relative_l2"]
        ),
        "rmse": result["vs_bf16"]["rmse"] - hao_error["rmse"],
    }
    return result


def main() -> None:
    args = parse_args()
    policies = args.policies or list(DEFAULT_POLICIES)
    seeds = args.seeds or [0, 1, 2, 3]
    if args.batch < 1 or args.heads < 1 or args.seqlen % 256:
        raise ValueError("batch/heads must be positive and S divisible by 256")
    if args.warmup_ms < 1 or args.rep_ms < 1:
        raise ValueError("benchmark durations must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = environment()
    built: dict[str, tuple[Path, str]] = {}
    raw: dict[str, list[dict[str, Any]]] = {}

    for policy in policies:
        extension, module, build_log = build_policy(
            args=args,
            policy=policy,
            env=env,
        )
        build_log_path = args.output_dir / f"build_{policy}.log"
        reused_extension = build_log == "reused existing extension\n"
        if not reused_extension or not build_log_path.exists():
            build_log_path.write_text(build_log, encoding="utf-8")
        built[policy] = (extension, module)
        raw[policy] = []
        for seed in seeds:
            result_path = args.output_dir / f"{policy}_seed{seed}.json"
            if args.reuse_results and result_path.exists():
                result = json.loads(result_path.read_text(encoding="utf-8"))
                actual_id = result["topology"].get("nv_policy_id")
                if actual_id != POLICY_IDS[policy]:
                    raise RuntimeError(
                        f"{policy} emitted policy ID {actual_id}, "
                        f"expected {POLICY_IDS[policy]}"
                    )
                raw[policy].append(result)
                continue
            result, benchmark_log = benchmark(
                args=args,
                extension=extension,
                module=module,
                seed=seed,
                env=env,
                tk_only=True,
            )
            actual_id = result["topology"].get("nv_policy_id")
            if actual_id != POLICY_IDS[policy]:
                raise RuntimeError(
                    f"{policy} emitted policy ID {actual_id}, "
                    f"expected {POLICY_IDS[policy]}"
                )
            raw[policy].append(result)
            write_json(result_path, result)
            (args.output_dir / f"{policy}_seed{seed}.log").write_text(
                benchmark_log,
                encoding="utf-8",
            )

    hao_reference = None
    if not args.skip_hao:
        reference_policy = "fast" if "fast" in built else policies[0]
        extension, module = built[reference_policy]
        hao_records = []
        for seed in seeds:
            hao_record, hao_log = benchmark(
                args=args,
                extension=extension,
                module=module,
                seed=seed,
                env=env,
                tk_only=False,
            )
            hao_records.append(hao_record)
            write_json(
                args.output_dir / f"hao_reference_seed{seed}.json",
                hao_record,
            )
            (args.output_dir / f"hao_reference_seed{seed}.log").write_text(
                hao_log,
                encoding="utf-8",
            )
        timing_names = (
            "hao_native_bf16",
            "hao_native_nvfp4_nvfp4pv",
        )
        error_names = (
            "cosine",
            "relative_l2",
            "rmse",
            "mean_abs",
            "max_abs",
        )
        hao_reference = {
            "seeds": seeds,
            "timing_ms": {
                name: statistics.fmean(
                    record["timing_ms"][name]
                    for record in hao_records
                )
                for name in timing_names
            },
            "vs_bf16": {
                name: statistics.fmean(
                    record["correctness"]["hao_vs_bf16_output"][name]
                    for record in hao_records
                )
                for name in error_names
            },
        }
        write_json(args.output_dir / "hao_reference.json", hao_reference)

    summary = {
        "schema": "tk_hao_nv_policy_suite_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "shape": {
            "batch": args.batch,
            "seqlen": args.seqlen,
            "heads": args.heads,
            "dim": 128,
        },
        "protocol": {
            "warmup_ms": args.warmup_ms,
            "rep_ms": args.rep_ms,
            "seeds": seeds,
            "qk_format": "nvfp4",
            "pv_format": "nvfp4",
        },
        "hao_reference": hao_reference,
        "policies": [
            summarize_policy(policy, raw[policy], hao_reference)
            for policy in policies
        ],
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
