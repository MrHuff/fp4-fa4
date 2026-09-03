#!/usr/bin/env python3
"""Sweep the stabilized NVFP4 probability-scale lift on one HAO input."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
HAO_ROOT = Path("/workspace/codebases/flash-attention-fp4")
DEFAULT_G = (
    1.0,
    2.0,
    4.0,
    8.0,
    16.0,
    32.0,
    64.0,
    128.0,
    256.0,
    288.0,
    320.0,
    352.0,
    384.0,
    416.0,
    448.0,
    480.0,
    512.0,
    1024.0,
    2048.0,
    2688.0,
)
QK_SCALE_MODE = {"nvfp4": 0, "mxfp4": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seqlen", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument(
        "--qk-format",
        action="append",
        choices=tuple(QK_SCALE_MODE),
        help="Repeat to select formats; defaults to both NVFP4 and MXFP4.",
    )
    parser.add_argument(
        "--g",
        action="append",
        type=float,
        help="Repeat to override the default scale-lift grid.",
    )
    parser.add_argument("--warmup-ms", type=int, default=10)
    parser.add_argument("--rep-ms", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", default="0")
    parser.add_argument(
        "--build-root",
        type=Path,
        default=Path("/tmp/tk_hao_p_scale_sweep"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild and rerun cases already present in the output directory.",
    )
    return parser.parse_args()


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


def extract_json(stdout: str) -> dict[str, Any]:
    start = stdout.rfind("\n{")
    if start >= 0:
        start += 1
    else:
        start = stdout.find("{")
    if start < 0:
        raise RuntimeError("benchmark produced no JSON")
    return json.loads(stdout[start:])


def factor_label(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def c_float_literal(value: float) -> str:
    text = f"{value:.17g}"
    if "." not in text and "e" not in text:
        text += ".0"
    return text + "f"


def build_case(
    *,
    args: argparse.Namespace,
    qk_format: str,
    g: float,
    extension: Path,
    module: str,
    env: dict[str, str],
) -> str:
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
        f"HAO_QK_SCALE_MODE={QK_SCALE_MODE[qk_format]}",
        "HAO_PV_SCALE_MODE=0",
        "HAO_FP4PV_NV_SHIFTLESS_SOFTMAX=0",
        "HAO_FP4PV_NV_PWL_EXP2=0",
        "HAO_FP4PV_NV_QUARTER_SCALE=0",
        "HAO_FP4PV_NV_QUARTER_SCHEDULE=0",
        "HAO_FP4PV_NV_EARLY_DIRECT_SCALE=0",
        "HAO_FP4PV_EARLY_P=0",
        "HAO_FP4PV_NV_P_GLOBAL_LOG2="
        f"{c_float_literal(math.log2(g))}",
        "NVCC_SPLIT_COMPILE=1",
    ]
    completed = run(command, env=env)
    return completed.stdout + completed.stderr


def benchmark_case(
    *,
    args: argparse.Namespace,
    qk_format: str,
    extension: Path,
    module: str,
    env: dict[str, str],
    tk_only: bool,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "hao_direct_fp4pv_benchmark.py",
        "--extension",
        str(extension),
        "--extension-module",
        module,
        "--qk-format",
        qk_format,
        "--pv-format",
        "nvfp4",
        "--warmup-ms",
        str(args.warmup_ms),
        "--rep-ms",
        str(args.rep_ms),
        "--seed",
        str(args.seed),
        "--summary-only",
    ]
    if tk_only:
        command.append("--tk-only")
    return extract_json(run(command, env=env).stdout)


def compact_summary(output: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for result in output["results"]:
        correctness = result["benchmark"]["correctness_global"]
        rows.append(
            {
                "qk_format": result["qk_format"],
                "pv_format": result["pv_format"],
                "G": result["G"],
                "log2_G": result["log2_G"],
                "timing_ms": result["benchmark"]["timing_ms"],
                "cosine_vs_bf16": correctness["cosine"],
                "relative_l2_vs_bf16": correctness["relative_l2"],
                "rmse_vs_bf16": correctness["rmse"],
            }
        )

    best_by_qk = {}
    for qk_format in sorted({row["qk_format"] for row in rows}):
        format_rows = [row for row in rows if row["qk_format"] == qk_format]
        best_by_qk[qk_format] = {
            "best_cosine": max(
                format_rows, key=lambda row: row["cosine_vs_bf16"]
            ),
            "best_relative_l2": min(
                format_rows, key=lambda row: row["relative_l2_vs_bf16"]
            ),
        }

    normalization = output["hao_normalization"]
    hao_correctness = normalization["correctness"]["hao_vs_bf16_output"]
    return {
        "schema": "tk_hao_p_scale_sweep_summary_v1",
        "created_utc": output["created_utc"],
        "shape": output["shape"],
        "protocol": output["protocol"],
        "factors": output["factors"],
        "hao_native_nvfp4_nvfp4": {
            "timing_ms": normalization["timing_ms"][
                "hao_native_nvfp4_nvfp4pv"
            ],
            "bf16_timing_ms": normalization["timing_ms"]["hao_native_bf16"],
            "cosine_vs_bf16": hao_correctness["cosine"],
            "relative_l2_vs_bf16": hao_correctness["relative_l2"],
            "rmse_vs_bf16": hao_correctness["rmse"],
        },
        "best_by_qk": best_by_qk,
        "results": rows,
    }


def main() -> None:
    args = parse_args()
    if args.batch < 1 or args.heads < 1 or args.seqlen % 256:
        raise SystemExit("batch/heads must be positive and seqlen divisible by 256")
    qk_formats = tuple(args.qk_format or QK_SCALE_MODE)
    factors = tuple(args.g or DEFAULT_G)
    if any(g < 1.0 or g > 2688.0 for g in factors):
        raise SystemExit("every G must be in the safe stabilized range [1, 2688]")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases_dir = args.output_dir / "cases"
    logs_dir = args.output_dir / "build_logs"
    cases_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    args.build_root.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": args.gpu,
            "PYTHONPATH": str(HAO_ROOT),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    results: list[dict[str, Any]] = []
    first_case: tuple[str, float, Path, str] | None = None

    for qk_format in qk_formats:
        for g in factors:
            label = f"{qk_format}_nvfp4_g{factor_label(g)}"
            module = f"_C_tk_hao_p_scale_{label}"
            extension = args.build_root / f"{label}.so"
            case_path = cases_dir / f"{label}.json"
            if first_case is None:
                first_case = (qk_format, g, extension, module)
            if case_path.exists() and extension.exists() and not args.rebuild:
                print(f"[skip] {label}", flush=True)
                results.append(json.loads(case_path.read_text()))
                continue
            print(f"[case] {label}", flush=True)
            if not args.skip_build:
                build_log = build_case(
                    args=args,
                    qk_format=qk_format,
                    g=g,
                    extension=extension,
                    module=module,
                    env=env,
                )
                (logs_dir / f"{label}.log").write_text(build_log)
            benchmark = benchmark_case(
                args=args,
                qk_format=qk_format,
                extension=extension,
                module=module,
                env=env,
                tk_only=True,
            )
            case = {
                "qk_format": qk_format,
                "pv_format": "nvfp4",
                "G": g,
                "log2_G": math.log2(g),
                "benchmark": benchmark,
            }
            case_path.write_text(
                json.dumps(case, indent=2, sort_keys=True) + "\n"
            )
            results.append(case)

    assert first_case is not None
    qk_format, g, extension, module = first_case
    print("[normalization] native HAO NV/NV versus HAO BF16", flush=True)
    normalized = benchmark_case(
        args=args,
        qk_format=qk_format,
        extension=extension,
        module=module,
        env=env,
        tk_only=False,
    )
    output = {
        "schema": "tk_hao_p_scale_sweep_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "shape": {
            "batch": args.batch,
            "seqlen": args.seqlen,
            "heads": args.heads,
            "dim": 128,
        },
        "protocol": {
            "seed": args.seed,
            "warmup_ms": args.warmup_ms,
            "rep_ms": args.rep_ms,
            "factory": "HAO create_nvfp4_attention_tensors",
            "stabilization": "exact online row maximum",
            "reference": "torch BF16 SDPA for sweep; HAO BF16 cross-check below",
        },
        "factors": list(factors),
        "results": results,
        "hao_normalization": normalized,
    }
    (args.output_dir / "sweep.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(compact_summary(output), indent=2, sort_keys=True) + "\n"
    )
    print(
        f"[done] {len(results)} cases written to "
        f"{args.output_dir / 'sweep.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
