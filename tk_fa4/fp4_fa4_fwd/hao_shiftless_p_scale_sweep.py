#!/usr/bin/env python3
"""Sweep exponent-rebased NVFP4 P-scale lifts on the fast shiftless policy."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from hao_p_scale_sweep import benchmark_case, c_float_literal, run


REPO_ROOT = HERE.parents[1]
HAO_ROOT = Path(
    os.environ.get(
        "HAO_FLASH_ATTN_ROOT",
        REPO_ROOT / "third_party/hao_flash_attention_fp4",
    )
).resolve()
QK_SCALE_MODE = {"nvfp4": 0, "mxfp4": 1}

# Cover the useful rebased range densely and retain high-exponent controls.
# G=1.25, 1.625, and 1.75 are low-exponent representatives of the stabilized
# optima at 320, 416, and 448, respectively.
DEFAULT_G = (
    1.0,
    1.125,
    1.25,
    1.375,
    1.40625,
    1.4375,
    1.46875,
    1.5,
    1.53125,
    1.5625,
    1.59375,
    1.625,
    1.75,
    1.875,
    2.0,
    2.25,
    2.5,
    2.75,
    3.0,
    3.25,
    3.5,
    3.75,
    4.0,
    4.25,
    4.5,
    4.75,
    5.0,
    5.5,
    6.0,
    6.5,
    7.0,
    8.0,
    16.0,
    32.0,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seqlen", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument(
        "--qk-format",
        action="append",
        choices=tuple(QK_SCALE_MODE),
        help="Repeat to select formats; defaults to NVFP4.",
    )
    parser.add_argument(
        "--g",
        action="append",
        type=float,
        help="Repeat to override the default rebased scale grid.",
    )
    parser.add_argument("--warmup-ms", type=int, default=10)
    parser.add_argument("--rep-ms", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", default="0")
    parser.add_argument(
        "--build-root",
        type=Path,
        default=Path("/tmp/tk_hao_shiftless_p_scale_sweep"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument(
        "--skip-normalization",
        action="store_true",
        help="Write per-case records only; useful for bounded build batches.",
    )
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def factor_label(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def mantissa_phase(value: float) -> float:
    return value / (2.0 ** math.floor(math.log2(value)))


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
        "HAO_FP4PV_NV_SHIFTLESS_SOFTMAX=1",
        "HAO_FP4PV_NV_SHIFTLESS_SCALE_LIFT=1",
        "HAO_FP4PV_NV_SCALE_SATFINITE=1",
        "HAO_FP4PV_NV_P_GLOBAL_LOG2="
        f"{c_float_literal(math.log2(g))}",
        "HAO_FP4PV_NV_PWL_EXP2=9",
        "HAO_FP4PV_NV_STAGE0_AFFINE_MASK=14",
        "HAO_FP4PV_NV_STAGE1_AFFINE_MASK=14",
        "HAO_FP4PV_NV_QUARTER_SCALE=1",
        "HAO_FP4PV_NV_SCALE_ENCODE=2",
        "HAO_FP4PV_NV_QUARTER_SCHEDULE=1",
        "HAO_FP4PV_NV_EARLY_DIRECT_SCALE=1",
        "HAO_FP4PV_MX_DELAYED_HALF_Q2=1",
        "HAO_FP4PV_MX_DELAYED_EARLY_Q3=1",
        "HAO_FP4PV_MX_EARLY_Q2_REDUCE=1",
        "HAO_FP4PV_MX_SHIFTLESS_CORR_BYPASS=1",
        "HAO_FP4PV_EARLY_P=1",
        "NVCC_SPLIT_COMPILE=1",
    ]
    completed = run(command, env=env)
    return completed.stdout + completed.stderr


def compact_summary(output: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for result in output["results"]:
        correctness = result["benchmark"]["correctness_global"]
        rows.append(
            {
                "qk_format": result["qk_format"],
                "pv_format": "nvfp4",
                "G": result["G"],
                "log2_G": result["log2_G"],
                "mantissa_phase": result["mantissa_phase"],
                "timing_ms": result["benchmark"]["timing_ms"],
                "cosine_vs_bf16": correctness["cosine"],
                "relative_l2_vs_bf16": correctness["relative_l2"],
                "rmse_vs_bf16": correctness["rmse"],
                "finite": all(
                    math.isfinite(correctness[key])
                    for key in ("cosine", "relative_l2", "rmse")
                ),
            }
        )

    best_by_qk = {}
    for qk_format in sorted({row["qk_format"] for row in rows}):
        finite_rows = [
            row
            for row in rows
            if row["qk_format"] == qk_format and row["finite"]
        ]
        best_by_qk[qk_format] = {
            "best_cosine": max(
                finite_rows, key=lambda row: row["cosine_vs_bf16"]
            ),
            "best_relative_l2": min(
                finite_rows, key=lambda row: row["relative_l2_vs_bf16"]
            ),
        }

    normalization = output["hao_normalization"]
    hao_correctness = normalization["correctness"]["hao_vs_bf16_output"]
    return {
        "schema": "tk_hao_shiftless_p_scale_sweep_summary_v1",
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
    qk_formats = tuple(args.qk_format or ("nvfp4",))
    factors = tuple(args.g or DEFAULT_G)
    if any(not math.isfinite(g) or g < 0.125 or g > 2688.0 for g in factors):
        raise SystemExit("every experimental G must be in [0.125, 2688]")

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
    first_case: tuple[str, Path, str] | None = None

    for qk_format in qk_formats:
        for g in factors:
            label = (
                f"{qk_format}_nvfp4_shiftless_g{factor_label(g)}"
                f"_b{args.batch}s{args.seqlen}h{args.heads}"
            )
            module = f"_C_tk_hao_{label}"
            extension = args.build_root / f"{label}.so"
            case_path = cases_dir / f"{label}.json"
            if first_case is None:
                first_case = (qk_format, extension, module)
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
                "mantissa_phase": mantissa_phase(g),
                "benchmark": benchmark,
            }
            case_path.write_text(
                json.dumps(case, indent=2, sort_keys=True) + "\n"
            )
            results.append(case)

    if args.skip_normalization:
        print(
            f"[done] {len(results)} per-case records written; "
            "normalization skipped",
            flush=True,
        )
        return

    assert first_case is not None
    qk_format, extension, module = first_case
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
        "schema": "tk_hao_shiftless_p_scale_sweep_v1",
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
            "policy": "fast shiftless mode 9, affine masks 14/14",
            "reference": "torch BF16 SDPA; HAO BF16 cross-check",
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
