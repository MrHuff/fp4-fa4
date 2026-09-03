#!/usr/bin/env python3
"""Build the causal-GQA FP8-P/V forward port without editing forward source."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sysconfig
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
FORWARD_SOURCE = REPO_ROOT / "tk_fa4" / "fp4_fa4_fwd"
FORWARD_PATCH = HERE / "causal_gqa_fp8pv_forward.patch"
PROBABILITY_POLICIES = {
    "exact": (0, 0xF, "1.62330034f", "0.92083546f"),
    "cubic": (3, 0xF, "1.62330034f", "0.92083546f"),
    "taylor": (1, 0x0, "0.69314718f", "1.0f"),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--sequence", type=int, default=4096)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, choices=(64, 128), default=128)
    parser.add_argument("--gpu", choices=("B200", "B300"), default="B200")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--nvcc-threads", type=int, default=4)
    parser.add_argument("--nvcc-split-compile", type=int, default=4)
    parser.add_argument(
        "--probability-policy",
        choices=tuple(PROBABILITY_POLICIES),
        default="exact",
        help=(
            "exact is the numerical default; cubic is the wider-range "
            "approximation; taylor is the fast undersaturated policy"
        ),
    )
    parser.add_argument(
        "--fixed-p-ceiling",
        action="store_true",
        help="diagnostic only: replace every probability by one",
    )
    parser.add_argument(
        "--score-pack-ceiling",
        action="store_true",
        help="diagnostic only: pack raw scores without softmax",
    )
    parser.add_argument("--module")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--keep-workdir", action="store_true")
    args = parser.parse_args()
    if args.batch <= 0:
        parser.error("--batch must be positive")
    if args.sequence % 256:
        parser.error("--sequence must be divisible by 256")
    if args.q_heads <= 0 or args.q_heads % args.kv_heads:
        parser.error("--q-heads must be divisible by --kv-heads")
    if args.fixed_p_ceiling and args.score_pack_ceiling:
        parser.error("the two ceiling diagnostics are mutually exclusive")
    return args


def main() -> None:
    args = _parse_args()
    mode, cubic_mask, affine_a, affine_b = PROBABILITY_POLICIES[
        args.probability_policy
    ]
    module = args.module or (
        "_C_tk_causal_gqa_nvfp4_fp8pv_"
        f"{args.probability_policy}_b{args.batch}s{args.sequence}h{args.q_heads}"
        f"kv{args.kv_heads}d{args.head_dim}"
    )
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not suffix:
        raise RuntimeError("Python extension suffix is unavailable")
    output = args.output or Path("/tmp") / f"{module}{suffix}"
    output = output.resolve()
    workdir = Path(
        tempfile.mkdtemp(
            prefix=".causal_gqa_fp8pv_build_",
            dir=FORWARD_SOURCE.parent,
        )
    )
    try:
        shutil.rmtree(workdir)
        shutil.copytree(FORWARD_SOURCE, workdir)
        subprocess.run(
            [
                "patch",
                "--batch",
                "--silent",
                "--input",
                str(FORWARD_PATCH),
            ],
            cwd=workdir,
            check=True,
        )
        command = [
            "make",
            "-f",
            "Makefile.hao_direct",
            f"GPU={args.gpu}",
            f"HAO_BATCH={args.batch}",
            f"HAO_SEQ_LEN={args.sequence}",
            f"HAO_HEADS={args.q_heads}",
            f"HAO_KV_HEADS={args.kv_heads}",
            f"HAO_HEAD_DIM={args.head_dim}",
            "HAO_CAUSAL=1",
            "HAO_FIXED_ROUTE_FASTPATH=1",
            f"HAO_FP8PV_SHIFTLESS_MODE={mode}",
            f"HAO_FP8PV_CUBIC_QUARTER_MASK={cubic_mask}",
            f"HAO_FP8PV_AFFINE_A={affine_a}",
            f"HAO_FP8PV_AFFINE_B={affine_b}",
            f"HAO_FP8PV_FIXED_P_CEILING={int(args.fixed_p_ceiling)}",
            f"HAO_FP8PV_SCORE_PACK_CEILING={int(args.score_pack_ceiling)}",
            f"OUT={output}",
            f"MODULE={module}",
            f"NVCC_THREADS={args.nvcc_threads}",
            f"NVCC_SPLIT_COMPILE={args.nvcc_split_compile}",
            f"-j{args.jobs}",
        ]
        subprocess.run(command, cwd=workdir, check=True)
        print(output)
    finally:
        if args.keep_workdir:
            print(f"kept build directory: {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
