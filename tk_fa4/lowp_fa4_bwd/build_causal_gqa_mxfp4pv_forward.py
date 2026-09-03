#!/usr/bin/env python3
"""Build the authenticated causal-GQA D64 NVFP4-QK/MXFP4-PV route."""

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
MX_POLICIES = {
    # Both policies retain native density four. d4q01 spends native EX2 only
    # on Q0/Q1 and is the reviewed saturated-training candidate.
    "d4q01": 0x3,
    "d4all": 0xF,
}
MX_VARIANTS = {
    "anchored": (),
    # This is the historically measured splitmix-v6 topology.  Keep every
    # late override explicit: relying on the causal-accurate policy defaults
    # would silently turn this experiment back into the anchor32 candidate.
    "unanchored-splitmix-v6": (
        "HAO_FP4PV_MX_GLOBAL_ANCHOR32_OVERRIDE=0",
        "HAO_FP4PV_MX_GLOBAL_ANCHOR128_OVERRIDE=0",
        "HAO_FP4PV_MX_GLOBAL_ANCHOR_MARGIN_LOG2_OVERRIDE=0",
        "HAO_FP4PV_MX_ANCHOR_AFFINE_HOIST_OVERRIDE=0",
        "HAO_FP4PV_MX_STORED_SCALE_SHIFT_LOG2_OVERRIDE=16",
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--sequence", type=int, default=4096)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, choices=(64,), default=64)
    parser.add_argument("--gpu", choices=("B200",), default="B200")
    parser.add_argument("--num-sm", type=int, default=152)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--nvcc-threads", type=int, default=1)
    parser.add_argument("--nvcc-split-compile", type=int, default=1)
    parser.add_argument(
        "--mx-policy",
        choices=tuple(MX_POLICIES),
        default="d4q01",
    )
    parser.add_argument(
        "--variant",
        choices=tuple(MX_VARIANTS),
        default="anchored",
        help=(
            "Select the reviewed anchor32 route or the separately named "
            "historical unanchored splitmix-v6 experiment"
        ),
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
    if args.num_sm <= 0:
        parser.error("--num-sm must be positive")
    return args


def main() -> None:
    args = _parse_args()
    quarter_mask = MX_POLICIES[args.mx_policy]
    shape = (
        f"b{args.batch}s{args.sequence}h{args.q_heads}"
        f"kv{args.kv_heads}d{args.head_dim}"
    )
    variant_tag = args.variant.replace("-", "_")
    default_variant_tag = "" if args.variant == "anchored" else f"_{variant_tag}"
    module = args.module or (
        f"_C_cfwd_mx_{args.mx_policy}{default_variant_tag}_{shape}"
    )
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not suffix:
        raise RuntimeError("Python extension suffix is unavailable")
    output = (args.output or Path("/tmp") / f"{module}{suffix}").resolve()
    symbol_tag = (
        f"cfwd_mx_{args.mx_policy}{default_variant_tag}_{shape}"
    )
    workdir = Path(
        tempfile.mkdtemp(
            prefix=".causal_gqa_mxfp4pv_build_",
            dir=FORWARD_SOURCE.parent,
        )
    )
    try:
        shutil.rmtree(workdir)
        shutil.copytree(FORWARD_SOURCE, workdir)
        command = [
            "make",
            "-B",
            "-f",
            "Makefile.hao_direct_fp4pv",
            f"-j{args.jobs}",
            f"GPU={args.gpu}",
            f"HAO_BATCH={args.batch}",
            f"HAO_SEQ_LEN={args.sequence}",
            f"HAO_HEADS={args.q_heads}",
            f"HAO_KV_HEADS={args.kv_heads}",
            f"HAO_HEAD_DIM={args.head_dim}",
            f"HAO_NUM_SM={args.num_sm}",
            "HAO_CAUSAL=1",
            "HAO_FIXED_ROUTE_FASTPATH=1",
            "HAO_CAUSAL_INTERLEAVED_KV=1",
            "HAO_FP4PV_MX_POLICY=causal-accurate",
            "HAO_FP4PV_MX_MODE23_NATIVE_DENSITY_OVERRIDE=4",
            (
                "HAO_FP4PV_MX_MODE23_NATIVE_QUARTER_MASK_OVERRIDE="
                f"{quarter_mask}"
            ),
            *MX_VARIANTS[args.variant],
            "HAO_EXTENSION_SYMBOLIC_BIND=1",
            f"HAO_KERNEL_SYMBOL_TAG={symbol_tag}",
            f"MODULE={module}",
            f"OUT={output}",
            f"NVCC_THREADS={args.nvcc_threads}",
            f"NVCC_SPLIT_COMPILE={args.nvcc_split_compile}",
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
