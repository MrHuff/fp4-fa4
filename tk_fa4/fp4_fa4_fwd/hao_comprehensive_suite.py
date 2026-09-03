#!/usr/bin/env python3
"""Run TK and native HAO attention providers over HAO's published shape grid."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
HAO_ROOT = Path(
    os.environ.get(
        "HAO_FLASH_ATTN_ROOT",
        REPO_ROOT / "third_party/hao_flash_attention_fp4",
    )
).resolve()

HEADLINE_SHAPES = (
    (1, 256, 16, 128),
    (1, 1024, 16, 128),
    (4, 4096, 16, 128),
    (1, 32768, 16, 128),
    (4, 4096, 32, 128),
    (1, 4096, 12, 128),
    (1, 32768, 12, 128),
    (1, 4096, 24, 128),
    (1, 32768, 24, 128),
    (1, 32768, 24, 64),
)
SWEEP_SEQUENCES = (1024, 2048, 4096, 8192, 16384, 32768)
SWEEP_HEADS = (12, 24, 32)
D64_SWEEP_HEADS = (12, 24, 32, 64)

PURE_FP4_FLAGS = {
    "HAO_FP4PV_NV_SHIFTLESS_SOFTMAX": "1",
    "HAO_FP4PV_NV_PWL_EXP2": "9",
    "HAO_FP4PV_NV_STAGE0_AFFINE_MASK": "14",
    "HAO_FP4PV_NV_STAGE1_AFFINE_MASK": "14",
    "HAO_FP4PV_NV_QUARTER_SCALE": "1",
    "HAO_FP4PV_NV_SCALE_ENCODE": "2",
    "HAO_FP4PV_NV_QUARTER_SCHEDULE": "1",
    "HAO_FP4PV_NV_EARLY_DIRECT_SCALE": "1",
    "HAO_FP4PV_MX_DELAYED_HALF_Q2": "1",
    "HAO_FP4PV_MX_DELAYED_EARLY_Q3": "1",
    "HAO_FP4PV_MX_EARLY_Q2_REDUCE": "1",
    "HAO_FP4PV_MX_SHIFTLESS_CORR_BYPASS": "1",
    "HAO_FP4PV_EARLY_P": "1",
    "NVCC_SPLIT_COMPILE": "1",
}
MX_PV_FLAGS = {
    "HAO_PV_SCALE_MODE": "1",
    "HAO_FP4PV_MX_SHIFTLESS_SOFTMAX": "1",
    # Match the NV throughput policy's aggressive transform budget on both
    # query stages. Leaving either stage cubic creates a software-only tax.
    "HAO_FP4PV_MX_PWL_EXP2": "23",
    "HAO_FP4PV_MX_STAGE0_AFFINE_MASK": "14",
    "HAO_FP4PV_MX_STAGE1_AFFINE_MASK": "14",
    "HAO_FP4PV_MX_PAIR_LOAD_SCAN": "1",
    "HAO_FP4PV_MX_DELAYED_HALF_Q2": "1",
    "HAO_FP4PV_MX_DELAYED_EARLY_Q3": "1",
    "HAO_FP4PV_MX_EARLY_Q2_REDUCE": "1",
    "HAO_FP4PV_MX_EARLY_P": "1",
    "HAO_FP4PV_MX_EARLY_DIRECT_SCALE": "1",
    "HAO_FP4PV_MX_EARLY_ASYNC_SCALE": "1",
    "HAO_FP4PV_MX_SHIFTLESS_CORR_BYPASS": "1",
    "NVCC_SPLIT_COMPILE": "1",
}
FP4_FORMATS = {
    "nv-nv": ("nvfp4", "nvfp4"),
    "nv-nv-bounded": ("nvfp4", "nvfp4"),
    "mx-nv": ("mxfp4", "nvfp4"),
    "nv-mx": ("nvfp4", "mxfp4"),
    "mx-mx": ("mxfp4", "mxfp4"),
}
NVMX_PARETO_POLICIES = {
    "nvmx-fast": "fast",
    "nvmx-balanced": "balanced",
    "nvmx-accurate": "accurate",
}
NVMX_FOLDED_QK_VARIANTS = {"nvmx-fast", "nvmx-balanced"}
FP4_FLAGS = {
    "nv-nv": {
        **PURE_FP4_FLAGS,
        "HAO_QK_SCALE_MODE": "0",
        "HAO_PV_SCALE_MODE": "0",
    },
    "nv-nv-bounded": {
        **PURE_FP4_FLAGS,
        "HAO_QK_SCALE_MODE": "0",
        "HAO_PV_SCALE_MODE": "0",
        "HAO_FP4PV_NV_SCALE_SATFINITE": "1",
        "HAO_FP4PV_NV_SCALE_ENCODE": "4",
    },
    "mx-nv": {
        **PURE_FP4_FLAGS,
        "HAO_QK_SCALE_MODE": "1",
        "HAO_PV_SCALE_MODE": "0",
    },
    "nv-mx": {
        **MX_PV_FLAGS,
        "HAO_QK_SCALE_MODE": "0",
    },
    "mx-mx": {
        **MX_PV_FLAGS,
        "HAO_QK_SCALE_MODE": "1",
    },
}
FP8_FLAGS = {
    # Mode 4 uses the selected one-pass cubic approximation and sampled
    # denominator while retaining E4M3 P/V and K32 PV publication.
    "HAO_FP8PV_SHIFTLESS_MODE": "4",
}


def unique_shapes(values: Iterable[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    seen: set[tuple[int, int, int, int]] = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def suite_shapes(name: str) -> list[tuple[int, int, int, int]]:
    d128_sweeps = (
        (1, seqlen, heads, 128)
        for heads in SWEEP_HEADS
        for seqlen in SWEEP_SEQUENCES
    )
    d64_sweeps = (
        (1, seqlen, heads, 64)
        for heads in D64_SWEEP_HEADS
        for seqlen in SWEEP_SEQUENCES
    )
    if name == "headline":
        return list(HEADLINE_SHAPES)
    if name == "sweeps":
        return unique_shapes(d128_sweeps)
    if name == "d64-sweeps":
        return unique_shapes(d64_sweeps)
    return unique_shapes((*HEADLINE_SHAPES, *d128_sweeps))


def parse_shape(value: str) -> tuple[int, int, int, int]:
    try:
        shape = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape must be B,S,H,D") from exc
    if len(shape) != 4:
        raise argparse.ArgumentTypeError("shape must be B,S,H,D")
    batch, seqlen, heads, dim = shape
    if batch < 1 or heads < 1 or seqlen % 256 or dim not in (64, 128):
        raise argparse.ArgumentTypeError(
            "B/H must be positive, S divisible by 256, and D 64 or 128"
        )
    return batch, seqlen, heads, dim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape-set",
        choices=("headline", "sweeps", "d64-sweeps", "all"),
        default="all",
    )
    parser.add_argument(
        "--shape",
        action="append",
        type=parse_shape,
        help="Override the suite with one or more B,S,H,D shapes",
    )
    parser.add_argument(
        "--variant",
        choices=(
            "pure-fp4",
            "nv-nv",
            "nv-nv-bounded",
            "mx-nv",
            "nv-mx",
            "mx-mx",
            "fp4-matrix",
            "nvmx-fast",
            "nvmx-balanced",
            "nvmx-accurate",
            "nvmx-pareto",
            "fp8",
            "both",
            "all",
        ),
        default="both",
    )
    parser.add_argument("--warmup-ms", type=int, default=10)
    parser.add_argument("--rep-ms", type=int, default=25)
    parser.add_argument("--cooldown-seconds", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", default="0")
    parser.add_argument(
        "--skip-hao-fp4",
        action="store_true",
        help=(
            "Skip HAO's full-FP4 provider but retain HAO BF16 timing "
            "and TK-vs-BF16 correctness"
        ),
    )
    parser.add_argument(
        "--target-gpu",
        choices=("B200", "B300"),
        default="B200",
    )
    parser.add_argument("--num-sm", type=int)
    parser.add_argument("--kv-stages", type=int)
    parser.add_argument("--task-order", type=int, choices=(0, 1))
    parser.add_argument("--physical-grid-cap", type=int)
    register_choices = range(24, 257, 8)
    parser.add_argument(
        "--softmax-stage0-regs", type=int, choices=register_choices
    )
    parser.add_argument(
        "--softmax-stage1-regs", type=int, choices=register_choices
    )
    parser.add_argument("--correction-regs", type=int, choices=register_choices)
    parser.add_argument("--producer-regs", type=int, choices=register_choices)
    parser.add_argument("--sm103-ldred", type=int, choices=(0, 1))
    parser.add_argument(
        "--sm103-ldred-mask",
        type=lambda value: int(value, 0),
        help="Four-bit mask selecting fused Q0..Q3 score reductions",
    )
    parser.add_argument(
        "--sm103-ldred-stage0-mask",
        type=lambda value: int(value, 0),
        help="Stage-0 override for fused Q0..Q3 score reductions",
    )
    parser.add_argument(
        "--sm103-ldred-stage1-mask",
        type=lambda value: int(value, 0),
        help="Stage-1 override for fused Q0..Q3 score reductions",
    )
    parser.add_argument(
        "--nv-scale-satfinite",
        type=int,
        choices=(0, 1),
        help="Clamp NVFP4 P scales and denominator inputs to finite bounds",
    )
    parser.add_argument(
        "--nv-scale-encode",
        type=int,
        choices=range(6),
        help="NVFP4 P-scale encoding mode",
    )
    parser.add_argument(
        "--mx-native-density",
        type=int,
        choices=(0, 1, 2, 4),
        help="Mode-23 native exp2 pair density",
    )
    parser.add_argument(
        "--mx-native-quarter-mask",
        type=lambda value: int(value, 0),
        help="Four-bit mask selecting quarters with native exp2 pairs",
    )
    parser.add_argument(
        "--mx-self-stage0-native",
        type=int,
        choices=(0, 1),
        help="Allow native exp2 in self-max-owned stage 0",
    )
    parser.add_argument(
        "--mx-early-native",
        type=int,
        choices=(0, 1),
        help="Issue mode-23 native exp2 samples before the affine pairs",
    )
    parser.add_argument(
        "--mx-early-native-stage-mask",
        type=lambda value: int(value, 0),
        help="Two-bit mask selecting score stages for early native exp2",
    )
    parser.add_argument(
        "--mx-early-native-quarter-mask",
        type=lambda value: int(value, 0),
        help="Four-bit mask selecting quarters for early native exp2",
    )
    parser.add_argument(
        "--mx-early-native-lookahead",
        type=int,
        choices=(1, 2, 3, 4),
        help="Number of native sample pairs issued before affine work",
    )
    parser.add_argument(
        "--mx-early-native-order",
        type=int,
        choices=(0, 1, 2),
        help="Native sample issue order (orders 1/2 require lookahead 4)",
    )
    parser.add_argument(
        "--mx-stage0-affine-mask",
        type=lambda value: int(value, 0),
        help="Four-bit stage-0 affine approximation mask",
    )
    parser.add_argument(
        "--mx-stage1-affine-mask",
        type=lambda value: int(value, 0),
        help="Four-bit stage-1 affine approximation mask",
    )
    parser.add_argument(
        "--mx-full-approx-denom",
        type=int,
        choices=range(6),
        help="Mode-23 denominator from all produced approximation values",
    )
    parser.add_argument(
        "--mx-pair-scale-reuse",
        type=lambda value: int(value, 0),
        help="Two-bit override for Q0/Q1 and Q2/Q3 scale reuse",
    )
    parser.add_argument(
        "--mx-pair-scale-stage-mask",
        type=lambda value: int(value, 0),
        help="Two-bit override selecting stages with pair-scale reuse",
    )
    parser.add_argument(
        "--mx-q1-self-max",
        type=lambda value: int(value, 0),
        help="Two-bit override for stage-owned Q1 maximum reduction",
    )
    parser.add_argument(
        "--mx-q3-correction-wg",
        type=int,
        choices=(0, 1),
        help="Offload stage-0 Q3 construction to the correction warp group",
    )
    parser.add_argument(
        "--mx-dual-q3-correction-wg",
        type=int,
        choices=(0, 1),
        help="Offload both stages' Q3 construction to the correction warp group",
    )
    parser.add_argument(
        "--mx-dual-q3-smem-wg",
        type=int,
        choices=(0, 1),
        help="Offload both Q3 quarters through a shared-memory checkpoint",
    )
    parser.add_argument(
        "--mx-dual-q3-tmem-wg",
        type=int,
        choices=(0, 1),
        help="Offload both Q3 quarters through a temporary TMEM checkpoint",
    )
    parser.add_argument(
        "--mx-split-stage-pv",
        type=int,
        choices=(0, 1),
        help="Interleave the two independent output-accumulator PV chains",
    )
    parser.add_argument(
        "--mx-qk-scale-preload",
        type=int,
        choices=(0, 1),
        help="Override QK scale preload after the selected MX policy",
    )
    parser.add_argument(
        "--mx-qk-scale-preload-before-p",
        type=int,
        choices=(0, 1),
        help="Override stage-0 QK scale preload placement",
    )
    parser.add_argument(
        "--nv-qk-preload-page-mask",
        type=lambda value: int(value, 0),
        choices=(1, 2, 3),
        help="Choose folded NVFP4 Q/K scale pages copied before the PV tail",
    )
    parser.add_argument(
        "--ex2-emu-mask",
        type=lambda value: int(value, 0),
        help="16-bit mask selecting ALU-emulated exp2 pairs",
    )
    parser.add_argument("--ex2-alu-degree", type=int, choices=(1, 2, 3))
    parser.add_argument("--fixed-p-ceiling", type=int, choices=(0, 1))
    parser.add_argument("--score-pack-ceiling", type=int, choices=(0, 1))
    parser.add_argument("--rowmax-pack-ceiling", type=int, choices=(0, 1))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, default=Path("/tmp/tk_hao_comprehensive"))
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    if args.num_sm is not None and args.num_sm < 1:
        parser.error("--num-sm must be positive")
    for name in (
        "sm103_ldred_mask",
        "sm103_ldred_stage0_mask",
        "sm103_ldred_stage1_mask",
    ):
        value = getattr(args, name)
        if value is not None and not 0 <= value <= 15:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and 15")
    for name in (
        "mx_native_quarter_mask",
        "mx_early_native_quarter_mask",
        "mx_stage0_affine_mask",
        "mx_stage1_affine_mask",
    ):
        value = getattr(args, name)
        if value is not None and not 0 <= value <= 15:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and 15")
    for name in (
        "mx_pair_scale_reuse",
        "mx_pair_scale_stage_mask",
        "mx_q1_self_max",
        "mx_early_native_stage_mask",
    ):
        value = getattr(args, name)
        if value is not None and not 0 <= value <= 3:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and 3")
    if (
        args.mx_early_native_order in (1, 2)
        and args.mx_early_native_lookahead not in (None, 4)
    ):
        parser.error("--mx-early-native-order 1/2 requires lookahead 4")
    ceiling_modes = (
        args.fixed_p_ceiling,
        args.score_pack_ceiling,
        args.rowmax_pack_ceiling,
    )
    if sum(value == 1 for value in ceiling_modes) > 1:
        parser.error("select at most one P-chain ceiling mode")
    return args


def run(
    command: list[str],
    *,
    env: dict[str, str],
    cwd: Path = HERE,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )


def git_revision(path: Path) -> dict[str, Any]:
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        text=True,
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=path,
            text=True,
        ).strip()
    )
    return {"path": str(path), "commit": revision, "dirty": dirty}


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def software_record(env: dict[str, str]) -> dict[str, Any]:
    script = """
import json, platform
import torch, triton, cutlass, flashinfer
print(json.dumps({
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "triton": triton.__version__,
    "cutlass": getattr(cutlass, "__version__", "unknown"),
    "flashinfer": getattr(flashinfer, "__version__", "unknown"),
}))
"""
    return json.loads(
        run(
            [sys.executable, "-c", script],
            env=env,
            cwd=REPO_ROOT,
        ).stdout
    )


def gpu_record(env: dict[str, str]) -> dict[str, Any]:
    query = run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total,clocks.max.sm,power.limit",
            "--format=csv,noheader,nounits",
            "-i",
            "0",
        ],
        env=env,
        cwd=REPO_ROOT,
    ).stdout.strip()
    fields = [item.strip() for item in query.split(",")]
    return {
        "name": fields[0],
        "uuid": fields[1],
        "driver": fields[2],
        "memory_mib": int(fields[3]),
        "max_sm_clock_mhz": int(fields[4]),
        "power_limit_w": float(fields[5]),
    }


def shape_label(shape: tuple[int, int, int, int]) -> str:
    batch, seqlen, heads, dim = shape
    return f"b{batch}_s{seqlen}_h{heads}_d{dim}"


def extract_json(stdout: str) -> dict[str, Any]:
    start = stdout.rfind("\n{")
    if start >= 0:
        start += 1
    else:
        start = stdout.find("{")
    if start < 0:
        raise RuntimeError("benchmark produced no JSON")
    return json.loads(stdout[start:])


def build_record(log: str) -> dict[str, Any]:
    registers = [int(value) for value in re.findall(r"Used (\d+) registers", log)]
    barriers = [int(value) for value in re.findall(r"used (\d+) barriers", log)]
    static_smem = [int(value) for value in re.findall(r"(\d+) bytes smem", log)]
    spills = re.findall(
        r"(\d+) bytes spill stores, (\d+) bytes spill loads",
        log,
    )
    return {
        "registers": max(registers) if registers else None,
        "barriers": max(barriers) if barriers else None,
        "static_smem_bytes": max(static_smem) if static_smem else None,
        "spill_store_bytes": max((int(store) for store, _ in spills), default=0),
        "spill_load_bytes": max((int(load) for _, load in spills), default=0),
    }


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def build_extension(
    *,
    variant: str,
    shape: tuple[int, int, int, int],
    extension: Path,
    kv_stages: int | None,
    task_order: int | None,
    physical_grid_cap: int | None,
    softmax_stage0_regs: int | None,
    softmax_stage1_regs: int | None,
    correction_regs: int | None,
    producer_regs: int | None,
    sm103_ldred: int | None,
    sm103_ldred_mask: int | None,
    sm103_ldred_stage0_mask: int | None,
    sm103_ldred_stage1_mask: int | None,
    nv_scale_satfinite: int | None,
    nv_scale_encode: int | None,
    mx_native_density: int | None,
    mx_native_quarter_mask: int | None,
    mx_self_stage0_native: int | None,
    mx_early_native: int | None,
    mx_early_native_stage_mask: int | None,
    mx_early_native_quarter_mask: int | None,
    mx_early_native_lookahead: int | None,
    mx_early_native_order: int | None,
    mx_stage0_affine_mask: int | None,
    mx_stage1_affine_mask: int | None,
    mx_full_approx_denom: int | None,
    mx_pair_scale_reuse: int | None,
    mx_pair_scale_stage_mask: int | None,
    mx_q1_self_max: int | None,
    mx_q3_correction_wg: int | None,
    mx_dual_q3_correction_wg: int | None,
    mx_dual_q3_smem_wg: int | None,
    mx_dual_q3_tmem_wg: int | None,
    mx_split_stage_pv: int | None,
    mx_qk_scale_preload: int | None,
    mx_qk_scale_preload_before_p: int | None,
    nv_qk_preload_page_mask: int | None,
    ex2_emu_mask: int | None,
    ex2_alu_degree: int | None,
    fixed_p_ceiling: int | None,
    score_pack_ceiling: int | None,
    rowmax_pack_ceiling: int | None,
    target_gpu: str,
    num_sm: int | None,
    env: dict[str, str],
) -> tuple[str, dict[str, Any], str]:
    batch, seqlen, heads, dim = shape
    fp4_variant = "nv-nv" if variant == "pure-fp4" else variant
    if variant in NVMX_PARETO_POLICIES:
        makefile = "Makefile.hao_direct_fp4pv"
        module = "_C_tk_hao_direct_fp4pv"
        policy = NVMX_PARETO_POLICIES[variant]
        if dim == 64 and policy == "fast":
            policy = "fast-d64"
        flags = {
            "HAO_QK_SCALE_MODE": "0",
            "HAO_PV_SCALE_MODE": "1",
            "HAO_FP4PV_MX_POLICY": policy,
            "NVCC_SPLIT_COMPILE": "1" if seqlen >= 4096 else "2",
        }
    elif fp4_variant in FP4_FORMATS:
        makefile = "Makefile.hao_direct_fp4pv"
        module = "_C_tk_hao_direct_fp4pv"
        flags = FP4_FLAGS[fp4_variant]
    else:
        makefile = "Makefile.hao_direct"
        module = "_C_tk_hao_direct"
        flags = FP8_FLAGS
    command = [
        "make",
        "-B",
        "-f",
        makefile,
        "-j1",
        f"GPU={target_gpu}",
        f"OUT={extension}",
        f"MODULE={module}",
        f"HAO_BATCH={batch}",
        f"HAO_SEQ_LEN={seqlen}",
        f"HAO_HEADS={heads}",
        f"HAO_HEAD_DIM={dim}",
    ]
    command.extend(f"{key}={value}" for key, value in flags.items())
    if kv_stages is not None:
        command.append(f"HAO_KV_STAGES_OVERRIDE={kv_stages}")
    if task_order is not None:
        command.append(f"HAO_TASK_ORDER={task_order}")
    if physical_grid_cap is not None:
        command.append(f"HAO_PHYSICAL_GRID_CAP={physical_grid_cap}")
    if softmax_stage0_regs is not None:
        command.append(
            "HAO_FP4PV_SOFTMAX_STAGE0_REGS_OVERRIDE="
            f"{softmax_stage0_regs}"
        )
    if softmax_stage1_regs is not None:
        command.append(
            f"HAO_FP4PV_SOFTMAX_REGS_OVERRIDE={softmax_stage1_regs}"
        )
    if correction_regs is not None:
        command.append(
            f"HAO_FP4PV_CORRECTION_REGS_OVERRIDE={correction_regs}"
        )
    if producer_regs is not None:
        command.append(
            f"HAO_FP4PV_PRODUCER_REGS_OVERRIDE={producer_regs}"
        )
    if sm103_ldred is not None:
        command.append(f"HAO_FP4PV_SM103_LDRED={sm103_ldred}")
    if sm103_ldred_mask is not None:
        command.append(f"HAO_FP4PV_SM103_LDRED_MASK={sm103_ldred_mask}")
    if sm103_ldred_stage0_mask is not None:
        command.append(
            "HAO_FP4PV_SM103_LDRED_STAGE0_MASK="
            f"{sm103_ldred_stage0_mask}"
        )
    if sm103_ldred_stage1_mask is not None:
        command.append(
            "HAO_FP4PV_SM103_LDRED_STAGE1_MASK="
            f"{sm103_ldred_stage1_mask}"
        )
    if nv_scale_satfinite is not None:
        command.append(
            f"HAO_FP4PV_NV_SCALE_SATFINITE={nv_scale_satfinite}"
        )
    if nv_scale_encode is not None:
        command.append(f"HAO_FP4PV_NV_SCALE_ENCODE={nv_scale_encode}")
    if mx_native_density is not None:
        command.append(
            f"HAO_FP4PV_MX_MODE23_NATIVE_DENSITY_OVERRIDE={mx_native_density}"
        )
    if mx_native_quarter_mask is not None:
        command.append(
            "HAO_FP4PV_MX_MODE23_NATIVE_QUARTER_MASK_OVERRIDE="
            f"{mx_native_quarter_mask}"
        )
    if mx_self_stage0_native is not None:
        command.append(
            "HAO_FP4PV_MX_MODE23_SELF_STAGE0_NATIVE="
            f"{mx_self_stage0_native}"
        )
    if mx_early_native is not None:
        command.append(
            f"HAO_FP4PV_MX_MODE23_EARLY_NATIVE={mx_early_native}"
        )
    if mx_early_native_stage_mask is not None:
        command.append(
            "HAO_FP4PV_MX_MODE23_EARLY_NATIVE_STAGE_MASK="
            f"{mx_early_native_stage_mask}"
        )
    if mx_early_native_quarter_mask is not None:
        command.append(
            "HAO_FP4PV_MX_MODE23_EARLY_NATIVE_QUARTER_MASK="
            f"{mx_early_native_quarter_mask}"
        )
    if mx_early_native_lookahead is not None:
        command.append(
            "HAO_FP4PV_MX_MODE23_EARLY_NATIVE_LOOKAHEAD="
            f"{mx_early_native_lookahead}"
        )
    if mx_early_native_order is not None:
        command.append(
            f"HAO_FP4PV_MX_MODE23_EARLY_NATIVE_ORDER={mx_early_native_order}"
        )
    if mx_stage0_affine_mask is not None:
        command.append(
            f"HAO_FP4PV_MX_STAGE0_AFFINE_MASK_OVERRIDE={mx_stage0_affine_mask}"
        )
    if mx_stage1_affine_mask is not None:
        command.append(
            f"HAO_FP4PV_MX_STAGE1_AFFINE_MASK_OVERRIDE={mx_stage1_affine_mask}"
        )
    if mx_full_approx_denom is not None:
        command.append(
            f"HAO_FP4PV_MX_FULL_APPROX_DENOM_OVERRIDE={mx_full_approx_denom}"
        )
    if mx_pair_scale_reuse is not None:
        command.append(
            f"HAO_FP4PV_MX_PAIR_SCALE_REUSE_OVERRIDE={mx_pair_scale_reuse}"
        )
    if mx_pair_scale_stage_mask is not None:
        command.append(
            "HAO_FP4PV_MX_PAIR_SCALE_STAGE_MASK_OVERRIDE="
            f"{mx_pair_scale_stage_mask}"
        )
    if mx_q1_self_max is not None:
        command.append(
            f"HAO_FP4PV_MX_Q1_SELF_MAX_OVERRIDE={mx_q1_self_max}"
        )
    if mx_q3_correction_wg is not None:
        command.append(
            f"HAO_FP4PV_MX_Q3_CORR_WG={mx_q3_correction_wg}"
        )
    if mx_dual_q3_correction_wg is not None:
        command.append(
            "HAO_FP4PV_MX_DUAL_Q3_CORR_WG="
            f"{mx_dual_q3_correction_wg}"
        )
    if mx_dual_q3_smem_wg is not None:
        command.append(
            f"HAO_FP4PV_MX_DUAL_Q3_SMEM_WG={mx_dual_q3_smem_wg}"
        )
    if mx_dual_q3_tmem_wg is not None:
        command.append(
            f"HAO_FP4PV_MX_DUAL_Q3_TMEM_WG={mx_dual_q3_tmem_wg}"
        )
    if mx_split_stage_pv is not None:
        command.append(
            "HAO_FP4PV_MX_SPLIT_STAGE_PV_OVERRIDE="
            f"{mx_split_stage_pv}"
        )
    if mx_qk_scale_preload is not None:
        command.append(
            "HAO_FP4PV_MX_QK_SCALE_PRELOAD_OVERRIDE="
            f"{mx_qk_scale_preload}"
        )
    if mx_qk_scale_preload_before_p is not None:
        command.append(
            "HAO_FP4PV_MX_QK_SCALE_PRELOAD_BEFORE_P_OVERRIDE="
            f"{mx_qk_scale_preload_before_p}"
        )
    if nv_qk_preload_page_mask is not None:
        command.append(
            "HAO_FP4PV_NV_QK_PRELOAD_PAGE_MASK_OVERRIDE="
            f"{nv_qk_preload_page_mask}"
        )
    if ex2_emu_mask is not None:
        command.append(f"HAO_FP4PV_EX2_EMU_MASK=0x{ex2_emu_mask:04x}")
    if ex2_alu_degree is not None:
        command.append(f"HAO_FP4PV_EX2_ALU_DEGREE={ex2_alu_degree}")
    if fixed_p_ceiling is not None:
        command.append(f"HAO_FP4PV_FIXED_P_CEILING={fixed_p_ceiling}")
    if score_pack_ceiling is not None:
        command.append(f"HAO_FP4PV_SCORE_PACK_CEILING={score_pack_ceiling}")
    if rowmax_pack_ceiling is not None:
        command.append(f"HAO_FP4PV_ROWMAX_PACK_CEILING={rowmax_pack_ceiling}")
    if num_sm is not None:
        command.append(f"HAO_NUM_SM={num_sm}")
    completed = run(command, env=env)
    log = completed.stdout + completed.stderr
    return module, build_record(log), log


def benchmark_extension(
    *,
    variant: str,
    shape: tuple[int, int, int, int],
    extension: Path,
    module: str,
    args: argparse.Namespace,
    env: dict[str, str],
) -> dict[str, Any]:
    batch, seqlen, heads, dim = shape
    if args.cooldown_seconds > 0:
        time.sleep(args.cooldown_seconds)
    fp4_variant = "nv-nv" if variant == "pure-fp4" else variant
    if variant in NVMX_PARETO_POLICIES:
        qk_format, pv_format = ("nvfp4", "mxfp4")
    elif fp4_variant in FP4_FORMATS:
        qk_format, pv_format = FP4_FORMATS[fp4_variant]
    else:
        qk_format = pv_format = None
    if qk_format is not None:
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
            pv_format,
        ]
        if variant in ("nvmx-balanced", "nvmx-accurate"):
            command.extend(
                ("--global-anchor-kv", "--global-anchor-samples", "32")
            )
        if dim == 128 and variant in NVMX_FOLDED_QK_VARIANTS:
            command.extend(
                (
                    "--nv-qk-fold-k64-scales",
                    "both",
                    "--nv-qk-fold-scale-select",
                    "mse",
                )
            )
        if args.skip_hao_fp4:
            command.append("--skip-hao-fp4")
    else:
        command = [
            sys.executable,
            "hao_direct_benchmark.py",
            "--extension",
            str(extension),
            "--qk-format",
            "nvfp4",
            "--batch",
            str(batch),
            "--seqlen",
            str(seqlen),
            "--heads",
            str(heads),
        ]
    command.extend(
        (
            "--warmup-ms",
            str(args.warmup_ms),
            "--rep-ms",
            str(args.rep_ms),
            "--seed",
            str(args.seed),
        )
    )
    completed = run(command, env=env)
    return extract_json(completed.stdout)


def benchmark_native_reference(
    *,
    shape: tuple[int, int, int, int],
    args: argparse.Namespace,
    env: dict[str, str],
) -> dict[str, Any]:
    batch, seqlen, heads, dim = shape
    if args.cooldown_seconds > 0:
        time.sleep(args.cooldown_seconds)
    completed = run(
        [
            sys.executable,
            "hao_native_reference_benchmark.py",
            "--batch",
            str(batch),
            "--seqlen",
            str(seqlen),
            "--heads",
            str(heads),
            "--dim",
            str(dim),
            "--warmup-ms",
            str(args.warmup_ms),
            "--rep-ms",
            str(args.rep_ms),
            "--seed",
            str(args.seed),
        ],
        env=env,
    )
    return extract_json(completed.stdout)


def add_throughput(
    benchmark: dict[str, Any],
    shape: tuple[int, int, int, int],
) -> None:
    batch, seqlen, heads, dim = shape
    flops = batch * heads * 2 * seqlen * seqlen * (dim + dim)
    benchmark["tflops"] = {
        name: flops / (milliseconds * 1e-3) / 1e12
        for name, milliseconds in benchmark["timing_ms"].items()
    }


def main() -> None:
    args = parse_args()
    shapes = list(args.shape or suite_shapes(args.shape_set))
    if args.variant == "both":
        variants = ("pure-fp4", "fp8")
    elif args.variant == "fp4-matrix":
        variants = tuple(FP4_FORMATS)
    elif args.variant == "nvmx-pareto":
        variants = tuple(NVMX_PARETO_POLICIES)
    elif args.variant == "all":
        variants = (*FP4_FORMATS, *NVMX_PARETO_POLICIES, "fp8")
    else:
        variants = (args.variant,)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases_dir = args.output_dir / "cases"
    logs_dir = args.output_dir / "build_logs"
    cases_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    args.build_root.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    python_paths = [
        str(REPO_ROOT / "TK_quantisation/mxfp4_v3"),
        str(HAO_ROOT),
    ]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": args.gpu,
            "PYTHONPATH": os.pathsep.join(python_paths),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    created = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        # Policy definitions evolve during tuning. Keep resumed manifests honest
        # even when every case is already complete and therefore skipped.
        manifest["variant_flags"] = {
            "pure-fp4": PURE_FP4_FLAGS,
            **FP4_FLAGS,
            **{
                name: {
                    "HAO_FP4PV_MX_POLICY": policy,
                    "D64_POLICY": (
                        "fast-d64" if policy == "fast" else policy
                    ),
                    "GLOBAL_ANCHOR_SAMPLES": (
                        "32" if policy != "fast" else "0"
                    ),
                    "NV_QK_FOLDED_K64_SCALES": (
                        "both"
                        if name in NVMX_FOLDED_QK_VARIANTS
                        else "none"
                    ),
                }
                for name, policy in NVMX_PARETO_POLICIES.items()
            },
            "fp8": FP8_FLAGS,
        }
        atomic_json(manifest_path, manifest)
    else:
        manifest = {
            "schema": "tk_hao_comprehensive_v1",
            "created_utc": created,
            "complete": False,
            "protocol": {
                "seed": args.seed,
                "warmup_ms": args.warmup_ms,
                "rep_ms": args.rep_ms,
                "cooldown_seconds": args.cooldown_seconds,
                "skip_hao_fp4": args.skip_hao_fp4,
                "kv_stages_override": args.kv_stages,
                "task_order_override": args.task_order,
                "physical_grid_cap_override": args.physical_grid_cap,
                "softmax_stage0_regs_override": args.softmax_stage0_regs,
                "softmax_stage1_regs_override": args.softmax_stage1_regs,
                "correction_regs_override": args.correction_regs,
                "producer_regs_override": args.producer_regs,
                "sm103_ldred_override": args.sm103_ldred,
                "sm103_ldred_mask_override": args.sm103_ldred_mask,
                "sm103_ldred_stage0_mask_override": (
                    args.sm103_ldred_stage0_mask
                ),
                "sm103_ldred_stage1_mask_override": (
                    args.sm103_ldred_stage1_mask
                ),
                "mx_native_density_override": args.mx_native_density,
                "mx_native_quarter_mask_override": args.mx_native_quarter_mask,
                "mx_self_stage0_native_override": args.mx_self_stage0_native,
                "mx_early_native_override": args.mx_early_native,
                "mx_early_native_stage_mask_override": (
                    args.mx_early_native_stage_mask
                ),
                "mx_early_native_quarter_mask_override": (
                    args.mx_early_native_quarter_mask
                ),
                "mx_early_native_lookahead_override": (
                    args.mx_early_native_lookahead
                ),
                "mx_early_native_order_override": args.mx_early_native_order,
                "mx_stage0_affine_mask_override": args.mx_stage0_affine_mask,
                "mx_stage1_affine_mask_override": args.mx_stage1_affine_mask,
                "mx_full_approx_denom_override": args.mx_full_approx_denom,
                "mx_pair_scale_reuse_override": args.mx_pair_scale_reuse,
                "mx_pair_scale_stage_mask_override": (
                    args.mx_pair_scale_stage_mask
                ),
                "mx_q1_self_max_override": args.mx_q1_self_max,
                "mx_q3_correction_wg_override": args.mx_q3_correction_wg,
                "mx_dual_q3_correction_wg_override": (
                    args.mx_dual_q3_correction_wg
                ),
                "mx_dual_q3_smem_wg_override": args.mx_dual_q3_smem_wg,
                "mx_dual_q3_tmem_wg_override": args.mx_dual_q3_tmem_wg,
                "mx_split_stage_pv_override": args.mx_split_stage_pv,
                "mx_qk_scale_preload_override": args.mx_qk_scale_preload,
                "mx_qk_scale_preload_before_p_override": (
                    args.mx_qk_scale_preload_before_p
                ),
                "nv_qk_preload_page_mask_override": (
                    args.nv_qk_preload_page_mask
                ),
                "ex2_emu_mask_override": args.ex2_emu_mask,
                "ex2_alu_degree_override": args.ex2_alu_degree,
                "fixed_p_ceiling_override": args.fixed_p_ceiling,
                "score_pack_ceiling_override": args.score_pack_ceiling,
                "rowmax_pack_ceiling_override": args.rowmax_pack_ceiling,
                "target_gpu": args.target_gpu,
                "num_sm_override": args.num_sm,
                "timer": "triton.testing.do_bench median",
                "causal": False,
                "factory": "HAO create_nvfp4_attention_tensors",
                "flop_convention": "B*H*2*Sq*Sk*(Dqk+Dvo)",
            },
            "sources": {
                "tk": {
                    **git_revision(REPO_ROOT),
                    "files": [
                        file_record(HERE / "hao_direct_fp4pv_kernel.inc"),
                        file_record(HERE / "hao_direct_fp4pv_softmax_reader.inc"),
                        file_record(HERE / "hao_direct_kernel.inc"),
                        file_record(HERE / "hao_direct_softmax_reader.inc"),
                        file_record(HERE / "hao_comprehensive_suite.py"),
                    ],
                },
                "hao": {
                    **git_revision(HAO_ROOT),
                    "files": [
                        file_record(
                            HAO_ROOT / "flash_attn/cute/flash_fwd_sm100_fp4.py"
                        ),
                        file_record(
                            HAO_ROOT / "flash_attn/cute/benchmarks/bench_fp4.py"
                        ),
                        file_record(HAO_ROOT / "flash_attn/cute/README.md"),
                    ],
                },
            },
            "software": software_record(env),
            "gpu": gpu_record(env),
            "shapes": [
                {"batch": b, "seqlen": s, "heads": h, "dim": d}
                for b, s, h, d in shapes
            ],
            "variants": list(variants),
            "variant_flags": {
                "pure-fp4": PURE_FP4_FLAGS,
                **FP4_FLAGS,
                **{
                    name: {
                        "HAO_FP4PV_MX_POLICY": policy,
                        "D64_POLICY": (
                            "fast-d64" if policy == "fast" else policy
                        ),
                        "GLOBAL_ANCHOR_SAMPLES": (
                            "32" if policy != "fast" else "0"
                        ),
                        "NV_QK_FOLDED_K64_SCALES": (
                            "both"
                            if name in NVMX_FOLDED_QK_VARIANTS
                            else "none"
                        ),
                    }
                    for name, policy in NVMX_PARETO_POLICIES.items()
                },
                "fp8": FP8_FLAGS,
            },
            "results": [],
            "failures": [],
        }
        atomic_json(manifest_path, manifest)

    completed_keys = {
        (item["label"], item["variant"])
        for item in manifest["results"]
    }
    for shape in shapes:
        label = shape_label(shape)
        for variant in variants:
            key = (label, variant)
            if key in completed_keys and not args.rebuild:
                print(f"[skip] {label} {variant}", flush=True)
                continue
            if key in completed_keys:
                manifest["results"] = [
                    item
                    for item in manifest["results"]
                    if (item["label"], item["variant"]) != key
                ]
                manifest["failures"] = [
                    item
                    for item in manifest["failures"]
                    if (item["label"], item["variant"]) != key
                ]
                completed_keys.discard(key)
                atomic_json(manifest_path, manifest)
            else:
                manifest["failures"] = [
                    item
                    for item in manifest["failures"]
                    if (item["label"], item["variant"]) != key
                ]
                atomic_json(manifest_path, manifest)
            batch, seqlen, heads, dim = shape
            fp4_variant = "nv-nv" if variant == "pure-fp4" else variant
            d64_fp4_supported = dim == 64 and (
                fp4_variant in ("nv-nv", "nv-nv-bounded", "nv-mx") or
                variant in NVMX_PARETO_POLICIES
            )
            if dim != 128 and not d64_fp4_supported:
                unsupported = {
                    "label": label,
                    "variant": variant,
                    "shape": {
                        "batch": batch,
                        "seqlen": seqlen,
                        "heads": heads,
                        "dim": dim,
                    },
                    "status": "unsupported",
                    "reason": (
                        "D64 currently supports NVFP4 QK with NVFP4 or "
                        "MXFP4 PV only"
                    ),
                }
                manifest["results"].append(unsupported)
                completed_keys.add(key)
                atomic_json(manifest_path, manifest)
                print(f"[unsupported] {label} {variant}", flush=True)
                continue
            extension = args.build_root / f"{label}_{variant}.so"
            case_path = cases_dir / f"{label}_{variant}.json"
            log_path = logs_dir / f"{label}_{variant}.log"
            print(f"[build] {label} {variant}", flush=True)
            try:
                module, build, build_log = build_extension(
                    variant=variant,
                    shape=shape,
                    extension=extension,
                    kv_stages=args.kv_stages,
                    task_order=args.task_order,
                    physical_grid_cap=args.physical_grid_cap,
                    softmax_stage0_regs=args.softmax_stage0_regs,
                    softmax_stage1_regs=args.softmax_stage1_regs,
                    correction_regs=args.correction_regs,
                    producer_regs=args.producer_regs,
                    sm103_ldred=args.sm103_ldred,
                    sm103_ldred_mask=args.sm103_ldred_mask,
                    sm103_ldred_stage0_mask=(
                        args.sm103_ldred_stage0_mask
                    ),
                    sm103_ldred_stage1_mask=(
                        args.sm103_ldred_stage1_mask
                    ),
                    nv_scale_satfinite=args.nv_scale_satfinite,
                    nv_scale_encode=args.nv_scale_encode,
                    mx_native_density=args.mx_native_density,
                    mx_native_quarter_mask=args.mx_native_quarter_mask,
                    mx_self_stage0_native=args.mx_self_stage0_native,
                    mx_early_native=args.mx_early_native,
                    mx_early_native_stage_mask=(
                        args.mx_early_native_stage_mask
                    ),
                    mx_early_native_quarter_mask=(
                        args.mx_early_native_quarter_mask
                    ),
                    mx_early_native_lookahead=(
                        args.mx_early_native_lookahead
                    ),
                    mx_early_native_order=args.mx_early_native_order,
                    mx_stage0_affine_mask=args.mx_stage0_affine_mask,
                    mx_stage1_affine_mask=args.mx_stage1_affine_mask,
                    mx_full_approx_denom=args.mx_full_approx_denom,
                    mx_pair_scale_reuse=args.mx_pair_scale_reuse,
                    mx_pair_scale_stage_mask=args.mx_pair_scale_stage_mask,
                    mx_q1_self_max=args.mx_q1_self_max,
                    mx_q3_correction_wg=args.mx_q3_correction_wg,
                    mx_dual_q3_correction_wg=(
                        args.mx_dual_q3_correction_wg
                    ),
                    mx_dual_q3_smem_wg=args.mx_dual_q3_smem_wg,
                    mx_dual_q3_tmem_wg=args.mx_dual_q3_tmem_wg,
                    mx_split_stage_pv=args.mx_split_stage_pv,
                    mx_qk_scale_preload=args.mx_qk_scale_preload,
                    mx_qk_scale_preload_before_p=(
                        args.mx_qk_scale_preload_before_p
                    ),
                    nv_qk_preload_page_mask=args.nv_qk_preload_page_mask,
                    ex2_emu_mask=args.ex2_emu_mask,
                    ex2_alu_degree=args.ex2_alu_degree,
                    fixed_p_ceiling=args.fixed_p_ceiling,
                    score_pack_ceiling=args.score_pack_ceiling,
                    rowmax_pack_ceiling=args.rowmax_pack_ceiling,
                    target_gpu=args.target_gpu,
                    num_sm=args.num_sm,
                    env=env,
                )
                log_path.write_text(build_log)
                print(f"[bench] {label} {variant}", flush=True)
                benchmark = benchmark_extension(
                    variant=variant,
                    shape=shape,
                    extension=extension,
                    module=module,
                    args=args,
                    env=env,
                )
                for comparison_name in (
                    "tk_vs_bf16_output",
                    "tk_vs_bf16_lse",
                ):
                    comparison = benchmark["correctness"][comparison_name]
                    actual_nonfinite = comparison["actual_nonfinite"]
                    reference_nonfinite = comparison["reference_nonfinite"]
                    if actual_nonfinite or reference_nonfinite:
                        atomic_json(
                            case_path,
                            {
                                "label": label,
                                "variant": variant,
                                "shape": {
                                    "batch": batch,
                                    "seqlen": seqlen,
                                    "heads": heads,
                                    "dim": dim,
                                },
                                "status": "invalid",
                                "build": build,
                                "benchmark": benchmark,
                            },
                        )
                        raise RuntimeError(
                            f"{comparison_name} contains non-finite values: "
                            f"actual={actual_nonfinite}, "
                            f"reference={reference_nonfinite}"
                        )
                add_throughput(benchmark, shape)
                result = {
                    "label": label,
                    "variant": variant,
                    "shape": {
                        "batch": batch,
                        "seqlen": seqlen,
                        "heads": heads,
                        "dim": dim,
                    },
                    "status": "complete",
                    "build": build,
                    "benchmark": benchmark,
                }
                atomic_json(case_path, result)
                manifest["results"].append(result)
                completed_keys.add(key)
                atomic_json(manifest_path, manifest)
                timing = benchmark["timing_ms"]
                tk_key = next(name for name in timing if name.startswith("tk_"))
                print(
                    f"[done] {label} {variant}: {timing[tk_key]:.6f} ms",
                    flush=True,
                )
            except Exception as exc:
                failure = {
                    "label": label,
                    "variant": variant,
                    "exception": repr(exc),
                }
                if isinstance(exc, subprocess.CalledProcessError):
                    failure["command"] = exc.cmd
                    failure["stdout"] = exc.stdout
                    failure["stderr"] = exc.stderr
                manifest["failures"].append(failure)
                atomic_json(manifest_path, manifest)
                print(f"[failed] {label} {variant}: {exc}", file=sys.stderr, flush=True)
                if isinstance(exc, subprocess.CalledProcessError):
                    if exc.stdout:
                        print(exc.stdout, file=sys.stderr, flush=True)
                    if exc.stderr:
                        print(exc.stderr, file=sys.stderr, flush=True)
                raise

    manifest["complete"] = not manifest["failures"]
    manifest["completed_utc"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    atomic_json(manifest_path, manifest)
    print(f"[report] {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
