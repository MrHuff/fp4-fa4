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
        HERE.parents[4] / "third_party" / "hao_flash_attention_fp4",
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
    sweeps = (
        (1, seqlen, heads, 128)
        for heads in SWEEP_HEADS
        for seqlen in SWEEP_SEQUENCES
    )
    if name == "headline":
        return list(HEADLINE_SHAPES)
    if name == "sweeps":
        return unique_shapes(sweeps)
    return unique_shapes((*HEADLINE_SHAPES, *sweeps))


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
        choices=("headline", "sweeps", "all"),
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, default=Path("/tmp/tk_hao_comprehensive"))
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


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
    spill = re.search(
        r"(\d+) bytes spill stores, (\d+) bytes spill loads",
        log,
    )
    return {
        "registers": registers[0] if registers else None,
        "barriers": barriers[0] if barriers else None,
        "static_smem_bytes": static_smem[0] if static_smem else None,
        "spill_store_bytes": int(spill.group(1)) if spill else 0,
        "spill_load_bytes": int(spill.group(2)) if spill else 0,
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
    env: dict[str, str],
) -> tuple[str, dict[str, Any], str]:
    batch, seqlen, heads, dim = shape
    if dim != 128:
        raise ValueError(f"{variant} TK extension supports D128 only")
    fp4_variant = "nv-nv" if variant == "pure-fp4" else variant
    if variant in NVMX_PARETO_POLICIES:
        makefile = "Makefile.hao_direct_fp4pv"
        module = "_C_tk_hao_direct_fp4pv"
        flags = {
            "HAO_QK_SCALE_MODE": "0",
            "HAO_PV_SCALE_MODE": "1",
            "HAO_FP4PV_MX_POLICY": NVMX_PARETO_POLICIES[variant],
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
        f"OUT={extension}",
        f"MODULE={module}",
        f"HAO_BATCH={batch}",
        f"HAO_SEQ_LEN={seqlen}",
        f"HAO_HEADS={heads}",
    ]
    command.extend(f"{key}={value}" for key, value in flags.items())
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
    batch, seqlen, heads, _ = shape
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
        if variant in NVMX_FOLDED_QK_VARIANTS:
            command.extend(
                (
                    "--nv-qk-fold-k64-scales",
                    "both",
                    "--nv-qk-fold-scale-select",
                    "mse",
                )
            )
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
            if dim != 128 and (
                fp4_variant in FP4_FORMATS or
                variant in NVMX_PARETO_POLICIES
            ):
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
                    "reason": "TK comprehensive kernels are specialized for D128",
                }
                manifest["results"].append(unsupported)
                completed_keys.add(key)
                atomic_json(manifest_path, manifest)
                print(f"[unsupported] {label} {variant}", flush=True)
                continue
            if dim != 128:
                print(f"[reference] {label} {variant}", flush=True)
                benchmark = benchmark_native_reference(
                    shape=shape,
                    args=args,
                    env=env,
                )
                result = {
                    "label": label,
                    "variant": variant,
                    "shape": {
                        "batch": batch,
                        "seqlen": seqlen,
                        "heads": heads,
                        "dim": dim,
                    },
                    "status": "reference-only",
                    "reason": "TK comprehensive kernels are specialized for D128",
                    "benchmark": benchmark,
                }
                atomic_json(cases_dir / f"{label}_{variant}.json", result)
                manifest["results"].append(result)
                completed_keys.add(key)
                atomic_json(manifest_path, manifest)
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
                manifest["failures"].append(failure)
                atomic_json(manifest_path, manifest)
                print(f"[failed] {label} {variant}: {exc}", file=sys.stderr, flush=True)
                raise

    manifest["complete"] = not manifest["failures"]
    manifest["completed_utc"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    atomic_json(manifest_path, manifest)
    print(f"[report] {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
