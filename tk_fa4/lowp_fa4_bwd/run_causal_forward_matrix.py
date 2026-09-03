#!/usr/bin/env python3
"""Build and run the matched causal D64 forward matrix, one shape at a time.

The driver intentionally serializes every expensive operation.  For each
shape it builds the reconstructed causal-accurate MXFP4-PV extension with
density 4, builds the exact mode-0 FP8-PV control, and then invokes
:mod:`benchmark_causal_forward_matrix` in a fresh Python process.  The MX
route names are ``d4q01`` for quarter mask 3 (Q0/Q1) and ``d4all`` for quarter
mask 15 (all quarters); ``d4all`` is the default.  Commands, source and
artifact hashes, resource snapshots, runtime topology, and a compact
incremental summary are retained under the result directory.

No command is passed through a shell.  ``--dry-run`` prints the complete plan
without creating directories, compiling CUDA, or touching a GPU.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
FORWARD_ROOT = REPO_ROOT / "tk_fa4" / "fp4_fa4_fwd"
MX_MAKEFILE = FORWARD_ROOT / "Makefile.hao_direct_fp4pv"
MX_CONFIG = FORWARD_ROOT / "hao_direct_fp4pv_config.inc"
MX_HOST = FORWARD_ROOT / "hao_direct_fp4pv_host.inc"
FP8_HOST = FORWARD_ROOT / "hao_direct_host.inc"
TK_COMMON = REPO_ROOT / "ThunderKittens" / "kernels" / "common.mk"
EXACT_BUILDER = HERE / "build_causal_gqa_fp8pv_forward.py"
EXACT_PATCH = HERE / "causal_gqa_fp8pv_forward.patch"
WORKER = HERE / "benchmark_causal_forward_matrix.py"
GIB = 1 << 30


@dataclass(frozen=True, order=True)
class Shape:
    sequence: int
    q_heads: int
    kv_heads: int
    head_dim: int = 64

    def __post_init__(self) -> None:
        if self.sequence <= 0 or self.sequence % 256:
            raise ValueError("sequence must be positive and divisible by 256")
        if self.q_heads <= 0 or self.kv_heads <= 0:
            raise ValueError("head counts must be positive")
        if self.q_heads % self.kv_heads:
            raise ValueError("q_heads must be divisible by kv_heads")
        if self.q_heads % 2 or self.kv_heads % 2:
            raise ValueError("the paired D64 projection requires even head counts")
        if self.head_dim != 64:
            raise ValueError("this reconstructed forward matrix is D64-only")

    @property
    def label(self) -> str:
        return (
            f"s{self.sequence}_h{self.q_heads}_kv{self.kv_heads}"
            f"_d{self.head_dim}"
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "batch": 1,
            "sequence": self.sequence,
            "q_heads": self.q_heads,
            "kv_heads": self.kv_heads,
            "head_dim": self.head_dim,
            "gqa_ratio": self.q_heads // self.kv_heads,
        }


DEFAULT_SHAPES = (
    Shape(512, 32, 8),
    Shape(1024, 32, 8),
    Shape(2048, 32, 8),
    Shape(4096, 32, 8),
    Shape(8192, 32, 8),
    Shape(4096, 16, 4),
    Shape(4096, 64, 16),
)

MX_QUARTER_MASK_NAMES = {3: "d4q01", 15: "d4all"}

EXPECTED_MX_TOPOLOGY_BASE: dict[str, Any] = {
    "causal": True,
    "qk_format": "nvfp4_e4m3_block16",
    "pv_format": "mxfp4_e8m0_block32",
    "fixed_route_fastpath": True,
    "route_env_guard_per_launch": False,
    "kernel_attribute_init": "once_per_host_thread_and_cuda_device",
    "tma_descriptor_cache": "bounded_thread_local_gl_descriptors",
    "tma_descriptor_cache_capacity": 256,
    "tma_descriptor_cache_lookup": (
        "splitmix64_device_pointer_four_way_set_associative"
    ),
    "tma_descriptor_cache_set_hash": "splitmix64_device_pointer_v1",
    "tma_descriptor_cache_sets": 64,
    "tma_descriptor_cache_ways": 4,
    "tma_descriptor_cache_capacity_scope": "per_compile_time_gl_slot",
    "tma_descriptor_cache_gl_slots": 10,
    "tma_descriptor_cache_total_entry_ceiling": 2560,
    "tma_descriptor_cache_key": (
        "cuda_device_data_ptr_and_compile_time_gl_slot"
    ),
    "tma_descriptor_cache_owns_tensors": False,
    "tma_descriptor_cache_counter_scope": "calling_host_thread",
    "causal_interleaved_kv": True,
    "mx_mode23_native_density": 4,
    "mx_stage0_affine_mask": 0,
    "mx_stage1_affine_mask": 0,
    "fixed_p_ceiling": False,
    "score_pack_ceiling": False,
}
EXPECTED_FP8_TOPOLOGY: dict[str, Any] = {
    "causal": True,
    "qk_format": "nvfp4_e4m3_block16",
    "pv_format": "e4m3_fp8",
    "fixed_route_fastpath": True,
    "route_env_guard_per_launch": False,
    "kernel_attribute_init": "once_per_host_thread_and_cuda_device",
    "tma_descriptor_cache": "bounded_thread_local_gl_descriptors",
    "tma_descriptor_cache_capacity": 256,
    "tma_descriptor_cache_lookup": (
        "splitmix64_device_pointer_four_way_set_associative"
    ),
    "tma_descriptor_cache_set_hash": "splitmix64_device_pointer_v1",
    "tma_descriptor_cache_sets": 64,
    "tma_descriptor_cache_ways": 4,
    "tma_descriptor_cache_capacity_scope": "per_compile_time_gl_slot",
    "tma_descriptor_cache_gl_slots": 9,
    "tma_descriptor_cache_total_entry_ceiling": 2304,
    "tma_descriptor_cache_key": (
        "cuda_device_data_ptr_and_compile_time_gl_slot"
    ),
    "tma_descriptor_cache_owns_tensors": False,
    "tma_descriptor_cache_counter_scope": "calling_host_thread",
    "shiftless_fp8_mode": 0,
    "fixed_p_ceiling": False,
    "score_pack_ceiling": False,
}


def _expected_mx_topology(quarter_mask: int) -> dict[str, Any]:
    return {
        **EXPECTED_MX_TOPOLOGY_BASE,
        "mx_mode23_native_quarter_mask": quarter_mask,
    }


class CommandFailed(RuntimeError):
    def __init__(self, name: str, record: dict[str, Any]) -> None:
        super().__init__(
            f"{name} failed with status {record['returncode']}; "
            f"see {record['log']}"
        )
        self.name = name
        self.record = record


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _default_run_tag() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sanitize_tag(value: str) -> str:
    tag = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")
    if not tag:
        raise ValueError("run tag must contain an ASCII letter or digit")
    if len(tag) > 48:
        raise ValueError("run tag must be at most 48 normalized characters")
    return tag


def _parse_shape(value: str) -> Shape:
    fields = re.split(r"[/,:x]", value)
    if len(fields) != 3:
        raise argparse.ArgumentTypeError("shape must be S/Hq/Hkv")
    try:
        return Shape(*(int(field) for field in fields))
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _positive_finite(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape",
        action="append",
        type=_parse_shape,
        metavar="S/HQ/HKV",
        help=(
            "shape to run; repeat to replace the seven-shape D64 default "
            "matrix"
        ),
    )
    parser.add_argument("--run-tag", default=_default_run_tag())
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument(
        "--projection-extension",
        type=Path,
        help="projection publisher extension passed to the one-shape worker",
    )
    parser.add_argument("--projection-module", default="_C_b300_lowp_bwd")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used for ABI discovery, builds, and workers",
    )
    parser.add_argument("--gpu-index", type=_nonnegative_int, default=0)
    parser.add_argument("--gpu-arch", choices=("B200", "B300"), default="B200")
    parser.add_argument("--num-sm", type=int, default=152)
    parser.add_argument(
        "--mx-quarter-mask",
        type=int,
        choices=tuple(MX_QUARTER_MASK_NAMES),
        default=15,
        help=(
            "density-4 native quarter mask: 3 is d4q01 (Q0/Q1), "
            "15 is d4all (default)"
        ),
    )
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--warmups", type=_nonnegative_int, default=5)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument(
        "--minimum-free-system-gib", type=_positive_finite, default=64.0
    )
    parser.add_argument(
        "--minimum-free-gpu-gib", type=_positive_finite, default=16.0
    )
    parser.add_argument(
        "--nvidia-smi", default="nvidia-smi", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="continue to later shapes after preserving a failed manifest",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print commands and provenance plan without writing or executing",
    )
    args = parser.parse_args(argv)

    try:
        args.run_tag = _sanitize_tag(args.run_tag)
    except ValueError as error:
        parser.error(str(error))
    args.shapes = tuple(args.shape or DEFAULT_SHAPES)
    args.mx_route_name = MX_QUARTER_MASK_NAMES[args.mx_quarter_mask]
    if len(set(args.shapes)) != len(args.shapes):
        parser.error("duplicate --shape entries are not allowed")
    if args.num_sm <= 0:
        parser.error("--num-sm must be positive")
    if args.hidden <= 0 or args.hidden % 128:
        parser.error("--hidden must be positive and divisible by 128")
    if args.samples <= 0:
        parser.error("--samples must be positive")
    if not args.projection_module.isidentifier():
        parser.error("--projection-module must be a Python identifier")
    if args.result_dir is None:
        args.result_dir = (
            REPO_ROOT / "results" / f"causal_forward_matrix_{args.run_tag}"
        )
    if args.artifact_dir is None:
        args.artifact_dir = Path("/tmp") / f"causal_forward_matrix_{args.run_tag}"
    return args


def _resolve_executable(value: str) -> Path:
    candidate = shutil.which(value)
    if candidate is None:
        path = Path(value).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            candidate = str(path)
    if candidate is None:
        raise FileNotFoundError(f"executable not found: {value}")
    # Do not call Path.resolve() here.  A venv's Python is normally a symlink
    # to the base interpreter, but invoking the symlink is what activates the
    # venv through the adjacent pyvenv.cfg.  Preserve that selected path while
    # normalizing it to an absolute path and validating the followed target.
    selected = Path(os.path.abspath(Path(candidate).expanduser()))
    if not selected.is_file() or not os.access(selected, os.X_OK):
        raise FileNotFoundError(f"executable is not runnable: {selected}")
    return selected


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, *, require: bool = True) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        if require:
            raise FileNotFoundError(resolved)
        return {"path": str(resolved), "exists": False}
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "exists": True,
        "bytes": stat.st_size,
        "sha256": _sha256(resolved),
    }


def _git_record() -> dict[str, Any]:
    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    return {
        "root": str(REPO_ROOT),
        "head": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "tracked_dirty": bool(run("status", "--short", "--untracked-files=no")),
    }


def _python_record(python: Path) -> tuple[dict[str, Any], dict[str, str]]:
    probe = r"""
import json
import sys
import sysconfig
import cutlass.cute
from torch.utils.cpp_extension import include_paths, library_paths

print(json.dumps({
    "executable": sys.executable,
    "version": sys.version,
    "ext_suffix": sysconfig.get_config_var("EXT_SUFFIX"),
    "ldversion": sysconfig.get_config_var("LDVERSION"),
    "include": sysconfig.get_path("include"),
    "libdir": sysconfig.get_config_var("LIBDIR"),
    "torch_include_paths": include_paths(),
    "torch_library_paths": library_paths(),
    "cutlass_cute_imported": True,
}))
"""
    completed = subprocess.run(
        [str(python), "-c", probe],
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = json.loads(completed.stdout)
    pybind = subprocess.run(
        [str(python), "-m", "pybind11", "--includes"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    required = ("ext_suffix", "ldversion", "include", "libdir")
    missing = [name for name in required if not metadata.get(name)]
    if missing:
        raise RuntimeError(f"Python ABI probe omitted {', '.join(missing)}")
    build_environment = {
        "PYTHON_VERSION": str(metadata["ldversion"]),
        "PYTHON_INCLUDES": f"-I{metadata['include']}",
        "PYTHON_LIBDIR": f"-L{metadata['libdir']}",
        "PYBIND_INCLUDES": pybind,
        "PYTORCH_INCLUDES": " ".join(
            f"-I{path}" for path in metadata["torch_include_paths"]
        ),
        "PYTORCH_LIBRARY_PATHS": " ".join(metadata["torch_library_paths"]),
    }
    metadata["pybind_includes"] = pybind
    return metadata, build_environment


def _source_records() -> dict[str, dict[str, Any]]:
    return {
        "worker": _file_record(WORKER),
        "mx_makefile": _file_record(MX_MAKEFILE),
        "mx_config": _file_record(MX_CONFIG),
        "mx_host": _file_record(MX_HOST),
        "fp8_host": _file_record(FP8_HOST),
        "thunderkittens_common": _file_record(TK_COMMON),
        "exact_builder": _file_record(EXACT_BUILDER),
        "exact_patch": _file_record(EXACT_PATCH),
        "driver": _file_record(Path(__file__)),
    }


def _modules(
    shape: Shape,
    run_tag: str,
    mx_quarter_mask: int,
) -> tuple[str, str, str]:
    suffix = shape.label.replace("_", "")
    route_name = MX_QUARTER_MASK_NAMES[mx_quarter_mask]
    mx = f"_C_cfwd_mx_{route_name}_i1_{suffix}_{run_tag}"
    fp8 = f"_C_cfwd_fp8exact0_for_{route_name}_{suffix}_{run_tag}"
    symbol_tag = f"cfwd_mx_{route_name}_i1_{suffix}_{run_tag}"
    return mx, fp8, symbol_tag


def _commands(
    args: argparse.Namespace,
    shape: Shape,
    python: Path,
    extension_suffix: str,
    result_path: Path,
) -> dict[str, dict[str, Any]]:
    mx_module, fp8_module, symbol_tag = _modules(
        shape,
        args.run_tag,
        args.mx_quarter_mask,
    )
    mx_extension = args.artifact_dir / f"{mx_module}{extension_suffix}"
    fp8_extension = args.artifact_dir / f"{fp8_module}{extension_suffix}"
    mx_command = [
        "make",
        "-B",
        "-f",
        MX_MAKEFILE.name,
        "-j1",
        f"GPU={args.gpu_arch}",
        "HAO_BATCH=1",
        f"HAO_SEQ_LEN={shape.sequence}",
        f"HAO_HEADS={shape.q_heads}",
        f"HAO_KV_HEADS={shape.kv_heads}",
        "HAO_HEAD_DIM=64",
        f"HAO_NUM_SM={args.num_sm}",
        "HAO_CAUSAL=1",
        "HAO_FIXED_ROUTE_FASTPATH=1",
        "HAO_CAUSAL_INTERLEAVED_KV=1",
        "HAO_FP4PV_MX_POLICY=causal-accurate",
        "HAO_FP4PV_MX_MODE23_NATIVE_DENSITY_OVERRIDE=4",
        (
            "HAO_FP4PV_MX_MODE23_NATIVE_QUARTER_MASK_OVERRIDE="
            f"{args.mx_quarter_mask}"
        ),
        "HAO_FP4PV_MX_GLOBAL_ANCHOR32_OVERRIDE=1",
        "HAO_FP4PV_MX_GLOBAL_ANCHOR128_OVERRIDE=0",
        "HAO_FP4PV_MX_GLOBAL_ANCHOR_BIAS_X8_OVERRIDE=0",
        "HAO_FP4PV_MX_GLOBAL_ANCHOR_MARGIN_LOG2_OVERRIDE=64",
        "HAO_FP4PV_MX_STORED_SCALE_SHIFT_LOG2_OVERRIDE=16",
        "HAO_FP4PV_MX_ANCHOR_AFFINE_HOIST_OVERRIDE=1",
        "HAO_EXTENSION_SYMBOLIC_BIND=1",
        f"HAO_KERNEL_SYMBOL_TAG={symbol_tag}",
        "NVCC_THREADS=1",
        "NVCC_SPLIT_COMPILE=1",
        f"MODULE={mx_module}",
        f"OUT={mx_extension}",
    ]
    fp8_command = [
        str(python),
        str(EXACT_BUILDER),
        "--sequence",
        str(shape.sequence),
        "--q-heads",
        str(shape.q_heads),
        "--kv-heads",
        str(shape.kv_heads),
        "--head-dim",
        "64",
        "--gpu",
        args.gpu_arch,
        "--jobs",
        "1",
        "--nvcc-threads",
        "1",
        "--nvcc-split-compile",
        "1",
        "--probability-policy",
        "exact",
        "--module",
        fp8_module,
        "--output",
        str(fp8_extension),
    ]
    worker_command = [
        str(python),
        "-B",
        str(WORKER),
        "--mx-extension",
        str(mx_extension),
        "--mx-module",
        mx_module,
        "--fp8-extension",
        str(fp8_extension),
        "--fp8-module",
        fp8_module,
        "--sequence",
        str(shape.sequence),
        "--q-heads",
        str(shape.q_heads),
        "--kv-heads",
        str(shape.kv_heads),
        "--head-dim",
        "64",
        "--hidden",
        str(args.hidden),
        "--projection-format",
        "auto",
        "--seed",
        str(args.seed),
        "--warmups",
        str(args.warmups),
        "--samples",
        str(args.samples),
        "--gpu",
        "0",
        "--minimum-free-gib",
        str(args.minimum_free_gpu_gib),
        "--causal-leakage-check",
        "--output",
        str(result_path),
    ]
    if args.projection_extension is not None:
        worker_command.extend(
            [
                "--projection-extension",
                str(args.projection_extension),
                "--projection-module",
                args.projection_module,
            ]
        )
    return {
        "build_mx": {
            "argv": mx_command,
            "cwd": str(FORWARD_ROOT),
            "artifact": str(mx_extension),
            "module": mx_module,
            "kernel_symbol_tag": symbol_tag,
            "mx_route_name": args.mx_route_name,
            "mx_quarter_mask": args.mx_quarter_mask,
        },
        "build_fp8_exact": {
            "argv": fp8_command,
            "cwd": str(REPO_ROOT),
            "artifact": str(fp8_extension),
            "module": fp8_module,
        },
        "run_worker": {
            "argv": worker_command,
            "cwd": str(REPO_ROOT),
            "result": str(result_path),
        },
    }


def _system_memory_snapshot() -> dict[str, Any]:
    values: dict[str, int] = {}
    with Path("/proc/meminfo").open() as handle:
        for line in handle:
            key, raw = line.split(":", 1)
            fields = raw.strip().split()
            if fields and fields[0].isdigit():
                multiplier = 1024 if len(fields) > 1 and fields[1] == "kB" else 1
                values[key] = int(fields[0]) * multiplier
    if "MemAvailable" not in values or "MemTotal" not in values:
        raise RuntimeError("/proc/meminfo does not expose MemAvailable/MemTotal")
    return {
        "available_bytes": values["MemAvailable"],
        "available_gib": values["MemAvailable"] / GIB,
        "total_bytes": values["MemTotal"],
        "total_gib": values["MemTotal"] / GIB,
    }


def _gpu_snapshot(nvidia_smi: Path, gpu_index: int) -> dict[str, Any]:
    query = subprocess.run(
        [
            str(nvidia_smi),
            "-i",
            str(gpu_index),
            "--query-gpu=index,uuid,name,memory.free,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    rows = [line for line in query.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(
            f"expected one nvidia-smi row for GPU {gpu_index}, got {len(rows)}"
        )
    fields = [field.strip() for field in rows[0].split(",", 5)]
    if len(fields) != 6:
        raise RuntimeError(f"unexpected nvidia-smi GPU row: {rows[0]!r}")
    index, uuid, name, free_mib, total_mib, utilization = fields
    applications = subprocess.run(
        [
            str(nvidia_smi),
            "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    processes = []
    for line in applications.splitlines():
        app_fields = [field.strip() for field in line.split(",")]
        if len(app_fields) != 3 or app_fields[0] != uuid:
            continue
        processes.append(
            {
                "pid": int(app_fields[1]),
                "used_memory_mib": int(app_fields[2]),
            }
        )
    return {
        "index": int(index),
        "uuid": uuid,
        "name": name,
        "free_memory_mib": int(free_mib),
        "free_memory_gib": int(free_mib) / 1024.0,
        "total_memory_mib": int(total_mib),
        "utilization_percent": int(utilization),
        "compute_processes": processes,
    }


def _enforce_resources(
    args: argparse.Namespace,
    nvidia_smi: Path,
    *,
    require_idle_gpu: bool,
) -> dict[str, Any]:
    snapshot: dict[str, Any] = {"checked_at": _utc_now()}
    memory = _system_memory_snapshot()
    snapshot["system_memory"] = memory
    if memory["available_gib"] < args.minimum_free_system_gib:
        raise RuntimeError(
            "system RAM reserve violated: "
            f"{memory['available_gib']:.2f} GiB available < "
            f"{args.minimum_free_system_gib:.2f} GiB required"
        )
    if require_idle_gpu:
        gpu = _gpu_snapshot(nvidia_smi, args.gpu_index)
        snapshot["gpu"] = gpu
        if gpu["free_memory_gib"] < args.minimum_free_gpu_gib:
            raise RuntimeError(
                "GPU memory reserve violated: "
                f"{gpu['free_memory_gib']:.2f} GiB free < "
                f"{args.minimum_free_gpu_gib:.2f} GiB required"
            )
        if gpu["compute_processes"]:
            pids = ", ".join(str(item["pid"]) for item in gpu["compute_processes"])
            raise RuntimeError(
                f"GPU {args.gpu_index} already has compute processes: {pids}"
            )
    return snapshot


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _command_record(specification: dict[str, Any]) -> dict[str, Any]:
    return {
        **specification,
        "shell_rendering": shlex.join(specification["argv"]),
        "status": "pending",
    }


def _run_logged(
    name: str,
    command: dict[str, Any],
    log_path: Path,
    environment: dict[str, str],
    on_update: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    record = _command_record(command)
    record["status"] = "running"
    record["started_at"] = _utc_now()
    record["log"] = str(log_path)
    if on_update is not None:
        on_update(record)
    started = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        completed = subprocess.run(
            command["argv"],
            cwd=command["cwd"],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    record["elapsed_seconds"] = time.monotonic() - started
    record["finished_at"] = _utc_now()
    record["returncode"] = completed.returncode
    record["status"] = "complete" if completed.returncode == 0 else "failed"
    if on_update is not None:
        on_update(record)
    if completed.returncode:
        raise CommandFailed(name, record)
    return record


def _validate_topology(
    result: dict[str, Any],
    shape: Shape,
    mx_quarter_mask: int,
) -> None:
    topology = result.get("topology", {})
    mx = topology.get("nvfp4_qk_mxfp4_pv", {})
    fp8 = topology.get("nvfp4_qk_fp8_pv_exact", {})
    shape_expected = {
        "batch": 1,
        "seqlen": shape.sequence,
        "heads": shape.q_heads,
        "kv_heads": shape.kv_heads,
        "dqk": 64,
        "dvo": 64,
    }
    mismatches = []
    for route, actual, expected in (
        (
            "mx",
            mx,
            {**shape_expected, **_expected_mx_topology(mx_quarter_mask)},
        ),
        ("fp8", fp8, {**shape_expected, **EXPECTED_FP8_TOPOLOGY}),
    ):
        for key, expected_value in expected.items():
            if actual.get(key) != expected_value:
                mismatches.append(
                    f"{route}.{key}={actual.get(key)!r}, expected {expected_value!r}"
                )
    if mismatches:
        raise RuntimeError("runtime topology mismatch: " + "; ".join(mismatches))


def _summary_row(
    shape: Shape,
    result: dict[str, Any],
    manifest_path: Path,
    result_path: Path,
    mx_quarter_mask: int,
) -> dict[str, Any]:
    providers = result["timing"]["providers"]
    correctness = result["correctness"]
    expected_mx = _expected_mx_topology(mx_quarter_mask)
    return {
        "shape": shape.as_dict(),
        "status": "complete",
        "manifest": str(manifest_path),
        "result": str(result_path),
        "mx_route": {
            "name": MX_QUARTER_MASK_NAMES[mx_quarter_mask],
            "native_density": 4,
            "native_quarter_mask": mx_quarter_mask,
        },
        "median_us": {
            name: values["median_us"] for name, values in providers.items()
        },
        "speedup": result["speedup"],
        "output_cosine": {
            "mxfp4_pv_vs_bf16": correctness[
                "nvfp4_qk_mxfp4_pv_vs_bf16"
            ]["output"]["cosine"],
            "exact_fp8_pv_vs_bf16": correctness[
                "nvfp4_qk_fp8_pv_exact_vs_bf16"
            ]["output"]["cosine"],
            "mxfp4_pv_vs_exact_fp8_pv": correctness[
                "mxfp4_pv_vs_exact_fp8_pv"
            ]["output"]["cosine"],
        },
        "causal_leakage_all_passed": result.get("causal_leakage", {}).get(
            "all_passed"
        ),
        "topology": {
            "mx": {
                key: result["topology"]["nvfp4_qk_mxfp4_pv"].get(key)
                for key in expected_mx
            },
            "fp8": {
                key: result["topology"]["nvfp4_qk_fp8_pv_exact"].get(key)
                for key in EXPECTED_FP8_TOPOLOGY
            },
        },
    }


def _refresh_summary(summary: dict[str, Any]) -> None:
    statuses = [entry["status"] for entry in summary["shapes"]]
    summary["updated_at"] = _utc_now()
    summary["counts"] = {
        "total": len(statuses),
        "pending": statuses.count("pending"),
        "running": statuses.count("running"),
        "complete": statuses.count("complete"),
        "failed": statuses.count("failed"),
    }


@contextlib.contextmanager
def _gpu_lock(gpu_index: int) -> Iterator[Path]:
    path = Path("/tmp") / f"fp4_fa4_causal_forward_gpu{gpu_index}.lock"
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"another causal-forward matrix driver holds {path}"
            ) from error
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} started={_utc_now()}\n")
        handle.flush()
        yield path


def _plan(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, str]]:
    python = _resolve_executable(args.python)
    python_metadata, build_environment = _python_record(python)
    result_dir = args.result_dir.expanduser().resolve()
    artifact_dir = args.artifact_dir.expanduser().resolve()
    args.result_dir = result_dir
    args.artifact_dir = artifact_dir
    if args.projection_extension is not None:
        args.projection_extension = args.projection_extension.expanduser().resolve()
    plans = []
    for shape in args.shapes:
        result_path = result_dir / "shapes" / shape.label / "result.json"
        commands = _commands(
            args,
            shape,
            python,
            python_metadata["ext_suffix"],
            result_path,
        )
        plans.append(
            {
                "shape": shape.as_dict(),
                "label": shape.label,
                "commands": {
                    key: _command_record(value) for key, value in commands.items()
                },
            }
        )
    projection = (
        _file_record(args.projection_extension, require=not args.dry_run)
        if args.projection_extension is not None
        else None
    )
    return (
        {
            "schema": "causal_forward_matrix_driver_v1",
            "created_at": _utc_now(),
            "run_tag": args.run_tag,
            "result_dir": str(result_dir),
            "artifact_dir": str(artifact_dir),
            "serialized": True,
            "mx_route": {
                "name": args.mx_route_name,
                "native_density": 4,
                "native_quarter_mask": args.mx_quarter_mask,
                "causal_interleaved_kv": True,
            },
            "resource_policy": {
                "minimum_free_system_gib": args.minimum_free_system_gib,
                "minimum_free_gpu_gib": args.minimum_free_gpu_gib,
                "gpu_index": args.gpu_index,
                "reject_existing_compute_processes": True,
                "make_jobs": 1,
                "nvcc_threads": 1,
                "nvcc_split_compile": 1,
            },
            "expected_topology": {
                "mx": _expected_mx_topology(args.mx_quarter_mask),
                "fp8": EXPECTED_FP8_TOPOLOGY,
            },
            "python": python_metadata,
            "recorded_build_environment": build_environment,
            "projection_extension": projection,
            "git": _git_record(),
            "sources": _source_records(),
            "shapes": plans,
        },
        build_environment,
    )


def _ensure_new_output_roots(args: argparse.Namespace) -> None:
    for path, label in (
        (args.result_dir, "result"),
        (args.artifact_dir, "artifact"),
    ):
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(
                f"{label} directory is not empty; choose a new --run-tag: {path}"
            )
        path.mkdir(parents=True, exist_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    plan, build_environment = _plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    nvidia_smi = _resolve_executable(args.nvidia_smi)
    _ensure_new_output_roots(args)
    plan_path = args.result_dir / "plan.json"
    summary_path = args.result_dir / "summary.json"
    _atomic_write_json(plan_path, plan)
    summary: dict[str, Any] = {
        "schema": "causal_forward_matrix_summary_v1",
        "run_tag": args.run_tag,
        "created_at": _utc_now(),
        "plan": str(plan_path),
        "mx_route": plan["mx_route"],
        "shapes": [
            {"shape": shape.as_dict(), "status": "pending"}
            for shape in args.shapes
        ],
    }
    _refresh_summary(summary)
    _atomic_write_json(summary_path, summary)

    environment = os.environ.copy()
    environment.update(build_environment)
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    failures = 0
    with _gpu_lock(args.gpu_index) as lock_path:
        plan["gpu_lock"] = str(lock_path)
        _atomic_write_json(plan_path, plan)
        for index, shape in enumerate(args.shapes):
            shape_dir = args.result_dir / "shapes" / shape.label
            result_path = shape_dir / "result.json"
            manifest_path = shape_dir / "manifest.json"
            commands = _commands(
                args,
                shape,
                _resolve_executable(args.python),
                plan["python"]["ext_suffix"],
                result_path,
            )
            manifest: dict[str, Any] = {
                "schema": "causal_forward_matrix_shape_v1",
                "run_tag": args.run_tag,
                "shape": shape.as_dict(),
                "status": "running",
                "started_at": _utc_now(),
                "expected_topology": plan["expected_topology"],
                "mx_route": plan["mx_route"],
                "commands": {
                    name: _command_record(specification)
                    for name, specification in commands.items()
                },
                "resource_snapshots": [],
                "sources": plan["sources"],
                "projection_extension": plan["projection_extension"],
            }
            summary["shapes"][index] = {
                "shape": shape.as_dict(),
                "status": "running",
                "manifest": str(manifest_path),
            }
            _refresh_summary(summary)
            _atomic_write_json(summary_path, summary)
            _atomic_write_json(manifest_path, manifest)

            def update_command(
                command_name: str,
            ) -> Callable[[dict[str, Any]], None]:
                def update(record: dict[str, Any]) -> None:
                    manifest["commands"][command_name] = record
                    _atomic_write_json(manifest_path, manifest)

                return update

            print(f"[{index + 1}/{len(args.shapes)}] {shape.label}: build MX", flush=True)
            try:
                manifest["resource_snapshots"].append(
                    _enforce_resources(args, nvidia_smi, require_idle_gpu=True)
                )
                manifest["commands"]["build_mx"] = _run_logged(
                    "build_mx",
                    commands["build_mx"],
                    shape_dir / "build_mx.log",
                    environment,
                    update_command("build_mx"),
                )
                mx_path = Path(commands["build_mx"]["artifact"])
                manifest.setdefault("artifacts", {})["mx"] = _file_record(mx_path)
                _atomic_write_json(manifest_path, manifest)

                print(
                    f"[{index + 1}/{len(args.shapes)}] {shape.label}: build exact FP8",
                    flush=True,
                )
                manifest["resource_snapshots"].append(
                    _enforce_resources(args, nvidia_smi, require_idle_gpu=False)
                )
                manifest["commands"]["build_fp8_exact"] = _run_logged(
                    "build_fp8_exact",
                    commands["build_fp8_exact"],
                    shape_dir / "build_fp8_exact.log",
                    environment,
                    update_command("build_fp8_exact"),
                )
                fp8_path = Path(commands["build_fp8_exact"]["artifact"])
                manifest.setdefault("artifacts", {})["fp8_exact"] = _file_record(
                    fp8_path
                )
                _atomic_write_json(manifest_path, manifest)

                print(
                    f"[{index + 1}/{len(args.shapes)}] {shape.label}: run worker",
                    flush=True,
                )
                manifest["resource_snapshots"].append(
                    _enforce_resources(args, nvidia_smi, require_idle_gpu=True)
                )
                manifest["commands"]["run_worker"] = _run_logged(
                    "run_worker",
                    commands["run_worker"],
                    shape_dir / "worker.log",
                    environment,
                    update_command("run_worker"),
                )
                result = json.loads(result_path.read_text())
                manifest["runtime_topology"] = result.get("topology")
                _validate_topology(result, shape, args.mx_quarter_mask)
                manifest["result"] = _file_record(result_path)
                manifest["status"] = "complete"
                manifest["finished_at"] = _utc_now()
                summary["shapes"][index] = _summary_row(
                    shape,
                    result,
                    manifest_path,
                    result_path,
                    args.mx_quarter_mask,
                )
            except Exception as error:
                failures += 1
                manifest["status"] = "failed"
                manifest["finished_at"] = _utc_now()
                manifest["error"] = {
                    "type": type(error).__name__,
                    "message": str(error),
                }
                summary["shapes"][index] = {
                    "shape": shape.as_dict(),
                    "status": "failed",
                    "manifest": str(manifest_path),
                    "error": manifest["error"],
                }
                _atomic_write_json(manifest_path, manifest)
                _refresh_summary(summary)
                _atomic_write_json(summary_path, summary)
                print(f"{shape.label}: {error}", file=sys.stderr, flush=True)
                if not args.keep_going:
                    break
            else:
                _atomic_write_json(manifest_path, manifest)
                _refresh_summary(summary)
                _atomic_write_json(summary_path, summary)
                print(f"[{index + 1}/{len(args.shapes)}] {shape.label}: complete", flush=True)

    _refresh_summary(summary)
    summary["finished_at"] = _utc_now()
    _atomic_write_json(summary_path, summary)
    print(summary_path)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
