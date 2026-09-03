#!/usr/bin/env python3
"""Measure the saturated D64 QKV-projection x PV-format training boundary.

This harness isolates the four B16/S4096 Llama-1.2B attention-forward cases:

* E4M3 QKV projection with exact E4M3 FP8-PV;
* E4M3 QKV projection with MXFP4-PV;
* native NVFP4 QKV projection with exact E4M3 FP8-PV; and
* native NVFP4 QKV projection with MXFP4-PV.

All cases consume one deterministic BF16 activation and packed learned-QKV
weight draw.  Each case owns a caller-allocated publication workspace and
attention output/LSE pair.  The shape-bound projection binders perform their
allocating legacy-versus-compact ABI check before timing; measured projection
calls use only their unchecked out-parameter symbols.  Provider order rotates
for every CUDA-event sample.

The projection specializations publish the backward operands retained by the
training route, but this harness executes no backward kernel.  The current
operand-preparation helpers are functional allocating APIs.  Projection and
attention use caller-owned output buffers; without allocator tracing that is
an API contract, not proof that their implementations make no transient CUDA
allocations.  The result schema records that distinction explicitly.

After timing, an untimed BF16 projection + causal SDPA reference is computed
from the same activation, learned weight, and RoPE tables.  Every low-precision
case must satisfy configurable output cosine and relative-L2 thresholds or the
harness exits nonzero.

``--dry-run`` is side-effect free: it does not import Torch, load extensions,
touch CUDA, or create the requested output directory.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import platform
import shutil
import socket
import stat
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
GIB = 1 << 30
BATCH = 16
SEQUENCE = 4096
HIDDEN = 2048
Q_HEADS = 32
KV_HEADS = 8
HEAD_DIM = 64
QKV_WIDTH = (Q_HEADS + 2 * KV_HEADS) * HEAD_DIM
ROPE_THETA = 500_000.0
ROPE_FACTOR = 32.0
ROPE_ORIGINAL_CONTEXT = 8192
ROPE_LOW_FREQUENCY_FACTOR = 1.0
ROPE_HIGH_FREQUENCY_FACTOR = 4.0

PROJECTION_FORMATS = ("e4m3", "nvfp4")
PV_FORMATS = ("fp8", "mx")
STAGE_NAMES = (
    "operand_preparation",
    "projection_publication",
    "attention",
    "prepared_projection_attention",
    "full_combined",
)

# CUDA/PyTorch imports stay worker-lazy so source-contract tests and dry runs
# do not initialize CUDA or opportunistically import a worktree extension.
torch: Any = None
tk_interface: Any = None


@dataclass(frozen=True)
class Case:
    projection_format: str
    pv_format: str

    @property
    def key(self) -> str:
        return f"{self.projection_format}_qkv__{self.pv_format}_pv"

    @property
    def publish_mxfp4_v(self) -> bool:
        return self.pv_format == "mx"

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "qkv_projection_format": self.projection_format,
            "pv_format": ("mxfp4_e8m0_block32" if self.publish_mxfp4_v else "e4m3_fp8"),
            "publish_mxfp4_v": self.publish_mxfp4_v,
            "causal_kv_order": (
                "quarter_interleaved" if self.publish_mxfp4_v else "logical"
            ),
            "projection_publishes_training_backward_operands": True,
            "backward_kernel_executed": False,
        }


CASES = tuple(
    Case(projection_format, pv_format)
    for projection_format in PROJECTION_FORMATS
    for pv_format in PV_FORMATS
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _positive_finite(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def _cosine_threshold(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < -1.0 or parsed > 1.0:
        raise argparse.ArgumentTypeError("value must be finite and in [-1, 1]")
    return parsed


def _default_module(path: Path) -> str:
    return path.name.split(".", 1)[0]


def _selected_absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(Path(path).expanduser()))


def _resolve_executable(value: str) -> Path:
    candidate = shutil.which(value)
    if candidate is None:
        expanded = Path(value).expanduser()
        if expanded.is_file() and os.access(expanded, os.X_OK):
            candidate = str(expanded)
    if candidate is None:
        raise FileNotFoundError(f"Python executable not found: {value}")
    # Keep a venv's lexical bin/python path: resolving that symlink would lose
    # the adjacent pyvenv.cfg that selects the requested environment.
    selected = _selected_absolute(candidate)
    if not selected.is_file() or not os.access(selected, os.X_OK):
        raise FileNotFoundError(f"Python executable is not runnable: {selected}")
    return selected


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mx-extension",
        type=Path,
        required=True,
        help="authenticated B16/S4096 causal MXFP4-PV forward extension",
    )
    parser.add_argument("--mx-module")
    parser.add_argument(
        "--fp8-extension",
        type=Path,
        required=True,
        help="authenticated B16/S4096 exact FP8-PV forward extension",
    )
    parser.add_argument("--fp8-module")
    parser.add_argument(
        "--projection-extension",
        type=Path,
        required=True,
        help=(
            "low-precision extension exporting both compact E4M3 and native "
            "NVFP4 D64 projection-out ABIs"
        ),
    )
    parser.add_argument(
        "--projection-module",
        choices=("_C_b300_lowp_bwd",),
        default="_C_b300_lowp_bwd",
        help="fixed CPython module name exported by the projection extension",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="selected Python/venv used by the isolated worker process",
    )
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--warmups", type=_nonnegative_int, default=4)
    parser.add_argument(
        "--samples",
        type=_positive_int,
        default=40,
        help="CUDA-event samples per case and stage; must be divisible by four",
    )
    parser.add_argument("--gpu", type=_nonnegative_int, default=0)
    parser.add_argument(
        "--minimum-free-gib",
        type=_positive_finite,
        default=24.0,
        help="fail before allocation if the selected GPU has less free HBM",
    )
    parser.add_argument(
        "--minimum-free-system-gib",
        type=_positive_finite,
        default=32.0,
        help="fail before importing Torch if the host has less available RAM",
    )
    parser.add_argument(
        "--input-std",
        type=_positive_finite,
        default=1.0,
        help="BF16 post-RMSNorm activation standard deviation",
    )
    parser.add_argument(
        "--weight-std",
        type=_positive_finite,
        default=0.02,
        help="BF16 packed learned-QKV weight standard deviation",
    )
    parser.add_argument("--q-quant-scale", type=_positive_finite, default=2.25)
    parser.add_argument("--k-quant-scale", type=_positive_finite, default=2.0)
    parser.add_argument(
        "--minimum-bf16-output-cosine",
        type=_cosine_threshold,
        default=0.95,
        help="fail if any low-precision output has lower cosine vs BF16 SDPA",
    )
    parser.add_argument(
        "--maximum-bf16-output-relative-l2",
        type=_positive_finite,
        default=0.35,
        help="fail if any low-precision output exceeds this relative L2 vs BF16",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the side-effect-free worker plan without touching CUDA",
    )
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    modules = (
        args.mx_module,
        args.fp8_module,
        args.projection_module,
    )
    for module in modules:
        if module is not None and not module.isidentifier():
            parser.error("extension module names must be Python identifiers")
    effective_modules = (
        args.mx_module or _default_module(args.mx_extension),
        args.fp8_module or _default_module(args.fp8_extension),
        args.projection_module,
    )
    if len(set(effective_modules)) != len(effective_modules):
        parser.error("MX, FP8, and projection module names must be distinct")
    if args.samples % len(CASES):
        parser.error(
            f"--samples must be divisible by {len(CASES)} for balanced "
            "rotating provider order"
        )
    return args


def _worker_command(
    args: argparse.Namespace,
    selected_python: Path,
) -> list[str]:
    command = [
        str(selected_python),
        str(Path(__file__).absolute()),
        "--_worker",
        "--python",
        str(selected_python),
        "--mx-extension",
        str(_selected_absolute(args.mx_extension)),
        "--mx-module",
        args.mx_module or _default_module(args.mx_extension),
        "--fp8-extension",
        str(_selected_absolute(args.fp8_extension)),
        "--fp8-module",
        args.fp8_module or _default_module(args.fp8_extension),
        "--projection-extension",
        str(_selected_absolute(args.projection_extension)),
        "--projection-module",
        args.projection_module,
        "--seed",
        str(args.seed),
        "--warmups",
        str(args.warmups),
        "--samples",
        str(args.samples),
        "--gpu",
        str(args.gpu),
        "--minimum-free-gib",
        str(args.minimum_free_gib),
        "--minimum-free-system-gib",
        str(args.minimum_free_system_gib),
        "--input-std",
        str(args.input_std),
        "--weight-std",
        str(args.weight_std),
        "--q-quant-scale",
        str(args.q_quant_scale),
        "--k-quant-scale",
        str(args.k_quant_scale),
        "--minimum-bf16-output-cosine",
        str(args.minimum_bf16_output_cosine),
        "--maximum-bf16-output-relative-l2",
        str(args.maximum_bf16_output_relative_l2),
        "--output",
        str(_selected_absolute(args.output)),
    ]
    return command


def _allocation_contract() -> dict[str, Any]:
    return {
        "scope": "caller-owned CUDA output API contract",
        "allocator_tracing_performed": False,
        "transient_cuda_allocation_freedom_proven": False,
        "host_python_object_allocation_out_of_scope": True,
        "stages": {
            "operand_preparation": {
                "caller_owned_output_api": False,
                "functional_output_allocation_expected": True,
                "reason": (
                    "current b300_prepare_e4m3_projection_* and "
                    "b300_prepare_nvfp4_projection_* functional APIs allocate "
                    "their returned payload and scale tensors"
                ),
            },
            "projection_publication": {
                "caller_owned_output_api": True,
                "functional_output_allocation_expected": False,
                "reason": (
                    "shape-bound unchecked projection-out ABI writes into a "
                    "caller-owned publication workspace; transient allocator "
                    "activity was not traced"
                ),
            },
            "attention": {
                "caller_owned_output_api": True,
                "functional_output_allocation_expected": False,
                "reason": (
                    "attention output and LSE are caller-preallocated; transient "
                    "allocator activity was not traced"
                ),
            },
            "prepared_projection_attention": {
                "caller_owned_output_api": True,
                "functional_output_allocation_expected": False,
                "reason": (
                    "prepared operands, publication workspace, output, and "
                    "LSE are all reused; transient allocator activity was not traced"
                ),
            },
            "full_combined": {
                "caller_owned_output_api": False,
                "functional_output_allocation_expected": True,
                "reason": (
                    "includes the current allocating operand-preparation APIs; "
                    "projection publication and attention outputs remain "
                    "preallocated"
                ),
            },
        },
    }


def _dry_run_plan(
    args: argparse.Namespace,
    selected_python: Path,
    command: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema": "b16_s4096_forward_factorial_plan_v2",
        "dry_run": True,
        "touches_cuda": False,
        "imports_torch": False,
        "creates_output": False,
        "selected_python": str(selected_python),
        "worker_command": list(command),
        "shape": {
            "batch": BATCH,
            "sequence": SEQUENCE,
            "hidden": HIDDEN,
            "q_heads": Q_HEADS,
            "kv_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "qkv_width": QKV_WIDTH,
            "causal": True,
        },
        "cases": [case.as_dict() for case in CASES],
        "timed_stages": list(STAGE_NAMES),
        "timing": {
            "method": "balanced rotating-provider CUDA events",
            "warmups_per_stage": args.warmups,
            "samples_per_case_per_stage": args.samples,
        },
        "allocation_contract": _allocation_contract(),
        "correctness_policy": {
            "reference": "untimed_bf16_projection_causal_sdpa",
            "minimum_bf16_output_cosine": args.minimum_bf16_output_cosine,
            "maximum_bf16_output_relative_l2": (
                args.maximum_bf16_output_relative_l2
            ),
            "fail_closed": True,
        },
        "extension_bindings": {
            "mx": {
                "path": str(_selected_absolute(args.mx_extension)),
                "module": args.mx_module or _default_module(args.mx_extension),
            },
            "fp8": {
                "path": str(_selected_absolute(args.fp8_extension)),
                "module": args.fp8_module or _default_module(args.fp8_extension),
            },
            "projection": {
                "path": str(_selected_absolute(args.projection_extension)),
                "module": args.projection_module,
            },
        },
        "output": str(_selected_absolute(args.output)),
    }


def _load_runtime(projection_path: Path) -> None:
    global tk_interface, torch
    # Select the exact regular file before importing the tk_fa4 package. This
    # uses interface.py's fail-closed override loader and avoids loading the
    # same pybind extension a second time under a different module identity.
    os.environ["TK_FA4_LOWP_BWD_EXTENSION_SOURCE"] = str(projection_path)
    import torch as loaded_torch
    import tk_fa4.interface as loaded_interface

    torch = loaded_torch
    tk_interface = loaded_interface


def _load_extension(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import extension {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file_provenance(path: Path) -> dict[str, Any]:
    selected = _selected_absolute(path)
    selected_stat = selected.lstat()
    if not stat.S_ISREG(selected_stat.st_mode):
        raise RuntimeError(f"extension must be a regular non-symlink file: {selected}")
    resolved = selected.resolve(strict=True)
    observed = resolved.stat()
    return {
        "selected_path": str(selected),
        "resolved_path": str(resolved),
        "sha256": _sha256(resolved),
        "bytes": observed.st_size,
        "mtime_ns": observed.st_mtime_ns,
    }


def _authenticate_loaded_extension(
    label: str,
    module: Any,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    loaded_file = getattr(module, "__file__", None)
    if loaded_file is None:
        raise RuntimeError(f"loaded {label} extension has no __file__")
    loaded_path = Path(str(loaded_file)).resolve(strict=True)
    expected_path = Path(provenance["resolved_path"]).resolve(strict=True)
    if loaded_path != expected_path:
        raise RuntimeError(
            f"loaded {label} extension path changed: "
            f"{loaded_path} != {expected_path}"
        )
    post_load = _regular_file_provenance(Path(provenance["selected_path"]))
    if Path(post_load["resolved_path"]) != expected_path:
        raise RuntimeError(
            f"selected {label} extension resolved path changed after load"
        )
    if post_load["sha256"] != provenance["sha256"]:
        raise RuntimeError(
            f"selected {label} extension bytes changed between hash and load"
        )
    return {
        "loaded_file": str(loaded_path),
        "post_load_sha256": post_load["sha256"],
        "post_load_bytes": post_load["bytes"],
        "post_load_mtime_ns": post_load["mtime_ns"],
        "path_and_sha256_match_preload_receipt": True,
    }


def _git_output(*arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ("git", "-C", str(REPO_ROOT), *arguments),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.rstrip("\n")


def _git_provenance() -> dict[str, Any]:
    tracked = _git_output("status", "--porcelain=v1", "--untracked-files=no")
    full = _git_output("status", "--porcelain=v1")
    return {
        "root": str(REPO_ROOT),
        "head": _git_output("rev-parse", "HEAD"),
        "branch": _git_output("branch", "--show-current"),
        "tracked_dirty": bool(tracked) if tracked is not None else None,
        "tracked_status": tracked,
        "dirty_including_untracked": bool(full) if full is not None else None,
        "full_status": full,
    }


def _host_available_bytes() -> int | None:
    try:
        lines = Path("/proc/meminfo").read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        if line.startswith("MemAvailable:"):
            fields = line.split()
            if len(fields) >= 2:
                return int(fields[1]) * 1024
    return None


def _write_new_atomic(path: Path, content: str) -> None:
    destination = _selected_absolute(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        # Atomic and fail-closed: never replace a result another process wrote.
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _rotating_orders(names: Sequence[str], rounds: int) -> list[list[str]]:
    if not names:
        raise ValueError("at least one timing provider is required")
    values = list(names)
    return [
        values[index % len(values) :] + values[: index % len(values)]
        for index in range(rounds)
    ]


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _timing_summary(
    values: Sequence[float],
    *,
    unit: str,
) -> dict[str, Any]:
    if not values:
        raise ValueError("cannot summarize empty timings")
    if unit == "microseconds":
        suffix = "us"
    elif unit == "nanoseconds":
        suffix = "ns"
    else:
        raise ValueError(f"unsupported timing unit: {unit}")
    return {
        "unit": unit,
        f"median_{suffix}": statistics.median(values),
        f"mean_{suffix}": statistics.fmean(values),
        f"minimum_{suffix}": min(values),
        f"p10_{suffix}": _percentile(values, 0.10),
        f"p90_{suffix}": _percentile(values, 0.90),
        f"maximum_{suffix}": max(values),
        f"samples_{suffix}": list(values),
    }


def _measure_interleaved(
    functions: dict[str, Callable[[], Any]],
    *,
    warmups: int,
    samples: int,
) -> dict[str, Any]:
    names = list(functions)
    expected = [case.key for case in CASES]
    if names != expected:
        raise ValueError(
            f"timing providers must preserve factorial order {expected}, got {names}"
        )
    for order in _rotating_orders(names, warmups):
        retained = [functions[name]() for name in order]
        torch.cuda.synchronize()
        del retained

    device_values: dict[str, list[float]] = {name: [] for name in names}
    host_values: dict[str, list[float]] = {name: [] for name in names}
    orders = _rotating_orders(names, samples)
    for order in orders:
        events = []
        retained = []
        for name in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            host_start = time.perf_counter_ns()
            retained.append(functions[name]())
            host_values[name].append(float(time.perf_counter_ns() - host_start))
            end.record()
            events.append((name, start, end))
        events[-1][2].synchronize()
        for name, start, end in events:
            device_values[name].append(float(start.elapsed_time(end) * 1000.0))
        del retained
    return {
        "providers": {
            name: {
                "device": _timing_summary(device_values[name], unit="microseconds"),
                "host_launch": _timing_summary(host_values[name], unit="nanoseconds"),
            }
            for name in names
        },
        "sample_orders": orders,
    }


def _validate_topology(
    label: str,
    topology: dict[str, Any],
    *,
    require_runtime_valid: bool,
) -> None:
    expected = {
        "batch": BATCH,
        "seqlen": SEQUENCE,
        "heads": Q_HEADS,
        "kv_heads": KV_HEADS,
        "dqk": HEAD_DIM,
        "dvo": HEAD_DIM,
        "causal": True,
        "qk_format": "nvfp4_e4m3_block16",
        "fixed_route_fastpath": True,
        "route_env_guard_per_launch": False,
    }
    for name, expected_value in expected.items():
        if topology.get(name) != expected_value:
            raise ValueError(
                f"{label} topology {name}={topology.get(name)!r}; "
                f"expected {expected_value!r}"
            )
    if require_runtime_valid and topology.get("valid") != 1:
        raise ValueError(f"{label} topology was not populated by a real launch")
    if bool(topology.get("fixed_p_ceiling", False)) or bool(
        topology.get("score_pack_ceiling", False)
    ):
        raise ValueError(f"{label} is a diagnostic ceiling build")
    if label == "mx":
        expected_mx = {
            "route": "real_fwd_tk_hao_direct_nvfp4_mxfp4pv",
            "pv_format": "mxfp4_e8m0_block32",
            "causal_interleaved_kv": True,
            "mx_mode23_native_density": 4,
            "mx_mode23_native_quarter_mask": 3,
        }
        for name, expected_value in expected_mx.items():
            if topology.get(name) != expected_value:
                raise ValueError(
                    f"MX topology {name}={topology.get(name)!r}; "
                    f"expected {expected_value!r}"
                )
    else:
        expected_fp8 = {
            "route": "real_fwd_tk_hao_direct_causal_gqa_nvfp4_fp8pv",
            "pv_format": "e4m3_fp8",
            "causal_interleaved_kv": False,
            "shiftless_fp8_mode": 0,
        }
        for name, expected_value in expected_fp8.items():
            if topology.get(name) != expected_value:
                raise ValueError(
                    f"FP8 topology {name}={topology.get(name)!r}; "
                    f"expected {expected_value!r}"
                )


def _make_llama3_rope() -> tuple[Any, Any]:
    pair_count = HEAD_DIM // 2
    positions = torch.arange(SEQUENCE, device="cuda", dtype=torch.float32)
    frequencies = 1.0 / (
        ROPE_THETA
        ** (torch.arange(pair_count, device="cuda", dtype=torch.float32) / pair_count)
    )
    wavelengths = 2.0 * math.pi / frequencies
    low_wavelength = ROPE_ORIGINAL_CONTEXT / ROPE_LOW_FREQUENCY_FACTOR
    high_wavelength = ROPE_ORIGINAL_CONTEXT / ROPE_HIGH_FREQUENCY_FACTOR
    scaled = torch.where(
        wavelengths > low_wavelength,
        frequencies / ROPE_FACTOR,
        frequencies,
    )
    smooth = (ROPE_ORIGINAL_CONTEXT / wavelengths - ROPE_LOW_FREQUENCY_FACTOR) / (
        ROPE_HIGH_FREQUENCY_FACTOR - ROPE_LOW_FREQUENCY_FACTOR
    )
    smoothed = (1.0 - smooth) * scaled / ROPE_FACTOR + smooth * scaled
    medium = ~((wavelengths < high_wavelength) | (wavelengths > low_wavelength))
    frequencies = torch.where(medium, smoothed, scaled)
    angles = positions[:, None] * frequencies[None, :]
    cosine = angles.cos()[None].repeat(BATCH, 1, 1).bfloat16().contiguous()
    sine = angles.sin()[None].repeat(BATCH, 1, 1).bfloat16().contiguous()
    return cosine, sine


def _new_normal(
    shape: Sequence[int],
    *,
    std: float,
    generator: Any,
) -> Any:
    tensor = torch.empty(*shape, device="cuda", dtype=torch.bfloat16)
    return tensor.normal_(mean=0.0, std=std, generator=generator)


def _make_shared_state(args: argparse.Namespace) -> dict[str, Any]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed)
    rows = _new_normal(
        (BATCH * SEQUENCE, HIDDEN),
        std=args.input_std,
        generator=generator,
    )
    qkv_weight = _new_normal(
        (QKV_WIDTH, HIDDEN),
        std=args.weight_std,
        generator=generator,
    )
    qk_scales = torch.zeros(
        BATCH,
        Q_HEADS // 2,
        7,
        device="cuda",
        dtype=torch.float32,
    )
    qk_scales[..., 0] = args.q_quant_scale
    qk_scales[..., 1] = args.k_quant_scale
    rope_cosine, rope_sine = _make_llama3_rope()
    paired_rope = tk_interface.b300_pack_gqa_d64_paired_rope(rope_cosine, rope_sine)
    return {
        "rows": rows,
        "qkv_weight": qkv_weight,
        "qk_scales": qk_scales,
        "rope_cosine": rope_cosine,
        "rope_sine": rope_sine,
        "paired_rope": paired_rope,
        "receipt": {
            "seed": args.seed,
            "rows": {
                "shape": list(rows.shape),
                "dtype": str(rows.dtype),
                "requested_std": args.input_std,
            },
            "qkv_weight": {
                "shape": list(qkv_weight.shape),
                "dtype": str(qkv_weight.dtype),
                "layout": "canonical_packed_q_then_k_then_v_rows",
                "requested_std": args.weight_std,
            },
            "qk_policy_scales": {
                "shape": list(qk_scales.shape),
                "q": args.q_quant_scale,
                "k": args.k_quant_scale,
            },
            "rope": "llama3.2_scaled_rope_theta500000_factor32",
        },
    }


def _apply_pair_rope_bf16_reference(
    tensor: Any,
    cosine: Any,
    sine: Any,
) -> Any:
    pairs = tensor.float().reshape(*tensor.shape[:-1], HEAD_DIM // 2, 2)
    first, second = pairs[..., 0], pairs[..., 1]
    cosine_f = cosine.float().unsqueeze(2)
    sine_f = sine.float().unsqueeze(2)
    return torch.stack(
        (
            first * cosine_f - second * sine_f,
            first * sine_f + second * cosine_f,
        ),
        dim=-1,
    ).flatten(-2).bfloat16()


def _run_bf16_causal_sdpa_reference(state: dict[str, Any]) -> Any:
    """Compute one untimed BF16 control from the exact shared master tensors."""
    with torch.no_grad():
        projected = torch.nn.functional.linear(
            state["rows"],
            state["qkv_weight"],
        )
        q_width = Q_HEADS * HEAD_DIM
        kv_width = KV_HEADS * HEAD_DIM
        q, k, v = projected.split((q_width, kv_width, kv_width), dim=-1)
        q = q.reshape(BATCH, SEQUENCE, Q_HEADS, HEAD_DIM)
        k = k.reshape(BATCH, SEQUENCE, KV_HEADS, HEAD_DIM)
        v = v.reshape(BATCH, SEQUENCE, KV_HEADS, HEAD_DIM)
        q = _apply_pair_rope_bf16_reference(
            q,
            state["rope_cosine"],
            state["rope_sine"],
        )
        k = _apply_pair_rope_bf16_reference(
            k,
            state["rope_cosine"],
            state["rope_sine"],
        )
        output = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=True,
            enable_gqa=True,
        )
        return output.transpose(1, 2).contiguous()


def _allocate_publication_workspace(device: Any) -> Any:
    q_payload = torch.empty(
        BATCH,
        Q_HEADS,
        SEQUENCE,
        HEAD_DIM // 2,
        device=device,
        dtype=torch.uint8,
    )
    k_payload = torch.empty(
        BATCH,
        KV_HEADS,
        SEQUENCE,
        HEAD_DIM // 2,
        device=device,
        dtype=torch.uint8,
    )
    q_scale_pages = torch.empty(
        BATCH,
        SEQUENCE // 128,
        Q_HEADS,
        512,
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    q_global_scale = torch.empty(BATCH, Q_HEADS, device=device, dtype=torch.float32)
    k_scale_pages = torch.empty(
        BATCH,
        SEQUENCE // 64,
        KV_HEADS,
        512,
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    k_global_scale = torch.empty(BATCH, KV_HEADS, device=device, dtype=torch.float32)
    v_mxfp4_payload = torch.empty(
        BATCH,
        KV_HEADS,
        HEAD_DIM,
        SEQUENCE // 2,
        device=device,
        dtype=torch.float4_e2m1fn_x2,
    )
    v_mxfp4_scale_pages = torch.empty(
        BATCH,
        SEQUENCE // 128,
        KV_HEADS,
        512,
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    v_fp8_payload = torch.empty(
        BATCH,
        KV_HEADS,
        HEAD_DIM,
        SEQUENCE,
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    v_backward_fp8 = torch.empty(
        BATCH,
        SEQUENCE,
        KV_HEADS,
        HEAD_DIM,
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    q_backward_fp8 = torch.empty(
        BATCH,
        SEQUENCE,
        Q_HEADS,
        HEAD_DIM,
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    k_backward_fp8 = torch.empty(
        BATCH,
        SEQUENCE,
        KV_HEADS,
        HEAD_DIM,
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    return tk_interface.B300E4M3QKVForwardWorkspace(
        q_payload=q_payload,
        k_payload=k_payload,
        q_scale_pages=q_scale_pages,
        q_global_scale=q_global_scale,
        k_scale_pages=k_scale_pages,
        k_global_scale=k_global_scale,
        v_mxfp4_payload=v_mxfp4_payload,
        v_mxfp4_scale_pages=v_mxfp4_scale_pages,
        v_fp8_payload=v_fp8_payload,
        v_backward_fp8=v_backward_fp8,
        q_backward_fp8=q_backward_fp8,
        k_backward_fp8=k_backward_fp8,
        q_payload_fp4=q_payload.view(torch.float4_e2m1fn_x2),
        k_payload_fp4=k_payload.view(torch.float4_e2m1fn_x2),
        empty_bf16=torch.empty((0,), device=device, dtype=torch.bfloat16),
        empty_byte=torch.empty((0,), device=device, dtype=torch.uint8),
        empty_fp8=torch.empty((0,), device=device, dtype=torch.float8_e4m3fn),
        empty_fp4=torch.empty((0,), device=device, dtype=torch.float4_e2m1fn_x2),
    )


def _workspace_owner_tensors(workspace: Any) -> dict[str, Any]:
    names = (
        "q_payload",
        "k_payload",
        "q_scale_pages",
        "q_global_scale",
        "k_scale_pages",
        "k_global_scale",
        "v_mxfp4_payload",
        "v_mxfp4_scale_pages",
        "v_backward_fp8",
        "q_backward_fp8",
        "k_backward_fp8",
        "v_fp8_payload",
    )
    return {name: getattr(workspace, name) for name in names}


def _workspace_receipt(workspace: Any) -> dict[str, Any]:
    owners = _workspace_owner_tensors(workspace)
    pointers = [int(tensor.data_ptr()) for tensor in owners.values()]
    return {
        "owner_count": len(owners),
        "owner_pointers_unique": len(set(pointers)) == len(pointers),
        "owner_pointers": {
            name: int(tensor.data_ptr()) for name, tensor in owners.items()
        },
        "owner_shapes": {name: list(tensor.shape) for name, tensor in owners.items()},
        "owner_dtypes": {name: str(tensor.dtype) for name, tensor in owners.items()},
        "q_typed_alias_matches_owner": (
            int(workspace.q_payload_fp4.data_ptr())
            == int(workspace.q_payload.data_ptr())
        ),
        "k_typed_alias_matches_owner": (
            int(workspace.k_payload_fp4.data_ptr())
            == int(workspace.k_payload.data_ptr())
        ),
        "total_owner_bytes": sum(
            tensor.numel() * tensor.element_size() for tensor in owners.values()
        ),
    }


def _require_workspace_stable(
    workspace: Any,
    receipt: dict[str, Any],
) -> None:
    observed = {
        name: int(tensor.data_ptr())
        for name, tensor in _workspace_owner_tensors(workspace).items()
    }
    if observed != receipt["owner_pointers"]:
        raise RuntimeError("caller-owned projection workspace pointer changed")
    if not receipt["owner_pointers_unique"]:
        raise RuntimeError("caller-owned projection workspace owners alias")
    if not (
        receipt["q_typed_alias_matches_owner"]
        and receipt["k_typed_alias_matches_owner"]
    ):
        raise RuntimeError("typed Q/K aliases do not match workspace owners")


def _prepare_operands(
    projection_format: str,
    state: dict[str, Any],
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    if projection_format == "e4m3":
        input_operand = tuple(
            tk_interface.b300_prepare_e4m3_projection_operand(state["rows"])
        )
        weight_operand = tuple(
            tk_interface.b300_prepare_e4m3_projection_weight(state["qkv_weight"])
        )
    elif projection_format == "nvfp4":
        input_operand = tuple(
            tk_interface.b300_prepare_nvfp4_projection_operand(state["rows"])
        )
        weight_operand = tuple(
            tk_interface.b300_prepare_nvfp4_projection_weight(state["qkv_weight"])
        )
    else:
        raise ValueError(f"unknown projection format: {projection_format}")
    return input_operand, weight_operand


def _bind_projector(case: Case) -> Any:
    common = {
        "batch": BATCH,
        "seqlen": SEQUENCE,
        "q_heads": Q_HEADS,
        "kv_heads": KV_HEADS,
        "publish_mxfp4_v": case.publish_mxfp4_v,
        "v_mxfp4_scale_2d": False,
    }
    if case.projection_format == "e4m3":
        return tk_interface.b300_bind_qkv_gqa_d64_paired_unified_lowp_e4m3_projection(
            **common,
            represented_backward=True,
            per_block_qk_scales=True,
            experimental_split_v_backward=case.publish_mxfp4_v,
        )
    return tk_interface.b300_bind_qkv_gqa_d64_paired_unified_lowp_nvfp4_projection(
        **common
    )


def _project(
    projector: Any,
    operands: tuple[tuple[Any, ...], tuple[Any, ...]],
    state: dict[str, Any],
    workspace: Any,
) -> Any:
    input_operand, weight_operand = operands
    return projector(
        input_operand,
        weight_operand,
        state["qk_scales"],
        state["paired_rope"],
        forward_workspace=workspace,
    )


def _run_attention(
    case: Case,
    extensions: dict[str, Any],
    bundle: Any,
    output: Any,
    lse: Any,
) -> None:
    if case.publish_mxfp4_v:
        extensions["mx"].forward_hao_direct_fp4pv(
            *bundle.forward_operands(), output, lse, 0, True, True
        )
        return
    forward_v = bundle.v_forward_fp8
    if forward_v is None:
        raise RuntimeError(
            "exact FP8-PV requires projection-native feature-major E4M3 V"
        )
    extensions["fp8"].forward_hao_direct_fp8pv(
        *bundle.qk_forward_operands(),
        forward_v,
        output,
        lse,
        0,
        True,
        True,
    )


def _chunked_metrics(
    reference: Any,
    actual: Any,
    *,
    chunk_elements: int = 1 << 20,
) -> dict[str, Any]:
    if reference.shape != actual.shape:
        raise ValueError("metric tensors have different shapes")
    reference_flat = reference.detach().reshape(-1)
    actual_flat = actual.detach().reshape(-1)
    reference_sq = 0.0
    actual_sq = 0.0
    difference_sq = 0.0
    dot = 0.0
    finite = True
    for start in range(0, reference_flat.numel(), chunk_elements):
        stop = min(start + chunk_elements, reference_flat.numel())
        reference_chunk = reference_flat[start:stop].float()
        actual_chunk = actual_flat[start:stop].float()
        difference = actual_chunk - reference_chunk
        reference_sq += float(reference_chunk.square().sum())
        actual_sq += float(actual_chunk.square().sum())
        difference_sq += float(difference.square().sum())
        dot += float((reference_chunk * actual_chunk).sum())
        finite = (
            finite
            and bool(torch.isfinite(reference_chunk).all())
            and bool(torch.isfinite(actual_chunk).all())
        )
    reference_norm = math.sqrt(max(reference_sq, 1.0e-40))
    actual_norm = math.sqrt(max(actual_sq, 1.0e-40))
    return {
        "finite": finite,
        "cosine": dot / (reference_norm * actual_norm),
        "relative_l2": math.sqrt(difference_sq) / reference_norm,
        "norm_ratio": actual_norm / reference_norm,
    }


def _bf16_output_verdict(
    metrics: dict[str, Any],
    *,
    minimum_cosine: float,
    maximum_relative_l2: float,
) -> dict[str, Any]:
    checks = {
        "finite": bool(metrics["finite"]),
        "minimum_cosine": float(metrics["cosine"]) >= minimum_cosine,
        "maximum_relative_l2": (
            float(metrics["relative_l2"]) <= maximum_relative_l2
        ),
    }
    return {
        "thresholds": {
            "minimum_cosine": minimum_cosine,
            "maximum_relative_l2": maximum_relative_l2,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def _case_functions(
    state: dict[str, Any],
    extensions: dict[str, Any],
    projectors: dict[str, Any],
    prepared: dict[str, tuple[tuple[Any, ...], tuple[Any, ...]]],
    workspaces: dict[str, Any],
    bundles: dict[str, Any],
    outputs: dict[str, Any],
    lses: dict[str, Any],
) -> dict[str, dict[str, Callable[[], Any]]]:
    stages: dict[str, dict[str, Callable[[], Any]]] = {name: {} for name in STAGE_NAMES}
    for case in CASES:
        key = case.key

        def prepare(case: Case = case) -> Any:
            return _prepare_operands(case.projection_format, state)

        def project(case: Case = case, key: str = key) -> Any:
            return _project(
                projectors[key],
                prepared[case.projection_format],
                state,
                workspaces[key],
            )

        def attention(case: Case = case, key: str = key) -> None:
            _run_attention(
                case,
                extensions,
                bundles[key],
                outputs[key],
                lses[key],
            )

        def prepared_boundary(case: Case = case, key: str = key) -> Any:
            bundle = _project(
                projectors[key],
                prepared[case.projection_format],
                state,
                workspaces[key],
            )
            _run_attention(
                case,
                extensions,
                bundle,
                outputs[key],
                lses[key],
            )
            return bundle

        def full_combined(case: Case = case, key: str = key) -> Any:
            operands = _prepare_operands(case.projection_format, state)
            bundle = _project(projectors[key], operands, state, workspaces[key])
            _run_attention(
                case,
                extensions,
                bundle,
                outputs[key],
                lses[key],
            )
            # Retain allocating preparation outputs until the CUDA-event round
            # is synchronized, even though the publication kernel consumed them.
            return operands, bundle

        stages["operand_preparation"][key] = prepare
        stages["projection_publication"][key] = project
        stages["attention"][key] = attention
        stages["prepared_projection_attention"][key] = prepared_boundary
        stages["full_combined"][key] = full_combined
    return stages


def _run_worker(args: argparse.Namespace, selected_python: Path) -> int:
    output_path = _selected_absolute(args.output)
    if output_path.exists():
        raise RuntimeError(f"refusing to overwrite benchmark output: {output_path}")
    host_available = _host_available_bytes()
    if (
        host_available is not None
        and host_available < args.minimum_free_system_gib * GIB
    ):
        raise RuntimeError(
            f"host has {host_available / GIB:.2f} GiB available; requires "
            f"{args.minimum_free_system_gib:.2f} GiB"
        )

    artifacts = {
        "mx": _regular_file_provenance(args.mx_extension),
        "fp8": _regular_file_provenance(args.fp8_extension),
        "projection": _regular_file_provenance(args.projection_extension),
    }
    _load_runtime(Path(artifacts["projection"]["resolved_path"]))
    if args.gpu >= torch.cuda.device_count():
        raise RuntimeError(
            f"GPU {args.gpu} is unavailable ({torch.cuda.device_count()} visible)"
        )
    torch.cuda.set_device(args.gpu)
    free_bytes, total_bytes = torch.cuda.mem_get_info(args.gpu)
    if free_bytes < args.minimum_free_gib * GIB:
        raise RuntimeError(
            f"GPU {args.gpu} has {free_bytes / GIB:.2f} GiB free; requires "
            f"{args.minimum_free_gib:.2f} GiB"
        )

    projection = tk_interface._C_b300_lowp_bwd
    extensions = {
        "mx": _load_extension(
            Path(artifacts["mx"]["resolved_path"]),
            args.mx_module or _default_module(args.mx_extension),
        ),
        "fp8": _load_extension(
            Path(artifacts["fp8"]["resolved_path"]),
            args.fp8_module or _default_module(args.fp8_extension),
        ),
    }
    loaded_artifact_authentication = {
        "projection": _authenticate_loaded_extension(
            "projection",
            projection,
            artifacts["projection"],
        ),
        **{
            name: _authenticate_loaded_extension(
                name,
                extension,
                artifacts[name],
            )
            for name, extension in extensions.items()
        },
    }
    topologies = {
        name: dict(extension.read_hao_direct_topology())
        for name, extension in extensions.items()
    }
    for name, topology in topologies.items():
        _validate_topology(name, topology, require_runtime_valid=False)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    state = _make_shared_state(args)
    device = state["rows"].device
    projectors = {case.key: _bind_projector(case) for case in CASES}
    workspaces = {case.key: _allocate_publication_workspace(device) for case in CASES}
    workspace_receipts = {
        key: _workspace_receipt(workspace) for key, workspace in workspaces.items()
    }
    if not all(
        receipt["owner_pointers_unique"]
        and receipt["q_typed_alias_matches_owner"]
        and receipt["k_typed_alias_matches_owner"]
        for receipt in workspace_receipts.values()
    ):
        raise RuntimeError("invalid caller-owned publication workspace")

    prepared = {
        projection_format: _prepare_operands(projection_format, state)
        for projection_format in PROJECTION_FORMATS
    }
    output_shape = (BATCH, SEQUENCE, Q_HEADS, HEAD_DIM)
    lse_shape = (BATCH, Q_HEADS, 1, SEQUENCE)
    outputs = {
        case.key: torch.empty(output_shape, device=device, dtype=torch.bfloat16)
        for case in CASES
    }
    lses = {
        case.key: torch.empty(lse_shape, device=device, dtype=torch.float32)
        for case in CASES
    }
    output_pointers = {key: int(output.data_ptr()) for key, output in outputs.items()}
    lse_pointers = {key: int(lse.data_ptr()) for key, lse in lses.items()}

    # First use intentionally invokes the allocating legacy ABI and the
    # checked compact out ABI, then verifies them bitwise. Keep that work
    # completely outside every timing interval. The second calls below are
    # therefore guaranteed to use only the unchecked compact symbols.
    bundles = {}
    with torch.no_grad():
        for case in CASES:
            key = case.key
            _project(
                projectors[key],
                prepared[case.projection_format],
                state,
                workspaces[key],
            )
        torch.cuda.synchronize()
        for case in CASES:
            key = case.key
            projector = projectors[key]
            if not projector.forward_workspace_abi_validated:
                raise RuntimeError(f"{key} projection ABI was not validated")
            if projector.validated_forward_workspace_count != 1:
                raise RuntimeError(
                    f"{key} validated an unexpected number of workspaces"
                )
            bundles[key] = _project(
                projector,
                prepared[case.projection_format],
                state,
                workspaces[key],
            )
            _run_attention(
                case,
                extensions,
                bundles[key],
                outputs[key],
                lses[key],
            )
        torch.cuda.synchronize()

        # Refresh runtime-populated topology only after each exact forward
        # extension has executed at the selected shape.
        topologies = {
            name: dict(extension.read_hao_direct_topology())
            for name, extension in extensions.items()
        }
        for name, topology in topologies.items():
            _validate_topology(name, topology, require_runtime_valid=True)

        stage_functions = _case_functions(
            state,
            extensions,
            projectors,
            prepared,
            workspaces,
            bundles,
            outputs,
            lses,
        )
        timing_stages = {
            stage_name: _measure_interleaved(
                stage_functions[stage_name],
                warmups=args.warmups,
                samples=args.samples,
            )
            for stage_name in STAGE_NAMES
        }

        # Leave outputs/LSE in one explicit comparable state after the final
        # timed stage, which otherwise ends on a rotated provider order.
        for case in CASES:
            key = case.key
            bundles[key] = _project(
                projectors[key],
                prepared[case.projection_format],
                state,
                workspaces[key],
            )
            _run_attention(
                case,
                extensions,
                bundles[key],
                outputs[key],
                lses[key],
            )
        torch.cuda.synchronize()

    for key, workspace in workspaces.items():
        _require_workspace_stable(workspace, workspace_receipts[key])
    if output_pointers != {
        key: int(output.data_ptr()) for key, output in outputs.items()
    }:
        raise RuntimeError("preallocated attention output pointer changed")
    if lse_pointers != {key: int(lse.data_ptr()) for key, lse in lses.items()}:
        raise RuntimeError("preallocated attention LSE pointer changed")

    # Keep the BF16 control completely outside all timing stages.  Only its
    # final BSHD output is retained for chunked metrics.
    bf16_reference = _run_bf16_causal_sdpa_reference(state)
    torch.cuda.synchronize()

    comparisons = {}
    pairwise_all_finite = True
    for projection_format in PROJECTION_FORMATS:
        fp8_key = Case(projection_format, "fp8").key
        mx_key = Case(projection_format, "mx").key
        output_metrics = _chunked_metrics(outputs[fp8_key], outputs[mx_key])
        lse_metrics = _chunked_metrics(lses[fp8_key], lses[mx_key])
        comparisons[f"{projection_format}_mx_vs_fp8"] = {
            "output": output_metrics,
            "lse": lse_metrics,
        }
        pairwise_all_finite = (
            pairwise_all_finite
            and output_metrics["finite"]
            and lse_metrics["finite"]
        )
    for pv_format in PV_FORMATS:
        e4m3_key = Case("e4m3", pv_format).key
        nvfp4_key = Case("nvfp4", pv_format).key
        output_metrics = _chunked_metrics(outputs[e4m3_key], outputs[nvfp4_key])
        lse_metrics = _chunked_metrics(lses[e4m3_key], lses[nvfp4_key])
        comparisons[f"nvfp4_vs_e4m3_{pv_format}"] = {
            "output": output_metrics,
            "lse": lse_metrics,
        }
        pairwise_all_finite = (
            pairwise_all_finite
            and output_metrics["finite"]
            and lse_metrics["finite"]
        )

    bf16_case_metrics = {}
    for case in CASES:
        metrics = _chunked_metrics(bf16_reference, outputs[case.key])
        bf16_case_metrics[case.key] = {
            "output": metrics,
            **_bf16_output_verdict(
                metrics,
                minimum_cosine=args.minimum_bf16_output_cosine,
                maximum_relative_l2=(
                    args.maximum_bf16_output_relative_l2
                ),
            ),
        }
    correctness_passed = pairwise_all_finite and all(
        record["passed"] for record in bf16_case_metrics.values()
    )

    properties = torch.cuda.get_device_properties(args.gpu)
    projection_contracts = {}
    for case in CASES:
        projector = projectors[case.key]
        projection_contracts[case.key] = {
            "symbol": projector.symbol,
            "checked_symbol": projector.checked_symbol,
            "unchecked_symbol": projector.unchecked_symbol,
            "abi_validation_symbol": projector.abi_validation_symbol,
            "forward_workspace_abi_validated": (
                projector.forward_workspace_abi_validated
            ),
            "validated_forward_workspace_count": (
                projector.validated_forward_workspace_count
            ),
            "steady_state_uses_unchecked_out_parameter_abi": True,
            "first_use_authentication_excluded_from_timing": True,
            "publishes_training_backward_operands": True,
            "represented_backward": bool(projector.represented_backward),
            "per_block_qk_scales": bool(projector.per_block_qk_scales),
            "experimental_split_v_backward": bool(
                projector.experimental_split_v_backward
            ),
            "backward_kernel_executed": False,
        }
    result = {
        "schema": "b16_s4096_qkv_projection_pv_forward_factorial_v2",
        "created_utc": (
            dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
        ),
        "shape": {
            "batch": BATCH,
            "sequence": SEQUENCE,
            "hidden": HIDDEN,
            "q_heads": Q_HEADS,
            "kv_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "qkv_width": QKV_WIDTH,
            "causal": True,
        },
        "scope": {
            "included": [
                "QKV activation and learned-weight operand preparation",
                "QKV GEMM plus RoPE plus Q/K/V publication",
                "causal attention forward with training LSE publication",
            ],
            "excluded": [
                "embedding and RMSNorm",
                "attention output projection",
                "residual and MLP",
                "cross entropy",
                "backward and optimizer",
            ],
            "forward_execution_only": True,
            "projection_publishes_training_backward_operands": True,
            "backward_kernel_executed": False,
            "shared_inputs": (
                "one deterministic BF16 post-RMSNorm activation and packed "
                "learned-QKV weight draw across all four cases"
            ),
        },
        "cases": [case.as_dict() for case in CASES],
        "shared_state": state["receipt"],
        "allocation_contract": _allocation_contract(),
        "projection_contracts": projection_contracts,
        "workspace_contract": {
            "one_private_workspace_per_case": True,
            "attention_output_preallocated_per_case": True,
            "attention_lse_preallocated_per_case": True,
            "output_pointers": output_pointers,
            "lse_pointers": lse_pointers,
            "cases": workspace_receipts,
        },
        "timing": {
            "method": (
                "balanced rotating-provider CUDA-event device intervals; "
                "unsynchronized host launch intervals are diagnostic only"
            ),
            "warmups_per_stage": args.warmups,
            "samples_per_case_per_stage": args.samples,
            "case_order": [case.key for case in CASES],
            "balanced_provider_positions": args.samples % len(CASES) == 0,
            "stages": timing_stages,
        },
        "correctness": {
            "passed": correctness_passed,
            "fail_closed": True,
            "pairwise_low_precision_outputs_and_lse_finite": (
                pairwise_all_finite
            ),
            "reference": {
                "kind": "bf16_projection_plus_causal_torch_sdpa",
                "timed": False,
                "same_rows_weight_and_rope_as_low_precision_cases": True,
                "output_shape": list(bf16_reference.shape),
            },
            "thresholds": {
                "minimum_bf16_output_cosine": (
                    args.minimum_bf16_output_cosine
                ),
                "maximum_bf16_output_relative_l2": (
                    args.maximum_bf16_output_relative_l2
                ),
            },
            "bf16_reference_per_case": bf16_case_metrics,
            "pairwise_low_precision_comparisons": comparisons,
        },
        "topology": topologies,
        "memory": {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(args.gpu),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(args.gpu),
            "free_bytes_at_start": free_bytes,
            "total_bytes": total_bytes,
        },
        "provenance": {
            "argv": list(sys.argv),
            "repository": _git_provenance(),
            "selected_python": {
                "selected_path": str(selected_python),
                "resolved_path": str(selected_python.resolve()),
                "running_executable": sys.executable,
                "version": sys.version,
            },
            "extensions": {
                "mx": {
                    **artifacts["mx"],
                    **loaded_artifact_authentication["mx"],
                    "module": args.mx_module or _default_module(args.mx_extension),
                },
                "fp8": {
                    **artifacts["fp8"],
                    **loaded_artifact_authentication["fp8"],
                    "module": args.fp8_module or _default_module(args.fp8_extension),
                },
                "projection": {
                    **artifacts["projection"],
                    **loaded_artifact_authentication["projection"],
                    "module": args.projection_module,
                },
            },
            "runtime": {
                "platform": platform.platform(),
                "hostname": socket.gethostname(),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "gpu_index": args.gpu,
                "gpu_name": properties.name,
                "gpu_capability": [properties.major, properties.minor],
                "host_available_memory_bytes_at_start": host_available,
                "minimum_free_device_gib": args.minimum_free_gib,
                "minimum_free_system_gib": args.minimum_free_system_gib,
            },
        },
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(encoded, end="")
    _write_new_atomic(args.output, encoded)
    return 0 if correctness_passed else 2


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    selected_python = _resolve_executable(args.python)
    command = _worker_command(args, selected_python)
    if args.dry_run:
        print(json.dumps(_dry_run_plan(args, selected_python, command), indent=2))
        return 0
    if args._worker:
        return _run_worker(args, selected_python)
    completed = subprocess.run(command, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
