#!/usr/bin/env python3
"""Measure the causal D64 E4M3 publisher-to-attention boundary.

This harness is deliberately narrower than the full Llama training benchmark
and broader than :mod:`benchmark_causal_forward_matrix`.  It answers whether
an isolated MXFP4-PV attention advantage survives the projection-native QKV
publisher and the LSE contract used by training.

One deterministic token draw, normalized embedding activation, and set of
one-layer weights is shared by every route and publication mode.  The two
forward extensions are loaded into one process, and every measured stage uses
rotating provider order.  The deployed contract is explicit: E4M3 QKV,
channelwise E4M3 QKV weights, 2D NVFP4 output-projection weights, and 1D
MXFP4 V scaling.

``--dry-run`` prints the complete execution plan without checking extension
files, importing them, creating output directories, or touching CUDA.  The
selected Python path is intentionally not symlink-resolved: invoking a venv's
``bin/python`` symlink is what selects its adjacent ``pyvenv.cfg``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

# Keep direct-file and ``python -m`` execution equivalent.
_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
GIB = 1 << 30
MX_ROUTE = "nvfp4_qk_mxfp4_pv"
FP8_ROUTE = "nvfp4_qk_fp8_pv_exact"
ROUTES = (MX_ROUTE, FP8_ROUTE)
PROJECTION_SYMBOL_BASE = (
    "project_qkv_gqa_d64_paired_unified_fp8_e4m3_rope_packed_"
    "interleaved_causal"
)

# CUDA/PyTorch imports are intentionally worker-lazy.  In particular, a
# dry-run must not opportunistically import the worktree projection binary.
torch: Any = None
F: Any = None
tk_interface: Any = None
_load_extension: Any = None
_make_rope: Any = None


def _load_runtime() -> None:
    global F, _load_extension, _make_rope, tk_interface, torch
    import torch as loaded_torch
    import torch.nn.functional as loaded_functional

    import tk_fa4.interface as loaded_interface
    from tk_fa4.lowp_fa4_bwd.profile_gqa_d128_chain import (
        _load_extension as loaded_extension_loader,
        _make_rope as loaded_rope,
    )

    torch = loaded_torch
    F = loaded_functional
    tk_interface = loaded_interface
    _load_extension = loaded_extension_loader
    _make_rope = loaded_rope


@dataclass(frozen=True)
class PublicationMode:
    name: str
    represented_backward: bool
    per_block_qk_scales: bool
    experimental_split_v_backward: bool = False

    @property
    def projection_symbol(self) -> str:
        symbol = PROJECTION_SYMBOL_BASE
        if self.represented_backward:
            symbol += "_represented_backward"
        if self.per_block_qk_scales:
            symbol += "_perblock_qk"
        if self.experimental_split_v_backward:
            symbol += "_split_v_backward"
        return symbol

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "represented_backward": self.represented_backward,
            "per_block_qk_scales": self.per_block_qk_scales,
            "experimental_split_v_backward": self.experimental_split_v_backward,
            "projection_symbol": self.projection_symbol,
        }


PUBLICATION_MODES = (
    PublicationMode("independent_headwide", False, False),
    PublicationMode("represented_headwide", True, False),
    PublicationMode("represented_k16", True, True),
)
SPLIT_V_MODE = PublicationMode("represented_k16_split_v", True, True, True)


@dataclass(frozen=True)
class Case:
    route: str
    mode: PublicationMode

    @property
    def key(self) -> str:
        return f"{self.route}__{self.mode.name}"

    def as_dict(self) -> dict[str, Any]:
        return {"key": self.key, "route": self.route, **self.mode.as_dict()}


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


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mx-extension", type=Path, required=True)
    parser.add_argument("--mx-module")
    parser.add_argument("--fp8-extension", type=Path, required=True)
    parser.add_argument("--fp8-module")
    parser.add_argument("--projection-extension", type=Path, required=True)
    parser.add_argument("--projection-module", default="_C_b300_lowp_bwd")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="selected Python/venv used to load all three extension modules",
    )
    parser.add_argument("--sequence", type=_positive_int, default=4096)
    parser.add_argument("--q-heads", type=_positive_int, default=32)
    parser.add_argument("--kv-heads", type=_positive_int, default=8)
    parser.add_argument("--hidden", type=_positive_int, default=2048)
    parser.add_argument("--vocab", type=_positive_int, default=8192)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--input-std", type=_positive_finite, default=0.02)
    parser.add_argument("--weight-std", type=_positive_finite, default=0.02)
    parser.add_argument("--rms-epsilon", type=_positive_finite, default=1.0e-5)
    parser.add_argument("--q-quant-scale", type=_positive_finite, default=2.25)
    parser.add_argument("--k-quant-scale", type=_positive_finite, default=2.0)
    parser.add_argument("--warmups", type=_nonnegative_int, default=3)
    parser.add_argument("--samples", type=_positive_int, default=42)
    parser.add_argument("--gpu", type=_nonnegative_int, default=0)
    parser.add_argument(
        "--minimum-free-gib", type=_positive_finite, default=16.0
    )
    parser.add_argument(
        "--minimum-free-system-gib", type=_positive_finite, default=32.0
    )
    parser.add_argument(
        "--mx-native-quarter-mask",
        type=int,
        choices=(3, 15),
        default=3,
        help="expected density-4 MX native mask (3 is deployed d4q01)",
    )
    parser.add_argument(
        "--projection-weight-scaling",
        choices=("2d",),
        default="2d",
        help="fixed deployed NVFP4 output-projection weight contract",
    )
    parser.add_argument(
        "--v-mxfp4-scaling",
        choices=("1d",),
        default="1d",
        help="fixed deployed forward MXFP4 V scaling contract",
    )
    parser.add_argument(
        "--strict-modes",
        action="store_true",
        help="fail instead of recording/skipping a missing publisher symbol",
    )
    parser.add_argument(
        "--experimental-split-v",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "include the MX-only represented-K16 split-V publisher when its "
            "projection symbol is present"
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print a side-effect-free plan without loading binaries or CUDA",
    )
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.sequence % 256:
        parser.error("--sequence must be divisible by 256")
    if args.q_heads % 2 or args.kv_heads % 2:
        parser.error("paired D64 publication requires even head counts")
    if args.q_heads % args.kv_heads:
        parser.error("--q-heads must be divisible by --kv-heads")
    if args.hidden % 128:
        parser.error("--hidden must be divisible by 128")
    modules = (args.mx_module, args.fp8_module, args.projection_module)
    for module in modules:
        if module is not None and not module.isidentifier():
            parser.error("extension module names must be Python identifiers")
    effective_modules = (
        args.mx_module or _default_module(args.mx_extension),
        args.fp8_module or _default_module(args.fp8_extension),
        args.projection_module,
    )
    if len(set(effective_modules)) != len(effective_modules):
        parser.error("MX, exact-FP8, and projection module names must be distinct")
    planned_cases = _all_cases(include_split_v=args.experimental_split_v)
    if args.samples % len(planned_cases):
        parser.error(
            f"--samples must be a multiple of {len(planned_cases)} for "
            "balanced provider order"
        )
    return args


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
    # Do not resolve a venv symlink; its lexical path selects pyvenv.cfg.
    selected = _selected_absolute(candidate)
    if not selected.is_file() or not os.access(selected, os.X_OK):
        raise FileNotFoundError(f"Python executable is not runnable: {selected}")
    return selected


def _default_module(path: Path) -> str:
    return path.name.split(".", 1)[0]


def _worker_command(args: argparse.Namespace, python: Path) -> list[str]:
    mx_module = args.mx_module or _default_module(args.mx_extension)
    fp8_module = args.fp8_module or _default_module(args.fp8_extension)
    command = [
        str(python),
        str(Path(__file__).absolute()),
        "--_worker",
        "--python",
        str(python),
        "--mx-extension",
        str(_selected_absolute(args.mx_extension)),
        "--mx-module",
        mx_module,
        "--fp8-extension",
        str(_selected_absolute(args.fp8_extension)),
        "--fp8-module",
        fp8_module,
        "--projection-extension",
        str(_selected_absolute(args.projection_extension)),
        "--projection-module",
        args.projection_module,
        "--sequence",
        str(args.sequence),
        "--q-heads",
        str(args.q_heads),
        "--kv-heads",
        str(args.kv_heads),
        "--hidden",
        str(args.hidden),
        "--vocab",
        str(args.vocab),
        "--seed",
        str(args.seed),
        "--input-std",
        str(args.input_std),
        "--weight-std",
        str(args.weight_std),
        "--rms-epsilon",
        str(args.rms_epsilon),
        "--q-quant-scale",
        str(args.q_quant_scale),
        "--k-quant-scale",
        str(args.k_quant_scale),
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
        "--mx-native-quarter-mask",
        str(args.mx_native_quarter_mask),
        "--projection-weight-scaling",
        args.projection_weight_scaling,
        "--v-mxfp4-scaling",
        args.v_mxfp4_scaling,
    ]
    if args.strict_modes:
        command.append("--strict-modes")
    command.append(
        "--experimental-split-v"
        if args.experimental_split_v
        else "--no-experimental-split-v"
    )
    if args.output is not None:
        command.extend(("--output", str(_selected_absolute(args.output))))
    return command


def _all_cases(
    modes: Sequence[PublicationMode] = PUBLICATION_MODES,
    *,
    include_split_v: bool = True,
) -> tuple[Case, ...]:
    cases = [Case(route, mode) for mode in modes for route in ROUTES]
    if include_split_v:
        cases.append(Case(MX_ROUTE, SPLIT_V_MODE))
    return tuple(cases)


def _rotating_orders(names: Sequence[str], rounds: int) -> list[list[str]]:
    if not names:
        raise ValueError("at least one timing provider is required")
    values = list(names)
    return [
        values[index % len(values) :] + values[: index % len(values)]
        for index in range(rounds)
    ]


def _dry_run_plan(
    args: argparse.Namespace,
    python: Path,
    command: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema": "causal_forward_boundary_ab_plan_v1",
        "dry_run": True,
        "touches_cuda": False,
        "creates_output": False,
        "selected_python": str(python),
        "worker_command": list(command),
        "extension_bindings": {
            "mx": {
                "path": str(_selected_absolute(args.mx_extension)),
                "module": args.mx_module or _default_module(args.mx_extension),
            },
            "fp8_exact": {
                "path": str(_selected_absolute(args.fp8_extension)),
                "module": args.fp8_module or _default_module(args.fp8_extension),
            },
            "projection": {
                "path": str(_selected_absolute(args.projection_extension)),
                "module": args.projection_module,
            },
        },
        "shape": {
            "batch": 1,
            "sequence": args.sequence,
            "q_heads": args.q_heads,
            "kv_heads": args.kv_heads,
            "head_dim": 64,
            "hidden": args.hidden,
            "layers": 1,
        },
        "deployed_contract": {
            "qkv_projection": "e4m3",
            "qkv_weight_scaling": "one decode per output channel",
            "output_projection_weight_scaling": args.projection_weight_scaling,
            "mx_v_scaling": args.v_mxfp4_scaling,
            "mx_v_scale_2d_argument": False,
            "mx_native_density": 4,
            "mx_native_quarter_mask": args.mx_native_quarter_mask,
        },
        "publication_modes": [
            mode.as_dict() for mode in (*PUBLICATION_MODES, SPLIT_V_MODE)
        ],
        "cases": [
            case.as_dict()
            for case in _all_cases(include_split_v=args.experimental_split_v)
        ],
        "timed_stages": [
            "qkv_weight_concat",
            "input_e4m3_rowwise_pack",
            "qkv_weight_e4m3_channelwise_pack",
            "qkv_rope_publication",
            "attention_store_lse_false",
            "attention_store_lse_true",
            "prepacked_publication_attention_store_lse_false",
            "prepacked_publication_attention_store_lse_true",
            "allocated_publication_attention_store_lse_true",
            "output_activation_nvfp4_pack",
            "output_weight_nvfp4_2d_pack",
            "output_projection",
            "full_one_layer_attention_boundary_preallocated_store_lse_true",
            "full_one_layer_attention_boundary_allocated_store_lse_true",
        ],
        "timing": {
            "method": "rotating-provider CUDA events plus host launch wall time",
            "warmups": args.warmups,
            "samples": args.samples,
            "minimum_free_device_gib": args.minimum_free_gib,
            "minimum_free_system_gib": args.minimum_free_system_gib,
        },
        "output": (
            str(_selected_absolute(args.output)) if args.output is not None else None
        ),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_provenance(selected_path: Path) -> dict[str, Any]:
    selected = _selected_absolute(selected_path)
    resolved = selected.resolve(strict=True)
    stat = resolved.stat()
    return {
        "selected_path": str(selected),
        "resolved_path": str(resolved),
        "sha256": _sha256(resolved),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
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
    tracked_status = _git_output(
        "status", "--porcelain=v1", "--untracked-files=no"
    )
    full_status = _git_output("status", "--porcelain=v1")
    harness_relative = Path(__file__).resolve().relative_to(REPO_ROOT.resolve())
    harness_tracked = _git_output(
        "ls-files", "--error-unmatch", str(harness_relative)
    )
    return {
        "root": str(REPO_ROOT),
        "head": _git_output("rev-parse", "HEAD"),
        "branch": _git_output("branch", "--show-current"),
        "tracked_dirty": (
            bool(tracked_status) if tracked_status is not None else None
        ),
        "tracked_status": tracked_status,
        "dirty_including_untracked": (
            bool(full_status) if full_status is not None else None
        ),
        "full_status": full_status,
        "harness_tracked": bool(harness_tracked),
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
    """Publish a complete new result without replacing an existing artifact."""
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
        # A hard-link publication is atomic and fails rather than overwriting
        # if another process created the requested result concurrently.
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _timing_summary(values: Sequence[float], unit: str) -> dict[str, Any]:
    if not values:
        raise ValueError("cannot summarize an empty timing sample")
    suffix = "us" if unit == "microseconds" else "ns"
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
    if not names:
        raise ValueError("at least one timing function is required")
    for order in _rotating_orders(names, warmups):
        retained = [functions[name]() for name in order]
        torch.cuda.synchronize()
        del retained

    device_values: dict[str, list[float]] = {name: [] for name in names}
    host_values: dict[str, list[float]] = {name: [] for name in names}
    orders = _rotating_orders(names, samples)
    for order in orders:
        events: list[tuple[str, Any, Any]] = []
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
                "device": _timing_summary(device_values[name], "microseconds"),
                "host_launch": _timing_summary(host_values[name], "nanoseconds"),
            }
            for name in names
        },
        "sample_orders": orders,
    }


def _validate_topology(
    label: str,
    topology: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    expected = {
        "batch": 1,
        "seqlen": args.sequence,
        "heads": args.q_heads,
        "kv_heads": args.kv_heads,
        "dqk": 64,
        "dvo": 64,
        "causal": True,
        "qk_format": "nvfp4_e4m3_block16",
    }
    for key, expected_value in expected.items():
        if topology.get(key) != expected_value:
            raise ValueError(
                f"{label} topology {key}={topology.get(key)!r}; "
                f"expected {expected_value!r}"
            )
    if bool(topology.get("fixed_p_ceiling", False)) or bool(
        topology.get("score_pack_ceiling", False)
    ):
        raise ValueError(f"{label} is a diagnostic ceiling build")
    if label == "mx":
        if topology.get("pv_format") != "mxfp4_e8m0_block32":
            raise ValueError("MX extension is not the MXFP4-PV route")
        if not bool(topology.get("causal_interleaved_kv", False)):
            raise ValueError("MX extension must consume interleaved causal K/V")
        if int(topology.get("mx_mode23_native_density", -1)) != 4:
            raise ValueError("MX extension must use native density 4")
        observed_mask = int(
            topology.get("mx_mode23_native_quarter_mask", -1)
        )
        if observed_mask != args.mx_native_quarter_mask:
            raise ValueError(
                f"MX native quarter mask {observed_mask}; expected "
                f"{args.mx_native_quarter_mask}"
            )
    else:
        if topology.get("pv_format") != "e4m3_fp8":
            raise ValueError("exact extension is not the E4M3 FP8-PV route")
        if int(topology.get("shiftless_fp8_mode", -1)) != 0:
            raise ValueError("exact FP8-PV must use shiftless mode 0")
        if bool(topology.get("causal_interleaved_kv", False)):
            raise ValueError("exact FP8-PV must consume normal-order K/V")


def _tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    values = tensor.float()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "mean": float(values.mean()),
        "rms": float(values.square().mean().sqrt()),
        "max_abs": float(values.abs().max()),
    }


def _token_digest(tokens: torch.Tensor) -> str:
    values = tokens.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(values).hexdigest()


def _new_normal(
    shape: Sequence[int],
    *,
    std: float,
    generator: torch.Generator,
) -> torch.Tensor:
    value = torch.empty(*shape, device="cuda", dtype=torch.bfloat16)
    return value.normal_(mean=0.0, std=std, generator=generator)


def _make_shared_state(args: argparse.Namespace) -> dict[str, Any]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed)
    tokens = torch.randint(
        args.vocab,
        (1, args.sequence),
        device="cuda",
        generator=generator,
    )
    embedding_weight = _new_normal(
        (args.vocab, args.hidden), std=args.input_std, generator=generator
    )
    embedded = F.embedding(tokens, embedding_weight)
    variance = embedded.float().square().mean(dim=-1, keepdim=True)
    rows = (embedded.float() * torch.rsqrt(variance + args.rms_epsilon))
    rows = rows.bfloat16().reshape(args.sequence, args.hidden).contiguous()

    q_width = args.q_heads * 64
    kv_width = args.kv_heads * 64
    q_weight = _new_normal(
        (q_width, args.hidden), std=args.weight_std, generator=generator
    )
    k_weight = _new_normal(
        (kv_width, args.hidden), std=args.weight_std, generator=generator
    )
    v_weight = _new_normal(
        (kv_width, args.hidden), std=args.weight_std, generator=generator
    )
    out_weight = _new_normal(
        (args.hidden, q_width), std=args.weight_std, generator=generator
    )
    qkv_weight = torch.cat((q_weight, k_weight, v_weight), dim=0).contiguous()
    rope_cos, rope_sin = _make_rope(args.sequence, 64)
    paired_rope = tk_interface.b300_pack_gqa_d64_paired_rope(
        rope_cos, rope_sin
    )
    qk_scales = torch.zeros(
        1, args.q_heads // 2, 7, device="cuda", dtype=torch.float32
    )
    qk_scales[..., 0] = args.q_quant_scale
    qk_scales[..., 1] = args.k_quant_scale
    summary = {
        "tokens": {
            "shape": list(tokens.shape),
            "dtype": str(tokens.dtype),
            "vocab": args.vocab,
            "sha256": _token_digest(tokens),
        },
        "normalized_rows": _tensor_summary(rows),
        "q_weight": _tensor_summary(q_weight),
        "k_weight": _tensor_summary(k_weight),
        "v_weight": _tensor_summary(v_weight),
        "out_weight": _tensor_summary(out_weight),
    }
    return {
        "tokens": tokens,
        "rows": rows,
        "q_weight": q_weight,
        "k_weight": k_weight,
        "v_weight": v_weight,
        "out_weight": out_weight,
        "qkv_weight": qkv_weight,
        "paired_rope": paired_rope,
        "qk_scales": qk_scales,
        "summary": summary,
    }


def _publish(
    args: argparse.Namespace,
    state: dict[str, Any],
    case: Case,
    *,
    input_operand: tuple[torch.Tensor, torch.Tensor],
    weight_operand: tuple[torch.Tensor, torch.Tensor],
) -> Any:
    is_mx = case.route == MX_ROUTE
    return tk_interface.b300_project_qkv_gqa_d64_paired_unified_lowp_e4m3(
        input_operand,
        weight_operand,
        state["qk_scales"],
        state["paired_rope"],
        batch=1,
        seqlen=args.sequence,
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
        publish_mxfp4_v=is_mx,
        v_mxfp4_scale_2d=False,
        interleave_causal_kv=is_mx,
        represented_backward=case.mode.represented_backward,
        per_block_qk_scales=case.mode.per_block_qk_scales,
        experimental_split_v_backward=(
            case.mode.experimental_split_v_backward
        ),
    )


def _run_attention(
    case: Case,
    extensions: dict[str, Any],
    topologies: dict[str, dict[str, Any]],
    bundle: Any,
    output: Any,
    lse: Any,
    *,
    store_lse: bool,
) -> None:
    topology = topologies[case.route]
    extension = extensions[case.route]
    os.environ["TK_FA4_FP4PV_FWD_CONFIG"] = str(topology["route"])
    if case.route == MX_ROUTE:
        extension.forward_hao_direct_fp4pv(
            *bundle.forward_operands(), output, lse, 0, True, store_lse
        )
        return
    forward_v = bundle.v_forward_fp8
    if forward_v is None:
        forward_v = bundle.v_backward_fp8.permute(0, 2, 3, 1).contiguous()
    extension.forward_hao_direct_fp8pv(
        *bundle.qk_forward_operands(),
        forward_v,
        output,
        lse,
        0,
        True,
        store_lse,
    )


def _chunked_metrics(
    reference: torch.Tensor,
    actual: torch.Tensor,
    *,
    chunk_elements: int = 1 << 20,
) -> dict[str, Any]:
    reference_flat = reference.detach().reshape(-1)
    actual_flat = actual.detach().reshape(-1)
    if reference_flat.numel() != actual_flat.numel():
        raise ValueError("metric tensors must have the same number of elements")
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
        finite = finite and bool(torch.isfinite(actual_chunk).all())
    reference_norm = math.sqrt(max(reference_sq, 1.0e-40))
    actual_norm = math.sqrt(max(actual_sq, 1.0e-40))
    return {
        "finite": finite,
        "cosine": dot / (reference_norm * actual_norm),
        "relative_l2": math.sqrt(difference_sq) / reference_norm,
        "norm_ratio": actual_norm / reference_norm,
    }


def _operands_bitwise_equal(
    left: Sequence[torch.Tensor], right: Sequence[torch.Tensor]
) -> bool:
    if len(left) != len(right):
        return False
    for left_item, right_item in zip(left, right, strict=True):
        if (
            left_item.dtype != right_item.dtype
            or left_item.shape != right_item.shape
            or left_item.device != right_item.device
        ):
            return False
        # torch.equal does not implement compare_eq_ne_cuda for packed
        # float4.  The publication audit is deliberately bitwise, so compare
        # the contiguous storage bytes for every dtype instead.
        left_bytes = left_item.contiguous().view(torch.uint8)
        right_bytes = right_item.contiguous().view(torch.uint8)
        if not torch.equal(left_bytes, right_bytes):
            return False
    return True


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return repr(value)


def _run_worker(args: argparse.Namespace, selected_python: Path) -> int:
    host_available = _host_available_bytes()
    if (
        host_available is not None
        and host_available < args.minimum_free_system_gib * GIB
    ):
        raise RuntimeError(
            f"host has {host_available / GIB:.2f} GiB available; requires "
            f"{args.minimum_free_system_gib:.2f} GiB"
        )
    _load_runtime()
    if args.gpu >= torch.cuda.device_count():
        raise RuntimeError(
            f"GPU {args.gpu} is unavailable ({torch.cuda.device_count()} visible)"
        )
    torch.cuda.set_device(args.gpu)
    free_bytes, total_bytes = torch.cuda.mem_get_info(args.gpu)
    if free_bytes < args.minimum_free_gib * GIB:
        raise RuntimeError(
            f"GPU {args.gpu} has {free_bytes / GIB:.2f} GiB free; "
            f"requires {args.minimum_free_gib:.2f} GiB"
        )

    mx_path = _selected_absolute(args.mx_extension)
    fp8_path = _selected_absolute(args.fp8_extension)
    projection_path = _selected_absolute(args.projection_extension)
    mx_module = args.mx_module or _default_module(mx_path)
    fp8_module = args.fp8_module or _default_module(fp8_path)
    projection = _load_extension(
        projection_path.resolve(strict=True), args.projection_module
    )
    tk_interface._C_b300_lowp_bwd = projection
    tk_interface._LOWP_BWD_IMPORT_ERROR = None
    extensions = {
        MX_ROUTE: _load_extension(mx_path.resolve(strict=True), mx_module),
        FP8_ROUTE: _load_extension(fp8_path.resolve(strict=True), fp8_module),
    }
    topologies = {
        route: dict(extension.read_hao_direct_topology())
        for route, extension in extensions.items()
    }
    _validate_topology("mx", topologies[MX_ROUTE], args)
    _validate_topology("fp8", topologies[FP8_ROUTE], args)

    available_modes = tuple(
        mode
        for mode in PUBLICATION_MODES
        if hasattr(projection, mode.projection_symbol)
    )
    missing_modes = tuple(
        mode for mode in PUBLICATION_MODES if mode not in available_modes
    )
    split_v_available = bool(
        args.experimental_split_v
        and hasattr(projection, SPLIT_V_MODE.projection_symbol)
    )
    missing_requested_modes = (
        *missing_modes,
        *(
            (SPLIT_V_MODE,)
            if args.experimental_split_v and not split_v_available
            else ()
        ),
    )
    if missing_requested_modes and args.strict_modes:
        missing = ", ".join(
            mode.projection_symbol for mode in missing_requested_modes
        )
        raise RuntimeError(f"projection extension is missing: {missing}")
    if not available_modes:
        raise RuntimeError("projection extension has none of the requested modes")
    cases = _all_cases(
        available_modes,
        include_split_v=split_v_available,
    )
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    state = _make_shared_state(args)
    case_keys = [case.key for case in cases]
    previous_route = os.environ.get("TK_FA4_FP4PV_FWD_CONFIG")
    try:
        stages: dict[str, Any] = {}
        stages["qkv_weight_concat"] = _measure_interleaved(
            {
                "shared": lambda: torch.cat(
                    (state["q_weight"], state["k_weight"], state["v_weight"]),
                    dim=0,
                ).contiguous()
            },
            warmups=args.warmups,
            samples=args.samples,
        )
        stages["input_e4m3_rowwise_pack"] = _measure_interleaved(
            {
                "shared": lambda: tuple(
                    tk_interface.b300_prepare_e4m3_projection_operand(
                        state["rows"]
                    )
                )
            },
            warmups=args.warmups,
            samples=args.samples,
        )
        stages["qkv_weight_e4m3_channelwise_pack"] = _measure_interleaved(
            {
                "shared": lambda: tuple(
                    tk_interface.b300_prepare_e4m3_projection_weight(
                        state["qkv_weight"]
                    )
                )
            },
            warmups=args.warmups,
            samples=args.samples,
        )

        input_operand = tuple(
            tk_interface.b300_prepare_e4m3_projection_operand(state["rows"])
        )
        weight_operand = tuple(
            tk_interface.b300_prepare_e4m3_projection_weight(
                state["qkv_weight"]
            )
        )

        publication_functions = {
            case.key: (
                lambda case=case: _publish(
                    args,
                    state,
                    case,
                    input_operand=input_operand,
                    weight_operand=weight_operand,
                )
            )
            for case in cases
        }
        stages["qkv_rope_publication"] = _measure_interleaved(
            publication_functions,
            warmups=args.warmups,
            samples=args.samples,
        )
        bundles = {name: function() for name, function in publication_functions.items()}
        torch.cuda.synchronize()

        output_shape = (1, args.sequence, args.q_heads, 64)
        lse_shape = (1, args.q_heads, 1, args.sequence)
        outputs = {
            case.key: torch.empty(
                output_shape, device="cuda", dtype=torch.bfloat16
            )
            for case in cases
        }
        lses = {
            case.key: torch.empty(lse_shape, device="cuda", dtype=torch.float32)
            for case in cases
        }

        store_lse_support: dict[str, dict[str, Any]] = {}
        for case in cases:
            _run_attention(
                case,
                extensions,
                topologies,
                bundles[case.key],
                outputs[case.key],
                lses[case.key],
                store_lse=True,
            )
            supports_false = True
            false_error = None
            try:
                _run_attention(
                    case,
                    extensions,
                    topologies,
                    bundles[case.key],
                    outputs[case.key],
                    lses[case.key],
                    store_lse=False,
                )
            except TypeError as error:
                supports_false = False
                false_error = str(error)
            store_lse_support[case.key] = {
                "true": True,
                "false": supports_false,
                "false_limitation": false_error,
            }
        torch.cuda.synchronize()

        def attention_functions(store_lse: bool) -> dict[str, Callable[[], None]]:
            return {
                case.key: (
                    lambda case=case, store_lse=store_lse: _run_attention(
                        case,
                        extensions,
                        topologies,
                        bundles[case.key],
                        outputs[case.key],
                        lses[case.key],
                        store_lse=store_lse,
                    )
                )
                for case in cases
                if store_lse or store_lse_support[case.key]["false"]
            }

        false_attention = attention_functions(False)
        if false_attention:
            stages["attention_store_lse_false"] = _measure_interleaved(
                false_attention,
                warmups=args.warmups,
                samples=args.samples,
            )
        stages["attention_store_lse_true"] = _measure_interleaved(
            attention_functions(True),
            warmups=args.warmups,
            samples=args.samples,
        )

        def composed_functions(store_lse: bool) -> dict[str, Callable[[], Any]]:
            functions: dict[str, Callable[[], Any]] = {}
            for case in cases:
                if not store_lse and not store_lse_support[case.key]["false"]:
                    continue

                def run(case: Case = case, store_lse: bool = store_lse) -> Any:
                    bundle = _publish(
                        args,
                        state,
                        case,
                        input_operand=input_operand,
                        weight_operand=weight_operand,
                    )
                    _run_attention(
                        case,
                        extensions,
                        topologies,
                        bundle,
                        outputs[case.key],
                        lses[case.key],
                        store_lse=store_lse,
                    )
                    return bundle

                functions[case.key] = run
            return functions

        false_composed = composed_functions(False)
        if false_composed:
            stages[
                "prepacked_publication_attention_store_lse_false"
            ] = _measure_interleaved(
                false_composed,
                warmups=args.warmups,
                samples=args.samples,
            )
        stages[
            "prepacked_publication_attention_store_lse_true"
        ] = _measure_interleaved(
            composed_functions(True),
            warmups=args.warmups,
            samples=args.samples,
        )

        def allocated_composed(case: Case) -> Any:
            bundle = _publish(
                args,
                state,
                case,
                input_operand=input_operand,
                weight_operand=weight_operand,
            )
            current_output = torch.empty(
                output_shape, device="cuda", dtype=torch.bfloat16
            )
            current_lse = torch.empty(
                lse_shape, device="cuda", dtype=torch.float32
            )
            _run_attention(
                case,
                extensions,
                topologies,
                bundle,
                current_output,
                current_lse,
                store_lse=True,
            )
            return bundle, current_output, current_lse

        stages[
            "allocated_publication_attention_store_lse_true"
        ] = _measure_interleaved(
            {
                case.key: (lambda case=case: allocated_composed(case))
                for case in cases
            },
            warmups=args.warmups,
            samples=args.samples,
        )

        # Re-establish matching output/LSE pairs before correctness and output
        # projection timing; the preceding false-LSE calls intentionally leave
        # the LSE buffers stale.
        for case in cases:
            _run_attention(
                case,
                extensions,
                topologies,
                bundles[case.key],
                outputs[case.key],
                lses[case.key],
                store_lse=True,
            )
        torch.cuda.synchronize()

        stages["output_activation_nvfp4_pack"] = _measure_interleaved(
            {
                case.key: (
                    lambda case=case: tuple(
                        tk_interface.b300_prepare_nvfp4_projection_operand(
                            outputs[case.key].reshape(
                                args.sequence, args.q_heads * 64
                            )
                        )
                    )
                )
                for case in cases
            },
            warmups=args.warmups,
            samples=args.samples,
        )
        stages["output_weight_nvfp4_2d_pack"] = _measure_interleaved(
            {
                "shared": lambda: tuple(
                    tk_interface.b300_prepare_nvfp4_projection_weight(
                        state["out_weight"]
                    )
                )
            },
            warmups=args.warmups,
            samples=args.samples,
        )
        output_operands = {
            case.key: tuple(
                tk_interface.b300_prepare_nvfp4_projection_operand(
                    outputs[case.key].reshape(args.sequence, args.q_heads * 64)
                )
            )
            for case in cases
        }
        output_weight_operand = tuple(
            tk_interface.b300_prepare_nvfp4_projection_weight(
                state["out_weight"]
            )
        )
        stages["output_projection"] = _measure_interleaved(
            {
                case.key: (
                    lambda case=case: tk_interface.b300_project_nvfp4(
                        output_operands[case.key], output_weight_operand
                    )
                )
                for case in cases
            },
            warmups=args.warmups,
            samples=args.samples,
        )

        def full_boundary(case: Case, *, allocate_attention_buffers: bool) -> Any:
            qkv_weight = torch.cat(
                (state["q_weight"], state["k_weight"], state["v_weight"]),
                dim=0,
            ).contiguous()
            current_input = tuple(
                tk_interface.b300_prepare_e4m3_projection_operand(state["rows"])
            )
            current_weight = tuple(
                tk_interface.b300_prepare_e4m3_projection_weight(qkv_weight)
            )
            bundle = _publish(
                args,
                state,
                case,
                input_operand=current_input,
                weight_operand=current_weight,
            )
            current_attention_output = outputs[case.key]
            current_lse = lses[case.key]
            if allocate_attention_buffers:
                current_attention_output = torch.empty(
                    output_shape, device="cuda", dtype=torch.bfloat16
                )
                current_lse = torch.empty(
                    lse_shape, device="cuda", dtype=torch.float32
                )
            _run_attention(
                case,
                extensions,
                topologies,
                bundle,
                current_attention_output,
                current_lse,
                store_lse=True,
            )
            current_output = tuple(
                tk_interface.b300_prepare_nvfp4_projection_operand(
                    current_attention_output.reshape(
                        args.sequence, args.q_heads * 64
                    )
                )
            )
            current_output_weight = tuple(
                tk_interface.b300_prepare_nvfp4_projection_weight(
                    state["out_weight"]
                )
            )
            projected = tk_interface.b300_project_nvfp4(
                current_output, current_output_weight
            )
            return bundle, current_attention_output, current_lse, projected

        stages[
            "full_one_layer_attention_boundary_preallocated_store_lse_true"
        ] = _measure_interleaved(
            {
                case.key: (
                    lambda case=case: full_boundary(
                        case, allocate_attention_buffers=False
                    )
                )
                for case in cases
            },
            warmups=args.warmups,
            samples=args.samples,
        )
        stages[
            "full_one_layer_attention_boundary_allocated_store_lse_true"
        ] = _measure_interleaved(
            {
                case.key: (
                    lambda case=case: full_boundary(
                        case, allocate_attention_buffers=True
                    )
                )
                for case in cases
            },
            warmups=args.warmups,
            samples=args.samples,
        )

        correctness: dict[str, Any] = {}
        publication_audit: dict[str, Any] = {}
        all_finite = True
        for mode in available_modes:
            mx_key = Case(MX_ROUTE, mode).key
            fp8_key = Case(FP8_ROUTE, mode).key
            output_metrics = _chunked_metrics(outputs[fp8_key], outputs[mx_key])
            lse_metrics = _chunked_metrics(lses[fp8_key], lses[mx_key])
            correctness[mode.name] = {
                "mxfp4_pv_vs_exact_fp8_pv": {
                    "output": output_metrics,
                    "lse": lse_metrics,
                }
            }
            all_finite = (
                all_finite
                and output_metrics["finite"]
                and lse_metrics["finite"]
            )
            mx_bundle = bundles[mx_key]
            fp8_bundle = bundles[fp8_key]
            publication_audit[mode.name] = {
                "q_payload_bitwise_equal_across_routes": torch.equal(
                    mx_bundle.backward.score_q_fp4,
                    fp8_bundle.backward.score_q_fp4,
                ),
                "q_scale_pages_bitwise_equal_across_routes": torch.equal(
                    mx_bundle.q_forward_scales,
                    fp8_bundle.q_forward_scales,
                ),
                "exact_direct_feature_major_v": fp8_bundle.v_forward_fp8 is not None,
                "mx_has_no_redundant_feature_major_fp8_v": (
                    mx_bundle.v_forward_fp8 is None
                ),
            }

        if split_v_available:
            split_key = Case(MX_ROUTE, SPLIT_V_MODE).key
            represented_k16 = PUBLICATION_MODES[-1]
            deployed_mx_key = Case(MX_ROUTE, represented_k16).key
            exact_k16_key = Case(FP8_ROUTE, represented_k16).key
            split_vs_exact_output = _chunked_metrics(
                outputs[exact_k16_key], outputs[split_key]
            )
            split_vs_exact_lse = _chunked_metrics(
                lses[exact_k16_key], lses[split_key]
            )
            split_vs_deployed_output = _chunked_metrics(
                outputs[deployed_mx_key], outputs[split_key]
            )
            split_vs_deployed_lse = _chunked_metrics(
                lses[deployed_mx_key], lses[split_key]
            )
            correctness[SPLIT_V_MODE.name] = {
                "split_mx_vs_exact_fp8_represented_k16": {
                    "output": split_vs_exact_output,
                    "lse": split_vs_exact_lse,
                },
                "split_mx_vs_deployed_mx_represented_k16": {
                    "output": split_vs_deployed_output,
                    "lse": split_vs_deployed_lse,
                },
            }
            all_finite = all_finite and all(
                metric["finite"]
                for metric in (
                    split_vs_exact_output,
                    split_vs_exact_lse,
                    split_vs_deployed_output,
                    split_vs_deployed_lse,
                )
            )
            split_bundle = bundles[split_key]
            deployed_mx_bundle = bundles[deployed_mx_key]
            publication_audit[SPLIT_V_MODE.name] = {
                "comparison": "MX-only A/B against represented_k16",
                "forward_operands_bitwise_equal_to_deployed_mx": (
                    _operands_bitwise_equal(
                        split_bundle.forward_operands(),
                        deployed_mx_bundle.forward_operands(),
                    )
                ),
                "q_backward_bitwise_equal_to_deployed_mx": torch.equal(
                    split_bundle.q_backward_fp8,
                    deployed_mx_bundle.q_backward_fp8,
                ),
                "k_backward_bitwise_equal_to_deployed_mx": torch.equal(
                    split_bundle.k_backward_fp8,
                    deployed_mx_bundle.k_backward_fp8,
                ),
                "v_backward_bitwise_equal_to_deployed_mx": torch.equal(
                    split_bundle.v_backward_fp8,
                    deployed_mx_bundle.v_backward_fp8,
                ),
                "expected_difference": (
                    "backward V is direct accumulator E4M3 instead of the "
                    "forward-MXFP4 represented round trip"
                ),
            }

        properties = torch.cuda.get_device_properties(args.gpu)
        now = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
        result = {
            "schema": "causal_forward_boundary_ab_v1",
            "created_utc": now,
            "shape": {
                "batch": 1,
                "sequence": args.sequence,
                "q_heads": args.q_heads,
                "kv_heads": args.kv_heads,
                "head_dim": 64,
                "hidden": args.hidden,
                "layers": 1,
                "causal": True,
            },
            "deployed_contract": {
                "qkv_projection": "e4m3",
                "qkv_activation_scaling": "one decode per input row",
                "qkv_weight_scaling": "one decode per output channel",
                "output_projection_weight_scaling": "2d_nvfp4_16x16",
                "mx_v_scaling": "1d_depth_row_by_sequence_block32",
                "v_mxfp4_scale_2d_argument": False,
                "mx_native_density": 4,
                "mx_native_quarter_mask": args.mx_native_quarter_mask,
                "attention_store_lse_training_value": True,
            },
            "scope": {
                "shared_input": (
                    "one deterministic token draw, normalized embedding "
                    "activation, and one-layer Q/K/V/O weight draw"
                ),
                "allocation": (
                    "attention-only and prepacked stages reuse output/LSE; "
                    "explicit allocated stages create both buffers between "
                    "publication and attention; publisher/packer return "
                    "allocations remain inside their named stages"
                ),
                "excluded": [
                    "embedding lookup and RMSNorm",
                    "residual and MLP",
                    "final norm, vocabulary projection, and loss",
                    "backward and optimizer",
                ],
            },
            "capabilities": {
                "requested_modes": [
                    mode.as_dict()
                    for mode in (
                        *PUBLICATION_MODES,
                        *(
                            (SPLIT_V_MODE,)
                            if args.experimental_split_v
                            else ()
                        ),
                    )
                ],
                "executed_modes": [
                    mode.as_dict()
                    for mode in (
                        *available_modes,
                        *((SPLIT_V_MODE,) if split_v_available else ()),
                    )
                ],
                "skipped_modes": [
                    {
                        **mode.as_dict(),
                        "reason": "projection extension does not export symbol",
                    }
                    for mode in missing_requested_modes
                ],
                "store_lse": store_lse_support,
            },
            "cases": [case.as_dict() for case in cases],
            "topology": {
                route: _jsonable(topology) for route, topology in topologies.items()
            },
            "shared_state": state["summary"],
            "publication_audit": publication_audit,
            "timing": {
                "method": (
                    "rotating-provider CUDA-event device intervals and "
                    "unsynchronized host launch wall intervals"
                ),
                "warmups_per_provider": args.warmups,
                "samples_per_provider": args.samples,
                "balanced_case_order": args.samples % len(case_keys) == 0,
                "case_order": case_keys,
                "stages": stages,
            },
            "correctness": {
                "all_outputs_and_lse_finite": all_finite,
                "by_publication_mode": correctness,
            },
            "provenance": {
                "argv": list(sys.argv),
                "repository": _git_provenance(),
                "worker": _file_provenance(Path(__file__)),
                "selected_python": {
                    "selected_path": str(selected_python),
                    "resolved_path": str(selected_python.resolve()),
                    "running_executable": sys.executable,
                    "version": sys.version,
                },
                "extensions": {
                    "mx": {
                        **_file_provenance(mx_path),
                        "module": mx_module,
                    },
                    "fp8_exact": {
                        **_file_provenance(fp8_path),
                        "module": fp8_module,
                    },
                    "projection": {
                        **_file_provenance(projection_path),
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
                    "gpu_total_memory_bytes": total_bytes,
                    "gpu_free_memory_bytes_at_start": free_bytes,
                    "host_available_memory_bytes_at_start": host_available,
                    "minimum_free_device_gib": args.minimum_free_gib,
                    "minimum_free_system_gib": args.minimum_free_system_gib,
                },
            },
        }
        encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
        print(encoded, end="")
        if args.output is not None:
            _write_new_atomic(args.output, encoded)
        return 0 if all_finite else 2
    finally:
        if previous_route is None:
            os.environ.pop("TK_FA4_FP4PV_FWD_CONFIG", None)
        else:
            os.environ["TK_FA4_FP4PV_FWD_CONFIG"] = previous_route


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    selected_python = _resolve_executable(args.python)
    if args._worker:
        return _run_worker(args, selected_python)
    command = _worker_command(args, selected_python)
    if args.dry_run:
        print(json.dumps(_dry_run_plan(args, selected_python, command), indent=2))
        return 0
    completed = subprocess.run(command, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
