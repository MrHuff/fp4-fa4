#!/usr/bin/env python3
"""Matched GB200 timing for native-TK v384/v387 and exact CuTe cd57.

The native cells remain runnable when the exact CuTe dependency chain is not
available.  In that case the JSON contains an explicit unavailable CuTe cell;
it never silently substitutes another CuTe source or another implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch


Q_HEADS = 32
KV_HEADS = 8
DEPTH = 64
SOFTMAX_SCALE = DEPTH**-0.5
CD57_SHA256 = (
    "cd57e3360082abe4bad7560c51a7793a4e9bfd4d16efc1259b92ce20238b99e1"
)
CD57_BYTES = 220876
CD57_CUTLASS_COMMIT = "b2ca083d2bb96c41d9b3c5a930637c641f6669bf"
THUNDERKITTENS_COMMIT = "9ee85b4afcdea1478b4dda8bb01f8907ab7edb0b"
FLASH_ATTENTION_COMMIT = "9743edaf3227a25f6afc4fa7be8b5e8498610553"
QUTLASS_COMMIT = "406e86fb2d7df436e94f825bcda8e59b1a7250a6"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path(
    "/volt/artifacts/native_tk_d64_ptx_adaptation_20260829/"
    "native_tk_d64_v387_b16_s4096_timing.json"
)
BUILD_INPUTS = (
    "tk_fa4/native_gqa_tk_bwd/Makefile.v384",
    "tk_fa4/native_gqa_tk_bwd/Makefile.v387",
    "tk_fa4/native_gqa_tk_bwd/v384_d64_gqa_e4m3_two_wg.cu",
    "tk_fa4/native_gqa_tk_bwd/v384_d64_gqa_e4m3_two_wg.cuh",
    "tk_fa4/native_gqa_tk_bwd/v385_d64_gqa_e4m3_k128q128.cuh",
    "tk_fa4/native_gqa_tk_bwd/v386_d64_gqa_e4m3_k128q128_halfcols.cuh",
    "tk_fa4/native_gqa_tk_bwd/v387_d64_gqa_e4m3_async_pipeline.cu",
    "tk_fa4/native_gqa_tk_bwd/v387_d64_gqa_e4m3_async_pipeline.cuh",
    "tk_fa4/native_gqa_tk_bwd/native_gqa_tk_bwd_pipelined.cuh",
    "tk_fa4/deprecated/fa4_common.cuh",
)


@dataclass(frozen=True)
class Runner:
    """One measured boundary and its persistent outputs."""

    prepare: Callable[[], None]
    call: Callable[[], None]
    outputs: Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    boundary: str


@dataclass
class Fixture:
    """Common BHSD operand/statistics tensors consumed by native candidates."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    dout: torch.Tensor
    l_aux: torch.Tensor
    delta: torch.Tensor
    mode: str
    exact_state: Any | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v384-extension", required=True, type=Path)
    parser.add_argument(
        "--v384-module", default="_C_b300_gqa_tk_bwd_v384"
    )
    parser.add_argument("--v387-extension", required=True, type=Path)
    parser.add_argument(
        "--v387-module",
        default="_C_sm100_gqa_tk_v387_d64_e4m3_async_pipeline",
    )
    parser.add_argument("--v384-resources", required=True, type=Path)
    parser.add_argument("--v387-resources", required=True, type=Path)
    parser.add_argument("--control-source", required=True, type=Path)
    parser.add_argument("--cute-setup-log", type=Path)
    parser.add_argument("--batch", default=16, type=int)
    parser.add_argument("--sequence", default=4096, type=int)
    parser.add_argument("--seed", default=20260825, type=int)
    parser.add_argument("--warmups", default=10, type=int)
    parser.add_argument("--samples", default=51, type=int)
    parser.add_argument("--output", default=DEFAULT_OUTPUT, type=Path)
    args = parser.parse_args()
    if args.batch <= 0 or args.batch > 65535:
        parser.error("--batch must be in [1, 65535]")
    if args.sequence <= 0 or args.sequence % 128:
        parser.error("--sequence must be a positive multiple of 128")
    if args.warmups != 10:
        parser.error("the publication protocol requires exactly 10 warmups")
    if args.samples != 51:
        parser.error("the publication protocol requires exactly 51 samples")
    return args


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def identify_file(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    payload = resolved.read_bytes()
    return {
        "path": str(resolved),
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


def authenticate_cd57(path: Path) -> dict[str, object]:
    identity = identify_file(path)
    if (
        identity["bytes"] != CD57_BYTES
        or identity["sha256"] != CD57_SHA256
    ):
        raise RuntimeError(
            "exact CuTe control drift: "
            f"bytes={identity['bytes']}, sha256={identity['sha256']}"
        )
    return identity


def run_text(command: list[str], *, cwd: Path = ROOT) -> str:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def optional_git_head(path: Path) -> str | None:
    try:
        return run_text(["git", "rev-parse", "HEAD"], cwd=path)
    except (OSError, subprocess.CalledProcessError):
        return None


def git_provenance() -> dict[str, object]:
    root_head = optional_git_head(ROOT)
    if root_head is not None:
        commit = root_head
        branch_or_head = run_text(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"]
        )
        tracked = {
            relative: bool(
                run_text(["git", "ls-files", "--error-unmatch", relative])
            )
            for relative in BUILD_INPUTS
        }
        dirty: bool | None = bool(
            run_text(
                ["git", "status", "--porcelain", "--untracked-files=no"]
            )
        )
        asset_commit = root_head
        provenance_mode = "repository_git_metadata"
    else:
        commit = os.environ.get("TKFA4_CANONICAL_SOURCE_COMMIT", "")
        asset_commit = os.environ.get("TKFA4_ASSET_COMMIT", "")
        if len(commit) != 40 or len(asset_commit) != 40:
            raise RuntimeError(
                "git-less code asset requires canonical and asset commit IDs"
            )
        branch_or_head = "git_metadata_intentionally_excluded_from_code_asset"
        tracked = {relative: True for relative in BUILD_INPUTS}
        dirty = None
        provenance_mode = "sha256_manifest_verified_gitless_asset"
    return {
        "commit": commit,
        "asset_commit": asset_commit,
        "branch_or_head": branch_or_head,
        "tracked_build_inputs": tracked,
        "tracked_worktree_dirty": dirty,
        "provenance_mode": provenance_mode,
        "git_asset_mode": os.environ.get("TKFA4_GIT_ASSET_MODE", "unknown"),
        "submodules": {
            "ThunderKittens": {
                "pinned": THUNDERKITTENS_COMMIT,
                "materialized": optional_git_head(ROOT / "ThunderKittens"),
            },
            "flash-attention": {
                "pinned": FLASH_ATTENTION_COMMIT,
                "materialized": optional_git_head(ROOT / "flash-attention"),
            },
            "qutlass": {
                "pinned": QUTLASS_COMMIT,
                "materialized": optional_git_head(ROOT / "qutlass"),
            },
            "exact_cd57_cutlass": {
                "required": CD57_CUTLASS_COMMIT,
                "materialized": optional_git_head(
                    Path(os.environ.get("TKFA4_CD57_CUTLASS_ROOT", "/missing"))
                ),
            },
        },
    }


def load_extension(name: str, path: Path) -> Any:
    resolved = path.resolve(strict=True)
    spec = importlib.util.spec_from_file_location(name, resolved)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot create extension spec for {resolved}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def percentile(sorted_values: list[float], fraction: float) -> float:
    position = fraction * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return (
        sorted_values[lower] * (1.0 - weight)
        + sorted_values[upper] * weight
    )


def summarize(values: list[float]) -> dict[str, object]:
    ordered = sorted(values)
    mean = statistics.fmean(ordered)
    stdev = statistics.pstdev(ordered)
    return {
        "unit": "ms",
        "count": len(ordered),
        "min": ordered[0],
        "p25": percentile(ordered, 0.25),
        "p50": percentile(ordered, 0.50),
        "median": percentile(ordered, 0.50),
        "p75": percentile(ordered, 0.75),
        "p95": percentile(ordered, 0.95),
        "max": ordered[-1],
        "mean": mean,
        "stdev": stdev,
        "cv": stdev / mean,
        "samples_in_collection_order": values,
    }


def compare(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, object]:
    reference_float = reference.float()
    actual_float = actual.float()
    difference = actual_float - reference_float
    reference_norm = reference_float.norm().clamp_min(1.0e-30)
    actual_norm = actual_float.norm().clamp_min(1.0e-30)
    return {
        "reference_finite": bool(torch.isfinite(reference_float).all()),
        "actual_finite": bool(torch.isfinite(actual_float).all()),
        "cosine": float(
            (reference_float * actual_float).sum()
            / (reference_norm * actual_norm)
        ),
        "relative_l2": float(difference.norm() / reference_norm),
        "norm_ratio": float(actual_norm / reference_norm),
        "max_abs": float(difference.abs().max()),
    }


def finite_outputs(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, bool]:
    return {
        name: bool(torch.isfinite(value).all())
        for name, value in zip(("dq", "dk", "dv"), outputs, strict=True)
    }


def sanitize_error(error: BaseException) -> str:
    message = str(error).replace("\n", " ")[:2000]
    message = re.sub(r"https://[^/@\s]+@", "https://<redacted>@", message)
    message = re.sub(
        r"(?i)(token|password|secret|access[_-]?key)(=|:)[^\s,;]+",
        r"\1\2<redacted>",
        message,
    )
    return message


def represented_e4m3(
    shape: tuple[int, ...], standard_deviation: float
) -> torch.Tensor:
    source = torch.randn(shape, device="cuda", dtype=torch.float32)
    source.mul_(standard_deviation)
    encoded = source.bfloat16().float().mul_(4.0).to(torch.float8_e4m3fn)
    del source
    return encoded


def make_native_fallback_fixture(args: argparse.Namespace) -> Fixture:
    """Make a stable native-only fixture without impersonating exact CuTe."""
    torch.manual_seed(args.seed)
    q_shape = (args.batch, Q_HEADS, args.sequence, DEPTH)
    kv_shape = (args.batch, KV_HEADS, args.sequence, DEPTH)
    q = represented_e4m3(q_shape, 0.25)
    k = represented_e4m3(kv_shape, 0.25)
    v = represented_e4m3(kv_shape, 0.25)
    dout = represented_e4m3(q_shape, 0.10)
    stats_shape = (args.batch, Q_HEADS, 1, args.sequence)
    direct_lse_log2 = torch.full(
        stats_shape, -4.0, device="cuda", dtype=torch.float32
    )
    l_aux = direct_lse_log2.div(SOFTMAX_SCALE * math.log2(math.e))
    delta = torch.zeros(stats_shape, device="cuda", dtype=torch.float32)
    return Fixture(
        q=q,
        k=k,
        v=v,
        dout=dout,
        l_aux=l_aux.contiguous(),
        delta=delta.contiguous(),
        mode="native_only_seeded_fixture_exact_cute_unavailable",
    )


def try_exact_cute(
    args: argparse.Namespace,
) -> tuple[Fixture | None, Any | None, dict[str, object]]:
    status: dict[str, object] = {
        "implementation": "exact CuTe-DSL cd57 public boundary",
        "required_source": {
            "bytes": CD57_BYTES,
            "sha256": CD57_SHA256,
            "cutlass_commit": CD57_CUTLASS_COMMIT,
        },
        "substitution_allowed": False,
    }
    if args.cute_setup_log is not None and args.cute_setup_log.exists():
        status["setup_log"] = identify_file(args.cute_setup_log)
    try:
        status["source"] = authenticate_cd57(args.control_source)
        from tk_fa4.lowp_fa4_bwd.tune_d64_gqa_cute import _load_control
        from tk_fa4.lowp_fa4_bwd.validate_causal_gqa_exact_backward_batch import (
            _build_lowp,
            _make_state,
            _publish_workspace_statistics,
        )

        state = _make_state(
            batch=args.batch,
            sequence=args.sequence,
            q_heads=Q_HEADS,
            kv_heads=KV_HEADS,
            seed=args.seed,
        )
        control = _load_control(
            fp8_p_storage="tmem",
            direct_tma_dkdv=True,
            precomposed_control_source=args.control_source,
            precomposed_control_sha256=CD57_SHA256,
            precomposed_control_bytes=CD57_BYTES,
        )
        cute_backward = _build_lowp(
            control, state, q_heads=Q_HEADS, kv_heads=KV_HEADS
        )
        _publish_workspace_statistics(cute_backward, state)
        # Compile before deciding that the exact cell is available.
        cute_backward.run(reset=True)
        torch.cuda.synchronize()

        fixture = Fixture(
            q=state.q_fp8.permute(0, 2, 1, 3).contiguous(),
            k=state.k_fp8.permute(0, 2, 1, 3).contiguous(),
            v=state.v_fp8.permute(0, 2, 1, 3).contiguous(),
            dout=state.dout_fp8.permute(0, 2, 1, 3).contiguous(),
            l_aux=(
                state.direct_lse_log2
                / (SOFTMAX_SCALE * math.log2(math.e))
            ).contiguous(),
            delta=(-state.direct_dpsum / 16.0).contiguous(),
            mode="exact_cd57_represented_forward_fixture",
            exact_state=state,
        )
        status["status"] = "available"
        status["public_boundary"] = "CompiledGqaBackward.run(reset=True)"
        return fixture, cute_backward, status
    except Exception as error:  # the unavailable cell must not kill native AB
        status.update(
            {
                "status": "unavailable",
                "reason": "exact cd57 dependency/build/runtime setup failed",
                "error_type": type(error).__name__,
                "error": sanitize_error(error),
                "replacement_used": False,
                "timing": None,
            }
        )
        torch.cuda.empty_cache()
        return None, None, status


def parse_resource_file(
    path: Path, *, kernel_marker: str
) -> dict[str, object]:
    identity = identify_file(path)
    text = path.read_text()
    pattern = re.compile(
        r"^ Function (?P<symbol>[^\n]*"
        + re.escape(kernel_marker)
        + r"[^\n]*):\n\s+(?P<usage>REG:[^\n]+)$",
        re.MULTILINE,
    )
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one {kernel_marker} resource record, found {len(matches)}"
        )
    fields = {
        name.lower(): int(value)
        for name, value in re.findall(r"([A-Z]+):([0-9]+)", matches[0]["usage"])
    }
    required = {"reg", "stack", "shared", "local"}
    if not required.issubset(fields):
        raise RuntimeError(f"incomplete resource record in {path}: {fields}")
    return {
        "artifact": identity,
        "kernel_symbol": matches[0]["symbol"],
        "static_resources": fields,
    }


def package_versions() -> dict[str, str | None]:
    names = (
        "torch",
        "nvidia-cutlass-dsl",
        "flashinfer-python",
        "apache-tvm-ffi",
        "torch-c-dlpack-ext",
        "quack-kernels",
        "ninja",
        "pybind11",
        "einops",
    )
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def machine_metadata() -> dict[str, object]:
    properties = torch.cuda.get_device_properties(0)
    try:
        smi = run_text(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,compute_cap,driver_version,"
                "memory.total,multiprocessor_count,clocks.max.sm",
                "--format=csv,noheader,nounits",
            ]
        )
    except (OSError, subprocess.CalledProcessError) as error:
        smi = f"unavailable: {sanitize_error(error)}"
    return {
        "requested": {
            "cluster": os.environ.get("TKFA4_VOLT_CLUSTER"),
            "machine_type": os.environ.get("TKFA4_VOLT_MACHINE_TYPE"),
            "num_gpus": int(os.environ.get("TKFA4_VOLT_NUM_GPUS", "1")),
            "image": os.environ.get("TKFA4_VOLT_IMAGE"),
        },
        "observed": {
            "cuda_device_count": torch.cuda.device_count(),
            "device_name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "multiprocessor_count": properties.multi_processor_count,
            "total_memory_bytes": properties.total_memory,
            "nvidia_smi": smi,
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "packages": package_versions(),
        },
    }


def snapshot_and_check(
    runners: dict[str, Runner]
) -> dict[str, dict[str, bool]]:
    result: dict[str, dict[str, bool]] = {}
    for name, runner in runners.items():
        runner.prepare()
        runner.call()
        torch.cuda.synchronize()
        result[name] = finite_outputs(runner.outputs())
    return result


def time_rotated(
    runners: dict[str, Runner], *, warmups: int, samples: int
) -> dict[str, dict[str, object]]:
    names = tuple(runners)
    for iteration in range(warmups):
        for offset in range(len(names)):
            runner = runners[names[(iteration + offset) % len(names)]]
            runner.prepare()
            runner.call()
    torch.cuda.synchronize()

    values = {name: [] for name in names}
    for iteration in range(samples):
        for offset in range(len(names)):
            name = names[(iteration + offset) % len(names)]
            runner = runners[name]
            # v387 main's caller-owned pre-clear is deliberately outside the
            # event boundary.  Other cells clear inside Runner.call().
            runner.prepare()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            runner.call()
            end.record()
            end.synchronize()
            values[name].append(float(start.elapsed_time(end)))
    return {name: summarize(samples_ms) for name, samples_ms in values.items()}


def pairwise_median_speedups(
    timing: dict[str, dict[str, object]]
) -> dict[str, float]:
    medians = {name: float(values["p50"]) for name, values in timing.items()}
    result: dict[str, float] = {}
    for baseline, baseline_ms in medians.items():
        for candidate, candidate_ms in medians.items():
            if baseline == candidate:
                continue
            result[f"{baseline}_over_{candidate}"] = baseline_ms / candidate_ms
    return result


def main() -> None:
    args = parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("benchmark requires exactly one visible GPU")
    torch.cuda.set_device(0)

    git = git_provenance()
    if git["tracked_worktree_dirty"] is True:
        raise RuntimeError("job code asset has tracked worktree modifications")
    if not all(git["tracked_build_inputs"].values()):
        raise RuntimeError("one or more v384/v387 build inputs are untracked")

    artifacts = {
        "build_inputs": {
            relative: identify_file(ROOT / relative) for relative in BUILD_INPUTS
        },
        "extensions": {
            "tk_v384": identify_file(args.v384_extension),
            "tk_v387": identify_file(args.v387_extension),
        },
        "cuobjdump": {
            "tk_v384": parse_resource_file(
                args.v384_resources,
                kernel_marker="v384_d64_gqa_e4m3_two_wg11main_kernel",
            ),
            "tk_v387": parse_resource_file(
                args.v387_resources,
                kernel_marker="v387_d64_gqa_e4m3_async_pipeline11main_kernel",
            ),
        },
    }

    fixture, cute_backward, cute_status = try_exact_cute(args)
    if fixture is None:
        fixture = make_native_fallback_fixture(args)

    v384 = load_extension(args.v384_module, args.v384_extension)
    v387 = load_extension(args.v387_module, args.v387_extension)
    q_shape = (args.batch, Q_HEADS, args.sequence, DEPTH)
    kv_shape = (args.batch, KV_HEADS, args.sequence, DEPTH)
    v384_outputs = (
        torch.empty(q_shape, device="cuda", dtype=torch.float32),
        torch.empty(kv_shape, device="cuda", dtype=torch.float32),
        torch.empty(kv_shape, device="cuda", dtype=torch.float32),
    )
    v387_main_outputs = (
        torch.empty(q_shape, device="cuda", dtype=torch.bfloat16),
        torch.empty(kv_shape, device="cuda", dtype=torch.bfloat16),
        torch.empty(kv_shape, device="cuda", dtype=torch.bfloat16),
    )
    v387_full_outputs = (
        torch.empty(q_shape, device="cuda", dtype=torch.bfloat16),
        torch.empty(kv_shape, device="cuda", dtype=torch.bfloat16),
        torch.empty(kv_shape, device="cuda", dtype=torch.bfloat16),
    )

    def clear_v387_main() -> None:
        for output in v387_main_outputs:
            output.zero_()

    def run_v387_main() -> None:
        v387.main_e4m3_bhsd(
            fixture.q,
            fixture.k,
            fixture.v,
            fixture.dout,
            fixture.l_aux,
            fixture.delta,
            *v387_main_outputs,
            SOFTMAX_SCALE,
        )

    def run_v387_full() -> None:
        v387.backward_e4m3_bhsd_out(
            fixture.q,
            fixture.k,
            fixture.v,
            fixture.dout,
            fixture.l_aux,
            fixture.delta,
            *v387_full_outputs,
            SOFTMAX_SCALE,
        )

    def run_v384_public() -> None:
        for output in v384_outputs:
            output.zero_()
        v384.main_e4m3_bhsd(
            fixture.q,
            fixture.k,
            fixture.v,
            fixture.dout,
            fixture.l_aux,
            fixture.delta,
            *v384_outputs,
            SOFTMAX_SCALE,
        )

    runners = {
        "tk_v387_main_preclear_excluded": Runner(
            prepare=clear_v387_main,
            call=run_v387_main,
            outputs=lambda: v387_main_outputs,
            boundary="caller pre-clears outputs before the start event",
        ),
        "tk_v387_full_with_clear": Runner(
            prepare=lambda: None,
            call=run_v387_full,
            outputs=lambda: v387_full_outputs,
            boundary=(
                "backward_e4m3_bhsd_out; three internal cudaMemsetAsync "
                "operations are inside the events"
            ),
        ),
        "tk_v384_public_with_clear": Runner(
            prepare=lambda: None,
            call=run_v384_public,
            outputs=lambda: v384_outputs,
            boundary=(
                "three caller-owned output zero_ operations followed by "
                "main_e4m3_bhsd, all inside the events"
            ),
        ),
    }
    if cute_backward is not None:
        runners["cute_cd57_public_with_clear"] = Runner(
            prepare=lambda: None,
            call=lambda: cute_backward.run(reset=True),
            outputs=lambda: tuple(
                output.permute(0, 2, 1, 3)
                for output in (
                    cute_backward.dq,
                    cute_backward.dk,
                    cute_backward.dv,
                )
            ),
            boundary="CompiledGqaBackward.run(reset=True), including reset",
        )

    pre_finite = snapshot_and_check(runners)
    correctness: dict[str, object] = {
        "tk_v387_main_vs_full": {
            gradient: compare(reference, actual)
            for gradient, reference, actual in zip(
                ("dq", "dk", "dv"),
                v387_full_outputs,
                v387_main_outputs,
                strict=True,
            )
        },
        "tk_v387_main_vs_v384": {
            gradient: compare(reference, actual)
            for gradient, reference, actual in zip(
                ("dq", "dk", "dv"),
                v384_outputs,
                v387_main_outputs,
                strict=True,
            )
        },
    }
    if cute_backward is not None:
        cute_outputs = runners["cute_cd57_public_with_clear"].outputs()
        for candidate, outputs in (
            ("tk_v384_vs_cute_cd57", v384_outputs),
            ("tk_v387_main_vs_cute_cd57", v387_main_outputs),
            ("tk_v387_full_vs_cute_cd57", v387_full_outputs),
        ):
            correctness[candidate] = {
                gradient: compare(reference, actual)
                for gradient, reference, actual in zip(
                    ("dq", "dk", "dv"), cute_outputs, outputs, strict=True
                )
            }

    timing = time_rotated(
        runners, warmups=args.warmups, samples=args.samples
    )
    post_finite = snapshot_and_check(runners)
    finite_pass = all(
        value
        for phase in (pre_finite, post_finite)
        for runner in phase.values()
        for value in runner.values()
    )

    metadata = {
        "tk_v384": dict(v384.native_tk_d64_backward_metadata()),
        "tk_v387": dict(v387.native_tk_d64_backward_metadata()),
    }
    metadata["tk_v384"]["cuobjdump_static_resources"] = artifacts[
        "cuobjdump"
    ]["tk_v384"]["static_resources"]
    metadata["tk_v387"]["cuobjdump_static_resources"] = artifacts[
        "cuobjdump"
    ]["tk_v387"]["static_resources"]

    document = {
        "schema": "tkfa4.native_v387_matched_b16_s4096.v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "shape": {
            "batch": args.batch,
            "sequence": args.sequence,
            "query_heads": Q_HEADS,
            "kv_heads": KV_HEADS,
            "head_dim": DEPTH,
            "causal": True,
        },
        "protocol": {
            "seed": args.seed,
            "warmups": args.warmups,
            "samples_per_cell": args.samples,
            "timing": "torch CUDA events on one stream",
            "order": "cell order rotates once per warmup/sample",
            "fixture": fixture.mode,
            "operand_contract": "same contiguous BHSD represented E4M3(4*x)",
            "statistics_contract": "same contiguous FP32 l_aux and delta",
            "boundaries": {
                name: runner.boundary for name, runner in runners.items()
            },
        },
        "exact_cute_cd57": cute_status,
        "git": git,
        "artifacts": artifacts,
        "metadata": metadata,
        "machine": machine_metadata(),
        "finite_checks": {
            "before_timing": pre_finite,
            "after_timing": post_finite,
            "pass": finite_pass,
        },
        "correctness": correctness,
        "timing": timing,
        "pairwise_median_speedups": pairwise_median_speedups(timing),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(document, indent=2, sort_keys=True)
    args.output.write_text(rendered + "\n")
    print(rendered, flush=True)
    if not finite_pass:
        raise RuntimeError("one or more pre/post timing finite checks failed")


if __name__ == "__main__":
    main()
