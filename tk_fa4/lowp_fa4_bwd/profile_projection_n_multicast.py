#!/usr/bin/env python3
"""Correctness- and timing-gate the projection-N multicast prototype."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sysconfig
from pathlib import Path
from typing import Callable

import torch

from tk_fa4 import b300_prepare_nvfp4_projection_operand, b300_project_nvfp4


def _load_extension(path: Path, module: str):
    spec = importlib.util.spec_from_file_location(module, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load extension {path}")
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def _metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float | bool]:
    reference_f = reference.float()
    actual_f = actual.float()
    difference = actual_f - reference_f
    reference_norm = torch.linalg.vector_norm(reference_f).clamp_min(1.0e-20)
    actual_norm = torch.linalg.vector_norm(actual_f).clamp_min(1.0e-20)
    return {
        "finite": bool(torch.isfinite(actual_f).all()),
        "cosine": float(
            torch.sum(reference_f * actual_f) / (reference_norm * actual_norm)
        ),
        "relative_l2": float(torch.linalg.vector_norm(difference) / reference_norm),
        "max_abs": float(difference.abs().max()),
    }


def _time_cuda(
    function: Callable[[], object],
    *,
    warmups: int,
    samples: int,
) -> dict[str, float | list[float]]:
    for _ in range(warmups):
        function()
    torch.cuda.synchronize()
    timings: list[float] = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        timings.append(float(start.elapsed_time(end) * 1000.0))
    return {
        "median_us": statistics.median(timings),
        "minimum_us": min(timings),
        "samples_us": timings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--reduction", type=int, default=6144)
    parser.add_argument("--output-width", type=int, default=4096)
    parser.add_argument("--clusters", type=int, nargs="+", default=(4, 8, 16))
    parser.add_argument("--cluster-cap", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument(
        "--stress-runs",
        type=int,
        default=1,
        help="number of independently checked launches per cluster shape",
    )
    parser.add_argument("--seed", type=int, default=2026081601)
    parser.add_argument("--module", default="_C_tk_projection_n_multicast_probe")
    parser.add_argument("--extension", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise RuntimeError("profiling requires exactly one visible GPU")
    for name, value in (
        ("rows", args.rows),
        ("reduction", args.reduction),
        ("output-width", args.output_width),
    ):
        if value % 256:
            parser.error(f"--{name} must be divisible by 256")
    for cluster in args.clusters:
        if cluster not in (2, 4, 8, 16):
            parser.error("--clusters entries must be 2, 4, 8, or 16")
        if args.output_width % (cluster * 128):
            parser.error(
                f"output width {args.output_width} is not divisible by "
                f"the cluster-{cluster} N supertile"
            )

    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    extension = args.extension or Path("/tmp") / f"{args.module}{suffix}"
    probe = _load_extension(extension.resolve(), args.module)
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    a = (torch.randn(args.rows, args.reduction, device="cuda") * 0.1).bfloat16()
    b = (
        torch.randn(args.output_width, args.reduction, device="cuda") * 0.02
    ).bfloat16()
    a_operand = tuple(b300_prepare_nvfp4_projection_operand(a))
    b_operand = tuple(b300_prepare_nvfp4_projection_operand(b))
    reference = b300_project_nvfp4(a_operand, b_operand)
    torch.cuda.synchronize()

    quality: dict[str, dict[str, float | bool | int]] = {}
    for cluster in args.clusters:
        name = f"cluster_{cluster}"
        runs = []
        for _ in range(args.stress_runs):
            output = probe.project_nvfp4_n_multicast(
                *a_operand,
                *b_operand,
                cluster,
                args.cluster_cap,
            )
            runs.append(_metrics(reference, output))
        quality[name] = {
            "runs": len(runs),
            "all_finite": all(bool(run["finite"]) for run in runs),
            "minimum_cosine": min(float(run["cosine"]) for run in runs),
            "maximum_relative_l2": max(float(run["relative_l2"]) for run in runs),
            "maximum_abs": max(float(run["max_abs"]) for run in runs),
        }

    timing = {
        "retained_two_cta": _time_cuda(
            lambda: b300_project_nvfp4(a_operand, b_operand),
            warmups=args.warmups,
            samples=args.samples,
        )
    }
    for cluster in args.clusters:
        timing[f"cluster_{cluster}"] = _time_cuda(
            lambda cluster=cluster: probe.project_nvfp4_n_multicast(
                *a_operand,
                *b_operand,
                cluster,
                args.cluster_cap,
            ),
            warmups=args.warmups,
            samples=args.samples,
        )

    retained_us = float(timing["retained_two_cta"]["median_us"])
    speedup = {
        name: retained_us / float(values["median_us"])
        for name, values in timing.items()
        if name != "retained_two_cta"
    }
    result = {
        "shape": {
            "rows": args.rows,
            "reduction": args.reduction,
            "output_width": args.output_width,
        },
        "cluster_cap": args.cluster_cap,
        "quality_vs_retained": quality,
        "timing": timing,
        "speedup_vs_retained": speedup,
    }
    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
