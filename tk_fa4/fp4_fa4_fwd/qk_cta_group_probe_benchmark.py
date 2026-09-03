#!/usr/bin/env python3
"""Compare work-normalized CTA-group1 and CTA-group2 MXFP4 QK issue."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
from pathlib import Path

import torch


ROWS = 128
COLS = 128
K = 192
K_PACKED = K // 2
SCALE_RECORD_BYTES = 1536
TRACE_FIELDS = 16
FLOPS_PER_RANK_ITERATION = 2 * ROWS * COLS * K
ROUTES = (
    "group1_issue_ceiling",
    "group2_issue_ceiling",
    "group1_production_cadence",
    "group2_production_cadence",
)


def load_extension(path: Path):
    name = "_C_tk_qk_cta_group_probe"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load extension from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def event_sample(fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def distribution(values: list[float]) -> dict[str, float | list[float]]:
    return {
        "min_ms": min(values),
        "median_ms": statistics.median(values),
        "max_ms": max(values),
        "mean_ms": statistics.fmean(values),
        "samples_ms": values,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--physical-ctas", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--samples", type=int, default=60)
    args = parser.parse_args()
    if args.physical_ctas % 2:
        raise ValueError("physical CTA count must be even")

    extension = load_extension(args.extension.resolve())
    device = torch.device("cuda:0")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260803)
    physical_ctas = args.physical_ctas
    clusters = physical_ctas // 2

    q_bytes = torch.randint(
        0, 256, (physical_ctas, ROWS, K_PACKED),
        generator=generator, dtype=torch.uint8,
    ).to(device)
    k_halves = torch.randint(
        0, 256, (clusters, 2, ROWS // 2, K_PACKED),
        generator=generator, dtype=torch.uint8,
    ).to(device)
    k_full = k_halves.reshape(clusters, ROWS, K_PACKED)
    k_group1_bytes = k_full.repeat_interleave(2, dim=0).contiguous()
    k_group2_bytes = k_halves.reshape(
        physical_ctas, ROWS // 2, K_PACKED
    ).contiguous()
    q = q_bytes.view(torch.float4_e2m1fn_x2)
    k_group1 = k_group1_bytes.view(torch.float4_e2m1fn_x2)
    k_group2 = k_group2_bytes.view(torch.float4_e2m1fn_x2)
    q_scale = torch.full(
        (physical_ctas * SCALE_RECORD_BYTES,), 0x38,
        dtype=torch.uint8, device=device,
    )
    k_scale = torch.full_like(q_scale, 0x38)
    scores = {
        route: torch.empty(
            (physical_ctas, ROWS, COLS), dtype=torch.float32, device=device
        )
        for route in ROUTES
    }
    traces = {
        route: torch.zeros(
            (physical_ctas, TRACE_FIELDS), dtype=torch.int64, device=device
        )
        for route in ROUTES
    }

    def launch(route: str, iterations: int, store_scores: bool) -> None:
        k = k_group2 if route.startswith("group2") else k_group1
        getattr(extension, f"mxfp4_qk_{route}")(
            q, q_scale, k, k_scale, scores[route], traces[route],
            iterations, store_scores,
        )

    # Check collective assembly before timing. Both routes see identical Q/K.
    for route in ROUTES:
        launch(route, 1, True)
    torch.cuda.synchronize()
    correctness = {}
    reference = scores["group1_issue_ceiling"]
    for route in ROUTES:
        candidate = scores[route]
        delta = (candidate - reference).abs()
        trace = traces[route].cpu()
        expected_group = 2 if route.startswith("group2") else 1
        issue_rows = trace[:, 5] == 0
        idle_rows = ~issue_rows
        timer_valid = bool(
            (trace[issue_rows, 9] > 0).all()
            and (trace[issue_rows, 10] > trace[issue_rows, 9]).all()
            and (
                expected_group == 1
                or (
                    (trace[idle_rows, 9] == 0).all()
                    and (trace[idle_rows, 10] == 0).all()
                )
            )
        )
        valid_trace = bool(
            (trace[:, 1] == expected_group).all()
            and (trace[:, 3] == 1).all()
            and (trace[:, 7] == physical_ctas).all()
            and timer_valid
        )
        correctness[route] = {
            "finite": bool(torch.isfinite(candidate).all().item()),
            "max_abs_vs_group1": float(delta.max().item()),
            "mean_abs_vs_group1": float(delta.mean().item()),
            "trace_valid": valid_trace,
        }
    if not all(
        row["finite"] and row["trace_valid"]
        and row["max_abs_vs_group1"] == 0.0
        for row in correctness.values()
    ):
        raise RuntimeError(f"correctness check failed: {correctness}")

    def timed_launch(route: str) -> None:
        launch(route, args.iterations, False)

    for sample_index in range(args.warmup):
        order = ROUTES if sample_index % 2 == 0 else tuple(reversed(ROUTES))
        for route in order:
            timed_launch(route)
    torch.cuda.synchronize()

    samples = {route: [] for route in ROUTES}
    for sample_index in range(args.samples):
        order = ROUTES if sample_index % 2 == 0 else tuple(reversed(ROUTES))
        for route in order:
            samples[route].append(event_sample(lambda route=route: timed_launch(route)))

    for route in ROUTES:
        timed_launch(route)
    torch.cuda.synchronize()

    results = {}
    total_flops = (
        FLOPS_PER_RANK_ITERATION * physical_ctas * args.iterations
    )
    for route in ROUTES:
        stats = distribution(samples[route])
        trace = traces[route].cpu()
        elapsed_cycles = (trace[:, 12] - trace[:, 11])
        elapsed_cycles = elapsed_cycles[elapsed_cycles > 0].tolist()
        results[route] = {
            **stats,
            "tflops": total_flops / (float(stats["median_ms"]) * 1.0e9),
            "median_issue_cycles": statistics.median(elapsed_cycles),
        }

    payload = {
        "schema": "mxfp4_qk_cta_group_ab_v2",
        "device": torch.cuda.get_device_properties(device).name,
        "physical_ctas": physical_ctas,
        "logical_m128_n128_k192_products": physical_ctas,
        "iterations": args.iterations,
        "correctness": correctness,
        "routes": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
