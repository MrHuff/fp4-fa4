#!/usr/bin/env python3
"""Measure saturated K128 TCGEN throughput with matched work per route."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
from pathlib import Path

import torch


ROUTES = {
    "raw_e2m1": 0,
    "scaled_mxfp4": 1,
    "bf16": 2,
    "dense_nvfp4": 6,
    "dedicated_mxfp4": 7,
    "dual_chain_mxfp4": 8,
    "triple_chain_mxfp4": 9,
    "dual_chain_k192_k64_mxfp4": 10,
    "dual_chain_k192_k96_mxfp4": 11,
    "dual_chain_nvfp4": 12,
    "dual_chain_k192_k64_nvfp4": 13,
    "dual_chain_k192_k96_nvfp4": 14,
    "parallel_dual_chain_mxfp4": 15,
    "parallel_triple_chain_mxfp4": 16,
    "four_cta_n64_mxfp4": 17,
    "four_cta_n96_mxfp4": 18,
    "eight_cta_n32_mxfp4": 19,
    "one_cta_n256_mxfp4": 20,
    "one_cta_attention_n96_k96_mxfp4": 21,
    "two_cta_attention_n96_k96_mxfp4": 22,
    "one_cta_attention_n128_split_qk_mxfp4": 23,
    "two_cta_attention_n128_split_qk_mxfp4": 24,
    "one_cta_attention_n96_nvqk_k96_mxpv": 27,
    "two_cta_attention_n96_nvqk_k96_mxpv": 28,
    "one_cta_attention_n96_production_scale_copy": 29,
    "two_cta_attention_n96_production_scale_copy": 30,
    "triple_chain_k192_k96_mxfp4": 31,
    "triple_chain_k192_k96_nvfp4": 32,
    "dual_chain_k192_k96_sm103_mxfp4": 33,
    "triple_chain_k192_k96_sm103_mxfp4": 34,
    "dual_score_collector_mxfp4": 35,
    "dual_score_collector_nvfp4": 36,
    "one_cta_attention_n192_ultra_nvqk_mxpv": 37,
    "one_cta_attention_n256_ultra_nvqk_mxpv": 38,
    "one_cta_attention_n256_dual_query_prefix_swap": 39,
}
DEFAULT_ROUTES = tuple(
    name
    for name in ROUTES
    if "k192" not in name
    and "parallel_" not in name
    and "_cta_" not in name
)
LOGICAL_K_BY_ROUTE = {
    name: (
        256
        if "attention_n256" in name
        else 192
        if "k192" in name or "attention_n192" in name
        else 128
    )
    for name in ROUTES
}
OUTPUT_N_BY_ROUTE = {
    name: (
        256
        if "n256" in name
        else 192
        if "n192" in name
        else 32
        if "n32" in name
        else 64
        if "n64" in name
        else 96
        if "n96" in name
        else 128
    )
    for name in ROUTES
}
BLOCK_RATIO_BY_ROUTE = {
    name: (
        (2, 1)
        if name == "four_cta_n64_mxfp4"
        else (1, 2)
        if name == "one_cta_n256_mxfp4"
        else (2, 3)
        if name == "one_cta_attention_n192_ultra_nvqk_mxpv"
        else (1, 2)
        if name == "one_cta_attention_n256_ultra_nvqk_mxpv"
        else (1, 4)
        if name == "one_cta_attention_n256_dual_query_prefix_swap"
        else (1, 1)
    )
    for name in ROUTES
}


def load_extension(path: Path):
    module_name = "_C_tk_raw_fp4_throughput"
    spec = importlib.util.spec_from_file_location(module_name, path)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sm-count", type=int, required=True)
    parser.add_argument("--blocks", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--routes", default=",".join(DEFAULT_ROUTES))
    args = parser.parse_args()

    selected = [name for name in args.routes.split(",") if name]
    unknown = set(selected) - set(ROUTES)
    if unknown:
        raise ValueError(f"unknown routes: {sorted(unknown)}")
    blocks = args.blocks or args.sm_count * 12

    extension = load_extension(args.extension.resolve())
    device = torch.device("cuda:0")
    if (
        any("k96" in name or "ultra" in name for name in selected)
        and torch.cuda.get_device_capability(device) < (10, 3)
    ):
        raise RuntimeError("the scaled-FP4 K96 routes require SM103 or newer")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260803)
    # The N192 Ultra probe needs 192 packed rows. Existing routes consume only
    # the leading 128 rows, so one shared allocation keeps their inputs intact.
    a_u8 = torch.randint(
        0, 256, (256, 64), generator=generator, dtype=torch.uint8
    ).to(device)
    b_u8 = torch.randint(
        0, 256, (256, 64), generator=generator, dtype=torch.uint8
    ).to(device)
    a_fp4 = a_u8.view(torch.float4_e2m1fn_x2)
    b_fp4 = b_u8.view(torch.float4_e2m1fn_x2)
    a_bf16 = torch.randn(
        (128, 128), generator=generator, dtype=torch.float32
    ).to(device, torch.bfloat16)
    b_bf16 = torch.randn(
        (128, 128), generator=generator, dtype=torch.float32
    ).to(device, torch.bfloat16)

    launches = {}
    route_traces = {}
    route_blocks = {}
    for name in selected:
        # Match each route's total arithmetic to the requested N128 block
        # count. N96 and N32 remain separate component measurements whose
        # summed work is exactly one N128 tile.
        block_numerator, block_denominator = BLOCK_RATIO_BY_ROUTE[name]
        scaled_blocks = blocks * block_numerator
        if scaled_blocks % block_denominator:
            raise ValueError(
                f"{name}: base block count {blocks} is not divisible by "
                f"{block_denominator}"
            )
        blocks_for_route = scaled_blocks // block_denominator
        if blocks_for_route > 4096:
            raise ValueError(
                f"{name}: equal-work block count {blocks_for_route} exceeds 4096"
            )
        route_blocks[name] = blocks_for_route
        traces = (
            torch.zeros(blocks_for_route, device=device, dtype=torch.int64),
            torch.zeros(blocks_for_route, device=device, dtype=torch.int64),
            torch.zeros(blocks_for_route, device=device, dtype=torch.int64),
            torch.full(
                (blocks_for_route,), -1, device=device, dtype=torch.int32
            ),
        )

        def launch(
            mode=ROUTES[name],
            traces=traces,
            blocks_for_route=blocks_for_route,
        ):
            extension.raw_fp4_throughput_probe(
                a_fp4,
                b_fp4,
                a_bf16,
                b_bf16,
                *traces,
                mode,
                args.iterations,
                blocks_for_route,
            )

        launches[name] = launch
        route_traces[name] = traces

    for sample_index in range(args.warmup):
        order = selected if sample_index % 2 == 0 else reversed(selected)
        for name in order:
            launches[name]()
    torch.cuda.synchronize()

    samples_by_route = {name: [] for name in selected}
    for sample_index in range(args.samples):
        order = selected if sample_index % 2 == 0 else reversed(selected)
        for name in order:
            samples_by_route[name].append(event_sample(launches[name]))

    for name in selected:
        launches[name]()
    torch.cuda.synchronize()

    results = {}
    for name in selected:
        times_ms = samples_by_route[name]
        cycles, starts, ends, smids = (
            tensor.cpu() for tensor in route_traces[name]
        )
        valid = (cycles > 0) & (ends > starts) & (smids >= 0)
        blocks_for_route = route_blocks[name]
        if int(valid.sum()) != blocks_for_route:
            raise RuntimeError(
                f"{name}: valid records "
                f"{int(valid.sum())}/{blocks_for_route}"
            )
        median_ms = statistics.median(times_ms)
        logical_k = LOGICAL_K_BY_ROUTE[name]
        output_n = OUTPUT_N_BY_ROUTE[name]
        if "attention_n96" in name:
            # One QK M128xN96xK128 plus one PV M128xN128xK96.
            flops_per_cta_iteration = 2 * 128 * (
                96 * 128 + 128 * 96
            )
        elif "attention_n128_split_qk" in name:
            # One QK and one PV, both M128xN128xK128. QK is issued as
            # disjoint N96 and N32 destinations to expose scale scratch.
            flops_per_cta_iteration = 2 * 128 * 128 * (128 + 128)
        elif "attention_n192" in name:
            # One QK M128xN192xK128 plus one PV M128xN128xK192.
            flops_per_cta_iteration = 2 * 128 * (
                192 * 128 + 128 * 192
            )
        elif "attention_n256" in name:
            # One QK M128xN256xK128 plus one PV M128xN128xK256.
            flops_per_cta_iteration = 2 * 128 * (
                256 * 128 + 128 * 256
            )
            if "dual_query" in name:
                flops_per_cta_iteration *= 2
        else:
            flops_per_cta_iteration = 2 * 128 * output_n * logical_k
        total_flops = (
            flops_per_cta_iteration
            * args.iterations
            * blocks_for_route
        )
        results[name] = {
            "mode": ROUTES[name],
            "output_n": output_n,
            "blocks": blocks_for_route,
            "median_ms": median_ms,
            "min_ms": min(times_ms),
            "max_ms": max(times_ms),
            "tflops": total_flops / (median_ms * 1.0e9),
            "logical_k_per_cta_iteration": logical_k,
            "flops_per_cta_iteration": flops_per_cta_iteration,
            "median_cta_cycles": statistics.median(
                int(value) for value in cycles[valid]
            ),
            "median_cta_ns": statistics.median(
                int(value) for value in (ends - starts)[valid]
            ),
            "observed_sms": len(set(int(value) for value in smids[valid])),
            "samples_ms": times_ms,
        }

    properties = torch.cuda.get_device_properties(device)
    payload = {
        "schema": "tcgen05_mxfp4_throughput_v4_equal_work",
        "device": properties.name,
        "sm_count_requested": args.sm_count,
        "base_n128_blocks": blocks,
        "iterations": args.iterations,
        "routes": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
