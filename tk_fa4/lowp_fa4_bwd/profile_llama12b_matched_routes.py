#!/usr/bin/env python3
"""Profile matched MXFP4-PV and FP8-PV routes after full-step acclimation."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import torch

from tk_fa4.lowp_fa4_bwd import benchmark_llama12b_e2e as benchmark
from tk_fa4.lowp_fa4_bwd.benchmark_llama12b_e2e import (
    DEFAULT_MODEL_PRESET,
    MODEL_PRESETS,
    Config,
    Llama12B,
    _make_llama3_rope,
    config_from_model_preset,
)
from tk_fa4.lowp_fa4_bwd.compare_llama12b_mx_fp8pv import (
    LOWP_ROUTE_NAMES,
    _acclimate_block_iteration,
    _assign_model_lowp_runtime,
    _make_runtime,
    _make_legacy_rope,
    _matched_backward_contracts,
    _optimizer,
    _share_matched_backward_runner,
    _step,
)


MX_ROUTE, FP8_ROUTE = LOWP_ROUTE_NAMES


def _device_time_us(event: Any) -> float:
    for name in ("device_time_total", "cuda_time_total"):
        value = getattr(event, name, None)
        if value is not None:
            return float(value)
    return 0.0


def _profile_step(
    route: str,
    model: Llama12B,
    optimizer: torch.optim.Optimizer,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    config: Config,
    forward_routes: dict[str, str],
    trace: Path,
) -> dict[str, Any]:
    benchmark._PROFILE_STAGE_RANGES = True
    try:
        with torch.profiler.profile(
            activities=(
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        ) as profile:
            record = _step(
                route,
                model,
                optimizer,
                tokens,
                targets,
                config,
                0,
                0,
                0,
                forward_routes,
                warmup=False,
            )
    finally:
        benchmark._PROFILE_STAGE_RANGES = False
    trace.parent.mkdir(parents=True, exist_ok=True)
    profile.export_chrome_trace(str(trace))
    averages = list(profile.key_averages())
    stages = []
    for event in averages:
        if not event.key.startswith("lowp/"):
            continue
        stages.append(
            {
                "name": event.key,
                "calls": int(event.count),
                "cpu_total_us": float(event.cpu_time_total),
                "device_total_us": _device_time_us(event),
            }
        )
    stages.sort(key=lambda item: item["name"])
    kernels = []
    for event in averages:
        device_us = _device_time_us(event)
        if device_us <= 0.0 or event.key.startswith("lowp/"):
            continue
        kernels.append(
            {
                "name": event.key,
                "calls": int(event.count),
                "cpu_total_us": float(event.cpu_time_total),
                "device_total_us": device_us,
            }
        )
    kernels.sort(key=lambda item: item["device_total_us"], reverse=True)
    return {
        "timing": record,
        "stages": stages,
        "top_device_events": kernels[:80],
        "trace": str(trace.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-preset",
        choices=MODEL_PRESETS,
        default=DEFAULT_MODEL_PRESET,
    )
    parser.add_argument("--layers", type=int)
    parser.add_argument("--sequence", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--acclimation-steps", type=int, default=12)
    parser.add_argument("--timed-steps", type=int, default=12)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument(
        "--per-block-qk-scales",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="publish one forward Q/K scale per logical row x K16",
    )
    parser.add_argument(
        "--profile-each-block",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="capture one CUDA trace after every timed route block",
    )
    parser.add_argument("--mx-extension", type=Path, required=True)
    parser.add_argument("--mx-module", required=True)
    parser.add_argument("--fp8-extension", type=Path, required=True)
    parser.add_argument("--fp8-module", required=True)
    parser.add_argument("--backward-control-source", type=Path)
    parser.add_argument("--backward-control-sha256")
    parser.add_argument("--backward-control-bytes", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-directory", type=Path, required=True)
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("expose exactly one GPU to the profiler")
    if min(args.acclimation_steps, args.timed_steps, args.cycles) < 1:
        raise ValueError("acclimation steps, timed steps, and cycles must be positive")

    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    config = config_from_model_preset(
        args.model_preset,
        sequence=args.sequence,
        layers=args.layers,
    )
    control_identity = (
        args.backward_control_source,
        args.backward_control_sha256,
        args.backward_control_bytes,
    )
    if config.head_dim == 64 and not all(
        value is not None for value in control_identity
    ):
        raise ValueError(
            "D64 profiling requires the backward control source, SHA256, and byte size"
        )
    if config.head_dim == 128 and any(
        value is not None for value in control_identity
    ):
        raise ValueError(
            "D128 uses generated shared-P control; omit D64 control arguments"
        )
    rope = (
        _make_llama3_rope(config)
        if config.head_dim == 128
        else _make_legacy_rope(config.sequence, config.head_dim)
    )
    d128 = config.head_dim == 128
    common = {
        "backward_probability_correction": 1.0 if d128 else None,
        "q_quant_scale": 2.25,
        "k_quant_scale": 2.0,
        "projection_weight_scale_2d": True,
        "v_mxfp4_scale_2d": False,
        "backward_q_gain": None,
        "backward_k_gain": None,
        "backward_v_gain": None,
        "backward_v_weight_gain": None,
        "backward_exp2_degree": 1,
        "backward_exp2_period": 0 if d128 else 2,
        "backward_control_source": args.backward_control_source,
        "backward_control_sha256": args.backward_control_sha256,
        "backward_control_bytes": args.backward_control_bytes,
        "backward_reuse_quantized_p": d128,
        "backward_match_forward_operands": not d128,
        "per_block_qk_scales": args.per_block_qk_scales,
        "backward_forward_mx_probability_replay": False,
        "backward_forward_mx_probability_scale_handoff": False,
        "qkv_projection_format": "nvfp4" if d128 else "e4m3",
    }
    mx_runtime, mx_topology = _make_runtime(
        config,
        rope,
        args.mx_extension,
        args.mx_module,
        route_slot="mx",
        experimental_split_v_backward=not d128,
        **common,
    )
    fp8_runtime, fp8_topology = _make_runtime(
        config,
        rope,
        args.fp8_extension,
        args.fp8_module,
        route_slot="fp8",
        experimental_split_v_backward=False,
        shared_backward_runtime=mx_runtime,
        **common,
    )
    backward_contracts = _matched_backward_contracts(
        mx_runtime,
        fp8_runtime,
    )
    shared = _share_matched_backward_runner(mx_runtime, fp8_runtime)
    if not all(shared.values()):
        raise RuntimeError("the two routes do not share one physical backward")
    runtimes = {MX_ROUTE: mx_runtime, FP8_ROUTE: fp8_runtime}
    forward_routes = {
        MX_ROUTE: str(mx_topology["route"]),
        FP8_ROUTE: str(fp8_topology["route"]),
    }

    torch.manual_seed(args.seed)
    model = Llama12B(config, rope, mx_runtime)
    optimizer = _optimizer(model, 0.0)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed + 101)
    tokens = torch.randint(
        config.vocab,
        (1, config.sequence),
        generator=generator,
        device="cuda",
    )
    targets = torch.roll(tokens, shifts=-1, dims=1)
    model.require_lowp_forward_workspace_stream()

    # Authenticate both compact projection ABIs before any measured block.
    for route in LOWP_ROUTE_NAMES:
        _assign_model_lowp_runtime(model, runtimes[route])
        _acclimate_block_iteration(
            route,
            model,
            optimizer,
            tokens,
            targets,
            config,
            forward_routes,
        )

    records: dict[str, list[dict[str, Any]]] = {
        route: [] for route in LOWP_ROUTE_NAMES
    }
    profiles: list[dict[str, Any]] = []
    block_records: list[dict[str, Any]] = []
    abba_order = (MX_ROUTE, FP8_ROUTE, FP8_ROUTE, MX_ROUTE)
    baab_order = (FP8_ROUTE, MX_ROUTE, MX_ROUTE, FP8_ROUTE)
    for cycle in range(args.cycles):
        order = abba_order if cycle % 2 == 0 else baab_order
        for position, route in enumerate(order):
            _assign_model_lowp_runtime(model, runtimes[route])
            for _ in range(args.acclimation_steps):
                _acclimate_block_iteration(
                    route,
                    model,
                    optimizer,
                    tokens,
                    targets,
                    config,
                    forward_routes,
                )
            block_samples = []
            for step_index in range(args.timed_steps):
                sample = _step(
                    route,
                    model,
                    optimizer,
                    tokens,
                    targets,
                    config,
                    step_index,
                    0,
                    position,
                    forward_routes,
                    warmup=False,
                )
                records[route].append(sample)
                block_samples.append(sample)
            block_records.append(
                {
                    "cycle": cycle,
                    "position": position,
                    "route": route,
                    "medians": {
                        field: statistics.median(
                            float(sample[field]) for sample in block_samples
                        )
                        for field in (
                            "forward_ms",
                            "backward_ms",
                            "optimizer_ms",
                            "step_ms",
                            "wall_ms",
                        )
                    },
                }
            )
            if args.profile_each_block:
                trace = args.trace_directory / (
                    f"cycle{cycle}_position{position}_{route}.json"
                )
                profile_result = _profile_step(
                    route,
                    model,
                    optimizer,
                    tokens,
                    targets,
                    config,
                    forward_routes,
                    trace,
                )
                profile_result.update(
                    {"cycle": cycle, "position": position, "route": route}
                )
                profiles.append(profile_result)

    timing = {}
    for route, route_records in records.items():
        timing[route] = {
            field: statistics.median(float(record[field]) for record in route_records)
            for field in ("forward_ms", "backward_ms", "optimizer_ms", "step_ms", "wall_ms")
        }
        timing[route]["samples"] = len(route_records)
    result = {
        "schema": "matched_llama_sustained_route_profile_v2",
        "configuration": {
            "model_preset": args.model_preset,
            "layers": config.layers,
            "sequence": config.sequence,
            "acclimation_steps": args.acclimation_steps,
            "timed_steps": args.timed_steps,
            "cycles": args.cycles,
            "profile_each_block": args.profile_each_block,
            "per_block_qk_scales": args.per_block_qk_scales,
            "seed": args.seed,
        },
        "shared_backward": shared,
        "backward_contracts": backward_contracts,
        "forward_topologies": {
            MX_ROUTE: mx_topology,
            FP8_ROUTE: fp8_topology,
        },
        "timing": timing,
        "blocks": block_records,
        "profiles": profiles,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
