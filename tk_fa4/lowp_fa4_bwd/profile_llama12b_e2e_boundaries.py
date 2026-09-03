#!/usr/bin/env python3
"""Profile the materialization boundaries in one assembled lowp Llama layer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from tk_fa4.lowp_fa4_bwd import benchmark_llama12b_e2e as benchmark
from tk_fa4.lowp_fa4_bwd.benchmark_llama12b_e2e import (
    Config,
    Llama12B,
    LowpAttentionRuntime,
    _load_forward,
    _make_rope,
)


def _device_time_us(event: Any) -> float:
    for name in ("device_time_total", "cuda_time_total"):
        value = getattr(event, name, None)
        if value is not None:
            return float(value)
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument(
        "--projection-dgrad", choices=("bf16", "nvfp4"), default="bf16"
    )
    parser.add_argument(
        "--forward-extension",
        type=Path,
        default=Path(
            "/tmp/_C_tk_gb200_causal_s4096_h32_d64."
            "cpython-312-aarch64-linux-gnu.so"
        ),
    )
    parser.add_argument(
        "--forward-module", default="_C_tk_gb200_causal_s4096_h32_d64"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trace", type=Path)
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("expose exactly one GPU to the profiler")
    if args.warmups < 1:
        raise ValueError("at least one warmup is required")
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    benchmark._PROFILE_STAGE_RANGES = True

    config = Config(layers=1)
    extension, topology = _load_forward(
        args.forward_extension, args.forward_module, config
    )
    rope = _make_rope(config.sequence, config.head_dim)
    runtime = LowpAttentionRuntime(
        config,
        rope,
        forward_extension=extension,
        forward_topology=topology,
        loss_scale=2.0**16,
        gradient_global_scale=2.0**-8,
        projection_dgrad=args.projection_dgrad,
    )
    model = Llama12B(config, rope, runtime)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed + 1)
    tokens = torch.randint(
        config.vocab,
        (1, config.sequence),
        generator=generator,
        device="cuda",
    )
    targets = torch.roll(tokens, shifts=-1, dims=1)

    def step() -> torch.Tensor:
        model.zero_grad(set_to_none=True)
        logits = model(tokens)
        loss = F.cross_entropy(
            logits.reshape(-1, config.vocab),
            targets.reshape(-1),
            reduction="mean",
        )
        loss.backward()
        return loss

    for _ in range(args.warmups):
        loss = step()
        torch.cuda.synchronize()
        if not torch.isfinite(loss):
            raise RuntimeError("warmup produced a non-finite loss")

    with torch.profiler.profile(
        activities=(
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as profile:
        loss = step()
        torch.cuda.synchronize()

    averages = list(profile.key_averages())
    boundaries = []
    for event in averages:
        if not event.key.startswith("lowp/"):
            continue
        boundaries.append(
            {
                "name": event.key,
                "calls": int(event.count),
                "cpu_total_us": float(event.cpu_time_total),
                "device_total_us": _device_time_us(event),
            }
        )
    boundaries.sort(key=lambda value: value["device_total_us"], reverse=True)

    top_device = []
    for event in sorted(
        averages,
        key=_device_time_us,
        reverse=True,
    )[:80]:
        device_us = _device_time_us(event)
        if device_us <= 0.0:
            continue
        top_device.append(
            {
                "name": event.key,
                "calls": int(event.count),
                "cpu_total_us": float(event.cpu_time_total),
                "device_total_us": device_us,
            }
        )

    result = {
        "configuration": {
            **config.__dict__,
            "batch": 1,
            "warmups": args.warmups,
            "seed": args.seed,
            "projection_dgrad": args.projection_dgrad,
            "backward_exp2_degree": runtime.backward_exp2_degree,
            "backward_exp2_period": runtime.backward_exp2_period,
            "backward_exp2_requested_degree": (
                runtime.backward_exp2_requested_degree
            ),
            "backward_exp2_requested_period": (
                runtime.backward_exp2_requested_period
            ),
            "backward_exp2_policy": runtime.backward_exp2_policy,
            "backward_detached_fp8_p_tmem": (
                runtime.backward_detached_fp8_p_tmem
            ),
            "backward_probability_tmem_policy": (
                runtime.backward_probability_tmem_policy
            ),
            "backward_head_fast_raster": runtime.backward_head_fast_raster,
            "backward_raster_policy": runtime.backward_raster_policy,
        },
        "loss": float(loss.detach()),
        "boundaries": boundaries,
        "top_device_events": top_device,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    if args.trace is not None:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        profile.export_chrome_trace(str(args.trace))


if __name__ == "__main__":
    main()
