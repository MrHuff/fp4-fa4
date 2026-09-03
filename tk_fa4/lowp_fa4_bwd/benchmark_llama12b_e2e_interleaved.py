#!/usr/bin/env python3
"""Alternate BF16 and lowp full steps to remove sequential device drift."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from tk_fa4.lowp_fa4_bwd.benchmark_llama12b_e2e import (
    Config,
    Llama12B,
    LowpAttentionRuntime,
    _load_forward,
    _make_rope,
    _useful_flops,
)


def _optimizer(model: torch.nn.Module, learning_rate: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1.0e-8,
        weight_decay=0.0,
        fused=True,
    )


def _step(
    name: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    config: Config,
    round_index: int,
    *,
    warmup: bool,
) -> dict[str, float | bool | str]:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    forward_done = torch.cuda.Event(enable_timing=True)
    backward_done = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record()
    optimizer.zero_grad(set_to_none=True)
    logits = model(tokens)
    loss = F.cross_entropy(
        logits.reshape(-1, config.vocab),
        targets.reshape(-1),
        reduction="mean",
    )
    forward_done.record()
    loss.backward()
    backward_done.record()
    optimizer.step()
    end.record()
    end.synchronize()
    record: dict[str, float | bool | str] = {
        "route": name,
        "round": float(round_index),
        "warmup": warmup,
        "loss": float(loss.detach()),
        "finite": math.isfinite(float(loss.detach())),
        "forward_ms": float(start.elapsed_time(forward_done)),
        "backward_ms": float(forward_done.elapsed_time(backward_done)),
        "optimizer_ms": float(backward_done.elapsed_time(end)),
        "step_ms": float(start.elapsed_time(end)),
        "wall_ms": (time.perf_counter() - wall_start) * 1000.0,
    }
    print(
        f"round={round_index} route={name} warmup={warmup} "
        f"loss={record['loss']:.6f} step={record['step_ms']:.3f} ms",
        flush=True,
    )
    return record


def _summary(records: list[dict[str, Any]], config: Config) -> dict[str, float]:
    medians = {
        key: statistics.median(float(record[key]) for record in records)
        for key in (
            "forward_ms",
            "backward_ms",
            "optimizer_ms",
            "step_ms",
            "wall_ms",
        )
    }
    step_seconds = medians["step_ms"] / 1000.0
    useful_flops = _useful_flops(config)
    return {
        **medians,
        "tokens_per_second": config.sequence / step_seconds,
        "useful_tflops": useful_flops / step_seconds / 1.0e12,
        "mfu_at_2250_tflops": useful_flops / step_seconds / 2.25e15,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
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
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("expose exactly one GPU to the benchmark")
    if args.rounds < 2:
        raise ValueError("at least two alternating rounds are required")
    torch.cuda.set_device(0)
    config = Config()
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
        projection_dgrad="bf16",
    )

    torch.manual_seed(args.seed)
    bf16_model = Llama12B(config, rope, None)
    torch.manual_seed(args.seed)
    lowp_model = Llama12B(config, rope, runtime)
    bf16_optimizer = _optimizer(bf16_model, args.learning_rate)
    lowp_optimizer = _optimizer(lowp_model, args.learning_rate)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed + 101)
    tokens = torch.randint(
        config.vocab,
        (1, config.sequence),
        generator=generator,
        device="cuda",
    )
    targets = torch.roll(tokens, shifts=-1, dims=1)

    records: dict[str, list[dict[str, Any]]] = {
        "bf16_cute": [],
        "fp4_fa4_fused_qkv_rope": [],
    }
    routes = {
        "bf16_cute": (bf16_model, bf16_optimizer),
        "fp4_fa4_fused_qkv_rope": (lowp_model, lowp_optimizer),
    }
    # Compile and initialize optimizer state before the alternating samples.
    for name in ("bf16_cute", "fp4_fa4_fused_qkv_rope"):
        model, optimizer = routes[name]
        _step(
            name,
            model,
            optimizer,
            tokens,
            targets,
            config,
            -1,
            warmup=True,
        )

    torch.cuda.reset_peak_memory_stats()
    for round_index in range(args.rounds):
        order = (
            ("bf16_cute", "fp4_fa4_fused_qkv_rope")
            if round_index % 2 == 0
            else ("fp4_fa4_fused_qkv_rope", "bf16_cute")
        )
        for name in order:
            model, optimizer = routes[name]
            records[name].append(
                _step(
                    name,
                    model,
                    optimizer,
                    tokens,
                    targets,
                    config,
                    round_index,
                    warmup=False,
                )
            )

    bf16 = _summary(records["bf16_cute"], config)
    lowp = _summary(records["fp4_fa4_fused_qkv_rope"], config)
    result = {
        "configuration": {
            **config.__dict__,
            "batch": 1,
            "rounds": args.rounds,
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "measurement": "alternating_route_order",
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
            "backward_fp8_ds_lift": runtime.backward_fp8_ds_lift,
            "backward_probability_correction": (
                runtime.backward_probability_correction
            ),
            "forward_topology": topology,
        },
        "records": records,
        "bf16": bf16,
        "lowp": lowp,
        "comparison": {
            "speedup_lowp_over_bf16": bf16["step_ms"] / lowp["step_ms"],
            "step_time_reduction_percent": (
                1.0 - lowp["step_ms"] / bf16["step_ms"]
            ) * 100.0,
            "mfu_delta_percentage_points": 100.0
            * (lowp["mfu_at_2250_tflops"] - bf16["mfu_at_2250_tflops"]),
        },
        "memory": {
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2.0**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2.0**30,
        },
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
