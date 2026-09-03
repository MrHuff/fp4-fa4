#!/usr/bin/env python3
"""Validate and time the D128 QKV weight-gradient stitch."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Callable

import torch

import tk_fa4.interface as tk_interface
from tk_fa4 import b300_stitch_gqa_d128_inverse_rope_gradient
from tk_fa4.lowp_fa4_bwd.benchmark_llama12b_e2e import (
    _make_llama3_rope,
    config_from_model_preset,
)
from tk_fa4.lowp_fa4_bwd.profile_gqa_d128_chain import (
    _inverse_rope_pair_native,
)


def _time_cuda(
    function: Callable[[], object],
    *,
    warmups: int,
    samples: int,
) -> dict[str, Any]:
    for _ in range(warmups):
        function()
    torch.cuda.synchronize()
    values = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end) * 1000.0))
    return {
        "median_us": statistics.median(values),
        "minimum_us": min(values),
        "maximum_us": max(values),
        "samples_us": values,
    }


def _file_identity(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, choices=(1, 2), default=1)
    parser.add_argument("--sequence", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--warmups", type=int, default=8)
    parser.add_argument("--samples", type=int, default=25)
    parser.add_argument("--q-gradient-scale", type=float, default=0.375)
    parser.add_argument("--k-gradient-scale", type=float, default=0.25)
    parser.add_argument("--v-gradient-scale", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("expose exactly one GPU to the validator")
    if min(args.sequence, args.warmups, args.samples) < 1:
        raise ValueError("sequence, warmups, and samples must be positive")
    scales = (
        args.q_gradient_scale,
        args.k_gradient_scale,
        args.v_gradient_scale,
    )
    if any(not value > 0.0 for value in scales):
        raise ValueError("Q/K/V gradient scales must be positive")

    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    config = config_from_model_preset(
        "llama3.1-8b",
        sequence=args.sequence,
        layers=1,
        batch=args.batch,
    )
    rope = _make_llama3_rope(config)
    dq = torch.randn(
        (config.batch, config.sequence, config.q_heads, config.head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    dk = torch.randn(
        (config.batch, config.sequence, config.kv_heads, config.head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    dv = torch.randn_like(dk)

    def reference() -> torch.Tensor:
        dq_inverse = _inverse_rope_pair_native(dq, *rope)
        dk_inverse = _inverse_rope_pair_native(dk, *rope)
        dq_inverse.mul_(args.q_gradient_scale)
        dk_inverse.mul_(args.k_gradient_scale)
        dv_decoded = (
            dv.float().mul_(args.v_gradient_scale).bfloat16()
        )
        return torch.cat(
            (
                dq_inverse.reshape(config.batch * config.sequence, -1),
                dk_inverse.reshape(config.batch * config.sequence, -1),
                dv_decoded.reshape(config.batch * config.sequence, -1),
            ),
            dim=1,
        ).contiguous()

    def fused() -> torch.Tensor:
        return b300_stitch_gqa_d128_inverse_rope_gradient(
            dq,
            dk,
            dv,
            *rope,
            q_gradient_scale=args.q_gradient_scale,
            k_gradient_scale=args.k_gradient_scale,
            v_gradient_scale=args.v_gradient_scale,
        )

    expected = reference()
    actual = fused()
    torch.cuda.synchronize()
    difference = (actual.float() - expected.float()).abs()
    exact = torch.equal(actual, expected)
    extension = Path(tk_interface._C_b300_lowp_bwd.__file__)
    reference_timing = _time_cuda(
        reference,
        warmups=args.warmups,
        samples=args.samples,
    )
    fused_timing = _time_cuda(
        fused,
        warmups=args.warmups,
        samples=args.samples,
    )
    result = {
        "schema": "d128_qkv_weight_gradient_stitch_v2",
        "configuration": {
            "batch": config.batch,
            "sequence": config.sequence,
            "q_heads": config.q_heads,
            "kv_heads": config.kv_heads,
            "head_dim": config.head_dim,
            "seed": args.seed,
            "warmups": args.warmups,
            "samples": args.samples,
            "q_gradient_scale": args.q_gradient_scale,
            "k_gradient_scale": args.k_gradient_scale,
            "v_gradient_scale": args.v_gradient_scale,
        },
        "device": {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
        },
        "extension": _file_identity(extension),
        "numerics": {
            "bitwise_equal": exact,
            "maximum_absolute_error": float(difference.max()),
            "mean_absolute_error": float(difference.mean()),
        },
        "timing": {
            "functional_reference": reference_timing,
            "fused": fused_timing,
            "speedup": (
                reference_timing["median_us"] / fused_timing["median_us"]
            ),
        },
    }
    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    if not exact:
        raise RuntimeError("fused D128 publication is not bitwise exact")


if __name__ == "__main__":
    main()
