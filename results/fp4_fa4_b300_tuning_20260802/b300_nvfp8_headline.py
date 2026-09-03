#!/usr/bin/env python3
"""Measure optimized TK NVFP4-QK/FP8-PV at HAO's GB300 headline shapes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any

import torch
import triton.testing

import hao_direct_benchmark as fp8_bench
import hao_direct_fp4pv_benchmark as fp4_bench


HAO_PUBLISHED_GB300 = {
    "b1_s32768_h24_d128": {
        "nvfp4_bf16_tflops": 2235,
        "nvfp4_fp8_tflops": 2677,
        "nvfp4_nvfp4_tflops": 1725,
        "nvfp4_mxfp8_tflops": 1809,
        "bf16_tflops": 1533,
        "nvfp4_fp8_cosine": 0.9899,
        "nvfp4_fp8_max_diff": 0.0142,
        "nvfp4_fp8_mean_diff": 0.0010,
    },
    "b1_s32768_h24_d64": {
        "nvfp4_bf16_tflops": 1209,
        "nvfp4_fp8_tflops": 1203,
        "bf16_tflops": 1221,
        "nvfp4_fp8_cosine": 0.9892,
        "nvfp4_fp8_max_diff": 0.0223,
        "nvfp4_fp8_mean_diff": 0.0011,
        "block_scaled_pv_supported": False,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d128-extension", type=Path, required=True)
    parser.add_argument("--d128-module", required=True)
    parser.add_argument("--d64-extension", type=Path, required=True)
    parser.add_argument("--d64-module", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--warmup-ms", type=int, default=10)
    parser.add_argument("--rep-ms", type=int, default=25)
    parser.add_argument("--windows", type=int, default=5)
    parser.add_argument("--cooldown-s", type=float, default=0.8)
    return parser.parse_args()


def measure_shape(
    extension: Any,
    *,
    dim: int,
    seed: int,
    warmup_ms: int,
    rep_ms: int,
    windows: int,
    cooldown_s: float,
) -> dict[str, Any]:
    batch, seqlen, heads = 1, 32768, 24
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    q_ref = torch.randn(
        batch, seqlen, heads, dim, device="cuda", dtype=torch.float32
    )
    k_ref = torch.randn(
        batch, seqlen, heads, dim, device="cuda", dtype=torch.float32
    )
    v_ref = torch.randn(
        batch, seqlen, heads, dim, device="cuda", dtype=torch.float32
    )
    q_nv, q_nv_scale = fp4_bench.quantize_nvfp4_qk(q_ref, 1.0)
    k_nv, k_nv_scale = fp4_bench.quantize_nvfp4_qk(k_ref, 1.0)
    prepared = fp8_bench.prepare_nvfp4_tk_inputs(
        q_nv,
        k_nv,
        v_ref.to(torch.float8_e4m3fn),
        q_nv_scale,
        k_nv_scale,
    )

    output = torch.empty(
        batch, seqlen, heads, dim, device="cuda", dtype=torch.bfloat16
    )
    lse = torch.empty(
        batch, heads, 1, seqlen, device="cuda", dtype=torch.float32
    )
    reference = torch.nn.functional.scaled_dot_product_attention(
        q_ref.to(torch.bfloat16).transpose(1, 2),
        k_ref.to(torch.bfloat16).transpose(1, 2),
        v_ref.to(torch.bfloat16).transpose(1, 2),
        is_causal=False,
    ).transpose(1, 2)

    def run(store_lse: bool = False) -> None:
        extension.forward_hao_direct_fp8pv(
            prepared.q_fp4_bhsd,
            prepared.q_scale_prepared,
            prepared.q_global_scale,
            prepared.k_fp4_bhsd,
            prepared.k_scale_prepared,
            prepared.k_global_scale,
            prepared.v_fp8_bhds,
            output,
            lse,
            0,
            True,
            store_lse,
        )

    topology = dict(extension.read_hao_direct_topology())
    route = str(topology["route"])
    previous_route = os.environ.get("TK_FA4_FP4PV_FWD_CONFIG")
    try:
        os.environ["TK_FA4_FP4PV_FWD_CONFIG"] = route
        run(store_lse=True)
        torch.cuda.synchronize()
        samples_ms = []
        for _ in range(windows):
            time.sleep(cooldown_s)
            samples_ms.append(
                float(
                    triton.testing.do_bench(
                        run,
                        warmup=warmup_ms,
                        rep=rep_ms,
                        return_mode="median",
                    )
                )
            )
        timing_ms = statistics.median(samples_ms)
        topology = dict(extension.read_hao_direct_topology())
    finally:
        if previous_route is None:
            os.environ.pop("TK_FA4_FP4PV_FWD_CONFIG", None)
        else:
            os.environ["TK_FA4_FP4PV_FWD_CONFIG"] = previous_route

    operation_count = batch * heads * 2 * seqlen * seqlen * (dim + dim)
    return {
        "shape": {
            "batch": batch,
            "seqlen": seqlen,
            "heads": heads,
            "dim": dim,
        },
        "provider": "tk_nvfp4_fp8_optimized",
        "timing_ms": timing_ms,
        "timing_samples_ms": samples_ms,
        "tflops": operation_count / (timing_ms * 1.0e9),
        "correctness_global": fp4_bench.localized_comparison(
            output, reference
        )["global"],
        "topology": topology,
    }


def main() -> None:
    args = parse_args()
    torch.cuda.set_device(0)
    d128 = fp8_bench.load(args.d128_extension, args.d128_module)
    d64 = fp8_bench.load(args.d64_extension, args.d64_module)
    rows = []
    for extension, dim in ((d128, 128), (d64, 64)):
        rows.append(
            measure_shape(
                extension,
                dim=dim,
                seed=args.seed,
                warmup_ms=args.warmup_ms,
                rep_ms=args.rep_ms,
                windows=args.windows,
                cooldown_s=args.cooldown_s,
            )
        )
        torch.cuda.empty_cache()

    properties = torch.cuda.get_device_properties(0)
    try:
        max_clock_mhz = int(
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=clocks.max.sm",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip()
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        max_clock_mhz = None
    try:
        tk_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        tk_commit = None

    result = {
        "hardware": {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "multiprocessor_count": properties.multi_processor_count,
            "max_sm_clock_mhz": max_clock_mhz,
        },
        "protocol": {
            "seed": args.seed,
            "timer": "triton.testing.do_bench median",
            "warmup_ms": args.warmup_ms,
            "rep_ms": args.rep_ms,
            "independent_windows": args.windows,
            "cooldown_s": args.cooldown_s,
            "tk_commit": tk_commit,
            "hao_reference": (
                "HAO flash-attention-fp4 fp4 branch README GB300 tables"
            ),
        },
        "hao_published_gb300": HAO_PUBLISHED_GB300,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
