#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch


SEED = 2024
DEFAULT_SEQS = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
ROOT = Path(__file__).resolve().parents[1]
TK_CAUSAL_DIR = ROOT / "ThunderKittens" / "kernels" / "attention" / "bf16_b300_mha_causal"
TK_NONCAUSAL_DIR = ROOT / "ThunderKittens" / "kernels" / "attention" / "bf16_b300_mha_noncausal"
FLASH_ATTENTION_DIR = ROOT / "flash-attention"
JSON_MARKER = "JSON_RESULT="


def splitmix_bf16(count: int, seed: int, min_val: float, max_val: float) -> torch.Tensor:
    idx = np.arange(count, dtype=np.uint64)
    x = np.uint64(seed) + idx
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    x = x ^ (x >> np.uint64(31))
    u = (x >> np.uint64(40)).astype(np.float32) * np.float32(1.0 / 16777216.0)
    vals = u * np.float32(max_val - min_val) + np.float32(min_val)
    return torch.from_numpy(vals).to(torch.bfloat16).cuda()


def flops_fwd(batch: int, seqlen: int, heads: int, d_qk: int, d_v: int, causal: bool) -> float:
    flops = 2.0 * batch * heads * seqlen * seqlen * (d_qk + d_v)
    return flops * 0.5 if causal else flops


def create_qkv(batch: int, seqlen: int, heads: int, d_qk: int, d_v: int, seed: int):
    count_qk = batch * seqlen * heads * d_qk
    count_v = batch * seqlen * heads * d_v
    q = splitmix_bf16(count_qk, seed, -1.0, 1.0).view(batch, seqlen, heads, d_qk)
    k = splitmix_bf16(count_qk, seed + 1, -1.0, 1.0).view(batch, seqlen, heads, d_qk)
    v = splitmix_bf16(count_v, seed + 2, -1.0, 1.0).view(batch, seqlen, heads, d_v)
    return q, k, v


def create_tk_inputs(batch: int, seqlen: int, heads: int, d_qk: int, d_v: int, seed: int):
    q, k, v = create_qkv(batch, seqlen, heads, d_qk, d_v, seed)
    o = torch.zeros(batch, seqlen, heads, d_v, dtype=torch.bfloat16, device="cuda")
    lse = torch.zeros(batch, heads, 1, seqlen, dtype=torch.float32, device="cuda")
    return q, k, v, o, lse


def tk_group_count(batch: int, seqlen: int, heads: int, d_qk: int, d_v: int) -> int:
    l2_cache_size = torch.cuda.get_device_properties(0).L2_cache_size
    arg_size = (2 * batch * seqlen * heads * d_qk + batch * seqlen * heads * d_v) * 2
    ideal_arg_size = l2_cache_size * 3
    return 1 if arg_size > ideal_arg_size else (ideal_arg_size // arg_size) + 1


def load_tk_module(causal: bool):
    tk_dir = TK_CAUSAL_DIR if causal else TK_NONCAUSAL_DIR
    sys.path.insert(0, str(tk_dir))
    import _C  # noqa: PLC0415

    return _C


def load_fa4():
    sys.path.insert(0, str(FLASH_ATTENTION_DIR))
    from flash_attn.cute.interface import flash_attn_func  # noqa: PLC0415

    return flash_attn_func


@torch.inference_mode()
def bench_tk_shape(batch: int, seqlen: int, heads: int, d_qk: int, d_v: int, causal: bool, warmup: int, iters: int):
    module = load_tk_module(causal)
    kernel = module.forward_persistent if causal and seqlen <= 4096 else module.forward
    group_count = tk_group_count(batch, seqlen, heads, d_qk, d_v)
    groups = [
        create_tk_inputs(batch, seqlen, heads, d_qk, d_v, SEED + idx * 100)
        for idx in range(group_count)
    ]

    time.sleep(0.5)
    for idx in range(warmup):
        kernel(*groups[idx % group_count])

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for idx in range(iters):
        kernel(*groups[idx % group_count])
    end.record()
    end.synchronize()

    total_ms = start.elapsed_time(end)
    avg_us = total_ms * 1000.0 / iters
    tflops = flops_fwd(batch, seqlen, heads, d_qk, d_v, causal) / (avg_us * 1e-6) / 1e12
    return avg_us, tflops


@torch.inference_mode()
def bench_fa4_shape(batch: int, seqlen: int, heads: int, d_qk: int, d_v: int, causal: bool, warmup: int, iters: int):
    flash_attn_func = load_fa4()
    softmax_scale = 1.0 / math.sqrt(d_qk)
    group_count = tk_group_count(batch, seqlen, heads, d_qk, d_v)
    groups = [
        create_qkv(batch, seqlen, heads, d_qk, d_v, SEED + idx * 100)
        for idx in range(group_count)
    ]

    time.sleep(0.5)
    for idx in range(warmup):
        out = flash_attn_func(*groups[idx % group_count], softmax_scale=softmax_scale, causal=causal)
        if isinstance(out, tuple):
            out = out[0]

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for idx in range(iters):
        out = flash_attn_func(*groups[idx % group_count], softmax_scale=softmax_scale, causal=causal)
        if isinstance(out, tuple):
            out = out[0]
    end.record()
    end.synchronize()

    total_ms = start.elapsed_time(end)
    avg_us = total_ms * 1000.0 / iters
    tflops = flops_fwd(batch, seqlen, heads, d_qk, d_v, causal) / (avg_us * 1e-6) / 1e12
    return avg_us, tflops


def run_worker(args) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run this script with the fp4_matmul .venv.")

    causal = args.mode == "causal"
    bench_fn = bench_tk_shape if args.backend == "tk" else bench_fa4_shape
    results = []
    for seqlen in args.seqs:
        avg_us, tflops = bench_fn(
            batch=args.batch,
            seqlen=seqlen,
            heads=args.heads,
            d_qk=args.d_qk,
            d_v=args.d_v,
            causal=causal,
            warmup=args.warmup,
            iters=args.iters,
        )
        results.append(
            {
                "backend": args.backend,
                "mode": args.mode,
                "batch": args.batch,
                "seqlen": seqlen,
                "heads": args.heads,
                "d_qk": args.d_qk,
                "d_v": args.d_v,
                "warmup": args.warmup,
                "iters": args.iters,
                "avg_us": avg_us,
                "tflops": tflops,
            }
        )

    return {
        "backend": args.backend,
        "mode": args.mode,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "results": results,
    }


def format_worker_report(payload: dict):
    print(f"{payload['backend'].upper()} {payload['mode']} on {payload['gpu']}")
    for row in payload["results"]:
        print(
            f"  S={row['seqlen']:>5}  avg={row['avg_us']:>10.2f} us"
            f"  perf={row['tflops']:>8.2f} TFLOPs"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark ThunderKittens BF16 B300 attention kernels against flash-attention CuTe FA4."
    )
    parser.add_argument("--backend", choices=["tk", "fa4"], required=True)
    parser.add_argument("--mode", choices=["causal", "noncausal"], default="causal")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--d-qk", type=int, default=192, dest="d_qk")
    parser.add_argument("--d-v", type=int, default=128, dest="d_v")
    parser.add_argument("--seqs", type=int, nargs="+", default=DEFAULT_SEQS)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--json-path")
    return parser.parse_args()


def main():
    args = parse_args()
    payload = run_worker(args)
    if args.json_path:
        Path(args.json_path).write_text(json.dumps(payload, sort_keys=True))
    elif args.json:
        print(f"{JSON_MARKER}{json.dumps(payload, sort_keys=True)}")
    else:
        format_worker_report(payload)


if __name__ == "__main__":
    main()
