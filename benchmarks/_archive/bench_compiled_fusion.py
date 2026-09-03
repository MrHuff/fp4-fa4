#!/usr/bin/env python3
"""
Benchmark: What is the actual overhead of separate norm+silu vs a proper fused version?

This measures:
1. F.rms_norm + F.silu (separate, optimized PyTorch kernels)
2. torch.compile'd fusion of norm+silu
3. V7 kernel (norm+silu mode, output BF16)
4. Full pipeline: norm+silu+TE_quant
"""

import os
os.environ["CUDA_MODULE_LOADING"] = "EAGER"

import torch
import torch.nn.functional as F
import time
import sys

sys.path.insert(0, "/workspace/low-bits-training")
import transformer_engine.pytorch as te
from transformer_engine.pytorch import NVFP4Quantizer


def bench_fn(fn, warmup=50, iters=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000  # ms


def fused_norm_silu(x, w, eps):
    """Torch-compilable fused rmsnorm+silu"""
    rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return F.silu(x * rms * w)

# Pre-compile
fused_norm_silu_compiled = torch.compile(fused_norm_silu, mode="max-autotune")


def main():
    torch.manual_seed(42)
    device = torch.device("cuda:0")
    eps = 1e-5

    shapes = [
        (2048, 4096),
        (4096, 4096),
        (4096, 8192),
        (8192, 8192),
        (4096, 16384),
        (8192, 16384),
    ]

    print("=" * 100)
    print("Norm+SiLU Fusion Benchmark")
    print("=" * 100)
    
    print(f"{'M':>8} {'K':>8} | {'sep(n+s)':>10} {'compiled':>10} | {'TE_quant':>10} | {'sep+Q':>10} {'comp+Q':>10} | {'speedup':>8}")
    print("-" * 100)

    for M, K in shapes:
        x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        w = torch.ones(K, dtype=torch.bfloat16, device=device)
        quantizer = NVFP4Quantizer()

        # 1. Separate F.rms_norm + F.silu
        def sep_norm_silu():
            n = F.rms_norm(x, (K,), w, eps)
            return F.silu(n)
        t_sep = bench_fn(sep_norm_silu)

        # 2. torch.compile'd fusion - warmup first
        print(f"  Compiling for shape ({M}, {K})...", end="", flush=True)
        for _ in range(5):
            fused_norm_silu_compiled(x, w, eps)
        torch.cuda.synchronize()
        print(" done")
        t_compiled = bench_fn(lambda: fused_norm_silu_compiled(x, w, eps))

        # 3. TE quant only
        activated = F.silu(F.rms_norm(x, (K,), w, eps))
        t_quant = bench_fn(lambda: quantizer.quantize(activated))

        # 4. Full pipeline: sep norm+silu + TE quant
        def sep_full():
            n = F.rms_norm(x, (K,), w, eps)
            a = F.silu(n)
            return quantizer.quantize(a)
        t_sep_full = bench_fn(sep_full)

        # 5. Full pipeline: compiled norm+silu + TE quant
        def comp_full():
            a = fused_norm_silu_compiled(x, w, eps)
            return quantizer.quantize(a)
        t_comp_full = bench_fn(comp_full)

        speedup = t_sep_full / t_comp_full

        print(f"{M:8d} {K:8d} | {t_sep:10.4f}ms {t_compiled:10.4f}ms | {t_quant:10.4f}ms | "
              f"{t_sep_full:10.4f}ms {t_comp_full:10.4f}ms | {speedup:8.2f}x")

    print()
    print("speedup > 1.0 means compiled fusion is faster than separate kernels")
    print("The savings come from eliminating one BF16 write+read between norm and silu")


if __name__ == "__main__":
    main()
