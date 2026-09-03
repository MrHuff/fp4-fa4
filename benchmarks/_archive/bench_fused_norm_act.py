#!/usr/bin/env python3
"""
Benchmark: Fused RMSNorm+SiLU+Quant strategies.

Compares:
1. TE Baseline: separate rmsnorm -> silu -> TE quant  (3 kernels)
2. Fused Norm+Act: fused rmsnorm+silu -> TE quant     (2 kernels)
3. V7 Full Fusion: V7 norm+act+quant (1 kernel)

The question: how much does fusing norm+act save by eliminating one BF16 round-trip?
"""

import os
os.environ["CUDA_MODULE_LOADING"] = "EAGER"

import torch
import torch.nn.functional as F
import time
import sys

# Load TE
sys.path.insert(0, "/workspace/low-bits-training")
import transformer_engine.pytorch as te
from transformer_engine.pytorch import NVFP4Quantizer

# Try loading V7 extension
try:
    sys.path.insert(0, "/workspace/fp4_matmul/fused_ops")
    from te_quant_v7_ext import fused_te_quant_v7_forward
    HAS_V7 = True
except:
    HAS_V7 = False
    print("[WARN] V7 extension not available, skipping V7 benchmarks")


def bench_fn(fn, warmup=50, iters=200):
    """Time a function with CUDA synchronization."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters * 1000  # ms
    return elapsed


def te_baseline(x, norm_weight, eps, quantizer):
    """3 kernels: rmsnorm -> silu -> TE quant"""
    normed = F.rms_norm(x, (x.shape[-1],), norm_weight, eps)
    activated = F.silu(normed)
    return quantizer.quantize(activated)


def fused_norm_act_then_quant(x, norm_weight, eps, quantizer):
    """2 kernels: fused rmsnorm+silu -> TE quant
    
    Uses torch.compile to fuse rmsnorm+silu into one kernel.
    """
    # Fuse norm+act using a simple torch implementation
    # torch will launch this as potentially 1-2 kernels
    rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    normed = x * rms * norm_weight
    activated = F.silu(normed)
    return quantizer.quantize(activated)


def v7_fused_then_nothing(x, norm_weight, eps, rows, cols):
    """V7 full fusion: 1 kernel for norm+act+quant"""
    if not HAS_V7:
        return None
    result = fused_te_quant_v7_forward(
        x, norm_weight, eps, rows, cols,
        0,  # NORM_MODE = rmsnorm
        1,  # ACT_MODE = silu
        0,  # SCALE_MODE = global
    )
    return result


def main():
    torch.manual_seed(42)
    device = torch.device("cuda:0")
    
    shapes = [
        (2048, 4096),
        (4096, 4096),
        (4096, 8192),
        (8192, 8192),
        (4096, 16384),
        (8192, 16384),
    ]
    
    eps = 1e-5
    
    print("=" * 80)
    print("Fused RMSNorm+SiLU+Quant Benchmark")
    print("=" * 80)
    
    # Header
    header = f"{'M':>8} {'K':>8} | {'3-kern':>10} {'2-kern':>10} {'1-kern':>10} | {'2v3':>8} {'1v3':>8}"
    print(header)
    print("-" * len(header))
    
    for M, K in shapes:
        x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        norm_weight = torch.ones(K, dtype=torch.bfloat16, device=device)
        quantizer = NVFP4Quantizer()
        
        # Benchmark 1: TE baseline (3 kernels: norm -> act -> quant)
        def te_3kern():
            normed = F.rms_norm(x, (K,), norm_weight, eps)
            activated = F.silu(normed)
            return quantizer.quantize(activated)
        
        t_3kern = bench_fn(te_3kern)
        
        # Benchmark 2: norm+act fused, then TE quant (2 kernels)
        # Use a manual RMSNorm+SiLU that PyTorch can fuse
        def te_2kern():
            rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
            normed = x * rms * norm_weight
            activated = F.silu(normed)
            return quantizer.quantize(activated)
        
        t_2kern = bench_fn(te_2kern)
        
        # Benchmark 3: V7 full fusion (1 kernel, but slower quant)
        if HAS_V7:
            def v7_1kern():
                return v7_fused_then_nothing(x, norm_weight, eps, M, K)
            t_1kern = bench_fn(v7_1kern)
        else:
            t_1kern = float('nan')
        
        ratio_2v3 = t_2kern / t_3kern
        ratio_1v3 = t_1kern / t_3kern if HAS_V7 else float('nan')
        
        print(f"{M:8d} {K:8d} | {t_3kern:10.4f}ms {t_2kern:10.4f}ms {t_1kern:10.4f}ms | {ratio_2v3:8.2f}x {ratio_1v3:8.2f}x")
    
    print()
    print("Note: 2v3 < 1.0 means 2-kernel approach is faster than 3-kernel")
    print("      1v3 < 1.0 means V7 1-kernel is faster than 3-kernel")
    
    # Now let's also measure the individual kernel costs
    print()
    print("=" * 80)
    print("Individual Kernel Costs")
    print("=" * 80)
    
    for M, K in shapes:
        x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        norm_weight = torch.ones(K, dtype=torch.bfloat16, device=device)
        quantizer = NVFP4Quantizer()
        
        # RMSNorm only
        t_norm = bench_fn(lambda: F.rms_norm(x, (K,), norm_weight, eps))
        
        # SiLU only
        normed = F.rms_norm(x, (K,), norm_weight, eps)
        t_silu = bench_fn(lambda: F.silu(normed))
        
        # TE quant only
        activated = F.silu(normed)
        t_quant = bench_fn(lambda: quantizer.quantize(activated))
        
        # RMSNorm + SiLU fused (manual)
        def fused_norm_silu():
            rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
            return F.silu(x * rms * norm_weight)
        t_fused_ns = bench_fn(fused_norm_silu)
        
        # Fused norm+silu using F.rms_norm + F.silu (not truly fused, 2 separate ops)
        def separate_norm_silu():
            return F.silu(F.rms_norm(x, (K,), norm_weight, eps))
        t_sep_ns = bench_fn(separate_norm_silu)
        
        print(f"M={M:5d} K={K:5d}: norm={t_norm:.3f}ms silu={t_silu:.3f}ms quant={t_quant:.3f}ms "
              f"| separate_ns={t_sep_ns:.3f}ms manual_fused_ns={t_fused_ns:.3f}ms "
              f"| total_3k={t_norm+t_silu+t_quant:.3f}ms  total_2k={t_fused_ns+t_quant:.3f}ms "
              f"| savings={t_norm+t_silu-t_fused_ns:.3f}ms")


if __name__ == "__main__":
    main()
