#!/usr/bin/env python3
"""
TRUE E2E Benchmark: FP4 Linear Modules — Forward + loss.backward()
===================================================================

Compares actual nn.Module implementations:
  - Baseline (unfused): nn.RMSNorm + SiLU + TELinearFP4
  - Fused (TE-2pass):   NormTELinearFP4 (absorbed RMSNorm + SiLU + quantization)

Each variant is timed as a single forward + backward pass with a real scalar loss.
"""

import os, sys, ctypes
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Pre-load shared libraries to avoid symbol errors ──
for dep in ['/usr/local/cuda/lib64/libnvrtc.so', '/usr/local/cuda/lib64/libcudart.so']:
    if os.path.exists(dep):
        ctypes.CDLL(dep, mode=ctypes.RTLD_GLOBAL)
te_lib = '/workspace/low-bits-training/TransformerEngine/build/cmake/libtransformer_engine.so'
if os.path.exists(te_lib):
    ctypes.CDLL(te_lib, mode=ctypes.RTLD_GLOBAL)

# Add project paths
sys.path.insert(0, '/workspace/low-bits-training')

from low_bits_training.quantization.fused_te_linear import (
    TELinearFP4,
    NormTELinearFP4,
)


class RMSNorm(nn.Module):
    """Simple RMSNorm for the unfused baseline."""
    def __init__(self, dim, eps=1e-5, dtype=torch.bfloat16, device=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))

    def forward(self, x):
        var = x.float().pow(2).mean(dim=-1, keepdim=True)
        inv_rms = torch.rsqrt(var + self.eps)
        return (x * inv_rms.to(x.dtype)) * self.weight


class UnfusedNormLinearFP4(nn.Module):
    """
    Baseline: separate RMSNorm → SiLU → TELinearFP4.
    This is what you'd have without any kernel fusion.
    """
    def __init__(self, in_features, out_features, norm_eps=1e-5,
                 bias=False, device=None, dtype=torch.bfloat16):
        super().__init__()
        self.norm = RMSNorm(in_features, eps=norm_eps, device=device, dtype=dtype)
        self.linear = TELinearFP4(in_features, out_features, bias=bias, device=device, dtype=dtype)

    def forward(self, x):
        x = self.norm(x)
        x = F.silu(x)
        return self.linear(x)

    def invalidate_weight_cache(self):
        self.linear.invalidate_weight_cache()


class FusedNormLinearFP4(nn.Module):
    """
    Fused: NormTELinearFP4 (RMSNorm + SiLU + FP4 quant in 2 CUDA passes).
    """
    def __init__(self, in_features, out_features, norm_eps=1e-5,
                 bias=False, device=None, dtype=torch.bfloat16):
        super().__init__()
        self.linear = NormTELinearFP4(
            in_features, out_features, bias=bias,
            norm_eps=norm_eps, device=device, dtype=dtype
        )

    def forward(self, x):
        return self.linear(x)

    def invalidate_weight_cache(self):
        self.linear.invalidate_weight_cache()


def bench_module(module, x, warmup=20, steps=100):
    """Time forward + backward of a module, returning ms per step."""
    # Warmup
    for _ in range(warmup):
        module.invalidate_weight_cache()
        y = module(x)
        loss = y.sum()
        loss.backward()

    torch.cuda.synchronize()
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
    end_events   = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]

    for i in range(steps):
        module.invalidate_weight_cache()
        start_events[i].record()
        y = module(x)
        loss = y.sum()
        loss.backward()
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    # Use median to reduce variance
    times.sort()
    median = times[len(times) // 2]
    return median


def bench_forward_only(module, x, warmup=20, steps=100):
    """Time forward only (no backward), returning ms per step."""
    with torch.no_grad():
        for _ in range(warmup):
            module.invalidate_weight_cache()
            module(x)

        torch.cuda.synchronize()
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
        end_events   = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]

        for i in range(steps):
            module.invalidate_weight_cache()
            start_events[i].record()
            module(x)
            end_events[i].record()

        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        times.sort()
        return times[len(times) // 2]


def main():
    parser = argparse.ArgumentParser(description='True E2E Module Benchmark')
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--steps', type=int, default=100)
    args = parser.parse_args()

    device = 'cuda'
    dtype = torch.bfloat16

    # Size configs: (M, K, N) — M=batch*seq, K=in_features, N=out_features
    sizes = [
        (4096,  2048,  2048),
        (4096,  8192,  2048),
        (4096,  2048,  8192),
        (8192,  8192,  8192),
        (16384, 8192,  2048),
    ]

    print("=" * 90)
    print("  TRUE E2E BENCHMARK: FP4 Linear Module (forward + loss.backward())")
    print("  Unfused: RMSNorm → SiLU → TELinearFP4 (standard TE pipeline)")
    print("  Fused:   NormTELinearFP4 (uses fused pass1 + TE pass2 + fused backward)")
    print("=" * 90)

    # Header
    print(f"\n{'M':>7} {'K':>7} {'N':>7} | {'Unfused FWD':>12} {'Fused FWD':>12} {'FWD speedup':>12}"
          f" | {'Unfused F+B':>12} {'Fused F+B':>12} {'F+B speedup':>12}")
    print("-" * 104)

    for M, K, N in sizes:
        # 2D input (acts like batch*seq flattened)
        x = torch.randn(M, K, device=device, dtype=dtype, requires_grad=True)

        # Create modules
        unfused = UnfusedNormLinearFP4(K, N, device=device, dtype=dtype).to(device)
        fused   = FusedNormLinearFP4(K, N, device=device, dtype=dtype).to(device)

        # Copy weights so both use the same parameters
        with torch.no_grad():
            fused.linear.weight.copy_(unfused.linear.weight)
            fused.linear.norm_weight.copy_(unfused.norm.weight)

        # ── Forward only ──
        t_fwd_unfused = bench_forward_only(unfused, x, warmup=args.warmup, steps=args.steps)
        t_fwd_fused   = bench_forward_only(fused, x, warmup=args.warmup, steps=args.steps)
        fwd_speedup = t_fwd_unfused / t_fwd_fused

        # ── Forward + Backward ──
        t_fb_unfused = bench_module(unfused, x, warmup=args.warmup, steps=args.steps)
        t_fb_fused   = bench_module(fused, x, warmup=args.warmup, steps=args.steps)
        fb_speedup = t_fb_unfused / t_fb_fused

        print(f"{M:>7} {K:>7} {N:>7} | {t_fwd_unfused:>10.3f}ms {t_fwd_fused:>10.3f}ms {fwd_speedup:>10.2f}x"
              f" | {t_fb_unfused:>10.3f}ms {t_fb_fused:>10.3f}ms {fb_speedup:>10.2f}x")

        # Cleanup
        del unfused, fused, x
        torch.cuda.empty_cache()

    print()
    print("Speedup > 1.0 means Fused is faster.")


if __name__ == '__main__':
    main()
