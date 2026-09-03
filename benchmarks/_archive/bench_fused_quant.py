"""
bench_fused_quant.py — Benchmark Fused RMSNorm+Act+NVFP4 vs TE Baseline

Compares:
  1. TE Baseline: separate torch.nn.RMSNorm → tex.quantize()
  2. Our Fused Kernel: single kernel (RMSNorm + SiLU + NVFP4 quant)

Both produce TE-compatible output (packed FP4 + FP8 scales + global scale).
The fused kernel is compiled on-the-fly using torch.utils.cpp_extension.

Usage:
    python3 benchmarks/bench_fused_quant.py
    python3 benchmarks/bench_fused_quant.py --m-sizes=4096,8192,16384 --k=8192
"""

import sys
import os
import torch
import torch.nn as nn
import argparse

# TE imports
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch.constants import TE_DType
from transformer_engine.pytorch import NVFP4Quantizer


def _time_fn(fn, steps, warmup):
    """Time a function using CUDA events. Returns avg ms per call."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(steps):
        fn()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / steps


# =====================================================================
# TE Baseline: separate RMSNorm + quantize
# =====================================================================

def bench_te_rmsnorm_plus_quant(m, k, steps, warmup):
    """
    TE baseline: torch RMSNorm → tex.quantize() (rowwise only, forward pass).
    Returns (rmsnorm_ms, quant_ms, total_ms).
    """
    device = "cuda"
    x_bf16 = torch.randn(m, k, device=device, dtype=torch.bfloat16)

    # RMSNorm
    rms_norm = nn.RMSNorm(k, eps=1e-5, device=device, dtype=torch.bfloat16)

    # Quantizer
    te_dtype = tex.DType.kFloat4E2M1
    quantizer = NVFP4Quantizer(
        fp4_dtype=te_dtype, rowwise=True, columnwise=False,
        with_amax_reduction=False, amax_reduction_group=None,
        with_rht=False, with_post_rht_amax=False,
    )
    x_nvfp4 = quantizer.make_empty(
        (m, k), dtype=torch.bfloat16, device=device, requires_grad=False
    )

    # Time RMSNorm only
    def run_rmsnorm():
        rms_norm(x_bf16)
    rmsnorm_ms = _time_fn(run_rmsnorm, steps, warmup)

    # Time quant only
    x_normed = rms_norm(x_bf16)
    def run_quant():
        quantizer.update_quantized(x_normed, x_nvfp4)
    quant_ms = _time_fn(run_quant, steps, warmup)

    # Time total (RMSNorm + quant)
    def run_total():
        out = rms_norm(x_bf16)
        quantizer.update_quantized(out, x_nvfp4)
    total_ms = _time_fn(run_total, steps, warmup)

    return rmsnorm_ms, quant_ms, total_ms


def bench_te_rmsnorm_silu_plus_quant(m, k, steps, warmup):
    """
    TE baseline with SiLU: torch RMSNorm → SiLU → tex.quantize().
    This is the "full fusion target" — what we want to do in one kernel.
    Returns (total_ms).
    """
    device = "cuda"
    x_bf16 = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    w_bf16 = torch.ones(k, device=device, dtype=torch.bfloat16)

    rms_norm = nn.RMSNorm(k, eps=1e-5, device=device, dtype=torch.bfloat16)
    silu = nn.SiLU()

    te_dtype = tex.DType.kFloat4E2M1
    quantizer = NVFP4Quantizer(
        fp4_dtype=te_dtype, rowwise=True, columnwise=False,
        with_amax_reduction=False, amax_reduction_group=None,
        with_rht=False, with_post_rht_amax=False,
    )
    x_nvfp4 = quantizer.make_empty(
        (m, k), dtype=torch.bfloat16, device=device, requires_grad=False
    )

    def run():
        out = rms_norm(x_bf16) 
        out = silu(out)
        out = out * w_bf16  # gated linear unit / gain
        quantizer.update_quantized(out, x_nvfp4)

    total_ms = _time_fn(run, steps, warmup)
    return total_ms


# =====================================================================
# Our Fused Kernel (compiled on the fly)
# =====================================================================

def try_load_fused_kernel():
    """
    Try to load the fused kernel via torch.utils.cpp_extension.
    Returns the module, or None if compilation fails.
    """
    try:
        from torch.utils.cpp_extension import load
        
        csrc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                '../fused_ops/csrc')
        
        # Build a minimal torch extension that wraps our kernel
        module = load(
            name='fused_te_quant_ext',
            sources=[
                os.path.join(csrc_dir, 'fused_te_quant.cu'),
            ],
            extra_include_paths=[csrc_dir],
            extra_cuda_cflags=[
                '-std=c++20', '-O3', '-lineinfo',
                '--expt-relaxed-constexpr',
                '-gencode=arch=compute_100a,code=sm_100a',
                '-rdc=true',  # Required for cooperative launch
            ],
            extra_ldflags=['-lcudadevrt'],
            verbose=True,
            is_python_module=False,
        )
        return module
    except Exception as e:
        print(f"  [WARN] Could not compile fused kernel: {e}")
        return None


def bench_fused_kernel(m, k, steps, warmup, norm_mode=0, act_mode=0):
    """
    Our fused kernel: single kernel doing RMSNorm + act + NVFP4 quant.
    This uses a C++ standalone binary approach since torch extension doesn't 
    easily support cooperative launch.
    
    For now, we'll measure using the existing fused_ops bindings if available,
    or skip.
    """
    # Try importing the existing _fused_ops module
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../fused_ops/python'))
        import fused_ops
        has_fused = True
    except ImportError:
        has_fused = False
    
    if not has_fused:
        return None
    
    device = "cuda"
    x_bf16 = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    w_bf16 = torch.ones(k, device=device, dtype=torch.bfloat16)
    
    # Allocate outputs in TE-compatible format
    out_fp4 = torch.empty(m, k // 2, device=device, dtype=torch.uint8)
    out_scales = torch.empty(m, k // 16, device=device, dtype=torch.uint8)  # fp8
    global_scale = torch.zeros(1, device=device, dtype=torch.float32)
    inv_rms = torch.empty(m, device=device, dtype=torch.float32)
    
    def run():
        fused_ops.fused_te_quant(
            out_fp4, out_scales, global_scale.squeeze(), inv_rms,
            x_bf16, w_bf16, 1e-5, norm_mode, act_mode
        )
    
    total_ms = _time_fn(run, steps, warmup)
    return total_ms


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Fused Quant Benchmark")
    parser.add_argument("--m-sizes", type=str, default="1024,2048,4096,8192,16384",
                        help="Comma-separated M dimensions")
    parser.add_argument("--k", type=int, default=8192,
                        help="K dimension (hidden dim)")
    parser.add_argument("--steps", type=int, default=500,
                        help="Timed iterations")
    parser.add_argument("--warmup", type=int, default=100,
                        help="Warmup iterations")
    args = parser.parse_args()

    m_sizes = [int(s) for s in args.m_sizes.split(",")]
    k = args.k

    print("=" * 130)
    print("  Fused RMSNorm + SiLU + NVFP4 Benchmark")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  K: {k}, Steps: {args.steps}, Warmup: {args.warmup}")
    print("=" * 130)

    # Header
    print(f"\n{'M':>8} | {'--- TE Sep (RMS+Quant) ---':^30} | {'--- TE Sep (RMS+SiLU+Quant) ---':^32} |"
          f" {'--- Budget to Match ---':^24} |")
    print(f"{'':>8} | {'RMS ms':>8} {'Quant ms':>9} {'Total ms':>9} {'GB/s':>7} |"
          f" {'Total ms':>10} {'GB/s':>7} {'Overhead':>12} |"
          f" {'Fused Target':>12} {'Saving':>10} |")
    print("-" * 130)

    for m in m_sizes:
        elements = m * k
        bytes_bf16 = elements * 2
        bytes_gb = bytes_bf16 / 1e9

        # TE baseline: RMSNorm + quant (no activation)
        rms_ms, quant_ms, total_rms_quant = bench_te_rmsnorm_plus_quant(
            m, k, args.steps, args.warmup
        )
        bw_rms_quant = bytes_gb / (total_rms_quant * 1e-3)

        # TE baseline: RMSNorm + SiLU + quant (full fusion target)
        total_rms_silu_quant = bench_te_rmsnorm_silu_plus_quant(
            m, k, args.steps, args.warmup
        )
        bw_rms_silu_quant = bytes_gb / (total_rms_silu_quant * 1e-3)
        overhead_ms = total_rms_silu_quant - quant_ms

        # The "budget to match" = total_rms_silu_quant 
        # If fused does it in <= quant_ms, we've eliminated RMS+SiLU overhead
        fused_target = quant_ms  # Best case: norm+act is "free"
        saving_ms = total_rms_silu_quant - fused_target

        print(f"{m:>8} | {rms_ms:>8.3f} {quant_ms:>9.3f} {total_rms_quant:>9.3f} {bw_rms_quant:>7.0f} |"
              f" {total_rms_silu_quant:>10.3f} {bw_rms_silu_quant:>7.0f} {overhead_ms:>9.3f} ms |"
              f" {fused_target:>9.3f} ms {saving_ms:>7.3f} ms |")

    print()
    print("Legend:")
    print("  TE Sep (RMS+Quant)     = torch.nn.RMSNorm → tex.quantize (rowwise only)")
    print("  TE Sep (RMS+SiLU+Quant)= torch.nn.RMSNorm → SiLU → tex.quantize (full pipeline)")
    print("  Budget to Match        = max latency for fused kernel to break even")
    print("  Fused Target           = quant-only time (RMS+SiLU fused 'for free')")
    print("  Saving                 = time saved if fusion achieves target")
    print("  GB/s                   = BF16 input read bandwidth")


if __name__ == "__main__":
    main()
