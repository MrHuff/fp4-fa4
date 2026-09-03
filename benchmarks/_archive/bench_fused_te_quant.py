"""
bench_fused_te_quant.py — Build & benchmark our fused kernel vs TE baseline

Compiles fused_te_quant.cu on-the-fly via torch.utils.cpp_extension,
then benchmarks:
  1. TE baseline: separate RMSNorm → SiLU → tex.quantize()  
  2. TE quant-only: tex.quantize() alone
  3. Our fused kernel: single kernel (RMSNorm + SiLU + NVFP4)
  4. FP4 GEMM: tex.generic_gemm (for proportion context)

Usage:
    python3 benchmarks/bench_fused_te_quant.py
"""

import os
import sys
import time
import torch
import torch.nn as nn
import argparse

# -------- Build fused kernel on-the-fly --------
print("Compiling fused_te_quant kernel...")
t0 = time.time()

from torch.utils.cpp_extension import load

CSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fused_ops', 'csrc')

fused_te = load(
    name='fused_te_quant_ext',
    sources=[
        os.path.join(CSRC, 'fused_te_quant_torch.cpp'),
        os.path.join(CSRC, 'fused_te_quant.cu'),
    ],
    extra_include_paths=[CSRC],
    extra_cuda_cflags=[
        '-std=c++20', '-O3', '-lineinfo',
        '--expt-relaxed-constexpr',
        '-gencode=arch=compute_100a,code=sm_100a',
    ],
    verbose=False,
)
print(f"Compiled in {time.time()-t0:.1f}s")

# -------- TE imports --------
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch.constants import TE_DType
from transformer_engine.pytorch import NVFP4Quantizer


def time_fn(fn, steps=300, warmup=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(steps):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m-sizes", type=str, default="1024,2048,4096,8192,16384")
    parser.add_argument("--k", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=100)
    args = parser.parse_args()

    m_sizes = [int(s) for s in args.m_sizes.split(",")]
    k = args.k
    n = k
    device = 'cuda'
    te_dtype = tex.DType.kFloat4E2M1

    print("=" * 110)
    print("  Fused TE Quant Benchmark")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  K={k}, N={n}, Steps={args.steps}, Warmup={args.warmup}")
    print("=" * 110)

    print(f"\n{'M':>8} | {'TE Quant':>9} {'RMS+SiLU+Q':>11} {'FP4 GEMM':>9} |"
          f" {'Fused':>8} | {'vs TEQ':>8} {'vs Full':>8} {'Q+G Saved':>10}")
    print("-" * 110)

    for m in m_sizes:
        x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
        w = torch.ones(k, device=device, dtype=torch.bfloat16)

        # ----- TE quant only -----
        quantizer = NVFP4Quantizer(fp4_dtype=te_dtype, rowwise=True, columnwise=False)
        x_q = quantizer.make_empty((m, k), dtype=torch.bfloat16, device=device)
        te_quant_ms = time_fn(lambda: quantizer.update_quantized(x, x_q), args.steps, args.warmup)

        # ----- TE full pipeline (RMS + SiLU + quant) -----
        rms = nn.RMSNorm(k, eps=1e-5, device=device, dtype=torch.bfloat16)
        silu = nn.SiLU()
        def te_full():
            out = rms(x)
            out = silu(out)
            out = out * w
            quantizer.update_quantized(out, x_q)
        te_full_ms = time_fn(te_full, args.steps, args.warmup)

        # ----- FP4 GEMM only -----
        wt = torch.randn(n, k, device=device, dtype=torch.bfloat16)
        xq = NVFP4Quantizer(fp4_dtype=te_dtype, rowwise=True, columnwise=True)
        wq = NVFP4Quantizer(fp4_dtype=te_dtype, rowwise=True, columnwise=True)
        x_nvfp4 = xq.make_empty((m, k), dtype=torch.bfloat16, device=device)
        w_nvfp4 = wq.make_empty((n, k), dtype=torch.bfloat16, device=device)
        xq.update_quantized(x, x_nvfp4)
        wq.update_quantized(wt, w_nvfp4)
        out_gemm = torch.empty(m, n, device=device, dtype=torch.bfloat16)
        workspace = torch.empty(4, dtype=torch.uint8, device=device)
        out_dtype = TE_DType[torch.bfloat16]
        bias_dtype = TE_DType[torch.bfloat16]
        def run_gemm():
            tex.generic_gemm(
                w_nvfp4, True, x_nvfp4, False,
                out_gemm, None, out_dtype,
                None, bias_dtype,
                False, None, False, workspace,
                workspace.shape[0], False, False,
            )
        gemm_ms = time_fn(run_gemm, args.steps, args.warmup)

        # ----- Our fused kernel -----
        fused_ms = time_fn(
            lambda: fused_te.forward_full(x, w, 1e-5, 0, 0),  # RMS + SiLU
            args.steps, args.warmup
        )

        # Compute comparisons
        vs_teq = fused_ms / te_quant_ms
        vs_full = fused_ms / te_full_ms

        # Q+G savings: compare (fused + gemm) vs (te_full + gemm)
        old_total = te_full_ms + gemm_ms
        new_total = fused_ms + gemm_ms
        saved_pct = (1 - new_total / old_total) * 100

        print(f"{m:>8} | {te_quant_ms:>8.3f}ms {te_full_ms:>10.3f}ms {gemm_ms:>8.3f}ms |"
              f" {fused_ms:>7.3f}ms | {vs_teq:>7.2f}x {vs_full:>7.2f}x {saved_pct:>8.1f}%")

    print()
    print("Legend:")
    print("  TE Quant    = tex.quantize() only (quant baseline)")
    print("  RMS+SiLU+Q  = separate RMSNorm → SiLU → quant (fusion target)")
    print("  FP4 GEMM    = tex.generic_gemm (cuBLASLt, pre-quantized)")
    print("  Fused       = our single-kernel RMS+SiLU+quant")
    print("  vs TEQ      = fused / quant-only (>1 = slower, <1 = faster)")
    print("  vs Full     = fused / full-pipeline (<1 = speedup!)")
    print("  Q+G Saved   = % time saved on (quant+GEMM) pipeline")


if __name__ == "__main__":
    main()
