"""
bench_e2e_pipeline.py — End-to-End Pipeline Comparison

Compares 3 pipelines for a single linear layer forward pass:
  A) BF16 Baseline:     RMSNorm → SiLU → BF16 GEMM           (torch.compile)
  B) TE Separate:       RMSNorm → SiLU → tex.quantize → FP4 GEMM
  C) Fused + FP4 GEMM:  fused(RMS+SiLU+quant) → FP4 GEMM

All pipelines compute:  y = (SiLU(RMSNorm(x)) * w_gain) @ W.T
where x is [M, K] activations and W is [N, K] weight matrix.

Usage:
    python3 benchmarks/bench_e2e_pipeline.py
    python3 benchmarks/bench_e2e_pipeline.py --m-sizes=1024,4096,8192,16384 --k=8192 --n=8192
"""

import os
import sys
import time
import torch
import torch.nn as nn
import argparse

# -------- Build fused kernel --------
print("Compiling fused kernel...")
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
    """Time function using CUDA events, returns average ms."""
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


def setup_bf16_pipeline(m, k, n, device='cuda'):
    """Setup BF16 pipeline: RMSNorm → SiLU → GEMM."""
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    w_gain = torch.ones(k, device=device, dtype=torch.bfloat16)
    W = torch.randn(n, k, device=device, dtype=torch.bfloat16)
    rms = nn.RMSNorm(k, eps=1e-5, device=device, dtype=torch.bfloat16)
    silu = nn.SiLU()

    def pipeline():
        h = rms(x)
        h = silu(h) * w_gain
        return torch.mm(h, W.T)

    return x, pipeline


def setup_bf16_compiled(m, k, n, device='cuda'):
    """Setup torch.compile'd BF16 pipeline."""
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    w_gain = torch.ones(k, device=device, dtype=torch.bfloat16)
    W = torch.randn(n, k, device=device, dtype=torch.bfloat16)
    rms = nn.RMSNorm(k, eps=1e-5, device=device, dtype=torch.bfloat16)
    silu = nn.SiLU()

    @torch.compile(mode="reduce-overhead")
    def pipeline(x_in):
        h = rms(x_in)
        h = silu(h) * w_gain
        return torch.mm(h, W.T)

    # Warmup compile
    for _ in range(5):
        pipeline(x)
    torch.cuda.synchronize()

    return x, lambda: pipeline(x)


def setup_te_separate(m, k, n, device='cuda'):
    """Setup TE separate pipeline: RMSNorm → SiLU → quant → FP4 GEMM."""
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    w_gain = torch.ones(k, device=device, dtype=torch.bfloat16)
    W = torch.randn(n, k, device=device, dtype=torch.bfloat16)
    rms = nn.RMSNorm(k, eps=1e-5, device=device, dtype=torch.bfloat16)
    silu = nn.SiLU()

    te_dtype = tex.DType.kFloat4E2M1
    xq = NVFP4Quantizer(fp4_dtype=te_dtype, rowwise=True, columnwise=True)
    wq = NVFP4Quantizer(fp4_dtype=te_dtype, rowwise=True, columnwise=True)
    x_nvfp4 = xq.make_empty((m, k), dtype=torch.bfloat16, device=device)
    w_nvfp4 = wq.make_empty((n, k), dtype=torch.bfloat16, device=device)

    # Pre-quantize weights (done once, not timed)
    wq.update_quantized(W, w_nvfp4)

    out = torch.empty(m, n, device=device, dtype=torch.bfloat16)
    workspace = torch.empty(4, dtype=torch.uint8, device=device)
    out_dtype = TE_DType[torch.bfloat16]
    bias_dtype = TE_DType[torch.bfloat16]

    def pipeline():
        h = rms(x)
        h = silu(h) * w_gain
        xq.update_quantized(h, x_nvfp4)
        tex.generic_gemm(
            w_nvfp4, True, x_nvfp4, False,
            out, None, out_dtype,
            None, bias_dtype,
            False, None, False, workspace,
            workspace.shape[0], False, False,
        )
        return out

    return x, pipeline


def setup_fused_fp4(m, k, n, device='cuda'):
    """Setup fused pipeline: fused(RMS+SiLU+quant) → FP4 GEMM."""
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    w_gain = torch.ones(k, device=device, dtype=torch.bfloat16)
    W = torch.randn(n, k, device=device, dtype=torch.bfloat16)

    te_dtype = tex.DType.kFloat4E2M1
    wq = NVFP4Quantizer(fp4_dtype=te_dtype, rowwise=True, columnwise=True)
    w_nvfp4 = wq.make_empty((n, k), dtype=torch.bfloat16, device=device)
    wq.update_quantized(W, w_nvfp4)

    # We need to wrap fused output into TE-compatible quantized tensor
    # For now, measure fused quant + GEMM separately since we need the
    # TE quantized tensor format for generic_gemm
    xq_quantizer = NVFP4Quantizer(fp4_dtype=te_dtype, rowwise=True, columnwise=True)
    x_nvfp4 = xq_quantizer.make_empty((m, k), dtype=torch.bfloat16, device=device)

    out = torch.empty(m, n, device=device, dtype=torch.bfloat16)
    workspace = torch.empty(4, dtype=torch.uint8, device=device)
    out_dtype = TE_DType[torch.bfloat16]
    bias_dtype = TE_DType[torch.bfloat16]

    def pipeline():
        # Fused: RMSNorm + SiLU + quant → returns (fp4_data, scales, global_scale, inv_rms)
        fp4_data, scales, global_scale, inv_rms = fused_te.forward_full(
            x, w_gain, 1e-5, 0, 0  # norm_mode=RMS, act_mode=SiLU
        )
        # For now, we still need to go through TE quant for GEMM compatibility
        # TODO: Wire fp4_data directly into cuBLASLt
        # Fallback: use separate TE quant + GEMM
        h = nn.functional.rms_norm(x, (k,), w_gain, eps=1e-5)
        h = torch.nn.functional.silu(h) * w_gain
        xq_quantizer.update_quantized(h, x_nvfp4)
        tex.generic_gemm(
            w_nvfp4, True, x_nvfp4, False,
            out, None, out_dtype,
            None, bias_dtype,
            False, None, False, workspace,
            workspace.shape[0], False, False,
        )
        return out

    # Actually, let's measure the fused quant time + GEMM time separately
    # and report the sum, since we can't yet feed our output directly to cuBLASLt
    def fused_quant_only():
        return fused_te.forward_full(x, w_gain, 1e-5, 0, 0)

    def gemm_only():
        tex.generic_gemm(
            w_nvfp4, True, x_nvfp4, False,
            out, None, out_dtype,
            None, bias_dtype,
            False, None, False, workspace,
            workspace.shape[0], False, False,
        )

    # Pre-quantize x for GEMM baseline
    h_temp = torch.nn.functional.silu(
        nn.functional.rms_norm(x, (k,), w_gain, eps=1e-5)
    ) * w_gain
    xq_quantizer.update_quantized(h_temp, x_nvfp4)

    return x, fused_quant_only, gemm_only


def main():
    parser = argparse.ArgumentParser(description="E2E Pipeline Comparison")
    parser.add_argument("--m-sizes", type=str, default="1024,2048,4096,8192,16384")
    parser.add_argument("--k", type=int, default=8192)
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--no-compile", action="store_true", help="Skip torch.compile")
    args = parser.parse_args()

    m_sizes = [int(s) for s in args.m_sizes.split(",")]
    k, n = args.k, args.n
    device = 'cuda'

    print("=" * 130)
    print("  E2E Pipeline Comparison: BF16 vs TE Separate vs Fused+FP4")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  K={k}, N={n}, Steps={args.steps}, Warmup={args.warmup}")
    print("  Pipelines:")
    print("    A) BF16:         RMSNorm → SiLU → BF16 GEMM (torch.compile)")
    print("    B) TE Separate:  RMSNorm → SiLU → tex.quantize → FP4 GEMM")
    print("    C) Fused+FP4:    fused(RMS+SiLU+quant) → FP4 GEMM")
    print("=" * 130)

    flops_per_gemm = 2.0 * n * k  # per row

    print(f"\n{'M':>8} | {'BF16 (A)':>10} {'TFLOPS':>7} | {'TE Sep (B)':>11} {'TFLOPS':>7} |"
          f" {'Fused (C)':>10} {'TFLOPS':>7} | {'B/A':>6} {'C/A':>6} {'C/B':>6}")
    print("-" * 130)

    for m in m_sizes:
        total_flops = flops_per_gemm * m

        # --- A) BF16 baseline ---
        try:
            if not args.no_compile:
                _, bf16_fn = setup_bf16_compiled(m, k, n)
            else:
                _, bf16_fn = setup_bf16_pipeline(m, k, n)
            bf16_ms = time_fn(bf16_fn, args.steps, args.warmup)
        except Exception as e:
            print(f"  [WARN] BF16 compile failed for M={m}: {e}")
            _, bf16_fn = setup_bf16_pipeline(m, k, n)
            bf16_ms = time_fn(bf16_fn, args.steps, args.warmup)

        bf16_tflops = total_flops / (bf16_ms * 1e-3) / 1e12

        # --- B) TE Separate ---
        _, te_fn = setup_te_separate(m, k, n)
        te_ms = time_fn(te_fn, args.steps, args.warmup)
        te_tflops = total_flops / (te_ms * 1e-3) / 1e12

        # --- C) Fused + FP4 ---
        _, fused_fn, gemm_fn = setup_fused_fp4(m, k, n)
        fused_quant_ms = time_fn(fused_fn, args.steps, args.warmup)
        gemm_ms = time_fn(gemm_fn, args.steps, args.warmup)
        fused_total_ms = fused_quant_ms + gemm_ms
        fused_tflops = total_flops / (fused_total_ms * 1e-3) / 1e12

        # Ratios
        b_over_a = te_ms / bf16_ms
        c_over_a = fused_total_ms / bf16_ms
        c_over_b = fused_total_ms / te_ms

        print(f"{m:>8} | {bf16_ms:>8.3f}ms {bf16_tflops:>7.1f} |"
              f" {te_ms:>9.3f}ms {te_tflops:>7.1f} |"
              f" {fused_total_ms:>8.3f}ms {fused_tflops:>7.1f} |"
              f" {b_over_a:>5.2f}x {c_over_a:>5.2f}x {c_over_b:>5.2f}x")

    print()
    print("Ratios: B/A = TE-separate vs BF16 (>1 = FP4 slower)")
    print("        C/A = Fused+FP4 vs BF16   (<1 = FP4 wins!)")
    print("        C/B = Fused vs TE-separate (<1 = fusion helps)")
    print()
    print("Note: Fused (C) = fused_quant_time + FP4_GEMM_time (added, not pipelined)")
    print("      In practice, quant and GEMM could overlap on different SMs.")
    print("      TFLOPS = 2*M*N*K / time (effective throughput)")


if __name__ == "__main__":
    main()
