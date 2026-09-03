"""
bench_tk_vs_te.py — ThunderKittens vs TransformerEngine NVFP4 Benchmark

Compares:
  1. Quantisation speed: TK nvfp4_quantize vs TE NVFP4Quantizer
  2. GEMM speed: TK nvfp4_gemm vs TE (cuBLASLt) vs BF16
  3. Full pipeline: quant + GEMM with breakdown of time per step

Usage:
    python benchmarks/bench_tk_vs_te.py
    python benchmarks/bench_tk_vs_te.py --sizes=4096,8192,16384 --steps=100
"""

import sys
import os
import torch
import argparse

# Add TK extension to path
TK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      '../ThunderKittens/kernels/gemm/nvfp4_b200')
sys.path.insert(0, TK_DIR)

from _C import nvfp4_gemm, nvfp4_quantize  # type: ignore

# TE imports — same as bench_quant_gemm.py
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch.constants import TE_DType
from transformer_engine.pytorch import NVFP4Quantizer


# =====================================================================
# Timing helper
# =====================================================================

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
# TK helpers
# =====================================================================

def tk_quantize_alloc(M, K, device="cuda"):
    """Allocate TK quantisation output buffers."""
    A_fp4x2 = torch.empty(M, K // 2, dtype=torch.float4_e2m1fn_x2, device=device)
    A_sc = torch.empty(M // 128, K // 64, 512, dtype=torch.float8_e4m3fn, device=device)
    A_sc_global = torch.empty(1, dtype=torch.float32, device=device)
    return A_fp4x2, A_sc, A_sc_global


# =====================================================================
# TE helpers — exactly match bench_quant_gemm.py's working pattern
# =====================================================================

def _make_te_quantizer():
    """Create TE NVFP4Quantizer matching bench_quant_gemm.py."""
    te_dtype = tex.DType.kFloat4E2M1
    q = NVFP4Quantizer(
        fp4_dtype=te_dtype, rowwise=True, columnwise=True,
        with_amax_reduction=False, amax_reduction_group=None,
        with_rht=False, with_post_rht_amax=False,
    )
    # Safety: ensure base class attrs are set (needed by tex.quantize C++ backend)
    q.optimize_for_gemm = getattr(q, 'optimize_for_gemm', False)
    q.internal = getattr(q, 'internal', False)
    return q


def _make_te_quantizer_quant_only():
    """Create TE NVFP4Quantizer for quant-only (rowwise only)."""
    te_dtype = tex.DType.kFloat4E2M1
    q = NVFP4Quantizer(
        fp4_dtype=te_dtype, rowwise=True, columnwise=False,
        with_amax_reduction=False, amax_reduction_group=None,
        with_rht=False, with_post_rht_amax=False,
    )
    q.optimize_for_gemm = getattr(q, 'optimize_for_gemm', False)
    q.internal = getattr(q, 'internal', False)
    return q


def _patch_nvfp4_tensor(t):
    """Monkey-patch attributes needed by tex.generic_gemm on NVFP4Tensor."""
    if not hasattr(t, '_with_gemm_swizzled_scales'):
        t._with_gemm_swizzled_scales = False
    return t


# =====================================================================
# 1. Quantisation-only benchmarks
# =====================================================================

def bench_quant_tk(M, K, steps, warmup):
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    A_fp4x2, A_sc, A_sc_global = tk_quantize_alloc(M, K)
    return _time_fn(lambda: nvfp4_quantize(A, A_fp4x2, A_sc, A_sc_global, False), steps, warmup)


def bench_quant_te(M, K, steps, warmup):
    quantizer = _make_te_quantizer_quant_only()
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    A_nvfp4 = quantizer.make_empty((M, K), dtype=torch.bfloat16, device="cuda", requires_grad=False)
    return _time_fn(lambda: quantizer.update_quantized(A, A_nvfp4), steps, warmup)


# =====================================================================
# 2. GEMM-only benchmarks
# =====================================================================

def bench_gemm_tk(M, N, K, steps, warmup):
    """TK GEMM on pre-quantised FP4 data."""
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") / K ** 0.25
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / K ** 0.25
    C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

    A_fp4x2, A_sc, A_sc_global = tk_quantize_alloc(M, K)
    B_fp4x2, B_sc, B_sc_global = tk_quantize_alloc(N, K)
    nvfp4_quantize(A, A_fp4x2, A_sc, A_sc_global, False)
    nvfp4_quantize(B, B_fp4x2, B_sc, B_sc_global, False)
    torch.cuda.synchronize()

    return _time_fn(lambda: nvfp4_gemm(A_fp4x2, A_sc, A_sc_global, B_fp4x2, B_sc, B_sc_global, C), steps, warmup)


def bench_bf16(M, N, K, steps, warmup):
    """BF16 matmul via torch.matmul (cuBLAS)."""
    a = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
    b = torch.randn(K, N, device='cuda', dtype=torch.bfloat16)
    out = torch.empty(M, N, device='cuda', dtype=torch.bfloat16)
    return _time_fn(lambda: torch.matmul(a, b, out=out), steps, warmup)


def bench_gemm_te(M, N, K, steps, warmup):
    """TE GEMM via tex.generic_gemm — same pattern as bench_quant_gemm.py."""
    device = "cuda"

    x_bf16 = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    w_bf16 = torch.randn(N, K, device=device, dtype=torch.bfloat16)

    x_quantizer = _make_te_quantizer()
    w_quantizer = _make_te_quantizer()

    x_nvfp4 = x_quantizer.make_empty((M, K), dtype=torch.bfloat16, device=device, requires_grad=False)
    x_nvfp4 = x_quantizer.update_quantized(x_bf16, x_nvfp4)
    _patch_nvfp4_tensor(x_nvfp4)

    w_nvfp4 = w_quantizer.make_empty((N, K), dtype=torch.bfloat16, device=device, requires_grad=False)
    w_nvfp4 = w_quantizer.update_quantized(w_bf16, w_nvfp4)
    _patch_nvfp4_tensor(w_nvfp4)

    transa = True
    transb = False
    out_dtype = TE_DType[torch.bfloat16]
    workspace = torch.empty(4, dtype=torch.uint8, device=device)
    out = torch.empty(M, N, device=device, dtype=torch.bfloat16)

    def run():
        tex.generic_gemm(
            w_nvfp4, transa, x_nvfp4, transb,
            out, None, out_dtype,
            None, TE_DType[torch.bfloat16],
            False, None, False, workspace,
            workspace.shape[0], False, False,
        )
    return _time_fn(run, steps, warmup)


# =====================================================================
# 3. Full pipeline: quant + GEMM (weights pre-quantised)
# =====================================================================

def bench_pipeline_tk(M, N, K, steps, warmup):
    """TK: quantise activations + GEMM. Returns (quant_ms, gemm_ms, total_ms)."""
    A_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") / K ** 0.25
    B_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / K ** 0.25
    C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

    B_fp4x2, B_sc, B_sc_global = tk_quantize_alloc(N, K)
    nvfp4_quantize(B_bf16, B_fp4x2, B_sc, B_sc_global, False)
    torch.cuda.synchronize()

    A_fp4x2, A_sc, A_sc_global = tk_quantize_alloc(M, K)

    quant_ms = _time_fn(lambda: nvfp4_quantize(A_bf16, A_fp4x2, A_sc, A_sc_global, False), steps, warmup)

    nvfp4_quantize(A_bf16, A_fp4x2, A_sc, A_sc_global, False)
    torch.cuda.synchronize()

    gemm_ms = _time_fn(lambda: nvfp4_gemm(A_fp4x2, A_sc, A_sc_global, B_fp4x2, B_sc, B_sc_global, C), steps, warmup)

    def pipeline():
        nvfp4_quantize(A_bf16, A_fp4x2, A_sc, A_sc_global, False)
        nvfp4_gemm(A_fp4x2, A_sc, A_sc_global, B_fp4x2, B_sc, B_sc_global, C)
    total_ms = _time_fn(pipeline, steps, warmup)

    return quant_ms, gemm_ms, total_ms


def bench_pipeline_te(M, N, K, steps, warmup):
    """TE: quantise activations + GEMM — same as bench_quant_gemm.py's bench_te_fp4."""
    device = "cuda"

    x_bf16 = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    w_bf16 = torch.randn(N, K, device=device, dtype=torch.bfloat16)

    x_quantizer = _make_te_quantizer()
    w_quantizer = _make_te_quantizer()

    w_nvfp4 = w_quantizer.make_empty((N, K), dtype=torch.bfloat16, device=device, requires_grad=False)
    w_nvfp4 = w_quantizer.update_quantized(w_bf16, w_nvfp4)
    _patch_nvfp4_tensor(w_nvfp4)

    x_nvfp4 = x_quantizer.make_empty((M, K), dtype=torch.bfloat16, device=device, requires_grad=False)
    _patch_nvfp4_tensor(x_nvfp4)

    transa = True
    transb = False
    out_dtype = TE_DType[torch.bfloat16]
    workspace = torch.empty(4, dtype=torch.uint8, device=device)
    out = torch.empty(M, N, device=device, dtype=torch.bfloat16)

    # Quant only
    quant_ms = _time_fn(lambda: x_quantizer.update_quantized(x_bf16, x_nvfp4), steps, warmup)

    x_quantizer.update_quantized(x_bf16, x_nvfp4)
    torch.cuda.synchronize()

    # GEMM only
    def gemm_only():
        tex.generic_gemm(
            w_nvfp4, transa, x_nvfp4, transb,
            out, None, out_dtype,
            None, TE_DType[torch.bfloat16],
            False, None, False, workspace,
            workspace.shape[0], False, False,
        )
    gemm_ms = _time_fn(gemm_only, steps, warmup)

    # Total
    def pipeline():
        x_quantizer.update_quantized(x_bf16, x_nvfp4)
        tex.generic_gemm(
            w_nvfp4, transa, x_nvfp4, transb,
            out, None, out_dtype,
            None, TE_DType[torch.bfloat16],
            False, None, False, workspace,
            workspace.shape[0], False, False,
        )
    total_ms = _time_fn(pipeline, steps, warmup)

    return quant_ms, gemm_ms, total_ms


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="TK vs TE NVFP4 Benchmark")
    parser.add_argument("--sizes", type=str, default="2048,4096,8192,16384",
                        help="Comma-separated square matrix sizes (M=N=K)")
    parser.add_argument("--steps", type=int, default=100, help="Timed iterations")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations")
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]

    print("=" * 110)
    print("  ThunderKittens vs TransformerEngine — NVFP4 Benchmark")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  Steps: {args.steps}, Warmup: {args.warmup}")
    print("=" * 110)

    # ------------------------------------------------------------------
    # Section 1: Quantisation only
    # ------------------------------------------------------------------
    print()
    print("=" * 80)
    print("  Section 1: Quantisation Only (BF16 → FP4)")
    print("=" * 80)
    print(f"{'M':>8} {'K':>8} | {'TK (ms)':>10} {'TE (ms)':>10} | {'TK/TE':>8} {'Winner':>8}")
    print("-" * 70)

    for sz in sizes:
        M, K = sz, sz
        tk_ms = bench_quant_tk(M, K, args.steps, args.warmup)
        te_ms = bench_quant_te(M, K, args.steps, args.warmup)
        ratio = tk_ms / te_ms
        winner = "TK" if tk_ms < te_ms else "TE"
        gb = M * K * (2 + 0.5 + 1/16) * 1e-9
        tk_gbps = gb / (tk_ms * 1e-3)
        te_gbps = gb / (te_ms * 1e-3)
        print(f"{M:>8} {K:>8} | {tk_ms:>9.4f}  {te_ms:>9.4f}  | {ratio:>7.2f}x  {winner:>6}"
              f"   ({tk_gbps:.0f} vs {te_gbps:.0f} GB/s)")

    # ------------------------------------------------------------------
    # Section 2: GEMM only (pre-quantised)
    # ------------------------------------------------------------------
    print()
    print("=" * 110)
    print("  Section 2: GEMM Only (Pre-quantised FP4)")
    print("=" * 110)
    print(f"{'Size':>8} | {'BF16 (ms)':>10} {'TK (ms)':>10} {'TE (ms)':>10}"
          f" | {'TK TFLOPS':>11} {'TE TFLOPS':>11} {'BF16 TFLOPS':>12} | {'TK/TE':>7}")
    print("-" * 110)

    for sz in sizes:
        M, N, K = sz, sz, sz
        flops = 2.0 * M * N * K

        bf16_ms = bench_bf16(M, N, K, args.steps, args.warmup)
        tk_ms = bench_gemm_tk(M, N, K, args.steps, args.warmup)
        te_ms = bench_gemm_te(M, N, K, args.steps, args.warmup)

        bf16_tflops = flops / (bf16_ms * 1e-3) / 1e12
        tk_tflops = flops / (tk_ms * 1e-3) / 1e12
        te_tflops = flops / (te_ms * 1e-3) / 1e12
        ratio = tk_ms / te_ms

        print(f"{sz:>8} | {bf16_ms:>10.3f} {tk_ms:>10.3f} {te_ms:>10.3f}"
              f" | {tk_tflops:>10.1f}  {te_tflops:>10.1f}  {bf16_tflops:>11.1f}"
              f"  | {ratio:>6.2f}x")

    # ------------------------------------------------------------------
    # Section 3: Full pipeline (quant + GEMM)
    # ------------------------------------------------------------------
    print()
    print("=" * 120)
    print("  Section 3: Full Pipeline — Quantise Activations + GEMM (weights pre-quantised)")
    print("=" * 120)
    print(f"{'Size':>8} | {'--------- ThunderKittens ---------':^38} | {'---------- TransformerEngine ----------':^42} | {'Winner':>7}")
    print(f"{'':>8} | {'Quant':>8} {'GEMM':>8} {'Total':>8} {'Q%':>6} | {'Quant':>8} {'GEMM':>8} {'Total':>8} {'Q%':>6}  |")
    print("-" * 120)

    for sz in sizes:
        M, N, K = sz, sz, sz

        tk_q, tk_g, tk_t = bench_pipeline_tk(M, N, K, args.steps, args.warmup)
        te_q, te_g, te_t = bench_pipeline_te(M, N, K, args.steps, args.warmup)

        tk_qpct = tk_q / tk_t * 100 if tk_t > 0 else 0
        te_qpct = te_q / te_t * 100 if te_t > 0 else 0
        winner = "TK" if tk_t < te_t else "TE"

        print(f"{sz:>8} | {tk_q:>7.3f}  {tk_g:>7.3f}  {tk_t:>7.3f}  {tk_qpct:>5.1f}%"
              f" | {te_q:>7.3f}  {te_g:>7.3f}  {te_t:>7.3f}  {te_qpct:>5.1f}%"
              f"  | {winner:>6}")

    print()
    print("Legend:")
    print("  Quant   = Activation quantisation only (BF16 → FP4) in ms")
    print("  GEMM    = FP4 GEMM only (pre-quantised data) in ms")
    print("  Total   = Quant + GEMM combined in ms")
    print("  Q%      = Quant overhead as % of total pipeline time")
    print("  TK/TE   = Ratio (< 1 means TK is faster)")
    print()
    print("Notes:")
    print("  - Weights are pre-quantised and NOT included in timing")
    print("  - TK GEMM: ThunderKittens custom kernel (Blackwell tcgen05 MMA + TMA)")
    print("  - TE GEMM: cuBLASLt NVFP4 under the hood")
    print("  - Theoretical max FP4 GEMM speedup vs BF16 = 4x")


if __name__ == "__main__":
    main()
