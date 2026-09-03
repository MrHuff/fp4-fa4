#!/usr/bin/env python3
"""
bench_standalone_gemm.py — Compare full quant+GEMM pipelines:
  1. TK standalone quant → TK GEMM  (no TE dependency)
  2. TE quant → TK GEMM             (current pipeline via _fast_quantize)
  3. TE quant → TE GEMM              (baseline)

Also checks GEMM output numerical parity.

Usage:
    cd /workspace/low-bits-training
    CUDA_VISIBLE_DEVICES=0 USE_TK_GEMM=1 NVTE_CUSTOM_QUANT=1 \
        python3 /workspace/fp4_matmul/TK_quantisation/bench_standalone_gemm.py
"""
import sys, os
os.environ.setdefault('NVTE_NVFP4_DISABLE_RHT', '1')
os.environ.setdefault('NVTE_NVFP4_DISABLE_2D_QUANTIZATION', '1')
os.environ.setdefault('NVTE_NVFP4_ENCODE_CENTRIC', '0')
os.environ.setdefault('NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING', '1')
os.environ.setdefault('NVTE_CUSTOM_QUANT', '1')
os.environ.setdefault('USE_TK_GEMM', '1')
os.environ.setdefault('FUSED_TE_QUANT', '0')

sys.path.insert(0, '/workspace/low-bits-training')
sys.path.insert(0, '/workspace/fp4_matmul/TK_quantisation/nvfp4')

import torch
import _tk_quant as tk_standalone  # Our module, no collision with GEMM's _C

# Import TE and TK GEMM wrappers
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch.constants import TE_DType
from low_bits_training.quantization.fused_te_linear import _fast_quantize, _TKQuantized
from low_bits_training.quantization.tk_gemm import _get_tk, tk_forward_gemm


def _time_fn(fn, steps=200, warmup=50):
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


def te_gemm(a_q, b_q, M, N):
    """TE generic_gemm for forward: y = A @ B^T."""
    ws = torch.empty(32*1024*1024, dtype=torch.uint8, device='cuda')
    out = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
    tex.generic_gemm(
        b_q, True, a_q, False,
        out, None, TE_DType[torch.bfloat16],
        None, TE_DType[torch.bfloat16],
        False, None, False, ws, ws.shape[0], False, False,
    )
    return out


# =====================================================================
# GEMM Parity Check
# =====================================================================
def check_gemm_parity(M, N, K):
    """Feed same FP4 data from both TK standalone and TE quant into TK GEMM,
    compare outputs. Also compare TE GEMM output."""
    tk = _get_tk()
    x_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device='cuda') * 0.01
    w_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device='cuda') * 0.01

    # Path A: TK standalone quant → TK GEMM
    x_sa = tk_standalone.tk_quantize_for_gemm(x_bf16, True)
    w_sa = tk_standalone.tk_quantize_for_gemm(w_bf16, True)
    # x_sa = (fp4, sc_3d, fp4_t, sc_3d_t, sg, sg)
    out_sa = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
    tk.nvfp4_gemm(x_sa[0], x_sa[1], x_sa[4],
                  w_sa[0], w_sa[1], w_sa[4], out_sa)

    # Path B: TE quant → TK GEMM (current pipeline)
    x_te = _fast_quantize(x_bf16, tk_swizzle=True)
    w_te = _fast_quantize(w_bf16, tk_swizzle=True)
    out_te_tk = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
    tk_forward_gemm(x_te, w_te, out_te_tk)

    # Path C: TE quant → TE GEMM
    x_te_only = _fast_quantize(x_bf16, tk_swizzle=False)
    w_te_only = _fast_quantize(w_bf16, tk_swizzle=False)
    out_te_te = te_gemm(x_te_only, w_te_only, M, N)

    # BF16 reference
    ref = x_bf16 @ w_bf16.T

    # Compare
    def rmse(a, b):
        return (a.float() - b.float()).pow(2).mean().sqrt().item()

    def max_diff(a, b):
        return (a.float() - b.float()).abs().max().item()

    sa_vs_tetk = rmse(out_sa, out_te_tk)
    sa_vs_tete = rmse(out_sa, out_te_te)
    tetk_vs_ref = rmse(out_te_tk, ref)
    sa_vs_ref = rmse(out_sa, ref)
    tete_vs_ref = rmse(out_te_te, ref)

    bitmatch = torch.equal(out_sa.view(torch.uint8), out_te_tk.view(torch.uint8))

    return {
        'shape': f'({M},{N},{K})',
        'sa_v_tetk_rmse': sa_vs_tetk,
        'sa_v_tetk_maxd': max_diff(out_sa, out_te_tk),
        'sa_v_tete_rmse': sa_vs_tete,
        'sa_v_ref_rmse': sa_vs_ref,
        'tetk_v_ref_rmse': tetk_vs_ref,
        'tete_v_ref_rmse': tete_vs_ref,
        'bitmatch_sa_tetk': bitmatch,
    }


# =====================================================================
# Benchmark
# =====================================================================
def bench_forward(M, N, K, steps=200, warmup=50):
    """Benchmark forward GEMM for all three paths."""
    tk = _get_tk()
    x_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device='cuda') * 0.01
    w_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device='cuda') * 0.01

    # Path A: TK standalone quant + TK GEMM (full pipeline)
    def sa_pipeline():
        xq = tk_standalone.tk_quantize_for_gemm(x_bf16, False)
        wq = tk_standalone.tk_quantize_for_gemm(w_bf16, False)
        out = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
        tk.nvfp4_gemm(xq[0], xq[1], xq[4], wq[0], wq[1], wq[4], out)
        return out

    # Path A quant-only
    def sa_quant():
        tk_standalone.tk_quantize_for_gemm(x_bf16, False)
        tk_standalone.tk_quantize_for_gemm(w_bf16, False)

    # Path A gemm-only (pre-quantized)
    xq_sa = tk_standalone.tk_quantize_for_gemm(x_bf16, False)
    wq_sa = tk_standalone.tk_quantize_for_gemm(w_bf16, False)
    out_sa = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
    def sa_gemm():
        tk.nvfp4_gemm(xq_sa[0], xq_sa[1], xq_sa[4],
                      wq_sa[0], wq_sa[1], wq_sa[4], out_sa)

    # Path B: TE quant + TK GEMM
    def te_tk_pipeline():
        xq = _fast_quantize(x_bf16, tk_swizzle=True)
        wq = _fast_quantize(w_bf16, tk_swizzle=True)
        out = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
        tk_forward_gemm(xq, wq, out)
        return out

    def te_tk_quant():
        _fast_quantize(x_bf16, tk_swizzle=True)
        _fast_quantize(w_bf16, tk_swizzle=True)

    x_tetk = _fast_quantize(x_bf16, tk_swizzle=True)
    w_tetk = _fast_quantize(w_bf16, tk_swizzle=True)
    out_tetk = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
    def te_tk_gemm():
        tk_forward_gemm(x_tetk, w_tetk, out_tetk)

    # Path C: TE quant + TE GEMM
    ws = torch.empty(32*1024*1024, dtype=torch.uint8, device='cuda')
    def te_te_pipeline():
        xq = _fast_quantize(x_bf16, tk_swizzle=False)
        wq = _fast_quantize(w_bf16, tk_swizzle=False)
        out = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
        tex.generic_gemm(
            wq, True, xq, False,
            out, None, TE_DType[torch.bfloat16],
            None, TE_DType[torch.bfloat16],
            False, None, False, ws, ws.shape[0], False, False,
        )
        return out

    x_tete = _fast_quantize(x_bf16, tk_swizzle=False)
    w_tete = _fast_quantize(w_bf16, tk_swizzle=False)
    out_tete = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
    def te_te_gemm():
        tex.generic_gemm(
            w_tete, True, x_tete, False,
            out_tete, None, TE_DType[torch.bfloat16],
            None, TE_DType[torch.bfloat16],
            False, None, False, ws, ws.shape[0], False, False,
        )

    # Run
    sa_q_ms  = _time_fn(sa_quant, steps, warmup)
    sa_g_ms  = _time_fn(sa_gemm, steps, warmup)
    sa_t_ms  = _time_fn(sa_pipeline, steps, warmup)

    tetk_q_ms = _time_fn(te_tk_quant, steps, warmup)
    tetk_g_ms = _time_fn(te_tk_gemm, steps, warmup)
    tetk_t_ms = _time_fn(te_tk_pipeline, steps, warmup)

    tete_g_ms = _time_fn(te_te_gemm, steps, warmup)
    tete_t_ms = _time_fn(te_te_pipeline, steps, warmup)

    flops = 2.0 * M * N * K
    return {
        'M': M, 'N': N, 'K': K,
        'sa_q': sa_q_ms, 'sa_g': sa_g_ms, 'sa_t': sa_t_ms,
        'tetk_q': tetk_q_ms, 'tetk_g': tetk_g_ms, 'tetk_t': tetk_t_ms,
        'tete_g': tete_g_ms, 'tete_t': tete_t_ms,
        'sa_tflops': flops / (sa_g_ms * 1e-3) / 1e12,
        'tetk_tflops': flops / (tetk_g_ms * 1e-3) / 1e12,
        'tete_tflops': flops / (tete_g_ms * 1e-3) / 1e12,
    }


def main():
    torch.manual_seed(42)
    print("=" * 120)
    print("  TK Standalone vs TE: Quant + GEMM Pipeline Benchmark")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print("=" * 120)

    # ===== Section 1: GEMM Output Parity =====
    print("\n" + "=" * 120)
    print("  Section 1: GEMM Output Parity — Same input, different quant paths → GEMM")
    print("=" * 120)
    print(f"  {'Shape':<20} {'SA↔TETK RMSE':<14} {'SA↔TETK MaxD':<14} {'SA↔TETE RMSE':<14} "
          f"{'SA↔Ref RMSE':<14} {'TETK↔Ref':<14} {'TETE↔Ref':<14} {'Bitmatch':<10}")
    print("-" * 120)

    parity_shapes = [
        (2048, 2048, 2048),
        (4096, 2048, 2048),
        (16384, 2048, 2048),
        (65536, 2048, 2048),
        (65536, 6144, 2048),   # QKV
        (65536, 2048, 5632),   # FFN w2
    ]
    for M, N, K in parity_shapes:
        r = check_gemm_parity(M, N, K)
        bm = "✓" if r['bitmatch_sa_tetk'] else "✗"
        print(f"  {r['shape']:<20} {r['sa_v_tetk_rmse']:<14.6f} {r['sa_v_tetk_maxd']:<14.6f} "
              f"{r['sa_v_tete_rmse']:<14.6f} {r['sa_v_ref_rmse']:<14.6f} "
              f"{r['tetk_v_ref_rmse']:<14.6f} {r['tete_v_ref_rmse']:<14.6f} {bm:<10}")

    # ===== Section 2: Pipeline Benchmark =====
    print("\n" + "=" * 120)
    print("  Section 2: Pipeline Benchmark — Quant + Forward GEMM")
    print("  SA = TK standalone quant + TK GEMM")
    print("  TETK = TE quant + TK GEMM")
    print("  TETE = TE quant + TE GEMM (baseline)")
    print("=" * 120)
    print(f"  {'M':>6} {'N':>6} {'K':>6} | "
          f"{'SA_Q':>7} {'SA_G':>7} {'SA_T':>7} | "
          f"{'TETK_Q':>7} {'TETK_G':>7} {'TETK_T':>7} | "
          f"{'TETE_G':>7} {'TETE_T':>7} | "
          f"{'SA/TETK':>7} {'SA/TETE':>7} {'SA_TF':>7} {'TETE_TF':>7}")
    print("-" * 120)

    bench_shapes = [
        (2048, 2048, 2048),
        (16384, 2048, 2048),
        (65536, 2048, 2048),
        (65536, 6144, 2048),   # QKV
        (65536, 5632, 2048),   # FFN w1/w3
        (65536, 2048, 5632),   # FFN w2
    ]
    steps, warmup = 200, 50
    for M, N, K in bench_shapes:
        r = bench_forward(M, N, K, steps, warmup)
        sa_vs_tetk = r['sa_t'] / r['tetk_t']
        sa_vs_tete = r['sa_t'] / r['tete_t']
        print(f"  {M:>6} {N:>6} {K:>6} | "
              f"{r['sa_q']:>6.3f}  {r['sa_g']:>6.3f}  {r['sa_t']:>6.3f} | "
              f"{r['tetk_q']:>6.3f}  {r['tetk_g']:>6.3f}  {r['tetk_t']:>6.3f} | "
              f"{r['tete_g']:>6.3f}  {r['tete_t']:>6.3f} | "
              f"{sa_vs_tetk:>6.2f}x {sa_vs_tete:>6.2f}x {r['sa_tflops']:>6.0f}  {r['tete_tflops']:>6.0f}")

    print()
    print("Legend:")
    print("  SA_Q/TETK_Q = Quant time (ms), SA_G/TETK_G/TETE_G = GEMM time (ms)")
    print("  SA_T/TETK_T/TETE_T = Total pipeline time (ms)")
    print("  SA/TETK = TK standalone total / TE+TK total (<1 = SA faster)")
    print("  SA/TETE = TK standalone total / TE+TE total (<1 = SA faster)")
    print("  SA_TF/TETE_TF = TFLOPS (GEMM only)")


if __name__ == '__main__':
    main()
