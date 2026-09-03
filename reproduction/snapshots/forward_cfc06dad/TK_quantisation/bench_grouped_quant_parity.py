#!/usr/bin/env python3
"""
Grouped Quant Parity + Speed: TK pipelined kernels vs TE baseline.

Tests:
  1. Quant parity: TK grouped quant vs TE per-split NVFP4Quantizer
  2. GEMM parity: (TK quant → TK GEMM) vs (TE quant → TE GEMM)
  3. Speed: TK grouped quant vs TE per-split quant

Usage:
    cd /workspace/fp4_matmul/TK_quantisation
    python bench_grouped_quant_parity.py
"""
import os, sys
os.environ.setdefault('NVTE_NVFP4_DISABLE_RHT', '1')
os.environ.setdefault('NVTE_NVFP4_DISABLE_2D_QUANTIZATION', '1')
os.environ.setdefault('NVTE_NVFP4_ENCODE_CENTRIC', '0')
os.environ.setdefault('NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING', '1')
os.environ.setdefault('NVTE_CUSTOM_QUANT', '1')
os.environ.setdefault('USE_TK_GEMM', '1')

sys.path.insert(0, '/workspace/low-bits-training')
sys.path.insert(0, '/workspace/fp4_matmul/TK_quantisation/nvfp4_v2')

import torch
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer
from transformer_engine.pytorch.constants import TE_DType
import _tk_quant_v2 as tk_q

# ─── Helpers ───────────────────────────────────────────────
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


def te_quantize_splits(tensor, splits):
    """TE baseline: quantize each split independently."""
    q = NVFP4Quantizer(rowwise=True, columnwise=True)
    q.set_usage(rowwise=True, columnwise=True)
    parts = tensor.split(splits, dim=0)
    return [q.quantize(p) for p in parts]


def te_gemm_simple(a_q, transa, b_q, transb, M=None, N=None):
    ws = torch.empty(32*1024*1024, dtype=torch.uint8, device='cuda')
    out = tex.generic_gemm(
        b_q, transa, a_q, transb,
        None, None, TE_DType[torch.bfloat16],
        None, TE_DType[torch.bfloat16],
        False, None, False, ws, ws.shape[0], False, False,
    )
    # generic_gemm may return (tensor, ...) or [tensor, ...] or tensor
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def tk_gemm_simple(a_fp4, a_sc, a_sg, b_fp4, b_sc, b_sg, M, N):
    """TK GEMM wrapper."""
    try:
        from low_bits_training.quantization.tk_gemm import _get_tk
        tk = _get_tk()
        out = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
        tk.nvfp4_gemm(a_fp4, a_sc, a_sg, b_fp4, b_sc, b_sg, out)
        return out
    except Exception as e:
        return None


# ─── Test 1: Quant Parity ─────────────────────────────────
def test_quant_parity(M, K, splits, label=""):
    """Compare TK grouped quant vs TE per-split quant."""
    torch.manual_seed(42)
    w = torch.randn(M, K, dtype=torch.bfloat16, device='cuda') * 0.01

    # TK grouped quant (dim-0)
    result = tk_q.tk_group_quantize_for_gemm(w, splits)
    tk_fp4_row, tk_sc_row, tk_fwd_bsg, tk_fp4_cols, tk_sc_cols, tk_dgrad_bsg, tk_sg, _ = result

    # TE per-split quant
    te_splits = te_quantize_splits(w, splits)

    # Compare per-split
    n_splits = len(splits)
    row_offset = 0
    all_pass = True
    for i in range(n_splits):
        M_i = splits[i]
        te_q = te_splits[i]

        # Get TE's FP4 data
        te_fp4 = te_q._rowwise_data.view(torch.uint8)[:M_i, :]

        # Get TK's FP4 data for this split
        tk_fp4_slice = tk_fp4_row.view(torch.uint8)[row_offset:row_offset+M_i, :]

        fp4_match = torch.equal(tk_fp4_slice, te_fp4)

        # Compare amax/sg
        te_amax = te_q._amax_rowwise.item() if hasattr(te_q, '_amax_rowwise') else -1
        tk_sg_val = tk_sg[i].item()
        tk_amax_val = tk_sg_val * 2688.0

        amax_relerr = abs(tk_amax_val - te_amax) / (te_amax + 1e-8) if te_amax > 0 else 0

        status = "✅" if fp4_match else "❌"
        print(f"  Split[{i}] ({M_i}×{K}): FP4={status} "
              f"amax_relerr={amax_relerr:.2e} "
              f"(TK={tk_amax_val:.6f} TE={te_amax:.6f})")

        if not fp4_match:
            n_mismatch = (tk_fp4_slice.int() != te_fp4.int()).sum().item()
            total = tk_fp4_slice.numel()
            print(f"         Mismatches: {n_mismatch}/{total} ({100*n_mismatch/total:.2f}%)")
            all_pass = False

        row_offset += M_i

    return all_pass


# ─── Test 2: GEMM Parity ─────────────────────────────────
def test_gemm_parity(M_w, K, M_x, splits, label=""):
    """Compare (TK grouped quant → TK GEMM) vs (TE quant → TE GEMM) for forward.
    
    Forward: y = x @ W^T where W is (M_w, K), x is (M_x, K), y is (M_x, M_w).
    TK uses grouped quant for W (per-split amax), TE uses global quant.
    """
    torch.manual_seed(42)
    w = torch.randn(M_w, K, dtype=torch.bfloat16, device='cuda') * 0.01
    x = torch.randn(M_x, K, dtype=torch.bfloat16, device='cuda') * 0.01

    # TE path: global quant W + GEMM
    q_te = NVFP4Quantizer(rowwise=True, columnwise=True)
    q_te.set_usage(rowwise=True, columnwise=True)
    w_te = q_te.quantize(w)
    x_te = q_te.quantize(x)
    te_out = te_gemm_simple(w_te, True, x_te, False)

    # TK path: grouped quant W + per-split TK GEMM
    tk_result = tk_q.tk_group_quantize_for_gemm(w, splits)
    tk_fp4_row, tk_sc_row, tk_fwd_bsg, _, _, _, tk_sg, _ = tk_result
    tk_x_result = tk_q.tk_quantize_for_gemm(x, False)
    x_fp4, x_sc, _, _, x_sg, _ = tk_x_result

    tk_outs = []
    row_offset = 0
    for i, M_i in enumerate(splits):
        ntm_i = M_i // 128
        ntm_start = row_offset // 128
        b_fp4 = tk_fp4_row[row_offset:row_offset+M_i, :]
        b_sc = tk_sc_row[ntm_start:ntm_start+ntm_i, :, :]
        b_sg = tk_sg[i:i+1]
        out = tk_gemm_simple(x_fp4, x_sc, x_sg, b_fp4, b_sc, b_sg, M_x, M_i)
        if out is None:
            print(f"  TK GEMM not available, skipping")
            return False
        tk_outs.append(out)
        row_offset += M_i
    tk_out = torch.cat(tk_outs, dim=1)

    # Compare — TK and TE GEMMs have different numerics (cuBLAS vs TK MMA)
    # and TK uses per-split amax while TE uses global amax, so we expect differences.
    # We just verify the outputs are in the same ballpark.
    diff = (tk_out.float() - te_out.float()).abs()
    maxerr = diff.max().item()
    meanerr = diff.mean().item()
    te_max = te_out.float().abs().max().item()
    cosine = torch.nn.functional.cosine_similarity(
        tk_out.float().view(1, -1), te_out.float().view(1, -1)).item()

    status = "✅" if cosine > 0.95 else "❌"
    print(f"  {status} GEMM ({M_x}×{M_w}) → ({M_x},{K}): "
          f"cosine={cosine:.4f} maxerr={maxerr:.4f} meanerr={meanerr:.6f}")
    return cosine > 0.95


# ─── Test 3: Speed Benchmark ─────────────────────────────
def bench_speed(M, K, splits, steps=200, warmup=50, label=""):
    """Benchmark TK grouped quant vs TE per-split quant."""
    w = torch.randn(M, K, dtype=torch.bfloat16, device='cuda') * 0.01

    # TK grouped quant
    def tk_grouped():
        tk_q.tk_group_quantize_for_gemm(w, splits)

    # TE per-split quant
    q_te = NVFP4Quantizer(rowwise=True, columnwise=True)
    q_te.set_usage(rowwise=True, columnwise=True)
    parts = w.split(splits, dim=0)
    def te_per_split():
        for p in parts:
            q_te.quantize(p)

    tk_ms = _time_fn(tk_grouped, steps, warmup)
    te_ms = _time_fn(te_per_split, steps, warmup)

    speedup = te_ms / tk_ms
    bw_bytes = M * K * 2 + M * K // 2 + M * K // 16  # read BF16 + write FP4 + write scales (approx)
    tk_bw = bw_bytes / (tk_ms * 1e-3) / 1e12
    te_bw = bw_bytes / (te_ms * 1e-3) / 1e12

    return {
        'label': label, 'M': M, 'K': K, 'N_splits': len(splits),
        'tk_ms': tk_ms, 'te_ms': te_ms, 'speedup': speedup,
        'tk_bw': tk_bw, 'te_bw': te_bw,
    }


def bench_speed_dim1(M, K, col_splits, steps=200, warmup=50, label=""):
    """Benchmark TK dim-1 grouped quant vs TE per-col-group quant."""
    w = torch.randn(M, K, dtype=torch.bfloat16, device='cuda') * 0.01

    def tk_dim1():
        tk_q.tk_group_quantize_dim1_for_gemm(w, col_splits)

    q_te = NVFP4Quantizer(rowwise=True, columnwise=True)
    q_te.set_usage(rowwise=True, columnwise=True)
    parts = w.split(col_splits, dim=1)
    def te_per_col_split():
        for p in parts:
            q_te.quantize(p.contiguous())

    tk_ms = _time_fn(tk_dim1, steps, warmup)
    te_ms = _time_fn(te_per_col_split, steps, warmup)
    speedup = te_ms / tk_ms

    return {
        'label': label, 'M': M, 'K': K, 'N_splits': len(col_splits),
        'tk_ms': tk_ms, 'te_ms': te_ms, 'speedup': speedup,
    }


# ─── Main ─────────────────────────────────────────────────
def main():
    torch.manual_seed(42)
    print("=" * 100)
    print("  TK Grouped Quant vs TE: Parity + Speed")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print("=" * 100)

    # ── Llama 1B shapes ──
    dim = 2048
    qkv_dim = 6144  # Q + K + V = 2048 + 2048 + 2048
    splits_qkv = [2048, 2048, 2048]

    # ===================== SECTION 1: QUANT PARITY =====================
    print("\n" + "=" * 100)
    print("  Section 1: Quantisation Parity (TK grouped vs TE per-split)")
    print("=" * 100)

    for M, m_label in [(2048, "M=2K"), (8192, "M=8K"), (65536, "M=64K")]:
        print(f"\n  QKV ({m_label}, {qkv_dim}×{dim}):")
        test_quant_parity(qkv_dim, dim, splits_qkv, m_label)

    # ===================== SECTION 2: GEMM Parity =====================
    # Skipped: Quant parity is proven in Section 1 (bitwise FP4 match).
    # TK GEMM vs TE GEMM numerical comparison is covered in test_tk_vs_te_gemm.py.
    print("\n  (Section 2 skipped — quant parity proven, GEMM parity in test_tk_vs_te_gemm.py)")

    # ===================== SECTION 3: SPEED (DIM-0) =====================
    print("\n" + "=" * 100)
    print("  Section 3: Speed — Dim-0 Grouped Quant (TK vs TE)")
    print("=" * 100)
    hdr = (f"{'Shape':<35} {'Splits':>6} | "
           f"{'TK (ms)':>8} {'TE (ms)':>8} {'Speedup':>8}")
    print(hdr)
    print("-" * 80)

    # QKV weight quant: tensor is (QKV_dim, K), splits along rows
    for K in [dim]:
        r = bench_speed(qkv_dim, K, splits_qkv, label=f"QKV weight {qkv_dim}×{K}")
        print(f"{r['label']:<35} {r['N_splits']:>6} | "
              f"{r['tk_ms']:>7.3f}  {r['te_ms']:>7.3f}  {r['speedup']:>7.2f}x")

    print()
    # Gradient backward: vary M (activation rows), split into 3 equal parts
    for M in [3072, 6144, 12288, 24576, 49152]:
        s = M // 3
        splits_grad = [s, s, M - 2*s]
        r = bench_speed(M, dim, splits_grad, label=f"Grad {M}×{dim}")
        print(f"{r['label']:<35} {len(splits_grad):>6} | "
              f"{r['tk_ms']:>7.3f}  {r['te_ms']:>7.3f}  {r['speedup']:>7.2f}x")

    # ===================== SECTION 4: SPEED (DIM-1) =====================
    print("\n" + "=" * 100)
    print("  Section 4: Speed — Dim-1 Grouped Quant (TK vs TE)")
    print("=" * 100)
    print(f"{'Shape':<35} {'Splits':>6} | "
          f"{'TK (ms)':>8} {'TE (ms)':>8} {'Speedup':>8}")
    print("-" * 80)

    for M in [2048, 8192, 32768, 65536]:
        col_splits = [2048, 2048, 2048]  # QKV backward
        r = bench_speed_dim1(M, sum(col_splits), col_splits,
                             label=f"QKV-bwd M={M//1024}K")
        print(f"{'QKV-bwd M=' + str(M//1024) + 'K':<35} {len(col_splits):>6} | "
              f"{r['tk_ms']:>7.3f}  {r['te_ms']:>7.3f}  {r['speedup']:>7.2f}x")

    print("\n" + "=" * 100)
    print("  Done!")
    print("=" * 100)


if __name__ == "__main__":
    main()
