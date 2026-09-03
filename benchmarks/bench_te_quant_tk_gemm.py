"""
bench_te_quant_tk_gemm.py — TE Quantisation + TK GEMM Integration Test

Tests:
  1. Correctness: TE quant → TK GEMM vs TE quant → TE GEMM vs TK quant → TK GEMM
  2. Benchmark: Full pipeline timing comparison

Usage:
  python benchmarks/bench_te_quant_tk_gemm.py [--steps=200] [--warmup=100]
"""

import argparse
import sys
import torch
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import NVFP4Quantizer
from transformer_engine.pytorch.constants import TE_DType

sys.path.insert(0, "/workspace/fp4_matmul/ThunderKittens/kernels/gemm/nvfp4_b200")
from _C import nvfp4_gemm, nvfp4_quantize  # type: ignore


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
# TE quantiser helpers
# =====================================================================
def _make_te_quantizer(optimize_for_gemm=False):
    """Create TE NVFP4Quantizer.
    
    Note: optimize_for_gemm (kernel-side scale swizzle) requires a rebuilt TE.
    The Python-side swizzle in te_nvfp4_to_tk_format handles this instead.
    """
    q = NVFP4Quantizer(
        fp4_dtype=tex.DType.kFloat4E2M1, rowwise=True, columnwise=True,
        with_amax_reduction=False, amax_reduction_group=None,
        with_rht=False, with_post_rht_amax=False,
    )
    q.internal = getattr(q, 'internal', False)
    return q


def _make_te_quantizer_rowonly(optimize_for_gemm=False):
    """Create TE NVFP4Quantizer (rowwise only).
    
    Note: optimize_for_gemm not used — Python adapter handles swizzle.
    """
    q = NVFP4Quantizer(
        fp4_dtype=tex.DType.kFloat4E2M1, rowwise=True, columnwise=False,
        with_amax_reduction=False, amax_reduction_group=None,
        with_rht=False, with_post_rht_amax=False,
    )
    q.internal = getattr(q, 'internal', False)
    return q


# =====================================================================
# Scale swizzle helper (precomputed index map)
# =====================================================================
_SWIZZLE_IDX = None

def _apply_scale_swizzle(tiles):
    """Vectorized swizzle: [batch..., 128, 4] → [batch..., 512] using precomputed indices."""
    global _SWIZZLE_IDX
    if _SWIZZLE_IDX is None or _SWIZZLE_IDX.device != tiles.device:
        idx = torch.empty(512, dtype=torch.long, device=tiles.device)
        for row in range(128):
            for k in range(4):
                dst = (row % 32) * 16 + (row // 32) * 4 + k
                src = row * 4 + k
                idx[dst] = src
        _SWIZZLE_IDX = idx

    shape = tiles.shape[:-2]  # batch dims
    flat = tiles.reshape(-1, 512)  # merge batch dims, flatten 128*4
    swizzled = flat[:, _SWIZZLE_IDX]  # vectorized gather
    return swizzled.reshape(*shape, 512)


# =====================================================================
# Adapter: TE quant output → TK GEMM input format
# =====================================================================
def te_nvfp4_to_tk_format(nvfp4_tensor, M, K):
    """
    Convert TE NVFP4Tensor to TK's expected format.

    TE output:
      - _rowwise_data: [M, K//2] uint8 (FP4 packed pairs)
      - _rowwise_scale_inv: [M, K//16] FP8E4M3 (flat or swizzled)
      - _amax_rowwise: scalar float32 (raw amax)

    TK expects:
      - A_fp4x2: [M, K//2] float4_e2m1fn_x2
      - A_sc: [M//128, K//64, 512] float8_e4m3fn (swizzled scales)
      - A_sc_global: [1] float32 = amax / (6.0 * 448.0)
    """
    # FP4 data — same bytes, just reinterpret dtype
    fp4x2 = nvfp4_tensor._rowwise_data.view(torch.float4_e2m1fn_x2)

    n_tile_m = M // 128
    n_tile_k = K // 64  # K/16/4

    if getattr(nvfp4_tensor, '_with_gemm_swizzled_scales', False):
        # Scales already in swizzled tile layout — just reshape
        raw_bytes = nvfp4_tensor._rowwise_scale_inv.contiguous().view(-1)
        sc = raw_bytes.reshape(n_tile_m, n_tile_k, 512).view(torch.float8_e4m3fn)
    else:
        # Scales in flat row-major [M, K/16] — tile + swizzle
        scales_flat = nvfp4_tensor._rowwise_scale_inv.contiguous().view(torch.uint8)
        scales_tiled = scales_flat.reshape(n_tile_m, 128, n_tile_k, 4).permute(0, 2, 1, 3).contiguous()
        sc = _apply_scale_swizzle(scales_tiled).view(torch.float8_e4m3fn)

    # Global scale — TE stores raw amax, TK stores amax/(6.0*448.0)
    sc_global = nvfp4_tensor._amax_rowwise / (6.0 * 448.0)

    return fp4x2, sc, sc_global


# =====================================================================
# TK helpers (from bench_tk_vs_te.py)
# =====================================================================
def tk_quantize_alloc(M, K, device="cuda"):
    """Allocate TK quantisation output buffers."""
    A_fp4x2 = torch.empty(M, K // 2, dtype=torch.float4_e2m1fn_x2, device=device)
    A_sc = torch.empty(M // 128, K // 64, 512, dtype=torch.float8_e4m3fn, device=device)
    A_sc_global = torch.empty(1, dtype=torch.float32, device=device)
    return A_fp4x2, A_sc, A_sc_global


# =====================================================================
# GEMM wrappers
# =====================================================================
def do_te_gemm(x_nvfp4, w_nvfp4, M, N):
    """TE GEMM via cuBLASLt."""
    ws = torch.empty(4, dtype=torch.uint8, device='cuda')
    out = torch.empty(M, N, device='cuda', dtype=torch.bfloat16)
    tex.generic_gemm(
        w_nvfp4, True, x_nvfp4, False, out, None, TE_DType[torch.bfloat16],
        None, TE_DType[torch.bfloat16], False, None, False, ws, ws.shape[0], False, False,
    )
    return out


def do_tk_gemm(A_fp4x2, A_sc, A_sc_global, B_fp4x2, B_sc, B_sc_global, M, N):
    """TK GEMM via ThunderKittens."""
    C = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
    nvfp4_gemm(A_fp4x2, A_sc, A_sc_global, B_fp4x2, B_sc, B_sc_global, C)
    return C


# =====================================================================
# Correctness test
# =====================================================================
def test_correctness(sizes, verbose=True):
    """Compare three paths: TE→TE, TE→TK, TK→TK"""
    print("=" * 80)
    print("  Correctness: TE→TE vs TE→TK vs TK→TK")
    print("=" * 80)

    for sz in sizes:
        M, N, K = sz, sz, sz
        torch.manual_seed(42)
        x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda') / K ** 0.25
        w = torch.randn(N, K, dtype=torch.bfloat16, device='cuda') / K ** 0.25

        # --- Path 1: TE quant + TE GEMM ---
        qx1 = _make_te_quantizer(optimize_for_gemm=True)
        qw1 = _make_te_quantizer(optimize_for_gemm=True)
        x1 = qx1.make_empty((M, K), dtype=torch.bfloat16, device='cuda', requires_grad=False)
        w1 = qw1.make_empty((N, K), dtype=torch.bfloat16, device='cuda', requires_grad=False)
        qx1.update_quantized(x, x1)
        qw1.update_quantized(w, w1)
        out_te_te = do_te_gemm(x1, w1, M, N)

        # --- Path 2: TE quant + TK GEMM ---
        # Reuse same quantized data from Path 1
        A_fp4x2, A_sc, A_sc_global = te_nvfp4_to_tk_format(x1, M, K)
        B_fp4x2, B_sc, B_sc_global = te_nvfp4_to_tk_format(w1, N, K)
        out_te_tk = do_tk_gemm(A_fp4x2, A_sc, A_sc_global, B_fp4x2, B_sc, B_sc_global, M, N)

        # --- Path 3: TK quant + TK GEMM ---
        A_fp4x2_tk, A_sc_tk, A_sc_global_tk = tk_quantize_alloc(M, K)
        B_fp4x2_tk, B_sc_tk, B_sc_global_tk = tk_quantize_alloc(N, K)
        nvfp4_quantize(x, A_fp4x2_tk, A_sc_tk, A_sc_global_tk, False)
        nvfp4_quantize(w, B_fp4x2_tk, B_sc_tk, B_sc_global_tk, False)
        out_tk_tk = do_tk_gemm(A_fp4x2_tk, A_sc_tk, A_sc_global_tk,
                               B_fp4x2_tk, B_sc_tk, B_sc_global_tk, M, N)

        # Compare
        diff_te_tk = (out_te_te - out_te_tk).abs()
        diff_tk_tk = (out_te_te - out_tk_tk).abs()

        # Use relative comparison
        ref_norm = out_te_te.abs().mean()
        pass_te_tk = diff_te_tk.mean() / ref_norm < 0.1  # 10% relative error OK for FP4
        pass_tk_tk = diff_tk_tk.mean() / ref_norm < 0.1

        tag_te_tk = "PASS" if pass_te_tk else "FAIL"
        tag_tk_tk = "PASS" if pass_tk_tk else "FAIL"

        if verbose:
            print(f"  sz={sz:5d}:")
            print(f"    TE→TK vs TE→TE: {tag_te_tk}  max={diff_te_tk.max():.4f}  mean={diff_te_tk.mean():.4f}  rel={diff_te_tk.mean()/ref_norm:.4f}")
            print(f"    TK→TK vs TE→TE: {tag_tk_tk}  max={diff_tk_tk.max():.4f}  mean={diff_tk_tk.mean():.4f}  rel={diff_tk_tk.mean()/ref_norm:.4f}")


# =====================================================================
# Benchmark
# =====================================================================
def benchmark(sizes, steps, warmup):
    """Benchmark three full pipelines."""
    print()
    print("=" * 100)
    print("  Full Pipeline Benchmark: Quant + GEMM (weights pre-quantised)")
    print("=" * 100)
    print(f"  {'Size':>6} | {'TE→TE (ms)':>12} {'TE→TK (ms)':>12} {'TK→TK (ms)':>12} | {'TE→TK/TE→TE':>12} {'TK→TK/TE→TE':>12}")
    print("-" * 100)

    for sz in sizes:
        M, N, K = sz, sz, sz
        x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda') / K ** 0.25
        w = torch.randn(N, K, dtype=torch.bfloat16, device='cuda') / K ** 0.25

        # --- TE→TE pipeline ---
        qxte = _make_te_quantizer(optimize_for_gemm=True)
        qwte = _make_te_quantizer(optimize_for_gemm=True)
        xte = qxte.make_empty((M, K), dtype=torch.bfloat16, device='cuda', requires_grad=False)
        wte = qwte.make_empty((N, K), dtype=torch.bfloat16, device='cuda', requires_grad=False)
        qwte.update_quantized(w, wte)
        out_te = torch.empty(M, N, device='cuda', dtype=torch.bfloat16)
        ws = torch.empty(4, dtype=torch.uint8, device='cuda')

        def te_te_pipeline():
            qxte.update_quantized(x, xte)
            tex.generic_gemm(wte, True, xte, False, out_te, None, TE_DType[torch.bfloat16],
                             None, TE_DType[torch.bfloat16], False, None, False, ws, ws.shape[0], False, False)

        te_te_ms = _time_fn(te_te_pipeline, steps, warmup)

        # --- TE→TK pipeline ---
        qxtk = _make_te_quantizer_rowonly(optimize_for_gemm=True)
        qwtk = _make_te_quantizer_rowonly(optimize_for_gemm=True)
        xtk = qxtk.make_empty((M, K), dtype=torch.bfloat16, device='cuda', requires_grad=False)
        wtk = qwtk.make_empty((N, K), dtype=torch.bfloat16, device='cuda', requires_grad=False)
        qwtk.update_quantized(w, wtk)
        B_fp4x2, B_sc, B_sc_global = te_nvfp4_to_tk_format(wtk, N, K)
        C_tk = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')

        # Pre-compute views once — update_quantized writes to same buffers
        qxtk.update_quantized(x, xtk)  # initial quant to set up buffers
        A_fp4_pre, A_sc_pre, _ = te_nvfp4_to_tk_format(xtk, M, K)

        def te_tk_pipeline():
            qxtk.update_quantized(x, xtk)
            # Data/scale views are pre-computed; only global scale needs refresh
            A_sg = xtk._amax_rowwise / (6.0 * 448.0)
            nvfp4_gemm(A_fp4_pre, A_sc_pre, A_sg, B_fp4x2, B_sc, B_sc_global, C_tk)

        te_tk_ms = _time_fn(te_tk_pipeline, steps, warmup)

        # --- TK→TK pipeline ---
        B_fp4x2_tk, B_sc_tk, B_sc_global_tk = tk_quantize_alloc(N, K)
        nvfp4_quantize(w, B_fp4x2_tk, B_sc_tk, B_sc_global_tk, False)
        A_fp4x2_tk, A_sc_tk, A_sc_global_tk = tk_quantize_alloc(M, K)
        C_tktk = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')

        def tk_tk_pipeline():
            nvfp4_quantize(x, A_fp4x2_tk, A_sc_tk, A_sc_global_tk, False)
            nvfp4_gemm(A_fp4x2_tk, A_sc_tk, A_sc_global_tk, B_fp4x2_tk, B_sc_tk, B_sc_global_tk, C_tktk)

        tk_tk_ms = _time_fn(tk_tk_pipeline, steps, warmup)

        ratio_te_tk = te_tk_ms / te_te_ms
        ratio_tk_tk = tk_tk_ms / te_te_ms
        print(f"  {sz:6d} | {te_te_ms:12.4f} {te_tk_ms:12.4f} {tk_tk_ms:12.4f} | {ratio_te_tk:12.2f}x {ratio_tk_tk:12.2f}x")


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=100)
    args = parser.parse_args()

    sizes = [256, 512, 1024, 2048, 4096, 8192]

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Steps: {args.steps}, Warmup: {args.warmup}")
    print()

    test_correctness(sizes)
    benchmark(sizes, args.steps, args.warmup)


if __name__ == "__main__":
    main()
