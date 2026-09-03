"""
End-to-end test: v3 quantize → TK GEMM correctness and performance.
Tests all 3 quant variants against v2 quantize → TK GEMM.
"""
import sys, os, torch, time

# Load v2 and v3 modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'nvfp4_v2'))

import _tk_quant_v3 as v3

# v2 is _tk_quant in nvfp4_v2 (or the symlinked nvfp4 dir)
v2_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'nvfp4')
if os.path.isdir(v2_dir):
    sys.path.insert(0, v2_dir)
else:
    v2_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'nvfp4_v2')
    sys.path.insert(0, v2_dir)
import _tk_quant as v2

# Load TK GEMM
tk_dir = '/workspace/fp4_matmul/ThunderKittens/kernels/gemm/nvfp4_b200'
if os.path.isdir(tk_dir):
    sys.path.insert(0, tk_dir)
from _C import nvfp4_gemm, nvfp4_grouped_gemm, nvfp4_grouped_k_gemm


def test_basic_gemm():
    """Test: v3 basic quantize → TK GEMM vs v2 basic quantize → TK GEMM."""
    print("=" * 80)
    print("TEST 1: Basic quantize → TK GEMM (forward)")
    print("=" * 80)

    shapes = [
        (256, 4096, 4096),   # small
        (1024, 4096, 4096),  # medium
        (4096, 4096, 4096),  # Llama 1B
        (4096, 4096, 14336), # Llama 1B FFN
        (8192, 8192, 8192),  # large
    ]

    for M, K, N in shapes:
        x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
        w = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')

        # v2 quantize
        x_fp4_v2, x_sc_v2, _, _, x_sg_v2, _ = v2.tk_quantize_for_gemm(x, False)
        w_fp4_v2, w_sc_v2, _, _, w_sg_v2, _ = v2.tk_quantize_for_gemm(w, False)

        out_v2 = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
        nvfp4_gemm(x_fp4_v2, x_sc_v2, x_sg_v2, w_fp4_v2, w_sc_v2, w_sg_v2, out_v2)

        # v3 quantize
        x_fp4_v3, x_sc_v3, _, _, x_sg_v3, _ = v3.tk_quantize_for_gemm(x, False)
        w_fp4_v3, w_sc_v3, _, _, w_sg_v3, _ = v3.tk_quantize_for_gemm(w, False)

        out_v3 = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
        nvfp4_gemm(x_fp4_v3, x_sc_v3, x_sg_v3, w_fp4_v3, w_sc_v3, w_sg_v3, out_v3)

        # Compare
        match = torch.equal(out_v2, out_v3)
        maxdiff = (out_v2 - out_v3).abs().max().item()
        status = "✅" if match else ("⚠️" if maxdiff < 1e-3 else "❌")
        print(f"  {status} M={M:5d} K={K:5d} N={N:5d} | match={match} maxdiff={maxdiff:.6f}")
        torch.cuda.synchronize()

    print()


def test_grouped_gemm():
    """Test: v3 grouped quantize → TK grouped GEMM vs v2."""
    print("=" * 80)
    print("TEST 2: Grouped dim=0 quantize → TK grouped GEMM (QKV forward)")
    print("=" * 80)

    Nb = 256
    shapes = [
        # (total_N, K, splits)   — like stacked QKV weights
        (3072, 4096, [1024, 1024, 1024]),     # small: q=k=v=1024
        (12288, 4096, [4096, 4096, 4096]),     # Llama 1B
    ]

    for total_N, K, splits in shapes:
        w = torch.randn(total_N, K, dtype=torch.bfloat16, device='cuda')
        M = 4096  # batch
        x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')

        # v2
        wc_fp4_r_v2, wc_sc_r_v2, fwd_b_sg_v2, \
            fp4_c_v2, sc_c_v2, dgrad_b_sg_v2, sg_cat_v2, _ = \
            v2.tk_group_quantize_for_gemm(w, splits)
        x_fp4, x_sc, _, _, x_sg, _ = v2.tk_quantize_for_gemm(x, False)

        out_v2 = torch.empty(M, total_N, dtype=torch.bfloat16, device='cuda')
        nvfp4_grouped_gemm(x_fp4, x_sc, x_sg, wc_fp4_r_v2, wc_sc_r_v2, fwd_b_sg_v2, out_v2)

        # v3
        wc_fp4_r_v3, wc_sc_r_v3, fwd_b_sg_v3, \
            fp4_c_v3, sc_c_v3, dgrad_b_sg_v3, sg_cat_v3, _ = \
            v3.tk_group_quantize_for_gemm(w, splits)
        x_fp4_3, x_sc_3, _, _, x_sg_3, _ = v3.tk_quantize_for_gemm(x, False)

        out_v3 = torch.empty(M, total_N, dtype=torch.bfloat16, device='cuda')
        nvfp4_grouped_gemm(x_fp4_3, x_sc_3, x_sg_3, wc_fp4_r_v3, wc_sc_r_v3, fwd_b_sg_v3, out_v3)

        # Compare
        match = torch.equal(out_v2, out_v3)
        maxdiff = (out_v2 - out_v3).abs().max().item()
        status = "✅" if match else ("⚠️" if maxdiff < 1e-3 else "❌")
        print(f"  {status} N_total={total_N:5d} K={K:5d} splits={splits} | match={match} maxdiff={maxdiff:.6f}")
        torch.cuda.synchronize()

    print()


def test_grouped_dim1_gemm():
    """Test: v3 grouped dim=1 quantize vs v2 — compare quantized outputs."""
    print("=" * 80)
    print("TEST 3: Grouped dim=1 quantize (QKV backward dgrad)")
    print("=" * 80)

    shapes = [
        # (M, N_total, col_splits) — e.g. dy_cat for QKV backward
        (4096, 3072, [1024, 1024, 1024]),
        (4096, 12288, [4096, 4096, 4096]),
    ]

    for M, N_total, col_splits in shapes:
        dy = torch.randn(M, N_total, dtype=torch.bfloat16, device='cuda')

        # v2
        fp4_row_v2, sc_row_v2, sg_v2, fp4_col_v2, sc_col_v2 = \
            v2.tk_group_quantize_dim1_for_gemm(dy, col_splits)

        # v3
        fp4_row_v3, sc_row_v3, sg_v3, fp4_col_v3, sc_col_v3 = \
            v3.tk_group_quantize_dim1_for_gemm(dy, col_splits)

        # Compare per-group outputs
        all_match = True
        for g in range(len(col_splits)):
            fp4_match = torch.equal(
                fp4_row_v2[g].view(torch.uint8),
                fp4_row_v3[g].view(torch.uint8))
            sc_match = torch.equal(
                sc_row_v2[g].view(torch.uint8),
                sc_row_v3[g].view(torch.uint8))
            col_match = torch.equal(
                fp4_col_v2[g].view(torch.uint8),
                fp4_col_v3[g].view(torch.uint8))
            if not (fp4_match and sc_match and col_match):
                all_match = False

        sg_diff = (sg_v2 - sg_v3).abs().max().item()
        status = "✅" if all_match and sg_diff == 0 else "❌"
        print(f"  {status} M={M:5d} N={N_total:5d} groups={col_splits} | match={all_match} sg_diff={sg_diff:.8f}")
        torch.cuda.synchronize()

    print()


def benchmark_basic_quant_gemm():
    """Benchmark: v3 vs v2 quantize+GEMM end-to-end."""
    print("=" * 80)
    print("BENCHMARK: v3 vs v2 quantize+GEMM end-to-end")
    print("=" * 80)

    shapes = [
        (256, 4096, 4096),
        (1024, 4096, 4096),
        (4096, 4096, 14336),
        (8192, 4096, 4096),
        (16384, 4096, 4096),
    ]

    warmup = 5
    iters = 20

    for M, K, N in shapes:
        x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
        w = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')

        # v2
        for _ in range(warmup):
            xf, xs, _, _, xsg, _ = v2.tk_quantize_for_gemm(x, False)
            wf, ws, _, _, wsg, _ = v2.tk_quantize_for_gemm(w, False)
            out = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
            nvfp4_gemm(xf, xs, xsg, wf, ws, wsg, out)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(iters):
            xf, xs, _, _, xsg, _ = v2.tk_quantize_for_gemm(x, False)
            wf, ws, _, _, wsg, _ = v2.tk_quantize_for_gemm(w, False)
            out = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
            nvfp4_gemm(xf, xs, xsg, wf, ws, wsg, out)
        torch.cuda.synchronize()
        t_v2 = (time.perf_counter() - t0) / iters * 1e6

        # v3
        for _ in range(warmup):
            xf, xs, _, _, xsg, _ = v3.tk_quantize_for_gemm(x, False)
            wf, ws, _, _, wsg, _ = v3.tk_quantize_for_gemm(w, False)
            out = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
            nvfp4_gemm(xf, xs, xsg, wf, ws, wsg, out)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(iters):
            xf, xs, _, _, xsg, _ = v3.tk_quantize_for_gemm(x, False)
            wf, ws, _, _, wsg, _ = v3.tk_quantize_for_gemm(w, False)
            out = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
            nvfp4_gemm(xf, xs, xsg, wf, ws, wsg, out)
        torch.cuda.synchronize()
        t_v3 = (time.perf_counter() - t0) / iters * 1e6

        speedup = t_v2 / t_v3
        emoji = "🚀" if speedup > 1.02 else ("⚠️" if speedup < 0.95 else "➡️")
        print(f"  {emoji} M={M:5d} K={K:5d} N={N:5d} | v2={t_v2:8.1f}μs  v3={t_v3:8.1f}μs  speedup={speedup:.2f}x")

    print()


if __name__ == "__main__":
    torch.manual_seed(42)
    test_basic_gemm()
    test_grouped_gemm()
    test_grouped_dim1_gemm()
    benchmark_basic_quant_gemm()
    print("✅ ALL TESTS COMPLETE")
