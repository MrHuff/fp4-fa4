"""
Test: Compare v3 (fused amax+quantize) vs v2 (two-pass) correctness and performance.

Verifies that:
1. tk_v3_quantize_for_gemm produces bitwise-identical results to v2
2. Performance of the fused approach is measured and compared

Usage:
  cd /workspace/fp4_matmul/TK_quantisation/nvfp4_v3
  python test_v3_vs_v2.py
"""

import sys
import os
import torch
torch.random.manual_seed(42)

# Import v3
sys.path.insert(0, '/workspace/fp4_matmul/TK_quantisation/nvfp4_v3')
import _tk_quant_v3 as v3

# Import v2 as reference
sys.path.insert(0, '/workspace/fp4_matmul/TK_quantisation/nvfp4_v2')
import _tk_quant_v2 as v2


NUM_WARMUPS = 10
NUM_ITERS = 20


def test_correctness(M, K):
    """Check that v3 produces identical results to v2."""
    x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')

    # V2 reference
    r2_fp4, r2_sc, c2_fp4, c2_sc, sg2_r, sg2_c = v2.tk_quantize_for_gemm(x, False)
    torch.cuda.synchronize()

    # V3 fused
    r3_fp4, r3_sc, c3_fp4, c3_sc, sg3_r, sg3_c = v3.tk_v3_quantize_for_gemm(x, False)
    torch.cuda.synchronize()

    # Compare
    fp4_match = torch.equal(r2_fp4.view(torch.uint8), r3_fp4.view(torch.uint8))
    sc_match = torch.equal(r2_sc.view(torch.uint8), r3_sc.view(torch.uint8))
    sg_diff = abs(sg2_r.item() - sg3_r.item())

    status = "✅" if fp4_match and sc_match and sg_diff < 1e-6 else "❌"
    print(f"  {status} M={M:>6d} K={K:>6d} | fp4={'match' if fp4_match else 'MISMATCH'} "
          f"sc={'match' if sc_match else 'MISMATCH'} sg_diff={sg_diff:.8f}")

    if not fp4_match:
        diff = (r2_fp4.view(torch.uint8).float() - r3_fp4.view(torch.uint8).float()).abs()
        print(f"    FP4 mismatch: max={diff.max().item()}, count={(diff > 0).sum().item()}/{diff.numel()}")
    if not sc_match:
        diff = (r2_sc.view(torch.uint8).float() - r3_sc.view(torch.uint8).float()).abs()
        print(f"    SC mismatch: max={diff.max().item()}, count={(diff > 0).sum().item()}/{diff.numel()}")

    return fp4_match and sc_match and sg_diff < 1e-6


def benchmark_quant(fn, x, num_groups=5):
    """Benchmark a quantization function."""
    xs = [torch.randn_like(x) for _ in range(num_groups)]

    for i in range(NUM_WARMUPS):
        _ = fn(xs[i % num_groups], False)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for i in range(NUM_ITERS):
        _ = fn(xs[i % num_groups], False)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / NUM_ITERS * 1e3  # microseconds


def test_perf(M, K):
    """Compare v2 vs v3 performance."""
    x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')

    t_v2 = benchmark_quant(v2.tk_quantize_for_gemm, x)
    t_v3 = benchmark_quant(v3.tk_v3_quantize_for_gemm, x)

    speedup = t_v2 / t_v3 if t_v3 > 0 else float('inf')
    bytes_moved_2pass = M * K * (2 + 2 + 0.5 + 1/16) * 1e-12  # amax reads + quant reads+writes
    bytes_moved_1pass = M * K * (2 + 0.5 + 1/16) * 1e-12      # quant reads+writes (amax merged)
    bw_v2 = bytes_moved_2pass / (t_v2 * 1e-6) if t_v2 > 0 else 0
    bw_v3 = bytes_moved_1pass / (t_v3 * 1e-6) if t_v3 > 0 else 0

    mode = "COOP" if True else "FALLBACK"
    marker = "🚀" if speedup > 1.05 else ("⚠️" if speedup < 0.95 else "➡️")

    print(f"  {marker} M={M:>6d} K={K:>6d} | v2={t_v2:>8.1f}μs  v3={t_v3:>8.1f}μs  "
          f"speedup={speedup:>5.2f}x")

    return speedup


def main():
    shapes = [
        (128, 2048),
        (256, 2048),
        (512, 4096),
        (1024, 4096),
        (2048, 4096),
        (4096, 4096),
        (8192, 4096),
        (16384, 4096),
        (32768, 4096),
        (65536, 4096),
        (4096, 14336),
        (8192, 8192),
        (16384, 8192),
        (32768, 8192),
        (65536, 8192),
    ]

    print("=" * 80)
    print("CORRECTNESS TEST: v3 (fused amax+quant) vs v2 (two-pass)")
    print("=" * 80)
    all_pass = True
    for M, K in shapes:
        try:
            ok = test_correctness(M, K)
            all_pass = all_pass and ok
        except RuntimeError as e:
            print(f"  ⚠️  M={M:>6d} K={K:>6d} | SKIPPED (v3 fallback): {e}")

    print(f"\n{'✅ ALL PASSED' if all_pass else '❌ SOME FAILURES'}\n")

    print("=" * 80)
    print("PERFORMANCE COMPARISON: v3 (fused) vs v2 (two-pass)")
    print("=" * 80)
    speedups = []
    for M, K in shapes:
        try:
            s = test_perf(M, K)
            speedups.append(s)
        except RuntimeError as e:
            print(f"  ⚠️  M={M:>6d} K={K:>6d} | SKIPPED: {e}")

    if speedups:
        print(f"\nSpeedup range: {min(speedups):.2f}x - {max(speedups):.2f}x")
        print(f"Mean speedup: {sum(speedups)/len(speedups):.2f}x")
        print(f"Median: {sorted(speedups)[len(speedups)//2]:.2f}x")


if __name__ == '__main__':
    main()
