"""
Test: Compare pipelined amax (v2) vs original (v1) correctness and performance.

Verifies that:
1. tk_quantize_for_gemm produces bitwise-identical results
2. Performance of the pipelined amax is measured and compared

Usage:
  cd /workspace/fp4_matmul/TK_quantisation/nvfp4_v2
  python test_v2_vs_v1.py
"""

import sys
import os
import torch
torch.random.manual_seed(42)

# Import both modules
sys.path.insert(0, '/workspace/fp4_matmul/TK_quantisation/nvfp4_v2')
import _tk_quant_v2 as v2

sys.path.insert(0, '/workspace/fp4_matmul/TK_quantisation/nvfp4')
import _tk_quant as v1


NUM_WARMUPS = 10
NUM_ITERS = 20


def test_correctness(M, K):
    """Check that v2 produces identical results to v1."""
    x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')

    # V1
    r1_fp4, r1_sc, c1_fp4, c1_sc, sg1_r, sg1_c = v1.tk_quantize_for_gemm(x, False)
    torch.cuda.synchronize()

    # V2
    r2_fp4, r2_sc, c2_fp4, c2_sc, sg2_r, sg2_c = v2.tk_quantize_for_gemm(x, False)
    torch.cuda.synchronize()

    # Compare
    fp4_match = torch.equal(r1_fp4.view(torch.uint8), r2_fp4.view(torch.uint8))
    sc_match = torch.equal(r1_sc.view(torch.uint8), r2_sc.view(torch.uint8))
    sg_diff = abs(sg1_r.item() - sg2_r.item())

    status = "✅" if fp4_match and sc_match and sg_diff < 1e-6 else "❌"
    print(f"  {status} M={M:>6d} K={K:>6d} | fp4={'match' if fp4_match else 'MISMATCH'} "
          f"sc={'match' if sc_match else 'MISMATCH'} sg_diff={sg_diff:.8f}")

    if not fp4_match:
        diff = (r1_fp4.view(torch.uint8).float() - r2_fp4.view(torch.uint8).float()).abs()
        print(f"    FP4 mismatch: max={diff.max().item()}, count={(diff > 0).sum().item()}/{diff.numel()}")
    if not sc_match:
        diff = (r1_sc.view(torch.uint8).float() - r2_sc.view(torch.uint8).float()).abs()
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
    """Compare v1 vs v2 performance."""
    x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')

    t_v1 = benchmark_quant(v1.tk_quantize_for_gemm, x)
    t_v2 = benchmark_quant(v2.tk_quantize_for_gemm, x)

    speedup = t_v1 / t_v2 if t_v2 > 0 else float('inf')
    bytes_moved = M * K * (2 + 2 + 0.5 + 1/16) * 1e-12  # TB (approx: amax read + quant read + write)
    bw_v1 = bytes_moved / (t_v1 * 1e-6)  # TB/s
    bw_v2 = bytes_moved / (t_v2 * 1e-6)

    print(f"  M={M:>6d} K={K:>6d} | v1={t_v1:>8.1f}μs  v2={t_v2:>8.1f}μs  "
          f"speedup={speedup:>5.2f}x  BW_v1={bw_v1:.2f}T  BW_v2={bw_v2:.2f}T")

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
    print("CORRECTNESS TEST: v2 (pipelined amax) vs v1 (original)")
    print("=" * 80)
    all_pass = True
    for M, K in shapes:
        ok = test_correctness(M, K)
        all_pass = all_pass and ok

    print(f"\n{'✅ ALL PASSED' if all_pass else '❌ SOME FAILURES'}\n")

    print("=" * 80)
    print("PERFORMANCE COMPARISON: v2 (pipelined amax) vs v1 (original)")
    print("=" * 80)
    speedups = []
    for M, K in shapes:
        s = test_perf(M, K)
        speedups.append(s)

    print(f"\nSpeedup range: {min(speedups):.2f}x - {max(speedups):.2f}x")
    print(f"Mean speedup: {sum(speedups)/len(speedups):.2f}x")
    print(f"Median: {sorted(speedups)[len(speedups)//2]:.2f}x")


if __name__ == '__main__':
    main()
