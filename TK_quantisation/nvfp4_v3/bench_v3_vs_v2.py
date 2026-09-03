"""
Benchmark: v3 (fused single-pass) vs v2 (two-pass) quantization + TK GEMM.

v2: separate amax kernel + quantize kernel (2 HBM reads of activations)
v3: fused amax + quantize in single kernel (1 HBM read of activations)

Tests:
  1. Quant-only latency (isolate quantization cost)
  2. Quant + GEMM end-to-end latency
  3. Grouped dim=0 quant + grouped GEMM
  4. Grouped dim=1 quant-only
"""
import importlib.util, sys, os, torch, time

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

# Load modules
v2 = load_module("_tk_quant",
    "/workspace/fp4_matmul/TK_quantisation/nvfp4_v2/_tk_quant.cpython-312-aarch64-linux-gnu.so")
v3 = load_module("_tk_quant_v3",
    "/workspace/fp4_matmul/TK_quantisation/nvfp4_v3/_tk_quant_v3.cpython-312-aarch64-linux-gnu.so")

# TK GEMM
sys.path.insert(0, "/workspace/fp4_matmul/ThunderKittens/kernels/gemm/nvfp4_b200")
from _C import nvfp4_gemm, nvfp4_grouped_gemm, nvfp4_grouped_k_gemm


def cuda_timer(fn, warmup=10, iters=50):
    """Time a function using CUDA events. Returns microseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000  # ms → μs


def bench_quant_only():
    """Benchmark quantization-only latency."""
    print("=" * 90)
    print("BENCHMARK 1: Quantization-Only Latency (v2 two-pass vs v3 single-pass)")
    print("=" * 90)
    print(f"  {'M':>6s} {'K':>6s} | {'v2 (μs)':>10s} {'v3 (μs)':>10s} {'speedup':>8s} | {'note':>20s}")
    print("-" * 90)

    shapes = [
        (128,   128),
        (256,   4096),
        (1024,  4096),
        (2048,  4096),
        (4096,  4096),
        (8192,  4096),
        (16384, 4096),
        (32768, 4096),
        (65536, 4096),
    ]

    for M, K in shapes:
        x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')

        t2 = cuda_timer(lambda: v2.tk_quantize_for_gemm(x, True))
        t3 = cuda_timer(lambda: v3.tk_quantize_for_gemm(x, True))

        sp = t2 / t3
        note = "🚀 v3 wins" if sp > 1.05 else ("❌ v2 wins" if sp < 0.95 else "~tie")
        print(f"  {M:6d} {K:6d} | {t2:10.1f} {t3:10.1f} {sp:7.2f}x | {note:>20s}")

    print()


def bench_quant_gemm_e2e():
    """Benchmark quantize + GEMM end-to-end."""
    print("=" * 90)
    print("BENCHMARK 2: Quantize + GEMM End-to-End (forward pass)")
    print("=" * 90)
    print(f"  {'M':>6s} {'K':>6s} {'N':>6s} | {'v2 (μs)':>10s} {'v3 (μs)':>10s} {'speedup':>8s} | {'note':>20s}")
    print("-" * 90)

    shapes = [
        (256,   4096, 4096),
        (1024,  4096, 4096),
        (4096,  4096, 4096),
        (4096,  4096, 14336),  # FFN
        (8192,  4096, 4096),
        (16384, 4096, 4096),
        (32768, 4096, 4096),
        (4096,  8192, 8192),   # large model
        (8192,  8192, 8192),
    ]

    for M, K, N in shapes:
        x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
        w = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')

        # Pre-quantize weights (amortized in training)
        wf, ws, _, _, wsg, _ = v2.tk_quantize_for_gemm(w, False)

        def run_v2():
            xf, xs, _, _, xsg, _ = v2.tk_quantize_for_gemm(x, False)
            out = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
            nvfp4_gemm(xf, xs, xsg, wf, ws, wsg, out)
            return out

        def run_v3():
            xf, xs, _, _, xsg, _ = v3.tk_quantize_for_gemm(x, False)
            out = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
            nvfp4_gemm(xf, xs, xsg, wf, ws, wsg, out)
            return out

        t2 = cuda_timer(run_v2)
        t3 = cuda_timer(run_v3)

        sp = t2 / t3
        note = "🚀 v3 wins" if sp > 1.05 else ("❌ v2 wins" if sp < 0.95 else "~tie")
        print(f"  {M:6d} {K:6d} {N:6d} | {t2:10.1f} {t3:10.1f} {sp:7.2f}x | {note:>20s}")

    print()


def bench_grouped_dim0():
    """Benchmark grouped dim=0 quantize + grouped GEMM."""
    print("=" * 90)
    print("BENCHMARK 3: Grouped Dim=0 Quantize + Grouped GEMM (QKV forward)")
    print("=" * 90)
    print(f"  {'M':>6s} {'K':>6s} {'N_tot':>6s} {'splits':>20s} | {'v2 (μs)':>10s} {'v3 (μs)':>10s} {'speedup':>8s}")
    print("-" * 90)

    shapes = [
        (1024,  4096, 3072,  [1024, 1024, 1024]),
        (4096,  4096, 12288, [4096, 4096, 4096]),
        (8192,  4096, 12288, [4096, 4096, 4096]),
        (16384, 4096, 12288, [4096, 4096, 4096]),
        (32768, 4096, 12288, [4096, 4096, 4096]),
        (4096,  8192, 3072,  [1024, 1024, 1024]),  # Llama 70B heads
    ]

    for M, K, N_total, splits in shapes:
        x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
        w = torch.randn(N_total, K, dtype=torch.bfloat16, device='cuda')

        # Pre-quantize weights
        wf2, ws2, bsg2, _, _, _, _, _ = v2.tk_group_quantize_for_gemm(w, splits)
        wf3, ws3, bsg3, _, _, _, _, _ = v3.tk_group_quantize_for_gemm(w, splits)

        def run_v2():
            xf, xs, _, _, xsg, _ = v2.tk_quantize_for_gemm(x, False)
            out = torch.empty(M, N_total, dtype=torch.bfloat16, device='cuda')
            nvfp4_grouped_gemm(xf, xs, xsg, wf2, ws2, bsg2, out)
            return out

        def run_v3():
            xf, xs, _, _, xsg, _ = v3.tk_quantize_for_gemm(x, False)
            out = torch.empty(M, N_total, dtype=torch.bfloat16, device='cuda')
            nvfp4_grouped_gemm(xf, xs, xsg, wf3, ws3, bsg3, out)
            return out

        t2 = cuda_timer(run_v2)
        t3 = cuda_timer(run_v3)

        sp = t2 / t3
        note = "🚀" if sp > 1.05 else ("❌" if sp < 0.95 else "~")
        splits_s = str(splits)
        print(f"  {M:6d} {K:6d} {N_total:6d} {splits_s:>20s} | {t2:10.1f} {t3:10.1f} {sp:7.2f}x {note}")

    print()


def bench_grouped_dim1():
    """Benchmark grouped dim=1 quantize-only (used in QKV backward)."""
    print("=" * 90)
    print("BENCHMARK 4: Grouped Dim=1 Quantize-Only (QKV backward)")
    print("=" * 90)
    print(f"  {'M':>6s} {'N':>6s} {'groups':>20s} | {'v2 (μs)':>10s} {'v3 (μs)':>10s} {'speedup':>8s}")
    print("-" * 90)

    shapes = [
        (1024,  3072,  [1024, 1024, 1024]),
        (4096,  12288, [4096, 4096, 4096]),
        (8192,  12288, [4096, 4096, 4096]),
        (16384, 12288, [4096, 4096, 4096]),
        (32768, 12288, [4096, 4096, 4096]),
    ]

    for M, N_total, col_splits in shapes:
        dy = torch.randn(M, N_total, dtype=torch.bfloat16, device='cuda')

        t2 = cuda_timer(lambda: v2.tk_group_quantize_dim1_for_gemm(dy, col_splits))
        t3 = cuda_timer(lambda: v3.tk_group_quantize_dim1_for_gemm(dy, col_splits))

        sp = t2 / t3
        note = "🚀" if sp > 1.05 else ("❌" if sp < 0.95 else "~")
        splits_s = str(col_splits)
        print(f"  {M:6d} {N_total:6d} {splits_s:>20s} | {t2:10.1f} {t3:10.1f} {sp:7.2f}x {note}")

    print()


def bench_grouped_dim0_quant_only():
    """Benchmark grouped dim=0 quantize-only (weight quantization)."""
    print("=" * 90)
    print("BENCHMARK 5: Grouped Dim=0 Quantize-Only (weight quantization)")
    print("=" * 90)
    print(f"  {'M':>6s} {'K':>6s} {'splits':>20s} | {'v2 (μs)':>10s} {'v3 (μs)':>10s} {'speedup':>8s}")
    print("-" * 90)

    shapes = [
        (3072,  4096,  [1024, 1024, 1024]),
        (12288, 4096,  [4096, 4096, 4096]),
        (12288, 8192,  [4096, 4096, 4096]),
    ]

    for N_total, K, splits in shapes:
        w = torch.randn(N_total, K, dtype=torch.bfloat16, device='cuda')

        t2 = cuda_timer(lambda: v2.tk_group_quantize_for_gemm(w, splits))
        t3 = cuda_timer(lambda: v3.tk_group_quantize_for_gemm(w, splits))

        sp = t2 / t3
        note = "🚀" if sp > 1.05 else ("❌" if sp < 0.95 else "~")
        splits_s = str(splits)
        print(f"  {N_total:6d} {K:6d} {splits_s:>20s} | {t2:10.1f} {t3:10.1f} {sp:7.2f}x {note}")

    print()


if __name__ == "__main__":
    torch.manual_seed(42)
    print()
    print("  v2 = two-pass (separate amax kernel + quantize kernel)")
    print("  v3 = fused single-pass (amax + quantize in one kernel)")
    print()

    bench_quant_only()
    bench_quant_gemm_e2e()
    bench_grouped_dim0_quant_only()
    bench_grouped_dim0()
    bench_grouped_dim1()

    print("✅ ALL BENCHMARKS COMPLETE")
