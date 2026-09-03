"""
Benchmark: TE NVFP4Quantizer vs fast_nvfp4_quantize (v1: TE amax, v2: fused amax).

Compares:
  1. TE's Quantizer.quantize() — full Python wrapper path (4 kernels)
  2. fast_nvfp4_quantize()     — C++ bypass, TE amax (3 kernels + memset)
  3. fast_nvfp4_quantize_v2()  — C++ bypass, custom fused amax (2 kernels + memset)

Tests numerical parity and measures latency.

Usage:
    python benchmarks/bench_fp4_quantizer.py [--profile]
"""

import argparse
import os
import sys
import ctypes
import time
import torch
import torch.cuda

# TE imports
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch.tensor.nvfp4_tensor import (
    NVFP4Quantizer,
    NVFP4Tensor,
)


# =========================================================================
# JIT-compile the fused extension (same method as fused_te_linear.py)
# =========================================================================
_te_fused_ext = None

def _load_fused_ext():
    global _te_fused_ext
    if _te_fused_ext is not None:
        return _te_fused_ext

    from torch.utils.cpp_extension import load

    TE_ROOT = '/workspace/low-bits-training/TransformerEngine'
    TE_LIB_DIR = os.path.join(TE_ROOT, 'build/cmake')
    for _dep in ['/usr/local/cuda/lib64/libnvrtc.so',
                 '/usr/local/cuda/lib64/libcudart.so',
                 os.path.join(TE_LIB_DIR, 'libtransformer_engine.so')]:
        if os.path.exists(_dep):
            ctypes.CDLL(_dep, mode=ctypes.RTLD_GLOBAL)

    TE_INCLUDE = os.path.join(TE_ROOT, 'transformer_engine/common/include')
    CSRC = '/workspace/fp4_matmul/fused_ops/csrc'
    CUDA_LIB = '/usr/local/cuda/lib64'

    _te_fused_ext = load(
        name='te_fused_rmsnorm_ext_bench',
        sources=[
            os.path.join(CSRC, 'te_fused_rmsnorm_ext.cpp'),
            os.path.join(CSRC, 'te_fused_pass1.cu'),
            os.path.join(CSRC, 'fused_silu_rmsnorm_backward.cu'),
            os.path.join(CSRC, 'elementwise_mul.cu'),
            os.path.join(CSRC, 'fused_amax_bf16.cu'),
        ],
        extra_include_paths=[TE_INCLUDE, '/usr/local/cuda/include', CSRC],
        extra_cflags=['-std=c++17'],
        extra_cuda_cflags=['-std=c++17', '--expt-relaxed-constexpr', '-O3'],
        extra_ldflags=[
            f'-L{TE_LIB_DIR}', '-ltransformer_engine',
            f'-Wl,-rpath,{TE_LIB_DIR}',
            f'-L{CUDA_LIB}', '-lcudart', '-lnvrtc',
            f'-Wl,-rpath,{CUDA_LIB}',
        ],
        verbose=True,
    )
    return _te_fused_ext


def make_quantizer():
    """Create a standard NVFP4Quantizer matching training config."""
    return NVFP4Quantizer(
        fp4_dtype=tex.DType.kFloat4E2M1,
        rowwise=True,
        columnwise=True,
    )


def timed_call(fn, warmup=20, iters=100):
    """Time a CUDA function call, return mean and median in microseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        fn()
        end_events[i].record()

    torch.cuda.synchronize()
    times_us = [s.elapsed_time(e) * 1000.0 for s, e in zip(start_events, end_events)]
    times_us.sort()
    trim = max(1, len(times_us) // 10)
    trimmed = times_us[trim:-trim]
    return sum(trimmed) / len(trimmed), trimmed[len(trimmed) // 2]


def verify_parity(M, K, ext, fn_name, fn_callable):
    """Compare a quantize function's output against TE quantizer."""
    x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    quantizer = make_quantizer()

    # TE path
    te_result = quantizer.quantize(x)
    te_row_data = te_result._rowwise_data
    te_row_scale = te_result._rowwise_scale_inv
    te_col_data = te_result._columnwise_data
    te_col_scale = te_result._columnwise_scale_inv

    # Fast path
    fp4, si, fp4_t, si_t, amax, amax_t = fn_callable(x, False)

    row_match = torch.equal(te_row_data, fp4)
    col_match = torch.equal(te_col_data, fp4_t)
    row_scale_match = torch.equal(te_row_scale, si[:te_row_scale.shape[0], :te_row_scale.shape[1]])
    col_scale_match = torch.equal(te_col_scale, si_t[:te_col_scale.shape[0], :te_col_scale.shape[1]])

    te_amax_val = te_result._amax_rowwise.item() if te_result._amax_rowwise is not None else None
    fast_amax = amax.item()

    ok = row_match and col_match and row_scale_match and col_scale_match
    status = "PASS" if ok else "FAIL"

    print(f"  {fn_name} ({M}x{K}): {status}  "
          f"row_data={'✓' if row_match else '✗'}  "
          f"col_data={'✓' if col_match else '✗'}  "
          f"row_scale={'✓' if row_scale_match else '✗'}  "
          f"col_scale={'✓' if col_scale_match else '✗'}  "
          f"amax={fast_amax:.4f}")

    if not ok:
        row_diff = (te_row_data != fp4).sum().item()
        col_diff = (te_col_data != fp4_t).sum().item()
        print(f"    row diff: {row_diff}/{te_row_data.numel()}  "
              f"col diff: {col_diff}/{te_col_data.numel()}")

    return ok


def benchmark_all(M, K, ext, warmup=20, iters=100):
    """Benchmark all three paths."""
    x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    quantizer = make_quantizer()

    te_avg, te_p50 = timed_call(lambda: quantizer.quantize(x), warmup, iters)
    v1_avg, v1_p50 = timed_call(lambda: ext.fast_nvfp4_quantize(x, False), warmup, iters)
    v2_avg, v2_p50 = timed_call(lambda: ext.fast_nvfp4_quantize_v2(x, False), warmup, iters)

    print(f"  ({M:5d} x {K:5d}):  "
          f"TE={te_avg:7.1f}µs  "
          f"v1={v1_avg:7.1f}µs ({te_avg/v1_avg:.2f}x)  "
          f"v2={v2_avg:7.1f}µs ({te_avg/v2_avg:.2f}x)  "
          f"v2_save={te_avg-v2_avg:.0f}µs")


def profile_path(M, K, fn, label, warmup=5, profile_iters=3):
    """Profile a single path."""
    x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')

    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        for _ in range(profile_iters):
            fn(x)
        torch.cuda.synchronize()

    events = prof.key_averages()
    cuda_calls = sum(e.count for e in events if e.device_type == torch.autograd.DeviceType.CUDA)
    cuda_time = sum(e.self_cuda_time_total for e in events if e.device_type == torch.autograd.DeviceType.CUDA)

    print(f"\n--- {label} ({M}x{K}, {profile_iters} calls) ---")
    print(f"  CUDA launches: {cuda_calls} ({cuda_calls // profile_iters} per call)  "
          f"GPU time: {cuda_time:.0f}µs ({cuda_time/profile_iters:.1f}µs/call)")
    print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    print("Loading fused extension (JIT compile if needed)...")
    ext = _load_fused_ext()
    print("Extension loaded.\n")

    shapes = [
        (2048, 2048),
        (2048, 5504),
        (2048, 8192),
    ]

    # 1. Numerical parity
    print("=" * 80)
    print("NUMERICAL PARITY CHECK")
    print("=" * 80)
    all_ok = True
    for M, K in shapes:
        ok1 = verify_parity(M, K, ext, "v1 (TE amax)", ext.fast_nvfp4_quantize)
        ok2 = verify_parity(M, K, ext, "v2 (fused)  ", ext.fast_nvfp4_quantize_v2)
        all_ok = all_ok and ok1 and ok2
    print(f"\nOverall: {'ALL PASS' if all_ok else 'MISMATCH DETECTED'}")

    # 2. Performance comparison
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON  (TE / v1=TE_amax / v2=fused_amax)")
    print("=" * 80)
    for M, K in shapes:
        benchmark_all(M, K, ext)

    # 3. Profiler
    if args.profile:
        quantizer = make_quantizer()
        print("\n" + "=" * 80)
        print("PROFILER")
        print("=" * 80)
        M, K = 2048, 8192
        profile_path(M, K, lambda x: quantizer.quantize(x), "TE Quantizer")
        profile_path(M, K, lambda x: ext.fast_nvfp4_quantize(x, False), "v1 (TE amax)")
        profile_path(M, K, lambda x: ext.fast_nvfp4_quantize_v2(x, False), "v2 (fused amax)")

    print("\n" + "=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()
