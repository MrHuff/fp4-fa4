"""
bench_fused_v1_v2.py — Compare V1 (2-pass) vs V2 (single-pass cooperative)

Both produce TE-compatible output. V2 should be faster at large M because
it only reads data from HBM once.

V1 is loaded via torch.utils.cpp_extension (no -rdc needed).
V2 is loaded via ctypes from a pre-compiled .so (needs cooperative launch).

Usage:
    # First build v2:
    # mkdir -p fused_ops/lib && nvcc -std=c++20 -O3 --expt-relaxed-constexpr \
    #   -gencode=arch=compute_100a,code=sm_100a -rdc=true --compiler-options '-fPIC' \
    #   -shared -lcudadevrt -I fused_ops/csrc \
    #   fused_ops/csrc/fused_te_quant_v2.cu -o fused_ops/lib/fused_te_quant_v2.so
    #
    # python3 benchmarks/bench_fused_v1_v2.py
"""

import os
import sys
import time
import ctypes
import torch
import torch.nn as nn
import argparse

# -------- Build V1 (2-pass) via torch extension --------
print("Loading V1 (2-pass)...")
from torch.utils.cpp_extension import load
CSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fused_ops', 'csrc')
v1 = load(
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

# -------- Load V2 (single-pass cooperative) via ctypes --------
print("Loading V2 (single-pass cooperative)...")
V2_SO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fused_ops', 'lib', 'fused_te_quant_v2.so')
if not os.path.exists(V2_SO):
    print(f"ERROR: {V2_SO} not found. Build with:")
    print(f"  nvcc -std=c++20 -O3 --expt-relaxed-constexpr \\")
    print(f"    -gencode=arch=compute_100a,code=sm_100a -rdc=true --compiler-options '-fPIC' \\")
    print(f"    -shared -lcudadevrt -I fused_ops/csrc \\")
    print(f"    fused_ops/csrc/fused_te_quant_v2.cu -o fused_ops/lib/fused_te_quant_v2.so")
    sys.exit(1)

v2_lib = ctypes.CDLL(V2_SO)
v2_lib.launch_fused_te_quant_v2.argtypes = [
    ctypes.c_void_p,  # x
    ctypes.c_void_p,  # w
    ctypes.c_float,   # epsilon
    ctypes.c_int,     # rows
    ctypes.c_int,     # cols
    ctypes.c_int,     # norm_mode
    ctypes.c_int,     # act_mode
    ctypes.c_void_p,  # y (fp4 packed)
    ctypes.c_void_p,  # scales (fp8)
    ctypes.c_void_p,  # global_scale
    ctypes.c_void_p,  # inv_rms_cache
]
v2_lib.launch_fused_te_quant_v2.restype = None

# -------- TE imports --------
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch.constants import TE_DType
from transformer_engine.pytorch import NVFP4Quantizer


def time_fn(fn, steps=300, warmup=100):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m-sizes", type=str, default="1024,2048,4096,8192,16384")
    parser.add_argument("--k", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=100)
    args = parser.parse_args()

    m_sizes = [int(s) for s in args.m_sizes.split(",")]
    k = args.k
    device = 'cuda'

    print("=" * 120)
    print("  V1 (2-pass) vs V2 (single-pass cooperative) Benchmark")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  K={k}, Steps={args.steps}, Warmup={args.warmup}")
    print("  Norm modes: RMS(0), MXNorm-AbsMax(1), MXNorm-BlockRMS(2)")
    print("=" * 120)

    for norm_mode, norm_name in [(0, "RMSNorm"), (2, "MXNorm-BlockRMS")]:
        print(f"\n--- {norm_name} (norm={norm_mode}) + SiLU ---")
        print(f"{'M':>8} | {'TE Quant':>9} {'TE Full':>9} |"
              f" {'V1 (2p)':>9} {'V2 (1p)':>9} |"
              f" {'V2/V1':>7} {'V2/TEQ':>7} {'V2/Full':>8}")
        print("-" * 100)

        for m in m_sizes:
            x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
            w = torch.ones(k, device=device, dtype=torch.bfloat16)

            # TE quant-only baseline
            te_dtype = tex.DType.kFloat4E2M1
            quantizer = NVFP4Quantizer(fp4_dtype=te_dtype, rowwise=True, columnwise=False)
            x_q = quantizer.make_empty((m, k), dtype=torch.bfloat16, device=device)
            te_quant_ms = time_fn(lambda: quantizer.update_quantized(x, x_q), args.steps, args.warmup)

            # TE full pipeline
            rms = nn.RMSNorm(k, eps=1e-5, device=device, dtype=torch.bfloat16)
            silu_fn = nn.SiLU()
            def te_full():
                out = rms(x)
                out = silu_fn(out) * w
                quantizer.update_quantized(out, x_q)
            te_full_ms = time_fn(te_full, args.steps, args.warmup)

            # V1 (2-pass)
            v1_ms = time_fn(
                lambda: v1.forward_full(x, w, 1e-5, norm_mode, 0),
                args.steps, args.warmup
            )

            # V2 (single-pass cooperative) via ctypes
            out_fp4 = torch.empty(m, k // 2, device=device, dtype=torch.uint8)
            out_scales = torch.empty(m, k // 16, device=device, dtype=torch.uint8)
            global_scale = torch.zeros(1, device=device, dtype=torch.float32)
            inv_rms = torch.empty(m, device=device, dtype=torch.float32)

            def run_v2():
                v2_lib.launch_fused_te_quant_v2(
                    x.data_ptr(), w.data_ptr(),
                    1e-5, m, k,
                    norm_mode, 0,  # SiLU
                    out_fp4.data_ptr(), out_scales.data_ptr(),
                    global_scale.data_ptr(), inv_rms.data_ptr(),
                )

            v2_ms = time_fn(run_v2, args.steps, args.warmup)

            print(f"{m:>8} | {te_quant_ms:>8.3f}ms {te_full_ms:>8.3f}ms |"
                  f" {v1_ms:>8.3f}ms {v2_ms:>8.3f}ms |"
                  f" {v2_ms/v1_ms:>6.2f}x {v2_ms/te_quant_ms:>6.2f}x {v2_ms/te_full_ms:>7.2f}x")

    print()
    print("Legend:")
    print("  V2/V1   = single-pass / 2-pass (<1 = V2 faster)")
    print("  V2/TEQ  = single-pass / TE quant-only (target: <2x)")
    print("  V2/Full = single-pass / TE full pipeline (<1 = win)")


if __name__ == "__main__":
    main()
