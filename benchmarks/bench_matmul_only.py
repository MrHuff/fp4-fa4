"""
bench_matmul_only.py — FP4 GEMM-Only Benchmark (No Quantisation Overhead)

Uses TE's NVFP4Quantizer + tex.generic_gemm() to measure raw FP4 GEMM
throughput vs BF16 torch.matmul, isolating the GEMM from quantisation.

Inspired by: TransformerEngine/tests/pytorch/nvfp4/test_nvfp4_gemm_exact.py

Usage:
    python3 benchmarks/bench_matmul_only.py
"""

import torch
import time
import sys
import argparse

# TE imports
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch.constants import TE_DType
from transformer_engine.pytorch import NVFP4Quantizer


def benchmark_bf16_matmul(m, n, k, steps=200, warmup=50):
    """Benchmark BF16 matmul using torch (cuBLAS)."""
    a = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
    b = torch.randn(k, n, device='cuda', dtype=torch.bfloat16)
    c = torch.empty(m, n, device='cuda', dtype=torch.bfloat16)

    # Warmup
    for _ in range(warmup):
        torch.matmul(a, b, out=c)

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(steps):
        torch.matmul(a, b, out=c)
    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)
    return elapsed_ms / steps


def benchmark_fp4_gemm(m, n, k, steps=200, warmup=50):
    """
    Benchmark FP4 GEMM using TE's tex.generic_gemm on pre-quantised NVFP4 tensors.
    Quantisation is done once upfront and NOT included in timing.
    Output tensor is pre-allocated and reused.
    """
    te_dtype = tex.DType.kFloat4E2M1
    device = "cuda"

    # Create input tensors
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    w = torch.randn(n, k, device=device, dtype=torch.bfloat16)

    # Quantize once (NOT timed)
    x_quantizer = NVFP4Quantizer(
        fp4_dtype=te_dtype,
        rowwise=True,
        columnwise=True,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=False,
        with_post_rht_amax=False,
    )
    w_quantizer = NVFP4Quantizer(
        fp4_dtype=te_dtype,
        rowwise=True,
        columnwise=True,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=False,
        with_post_rht_amax=False,
    )

    x_nvfp4 = x_quantizer.make_empty(
        (m, k), dtype=torch.bfloat16, device=device, requires_grad=False
    )
    x_nvfp4 = x_quantizer.update_quantized(x, x_nvfp4)

    w_nvfp4 = w_quantizer.make_empty(
        (n, k), dtype=torch.bfloat16, device=device, requires_grad=False
    )
    w_nvfp4 = w_quantizer.update_quantized(w, w_nvfp4)

    # GEMM config (from test_nvfp4_gemm_exact.py)
    transa = True   # w is (N, K), transpose to get (K, N)
    transb = False  # x is (M, K), no transpose
    out_quantizer = None
    out_dtype = TE_DType[torch.bfloat16]
    bias = None
    bias_dtype = TE_DType[torch.bfloat16]
    use_gelu = False
    gelu_input = None
    use_grad = False
    use_split_accumulator = False
    workspace = torch.empty(4, dtype=torch.uint8, device=device)

    # Pre-allocate output tensor (avoids allocation overhead in timing loop)
    out = torch.empty(m, n, device=device, dtype=torch.bfloat16)

    # Warmup (run GEMM, not quantization)
    for _ in range(warmup):
        tex.generic_gemm(
            w_nvfp4, transa,
            x_nvfp4, transb,
            out,  # reuse pre-allocated output
            out_quantizer, out_dtype,
            bias, bias_dtype,
            use_gelu, gelu_input,
            use_grad, workspace,
            workspace.shape[0],
            False,  # accumulate
            use_split_accumulator,
        )

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(steps):
        tex.generic_gemm(
            w_nvfp4, transa,
            x_nvfp4, transb,
            out,  # reuse pre-allocated output
            out_quantizer, out_dtype,
            bias, bias_dtype,
            use_gelu, gelu_input,
            use_grad, workspace,
            workspace.shape[0],
            False,
            use_split_accumulator,
        )
    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)
    return elapsed_ms / steps


def main():
    parser = argparse.ArgumentParser(description="FP4 GEMM-Only Benchmark")
    parser.add_argument("--sizes", type=str, default="4096,8192,12288,16384",
                        help="Comma-separated matrix sizes (square M=N=K)")
    parser.add_argument("--steps", type=int, default=200,
                        help="Number of timed iterations")
    parser.add_argument("--warmup", type=int, default=50,
                        help="Number of warmup iterations")
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    steps = args.steps
    warmup = args.warmup

    print("=" * 80)
    print("  FP4 GEMM-Only Benchmark (No Quantisation Overhead)")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  Steps: {steps}, Warmup: {warmup}")
    print("  Output tensor: pre-allocated (no alloc overhead in timed loop)")
    print("=" * 80)
    print()

    header = f"{'Size':>8} {'BF16 (ms)':>12} {'BF16 TFLOPS':>13} {'FP4 (ms)':>12} {'FP4 TFLOPS':>13} {'Speedup':>10}"
    print(header)
    print("-" * len(header))

    for sz in sizes:
        m, n, k = sz, sz, sz
        flops = 2.0 * m * n * k

        bf16_ms = benchmark_bf16_matmul(m, n, k, steps, warmup)
        fp4_ms = benchmark_fp4_gemm(m, n, k, steps, warmup)

        bf16_tflops = flops / (bf16_ms * 1e-3) / 1e12
        fp4_tflops = flops / (fp4_ms * 1e-3) / 1e12
        speedup = bf16_ms / fp4_ms

        print(f"{sz:>8} {bf16_ms:>12.3f} {bf16_tflops:>13.1f} {fp4_ms:>12.3f} {fp4_tflops:>13.1f} {speedup:>9.2f}x")

    print()
    print("Note: FP4 TFLOPS = equivalent BF16 TFLOPS (2*M*N*K / time).")
    print("Theoretical max speedup = 4x (FP4 has 4x MMA throughput vs BF16 on Blackwell).")


if __name__ == "__main__":
    main()
