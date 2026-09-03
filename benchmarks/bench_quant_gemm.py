"""
bench_quant_gemm.py — FP4 Quantise+GEMM End-to-End Benchmark

Compares the full forward-pass cost (quantise activations + GEMM) of:
  1. BF16 matmul  (baseline, torch.matmul via cuBLAS)
  2. TE NVFP4     (NVFP4Quantizer.update_quantized + tex.generic_gemm)
  3. Quartet-II   (quant_fp4 + to_blocked + qutlass.matmul_nvf4_bf16_tn)

For TE and Quartet-II, weights are pre-quantised (simulating inference or a
training step where weights are quantised once at the start of the forward
pass). Only activation quantisation + GEMM are timed.

We also decompose the timing into:
  - Quant-only time (activation quantisation)
  - GEMM-only time  (matmul on pre-quantised data)
  - Total time      (quant + GEMM, what matters for e2e)

Usage:
    python3 benchmarks/bench_quant_gemm.py
    python3 benchmarks/bench_quant_gemm.py --sizes=4096,8192,16384 --steps=200
"""

import sys
import os
import torch
import argparse

# Add Quartet-II to path
script_dir = os.path.dirname(os.path.abspath(__file__))
quartet_kernels_path = os.path.join(script_dir, '../Quartet-II/kernels/python')
sys.path.insert(0, quartet_kernels_path)

# TE imports
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch.constants import TE_DType
from transformer_engine.pytorch import NVFP4Quantizer

# Quartet-II imports
from quartet2.quant import quant_fp4
from quartet2.linear import to_blocked, abs_max
import qutlass


# =====================================================================
# Timing helpers
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
# 1. BF16 Baseline
# =====================================================================

def bench_bf16(m, n, k, steps, warmup):
    """BF16 matmul via torch (cuBLAS). Returns total_ms."""
    a = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
    b = torch.randn(k, n, device='cuda', dtype=torch.bfloat16)
    out = torch.empty(m, n, device='cuda', dtype=torch.bfloat16)

    def run():
        torch.matmul(a, b, out=out)

    total_ms = _time_fn(run, steps, warmup)
    return total_ms


# =====================================================================
# 2. TE NVFP4: Quantise + GEMM
# =====================================================================

def bench_te_fp4(m, n, k, steps, warmup):
    """
    TE NVFP4 forward: quantise activations + GEMM.
    Weights are pre-quantised (not timed).
    Returns (quant_ms, gemm_ms, total_ms).
    """
    te_dtype = tex.DType.kFloat4E2M1
    device = "cuda"

    # Create BF16 input (activations) and weights
    x_bf16 = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    w_bf16 = torch.randn(n, k, device=device, dtype=torch.bfloat16)

    # Quantizer setup
    x_quantizer = NVFP4Quantizer(
        fp4_dtype=te_dtype, rowwise=True, columnwise=True,
        with_amax_reduction=False, amax_reduction_group=None,
        with_rht=False, with_post_rht_amax=False,
    )
    w_quantizer = NVFP4Quantizer(
        fp4_dtype=te_dtype, rowwise=True, columnwise=True,
        with_amax_reduction=False, amax_reduction_group=None,
        with_rht=False, with_post_rht_amax=False,
    )

    # Pre-quantise weights (NOT timed — done once)
    w_nvfp4 = w_quantizer.make_empty(
        (n, k), dtype=torch.bfloat16, device=device, requires_grad=False
    )
    w_nvfp4 = w_quantizer.update_quantized(w_bf16, w_nvfp4)

    # Pre-allocate activation quantised buffer
    x_nvfp4 = x_quantizer.make_empty(
        (m, k), dtype=torch.bfloat16, device=device, requires_grad=False
    )

    # GEMM config
    transa = True
    transb = False
    out_dtype = TE_DType[torch.bfloat16]
    workspace = torch.empty(4, dtype=torch.uint8, device=device)
    out = torch.empty(m, n, device=device, dtype=torch.bfloat16)

    # --- Time activation quantisation only ---
    def quant_only():
        x_quantizer.update_quantized(x_bf16, x_nvfp4)

    quant_ms = _time_fn(quant_only, steps, warmup)

    # Make sure x_nvfp4 is valid for GEMM timing
    x_quantizer.update_quantized(x_bf16, x_nvfp4)
    torch.cuda.synchronize()

    # --- Time GEMM only (pre-quantised) ---
    def gemm_only():
        tex.generic_gemm(
            w_nvfp4, transa, x_nvfp4, transb,
            out, None, out_dtype,
            None, TE_DType[torch.bfloat16],
            False, None, False, workspace,
            workspace.shape[0], False, False,
        )

    gemm_ms = _time_fn(gemm_only, steps, warmup)

    # --- Time total (quant + GEMM) ---
    def quant_and_gemm():
        x_quantizer.update_quantized(x_bf16, x_nvfp4)
        tex.generic_gemm(
            w_nvfp4, transa, x_nvfp4, transb,
            out, None, out_dtype,
            None, TE_DType[torch.bfloat16],
            False, None, False, workspace,
            workspace.shape[0], False, False,
        )

    total_ms = _time_fn(quant_and_gemm, steps, warmup)

    return quant_ms, gemm_ms, total_ms


# =====================================================================
# 3. Quartet-II: Quantise + GEMM
# =====================================================================

def bench_quartet(m, n, k, steps, warmup):
    """
    Quartet-II forward: quant_fp4 + to_blocked + qutlass GEMM.
    Weights are pre-quantised (not timed).
    Returns (quant_ms, gemm_ms, total_ms).
    """
    device = "cuda"

    # Create BF16 input and weights
    x_bf16 = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    w_bf16 = torch.randn(n, k, device=device, dtype=torch.bfloat16)

    # Pre-quantise weights (NOT timed)
    w_amax = abs_max(w_bf16)
    w_fp4 = quant_fp4(w_bf16, amax=w_amax, scale_override=1.0, four_over_six=True)
    w_scales_blocked = to_blocked(w_fp4.micro_scales)
    if w_scales_blocked.dtype != torch.float8_e4m3fn:
        w_scales_blocked = w_scales_blocked.view(torch.float8_e4m3fn)

    # Pre-compute input amax (stays constant for benchmark)
    x_amax = abs_max(x_bf16)

    # --- Time activation quantisation only ---
    def quant_only():
        x_fp4 = quant_fp4(x_bf16, amax=x_amax, scale_override=1.0, four_over_six=True)
        to_blocked(x_fp4.micro_scales)

    quant_ms = _time_fn(quant_only, steps, warmup)

    # Pre-quantise activations for GEMM timing
    x_fp4 = quant_fp4(x_bf16, amax=x_amax, scale_override=1.0, four_over_six=True)
    x_scales_blocked = to_blocked(x_fp4.micro_scales)
    if x_scales_blocked.dtype != torch.float8_e4m3fn:
        x_scales_blocked = x_scales_blocked.view(torch.float8_e4m3fn)
    alpha = x_fp4.tensor_scale * w_fp4.tensor_scale

    # --- Time GEMM only (pre-quantised) ---
    def gemm_only():
        qutlass.matmul_nvf4_bf16_tn(
            x_fp4.fp4, w_fp4.fp4,
            x_scales_blocked, w_scales_blocked,
            alpha=alpha
        )

    gemm_ms = _time_fn(gemm_only, steps, warmup)

    # --- Time total (quant + GEMM) ---
    def quant_and_gemm():
        xfp4 = quant_fp4(x_bf16, amax=x_amax, scale_override=1.0, four_over_six=True)
        xs_blocked = to_blocked(xfp4.micro_scales)
        if xs_blocked.dtype != torch.float8_e4m3fn:
            xs_blocked = xs_blocked.view(torch.float8_e4m3fn)
        a = xfp4.tensor_scale * w_fp4.tensor_scale
        qutlass.matmul_nvf4_bf16_tn(
            xfp4.fp4, w_fp4.fp4,
            xs_blocked, w_scales_blocked,
            alpha=a
        )

    total_ms = _time_fn(quant_and_gemm, steps, warmup)

    return quant_ms, gemm_ms, total_ms


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="FP4 Quantise+GEMM Benchmark")
    parser.add_argument("--sizes", type=str, default="4096,8192,12288,16384",
                        help="Comma-separated matrix sizes (square M=N=K)")
    parser.add_argument("--steps", type=int, default=200,
                        help="Timed iterations")
    parser.add_argument("--warmup", type=int, default=100,
                        help="Warmup iterations")
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]

    print("=" * 110)
    print("  FP4 Quantise + GEMM Benchmark (Forward Pass)")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  Steps: {args.steps}, Warmup: {args.warmup}")
    print("  Weights: pre-quantised (not timed). Only activation quant + GEMM timed.")
    print("=" * 110)

    # Table header
    print(f"\n{'Size':>8} | {'--- BF16 ---':^14} | {'---------- TE NVFP4 ----------':^40} | {'--------- Quartet-II ---------':^40}")
    print(f"{'':>8} | {'Total ms':>12} | {'Quant ms':>10} {'GEMM ms':>10} {'Total ms':>10} {'Speedup':>8} | {'Quant ms':>10} {'GEMM ms':>10} {'Total ms':>10} {'Speedup':>8}")
    print("-" * 115)

    for sz in sizes:
        m, n, k = sz, sz, sz
        flops = 2.0 * m * n * k

        # BF16 baseline
        bf16_ms = bench_bf16(m, n, k, args.steps, args.warmup)

        # TE NVFP4
        te_q, te_g, te_t = bench_te_fp4(m, n, k, args.steps, args.warmup)

        # Quartet-II
        try:
            q2_q, q2_g, q2_t = bench_quartet(m, n, k, args.steps, args.warmup)
        except Exception as e:
            print(f"  Quartet-II error at {sz}: {e}")
            q2_q, q2_g, q2_t = -1, -1, -1

        te_sp = bf16_ms / te_t if te_t > 0 else 0
        q2_sp = bf16_ms / q2_t if q2_t > 0 else 0

        q2_q_s = f"{q2_q:>10.3f}" if q2_q > 0 else f"{'ERR':>10}"
        q2_g_s = f"{q2_g:>10.3f}" if q2_g > 0 else f"{'ERR':>10}"
        q2_t_s = f"{q2_t:>10.3f}" if q2_t > 0 else f"{'ERR':>10}"
        q2_sp_s = f"{q2_sp:>7.2f}x" if q2_t > 0 else f"{'ERR':>8}"

        print(f"{sz:>8} | {bf16_ms:>12.3f} | {te_q:>10.3f} {te_g:>10.3f} {te_t:>10.3f} {te_sp:>7.2f}x | {q2_q_s} {q2_g_s} {q2_t_s} {q2_sp_s}")

    # Summary analysis
    print()
    print("Legend:")
    print("  BF16 Total   = torch.matmul (cuBLAS BF16)")
    print("  Quant ms     = Activation quantisation only (BF16 → FP4)")
    print("  GEMM ms      = FP4 GEMM only (pre-quantised data)")
    print("  Total ms     = Quant + GEMM combined")
    print("  Speedup      = BF16 Total / FP4 Total")
    print()
    print("Notes:")
    print("  - Weights are pre-quantised and NOT included in timing")
    print("  - TE uses cuBLASLt for GEMM; Quartet-II uses CUTLASS via qutlass")
    print("  - Quant overhead reduces effective speedup from the ~3.75x raw GEMM speedup")
    print("  - Theoretical max raw GEMM speedup = 4x (FP4 has 4x MMA throughput)")


if __name__ == "__main__":
    main()
