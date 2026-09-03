"""
bench_cutlass_grouped_gemm.py — CUTLASS Grouped GEMM vs TE Ungrouped (3 separate calls)

Sweeps over multiple (M, K, N_q, N_k, N_v) configurations modeling real
transformer attention QKV projections at different model sizes.

Usage:
    python benchmarks/bench_cutlass_grouped_gemm.py
    python benchmarks/bench_cutlass_grouped_gemm.py --steps=200 --warmup=100
"""

import sys
import os
import torch
import argparse

sys.path.insert(0, '/workspace/low-bits-training')

import transformer_engine.pytorch as te  # must import first to init shared libs
import transformer_engine_torch as tex
from transformer_engine.pytorch.constants import TE_DType
from transformer_engine.pytorch import NVFP4Quantizer
from low_bits_training.quantization.fused_te_linear import _fast_quantize

# CUTLASS extension
from low_bits_training.quantization.fused_te_linear import _get_cutlass_gemm_ext


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


def _patch_nvfp4_tensor(t):
    if not hasattr(t, '_with_gemm_swizzled_scales'):
        t._with_gemm_swizzled_scales = False
    return t


def bench_one_config(M, K, N_dims, steps, warmup, cutlass_ext, verbose=False):
    """Benchmark one (M, K, [N_q, N_k, N_v]) config. Returns (te_ms, cutlass_ms, cos_min)."""
    device = "cuda"
    num_groups = len(N_dims)
    total_flops = sum(2.0 * M * N * K for N in N_dims)

    torch.manual_seed(42)

    # Quantize
    x_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.1
    w_list = [torch.randn(N, K, dtype=torch.bfloat16, device=device) * 0.02 for N in N_dims]
    x_q = _fast_quantize(x_bf16)
    _patch_nvfp4_tensor(x_q)
    w_q_list = [_patch_nvfp4_tensor(_fast_quantize(w)) for w in w_list]

    # TE Baseline: N separate GEMMs
    workspace = torch.empty(4, dtype=torch.uint8, device=device)
    out_dtype = TE_DType[torch.bfloat16]
    te_outs = [torch.empty(M, N, device=device, dtype=torch.bfloat16) for N in N_dims]

    def te_n_gemms():
        for g in range(num_groups):
            tex.generic_gemm(
                w_q_list[g], True, x_q, False,
                te_outs[g], None, out_dtype,
                None, TE_DType[torch.bfloat16],
                False, None, False, workspace,
                workspace.shape[0], False, False,
            )

    te_ms = _time_fn(te_n_gemms, steps, warmup)

    # CUTLASS Grouped
    A_sf_cutlass = cutlass_ext.convert_te_sf_to_cutlass(x_q._rowwise_scale_inv, M, K)
    B_data_list = [w._rowwise_data for w in w_q_list]
    B_sf_list = [cutlass_ext.convert_te_sf_to_cutlass(w._rowwise_scale_inv, N, K)
                 for w, N in zip(w_q_list, N_dims)]

    fp8_max, fp4_max = 448.0, 6.0
    amax_A = x_q._amax_rowwise.item()
    S_enc_A = fp8_max * fp4_max / amax_A if amax_A > 0 else 1.0

    def cutlass_grouped():
        return cutlass_ext.forward(
            x_q._rowwise_data, A_sf_cutlass,
            B_data_list, B_sf_list,
            N_dims, M, K, 1.0 / S_enc_A,
        )

    cutlass_ms = _time_fn(cutlass_grouped, steps, warmup)

    # Correctness: quick check
    te_n_gemms()
    cutlass_outs = cutlass_grouped()
    # Post-scale for correctness check
    for g in range(num_groups):
        amax_B = w_q_list[g]._amax_rowwise.item()
        S_enc_B = fp8_max * fp4_max / amax_B if amax_B > 0 else 1.0
        cutlass_outs[g] = cutlass_outs[g] * (1.0 / S_enc_B)

    cos_vals = []
    for g in range(num_groups):
        cos = torch.nn.functional.cosine_similarity(
            te_outs[g].float().flatten(), cutlass_outs[g].float().flatten(), dim=0)
        cos_vals.append(cos.item())
    cos_min = min(cos_vals)

    return te_ms, cutlass_ms, cos_min, total_flops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=100)
    args = parser.parse_args()

    # Typical transformer configs: (M, K, [N_q, N_k, N_v])
    # M = sequence_length * batch (per-GPU tokens)
    # K = hidden_dim
    # N_q = n_heads * head_dim, N_k = n_kv_heads * head_dim, N_v = n_kv_heads * head_dim
    configs = [
        # Llama 8B: dim=4096, n_heads=32, n_kv=8, head_dim=128
        #   N_q=4096, N_k=1024, N_v=1024
        {"label": "Llama8B  M=1024",  "M": 1024,  "K": 4096,  "N": [4096, 1024, 1024]},
        {"label": "Llama8B  M=2048",  "M": 2048,  "K": 4096,  "N": [4096, 1024, 1024]},
        {"label": "Llama8B  M=4096",  "M": 4096,  "K": 4096,  "N": [4096, 1024, 1024]},
        {"label": "Llama8B  M=8192",  "M": 8192,  "K": 4096,  "N": [4096, 1024, 1024]},
        # Llama 70B: dim=8192, n_heads=64, n_kv=8, head_dim=128
        #   N_q=8192, N_k=1024, N_v=1024
        {"label": "Llama70B M=1024",  "M": 1024,  "K": 8192,  "N": [8192, 1024, 1024]},
        {"label": "Llama70B M=2048",  "M": 2048,  "K": 8192,  "N": [8192, 1024, 1024]},
        {"label": "Llama70B M=4096",  "M": 4096,  "K": 8192,  "N": [8192, 1024, 1024]},
        {"label": "Llama70B M=8192",  "M": 8192,  "K": 8192,  "N": [8192, 1024, 1024]},
        # FFN-like: 2 groups (w1, w3) with hidden_dim expansion
        {"label": "FFN 8B   M=2048",  "M": 2048,  "K": 4096,  "N": [14336, 14336]},
        {"label": "FFN 8B   M=4096",  "M": 4096,  "K": 4096,  "N": [14336, 14336]},
        # Square for reference
        {"label": "Square   4096",    "M": 4096,  "K": 4096,  "N": [4096]},
        {"label": "Square   8192",    "M": 8192,  "K": 8192,  "N": [8192]},
    ]

    cutlass_ext = _get_cutlass_gemm_ext()
    if cutlass_ext is None:
        print("❌ CUTLASS extension not available!")
        return

    print("=" * 120)
    print("  CUTLASS Grouped GEMM vs TE Ungrouped — Multi-Size Sweep")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  Steps={args.steps} Warmup={args.warmup}")
    print("=" * 120)
    print()
    print(f"{'Config':<22} | {'M':>5} {'K':>5} {'N':>20} {'Grp':>3}"
          f" | {'TE (ms)':>9} {'CUTLASS (ms)':>12} {'Speedup':>8}"
          f" | {'TE TFLOPS':>10} {'CUT TFLOPS':>11} {'cos_min':>8}")
    print("-" * 120)

    for cfg in configs:
        M, K, N_dims = cfg["M"], cfg["K"], cfg["N"]
        try:
            te_ms, cut_ms, cos_min, flops = bench_one_config(
                M, K, N_dims, args.steps, args.warmup, cutlass_ext)
            speedup = te_ms / cut_ms
            te_tflops = flops / (te_ms * 1e-3) / 1e12
            cut_tflops = flops / (cut_ms * 1e-3) / 1e12
            N_str = "+".join(str(n) for n in N_dims)
            status = "✅" if cos_min > 0.999 else "⚠️"
            print(f"{cfg['label']:<22} | {M:>5} {K:>5} {N_str:>20} {len(N_dims):>3}"
                  f" | {te_ms:>9.4f} {cut_ms:>12.4f} {speedup:>7.2f}x"
                  f" | {te_tflops:>10.1f} {cut_tflops:>11.1f} {status}{cos_min:>7.4f}")
        except Exception as e:
            print(f"{cfg['label']:<22} | ERROR: {e}")

    print()
    print("Legend: Speedup = TE_time / CUTLASS_time (>1 = CUTLASS wins)")
    print("        cos_min = min cosine similarity across groups (should be ~0.9999)")


if __name__ == "__main__":
    main()
