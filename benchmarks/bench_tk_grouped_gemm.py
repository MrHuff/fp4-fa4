"""
bench_tk_grouped_gemm.py — Four-way comparison: TE→TK vs TK vs CUTLASS grouped vs TE

For each (M, K, [N_q, N_k, N_v]) configuration, compares:
  1. TE: N separate tex.generic_gemm calls (cuBLASLt)
  2. CUTLASS: 1 grouped GEMM call 
  3. TK: N separate nvfp4_gemm calls (TK quant)
  4. TE→TK: N separate nvfp4_gemm calls (TE quant + TK GEMM)

Each method uses its own quantization. Only GEMM time is measured.

Usage:
    python benchmarks/bench_tk_grouped_gemm.py
"""

import sys
import os
import torch
import argparse

# TK imports
TK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      '../ThunderKittens/kernels/gemm/nvfp4_b200')
sys.path.insert(0, TK_DIR)
from _C import nvfp4_gemm, nvfp4_grouped_gemm, nvfp4_quantize  # type: ignore

# TE imports
sys.path.insert(0, '/workspace/low-bits-training')
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch.constants import TE_DType
from low_bits_training.quantization.fused_te_linear import _fast_quantize, _get_cutlass_gemm_ext
from bench_te_quant_tk_gemm import te_nvfp4_to_tk_format


def _time_fn(fn, steps, warmup):
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


def _patch(t):
    if not hasattr(t, '_with_gemm_swizzled_scales'):
        t._with_gemm_swizzled_scales = False
    return t


def tk_quantize(x_bf16):
    """Quantize using TK's nvfp4_quantize. Returns (fp4x2, sc, sc_global)."""
    M, K = x_bf16.shape
    assert M % 128 == 0 and K % 128 == 0
    fp4x2 = torch.empty(M, K // 2, dtype=torch.float4_e2m1fn_x2, device="cuda")
    sc = torch.empty(M // 128, K // 64, 512, dtype=torch.float8_e4m3fn, device="cuda")
    sc_global = torch.empty(1, dtype=torch.float32, device="cuda")
    nvfp4_quantize(x_bf16, fp4x2, sc, sc_global, False)
    return fp4x2, sc, sc_global


def bench_one_config(M, K, N_dims, steps, warmup, cutlass_ext):
    """Benchmark one config. Returns (te_ms, cutlass_ms, tk_ms, flops)."""
    device = "cuda"
    num_groups = len(N_dims)
    total_flops = sum(2.0 * M * N * K for N in N_dims)

    torch.manual_seed(42)
    x_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.1
    w_list = [torch.randn(N, K, dtype=torch.bfloat16, device=device) * 0.02 for N in N_dims]

    # ========== TE ==========
    x_q = _patch(_fast_quantize(x_bf16))
    w_q_list = [_patch(_fast_quantize(w)) for w in w_list]
    workspace = torch.empty(4, dtype=torch.uint8, device=device)
    out_dtype = TE_DType[torch.bfloat16]
    te_outs = [torch.empty(M, N, device=device, dtype=torch.bfloat16) for N in N_dims]

    def te_gemms():
        for g in range(num_groups):
            tex.generic_gemm(
                w_q_list[g], True, x_q, False,
                te_outs[g], None, out_dtype,
                None, TE_DType[torch.bfloat16],
                False, None, False, workspace,
                workspace.shape[0], False, False,
            )

    te_ms = _time_fn(te_gemms, steps, warmup)

    # ========== CUTLASS Grouped ==========
    cutlass_ms = None
    if cutlass_ext is not None:
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

    # ========== TK ==========
    # Quantize with TK format
    tk_x_fp4, tk_x_sc, tk_x_sc_global = tk_quantize(x_bf16)
    tk_w_quant = [tk_quantize(w) for w in w_list]
    tk_outs = [torch.zeros(M, N, dtype=torch.bfloat16, device=device) for N in N_dims]

    def tk_gemms():
        for g in range(num_groups):
            w_fp4, w_sc, w_sc_global = tk_w_quant[g]
            nvfp4_gemm(tk_x_fp4, tk_x_sc, tk_x_sc_global,
                       w_fp4, w_sc, w_sc_global, tk_outs[g])

    tk_ms = _time_fn(tk_gemms, steps, warmup)

    # ========== TE→TK (TE quant + TK GEMM) ==========
    # Use TE-quantized data, convert to TK format, run TK GEMM
    tetk_x_fp4, tetk_x_sc, tetk_x_sg = te_nvfp4_to_tk_format(x_q, M, K)
    tetk_w_quant = [te_nvfp4_to_tk_format(w, N, K) for w, N in zip(w_q_list, N_dims)]
    tetk_outs = [torch.zeros(M, N, dtype=torch.bfloat16, device=device) for N in N_dims]

    def tetk_gemms():
        for g in range(num_groups):
            w_fp4, w_sc, w_sc_global = tetk_w_quant[g]
            nvfp4_gemm(tetk_x_fp4, tetk_x_sc, tetk_x_sg,
                       w_fp4, w_sc, w_sc_global, tetk_outs[g])

    tetk_ms = _time_fn(tetk_gemms, steps, warmup)

    # ========== Full Pipeline: Grouped TE Quant + N×TK GEMM ==========
    # Use tex.split_quantize for single-kernel grouped quant, then TK GEMM
    from transformer_engine.pytorch import NVFP4Quantizer
    grp_quantizers = [
        NVFP4Quantizer(
            fp4_dtype=tex.DType.kFloat4E2M1, rowwise=True, columnwise=False,
            with_amax_reduction=False, amax_reduction_group=None,
            with_rht=False, with_post_rht_amax=False
        )
        for _ in range(num_groups)
    ]
    # Ensure optimize_for_gemm exists (older TE builds may lack it)
    for q in grp_quantizers:
        if not hasattr(q, 'optimize_for_gemm'):
            q.optimize_for_gemm = False

    # Pre-quantize weights with grouped quant (single kernel launch)
    grp_outs = [torch.zeros(M, N, dtype=torch.bfloat16, device=device) for N in N_dims]

    w_concat = torch.cat(w_list, dim=0)  # [sum(N), K]
    grp_w_splits = tex.split_quantize(w_concat, N_dims, grp_quantizers)
    grp_w_tk = [te_nvfp4_to_tk_format(w, N, K) for w, N in zip(grp_w_splits, N_dims)]

    # For activation: reuse tetk_x_* from TE→TK path (same activation, quantized once)

    def tetk_grp_gemms():
        for g in range(num_groups):
            w_fp4, w_sc, w_sc_global = grp_w_tk[g]
            nvfp4_gemm(tetk_x_fp4, tetk_x_sc, tetk_x_sg,
                       w_fp4, w_sc, w_sc_global, grp_outs[g])

    tetk_grp_ms = _time_fn(tetk_grp_gemms, steps, warmup)

    # ========== True Grouped: Concat Weights → 1 TK GEMM ==========
    # Instead of N separate GEMMs, concat weights → single large GEMM → slice output
    N_total = sum(N_dims)
    # Quantize concatenated weight with TK
    tk_wc_fp4, tk_wc_sc, tk_wc_sg = tk_quantize(w_concat)
    concat_out = torch.zeros(M, N_total, dtype=torch.bfloat16, device=device)

    def tk_concat_gemm():
        nvfp4_gemm(tk_x_fp4, tk_x_sc, tk_x_sc_global,
                   tk_wc_fp4, tk_wc_sc, tk_wc_sg, concat_out)

    concat_ms = _time_fn(tk_concat_gemm, steps, warmup)

    # ========== Grouped TK GEMM: 1 call with per-tile B_sg ==========
    # Pre-compute per-tile B_sg tensor ONCE (on GPU) — no host work in hot loop
    grp_wc_fp4 = torch.cat([w[0] for w in grp_w_tk], dim=0)
    grp_wc_sc = torch.cat([w[1] for w in grp_w_tk], dim=0)
    Nb = 256  # C::Nb
    b_sg_per_tile_list = []
    for gi, N in enumerate(N_dims):
        n_tiles = N // Nb
        b_sg_per_tile_list.extend([grp_w_tk[gi][2].item()] * n_tiles)
    grp_b_sg_per_tile = torch.tensor(b_sg_per_tile_list, dtype=torch.float32, device=device)
    grp_gemm_out = torch.zeros(M, N_total, dtype=torch.bfloat16, device=device)

    def tk_grouped_gemm():
        nvfp4_grouped_gemm(tetk_x_fp4, tetk_x_sc, tetk_x_sg,
                           grp_wc_fp4, grp_wc_sc, grp_b_sg_per_tile,
                           grp_gemm_out)

    grouped_ms = _time_fn(tk_grouped_gemm, steps, warmup)

    return te_ms, cutlass_ms, tk_ms, tetk_ms, tetk_grp_ms, concat_ms, grouped_ms, total_flops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=100)
    args = parser.parse_args()

    configs = [
        # Llama 8B QKV: dim=4096, n_heads=32, n_kv=8, head_dim=128
        {"label": "Llama8B  M=1024",  "M": 1024,  "K": 4096,  "N": [4096, 1024, 1024]},
        {"label": "Llama8B  M=2048",  "M": 2048,  "K": 4096,  "N": [4096, 1024, 1024]},
        {"label": "Llama8B  M=4096",  "M": 4096,  "K": 4096,  "N": [4096, 1024, 1024]},
        {"label": "Llama8B  M=8192",  "M": 8192,  "K": 4096,  "N": [4096, 1024, 1024]},
        # Llama 70B QKV: dim=8192, n_heads=64, n_kv=8, head_dim=128
        {"label": "Llama70B M=1024",  "M": 1024,  "K": 8192,  "N": [8192, 1024, 1024]},
        {"label": "Llama70B M=2048",  "M": 2048,  "K": 8192,  "N": [8192, 1024, 1024]},
        {"label": "Llama70B M=4096",  "M": 4096,  "K": 8192,  "N": [8192, 1024, 1024]},
        {"label": "Llama70B M=8192",  "M": 8192,  "K": 8192,  "N": [8192, 1024, 1024]},
        # FFN-like
        {"label": "FFN 8B   M=2048",  "M": 2048,  "K": 4096,  "N": [14336, 14336]},
        {"label": "FFN 8B   M=4096",  "M": 4096,  "K": 4096,  "N": [14336, 14336]},
        # Square
        {"label": "Square   4096",    "M": 4096,  "K": 4096,  "N": [4096]},
        {"label": "Square   8192",    "M": 8192,  "K": 8192,  "N": [8192]},
    ]

    cutlass_ext = _get_cutlass_gemm_ext()

    print("=" * 200)
    print("  Grouped NVFP4 GEMM Benchmark — 6-way: TE | CUTLASS | N×TK | TE→TK | GrpQ→TK | 1×TK(concat)")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  Steps={args.steps} Warmup={args.warmup}")
    print("=" * 200)
    print()
    hdr = (f"{'Config':<22} | {'M':>5} {'K':>5} {'N':>20} {'Grp':>3}"
           f" | {'TE':>8} {'CUT':>8} {'N×TK':>8} {'TE→TK':>8} {'GrpQ':>8} {'1×TK':>8} {'GrpTK':>8}"
           f" | {'N×TK/TE':>8} {'GrpTK/TE':>9} {'GrpTK/N×TK':>11}")
    print(hdr)
    print("-" * 200)

    for cfg in configs:
        M, K, N_dims = cfg["M"], cfg["K"], cfg["N"]
        try:
            te_ms, cut_ms, tk_ms, tetk_ms, grp_ms, concat_ms, grouped_ms, flops = bench_one_config(
                M, K, N_dims, args.steps, args.warmup, cutlass_ext)

            cut_str = f"{cut_ms:>8.4f}" if cut_ms else f"{'N/A':>8}"
            ntk_ratio = f"{te_ms / tk_ms:>8.2f}x"
            grptk_te_ratio = f"{te_ms / grouped_ms:>9.2f}x"
            grptk_ntk_ratio = f"{tk_ms / grouped_ms:>11.2f}x"

            N_str = "+".join(str(n) for n in N_dims)
            print(f"{cfg['label']:<22} | {M:>5} {K:>5} {N_str:>20} {len(N_dims):>3}"
                  f" | {te_ms:>8.4f} {cut_str} {tk_ms:>8.4f} {tetk_ms:>8.4f} {grp_ms:>8.4f} {concat_ms:>8.4f} {grouped_ms:>8.4f}"
                  f" | {ntk_ratio} {grptk_te_ratio} {grptk_ntk_ratio}")
        except Exception as e:
            print(f"{cfg['label']:<22} | ERROR: {e}")

    print()
    print("Legend:")
    print("  N×TK/TE    = TE_time / N×TK_time         (>1x = N separate TK GEMMs faster than TE)")
    print("  GrpTK/TE   = TE_time / GrpTK_time        (>1x = grouped TK GEMM faster than TE)")
    print("  GrpTK/N×TK = N×TK_time / GrpTK_time      (>1x = grouped TK faster than N separate)")
    print("  GEMM-only: quant is NOT included in timing")


if __name__ == "__main__":
    main()
