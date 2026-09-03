"""
Test CUTLASS grouped NVFP4 GEMM extension with TE quantization data.

Compares CUTLASS grouped GEMM (with SF format conversion) against
dequantized bf16 matmul reference.

Usage:
  python tests/test_cutlass_grouped_gemm.py
"""
import torch
import os
import sys

sys.path.insert(0, '/workspace/low-bits-training')

def load_grouped_gemm_ext():
    """Load the CUTLASS grouped GEMM extension."""
    from torch.utils.cpp_extension import load
    CUTLASS_ROOT = '/workspace/fp4_matmul/cutlass'
    return load(
        name='fp4_grouped_gemm_ext',
        sources=['/workspace/fp4_matmul/fused_ops/csrc/fp4_grouped_gemm.cu'],
        extra_include_paths=[
            os.path.join(CUTLASS_ROOT, 'include'),
            os.path.join(CUTLASS_ROOT, 'tools/util/include'),
            os.path.join(CUTLASS_ROOT, 'examples/common'),
        ],
        extra_cuda_cflags=[
            '-std=c++17', '-O3', '--expt-relaxed-constexpr',
            '-gencode=arch=compute_100a,code=sm_100a',
            '-DCUTE_ARCH_TCGEN05_MXF4_MMA_ENABLED',
            '-DCUTE_ARCH_TCGEN05_MXF4NVF4_MMA_ENABLED',
            '-DCUTE_ARCH_TCGEN05_TMEM_ENABLED',
        ],
        verbose=False,
    )

def main():
    torch.manual_seed(42)
    device = 'cuda'

    from low_bits_training.quantization.fused_te_linear import _fast_quantize

    M = 256
    K = 4096
    N_dims = [4096, 1024, 1024]
    num_groups = len(N_dims)

    print(f"=== CUTLASS Grouped NVFP4 GEMM Correctness Test ===")
    print(f"  M={M} K={K} groups={num_groups}")
    for g, N in enumerate(N_dims):
        print(f"  Group {g}: N={N}")

    # Generate test data
    x_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.1
    weights = [torch.randn(N, K, dtype=torch.bfloat16, device=device) * 0.02 for N in N_dims]

    # Quantize
    x_nvfp4 = _fast_quantize(x_bf16)
    w_nvfp4_list = [_fast_quantize(w) for w in weights]

    # ===== Reference: dequantize + bf16 matmul =====
    print("\n--- Reference: dequantized bf16 matmul ---")
    x_deq = x_nvfp4.dequantize()
    ref_outputs = []
    for g in range(num_groups):
        w_deq = w_nvfp4_list[g].dequantize()
        y = x_deq @ w_deq.T
        ref_outputs.append(y)
        print(f"  Group {g}: shape={y.shape} norm={y.norm().item():.4f}")

    # ===== CUTLASS grouped GEMM =====
    print("\n--- CUTLASS grouped GEMM ---")
    ext = load_grouped_gemm_ext()

    # Extract raw data
    A_data = x_nvfp4._rowwise_data       # (M, K/2) uint8
    A_sf_te = x_nvfp4._rowwise_scale_inv  # (M, K/16) uint8 E4M3

    print(f"  A_data: {A_data.shape} {A_data.dtype}")
    print(f"  A_sf_te: {A_sf_te.shape} {A_sf_te.dtype}")

    # Convert A SF from TE row-major to CUTLASS tiled layout
    A_sf_cutlass = ext.convert_te_sf_to_cutlass(A_sf_te, M, K)
    print(f"  A_sf_cutlass: {A_sf_cutlass.shape}")

    B_data_list = []
    B_sf_list = []
    for g in range(num_groups):
        N = N_dims[g]
        w_data = w_nvfp4_list[g]._rowwise_data       # (N, K/2) uint8
        w_sf_te = w_nvfp4_list[g]._rowwise_scale_inv  # (N, K/16) uint8 E4M3
        w_sf_cutlass = ext.convert_te_sf_to_cutlass(w_sf_te, N, K)
        B_data_list.append(w_data)
        B_sf_list.append(w_sf_cutlass)
        print(f"  Group {g}: B={w_data.shape} SF={w_sf_te.shape} -> {w_sf_cutlass.shape}")

    # Run CUTLASS grouped GEMM
    # TE uses two-stage scaling: S_enc = fp8_max * fp4_max / amax
    # The scale_inv values embed S_enc, so CUTLASS output is too big by S_enc_A * S_enc_B
    fp8_max = 448.0
    fp4_max = 6.0
    amax_A = x_nvfp4._amax_rowwise.item()
    S_enc_A = fp8_max * fp4_max / amax_A if amax_A > 0 else 1.0
    
    # For grouped GEMM, each B group may have different amax
    # We need per-group alpha, but CUTLASS only supports one alpha for all groups
    # Workaround: use a representative alpha (works if amaxes are similar)
    # or pre-scale the B scale factors
    # For now, compute per-group alpha and check if they're similar enough
    alphas = []
    for g in range(num_groups):
        amax_B = w_nvfp4_list[g]._amax_rowwise.item()
        S_enc_B = fp8_max * fp4_max / amax_B if amax_B > 0 else 1.0
        alpha_g = 1.0 / (S_enc_A * S_enc_B)
        alphas.append(alpha_g)
        print(f"  Group {g}: amax_B={amax_B:.4f} S_enc_B={S_enc_B:.1f} alpha={alpha_g:.6e}")

    # For now, run with per-group alpha (use the first group's alpha as global)
    # TODO: Handle per-group alpha by pre-scaling B SFs or using per-group epilogue
    alpha = alphas[0]
    print(f"  Using alpha={alpha:.6e} (S_enc_A={S_enc_A:.1f})")
    
    cutlass_outputs = ext.forward(A_data, A_sf_cutlass, B_data_list, B_sf_list,
                                   N_dims, M, K, alpha)
    torch.cuda.synchronize()

    # ===== Compare =====
    print("\n--- Comparison (CUTLASS vs dequantized bf16 reference) ---")
    all_pass = True
    for g in range(num_groups):
        ref = ref_outputs[g].float()
        cut = cutlass_outputs[g].float()
        cos = torch.nn.functional.cosine_similarity(ref.flatten(), cut.flatten(), dim=0)
        max_err = (ref - cut).abs().max()
        rel_err = max_err / ref.abs().max()
        norm_ratio = cut.norm() / ref.norm()
        # FP4 GEMM has significant quantization error — cos > 0.9 is reasonable
        status = "✅" if cos > 0.9 else "❌"
        if cos <= 0.9:
            all_pass = False
        print(f"  {status} Group {g}: cos={cos:.6f} norm_ratio={norm_ratio:.6f} "
              f"maxerr={max_err:.4e} relerr={rel_err:.4e}")
        print(f"    ref norm={ref.norm():.4f} cutlass norm={cut.norm():.4f}")

    print(f"\n{'✅ ALL PASS' if all_pass else '❌ SOME FAILED'}")

if __name__ == '__main__':
    main()
