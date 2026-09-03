"""
test_fused_gemm_e2e.py — Prove fused kernel output works with cuBLASLt GEMM end-to-end

Compares:
  1. TE pipeline: RMSNorm → SiLU → tex.quantize → tex.generic_gemm  
  2. Fused → GEMM: fused(RMS+SiLU+quant) → construct NVFP4Tensor → tex.generic_gemm

Both should produce similar output.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# TE must be imported first
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import NVFP4Quantizer
from transformer_engine.pytorch.constants import TE_DType
from transformer_engine.pytorch.tensor import NVFP4Tensor

# Build fused kernel
print("Compiling fused kernel...")
from torch.utils.cpp_extension import load
CSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fused_ops', 'csrc')
fused_te = load(
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
print("Compiled.")


def test_fused_gemm(m=256, k=4096, n=4096, seed=42):
    """Feed fused output into cuBLASLt GEMM and compare with TE pipeline."""
    torch.manual_seed(seed)
    device = 'cuda'

    x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    w_gain = torch.ones(k, device=device, dtype=torch.bfloat16)
    W = torch.randn(n, k, device=device, dtype=torch.bfloat16)

    # Pre-quantize weights (common to both paths)
    te_dtype = tex.DType.kFloat4E2M1
    wq = NVFP4Quantizer(fp4_dtype=te_dtype, rowwise=True, columnwise=True)
    w_nvfp4 = wq.make_empty((n, k), dtype=torch.bfloat16, device=device)
    wq.update_quantized(W, w_nvfp4)

    out_te = torch.empty(m, n, device=device, dtype=torch.bfloat16)
    out_fused = torch.empty(m, n, device=device, dtype=torch.bfloat16)
    workspace = torch.empty(4, dtype=torch.uint8, device=device)
    out_dtype = TE_DType[torch.bfloat16]
    bias_dtype = TE_DType[torch.bfloat16]

    # ====== Path A: TE Separate Pipeline ======
    rms = nn.RMSNorm(k, eps=1e-5, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        rms.weight.fill_(1.0)

    h_te = rms(x)
    h_te = F.silu(h_te)

    xq_te = NVFP4Quantizer(fp4_dtype=te_dtype, rowwise=True, columnwise=True)
    x_nvfp4_te = xq_te.make_empty((m, k), dtype=torch.bfloat16, device=device)
    xq_te.update_quantized(h_te, x_nvfp4_te)

    tex.generic_gemm(
        w_nvfp4, True, x_nvfp4_te, False,
        out_te, None, out_dtype,
        None, bias_dtype,
        False, None, False, workspace,
        workspace.shape[0], False, False,
    )

    # ====== Path B: Fused → construct NVFP4Tensor → GEMM ======
    fp4_data, scales, global_scale, inv_rms = fused_te.forward_full(
        x, w_gain, 1e-5, 0, 0  # RMSNorm + SiLU
    )

    # Construct NVFP4Tensor from fused output
    # We need: rowwise_data, rowwise_scale_inv, amax_rowwise
    # amax = global_scale * 6.0 * 448.0 (inverse of compute_te_global_scale)
    amax = torch.tensor(global_scale.item() * 6.0 * 448.0, device=device, dtype=torch.float32)

    # Make an empty NVFP4Tensor and fill it with our data
    xq_fused = NVFP4Quantizer(fp4_dtype=te_dtype, rowwise=True, columnwise=True)
    x_nvfp4_fused = xq_fused.make_empty((m, k), dtype=torch.bfloat16, device=device)

    # Overwrite internals with our fused output
    x_nvfp4_fused._rowwise_data = fp4_data.view(x_nvfp4_fused._rowwise_data.shape)
    x_nvfp4_fused._rowwise_scale_inv = scales.view(x_nvfp4_fused._rowwise_scale_inv.shape)
    x_nvfp4_fused._amax_rowwise = amax

    tex.generic_gemm(
        w_nvfp4, True, x_nvfp4_fused, False,
        out_fused, None, out_dtype,
        None, bias_dtype,
        False, None, False, workspace,
        workspace.shape[0], False, False,
    )

    # ====== Compare ======
    # Also compute BF16 reference
    ref = F.silu(F.rms_norm(x.float(), (k,), None, eps=1e-5)) @ W.T.float()

    cos_te_ref = F.cosine_similarity(out_te.float().flatten().unsqueeze(0),
                                      ref.flatten().unsqueeze(0)).item()
    cos_fused_ref = F.cosine_similarity(out_fused.float().flatten().unsqueeze(0),
                                         ref.flatten().unsqueeze(0)).item()
    cos_te_fused = F.cosine_similarity(out_te.float().flatten().unsqueeze(0),
                                        out_fused.float().flatten().unsqueeze(0)).item()

    diff_te_fused = (out_te.float() - out_fused.float()).abs()
    rel_err = diff_te_fused / (out_te.float().abs() + 1e-8)

    print(f"\n{'='*60}")
    print(f"  E2E GEMM Test: M={m}, K={k}, N={n}")
    print(f"{'='*60}")
    print(f"  Cosine sim (TE vs float ref):    {cos_te_ref:.6f}")
    print(f"  Cosine sim (Fused vs float ref): {cos_fused_ref:.6f}")
    print(f"  Cosine sim (TE vs Fused):        {cos_te_fused:.6f}")
    print(f"  Abs diff (TE vs Fused): mean={diff_te_fused.mean():.4f}, max={diff_te_fused.max():.4f}")
    print(f"  Rel err  (TE vs Fused): mean={rel_err.mean():.4f}, max={rel_err.max():.4f}")

    passed = cos_te_fused > 0.99
    print(f"  {'✅ PASS' if passed else '❌ FAIL'}")
    return passed


if __name__ == "__main__":
    r1 = test_fused_gemm(m=256, k=4096, n=4096)
    r2 = test_fused_gemm(m=1024, k=8192, n=8192)
    r3 = test_fused_gemm(m=4096, k=8192, n=8192)

    print(f"\n{'='*60}")
    all_pass = all([r1, r2, r3])
    print(f"  Overall: {'✅ ALL PASS' if all_pass else '❌ SOME FAILED'}")
    print(f"{'='*60}")
