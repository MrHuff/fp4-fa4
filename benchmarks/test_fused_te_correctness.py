"""
test_fused_te_correctness.py — Validate fused kernel output matches TE's NVFP4 format

Strategy: Dequantize both TE and fused outputs, compare the float values.

TE NVFP4Tensor internals:
  _rowwise_data:        packed FP4 bytes [M, K/2]
  _rowwise_scale_inv:   FP8 e4m3 block scales [padded]
  _amax_rowwise:        global amax scalar
  global_scale = _amax_rowwise / (6.0 * 448.0)

Usage:
    python3 benchmarks/test_fused_te_correctness.py
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# TE must be imported FIRST (loads C++ extension)
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import NVFP4Quantizer

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


def dequant_raw_fp4(fp4_bytes, scales_fp8, global_scale, m, k):
    """Dequantize raw FP4 bytes + FP8 block scales + global scale.

    Args:
        fp4_bytes: [M, K/2] uint8 packed FP4
        scales_fp8: [M, K/16] uint8 (interpreted as fp8 e4m3)
        global_scale: float scalar
        m, k: dimensions

    Returns:
        [M, K] float32 tensor
    """
    # FP4 E2M1 lookup table
    fp4_vals = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        device=fp4_bytes.device, dtype=torch.float32
    )

    # Unpack FP4: each byte → 2 nibbles
    data = fp4_bytes.view(torch.uint8).to(torch.int32)
    lo = data & 0x0F
    hi = data >> 4
    unpacked = torch.stack((lo, hi), dim=-1).reshape(m, k)  # [M, K]
    float_vals = fp4_vals[unpacked]  # [M, K] float32

    # Block scales: interpret uint8 as fp8 e4m3
    block_scales = scales_fp8.view(torch.float8_e4m3fn).to(torch.float32)
    block_scales = block_scales[:m, :k//16]  # trim padding

    # Apply scales: each group of 16 elements shares one block scale
    block_data = float_vals.view(-1, 16)  # [M*K/16, 16]
    block_data = block_data * global_scale * block_scales.reshape(-1, 1)

    return block_data.view(m, k)


def test_correctness(m=64, k=256, seed=42, verbose=True):
    """Compare fused kernel dequantized output vs TE dequantized output."""
    torch.manual_seed(seed)
    device = 'cuda'

    x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    w = torch.ones(k, device=device, dtype=torch.bfloat16)

    # ====== Reference: compute expected output in float ======
    x_f = x.float()
    rms = torch.sqrt(torch.mean(x_f ** 2, dim=-1, keepdim=True) + 1e-5)
    x_normed = x_f / rms
    x_act = F.silu(x_normed)  # SiLU(RMSNorm(x))

    # ====== TE Pipeline ======
    rms_module = nn.RMSNorm(k, eps=1e-5, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        rms_module.weight.fill_(1.0)
    h_te = rms_module(x)
    h_te = F.silu(h_te)

    te_dtype = tex.DType.kFloat4E2M1
    quantizer = NVFP4Quantizer(fp4_dtype=te_dtype, rowwise=True, columnwise=False)
    x_q = quantizer.make_empty((m, k), dtype=torch.bfloat16, device=device)
    quantizer.update_quantized(h_te, x_q)

    # Dequantize TE
    te_deq = x_q.dequantize(dtype=torch.float32)

    # ====== Fused Kernel ======
    fp4_data, scales, global_scale_tensor, inv_rms = fused_te.forward_full(
        x, w, 1e-5, 0, 0  # RMSNorm + SiLU
    )

    global_scale = global_scale_tensor.item()

    # Dequantize fused output
    fused_deq = dequant_raw_fp4(fp4_data, scales, global_scale, m, k)

    # ====== Compare ======
    if verbose:
        print(f"\n{'='*70}")
        print(f"  Correctness Test: M={m}, K={k}")
        print(f"{'='*70}")

    # 1. inv_rms check
    ref_inv_rms = (1.0 / rms.squeeze(-1))
    fused_inv_rms_f = inv_rms.float()
    inv_rms_err = torch.abs(ref_inv_rms.cpu() - fused_inv_rms_f.cpu()).max().item()

    # 2. Reference vs TE dequantized  (quantization error)
    ref_vs_te = torch.abs(x_act.cpu() - te_deq.cpu())

    # 3. Reference vs Fused dequantized (quantization error)
    ref_vs_fused = torch.abs(x_act.cpu() - fused_deq.cpu())

    # 4. TE vs Fused (format compatibility)
    te_vs_fused = torch.abs(te_deq.cpu() - fused_deq.cpu())

    if verbose:
        print(f"\n  Global Scale:")
        print(f"    TE amax:         {x_q._amax_rowwise.item():.6f}")
        print(f"    TE global_scale: {x_q._amax_rowwise.item() / (6*448):.6f}")
        print(f"    Fused:           {global_scale:.6f}")

        print(f"\n  Inv RMS max error: {inv_rms_err:.2e}")

        print(f"\n  Quantization Error (vs float reference):")
        print(f"    TE:    mean={ref_vs_te.mean():.4f}, max={ref_vs_te.max():.4f}")
        print(f"    Fused: mean={ref_vs_fused.mean():.4f}, max={ref_vs_fused.max():.4f}")

        print(f"\n  TE vs Fused (format match):")
        print(f"    mean={te_vs_fused.mean():.4f}, max={te_vs_fused.max():.4f}")
        print(f"    cosine_sim={F.cosine_similarity(te_deq.cpu().flatten().unsqueeze(0), fused_deq.cpu().flatten().unsqueeze(0)).item():.6f}")

    # 5. Check FP4 byte match
    te_fp4 = x_q._rowwise_data.cpu()
    fused_fp4 = fp4_data.cpu()

    if te_fp4.shape == fused_fp4.shape:
        byte_match = (te_fp4.view(torch.uint8) == fused_fp4.view(torch.uint8)).float().mean().item()
        if verbose:
            print(f"\n  FP4 byte match:   {byte_match*100:.1f}%")
    else:
        if verbose:
            print(f"\n  FP4 shape mismatch: TE={te_fp4.shape} vs Fused={fused_fp4.shape}")
        byte_match = 0.0

    # 6. Scale byte match
    te_scales = x_q._rowwise_scale_inv.cpu()
    fused_scales_cpu = scales.cpu()
    if verbose:
        print(f"  TE scale shape:   {te_scales.shape} dtype={te_scales.dtype}")
        print(f"  Fused scale shape: {fused_scales_cpu.shape} dtype={fused_scales_cpu.dtype}")

    # Summary
    cos_sim = F.cosine_similarity(
        te_deq.cpu().flatten().unsqueeze(0),
        fused_deq.cpu().flatten().unsqueeze(0)
    ).item()

    passed = cos_sim > 0.99
    status = "✅ PASS" if passed else "❌ FAIL"
    if verbose:
        print(f"\n  {status} (cosine_sim={cos_sim:.6f})")

    return passed, cos_sim


if __name__ == "__main__":
    results = []
    for m, k in [(64, 256), (256, 4096), (1024, 8192)]:
        passed, cos = test_correctness(m, k)
        results.append((m, k, passed, cos))

    print(f"\n{'='*70}")
    print("  Summary")
    print(f"{'='*70}")
    for m, k, passed, cos in results:
        status = "✅" if passed else "❌"
        print(f"  {status} M={m:>5}, K={k:>5}: cosine_sim={cos:.6f}")
