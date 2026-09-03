#!/usr/bin/env python3
"""
Deep-dive diagnostic: WHY does SA quant differ from TE quant?

Root cause hypothesis: amax computation differs.
  - TE uses bf16-domain max(__hmax, __habs) → converts to float only at final reduction
  - SA uses float-domain max(fabsf(__bfloat162float(x))) throughout
  
These CAN produce different results because:
  bf16 abs(x) operates on the bf16 representation (just clear sign bit),
  while float abs(bf16-to-float(x)) first promotes to float then takes abs.
  For normal values these are the same, but the intermediate max comparisons
  in bf16 may lose precision vs float comparisons.

Test plan:
  1. Compare amax values (TE vs SA) for shapes that show mismatches
  2. Feed TE's amax into SA's quantize kernel → do mismatches vanish?
  3. Feed SA's amax into TE's quantize kernel → do mismatches appear?
"""
import sys, os
os.environ['NVTE_NVFP4_DISABLE_RHT'] = '1'
os.environ['NVTE_NVFP4_DISABLE_2D_QUANTIZATION'] = '1'
os.environ['NVTE_NVFP4_ENCODE_CENTRIC'] = '0'
os.environ['NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING'] = '1'
os.environ['NVTE_CUSTOM_QUANT'] = '1'
os.environ['USE_TK_GEMM'] = '1'
os.environ['FUSED_TE_QUANT'] = '0'
sys.path.insert(0, '/workspace/low-bits-training')
sys.path.insert(0, '/workspace/fp4_matmul/TK_quantisation/nvfp4')

import torch
import _tk_quant as tk_standalone

# Load TE quant ext
sys.path.insert(0, '/root/.cache/torch_extensions/py312_cu130/fp4_quantize_ext')
import fp4_quantize_ext as te_ext

print("=" * 100)
print("  Deep-dive: Amax + Quantization Parity Diagnostic")
print("=" * 100)

shapes = [(256, 512), (2048, 2048), (4096, 2048), (65536, 2048)]

for M, K in shapes:
    torch.manual_seed(42)
    inp = torch.randn(M, K, dtype=torch.bfloat16, device='cuda') * 0.01

    # ---- TE path: get amax ----
    te_result = te_ext.fast_nvfp4_quantize_v2(inp, False, True, True)
    te_fp4 = te_result[0].view(torch.uint8)  # (M, K/2)
    te_amax = te_result[4].item()  # scalar float
    te_sg = te_result[6].item()    # scalar float

    # ---- SA path: get amax ----
    sa_result = tk_standalone.tk_quantize_for_gemm(inp, False)
    sa_fp4 = sa_result[0].view(torch.uint8)  # (M, K/2)
    # SA doesn't directly expose amax, but we can get it from sg: amax = sg * 2688
    sa_sg = sa_result[4].item()
    sa_amax = sa_sg * 2688.0

    # ---- Compare amax ----
    amax_match = (te_amax == sa_amax)
    amax_diff = abs(te_amax - sa_amax)

    # ---- Compare fp4 data ----
    fp4_match = torch.equal(te_fp4.reshape(-1), sa_fp4.reshape(-1))
    n_mismatch = (te_fp4.reshape(-1).int() != sa_fp4.reshape(-1).int()).sum().item()
    total = te_fp4.numel()

    # ---- Ground truth amax (computed on CPU in float64) ----
    gt_amax = inp.float().abs().max().item()

    print(f"\n--- Shape ({M}, {K}) ---")
    print(f"  Ground truth amax (f64):  {gt_amax:.10e}")
    print(f"  TE amax:                  {te_amax:.10e}")
    print(f"  SA amax:                  {sa_amax:.10e}")
    print(f"  TE sg:                    {te_sg:.10e}")
    print(f"  SA sg:                    {sa_sg:.10e}")
    print(f"  Amax match:               {amax_match} (diff = {amax_diff:.10e})")
    print(f"  FP4 match:                {fp4_match} ({n_mismatch}/{total} mismatches)")

    if not amax_match:
        # Hex comparison of amax
        import struct
        te_hex = struct.pack('f', te_amax).hex()
        sa_hex = struct.pack('f', float(sa_amax)).hex()
        gt_hex = struct.pack('f', float(gt_amax)).hex()
        print(f"  TE amax hex: {te_hex}")
        print(f"  SA amax hex: {sa_hex}")
        print(f"  GT amax hex: {gt_hex}")

    # ---- Test: feed TE's amax into SA's kernel ----
    if not fp4_match and not amax_match:
        # Use the raw quantize with TE's amax
        te_amax_tensor = torch.tensor([te_amax], dtype=torch.float32, device='cuda')
        sa_with_te_amax = tk_standalone.tk_quantize_transpose(inp, te_amax_tensor, te_amax_tensor, False)
        sa_te_fp4 = sa_with_te_amax[0].view(torch.uint8).reshape(-1)
        n_fixed = (te_fp4.reshape(-1).int() != sa_te_fp4.int()).sum().item()
        print(f"\n  ** SA with TE's amax: {n_fixed}/{total} mismatches (was {n_mismatch})")
        if n_fixed == 0:
            print(f"  ** ✓ CONFIRMED: amax difference is the sole cause of FP4 mismatches!")
        elif n_fixed < n_mismatch:
            print(f"  ** Partial fix: reduced from {n_mismatch} to {n_fixed}")
        else:
            print(f"  ** No improvement — amax is NOT the cause")

print("\n" + "=" * 100)
print("  Conclusion")
print("=" * 100)
