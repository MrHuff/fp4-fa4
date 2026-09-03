#!/usr/bin/env python3
"""
Integration test: verify TK standalone quant works in the training pipeline.

Tests:
1. Single quant via _fast_quantize (USE_TK_QUANT=1)
2. Grouped quant via tk_group_quantize_for_gemm
3. Parity between TK and TE quantization paths
"""
import os
os.environ['USE_TK_QUANT'] = '1'
os.environ['USE_TK_GEMM'] = '1'
os.environ['NVTE_CUSTOM_QUANT'] = '1'
os.environ['FUSED_TE_QUANT'] = '0'

import sys
sys.path.insert(0, '/workspace/low-bits-training')
sys.path.insert(0, '/workspace/fp4_matmul/TK_quantisation/nvfp4')

import torch

print("=" * 80)
print("  Integration Test: TK Standalone Quant in Pipeline")
print("=" * 80)

# ── Test 1: Single quant via _fast_quantize ──
print("\n--- Test 1: _fast_quantize with USE_TK_QUANT=1 ---")
from low_bits_training.quantization.fused_te_linear import (
    _fast_quantize, use_tk_quant, _TKQuantized
)
assert use_tk_quant(), "USE_TK_QUANT should be 1"

torch.manual_seed(42)
x = torch.randn(2048, 2048, dtype=torch.bfloat16, device='cuda') * 0.01

# TK quant path (USE_TK_QUANT=1, tk_swizzle=True)
result_tk = _fast_quantize(x, tk_swizzle=True)
assert isinstance(result_tk, _TKQuantized), f"Expected _TKQuantized, got {type(result_tk)}"
fp4_tk, sc_tk, sg_tk = result_tk._tk_row
fp4_t_tk, sc_t_tk, _ = result_tk._tk_col
print(f"  TK quant: fp4={fp4_tk.shape} sc={sc_tk.shape} sg={sg_tk.shape}")
print(f"  TK col:   fp4_t={fp4_t_tk.shape} sc_t={sc_t_tk.shape}")

# TE quant path (force tk_swizzle=False to go through TE)
os.environ['USE_TK_QUANT'] = '0'  # temporarily disable
result_te = _fast_quantize(x, tk_swizzle=True)
os.environ['USE_TK_QUANT'] = '1'  # re-enable
assert isinstance(result_te, _TKQuantized), f"Expected _TKQuantized, got {type(result_te)}"
fp4_te, sc_te, sg_te = result_te._tk_row

# Compare
fp4_match = torch.equal(fp4_tk.view(torch.uint8), fp4_te.view(torch.uint8))
sc_match = torch.equal(sc_tk.view(torch.uint8), sc_te.view(torch.uint8))
sg_diff = abs(sg_tk.item() - sg_te.item())
print(f"  FP4 match: {fp4_match}")
print(f"  SC match:  {sc_match}")
print(f"  SG diff:   {sg_diff:.2e}")
print(f"  ✅ Single quant works!" if fp4_match and sc_match else "  ❌ Mismatch!")

# ── Test 2: Grouped quant via tk_group_quantize_for_gemm ──
print("\n--- Test 2: tk_group_quantize_for_gemm ---")
import _tk_quant as tk_q

torch.manual_seed(42)
w = torch.randn(6144, 2048, dtype=torch.bfloat16, device='cuda') * 0.01
splits = [2048, 2048, 2048]  # Q, K, V

result = tk_q.tk_group_quantize_for_gemm(w, splits)
fp4_row, sc_row, fwd_b_sg, fp4_cols, sc_cols, dgrad_b_sg, sg_cat, mega = result

print(f"  fp4_row:    {fp4_row.shape} {fp4_row.dtype}")
print(f"  sc_row:     {sc_row.shape} {sc_row.dtype}")
print(f"  fwd_b_sg:   {fwd_b_sg.shape} ({fwd_b_sg.tolist()[:6]}...)")
print(f"  dgrad_b_sg: {dgrad_b_sg.shape}")
print(f"  sg_cat:     {sg_cat.shape} ({sg_cat.tolist()})")
print(f"  # col splits: {len(fp4_cols)}")
for i, (fp4_c, sc_c) in enumerate(zip(fp4_cols, sc_cols)):
    print(f"    split[{i}]: fp4_c={fp4_c.shape} sc_c={sc_c.shape}")

# Verify b_sg values match sg_cat
print(f"\n  Verifying b_sg consistency...")
# fwd_b_sg should be [sg0]*n0, [sg1]*n1, [sg2]*n2
Nb = 256
off = 0
ok = True
for i, s in enumerate(splits):
    tiles = s // Nb
    expected_sg = sg_cat[i].item()
    actual = fwd_b_sg[off:off+tiles]
    if not torch.allclose(actual, torch.full_like(actual, expected_sg)):
        print(f"    ❌ fwd_b_sg split {i}: expected {expected_sg}, got {actual.tolist()[:3]}...")
        ok = False
    off += tiles
if ok:
    print(f"  ✅ fwd_b_sg consistent with sg_cat")

# Check dgrad_b_sg
dgrad_tiles_per = 2048 // Nb
ok_d = True
for i in range(3):
    expected_sg = sg_cat[i].item()
    actual = dgrad_b_sg[i*dgrad_tiles_per:(i+1)*dgrad_tiles_per]
    if not torch.allclose(actual, torch.full_like(actual, expected_sg)):
        print(f"    ❌ dgrad_b_sg split {i}: expected {expected_sg}, got {actual.tolist()[:3]}...")
        ok_d = False
if ok_d:
    print(f"  ✅ dgrad_b_sg consistent with sg_cat")

# ── Test 3: Compare grouped quant between TK and TE ──
print("\n--- Test 3: Grouped quant parity (TK vs TE) ---")
from low_bits_training.quantization.fused_te_linear import _get_fp4_ext
fp4_ext = _get_fp4_ext()

te_result = fp4_ext.group_nvfp4_quantize_tk(w, splits)
te_fp4, te_sc, te_fwd, te_cols, te_sc_cols, te_dgrad, te_sg, te_mega = te_result

fp4_match_g = torch.equal(fp4_row.view(torch.uint8), te_fp4.view(torch.uint8))
sc_match_g = torch.equal(sc_row.view(torch.uint8), te_sc.view(torch.uint8))
sg_diff_g = (sg_cat - te_sg).abs().max().item()

print(f"  Row FP4 match:  {fp4_match_g}")
print(f"  Row SC match:   {sc_match_g}")
print(f"  SG max diff:    {sg_diff_g:.2e}")

if not fp4_match_g:
    n_mismatch = (fp4_row.view(torch.uint8).int() != te_fp4.view(torch.uint8).int()).sum().item()
    print(f"  FP4 mismatches: {n_mismatch}/{fp4_row.view(torch.uint8).numel()}")

for i in range(len(fp4_cols)):
    col_match = torch.equal(fp4_cols[i].view(torch.uint8), te_cols[i].view(torch.uint8))
    col_sc_match = torch.equal(sc_cols[i].view(torch.uint8), te_sc_cols[i].view(torch.uint8))
    print(f"  Col split[{i}]: fp4={col_match} sc={col_sc_match}")

print("\n" + "=" * 80)
print("  All tests complete!")
print("=" * 80)
