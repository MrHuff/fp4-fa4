#!/usr/bin/env python3
"""Test v3 fused amax kernels (all 4 functions) vs v2 reference."""

import sys
sys.path.insert(0, "/workspace/fp4_matmul/TK_quantisation/nvfp4_v2")
sys.path.insert(0, "/workspace/fp4_matmul/TK_quantisation/nvfp4_v3")

import torch
import importlib.util

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

print("Loading v2 …")
v2 = load_module("_tk_quant", "/workspace/fp4_matmul/TK_quantisation/nvfp4_v2/_tk_quant.cpython-312-aarch64-linux-gnu.so")
print("Loading v3 …")
v3 = load_module("_tk_quant_v3", "/workspace/fp4_matmul/TK_quantisation/nvfp4_v3/_tk_quant_v3.cpython-312-aarch64-linux-gnu.so")

device = "cuda"
torch.manual_seed(42)

def compare_tensors(name, a, b, tolerate_fp4=True):
    """Compare two tensors, return True if match."""
    a_u8 = a.contiguous().view(torch.uint8)
    b_u8 = b.contiguous().view(torch.uint8)
    if a_u8.shape != b_u8.shape:
        print(f"  {name}: SHAPE MISMATCH {a_u8.shape} vs {b_u8.shape}")
        return False
    match = (a_u8 == b_u8).all().item()
    if not match:
        diff = (a_u8 != b_u8).sum().item()
        total = a_u8.numel()
        print(f"  {name}: {diff}/{total} bytes differ ({100*diff/total:.2f}%)")
        return False
    print(f"  {name}: ✓ bitwise match")
    return True

# ───────────────────────────────────────────────────
# Test 1: tk_quantize_for_gemm
# ───────────────────────────────────────────────────
print("\n" + "="*60)
print("TEST 1: tk_quantize_for_gemm")
print("="*60)

shapes_basic = [(128, 128), (256, 512), (1024, 1024), (2048, 4096)]
for M, K in shapes_basic:
    x = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    r2 = v2.tk_quantize_for_gemm(x, True)
    r3 = v3.tk_quantize_for_gemm(x, True)
    
    print(f"\n  Shape ({M}, {K}):")
    ok = True
    ok &= compare_tensors("row_fp4", r2[0], r3[0])
    ok &= compare_tensors("row_sc",  r2[1], r3[1])
    ok &= compare_tensors("col_fp4", r2[2], r3[2])
    ok &= compare_tensors("col_sc",  r2[3], r3[3])
    # sg values
    sg2 = r2[4].item()
    sg3 = r3[4].item()
    sg_ok = abs(sg2 - sg3) < 1e-6
    print(f"  sg: v2={sg2:.6f} v3={sg3:.6f} {'✓' if sg_ok else '✗'}")
    ok &= sg_ok
    print(f"  → {'PASS' if ok else 'FAIL'}")

# ───────────────────────────────────────────────────
# Test 2: tk_group_quantize_for_gemm (dim=0)
# ───────────────────────────────────────────────────
print("\n" + "="*60)
print("TEST 2: tk_group_quantize_for_gemm (dim=0)")
print("="*60)

shapes_grp = [
    (384, 512, [128, 128, 128]),
    (768, 1024, [256, 256, 256]),
    (1536, 2048, [512, 512, 512]),
]
for M, K, splits in shapes_grp:
    x = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    r2 = v2.tk_group_quantize_for_gemm(x, splits)
    r3 = v3.tk_group_quantize_for_gemm(x, splits)
    
    print(f"\n  Shape ({M}, {K}), splits={splits}:")
    ok = True
    # r = (fp4_row, sc_row, fwd_b_sg, [fp4_col], [sc_col], dgrad_b_sg, sg_cat, mega_buf)
    ok &= compare_tensors("fp4_row", r2[0], r3[0])
    ok &= compare_tensors("sc_row",  r2[1], r3[1])
    
    # sg_cat
    sg2 = r2[6]
    sg3 = r3[6]
    sg_ok = torch.allclose(sg2, sg3, atol=1e-5)
    print(f"  sg_cat: v2={sg2.tolist()} v3={sg3.tolist()} {'✓' if sg_ok else '✗'}")
    ok &= sg_ok
    
    # Per-split col outputs
    N = len(splits)
    for i in range(N):
        ok &= compare_tensors(f"col_fp4[{i}]", r2[3][i], r3[3][i])
        ok &= compare_tensors(f"col_sc[{i}]",  r2[4][i], r3[4][i])
    
    # fwd_b_sg, dgrad_b_sg
    ok &= compare_tensors("fwd_b_sg", r2[2], r3[2])
    ok &= compare_tensors("dgrad_b_sg", r2[5], r3[5])
    
    print(f"  → {'PASS' if ok else 'FAIL'}")

# ───────────────────────────────────────────────────
# Test 3: tk_group_quantize_dim1_for_gemm
# ───────────────────────────────────────────────────
print("\n" + "="*60)
print("TEST 3: tk_group_quantize_dim1_for_gemm (dim=1)")
print("="*60)

shapes_dim1 = [
    (256, 384, [128, 128, 128]),
    (512, 768, [256, 256, 256]),
    (1024, 1536, [512, 512, 512]),
]
for M, N, col_splits in shapes_dim1:
    x = torch.randn(M, N, device=device, dtype=torch.bfloat16)
    r2 = v2.tk_group_quantize_dim1_for_gemm(x, col_splits)
    r3 = v3.tk_group_quantize_dim1_for_gemm(x, col_splits)
    
    print(f"\n  Shape ({M}, {N}), col_splits={col_splits}:")
    ok = True
    # r = ([fp4_row_i], [sc_row_i], sg_per_group, [fp4_col_i], [sc_col_i])
    G = len(col_splits)
    
    sg2 = r2[2]
    sg3 = r3[2]
    sg_ok = torch.allclose(sg2, sg3, atol=1e-5)
    print(f"  sg_per_group: v2={sg2.tolist()} v3={sg3.tolist()} {'✓' if sg_ok else '✗'}")
    ok &= sg_ok
    
    for g in range(G):
        ok &= compare_tensors(f"fp4_row[{g}]", r2[0][g], r3[0][g])
        ok &= compare_tensors(f"sc_row[{g}]",  r2[1][g], r3[1][g])
        ok &= compare_tensors(f"fp4_col[{g}]", r2[3][g], r3[3][g])
        ok &= compare_tensors(f"sc_col[{g}]",  r2[4][g], r3[4][g])
    
    print(f"  → {'PASS' if ok else 'FAIL'}")

print("\n" + "="*60)
print("ALL TESTS COMPLETE")
print("="*60)
