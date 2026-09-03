#!/usr/bin/env python3
"""
Deep-dive #2: Compare SCALE outputs between SA and TE paths.

Since amax is NOT the cause (proven), the difference must be in:
1. Scale buffer layout/padding — different effective scale_stride
2. TMA tensor map config — different global dims or box dims
3. Something in the kernel that behaves differently based on allocation

We test:
- Scale comparison (byte-level) between SA and TE
- FP4 comparison isolated to specific 128x128 chunks
- Whether the mismatch correlates with specific scale tile boundaries
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

sys.path.insert(0, '/root/.cache/torch_extensions/py312_cu130/fp4_quantize_ext')
import fp4_quantize_ext as te_ext

print("=" * 100)
print("  Deep-dive #2: Scale + FP4 Data Byte-Level Comparison")
print("=" * 100)

M, K = 2048, 2048  # Shape with mismatches
torch.manual_seed(42)
inp = torch.randn(M, K, dtype=torch.bfloat16, device='cuda') * 0.01

# TE path with tk_swizzle=True (to get same scale layout)
te_result = te_ext.fast_nvfp4_quantize_v2(inp, False, True, True)
te_fp4 = te_result[0].view(torch.uint8).reshape(M, K // 2)
te_sc = te_result[1]  # 3D swizzled (ntm, ntk, 512) fp8
te_amax = te_result[4]

# SA path with TE's amax
te_amax_scalar = te_amax.reshape(1)
sa_raw = tk_standalone.tk_quantize_transpose(inp, te_amax_scalar, te_amax_scalar, False)
sa_fp4 = sa_raw[0].reshape(M, K // 2)
sa_sc = sa_raw[1]  # flat (M, scale_stride) uint8

print(f"\nTE fp4 shape: {te_fp4.shape}, dtype: {te_fp4.dtype}")
print(f"TE scales shape: {te_sc.shape}, dtype: {te_sc.dtype}")
print(f"SA fp4 shape: {sa_fp4.shape}, dtype: {sa_fp4.dtype}")
print(f"SA scales shape: {sa_sc.shape}, dtype: {sa_sc.dtype}")

# Compare scales: reshape SA flat to match TE 3D
ntm = M // 128
ntk = K // 64
print(f"\nExpected scale tiles: ntm={ntm}, ntk={ntk}")
print(f"TE scale total bytes: {te_sc.view(torch.uint8).numel()}")
print(f"SA scale total bytes: {sa_sc.numel()}")

# Both should have same total bytes
te_sc_flat = te_sc.view(torch.uint8).reshape(-1)
sa_sc_flat = sa_sc.view(torch.uint8).reshape(-1)

# Take min length for comparison
min_len = min(te_sc_flat.numel(), sa_sc_flat.numel())
sc_match = torch.equal(te_sc_flat[:min_len], sa_sc_flat[:min_len])
sc_mismatch = (te_sc_flat[:min_len].int() != sa_sc_flat[:min_len].int()).sum().item()
print(f"\nScale comparison (first {min_len} bytes):")
print(f"  Match: {sc_match} ({sc_mismatch} mismatches)")

if sc_mismatch > 0:
    idx = (te_sc_flat[:min_len].int() != sa_sc_flat[:min_len].int()).nonzero(as_tuple=True)[0]
    print(f"  First 10 mismatch indices: {idx[:10].tolist()}")
    for i in idx[:5]:
        print(f"    byte[{i}]: TE=0x{te_sc_flat[i]:02x} SA=0x{sa_sc_flat[i]:02x}")

# Compare FP4 data
fp4_diff = (te_fp4.int() != sa_fp4.int())
n_mismatch = fp4_diff.sum().item()
print(f"\nFP4 comparison: {n_mismatch}/{te_fp4.numel()} mismatches")

# Analyze which 128x128 chunks have mismatches
if n_mismatch > 0:
    print("\n  Per-chunk mismatch counts (128x64 byte chunks):")
    for cy in range(M // 128):
        for cx in range(K // 128):
            # Each 128x128 logical chunk = 128 rows × 64 byte-columns
            chunk = fp4_diff[cy*128:(cy+1)*128, cx*64:(cx+1)*64]
            n = chunk.sum().item()
            if n > 0:
                print(f"    Chunk[{cy},{cx}]: {n} mismatches")
                # Show a few mismatched bytes
                mis_r, mis_c = torch.where(chunk)
                for j in range(min(3, len(mis_r))):
                    r, c = cy*128 + mis_r[j].item(), cx*64 + mis_c[j].item()
                    te_val = te_fp4[r, c].item()
                    sa_val = sa_fp4[r, c].item()
                    # Decode nibbles
                    te_hi, te_lo = te_val >> 4, te_val & 0xf
                    sa_hi, sa_lo = sa_val >> 4, sa_val & 0xf
                    print(f"      byte[{r},{c}]: TE=0x{te_val:02x} ({te_hi},{te_lo}) "
                          f"SA=0x{sa_val:02x} ({sa_hi},{sa_lo}) "
                          f"diff_hi={te_hi-sa_hi} diff_lo={te_lo-sa_lo}")

    # Now check: are mismatched positions correlated with block boundaries?
    mis_rows, mis_cols = torch.where(fp4_diff)
    # Each FP4 byte at (r, c) represents logical positions (r, 2*c) and (r, 2*c+1)
    # Scale block = 16 consecutive values → scale index = logical_col / 16
    # In byte terms: scale changes every 8 bytes
    byte_in_block = mis_cols % 8  # which byte within a 16-value FP4 block
    print(f"\n  Mismatch position within 16-value FP4 block (modulo 8 bytes):")
    for b in range(8):
        n = (byte_in_block == b).sum().item()
        if n > 0:
            print(f"    byte_pos {b}: {n} mismatches")

print("\n" + "=" * 100)

# Now test: use SA's GEMM-ready path for (2048,2048) and compare scales
sa_gemm = tk_standalone.tk_quantize_for_gemm(inp, False)
sa_gemm_sc = sa_gemm[1]  # (ntm, ntk, 512) fp8
sa_gemm_sc_flat = sa_gemm_sc.view(torch.uint8).reshape(-1)
print(f"\nSA gemm-ready scale shape: {sa_gemm_sc.shape}")
print(f"SA gemm-ready scale bytes: {sa_gemm_sc_flat.numel()}")

# Compare SA gemm scales vs TE scales
sc_cmp_len = min(te_sc_flat.numel(), sa_gemm_sc_flat.numel())
sc_gemm_match = torch.equal(te_sc_flat[:sc_cmp_len], sa_gemm_sc_flat[:sc_cmp_len])
print(f"SA gemm scales vs TE scales: match={sc_gemm_match}")
