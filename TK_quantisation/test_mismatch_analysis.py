#!/usr/bin/env python3
"""Determinism check + mismatch analysis for TK vs TE parity."""
import sys, os, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nvfp4'))
import _C as tk
sys.path.insert(0, '/root/.cache/torch_extensions/py312_cu130/fp4_quantize_ext')
import fp4_quantize_ext as te

# Check 1: Is TK itself deterministic?
print("=== TK Self-Determinism ===")
for M, K in [(256, 512), (65536, 2048)]:
    torch.manual_seed(42)
    inp = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    amax = inp.abs().max().unsqueeze(0).float()
    r1, s1, c1, cs1 = tk.tk_quantize_transpose(inp, amax, amax, True)
    r2, s2, c2, cs2 = tk.tk_quantize_transpose(inp, amax, amax, True)
    r_match = torch.equal(r1, r2)
    c_match = torch.equal(c1, c2)
    print(f"  ({M},{K}): row_fp4 self-match={r_match}, col_fp4 self-match={c_match}")

# Check 2: Is TE itself deterministic?
print("\n=== TE Self-Determinism ===")
for M, K in [(256, 512), (65536, 2048)]:
    torch.manual_seed(42)
    inp = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    te1 = te.fast_nvfp4_quantize_v2(inp, False, True, True)
    te2 = te.fast_nvfp4_quantize_v2(inp, False, True, True)
    r_match = torch.equal(te1[0].view(torch.uint8), te2[0].view(torch.uint8))
    c_match = torch.equal(te1[2].view(torch.uint8), te2[2].view(torch.uint8))
    print(f"  ({M},{K}): row_fp4 self-match={r_match}, col_fp4 self-match={c_match}")

# Check 3: What does the mismatch look like?
print("\n=== Mismatch Pattern Analysis ===")
for M, K in [(256, 512), (65536, 2048)]:
    torch.manual_seed(42)
    inp = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    
    te_fp4, te_si, te_fp4_t, te_si_t, te_amax, _, _ = te.fast_nvfp4_quantize_v2(inp, False, True, True)
    amax_t = te_amax.reshape(1)
    tk_fp4, tk_sc, tk_fp4_t, tk_sc_t = tk.tk_quantize_transpose(inp, amax_t, amax_t, True)
    
    te_u8 = te_fp4.view(torch.uint8).reshape(M, K//2)
    tk_u8 = tk_fp4.view(torch.uint8).reshape(M, K//2)
    diff = (te_u8.int() - tk_u8.int()).abs()
    mismatch_mask = diff > 0
    n_mismatch = mismatch_mask.sum().item()

    if n_mismatch > 0:
        print(f"\n  ({M},{K}): {n_mismatch} mismatches")
        # Check if mismatches are in specific chunks
        for cy in range(min(M // 128, 4)):
            for cx in range(min(K // 128, 4)):
                chunk = diff[cy*128:(cy+1)*128, cx*64:(cx+1)*64]
                n = (chunk > 0).sum().item()
                if n > 0:
                    print(f"    Chunk[{cy},{cx}]: {n} mismatches")

        # Analyze byte-level differences
        te_mismatched = te_u8[mismatch_mask]
        tk_mismatched = tk_u8[mismatch_mask]
        diffs = (te_mismatched.int() - tk_mismatched.int())
        unique_diffs = diffs.unique()
        print(f"    Unique diff values: {unique_diffs.tolist()[:20]}")
        
        # Check nibble-level: are diffs always ±1 in one nibble?
        te_hi = te_mismatched >> 4
        te_lo = te_mismatched & 0xf
        tk_hi = tk_mismatched >> 4
        tk_lo = tk_mismatched & 0xf
        hi_diff = (te_hi.int() - tk_hi.int()).abs()
        lo_diff = (te_lo.int() - tk_lo.int()).abs()
        hi_only = ((hi_diff > 0) & (lo_diff == 0)).sum().item()
        lo_only = ((hi_diff == 0) & (lo_diff > 0)).sum().item()
        both = ((hi_diff > 0) & (lo_diff > 0)).sum().item()
        print(f"    Nibble analysis: hi_only={hi_only}, lo_only={lo_only}, both={both}")
        print(f"    Max hi_nibble_diff={hi_diff.max().item()}, max lo_nibble_diff={lo_diff.max().item()}")
    else:
        print(f"\n  ({M},{K}): PERFECT MATCH ✓")
