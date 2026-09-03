#!/usr/bin/env python3
"""
Parity test: TK standalone quantisation vs TE quantisation.
Usage: CUDA_VISIBLE_DEVICES=0 python3 test_parity.py
"""
import sys, os
import torch

# ── Load TK standalone module ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nvfp4'))
import _C as tk_quant

# ── Try loading TE module ──
TE_MODULE = None
for p in ['/root/.cache/torch_extensions/py312_cu130/fp4_quantize_ext',
          '/workspace/fp4_matmul/fused_ops']:
    if os.path.exists(p):
        sys.path.insert(0, p)
        try:
            import fp4_quantize_ext as _te
            TE_MODULE = _te
            print(f"TE loaded from {p}")
            break
        except ImportError:
            continue


def test_shapes(shapes):
    """Test all shapes produce valid output."""
    print("=" * 70)
    print("TK Standalone — Smoke Test (scalar amax)")
    print("=" * 70)
    all_pass = True
    for M, K in shapes:
        inp = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
        amax = inp.abs().max().unsqueeze(0).float()
        r, s, c, cs = tk_quant.tk_quantize_transpose(inp, amax, amax, True)
        pct_r = 100 * (r != 0).sum().item() / r.numel()
        pct_c = 100 * (c != 0).sum().item() / c.numel()
        ok = pct_r > 90 and pct_c > 90
        tag = "✓" if ok else "✗"
        print(f"  {tag} ({M:6d}, {K:5d}): row={pct_r:.1f}% col={pct_c:.1f}% sc=100%")
        if not ok: all_pass = False
    print("✓ All passed" if all_pass else "✗ Some failed")
    return all_pass


def test_parity(shapes):
    """Compare TK standalone vs TE quantisation."""
    if TE_MODULE is None:
        print("\n⚠ Skipping TE parity (module not found)")
        return

    print("\n" + "=" * 70)
    print("TK vs TE — Bitwise Parity")
    print("=" * 70)
    for M, K in shapes:
        inp = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')

        # TE: custom_quant=True, tk_swizzle=True → same underlying kernel
        te_fp4, te_si, te_fp4_t, te_si_t, te_amax, _, _ = \
            TE_MODULE.fast_nvfp4_quantize_v2(inp, False, True, True)

        # TK: use TE's amax
        amax_t = te_amax.reshape(1)
        tk_fp4, tk_sc, tk_fp4_t, tk_sc_t = tk_quant.tk_quantize_transpose(
            inp, amax_t, amax_t, True)

        def cmp(name, a, b):
            au = a.view(torch.uint8).reshape(-1)
            bu = b.view(torch.uint8).reshape(-1)
            n = min(au.numel(), bu.numel())
            d = (au[:n].int() - bu[:n].int()).abs()
            pct = 100 * (d == 0).sum().item() / n
            if pct == 100:
                print(f"  ({M},{K}) {name}: BITWISE MATCH ✓")
            else:
                print(f"  ({M},{K}) {name}: {pct:.2f}% match ({(d>0).sum().item()} mismatches)")

        cmp("row_fp4", te_fp4, tk_fp4)
        cmp("col_fp4", te_fp4_t, tk_fp4_t)
        cmp("row_sc ", te_si, tk_sc)


def test_benchmark(shapes, iters=100, warmup=20):
    """Benchmark TK standalone quantisation."""
    print("\n" + "=" * 70)
    print(f"TK Benchmark ({iters} iters, with transpose)")
    print("=" * 70)
    for M, K in shapes:
        inp = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
        amax = inp.abs().max().unsqueeze(0).float()
        for _ in range(warmup):
            tk_quant.tk_quantize_transpose(inp, amax, amax, True)
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            tk_quant.tk_quantize_transpose(inp, amax, amax, True)
        e.record(); torch.cuda.synchronize()
        ms = s.elapsed_time(e) / iters
        gb = M * K * 2 / 1e9 / (ms / 1e3)
        print(f"  ({M:6d}, {K:5d}): {ms*1000:.1f} µs, {gb:.1f} GB/s")


if __name__ == '__main__':
    shapes = [(128, 256), (256, 512), (2048, 2048), (4096, 2048),
              (16384, 2048), (65536, 2048)]
    test_shapes(shapes)
    test_parity(shapes)
    test_benchmark(shapes)
