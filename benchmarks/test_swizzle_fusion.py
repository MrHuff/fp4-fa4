"""
test_swizzle_fusion.py — Verify that fused-swizzle quantisation produces
correct scales by comparing against separate quantise + swizzle.

Usage:
    python benchmarks/test_swizzle_fusion.py
"""

import torch
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import NVFP4Quantizer
from transformer_engine.pytorch.constants import TE_DType


def _make_quantizer():
    """Create quantizer with minimal args."""
    q = NVFP4Quantizer(
        fp4_dtype=tex.DType.kFloat4E2M1,
        rowwise=True, columnwise=True,
        with_amax_reduction=False, amax_reduction_group=None,
        with_rht=False, with_post_rht_amax=False,
    )
    q.optimize_for_gemm = getattr(q, 'optimize_for_gemm', False)
    q.internal = getattr(q, 'internal', False)
    return q


def test_fused_swizzle_correctness(M=256, K=256):
    """Compare: (1) quant(flat) + swizzle vs (2) quant(swizzled)."""
    device = "cuda"
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)

    # --- Path A: Baseline (flat quant + separate swizzle) ---
    q_flat = _make_quantizer()
    flat_out = q_flat.make_empty((M, K), dtype=torch.bfloat16, device=device, requires_grad=False)
    flat_out = q_flat.update_quantized(x, flat_out)

    # Get flat scales
    flat_scales = flat_out._rowwise_scale_inv.clone()

    # Do swizzle via TE's built-in
    if not hasattr(flat_out, '_with_gemm_swizzled_scales'):
        flat_out._with_gemm_swizzled_scales = False
    tex.swizzle_scales_for_gemm_(flat_out)
    swizzled_scales_ref = flat_out._rowwise_scale_inv.clone()

    # --- Path B: Fused swizzle (quant writes directly in swizzled format) ---
    q_fused = _make_quantizer()
    fused_out = q_fused.make_empty((M, K), dtype=torch.bfloat16, device=device, requires_grad=False)
    # Set the flag BEFORE quantizing so the kernel writes swizzled
    fused_out._with_gemm_swizzled_scales = True
    fused_out = q_fused.update_quantized(x, fused_out)

    fused_scales = fused_out._rowwise_scale_inv.clone()

    # --- Compare ---
    print(f"Test: M={M}, K={K}")
    print(f"  Flat scales shape:     {flat_scales.shape}")
    print(f"  Swizzled ref shape:    {swizzled_scales_ref.shape}")
    print(f"  Fused swizzled shape:  {fused_scales.shape}")

    # Both should have same number of bytes
    flat_bytes = flat_scales.view(torch.uint8)
    ref_bytes = swizzled_scales_ref.view(torch.uint8)
    fused_bytes = fused_scales.view(torch.uint8)

    print(f"  Flat bytes count:      {flat_bytes.numel()}")
    print(f"  Ref bytes count:       {ref_bytes.numel()}")
    print(f"  Fused bytes count:     {fused_bytes.numel()}")

    if ref_bytes.numel() == fused_bytes.numel():
        match = torch.equal(ref_bytes, fused_bytes)
        if match:
            print(f"  ✅ PASS: Fused swizzled scales match reference!")
        else:
            mismatches = (ref_bytes != fused_bytes).sum().item()
            total = ref_bytes.numel()
            print(f"  ❌ FAIL: {mismatches}/{total} bytes differ")
            # Show first few mismatches
            diff_idx = (ref_bytes != fused_bytes).nonzero(as_tuple=True)[0][:10]
            for idx in diff_idx:
                print(f"    byte[{idx.item()}]: ref={ref_bytes[idx].item():02x} fused={fused_bytes[idx].item():02x}")
    else:
        print(f"  ❌ FAIL: Different byte counts")

    # Also verify GEMM works with fused-swizzled output
    print("\n  Testing GEMM with fused-swizzled tensors...")
    try:
        w_bf16 = torch.randn(K, K, dtype=torch.bfloat16, device=device)
        q_w = _make_quantizer()
        w_out = q_w.make_empty((K, K), dtype=torch.bfloat16, device=device, requires_grad=False)
        w_out._with_gemm_swizzled_scales = True
        w_out = q_w.update_quantized(w_bf16, w_out)
        w_out._with_gemm_swizzled_scales = True

        workspace = torch.empty(4, dtype=torch.uint8, device=device)
        out = torch.empty(M, K, device=device, dtype=torch.bfloat16)
        out_dtype = TE_DType[torch.bfloat16]

        tex.generic_gemm(
            w_out, True, fused_out, False,
            out, None, out_dtype,
            None, TE_DType[torch.bfloat16],
            False, None, False, workspace,
            workspace.shape[0], False, False,
        )
        print(f"  ✅ GEMM succeeded! Output norm: {out.norm():.4f}")
    except Exception as e:
        print(f"  ❌ GEMM failed: {e}")

    return match if ref_bytes.numel() == fused_bytes.numel() else False


def bench_swizzle_overhead(M=4096, K=4096, steps=200, warmup=50):
    """Compare: quant+swizzle (old) vs quant_fused (new)."""
    device = "cuda"
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)

    # --- Old: quant(flat) + swizzle ---
    q_old = _make_quantizer()
    old_out = q_old.make_empty((M, K), dtype=torch.bfloat16, device=device, requires_grad=False)
    old_out._with_gemm_swizzled_scales = False

    def old_path():
        q_old.update_quantized(x, old_out)
        tex.swizzle_scales_for_gemm_(old_out)
        old_out._with_gemm_swizzled_scales = False  # reset for next iter

    for _ in range(warmup):
        old_path()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(steps):
        old_path()
    end.record()
    torch.cuda.synchronize()
    old_ms = start.elapsed_time(end) / steps

    # --- New: quant(swizzled) ---
    q_new = _make_quantizer()
    new_out = q_new.make_empty((M, K), dtype=torch.bfloat16, device=device, requires_grad=False)
    new_out._with_gemm_swizzled_scales = True

    def new_path():
        q_new.update_quantized(x, new_out)

    for _ in range(warmup):
        new_path()
    torch.cuda.synchronize()
    start2 = torch.cuda.Event(enable_timing=True)
    end2 = torch.cuda.Event(enable_timing=True)
    start2.record()
    for _ in range(steps):
        new_path()
    end2.record()
    torch.cuda.synchronize()
    new_ms = start2.elapsed_time(end2) / steps

    print(f"\nBenchmark: M={M}, K={K}")
    print(f"  Old (quant + swizzle): {old_ms:.4f} ms")
    print(f"  New (fused swizzle):   {new_ms:.4f} ms")
    print(f"  Speedup:               {old_ms/new_ms:.2f}x")


if __name__ == "__main__":
    print("=" * 60)
    print("  Fused Swizzle Quantisation Test")
    print("=" * 60)

    all_pass = True
    for M, K in [(128, 128), (256, 256), (1024, 1024), (4096, 4096)]:
        result = test_fused_swizzle_correctness(M, K)
        all_pass = all_pass and result
        print()

    if all_pass:
        print("\n✅ All correctness tests PASSED!")
        bench_swizzle_overhead()
    else:
        print("\n❌ Some tests FAILED! Skipping benchmark.")
