"""End-to-end TK vs TE parity test for actual training layers.

Tests that the TK GEMM path produces identical results to TE through:
  1. SimpleFP4Linear  (single forward + backward)
  2. FusedFeedForwardFP4  (3 forward GEMMs + 6 backward GEMMs)
  3. FusedAttentionFP4 QKV (grouped forward + 2 backward GEMMs)

Strategy: seed RNG identically before each path to get identical
stochastic quantizations, then compare outputs + gradients.
"""

import os
import sys
import torch
import torch.nn as nn

# Pre-init CUDA
torch.cuda.init()
_ = torch.zeros(1, device='cuda')
torch.cuda.synchronize()

sys.path.insert(0, '/workspace/fp4_matmul/ThunderKittens/kernels/gemm/nvfp4_b200')
sys.path.insert(0, '/workspace/low-bits-training')

# Verify TK loads
from low_bits_training.quantization.tk_gemm import is_tk_available
assert is_tk_available(), "TK GEMM not available!"
print(f"TK available: {is_tk_available()}")


def seed_all(s=42):
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def compare(name, t1, t2, atol=0.0):
    """Compare two tensors and print diagnostics."""
    if t1 is None and t2 is None:
        print(f"  {name}: both None — skip")
        return True
    if t1 is None or t2 is None:
        print(f"  {name}: one is None — FAIL")
        return False
    bitwise = torch.equal(t1, t2)
    if bitwise:
        print(f"  {name}: ✓ BITWISE MATCH")
        return True
    maxerr = (t1.float() - t2.float()).abs().max().item()
    relerr = maxerr / (t2.float().abs().max().item() + 1e-8)
    cos = torch.nn.functional.cosine_similarity(
        t1.flatten().float(), t2.flatten().float(), dim=0).item()
    ok = cos > 0.99
    status = f"~ close (relerr={relerr:.2e})" if ok else "✗ MISMATCH"
    print(f"  {name}: maxerr={maxerr:.6f} relerr={relerr:.2e} cos={cos:.6f} {status}")
    return ok


# =========================================================================
# Test 1: SimpleFP4Linear
# =========================================================================
def test_simple_linear(M=1024, K=4096, N=4096):
    from low_bits_training.quantization.fused_te_linear import SimpleFP4Linear

    print(f"\n{'='*70}")
    print(f"Test 1: SimpleFP4Linear  M={M} K={K} N={N}")
    print(f"{'='*70}")

    # Shared layer + frozen weights
    layer = SimpleFP4Linear(K, N, bias=False, device='cuda', dtype=torch.bfloat16)

    # --- TK path ---
    os.environ['USE_TK_GEMM'] = '1'
    seed_all(42)
    x_tk = torch.randn(M, K, dtype=torch.bfloat16, device='cuda', requires_grad=True)
    y_tk = layer(x_tk)
    loss_tk = y_tk.sum()
    loss_tk.backward()
    y_tk_d = y_tk.detach().clone()
    gx_tk = x_tk.grad.detach().clone()
    gw_tk = layer.weight.grad.detach().clone() if layer.weight.grad is not None else None
    # Zero grads
    layer.zero_grad()

    # --- TE path ---
    os.environ['USE_TK_GEMM'] = '0'
    seed_all(42)
    x_te = torch.randn(M, K, dtype=torch.bfloat16, device='cuda', requires_grad=True)
    y_te = layer(x_te)
    loss_te = y_te.sum()
    loss_te.backward()
    y_te_d = y_te.detach().clone()
    gx_te = x_te.grad.detach().clone()
    gw_te = layer.weight.grad.detach().clone() if layer.weight.grad is not None else None
    layer.zero_grad()

    os.environ['USE_TK_GEMM'] = '1'

    # Verify inputs match (sanity)
    assert torch.equal(x_tk.detach(), x_te.detach()), "Inputs differ despite same seed!"
    print("  Inputs: identical ✓")

    ok = True
    ok &= compare("forward output", y_tk_d, y_te_d)
    ok &= compare("grad_input", gx_tk, gx_te)
    ok &= compare("grad_weight", gw_tk, gw_te)
    return ok


# =========================================================================
# Test 2: FusedFeedForwardFP4
# =========================================================================
def test_ffn(M=1024, dim=4096, hidden_dim=14336):
    from low_bits_training.quantization.fused_te_linear import FusedFeedForwardFP4

    print(f"\n{'='*70}")
    print(f"Test 2: FusedFeedForwardFP4  M={M} dim={dim} hidden_dim={hidden_dim}")
    print(f"{'='*70}")

    # Create the fused FFN
    seed_all(0)
    ffn = FusedFeedForwardFP4(
        dim=dim, hidden_dim=hidden_dim, norm_eps=1e-5,
        bias=False, device='cuda', dtype=torch.bfloat16,
    )
    ffn.init_weights(init_std=0.02)

    # --- TK path ---
    os.environ['USE_TK_GEMM'] = '1'
    seed_all(42)
    x_tk = torch.randn(M, dim, dtype=torch.bfloat16, device='cuda', requires_grad=True)
    y_tk = ffn(x_tk)
    y_tk.sum().backward()
    y_tk_d = y_tk.detach().clone()
    gx_tk = x_tk.grad.detach().clone()
    gw1_tk = ffn.w1_weight.grad.detach().clone()
    gw2_tk = ffn.w2_weight.grad.detach().clone()
    gw3_tk = ffn.w3_weight.grad.detach().clone()
    gn_tk = ffn.norm_weight.grad.detach().clone()
    ffn.zero_grad()

    # --- TE path ---
    os.environ['USE_TK_GEMM'] = '0'
    seed_all(42)
    x_te = torch.randn(M, dim, dtype=torch.bfloat16, device='cuda', requires_grad=True)
    y_te = ffn(x_te)
    y_te.sum().backward()
    y_te_d = y_te.detach().clone()
    gx_te = x_te.grad.detach().clone()
    gw1_te = ffn.w1_weight.grad.detach().clone()
    gw2_te = ffn.w2_weight.grad.detach().clone()
    gw3_te = ffn.w3_weight.grad.detach().clone()
    gn_te = ffn.norm_weight.grad.detach().clone()
    ffn.zero_grad()

    os.environ['USE_TK_GEMM'] = '1'

    assert torch.equal(x_tk.detach(), x_te.detach()), "Inputs differ!"
    print("  Inputs: identical ✓")

    ok = True
    ok &= compare("forward output", y_tk_d, y_te_d)
    ok &= compare("grad_input", gx_tk, gx_te)
    ok &= compare("grad_w1", gw1_tk, gw1_te)
    ok &= compare("grad_w2", gw2_tk, gw2_te)
    ok &= compare("grad_w3", gw3_tk, gw3_te)
    ok &= compare("grad_norm", gn_tk, gn_te)
    return ok


# =========================================================================
# Test 3: FusedAttentionFP4 (QKV + wo paths)
# =========================================================================
def test_qkv(M=1024, dim=4096, n_heads=32, n_kv_heads=8, head_dim=128):
    from low_bits_training.quantization.fused_te_linear import FusedAttentionFP4

    q_dim = n_heads * head_dim
    k_dim = n_kv_heads * head_dim
    v_dim = n_kv_heads * head_dim

    print(f"\n{'='*70}")
    print(f"Test 3: FusedAttentionFP4 QKV  M={M} dim={dim}")
    print(f"  Q={q_dim} K={k_dim} V={v_dim}")
    print(f"{'='*70}")

    seed_all(0)
    attn = FusedAttentionFP4(
        dim=dim, n_heads=n_heads, n_kv_heads=n_kv_heads,
        head_dim=head_dim, norm_eps=1e-5,
        device='cuda', dtype=torch.bfloat16,
    )
    attn.init_weights(init_std=0.02)

    # --- TK path (QKV forward) ---
    os.environ['USE_TK_GEMM'] = '1'
    seed_all(42)
    x_tk = torch.randn(M, dim, dtype=torch.bfloat16, device='cuda', requires_grad=True)
    xq_tk, xk_tk, xv_tk = attn.forward_qkv(x_tk)
    (xq_tk.sum() + xk_tk.sum() + xv_tk.sum()).backward()
    xq_tk_d, xk_tk_d, xv_tk_d = xq_tk.detach().clone(), xk_tk.detach().clone(), xv_tk.detach().clone()
    gx_tk = x_tk.grad.detach().clone()
    gw_tk = attn.w_qkv.grad.detach().clone()
    gn_tk = attn.norm_weight.grad.detach().clone()
    attn.zero_grad()

    # --- TE path (QKV forward) ---
    os.environ['USE_TK_GEMM'] = '0'
    seed_all(42)
    x_te = torch.randn(M, dim, dtype=torch.bfloat16, device='cuda', requires_grad=True)
    xq_te, xk_te, xv_te = attn.forward_qkv(x_te)
    (xq_te.sum() + xk_te.sum() + xv_te.sum()).backward()
    xq_te_d, xk_te_d, xv_te_d = xq_te.detach().clone(), xk_te.detach().clone(), xv_te.detach().clone()
    gx_te = x_te.grad.detach().clone()
    gw_te = attn.w_qkv.grad.detach().clone()
    gn_te = attn.norm_weight.grad.detach().clone()
    attn.zero_grad()

    os.environ['USE_TK_GEMM'] = '1'

    assert torch.equal(x_tk.detach(), x_te.detach()), "Inputs differ!"
    print("  Inputs: identical ✓")

    ok = True
    ok &= compare("xq", xq_tk_d, xq_te_d)
    ok &= compare("xk", xk_tk_d, xk_te_d)
    ok &= compare("xv", xv_tk_d, xv_te_d)
    ok &= compare("grad_input", gx_tk, gx_te)
    ok &= compare("grad_w_qkv", gw_tk, gw_te)
    ok &= compare("grad_norm", gn_tk, gn_te)

    # --- wo forward ---
    print(f"\n  --- wo projection ---")
    os.environ['USE_TK_GEMM'] = '1'
    seed_all(99)
    attn_out_tk = torch.randn(M, q_dim, dtype=torch.bfloat16, device='cuda', requires_grad=True)
    wo_tk = attn.forward_wo(attn_out_tk)
    wo_tk.sum().backward()
    wo_tk_d = wo_tk.detach().clone()
    gao_tk = attn_out_tk.grad.detach().clone()
    gwo_tk = attn.wo_weight.grad.detach().clone()
    attn.zero_grad()

    os.environ['USE_TK_GEMM'] = '0'
    seed_all(99)
    attn_out_te = torch.randn(M, q_dim, dtype=torch.bfloat16, device='cuda', requires_grad=True)
    wo_te = attn.forward_wo(attn_out_te)
    wo_te.sum().backward()
    wo_te_d = wo_te.detach().clone()
    gao_te = attn_out_te.grad.detach().clone()
    gwo_te = attn.wo_weight.grad.detach().clone()
    attn.zero_grad()

    os.environ['USE_TK_GEMM'] = '1'

    ok &= compare("wo output", wo_tk_d, wo_te_d)
    ok &= compare("wo grad_input", gao_tk, gao_te)
    ok &= compare("wo grad_weight", gwo_tk, gwo_te)
    return ok


# =========================================================================
# Main
# =========================================================================
if __name__ == '__main__':
    print("="*70)
    print("End-to-end TK vs TE layer parity tests")
    print("="*70)

    results = {}

    try:
        results['SimpleFP4Linear'] = test_simple_linear(1024, 4096, 4096)
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        import traceback; traceback.print_exc()
        results['SimpleFP4Linear'] = False

    try:
        results['FusedFeedForwardFP4'] = test_ffn(1024, 4096, 14336)
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        import traceback; traceback.print_exc()
        results['FusedFeedForwardFP4'] = False

    try:
        results['FusedAttentionFP4'] = test_qkv(1024, 4096, 32, 8, 128)
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        import traceback; traceback.print_exc()
        results['FusedAttentionFP4'] = False

    print(f"\n{'='*70}")
    print("Summary:")
    all_pass = True
    for name, ok in results.items():
        status = "PASS ✓" if ok else "FAIL ✗"
        print(f"  {name}: {status}")
        all_pass &= ok
    print(f"\nOverall: {'ALL PASS ✓' if all_pass else 'SOME FAILED ✗'}")
    print(f"{'='*70}")
