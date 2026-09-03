#!/usr/bin/env python3
"""
TKTK vs TETE: Forward + Backward comparison for linear and grouped QKV layers.

TKTK = TK standalone quant (USE_TK_QUANT=1) + TK GEMM (USE_TK_GEMM=1)
TETE = TE quant + TE GEMM (no TK)

Tests:
  1. Single linear layer: _WoFunction forward + backward
  2. Grouped QKV layer: _FusedQKVFunction forward + backward
"""
import os, sys, torch
sys.path.insert(0, '/workspace/low-bits-training')
sys.path.insert(0, '/workspace/fp4_matmul/TK_quantisation/nvfp4')

# Import tex indirectly via the module that handles TE loading order
from low_bits_training.quantization.fused_te_linear import tex

def compare(name, a, b):
    """Compare two tensors and print stats."""
    if a is None and b is None:
        print(f"  {name}: both None")
        return True
    match = torch.equal(a, b)
    diff = (a.float() - b.float())
    max_abs = diff.abs().max().item()
    rms_a = a.float().pow(2).mean().sqrt().item()
    rms_b = b.float().pow(2).mean().sqrt().item()
    rms_d = diff.pow(2).mean().sqrt().item()
    rel = rms_d / (max(rms_a, rms_b) + 1e-12)
    tag = "✅" if match else ("⚠️" if rel < 0.01 else "❌")
    print(f"  {name}: {tag}  exact={match}  max_abs={max_abs:.2e}  rel_rms={rel:.2e}  "
          f"(rms: TK={rms_a:.2e} TE={rms_b:.2e})")
    return match

print("=" * 90)
print("  TKTK vs TETE: Forward + Backward Parity")
print("=" * 90)

# Llama 1B dimensions
M = 2048       # batch * seq
dim = 2048     # model dim
n_heads = 16
n_kv_heads = 8
head_dim = 128
q_dim = n_heads * head_dim       # 2048
k_dim = n_kv_heads * head_dim   # 1024
v_dim = n_kv_heads * head_dim   # 1024

device = 'cuda'
dtype = torch.bfloat16
torch.manual_seed(42)

# ========================================================================
# Test 1: Single Linear Layer (Wo-style)
# ========================================================================
print("\n" + "=" * 90)
print("  Test 1: Single Linear (Wo-style): y = W @ x")
print("=" * 90)

# Shared weights and inputs
w_wo = torch.randn(dim, q_dim, dtype=dtype, device=device) * 0.02
x_wo = torch.randn(M, q_dim, dtype=dtype, device=device) * 0.01
grad_out_wo = torch.randn(M, dim, dtype=dtype, device=device) * 0.001

from low_bits_training.quantization.fused_te_linear import (
    _WoFunction_TE, _WoFunction_TK, _fast_quantize, _TKQuantized,
    _FusedQKVFunction_TE, _FusedQKVFunction_TK, _get_te_fused,
    NVFP4Quantizer,
)
TE_DType = {torch.bfloat16: tex.DType.kBFloat16}

def make_quantizer():
    return NVFP4Quantizer(
        fp4_dtype=tex.DType.kFloat4E2M1, rowwise=True, columnwise=True,
        with_amax_reduction=False, amax_reduction_group=None,
        with_rht=False, with_post_rht_amax=False,
        with_2d_quantization=False, stochastic_rounding=False,
        with_random_sign_mask=True, encode_centric=False,
    )

workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=device)

# ── TETE path ──
print("\n--- TETE (TE quant + TE GEMM) ---")
os.environ['USE_TK_QUANT'] = '0'
os.environ['USE_TK_GEMM'] = '0'
os.environ['NVTE_CUSTOM_QUANT'] = '1'

x_te = x_wo.clone().requires_grad_(True)
w_te = w_wo.clone().requires_grad_(True)
y_te = _WoFunction_TE.apply(x_te, w_te, make_quantizer(), make_quantizer(), make_quantizer(), workspace)
y_te.backward(grad_out_wo)
print(f"  y_te: {y_te.shape}")
print(f"  dx_te: {x_te.grad.shape}")
print(f"  dw_te: {w_te.grad.shape}")

# ── TKTK path ──
print("\n--- TKTK (TK quant + TK GEMM) ---")
os.environ['USE_TK_QUANT'] = '1'
os.environ['USE_TK_GEMM'] = '1'

x_tk = x_wo.clone().requires_grad_(True)
w_tk = w_wo.clone().requires_grad_(True)
y_tk = _WoFunction_TK.apply(x_tk, w_tk, make_quantizer(), make_quantizer(), make_quantizer(), workspace)
y_tk.backward(grad_out_wo)
print(f"  y_tk: {y_tk.shape}")
print(f"  dx_tk: {x_tk.grad.shape}")
print(f"  dw_tk: {w_tk.grad.shape}")

# ── Compare ──
print("\n--- Comparison ---")
compare("Forward (y)", y_tk, y_te)
compare("dInput (dx)", x_tk.grad, x_te.grad)
compare("dWeight (dW)", w_tk.grad, w_te.grad)


# ========================================================================
# Test 2: Grouped QKV Layer (FusedQKVFunction)
# ========================================================================
print("\n" + "=" * 90)
print(f"  Test 2: Grouped QKV (q_dim={q_dim}, k_dim={k_dim}, v_dim={v_dim})")
print("=" * 90)

total_out = q_dim + k_dim + v_dim
w_qkv = torch.randn(total_out, dim, dtype=dtype, device=device) * 0.02
x_qkv = torch.randn(M, dim, dtype=dtype, device=device) * 0.01
norm_w = torch.ones(dim, dtype=dtype, device=device)
epsilon = 1e-5

grad_q = torch.randn(M, q_dim, dtype=dtype, device=device) * 0.001
grad_k = torch.randn(M, k_dim, dtype=dtype, device=device) * 0.001
grad_v = torch.randn(M, v_dim, dtype=dtype, device=device) * 0.001

# ── TETE path ──
print("\n--- TETE (TE quant + TE GEMM) ---")
os.environ['USE_TK_QUANT'] = '0'
os.environ['USE_TK_GEMM'] = '0'

x_te2 = x_qkv.clone().requires_grad_(True)
w_te2 = w_qkv.clone().requires_grad_(True)
nw_te2 = norm_w.clone().requires_grad_(True)

xq_te, xk_te, xv_te = _FusedQKVFunction_TE.apply(
    x_te2, w_te2, nw_te2, epsilon, q_dim, k_dim, v_dim,
    make_quantizer(), make_quantizer(), make_quantizer(), workspace)

loss_te = (xq_te * grad_q).sum() + (xk_te * grad_k).sum() + (xv_te * grad_v).sum()
loss_te.backward()
print(f"  xq_te: {xq_te.shape}, xk_te: {xk_te.shape}, xv_te: {xv_te.shape}")
print(f"  dx_te: {x_te2.grad.shape}")
print(f"  dw_te: {w_te2.grad.shape}")
print(f"  dnorm_te: {nw_te2.grad.shape}")

# ── TKTK path ──
print("\n--- TKTK (TK quant + TK GEMM) ---")
os.environ['USE_TK_QUANT'] = '1'
os.environ['USE_TK_GEMM'] = '1'

x_tk2 = x_qkv.clone().requires_grad_(True)
w_tk2 = w_qkv.clone().requires_grad_(True)
nw_tk2 = norm_w.clone().requires_grad_(True)

xq_tk, xk_tk, xv_tk = _FusedQKVFunction_TK.apply(
    x_tk2, w_tk2, nw_tk2, epsilon, q_dim, k_dim, v_dim,
    make_quantizer(), make_quantizer(), make_quantizer(), workspace)

loss_tk = (xq_tk * grad_q).sum() + (xk_tk * grad_k).sum() + (xv_tk * grad_v).sum()
loss_tk.backward()
print(f"  xq_tk: {xq_tk.shape}, xk_tk: {xk_tk.shape}, xv_tk: {xv_tk.shape}")
print(f"  dx_tk: {x_tk2.grad.shape}")
print(f"  dw_tk: {w_tk2.grad.shape}")
print(f"  dnorm_tk: {nw_tk2.grad.shape}")

# ── Compare ──
print("\n--- Comparison ---")
compare("Forward xq", xq_tk, xq_te)
compare("Forward xk", xk_tk, xk_te)
compare("Forward xv", xv_tk, xv_te)
compare("dInput (dx)", x_tk2.grad, x_te2.grad)
compare("dWeight (dW)", w_tk2.grad, w_te2.grad)
compare("dNorm (dgamma)", nw_tk2.grad, nw_te2.grad)

print("\n" + "=" * 90)
print("  All tests complete!")
print("=" * 90)
