"""Run a single step of training and compare TK vs TE for EVERY GEMM call.

Patches tk_forward_gemm/tk_dgrad_gemm/tk_wgrad_gemm to also run TE and compare.
"""
import os, sys
os.environ['USE_TK_GEMM'] = '1'
import torch
torch.cuda.init()
_ = torch.zeros(1, device='cuda')
torch.cuda.synchronize()

sys.path.insert(0, '/workspace/low-bits-training')

import low_bits_training.quantization.tk_gemm as tk_mod
import transformer_engine_torch as tex
from transformer_engine.pytorch.constants import TE_DType

# Monkey-patch the wrapper functions to ALSO run TE and compare
_gemm_counter = [0]
_workspace = [None]
_max_err = [0.0]

def _get_workspace(device):
    if _workspace[0] is None or _workspace[0].device != device:
        _workspace[0] = torch.empty(33554432, dtype=torch.int8, device=device)
    return _workspace[0]

_orig_forward = tk_mod.tk_forward_gemm
_orig_dgrad = tk_mod.tk_dgrad_gemm
_orig_wgrad = tk_mod.tk_wgrad_gemm

def patched_forward(x_nvfp4, w_nvfp4, out=None):
    result = _orig_forward(x_nvfp4, w_nvfp4, out)
    # Also run TE
    M, K = x_nvfp4.shape
    N = w_nvfp4.shape[0]
    ws = _get_workspace(result.device)
    te_out = torch.empty_like(result)
    tex.generic_gemm(
        w_nvfp4, True, x_nvfp4, False,
        te_out, None, TE_DType[torch.bfloat16],
        None, TE_DType[torch.bfloat16],
        False, None, False,
        ws, ws.shape[0], False, False,
    )
    maxerr = (result.float() - te_out.float()).abs().max().item()
    bw = torch.equal(result, te_out)
    _gemm_counter[0] += 1
    _max_err[0] = max(_max_err[0], maxerr)
    if not bw:
        relerr = maxerr / (te_out.float().abs().max().item() + 1e-8)
        print(f"  [FWD#{_gemm_counter[0]}] M={M} K={K} N={N} maxerr={maxerr:.4f} relerr={relerr:.2e} "
              f"TE_max={te_out.abs().max():.2f} TK_max={result.abs().max():.2f}", flush=True)
    return result

def patched_dgrad(dY_nvfp4, w_nvfp4, dx=None):
    result = _orig_dgrad(dY_nvfp4, w_nvfp4, dx)
    M, N = dY_nvfp4.shape
    N_w, K = w_nvfp4.shape
    ws = _get_workspace(result.device)
    te_out = tex.generic_gemm(
        w_nvfp4, False, dY_nvfp4, False,
        None, None, TE_DType[torch.bfloat16],
        None, TE_DType[torch.bfloat16],
        False, None, False,
        ws, ws.shape[0], False, False,
    )[0]
    maxerr = (result.float() - te_out.float()).abs().max().item()
    bw = torch.equal(result, te_out)
    _gemm_counter[0] += 1
    _max_err[0] = max(_max_err[0], maxerr)
    if not bw:
        relerr = maxerr / (te_out.float().abs().max().item() + 1e-8)
        print(f"  [DGRAD#{_gemm_counter[0]}] M={M} K={K} N={N_w} maxerr={maxerr:.4f} relerr={relerr:.2e} "
              f"TE_max={te_out.abs().max():.2f} TK_max={result.abs().max():.2f}", flush=True)
    return result

def patched_wgrad(x_nvfp4, dY_nvfp4, dW=None):
    result = _orig_wgrad(x_nvfp4, dY_nvfp4, dW)
    M, K = x_nvfp4.shape
    M_dy, N = dY_nvfp4.shape
    ws = _get_workspace(result.device)
    te_out = tex.generic_gemm(
        x_nvfp4, False, dY_nvfp4, True,
        None, None, TE_DType[torch.bfloat16],
        None, TE_DType[torch.bfloat16],
        False, None, False,
        ws, ws.shape[0], False, False,
    )[0]
    maxerr = (result.float() - te_out.float()).abs().max().item()
    bw = torch.equal(result, te_out)
    _gemm_counter[0] += 1
    _max_err[0] = max(_max_err[0], maxerr)
    if not bw:
        relerr = maxerr / (te_out.float().abs().max().item() + 1e-8)
        print(f"  [WGRAD#{_gemm_counter[0]}] M={M} K={K} N={N} maxerr={maxerr:.4f} relerr={relerr:.2e} "
              f"TE_max={te_out.abs().max():.2f} TK_max={result.abs().max():.2f}  "
              f"x_shape={x_nvfp4.shape} dY_shape={dY_nvfp4.shape} "
              f"x_col_data={x_nvfp4._columnwise_data.shape} dY_col_data={dY_nvfp4._columnwise_data.shape}", flush=True)
    return result

# Apply patches
tk_mod.tk_forward_gemm = patched_forward
tk_mod.tk_dgrad_gemm = patched_dgrad
tk_mod.tk_wgrad_gemm = patched_wgrad

# Now run a FusedFeedForwardFP4 + FusedAttentionFP4
from low_bits_training.quantization.fused_te_linear import FusedFeedForwardFP4, FusedAttentionFP4
import torch.nn as nn

dim = 2048
hidden_dim = 5632
M = 2048  # batch * seq_len

torch.manual_seed(42)

# Test FFN
print("="*60)
print("Testing FusedFeedForwardFP4")
print("="*60)
w1 = nn.Linear(dim, hidden_dim, bias=False, device='cuda', dtype=torch.bfloat16)
w2 = nn.Linear(hidden_dim, dim, bias=False, device='cuda', dtype=torch.bfloat16)
w3 = nn.Linear(dim, hidden_dim, bias=False, device='cuda', dtype=torch.bfloat16)
norm_weight = torch.randn(dim, device='cuda', dtype=torch.bfloat16)
ffn = FusedFeedForwardFP4(dim=dim, hidden_dim=hidden_dim, device='cuda')
ffn.w1.weight.data = w1.weight.data
ffn.w2.weight.data = w2.weight.data
ffn.w3.weight.data = w3.weight.data
ffn.norm_weight.data = norm_weight.data

x = torch.randn(M, dim, device='cuda', dtype=torch.bfloat16, requires_grad=True)
print(f"\nForward pass:")
_gemm_counter[0] = 0
y = ffn(x)
print(f"  TK forward done. Output: max={y.abs().max():.4f} std={y.float().std():.4f}")

print(f"\nBackward pass:")
dy = torch.randn_like(y)
_gemm_counter[0] = 0
y.backward(dy)
print(f"  TK backward done. grad_input: max={x.grad.abs().max():.4f} std={x.grad.float().std():.4f}")
print(f"\n  Max error across all GEMMs: {_max_err[0]:.6f}")
