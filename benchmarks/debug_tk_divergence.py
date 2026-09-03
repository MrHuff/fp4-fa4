"""Diagnostic: Find where TK diverges from TE with Llama 1B shapes.

Tests individual GEMMs with dim=2048 (Llama 1B) to catch shape-specific bugs.
"""
import os, sys, torch
os.environ['USE_TK_GEMM'] = '1'
torch.cuda.init()
_ = torch.zeros(1, device='cuda')
torch.cuda.synchronize()

sys.path.insert(0, '/workspace/fp4_matmul/ThunderKittens/kernels/gemm/nvfp4_b200')
sys.path.insert(0, '/workspace/low-bits-training')

from low_bits_training.quantization.tk_gemm import (
    is_tk_available, te_nvfp4_to_tk_format, te_nvfp4_to_tk_format_t,
    _get_tk, _convert_flat_scales
)
from low_bits_training.quantization.fused_te_linear import _fast_quantize
import transformer_engine_torch as tex
from transformer_engine.pytorch.constants import TE_DType

assert is_tk_available(), "TK not available!"
tk = _get_tk()

def test_shape(M, K, N, label=""):
    """Test a single forward/dgrad/wgrad GEMM at the given shape."""
    print(f"\n{'='*60}")
    print(f"{label}  M={M} K={K} N={N}")
    print(f"{'='*60}")

    torch.manual_seed(42)
    x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    w = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')
    dy = torch.randn(M, N, dtype=torch.bfloat16, device='cuda')

    x_nvfp4 = _fast_quantize(x)
    w_nvfp4 = _fast_quantize(w)
    dy_nvfp4 = _fast_quantize(dy)

    workspace = torch.empty(33554432, dtype=torch.int8, device='cuda')

    # ---- FORWARD: y = x @ W^T ----
    y_te = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
    tex.generic_gemm(
        w_nvfp4, True, x_nvfp4, False,
        y_te, None, TE_DType[torch.bfloat16],
        None, TE_DType[torch.bfloat16],
        False, None, False,
        workspace, workspace.shape[0], False, False,
    )

    y_tk = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
    a_fp4, a_sc, a_sg = te_nvfp4_to_tk_format(x_nvfp4, M, K)
    b_fp4, b_sc, b_sg = te_nvfp4_to_tk_format(w_nvfp4, N, K)
    tk.nvfp4_gemm(a_fp4, a_sc, a_sg, b_fp4, b_sc, b_sg, y_tk)

    bitwise = torch.equal(y_te, y_tk)
    maxerr = (y_te.float() - y_tk.float()).abs().max().item()
    relerr = maxerr / (y_te.float().abs().max().item() + 1e-8)
    print(f"  FWD:   {'BITWISE' if bitwise else f'DIFF maxerr={maxerr:.4f} relerr={relerr:.2e}'}")
    print(f"         TE: max={y_te.abs().max():.4f} std={y_te.float().std():.4f}")
    print(f"         TK: max={y_tk.abs().max():.4f} std={y_tk.float().std():.4f}")

    # ---- DGRAD: dx = dY @ W ----
    dx_te = tex.generic_gemm(
        w_nvfp4, False, dy_nvfp4, False,
        None, None, TE_DType[torch.bfloat16],
        None, TE_DType[torch.bfloat16],
        False, None, False,
        workspace, workspace.shape[0], False, False,
    )[0]

    dx_tk = torch.zeros(M, K, dtype=torch.bfloat16, device='cuda')
    a_fp4, a_sc, a_sg = te_nvfp4_to_tk_format(dy_nvfp4, M, N)
    b_fp4, b_sc, b_sg = te_nvfp4_to_tk_format_t(w_nvfp4, N, K)
    tk.nvfp4_gemm(a_fp4, a_sc, a_sg, b_fp4, b_sc, b_sg, dx_tk)

    bitwise = torch.equal(dx_te, dx_tk)
    maxerr = (dx_te.float() - dx_tk.float()).abs().max().item()
    relerr = maxerr / (dx_te.float().abs().max().item() + 1e-8)
    print(f"  DGRAD: {'BITWISE' if bitwise else f'DIFF maxerr={maxerr:.4f} relerr={relerr:.2e}'}")
    print(f"         TE: max={dx_te.abs().max():.4f} std={dx_te.float().std():.4f}")
    print(f"         TK: max={dx_tk.abs().max():.4f} std={dx_tk.float().std():.4f}")

    # ---- WGRAD: dW = dY^T @ x ----
    dw_te = tex.generic_gemm(
        x_nvfp4, False, dy_nvfp4, True,
        None, None, TE_DType[torch.bfloat16],
        None, TE_DType[torch.bfloat16],
        False, None, False,
        workspace, workspace.shape[0], False, False,
    )[0]

    dw_tk = torch.zeros(N, K, dtype=torch.bfloat16, device='cuda')
    a_fp4, a_sc, a_sg = te_nvfp4_to_tk_format_t(dy_nvfp4, M, N)
    b_fp4, b_sc, b_sg = te_nvfp4_to_tk_format_t(x_nvfp4, M, K)
    tk.nvfp4_gemm(a_fp4, a_sc, a_sg, b_fp4, b_sc, b_sg, dw_tk)

    bitwise = torch.equal(dw_te, dw_tk)
    maxerr = (dw_te.float() - dw_tk.float()).abs().max().item()
    relerr = maxerr / (dw_te.float().abs().max().item() + 1e-8)
    print(f"  WGRAD: {'BITWISE' if bitwise else f'DIFF maxerr={maxerr:.4f} relerr={relerr:.2e}'}")
    print(f"         TE: max={dw_te.abs().max():.4f} std={dw_te.float().std():.4f}")
    print(f"         TK: max={dw_tk.abs().max():.4f} std={dw_tk.float().std():.4f}")

    # Print scale diagnostics
    print(f"  SCALE DIAG:")
    print(f"    x amax={x_nvfp4._amax_rowwise.item():.4f} sc_global={x_nvfp4._amax_rowwise.item()/(6*448):.6f}")
    print(f"    w amax={w_nvfp4._amax_rowwise.item():.4f} sc_global={w_nvfp4._amax_rowwise.item()/(6*448):.6f}")
    print(f"    dy amax={dy_nvfp4._amax_rowwise.item():.4f}")

if __name__ == '__main__':
    # Llama 1B shapes
    dim = 2048
    hidden_dim = 5632  # Llama 1B
    n_heads = 16
    n_kv_heads = 4
    head_dim = 128
    M = 2048  # batch*seq (small for testing)

    print("Testing with Llama 1B shapes:")

    # FFN w1/w3: (M, dim) @ (hidden_dim, dim)^T → (M, hidden_dim)
    test_shape(M, dim, hidden_dim, "FFN w1/w3")

    # FFN w2: (M, hidden_dim) @ (dim, hidden_dim)^T → (M, dim)
    test_shape(M, hidden_dim, dim, "FFN w2")

    # QKV q: (M, dim) @ (q_dim, dim)^T
    test_shape(M, dim, n_heads * head_dim, "QKV q")

    # QKV k: (M, dim) @ (k_dim, dim)^T
    test_shape(M, dim, n_kv_heads * head_dim, "QKV k")

    # QKV v: same as k
    test_shape(M, dim, n_kv_heads * head_dim, "QKV v")

    # wo: (M, q_dim) @ (dim, q_dim)^T
    test_shape(M, n_heads * head_dim, dim, "wo")

    # Training batch size
    M_train = 65536
    test_shape(M_train, dim, hidden_dim, "FFN w1 (training M)")
    test_shape(M_train, dim, n_kv_heads * head_dim, "QKV k (training M)")
