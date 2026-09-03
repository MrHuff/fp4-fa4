"""Test grouped GEMM vs individual GEMMs for QKV with Llama 1B shapes."""
import os, sys, torch
os.environ['USE_TK_GEMM'] = '1'
torch.cuda.init()
_ = torch.zeros(1, device='cuda')
torch.cuda.synchronize()

sys.path.insert(0, '/workspace/fp4_matmul/ThunderKittens/kernels/gemm/nvfp4_b200')
sys.path.insert(0, '/workspace/low-bits-training')

from low_bits_training.quantization.tk_gemm import (
    tk_forward_gemm, tk_grouped_forward_gemm, te_nvfp4_to_tk_format,
    _get_tk, _convert_flat_scales
)
from low_bits_training.quantization.fused_te_linear import _fast_quantize, _get_fp4_ext

dim = 2048
q_dim = 2048  # 16 heads * 128
k_dim = 512   # 4 kv_heads * 128
v_dim = 512
M = 65536  # training batch

torch.manual_seed(42)
x = torch.randn(M, dim, dtype=torch.bfloat16, device='cuda')
wq = torch.randn(q_dim, dim, dtype=torch.bfloat16, device='cuda')
wk = torch.randn(k_dim, dim, dtype=torch.bfloat16, device='cuda')
wv = torch.randn(v_dim, dim, dtype=torch.bfloat16, device='cuda')

# Stack weights like training code does
w_qkv = torch.cat([wq, wk, wv], dim=0)  # (3072, 2048)
print(f"w_qkv shape: {w_qkv.shape}")

# Quantize input
x_nvfp4 = _fast_quantize(x)
print(f"x_nvfp4 shape: {x_nvfp4.shape}")
print(f"x amax: {x_nvfp4._amax_rowwise.item():.4f}")

# Grouped quantize (matches training path)
fp4_ext = _get_fp4_ext()
split_results = fp4_ext.group_nvfp4_quantize(w_qkv.contiguous(), [q_dim, k_dim, v_dim])
split_amaxes = [r[4].item() for r in split_results]
print(f"Split amaxes: {split_amaxes}")

# ---- Individual GEMMs (known correct) ----
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Tensor
from transformer_engine.pytorch import NVFP4Quantizer
import transformer_engine_torch as tex
from transformer_engine.pytorch.constants import TE_DType

individual_results = []
for gi, (fp4_data, si_data, fp4_t, si_t, amax) in enumerate(split_results):
    N_g = [q_dim, k_dim, v_dim][gi]
    w_nvfp4 = NVFP4Tensor(
        (N_g, dim), torch.bfloat16,
        rowwise_data=fp4_data, rowwise_scale_inv=si_data,
        columnwise_data=fp4_t, columnwise_scale_inv=si_t,
        amax_rowwise=amax, amax_columnwise=amax,
        fp4_dtype=tex.DType.kFloat4E2M1,
        quantizer=NVFP4Quantizer(fp4_dtype=tex.DType.kFloat4E2M1, rowwise=True, columnwise=True),
    )
    if not hasattr(w_nvfp4, '_with_gemm_swizzled_scales'):
        w_nvfp4._with_gemm_swizzled_scales = False
    
    # Individual TK forward
    out_tk = tk_forward_gemm(x_nvfp4, w_nvfp4)
    individual_results.append(out_tk)
    print(f"  Individual TK [{['Q','K','V'][gi]}]: max={out_tk.abs().max():.2f} std={out_tk.float().std():.4f}")

# ---- Grouped GEMM ----
N_dims = [q_dim, k_dim, v_dim]
grouped_outputs = tk_grouped_forward_gemm(x_nvfp4, split_results, N_dims, M, dim)
for gi, gout in enumerate(grouped_outputs):
    ind = individual_results[gi]
    bw = torch.equal(gout, ind)
    maxerr = (gout.float() - ind.float()).abs().max().item()
    relerr = maxerr / (ind.float().abs().max().item() + 1e-8)
    print(f"  Grouped vs Individual [{['Q','K','V'][gi]}]: "
          f"{'MATCH' if bw else f'DIFF maxerr={maxerr:.4f} relerr={relerr:.2e}'} "
          f"grouped_max={gout.abs().max():.2f} ind_max={ind.abs().max():.2f}")
