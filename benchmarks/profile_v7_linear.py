"""
Profile V7FusedLinear to find where the overhead comes from.
"""
import torch
import torch.nn.functional as F
import time

dev = 'cuda'
M, K, N = 4096, 8192, 4096

# ============================================================
# 1. Profile individual steps of V7FusedLinear  
# ============================================================
print("=" * 70)
print("Profiling V7FusedLinear overhead sources")
print("=" * 70)

# Load V7 extension
from torch.utils.cpp_extension import load
CSRC = '/workspace/fp4_matmul/fused_ops/csrc'
FL = ['-std=c++20', '-O3', '--expt-relaxed-constexpr',
      '-gencode=arch=compute_100a,code=sm_100a']
v7 = load(name='fused_te_quant_v7_profile',
    sources=[CSRC+'/fused_te_quant_v7_torch.cpp', CSRC+'/fused_te_quant_v7.cu'],
    extra_include_paths=[CSRC], extra_cuda_cflags=FL, verbose=False)

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import NVFP4Quantizer
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Tensor
from transformer_engine.pytorch.constants import TE_DType, NVFP4_BLOCK_SCALING_SIZE
from transformer_engine.pytorch.utils import round_up_to_nearest_multiple
import math

x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
w_norm = torch.ones(K, device=dev, dtype=torch.bfloat16)
weight = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.023

te_dtype = tex.DType.kFloat4E2M1
input_quantizer = NVFP4Quantizer(
    fp4_dtype=te_dtype, rowwise=True, columnwise=True,
    with_amax_reduction=False, amax_reduction_group=None,
    with_rht=False, with_post_rht_amax=False,
    with_2d_quantization=False, stochastic_rounding=False,
    with_random_sign_mask=True, encode_centric=False,
)
weight_quantizer = NVFP4Quantizer(
    fp4_dtype=te_dtype, rowwise=True, columnwise=True,
    with_amax_reduction=False, amax_reduction_group=None,
    with_rht=False, with_post_rht_amax=False,
    with_2d_quantization=False, stochastic_rounding=False,
    with_random_sign_mask=True, encode_centric=False,
)

# Warmup
for _ in range(5):
    fp4_data, scales, global_scale, inv_rms = v7.forward_full(x, w_norm, 1e-5, 0, 0, 0)
    h = F.silu(F.rms_norm(x, (K,), w_norm, 1e-5))
    x_nvfp4 = input_quantizer.quantize(h)
torch.cuda.synchronize()

def bench_cuda(fn, warmup=10, steps=50):
    for _ in range(warmup): fn(); torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(); s.record()
    for _ in range(steps): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / steps

# --- Step 1: V7 kernel only ---
t_v7_kernel = bench_cuda(lambda: v7.forward_full(x, w_norm, 1e-5, 0, 0, 0))
print(f"\n1. V7 kernel (norm+act+quant):    {t_v7_kernel:.4f} ms")

# --- Step 2: NVFP4Tensor construction (Python overhead) ---
fp4_data, scales, global_scale, inv_rms = v7.forward_full(x, w_norm, 1e-5, 0, 0, 0)
outer = round_up_to_nearest_multiple(M, 128)
inner = round_up_to_nearest_multiple(math.ceil(K / NVFP4_BLOCK_SCALING_SIZE), 4)

def construct_nvfp4():
    padded_scales = torch.zeros(outer, inner, dtype=torch.uint8, device=dev)
    src_cols = K // NVFP4_BLOCK_SCALING_SIZE
    padded_scales[:M, :src_cols] = scales[:M, :src_cols]
    amax = global_scale * (6.0 * 448.0)
    return NVFP4Tensor(
        shape=(M, K), dtype=torch.bfloat16,
        rowwise_data=fp4_data, rowwise_scale_inv=padded_scales,
        columnwise_data=None, columnwise_scale_inv=None,
        amax_rowwise=amax, amax_columnwise=None,
        fp4_dtype=tex.DType.kFloat4E2M1, quantizer=input_quantizer,
        requires_grad=False,
    )

t_construct = bench_cuda(construct_nvfp4)
print(f"2. NVFP4Tensor construction:      {t_construct:.4f} ms")

# --- Step 2b: Just the padding ---
def just_pad():
    padded_scales = torch.zeros(outer, inner, dtype=torch.uint8, device=dev)
    src_cols = K // NVFP4_BLOCK_SCALING_SIZE
    padded_scales[:M, :src_cols] = scales[:M, :src_cols]
    return padded_scales

t_pad = bench_cuda(just_pad)
print(f"   2a. Scale padding only:        {t_pad:.4f} ms")

# --- Step 2c: Just NVFP4Tensor init ---
padded_scales = torch.zeros(outer, inner, dtype=torch.uint8, device=dev)
amax = global_scale * (6.0 * 448.0)
def just_init():
    return NVFP4Tensor(
        shape=(M, K), dtype=torch.bfloat16,
        rowwise_data=fp4_data, rowwise_scale_inv=padded_scales,
        columnwise_data=None, columnwise_scale_inv=None,
        amax_rowwise=amax, amax_columnwise=None,
        fp4_dtype=tex.DType.kFloat4E2M1, quantizer=input_quantizer,
        requires_grad=False,
    )
t_init = bench_cuda(just_init)
print(f"   2b. NVFP4Tensor __init__ only:  {t_init:.4f} ms")

# --- Step 3: Weight quant ---
t_wquant = bench_cuda(lambda: weight_quantizer.quantize(weight))
print(f"3. Weight quantize (TE):           {t_wquant:.4f} ms")

# --- Step 4: Workspace allocation ---
t_workspace = bench_cuda(lambda: torch.empty(32*1024*1024, dtype=torch.uint8, device=dev))
print(f"4. Workspace alloc (32MB):         {t_workspace:.4f} ms")

# --- Step 5: GEMM only ---
x_nvfp4 = construct_nvfp4()
w_nvfp4 = weight_quantizer.quantize(weight)
workspace = torch.empty(32*1024*1024, dtype=torch.uint8, device=dev)
y = torch.empty(M, N, dtype=torch.bfloat16, device=dev)

def just_gemm():
    tex.generic_gemm(w_nvfp4, True, x_nvfp4, False, y, None,
        TE_DType[torch.bfloat16], None, TE_DType[torch.bfloat16],
        False, None, False, workspace, workspace.shape[0], False, False)

t_gemm = bench_cuda(just_gemm)
print(f"5. GEMM only (tex.generic_gemm):   {t_gemm:.4f} ms")

# --- Reference: TE norm+act+quant ---
t_te_norm = bench_cuda(lambda: F.silu(F.rms_norm(x, (K,), w_norm, 1e-5)))
print(f"\n--- Reference ---")
print(f"6. RMSNorm+SiLU (PyTorch):         {t_te_norm:.4f} ms")

t_te_quant = bench_cuda(lambda: input_quantizer.quantize(F.silu(F.rms_norm(x, (K,), w_norm, 1e-5))))
print(f"7. RMSNorm+SiLU+TE quant:          {t_te_quant:.4f} ms")

# --- Total comparison ---
print(f"\n--- Total Breakdown ---")
v7_total = t_v7_kernel + t_construct + t_wquant + t_workspace + t_gemm
te_total = t_te_norm + t_te_quant - t_te_norm + t_wquant + t_gemm  # norm already in quant
print(f"V7 pipeline:  kernel({t_v7_kernel:.3f}) + construct({t_construct:.3f}) + wquant({t_wquant:.3f}) + workspace({t_workspace:.3f}) + gemm({t_gemm:.3f}) = {v7_total:.3f} ms")

te_quant_only = t_te_quant - t_te_norm
te_total2 = t_te_norm + te_quant_only + t_wquant + t_gemm 
print(f"TE pipeline:  norm+act({t_te_norm:.3f}) + quant({te_quant_only:.3f}) + wquant({t_wquant:.3f}) + gemm({t_gemm:.3f}) = {te_total2:.3f} ms")
print(f"\nOverhead from V7 wrapper: {(t_construct + t_workspace):.3f} ms")
