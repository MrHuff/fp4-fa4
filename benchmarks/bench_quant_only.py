"""
Compare V7 quant-only vs TE quant-only to see if V7's quantisation is inherently slower.
V7 with identity norm + identity activation = pure quant.
"""
import torch
import torch.nn.functional as F

dev = 'cuda'

# Load V7
from torch.utils.cpp_extension import load
CSRC = '/workspace/fp4_matmul/fused_ops/csrc'
FL = ['-std=c++20', '-O3', '--expt-relaxed-constexpr',
      '-gencode=arch=compute_100a,code=sm_100a']
v7 = load(name='fused_te_quant_v7_qonly',
    sources=[CSRC+'/fused_te_quant_v7_torch.cpp', CSRC+'/fused_te_quant_v7.cu'],
    extra_include_paths=[CSRC], extra_cuda_cflags=FL, verbose=False)

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import NVFP4Quantizer

te_dtype = tex.DType.kFloat4E2M1
quantizer = NVFP4Quantizer(
    fp4_dtype=te_dtype, rowwise=True, columnwise=False,
    with_amax_reduction=False, amax_reduction_group=None,
    with_rht=False, with_post_rht_amax=False,
    with_2d_quantization=False, stochastic_rounding=False,
    with_random_sign_mask=False, encode_centric=False,
)

def bench(fn, warmup=20, steps=100):
    for _ in range(warmup): fn(); torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(); s.record()
    for _ in range(steps): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / steps

print("=" * 70)
print("V7 Quant-Only vs TE Quant-Only (no norm, no activation)")
print("=" * 70)
print(f"{'M':>8} {'K':>8} | {'V7 quant':>10} {'TE quant':>10} | {'ratio':>8}")
print("-" * 60)

configs = [
    (2048, 4096),
    (4096, 4096),
    (4096, 8192),
    (8192, 8192),
    (4096, 16384),
    (8192, 16384),
]

for M, K in configs:
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    w_ones = torch.ones(K, device=dev, dtype=torch.bfloat16)
    
    # V7 with identity norm (mode=2) + identity activation (mode=2) = pure quant
    t_v7 = bench(lambda: v7.forward_full(x, w_ones, 1e-5, 2, 2, 0))
    
    # TE quantizer
    t_te = bench(lambda: quantizer.quantize(x))
    
    ratio = t_v7 / t_te
    print(f"{M:>8} {K:>8} | {t_v7:>9.4f}ms {t_te:>9.4f}ms | {ratio:>7.2f}x")
    
    del x
    torch.cuda.empty_cache()

print()
print("=" * 70)
print("V7 Fused (norm+act+quant) vs TE Quant-Only")
print("=" * 70)
print(f"{'M':>8} {'K':>8} | {'V7 fused':>10} {'TE quant':>10} | {'ratio':>8} {'savings':>10}")
print("-" * 70)

for M, K in configs:
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    w_ones = torch.ones(K, device=dev, dtype=torch.bfloat16)
    
    # V7 fused: RMS norm + SiLU + quant
    t_v7_fused = bench(lambda: v7.forward_full(x, w_ones, 1e-5, 0, 0, 0))
    
    # TE quant only (no norm/act)
    t_te_quant = bench(lambda: quantizer.quantize(x))
    
    # Separate norm+act
    t_norm_act = bench(lambda: F.silu(F.rms_norm(x, (K,), w_ones, 1e-5)))
    
    # Total TE pipeline: norm+act+quant
    t_te_total = t_norm_act + t_te_quant
    
    ratio = t_v7_fused / t_te_quant
    savings_ms = t_te_total - t_v7_fused
    print(f"{M:>8} {K:>8} | {t_v7_fused:>9.4f}ms {t_te_quant:>9.4f}ms | {ratio:>7.2f}x   norm+act={t_norm_act:.4f} total_TE={t_te_total:.4f} save={savings_ms:.4f}")
    
    del x
    torch.cuda.empty_cache()
