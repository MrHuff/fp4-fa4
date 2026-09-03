"""bench_v7.py — V1 vs V3 vs V7 vs TE — all modes"""
import torch, torch.nn.functional as F
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import NVFP4Quantizer
from torch.utils.cpp_extension import load
import sys

CSRC = '/workspace/fp4_matmul/fused_ops/csrc'
FL = ['-std=c++20', '-O3', '-lineinfo', '--expt-relaxed-constexpr',
      '-gencode=arch=compute_100a,code=sm_100a']

print('Compiling...', flush=True)
v1 = load(name='fused_te_quant_v1',
    sources=[CSRC+'/fused_te_quant_torch.cpp', CSRC+'/fused_te_quant.cu'],
    extra_include_paths=[CSRC], extra_cuda_cflags=FL, verbose=False)
v3 = load(name='fused_te_quant_v3',
    sources=[CSRC+'/fused_te_quant_v3_torch.cpp', CSRC+'/fused_te_quant_v3.cu'],
    extra_include_paths=[CSRC], extra_cuda_cflags=FL, verbose=False)
v7 = load(name='fused_te_quant_v7',
    sources=[CSRC+'/fused_te_quant_v7_torch.cpp', CSRC+'/fused_te_quant_v7.cu'],
    extra_include_paths=[CSRC], extra_cuda_cflags=FL, verbose=False)
print('Done.\n', flush=True)

def dequant_fp4(fp4_bytes, scales_fp8, gs, m, k):
    lut = torch.tensor([0,0.5,1,1.5,2,3,4,6,-0,-0.5,-1,-1.5,-2,-3,-4,-6],
                       device=fp4_bytes.device, dtype=torch.float32)
    d = fp4_bytes.view(torch.uint8).to(torch.int32)
    u = torch.stack((d & 0x0F, d >> 4), dim=-1).reshape(m, k)
    fv = lut[u]
    sc = scales_fp8.view(torch.float8_e4m3fn).to(torch.float32)[:m,:k//16]
    return (fv.view(-1,16) * gs * sc.reshape(-1,1)).view(m, k)

def bench(fn, warmup=10, steps=20):
    for _ in range(warmup): fn(); torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(); s.record()
    for _ in range(steps): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / steps

dev = 'cuda'; td = tex.DType.kFloat4E2M1
configs = [
    (1024,8192), (4096,8192), (8192,8192), (16384,8192),
    (4096,16384), (8192,16384), (16384,16384),
    (8192,32768), (16384,32768),
    (32768,8192), (32768,16384), (32768,32768),
]

# ============================================================
# Performance benchmark: V1 vs V3 vs V7 vs TE (RMS + SiLU + decode)
# ============================================================
print(f'=== Performance: RMS + SiLU + decode ===', flush=True)
print(f'{"M":>8} {"K":>8} | {"V1":>8} {"V3":>8} {"V7":>8} {"TE":>8} | {"V7/V1":>6} {"V7/TE":>6}', flush=True)
print('-'*80, flush=True)

for M, K in configs:
    try:
        w = torch.ones(K, device=dev, dtype=torch.bfloat16)
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        rms = torch.nn.RMSNorm(K, eps=1e-5, device=dev, dtype=torch.bfloat16)
        with torch.no_grad(): rms.weight.fill_(1.0)
        h = F.silu(rms(x))
        q = NVFP4Quantizer(fp4_dtype=td, rowwise=True, columnwise=False)
        xq = q.make_empty((M, K), dtype=torch.bfloat16, device=dev)

        tv1 = bench(lambda: v1.forward_full(x, w, 1e-5, 0, 0))
        tv3 = bench(lambda: v3.forward_full(x, w, 1e-5, 0, 0, 0))
        tv7 = bench(lambda: v7.forward_full(x, w, 1e-5, 0, 0, 0))
        tte = bench(lambda: q.update_quantized(h, xq))

        print(f'{M:>8} {K:>8} | {tv1:>7.3f}ms {tv3:>7.3f}ms {tv7:>7.3f}ms {tte:>7.3f}ms | {tv7/tv1:>5.2f}x {tv7/tte:>5.2f}x', flush=True)
        del x, w, h, xq, rms, q; torch.cuda.empty_cache()
    except Exception as e:
        print(f'{M:>8} {K:>8} | ERROR: {e}', flush=True)
        torch.cuda.empty_cache()

# ============================================================
# Correctness: V7 vs V1 (byte match) and V7 vs V3 (all modes)
# ============================================================
print(f'\n=== Correctness ===', flush=True)
M, K = 4096, 8192
w = torch.ones(K, device=dev, dtype=torch.bfloat16)
x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)

# V7 vs V1 (decode, RMS, SiLU)
f1, s1, g1, _ = v1.forward_full(x, w, 1e-5, 0, 0)
f7, s7, g7, _ = v7.forward_full(x, w, 1e-5, 0, 0, 0)
bm = (f1.view(torch.uint8) == f7.view(torch.uint8)).float().mean().item()
d1 = dequant_fp4(f1, s1, g1.item(), M, K)
d7 = dequant_fp4(f7, s7, g7.item(), M, K)
cos = F.cosine_similarity(d1.flatten().unsqueeze(0), d7.flatten().unsqueeze(0)).item()
print(f'V7 vs V1 (RMS+SiLU+decode): byte={bm*100:.1f}% cos={cos:.6f} gs_v1={g1.item():.6f} gs_v7={g7.item():.6f}', flush=True)

# V7 vs V3 across modes
for nm, nm_name in [(0,'RMS'), (1,'AbsMax'), (2,'MXNorm')]:
    for sm, sm_name in [(0,'decode'), (1,'encode')]:
        f3, s3, g3, _ = v3.forward_full(x, w, 1e-5, nm, 0, sm)
        f7, s7, g7, _ = v7.forward_full(x, w, 1e-5, nm, 0, sm)
        bm = (f3.view(torch.uint8) == f7.view(torch.uint8)).float().mean().item()
        d3 = dequant_fp4(f3, s3, g3.item(), M, K)
        d7 = dequant_fp4(f7, s7, g7.item(), M, K)
        cos = F.cosine_similarity(d3.flatten().unsqueeze(0), d7.flatten().unsqueeze(0)).item()
        print(f'V7 vs V3 ({nm_name}+SiLU+{sm_name}): byte={bm*100:.1f}% cos={cos:.6f}', flush=True)
