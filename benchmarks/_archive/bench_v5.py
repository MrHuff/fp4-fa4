"""bench_v5.py — V1 vs V3 vs V5(2D TMA double-buf) vs TE — Extreme sizes"""
import os, torch, torch.nn.functional as F

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import NVFP4Quantizer

from torch.utils.cpp_extension import load
CSRC = '/workspace/fp4_matmul/fused_ops/csrc'
FL = ['-std=c++20','-O3','-lineinfo','--expt-relaxed-constexpr',
      '-gencode=arch=compute_100a,code=sm_100a']
LD = ['-L/usr/lib64', '-lcuda']  # Need CUDA driver API for cuTensorMapEncodeTiled

print('Compiling V1...'); v1 = load(name='fused_te_quant_v1',
    sources=[CSRC+'/fused_te_quant_torch.cpp',CSRC+'/fused_te_quant.cu'],
    extra_include_paths=[CSRC],extra_cuda_cflags=FL,verbose=False)
print('Compiling V3...'); v3 = load(name='fused_te_quant_v3',
    sources=[CSRC+'/fused_te_quant_v3_torch.cpp',CSRC+'/fused_te_quant_v3.cu'],
    extra_include_paths=[CSRC],extra_cuda_cflags=FL,verbose=False)
print('Compiling V5...'); v5 = load(name='fused_te_quant_v5',
    sources=[CSRC+'/fused_te_quant_v5_torch.cpp',CSRC+'/fused_te_quant_v5.cu'],
    extra_include_paths=[CSRC],extra_cuda_cflags=FL,extra_ldflags=LD,verbose=False)
print('Done.\n')

def dequant_fp4(fp4_bytes, scales_fp8, gs, m, k):
    lut = torch.tensor([0,0.5,1,1.5,2,3,4,6,-0,-0.5,-1,-1.5,-2,-3,-4,-6],
                       device=fp4_bytes.device, dtype=torch.float32)
    d = fp4_bytes.view(torch.uint8).to(torch.int32)
    u = torch.stack((d & 0x0F, d >> 4), dim=-1).reshape(m, k)
    fv = lut[u]
    sc = scales_fp8.view(torch.float8_e4m3fn).to(torch.float32)[:m,:k//16]
    return (fv.view(-1,16) * gs * sc.reshape(-1,1)).view(m, k)

def bench(fn, warmup=20, steps=50):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(steps): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/steps

dev='cuda'; td=tex.DType.kFloat4E2M1
# K must be multiple of 1024 for V5
configs = [
    (4096,8192),(8192,8192),(16384,8192),
    (4096,16384),(8192,16384),(16384,16384),
    (4096,32768),(8192,32768),(16384,32768),
    (32768,8192),(65536,8192),
    (32768,16384),(32768,32768),
]

print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'{"M":>8} {"K":>8} | {"V1":>9} {"V3":>9} {"V5-dec":>9} {"V5-enc":>9} {"TE":>9} | {"V5/V1":>7} {"V5/V3":>7} {"V5/TE":>7} | {"cos":>6} {"byte%":>5}')
print('-'*120)

for M,K in configs:
    try:
        w=torch.ones(K,device=dev,dtype=torch.bfloat16)
        x=torch.randn(M,K,device=dev,dtype=torch.bfloat16)
        rms=torch.nn.RMSNorm(K,eps=1e-5,device=dev,dtype=torch.bfloat16)
        with torch.no_grad(): rms.weight.fill_(1.0)
        h=F.silu(rms(x))
        q=NVFP4Quantizer(fp4_dtype=td,rowwise=True,columnwise=False)
        xq=q.make_empty((M,K),dtype=torch.bfloat16,device=dev)

        tv1=bench(lambda:v1.forward_full(x,w,1e-5,0,0))
        tv3=bench(lambda:v3.forward_full(x,w,1e-5,0,0,0))
        tv5d=bench(lambda:v5.forward_full(x,w,1e-5,0,0))
        tv5e=bench(lambda:v5.forward_full(x,w,1e-5,0,1))
        tte=bench(lambda:q.update_quantized(h,xq))

        # Correctness V1 vs V5
        f1,s1,g1,_=v1.forward_full(x,w,1e-5,0,0)
        f5,s5,g5,_=v5.forward_full(x,w,1e-5,0,0)
        d1=dequant_fp4(f1,s1,g1.item(),M,K)
        d5=dequant_fp4(f5,s5,g5.item(),M,K)
        cos=F.cosine_similarity(d1.flatten().unsqueeze(0),d5.flatten().unsqueeze(0)).item()
        bm=(f1.view(torch.uint8)==f5.view(torch.uint8)).float().mean().item()

        print(f'{M:>8} {K:>8} | {tv1:>8.3f}ms {tv3:>8.3f}ms {tv5d:>8.3f}ms {tv5e:>8.3f}ms {tte:>8.3f}ms | {tv5d/tv1:>6.2f}x {tv5d/tv3:>6.2f}x {tv5d/tte:>6.2f}x | {cos:>.4f} {bm*100:>4.1f}%')
        del x,w,h,xq,rms,q; torch.cuda.empty_cache()
    except Exception as ex:
        print(f'{M:>8} {K:>8} | ERROR: {ex}')
        torch.cuda.empty_cache()
