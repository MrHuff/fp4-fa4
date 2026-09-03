"""Extreme size benchmark: V1 vs V3 vs V4 vs TE"""
import os, torch, torch.nn.functional as F

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import NVFP4Quantizer

from torch.utils.cpp_extension import load
CSRC = '/workspace/fp4_matmul/fused_ops/csrc'
FL = ['-std=c++20','-O3','-lineinfo','--expt-relaxed-constexpr',
      '-gencode=arch=compute_100a,code=sm_100a']

print('Compiling V1...'); v1 = load(name='fused_te_quant_v1',
    sources=[CSRC+'/fused_te_quant_torch.cpp',CSRC+'/fused_te_quant.cu'],
    extra_include_paths=[CSRC],extra_cuda_cflags=FL,verbose=False)
print('Compiling V3...'); v3 = load(name='fused_te_quant_v3',
    sources=[CSRC+'/fused_te_quant_v3_torch.cpp',CSRC+'/fused_te_quant_v3.cu'],
    extra_include_paths=[CSRC],extra_cuda_cflags=FL,verbose=False)
print('Compiling V4...'); v4 = load(name='fused_te_quant_v4',
    sources=[CSRC+'/fused_te_quant_v4_torch.cpp',CSRC+'/fused_te_quant_v4.cu'],
    extra_include_paths=[CSRC],extra_cuda_cflags=FL,verbose=False)
print('Done.\n')

def bench(fn, warmup=20, steps=50):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(steps): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/steps

dev='cuda'; td=tex.DType.kFloat4E2M1
configs = [
    (4096,8192),(8192,8192),(16384,8192),
    (4096,16384),(4096,32768),
    (8192,16384),(8192,32768),
    (32768,8192),(65536,8192),
    (16384,16384),(32768,16384),
    (16384,32768),(32768,32768),
]

print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'{"M":>8} {"K":>8} | {"V1":>9} {"V3":>9} {"V4-dec":>9} {"V4-enc":>9} {"TE":>9} | {"V4/V1":>7} {"V3/V1":>7} {"V4/TE":>7} | {"BW":>7}')
print('-'*118)

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
        tv4d=bench(lambda:v4.forward_full(x,w,1e-5,0,0,0))
        tv4e=bench(lambda:v4.forward_full(x,w,1e-5,0,0,1))
        tte=bench(lambda:q.update_quantized(h,xq))

        db=2*M*K*2+M*K//2+M*K//16
        bw=db/(tv1*1e-3)/1e9
        print(f'{M:>8} {K:>8} | {tv1:>8.3f}ms {tv3:>8.3f}ms {tv4d:>8.3f}ms {tv4e:>8.3f}ms {tte:>8.3f}ms | {tv4d/tv1:>6.2f}x {tv3/tv1:>6.2f}x {tv4d/tte:>6.2f}x | {bw:>5.0f}G')
        del x,w,h,xq,rms,q; torch.cuda.empty_cache()
    except Exception as ex:
        print(f'{M:>8} {K:>8} | ERROR: {ex}')
        torch.cuda.empty_cache()
