"""
bench_v1_v3_v4_te.py — V1 vs V3 vs V4 vs TE Benchmark

V1: __nv_cvt_float2_to_fp4x2, global loads
V3: PTX mul+cvt, global loads
V4: PTX mul+cvt + TMA 1D bulk async copy
TE: tex.quantize (native TE quant-only)
"""

import os, sys, torch, torch.nn.functional as F, argparse

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import NVFP4Quantizer

print("Compiling V1 kernel...")
from torch.utils.cpp_extension import load
CSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fused_ops', 'csrc')
CUDA_FLAGS = ['-std=c++20', '-O3', '-lineinfo', '--expt-relaxed-constexpr',
              '-gencode=arch=compute_100a,code=sm_100a']

v1 = load(name='fused_te_quant_v1',
    sources=[os.path.join(CSRC, 'fused_te_quant_torch.cpp'),
             os.path.join(CSRC, 'fused_te_quant.cu')],
    extra_include_paths=[CSRC], extra_cuda_cflags=CUDA_FLAGS, verbose=False)

print("Compiling V3 kernel...")
v3 = load(name='fused_te_quant_v3',
    sources=[os.path.join(CSRC, 'fused_te_quant_v3_torch.cpp'),
             os.path.join(CSRC, 'fused_te_quant_v3.cu')],
    extra_include_paths=[CSRC], extra_cuda_cflags=CUDA_FLAGS, verbose=False)

print("Compiling V4 kernel...")
v4 = load(name='fused_te_quant_v4',
    sources=[os.path.join(CSRC, 'fused_te_quant_v4_torch.cpp'),
             os.path.join(CSRC, 'fused_te_quant_v4.cu')],
    extra_include_paths=[CSRC], extra_cuda_cflags=CUDA_FLAGS, verbose=False)
print("All compiled.\n")


def dequant_raw_fp4(fp4_bytes, scales_fp8, global_scale, m, k):
    lut = torch.tensor([0,0.5,1,1.5,2,3,4,6,-0,-0.5,-1,-1.5,-2,-3,-4,-6],
                       device=fp4_bytes.device, dtype=torch.float32)
    d = fp4_bytes.view(torch.uint8).to(torch.int32)
    lo, hi = d & 0x0F, d >> 4
    u = torch.stack((lo, hi), dim=-1).reshape(m, k)
    fv = lut[u]
    sc = scales_fp8.view(torch.float8_e4m3fn).to(torch.float32)[:m, :k//16]
    return (fv.view(-1, 16) * global_scale * sc.reshape(-1, 1)).view(m, k)


def bench(fn, warmup=50, steps=200):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(steps): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--warmup', type=int, default=50)
    p.add_argument('--steps', type=int, default=200)
    args = p.parse_args()

    K = 8192; dev = 'cuda'
    w = torch.ones(K, device=dev, dtype=torch.bfloat16)
    te_dtype = tex.DType.kFloat4E2M1

    print(f"{'='*120}")
    print(f"  V1 vs V3 vs V4(TMA) vs TE — GB200, K={K}")
    print(f"{'='*120}")
    print(f"\n{'M':>8} | {'V1':>9} {'V3':>9} {'V4-dec':>9} {'V4-enc':>9} {'TE':>9} |"
          f" {'V4/V1':>7} {'V4/V3':>7} {'V4/TE':>7} | {'cos(V1,V4)':>10} {'byte%':>6}")
    print('-' * 120)

    rms = torch.nn.RMSNorm(K, eps=1e-5, device=dev, dtype=torch.bfloat16)
    with torch.no_grad(): rms.weight.fill_(1.0)

    for M in [1024, 2048, 4096, 8192, 16384]:
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        quantizer = NVFP4Quantizer(fp4_dtype=te_dtype, rowwise=True, columnwise=False)
        x_q = quantizer.make_empty((M, K), dtype=torch.bfloat16, device=dev)
        h = F.silu(rms(x))

        t_te = bench(lambda: quantizer.update_quantized(h, x_q), args.warmup, args.steps)
        t_v1 = bench(lambda: v1.forward_full(x, w, 1e-5, 0, 0), args.warmup, args.steps)
        t_v3 = bench(lambda: v3.forward_full(x, w, 1e-5, 0, 0, 0), args.warmup, args.steps)
        t_v4d = bench(lambda: v4.forward_full(x, w, 1e-5, 0, 0, 0), args.warmup, args.steps)
        t_v4e = bench(lambda: v4.forward_full(x, w, 1e-5, 0, 0, 1), args.warmup, args.steps)

        # Correctness V1 vs V4
        fp4_v1, sc_v1, gs_v1, _ = v1.forward_full(x, w, 1e-5, 0, 0)
        fp4_v4, sc_v4, gs_v4, _ = v4.forward_full(x, w, 1e-5, 0, 0, 0)
        d1 = dequant_raw_fp4(fp4_v1, sc_v1, gs_v1.item(), M, K)
        d4 = dequant_raw_fp4(fp4_v4, sc_v4, gs_v4.item(), M, K)
        cos = F.cosine_similarity(d1.flatten().unsqueeze(0), d4.flatten().unsqueeze(0)).item()
        bm = (fp4_v1.view(torch.uint8) == fp4_v4.view(torch.uint8)).float().mean().item()

        print(f"{M:>8} | {t_v1:>8.3f}ms {t_v3:>8.3f}ms {t_v4d:>8.3f}ms {t_v4e:>8.3f}ms {t_te:>8.3f}ms |"
              f" {t_v4d/t_v1:>6.2f}x {t_v4d/t_v3:>6.2f}x {t_v4d/t_te:>6.2f}x |"
              f" {cos:>10.6f} {bm*100:>5.1f}%")

    print(f"\nNote: V1/V3/V4 include RMSNorm+SiLU; TE is quant-only")
    print(f"V4 uses TMA cp.async.bulk for global→shared memory loads")

if __name__ == '__main__':
    main()
