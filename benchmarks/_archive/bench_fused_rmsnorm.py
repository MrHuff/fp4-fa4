"""
Comprehensive benchmark: Fused RMSNorm + SiLU + NVFP4 Quantization

Compares 5 approaches:
  1. Eager:     torch.RMSNorm → SiLU → TE quantize  (3 separate kernels)
  2. Compiled:  torch.compile(RMSNorm + SiLU) → TE quantize  (2 kernels)
  3. V7 fused:  Single fused kernel (RMSNorm+SiLU+Quant)
  4. TE-only:   TE quantize only (no norm/act) — lower bound reference
  5. inv_rms + TE+SiLU:  Precompute inv_rms → TE quantize w/ SiLU activation

Also validates correctness: V7 fused vs eager reference (cosine similarity + max error).

Usage: python bench_fused_rmsnorm.py
"""

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import NVFP4Quantizer
import time
import sys

CSRC = '/workspace/fp4_matmul/fused_ops/csrc'
FL = ['-std=c++20', '-O3', '-lineinfo', '--expt-relaxed-constexpr',
      '-gencode=arch=compute_100a,code=sm_100a']

print('Compiling V7 kernel...', flush=True)
v7 = load(name='fused_te_quant_v7',
    sources=[CSRC+'/fused_te_quant_v7_torch.cpp', CSRC+'/fused_te_quant_v7.cu'],
    extra_include_paths=[CSRC], extra_cuda_cflags=FL, verbose=False)
print('Done.\n', flush=True)

td = tex.DType.kFloat4E2M1
dev = 'cuda'


# =========================================================================
# Helpers
# =========================================================================

def bench(fn, warmup=20, steps=50):
    """Time a function in milliseconds (GPU-side)."""
    for _ in range(warmup): fn(); torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(); s.record()
    for _ in range(steps): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / steps


def dequant_fp4(fp4_bytes, scales_fp8, gs, m, k):
    """Dequantize FP4 packed bytes → float32 for correctness checks."""
    lut = torch.tensor([0,0.5,1,1.5,2,3,4,6,-0,-0.5,-1,-1.5,-2,-3,-4,-6],
                       device=fp4_bytes.device, dtype=torch.float32)
    d = fp4_bytes.view(torch.uint8).to(torch.int32)
    u = torch.stack((d & 0x0F, d >> 4), dim=-1).reshape(m, k)
    fv = lut[u]
    sc = scales_fp8.view(torch.float8_e4m3fn).to(torch.float32)[:m,:k//16]
    return (fv.view(-1,16) * gs * sc.reshape(-1,1)).view(m, k)


def compute_inv_rms(x, eps=1e-5):
    """Compute inv_rms = 1/sqrt(mean(x^2, dim=-1) + eps)."""
    return torch.rsqrt(x.float().pow(2).mean(dim=-1) + eps)


# =========================================================================
# Part 1: Performance comparison across tensor sizes
# =========================================================================
configs = [
    (1024,  8192),
    (4096,  8192),
    (8192,  8192),
    (16384, 8192),
    (4096,  16384),
    (8192,  16384),
    (16384, 16384),
    (8192,  32768),
    (16384, 32768),
    (32768, 8192),
    (32768, 16384),
    (32768, 32768),
]

print(f'GPU: {torch.cuda.get_device_name()}')
print(f'\n{"="*100}')
print(f'  PERFORMANCE: Full Pipeline (RMSNorm + SiLU + NVFP4 Quant)')
print(f'{"="*100}')
hdr = (f'{"M":>8} {"K":>8} | {"eager":>9} {"compile":>9} {"V7fused":>9} '
       f'{"TEonly":>9} | {"V7/eagr":>8} {"V7/comp":>8} {"V7/TEq":>8}')
print(hdr)
print('-' * 100)

results = []
for M, K in configs:
    try:
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        w = torch.ones(K, device=dev, dtype=torch.bfloat16)
        rms = torch.nn.RMSNorm(K, eps=1e-5, device=dev, dtype=torch.bfloat16)
        with torch.no_grad(): rms.weight.fill_(1.0)

        q = NVFP4Quantizer(fp4_dtype=td, rowwise=True, columnwise=False)
        xq = q.make_empty((M, K), dtype=torch.bfloat16, device=dev)

        with torch.no_grad():
            h_pre = F.silu(rms(x))

            # Approach 1: Eager
            def pipeline_eager():
                h = F.silu(rms(x))
                q.update_quantized(h, xq)
            t_eager = bench(pipeline_eager)

            # Approach 2: torch.compile
            try:
                @torch.compile(mode='max-autotune', fullgraph=True)
                def compiled_norm_act(inp, norm):
                    return F.silu(norm(inp))
                for _ in range(5):
                    torch.compiler.cudagraph_mark_step_begin()
                    _ = compiled_norm_act(x, rms)
                torch.cuda.synchronize()

                def pipeline_compiled():
                    torch.compiler.cudagraph_mark_step_begin()
                    h = compiled_norm_act(x, rms)
                    q.update_quantized(h, xq)
                t_compiled = bench(pipeline_compiled)
                del compiled_norm_act
            except Exception:
                t_compiled = float('nan')

            # Approach 3: V7 fused
            t_v7 = bench(lambda: v7.forward_full(x, w, 1e-5, 0, 0, 0))

            # Approach 4: TE quant only (reference)
            t_te = bench(lambda: q.update_quantized(h_pre, xq))

        # Speedups
        sp_eager = t_eager / t_v7
        sp_comp = t_compiled / t_v7
        sp_te = t_v7 / t_te

        results.append((M, K, t_eager, t_compiled, t_v7, t_te, sp_eager, sp_comp, sp_te))
        print(f'{M:>8} {K:>8} | {t_eager:>8.3f}ms {t_compiled:>8.3f}ms {t_v7:>8.3f}ms '
              f'{t_te:>8.3f}ms | {sp_eager:>7.2f}x {sp_comp:>7.2f}x {sp_te:>7.2f}x', flush=True)

        del x, w, h_pre, rms, xq, q
        torch.cuda.empty_cache()
    except Exception as e:
        print(f'{M:>8} {K:>8} | ERROR: {e}', flush=True)
        import traceback; traceback.print_exc()
        torch.cuda.empty_cache()


# =========================================================================
# Part 2: Breakdown — component timings
# =========================================================================
print(f'\n{"="*100}')
print(f'  BREAKDOWN: Component Timings')
print(f'{"="*100}')
print(f'{"M":>8} {"K":>8} | {"RMSNorm":>9} {"SiLU":>9} {"inv_rms":>9} {"TE quant":>9} '
      f'{"sum":>9} | {"V7 fused":>9} {"savings":>8}')
print('-' * 100)

for M, K in [(4096,8192), (8192,16384), (16384,16384), (32768,32768)]:
    try:
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        w = torch.ones(K, device=dev, dtype=torch.bfloat16)
        rms = torch.nn.RMSNorm(K, eps=1e-5, device=dev, dtype=torch.bfloat16)
        with torch.no_grad(): rms.weight.fill_(1.0)
        q = NVFP4Quantizer(fp4_dtype=td, rowwise=True, columnwise=False)
        xq = q.make_empty((M, K), dtype=torch.bfloat16, device=dev)

        with torch.no_grad():
            h = F.silu(rms(x))
            h_normed = rms(x)

            # Component timings
            t_norm = bench(lambda: rms(x))
            t_silu = bench(lambda: F.silu(h_normed))
            t_inv_rms = bench(lambda: compute_inv_rms(x))
            t_quant = bench(lambda: q.update_quantized(h, xq))
            t_sum = t_norm + t_silu + t_quant
            t_v7 = bench(lambda: v7.forward_full(x, w, 1e-5, 0, 0, 0))
            savings = (1.0 - t_v7 / t_sum) * 100

        print(f'{M:>8} {K:>8} | {t_norm:>8.3f}ms {t_silu:>8.3f}ms {t_inv_rms:>8.3f}ms '
              f'{t_quant:>8.3f}ms {t_sum:>8.3f}ms | {t_v7:>8.3f}ms {savings:>6.1f}%', flush=True)
        del x, w, h, h_normed, rms, xq, q; torch.cuda.empty_cache()
    except Exception as e:
        print(f'{M:>8} {K:>8} | ERROR: {e}', flush=True)
        torch.cuda.empty_cache()


# =========================================================================
# Part 3: Correctness validation
# =========================================================================
print(f'\n{"="*100}')
print(f'  CORRECTNESS: V7 fused vs Reference (Eager RMSNorm + SiLU)')
print(f'{"="*100}')

for M, K in [(1024,8192), (4096,8192), (8192,16384), (16384,16384)]:
    try:
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        w = torch.ones(K, device=dev, dtype=torch.bfloat16)
        rms = torch.nn.RMSNorm(K, eps=1e-5, device=dev, dtype=torch.bfloat16)
        with torch.no_grad():
            rms.weight.fill_(1.0)

            # Reference: eager
            ref = F.silu(rms(x))

            # V7: fused
            f7, s7, g7, irms = v7.forward_full(x, w, 1e-5, 0, 0, 0)
            d7 = dequant_fp4(f7, s7, g7.item(), M, K)

            # Compute metrics
            cos = F.cosine_similarity(ref.flatten().float().unsqueeze(0),
                                       d7.flatten().unsqueeze(0)).item()
            abs_err = (ref.float() - d7).abs()
            max_err = abs_err.max().item()
            mean_err = abs_err.mean().item()
            rel_err = (abs_err / (ref.float().abs() + 1e-8)).mean().item()

            # Also check inv_rms
            inv_rms_ref = compute_inv_rms(x)
            irms_err = (inv_rms_ref - irms).abs().max().item()

        print(f'  [{M:>5}x{K:>5}] cos={cos:.6f}  max_err={max_err:.4f}  '
              f'mean_err={mean_err:.6f}  rel_err={rel_err:.4f}  inv_rms_err={irms_err:.2e}')
        del x, w, ref, f7, s7, g7, d7, irms; torch.cuda.empty_cache()
    except Exception as e:
        print(f'  [{M:>5}x{K:>5}] ERROR: {e}')
        torch.cuda.empty_cache()


# =========================================================================
# Part 4: All norm modes + scale modes
# =========================================================================
print(f'\n{"="*100}')
print(f'  ALL MODES: V7 Performance Across Norm/Act/Scale Modes (8192x16384)')
print(f'{"="*100}')

M, K = 8192, 16384
x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
w = torch.ones(K, device=dev, dtype=torch.bfloat16)

norm_names = ['RMS', 'AbsMax', 'MXNorm']
act_names = ['SiLU', 'GeLU', 'Identity']
scale_names = ['decode', 'encode']

print(f'{"Norm":>8} {"Act":>10} {"Scale":>8} | {"time(ms)":>10}')
print('-' * 50)

for nm, nm_name in enumerate(norm_names):
    for am, am_name in enumerate(act_names):
        for sm, sm_name in enumerate(scale_names):
            try:
                t = bench(lambda nm_=nm, am_=am, sm_=sm: v7.forward_full(x, w, 1e-5, nm_, am_, sm_))
                print(f'{nm_name:>8} {am_name:>10} {sm_name:>8} | {t:>9.3f}ms', flush=True)
            except Exception as e:
                print(f'{nm_name:>8} {am_name:>10} {sm_name:>8} | ERROR: {e}')

del x, w; torch.cuda.empty_cache()

print(f'\n{"="*100}')
print(f'  Summary')
print(f'{"="*100}')
if results:
    avg_speedup = sum(r[6] for r in results) / len(results)
    print(f'  Average speedup V7/eager: {avg_speedup:.2f}x')
    print(f'  V7 fused kernel eliminates BF16 memory round-trips between RMSNorm, SiLU, and Quantize.')
    print(f'  The nvfp4_transpose_fused.cuh (TE TMA-based) is also available for TE rebuild integration.')
print()
