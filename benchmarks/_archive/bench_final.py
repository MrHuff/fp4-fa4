"""
Final benchmark: Full pipeline (RMSNorm + SiLU + NVFP4 quant)

Compares:
  1. torch.compile(rmsnorm + silu) → TE quant
  2. Eager (rmsnorm + silu) → TE quant  
  3. V7 fused kernel (single call)
  4. TE quant only (no norm/act) — reference
"""
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import NVFP4Quantizer

CSRC = '/workspace/fp4_matmul/fused_ops/csrc'
FL = ['-std=c++20', '-O3', '-lineinfo', '--expt-relaxed-constexpr',
      '-gencode=arch=compute_100a,code=sm_100a']

print('Compiling V7...', flush=True)
v7 = load(name='fused_te_quant_v7',
    sources=[CSRC+'/fused_te_quant_v7_torch.cpp', CSRC+'/fused_te_quant_v7.cu'],
    extra_include_paths=[CSRC], extra_cuda_cflags=FL, verbose=False)
print('Done.\n', flush=True)

td = tex.DType.kFloat4E2M1

def bench(fn, warmup=20, steps=50):
    for _ in range(warmup): fn(); torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(); s.record()
    for _ in range(steps): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / steps

dev = 'cuda'

configs = [
    (1024, 8192), (4096, 8192), (8192, 8192), (16384, 8192),
    (4096, 16384), (8192, 16384), (16384, 16384),
    (8192, 32768), (16384, 32768),
    (32768, 8192), (32768, 16384), (32768, 32768),
]

# ============================================================
# Part 1: torch.compile with proper no_grad
# ============================================================
print(f'GPU: {torch.cuda.get_device_name()}')
print(f'\n=== Full Pipeline: RMSNorm + SiLU + NVFP4 Quant ===')
print(f'{"M":>8} {"K":>8} | {"comp+TE":>9} {"eag+TE":>9} {"V7":>9} {"TE only":>9} | {"speedup":>8} {"vs TE":>8}')
print(f'{"":>8} {"":>8} | {"norm+act":>9} {"":>9} {"fused":>9} {"quant":>9} | {"V7/eag":>8} {"V7/TEq":>8}')
print('-'*92)

for M, K in configs:
    try:
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        w = torch.ones(K, device=dev, dtype=torch.bfloat16)
        rms = torch.nn.RMSNorm(K, eps=1e-5, device=dev, dtype=torch.bfloat16)
        with torch.no_grad(): rms.weight.fill_(1.0)

        q = NVFP4Quantizer(fp4_dtype=td, rowwise=True, columnwise=False)
        xq = q.make_empty((M, K), dtype=torch.bfloat16, device=dev)

        # Compiled norm+silu — must use torch.no_grad() and fresh compile per shape
        @torch.compile(mode='max-autotune', fullgraph=True)
        def compiled_norm_act(inp, norm):
            return F.silu(norm(inp))

        with torch.no_grad():
            # Warmup compile (triggers codegen)
            for _ in range(5):
                torch.compiler.cudagraph_mark_step_begin()
                _ = compiled_norm_act(x, rms)
            torch.cuda.synchronize()

            # Approach 1: compiled(norm+silu) → TE quant
            def pipeline_compiled():
                torch.compiler.cudagraph_mark_step_begin()
                h = compiled_norm_act(x, rms)
                q.update_quantized(h, xq)
            t_compiled = bench(pipeline_compiled)

            # Approach 2: eager norm+silu → TE quant
            def pipeline_eager():
                h = F.silu(rms(x))
                q.update_quantized(h, xq)
            t_eager = bench(pipeline_eager)

            # Approach 3: V7 fused
            t_v7 = bench(lambda: v7.forward_full(x, w, 1e-5, 0, 0, 0))

            # Reference: TE quant only
            h_pre = F.silu(rms(x))
            t_te = bench(lambda: q.update_quantized(h_pre, xq))

        # speedup = eager/V7 (>1 means V7 is faster)
        speedup = t_eager / t_v7
        vs_te = t_v7 / t_te  # how much slower V7 is vs TE-only (V7 does more work)

        print(f'{M:>8} {K:>8} | {t_compiled:>8.3f}ms {t_eager:>8.3f}ms {t_v7:>8.3f}ms {t_te:>8.3f}ms '
              f'| {speedup:>7.2f}x {vs_te:>7.2f}x', flush=True)
        del x, w, h_pre, rms, xq, q, compiled_norm_act; torch.cuda.empty_cache()
    except Exception as e:
        print(f'{M:>8} {K:>8} | ERROR: {e}', flush=True)
        import traceback; traceback.print_exc()
        torch.cuda.empty_cache()

# ============================================================
# Part 2: Breakdown — cost of norm+act vs quant vs fused
# ============================================================
print(f'\n=== Breakdown: Where does the time go? ===')
print(f'{"M":>8} {"K":>8} | {"norm+act":>9} {"TE quant":>9} {"sum":>9} | {"V7 fused":>9} | {"savings":>8}')
print('-'*80)

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
            # Measure norm+act alone (eager)
            t_norm = bench(lambda: F.silu(rms(x)))
            # Measure TE quant alone
            t_quant = bench(lambda: q.update_quantized(h, xq))
            t_sum = t_norm + t_quant
            # Measure V7 fused
            t_v7 = bench(lambda: v7.forward_full(x, w, 1e-5, 0, 0, 0))
            savings = (1.0 - t_v7 / t_sum) * 100

        print(f'{M:>8} {K:>8} | {t_norm:>8.3f}ms {t_quant:>8.3f}ms {t_sum:>8.3f}ms '
              f'| {t_v7:>8.3f}ms | {savings:>6.1f}%', flush=True)
        del x, w, h, rms, xq, q; torch.cuda.empty_cache()
    except Exception as e:
        print(f'{M:>8} {K:>8} | ERROR: {e}', flush=True)
        torch.cuda.empty_cache()
