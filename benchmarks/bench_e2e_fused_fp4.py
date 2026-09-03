"""
bench_e2e_fused_fp4.py — End-to-End: Fused(RMSNorm+SiLU+FP4Quant)+FP4GEMM vs BF16

Compares 4 pipelines for a single linear layer forward pass:
  A) BF16 Eager:      RMSNorm → SiLU → BF16 GEMM
  B) BF16 Compiled:   RMSNorm → SiLU → BF16 GEMM  (torch.compile)
  C) TE Separate:     RMSNorm → SiLU → tex.quantize → FP4 GEMM
  D) Fused 2-pass:    fused(RMS+SiLU+quant) → FP4 GEMM  (our kernel)

All compute:  y = SiLU(RMSNorm(x)) @ W.T
where x is [M, K] activations and W is [N, K] weight matrix.

Usage:
    python3 benchmarks/bench_e2e_fused_fp4.py
    python3 benchmarks/bench_e2e_fused_fp4.py --m=4096 --k=8192 --n=8192
"""
import os, sys, time, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------- Build fused TE kernel --------
print("Compiling fused TE extension...", flush=True)
from torch.utils.cpp_extension import load
TE_ROOT = '/workspace/low-bits-training/TransformerEngine'
TE_INCLUDE = os.path.join(TE_ROOT, 'transformer_engine/common/include')
TE_LIB_DIR = os.path.join(TE_ROOT, 'build/cmake')
CSRC = '/workspace/fp4_matmul/fused_ops/csrc'
cuda_lib = '/usr/local/cuda/lib64'

te_fused = load(name='te_fused_rmsnorm_ext',
    sources=[
        os.path.join(CSRC, 'te_fused_rmsnorm_ext.cpp'),
        os.path.join(CSRC, 'te_fused_pass1.cu'),
    ],
    extra_include_paths=[TE_INCLUDE, '/usr/local/cuda/include'],
    extra_cflags=['-std=c++17'],
    extra_cuda_cflags=['-std=c++17', '--expt-relaxed-constexpr', '-O3'],
    extra_ldflags=[f'-L{TE_LIB_DIR}', '-ltransformer_engine', f'-Wl,-rpath,{TE_LIB_DIR}',
                   f'-L{cuda_lib}', '-lcudart', '-lnvrtc', f'-Wl,-rpath,{cuda_lib}'],
    verbose=False)
print("Compiled.\n", flush=True)

# -------- TE imports --------
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch.constants import TE_DType
from transformer_engine.pytorch import NVFP4Quantizer

# -------- Timing --------
def time_fn(fn, steps=200, warmup=50):
    """Time function using CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(steps):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / steps

# ====================================================================
# A) BF16 Eager
# ====================================================================
def setup_bf16_eager(m, k, n):
    x = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
    w = torch.ones(k, device='cuda', dtype=torch.bfloat16)
    W = torch.randn(n, k, device='cuda', dtype=torch.bfloat16)
    rms = nn.RMSNorm(k, eps=1e-5, device='cuda', dtype=torch.bfloat16)
    def fn():
        h = F.silu(rms(x))
        return h @ W.T
    return fn

# ====================================================================
# B) BF16 Compiled
# ====================================================================
def setup_bf16_compiled(m, k, n):
    x = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
    w = torch.ones(k, device='cuda', dtype=torch.bfloat16)
    W = torch.randn(n, k, device='cuda', dtype=torch.bfloat16)
    rms = nn.RMSNorm(k, eps=1e-5, device='cuda', dtype=torch.bfloat16)
    @torch.compile(mode="reduce-overhead")
    def compiled_fn(x_in):
        h = F.silu(rms(x_in))
        return h @ W.T
    # Warmup compile
    for _ in range(5):
        compiled_fn(x)
    torch.cuda.synchronize()
    return lambda: compiled_fn(x)

# ====================================================================
# C) TE Separate: RMSNorm → SiLU → tex.quantize → FP4 GEMM
# ====================================================================
def setup_te_separate(m, k, n):
    x = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
    W = torch.randn(n, k, device='cuda', dtype=torch.bfloat16)
    rms = nn.RMSNorm(k, eps=1e-5, device='cuda', dtype=torch.bfloat16)

    te_dtype = tex.DType.kFloat4E2M1
    xq = NVFP4Quantizer(fp4_dtype=te_dtype, rowwise=True, columnwise=True)
    wq = NVFP4Quantizer(fp4_dtype=te_dtype, rowwise=True, columnwise=True)
    x_nvfp4 = xq.make_empty((m, k), dtype=torch.bfloat16, device='cuda')
    w_nvfp4 = wq.make_empty((n, k), dtype=torch.bfloat16, device='cuda')

    # Pre-quantize weights (not timed)
    wq.update_quantized(W, w_nvfp4)

    out = torch.empty(m, n, device='cuda', dtype=torch.bfloat16)
    workspace = torch.empty(4, dtype=torch.uint8, device='cuda')
    out_dtype = TE_DType[torch.bfloat16]
    bias_dtype = TE_DType[torch.bfloat16]

    def fn():
        h = F.silu(rms(x))
        xq.update_quantized(h, x_nvfp4)
        tex.generic_gemm(
            w_nvfp4, True, x_nvfp4, False,
            out, None, out_dtype,
            None, bias_dtype,
            False, None, False, workspace,
            workspace.shape[0], False, False,
        )
        return out
    return fn

# ====================================================================
# D) Fused 2-pass: fused(RMS+SiLU+quant) → FP4 GEMM
# ====================================================================
def setup_fused_2pass(m, k, n):
    x = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
    w_norm = torch.ones(k, device='cuda', dtype=torch.bfloat16)
    W = torch.randn(n, k, device='cuda', dtype=torch.bfloat16)

    te_dtype = tex.DType.kFloat4E2M1
    wq = NVFP4Quantizer(fp4_dtype=te_dtype, rowwise=True, columnwise=True)
    w_nvfp4 = wq.make_empty((n, k), dtype=torch.bfloat16, device='cuda')
    wq.update_quantized(W, w_nvfp4)

    # Create an empty NVFP4 tensor for activations — we'll overwrite its internals
    xq = NVFP4Quantizer(fp4_dtype=te_dtype, rowwise=True, columnwise=False)
    x_nvfp4 = xq.make_empty((m, k), dtype=torch.bfloat16, device='cuda')

    out = torch.empty(m, n, device='cuda', dtype=torch.bfloat16)
    workspace = torch.empty(4, dtype=torch.uint8, device='cuda')
    out_dtype = TE_DType[torch.bfloat16]
    bias_dtype = TE_DType[torch.bfloat16]

    def fn():
        # Fused Pass1 + Pass2: computes inv_rms, amax, then quantizes
        fp4_data, scale_inv, inv_rms, amax = te_fused.fused_te_quantize_rmsnorm_silu_2pass(
            x, w_norm, 1e-5
        )
        # Inject into TE tensor storage
        x_nvfp4._rowwise_data = fp4_data
        x_nvfp4._rowwise_scale_inv = scale_inv
        x_nvfp4._amax_rowwise = amax

        # FP4 GEMM
        tex.generic_gemm(
            w_nvfp4, True, x_nvfp4, False,
            out, None, out_dtype,
            None, bias_dtype,
            False, None, False, workspace,
            workspace.shape[0], False, False,
        )
        return out
    return fn

# ====================================================================
# E) Fused kernel-only (amax given) + FP4 GEMM  
# ====================================================================
def setup_fused_kernel_only(m, k, n):
    x = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
    w_norm = torch.ones(k, device='cuda', dtype=torch.bfloat16)
    W = torch.randn(n, k, device='cuda', dtype=torch.bfloat16)
    eps = 1e-5

    te_dtype = tex.DType.kFloat4E2M1
    wq = NVFP4Quantizer(fp4_dtype=te_dtype, rowwise=True, columnwise=True)
    w_nvfp4 = wq.make_empty((n, k), dtype=torch.bfloat16, device='cuda')
    wq.update_quantized(W, w_nvfp4)

    xq = NVFP4Quantizer(fp4_dtype=te_dtype, rowwise=True, columnwise=False)
    x_nvfp4 = xq.make_empty((m, k), dtype=torch.bfloat16, device='cuda')

    # Pre-compute stats (simulates delayed scaling from previous iteration)
    inv_rms = torch.rsqrt(x.float().pow(2).mean(dim=-1) + eps)
    golden = F.silu(x.float() * inv_rms.unsqueeze(-1) * w_norm.float()).to(torch.bfloat16)
    ref_qt = NVFP4Quantizer(rowwise=True, columnwise=False).quantize(golden)
    amax = ref_qt._amax_rowwise.clone()

    out = torch.empty(m, n, device='cuda', dtype=torch.bfloat16)
    workspace = torch.empty(4, dtype=torch.uint8, device='cuda')
    out_dtype = TE_DType[torch.bfloat16]
    bias_dtype = TE_DType[torch.bfloat16]

    def fn():
        fp4_data, scale_inv = te_fused.fused_te_quantize_rmsnorm_silu(
            x, inv_rms, w_norm, amax
        )
        x_nvfp4._rowwise_data = fp4_data
        x_nvfp4._rowwise_scale_inv = scale_inv
        x_nvfp4._amax_rowwise = amax

        tex.generic_gemm(
            w_nvfp4, True, x_nvfp4, False,
            out, None, out_dtype,
            None, bias_dtype,
            False, None, False, workspace,
            workspace.shape[0], False, False,
        )
        return out
    return fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m-sizes", type=str, default="1024,2048,4096,8192,16384")
    parser.add_argument("--k", type=int, default=8192)
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    m_sizes = [int(s) for s in args.m_sizes.split(",")]
    k, n = args.k, args.n

    gpu = torch.cuda.get_device_name()
    print("=" * 140)
    print(f"  E2E Pipeline: Fused(RMSNorm+SiLU+FP4Quant)+GEMM vs BF16 GEMM")
    print(f"  GPU: {gpu}  |  K={k}, N={n}  |  Steps={args.steps}, Warmup={args.warmup}")
    print("  Pipelines:")
    print("    A) BF16 Eager:    RMSNorm → SiLU → BF16 GEMM")
    if not args.no_compile:
        print("    B) BF16 Compiled: RMSNorm → SiLU → BF16 GEMM (torch.compile)")
    print("    C) TE Separate:   RMSNorm → SiLU → tex.quantize → FP4 GEMM")
    print("    D) Fused 2-pass:  fused(RMS+SiLU+FP4quant) → FP4 GEMM")
    print("    E) Fused kern:    fused_kernel(amax given) → FP4 GEMM  [delayed scaling]")
    print("=" * 140)

    flops_per_gemm = 2.0 * n * k  # per row

    header = f"{'M':>8} |"
    header += f" {'BF16(A)':>9} {'TFLOP':>6} |"
    if not args.no_compile:
        header += f" {'BF16cc(B)':>9} {'TFLOP':>6} |"
    header += f" {'TESep(C)':>9} {'TFLOP':>6} |"
    header += f" {'Fus2p(D)':>9} {'TFLOP':>6} |"
    header += f" {'FusKn(E)':>9} {'TFLOP':>6} |"
    header += f" {'D/A':>6} {'D/C':>6} {'E/A':>6}"
    print(f"\n{header}")
    print("-" * 140)

    for m in m_sizes:
        total_flops = flops_per_gemm * m
        results = {}

        # A) BF16 Eager
        fn_a = setup_bf16_eager(m, k, n)
        ms_a = time_fn(fn_a, args.steps, args.warmup)
        results['A'] = ms_a

        # B) BF16 Compiled
        if not args.no_compile:
            try:
                fn_b = setup_bf16_compiled(m, k, n)
                ms_b = time_fn(fn_b, args.steps, args.warmup)
            except Exception as e:
                ms_b = ms_a  # fallback
                print(f"  [WARN] compile failed M={m}: {e}")
            results['B'] = ms_b

        # C) TE Separate
        fn_c = setup_te_separate(m, k, n)
        ms_c = time_fn(fn_c, args.steps, args.warmup)
        results['C'] = ms_c

        # D) Fused 2-pass
        fn_d = setup_fused_2pass(m, k, n)
        ms_d = time_fn(fn_d, args.steps, args.warmup)
        results['D'] = ms_d

        # E) Fused kernel-only (amax given)
        fn_e = setup_fused_kernel_only(m, k, n)
        ms_e = time_fn(fn_e, args.steps, args.warmup)
        results['E'] = ms_e

        def tflops(ms):
            return total_flops / (ms * 1e-3) / 1e12

        row = f"{m:>8} |"
        row += f" {ms_a:>7.3f}ms {tflops(ms_a):>6.1f} |"
        if not args.no_compile:
            row += f" {ms_b:>7.3f}ms {tflops(ms_b):>6.1f} |"
        row += f" {ms_c:>7.3f}ms {tflops(ms_c):>6.1f} |"
        row += f" {ms_d:>7.3f}ms {tflops(ms_d):>6.1f} |"
        row += f" {ms_e:>7.3f}ms {tflops(ms_e):>6.1f} |"
        row += f" {ms_a/ms_d:>5.2f}x {ms_c/ms_d:>5.2f}x {ms_a/ms_e:>5.2f}x"
        print(row)

    print()
    print("Ratios:  D/A = Fused-2pass vs BF16   (>1 = FP4 wins)")
    print("         D/C = Fused vs TE-separate   (>1 = fusion helps)")
    print("         E/A = Fused-kern vs BF16     (>1 = delayed-scaling FP4 wins)")
    print()
    print("TFLOP = effective throughput: 2*M*N*K / time  (higher = better)")

if __name__ == "__main__":
    main()
