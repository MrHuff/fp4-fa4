"""
Unified Benchmark: Fused RMSNorm + SiLU + NVFP4 Quantization (Forward & Backward)

Compares all relevant kernel variants against a common eager PyTorch baseline
with consistent timing methodology.

=== FORWARD ===
  Baseline:   torch.compile(RMSNorm + SiLU) + TE quantize (best without fused kernels)
  1. TE 2-pass:     Pass1=stats (te_fused_pass1.cu) + Pass2=TE kernel (nvfp4_transpose_fused.cuh)
  2. V7 fused:      Custom 2-pass (PTX mul+cvt, warp shuffles)
  3. TE kern-only:  TE kernel w/ pre-computed inv_rms+amax (lower bound)

=== BACKWARD ===
  1. Eager:         PyTorch autograd (F32 math)
  2. Fused CUDA:    fused_silu_rmsnorm_backward.cu (SiLU' + RMSNorm bwd + dgamma)

=== COMBINED ===
  Forward quant + GEMM + Backward (training-realistic wall-clock)

=== CORRECTNESS ===
  All variants vs eager reference (cosine similarity, max error)

Usage:
    LD_PRELOAD=.../libtransformer_engine.so python benchmarks/bench_unified_fused.py
"""

import os
import sys
import time
import argparse
import ctypes

# Pre-load TE dependencies then libtransformer_engine.so
# libtransformer_engine depends on libnvrtc + libcudart
for _dep in ['/usr/local/cuda/lib64/libnvrtc.so', '/usr/local/cuda/lib64/libcudart.so']:
    if os.path.exists(_dep):
        ctypes.CDLL(_dep, mode=ctypes.RTLD_GLOBAL)
_TE_LIB = os.path.join('/workspace/low-bits-training/TransformerEngine/build/cmake',
                        'libtransformer_engine.so')
if os.path.exists(_TE_LIB):
    ctypes.CDLL(_TE_LIB, mode=ctypes.RTLD_GLOBAL)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load

# ── Paths ──────────────────────────────────────────────────────────────
TE_ROOT = '/workspace/low-bits-training/TransformerEngine'
TE_INCLUDE = os.path.join(TE_ROOT, 'transformer_engine/common/include')
TE_LIB_DIR = os.path.join(TE_ROOT, 'build/cmake')
CSRC = '/workspace/fp4_matmul/fused_ops/csrc'
CUDA_LIB = '/usr/local/cuda/lib64'

# ── TE imports ─────────────────────────────────────────────────────────
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch.constants import TE_DType
from transformer_engine.pytorch import NVFP4Quantizer

te_dtype = tex.DType.kFloat4E2M1


# ── Compile extensions ─────────────────────────────────────────────────
def _compile_te_fused():
    print("Compiling TE-fused extension...", flush=True)
    ext = load(
        name='te_fused_rmsnorm_ext_bench',
        sources=[
            os.path.join(CSRC, 'te_fused_rmsnorm_ext.cpp'),
            os.path.join(CSRC, 'te_fused_pass1.cu'),
            os.path.join(CSRC, 'fused_silu_rmsnorm_backward.cu'),
        ],
        extra_include_paths=[TE_INCLUDE, '/usr/local/cuda/include', CSRC],
        extra_cflags=['-std=c++17'],
        extra_cuda_cflags=['-std=c++17', '--expt-relaxed-constexpr', '-O3'],
        extra_ldflags=[
            f'-L{TE_LIB_DIR}', '-ltransformer_engine', f'-Wl,-rpath,{TE_LIB_DIR}',
            f'-L{CUDA_LIB}', '-lcudart', '-lnvrtc', f'-Wl,-rpath,{CUDA_LIB}',
        ],
        verbose=False,
    )
    print("  ✓ TE-fused compiled.\n", flush=True)
    return ext


def _compile_v7():
    srcs = [os.path.join(CSRC, f) for f in ['fused_te_quant_v7.cu', 'fused_te_quant_v7_torch.cpp']]
    if not all(os.path.exists(s) for s in srcs):
        print("  V7 sources not found, skipping.\n")
        return None
    try:
        print("Compiling V7 kernel...", flush=True)
        ext = load(
            name='fused_te_quant_v7_bench',
            sources=srcs,
            extra_include_paths=[TE_INCLUDE, '/usr/local/cuda/include', CSRC],
            extra_cflags=['-std=c++17'],
            extra_cuda_cflags=[
                '-std=c++17', '--expt-relaxed-constexpr', '-O3',
                '-gencode=arch=compute_100a,code=sm_100a',
                f'-I{TE_INCLUDE}', f'-I{CSRC}',
            ],
            extra_ldflags=[
                f'-L{TE_LIB_DIR}', '-ltransformer_engine', f'-Wl,-rpath,{TE_LIB_DIR}',
                f'-L{CUDA_LIB}', '-lcudart', '-lnvrtc', f'-Wl,-rpath,{CUDA_LIB}',
            ],
            verbose=False,
        )
        print("  ✓ V7 compiled.\n", flush=True)
        return ext
    except Exception as e:
        print(f"  V7 compilation failed ({e.__class__.__name__}: {e}), skipping.\n")
        return None


# ── Timing utilities ───────────────────────────────────────────────────
def bench(fn, warmup=20, steps=100):
    """GPU-side timing in milliseconds using CUDA events."""
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


# ── FP4 dequantization (for correctness) ──────────────────────────────
FP4_LUT = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6]

def deq_fp4(fp4_bytes, sc_padded, amax_val, m, k):
    """Dequantize V7/custom FP4 packed bytes → float32."""
    lut = torch.tensor(FP4_LUT, device='cuda', dtype=torch.float32)
    d = fp4_bytes.view(torch.uint8).to(torch.int32)
    u = torch.stack((d & 0x0F, d >> 4), dim=-1).reshape(m, k)
    fv = lut[u]
    sc = sc_padded.view(torch.float8_e4m3fn).to(torch.float32)[:m, :k // 16]
    ts = amax_val / (6.0 * 448.0)
    return (fv.view(-1, 16) * ts * sc.reshape(-1, 1)).view(m, k)


# ── Reference implementations ─────────────────────────────────────────
def eager_forward(x, w, eps, quantizer):
    """Eager: RMSNorm → SiLU → TE quantize."""
    inv_rms = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    normed = x.float() * inv_rms * w.float()
    act = F.silu(normed).to(torch.bfloat16)
    return quantizer.quantize(act), inv_rms.squeeze(-1)


def eager_backward(grad_out, x_raw, w, inv_rms, eps):
    """Eager backward: SiLU' + RMSNorm backward + dgamma (F32 math)."""
    x_f = x_raw.float()
    w_f = w.float()
    inv_rms_u = inv_rms.unsqueeze(-1)
    x_norm = x_f * inv_rms_u * w_f
    sig = torch.sigmoid(x_norm)
    silu_grad = sig * (1 + x_norm * (1 - sig))
    d_y = grad_out.float() * silu_grad

    # dgamma = sum(d_y * x_hat)  where x_hat = x * inv_rms
    x_hat = x_f * inv_rms_u
    dgamma = (d_y * x_hat).sum(dim=0)

    # dx = inv_rms * (d_y * w - x_hat * mean(d_y * w * x_hat))
    d_z = d_y * w_f
    mean_term = (d_z * x_hat).mean(dim=-1, keepdim=True)
    dx = inv_rms_u * (d_z - x_hat * mean_term)
    return dx.to(torch.bfloat16), dgamma


# ══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Unified fused FP4 benchmark")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--skip-backward", action="store_true")
    parser.add_argument("--skip-combined", action="store_true")
    args = parser.parse_args()

    # Compile extensions
    te_fused = _compile_te_fused()
    v7_ext = _compile_v7()

    gpu = torch.cuda.get_device_name()
    dev = 'cuda'
    eps = 1e-5

    SIZES = [
        (4096,  2048),
        (4096,  8192),
        (8192,  8192),
        (8192,  16384),
        (16384, 8192),
        (16384, 16384),
        (32768, 8192),
    ]

    print("=" * 110)
    print(f"  Unified Fused FP4 Benchmark — {gpu}")
    print(f"  Warmup={args.warmup}  Steps={args.steps}")
    print("=" * 110)

    # ══════════════════════════════════════════════════════════════════
    # PART 1: FORWARD PERFORMANCE
    # ══════════════════════════════════════════════════════════════════
    # Baseline: torch.compile(RMSNorm + SiLU) + TE quantize
    # This is the best a user can do WITHOUT our fused kernels.
    print(f"\n{'=' * 110}")
    print("  FORWARD: RMSNorm + SiLU + NVFP4 Quantization")
    print("  Baseline = torch.compile(RMSNorm+SiLU) + TE quantize")
    print(f"{'=' * 110}")

    hdr = (f"{'M':>8} {'K':>8} | {'compiled':>9} {'TE-quant':>9} {'base(c+q)':>10} | "
           f"{'TE-2pass':>9} {'V7':>9} {'TEkern':>9} | "
           f"{'2p/base':>8} {'V7/base':>8}")
    print(hdr)
    print("-" * 120)

    fwd_results = []
    for M, K in SIZES:
        try:
            # Reset dynamo so each size gets a clean compile
            torch._dynamo.reset()

            x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
            w = torch.ones(K, device=dev, dtype=torch.bfloat16)
            rms = torch.nn.RMSNorm(K, eps=eps, device=dev, dtype=torch.bfloat16)
            with torch.no_grad():
                rms.weight.fill_(1.0)
            q = NVFP4Quantizer(fp4_dtype=te_dtype, rowwise=True, columnwise=False)

            with torch.no_grad():
                # Pre-compute reference for TE kernel-only
                inv_rms_ref = torch.rsqrt(x.float().pow(2).mean(dim=-1) + eps)
                ref_qt = q.quantize(F.silu(x.float() * inv_rms_ref.unsqueeze(-1) * w.float()).to(torch.bfloat16))
                ref_amax = ref_qt._amax_rowwise

                # ── Baseline: torch.compile(RMSNorm+SiLU) + TE quantize ──
                # Use reduce-overhead (not max-autotune which attempts cudagraphs and OOMs at large sizes)
                compiled_norm_act = torch.compile(
                    lambda inp, norm_mod: F.silu(norm_mod(inp)),
                    mode='reduce-overhead', fullgraph=True
                )
                # Warmup compile — these runs trigger compilation, NOT timed
                for _ in range(3):
                    _ = compiled_norm_act(x, rms)
                torch.cuda.synchronize()

                # Time compiled norm+act only (compilation already happened above)
                t_compiled = bench(lambda: compiled_norm_act(x, rms),
                                   args.warmup, args.steps)

                # Time TE quant only
                h_pre = F.silu(rms(x))
                t_te_quant = bench(lambda: q.quantize(h_pre), args.warmup, args.steps)

                # Time compiled + TE quant combined (the actual baseline)
                def fn_baseline():
                    h = compiled_norm_act(x, rms)
                    return q.quantize(h)
                t_baseline = bench(fn_baseline, args.warmup, args.steps)

                # ── Fused variants ──
                # TE 2-pass (production path)
                t_te2 = bench(lambda: te_fused.fused_te_quantize_rmsnorm_silu_2pass(x, w, eps),
                              args.warmup, args.steps)

                # V7 fused
                t_v7 = float('nan')
                if v7_ext is not None:
                    t_v7 = bench(lambda: v7_ext.forward_full(x, w, eps, 0, 0, 0),
                                 args.warmup, args.steps)

                # TE kernel-only (amax pre-computed — lower bound)
                t_tek = bench(lambda: te_fused.fused_te_quantize_rmsnorm_silu(x, inv_rms_ref, w, ref_amax),
                              args.warmup, args.steps)

            sp_te2 = t_baseline / t_te2
            sp_v7 = t_baseline / t_v7 if t_v7 == t_v7 else float('nan')
            fwd_results.append(dict(M=M, K=K, t_compiled=t_compiled, t_te_quant=t_te_quant,
                                    t_baseline=t_baseline, t_te2=t_te2, t_v7=t_v7, t_tek=t_tek,
                                    sp_te2=sp_te2, sp_v7=sp_v7))

            v7_str = f"{t_v7:>8.3f}ms" if t_v7 == t_v7 else f"{'n/a':>9}"
            v7_sp = f"{sp_v7:>7.2f}x" if sp_v7 == sp_v7 else f"{'n/a':>8}"
            print(f"{M:>8} {K:>8} | {t_compiled:>8.3f}ms {t_te_quant:>8.3f}ms {t_baseline:>9.3f}ms | "
                  f"{t_te2:>8.3f}ms {v7_str} {t_tek:>8.3f}ms | "
                  f"{sp_te2:>7.2f}x {v7_sp}", flush=True)

            del x, w, q, inv_rms_ref, ref_qt, ref_amax, rms
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"{M:>8} {K:>8} | ERROR: {e}", flush=True)
            import traceback; traceback.print_exc()
            torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════
    # PART 2: BACKWARD PERFORMANCE
    # ══════════════════════════════════════════════════════════════════
    if not args.skip_backward:
        print(f"\n{'=' * 110}")
        print("  BACKWARD: SiLU' + RMSNorm Backward + dgamma")
        print(f"{'=' * 110}")
        print(f"{'M':>8} {'K':>8} | {'eager':>9} {'fused':>9} | {'speedup':>8}")
        print("-" * 70)

        for M, K in SIZES:
            try:
                x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
                w = torch.ones(K, device=dev, dtype=torch.bfloat16)
                grad_out = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
                inv_rms = torch.rsqrt(x.float().pow(2).mean(dim=-1) + eps)

                with torch.no_grad():
                    # 1. Eager backward
                    def fn_eager_bwd():
                        return eager_backward(grad_out, x, w, inv_rms, eps)
                    t_eager_bwd = bench(fn_eager_bwd, args.warmup, args.steps)

                    # 2. Fused CUDA backward
                    def fn_fused_bwd():
                        return te_fused.fused_silu_rmsnorm_backward(grad_out, x, w, inv_rms)
                    t_fused_bwd = bench(fn_fused_bwd, args.warmup, args.steps)

                sp = t_eager_bwd / t_fused_bwd
                print(f"{M:>8} {K:>8} | {t_eager_bwd:>8.3f}ms {t_fused_bwd:>8.3f}ms | {sp:>7.2f}x",
                      flush=True)

                del x, w, grad_out, inv_rms
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"{M:>8} {K:>8} | ERROR: {e}", flush=True)
                torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════
    # PART 3: COMBINED FORWARD + BACKWARD (training-realistic)
    # ══════════════════════════════════════════════════════════════════
    if not args.skip_combined:
        print(f"\n{'=' * 110}")
        print("  COMBINED: Forward Quant + GEMM + Backward (training-realistic)")
        print(f"{'=' * 110}")
        print(f"{'M':>8} {'K':>8} | {'eager(F+B)':>11} {'fused(F+B)':>11} {'GEMM':>9} | {'speedup':>8} {'% saved':>8}")
        print("-" * 90)

        for M, K in SIZES:
            N = K  # square GEMM for simplicity
            try:
                x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
                w_norm = torch.ones(K, device=dev, dtype=torch.bfloat16)
                grad_out = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
                q = NVFP4Quantizer(fp4_dtype=te_dtype, rowwise=True, columnwise=False)

                with torch.no_grad():
                    inv_rms_ref = torch.rsqrt(x.float().pow(2).mean(dim=-1) + eps)

                    # Eager combined: forward + backward
                    def fn_eager_combined():
                        ir = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
                        n = x.float() * ir * w_norm.float()
                        a = F.silu(n).to(torch.bfloat16)
                        _ = q.quantize(a)
                        # backward
                        return eager_backward(grad_out, x, w_norm, ir.squeeze(-1), eps)
                    t_eager_fb = bench(fn_eager_combined, args.warmup, args.steps)

                    # Fused combined: forward + backward
                    def fn_fused_combined():
                        _ = te_fused.fused_te_quantize_rmsnorm_silu_2pass(x, w_norm, eps)
                        return te_fused.fused_silu_rmsnorm_backward(grad_out, x, w_norm, inv_rms_ref)
                    t_fused_fb = bench(fn_fused_combined, args.warmup, args.steps)

                    # GEMM only (for context)
                    wt = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
                    xq = NVFP4Quantizer(fp4_dtype=te_dtype, rowwise=True, columnwise=True)
                    wq = NVFP4Quantizer(fp4_dtype=te_dtype, rowwise=True, columnwise=True)
                    x_nvfp4 = xq.make_empty((M, K), dtype=torch.bfloat16, device=dev)
                    w_nvfp4 = wq.make_empty((N, K), dtype=torch.bfloat16, device=dev)
                    xq.update_quantized(x, x_nvfp4)
                    wq.update_quantized(wt, w_nvfp4)
                    out_gemm = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
                    workspace = torch.empty(4, dtype=torch.uint8, device=dev)
                    out_dt = TE_DType[torch.bfloat16]
                    bias_dt = TE_DType[torch.bfloat16]

                    def fn_gemm():
                        tex.generic_gemm(
                            w_nvfp4, True, x_nvfp4, False,
                            out_gemm, None, out_dt,
                            None, bias_dt,
                            False, None, False, workspace,
                            workspace.shape[0], False, False,
                        )
                    t_gemm = bench(fn_gemm, args.warmup, args.steps)

                sp = t_eager_fb / t_fused_fb
                pct = (1 - t_fused_fb / t_eager_fb) * 100
                print(f"{M:>8} {K:>8} | {t_eager_fb:>10.3f}ms {t_fused_fb:>10.3f}ms {t_gemm:>8.3f}ms | "
                      f"{sp:>7.2f}x {pct:>7.1f}%", flush=True)

                del x, w_norm, grad_out, wt, q, xq, wq
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"{M:>8} {K:>8} | ERROR: {e}", flush=True)
                torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════
    # PART 4: CORRECTNESS VALIDATION
    # ══════════════════════════════════════════════════════════════════
    if not args.skip_correctness:
        print(f"\n{'=' * 110}")
        print("  CORRECTNESS: All Forward Variants vs Eager Reference")
        print(f"{'=' * 110}")

        for M, K in [(1024, 8192), (4096, 8192), (8192, 16384)]:
            torch.manual_seed(42)
            x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
            w = torch.ones(K, device=dev, dtype=torch.bfloat16)

            with torch.no_grad():
                # Reference
                inv_rms = torch.rsqrt(x.float().pow(2).mean(dim=-1) + eps)
                normed = x.float() * inv_rms.unsqueeze(-1) * w.float()
                ref = F.silu(normed).to(torch.bfloat16)
                q = NVFP4Quantizer(rowwise=True, columnwise=False)
                ref_qt = q.quantize(ref)
                ref_deq = ref_qt.dequantize(dtype=torch.float32)
                ref_amax = ref_qt._amax_rowwise

                print(f"\n  [{M:>5}x{K:>5}]")

                # TE 2-pass
                fp4, sc, irms, amax = te_fused.fused_te_quantize_rmsnorm_silu_2pass(x, w, eps)
                deq = deq_fp4(fp4, sc, amax.item(), M, K)
                cos = F.cosine_similarity(deq.flatten().unsqueeze(0),
                                          ref_deq.flatten().unsqueeze(0)).item()
                irms_cos = F.cosine_similarity(inv_rms.unsqueeze(0),
                                               irms.unsqueeze(0)).item()
                print(f"    TE-2pass    cos={cos:.6f}  inv_rms_cos={irms_cos:.6f}  "
                      f"amax_err={abs(ref_amax.item()-amax.item()):.4f}")

                # TE kernel-only
                fp4k, sck = te_fused.fused_te_quantize_rmsnorm_silu(x, inv_rms, w, ref_amax)
                deqk = deq_fp4(fp4k, sck, ref_amax.item(), M, K)
                cos_k = F.cosine_similarity(deqk.flatten().unsqueeze(0),
                                            ref_deq.flatten().unsqueeze(0)).item()
                print(f"    TE-kern     cos={cos_k:.6f}")

                # V7
                if v7_ext is not None:
                    f7, s7, g7, irms7 = v7_ext.forward_full(x, w, eps, 0, 0, 0)
                    d7 = deq_fp4(f7, s7, g7.item(), M, K)
                    cos_v7 = F.cosine_similarity(d7.flatten().unsqueeze(0),
                                                 ref_deq.flatten().unsqueeze(0)).item()
                    print(f"    V7          cos={cos_v7:.6f}")

            del x, w
            torch.cuda.empty_cache()

        # Backward correctness
        print(f"\n{'=' * 110}")
        print("  CORRECTNESS: Fused Backward vs Eager Reference")
        print(f"{'=' * 110}")

        for M, K in [(1024, 8192), (4096, 8192), (8192, 16384)]:
            torch.manual_seed(42)
            x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
            w = torch.ones(K, device=dev, dtype=torch.bfloat16)
            grad_out = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
            inv_rms = torch.rsqrt(x.float().pow(2).mean(dim=-1) + eps)

            with torch.no_grad():
                dx_ref, dg_ref = eager_backward(grad_out, x, w, inv_rms, eps)
                dx_fused, dg_fused = te_fused.fused_silu_rmsnorm_backward(grad_out, x, w, inv_rms)

                dx_cos = F.cosine_similarity(dx_ref.float().flatten().unsqueeze(0),
                                             dx_fused.float().flatten().unsqueeze(0)).item()
                dg_cos = F.cosine_similarity(dg_ref.flatten().unsqueeze(0),
                                             dg_fused.flatten().unsqueeze(0)).item()
                dx_rel = (dx_ref.float() - dx_fused.float()).abs().mean().item() / (dx_ref.float().abs().mean().item() + 1e-8)

                print(f"  [{M:>5}x{K:>5}]  dx_cos={dx_cos:.6f}  dgamma_cos={dg_cos:.6f}  dx_rel_err={dx_rel:.6f}")

            del x, w, grad_out, inv_rms
            torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════
    # PART 5: E2E LINEAR LAYER (Component breakdown)
    # ══════════════════════════════════════════════════════════════════
    # Full pipeline from te_parity_linear_tex.py:
    #   FWD: RMSNorm+SiLU+Quant(x) + Quant(W) + GEMM_fwd
    #   BWD: Quant(dY) + re-Quant(W) + dgrad_GEMM + re-Quant(x) + wgrad_GEMM + norm_bwd
    # 3 quantizations total: x (fwd), dY (bwd), x (re-quant bwd for wgrad)
    # Fused kernels replace: (1) forward [RMSNorm+SiLU+Quant(x)], (2) backward [norm_bwd]
    # GEMMs and other quants are identical — we measure them once.
    print(f"\n{'=' * 110}")
    print("  E2E LINEAR LAYER: Component Breakdown (per training step)")
    print("  FWD: [RMSNorm+SiLU+Quant(x)] + Quant(W) + GEMM")
    print("  BWD: Quant(dY) + re-Quant(W) + dGrad + re-Quant(x) + wGrad + [norm_bwd]")
    print("  Brackets [] = where fused kernels differ from baseline")
    print(f"{'=' * 110}")

    E2E_SIZES = [
        (4096, 2048, 2048),
        (4096, 8192, 2048),     # Llama-like: up_proj
        (4096, 2048, 8192),     # Llama-like: down_proj
        (8192, 8192, 8192),     # square
        (16384, 8192, 2048),    # large batch
    ]

    for M, K, N in E2E_SIZES:
        try:
            torch._dynamo.reset()

            x_raw = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
            w_norm = torch.ones(K, device=dev, dtype=torch.bfloat16)
            W = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.023
            dY = torch.randn(M, N, device=dev, dtype=torch.bfloat16)
            rms = torch.nn.RMSNorm(K, eps=eps, device=dev, dtype=torch.bfloat16)
            with torch.no_grad():
                rms.weight.fill_(1.0)

            q_x = NVFP4Quantizer(fp4_dtype=te_dtype, rowwise=True, columnwise=True)
            q_w = NVFP4Quantizer(fp4_dtype=te_dtype, rowwise=True, columnwise=True)
            q_dy = NVFP4Quantizer(fp4_dtype=te_dtype, rowwise=True, columnwise=True)
            workspace_sz = 32 * 1024 * 1024
            workspace = torch.empty(workspace_sz, dtype=torch.uint8, device=dev)
            out_dt = TE_DType[torch.bfloat16]
            bias_dt = TE_DType[torch.bfloat16]

            inv_rms_cache = torch.rsqrt(x_raw.float().pow(2).mean(dim=-1) + eps)

            with torch.no_grad():
                # Pre-compute reference quantized tensors for GEMM timing
                eager_act = F.silu(x_raw.float() * inv_rms_cache.unsqueeze(-1) * w_norm.float()).to(torch.bfloat16)
                x_q_ref = q_x.quantize(eager_act)
                w_q_ref = q_w.quantize(W)
                dy_q_ref = q_dy.quantize(dY)
                x_q_bwd_ref = q_x.quantize(x_raw)

                # ── SHARED COMPONENTS (identical across variants) ──
                # Quant W (forward)
                t_quant_w = bench(lambda: q_w.quantize(W), args.warmup, args.steps)

                # GEMM fwd: Y = W_q @ X_q
                def gemm_fwd():
                    y = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
                    tex.generic_gemm(w_q_ref, True, x_q_ref, False, y, None, out_dt,
                                     None, bias_dt, False, None, False, workspace,
                                     workspace_sz, False, False)
                    return y
                t_gemm_fwd = bench(gemm_fwd, args.warmup, args.steps)

                # Quant dY (backward)
                t_quant_dy = bench(lambda: q_dy.quantize(dY), args.warmup, args.steps)

                # Re-quant W (backward, same as forward)
                t_requant_w = t_quant_w  # identical operation

                # dgrad GEMM: dX = W_q @ dY_q
                def gemm_dgrad():
                    return tex.generic_gemm(w_q_ref, False, dy_q_ref, False, None, None, out_dt,
                                            None, bias_dt, False, None, False, workspace,
                                            workspace_sz, False, False)[0]
                t_gemm_dgrad = bench(gemm_dgrad, args.warmup, args.steps)

                # Re-quant x (backward, for wgrad)
                t_requant_x = bench(lambda: q_x.quantize(x_raw), args.warmup, args.steps)

                # wgrad GEMM: dW = dY_q.T @ X_q
                def gemm_wgrad():
                    return tex.generic_gemm(x_q_bwd_ref, False, dy_q_ref, True, None, None, out_dt,
                                            None, bias_dt, False, None, False, workspace,
                                            workspace_sz, False, False)[0]
                t_gemm_wgrad = bench(gemm_wgrad, args.warmup, args.steps)

                # ── VARIANT COMPONENTS (differ across variants) ──

                # 1. Eager: RMSNorm + SiLU + TE quant(x)
                def eager_quant_x():
                    ir = torch.rsqrt(x_raw.float().pow(2).mean(dim=-1, keepdim=True) + eps)
                    n = x_raw.float() * ir * w_norm.float()
                    a = F.silu(n).to(torch.bfloat16)
                    return q_x.quantize(a)
                t_eager_quant_x = bench(eager_quant_x, args.warmup, args.steps)

                # 2. Compiled: torch.compile(RMSNorm+SiLU) + TE quant(x)
                compiled_norm_act = torch.compile(
                    lambda inp, norm_mod: F.silu(norm_mod(inp)),
                    mode='reduce-overhead', fullgraph=True
                )
                for _ in range(3):
                    _ = compiled_norm_act(x_raw, rms)
                torch.cuda.synchronize()

                def compiled_quant_x():
                    h = compiled_norm_act(x_raw, rms)
                    return q_x.quantize(h)
                t_compiled_quant_x = bench(compiled_quant_x, args.warmup, args.steps)

                # 3. TE-2pass fused: RMSNorm + SiLU + quant in one kernel
                t_fused_quant_x = bench(
                    lambda: te_fused.fused_te_quantize_rmsnorm_silu_2pass(x_raw, w_norm, eps),
                    args.warmup, args.steps)

                # 4. V7 fused
                t_v7_quant_x = float('nan')
                if v7_ext is not None:
                    t_v7_quant_x = bench(
                        lambda: v7_ext.forward_full(x_raw, w_norm, eps, 0, 0, 0),
                        args.warmup, args.steps)

                # Backward norm gradient variants
                # The norm backward receives dX_linear = dY @ W (M,K), NOT raw dY (M,N)
                dX_linear = dY @ W  # (M,N) @ (N,K) → (M,K)

                # Eager backward
                t_eager_norm_bwd = bench(
                    lambda: eager_backward(dX_linear, x_raw, w_norm, inv_rms_cache, eps),
                    args.warmup, args.steps)

                # Fused backward
                t_fused_norm_bwd = bench(
                    lambda: te_fused.fused_silu_rmsnorm_backward(dX_linear, x_raw, w_norm, inv_rms_cache),
                    args.warmup, args.steps)

            # ── Assemble totals ──
            shared_fwd = t_quant_w + t_gemm_fwd
            shared_bwd = t_quant_dy + t_requant_w + t_gemm_dgrad + t_requant_x + t_gemm_wgrad

            fwd_eager     = t_eager_quant_x     + shared_fwd
            fwd_compiled  = t_compiled_quant_x  + shared_fwd
            fwd_fused     = t_fused_quant_x     + shared_fwd
            fwd_v7        = t_v7_quant_x        + shared_fwd if t_v7_quant_x == t_v7_quant_x else float('nan')

            bwd_eager = shared_bwd + t_eager_norm_bwd
            bwd_fused = shared_bwd + t_fused_norm_bwd

            total_eager    = fwd_eager    + bwd_eager
            total_compiled = fwd_compiled + bwd_eager   # compiled only helps fwd
            total_fused    = fwd_fused    + bwd_fused   # fused helps both
            total_v7       = fwd_v7       + bwd_fused if fwd_v7 == fwd_v7 else float('nan')

            sp_e = total_eager / total_fused
            sp_c = total_compiled / total_fused
            sp_v7 = total_eager / total_v7 if total_v7 == total_v7 else float('nan')

            print(f"\n  ┌─ [{M}x{K}] → [{M}x{N}]  (GEMM: [{M},{K}]@[{N},{K}].T)")
            print(f"  │")
            print(f"  │  Shared components:")
            print(f"  │    Quant(W):   {t_quant_w:.3f}ms")
            print(f"  │    GEMM fwd:   {t_gemm_fwd:.3f}ms")
            print(f"  │    Quant(dY):  {t_quant_dy:.3f}ms")
            print(f"  │    dGrad GEMM: {t_gemm_dgrad:.3f}ms")
            print(f"  │    Quant(x↺):  {t_requant_x:.3f}ms")
            print(f"  │    wGrad GEMM: {t_gemm_wgrad:.3f}ms")
            print(f"  │")
            print(f"  │  Variant components (where fusion helps):")
            print(f"  │    [FWD] Eager  RMSNorm+SiLU+Quant(x):  {t_eager_quant_x:.3f}ms")
            print(f"  │    [FWD] Compile+TEq  Quant(x):          {t_compiled_quant_x:.3f}ms")
            print(f"  │    [FWD] TE-2pass fused Quant(x):        {t_fused_quant_x:.3f}ms")
            v7_str = f"{t_v7_quant_x:.3f}ms" if t_v7_quant_x == t_v7_quant_x else "n/a"
            print(f"  │    [FWD] V7 fused Quant(x):              {v7_str}")
            print(f"  │    [BWD] Eager norm bwd:                 {t_eager_norm_bwd:.3f}ms")
            print(f"  │    [BWD] Fused norm bwd:                 {t_fused_norm_bwd:.3f}ms")
            print(f"  │")
            print(f"  │  Totals (fwd + bwd):")
            print(f"  │         {'variant':<12} {'fwd':>8} {'bwd':>8} {'total':>8} {'vs eager':>9} {'vs compile':>10}")
            print(f"  │         {'eager':<12} {fwd_eager:>7.3f}ms {bwd_eager:>7.3f}ms {total_eager:>7.3f}ms {'1.00x':>9}")
            print(f"  │         {'compiled':<12} {fwd_compiled:>7.3f}ms {bwd_eager:>7.3f}ms {total_compiled:>7.3f}ms {total_eager/total_compiled:>8.2f}x")
            print(f"  │         {'TE-2pass':<12} {fwd_fused:>7.3f}ms {bwd_fused:>7.3f}ms {total_fused:>7.3f}ms {sp_e:>8.2f}x  {sp_c:>9.2f}x")
            if total_v7 == total_v7:
                print(f"  │         {'V7':<12} {fwd_v7:>7.3f}ms {bwd_fused:>7.3f}ms {total_v7:>7.3f}ms {sp_v7:>8.2f}x")
            print(f"  └{'─' * 90}")

            del x_raw, w_norm, W, dY, rms, q_x, q_w, q_dy, workspace
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"\n  ┌─ [{M}x{K}] → [{M}x{N}]  ERROR: {e}")
            import traceback; traceback.print_exc()
            torch.cuda.empty_cache()
    print()


if __name__ == "__main__":
    main()
