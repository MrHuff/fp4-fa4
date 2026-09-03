"""
bench_v1_v3_te.py — V1 vs V3 vs TE Quant-Only Benchmark

Compares quantization speed (no GEMM) across:
  V1: Original fused_te_quant with __nv_cvt_float2_to_fp4x2
  V3: PTX fused mul+cvt (cvt.rn.satfinite.e2m1x2)
  TE: tex.quantize (TE's native quantization)

Plus correctness: Dequantize V1 and V3, compare cosine similarity.
"""

import os
import sys
import time
import torch
import torch.nn.functional as F
import argparse

# TE first
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import NVFP4Quantizer

print("Compiling V1 kernel...")
from torch.utils.cpp_extension import load
CSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fused_ops', 'csrc')
v1 = load(
    name='fused_te_quant_v1',
    sources=[
        os.path.join(CSRC, 'fused_te_quant_torch.cpp'),
        os.path.join(CSRC, 'fused_te_quant.cu'),
    ],
    extra_include_paths=[CSRC],
    extra_cuda_cflags=[
        '-std=c++20', '-O3', '-lineinfo',
        '--expt-relaxed-constexpr',
        '-gencode=arch=compute_100a,code=sm_100a',
    ],
    verbose=False,
)

print("Compiling V3 kernel...")
v3 = load(
    name='fused_te_quant_v3',
    sources=[
        os.path.join(CSRC, 'fused_te_quant_v3_torch.cpp'),
        os.path.join(CSRC, 'fused_te_quant_v3.cu'),
    ],
    extra_include_paths=[CSRC],
    extra_cuda_cflags=[
        '-std=c++20', '-O3', '-lineinfo',
        '--expt-relaxed-constexpr',
        '-gencode=arch=compute_100a,code=sm_100a',
    ],
    verbose=False,
)
print("Compiled.")


def dequant_raw_fp4(fp4_bytes, scales_fp8, global_scale, m, k):
    fp4_vals = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        device=fp4_bytes.device, dtype=torch.float32
    )
    data = fp4_bytes.view(torch.uint8).to(torch.int32)
    lo = data & 0x0F
    hi = data >> 4
    unpacked = torch.stack((lo, hi), dim=-1).reshape(m, k)
    float_vals = fp4_vals[unpacked]
    block_scales = scales_fp8.view(torch.float8_e4m3fn).to(torch.float32)
    block_scales = block_scales[:m, :k//16]
    block_data = float_vals.view(-1, 16)
    block_data = block_data * global_scale * block_scales.reshape(-1, 1)
    return block_data.view(m, k)


def bench(fn, warmup=50, steps=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(steps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--steps', type=int, default=200)
    args = parser.parse_args()

    K = 8192
    device = 'cuda'
    w = torch.ones(K, device=device, dtype=torch.bfloat16)

    print(f"\n{'='*100}")
    print(f"  V1 vs V3 vs TE — Fused Quant-Only Benchmark")
    print(f"  GPU: {torch.cuda.get_device_name(0)}, K={K}")
    print(f"  V3 uses PTX fused mul+cvt (cvt.rn.satfinite.e2m1x2.f32)")
    print(f"{'='*100}")

    # Header
    print(f"\n{'M':>8} | {'V1':>10} {'V3-dec':>10} {'V3-enc':>10} {'TE':>10} |"
          f" {'V3/V1':>8} {'V3/TE':>8} | {'cos(V1,V3)':>10} {'byte%':>6}")
    print('-' * 100)

    te_dtype = tex.DType.kFloat4E2M1

    for M in [1024, 2048, 4096, 8192, 16384]:
        x = torch.randn(M, K, device=device, dtype=torch.bfloat16)

        # TE
        quantizer = NVFP4Quantizer(fp4_dtype=te_dtype, rowwise=True, columnwise=False)
        x_q = quantizer.make_empty((M, K), dtype=torch.bfloat16, device=device)
        # Pre-compute RMS+SiLU for TE (since TE quant doesn't include norm/act)
        rms = torch.nn.RMSNorm(K, eps=1e-5, device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            rms.weight.fill_(1.0)
        h = F.silu(rms(x))

        t_te = bench(lambda: quantizer.update_quantized(h, x_q), args.warmup, args.steps)

        # V1
        t_v1 = bench(lambda: v1.forward_full(x, w, 1e-5, 0, 0), args.warmup, args.steps)

        # V3 decode-centric
        t_v3_dec = bench(lambda: v3.forward_full(x, w, 1e-5, 0, 0, 0), args.warmup, args.steps)

        # V3 encode-centric
        t_v3_enc = bench(lambda: v3.forward_full(x, w, 1e-5, 0, 0, 1), args.warmup, args.steps)

        # Correctness: V1 vs V3 decode
        fp4_v1, sc_v1, gs_v1, _ = v1.forward_full(x, w, 1e-5, 0, 0)
        fp4_v3, sc_v3, gs_v3, _ = v3.forward_full(x, w, 1e-5, 0, 0, 0)

        deq_v1 = dequant_raw_fp4(fp4_v1, sc_v1, gs_v1.item(), M, K)
        deq_v3 = dequant_raw_fp4(fp4_v3, sc_v3, gs_v3.item(), M, K)
        cos = F.cosine_similarity(deq_v1.flatten().unsqueeze(0),
                                   deq_v3.flatten().unsqueeze(0)).item()
        byte_match = (fp4_v1.view(torch.uint8) == fp4_v3.view(torch.uint8)).float().mean().item()

        print(f"{M:>8} | {t_v1:>9.3f}ms {t_v3_dec:>9.3f}ms {t_v3_enc:>9.3f}ms {t_te:>9.3f}ms |"
              f" {t_v3_dec/t_v1:>7.2f}x {t_v3_dec/t_te:>7.2f}x | {cos:>10.6f} {byte_match*100:>5.1f}%")

    print(f"\nNote: V1 and V3 include RMSNorm+SiLU; TE is quant-only (norm+act done separately)")


if __name__ == '__main__':
    main()
