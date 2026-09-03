"""
Benchmark & Test: Fused TE RMSNorm + SiLU + NVFP4 Quant
Tests three modes:
  1. fused_te_quantize_rmsnorm_silu      — amax given externally
  2. fused_te_quantize_rmsnorm_silu_2pass — amax computed internally (Pass1+Pass2)
  3. V7 custom kernel (baseline)
"""
import torch, torch.nn.functional as F, os, time
from torch.utils.cpp_extension import load
import transformer_engine.pytorch as te
from transformer_engine.pytorch import NVFP4Quantizer

TE_ROOT = '/workspace/low-bits-training/TransformerEngine'
TE_INCLUDE = os.path.join(TE_ROOT, 'transformer_engine/common/include')
TE_LIB_DIR = os.path.join(TE_ROOT, 'build/cmake')
CSRC = '/workspace/fp4_matmul/fused_ops/csrc'
cuda_lib = '/usr/local/cuda/lib64'

print("Compiling TE-fused extension...", flush=True)
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
print("TE-fused extension compiled.\n", flush=True)

# Load V7 for comparison
V7_SOURCES = [os.path.join(CSRC, f) for f in ['fused_te_quant_v7.cu', 'fused_te_quant_v7_torch.cpp']]
if all(os.path.exists(s) for s in V7_SOURCES):
    try:
        print("Compiling V7 kernel...", flush=True)
        v7_ext = load(name='fused_te_quant_v7',
            sources=V7_SOURCES,
            extra_include_paths=[TE_INCLUDE, '/usr/local/cuda/include', CSRC],
            extra_cuda_cflags=['-std=c++17', '--expt-relaxed-constexpr', '-O3',
                               f'-I{TE_INCLUDE}', f'-I{CSRC}'],
            extra_cflags=['-std=c++17'],
            extra_ldflags=[f'-L{TE_LIB_DIR}', '-ltransformer_engine', f'-Wl,-rpath,{TE_LIB_DIR}',
                           f'-L{cuda_lib}', '-lcudart', '-lnvrtc', f'-Wl,-rpath,{cuda_lib}'],
            verbose=False)
        print("V7 compiled.\n", flush=True)
    except Exception as e:
        v7_ext = None
        print(f"V7 compilation failed ({e.__class__.__name__}), skipping.\n")
else:
    v7_ext = None
    print("V7 sources not found, skipping.\n")

# Dequant helper
FP4_LUT = [0,0.5,1,1.5,2,3,4,6,-0,-0.5,-1,-1.5,-2,-3,-4,-6]
FP4_MAX = 6.0; FP8_MAX = 448.0

def deq(fp4, sc_padded, amax, m, k):
    lut = torch.tensor(FP4_LUT, device='cuda', dtype=torch.float32)
    d = fp4.view(torch.uint8).to(torch.int32)
    u = torch.stack((d & 0x0F, d >> 4), dim=-1).reshape(m, k)
    fv = lut[u]
    sc = sc_padded.view(torch.float8_e4m3fn).to(torch.float32)[:m, :k//16]
    ts = amax / (FP4_MAX * FP8_MAX)
    return (fv.view(-1, 16) * ts * sc.reshape(-1, 1)).view(m, k)

def bench_fn(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000

# GPU info 
gpu_name = torch.cuda.get_device_name()
print(f"GPU: {gpu_name}\n")

SIZES = [(1024,8192), (4096,8192), (8192,16384), (16384,16384), (32768,32768)]

# ===== CORRECTNESS =====
print("=" * 100)
print("  CORRECTNESS")
print("=" * 100)

for M, K in SIZES:
    torch.manual_seed(42)
    x = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
    w = torch.ones(K, device='cuda', dtype=torch.bfloat16)
    eps = 1e-5

    # Golden reference: eager RMSNorm + SiLU + TE quantize
    inv_rms = torch.rsqrt(x.float().pow(2).mean(dim=-1) + eps)
    normed = x.float() * inv_rms.unsqueeze(-1) * w.float()
    act_out = F.silu(normed).to(torch.bfloat16)

    q = NVFP4Quantizer(rowwise=True, columnwise=False)
    ref_qt = q.quantize(act_out)
    ref_amax = ref_qt._amax_rowwise
    ref_deq = ref_qt.dequantize(dtype=torch.float32)

    # Mode 1: fused with given amax
    fp4_1, sc_1 = te_fused.fused_te_quantize_rmsnorm_silu(x, inv_rms, w, ref_amax)
    torch.cuda.synchronize()
    deq_1 = deq(fp4_1, sc_1, ref_amax.item(), M, K)
    cos_1 = F.cosine_similarity(deq_1.flatten().unsqueeze(0), ref_deq.flatten().unsqueeze(0)).item()

    # Mode 2: 2-pass (computes its own amax)
    fp4_2, sc_2, inv_rms_2, amax_2 = te_fused.fused_te_quantize_rmsnorm_silu_2pass(x, w, eps)
    torch.cuda.synchronize()
    deq_2 = deq(fp4_2, sc_2, amax_2.item(), M, K)
    cos_2 = F.cosine_similarity(deq_2.flatten().unsqueeze(0), ref_deq.flatten().unsqueeze(0)).item()

    # Check inv_rms agreement
    inv_rms_cos = F.cosine_similarity(inv_rms.unsqueeze(0), inv_rms_2.unsqueeze(0)).item()
    amax_err = abs(ref_amax.item() - amax_2.item())

    nz_1 = (fp4_1 != 0).sum().item() / fp4_1.numel() * 100
    nz_2 = (fp4_2 != 0).sum().item() / fp4_2.numel() * 100

    print(f"  [{M:5d}x{K:5d}]  amax_given cos={cos_1:.6f}  2pass cos={cos_2:.6f}  "
          f"inv_rms_cos={inv_rms_cos:.6f}  amax_err={amax_err:.4f}  nz1={nz_1:.1f}%  nz2={nz_2:.1f}%")

# ===== PERFORMANCE =====
print()
print("=" * 100)
print("  PERFORMANCE")
print("=" * 100)
print(f"{'M':>8} {'K':>8} | {'separated':>10} {'TE-2pass':>10} {'TEkrnl':>10} {'V7':>10} | {'2p/sep':>8} {'kern/sep':>8}")
print("-" * 100)

for M, K in SIZES:
    torch.manual_seed(42)
    x = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
    w = torch.ones(K, device='cuda', dtype=torch.bfloat16)
    eps = 1e-5

    inv_rms = torch.rsqrt(x.float().pow(2).mean(dim=-1) + eps)
    normed = x.float() * inv_rms.unsqueeze(-1) * w.float()
    act_out = F.silu(normed).to(torch.bfloat16)
    q = NVFP4Quantizer(rowwise=True, columnwise=False)
    ref_qt = q.quantize(act_out)
    ref_amax = ref_qt._amax_rowwise

    # Separated: RMSNorm + SiLU + TE quantize
    def separated():
        ir = torch.rsqrt(x.float().pow(2).mean(dim=-1) + eps)
        n = x.float() * ir.unsqueeze(-1) * w.float()
        a = F.silu(n).to(torch.bfloat16)
        return q.quantize(a)
    t_sep = bench_fn(separated)

    # TE kernel only (amax given)
    def te_kern():
        return te_fused.fused_te_quantize_rmsnorm_silu(x, inv_rms, w, ref_amax)
    t_kern = bench_fn(te_kern)

    # TE 2-pass (full pipeline, no Python overhead)
    def te_2pass():
        return te_fused.fused_te_quantize_rmsnorm_silu_2pass(x, w, eps)
    t_2pass = bench_fn(te_2pass)

    # V7
    t_v7 = 0
    if v7_ext is not None:
        def v7_fn():
            return v7_ext.forward(x, w, eps, 1.0)
        t_v7 = bench_fn(v7_fn)

    print(f"  {M:6d} {K:8d} | {t_sep:9.3f}ms {t_2pass:9.3f}ms {t_kern:9.3f}ms {t_v7:9.3f}ms | "
          f"{t_sep/t_2pass:7.2f}x {t_sep/t_kern:7.2f}x")

print()
print("=" * 100)
print("Done.")
