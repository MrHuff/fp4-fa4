#!/usr/bin/env python3
"""
TKTK vs TETE: Module-level benchmark at realistic Llama shapes.

Uses the same shapes as bench_tk_te_realistic.py but benchmarks through
the actual _WoFunction and _FusedQKVFunction autograd modules, measuring
the full quant+forward and quant+backward pipeline.

Llama 1B_legacy: dim=2048, n_heads=32, n_kv_heads=32, head_dim=64
  - q_dim=k_dim=v_dim=2048, N_total=6144
  - FFN hidden_dim=5632
  - M = BSZ × seq_len: 16384, 32768, 65536
"""
import os, sys, torch
sys.path.insert(0, '/workspace/low-bits-training')
sys.path.insert(0, '/workspace/fp4_matmul/TK_quantisation/nvfp4')

os.environ.setdefault('NVTE_NVFP4_DISABLE_RHT', '1')
os.environ.setdefault('NVTE_NVFP4_DISABLE_2D_QUANTIZATION', '1')
os.environ.setdefault('NVTE_NVFP4_ENCODE_CENTRIC', '0')
os.environ.setdefault('NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING', '1')
os.environ.setdefault('NVTE_CUSTOM_QUANT', '1')
os.environ.setdefault('FUSED_TE_QUANT', '0')

from low_bits_training.quantization.fused_te_linear import (
    tex, NVFP4Quantizer,
    _WoFunction_TE, _WoFunction_TK,
    _FusedQKVFunction_TE, _FusedQKVFunction_TK,
)

device = 'cuda'
dtype = torch.bfloat16
workspace = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=device)

def make_quantizer():
    return NVFP4Quantizer(
        fp4_dtype=tex.DType.kFloat4E2M1, rowwise=True, columnwise=True,
        with_amax_reduction=False, amax_reduction_group=None,
        with_rht=False, with_post_rht_amax=False,
        with_2d_quantization=False, stochastic_rounding=False,
        with_random_sign_mask=True, encode_centric=False,
    )

def _time(fn, steps=200, warmup=50):
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


# ─── Linear (Wo / FFN) benchmark ─────────────────────────────────
def bench_linear(M, K, N, steps=200, warmup=50):
    w = torch.randn(N, K, dtype=dtype, device=device) * 0.02
    x = torch.randn(M, K, dtype=dtype, device=device) * 0.01
    g = torch.randn(M, N, dtype=dtype, device=device) * 0.001

    def make_fwd(cls):
        os.environ['USE_TK_QUANT'] = '1' if cls == _WoFunction_TK else '0'
        os.environ['USE_TK_GEMM'] = '1' if cls == _WoFunction_TK else '0'
        def fn():
            with torch.no_grad():
                cls.apply(x, w, make_quantizer(), make_quantizer(), make_quantizer(), workspace)
        return fn

    def make_fwd_bwd(cls):
        os.environ['USE_TK_QUANT'] = '1' if cls == _WoFunction_TK else '0'
        os.environ['USE_TK_GEMM'] = '1' if cls == _WoFunction_TK else '0'
        def fn():
            x_ = x.clone().requires_grad_(True)
            w_ = w.clone().requires_grad_(True)
            y = cls.apply(x_, w_, make_quantizer(), make_quantizer(), make_quantizer(), workspace)
            y.backward(g)
        return fn

    os.environ['USE_TK_QUANT'] = '0'; os.environ['USE_TK_GEMM'] = '0'
    te_fwd = _time(make_fwd(_WoFunction_TE), steps, warmup)
    te_tot = _time(make_fwd_bwd(_WoFunction_TE), steps, warmup)
    te_bwd = te_tot - te_fwd

    os.environ['USE_TK_QUANT'] = '1'; os.environ['USE_TK_GEMM'] = '1'
    tk_fwd = _time(make_fwd(_WoFunction_TK), steps, warmup)
    tk_tot = _time(make_fwd_bwd(_WoFunction_TK), steps, warmup)
    tk_bwd = tk_tot - tk_fwd

    return te_fwd, te_bwd, te_tot, tk_fwd, tk_bwd, tk_tot


# ─── QKV benchmark ───────────────────────────────────────────────
def bench_qkv(M, dim, q_dim, k_dim, v_dim, steps=200, warmup=50):
    N_total = q_dim + k_dim + v_dim
    w = torch.randn(N_total, dim, dtype=dtype, device=device) * 0.02
    x = torch.randn(M, dim, dtype=dtype, device=device) * 0.01
    nw = torch.ones(dim, dtype=dtype, device=device)
    gq = torch.randn(M, q_dim, dtype=dtype, device=device) * 0.001
    gk = torch.randn(M, k_dim, dtype=dtype, device=device) * 0.001
    gv = torch.randn(M, v_dim, dtype=dtype, device=device) * 0.001

    def make_fwd(cls):
        os.environ['USE_TK_QUANT'] = '1' if cls == _FusedQKVFunction_TK else '0'
        os.environ['USE_TK_GEMM'] = '1' if cls == _FusedQKVFunction_TK else '0'
        def fn():
            with torch.no_grad():
                cls.apply(x, w, nw, 1e-5, q_dim, k_dim, v_dim,
                          make_quantizer(), make_quantizer(), make_quantizer(), workspace)
        return fn

    def make_fwd_bwd(cls):
        os.environ['USE_TK_QUANT'] = '1' if cls == _FusedQKVFunction_TK else '0'
        os.environ['USE_TK_GEMM'] = '1' if cls == _FusedQKVFunction_TK else '0'
        def fn():
            x_ = x.clone().requires_grad_(True)
            w_ = w.clone().requires_grad_(True)
            nw_ = nw.clone().requires_grad_(True)
            outs = cls.apply(x_, w_, nw_, 1e-5, q_dim, k_dim, v_dim,
                             make_quantizer(), make_quantizer(), make_quantizer(), workspace)
            loss = (outs[0] * gq).sum() + (outs[1] * gk).sum() + (outs[2] * gv).sum()
            loss.backward()
        return fn

    os.environ['USE_TK_QUANT'] = '0'; os.environ['USE_TK_GEMM'] = '0'
    te_fwd = _time(make_fwd(_FusedQKVFunction_TE), steps, warmup)
    te_tot = _time(make_fwd_bwd(_FusedQKVFunction_TE), steps, warmup)
    te_bwd = te_tot - te_fwd

    os.environ['USE_TK_QUANT'] = '1'; os.environ['USE_TK_GEMM'] = '1'
    tk_fwd = _time(make_fwd(_FusedQKVFunction_TK), steps, warmup)
    tk_tot = _time(make_fwd_bwd(_FusedQKVFunction_TK), steps, warmup)
    tk_bwd = tk_tot - tk_fwd

    return te_fwd, te_bwd, te_tot, tk_fwd, tk_bwd, tk_tot


def ratio(a, b):
    return a / b if b > 0.001 else float('nan')


def main():
    torch.manual_seed(42)

    print("=" * 130)
    print("  TKTK vs TETE: Module-Level Benchmark — Realistic Llama Shapes")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print("=" * 130)

    # Llama 1B_legacy: dim=2048, n_heads=32, n_kv_heads=32, head_dim=64
    dim = 2048
    hidden_dim = 5632
    q_dim = dim       # 32 × 64
    k_dim = dim       # 32 × 64
    v_dim = dim       # 32 × 64

    M_values = [
        (16384, "BSZ=16×1024"),
        (32768, "BSZ=32×1024"),
        (65536, "BSZ=64×1024"),
    ]

    steps = 100
    warmup = 20

    hdr = (f"  {'Layer':24s}  {'M':>6s}  "
           f"{'── TE Fwd ──':>11s}  {'── TK Fwd ──':>11s}  {'Fwd Spd':>7s}  │  "
           f"{'── TE Bwd ──':>11s}  {'── TK Bwd ──':>11s}  {'Bwd Spd':>7s}  │  "
           f"{'── TE Tot ──':>11s}  {'── TK Tot ──':>11s}  {'Tot Spd':>7s}")
    
    for M, m_label in M_values:
        print(f"\n{'─'*130}")
        print(f"  {m_label}  (M={M})")
        print(f"{'─'*130}")
        print(hdr)

        # QKV
        te_f, te_b, te_t, tk_f, tk_b, tk_t = bench_qkv(M, dim, q_dim, k_dim, v_dim, steps, warmup)
        print(f"  {'QKV (' + str(q_dim) + '+' + str(k_dim) + '+' + str(v_dim) + ')':24s}  {M:>6d}  "
              f"{te_f:>10.3f}ms  {tk_f:>10.3f}ms  {ratio(te_f,tk_f):>6.2f}x  │  "
              f"{te_b:>10.3f}ms  {tk_b:>10.3f}ms  {ratio(te_b,tk_b):>6.2f}x  │  "
              f"{te_t:>10.3f}ms  {tk_t:>10.3f}ms  {ratio(te_t,tk_t):>6.2f}x")

        # Wo: (M, q_dim) → (M, dim)
        te_f, te_b, te_t, tk_f, tk_b, tk_t = bench_linear(M, q_dim, dim, steps, warmup)
        print(f"  {'Wo (' + str(q_dim) + '→' + str(dim) + ')':24s}  {M:>6d}  "
              f"{te_f:>10.3f}ms  {tk_f:>10.3f}ms  {ratio(te_f,tk_f):>6.2f}x  │  "
              f"{te_b:>10.3f}ms  {tk_b:>10.3f}ms  {ratio(te_b,tk_b):>6.2f}x  │  "
              f"{te_t:>10.3f}ms  {tk_t:>10.3f}ms  {ratio(te_t,tk_t):>6.2f}x")

        # FFN w1/w3: (M, dim) → (M, hidden_dim)
        te_f, te_b, te_t, tk_f, tk_b, tk_t = bench_linear(M, dim, hidden_dim, steps, warmup)
        print(f"  {'FFN w1/w3 (' + str(dim) + '→' + str(hidden_dim) + ')':24s}  {M:>6d}  "
              f"{te_f:>10.3f}ms  {tk_f:>10.3f}ms  {ratio(te_f,tk_f):>6.2f}x  │  "
              f"{te_b:>10.3f}ms  {tk_b:>10.3f}ms  {ratio(te_b,tk_b):>6.2f}x  │  "
              f"{te_t:>10.3f}ms  {tk_t:>10.3f}ms  {ratio(te_t,tk_t):>6.2f}x")

        # FFN w2: (M, hidden_dim) → (M, dim)
        te_f, te_b, te_t, tk_f, tk_b, tk_t = bench_linear(M, hidden_dim, dim, steps, warmup)
        print(f"  {'FFN w2 (' + str(hidden_dim) + '→' + str(dim) + ')':24s}  {M:>6d}  "
              f"{te_f:>10.3f}ms  {tk_f:>10.3f}ms  {ratio(te_f,tk_f):>6.2f}x  │  "
              f"{te_b:>10.3f}ms  {tk_b:>10.3f}ms  {ratio(te_b,tk_b):>6.2f}x  │  "
              f"{te_t:>10.3f}ms  {tk_t:>10.3f}ms  {ratio(te_t,tk_t):>6.2f}x")

    print(f"\n{'='*130}")
    print("  All benchmarks complete!")
    print(f"{'='*130}")


if __name__ == "__main__":
    main()
