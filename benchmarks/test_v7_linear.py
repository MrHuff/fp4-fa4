"""
Test V7FusedLinear v2 (pre-allocated buffers): correctness + performance
"""
import torch
import torch.nn.functional as F

print("=" * 70)
print("Testing V7FusedLinear v2 (pre-allocated buffers)")
print("=" * 70)

dev = 'cuda'
M, K, N = 4096, 8192, 4096

# Create a standard linear to convert from
linear = torch.nn.Linear(K, N, bias=False, device=dev, dtype=torch.bfloat16)

# Create TEParityLinearTex baseline
import transformer_engine.pytorch
from low_bits_training.quantization.te_parity_linear_tex import TEParityLinearTex
te_linear = TEParityLinearTex(K, N, bias=False)
te_linear = te_linear.to(dev).to(torch.bfloat16)
with torch.no_grad():
    te_linear.weight.copy_(linear.weight)

# Create V7FusedLinear
from low_bits_training.quantization.v7_fused_linear import V7FusedLinear
v7_linear = V7FusedLinear(K, N, bias=False, max_batch_tokens=M)
v7_linear = v7_linear.to(dev).to(torch.bfloat16)
with torch.no_grad():
    v7_linear.weight.copy_(linear.weight)

x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)

print(f"\nForward pass test (M={M}, K={K}, N={N})...")
with torch.no_grad():
    y_v7 = v7_linear(x)
    print(f"  V7: {y_v7.shape}, mean={y_v7.float().mean():.4f}, std={y_v7.float().std():.4f}")

    h = F.silu(F.rms_norm(x, (K,), torch.ones(K, device=dev, dtype=torch.bfloat16), 1e-5))
    y_te = te_linear(h)
    print(f"  TE: {y_te.shape}, mean={y_te.float().mean():.4f}, std={y_te.float().std():.4f}")

    cos = F.cosine_similarity(y_v7.flatten().float().unsqueeze(0),
                               y_te.flatten().float().unsqueeze(0)).item()
    print(f"  Cosine similarity: {cos:.6f}")

# ============================================================
# Performance benchmark
# ============================================================
def bench(fn, warmup=20, steps=50):
    for _ in range(warmup): fn(); torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(); s.record()
    for _ in range(steps): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / steps

print(f"\n{'='*70}")
print("Performance: V7FusedLinear v2 vs TEParityLinearTex (forward only)")
print(f"{'='*70}")
print(f"{'M':>8} {'K':>8} {'N':>8} | {'V7Fused':>10} {'TE+norm':>10} | {'speedup':>8}")
print("-" * 70)

configs = [
    (2048, 4096, 4096),
    (4096, 4096, 4096),
    (4096, 8192, 4096),
    (4096, 8192, 8192),
    (8192, 8192, 8192),
    (4096, 16384, 4096),
    (4096, 16384, 16384),
    (8192, 16384, 16384),
]

for M, K, N in configs:
    try:
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        linear = torch.nn.Linear(K, N, bias=False, device=dev, dtype=torch.bfloat16)

        v7_l = V7FusedLinear(K, N, bias=False, max_batch_tokens=M).to(dev).to(torch.bfloat16)
        with torch.no_grad(): v7_l.weight.copy_(linear.weight)

        te_l = TEParityLinearTex(K, N, bias=False).to(dev).to(torch.bfloat16)
        with torch.no_grad(): te_l.weight.copy_(linear.weight)

        rms = torch.nn.RMSNorm(K, eps=1e-5, device=dev, dtype=torch.bfloat16)
        with torch.no_grad(): rms.weight.fill_(1.0)

        with torch.no_grad():
            # Warmup V7 (triggers buffer allocation)
            _ = v7_l(x)

            t_v7 = bench(lambda: v7_l(x))

            def te_pipeline():
                h = F.silu(rms(x))
                return te_l(h)
            t_te = bench(te_pipeline)

        speedup = t_te / t_v7
        marker = "**" if speedup > 1.0 else "  "
        print(f"{M:>8} {K:>8} {N:>8} | {t_v7:>9.3f}ms {t_te:>9.3f}ms | {speedup:>6.2f}x {marker}")

        del x, linear, v7_l, te_l, rms
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"{M:>8} {K:>8} {N:>8} | ERROR: {e}")
        import traceback; traceback.print_exc()
        torch.cuda.empty_cache()
