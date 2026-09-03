
import torch
import torch.nn as nn
import time
from fused_ops.fused_linear import fused_quant_fp4
from quartet2.quant import quant_fp4

def profile_breakdown():
    print("Profiling Breakdown: Fused vs Component-wise")
    device = "cuda"
    dtype = torch.bfloat16
    
    # Config
    batch_size = 4
    seq_len = 2048
    in_dim = 4096
    
    x = torch.randn(batch_size, seq_len, in_dim, device=device, dtype=dtype)
    norm_weight = torch.rand(in_dim, device=device, dtype=dtype) + 0.5
    epsilon = 1e-5
    
    # Warmup
    for _ in range(10):
        # Baseline components
        y = x * 1.0
        n = nn.RMSNorm(in_dim, eps=epsilon, dtype=dtype, device=device)
        a = nn.SiLU()
        # Fused components
        x_flat = x.reshape(-1, in_dim)
        _ = fused_quant_fp4(x_flat, norm_weight, epsilon, scale_override=1.0)
        
    torch.cuda.synchronize()
    
    iterations = 100
    
    # 1. Baseline: RMSNorm
    norm_layer = nn.RMSNorm(in_dim, eps=epsilon, dtype=dtype, device=device)
    norm_layer.weight.data.copy_(norm_weight)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iterations):
        x_norm = norm_layer(x)
    torch.cuda.synchronize()
    t_rms = (time.time() - start) / iterations * 1000
    
    # 2. Baseline: SiLU
    act_layer = nn.SiLU()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iterations):
        x_act = act_layer(x_norm)
    torch.cuda.synchronize()
    t_silu = (time.time() - start) / iterations * 1000
    
    # 3. Baseline: Quantization (Standard)
    # We use quartet2.quant.quant_fp4
    amax = x_act.abs().max()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iterations):
        # emulate finding amax? Standard quant usually requires finding amax first.
        # Let's include amax finding in timing if we want realistic "Auto" quant.
        # Or exclude it if we assume static?
        # Fused kernel does calculate amax. So we should include it.
        local_amax = x_act.abs().max().to(torch.float32)
        x_act_flat = x_act.reshape(-1, in_dim)
        _ = quant_fp4(x_act_flat, amax=local_amax, scale_override=1.0, four_over_six=False)
    torch.cuda.synchronize()
    t_quant = (time.time() - start) / iterations * 1000
    
    t_baseline_total = t_rms + t_silu + t_quant
    
    print(f"Baseline Component Breakdown:")
    print(f"  RMSNorm: {t_rms:.3f} ms")
    print(f"  SiLU:    {t_silu:.3f} ms")
    print(f"  Quant:   {t_quant:.3f} ms")
    print(f"  Total:   {t_baseline_total:.3f} ms")
    
    # 4. Fused Kernel
    x_flat = x.reshape(-1, in_dim)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iterations):
        _ = fused_quant_fp4(x_flat, norm_weight, epsilon, scale_override=1.0)
    torch.cuda.synchronize()
    t_fused = (time.time() - start) / iterations * 1000
    
    print(f"Fused Kernel:")
    print(f"  Total:   {t_fused:.3f} ms")
    
    print(f"Speedup vs Components: {t_baseline_total / t_fused:.2f}x")

if __name__ == "__main__":
    profile_breakdown()
