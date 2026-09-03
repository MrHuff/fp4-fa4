
import torch
import torch.nn as nn
from fused_ops.fused_linear import FusedQuartetLinear
import time

def benchmark_fused(in_dim, out_dim, batch_size, seq_len, dtype=torch.bfloat16):
    print(f"Benchmarking Fused Linear: In={in_dim}, Out={out_dim}, BS={batch_size}, Seq={seq_len}")
    
    device = "cuda"
    x = torch.randn(batch_size, seq_len, in_dim, device=device, dtype=dtype) * 0.1 # Small values
    
    # Baseline: RMSNorm -> SiLU -> Linear (BF16)
    class Baseline(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.RMSNorm(in_dim, eps=1e-5, dtype=dtype, device=device)
            self.act = nn.SiLU()
            self.linear = nn.Linear(in_dim, out_dim, bias=False, dtype=dtype, device=device)
        
        def forward(self, x):
            return self.linear(self.act(self.norm(x)))
            
    baseline = Baseline()
    
    # Fused: FusedQuartetLinear
    fused = FusedQuartetLinear(in_dim, out_dim, norm_eps=1e-5, device=device, dtype=dtype)
    # Copy weights for numerical sanity check (approx)
    # fused.weight.data.copy_(baseline.linear.weight.data) # Shape check: fused(Out, In), linear(Out, In)
    # fused.norm_weight.data.copy_(baseline.norm.weight.data)
    
    # Warmup
    for _ in range(10):
        _ = baseline(x)
        _ = fused(x)
        
    torch.cuda.synchronize()
    
    # Benchmark Baseline
    start = time.time()
    for _ in range(100):
        _ = baseline(x)
    torch.cuda.synchronize()
    baseline_time = (time.time() - start) / 100
    
    # Benchmark Fused
    start = time.time()
    for _ in range(100):
        _ = fused(x)
    torch.cuda.synchronize()
    fused_time = (time.time() - start) / 100
    
    print(f"Baseline Time: {baseline_time*1000:.3f} ms")
    print(f"Fused Time:    {fused_time*1000:.3f} ms")
    print(f"Speedup:       {baseline_time / fused_time:.2f}x")
    
    # Numerical Check
    # Output of Fused is float32 (from qutlass) but typically we want BF16 or Float
    # Let's check mean/std
    y_base = baseline(x)
    y_fused = fused(x)
    
    print(f"Baseline Mean: {y_base.mean().item():.4f}, Std: {y_base.std().item():.4f}")
    print(f"Fused Mean:    {y_fused.mean().item():.4f}, Std: {y_fused.std().item():.4f}")
    # Correlation?
    # Reshape
    y_base_flat = y_base.float().flatten()
    y_fused_flat = y_fused.float().flatten()
    # Normalize
    # Quantization noise is high for FP4. Just expect roughly same magnitude/stats.
    
if __name__ == "__main__":
    benchmark_fused(4096, 4096, 4, 2048)
    benchmark_fused(4096, 14336, 4, 2048)
