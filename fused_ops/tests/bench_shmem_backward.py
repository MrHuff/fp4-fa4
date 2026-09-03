import torch
import fused_ops._fused_ops as fused_ops
import time

def bench_backward(batch=16, seq=128, dim=2048):
    device = torch.device('cuda')
    rows = batch * seq
    cols = dim
    
    print(f"\nBenchmarking Shape: ({rows}, {cols})")
    
    # Inputs
    grad_output = torch.randn(rows, cols, device=device, dtype=torch.bfloat16)
    x = torch.randn(rows, cols, device=device, dtype=torch.bfloat16)
    weight = torch.randn(cols, device=device, dtype=torch.bfloat16)
    inv_rms = torch.ones(rows, device=device, dtype=torch.float32) * 0.5
    eps = 1e-5
    
    # Outputs
    grad_input_std = torch.empty_like(x)
    grad_input_shmem = torch.empty_like(x)
    
    # Warmup and Correctness Check
    fused_ops.fused_backward_opt(grad_output, x, weight, inv_rms, eps, grad_input_std)
    fused_ops.fused_backward_shmem(grad_output, x, weight, inv_rms, eps, grad_input_shmem)
    
    if torch.allclose(grad_input_std, grad_input_shmem, atol=1e-3, rtol=1e-3):
        print("Correctness Check: PASS")
    else:
        print("Correctness Check: FAIL (Continuing)")
        print(f"Max Diff: {(grad_input_std - grad_input_shmem).abs().max().item()}")
        # Continue to benchmark performance anyway

    # Benchmark Standard
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    iters = 100
    
    start.record()
    for _ in range(iters):
        fused_ops.fused_backward_opt(grad_output, x, weight, inv_rms, eps, grad_input_std)
    end.record()
    torch.cuda.synchronize()
    time_std = start.elapsed_time(end) / iters
    print(f"Standard Backward: {time_std:.3f} ms")

    # Benchmark Shmem
    start.record()
    for _ in range(iters):
        fused_ops.fused_backward_shmem(grad_output, x, weight, inv_rms, eps, grad_input_shmem)
    end.record()
    torch.cuda.synchronize()
    time_shmem = start.elapsed_time(end) / iters
    print(f"Shmem Backward:    {time_shmem:.3f} ms")
    
    print(f"Speedup: {time_std / time_shmem:.2f}x")

if __name__ == "__main__":
    bench_backward(batch=64, seq=2048, dim=2048)
    bench_backward(batch=64, seq=2048, dim=5632)
    bench_backward(batch=64, seq=5632, dim=2048)
    bench_backward(batch=32, seq=3072, dim=3072)
    bench_backward(batch=32, seq=3072, dim=8192)
