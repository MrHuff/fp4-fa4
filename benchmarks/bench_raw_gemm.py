import sys
import os
# Add Quartet-II/kernels/python to path
script_dir = os.path.dirname(os.path.abspath(__file__))
quartet_kernels_path = os.path.join(script_dir, '../Quartet-II/kernels/python')
sys.path.append(quartet_kernels_path)

import torch
import torch.nn.functional as F
import time
import qutlass
import matplotlib.pyplot as plt

def benchmark_raw_gemm(m, n, k, steps=100, warmup=10):
    print(f"Benchmarking {m}x{n}x{k}...")
    
    # -------------------------------------------------------
    # 1. BF16 Baseline (cuBLAS)
    # -------------------------------------------------------
    a_bf16 = torch.randn((m, k), dtype=torch.bfloat16, device='cuda')
    b_bf16 = torch.randn((k, n), dtype=torch.bfloat16, device='cuda') # KxN for torch.matmul
    
    # Warmup
    for _ in range(warmup):
        _ = torch.matmul(a_bf16, b_bf16)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(steps):
        _ = torch.matmul(a_bf16, b_bf16)
    torch.cuda.synchronize()
    bf16_time = (time.time() - start) / steps
    
    # -------------------------------------------------------
    # 2. FP4 Raw GEMM (qutlass)
    # -------------------------------------------------------
    # qutlass.matmul_nvf4_bf16_tn expects:
    # A: [M, K/2] (packed FP4) -- RowMajor
    # B: [N, K/2] (packed FP4) -- RowMajor (because it's TN: A * B^T, so B is stored as NxK logically, packed as Nx(K/2))
    # Scales A: [M, K/128] (FP8) -- Block blocked? 
    # Scales B: [N, K/128] (FP8)
    
    # Actually, let's look at `to_blocked` in linear.py
    # effective_k = k / 2 for packed storage
    
    # Dummy packed data (uint8, treating as packed FP4)
    # K must be multiple of 128 (block size) and 32 (alignment)
    assert k % 128 == 0
    
    a_fp4 = torch.randint(0, 255, (m, k // 2), dtype=torch.uint8, device='cuda')
    b_fp4 = torch.randint(0, 255, (n, k // 2), dtype=torch.uint8, device='cuda')
    
    # Scales (float8_e4m3fn)
    # Block size is 128 elements = 128/16 = 8 blocks of 16? 
    # No, qutlass expects specific blocked layout for scales.
    # In linear.py: input_scales_blocked = to_blocked(input_fp4.micro_scales)
    # micro_scales shape in linear.py is [M, K//16] ?? No, let's check quant.cu.
    # But for raw speed, we just need a tensor of the right size.
    # The `to_blocked` function rearranges [M, K//128] -> [something else].
    
    # Let's just create dummy scales of the size expected by qutlass.
    # If we pass aligned memory, it should run. 
    # The blocked layout size:
    # Re-arranged tensor of shape (32*ceil_div(H,128), 16*ceil_div(W,4)) where W is num_blocks?
    # Let's trust `to_blocked` logic from linear.py
    
    # Scale size: group_size=16. The blocked layout rearranges elements but keeps count same.
    # Input to to_blocked is [M, K//16]. Output is flattened same size.
    size_a_scales = m * (k // 16)
    size_b_scales = n * (k // 16)
    
    scales_a = torch.zeros(size_a_scales, dtype=torch.float8_e4m3fn, device='cuda')
    scales_b = torch.zeros(size_b_scales, dtype=torch.float8_e4m3fn, device='cuda')
    
    alpha = torch.tensor(1.0, dtype=torch.float32, device='cuda')
    
    # Warmup
    for _ in range(warmup):
        _ = qutlass.matmul_nvf4_bf16_tn(a_fp4, b_fp4, scales_a, scales_b, alpha)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(steps):
        _ = qutlass.matmul_nvf4_bf16_tn(a_fp4, b_fp4, scales_a, scales_b, alpha)
    torch.cuda.synchronize()
    fp4_time = (time.time() - start) / steps
    
    print(f"  BF16: {bf16_time*1000:.3f} ms")
    print(f"  FP4:  {fp4_time*1000:.3f} ms")
    print(f"  Speedup: {bf16_time / fp4_time:.2f}x")
    return bf16_time, fp4_time

if __name__ == "__main__":
    sizes = [4096, 6144, 8192, 12288, 16384, 20480, 24576]
    for size in sizes:
        benchmark_raw_gemm(size, size, size)
