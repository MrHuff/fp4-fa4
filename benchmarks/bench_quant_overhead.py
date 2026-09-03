import sys
import os
# Add Quartet-II/kernels/python to path
script_dir = os.path.dirname(os.path.abspath(__file__))
quartet_kernels_path = os.path.join(script_dir, '../Quartet-II/kernels/python')
sys.path.append(quartet_kernels_path)

import torch
import time
import qutlass
from quartet2.quant import quant_fp4
from quartet2.linear import abs_max, to_blocked

def benchmark_overhead(m, n, k, steps=100, warmup=10):
    print(f"--- Overhead Analysis: {m}x{n}x{k} ---")
    
    # Setup Data
    x = torch.randn((m, k), dtype=torch.bfloat16, device='cuda')
    w = torch.randn((n, k), dtype=torch.bfloat16, device='cuda') # Weight usually [Out, In]
    
    # 1. AbsMax
    for _ in range(warmup): abs_max(x)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(steps): abs_max(x)
    torch.cuda.synchronize()
    t_abs = (time.time() - start) / steps
    
    # 2. Quantization (includes absmax usually, but let's measure raw quant with pre-computed amax)
    amax = abs_max(x)
    
    for _ in range(warmup): quant_fp4(x, 1.0, amax=amax, four_over_six=True)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(steps): 
        q_out = quant_fp4(x, 1.0, amax=amax, four_over_six=True)
    torch.cuda.synchronize()
    t_quant = (time.time() - start) / steps
    
    # 3. Layout Transform (to_blocked)
    # micro_scales from quant_fp4 are [M, K//16]
    ms = q_out.micro_scales
    for _ in range(warmup): to_blocked(ms)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(steps): to_blocked(ms)
    torch.cuda.synchronize()
    t_layout = (time.time() - start) / steps
    
    # 4. Raw GEMM (from previous result, approx)
    # We can re-measure or just use previous values. Let's re-measure quickly.
    # We need packed inputs.
    x_fp4 = torch.randint(0, 255, (m, k // 2), dtype=torch.uint8, device='cuda')
    w_fp4 = torch.randint(0, 255, (n, k // 2), dtype=torch.uint8, device='cuda')
    # Scales
    size_scales_x = m * (k // 16)
    size_scales_w = n * (k // 16)
    sx = torch.zeros(size_scales_x, dtype=torch.float8_e4m3fn, device='cuda')
    sw = torch.zeros(size_scales_w, dtype=torch.float8_e4m3fn, device='cuda')
    alpha = torch.tensor(1.0, device='cuda')
    
    for _ in range(warmup): qutlass.matmul_nvf4_bf16_tn(x_fp4, w_fp4, sx, sw, alpha)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(steps): qutlass.matmul_nvf4_bf16_tn(x_fp4, w_fp4, sx, sw, alpha)
    torch.cuda.synchronize()
    t_gemm = (time.time() - start) / steps
    
    # 5. BF16 Gemm
    for _ in range(warmup): torch.matmul(x, w.T)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(steps): torch.matmul(x, w.T)
    torch.cuda.synchronize()
    t_bf16 = (time.time() - start) / steps
    
    # Report
    print(f"BF16 GEMM:    {t_bf16*1000:.3f} ms")
    print(f"FP4 RAW GEMM: {t_gemm*1000:.3f} ms ({t_bf16/t_gemm:.2f}x speedup)")
    print(f"Overhead Breakdown:")
    print(f"  AbsMax:     {t_abs*1000:.3f} ms")
    print(f"  Quant:      {t_quant*1000:.3f} ms")
    print(f"  Layout:     {t_layout*1000:.3f} ms")
    
    # Total Quartet Forward Estimate (Input Quant + Weight Quant + GEMM)
    # Assuming Weight Quant is same as Input Quant
    # And we do Layout for both
    t_total_est = t_gemm + 2*t_quant + 2*t_layout + 2*t_abs 
    # Actually quant includes absmax usually if not checking? No, in linear.py:
    #   input_amax = abs_max(flat_input)
    #   input_fp4 = quant_fp4(..., amax=input_amax)
    # So they are additive.
    
    print(f"Total FP4 Est:{t_total_est*1000:.3f} ms")
    print(f"Effective Speedup: {t_bf16/t_total_est:.2f}x")
    return t_bf16, t_total_est

if __name__ == "__main__":
    sizes = [4096, 8192, 16384]
    for s in sizes:
        benchmark_overhead(s, s, s)
