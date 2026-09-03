
import torch
import fused_ops._fused_ops as fused_ops
import time
import pandas as pd

def benchmark_fn(fn, warmup=10, iters=100):
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    
    # Timing
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    
    return start.elapsed_time(end) / iters

def run_benchmarks():
    shapes = [
        (128, 4096),
        (256, 4096),
        (512, 4096),
        (1024, 4096),
        (128, 8192),
        (256, 8192),
        (512, 8192),
    ]
    
    results = []
    
    print(f"{'Shape':<20} | {'Baseline':<15} | {'Mode0(RMS)':<15} | {'Mode1(AbsMax)':<15} | {'Mode2(BlkMax)':<15} | {'Fast(AbsMax)':<15} | {'Fast(BlkMax)':<15}")
    print("-" * 160)
    
    for rows, cols in shapes:
        try:
            torch.manual_seed(42)
            x = torch.randn(rows, cols, dtype=torch.bfloat16, device='cuda')
            w = torch.randn(cols, dtype=torch.bfloat16, device='cuda')
            epsilon = 1e-6
            scale_override = 1.0
            
            # Alloc outputs (reused)
            out_packed = torch.zeros(rows, cols // 2, dtype=torch.uint8, device='cuda')
            scales = torch.zeros(rows * cols // 16, dtype=torch.uint8, device='cuda')
            global_scale = torch.zeros((), dtype=torch.float32, device='cuda')
            inv_rms_cache = torch.zeros(rows, dtype=torch.float32, device='cuda')
            
            # 1. Baseline: fused_rmsnorm_act_quant_v2 (Optimized V2)
            def run_baseline():
                fused_ops.fused_rmsnorm_act_quant_v2(
                    out_packed, scales, global_scale, inv_rms_cache,
                    x, w, epsilon, scale_override, True 
                )
                
            t_baseline = benchmark_fn(run_baseline)
            
            # 2. MXNorm Mode 0 (RMS)
            def run_mx_mode0():
                fused_ops.fused_mxnorm(
                    out_packed, scales, global_scale, inv_rms_cache,
                    x, w, epsilon, scale_override, True, 0
                )
            t_mode0 = benchmark_fn(run_mx_mode0)
            
            # 3. MXNorm Mode 1 (AbsMax)
            def run_mx_mode1():
                fused_ops.fused_mxnorm(
                    out_packed, scales, global_scale, inv_rms_cache,
                    x, w, epsilon, scale_override, True, 1
                )
            t_mode1 = benchmark_fn(run_mx_mode1)

            # 4. MXNorm Mode 2 (BlockMax)
            def run_mx_mode2():
                fused_ops.fused_mxnorm(
                    out_packed, scales, global_scale, inv_rms_cache,
                    x, w, epsilon, scale_override, True, 2
                )
            t_mode2 = benchmark_fn(run_mx_mode2)
            
            # 4. Mode 2: Block-Max RMS
            def run_mx_mode2():
                fused_ops.fused_mxnorm(
                    out_packed, scales, global_scale, inv_rms_cache,
                    x, w, epsilon, scale_override, True, 2  # norm_mode=2
                )
            t_mode2 = benchmark_fn(run_mx_mode2)

            # 5. Fast MXNorm (AbsMax Only)
            def run_mx_fast():
                fused_ops.fused_mxnorm_fast(
                    out_packed, scales, global_scale, inv_rms_cache,
                    x, w, epsilon, scale_override, True
                )
            t_fast = benchmark_fn(run_mx_fast)

            # 6. Fast MXNorm (Block-Max Only)
            def run_mx_fast_block():
                fused_ops.fused_mxnorm_fast_block(
                    out_packed, scales, global_scale, inv_rms_cache,
                    x, w, epsilon, scale_override, True
                )
            t_fast_block = benchmark_fn(run_mx_fast_block)
            
            print(f"({rows}, {cols:<5}) | {t_baseline:.4f} ms      | {t_mode0:.4f} ms      | {t_mode1:.4f} ms | {t_mode2:.4f} ms | {t_fast:.4f} ms | {t_fast_block:.4f} ms")
            
            results.append({
                "rows": rows, "cols": cols,
                "baseline_ms": t_baseline,
                "mx_rms_ms": t_mode0,
                "mx_absmax_ms": t_mode1,
                "mx_block_ms": t_mode2,
                "mx_fast_ms": t_fast,
                "mx_fast_block_ms": t_fast_block
            })
        except RuntimeError as e:
            print(f"({rows}, {cols:<5}) | Skipped: {e}")

if __name__ == "__main__":
    run_benchmarks()
