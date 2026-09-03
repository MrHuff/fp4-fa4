import torch
import time
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import NVFP4BlockScaling, MXFP4BlockScaling, Format

def benchmark_te(m, n, k, recipe_name="NVFP4", steps=100, warmup=20):
    # Setup inputs
    x = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
    w = torch.randn(n, k, device='cuda', dtype=torch.bfloat16) # TE Linear takes (out_features, in_features)
    bias = torch.zeros(n, device='cuda', dtype=torch.bfloat16)

    # Setup TE Linear
    # bias=True to match typical usage, or False? 
    # qutlass benchmark was without bias. Let's do bias=False to be comparable.
    layer = te.Linear(k, n, bias=False).to('cuda').to(torch.bfloat16)
    
    # Configure recipe
    if recipe_name == "NVFP4":
        recipe = NVFP4BlockScaling(fp4_format=Format.E2M1, fp8_format=Format.E4M3)
    elif recipe_name == "MXFP4":
        recipe = MXFP4BlockScaling(fp4_format=Format.E2M1, fp8_format=Format.E4M3)
    elif recipe_name == "BF16":
        recipe = None # standard BF16
    else:
        raise ValueError(f"Unknown recipe: {recipe_name}")

    # Warmup
    with te.fp8_autocast(enabled=(recipe is not None), fp8_recipe=recipe):
        for _ in range(warmup):
            out = layer(x)
    
    torch.cuda.synchronize()
    start = time.time()
    
    with te.fp8_autocast(enabled=(recipe is not None), fp8_recipe=recipe):
        for _ in range(steps):
            out = layer(x)
            
    torch.cuda.synchronize()
    end = time.time()
    
    avg_time = (end - start) / steps * 1000 # ms
    print(f"[{recipe_name}] {m}x{n}x{k}: {avg_time:.4f} ms")
    return avg_time

if __name__ == "__main__":
    sizes = [4096, 8192, 16384]
    
    print("--- Transformer Engine Benchmarks ---")
    for size in sizes:
        print(f"\nSize: {size}")
        t_bf16 = benchmark_te(size, size, size, "BF16")
        t_nvfp4 = benchmark_te(size, size, size, "NVFP4")
        t_mxfp4 = benchmark_te(size, size, size, "MXFP4")
        
        print(f"Speedup NVFP4 vs BF16: {t_bf16 / t_nvfp4:.2f}x")
        print(f"Speedup MXFP4 vs BF16: {t_bf16 / t_mxfp4:.2f}x")
