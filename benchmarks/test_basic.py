"""Minimal test: just quantize + ungrouped GEMM to verify .so build integrity."""
import sys, torch

torch.cuda.init()
_ = torch.zeros(1, device='cuda')
torch.cuda.synchronize()
print("1. CUDA OK", flush=True)

sys.path.insert(0, '/workspace/fp4_matmul/ThunderKittens/kernels/gemm/nvfp4_b200')
from _C import nvfp4_gemm, nvfp4_quantize
print("2. Import OK", flush=True)


def tkq(t):
    r, c = t.shape
    f = torch.empty(r, c // 2, dtype=torch.float4_e2m1fn_x2, device='cuda')
    s = torch.empty(r, c // 16, dtype=torch.float8_e4m3fn, device='cuda')
    g = torch.empty(1, dtype=torch.float32, device='cuda')
    nvfp4_quantize(t, f, s, g, False)
    torch.cuda.synchronize()
    return f, s, g


# Quant tests
for M, K in [(256, 256), (1024, 1024), (2048, 4096)]:
    try:
        x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda') * 0.1
        xf, xs, xg = tkq(x)
        print("3. Quant {}x{}: sg={:.6f}".format(M, K, xg.item()), flush=True)
    except Exception as e:
        print("3. Quant {}x{} FAILED: {}".format(M, K, str(e)[:80]), flush=True)

# GEMM test
M, K, N = 2048, 4096, 4096
try:
    x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda') * 0.1
    w = torch.randn(N, K, dtype=torch.bfloat16, device='cuda') * 0.02
    xf, xs, xg = tkq(x)
    wf, ws, wg = tkq(w)
    o = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
    nvfp4_gemm(xf, xs, xg, wf, ws, wg, o)
    torch.cuda.synchronize()
    print("4. GEMM {}x{}x{} OK, nan={}".format(M, K, N, o.isnan().any().item()), flush=True)
except Exception as e:
    print("4. GEMM {}x{}x{} FAILED: {}".format(M, K, N, str(e)[:80]), flush=True)

print("DONE", flush=True)
