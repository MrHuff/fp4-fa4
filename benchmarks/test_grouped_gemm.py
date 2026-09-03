"""Test grouped TK GEMM correctness using TE quantize (bypasses TK quantize bug)."""
import sys, os
sys.path.insert(0, '/workspace/fp4_matmul/ThunderKittens/kernels/gemm/nvfp4_b200')
sys.path.insert(0, '/workspace/fp4_matmul/benchmarks')

import torch
torch.cuda.init()
_ = torch.zeros(1, device='cuda')
torch.cuda.synchronize()
print("CUDA OK", flush=True)

from _C import nvfp4_gemm, nvfp4_grouped_gemm
print("TK Import OK", flush=True)

# Use TE quantize
import transformer_engine.pytorch as te
from bench_te_quant_tk_gemm import te_nvfp4_to_tk_format, _make_te_quantizer_rowonly
print("TE Import OK", flush=True)


def te_quant_for_tk(x_bf16):
    """Quantize using TE, convert to TK format."""
    M, K = x_bf16.shape
    from bench_te_quant_tk_gemm import NVFP4Quantizer, tex
    q = NVFP4Quantizer(
        fp4_dtype=tex.DType.kFloat4E2M1, rowwise=True, columnwise=False,
        with_amax_reduction=False, amax_reduction_group=None,
        with_rht=False, with_post_rht_amax=False,
    )
    q.optimize_for_gemm = False  # TE's C++ binding checks this attribute
    nvfp4 = q(x_bf16)
    fp4, sc, sg = te_nvfp4_to_tk_format(nvfp4, M, K)
    return fp4, sc, sg


def cs(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().flatten().unsqueeze(0),
        b.float().flatten().unsqueeze(0)
    ).item()


torch.manual_seed(42)
M, K = 2048, 4096
N_dims = [4096, 1024, 1024]  # QKV

x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda') * 0.1
w_list = [torch.randn(N, K, dtype=torch.bfloat16, device='cuda') * 0.02 for N in N_dims]
ref = [torch.mm(x.float(), w.float().T).bfloat16() for w in w_list]
print("Data generated", flush=True)

# TE quantize
txf, txs, txg = te_quant_for_tk(x)
tw_list = [te_quant_for_tk(w) for w in w_list]
torch.cuda.synchronize()
print(f"TE Quant OK: A_sg={txg.item():.8f} B_sg={[round(t[2].item(),8) for t in tw_list]}", flush=True)

# === N × separate TK GEMM ===
sep_outs = []
for g in range(len(N_dims)):
    o = torch.zeros(M, N_dims[g], dtype=torch.bfloat16, device='cuda')
    nvfp4_gemm(txf, txs, txg, tw_list[g][0], tw_list[g][1], tw_list[g][2], o)
    sep_outs.append(o)
torch.cuda.synchronize()
print("N×TK separate OK", flush=True)

# === 1 × Grouped TK GEMM ===
wc_fp4 = torch.cat([t[0] for t in tw_list], dim=0)
wc_sc = torch.cat([t[1] for t in tw_list], dim=0)
# Pre-compute per-tile B_sg tensor on GPU (new API)
Nb = 256
b_sg_per_tile_list = []
for gi, N in enumerate(N_dims):
    n_tiles = N // Nb
    b_sg_per_tile_list.extend([tw_list[gi][2].item()] * n_tiles)
b_sg_per_tile = torch.tensor(b_sg_per_tile_list, dtype=torch.float32, device='cuda')
go = torch.zeros(M, sum(N_dims), dtype=torch.bfloat16, device='cuda')
nvfp4_grouped_gemm(txf, txs, txg, wc_fp4, wc_sc, b_sg_per_tile, go)
torch.cuda.synchronize()
print("Grouped TK GEMM OK", flush=True)

# === Compare ===
grp_outs = list(torch.split(go, N_dims, dim=1))
print()
print(f"{'Split':<6}| {'Sep maxerr':>12} {'cos':>8} | {'Grp maxerr':>12} {'cos':>8} | {'Match':>10}")
print("-" * 70)
all_ok = True
for g, gn in enumerate("QKV"):
    sep = sep_outs[g]
    grp = grp_outs[g].contiguous()
    ds = (sep.float() - ref[g].float()).abs()
    dg = (grp.float() - ref[g].float()).abs()
    bw = torch.equal(sep, grp)
    if not bw:
        diff = (sep.float() - grp.float()).abs()
        all_ok = False
    print(f"  {gn:<4}| {ds.max().item():>12.6f} {cs(sep,ref[g]):>8.5f} | {dg.max().item():>12.6f} {cs(grp,ref[g]):>8.5f} | {'✓ bitwise' if bw else '✗ DIFF'}", flush=True)
    if not bw:
        print(f"      | sep vs grp: max_diff={diff.max().item():.6f} cos={cs(sep,grp):.6f}", flush=True)

print()
print(f"Per-group B_sg: {b_sg.tolist()}")
print(f"Shared   A_sg:  {txg.item()}")
print(f"\nResult: {'ALL BITWISE MATCH ✓' if all_ok else 'SOME DIFFER ✗'}")
print("DONE", flush=True)
