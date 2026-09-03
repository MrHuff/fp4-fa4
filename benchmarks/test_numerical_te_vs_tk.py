"""
Numerical correctness: TE GEMM vs TK GEMM vs BF16 reference.

Uses the same import/init pattern as the working bench_tk_grouped_gemm.py.
Runs TE and TK GEMMs on separate QKV-like shapes and compares outputs.
"""
import sys, os, torch

# TK imports — same as bench_tk_grouped_gemm.py
TK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      '../ThunderKittens/kernels/gemm/nvfp4_b200')
sys.path.insert(0, TK_DIR)
from _C import nvfp4_gemm, nvfp4_quantize

# TE imports
sys.path.insert(0, '/workspace/low-bits-training')
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch.constants import TE_DType
from low_bits_training.quantization.fused_te_linear import _fast_quantize
from bench_te_quant_tk_gemm import te_nvfp4_to_tk_format


def _patch(t):
    if not hasattr(t, '_with_gemm_swizzled_scales'):
        t._with_gemm_swizzled_scales = False
    return t


def tk_quantize(x_bf16):
    M, K = x_bf16.shape
    fp4 = torch.empty(M, K // 2, dtype=torch.float4_e2m1fn_x2, device="cuda")
    sc = torch.empty(M, K // 16, dtype=torch.float8_e4m3fn, device="cuda")
    sg = torch.empty(1, dtype=torch.float32, device="cuda")
    nvfp4_quantize(x_bf16, fp4, sc, sg, False)
    return fp4, sc, sg


def cos_sim(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().flatten().unsqueeze(0),
        b.float().flatten().unsqueeze(0)).item()


def main():
    device = "cuda"
    configs = [
        {"label": "Llama8B Q",  "M": 2048, "K": 4096, "N": 4096},
        {"label": "Llama8B KV", "M": 2048, "K": 4096, "N": 1024},
        {"label": "Llama70B Q", "M": 2048, "K": 8192, "N": 8192},
        {"label": "FFN gate",   "M": 2048, "K": 4096, "N": 14336},
        {"label": "Square 4k",  "M": 4096, "K": 4096, "N": 4096},
    ]

    print("=" * 110)
    print("  Numerical Comparison: TE GEMM vs TK GEMM vs BF16 Reference (per-matrix amax)")
    print("  Each matrix quantized SEPARATELY — respects per-split amaxes")
    print("=" * 110)
    print()
    print(f"  {'Config':<14} | {'Method':<20} | {'MaxAbsErr':>10} {'MeanAbsErr':>10} {'CosSim':>10} | {'Notes'}")
    print(f"  {'-'*100}")

    for cfg in configs:
        M, K, N = cfg["M"], cfg["K"], cfg["N"]
        label = cfg["label"]

        torch.manual_seed(42)
        x = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.1
        w = torch.randn(N, K, dtype=torch.bfloat16, device=device) * 0.02

        # BF16 reference
        ref = torch.mm(x.float(), w.float().T).bfloat16()

        # ---- TE GEMM (cuBLASLt) ----
        xq = _patch(_fast_quantize(x))
        wq = _patch(_fast_quantize(w))
        workspace = torch.empty(33554432, dtype=torch.uint8, device=device)
        te_out = torch.empty(M, N, device=device, dtype=torch.bfloat16)
        tex.generic_gemm(
            wq, True, xq, False, te_out, None, TE_DType[torch.bfloat16],
            None, TE_DType[torch.bfloat16], False, None, False, workspace,
            workspace.shape[0], False, False,
        )

        # ---- N×TK (TK quant + TK GEMM) ----
        txf, txs, txg = tk_quantize(x)
        twf, tws, twg = tk_quantize(w)
        tk_out = torch.zeros(M, N, dtype=torch.bfloat16, device=device)
        nvfp4_gemm(txf, txs, txg, twf, tws, twg, tk_out)

        # ---- TE→TK (TE quant + TK GEMM) ----
        exf, exs, exg = te_nvfp4_to_tk_format(xq, M, K)
        ewf, ews, ewg = te_nvfp4_to_tk_format(wq, N, K)
        tetk_out = torch.zeros(M, N, dtype=torch.bfloat16, device=device)
        nvfp4_gemm(exf, exs, exg, ewf, ews, ewg, tetk_out)

        torch.cuda.synchronize()

        # Global scale comparison
        amax_x = xq._amax_rowwise.item()
        amax_w = wq._amax_rowwise.item()
        te_alpha = amax_x * amax_w / (6*6*448*448)
        tk_gs = txg.item() * twg.item()

        # Print vs BF16
        for name, out in [("TE (cuBLASLt)", te_out), ("N×TK (TK quant)", tk_out), ("TE→TK (TE quant)", tetk_out)]:
            d = (out.float() - ref.float()).abs()
            cs = cos_sim(out, ref)
            notes = ""
            if name == "TE (cuBLASLt)":
                notes = f"alpha={te_alpha:.2e}"
            elif name == "N×TK (TK quant)":
                notes = f"gs={tk_gs:.2e}"
            print(f"  {label:<14} | {name:<20} | {d.max().item():>10.6f} {d.mean().item():>10.6f} {cs:>10.6f} | {notes}")

        # Direct TE vs TK
        d_te_tk = (te_out.float() - tk_out.float()).abs()
        d_te_tetk = (te_out.float() - tetk_out.float()).abs()
        bw = torch.equal(te_out, tetk_out)
        print(f"  {'':14} | {'TE↔TK (diff quant)':<20} | {d_te_tk.max().item():>10.6f} {d_te_tk.mean().item():>10.6f} {cos_sim(te_out, tk_out):>10.6f} |")
        print(f"  {'':14} | {'TE↔TE→TK (same q!)':<20} | {d_te_tetk.max().item():>10.6f} {d_te_tetk.mean().item():>10.6f} {cos_sim(te_out, tetk_out):>10.6f} | bw={'✓' if bw else '✗'}")
        print(f"  {'-'*100}")

    # Scale equivalence summary
    print()
    print("  Global scale equivalence check:")
    print("  TE: alpha = amax_A * amax_B / (6² × 448²)")
    print("  TK: global_scale = sg_A × sg_B, where sg = amax / (6 × 448)")
    print("  → TE_alpha ≡ TK_global_scale (mathematically identical)")


if __name__ == "__main__":
    main()
