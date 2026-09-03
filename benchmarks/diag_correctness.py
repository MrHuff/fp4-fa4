"""Deep diagnostic: Is the fused kernel quantization CORRECT?
Compare dequant of fused kernel vs the UNQUANTIZED rmsnorm+silu output."""
import torch, torch.nn.functional as F, os
from torch.utils.cpp_extension import load
import transformer_engine.pytorch as te
from transformer_engine.pytorch import NVFP4Quantizer

TE_ROOT = '/workspace/low-bits-training/TransformerEngine'
TE_INCLUDE = os.path.join(TE_ROOT, 'transformer_engine/common/include')
TE_LIB_DIR = os.path.join(TE_ROOT, 'build/cmake')
CSRC = '/workspace/fp4_matmul/fused_ops/csrc'
cuda_lib = '/usr/local/cuda/lib64'

print("Loading extension...", flush=True)
te_fused = load(name='te_fused_rmsnorm_ext',
    sources=[os.path.join(CSRC, 'te_fused_rmsnorm_ext.cpp')],
    extra_include_paths=[os.path.join(TE_ROOT, 'transformer_engine/common/include'), '/usr/local/cuda/include'],
    extra_cflags=['-std=c++17'],
    extra_ldflags=[f'-L{TE_LIB_DIR}', '-ltransformer_engine', f'-Wl,-rpath,{TE_LIB_DIR}',
                   f'-L{cuda_lib}', '-lcudart', '-lnvrtc', f'-Wl,-rpath,{cuda_lib}'],
    verbose=False)
print("Extension loaded.\n", flush=True)

FP4_LUT = [0,0.5,1,1.5,2,3,4,6,-0,-0.5,-1,-1.5,-2,-3,-4,-6]
FP4_MAX = 6.0
FP8_MAX = 448.0

def deq(fp4, sc_padded, amax, m, k):
    lut = torch.tensor(FP4_LUT, device='cuda', dtype=torch.float32)
    d = fp4.view(torch.uint8).to(torch.int32)
    u = torch.stack((d & 0x0F, d >> 4), dim=-1).reshape(m, k)
    fv = lut[u]
    sc = sc_padded.view(torch.float8_e4m3fn).to(torch.float32)[:m, :k//16]
    ts = amax / (FP4_MAX * FP8_MAX)
    return (fv.view(-1, 16) * ts * sc.reshape(-1, 1)).view(m, k)

M, K = 128, 256
torch.manual_seed(42)
x = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
w = torch.ones(K, device='cuda', dtype=torch.bfloat16)
inv_rms = torch.rsqrt(x.float().pow(2).mean(dim=-1) + 1e-5)

# Golden: RMSNorm + SiLU in eager (unquantized, float32)
normed = x.float() * inv_rms.unsqueeze(-1) * w.float()
golden = F.silu(normed)  # float32
golden_bf16 = golden.to(torch.bfloat16)

# TE reference: quantize the bf16 golden
q = NVFP4Quantizer(rowwise=True, columnwise=False)
ref_qt = q.quantize(golden_bf16)
ref_amax = ref_qt._amax_rowwise
ref_deq = ref_qt.dequantize(dtype=torch.float32)

# Fused: pass the SAME amax as reference
fp4_fused, sc_fused = te_fused.fused_te_quantize_rmsnorm_silu(x, inv_rms, w, ref_amax)
torch.cuda.synchronize()
fused_deq = deq(fp4_fused, sc_fused, ref_amax.item(), M, K)

# === Compare ===
cos_ref_vs_golden = F.cosine_similarity(ref_deq.flatten().unsqueeze(0), golden.flatten().unsqueeze(0)).item()
cos_fused_vs_golden = F.cosine_similarity(fused_deq.flatten().unsqueeze(0), golden.flatten().unsqueeze(0)).item()
cos_fused_vs_ref = F.cosine_similarity(fused_deq.flatten().unsqueeze(0), ref_deq.flatten().unsqueeze(0)).item()

print(f"cos(TE-deq, golden-fp32):     {cos_ref_vs_golden:.6f}")
print(f"cos(fused-deq, golden-fp32):  {cos_fused_vs_golden:.6f}")
print(f"cos(fused-deq, TE-deq):       {cos_fused_vs_ref:.6f}")

# Check errors
err_ref = (ref_deq - golden).abs()
err_fused = (fused_deq - golden).abs()
print(f"\nTE-ref error:   max={err_ref.max():.4f}  mean={err_ref.mean():.6f}")
print(f"Fused error:    max={err_fused.max():.4f}  mean={err_fused.mean():.6f}")

# Row 0 element-by-element comparison (first 16 elements)
print(f"\nRow 0 first 16 elements:")
print(f"  golden:    {golden[0,:16].tolist()}")
print(f"  TE deq:    {ref_deq[0,:16].tolist()}")
print(f"  fused deq: {fused_deq[0,:16].tolist()}")

# Check SIGNS
te_signs = torch.sign(ref_deq[0,:16])
fused_signs = torch.sign(fused_deq[0,:16])
golden_signs = torch.sign(golden[0,:16])
sign_match_te = (te_signs == golden_signs).sum().item()
sign_match_fused = (fused_signs == golden_signs).sum().item()
print(f"\n  Sign match TE:    {sign_match_te}/16")
print(f"  Sign match fused: {sign_match_fused}/16")

# Check if fused output is ZERO
print(f"\n  fused fp4 nonzero: {(fp4_fused != 0).sum().item()}/{fp4_fused.numel()}")
print(f"  fused fp4 row0 first32: {fp4_fused[0,:32].tolist()}")

# Check scale values
print(f"\n  fused scales row0: {sc_fused.view(torch.float8_e4m3fn).to(torch.float32)[0,:K//16].tolist()}")
print(f"  TE scales row0:    {ref_qt._rowwise_scale_inv.view(torch.float8_e4m3fn).to(torch.float32)[0,:K//16].tolist()}")

# Additional: try with nvte_quantize_rmsnorm (NO silu, just rmsnorm + quant)
# This should quantize rmsnorm(x) = x * inv_rms * w = x * inv_rms (since w=1)
rmsnorm_out = (x.float() * inv_rms.unsqueeze(-1) * w.float()).to(torch.bfloat16)
ref_rmsnorm_qt = q.quantize(rmsnorm_out) 
ref_rmsnorm_amax = ref_rmsnorm_qt._amax_rowwise

fp4_rmsnorm, sc_rmsnorm = te_fused.fused_te_quantize_rmsnorm(x, inv_rms, w, ref_rmsnorm_amax)
torch.cuda.synchronize()

rmsnorm_deq = deq(fp4_rmsnorm, sc_rmsnorm, ref_rmsnorm_amax.item(), M, K)
ref_rmsnorm_deq = ref_rmsnorm_qt.dequantize(dtype=torch.float32)

cos_rmsnorm = F.cosine_similarity(rmsnorm_deq.flatten().unsqueeze(0), ref_rmsnorm_deq.flatten().unsqueeze(0)).item()
print(f"\n=== RMSNorm-only (no SiLU) ===")
print(f"  cos(fused-rmsnorm-deq, TE-deq): {cos_rmsnorm:.6f}")

# Now test: what if we DON'T use the rmsnorm path at all?
# Just quantize golden_bf16 through our extension using identity rmsnorm
inv_rms_ones = torch.ones(M, device='cuda', dtype=torch.float32)
fp4_ident, sc_ident = te_fused.fused_te_quantize_rmsnorm(golden_bf16, inv_rms_ones, w, ref_amax)
torch.cuda.synchronize()
ident_deq = deq(fp4_ident, sc_ident, ref_amax.item(), M, K)
cos_ident = F.cosine_similarity(ident_deq.flatten().unsqueeze(0), ref_deq.flatten().unsqueeze(0)).item()
print(f"\n=== Identity RMSNorm quant (fused path, no activation) vs TE ref ===")
print(f"  cos: {cos_ident:.6f}")

# Test: quantize through rmsnorm_silu extension with inv_rms=1,w=1 (=just SiLU path)
act_bf16 = F.silu(x.float()).to(torch.bfloat16)
ref_act_qt = q.quantize(act_bf16)
ref_act_amax = ref_act_qt._amax_rowwise

inv_rms_ones = torch.ones(M, device='cuda', dtype=torch.float32)
fp4_silu, sc_silu = te_fused.fused_te_quantize_rmsnorm_silu(x, inv_rms_ones, w, ref_act_amax)
torch.cuda.synchronize()
silu_deq = deq(fp4_silu, sc_silu, ref_act_amax.item(), M, K)
ref_act_deq = ref_act_qt.dequantize(dtype=torch.float32)

cos_silu_only = F.cosine_similarity(silu_deq.flatten().unsqueeze(0), ref_act_deq.flatten().unsqueeze(0)).item()
print(f"\n=== SiLU-only (inv_rms=1) through fused path vs TE ref ===")
print(f"  cos: {cos_silu_only:.6f}")

# Finally: compare first 8 elements of dequant to see WHAT the fused kernel is producing
print(f"\n=== SiLU-only element comparison row 0 first 8 ===")
print(f"  golden silu:  {act_bf16[0,:8].float().tolist()}")
print(f"  TE deq:       {ref_act_deq[0,:8].tolist()}")
print(f"  fused deq:    {silu_deq[0,:8].tolist()}")

print("\nDone.", flush=True)
