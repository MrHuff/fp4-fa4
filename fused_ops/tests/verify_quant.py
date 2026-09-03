
import torch
from fused_ops.fused_linear import fused_quant_fp4
from quartet2.quant import quant_fp4

def verify_quant():
    print("Verifying Fused Quantization Correctness...")
    device = "cuda"
    rows, cols = 32, 1024 # Small size for inspection
    x = torch.randn(rows, cols, device=device, dtype=torch.bfloat16)
    
    # Weights for RMSNorm
    norm_weight = torch.rand(cols, device=device, dtype=torch.bfloat16) + 0.5
    epsilon = 1e-5
    
    # Reference: RMSNorm -> SiLU -> Quant
    # Manual RMSNorm
    x_float = x.float()
    w_float = norm_weight.float()
    var = x_float.pow(2).mean(dim=-1, keepdim=True)
    x_norm = x_float * torch.rsqrt(var + epsilon) * w_float
    
    # Manual SiLU
    x_act = torch.nn.functional.silu(x_norm)
    
    # Quantize
    x_act_bf16 = x_act.to(torch.bfloat16)
    # Note: Quartet-II quant_fp4 computes global amax internally if not provided?
    # Or takes amax.
    # To match fused, we should compute global amax of x_act
    amax_ref = x_act_bf16.abs().max()
    
    ref_quant = quant_fp4(x_act_bf16, amax=amax_ref.to(torch.float32), scale_override=1.0, four_over_six=False) # Fused is RTN currently
    
    # Fused
    fused_quant = fused_quant_fp4(x, norm_weight, epsilon, scale_override=1.0)
    
    # Compare
    # 1. Global Amax (Fused computes it internally? But outputs global_scale? No, outputs NVFP4Quant.tensor_scale)
    # NVFP4Quant.tensor_scale for fused is "scale" (which is amax/448/6).
    # tensor_scale in quant_fp4 behaves similarly.
    
    print(f"Ref Scalar: {ref_quant.tensor_scale.item()}")
    print(f"Fused Scalar: {fused_quant.tensor_scale.item()}")
    
    # 2. FP4 Bits
    # This is hard to compare exactly bit-exact due to floating point ordering in RMSNorm accumulation?
    # But should be close.
    # Check match percentage
    match = (ref_quant.fp4 == fused_quant.fp4).float().mean()
    print(f"FP4 Bit Match: {match.item()*100:.2f}%")
    
    # 3. Micro Scales
    # FP8 e4m3
    # Convert to float to compare
    ref_scales = ref_quant.micro_scales.to(torch.float32)
    fused_scales = fused_quant.micro_scales.to(torch.float32)
    diff = (ref_scales - fused_scales).abs().mean()
    print(f"Scales Mean Diff: {diff.item()}")

if __name__ == "__main__":
    verify_quant()
