
import torch
import fused_ops
import numpy as np

def verify_backward_absmax():
    print("--- Verifying Fused Backward AbsMax Kernel ---")
    
    rows, cols = 128, 4096
    epsilon = 1e-6
    
    # Inputs
    x = torch.randn(rows, cols, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(cols, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    grad_output = torch.randn(rows, cols, device='cuda', dtype=torch.bfloat16) # dy
    
    # --------------------------------------------------------
    # 1. Python Exact Reference
    # --------------------------------------------------------
    print("Running Python Reference...")
    # Need to reimplement logic manually or use CustomNormFunc if importable.
    # Manual reimplementation to be safe and explicit here.
    
    x_ref = x.detach().clone().float()
    w_ref = w.detach().clone().float()
    dy_ref = grad_output.detach().clone().float()
    
    x_ref.requires_grad = True
    
    # Forward: SiLU -> AbsMax -> Scale
    z = torch.nn.functional.silu(x_ref)
    absmax_val = torch.max(torch.abs(z), dim=-1, keepdim=True)[0] # No epsilon in max, but usually +eps?
    # Kernel assumes we pass inv_rms_cache.
    # Forward logic: 
    # forward kernel computes s = max(|silu(x)*w|)? No, usually max(|silu(x)|) or max(|silu(x)*w|) depending on design.
    # Let's check fused_mxnorm_fast.cu:
    #   act_val = silu(val);
    #   act_with_gain = fabsf(act_val * w);
    #   local_max = fmaxf...
    # So it computes max of (SiLU(x) * w).
    
    # Wait, my backward derivation assumed s = max(|x|) or max(|z|).
    # If s = max(|z * w|), then w is inside the norm...
    # fused_mxnorm_fast.cu:
    #   norm_val = act_val * inv_rms * w;
    #   Wait, inv_rms is calculated from max(|act_val * w|)?
    #   Yes: "float act_with_gain = fabsf(act_val * w); local_max..."
    #   Then: "s_inv_rms = rsqrtf(row_block_sum_sq...)" for Block mode.
    #   For AbsMax mode (fast): 
    #     global_max calculation is usually done on scaled data?
    # Actually, fused_mxnorm_fast.cu (AbsMax mode):
    #   Computes global_amax for quantization scale.
    #   But what is "inv_rms"?
    #   AbsMax normalization usually means dividing by max(|x|).
    
    # Let's re-read fused_mxnorm_fast.cu carefully.
    # Phase 1: 
    #   act_with_gain = fabsf(act_val * w); 
    #   max_abs_val = max(act_with_gain);
    # Reduction -> row_max_abs_val.
    # s_inv_rms = 1.0f / (row_max_abs_val + epsilon);
    # Output: act_val * s_inv_rms * w? 
    #   No: "float norm_val = act_val * inv_rms * w;"
    #   If inv_rms = 1/max(|act*w|), then norm_val = (act*w) / max(|act*w|). 
    #   This normalizes the *weighted* activation to range [-1, 1].
    
    # OK, so Forward is:
    # z = SiLU(x)
    # v = z * w
    # s = max(|v|)
    # y = v / s
    
    # Backward must match this.
    # My derived backward in `backward_math.md` assumed difference.
    # It assumed y = (x/s) * w where s = max(|x|).
    
    # CRITICAL CHECK: Does `fused_backward_absmax.cu` match the Kernel Forward?
    # My kernel implementation:
    #   dz = dy * w * inv_s
    #   Is this correct if s depends on w?
    #   If s = max(|z*w|), then s depends on w!
    #   Then ds/dw is non-zero.
    #   This complicates dw calculation, but for dx (fixed w)?
    #   s depends on z. ds/dz is sparse.
    #   y = (z * w) / s(z).
    #   dy/dz = w/s - (z*w)/s^2 * ds/dz
    #   = w/s - y/s * ds/dz.
    #   ds/dz: s = max(|z*w|). ds/dz_i = w_i * sign(z_i*w_i) * I(i=kmax).
    #   So correction: y/s * w * sign * I
    #   = y/s * w * sign...
    
    # My Kernel Logic in `fused_backward_absmax.cu`:
    #   float dz = dy * w * inv_s;
    #   correction = (sum(dy * w * z) / s) * inv_s ...? 
    #   My kernel computes dp_sum = sum(dy * w * z).
    #   Kernel uses: correction = dp_sum * inv_s * inv_s = sum(dy*w*z)/s^2 = sum(dy * y)/s.
    #   And subtracts: correction * sign(z).
    #
    #   Wait, if s = max(|z*w|), then ds/dz includes w!
    #   The logic in derivation was for s = max(|z|).
    
    #   If s = max(|z*w|):
    #   ds/dz_i = w_i * sign(z_i * w_i) * I(...)
    #   Term 2: - (y / s) * (w_i * sign * I)
    
    #   My kernel subtracts: (sum(dy*y)/s) * sign(z_i).
    #   It MISSES the `w_i` factor in the correction term IF `s` is defined as max(|z*w|).
    
    #   HOWEVER, standard RMSNorm usually normalizes x (or z), then applies w.
    #   Let's check `fused_mxnorm_fast.cu` again.
    #   "float norm_val = act_val * inv_rms * w;"
    #   "s_inv_rms = 1.0f / (row_max_abs_val + epsilon);"
    #   "row_max_abs_val = ... max(fabsf(act_val * w))"
    
    #   Yes, it normalizes by the weighted max.
    
    #   Does this matter?
    #   If we define s' = max(|z|), and y' = z/s' * w.
    #   vs s = max(|z*w|), and y = z*w / s.
    #   If w is roughly constant magnitude, s approx s' * w_scale.
    #   y = z*w / (s' * w_scale) approx z/s'.
    #   Effectively the same output range [-1, 1].
    
    #   But mathematically, the gradient w.r.t z is different if s depends on w.
    #   If s depends on w, then correction term needs w.
    
    #   Let's verify via Python.
    
    z = torch.nn.functional.silu(x_ref)
    v = z * w_ref
    absmax = torch.max(torch.abs(v), dim=-1, keepdim=True)[0] + 1e-6
    y = v / absmax
    
    y.backward(dy_ref)
    dx_ref = x_ref.grad
    
    # --------------------------------------------------------
    # 2. CUDA Kernel
    # --------------------------------------------------------
    print("Running CUDA Kernel...")
    
    inv_s = 1.0 / absmax.squeeze().detach().to(dtype=torch.float32) # [rows]
    
    grad_input = torch.zeros_like(x)
    
    # Kernel expects inv_rms_cache.
    fused_ops.fused_backward_absmax(
        grad_input,     # dx output
        grad_output,    # dy input
        x,              # x input
        w,              # w input
        inv_s           # 1/s input
    )
    
    # --------------------------------------------------------
    # 3. Compare
    # --------------------------------------------------------
    
    dx_cuda = grad_input.float()
    
    # Check max diff
    diff = (dx_ref - dx_cuda).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"Max Diff: {max_diff:.8f}")
    print(f"Mean Diff: {mean_diff:.8f}")
    
    if max_diff < 1e-3: # BF16 precision tolerance
        print("SUCCESS: Gradients match!")
    else:
        print("FAILURE: Gradients mismatch.")
        # Debug printing
        print("Ref sample:", dx_ref[0, :5])
        print("Ker sample:", dx_cuda[0, :5])

if __name__ == "__main__":
    verify_backward_absmax()
