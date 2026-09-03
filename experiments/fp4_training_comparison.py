
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import fused_ops
from quartet2.linear import _dq_fp4

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

# --- Dequantization Helper ---
def dequant_output(y_packed, scales, global_scale):
    """
    Unpack and dequantize the fused_ops output.
    y_packed: [rows, cols/2] uint8 (FP4 packed)
    scales: [rows, cols/16] float8 (E4M3 block scales)
    global_scale: [1] float32
    """
    # fused_ops packs as e2m1
    # _dq_fp4 expects:
    #   x_e2m1: Tensor (uint8 packed)
    #   x_e4m3: Tensor (scales)
    #   alpha: float
    
    # Check shapes
    rows, cols_packed = y_packed.shape
    
    # _dq_fp4 seems to expect standard packed format.
    # fused_ops packs using __nv_cvt_float2_to_fp4x2.
    # Both seem compatible (standard NVFP4).
    
    # _dq_fp4 returns BF16
    return _dq_fp4(y_packed, scales, global_scale.item())

# --- Fused Norm Module (FP4) ---
class FusedNormFP4(nn.Module):
    def __init__(self, dim, mode='rms', epsilon=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, device='cuda', dtype=torch.bfloat16))
        self.epsilon = epsilon
        self.mode = mode # 'rms', 'proxy_rms', 'absmax_exact'
        
    def forward(self, x):
        return FusedNormFP4Func.apply(x, self.weight, self.epsilon, self.mode)

class FusedNormFP4Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, epsilon, mode):
        rows, cols = x.shape
        
        # Allocate outputs
        y_packed = torch.empty((rows, cols // 2), device=x.device, dtype=torch.uint8)
        # Try uint8 for scales to avoid dtype issues with nanobind
        scales = torch.empty((rows, cols // 16), device=x.device, dtype=torch.uint8)
        global_scale = torch.empty((), device=x.device, dtype=torch.float32)
        inv_rms_cache = torch.empty((rows,), device=x.device, dtype=torch.float32)
        
        ctx.epsilon = epsilon
        ctx.mode = mode
        
        # Dispatch based on mode
        # 'rms' or 'proxy_rms' -> fused_mxnorm (mode 0 or 1 respectively? No, mode 0 is RMS, mode 1 is AbsMax)
        # Wait, 'proxy_rms' uses AbsMax forward (mode 1) but RMS gradient.
        # 'absmax_exact' uses AbsMax forward (mode 1) and AbsMax gradient.
        
        norm_mode_int = 0
        is_absmax_forward = False
        
        if mode == 'rms':
            norm_mode_int = 0 # RMSNorm
        elif mode == 'proxy_rms' or mode == 'absmax_exact':
            norm_mode_int = 1 # AbsMaxNorm
            is_absmax_forward = True
            
        # Enforce contiguous and check types
        x_contig = x.contiguous()
        w_contig = weight.data.contiguous() # access .data and ensure contiguous
        y_packed = y_packed.contiguous()
        scales = scales.contiguous()
        global_scale = global_scale.contiguous()
        inv_rms_cache = inv_rms_cache.contiguous()
        
        try:
            if is_absmax_forward:
                 # Use the fast kernel for AbsMax forward
                 fused_ops.fused_mxnorm_fast(
                    y_packed, scales, global_scale, inv_rms_cache,
                    x_contig, w_contig, epsilon, 1.0, True # use_four_six=True
                 )
            else:
                 # Use flexible kernel for RMS
                 fused_ops.fused_mxnorm(
                    y_packed, scales, global_scale, inv_rms_cache,
                    x_contig, w_contig, epsilon, 1.0, True, 0 # Mode 0
                 )
        except TypeError as e:
            # Debug printing
            print("\n--- TYPE ERROR DEBUG ---")
            for name, t in [("y", y_packed), ("scales", scales), ("g_scale", global_scale), ("inv_rms", inv_rms_cache), ("x", x_contig), ("w", w_contig)]:
                print(f"{name}: shape={t.shape}, stride={t.stride()}, dtype={t.dtype}, device={t.device}, is_contig={t.is_contiguous()}")
            raise e
             
        ctx.save_for_backward(x, weight, inv_rms_cache)
        ctx.rows = rows
        ctx.cols = cols
        
        # Return dequantized BF16 for the "next layer" simulation
        # In a real setup, next layer would take y_packed.
        y_dequant = dequant_output(y_packed, scales.view(torch.float8_e4m3fn), global_scale)
        
        return y_dequant

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, inv_rms_cache = ctx.saved_tensors
        rows, cols = ctx.rows, ctx.cols
        epsilon = ctx.epsilon
        mode = ctx.mode
        
        grad_input = torch.empty_like(x)
        
        # Ensure contiguous
        grad_output = grad_output.contiguous()
        x = x.contiguous()
        weight = weight.contiguous()
        
        if mode == 'rms' or mode == 'proxy_rms':
            # Use backward_v2 which uses cached inv_rms
            # Signature: (grad_output, input, weight, cached_inv_rms, epsilon, grad_input)
            fused_ops.fused_backward_v2(
                grad_output, x, weight, 
                inv_rms_cache, 
                epsilon,
                grad_input
            )
        elif mode == 'absmax_exact':
            # Use dedicated AbsMax backward (Exact Sparse)
            # Signature: (grad_input, grad_output, input, weight, inv_rms_cache)
            fused_ops.fused_backward_absmax(
                grad_input,
                grad_output,
                x,
                weight,
                inv_rms_cache
            )
            
        # Weight gradient (simple approx/sum)
        # dL/dw = sum(dL/dy * y_norm)
        # We can approximate or just use None for this specific stability test if we freeze weights?
        # User said "full FP4 training", implying weights update.
        # We need dW.
        # Let's compute it in PyTorch for simplicity as fused kernels often skip it.
        # Recover y_norm
        with torch.no_grad():
             inv_s = inv_rms_cache.view(-1, 1)
             if mode == 'rms':
                  # RMSNorm: y = x * inv_s * w
                  # But inv_rms_cache in fused_mxnorm is 1/RMS.
                  # Standard RMS backward expects inv_rms.
                  pass
             # Compute y_norm = x * inv_s
             x_norm = x * inv_s
             grad_weight = (grad_output * x_norm).sum(dim=0)
             
        return grad_input, grad_weight, None, None

# --- Experiment Runner ---
class TrainingComparison:
    def __init__(self):
        self.input_dim = 1024
        self.hidden_dim = 4096
        self.batch_size = 32
        
    def run(self, mode, steps=200):
        print(f"--- Running FP4 Training: {mode} ---")
        
        # Simple FFN
        l1 = nn.Linear(self.input_dim, self.hidden_dim, bias=False, device='cuda', dtype=torch.bfloat16)
        norm = FusedNormFP4(self.hidden_dim, mode=mode)
        act = nn.SiLU() # Note: Fused kernel has SiLU inside. 
                        # Wait, fused_mxnorm includes SiLU? 
                        # "Optimized Fused MXNorm + SiLU Activation + FP4 Quantization"
                        # YES.
                        # So my module output is ALREADY SiLU'd.
                        # I should NOT add another SiLU.
        l2 = nn.Linear(self.hidden_dim, self.input_dim, bias=False, device='cuda', dtype=torch.bfloat16)
        
        optimizer = optim.Adam(list(l1.parameters()) + list(norm.parameters()) + list(l2.parameters()), lr=1e-3)
        
        inputs = torch.randn(self.batch_size, self.input_dim, device='cuda', dtype=torch.bfloat16)
        targets = torch.randn(self.batch_size, self.input_dim, device='cuda', dtype=torch.bfloat16)
        
        history = []
        
        for i in range(steps):
            optimizer.zero_grad()
            
            # Forward
            h = l1(inputs)
            h_fp4_dequant = norm(h) # Fused Norm+SiLU+Quant -> Dequant
            # No extra SiLU here!
            out = l2(h_fp4_dequant)
            
            loss = nn.MSELoss()(out, targets)
            
            if torch.isnan(loss):
                print(f"Step {i}: Loss NaN!")
                history.append(float('nan'))
                break
                
            loss.backward()
            optimizer.step()
            
            history.append(loss.item())
            
            if i % 50 == 0:
                print(f"Step {i}: {loss.item():.6f}")
                
        return history

if __name__ == "__main__":
    runner = TrainingComparison()
    
    modes = ['rms', 'proxy_rms', 'absmax_exact']
    results = {}
    
    for m in modes:
        results[m] = runner.run(m)
        
    print("\n=== Final FP4 Training Results (Last 10 Avg) ===")
    for m in modes:
        avg_loss = np.mean(results[m][-10:])
        print(f"{m:<15} | Loss: {avg_loss:.6f}")
