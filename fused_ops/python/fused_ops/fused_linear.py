import torch
from . import _fused_ops
from quartet2.linear import _fp4_mm, to_blocked
from collections import namedtuple

# Torch-compile compatible wrapper
@torch.library.custom_op("clover::fused_pre_quant", mutates_args=("out", "scales", "global_scale_out", "global_amax"))
def _fused_pre_quant_op(
    out: torch.Tensor, 
    scales: torch.Tensor, 
    global_scale_out: torch.Tensor, 
    input: torch.Tensor, 
    weight: torch.Tensor, 
    epsilon: float, 
    global_amax: torch.Tensor, 
    scale_override: float
) -> None:
    _fused_ops.fused_pre_quant(
        out, 
        scales,  # already uint8 — no .view() needed
        global_scale_out, 
        input, 
        weight.data, 
        epsilon, 
        global_amax, 
        scale_override
    )

NVFP4Quant = namedtuple("NVFP4Quant", ["fp4", "micro_scales", "tensor_scale"])

def fused_quant_fp4(x, norm_weight, epsilon=1e-5, scale_override=1.0, global_amax=None) -> NVFP4Quant:
    rows, cols = x.shape
    device = x.device
    
    out = torch.empty((rows, cols // 2), device=device, dtype=torch.uint8)
    scales = torch.empty((rows, cols // 16), device=device, dtype=torch.uint8)
    global_scale_out = torch.empty((), device=device, dtype=torch.float32)
    
    if global_amax is None:
        global_amax = torch.zeros((), device=device, dtype=torch.float32)
    else:
        global_amax.fill_(0)
        
    _fused_pre_quant_op(
        out, scales, global_scale_out, 
        x, norm_weight, epsilon, 
        global_amax, scale_override
    )
    
    return NVFP4Quant(out, scales, global_scale_out)

class FusedQuartetFn(torch.autograd.Function):

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: (B*S, Out) or reshaped?
        # Linear Backward: d(L)/d(Y_quant).
        # We need to perform Linear Backward (dY @ W.T) to get d(Input_Quant).
        # Quartet-II's `_fp4_mm` is forward.
        # Backward of Quantization is usually STE.
        # But Quartet-II uses `dequant_tp_had_quant` for backward inputs?
        # For simplicity in this demo, we assume standard BF16 Linear Backward for the GEMM part,
        # then we fuse the RMSNorm+Act backward into the result.
        
        input, weight, norm_weight, had = ctx.saved_tensors
        epsilon = ctx.epsilon
        scale_override = ctx.scale_override
        
        # 1. Linear Backward (BF16 for simplicity/baseline)
        # grad_input_linear = grad_output @ weight
        grad_linear_input = grad_output.mm(weight) # Standard Matmul (BF16)
        
        # 2. Fused Pre-Quant Backward
        # Inputs: grad_linear_input (dY), input (X), norm_weight (W)
        # Output: dX
        # Note: We need dW_norm too. The fused kernel only computes dX currently.
        # We will compute dX with kernel.
        
        rows, cols = input.shape
        grad_input = torch.empty_like(input)
        
        # Ensure contiguous
        grad_linear_input = grad_linear_input.contiguous()
        input = input.contiguous()
        norm_weight = norm_weight.contiguous()
        
        _fused_backward_op(grad_linear_input, input, norm_weight, epsilon, grad_input)
        
        # Compute dW_norm separately for now (or fuse later)
        # d(Loss)/d(W) = sum( d(Loss)/d(Act_Out) * d(Act)/d(Norm_Out) * d(Norm)/d(W) )
        # fused_backward_kernel already recomputes inv_rms and d_y.
        # If we can't get dW from kernel, we do it in Pytorch (slow but functional)
        # Recompute Forward stats
        with torch.no_grad():
             var = input.float().pow(2).mean(dim=-1, keepdim=True)
             inv_rms = torch.rsqrt(var + epsilon)
             x_norm = input * inv_rms
             # y = x_norm * w
             # act = silu(y)
             # d(act) = grad_linear_input
             # d(y) = d(act) * silu'(y)
             
             y = x_norm * norm_weight
             sig = torch.sigmoid(y)
             d_act_dy = sig * (1 + y * (1 - sig))
             d_y = grad_linear_input * d_act_dy
             
             grad_norm_weight = (d_y * x_norm).sum(dim=0)
             
        grad_weight = grad_output.t().mm(input) # Incorrect inputs for dW? No dW = dY.T @ X_quant? 
        # But we don't return dW here, we return grad_input, grad_weight, grad_norm_weight
        
        # For this fused kernel demo, we return None for weight gradients to focus on Input gradient fusion
        return grad_input, None, grad_norm_weight, None, None, None, None, None

@torch.library.custom_op("clover::fused_backward", mutates_args=("grad_input",))
def _fused_backward_op(grad_output: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float, grad_input: torch.Tensor) -> None:
    _fused_ops.fused_backward(grad_output, input, weight, epsilon, grad_input)


class FusedQuartetLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, norm_eps=1e-5, bias=False, device=None, dtype=torch.bfloat16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.norm_eps = norm_eps
        
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        self.norm_weight = torch.nn.Parameter(torch.ones(in_features, device=device, dtype=dtype))
        
        # Hadamard matrix for potential future use (Quartet needs it for backward/rotation?)
        # For this fused kernel demo, we might skip rotation or use identity
        # But `_fp4_mm` is just checking inputs.
        
    def forward(self, x):
        # x: (Batch, Seq, In)
        batch, seq, hidden = x.shape
        x_flat = x.reshape(-1, hidden)
        
        # Call Fused Function
        # We need to manually invoke our fused pipeline
        # For simplicity, inline it or use the class
        
        # 1. Quantize Input (Fused)
        global_amax = torch.zeros((), device=x.device, dtype=torch.float32)
        input_fp4 = fused_quant_fp4(x_flat, self.norm_weight, self.norm_eps, global_amax=global_amax)
        
        # 2. Quantize Weight (Standard)
        # In real training, we might cache this
        weight_amax = self.weight.abs().max().to(torch.float32)
        from quartet2.quant import quant_fp4
        # Note: scale_override?
        weight_fp4 = quant_fp4(self.weight, amax=weight_amax, scale_override=1.0, four_over_six=True)
        
        # 3. GEMM
        alpha = input_fp4.tensor_scale * weight_fp4.tensor_scale
        
        # Need to block the weight scales
        w_scales_blocked = to_blocked(weight_fp4.micro_scales)
        
        res = _fp4_mm(
            input_fp4.fp4, 
            weight_fp4.fp4,
            input_fp4.micro_scales,  # already uint8
            w_scales_blocked,  # already uint8
            alpha
        )
        
        return res.reshape(batch, seq, self.out_features)
