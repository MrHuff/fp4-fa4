
import torch
import torch.nn as nn
from torch.autograd import Function

class CustomNormFunc(Function):
    @staticmethod
    def forward(ctx, x, weight, epsilon, forward_mode, backward_mode):
        """
        forward_mode: 'rms' or 'absmax'
        backward_mode: 'exact', 'stopgrad', 'proxy_rms'
        """
        ctx.save_for_backward(x, weight)
        ctx.epsilon = epsilon
        ctx.forward_mode = forward_mode
        ctx.backward_mode = backward_mode

        if forward_mode == 'rms':
            # Standard RMSNorm
            rms = torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + epsilon)
            x_norm = x / rms
            ctx.rms = rms # Save for backward if needed
        elif forward_mode == 'absmax':
            # AbsMax Norm
            absmax = torch.max(torch.abs(x), dim=-1, keepdim=True)[0] + epsilon
            x_norm = x / absmax
            ctx.absmax = absmax
        else:
            raise ValueError(f"Unknown forward_mode: {forward_mode}")

        output = x_norm * weight
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        epsilon = ctx.epsilon
        forward_mode = ctx.forward_mode
        backward_mode = ctx.backward_mode

        # Calculate common terms
        N = x.shape[-1]
        
        # Determine scaling factor s used in forward pass
        if forward_mode == 'rms':
            s = ctx.rms
        else:
            s = ctx.absmax

        # Normalized input (pre-weight)
        x_norm = x / s
        
        # Gradient w.r.t input (dx) logic
        
        # 1. StopGrad Approximation (Applies to any forward mode)
        # dx = dy * g / s
        # This assumes s is constant.
        if backward_mode == 'stopgrad':
            grad_x = grad_output * weight / s
            
        # 2. Proxy RMS (Applies RMS-style projection to whatever s we used)
        # dx = (g/s) * (dy - y * sum(dy*y)/sum(y^2))
        elif backward_mode == 'proxy_rms' or (forward_mode == 'rms' and backward_mode == 'exact'):
            # This is the standard RMSNorm gradient structure
            # If forward_mode='absmax', we use 's' = absmax but 'project' out the gain direction like RMS.
            
            w_grad_output = grad_output * weight # dy * g
            
            # Dot product term: sum(dy * y)
            # Here y = x_norm (without weight, to simplify derivation matching standard forms)
            # Actually standard form x_hat = x/s. output = x_hat * w. 
            # dy_output = grad_output.
            # dx_hat = grad_output * w.
            
            dx_hat = grad_output * weight
            
            # term 1: mean(dx_hat * x_hat)
            term1 = torch.mean(dx_hat * x_norm, dim=-1, keepdim=True)
            
            # term 2: x_hat
            grad_x = (dx_hat - x_norm * term1) / s
            
            # Note: The standard RMS backward usually assumes mean(x^2)=1 for the projection term simplifiction.
            # Let's use the explicit projection: 
            # dx = 1/s * (I - x x^T / ||x||^2) * dy
            # For RMS: ||x||^2 = N * s^2. 
            # For AbsMax: This projection is 'ProxyRMS'.
            
        # 3. Exact AbsMax (Sparse update)
        elif forward_mode == 'absmax' and backward_mode == 'exact':
            # dx_i = (dy_i * g_i)/s - (sum_j dy_j y_j)/s * I(i=k_max) * sign(x_i)
            # where y_j = x_j/s * g_j
            
            dx_scaled = (grad_output * weight) / s
            
            # The correction term relies on sum(grad_output * output)
            # output = x/s * w
            # sum_j (grad_output_j * output_j) = sum(dy * y)
            dot_prod = torch.sum(grad_output * (x_norm * weight), dim=-1, keepdim=True)
            
            # Find max index
            # We need to find k_max s.t. |x_k| = s (approx)
            # Or just use torch.max
            max_vals, max_indices = torch.max(torch.abs(x), dim=-1, keepdim=True)
            
            # Create a one-hot mask for the max element
            # Note: Handing ties correctly? torch.max takes first.
            mask = torch.zeros_like(x)
            mask.scatter_(-1, max_indices, 1.0)
            
            # Sign of the max element
            # Actually derivative of |x| is sign(x). 
            # s = |x_k|. ds/dx_k = sign(x_k).
            sign_x = torch.sign(x)
            
            # Correction term: (1/s) * dot_prod * mask * sign
            correction = (1.0 / s) * dot_prod * mask * sign_x
            
            grad_x = dx_scaled - correction

        else:
             raise ValueError(f"Unknown combination: {forward_mode}, {backward_mode}")

        # Gradient w.r.t weight
        grad_weight = torch.sum(grad_output * x_norm, dim=0) # Sum over batch if 2D, or broadcast handling
        # Assuming simple [B, L, D] or [B, D] input, weight is [D]. Sum over all batch dims.
        grad_weight = torch.sum(grad_output * x_norm, dim=tuple(range(x.ndim - 1)))
        
        return grad_x, grad_weight, None, None, None

class CustomNorm(nn.Module):
    def __init__(self, dim, forward_mode='rms', backward_mode='exact', epsilon=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.epsilon = epsilon
        self.forward_mode = forward_mode
        self.backward_mode = backward_mode

    def forward(self, x):
        return CustomNormFunc.apply(x, self.weight, self.epsilon, self.forward_mode, self.backward_mode)
