
import torch
from fused_ops.fused_linear import _fused_backward_op

def verify_backward():
    print("Verifying Fused Backward Correctness...")
    device = "cuda"
    rows, cols = 32, 1024
    
    # Inputs
    x = torch.randn(rows, cols, device=device, dtype=torch.bfloat16, requires_grad=True)
    w = torch.rand(cols, device=device, dtype=torch.bfloat16) + 0.5
    epsilon = 1e-5
    
    # Reference
    # y = RMSNorm(x)
    x_float = x.float()
    w_float = w.float()
    var = x_float.pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(var + epsilon)
    y = x_float * inv_rms * w_float
    
    # act = SiLU(y)
    act = torch.nn.functional.silu(y)
    
    # loss = sum(act * grad_output)
    # Generate random grad_output (dY)
    grad_output = torch.randn_like(x)
    
    loss = (act * grad_output.float()).sum()
    loss.backward()
    
    ref_grad = x.grad.clone()
    
    # Fused
    fused_grad = torch.empty_like(x)
    # Note: _fused_backward_op takes (grad_output, input, weight, epsilon, grad_input)
    # Inputs must be contiguous.
    _fused_backward_op(grad_output, x.detach(), w.detach(), epsilon, fused_grad)
    
    # Compare
    ref_g = ref_grad.float()
    fused_g = fused_grad.float()
    
    # Expected relative error? BF16 precision.
    diff = (ref_g - fused_g).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    rel_error = diff / (ref_g.abs() + 1e-6)
    max_rel = rel_error.max().item()
    
    print(f"Max Diff: {max_diff}")
    print(f"Mean Diff: {mean_diff}")
    print(f"Max Rel Error: {max_rel}")
    
    if mean_diff < 1e-3: # Loose tolerance for BF16 aggregation
        print("Backward Pass Verified!")
    else:
        print("Backward Pass FAILED verification.")

if __name__ == "__main__":
    verify_backward()
