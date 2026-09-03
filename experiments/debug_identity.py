
import torch
import torch.nn as nn
import torch.optim as optim

class MockNormFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        # Returns: uint8, uint8, float, bf16
        ctx.save_for_backward(x, weight)
        rows, cols = x.shape
        y_packed = torch.zeros((rows, cols//2), device=x.device, dtype=torch.uint8)
        scales = torch.zeros((rows, cols//16), device=x.device, dtype=torch.uint8)
        global_scale = torch.tensor(1.0, device=x.device, dtype=torch.float32)
        transport = x.clone()
        return y_packed, scales, global_scale, transport

    @staticmethod
    def backward(ctx, grad_y, grad_scales, grad_global, grad_transport):
        # grad_y: None
        # grad_scales: None
        # grad_global: None? (Since global_scale created in forward without dependency?)
        # Wait, in real code global_scale depends on x implicitly via kernel?
        # But here global_scale is new tensor.
        # But grad is still passed?
        
        x, weight = ctx.saved_tensors
        if grad_transport is None:
             grad_x = None
        else:
             grad_x = grad_transport
        
        return grad_x, None

class MockNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, device='cuda', dtype=torch.bfloat16))
    def forward(self, x):
        return MockNormFunc.apply(x, self.weight)

class MockLinearFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_packed, scales, global_scale, transport, weight):
        ctx.save_for_backward(transport, weight)
        return transport.mm(weight.t())

    @staticmethod
    def backward(ctx, grad_output):
        transport, weight = ctx.saved_tensors
        grad_transport = grad_output.mm(weight)
        grad_weight = grad_output.t().mm(transport)
        grad_global = torch.zeros((), device=grad_output.device, dtype=torch.float32)
        
        return None, None, grad_global, grad_transport, grad_weight

class MockLinear(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim, dim, device='cuda', dtype=torch.bfloat16))
    def forward(self, x_packed, scales, global_scale, transport):
        return MockLinearFunc.apply(x_packed, scales, global_scale, transport, self.weight)

def run():
    print("Running Mock Full Signature...")
    norm = MockNorm(128).cuda()
    lin = MockLinear(128).cuda()
    
    x = torch.randn(32, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    tgt = torch.randn(32, 128, device='cuda', dtype=torch.bfloat16)
    
    # Forward
    h_packed, scales, global_scale, transport = norm(x)
    # Check requires_grad
    print(f"Global Scale Grad: {global_scale.requires_grad}") # Should be True because it came from Function?
    # Actually, if Function inputs have grad, outputs have grad_fn.
    # So `global_scale` has grad_fn.
    
    z = lin(h_packed, scales, global_scale, transport)
    loss = ((z - tgt)**2).mean()
    loss.backward()
    print("Success!")

if __name__ == "__main__":
    run()
