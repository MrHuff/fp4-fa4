
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from custom_norm import CustomNorm

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def run_experiment(forward_mode, backward_mode, steps=500, lr=0.01):
    input_dim = 1024
    hidden_dim = 4096
    batch_size = 32
    
    # Simple FFN: Linear -> Norm -> SiLU -> Linear
    class SimpleFFN(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(input_dim, hidden_dim, bias=False)
            self.norm = CustomNorm(hidden_dim, forward_mode=forward_mode, backward_mode=backward_mode)
            self.act = nn.SiLU()
            self.l2 = nn.Linear(hidden_dim, input_dim, bias=False)
            
        def forward(self, x):
            x = self.l1(x)
            x_pre_norm = x.clone() # Keep for logging if needed
            x = self.norm(x)
            x = self.act(x)
            x = self.l2(x)
            return x, x_pre_norm

    model = SimpleFFN().cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Dummy data
    inputs = torch.randn(batch_size, input_dim).cuda()
    targets = torch.randn(batch_size, input_dim).cuda()
    
    loss_history = []
    grad_norm_history = []
    grad_sparsity_history = [] # Measure of peakiness (L_inf / L2)
    
    # Hook to capture gradients explicitly at the norm input
    input_grads = []
    
    def save_grad_hook(grad):
        input_grads.append(grad.detach().clone())
        
    # Register hook on the output of l1 (input to norm)
    # We can't access intermediate tensor easily without rewriting forward or using hooks on module
    # Register hook on the tensor inside forward? Or hook on L1 output?
    # Backward hook on Linear is easiest.
    
    # Actually, simpler: we want grad of Loss w.r.t input of Norm.
    # In backward(), CustomNormFunc receives grad_output and returns grad_input.
    # We can capture it in the CustomNormFunc if we modify it, or we can use register_full_backward_hook on the layer.
    
    # Let's use register_full_backward_hook on model.norm
    def hook_fn(module, grad_input, grad_output):
        # grad_input is a tuple (grad_x, grad_weight, ...) based on forward args
        # The first element is grad_x
        gx = grad_input[0]
        if gx is not None:
             input_grads.append(gx.detach())

    model.norm.register_full_backward_hook(hook_fn)

    print(f"--- Running Experiment: Fwd={forward_mode}, Bwd={backward_mode} ---")
    
    for i in range(steps):
        optimizer.zero_grad()
        output, _ = model(inputs)
        loss = nn.MSELoss()(output, targets)
        
        if torch.isnan(loss):
            print(f"Step {i}: Loss went NaN!")
            break
            
        loss.backward()
        
        # Log metrics from the hook
        if len(input_grads) > 0:
            g = input_grads[-1] # [B, H]
            g_flat = g.view(-1)
            
            # L2 Norm
            l2 = torch.norm(g_flat, p=2).item()
            
            # L_inf Norm
            linf = torch.norm(g_flat, p=float('inf')).item()
            
            # Peakiness Ratio (L_inf / L2_adj). 
            # For uniform vec, L_inf / (L2/sqrt(N)) approx 1?
            # Let's just track Linf for now.
            
            grad_norm_history.append(l2)
            grad_sparsity_history.append(linf)
            
            input_grads.clear() # Clear for next step
            
        loss_history.append(loss.item())
        optimizer.step()
        
        if i % 100 == 0:
            print(f"Step {i}: Loss = {loss.item():.6f}")

    return {
        'loss': loss_history,
        'grad_l2': grad_norm_history,
        'grad_linf': grad_sparsity_history
    }

if __name__ == "__main__":
    configs = [
        ('rms', 'exact'),       # Baseline
        ('absmax', 'exact'),    # Theoretically correct but sparse
        ('absmax', 'stopgrad'), # Fast approx
        ('absmax', 'proxy_rms') # Smooth approx
    ]
    
    results = {}
    
    for fwd, bwd in configs:
        key = f"{fwd}_{bwd}"
        results[key] = run_experiment(fwd, bwd)
        
    # Print Summary
    print("\n\n=== Final Summary (Last 10 steps avg) ===")
    print(f"{'Config':<20} | {'Loss':<10} | {'Grad L2':<10} | {'Grad Linf':<10}")
    print("-" * 60)
    for key, res in results.items():
        if len(res['loss']) > 0:
            final_loss = np.mean(res['loss'][-10:])
            final_l2 = np.mean(res['grad_l2'][-10:])
            final_linf = np.mean(res['grad_linf'][-10:])
            print(f"{key:<20} | {final_loss:.6f}   | {final_l2:.4f}     | {final_linf:.4f}")
        else:
            print(f"{key:<20} | FAILED")

