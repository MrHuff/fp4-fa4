#!/usr/bin/env python
"""
Pure FP4 Training Comparison

Uses Quartet-II recipe as base with our fused kernels:
- Forward: fused rmsnorm/mxnorm + activation + FP4 quantization
- Backward: Quartet-II style linear backward + fused activation gradients

NOTE: Uses fused_ops.matmul_nvf4_bf16_tn directly since qutlass wrapper has issues.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import time
from scipy.linalg import hadamard

import fused_ops
from quartet2.quant import quant_fp4, quant_had_eden, dequant_tp_had_eden
from quartet2.linear import to_blocked, _dq_fp4


def get_hadamard_matrix(group_size: int, dtype: torch.dtype, device: torch.device):
    return torch.tensor(
        hadamard(group_size) * group_size**-0.5,
        dtype=dtype,
        device=device,
        requires_grad=False,
    )


def rerotate_hadamard(hadamard_matrix):
    signs = torch.randint(
        0, 2, (hadamard_matrix.size(0),),
        device=hadamard_matrix.device,
        dtype=hadamard_matrix.dtype
    ) * 2 - 1
    return hadamard_matrix * signs[None, :]


def fp4_mm(out, x_fp4, w_fp4, x_scales, w_scales, alpha):
    # print(f"DEBUG: out={out.shape} out_stride={out.stride()}")
    # print(f"DEBUG: x={x_fp4.shape} x_stride={x_fp4.stride()}")
    # print(f"DEBUG: w={w_fp4.shape} w_stride={w_fp4.stride()}")
    xs = to_blocked(x_scales).contiguous().view(torch.uint8)
    ws = to_blocked(w_scales).contiguous().view(torch.uint8)
    # print(f"DEBUG: xs={xs.shape} xs_stride={xs.stride()}")
    # print(f"DEBUG: ws={ws.shape} ws_stride={ws.stride()}")
    
    fused_ops.matmul_nvf4_bf16_tn(
        out, x_fp4, w_fp4,
        xs,
        ws,
        alpha
    )


# =============================================================================
# FP4 Linear with Fused Norm+Activation in Forward
# =============================================================================
class FusedFP4Linear_fn(torch.autograd.Function):
    """
    FP4 Linear layer with fused normalization + activation + quantization.
    """
    
    @staticmethod
    def forward(ctx, x, weight, norm_weight, had, mode='fused_absmax', epsilon=1e-5,
                disable_backward_quant=True):
        ctx.mode = mode
        ctx.epsilon = epsilon
        ctx.disable_backward_quant = disable_backward_quant
        
        # Start Profiling
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        quant_start = torch.cuda.Event(enable_timing=True)
        gemm_start = torch.cuda.Event(enable_timing=True)
        
        if _profiling_enabled: start_event.record()
        
        # Handle 3D input (batch, seq, dim) -> (batch*seq, dim)
        original_shape = x.shape
        if x.dim() == 3:
            ctx.batch = x.shape[0]
            ctx.seq = x.shape[1]
            flat_x = x.reshape(-1, x.shape[-1])
        else:
            ctx.batch = x.shape[0]
            ctx.seq = 1
            flat_x = x
        
        ctx.in_dim = weight.shape[1]
        ctx.out_dim = weight.shape[0]
        
        rows, cols = flat_x.shape
        
        if _profiling_enabled: quant_start.record()
        
        # Allocate outputs for fused quantization
        x_packed = torch.empty((rows, cols // 2), device=x.device, dtype=torch.uint8)
        x_scales = torch.empty((rows, cols // 16), device=x.device, dtype=torch.uint8)
        x_global_scale = torch.empty((), device=x.device, dtype=torch.float32)
        inv_rms_cache = torch.empty((rows,), device=x.device, dtype=torch.float32)
        
        # Fused norm + activation + quantization
        if mode == 'fused_absmax' or mode == 'fused_absmax_rms_bwd':
            fused_ops.fused_mxnorm_fast(
                x_packed, x_scales, x_global_scale, inv_rms_cache,
                flat_x.contiguous(), norm_weight.data.contiguous(),
                epsilon, 1.0, True  # apply_silu=True
            )
        elif mode == 'fused_rms':
            fused_ops.fused_mxnorm(
                x_packed, x_scales, x_global_scale, inv_rms_cache,
                flat_x.contiguous(), norm_weight.data.contiguous(),
                epsilon, 1.0, True, 0  # mode 0 = RMS
            )
        else:  # standard - use separate ops
            rms = torch.sqrt(torch.mean(flat_x ** 2, dim=-1, keepdim=True) + epsilon)
            inv_rms_cache = (1.0 / rms).squeeze(-1)
            x_norm = flat_x * inv_rms_cache.unsqueeze(-1) * norm_weight
            x_act = x_norm * torch.sigmoid(x_norm)  # SiLU
            
            x_fp4_result = quant_fp4(x_act, 1.0)
            x_packed = x_fp4_result.fp4
            x_scales = x_fp4_result.micro_scales.view(torch.uint8)
            x_global_scale = x_fp4_result.tensor_scale
        
        # Quantize weight
        weight_fp4 = quant_fp4(weight, 1.0)
        
        if _profiling_enabled: gemm_start.record()
        
        # FP4 GEMM (use fused_ops directly)
        out = torch.empty((rows, weight.shape[0]), device=x.device, dtype=torch.bfloat16)
        fp4_mm(out, x_packed, weight_fp4.fp4,
               x_scales.view(torch.float8_e4m3fn), weight_fp4.micro_scales,
               (x_global_scale * weight_fp4.tensor_scale).item())
        
        if _profiling_enabled: 
            end_event.record()
            torch.cuda.synchronize()
            _profiling_stats['fwd_quant'].append(quant_start.elapsed_time(gemm_start))
            _profiling_stats['fwd_gemm'].append(gemm_start.elapsed_time(end_event))
            _profiling_stats['fwd_overhead'].append(start_event.elapsed_time(quant_start))
        
        # Save for backward
        ctx.save_for_backward(
            flat_x, norm_weight.data, inv_rms_cache,
            x_packed, x_scales.view(torch.float8_e4m3fn), x_global_scale,
            weight_fp4.fp4, weight_fp4.micro_scales, weight_fp4.tensor_scale,
            had, torch.tensor(disable_backward_quant)
        )
        
        if len(original_shape) == 3:
            return out.reshape(ctx.batch, ctx.seq, ctx.out_dim)
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        (flat_x, norm_weight, inv_rms_cache,
         xfp4, xs, xts,
         wfp4, ws, wts,
         had, disable_backward_quant) = ctx.saved_tensors
        
        # Profiling
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        quant_start = torch.cuda.Event(enable_timing=True)
        gemm_start = torch.cuda.Event(enable_timing=True)
        fused_bwd_start = torch.cuda.Event(enable_timing=True)
        
        if _profiling_enabled: start_event.record()
        
        flat_grad = grad_output.reshape(-1, grad_output.shape[-1])
        
        grad_x_linear = None
        grad_weight = None

        # Step 1: Linear backward
        if disable_backward_quant:
            # Always use dequant + BF16 for now (disable_backward_quant=True)
            xr = _dq_fp4(xfp4, xs, xts.item())
            wr = _dq_fp4(wfp4, ws, wts.item())
            grad_x_linear = flat_grad @ wr
            grad_weight = flat_grad.T @ xr
        else:
            # FP4 Backward
            backward_scale_override = (17 / 16) * 0.93
            had = rerotate_hadamard(had)
            scratch_amax = torch.empty((), dtype=torch.int32, device=grad_output.device)
            
            if _profiling_enabled: quant_start.record()
            
            # dL/dX = dL/dY @ W.T
            # Quantize dL/dY
            e_ht_fp4, e_ht_ms, e_ht_ts = quant_had_eden(x=flat_grad, h=had, scale_override=backward_scale_override, scratch_amax=scratch_amax)
            # Dequant-Requant W (Transposed)? Reference uses dequant_tp_had_eden on wfp4
            wt_ht_fp4, wt_ht_ms, wt_ht_ts = dequant_tp_had_eden(x=wfp4, x_group_scales=ws, x_tensor_scale=wts, h=had, scale_override=backward_scale_override, scratch_amax=scratch_amax)
            
            if _profiling_enabled: gemm_start.record()
            
            grad_x_linear = torch.empty((flat_grad.shape[0], wfp4.shape[1] * 2), device=flat_grad.device, dtype=torch.bfloat16)
            fp4_mm(grad_x_linear, e_ht_fp4, wt_ht_fp4, e_ht_ms, wt_ht_ms, (e_ht_ts * wt_ht_ts).item())
            
            # dL/dW = (dL/dY).T @ X
            # Quantize dL/dY (Transposed)
            et_ht_fp4, et_ht_ms, et_ht_ts = quant_had_eden(x=flat_grad, h=had, scale_override=backward_scale_override, transpose=True, scratch_amax=scratch_amax)
            # Dequant-Requant X
            xt_ht_fp4, xt_ht_ms, xt_ht_ts = dequant_tp_had_eden(x=xfp4, x_group_scales=xs, x_tensor_scale=xts, h=had, scale_override=backward_scale_override, scratch_amax=scratch_amax)
            
            grad_weight = torch.empty((flat_grad.shape[1], xfp4.shape[1] * 2), device=flat_grad.device, dtype=torch.bfloat16)
            fp4_mm(grad_weight, et_ht_fp4, xt_ht_fp4, et_ht_ms, xt_ht_ms, (et_ht_ts * xt_ht_ts).item())
            
        if _profiling_enabled:
            fused_bwd_start.record()
            _profiling_stats['bwd_quant'].append(quant_start.elapsed_time(gemm_start))
            _profiling_stats['bwd_gemm'].append(gemm_start.elapsed_time(fused_bwd_start))
            _profiling_stats['bwd_overhead'].append(start_event.elapsed_time(quant_start))
        
        # Step 2: Fused backward for norm + activation
        grad_x = torch.empty_like(flat_x)
        
        # Mode-specific backward dispatch
        if ctx.mode == 'fused_absmax':
             # Exact AbsMax backward
             fused_ops.fused_backward_absmax(
                 grad_x, grad_x_linear, flat_x, norm_weight, inv_rms_cache
             )
        else:
             # Standard RMS backward (used for fused_rms AND fused_absmax_rms_bwd)
             # For fused_absmax_rms_bwd, this is the "Straight Through" approximation
             fused_ops.fused_backward_v2(
                 grad_x_linear, flat_x, norm_weight, inv_rms_cache,
                 ctx.epsilon, grad_x
             )
             
        if _profiling_enabled:
            end_event.record()
            torch.cuda.synchronize()
            _profiling_stats['bwd_fused'].append(fused_bwd_start.elapsed_time(end_event))
        
        if ctx.seq > 1:
            grad_x = grad_x.reshape(ctx.batch, ctx.seq, ctx.in_dim)
        
        return grad_x, grad_weight, None, None, None, None, None, None


class FusedFP4Linear(nn.Module):
    """FP4 Linear with fused normalization."""
    
    def __init__(self, in_features, out_features, mode='fused_absmax', epsilon=1e-5,
                 dtype=torch.bfloat16, device='cuda'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mode = mode
        self.epsilon = epsilon
        
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features, device=device, dtype=dtype) / (in_features ** 0.5)
        )
        self.norm_weight = nn.Parameter(
            torch.ones(in_features, device=device, dtype=dtype)
        )
        self.register_buffer("had", get_hadamard_matrix(128, dtype, device))
        self.register_buffer("scratch_amax", torch.empty((), dtype=torch.int32, device=device))
    
    def forward(self, x, disable_backward_quant=True):
        return FusedFP4Linear_fn.apply(
            x, self.weight, self.norm_weight, self.had, self.mode,
            self.epsilon, disable_backward_quant
        )


# =============================================================================
# Simple FP4 Linear (for first layer, no fused norm)
# =============================================================================
# =============================================================================
# Simple FP4 Linear (for first layer, no fused norm)
# =============================================================================

# =============================================================================
# Profiling Utils
# =============================================================================
_profiling_stats = {
    'fwd_quant': [], 'fwd_gemm': [], 'fwd_overhead': [],
    'bwd_quant': [], 'bwd_gemm': [], 'bwd_overhead': [],
    'bwd_fused': []
}
_profiling_enabled = False

def enable_profiling():
    global _profiling_enabled
    _profiling_enabled = True

def disable_profiling():
    global _profiling_enabled
    _profiling_enabled = False

def print_profiling_stats(steps):
    if not _profiling_stats['fwd_gemm']:
        return
    print("\n" + "=" * 60)
    print("PROFILING STATS (Avg ms/step)")
    print("=" * 60)
    
    headers = [k for k in _profiling_stats.keys()]
    avgs = [sum(_profiling_stats[k]) / max(1, len(_profiling_stats[k])) for k in headers]
    
    for h, avg in zip(headers, avgs):
        print(f"{h:<20} | {avg:.3f} ms")


class SimpleFP4Linear_fn(torch.autograd.Function):
    """Simple FP4 Linear without fused norm."""
    
    @staticmethod
    def forward(ctx, x, weight, had, disable_backward_quant=False):
        ctx.disable_backward_quant = disable_backward_quant
        
        # Start Profiling
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        quant_start = torch.cuda.Event(enable_timing=True)
        gemm_start = torch.cuda.Event(enable_timing=True)
        
        if _profiling_enabled: start_event.record()
        
        # Handle 3D input (batch, seq, dim) -> (batch*seq, dim)
        original_shape = x.shape
        if x.dim() == 3:
            ctx.batch = x.shape[0]
            ctx.seq = x.shape[1]
            flat_x = x.reshape(-1, x.shape[-1])
        else:
            ctx.batch = x.shape[0]
            ctx.seq = 1
            flat_x = x
        
        if _profiling_enabled: quant_start.record()
        
        # Quantize input and weight
        x_fp4 = quant_fp4(flat_x, 1.0)
        w_fp4 = quant_fp4(weight, 1.0)
        
        if _profiling_enabled: gemm_start.record()
        
        # FP4 GEMM
        out = torch.empty((flat_x.shape[0], weight.shape[0]), device=x.device, dtype=torch.bfloat16)
        fp4_mm(out, x_fp4.fp4, w_fp4.fp4,
               x_fp4.micro_scales, w_fp4.micro_scales,
               (x_fp4.tensor_scale * w_fp4.tensor_scale).item())
        
        if _profiling_enabled: 
            end_event.record()
            torch.cuda.synchronize()
            _profiling_stats['fwd_quant'].append(quant_start.elapsed_time(gemm_start))
            _profiling_stats['fwd_gemm'].append(gemm_start.elapsed_time(end_event))
            _profiling_stats['fwd_overhead'].append(start_event.elapsed_time(quant_start))
        
        # Save for backward
        ctx.save_for_backward(
            x_fp4.fp4, x_fp4.micro_scales, x_fp4.tensor_scale,
            w_fp4.fp4, w_fp4.micro_scales, w_fp4.tensor_scale,
            had
        )
        
        if len(original_shape) == 3:
            return out.reshape(ctx.batch, ctx.seq, -1)
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        (xfp4, xs, xts, wfp4, ws, wts, had) = ctx.saved_tensors
        
        # Profiling
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        quant_start = torch.cuda.Event(enable_timing=True)
        gemm_start = torch.cuda.Event(enable_timing=True)
        
        if _profiling_enabled: start_event.record()
        
        flat_grad = grad_output.reshape(-1, grad_output.shape[-1])
        
        grad_x = None
        grad_weight = None

        if ctx.disable_backward_quant:
            # Dequantize and use BF16 backward
            xr = _dq_fp4(xfp4, xs, xts.item())
            wr = _dq_fp4(wfp4, ws, wts.item())
            grad_x = flat_grad @ wr
            grad_weight = flat_grad.T @ xr
        else:
             # FP4 Backward
            backward_scale_override = (17 / 16) * 0.93
            had = rerotate_hadamard(had)
            scratch_amax = torch.empty((), dtype=torch.int32, device=grad_output.device)
            
            if _profiling_enabled: quant_start.record()
            
            # dL/dX
            e_ht_fp4, e_ht_ms, e_ht_ts = quant_had_eden(x=flat_grad, h=had, scale_override=backward_scale_override, scratch_amax=scratch_amax)
            wt_ht_fp4, wt_ht_ms, wt_ht_ts = dequant_tp_had_eden(x=wfp4, x_group_scales=ws, x_tensor_scale=wts, h=had, scale_override=backward_scale_override, scratch_amax=scratch_amax)
            
            if _profiling_enabled: gemm_start.record()
            
            grad_x = torch.empty((flat_grad.shape[0], wfp4.shape[1] * 2), device=flat_grad.device, dtype=torch.bfloat16)
            fp4_mm(grad_x, e_ht_fp4, wt_ht_fp4, e_ht_ms, wt_ht_ms, (e_ht_ts * wt_ht_ts).item())
            
            # dL/dW
            et_ht_fp4, et_ht_ms, et_ht_ts = quant_had_eden(x=flat_grad, h=had, scale_override=backward_scale_override, transpose=True, scratch_amax=scratch_amax)
            xt_ht_fp4, xt_ht_ms, xt_ht_ts = dequant_tp_had_eden(x=xfp4, x_group_scales=xs, x_tensor_scale=xts, h=had, scale_override=backward_scale_override, scratch_amax=scratch_amax)
            
            grad_weight = torch.empty((flat_grad.shape[1], xfp4.shape[1] * 2), device=flat_grad.device, dtype=torch.bfloat16)
            fp4_mm(grad_weight, et_ht_fp4, xt_ht_fp4, et_ht_ms, xt_ht_ms, (et_ht_ts * xt_ht_ts).item())
            
        if _profiling_enabled:
            end_event.record()
            torch.cuda.synchronize()
            _profiling_stats['bwd_quant'].append(quant_start.elapsed_time(gemm_start))
            _profiling_stats['bwd_gemm'].append(gemm_start.elapsed_time(end_event))
            _profiling_stats['bwd_overhead'].append(start_event.elapsed_time(quant_start))

        if ctx.seq > 1:
            grad_x = grad_x.reshape(ctx.batch, ctx.seq, -1)
            
        return grad_x, grad_weight, None, None, None


class SimpleFP4Linear(nn.Module):
    """Simple FP4 Linear for first layer."""
    
    def __init__(self, in_features, out_features, dtype=torch.bfloat16, device='cuda'):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features, device=device, dtype=dtype) / (in_features ** 0.5)
        )
        self.register_buffer("had", get_hadamard_matrix(128, dtype, device))
        self.register_buffer("scratch_amax", torch.empty((), dtype=torch.int32, device=device))
    
    def forward(self, x, disable_backward_quant=False):
        return SimpleFP4Linear_fn.apply(x, self.weight, self.had, disable_backward_quant)



# =============================================================================
# Compiled RMSNorm + Simple FP4 Linear
# =============================================================================

@torch.compile
def compiled_rms_silu(x, norm_weight, epsilon):
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + epsilon)
    x_norm = x * (1.0 / rms) * norm_weight
    return x_norm * torch.sigmoid(x_norm) # SiLU

class CompiledRMSLinear(nn.Module):
    """RMSNorm + SiLU (compiled) -> SimpleFP4Linear."""
    
    def __init__(self, in_features, out_features, dtype=torch.bfloat16, device='cuda'):
        super().__init__()
        self.epsilon = 1e-5
        self.norm_weight = nn.Parameter(torch.ones(in_features, device=device, dtype=dtype))
        self.linear = SimpleFP4Linear(in_features, out_features, dtype, device)
        
    def forward(self, x):
        h = compiled_rms_silu(x, self.norm_weight, self.epsilon)
        return self.linear(h, disable_backward_quant=False)


# =============================================================================
# Compiled RMSNorm + Pure BF16 Linear
# =============================================================================

class CompiledBF16Linear(nn.Module):
    """RMSNorm + SiLU (compiled) -> Standard BF16 Linear."""
    
    def __init__(self, in_features, out_features, dtype=torch.bfloat16, device='cuda'):
        super().__init__()
        self.epsilon = 1e-5
        self.norm_weight = nn.Parameter(torch.ones(in_features, device=device, dtype=dtype))
        self.linear = nn.Linear(in_features, out_features, bias=False, device=device, dtype=dtype)
        # Initialize weight similarly to others
        torch.nn.init.normal_(self.linear.weight, std=in_features**-0.5)
        
    def forward(self, x):
        h = compiled_rms_silu(x, self.norm_weight, self.epsilon)
        return self.linear(h)


# =============================================================================
# Experiment Runner
# =============================================================================
class FP4TrainingComparison:
    def __init__(self, input_dim=1024, hidden_dim=4096, batch_size=256, num_layers=4):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
    
    def run_method(self, method, steps=200):
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        
        # Reset stats
        for k in _profiling_stats: _profiling_stats[k] = []
        enable_profiling()
        
        # Parse method
        if method == 'fused_absmax':
            mode = 'fused_absmax'
        elif method == 'fused_rms':
            mode = 'fused_rms'
        elif method == 'fused_absmax_rms_bwd':
            mode = 'fused_absmax_rms_bwd'
        elif method == 'fused_absmax_fp4_bwd':
            mode = 'fused_absmax' # Reuse absmax mode for forward, just change bwd arg
        elif method == 'standard':
            mode = 'standard'
        elif method == 'compiled_rms':
            mode = 'compiled_rms'
        elif method == 'compiled_bf16':
            mode = 'compiled_bf16'
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Build model: l1 -> l2 -> ... -> ln
        layers = []
        
        # L1
        if method == 'compiled_bf16':
             l1 = nn.Linear(self.input_dim, self.hidden_dim, bias=False, device='cuda', dtype=torch.bfloat16)
        else:
             l1 = SimpleFP4Linear(self.input_dim, self.hidden_dim) # Defaults to FP4 bwd
        layers.append(l1)
        
        # L2...Ln
        for _ in range(self.num_layers - 1):
            if method == 'compiled_rms':
                l = CompiledRMSLinear(self.hidden_dim, self.hidden_dim) # Assume hidden->hidden for robust scaling
            elif method == 'compiled_bf16':
                l = CompiledBF16Linear(self.hidden_dim, self.hidden_dim)
            else:
                l = FusedFP4Linear(self.hidden_dim, self.hidden_dim, mode=mode)
                original_forward = l.forward
                l.forward = lambda x: original_forward(x, disable_backward_quant=False)
            layers.append(l)

        # Make output dim match input for simple chaining or keep hidden? 
        # Actually for "hidden -> hidden" layers it works. 
        # But if input != hidden, the first layer handles transition.
        # Wait, if input_dim != hidden_dim, subsequent layers must be hidden->hidden.
        # My previous L2 was hidden->input (project back). 
        # To scale "layers", usually we do hidden->hidden.
        # Let's assume standard Transformer block: In->Hidden (Act) -> Output if MLP.
        # But here we just stack Linear layers. 
        # Let's adjust L2..Ln to be Hidden->Hidden.
        # And ensure final layer output size is handled for Loss calculation.
        # Target is (Batch, Input). If final output is Hidden, we need target to matching Hidden.
        
        model = nn.Sequential(*layers)
        
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        # Data
        inputs = torch.randn(self.batch_size, self.input_dim, device='cuda', dtype=torch.bfloat16)
        # Target shape depends on last layer output
        last_dim = self.hidden_dim
        targets = torch.randn(self.batch_size, last_dim, device='cuda', dtype=torch.bfloat16)
        
        history = []
        
        # Warmup
        torch.cuda.synchronize()
        for _ in range(5):
            optimizer.zero_grad()
            out = model(inputs)
            loss = nn.MSELoss()(out, targets)
            loss.backward()
            optimizer.step()
        
        torch.cuda.synchronize()
        start = time.time()
        
        for i in range(steps):
            optimizer.zero_grad()
            out = model(inputs)
            loss = nn.MSELoss()(out, targets)
            loss.backward()
            optimizer.step()
            
            if i % 50 == 0:
                print(f"  Step {i}: Loss={loss.item():.6f}")
            history.append(loss.item())
            
            if i == 50: disable_profiling() # Stop profiling after 50 steps
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        if method == 'fused_absmax': # Only print for one relevant method to avoid noise
            print_profiling_stats(steps)
        
        return {
            'final_loss': history[-1],
            'history': history,
            'time': elapsed,
            'steps_per_sec': steps / elapsed
        }
    
    def run_comparison(self, steps=1000):
        # All methods will now use FP4 Backward
        methods = ['compiled_bf16', 'fused_absmax', 'compiled_rms', 'fused_rms', 'fused_absmax_rms_bwd'] 
        results = {}
        
        for method in methods:
            print(f"\n--- {method.upper()} ---")
            try:
                results[method] = self.run_method(method, steps)
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                results[method] = {'error': str(e)}
        
        return results


if __name__ == '__main__':
    print("=" * 60)
    print("Pure FP4 Training Comparison")
    print("=" * 60)
    
    # Iterate sizes
    configs = [
        # (Input, Hidden, Batch)
        (1024, 4096, 256),   # Standard test
        (4096, 14336, 128),  # Llama 3 8B SwiGLU Size (approx) / 2 for BF16 fits
        (2048, 8192, 256),   # Intermediate
    ]
    
    for in_dim, hid_dim, bs in configs:
        print(f"\n>>>> CONFIG: In={in_dim}, Hidden={hid_dim}, Batch={bs} <<<<")
        runner = FP4TrainingComparison(input_dim=in_dim, hidden_dim=hid_dim, batch_size=bs, num_layers=4)
        results = runner.run_comparison(steps=200) # Keep steps moderate for multiple runs
        
        print("\n" + "=" * 60)
        print(f"RESULTS SUMMARY (In={in_dim}, Hid={hid_dim})")
        print("=" * 60)
        print(f"{'Method':<20} | {'Final Loss':<12} | {'Time (s)':<10} | {'Steps/s':<10}")
        print("-" * 60)
        
        for method, res in results.items():
            if 'error' in res:
                print(f"{method:<20} | ERROR: {res['error'][:40]}")
            else:
                print(f"{method:<20} | {res['final_loss']:<12.6f} | {res['time']:<10.2f} | {res['steps_per_sec']:<10.1f}")

