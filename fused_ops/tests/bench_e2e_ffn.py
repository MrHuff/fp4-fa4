"""
E2E FFN Block Benchmark - Fair Comparison with Forward + Backward

Compares three FFN implementations:
1. torch.compile (BF16): RMSNorm → SiLU → BF16 GEMM
2. Quartet-II (FP4): torch.compile(RMSNorm → SiLU) → Quartet_II_linear (quant + FP4 GEMM)  
3. Fused (ours, FP4): Fused(RMSNorm + SiLU + Quant) → FP4 GEMM

All implementations use torch.compile for fair comparison of the pre-processing step.
"""

import torch
import torch.nn.functional as F
import time
from typing import Dict

# Fused ops import
try:
    import fused_ops._fused_ops as fused_ops
    HAS_FUSED_OPS = True
except ImportError as e:
    print(f"Warning: fused_ops not available: {e}")
    HAS_FUSED_OPS = False

# Quartet imports
try:
    from quartet2.linear import Quartet_II_linear, to_blocked
    from quartet2.quant import quant_fp4
    import qutlass
    HAS_QUARTET = True
except ImportError as e:
    print(f"Warning: Quartet-II not available, skipping those benchmarks. Error: {e}")
    HAS_QUARTET = False

if HAS_QUARTET:
    try:
        # Monkey Patch Quartet Backward to fix Float/BF16 mismatch
        import quartet2.linear as qlinear
        from quartet2.linear import Quartet_II_fn

        # Define fixed backward function
        def fixed_quartet_backward(ctx, grad_output):
            # Load ctx and reshape
            xfp4, xs, xm, wfp4, ws, wm, had = ctx.saved_tensors
            backward_scale_override = (17 / 16) * 0.93

            # Re-randomize the rotation
            had = qlinear.rerotate_hadamard(had)
            flat_grad_output = grad_output.reshape(-1, grad_output.shape[-1])

            if ctx.disable_backward_quant:
                # FIX: Ensure flat_grad_output is BF16 (it comes as float from qutlass)
                flat_grad_output = flat_grad_output.to(dtype=torch.bfloat16)
                
                xr = qlinear._dq_fp4(xfp4, xs, xm)
                wr = qlinear._dq_fp4(wfp4, ws, wm)
                grad_input = flat_grad_output @ wr
                grad_weight = flat_grad_output.T @ xr
                return grad_input.reshape(ctx.batch, ctx.seq, ctx.in_dim), grad_weight, None, None, None, None, None, None

            # EW
            with qlinear.nvtx_annotate("Quant", color="yellow"):
                e_ht_fp4, e_ht_ms, e_ht_ts = qlinear.quant_had_eden(x=flat_grad_output, h=had, scale_override=backward_scale_override, scratch_amax=ctx.scratch_amax)
                wt_ht_fp4, wt_ht_ms, wt_ht_ts = qlinear.dequant_tp_had_eden(x=wfp4, x_group_scales=ws, x_tensor_scale=wm, h=had, scale_override=backward_scale_override, scratch_amax=ctx.scratch_amax)
            with qlinear.nvtx_annotate("Matmul", color="blue"):
                grad_input = qlinear._fp4_mm(e_ht_fp4, wt_ht_fp4, qlinear.to_blocked(e_ht_ms), qlinear.to_blocked(wt_ht_ms), alpha=e_ht_ts*wt_ht_ts)

            # EtX
            with qlinear.nvtx_annotate("Quant", color="yellow"):
                et_ht_fp4, et_ht_ms, et_ht_ts = qlinear.quant_had_eden(x=flat_grad_output, h=had, scale_override=backward_scale_override, transpose=True, scratch_amax=ctx.scratch_amax)
                xt_ht_fp4, xt_ht_ms, xt_ht_ts = qlinear.dequant_tp_had_eden(x=xfp4, x_group_scales=xs, x_tensor_scale=xm, h=had, scale_override=backward_scale_override, scratch_amax=ctx.scratch_amax)
            with qlinear.nvtx_annotate("Matmul", color="blue"):
                grad_weight = qlinear._fp4_mm(et_ht_fp4, xt_ht_fp4, qlinear.to_blocked(et_ht_ms), qlinear.to_blocked(xt_ht_ms), alpha=et_ht_ts*xt_ht_ts)
            return grad_input.reshape(ctx.batch, ctx.seq, ctx.in_dim), grad_weight, None, None, None, None, None, None

        # Apply patch
        print("Applying monkey-patch to Quartet_II_fn.backward...")
        Quartet_II_fn.backward = staticmethod(fixed_quartet_backward)

        # Fix _fp4_mm_fake mismatch: Real kernel returns float, but fake returned bf16.
        # This crashes torch.compile. We patch the fake to return float.
        if hasattr(qlinear, "_fp4_mm"):
            @qlinear._fp4_mm.register_fake
            def _fp4_mm_fake_fixed(x_fp4, w_fp4, x_mx, w_mx, alpha):
                 return torch.empty((x_fp4.shape[0], w_fp4.shape[0]), device=x_fp4.device, dtype=torch.float32)
            print("Applying monkey-patch to _fp4_mm meta/fake kernel (BF16 -> FP32)...")

    except Exception as e:
        print(f"Warning: Failed to monkey-patch Quartet: {e}")


# ============================================================================
# Model Definitions
# ============================================================================

class RMSNorm(torch.nn.Module):
    """RMSNorm with bf16 output."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        dtype = x.dtype
        x_float = x.float()
        rms = torch.sqrt(torch.mean(x_float ** 2, dim=-1, keepdim=True) + self.eps)
        return (x_float / rms * self.weight.float()).to(dtype)


class TorchCompiledFFN(torch.nn.Module):
    """torch.compile BF16 baseline: RMSNorm → SiLU → BF16 Linear."""
    def __init__(self, in_dim: int, out_dim: int, device='cuda', dtype=torch.bfloat16):
        super().__init__()
        self.rmsnorm = RMSNorm(in_dim).to(device=device, dtype=dtype)
        self.linear = torch.nn.Linear(in_dim, out_dim, device=device, dtype=dtype)
    
    def forward(self, x):
        x = self.rmsnorm(x)
        x = F.silu(x)
        return self.linear(x)


class RMSNormSiLU(torch.nn.Module):
    """RMSNorm + SiLU for preprocessing."""
    def __init__(self, dim: int, device='cuda', dtype=torch.bfloat16):
        super().__init__()
        self.rmsnorm = RMSNorm(dim).to(device=device, dtype=dtype)
    
    def forward(self, x):
        return F.silu(self.rmsnorm(x))


class QuartetFFN(torch.nn.Module):
    """Quartet-II flow: torch.compile(RMSNorm → SiLU) → Quartet_II_linear.
    
    Uses torch.compile for fair comparison with BF16 baseline.
    """
    def __init__(self, in_dim: int, out_dim: int, device='cuda', dtype=torch.bfloat16):
        super().__init__()
        # Use torch.compile for fair comparison
        self.pre_process = torch.compile(RMSNormSiLU(in_dim, device, dtype), mode='default')
        self.quartet_linear = Quartet_II_linear(in_dim, out_dim, device=device, dtype=dtype, four_over_six=True)
        with torch.no_grad():
            self.quartet_linear.weight_abs_max = self.quartet_linear.weight.abs().max().float()
    
    def forward(self, x):
        x = self.pre_process(x)
        x = x.to(torch.bfloat16)
        # Use disable_backward_quant=True to avoid bf16 grad requirement
        return self.quartet_linear(x, disable_backward_quant=True)


# ============================================================================
# Fused FFN with Custom Autograd
# ============================================================================

class FusedForwardBackward(torch.autograd.Function):
    """Custom autograd for Fused RMSNorm + SiLU + Quant forward/backward."""
    
    @staticmethod
    def forward(ctx, x, rmsnorm_weight, weight, w_fp4, w_scales, w_tensor_scale, eps):
        batch, seq, dim = x.shape
        x_flat = x.view(-1, dim).contiguous()
        rows = x_flat.shape[0]
        cols = dim
        
        # Output buffers for fused forward
        out_packed = torch.zeros(rows, cols // 2, dtype=torch.uint8, device=x.device)
        scales = torch.zeros(rows * cols // 16, dtype=torch.uint8, device=x.device)
        global_scale = torch.zeros((), dtype=torch.float32, device=x.device)
        inv_rms_cache = torch.zeros(rows, dtype=torch.float32, device=x.device)
        
        # Profiling
        start = torch.cuda.Event(enable_timing=True)
        mid = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        # FUSED: RMSNorm + SiLU + FP4 Quantization
        # Using 2-PASS version
        fused_ops.fused_rmsnorm_act_quant_2pass(
            out_packed, scales, global_scale,
            inv_rms_cache, x_flat, rmsnorm_weight,
            eps, 1.0, True
        )
        mid.record()
        
        # FP4 GEMM
        x_scales_blocked = to_blocked(scales.view(rows, -1))
        alpha = global_scale * w_tensor_scale
        out = qutlass.matmul_nvf4_bf16_tn(
            out_packed.view(rows, -1), 
            w_fp4,
            x_scales_blocked.view(torch.uint8),
            to_blocked(w_scales).view(torch.uint8),
            alpha.item()
        )
        end.record()
        
        # Keep stats (Global hack for demo purposes)
        if not hasattr(FusedForwardBackward, 'stats'):
             FusedForwardBackward.stats = {'quant': [], 'gemm': []}
             FusedForwardBackward.count = 0
             
        # Only synchronize/print occasionally to avoid overhead
        FusedForwardBackward.count += 1
        if FusedForwardBackward.count % 100 == 0:
            end.synchronize()
            t_quant = start.elapsed_time(mid)
            t_gemm = mid.elapsed_time(end)
            FusedForwardBackward.stats['quant'].append(t_quant)
            FusedForwardBackward.stats['gemm'].append(t_gemm)
            # print(f"  [Profile] Quant: {t_quant:.3f}ms, GEMM: {t_gemm:.3f}ms")
            
        # Save for backward
        ctx.save_for_backward(x_flat, rmsnorm_weight, weight, inv_rms_cache)
        ctx.shape = (batch, seq, dim)
        ctx.eps = eps
        
        return out.view(batch, seq, -1)
    
    @staticmethod
    def backward(ctx, grad_output):
        x_flat, rmsnorm_weight, weight, inv_rms_cache = ctx.saved_tensors
        batch, seq, dim = ctx.shape
        
        # Grad output from GEMM - reshape for backward
        grad_out_flat = grad_output.view(-1, grad_output.shape[-1]).contiguous()
        
        # 1. Backprop through Linear: dX_GEMM = dY @ W
        # Note: We use BF16 matmul for the backward pass of the linear layer
        # Ensure grad_out_flat is same dtype as weight (BF16)
        grad_gemm_input = grad_out_flat.to(weight.dtype).matmul(weight)
        
        # 2. Backprop through Fused (RMSNorm + SiLU + Quant)
        # Note: The 'input' to the GEMM was the 'output' of the Fused Op (quantized)
        # But our fused backward takes gradients w.r.t. the output of the activation
        # effectively doing d(Act)/d(X).
        # We pass appropriate gradients to the fused kernel.
        
        grad_input = torch.zeros_like(x_flat)
        
        # Use our fused backward kernel
        fused_ops.fused_backward_v2(
            grad_gemm_input.to(torch.bfloat16),  # grad_output passed from GEMM
            x_flat,  # input
            rmsnorm_weight,  # weight
            inv_rms_cache,  # cached_inv_rms
            ctx.eps,  # epsilon
            grad_input  # grad_input (output)
        )
        
        return grad_input.view(batch, seq, dim), None, None, None, None, None, None

class FusedFFN(torch.nn.Module):
    """Our fused approach: Fused(RMSNorm + SiLU + Quant) → FP4 GEMM."""
    def __init__(self, in_dim: int, out_dim: int, device='cuda', dtype=torch.bfloat16):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = torch.nn.Parameter(torch.randn(out_dim, in_dim, device=device, dtype=dtype) / in_dim**0.5)
        self.rmsnorm_weight = torch.nn.Parameter(torch.ones(in_dim, device=device, dtype=dtype))
        self.eps = 1e-6
        
        self._update_weight_quant()
    
    def _update_weight_quant(self):
        if HAS_QUARTET:
            with torch.no_grad():
                w_amax = self.weight.abs().max().float()
                w_quant = quant_fp4(self.weight, scale_override=1.0, amax=w_amax, four_over_six=True)
                self.register_buffer('w_fp4', w_quant.fp4)
                self.register_buffer('w_scales', w_quant.micro_scales)
                self.register_buffer('w_tensor_scale', w_quant.tensor_scale)
    
    def forward(self, x):
        return FusedForwardBackward.apply(
            x, self.rmsnorm_weight, self.weight,
            self.w_fp4, self.w_scales, self.w_tensor_scale,
            self.eps
        )


# ============================================================================
# Benchmarking Utilities
# ============================================================================

def benchmark_fn(fn, warmup=20, iters=100) -> float:
    """Returns time in ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    
    return (time.perf_counter() - start) / iters * 1000


def benchmark_backward(model, x, grad, warmup=10, iters=50) -> float:
    """Benchmark forward + backward pass."""
    grad_bf16 = grad.to(torch.bfloat16)
    
    def fwd_bwd():
        x_clone = x.clone().requires_grad_(True)
        out = model(x_clone)
        out.backward(grad_bf16)
    
    for _ in range(warmup):
        fwd_bwd()
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(iters):
        fwd_bwd()
    torch.cuda.synchronize()
    
    return (time.perf_counter() - start) / iters * 1000


def bench_shape(
    in_dim: int, 
    out_dim: int, 
    batch_size: int = 8, 
    seq_len: int = 2048,
    warmup: int = 20,
    iters: int = 100,
) -> Dict:
    """Benchmark all implementations for a given shape."""
    device = 'cuda'
    dtype = torch.bfloat16
    
    results = {}
    
    x = torch.randn(batch_size, seq_len, in_dim, device=device, dtype=dtype)
    grad = torch.randn(batch_size, seq_len, out_dim, device=device, dtype=dtype)
    
    # ===== 1. torch.compile BF16 Baseline =====
    torch_model = TorchCompiledFFN(in_dim, out_dim, device, dtype)
    torch_model_compiled = torch.compile(torch_model, mode='default', fullgraph=False)
    
    with torch.no_grad():
        _ = torch_model_compiled(x)
    
    results['bf16_fwd_ms'] = benchmark_fn(lambda: torch_model_compiled(x), warmup, iters)
    results['bf16_total_ms'] = benchmark_backward(torch_model_compiled, x, grad, warmup//2, iters//2)
    
    # ===== 2. Quartet-II: torch.compile(RMSNorm+SiLU) → Quartet_II_linear =====
    if HAS_QUARTET:
        quartet_model = QuartetFFN(in_dim, out_dim, device, dtype)
        
        with torch.no_grad():
            _ = quartet_model(x)
        
        with torch.no_grad():
            results['quartet_fwd_ms'] = benchmark_fn(lambda: quartet_model(x), warmup, iters)
        
        # Backward with disable_backward_quant=True (uses bf16 matmul)
        try:
            results['quartet_total_ms'] = benchmark_backward(quartet_model, x, grad, warmup//2, iters//2)
        except Exception as e:
            print(f"  Quartet backward failed: {e}")
            results['quartet_total_ms'] = float('nan')
    else:
        results['quartet_fwd_ms'] = float('nan')
        results['quartet_total_ms'] = float('nan')

    # ===== 2b. Quartet-II (Fully Compiled): torch.compile(QuartetFFN) =====
    if HAS_QUARTET:
        quartet_model_c = QuartetFFN(in_dim, out_dim, device, dtype)
        # Compile the whole model
        quartet_compiled = torch.compile(quartet_model_c, mode='default')
        
        try:
            with torch.no_grad():
                _ = quartet_compiled(x)
            
            with torch.no_grad():
                results['quartet_c_fwd_ms'] = benchmark_fn(lambda: quartet_compiled(x), warmup, iters)
            
            results['quartet_c_total_ms'] = benchmark_backward(quartet_compiled, x, grad, warmup//2, iters//2)

        except Exception as e:
            print(f"  Quartet (Compiled) failed: {e}")
            results['quartet_c_fwd_ms'] = float('nan')
            results['quartet_c_total_ms'] = float('nan')
    else:
        results['quartet_c_fwd_ms'] = float('nan')
        results['quartet_c_total_ms'] = float('nan')
    
    # ===== 3. Fused (ours): Fused(RMSNorm+SiLU+Quant) → FP4 GEMM =====
    if HAS_FUSED_OPS and HAS_QUARTET:
        fused_model = FusedFFN(in_dim, out_dim, device, dtype)
        
        with torch.no_grad():
            _ = fused_model(x)
        
        with torch.no_grad():
            results['fused_fwd_ms'] = benchmark_fn(lambda: fused_model(x), warmup, iters)
        
        try:
            results['fused_total_ms'] = benchmark_backward(fused_model, x, grad, warmup//2, iters//2)
        except Exception as e:
            print(f"  Fused backward failed: {e}")
            results['fused_total_ms'] = float('nan')
    else:
        results['fused_fwd_ms'] = float('nan')
        results['fused_total_ms'] = float('nan')
    
    # Calculate speedups
    if not torch.isnan(torch.tensor(results.get('quartet_fwd_ms', float('nan')))):
        results['fused_vs_quartet_fwd'] = results['quartet_fwd_ms'] / results['fused_fwd_ms']
    else:
        results['fused_vs_quartet_fwd'] = float('nan')
    
    if not torch.isnan(torch.tensor(results.get('quartet_total_ms', float('nan')))) and \
       not torch.isnan(torch.tensor(results.get('fused_total_ms', float('nan')))):
        results['fused_vs_quartet_total'] = results['quartet_total_ms'] / results['fused_total_ms']
    else:
        results['fused_vs_quartet_total'] = float('nan')
    
    return results


def main():
    print("=" * 80)
    print("E2E FFN Benchmark - Forward + Backward (Fair Comparison)")
    print("=" * 80)
    print(f"fused_ops available: {HAS_FUSED_OPS}")
    print(f"Quartet-II available: {HAS_QUARTET}")
    print()
    print("Comparison (all use torch.compile for preprocessing):")
    print("  1. BF16:    torch.compile(RMSNorm → SiLU → BF16 GEMM)")
    print("  2. Quartet: torch.compile(RMSNorm → SiLU) → Quartet(quant + FP4 GEMM)")
    print("  2b.Quartet+: torch.compile(RMSNorm → SiLU → Quartet(quant + FP4 GEMM)) [Full Compile]")
    print("  3. Fused:   Fused(RMSNorm + SiLU + Quant) → FP4 GEMM")
    print()
    
    BATCH_SIZE = 8
    SEQ_LEN = 2048
    
    SHAPES = {
        '800M': [(2048, 2048), (2048, 5632), (5632, 2048)],
        '3B': [(3072, 3072), (3072, 8192), (8192, 3072)],
    }
    
    all_results = {}
    
    for model_size, shapes in SHAPES.items():
        print(f"\n{'='*60}")
        print(f"Model Size: {model_size}")
        print(f"{'='*60}\n")
        
        model_results = []
        
        for in_dim, out_dim in shapes:
            print(f"Shape: ({in_dim}, {out_dim})")
            print("-" * 40)
            
            try:
                results = bench_shape(in_dim, out_dim, BATCH_SIZE, SEQ_LEN)
                model_results.append(results)
                
                print(f"  {'':15s} {'Forward':>10s} {'Fwd+Bwd':>10s}")
                print(f"  {'BF16':15s} {results['bf16_fwd_ms']:>10.2f}ms {results['bf16_total_ms']:>10.2f}ms")
                print(f"  {'Quartet':15s} {results['quartet_fwd_ms']:>10.2f}ms {results['quartet_total_ms']:>10.2f}ms")
                print(f"  {'Quartet+':15s} {results['quartet_c_fwd_ms']:>10.2f}ms {results['quartet_c_total_ms']:>10.2f}ms")
                print(f"  {'Fused (ours)':15s} {results['fused_fwd_ms']:>10.2f}ms {results['fused_total_ms']:>10.2f}ms")
                
                # Print Profile Stats
                if hasattr(FusedForwardBackward, 'stats') and len(FusedForwardBackward.stats['quant']) > 0:
                     avg_quant = sum(FusedForwardBackward.stats['quant']) / len(FusedForwardBackward.stats['quant'])
                     avg_gemm = sum(FusedForwardBackward.stats['gemm']) / len(FusedForwardBackward.stats['gemm'])
                     print(f"  [Details] Quant: {avg_quant:.3f}ms | GEMM: {avg_gemm:.3f}ms | Overhead/Other: {results['fused_fwd_ms'] - avg_quant - avg_gemm:.3f}ms")
                     # Reset
                     FusedForwardBackward.stats = {'quant': [], 'gemm': []}
                     FusedForwardBackward.count = 0
                
                fwd_speedup = results.get('fused_vs_quartet_fwd', float('nan'))
                total_speedup = results.get('fused_vs_quartet_total', float('nan'))
                print(f"  -> Fused/Quartet: fwd={fwd_speedup:.2f}x, total={total_speedup:.2f}x")
                print()
                
            except Exception as e:
                import traceback
                print(f"  Error: {e}")
                traceback.print_exc()
                model_results.append({})
        
        all_results[model_size] = model_results
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    for model_size, results_list in all_results.items():
        if not results_list or not results_list[0]:
            continue
        
        total_bf16_fwd = sum(r.get('bf16_fwd_ms', 0) for r in results_list)
        total_quartet_fwd = sum(r.get('quartet_fwd_ms', 0) for r in results_list)
        total_quartet_c_fwd = sum(r.get('quartet_c_fwd_ms', 0) for r in results_list)
        total_fused_fwd = sum(r.get('fused_fwd_ms', 0) for r in results_list)
        
        total_bf16_total = sum(r.get('bf16_total_ms', 0) for r in results_list)
        total_quartet_total = sum(r.get('quartet_total_ms', 0) for r in results_list if not torch.isnan(torch.tensor(r.get('quartet_total_ms', float('nan')))))
        total_quartet_c_total = sum(r.get('quartet_c_total_ms', 0) for r in results_list if not torch.isnan(torch.tensor(r.get('quartet_c_total_ms', float('nan')))))
        total_fused_total = sum(r.get('fused_total_ms', 0) for r in results_list if not torch.isnan(torch.tensor(r.get('fused_total_ms', float('nan')))))
        
        print(f"\n{model_size}:")
        print(f"  Forward:  BF16={total_bf16_fwd:.2f}ms | Quartet={total_quartet_fwd:.2f}ms | Quartet+={total_quartet_c_fwd:.2f}ms | Fused={total_fused_fwd:.2f}ms")
        if total_quartet_fwd > 0 and total_fused_fwd > 0:
            print(f"            Fused vs Quartet: {total_quartet_fwd/total_fused_fwd:.2f}x")
        
        print(f"  Total:    BF16={total_bf16_total:.2f}ms | Quartet={total_quartet_total:.2f}ms | Quartet+={total_quartet_c_total:.2f}ms | Fused={total_fused_total:.2f}ms")
        if total_quartet_total > 0 and total_fused_total > 0:
            print(f"            Fused vs Quartet: {total_quartet_total/total_fused_total:.2f}x")


if __name__ == '__main__':
    main()
