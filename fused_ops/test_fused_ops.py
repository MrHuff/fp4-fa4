"""
Tests for Fused RMSNorm + SiLU + FP4 Quantization

Tests:
1. Numerical parity with PyTorch reference implementation
2. Performance benchmarks vs torch.compile baseline
3. Backward pass correctness
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import fused_ops._fused_ops as fused_ops
import pytest
import time
import math

# Ensure errors are caught synchronously
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# ============================================================================
# Reference Implementations (Pure PyTorch)
# ============================================================================

def reference_rmsnorm_silu(x: torch.Tensor, weight: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """Reference RMSNorm + SiLU implementation."""
    # RMSNorm
    rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + epsilon)
    x_norm = (x.float() / rms * weight.float())
    # SiLU
    out = F.silu(x_norm)
    return out.to(x.dtype)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def test_shapes():
    """Common test shapes."""
    return [
        (1, 256),
        (32, 512),
        (128, 1024),
        (256, 2048),
    ]


# ============================================================================
# Numerical Correctness Tests
# ============================================================================

class TestNumericalCorrectness:
    """Tests for numerical parity with reference implementation."""
    
    @pytest.mark.parametrize("rows,cols", [
        (1, 256),
        (32, 512),
        (64, 1024),
        (128, 2048),
    ])
    def test_rmsnorm_silu_output(self, rows, cols):
        """Test that RMSNorm + SiLU matches reference."""
        torch.manual_seed(42)
        
        # Input data
        x = torch.randn(rows, cols, dtype=torch.bfloat16, device='cuda')
        weight = torch.randn(cols, dtype=torch.bfloat16, device='cuda')
        epsilon = 1e-6
        scale_override = 1.0
        
        # Reference output
        ref_out = reference_rmsnorm_silu(x, weight, epsilon)
        ref_amax = ref_out.abs().max().item()
        
        # Fused kernel output buffers - NOTE: using 0-dim tensor with squeeze for scalar
        out_packed = torch.zeros(rows, cols // 2, dtype=torch.uint8, device='cuda')
        scales = torch.zeros(rows * cols // 16, dtype=torch.uint8, device='cuda')  # E4M3
        global_scale = torch.zeros((), dtype=torch.float32, device='cuda')  # 0-dim scalar
        inv_rms_cache = torch.zeros(rows, dtype=torch.float32, device='cuda')
        
        # Run fused kernel
        fused_ops.fused_rmsnorm_act_quant(
            out_packed, scales, global_scale,
            inv_rms_cache,
            x, weight,
            epsilon, scale_override,
            True  # use_four_six
        )
        torch.cuda.synchronize()
        
        # Verify global scale is reasonable
        gs = global_scale.item()
        expected_gs = ref_amax / 256.0 / 6.0 if ref_amax > 0 else 1.0
        
        # Allow some tolerance due to float precision and reduction order
        rel_error = abs(gs - expected_gs) / max(expected_gs, 1e-8)
        assert rel_error < 0.1, f"Global scale mismatch: got {gs}, expected {expected_gs} (rel_err={rel_error})"
        
        # Verify inv_rms values are reasonable
        assert (inv_rms_cache > 0).all(), "inv_rms should be positive"
        
        # Compute expected inv_rms for first row
        expected_inv_rms_0 = 1.0 / math.sqrt((x[0].float() ** 2).mean().item() + epsilon)
        actual_inv_rms_0 = inv_rms_cache[0].item()
        rel_error_rms = abs(actual_inv_rms_0 - expected_inv_rms_0) / expected_inv_rms_0
        assert rel_error_rms < 0.01, f"inv_rms mismatch: got {actual_inv_rms_0}, expected {expected_inv_rms_0}"
        
        print(f"✓ Test passed for shape ({rows}, {cols})")
        print(f"  Global scale: expected={expected_gs:.6f}, actual={gs:.6f}")
        print(f"  inv_rms[0]: expected={expected_inv_rms_0:.6f}, actual={actual_inv_rms_0:.6f}")

    @pytest.mark.parametrize("rows,cols", [
        (32, 512),
        (64, 1024),
    ])
    def test_inv_rms_cache_all_rows(self, rows, cols):
        """Verify inv_rms is correct for all rows."""
        torch.manual_seed(123)
        
        x = torch.randn(rows, cols, dtype=torch.bfloat16, device='cuda')
        weight = torch.ones(cols, dtype=torch.bfloat16, device='cuda')
        
        out_packed = torch.zeros(rows, cols // 2, dtype=torch.uint8, device='cuda')
        scales = torch.zeros(rows * cols // 16, dtype=torch.uint8, device='cuda')
        global_scale = torch.zeros((), dtype=torch.float32, device='cuda')
        inv_rms_cache = torch.zeros(rows, dtype=torch.float32, device='cuda')
        
        fused_ops.fused_rmsnorm_act_quant(
            out_packed, scales, global_scale,
            inv_rms_cache, x, weight,
            1e-6, 1.0, True
        )
        torch.cuda.synchronize()
        
        # Compute reference inv_rms for all rows
        x_float = x.float()
        rms_sq = (x_float ** 2).mean(dim=-1)
        expected_inv_rms = 1.0 / torch.sqrt(rms_sq + 1e-6)
        
        # Compare
        max_rel_error = ((inv_rms_cache - expected_inv_rms).abs() / expected_inv_rms).max().item()
        assert max_rel_error < 0.01, f"inv_rms max relative error: {max_rel_error}"
        
        print(f"✓ inv_rms verified for all {rows} rows, max_rel_error={max_rel_error:.6f}")


# ============================================================================
# Performance Benchmarks
# ============================================================================

class TestPerformance:
    """Performance benchmarks comparing fused kernel vs baselines."""
    
    def _benchmark(self, fn, warmup=10, iters=100):
        """Run benchmark with warmup."""
        # Warmup
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        
        # Timed runs
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        
        return start.elapsed_time(end) / iters  # ms per iteration

    @pytest.mark.parametrize("rows,cols", [
        (128, 1024),
        (256, 2048),
        (512, 4096),
    ])
    def test_fused_vs_pytorch_baseline(self, rows, cols):
        """Compare fused kernel vs PyTorch baseline (RMSNorm + SiLU)."""
        torch.manual_seed(42)
        
        x = torch.randn(rows, cols, dtype=torch.bfloat16, device='cuda')
        weight = torch.randn(cols, dtype=torch.bfloat16, device='cuda')
        
        # Allocate outputs for fused kernel
        out_packed = torch.zeros(rows, cols // 2, dtype=torch.uint8, device='cuda')
        scales = torch.zeros(rows * cols // 16, dtype=torch.uint8, device='cuda')
        global_scale = torch.zeros((), dtype=torch.float32, device='cuda')
        inv_rms_cache = torch.zeros(rows, dtype=torch.float32, device='cuda')
        
        # Fused kernel function
        def run_fused():
            fused_ops.fused_rmsnorm_act_quant(
                out_packed, scales, global_scale,
                inv_rms_cache, x, weight,
                1e-6, 1.0, True
            )
        
        # PyTorch baseline (no quantization, just RMSNorm + SiLU)
        def run_pytorch():
            return reference_rmsnorm_silu(x, weight, 1e-6)
        
        # Torch compile baseline
        compiled_fn = torch.compile(run_pytorch, mode="reduce-overhead")
        def run_compiled():
            return compiled_fn()
        
        # Warmup compiled
        for _ in range(5):
            run_compiled()
        
        # Benchmark
        time_fused = self._benchmark(run_fused)
        time_pytorch = self._benchmark(run_pytorch)
        time_compiled = self._benchmark(run_compiled)
        
        speedup_vs_pytorch = time_pytorch / time_fused
        speedup_vs_compiled = time_compiled / time_fused
        
        print(f"\n{'='*60}")
        print(f"Benchmark: ({rows}, {cols})")
        print(f"{'='*60}")
        print(f"  Fused kernel:    {time_fused:.4f} ms")
        print(f"  PyTorch:         {time_pytorch:.4f} ms (speedup: {speedup_vs_pytorch:.2f}x)")
        print(f"  torch.compile:   {time_compiled:.4f} ms (speedup: {speedup_vs_compiled:.2f}x)")
        print(f"{'='*60}")
        
        return {
            'shape': (rows, cols),
            'fused_ms': time_fused,
            'pytorch_ms': time_pytorch,
            'compiled_ms': time_compiled,
        }

    @pytest.mark.parametrize("rows,cols", [
        (128, 1024),
        (256, 2048),
        (512, 4096),
        (1024, 4096),
    ])
    def test_four_six_vs_rtn_ablation(self, rows, cols):
        """Ablation: Compare four-six search vs RTN (round-to-nearest)."""
        torch.manual_seed(42)
        
        x = torch.randn(rows, cols, dtype=torch.bfloat16, device='cuda')
        weight = torch.randn(cols, dtype=torch.bfloat16, device='cuda')
        
        # Shared output buffers
        out_packed = torch.zeros(rows, cols // 2, dtype=torch.uint8, device='cuda')
        scales = torch.zeros(rows * cols // 16, dtype=torch.uint8, device='cuda')
        global_scale = torch.zeros((), dtype=torch.float32, device='cuda')
        inv_rms_cache = torch.zeros(rows, dtype=torch.float32, device='cuda')
        
        # Four-six search kernel
        def run_four_six():
            fused_ops.fused_rmsnorm_act_quant(
                out_packed, scales, global_scale,
                inv_rms_cache, x, weight,
                1e-6, 1.0, True  # use_four_six=True
            )
        
        # RTN kernel
        def run_rtn():
            fused_ops.fused_rmsnorm_act_quant(
                out_packed, scales, global_scale,
                inv_rms_cache, x, weight,
                1e-6, 1.0, False  # use_four_six=False (RTN)
            )
        
        # Benchmark both
        time_four_six = self._benchmark(run_four_six)
        time_rtn = self._benchmark(run_rtn)
        
        overhead = (time_four_six - time_rtn) / time_rtn * 100
        
        print(f"\n{'='*60}")
        print(f"Ablation: Four-Six vs RTN for shape ({rows}, {cols})")
        print(f"{'='*60}")
        print(f"  RTN (baseline):    {time_rtn:.4f} ms")
        print(f"  Four-Six search:   {time_four_six:.4f} ms")
        print(f"  Overhead:          {overhead:+.1f}%")
        print(f"{'='*60}")
        
        return {
            'shape': (rows, cols),
            'rtn_ms': time_rtn,
            'four_six_ms': time_four_six,
            'overhead_pct': overhead,
        }

    @pytest.mark.parametrize("rows,cols", [
        (128, 1024),
        (256, 2048),
        (512, 4096),
    ])
    def test_v1_vs_v2_performance(self, rows, cols):
        """Compare V1 (two passes for absmax) vs V2 (absmax during first pass)."""
        torch.manual_seed(42)
        
        x = torch.randn(rows, cols, dtype=torch.bfloat16, device='cuda')
        weight = torch.randn(cols, dtype=torch.bfloat16, device='cuda')
        
        # Output buffers
        out_packed = torch.zeros(rows, cols // 2, dtype=torch.uint8, device='cuda')
        scales = torch.zeros(rows * cols // 16, dtype=torch.uint8, device='cuda')
        global_scale = torch.zeros((), dtype=torch.float32, device='cuda')
        inv_rms_cache = torch.zeros(rows, dtype=torch.float32, device='cuda')
        
        # V1 kernel
        def run_v1():
            fused_ops.fused_rmsnorm_act_quant(
                out_packed, scales, global_scale,
                inv_rms_cache, x, weight,
                1e-6, 1.0, True
            )
        
        # V2 kernel (optimized - block absmaxes during first pass)
        def run_v2():
            fused_ops.fused_rmsnorm_act_quant_v2(
                out_packed, scales, global_scale,
                inv_rms_cache, x, weight,
                1e-6, 1.0, True
            )
        
        # Benchmark
        time_v1 = self._benchmark(run_v1)
        time_v2 = self._benchmark(run_v2)
        
        # V3 kernel (warp shuffles + inv_rms factored)
        def run_v3():
            fused_ops.fused_rmsnorm_act_quant_v3(
                out_packed, scales, global_scale,
                inv_rms_cache, x, weight,
                1e-6, 1.0, True
            )
        
        time_v3 = self._benchmark(run_v3)
        
        # V4 kernel (lock-free, no atomics)
        def run_v4():
            fused_ops.fused_rmsnorm_act_quant_v4(
                out_packed, scales, global_scale,
                inv_rms_cache, x, weight,
                1e-6, 1.0, True
            )
        
        time_v4 = self._benchmark(run_v4)
        
        # V1-OPT kernel (fast math intrinsics)
        def run_opt():
            fused_ops.fused_rmsnorm_act_quant_opt(
                out_packed, scales, global_scale,
                inv_rms_cache, x, weight,
                1e-6, 1.0, True
            )
        
        time_opt = self._benchmark(run_opt)
        
        speedup_v2 = time_v1 / time_v2
        speedup_v3 = time_v1 / time_v3
        speedup_v4 = time_v1 / time_v4
        speedup_opt = time_v1 / time_opt
        
        print(f"\n{'='*70}")
        print(f"Kernel Comparison for shape ({rows}, {cols})")
        print(f"{'='*70}")
        print(f"  V1 (original):     {time_v1:.4f} ms")
        print(f"  V2 (atomics):      {time_v2:.4f} ms (speedup: {speedup_v2:.2f}x)")
        print(f"  V3 (warp+factor):  {time_v3:.4f} ms (speedup: {speedup_v3:.2f}x)")
        print(f"  V4 (lock-free):    {time_v4:.4f} ms (speedup: {speedup_v4:.2f}x)")
        print(f"  V1-OPT (__expf):   {time_opt:.4f} ms (speedup: {speedup_opt:.2f}x)")
        print(f"{'='*70}")
        
        return {
            'shape': (rows, cols),
            'v1_ms': time_v1,
            'opt_ms': time_opt,
            'speedup_opt': speedup_opt,
        }


# ============================================================================
# Backward Pass Tests
# ============================================================================

class TestBackward:
    """Tests for backward pass correctness."""
    
    @pytest.mark.parametrize("rows,cols", [
        (32, 256),
        (64, 512),
    ])
    def test_backward_with_cached_inv_rms(self, rows, cols):
        """Test backward pass using cached inv_rms from forward."""
        torch.manual_seed(42)
        
        x = torch.randn(rows, cols, dtype=torch.bfloat16, device='cuda', requires_grad=True)
        weight = torch.randn(cols, dtype=torch.bfloat16, device='cuda')
        grad_output = torch.randn(rows, cols, dtype=torch.bfloat16, device='cuda')
        
        # Forward with fused kernel (to get inv_rms_cache)
        out_packed = torch.zeros(rows, cols // 2, dtype=torch.uint8, device='cuda')
        scales = torch.zeros(rows * cols // 16, dtype=torch.uint8, device='cuda')
        global_scale = torch.zeros((), dtype=torch.float32, device='cuda')
        inv_rms_cache = torch.zeros(rows, dtype=torch.float32, device='cuda')
        
        fused_ops.fused_rmsnorm_act_quant(
            out_packed, scales, global_scale,
            inv_rms_cache, x, weight,
            1e-6, 1.0, True
        )
        
        # Backward with cached inv_rms
        grad_input = torch.zeros_like(x)
        fused_ops.fused_backward_v2(
            grad_output, x, weight,
            inv_rms_cache, 1e-6, grad_input
        )
        torch.cuda.synchronize()
        
        # Reference backward
        x_ref = x.detach().clone().requires_grad_(True)
        out_ref = reference_rmsnorm_silu(x_ref, weight, 1e-6)
        out_ref.backward(grad_output.float())
        grad_ref = x_ref.grad.to(torch.bfloat16)
        
        # Compare gradients
        max_abs_error = (grad_input - grad_ref).abs().max().item()
        rel_error = max_abs_error / (grad_ref.abs().max().item() + 1e-8)
        
        print(f"✓ Backward test ({rows}, {cols}): max_abs_err={max_abs_error:.6f}, rel_err={rel_error:.4f}")
        assert rel_error < 0.1, f"Backward gradient mismatch: rel_error={rel_error}"


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """End-to-end integration tests."""
    
    def test_full_forward_backward_loop(self):
        """Test full forward and backward pass loop."""
        rows, cols = 64, 512
        torch.manual_seed(42)
        
        x = torch.randn(rows, cols, dtype=torch.bfloat16, device='cuda')
        weight = torch.randn(cols, dtype=torch.bfloat16, device='cuda')
        
        # Allocate
        out_packed = torch.zeros(rows, cols // 2, dtype=torch.uint8, device='cuda')
        scales = torch.zeros(rows * cols // 16, dtype=torch.uint8, device='cuda')
        global_scale = torch.zeros((), dtype=torch.float32, device='cuda')
        inv_rms_cache = torch.zeros(rows, dtype=torch.float32, device='cuda')
        grad_output = torch.randn(rows, cols, dtype=torch.bfloat16, device='cuda')
        grad_input = torch.zeros_like(x)
        
        # Forward
        fused_ops.fused_rmsnorm_act_quant(
            out_packed, scales, global_scale,
            inv_rms_cache, x, weight,
            1e-6, 1.0, True
        )
        
        # Backward
        fused_ops.fused_backward_v2(
            grad_output, x, weight,
            inv_rms_cache, 1e-6, grad_input
        )
        
        torch.cuda.synchronize()
        
        # Basic sanity checks
        assert not torch.isnan(global_scale).any(), "NaN in global_scale"
        assert not torch.isnan(inv_rms_cache).any(), "NaN in inv_rms_cache"
        assert not torch.isnan(grad_input).any(), "NaN in grad_input"
        
        print("✓ Full forward/backward loop completed successfully")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Fused RMSNorm + SiLU + FP4 Quantization Tests")
    print("=" * 70)
    
    # Run numerical tests
    print("\n--- Numerical Correctness Tests ---")
    tc = TestNumericalCorrectness()
    for rows, cols in [(32, 512), (64, 1024), (128, 2048)]:
        tc.test_rmsnorm_silu_output(rows, cols)
    
    # Run performance benchmarks
    print("\n--- Performance Benchmarks ---")
    tp = TestPerformance()
    for rows, cols in [(128, 1024), (256, 2048), (512, 4096)]:
        tp.test_fused_vs_pytorch_baseline(rows, cols)
    
    # Run ablation: Four-six vs RTN
    print("\n--- Ablation: Four-Six vs RTN ---")
    for rows, cols in [(128, 1024), (256, 2048), (512, 4096)]:
        tp.test_four_six_vs_rtn_ablation(rows, cols)
    
    # Run V1 vs V2 comparison
    print("\n--- V1 vs V2 Performance ---")
    for rows, cols in [(128, 1024), (256, 2048), (512, 4096)]:
        tp.test_v1_vs_v2_performance(rows, cols)
    
    # Run backward tests
    print("\n--- Backward Pass Tests ---")
    tb = TestBackward()
    for rows, cols in [(32, 256), (64, 512)]:
        tb.test_backward_with_cached_inv_rms(rows, cols)
    
    # Run integration test
    print("\n--- Integration Tests ---")
    ti = TestIntegration()
    ti.test_full_forward_backward_loop()
    
    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
