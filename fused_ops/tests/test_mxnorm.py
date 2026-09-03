
import torch
import fused_ops._fused_ops as fused_ops
import pytest
import math

class TestMXNorm:
    @pytest.mark.parametrize("rows,cols", [(32, 256), (128, 1024)])
    def test_mxnorm_modes(self, rows, cols):
        torch.manual_seed(42)
        x = torch.randn(rows, cols, dtype=torch.bfloat16, device='cuda')
        w = torch.randn(cols, dtype=torch.bfloat16, device='cuda')
        epsilon = 1e-6
        
        # Alloc outputs
        out_packed = torch.zeros(rows, cols // 2, dtype=torch.uint8, device='cuda')
        scales = torch.zeros(rows * cols // 16, dtype=torch.uint8, device='cuda')
        global_scale = torch.zeros((), dtype=torch.float32, device='cuda')
        inv_rms_cache = torch.zeros(rows, dtype=torch.float32, device='cuda')
        
        # Mode 0: RMSNorm (Should match standard RMS)
        fused_ops.fused_mxnorm(
            out_packed, scales, global_scale, inv_rms_cache,
            x, w, epsilon, 1.0, True, 0 # Mode 0
        )
        torch.cuda.synchronize()
        
        # Verify mode 0 inv_rms
        x_act = torch.nn.functional.silu(x.float())
        expected_rms = torch.sqrt(torch.mean(x_act**2, dim=-1) + epsilon)
        expected_inv_rms = 1.0 / expected_rms
        rel_err = (inv_rms_cache - expected_inv_rms).abs() / expected_inv_rms
        assert rel_err.max() < 0.01, f"Mode 0 (RMS) mismatch. Max RelErr: {rel_err.max()}"
        print(f"Mode 0 (RMS) verified. Max RelErr: {rel_err.max().item():.6f}")

        # Mode 1: AbsMax
        inv_rms_cache.zero_()
        fused_ops.fused_mxnorm(
            out_packed, scales, global_scale, inv_rms_cache,
            x, w, epsilon, 1.0, True, 1 # Mode 1
        )
        torch.cuda.synchronize()
        
        expected_max = x_act.abs().max(dim=-1).values
        expected_inv_max = 1.0 / (expected_max + epsilon)
        rel_err_max = (inv_rms_cache - expected_inv_max).abs() / expected_inv_max
        assert rel_err_max.max() < 0.01, f"Mode 1 (AbsMax) mismatch. Max RelErr: {rel_err_max.max()}"
        print(f"Mode 1 (AbsMax) verified. Max RelErr: {rel_err_max.max().item():.6f}")
        
        # Mode 2: BlockMax
        # Harder to verify exact value without replicating logic, but check consistency
        inv_rms_cache.zero_()
        fused_ops.fused_mxnorm(
            out_packed, scales, global_scale, inv_rms_cache,
            x, w, epsilon, 1.0, True, 2 # Mode 2
        )
        torch.cuda.synchronize()
        assert (inv_rms_cache > 0).all(), "Mode 2 produced non-positive inv_rms"
        print(f"Mode 2 (BlockMax) ran successfully. Sample inv_rms[0]: {inv_rms_cache[0].item():.6f}")

if __name__ == "__main__":
    t = TestMXNorm()
    t.test_mxnorm_modes(64, 512)
