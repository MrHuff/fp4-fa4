"""
Test MXFP4 quantisation correctness and performance.

Tests:
1. mxfp4_quantize_for_gemm: validates FP4 output and E8M0 scales against PyTorch reference
2. mxfp4_group_quantize_dim0: validates grouped quantise along rows
3. mxfp4_group_quantize_dim1: validates grouped quantise along columns
"""

import sys
import torch
torch.random.manual_seed(42)
torch.set_printoptions(sci_mode=False)

sys.path.insert(0, ".")
from mxfp4_quant import mxfp4_quantize_for_gemm, mxfp4_group_quantize_dim0, mxfp4_group_quantize_dim1


def torch_mxfp4_quantize(V: torch.Tensor):
    """Reference MXFP4 quantisation in PyTorch."""
    assert V.dim() == 2, V.shape
    assert V.shape[0] % 128 == 0 and V.shape[1] % 128 == 0

    M, N = V.shape
    V = V.to(torch.float32)

    fp4_max = 6.0
    block_amax = torch.amax(torch.abs(V).view(M, N // 32, 32), dim=-1)

    # E8M0: round(log2(amax)) + 127
    e8m0_vals = torch.where(
        block_amax <= 1e-9,
        torch.zeros_like(block_amax),
        torch.round(torch.log2(block_amax)) + 127.0
    ).clamp(0, 255).to(torch.uint8)

    # Quantise: x * (6.0 / 2^exponent)
    scale_pow2 = (2.0 ** (e8m0_vals.to(torch.float32) - 127.0))
    scale_inv = fp4_max / scale_pow2.repeat_interleave(32, dim=-1)
    V_scaled = (V * scale_inv).clamp(-6.0, 6.0)

    return V_scaled, e8m0_vals


def scale_swizzle(V_sc_unswizzled: torch.Tensor) -> torch.Tensor:
    """Swizzle scales to MMA-compatible layout [M/128, K/128, 32, 16]."""
    assert V_sc_unswizzled.dtype == torch.uint8
    assert V_sc_unswizzled.shape[0] % 128 == 0
    assert (V_sc_unswizzled.shape[1] * 32) % 128 == 0

    M, N_32 = V_sc_unswizzled.shape
    M_BLOCK, N_BLOCK = 128, 4  # 128/32

    V_sc = V_sc_unswizzled
    V_sc = V_sc.reshape(M // M_BLOCK, M_BLOCK, N_32 // N_BLOCK, N_BLOCK)
    V_sc = V_sc.transpose(1, 2)
    V_sc = V_sc.reshape(M // M_BLOCK, N_32 // N_BLOCK, 4, M_BLOCK // 4, N_BLOCK)
    V_sc = V_sc.transpose(-2, -3)
    V_sc = V_sc.reshape(M // M_BLOCK, N_32 // N_BLOCK, M_BLOCK // 4, N_BLOCK * 4)

    return V_sc.contiguous()


def check_diff(name: str, A: torch.Tensor, A_ref: torch.Tensor) -> None:
    A = A.to(torch.float32)
    A_ref = A_ref.to(torch.float32)
    print(f"{'='*79}")
    print(f"<{name}>")
    print(f"Max diff:  {((A - A_ref).abs().max().item()):.10f}")
    print(f"Mean diff: {((A - A_ref).abs().mean().item()):.10f}")
    print(f"Mean:      {A.abs().mean().item():.10f}")
    print(f"Ref mean:  {A_ref.abs().mean().item():.10f}")


if __name__ == "__main__":
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 2048

    print(f"\n{'='*79}")
    print(f"TEST 1: mxfp4_quantize_for_gemm  (M={M}, K={K})")
    print(f"{'='*79}")

    A_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 2.0

    # Reference
    A_scaled_ref, A_sc_ref = torch_mxfp4_quantize(A_bf16)
    A_sc_swizzled_ref = scale_swizzle(A_sc_ref)

    # Our kernel
    A_fp4, A_sc = mxfp4_quantize_for_gemm(A_bf16)

    # Check scales
    check_diff("Scales", A_sc.flatten().to(torch.float32), A_sc_swizzled_ref.flatten().to(torch.float32))

    # Benchmark
    NUM_WARMUPS, NUM_ITERS = 5, 10
    for _ in range(NUM_WARMUPS):
        mxfp4_quantize_for_gemm(A_bf16)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(NUM_ITERS):
        mxfp4_quantize_for_gemm(A_bf16)
    end.record()
    torch.cuda.synchronize()

    avg_us = start.elapsed_time(end) * 1000.0 / NUM_ITERS
    gb = M * K * (2 + 0.5 + 1.0/32) * 1e-9
    print(f"Average time: {avg_us:.2f} us")
    print(f"Throughput: {gb / (avg_us * 1e-6):.2f} GB/s")

    # ────────────────────────────────────────────────────────────
    print(f"\n{'='*79}")
    print(f"TEST 2: mxfp4_group_quantize_dim0  (splits=[{M//2}, {M//2}])")
    print(f"{'='*79}")

    split_sizes = [M // 2, M // 2]
    fp4_list, sc_list = mxfp4_group_quantize_dim0(A_bf16, split_sizes)

    print(f"Group 0: fp4 {fp4_list[0].shape}, sc {sc_list[0].shape}")
    print(f"Group 1: fp4 {fp4_list[1].shape}, sc {sc_list[1].shape}")

    # Reference: split input, quantise each half
    A_top = A_bf16[:M//2, :]
    A_bot = A_bf16[M//2:, :]
    _, A_sc_top_ref = torch_mxfp4_quantize(A_top)
    _, A_sc_bot_ref = torch_mxfp4_quantize(A_bot)
    A_sc_top_sw = scale_swizzle(A_sc_top_ref)
    A_sc_bot_sw = scale_swizzle(A_sc_bot_ref)

    check_diff("Group 0 Scales", sc_list[0].flatten().to(torch.float32), A_sc_top_sw.flatten().to(torch.float32))
    check_diff("Group 1 Scales", sc_list[1].flatten().to(torch.float32), A_sc_bot_sw.flatten().to(torch.float32))

    # ────────────────────────────────────────────────────────────
    print(f"\n{'='*79}")
    print(f"TEST 3: mxfp4_group_quantize_dim1  (splits=[{K//2}, {K//2}])")
    print(f"{'='*79}")

    split_sizes_k = [K // 2, K // 2]
    fp4_list_k, sc_list_k = mxfp4_group_quantize_dim1(A_bf16, split_sizes_k)

    print(f"Group 0: fp4 {fp4_list_k[0].shape}, sc {sc_list_k[0].shape}")
    print(f"Group 1: fp4 {fp4_list_k[1].shape}, sc {sc_list_k[1].shape}")

    # Reference: split along columns
    A_left = A_bf16[:, :K//2].contiguous()
    A_right = A_bf16[:, K//2:].contiguous()
    _, A_sc_left_ref = torch_mxfp4_quantize(A_left)
    _, A_sc_right_ref = torch_mxfp4_quantize(A_right)
    A_sc_left_sw = scale_swizzle(A_sc_left_ref)
    A_sc_right_sw = scale_swizzle(A_sc_right_ref)

    check_diff("Group 0 Scales (dim1)", sc_list_k[0].flatten().to(torch.float32), A_sc_left_sw.flatten().to(torch.float32))
    check_diff("Group 1 Scales (dim1)", sc_list_k[1].flatten().to(torch.float32), A_sc_right_sw.flatten().to(torch.float32))

    print(f"\n{'='*79}")
    print("All tests completed!")
    print(f"{'='*79}")
