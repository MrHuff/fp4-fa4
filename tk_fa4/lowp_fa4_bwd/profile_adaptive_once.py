#!/usr/bin/env python3
"""Deterministic one-shape harness for Nsight Compute application replay."""

import argparse

import torch

from tk_fa4 import _C_b300_lowp_bwd as lowp
from tk_fa4.interface import b300_mha_fwd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqlen", type=int, default=8192)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026081801)
    parser.add_argument("--bf16-dq", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    shape_qk = (1, args.seqlen, args.heads, 192)
    shape_v = (1, args.seqlen, args.heads, 128)
    q = (torch.randn(shape_qk, device="cuda") * 0.1).bfloat16()
    k = (torch.randn(shape_qk, device="cuda") * 0.1).bfloat16()
    v = (torch.randn(shape_v, device="cuda") * 0.1).bfloat16()
    dout = (torch.randn(shape_v, device="cuda") * 0.1).bfloat16()
    out, lse = b300_mha_fwd(q, k, v, causal=True, return_lse=True)
    softmax_scale = 192**-0.5
    packed = lowp.quantize_fp4_dual_qk_adaptive(
        q,
        k,
        16.0,
        2**-12,
        0.325,
        2.75,
        softmax_scale,
        4096.0,
    )
    backward = (
        lowp.backward_fp4_fp8dpdv_x32_split_dk_adaptive_bf16dq_native
        if args.bf16_dq
        else lowp.backward_fp4_fp8dpdv_x32_split_dk_adaptive_native
    )

    def run():
        return backward(
            q,
            k,
            v,
            out,
            lse,
            dout,
            *packed,
            4096.0,
            True,
            softmax_scale,
            False,
        )

    for _ in range(args.warmup):
        run()
    run()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
