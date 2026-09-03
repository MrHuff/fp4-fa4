#!/usr/bin/env python3
"""Deterministic one-shape harness for profiling the prepacked mixed route."""

import argparse

import torch

from tk_fa4 import _C_b300_lowp_bwd as lowp
from tk_fa4.interface import b300_mha_fwd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqlen", type=int, default=8192)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026081314)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    qk_shape = (1, args.seqlen, args.heads, 192)
    value_shape = (1, args.seqlen, args.heads, 128)
    q = (torch.randn(qk_shape, device="cuda") * 0.1).bfloat16()
    k = (torch.randn(qk_shape, device="cuda") * 0.1).bfloat16()
    v = (torch.randn(value_shape, device="cuda") * 0.1).bfloat16()
    dout = (torch.randn(value_shape, device="cuda") * 0.1).bfloat16()
    out, lse = b300_mha_fwd(q, k, v, causal=True, return_lse=True)
    softmax_scale = 192**-0.5
    q_fp4, score_q_fp4, k_fp4, score_k_fp4, *_ = (
        lowp.quantize_fp4_dual_qk_blockscale(q, k, 16.0, 16.0)
    )
    mixed_v = lowp.prepack_mixed_v(v)

    def run():
        return lowp.backward_fp4_mixed_fp8_mxfp4dp_fp8dv_x32_split_dk_prepacked_v_native(
            q,
            k,
            v,
            out,
            lse,
            dout,
            q_fp4,
            score_q_fp4,
            k_fp4,
            score_k_fp4,
            16.0,
            16.0,
            4096.0,
            True,
            softmax_scale,
            False,
            mixed_v,
        )

    torch.cuda.synchronize()
    for _ in range(args.warmup):
        run()
    run()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
