#!/usr/bin/env python3
"""Profile one retained producer-native-V backward invocation."""

from __future__ import annotations

import argparse

import torch

from tk_fa4 import _C_b300_lowp_bwd as lowp
from tk_fa4.fp4_pv_experiments import _run_forward_streaming_live_mxfp4
from tk_fa4.interface import b300_project_nvfp4
from tk_fa4.lowp_fa4_bwd.evaluate_llama_attention_e2e import (
    QK_DIM,
    _project_qkv_lowp,
    build_problem,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqlen", type=int, default=8192)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026081317)
    args = parser.parse_args()

    problem = build_problem(
        args.seqlen,
        args.heads,
        args.hidden,
        args.seed,
        use_rope=False,
    )
    bundle = _project_qkv_lowp(
        problem,
        charge_input_pack=False,
        publish_fp8_backward=True,
    )
    assert bundle.q is not None and bundle.k is not None and bundle.v is not None
    assert bundle.v_backward_fp8 is not None
    out, lse_bhs = _run_forward_streaming_live_mxfp4(
        *bundle.forward_operands()
    )
    lse = lse_bhs.permute(0, 2, 1).contiguous()
    dout = b300_project_nvfp4(
        problem.dy_operand,
        problem.out_backward_weight_operand,
    ).reshape_as(out)

    def run() -> object:
        return lowp.backward_fp4_fp8dpdv_x32_split_dk_adaptive_qkv_bf16_prepacked_v_native(
            bundle.q,
            bundle.k,
            bundle.v,
            out,
            lse,
            dout,
            bundle.backward.q_fp4,
            bundle.backward.score_q_fp4,
            bundle.backward.k_fp4,
            bundle.backward.score_k_fp4,
            bundle.backward.qk_scales,
            bundle.v_backward_fp8,
            4096.0,
            True,
            float(QK_DIM**-0.5),
            False,
        )

    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()
    cudart = torch.cuda.cudart()
    if cudart.cudaProfilerStart() != 0:
        raise RuntimeError("cudaProfilerStart failed")
    run()
    torch.cuda.synchronize()
    if cudart.cudaProfilerStop() != 0:
        raise RuntimeError("cudaProfilerStop failed")


if __name__ == "__main__":
    main()
