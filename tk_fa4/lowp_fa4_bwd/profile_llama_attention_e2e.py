#!/usr/bin/env python3
"""Profile components of the composed low-precision attention sublayer.

This diagnostic shares the contract and problem construction used by
``evaluate_llama_attention_e2e.py``.  Dependencies are prepared once so each
reported number isolates one boundary: QKV projection, attention forward,
output projection, dO projection, attention backward, QKV dgrad projection,
or the two learned-weight gradients.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from tk_fa4 import _C_b300_lowp_bwd as lowp
from tk_fa4.interface import (
    b300_mha_fwd,
    b300_prepare_nvfp4_projection_operand,
    b300_project_nvfp4,
)
from tk_fa4.fp4_pv_experiments import _run_forward_streaming_live_mxfp4
from tk_fa4.lowp_fa4_bwd.evaluate_llama_attention_e2e import (
    QK_DIM,
    V_DIM,
    _interleave_gradients,
    _project_interleaved_bf16,
    _project_qkv_lowp,
    _projection_splits,
    build_problem,
    parse_shapes,
    time_rotated,
)


def profile_shape(
    sequence: int,
    heads: int,
    hidden: int,
    *,
    seed: int,
    warmups: int,
    samples: int,
) -> dict[str, Any]:
    problem = build_problem(sequence, heads, hidden, seed)

    q_bf16 = torch.mm(problem.x, problem.q_weight.T).reshape(
        1, sequence, heads, QK_DIM
    )
    k_bf16 = torch.mm(problem.x, problem.k_weight.T).reshape_as(q_bf16)
    v_bf16 = torch.mm(problem.x, problem.v_weight.T).reshape(
        1, sequence, heads, V_DIM
    )
    out_bf16, lse_bf16 = b300_mha_fwd(
        q_bf16,
        k_bf16,
        v_bf16,
        causal=True,
        return_lse=True,
    )
    dout_bf16 = torch.mm(problem.dy, problem.out_weight).reshape_as(out_bf16)
    dq_bf16, dk_bf16, dv_bf16 = lowp.backward_bf16_control(
        q_bf16,
        k_bf16,
        v_bf16,
        out_bf16,
        lse_bf16,
        dout_bf16,
        True,
        float(QK_DIM**-0.5),
        False,
    )
    qkv_grad_bf16 = _interleave_gradients(dq_bf16, dk_bf16, dv_bf16)

    bundle = _project_qkv_lowp(problem, charge_input_pack=False)
    assert bundle.q is not None and bundle.k is not None and bundle.v is not None
    out_lowp, lse_lowp_bhs = _run_forward_streaming_live_mxfp4(
        *bundle.forward_operands()
    )
    lse_lowp = lse_lowp_bhs.permute(0, 2, 1).contiguous()
    out_matrix = out_lowp.reshape(problem.rows, problem.v_width)
    dout_lowp = b300_project_nvfp4(
        problem.dy_operand,
        problem.out_backward_weight_operand,
    ).reshape_as(out_lowp)

    def fixed_backward() -> torch.Tensor:
        (qkv_grad,) = (
            lowp.backward_fp4_fp8dpdv_x32_split_dk_adaptive_qkv_bf16_native(
                bundle.q,
                bundle.k,
                bundle.v,
                out_lowp,
                lse_lowp,
                dout_lowp,
                bundle.backward.q_fp4,
                bundle.backward.score_q_fp4,
                bundle.backward.k_fp4,
                bundle.backward.score_k_fp4,
                bundle.backward.qk_scales,
                4096.0,
                True,
                float(QK_DIM**-0.5),
                False,
            )
        )
        return qkv_grad

    qkv_grad_lowp = fixed_backward()
    out_operand = tuple(b300_prepare_nvfp4_projection_operand(out_matrix))
    qkv_grad_matrix = qkv_grad_lowp.reshape(problem.rows, problem.qkv_width)
    qkv_grad_operand = tuple(
        b300_prepare_nvfp4_projection_operand(qkv_grad_matrix)
    )
    out_global_scale = out_operand[2]
    qkv_grad_global_scale = qkv_grad_operand[2]

    def bf16_qkv_projection() -> object:
        return (
            torch.mm(problem.x, problem.q_weight.T).reshape(
                1, sequence, heads, QK_DIM
            ),
            torch.mm(problem.x, problem.k_weight.T).reshape(
                1, sequence, heads, QK_DIM
            ),
            torch.mm(problem.x, problem.v_weight.T).reshape(
                1, sequence, heads, V_DIM
            ),
        )

    def lowp_qkv_projection() -> object:
        return _project_qkv_lowp(problem, charge_input_pack=False)

    def lowp_qkv_projection_materialized() -> object:
        return _project_qkv_lowp(problem, charge_input_pack=True)

    def bf16_attention_forward() -> object:
        return b300_mha_fwd(
            q_bf16,
            k_bf16,
            v_bf16,
            causal=True,
            return_lse=True,
        )

    def lowp_attention_forward() -> object:
        return _run_forward_streaming_live_mxfp4(*bundle.forward_operands())

    def nvfp4_output_full_scale() -> torch.Tensor:
        operand = tuple(b300_prepare_nvfp4_projection_operand(out_matrix))
        return b300_project_nvfp4(
            operand,
            problem.out_forward_weight_operand,
        )

    def nvfp4_output_delayed_scale() -> torch.Tensor:
        operand = tuple(
            b300_prepare_nvfp4_projection_operand(
                out_matrix,
                global_scale=out_global_scale,
            )
        )
        return b300_project_nvfp4(
            operand,
            problem.out_forward_weight_operand,
        )

    def bf16_attention_backward() -> torch.Tensor:
        dq, dk, dv = lowp.backward_bf16_control(
            q_bf16,
            k_bf16,
            v_bf16,
            out_bf16,
            lse_bf16,
            dout_bf16,
            True,
            float(QK_DIM**-0.5),
            False,
        )
        return _interleave_gradients(dq, dk, dv)

    def nvfp4_qkv_dgrad_full_scale() -> torch.Tensor:
        operand = tuple(
            b300_prepare_nvfp4_projection_operand(qkv_grad_matrix)
        )
        return b300_project_nvfp4(
            operand,
            problem.qkv_backward_weight_operand,
        )

    def nvfp4_qkv_dgrad_delayed_scale() -> torch.Tensor:
        operand = tuple(
            b300_prepare_nvfp4_projection_operand(
                qkv_grad_matrix,
                global_scale=qkv_grad_global_scale,
            )
        )
        return b300_project_nvfp4(
            operand,
            problem.qkv_backward_weight_operand,
        )

    components = {
        "qkv_projection/bf16_three_direct": bf16_qkv_projection,
        "qkv_projection/nvfp4_unified": lowp_qkv_projection,
        "qkv_projection/nvfp4_materialized_x": lowp_qkv_projection_materialized,
        "attention_forward/bf16": bf16_attention_forward,
        "attention_forward/fp4": lowp_attention_forward,
        "output_projection/bf16": lambda: torch.mm(
            out_matrix,
            problem.out_weight.T,
        ),
        "output_projection/nvfp4_gemm_only": lambda: b300_project_nvfp4(
            out_operand,
            problem.out_forward_weight_operand,
        ),
        "output_projection/nvfp4_delayed_scale": nvfp4_output_delayed_scale,
        "output_projection/nvfp4_full_scale": nvfp4_output_full_scale,
        "dout_projection/bf16": lambda: torch.mm(problem.dy, problem.out_weight),
        "dout_projection/nvfp4_gemm_only": lambda: b300_project_nvfp4(
            problem.dy_operand,
            problem.out_backward_weight_operand,
        ),
        "attention_backward/bf16": bf16_attention_backward,
        "attention_backward/fixed_fp4_fp8": fixed_backward,
        "qkv_dgrad_projection/bf16": lambda: _project_interleaved_bf16(
            qkv_grad_bf16,
            problem.qkv_weight_interleaved,
            splits=_projection_splits(problem.sequence, problem.heads),
        ),
        "qkv_dgrad_projection/nvfp4_gemm_only": lambda: b300_project_nvfp4(
            qkv_grad_operand,
            problem.qkv_backward_weight_operand,
        ),
        "qkv_dgrad_projection/nvfp4_delayed_scale": (
            nvfp4_qkv_dgrad_delayed_scale
        ),
        "qkv_dgrad_projection/nvfp4_full_scale": nvfp4_qkv_dgrad_full_scale,
        "weight_gradient/qkv": lambda: torch.mm(
            qkv_grad_matrix.T,
            problem.x,
        ),
        "weight_gradient/output": lambda: torch.mm(problem.dy.T, out_matrix),
        "packing/x_full_scale": lambda: b300_prepare_nvfp4_projection_operand(
            problem.x
        ),
        "packing/output_full_scale": lambda: (
            b300_prepare_nvfp4_projection_operand(out_matrix)
        ),
        "packing/output_delayed_scale": lambda: (
            b300_prepare_nvfp4_projection_operand(
                out_matrix,
                global_scale=out_global_scale,
            )
        ),
        "packing/qkv_grad_full_scale": lambda: (
            b300_prepare_nvfp4_projection_operand(qkv_grad_matrix)
        ),
        "packing/qkv_grad_delayed_scale": lambda: (
            b300_prepare_nvfp4_projection_operand(
                qkv_grad_matrix,
                global_scale=qkv_grad_global_scale,
            )
        ),
    }
    timing = time_rotated(components, warmups=warmups, samples=samples)
    return {
        "shape": {
            "batch": 1,
            "sequence": sequence,
            "heads": heads,
            "hidden": hidden,
            "qk_head_dim": QK_DIM,
            "v_head_dim": V_DIM,
        },
        "timing": timing,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shapes",
        default="4096x24x3072,4096x64x8192,8192x16x2048",
    )
    parser.add_argument("--seed", type=int, default=2026081418)
    parser.add_argument("--warmups", type=int, default=4)
    parser.add_argument("--samples", type=int, default=13)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("profiling requires exactly one visible GPU")
    torch.cuda.set_device(0)

    result = {
        "configuration": {
            "shapes": args.shapes,
            "seed": args.seed,
            "warmups": args.warmups,
            "samples": args.samples,
        },
        "records": [],
    }
    for index, shape in enumerate(parse_shapes(args.shapes)):
        record = profile_shape(
            *shape,
            seed=args.seed + index * 17,
            warmups=args.warmups,
            samples=args.samples,
        )
        result["records"].append(record)
        print(
            f"profiled S{shape[0]} H{shape[1]} K{shape[2]}",
            flush=True,
        )
        torch.cuda.empty_cache()

    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
