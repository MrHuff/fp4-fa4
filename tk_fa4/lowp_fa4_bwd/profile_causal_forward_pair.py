#!/usr/bin/env python3
"""Compare two read-only causal GQA forward artifacts on identical operands."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch

from tk_fa4 import (
    b300_pack_gqa_d128_rope,
    b300_pair_interleave_gqa_d128_qk_projection_weights,
    b300_prepare_nvfp4_projection_operand,
    b300_prepare_nvfp4_projection_weight,
    b300_project_nvfp4,
    b300_project_qkv_gqa_d128_unified_lowp_nvfp4,
    b300_stack_gqa_d128_qkv_projection_weights,
)
from tk_fa4.lowp_fa4_bwd.profile_gqa_d128_chain import (
    _bf16_gqa_attention_reference,
    _load_extension,
    _make_rope,
    _metrics,
    _time_cuda,
)


def _topology(module: Any) -> dict[str, Any]:
    return dict(module.read_hao_direct_topology())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--sequence", type=int, default=4096)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument(
        "--new-extension",
        type=Path,
        default=Path(
            "/tmp/_C_tk_gb200_causal_s4096_h32_d128." "cpython-312-aarch64-linux-gnu.so"
        ),
    )
    parser.add_argument("--new-module", default="_C_tk_gb200_causal_s4096_h32_d128")
    parser.add_argument(
        "--old-extension",
        type=Path,
        default=Path(
            "/tmp/_C_tk_causal_gqa_nvfp4_fp8pv_exact_builder_sm100."
            "cpython-312-aarch64-linux-gnu.so"
        ),
    )
    parser.add_argument(
        "--old-module",
        default="_C_tk_causal_gqa_nvfp4_fp8pv_exact_builder_sm100",
    )
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument(
        "--project-output",
        action="store_true",
        help="also propagate each attention output through an NVFP4 output projection",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.batch not in (1, 2):
        raise ValueError("D128 causal pair profiling supports batch 1 or 2")

    if torch.cuda.device_count() != 1:
        raise RuntimeError("profiling requires exactly one visible GPU")
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)

    new = _load_extension(args.new_extension, args.new_module)
    old = _load_extension(args.old_extension, args.old_module)
    new_topology = _topology(new)
    old_topology = _topology(old)
    expected = {
        "batch": args.batch,
        "heads": args.q_heads,
        "kv_heads": args.kv_heads,
        "seqlen": args.sequence,
        "dqk": 128,
        "dvo": 128,
        "causal": True,
    }
    for label, topology in (("new", new_topology), ("old", old_topology)):
        for name, value in expected.items():
            if topology[name] != value:
                raise ValueError(f"{label} topology {name}={topology[name]} != {value}")

    depth = 128
    rows = args.batch * args.sequence
    x = (torch.randn(rows, args.hidden, device="cuda") * 0.1).bfloat16()
    q_weight_raw = (
        torch.randn(args.q_heads * depth, args.hidden, device="cuda") * 0.02
    ).bfloat16()
    k_weight_raw = (
        torch.randn(args.kv_heads * depth, args.hidden, device="cuda") * 0.02
    ).bfloat16()
    v_weight = torch.randn_like(k_weight_raw.float()).mul_(0.02).bfloat16()
    q_weight, k_weight = b300_pair_interleave_gqa_d128_qk_projection_weights(
        q_weight_raw,
        k_weight_raw,
    )
    qkv_weight = b300_stack_gqa_d128_qkv_projection_weights(
        q_weight,
        k_weight,
        v_weight,
    )
    x_operand = tuple(b300_prepare_nvfp4_projection_operand(x))
    weight_operand = tuple(b300_prepare_nvfp4_projection_weight(qkv_weight))
    qk_scales = torch.zeros(
        args.batch,
        args.q_heads,
        7,
        device="cuda",
        dtype=torch.float32,
    )
    qk_scales[:, :, :2] = 16.0
    rope_cos, rope_sin = _make_rope(args.sequence)
    rope_cos = rope_cos.expand(args.batch, -1, -1).contiguous()
    rope_sin = rope_sin.expand(args.batch, -1, -1).contiguous()
    rope_packed = b300_pack_gqa_d128_rope(rope_cos, rope_sin)
    qkv = b300_project_qkv_gqa_d128_unified_lowp_nvfp4(
        x_operand,
        weight_operand,
        qk_scales,
        batch=args.batch,
        seqlen=args.sequence,
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
        store_bf16=True,
        publish_fp8_backward=True,
        rope_packed=rope_packed,
    )
    assert qkv.q is not None and qkv.k is not None and qkv.v is not None
    assert qkv.v_backward_fp8 is not None
    operands = qkv.forward_operands()
    old_v = qkv.v_backward_fp8.permute(0, 2, 3, 1).contiguous()
    output_shape = (args.batch, args.sequence, args.q_heads, depth)
    lse_shape = (args.batch, args.q_heads, 1, args.sequence)
    new_output = torch.empty(output_shape, device="cuda", dtype=torch.bfloat16)
    old_output = torch.empty_like(new_output)
    new_lse = torch.empty(lse_shape, device="cuda", dtype=torch.float32)
    old_lse = torch.empty_like(new_lse)

    def run_new() -> None:
        os.environ["TK_FA4_FP4PV_FWD_CONFIG"] = str(new_topology["route"])
        new.forward_hao_direct_fp4pv(*operands, new_output, new_lse, 0, True, True)

    def run_old() -> None:
        os.environ["TK_FA4_FP4PV_FWD_CONFIG"] = str(old_topology["route"])
        old.forward_hao_direct_fp8pv(
            *operands[:6], old_v, old_output, old_lse, 0, True, True
        )

    run_new()
    run_old()
    torch.cuda.synchronize()
    reference_pairs = [
        _bf16_gqa_attention_reference(
            qkv.q[index : index + 1],
            qkv.k[index : index + 1],
            qkv.v[index : index + 1],
        )
        for index in range(args.batch)
    ]
    reference_output = torch.cat(
        [pair[0] for pair in reference_pairs],
        dim=0,
    )
    reference_lse = torch.cat(
        [pair[1] for pair in reference_pairs],
        dim=0,
    )
    reference_lse = reference_lse.permute(0, 1, 3, 2).contiguous()
    result = {
        "shape": {
            "batch": args.batch,
            "sequence": args.sequence,
            "q_heads": args.q_heads,
            "kv_heads": args.kv_heads,
            "depth": depth,
            "hidden": args.hidden,
        },
        "new": {
            "topology": new_topology,
            "timing": _time_cuda(run_new, warmups=args.warmups, samples=args.samples),
            "output_vs_bf16": _metrics(reference_output, new_output),
            "lse_vs_bf16": _metrics(reference_lse, new_lse),
        },
        "old": {
            "topology": old_topology,
            "timing": _time_cuda(run_old, warmups=args.warmups, samples=args.samples),
            "output_vs_bf16": _metrics(reference_output, old_output),
            "lse_vs_bf16": _metrics(reference_lse, old_lse),
        },
        "new_vs_old": {
            "output": _metrics(old_output, new_output),
            "lse": _metrics(old_lse, new_lse),
        },
    }
    result["speedup_new_over_old"] = (
        result["old"]["timing"]["median_us"] / result["new"]["timing"]["median_us"]
    )
    if args.project_output:
        output_weight = (
            torch.randn(args.hidden, args.q_heads * depth, device="cuda") * 0.02
        ).bfloat16()
        weight_operand = tuple(b300_prepare_nvfp4_projection_weight(output_weight))
        reference_projected = torch.mm(
            reference_output.reshape(rows, -1),
            output_weight.T,
        )

        def project_output(output: torch.Tensor) -> torch.Tensor:
            operand = tuple(
                b300_prepare_nvfp4_projection_operand(output.reshape(rows, -1))
            )
            return b300_project_nvfp4(operand, weight_operand)

        new_projected = project_output(new_output)
        old_projected = project_output(old_output)
        result["through_output_projection"] = {
            "new_vs_bf16": _metrics(reference_projected, new_projected),
            "old_vs_bf16": _metrics(reference_projected, old_projected),
            "new_vs_old": _metrics(old_projected, new_projected),
            "new_materialized_timing": _time_cuda(
                lambda: project_output(new_output),
                warmups=args.warmups,
                samples=args.samples,
            ),
            "old_materialized_timing": _time_cuda(
                lambda: project_output(old_output),
                warmups=args.warmups,
                samples=args.samples,
            ),
        }
    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
