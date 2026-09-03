#!/usr/bin/env python3
"""Validate inverse-RoPE fusion in the delayed-scale NVFP4 handoff."""

from __future__ import annotations

import argparse
import json

import torch

from tk_fa4.interface import (
    b300_inverse_rope_interleaved_qkv_grad_,
    b300_prepare_nvfp4_projection_operand,
    b300_prepare_nvfp4_projection_operand_inverse_rope,
)
from tk_fa4.lowp_fa4_bwd.evaluate_llama_attention_e2e import (
    QKV_HEAD_WIDTH,
    _make_rope_tables,
    time_rotated,
)


def metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    reference_f = reference.float().reshape(-1)
    actual_f = actual.float().reshape(-1)
    difference = actual_f - reference_f
    reference_norm = torch.linalg.vector_norm(reference_f).clamp_min(1.0e-20)
    return {
        "cosine": float(
            torch.dot(reference_f, actual_f)
            / (
                reference_norm
                * torch.linalg.vector_norm(actual_f).clamp_min(1.0e-20)
            )
        ),
        "relative_l2": float(
            torch.linalg.vector_norm(difference) / reference_norm
        ),
        "max_abs": float(difference.abs().max()),
    }


def byte_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return bool(
        torch.equal(
            left.contiguous().view(torch.uint8),
            right.contiguous().view(torch.uint8),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2026081321)
    parser.add_argument("--warmups", type=int, default=4)
    parser.add_argument("--samples", type=int, default=13)
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("validation requires exactly one visible GPU")
    torch.cuda.set_device(0)
    if args.sequence % 128:
        raise ValueError("sequence must be divisible by 128")

    torch.manual_seed(args.seed)
    raw = (
        torch.randn(
            1,
            args.sequence,
            args.heads,
            QKV_HEAD_WIDTH,
            device="cuda",
        )
        * 0.01
    ).bfloat16()
    rope_cos, rope_sin = _make_rope_tables(args.sequence)

    reference = raw.clone()
    b300_inverse_rope_interleaved_qkv_grad_(
        reference,
        rope_cos,
        rope_sin,
    )
    reference_matrix = reference.reshape(args.sequence, -1)
    global_scale = b300_prepare_nvfp4_projection_operand(reference_matrix)[2]
    reference_operand = b300_prepare_nvfp4_projection_operand(
        reference_matrix,
        global_scale=global_scale,
    )

    no_publish_input = raw.clone()
    no_publish_operand = b300_prepare_nvfp4_projection_operand_inverse_rope(
        no_publish_input.reshape(args.sequence, -1),
        rope_cos,
        rope_sin,
        global_scale=global_scale,
        publish_inverse_bf16=False,
    )
    publish_input = raw.clone()
    publish_operand = b300_prepare_nvfp4_projection_operand_inverse_rope(
        publish_input.reshape(args.sequence, -1),
        rope_cos,
        rope_sin,
        global_scale=global_scale,
        publish_inverse_bf16=True,
    )
    torch.cuda.synchronize()

    no_publish_equal = [
        byte_equal(actual, expected)
        for actual, expected in zip(
            no_publish_operand[:2],
            reference_operand[:2],
            strict=True,
        )
    ]
    publish_equal = [
        byte_equal(actual, expected)
        for actual, expected in zip(
            publish_operand[:2],
            reference_operand[:2],
            strict=True,
        )
    ]

    separate_scratch = raw.clone()
    publish_scratch = raw.clone()

    def separate_inverse_then_pack() -> object:
        b300_inverse_rope_interleaved_qkv_grad_(
            separate_scratch,
            rope_cos,
            rope_sin,
        )
        return b300_prepare_nvfp4_projection_operand(
            separate_scratch.reshape(args.sequence, -1),
            global_scale=global_scale,
        )

    def fused_no_publish() -> object:
        return b300_prepare_nvfp4_projection_operand_inverse_rope(
            raw.reshape(args.sequence, -1),
            rope_cos,
            rope_sin,
            global_scale=global_scale,
            publish_inverse_bf16=False,
        )

    def fused_publish() -> object:
        return b300_prepare_nvfp4_projection_operand_inverse_rope(
            publish_scratch.reshape(args.sequence, -1),
            rope_cos,
            rope_sin,
            global_scale=global_scale,
            publish_inverse_bf16=True,
        )

    timing = time_rotated(
        {
            "inverse_only": lambda: b300_inverse_rope_interleaved_qkv_grad_(
                separate_scratch,
                rope_cos,
                rope_sin,
            ),
            "pack_only": lambda: b300_prepare_nvfp4_projection_operand(
                reference_matrix,
                global_scale=global_scale,
            ),
            "separate_inverse_then_pack": separate_inverse_then_pack,
            "fused_no_bf16_publication": fused_no_publish,
            "fused_with_bf16_publication": fused_publish,
        },
        warmups=args.warmups,
        samples=args.samples,
    )

    result = {
        "shape": {
            "batch": 1,
            "sequence": args.sequence,
            "heads": args.heads,
            "qk_head_dim": 192,
            "v_head_dim": 128,
        },
        "correctness": {
            "no_publish_operand_bitwise_equal": no_publish_equal,
            "publish_operand_bitwise_equal": publish_equal,
            "no_publish_input_bitwise_unchanged": bool(
                torch.equal(no_publish_input, raw)
            ),
            "published_bf16_bitwise_equal": bool(
                torch.equal(publish_input, reference)
            ),
            "published_bf16_metrics": metrics(reference, publish_input),
        },
        "timing": timing,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
