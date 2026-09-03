#!/usr/bin/env python3
"""Validate pair-native RoPE in the unified NVFP4 QKV epilogue."""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable

import torch

from tk_fa4 import (
    _C_b300_lowp_bwd as lowp,
    b300_inverse_rope_interleaved_qkv_grad_,
    b300_pair_interleave_qk_projection_weights,
    b300_prepare_nvfp4_projection_operand,
    b300_prepare_nvfp4_projection_weight,
    b300_project_qkv_unified_lowp_nvfp4,
    b300_rope_pair_qk_,
    b300_stack_qkv_projection_weights,
)


QK_DIM = 192
V_DIM = 128
HEAD_WIDTH = QK_DIM * 2 + V_DIM


def metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    reference_f = reference.float()
    actual_f = actual.float()
    difference = actual_f - reference_f
    return {
        "cosine": float(
            torch.sum(reference_f * actual_f)
            / (
                torch.linalg.vector_norm(reference_f).clamp_min(1.0e-20)
                * torch.linalg.vector_norm(actual_f).clamp_min(1.0e-20)
            )
        ),
        "relative_l2": float(
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(reference_f).clamp_min(1.0e-20)
        ),
        "max_abs": float(difference.abs().max()),
    }


def pair_interleave(tensor: torch.Tensor) -> torch.Tensor:
    first, second = tensor.chunk(2, dim=-1)
    return torch.stack((first, second), dim=-1).flatten(-2)


def rotate_pairs(
    tensor: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    pairs = tensor.float().reshape(*tensor.shape[:-1], QK_DIM // 2, 2)
    x = pairs[..., 0]
    y = pairs[..., 1]
    cosine_f = cosine.float().unsqueeze(2)
    sine_f = sine.float().unsqueeze(2)
    return torch.stack(
        (
            x * cosine_f - y * sine_f,
            y * cosine_f + x * sine_f,
        ),
        dim=-1,
    ).flatten(-2).bfloat16()


def make_rope(sequence: int) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(sequence, device="cuda", dtype=torch.float32)
    frequencies = 1.0 / (
        10_000.0
        ** (
            torch.arange(QK_DIM // 2, device="cuda", dtype=torch.float32)
            / (QK_DIM // 2)
        )
    )
    angles = positions[:, None] * frequencies[None, :]
    return angles.cos()[None].bfloat16(), angles.sin()[None].bfloat16()


def time_candidates(
    candidates: dict[str, Callable[[], object]],
    *,
    warmups: int,
    samples: int,
) -> dict[str, float]:
    names = list(candidates)
    for iteration in range(warmups):
        for offset in range(len(names)):
            candidates[names[(iteration + offset) % len(names)]]()
    torch.cuda.synchronize()
    values: dict[str, list[float]] = {name: [] for name in names}
    for iteration in range(samples):
        for offset in range(len(names)):
            name = names[(iteration + offset) % len(names)]
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            candidates[name]()
            end.record()
            end.synchronize()
            values[name].append(float(start.elapsed_time(end)))
    return {name: statistics.median(times) for name, times in values.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=9)
    args = parser.parse_args()

    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    x = (
        torch.randn(args.sequence, args.hidden, device="cuda") * 0.1
    ).bfloat16()
    q_weight = (
        torch.randn(args.heads * QK_DIM, args.hidden, device="cuda") * 0.02
    ).bfloat16()
    k_weight = torch.randn_like(q_weight.float()).mul_(0.02).bfloat16()
    v_weight = (
        torch.randn(args.heads * V_DIM, args.hidden, device="cuda") * 0.02
    ).bfloat16()
    q_pair_weight, k_pair_weight = b300_pair_interleave_qk_projection_weights(
        q_weight,
        k_weight,
    )
    stacked_weight = b300_stack_qkv_projection_weights(
        q_pair_weight,
        k_pair_weight,
        v_weight,
    )
    cosine, sine = make_rope(args.sequence)

    q_standard = torch.mm(x, q_weight.T).reshape(
        1, args.sequence, args.heads, QK_DIM
    )
    k_standard = torch.mm(x, k_weight.T).reshape_as(q_standard)
    q_pair_reference = rotate_pairs(pair_interleave(q_standard), cosine, sine)
    k_pair_reference = rotate_pairs(pair_interleave(k_standard), cosine, sine)
    adaptive_scales = lowp.quantize_fp4_dual_qk_adaptive(
        q_pair_reference,
        k_pair_reference,
        16.0,
        2.0**-12,
        0.325,
        2.75,
        float(QK_DIM**-0.5),
        4096.0,
    )[4]
    x_operand = tuple(b300_prepare_nvfp4_projection_operand(x))
    weight_operand = tuple(
        b300_prepare_nvfp4_projection_weight(stacked_weight)
    )
    unrotated = b300_project_qkv_unified_lowp_nvfp4(
        x_operand,
        weight_operand,
        adaptive_scales,
        batch=1,
        seqlen=args.sequence,
        heads=args.heads,
    )
    fused = b300_project_qkv_unified_lowp_nvfp4(
        x_operand,
        weight_operand,
        adaptive_scales,
        batch=1,
        seqlen=args.sequence,
        heads=args.heads,
        rope_cos=cosine,
        rope_sin=sine,
    )
    assert unrotated.q is not None and unrotated.k is not None
    assert unrotated.v is not None
    assert fused.q is not None and fused.k is not None and fused.v is not None
    expected_q = rotate_pairs(unrotated.q, cosine, sine)
    expected_k = rotate_pairs(unrotated.k, cosine, sine)
    standalone_q = unrotated.q.clone()
    standalone_k = unrotated.k.clone()
    b300_rope_pair_qk_(standalone_q, standalone_k, cosine, sine)
    standalone = lowp.quantize_fp4_dual_qk_precomputed_scales(
        fused.q,
        fused.k,
        adaptive_scales,
    )
    fused_layouts = (
        fused.backward.q_fp4,
        fused.backward.score_q_fp4,
        fused.backward.k_fp4,
        fused.backward.score_k_fp4,
    )

    raw_gradient = (
        torch.randn(
            1,
            args.sequence,
            args.heads,
            HEAD_WIDTH,
            device="cuda",
        )
        * 0.1
    ).bfloat16()
    rotated_gradient = raw_gradient.clone()
    rotated_gradient[..., :QK_DIM] = rotate_pairs(
        raw_gradient[..., :QK_DIM], cosine, sine
    )
    rotated_gradient[..., QK_DIM : 2 * QK_DIM] = rotate_pairs(
        raw_gradient[..., QK_DIM : 2 * QK_DIM], cosine, sine
    )
    b300_inverse_rope_interleaved_qkv_grad_(
        rotated_gradient,
        cosine,
        sine,
    )

    q_scratch = unrotated.q.clone()
    k_scratch = unrotated.k.clone()
    gradient_scratch = rotated_gradient.clone()

    def project_unrotated() -> object:
        return b300_project_qkv_unified_lowp_nvfp4(
            x_operand,
            weight_operand,
            adaptive_scales,
            batch=1,
            seqlen=args.sequence,
            heads=args.heads,
        )

    def project_fused() -> object:
        return b300_project_qkv_unified_lowp_nvfp4(
            x_operand,
            weight_operand,
            adaptive_scales,
            batch=1,
            seqlen=args.sequence,
            heads=args.heads,
            rope_cos=cosine,
            rope_sin=sine,
        )

    def project_then_rope() -> object:
        projected = project_unrotated()
        assert projected.q is not None and projected.k is not None
        return b300_rope_pair_qk_(
            projected.q,
            projected.k,
            cosine,
            sine,
        )

    timing = time_candidates(
        {
            "projection_unrotated": project_unrotated,
            "projection_fused_rope": project_fused,
            "projection_then_standalone_rope": project_then_rope,
            "standalone_rope_only": lambda: b300_rope_pair_qk_(
                q_scratch,
                k_scratch,
                cosine,
                sine,
            ),
            "inverse_qkv_gradient_only": lambda: (
                b300_inverse_rope_interleaved_qkv_grad_(
                    gradient_scratch,
                    cosine,
                    sine,
                )
            ),
        },
        warmups=args.warmups,
        samples=args.samples,
    )

    result = {
        "shape": {
            "batch": 1,
            "sequence": args.sequence,
            "heads": args.heads,
            "hidden": args.hidden,
        },
        "fused_vs_epilogue_reference": {
            "q": metrics(expected_q, fused.q),
            "k": metrics(expected_k, fused.k),
            "v_bitwise_equal": bool(torch.equal(unrotated.v, fused.v)),
        },
        "standalone_kernel_vs_epilogue_reference": {
            "q": metrics(expected_q, standalone_q),
            "k": metrics(expected_k, standalone_k),
        },
        "fused_vs_dense_standard_rope": {
            "q": metrics(q_pair_reference, fused.q),
            "k": metrics(k_pair_reference, fused.k),
        },
        "lowp_layouts_match_standalone": [
            bool(torch.equal(produced, reference))
            for produced, reference in zip(
                fused_layouts,
                standalone[:4],
                strict=True,
            )
        ],
        "inverse_gradient_roundtrip": {
            "q": metrics(
                raw_gradient[..., :QK_DIM],
                rotated_gradient[..., :QK_DIM],
            ),
            "k": metrics(
                raw_gradient[..., QK_DIM : 2 * QK_DIM],
                rotated_gradient[..., QK_DIM : 2 * QK_DIM],
            ),
            "v_bitwise_equal": bool(
                torch.equal(
                    raw_gradient[..., 2 * QK_DIM :],
                    rotated_gradient[..., 2 * QK_DIM :],
                )
            ),
        },
        "timing_median_ms": timing,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
