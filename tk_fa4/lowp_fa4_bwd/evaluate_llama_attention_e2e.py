#!/usr/bin/env python3
"""Evaluate a projected causal decoder-attention forward/backward chain.

This is an integration harness, not a claim of drop-in stock-Llama support.
The native TK kernels use equal Q/K/V head counts, QK depth 192, and V depth
128.  The timed block contains learned Q/K/V projections, optional full-head
RoPE, causal attention, output projection, output-projection dgrad, attention
backward, inverse RoPE, and QKV projection dgrad.  An optional training scope
also includes QKV and output projection weight gradients.

The retained FP4 forward implementation is consumed through its existing
Python entry point and remains source-read-only.  Q/K/V projection weights
and NVFP4 operands are cached outside the timed region.  Results distinguish
an upstream-native contract (X and dY are already packed) from a materialized
contract that charges their standalone NVFP4 packing launches.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from tk_fa4 import _C_b300_lowp_bwd as lowp
from tk_fa4.interface import (
    B300AdaptiveLowpOperands,
    B300UnifiedLowpQKV,
    b300_interleave_qkv_projection_weights,
    b300_inverse_rope_interleaved_qkv_grad_,
    b300_mha_bwd_adaptive_lowp,
    b300_mha_bwd_adaptive_lowp_hierarchical_nvfp4_qkv_projection_dgrad,
    b300_mha_bwd_adaptive_lowp_stacked_nvfp4_qkv_projection_dgrad,
    b300_mha_fwd,
    b300_pair_interleave_qk_projection_weights,
    b300_prepare_nvfp4_projection_operand,
    b300_prepare_nvfp4_projection_operand_inverse_rope,
    b300_project_dout_unified_lowp_nvfp4,
    b300_project_nvfp4,
    b300_project_qkv_unified_lowp_nvfp4,
    b300_rope_pair_qk_,
    b300_stack_qkv_projection_weights,
)
from tk_fa4.fp4_pv_experiments import _run_forward_streaming_live_mxfp4


QK_DIM = 192
V_DIM = 128
QKV_HEAD_WIDTH = QK_DIM * 2 + V_DIM


@dataclass(frozen=True)
class Problem:
    sequence: int
    heads: int
    hidden: int
    x: torch.Tensor
    dy: torch.Tensor
    q_weight: torch.Tensor
    k_weight: torch.Tensor
    q_native_weight: torch.Tensor
    k_native_weight: torch.Tensor
    v_weight: torch.Tensor
    qkv_weight_stacked: torch.Tensor
    qkv_weight_interleaved: torch.Tensor
    out_weight: torch.Tensor
    qk_scales: torch.Tensor
    rope_cos: torch.Tensor | None
    rope_sin: torch.Tensor | None
    x_operand: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    dy_operand: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    qkv_weight_operand: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    out_forward_weight_operand: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    out_backward_weight_operand: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    qkv_backward_weight_operand: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    qkv_backward_stacked_weight_operand: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]

    @property
    def rows(self) -> int:
        return self.sequence

    @property
    def q_width(self) -> int:
        return self.heads * QK_DIM

    @property
    def v_width(self) -> int:
        return self.heads * V_DIM

    @property
    def qkv_width(self) -> int:
        return self.heads * QKV_HEAD_WIDTH


def parse_shapes(value: str) -> list[tuple[int, int, int]]:
    shapes: list[tuple[int, int, int]] = []
    for raw in value.split(","):
        fields = raw.lower().split("x")
        if len(fields) != 3:
            raise ValueError("shapes must use sequence x heads x hidden")
        sequence, heads, hidden = map(int, fields)
        shapes.append((sequence, heads, hidden))
    return shapes


def _make_rope_tables(sequence: int) -> tuple[torch.Tensor, torch.Tensor]:
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


def _pair_interleave(tensor: torch.Tensor) -> torch.Tensor:
    first, second = tensor.chunk(2, dim=-1)
    return torch.stack((first, second), dim=-1).flatten(-2).contiguous()


def _apply_split_rope(
    tensor: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    first, second = tensor.float().chunk(2, dim=-1)
    cosine_f = cosine.float().unsqueeze(2)
    sine_f = sine.float().unsqueeze(2)
    return torch.cat(
        (
            first * cosine_f - second * sine_f,
            second * cosine_f + first * sine_f,
        ),
        dim=-1,
    ).bfloat16()


def _apply_pair_rope(
    tensor: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    pairs = tensor.float().reshape(*tensor.shape[:-1], QK_DIM // 2, 2)
    first = pairs[..., 0]
    second = pairs[..., 1]
    cosine_f = cosine.float().unsqueeze(2)
    sine_f = sine.float().unsqueeze(2)
    return torch.stack(
        (
            first * cosine_f - second * sine_f,
            second * cosine_f + first * sine_f,
        ),
        dim=-1,
    ).flatten(-2).bfloat16()


def _inverse_pair_rope(
    tensor: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    pairs = tensor.float().reshape(*tensor.shape[:-1], QK_DIM // 2, 2)
    first = pairs[..., 0]
    second = pairs[..., 1]
    cosine_f = cosine.float().unsqueeze(2)
    sine_f = sine.float().unsqueeze(2)
    return torch.stack(
        (
            first * cosine_f + second * sine_f,
            second * cosine_f - first * sine_f,
        ),
        dim=-1,
    ).flatten(-2).bfloat16()


def _inverse_split_rope(
    tensor: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    first, second = tensor.to(torch.bfloat16).float().chunk(2, dim=-1)
    cosine_f = cosine.float().unsqueeze(2)
    sine_f = sine.float().unsqueeze(2)
    return torch.cat(
        (
            first * cosine_f + second * sine_f,
            second * cosine_f - first * sine_f,
        ),
        dim=-1,
    ).bfloat16()


def metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, Any]:
    reference_f = reference.float()
    actual_f = actual.float()
    difference = actual_f - reference_f
    reference_norm = torch.linalg.vector_norm(reference_f).clamp_min(1.0e-20)
    actual_norm = torch.linalg.vector_norm(actual_f).clamp_min(1.0e-20)
    return {
        "finite": bool(torch.isfinite(actual_f).all()),
        "cosine": float(
            torch.sum(reference_f * actual_f)
            / (reference_norm * actual_norm)
        ),
        "relative_l2": float(
            torch.linalg.vector_norm(difference) / reference_norm
        ),
        "max_abs": float(difference.abs().max()),
        "norm_ratio": float(actual_norm / reference_norm),
    }


def time_rotated(
    candidates: dict[str, Callable[[], object]],
    *,
    warmups: int,
    samples: int,
) -> dict[str, dict[str, Any]]:
    names = list(candidates)
    for iteration in range(warmups):
        offset = iteration % len(names)
        for name in names[offset:] + names[:offset]:
            candidates[name]()
    torch.cuda.synchronize()

    elapsed: dict[str, list[float]] = {name: [] for name in names}
    for iteration in range(samples):
        offset = iteration % len(names)
        for name in names[offset:] + names[:offset]:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            candidates[name]()
            end.record()
            end.synchronize()
            elapsed[name].append(float(start.elapsed_time(end)))
    return {
        name: {
            "median_ms": statistics.median(values),
            "mean_ms": statistics.fmean(values),
            "minimum_ms": min(values),
            "maximum_ms": max(values),
            "samples_ms": values,
        }
        for name, values in elapsed.items()
    }


def _split_stacked_qkv(
    qkv: torch.Tensor,
    problem: Problem,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_end = problem.q_width
    k_end = q_end + problem.q_width
    q = qkv[:, :q_end].reshape(
        1, problem.sequence, problem.heads, QK_DIM
    ).contiguous()
    k = qkv[:, q_end:k_end].reshape_as(q).contiguous()
    v = qkv[:, k_end:].reshape(
        1, problem.sequence, problem.heads, V_DIM
    ).contiguous()
    return q, k, v


def _interleave_gradients(
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
) -> torch.Tensor:
    batch, sequence, heads, _ = dq.shape
    # The attention reference may retain dQ in FP32 accumulation storage,
    # while a learned projection consumes the rounded activation gradient.
    # Normalize every route at that real BF16 projection boundary.
    dq = dq.to(torch.bfloat16)
    dk = dk.to(torch.bfloat16)
    dv = dv.to(torch.bfloat16)
    return torch.cat(
        (
            dq.reshape(batch, sequence, heads, QK_DIM),
            dk.reshape(batch, sequence, heads, QK_DIM),
            dv.reshape(batch, sequence, heads, V_DIM),
        ),
        dim=-1,
    ).contiguous()


def _split_interleaved_gradients(
    qkv_grad: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        qkv_grad[..., :QK_DIM],
        qkv_grad[..., QK_DIM : QK_DIM * 2],
        qkv_grad[..., QK_DIM * 2 :],
    )


def _project_interleaved_bf16(
    qkv_grad: torch.Tensor,
    weight: torch.Tensor,
    *,
    splits: int,
) -> torch.Tensor:
    projection_input = qkv_grad.reshape(-1, qkv_grad.shape[2] * QKV_HEAD_WIDTH)
    split_width = projection_input.shape[1] // splits
    dx = torch.mm(
        projection_input[:, :split_width],
        weight[:split_width],
    )
    for split in range(1, splits):
        start = split * split_width
        end = start + split_width
        dx.addmm_(projection_input[:, start:end], weight[start:end])
    return dx


def _projection_splits(sequence: int, heads: int) -> int:
    if heads == 64:
        return 4
    if heads == 16 or (heads == 24 and sequence >= 8192):
        return 2
    return 1


def _algorithmic_matmul_flops(
    sequence: int,
    heads: int,
    hidden: int,
    *,
    weight_gradients: bool,
) -> int:
    """Count useful dense/causal matmul FLOPs at the model boundary.

    Causal attention counts the lower triangle.  Elementwise softmax, RoPE,
    packing, and reductions are deliberately excluded, making this a stable
    algorithmic-throughput numerator shared by BF16 and low-precision routes.
    """
    projection_activation = 4 * sequence * hidden * heads * (
        QKV_HEAD_WIDTH + V_DIM
    )
    attention_forward_backward = 3 * heads * sequence * sequence * (
        QK_DIM + V_DIM
    )
    projection_weight = (
        2 * sequence * hidden * heads * (QKV_HEAD_WIDTH + V_DIM)
        if weight_gradients
        else 0
    )
    return projection_activation + attention_forward_backward + projection_weight


def build_problem(
    sequence: int,
    heads: int,
    hidden: int,
    seed: int,
    *,
    use_rope: bool = False,
) -> Problem:
    if sequence % 128:
        raise ValueError("sequence must be divisible by 128")
    if hidden % 128:
        raise ValueError("hidden must be divisible by 128")
    supported_heads = {
        4096: {24, 64},
        8192: {8, 16, 24, 64},
        16384: {24, 64},
    }.get(sequence, set())
    if heads not in supported_heads:
        raise ValueError(
            f"unsupported native backward shape S{sequence}/H{heads}"
        )
    torch.manual_seed(seed)
    x = (torch.randn(sequence, hidden, device="cuda") * 0.1).bfloat16()
    dy = (torch.randn_like(x.float()) * 0.1).bfloat16()
    q_weight = (
        torch.randn(heads * QK_DIM, hidden, device="cuda") * 0.02
    ).bfloat16()
    k_weight = (
        torch.randn_like(q_weight.float()) * 0.02
    ).bfloat16()
    v_weight = (
        torch.randn(heads * V_DIM, hidden, device="cuda") * 0.02
    ).bfloat16()
    out_weight = (
        torch.randn(hidden, heads * V_DIM, device="cuda") * 0.02
    ).bfloat16()
    if use_rope:
        q_native_weight, k_native_weight = (
            b300_pair_interleave_qk_projection_weights(q_weight, k_weight)
        )
        rope_cos, rope_sin = _make_rope_tables(sequence)
    else:
        q_native_weight = q_weight
        k_native_weight = k_weight
        rope_cos = None
        rope_sin = None
    qkv_weight_stacked = b300_stack_qkv_projection_weights(
        q_native_weight,
        k_native_weight,
        v_weight,
    )
    qkv_weight_interleaved = b300_interleave_qkv_projection_weights(
        q_native_weight,
        k_native_weight,
        v_weight,
    )

    # Calibrate the seven-word adaptive record from an independent prior
    # batch.  In a model this record comes from delayed/device-resident scale
    # state or another upstream producer; it is deliberately outside the
    # current-batch projection hot path.
    torch.manual_seed(seed + 1_000_003)
    calibration_x = (
        torch.randn(sequence, hidden, device="cuda") * 0.1
    ).bfloat16()
    q_reference = torch.mm(calibration_x, q_native_weight.T).reshape(
        1, sequence, heads, QK_DIM
    ).contiguous()
    k_reference = torch.mm(
        calibration_x, k_native_weight.T
    ).reshape_as(q_reference)
    if use_rope:
        assert rope_cos is not None and rope_sin is not None
        q_reference = _apply_pair_rope(q_reference, rope_cos, rope_sin)
        k_reference = _apply_pair_rope(k_reference, rope_cos, rope_sin)
    adaptive = lowp.quantize_fp4_dual_qk_adaptive(
        q_reference,
        k_reference,
        16.0,
        2.0**-12,
        0.325,
        2.75,
        float(QK_DIM**-0.5),
        4096.0,
    )
    qk_scales = adaptive[4]

    return Problem(
        sequence=sequence,
        heads=heads,
        hidden=hidden,
        x=x,
        dy=dy,
        q_weight=q_weight,
        k_weight=k_weight,
        q_native_weight=q_native_weight,
        k_native_weight=k_native_weight,
        v_weight=v_weight,
        qkv_weight_stacked=qkv_weight_stacked,
        qkv_weight_interleaved=qkv_weight_interleaved,
        out_weight=out_weight,
        qk_scales=qk_scales,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        x_operand=tuple(b300_prepare_nvfp4_projection_operand(x)),
        dy_operand=tuple(b300_prepare_nvfp4_projection_operand(dy)),
        qkv_weight_operand=tuple(
            b300_prepare_nvfp4_projection_operand(qkv_weight_stacked)
        ),
        out_forward_weight_operand=tuple(
            b300_prepare_nvfp4_projection_operand(out_weight)
        ),
        out_backward_weight_operand=tuple(
            b300_prepare_nvfp4_projection_operand(out_weight.T.contiguous())
        ),
        qkv_backward_weight_operand=tuple(
            b300_prepare_nvfp4_projection_operand(
                qkv_weight_interleaved.T.contiguous()
            )
        ),
        qkv_backward_stacked_weight_operand=tuple(
            b300_prepare_nvfp4_projection_operand(
                qkv_weight_stacked.T.contiguous()
            )
        ),
    )


def _project_qkv_lowp(
    problem: Problem,
    *,
    charge_input_pack: bool,
    input_global_scale: torch.Tensor | None = None,
    publish_fp8_backward: bool = False,
) -> B300UnifiedLowpQKV:
    input_operand = (
        tuple(
            b300_prepare_nvfp4_projection_operand(
                problem.x,
                global_scale=input_global_scale,
            )
        )
        if charge_input_pack
        else problem.x_operand
    )
    return b300_project_qkv_unified_lowp_nvfp4(
        input_operand,
        problem.qkv_weight_operand,
        problem.qk_scales,
        batch=1,
        seqlen=problem.sequence,
        heads=problem.heads,
        store_bf16=True,
        publish_fp8_backward=publish_fp8_backward,
        rope_cos=problem.rope_cos,
        rope_sin=problem.rope_sin,
    )


def run_bf16(
    problem: Problem,
    *,
    weight_gradients: bool,
) -> dict[str, torch.Tensor]:
    # A stacked GEMM followed by three layout copies is substantially slower
    # at these widths.  Use the stronger Llama-style control: three learned
    # projections write their consumer-contiguous Q/K/V matrices directly.
    q = torch.mm(problem.x, problem.q_native_weight.T).reshape(
        1, problem.sequence, problem.heads, QK_DIM
    )
    k = torch.mm(problem.x, problem.k_native_weight.T).reshape_as(q)
    if problem.rope_cos is not None:
        assert problem.rope_sin is not None
        q, k = b300_rope_pair_qk_(
            q,
            k,
            problem.rope_cos,
            problem.rope_sin,
        )
    v = torch.mm(problem.x, problem.v_weight.T).reshape(
        1, problem.sequence, problem.heads, V_DIM
    )
    out, lse = b300_mha_fwd(q, k, v, causal=True, return_lse=True)
    out_matrix = out.reshape(problem.rows, problem.v_width)
    y = torch.mm(out_matrix, problem.out_weight.T)
    dout = torch.mm(problem.dy, problem.out_weight).reshape_as(out)
    dq, dk, dv = lowp.backward_bf16_control(
        q,
        k,
        v,
        out,
        lse,
        dout,
        True,
        float(QK_DIM**-0.5),
        False,
    )
    qkv_grad = _interleave_gradients(dq, dk, dv)
    if problem.rope_cos is not None:
        assert problem.rope_sin is not None
        b300_inverse_rope_interleaved_qkv_grad_(
            qkv_grad,
            problem.rope_cos,
            problem.rope_sin,
        )
    dx = _project_interleaved_bf16(
        qkv_grad,
        problem.qkv_weight_interleaved,
        splits=_projection_splits(problem.sequence, problem.heads),
    )
    result = {
        "y": y,
        "dx": dx,
        "out": out,
        "dout": dout,
        "qkv_grad": qkv_grad,
    }
    if weight_gradients:
        result["dqkv_weight"] = torch.mm(
            qkv_grad.reshape(problem.rows, problem.qkv_width).T,
            problem.x,
        )
        result["dout_weight"] = torch.mm(problem.dy.T, out_matrix)
    return result


def _lowp_dense_projection(
    input_matrix: torch.Tensor,
    cached_operand: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    weight_operand: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    global_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    input_operand = (
        cached_operand
        if cached_operand is not None
        else tuple(
            b300_prepare_nvfp4_projection_operand(
                input_matrix,
                global_scale=global_scale,
            )
        )
    )
    return b300_project_nvfp4(input_operand, weight_operand)


def run_fixed(
    problem: Problem,
    *,
    charge_input_pack: bool,
    lowp_dense_projections: bool,
    cache_dy_operand: bool,
    weight_gradients: bool,
    x_global_scale: torch.Tensor | None = None,
    dy_global_scale: torch.Tensor | None = None,
    out_global_scale: torch.Tensor | None = None,
    qkv_grad_global_scale: torch.Tensor | None = None,
    fuse_inverse_rope_handoff: bool = False,
    producer_native_fp8_backward: bool = False,
    producer_native_fp8_v_only: bool = False,
    producer_native_fp8_dout_v: bool = False,
    producer_native_fp8_dout_stats_from_packed: bool = True,
    hierarchical_qkv_projection: bool = False,
    stacked_qkv_projection: bool = False,
    bf16_attention_forward: bool = False,
    bf16_qkv_projection: bool = False,
) -> dict[str, torch.Tensor]:
    producer_modes = sum((
        producer_native_fp8_backward,
        producer_native_fp8_v_only,
        producer_native_fp8_dout_v,
    ))
    if producer_modes > 1:
        raise ValueError("FP8 producer modes are mutually exclusive")
    if hierarchical_qkv_projection and stacked_qkv_projection:
        raise ValueError("QKV projection-native modes are mutually exclusive")
    projection_native_qkv = (
        hierarchical_qkv_projection or stacked_qkv_projection
    )
    if projection_native_qkv and not (
        lowp_dense_projections
        and fuse_inverse_rope_handoff
        and producer_native_fp8_dout_v
        and problem.rope_cos is not None
        and qkv_grad_global_scale is not None
    ):
        raise ValueError(
            "hierarchical QKV projection requires the delayed-scale NVFP4 "
            "dense path, inverse RoPE, and producer-native FP8 dO/V"
        )
    if projection_native_qkv and weight_gradients:
        raise ValueError(
            "hierarchical QKV projection currently covers activation dgrad; "
            "the projection weight-gradient consumer is not integrated"
        )
    if bf16_qkv_projection:
        if not bf16_attention_forward:
            raise ValueError("BF16 QKV projection requires BF16 attention forward")
        if producer_modes:
            raise ValueError(
                "BF16 QKV projection does not publish producer-native FP8 V"
            )
        q = torch.mm(problem.x, problem.q_native_weight.T).reshape(
            1, problem.sequence, problem.heads, QK_DIM
        )
        k = torch.mm(problem.x, problem.k_native_weight.T).reshape_as(q)
        if problem.rope_cos is not None:
            assert problem.rope_sin is not None
            q, k = b300_rope_pair_qk_(
                q,
                k,
                problem.rope_cos,
                problem.rope_sin,
            )
        v = torch.mm(problem.x, problem.v_weight.T).reshape(
            1, problem.sequence, problem.heads, V_DIM
        )
        backward_operands = B300AdaptiveLowpOperands(
            *lowp.quantize_fp4_dual_qk_precomputed_scales(
                q,
                k,
                problem.qk_scales,
            )
        )
        bundle = None
    else:
        bundle = _project_qkv_lowp(
            problem,
            charge_input_pack=charge_input_pack,
            input_global_scale=x_global_scale,
            publish_fp8_backward=(
                producer_modes > 0
            ),
        )
        assert (
            bundle.q is not None
            and bundle.k is not None
            and bundle.v is not None
        )
        q, k, v = bundle.q, bundle.k, bundle.v
        backward_operands = bundle.backward
    if bf16_attention_forward:
        out, lse = b300_mha_fwd(
            q,
            k,
            v,
            causal=True,
            return_lse=True,
        )
    else:
        assert bundle is not None
        out, lse_bhs = _run_forward_streaming_live_mxfp4(
            *bundle.forward_operands()
        )
        lse = lse_bhs.permute(0, 2, 1).contiguous()
    out_matrix = out.reshape(problem.rows, problem.v_width)
    dout_bundle = None
    dout_quality = None
    if lowp_dense_projections:
        y = _lowp_dense_projection(
            out_matrix,
            None,
            problem.out_forward_weight_operand,
            global_scale=out_global_scale,
        )
        if producer_native_fp8_backward or producer_native_fp8_dout_v:
            dy_operand = (
                problem.dy_operand
                if cache_dy_operand
                else tuple(
                    b300_prepare_nvfp4_projection_operand(
                        problem.dy,
                        global_scale=dy_global_scale,
                    )
                )
            )
            dout_bundle = b300_project_dout_unified_lowp_nvfp4(
                dy_operand,
                problem.out_backward_weight_operand,
                out,
                lse,
                batch=1,
                seqlen=problem.sequence,
                heads=problem.heads,
                store_bf16=not (
                    producer_native_fp8_dout_v
                    and producer_native_fp8_dout_stats_from_packed
                ),
                publish_fp8_backward=True,
                publish_stats=producer_native_fp8_backward,
            )
            if (
                producer_native_fp8_dout_v
                and producer_native_fp8_dout_stats_from_packed
            ):
                assert dout_bundle.dout is None
                assert dout_bundle.dout_backward_fp8 is not None
                dout_matrix = dout_bundle.dout_storage.reshape(
                    problem.rows, problem.v_width
                )
                dout_quality = dout_bundle.dout_backward_fp8
            else:
                assert dout_bundle.dout is not None
                dout_matrix = dout_bundle.dout.reshape(
                    problem.rows, problem.v_width
                )
        else:
            dout_matrix = _lowp_dense_projection(
                problem.dy,
                problem.dy_operand if cache_dy_operand else None,
                problem.out_backward_weight_operand,
                global_scale=dy_global_scale,
            )
    else:
        y = torch.mm(out_matrix, problem.out_weight.T)
        dout_matrix = torch.mm(problem.dy, problem.out_weight)
    dout = dout_matrix.reshape(1, problem.sequence, problem.heads, V_DIM)
    if projection_native_qkv:
        assert bundle is not None
        assert dout_bundle is not None
        assert bundle.v_backward_fp8 is not None
        assert dout_bundle.dout_backward_fp8 is not None
        assert qkv_grad_global_scale is not None
        assert problem.rope_cos is not None
        assert problem.rope_sin is not None
        projection_backward = (
            b300_mha_bwd_adaptive_lowp_hierarchical_nvfp4_qkv_projection_dgrad
            if hierarchical_qkv_projection
            else b300_mha_bwd_adaptive_lowp_stacked_nvfp4_qkv_projection_dgrad
        )
        dx, dk_attention, dv_attention = (
            projection_backward(
                q,
                k,
                v,
                out,
                lse,
                dout,
                backward_operands,
                dout_bundle.dout_backward_fp8,
                bundle.v_backward_fp8,
                problem.qkv_backward_stacked_weight_operand,
                qkv_grad_global_scale,
                problem.rope_cos,
                problem.rope_sin,
                stats_from_packed_dout=(
                    producer_native_fp8_dout_stats_from_packed
                ),
                causal=True,
                softmax_scale=float(QK_DIM**-0.5),
                deterministic=False,
            )
        )
        return {
            "y": y,
            "dx": dx,
            "out": out,
            "dout": dout if dout_quality is None else dout_quality,
            "dk_attention": dk_attention,
            "dv_attention": dv_attention,
            "_rope_cos": problem.rope_cos,
            "_rope_sin": problem.rope_sin,
        }
    if producer_native_fp8_backward:
        assert bundle is not None
        assert dout_bundle is not None
        assert bundle.v_backward_fp8 is not None
        assert dout_bundle.dout_backward_fp8 is not None
        backward = lowp.backward_fp4_fp8dpdv_x32_split_dk_adaptive_qkv_bf16_prepacked_native
        producer_suffix = (
            dout_bundle.dout_backward_fp8,
            bundle.v_backward_fp8,
            dout_bundle.dpsum,
            dout_bundle.lse_log2,
        )
    elif producer_native_fp8_dout_v:
        assert bundle is not None
        assert dout_bundle is not None
        assert bundle.v_backward_fp8 is not None
        assert dout_bundle.dout_backward_fp8 is not None
        backward = getattr(
            lowp,
            "backward_fp4_fp8dpdv_x32_split_dk_adaptive_qkv_bf16_"
            "prepacked_dout_v_native",
        )
        producer_suffix = (
            dout_bundle.dout_backward_fp8,
            bundle.v_backward_fp8,
            producer_native_fp8_dout_stats_from_packed,
        )
    elif producer_native_fp8_v_only:
        assert bundle is not None
        assert bundle.v_backward_fp8 is not None
        backward = lowp.backward_fp4_fp8dpdv_x32_split_dk_adaptive_qkv_bf16_prepacked_v_native
        producer_suffix = (bundle.v_backward_fp8,)
    else:
        backward = lowp.backward_fp4_fp8dpdv_x32_split_dk_adaptive_qkv_bf16_native
        producer_suffix = ()
    (qkv_grad,) = (
        backward(
            q,
            k,
            v,
            out,
            lse,
            dout,
            backward_operands.q_fp4,
            backward_operands.score_q_fp4,
            backward_operands.k_fp4,
            backward_operands.score_k_fp4,
            backward_operands.qk_scales,
            *producer_suffix,
            4096.0,
            True,
            float(QK_DIM**-0.5),
            False,
        )
    )
    qkv_grad_operand = None
    if problem.rope_cos is not None:
        assert problem.rope_sin is not None
        if fuse_inverse_rope_handoff:
            if not lowp_dense_projections or qkv_grad_global_scale is None:
                raise ValueError(
                    "fused inverse-RoPE handoff requires delayed-scale "
                    "low-precision projection backward"
                )
            qkv_grad_operand = tuple(
                b300_prepare_nvfp4_projection_operand_inverse_rope(
                    qkv_grad.reshape(problem.rows, problem.qkv_width),
                    problem.rope_cos,
                    problem.rope_sin,
                    global_scale=qkv_grad_global_scale,
                    publish_inverse_bf16=True,
                )
            )
        else:
            b300_inverse_rope_interleaved_qkv_grad_(
                qkv_grad,
                problem.rope_cos,
                problem.rope_sin,
            )
    if lowp_dense_projections:
        dx = _lowp_dense_projection(
            qkv_grad.reshape(problem.rows, problem.qkv_width),
            qkv_grad_operand,
            problem.qkv_backward_weight_operand,
            global_scale=qkv_grad_global_scale,
        )
    else:
        dx = _project_interleaved_bf16(
            qkv_grad,
            problem.qkv_weight_interleaved,
            splits=_projection_splits(problem.sequence, problem.heads),
        )
    result = {
        "y": y,
        "dx": dx,
        "out": out,
        "dout": dout if dout_quality is None else dout_quality,
        "qkv_grad": qkv_grad,
    }
    if weight_gradients:
        result["dqkv_weight"] = torch.mm(
            qkv_grad.reshape(problem.rows, problem.qkv_width).T,
            problem.x,
        )
        result["dout_weight"] = torch.mm(problem.dy.T, out_matrix)
    return result


def run_mixed(
    problem: Problem,
    *,
    charge_input_pack: bool,
    weight_gradients: bool,
) -> dict[str, torch.Tensor]:
    bundle = _project_qkv_lowp(problem, charge_input_pack=charge_input_pack)
    assert bundle.q is not None and bundle.k is not None and bundle.v is not None
    mixed_v = lowp.prepack_mixed_v(bundle.v)
    out, lse_bhs = _run_forward_streaming_live_mxfp4(
        *bundle.forward_operands()
    )
    out_matrix = out.reshape(problem.rows, problem.v_width)
    y = torch.mm(out_matrix, problem.out_weight.T)
    dout = torch.mm(problem.dy, problem.out_weight).reshape_as(out)
    operands = B300AdaptiveLowpOperands(
        bundle.backward.q_fp4,
        bundle.backward.score_q_fp4,
        bundle.backward.k_fp4,
        bundle.backward.score_k_fp4,
        bundle.backward.qk_scales,
        mixed_v,
    )
    dq, dk, dv = b300_mha_bwd_adaptive_lowp(
        bundle.q,
        bundle.k,
        bundle.v,
        out,
        lse_bhs,
        dout,
        operands,
        route="mixed",
        causal=True,
        softmax_scale=float(QK_DIM**-0.5),
    )
    qkv_grad = _interleave_gradients(dq, dk, dv)
    if problem.rope_cos is not None:
        assert problem.rope_sin is not None
        b300_inverse_rope_interleaved_qkv_grad_(
            qkv_grad,
            problem.rope_cos,
            problem.rope_sin,
        )
    dq_projection, dk_projection, dv_projection = (
        _split_interleaved_gradients(qkv_grad)
    )
    dx = torch.mm(
        dq_projection.reshape(problem.rows, problem.q_width),
        problem.q_native_weight,
    )
    dx.addmm_(
        dk_projection.reshape(problem.rows, problem.q_width),
        problem.k_native_weight,
    )
    dx.addmm_(
        dv_projection.reshape(problem.rows, problem.v_width),
        problem.v_weight,
    )
    result = {
        "y": y,
        "dx": dx,
        "out": out,
        "dout": dout,
        "qkv_grad": qkv_grad,
    }
    if weight_gradients:
        result["dqkv_weight"] = torch.mm(
            qkv_grad.reshape(problem.rows, problem.qkv_width).T,
            problem.x,
        )
        result["dout_weight"] = torch.mm(problem.dy.T, out_matrix)
    return result


def quality_record(
    reference: dict[str, torch.Tensor],
    actual: dict[str, torch.Tensor],
) -> dict[str, Any]:
    common_names = ("y", "dx", "out", "dout")
    result = {
        name: metrics(
            reference[name],
            actual[name].float() * 0.25
            if name == "dout" and actual[name].dtype == torch.float8_e4m3fn
            else actual[name],
        )
        for name in common_names
    }
    dq_ref, dk_ref, dv_ref = _split_interleaved_gradients(
        reference["qkv_grad"]
    )
    if "qkv_grad" not in actual:
        dk = _inverse_pair_rope(
            actual["dk_attention"],
            actual["_rope_cos"],
            actual["_rope_sin"],
        )
        result["qkv_grad"] = {"available": False}
        result["dq"] = {"available": False}
        result["dk"] = metrics(dk_ref, dk)
        result["dv"] = metrics(dv_ref, actual["dv_attention"])
        return result
    result["qkv_grad"] = metrics(
        reference["qkv_grad"], actual["qkv_grad"]
    )
    dq, dk, dv = _split_interleaved_gradients(actual["qkv_grad"])
    result["dq"] = metrics(dq_ref, dq)
    result["dk"] = metrics(dk_ref, dk)
    result["dv"] = metrics(dv_ref, dv)
    if "dqkv_weight" in reference:
        result["dqkv_weight"] = metrics(
            reference["dqkv_weight"], actual["dqkv_weight"]
        )
        result["dout_weight"] = metrics(
            reference["dout_weight"], actual["dout_weight"]
        )
    return result


def _calibrate_delayed_dense_scales(
    problem: Problem,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create delayed projection scales from an independent prior batch."""
    torch.manual_seed(seed)
    calibration_x = (
        torch.randn_like(problem.x.float()) * 0.1
    ).bfloat16()
    calibration_dy = (
        torch.randn_like(problem.dy.float()) * 0.1
    ).bfloat16()
    q = torch.mm(calibration_x, problem.q_native_weight.T).reshape(
        1, problem.sequence, problem.heads, QK_DIM
    )
    k = torch.mm(calibration_x, problem.k_native_weight.T).reshape_as(q)
    if problem.rope_cos is not None:
        assert problem.rope_sin is not None
        q = _apply_pair_rope(q, problem.rope_cos, problem.rope_sin)
        k = _apply_pair_rope(k, problem.rope_cos, problem.rope_sin)
    v = torch.mm(calibration_x, problem.v_weight.T).reshape(
        1, problem.sequence, problem.heads, V_DIM
    )
    out, lse = b300_mha_fwd(q, k, v, causal=True, return_lse=True)
    dout = torch.mm(calibration_dy, problem.out_weight).reshape_as(out)
    dq, dk, dv = lowp.backward_bf16_control(
        q,
        k,
        v,
        out,
        lse,
        dout,
        True,
        float(QK_DIM**-0.5),
        False,
    )
    qkv_grad = _interleave_gradients(dq, dk, dv)
    if problem.rope_cos is not None:
        assert problem.rope_sin is not None
        b300_inverse_rope_interleaved_qkv_grad_(
            qkv_grad,
            problem.rope_cos,
            problem.rope_sin,
        )
    x_global_scale = b300_prepare_nvfp4_projection_operand(calibration_x)[2]
    dy_global_scale = b300_prepare_nvfp4_projection_operand(calibration_dy)[2]
    out_global_scale = b300_prepare_nvfp4_projection_operand(
        out.reshape(problem.rows, problem.v_width)
    )[2]
    qkv_grad_global_scale = b300_prepare_nvfp4_projection_operand(
        qkv_grad.reshape(problem.rows, problem.qkv_width)
    )[2]
    return (
        x_global_scale,
        dy_global_scale,
        out_global_scale,
        qkv_grad_global_scale,
    )


def evaluate_shape(
    sequence: int,
    heads: int,
    hidden: int,
    *,
    seed: int,
    warmups: int,
    samples: int,
    scope: str,
    use_rope: bool,
    peak_tflops: float | None,
) -> dict[str, Any]:
    problem = build_problem(
        sequence,
        heads,
        hidden,
        seed,
        use_rope=use_rope,
    )
    include_weight_gradients = scope in {"training", "both"}
    reference = run_bf16(
        problem,
        weight_gradients=include_weight_gradients,
    )
    # Model delayed/device-resident scaling state without adding a reduction
    # launch to the current step.  Calibrate from an independent prior batch
    # under the same weights; the full-scale candidates below still charge a
    # fresh current-batch amax reduction.
    (
        x_global_scale,
        dy_global_scale,
        out_global_scale,
        qkv_grad_global_scale,
    ) = (
        _calibrate_delayed_dense_scales(problem, seed=seed + 1_000_003)
    )
    candidates = {
        "fixed_upstream_native_bf16_dense": lambda: run_fixed(
            problem,
            charge_input_pack=False,
            lowp_dense_projections=False,
            cache_dy_operand=True,
            weight_gradients=include_weight_gradients,
        ),
        "fixed_materialized_bf16_dense": lambda: run_fixed(
            problem,
            charge_input_pack=True,
            lowp_dense_projections=False,
            cache_dy_operand=False,
            weight_gradients=include_weight_gradients,
        ),
        "fixed_upstream_native_nvfp4_dense": lambda: run_fixed(
            problem,
            charge_input_pack=False,
            lowp_dense_projections=True,
            cache_dy_operand=True,
            weight_gradients=include_weight_gradients,
        ),
        "fixed_materialized_nvfp4_dense": lambda: run_fixed(
            problem,
            charge_input_pack=True,
            lowp_dense_projections=True,
            cache_dy_operand=False,
            weight_gradients=include_weight_gradients,
        ),
        "fixed_upstream_native_nvfp4_delayed_scale_dense": lambda: run_fixed(
            problem,
            charge_input_pack=False,
            lowp_dense_projections=True,
            cache_dy_operand=True,
            weight_gradients=include_weight_gradients,
            out_global_scale=out_global_scale,
            qkv_grad_global_scale=qkv_grad_global_scale,
        ),
        "fixed_materialized_nvfp4_delayed_scale_dense": lambda: run_fixed(
            problem,
            charge_input_pack=True,
            lowp_dense_projections=True,
            cache_dy_operand=False,
            weight_gradients=include_weight_gradients,
            x_global_scale=x_global_scale,
            dy_global_scale=dy_global_scale,
            out_global_scale=out_global_scale,
            qkv_grad_global_scale=qkv_grad_global_scale,
        ),
        "mixed_upstream_native_bf16_dense": lambda: run_mixed(
            problem,
            charge_input_pack=False,
            weight_gradients=include_weight_gradients,
        ),
        "mixed_materialized_bf16_dense": lambda: run_mixed(
            problem,
            charge_input_pack=True,
            weight_gradients=include_weight_gradients,
        ),
    }
    if use_rope:
        candidates[
            "fixed_upstream_native_nvfp4_delayed_scale_dense_fused_inverse_rope"
        ] = lambda: run_fixed(
            problem,
            charge_input_pack=False,
            lowp_dense_projections=True,
            cache_dy_operand=True,
            weight_gradients=include_weight_gradients,
            out_global_scale=out_global_scale,
            qkv_grad_global_scale=qkv_grad_global_scale,
            fuse_inverse_rope_handoff=True,
        )
        candidates[
            "fixed_upstream_native_nvfp4_delayed_scale_dense_"
            "producer_native_fp8_fused_inverse_rope"
        ] = lambda: run_fixed(
            problem,
            charge_input_pack=False,
            lowp_dense_projections=True,
            cache_dy_operand=True,
            weight_gradients=include_weight_gradients,
            out_global_scale=out_global_scale,
            qkv_grad_global_scale=qkv_grad_global_scale,
            fuse_inverse_rope_handoff=True,
            producer_native_fp8_backward=True,
        )
        candidates[
            "fixed_upstream_native_nvfp4_delayed_scale_dense_"
            "producer_native_fp8_v_fused_inverse_rope"
        ] = lambda: run_fixed(
            problem,
            charge_input_pack=False,
            lowp_dense_projections=True,
            cache_dy_operand=True,
            weight_gradients=include_weight_gradients,
            out_global_scale=out_global_scale,
            qkv_grad_global_scale=qkv_grad_global_scale,
            fuse_inverse_rope_handoff=True,
            producer_native_fp8_v_only=True,
        )
        candidates[
            "fixed_upstream_native_nvfp4_delayed_scale_dense_"
            "producer_native_fp8_dout_v_fused_inverse_rope"
        ] = lambda: run_fixed(
            problem,
            charge_input_pack=False,
            lowp_dense_projections=True,
            cache_dy_operand=True,
            weight_gradients=include_weight_gradients,
            out_global_scale=out_global_scale,
            qkv_grad_global_scale=qkv_grad_global_scale,
            fuse_inverse_rope_handoff=True,
            producer_native_fp8_dout_v=True,
        )
        if not include_weight_gradients:
            candidates[
                "fixed_upstream_native_nvfp4_delayed_scale_dense_"
                "stacked_qkv_projection"
            ] = lambda: run_fixed(
                problem,
                charge_input_pack=False,
                lowp_dense_projections=True,
                cache_dy_operand=True,
                weight_gradients=False,
                out_global_scale=out_global_scale,
                qkv_grad_global_scale=qkv_grad_global_scale,
                fuse_inverse_rope_handoff=True,
                producer_native_fp8_dout_v=True,
                stacked_qkv_projection=True,
            )
            candidates[
                "fixed_upstream_native_nvfp4_delayed_scale_dense_"
                "hierarchical_qkv_projection"
            ] = lambda: run_fixed(
                problem,
                charge_input_pack=False,
                lowp_dense_projections=True,
                cache_dy_operand=True,
                weight_gradients=False,
                out_global_scale=out_global_scale,
                qkv_grad_global_scale=qkv_grad_global_scale,
                fuse_inverse_rope_handoff=True,
                producer_native_fp8_dout_v=True,
                hierarchical_qkv_projection=True,
            )
        candidates[
            "fixed_upstream_native_nvfp4_delayed_scale_dense_"
            "producer_native_fp8_dout_v_bf16_stats_fused_inverse_rope"
        ] = lambda: run_fixed(
            problem,
            charge_input_pack=False,
            lowp_dense_projections=True,
            cache_dy_operand=True,
            weight_gradients=include_weight_gradients,
            out_global_scale=out_global_scale,
            qkv_grad_global_scale=qkv_grad_global_scale,
            fuse_inverse_rope_handoff=True,
            producer_native_fp8_dout_v=True,
            producer_native_fp8_dout_stats_from_packed=False,
        )
        candidates[
            "fixed_upstream_native_nvfp4_delayed_scale_dense_bf16_forward_"
            "producer_native_fp8_v_fused_inverse_rope"
        ] = lambda: run_fixed(
            problem,
            charge_input_pack=False,
            lowp_dense_projections=True,
            cache_dy_operand=True,
            weight_gradients=include_weight_gradients,
            out_global_scale=out_global_scale,
            qkv_grad_global_scale=qkv_grad_global_scale,
            fuse_inverse_rope_handoff=True,
            producer_native_fp8_v_only=True,
            bf16_attention_forward=True,
        )
        candidates[
            "fixed_upstream_native_nvfp4_delayed_scale_dense_bf16_forward_"
            "producer_native_fp8_dout_v_fused_inverse_rope"
        ] = lambda: run_fixed(
            problem,
            charge_input_pack=False,
            lowp_dense_projections=True,
            cache_dy_operand=True,
            weight_gradients=include_weight_gradients,
            out_global_scale=out_global_scale,
            qkv_grad_global_scale=qkv_grad_global_scale,
            fuse_inverse_rope_handoff=True,
            producer_native_fp8_dout_v=True,
            bf16_attention_forward=True,
        )
        candidates[
            "fixed_bf16_qkv_forward_nvfp4_delayed_scale_dense_"
            "fused_inverse_rope"
        ] = lambda: run_fixed(
            problem,
            charge_input_pack=False,
            lowp_dense_projections=True,
            cache_dy_operand=True,
            weight_gradients=include_weight_gradients,
            out_global_scale=out_global_scale,
            qkv_grad_global_scale=qkv_grad_global_scale,
            fuse_inverse_rope_handoff=True,
            bf16_attention_forward=True,
            bf16_qkv_projection=True,
        )
    quality = {
        name: quality_record(reference, candidate())
        for name, candidate in candidates.items()
    }
    timed: dict[str, Callable[[], object]] = {
        "bf16": lambda: run_bf16(
            problem,
            weight_gradients=include_weight_gradients,
        )
    }
    timed.update(candidates)
    timing = time_rotated(timed, warmups=warmups, samples=samples)
    bf16_ms = timing["bf16"]["median_ms"]
    algorithmic_flops = _algorithmic_matmul_flops(
        sequence,
        heads,
        hidden,
        weight_gradients=include_weight_gradients,
    )
    for name, values in timing.items():
        values["speedup_vs_bf16"] = bf16_ms / values["median_ms"]
        values["algorithmic_mfu_multiplier_vs_bf16"] = (
            values["speedup_vs_bf16"]
        )
        values["algorithmic_matmul_tflops"] = (
            algorithmic_flops / (values["median_ms"] * 1.0e9)
        )
        if peak_tflops is not None:
            values["algorithmic_mfu_fraction_of_supplied_peak"] = (
                values["algorithmic_matmul_tflops"] / peak_tflops
            )
        values["time_reduction_percent"] = (
            100.0 * (1.0 - values["median_ms"] / bf16_ms)
        )
    return {
        "shape": {
            "batch": 1,
            "sequence": sequence,
            "heads": heads,
            "hidden": hidden,
            "qk_head_dim": QK_DIM,
            "v_head_dim": V_DIM,
            "equal_kv_heads": True,
            "rope": use_rope,
        },
        "scope": (
            "forward_backward_and_weight_gradients"
            if include_weight_gradients
            else "forward_backward_activation_gradients"
        ),
        "scale_contract": "precomputed_device_resident_per_head_record",
        "algorithmic_matmul_flops": algorithmic_flops,
        "mfu_peak_tflops": peak_tflops,
        "mfu_note": (
            "For identical useful FLOPs, the reported MFU multiplier equals "
            "the measured speedup versus BF16. Absolute algorithmic MFU is "
            "reported only when --peak-tflops supplies the agreed dense-peak "
            "denominator."
        ),
        "quality_vs_bf16": quality,
        "timing": timing,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shapes",
        default="4096x24x3072",
        help="comma-separated sequence x heads x hidden triples",
    )
    parser.add_argument("--seed", type=int, default=2026081402)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument(
        "--rope",
        action="store_true",
        help="fuse full-head RoPE into Q/K projection publication",
    )
    parser.add_argument(
        "--scope",
        choices=("activation", "training", "both"),
        default="both",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--peak-tflops",
        type=float,
        help="optional dense-peak denominator for absolute algorithmic MFU",
    )
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("evaluation requires exactly one visible GPU")
    torch.cuda.set_device(0)

    result: dict[str, Any] = {
        "contract": {
            "model_boundary": "dense causal decoder attention sublayer",
            "stock_llama_drop_in": False,
            "fp4_forward_implementation": (
                "causal D192 streaming FP4 path; not the optimized "
                "HAO-direct D64/D128 noncausal kernel"
            ),
            "bf16_qkv_baseline": "three direct consumer-layout GEMMs",
            "training_scope_excludes": [
                "optimizer step",
                "updated-weight repacking",
                "RMSNorm, residual, MLP, and loss",
            ],
            "missing_stock_llama_features": [
                "arbitrary Q/K head dimensions (Llama-3.2-1B uses D=64)",
                "grouped-query attention",
                "causal port of the optimized HAO-direct FP4 forward",
            ] + ([] if args.rope else ["RoPE-aware projection publication"]),
            "cached_outside_timing": [
                "all projection weight operands",
                "interleaved QKV weight view",
                "RoPE cosine/sine tables and pair-interleaved Q/K weights",
                "adaptive Q/K scale record",
                "delayed X/dY/output/QKV-gradient global scales for delayed-scale candidates",
            ],
        },
        "configuration": {
            "shapes": args.shapes,
            "seed": args.seed,
            "warmups": args.warmups,
            "samples": args.samples,
            "scope": args.scope,
            "rope": args.rope,
            "peak_tflops": args.peak_tflops,
        },
        "records": [],
    }
    scopes = ("activation", "training") if args.scope == "both" else (args.scope,)
    for shape_index, shape in enumerate(parse_shapes(args.shapes)):
        for scope in scopes:
            record = evaluate_shape(
                *shape,
                seed=args.seed + shape_index * 17,
                warmups=args.warmups,
                samples=args.samples,
                scope=scope,
                use_rope=args.rope,
                peak_tflops=args.peak_tflops,
            )
            result["records"].append(record)
            timing = record["timing"]
            winner = min(timing, key=lambda name: timing[name]["median_ms"])
            print(
                f"S{shape[0]} H{shape[1]} K{shape[2]} {scope}: "
                f"bf16={timing['bf16']['median_ms']:.6f} ms "
                f"winner={winner} {timing[winner]['median_ms']:.6f} ms "
                f"({timing[winner]['speedup_vs_bf16']:.3f}x)",
                flush=True,
            )
        torch.cuda.empty_cache()

    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
        print(f"wrote {args.output}")
    else:
        print(rendered)


if __name__ == "__main__":
    with torch.no_grad():
        main()
