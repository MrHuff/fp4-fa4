#!/usr/bin/env python3
"""Compare BF16 and FP4/FP8 training trajectories for one attention block.

This is a deterministic teacher/student convergence experiment for the native
B300 dense causal attention geometry.  It is deliberately smaller in scope
than language-model pretraining: QKV and output projection weights are learned
against fixed teacher activations, while the forward and backward paths are
the same kernels used by ``evaluate_llama_attention_e2e.py``.

The low-precision student uses projection-native NVFP4 QKV, the causal D192
streaming FP4 forward, NVFP4 output projections, and the adaptive
FP4-QK/FP8-dP,dV backward.  This is not the optimized HAO-direct D64/D128
forward family, which is currently noncausal.  Both students publish and apply
complete QKV and output-projection weight gradients.  Timing/MFU belongs to
the separate E2E harness; this file tests optimization trajectory only and
intentionally refreshes packed weight operands after every update.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from tk_fa4 import _C_b300_lowp_bwd as lowp
from tk_fa4.interface import (
    b300_interleave_qkv_projection_weights,
    b300_inverse_rope_interleaved_qkv_grad_,
    b300_mha_fwd,
    b300_prepare_nvfp4_projection_operand,
    b300_project_dout_unified_lowp_nvfp4,
    b300_project_nvfp4,
    b300_project_qkv_unified_lowp_nvfp4,
    b300_rope_pair_qk_,
    b300_stack_qkv_projection_weights,
)

from .evaluate_llama_attention_e2e import (
    QK_DIM,
    QKV_HEAD_WIDTH,
    V_DIM,
    Problem,
    _make_rope_tables,
    _run_forward_streaming_live_mxfp4,
    _split_interleaved_gradients,
    build_problem,
    metrics,
    parse_shapes,
)
from tk_fa4.fp4_pv_experiments import _run_qk_fp4_v_bf16


_WEIGHT_NAMES = ("q_weight", "k_weight", "v_weight", "out_weight")


@dataclass
class TrainableWeights:
    q: torch.nn.Parameter
    k: torch.nn.Parameter
    v: torch.nn.Parameter
    out: torch.nn.Parameter

    def parameters(self) -> list[torch.nn.Parameter]:
        return [self.q, self.k, self.v, self.out]

    def bf16(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return tuple(
            parameter.detach().bfloat16().contiguous()
            for parameter in self.parameters()
        )  # type: ignore[return-value]


@dataclass
class ForwardState:
    x: torch.Tensor
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    out: torch.Tensor
    lse: torch.Tensor
    out_weight: torch.Tensor
    attention_output_gain: float = 1.0
    attention_backward_policy: str = "chain_rule"
    forward_operands: Any | None = None
    backward_operands: Any | None = None
    v_backward_fp8: torch.Tensor | None = None


def _attention_backward_inputs(
    state: ForwardState,
) -> tuple[float, torch.Tensor]:
    """Return dO scale and saved O for the selected gain surrogate."""
    if state.attention_backward_policy == "chain_rule":
        return state.attention_output_gain, state.out
    if state.attention_backward_policy == "identity_ste":
        return 1.0, state.out
    if state.attention_backward_policy == "delta_corrected_ste":
        return 1.0, state.out * state.attention_output_gain
    raise ValueError(
        "unsupported attention backward policy "
        f"{state.attention_backward_policy!r}"
    )


def _make_student_weights(
    problem: Problem,
    *,
    seed: int,
    relative_noise: float,
) -> TrainableWeights:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    values = []
    for source in (
        problem.q_native_weight,
        problem.k_native_weight,
        problem.v_weight,
        problem.out_weight,
    ):
        master = source.float()
        noise = torch.randn(
            master.shape,
            device=master.device,
            dtype=master.dtype,
            generator=generator,
        )
        master = master + noise * (master.std() * relative_noise)
        values.append(torch.nn.Parameter(master))
    return TrainableWeights(*values)


def _clone_weights(source: TrainableWeights) -> TrainableWeights:
    return TrainableWeights(
        *(torch.nn.Parameter(parameter.detach().clone()) for parameter in source.parameters())
    )


def _bf16_forward(
    problem: Problem,
    x: torch.Tensor,
    weights: TrainableWeights,
) -> tuple[torch.Tensor, ForwardState]:
    q_weight, k_weight, v_weight, out_weight = weights.bf16()
    q = torch.mm(x, q_weight.T).reshape(
        1, problem.sequence, problem.heads, QK_DIM
    )
    k = torch.mm(x, k_weight.T).reshape_as(q)
    if problem.rope_cos is not None:
        assert problem.rope_sin is not None
        q, k = b300_rope_pair_qk_(
            q,
            k,
            problem.rope_cos,
            problem.rope_sin,
        )
    v = torch.mm(x, v_weight.T).reshape(
        1, problem.sequence, problem.heads, V_DIM
    )
    out, lse = b300_mha_fwd(q, k, v, causal=True, return_lse=True)
    y = torch.mm(out.reshape(problem.rows, problem.v_width), out_weight.T)
    return y, ForwardState(x, q, k, v, out, lse, out_weight)


def _lowp_forward(
    problem: Problem,
    x: torch.Tensor,
    weights: TrainableWeights,
    qk_scales: torch.Tensor,
    *,
    forward_p_quant_mode: str = "rte",
    forward_attention_mode: str = "mxfp4",
    attention_output_gain: float = 1.0,
    attention_backward_policy: str = "chain_rule",
) -> tuple[torch.Tensor, ForwardState, torch.Tensor]:
    q_weight, k_weight, v_weight, out_weight = weights.bf16()
    qkv_weight = b300_stack_qkv_projection_weights(
        q_weight,
        k_weight,
        v_weight,
    )
    bundle = b300_project_qkv_unified_lowp_nvfp4(
        tuple(b300_prepare_nvfp4_projection_operand(x)),
        tuple(b300_prepare_nvfp4_projection_operand(qkv_weight)),
        qk_scales,
        batch=1,
        seqlen=problem.sequence,
        heads=problem.heads,
        store_bf16=True,
        publish_fp8_backward=True,
        rope_cos=problem.rope_cos,
        rope_sin=problem.rope_sin,
    )
    assert bundle.q is not None and bundle.k is not None and bundle.v is not None
    assert bundle.v_backward_fp8 is not None
    forward_operands = bundle.forward_operands()
    if forward_attention_mode == "mxfp4":
        out, lse_bhs = _run_forward_streaming_live_mxfp4(
            *forward_operands,
            p_quant_mode=forward_p_quant_mode,
        )
        lse = lse_bhs.permute(0, 2, 1).contiguous()
    elif forward_attention_mode == "qk_fp4_v_bf16_control":
        out, lse_bhs = _run_qk_fp4_v_bf16(
            *forward_operands[:6],
            bundle.v,
            launch_mode="persistent",
        )
        lse = lse_bhs.permute(0, 2, 1).contiguous()
    elif forward_attention_mode == "bf16_on_lowp_qkv":
        out, lse = b300_mha_fwd(
            bundle.q,
            bundle.k,
            bundle.v,
            causal=True,
            return_lse=True,
        )
    else:
        raise ValueError(
            f"unsupported forward attention mode {forward_attention_mode!r}"
        )
    out_matrix = out.reshape(problem.rows, problem.v_width)
    out_operand = list(b300_prepare_nvfp4_projection_operand(out_matrix))
    if attention_output_gain != 1.0:
        out_operand[2] = (
            out_operand[2].float() * attention_output_gain
        ).contiguous()
    y = b300_project_nvfp4(
        tuple(out_operand),
        tuple(b300_prepare_nvfp4_projection_operand(out_weight)),
    )
    next_qk_scales = lowp.quantize_fp4_dual_qk_adaptive(
        bundle.q,
        bundle.k,
        16.0,
        2.0**-12,
        0.325,
        2.75,
        float(QK_DIM**-0.5),
        4096.0,
    )[4]
    return (
        y,
        ForwardState(
            x,
            bundle.q,
            bundle.k,
            bundle.v,
            out,
            lse,
            out_weight,
            attention_output_gain=attention_output_gain,
            attention_backward_policy=attention_backward_policy,
            forward_operands=forward_operands,
            backward_operands=bundle.backward,
            v_backward_fp8=bundle.v_backward_fp8,
        ),
        next_qk_scales,
    )


def _split_weight_gradient(
    qkv_gradient: torch.Tensor,
    problem: Problem,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    per_head = qkv_gradient.reshape(problem.heads, QKV_HEAD_WIDTH, problem.hidden)
    return (
        per_head[:, :QK_DIM].reshape(problem.q_width, problem.hidden),
        per_head[:, QK_DIM : 2 * QK_DIM].reshape(
            problem.q_width, problem.hidden
        ),
        per_head[:, 2 * QK_DIM :].reshape(problem.v_width, problem.hidden),
    )


def _projection_weight_gradients(
    problem: Problem,
    state: ForwardState,
    dy: torch.Tensor,
    qkv_grad: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    qkv_weight_gradient = torch.mm(
        qkv_grad.reshape(problem.rows, problem.qkv_width).T,
        state.x,
    )
    q_gradient, k_gradient, v_gradient = _split_weight_gradient(
        qkv_weight_gradient,
        problem,
    )
    out_gradient = torch.mm(
        dy.T,
        state.out.reshape(problem.rows, problem.v_width),
    ) * state.attention_output_gain
    return q_gradient, k_gradient, v_gradient, out_gradient


def _bf16_backward(
    problem: Problem,
    state: ForwardState,
    dy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dout_gain, backward_out = _attention_backward_inputs(state)
    dout = (
        torch.mm(dy, state.out_weight)
        * dout_gain
    ).reshape_as(state.out)
    dq, dk, dv = lowp.backward_bf16_control(
        state.q,
        state.k,
        state.v,
        backward_out,
        state.lse,
        dout,
        True,
        float(QK_DIM**-0.5),
        False,
    )
    qkv_grad = torch.cat(
        (
            dq.to(torch.bfloat16),
            dk.to(torch.bfloat16),
            dv.to(torch.bfloat16),
        ),
        dim=-1,
    ).contiguous()
    if problem.rope_cos is not None:
        assert problem.rope_sin is not None
        b300_inverse_rope_interleaved_qkv_grad_(
            qkv_grad,
            problem.rope_cos,
            problem.rope_sin,
        )
    return _projection_weight_gradients(problem, state, dy, qkv_grad)


def _lowp_backward(
    problem: Problem,
    state: ForwardState,
    dy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert state.backward_operands is not None
    assert state.v_backward_fp8 is not None
    dout_gain, backward_out = _attention_backward_inputs(state)
    dy_operand = list(b300_prepare_nvfp4_projection_operand(dy))
    if dout_gain != 1.0:
        dy_operand[2] = (
            dy_operand[2].float() * dout_gain
        ).contiguous()
    dout_bundle = b300_project_dout_unified_lowp_nvfp4(
        tuple(dy_operand),
        tuple(
            b300_prepare_nvfp4_projection_operand(
                state.out_weight.T.contiguous()
            )
        ),
        backward_out,
        state.lse,
        batch=1,
        seqlen=problem.sequence,
        heads=problem.heads,
        store_bf16=False,
        publish_fp8_backward=True,
        publish_stats=False,
    )
    assert dout_bundle.dout_backward_fp8 is not None
    dout = dout_bundle.dout_storage.reshape_as(state.out)
    operands = state.backward_operands
    (qkv_grad,) = (
        lowp.backward_fp4_fp8dpdv_x32_split_dk_adaptive_qkv_bf16_prepacked_dout_v_native(
            state.q,
            state.k,
            state.v,
            backward_out,
            state.lse,
            dout,
            operands.q_fp4,
            operands.score_q_fp4,
            operands.k_fp4,
            operands.score_k_fp4,
            operands.qk_scales,
            dout_bundle.dout_backward_fp8,
            state.v_backward_fp8,
            True,
            4096.0,
            True,
            float(QK_DIM**-0.5),
            False,
        )
    )
    if problem.rope_cos is not None:
        assert problem.rope_sin is not None
        b300_inverse_rope_interleaved_qkv_grad_(
            qkv_grad,
            problem.rope_cos,
            problem.rope_sin,
        )
    return _projection_weight_gradients(problem, state, dy, qkv_grad)


def _relative_mse_and_gradient(
    actual: torch.Tensor,
    target: torch.Tensor,
) -> tuple[float, torch.Tensor]:
    difference = actual.float() - target.float()
    denominator = target.float().square().mean().clamp_min(1.0e-12)
    loss = difference.square().mean() / denominator
    gradient = (
        difference * (2.0 / (difference.numel() * denominator))
    ).bfloat16()
    return float(loss), gradient


def _assign_gradients(
    weights: TrainableWeights,
    gradients: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    for parameter, gradient in zip(weights.parameters(), gradients):
        parameter.grad = gradient.float()


def _unscale_gradients(
    gradients: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    loss_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(
        (gradient.float() / loss_scale)
        for gradient in gradients
    )  # type: ignore[return-value]


def _gradient_quality(
    reference: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    actual: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, Any]:
    return {
        name: metrics(reference_gradient, actual_gradient)
        for name, reference_gradient, actual_gradient in zip(
            _WEIGHT_NAMES,
            reference,
            actual,
        )
    }


def _sampled_tensor_statistics(
    tensor: torch.Tensor,
    *,
    max_samples: int = 65_536,
) -> dict[str, Any]:
    """Return compact tail/scale diagnostics without serializing tensors."""
    flat = tensor.detach().float().reshape(-1)
    stride = max((flat.numel() + max_samples - 1) // max_samples, 1)
    sample = flat[::stride][:max_samples]
    finite = torch.isfinite(sample)
    finite_sample = sample[finite]
    if finite_sample.numel() == 0:
        return {
            "numel": flat.numel(),
            "sample_count": sample.numel(),
            "finite_fraction": 0.0,
        }
    absolute = finite_sample.abs()
    signed_quantiles = torch.quantile(
        finite_sample,
        torch.tensor(
            (0.001, 0.01, 0.5, 0.99, 0.999),
            device=finite_sample.device,
        ),
    )
    absolute_quantiles = torch.quantile(
        absolute,
        torch.tensor(
            (0.5, 0.9, 0.99, 0.999),
            device=absolute.device,
        ),
    )
    rms = torch.sqrt(torch.mean(finite_sample.square())).clamp_min(1.0e-30)
    return {
        "numel": flat.numel(),
        "sample_count": sample.numel(),
        "finite_fraction": float(finite.float().mean()),
        "zero_fraction": float((finite_sample == 0).float().mean()),
        "positive_fraction": float((finite_sample > 0).float().mean()),
        "mean": float(finite_sample.mean()),
        "rms": float(rms),
        "max_abs": float(absolute.max()),
        "max_to_rms": float(absolute.max() / rms),
        "signed_quantiles": {
            name: float(value)
            for name, value in zip(
                ("p001", "p01", "p50", "p99", "p999"),
                signed_quantiles,
            )
        },
        "absolute_quantiles": {
            name: float(value)
            for name, value in zip(
                ("p50", "p90", "p99", "p999"),
                absolute_quantiles,
            )
        },
    }


def _named_tensor_statistics(
    tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, Any]:
    return {
        name: _sampled_tensor_statistics(tensor)
        for name, tensor in zip(_WEIGHT_NAMES, tensors)
    }


def _snapshot_weights(weights: TrainableWeights) -> tuple[torch.Tensor, ...]:
    return tuple(parameter.detach().clone() for parameter in weights.parameters())


def _forward_diagnostics(
    problem: Problem,
    x: torch.Tensor,
    target: torch.Tensor,
    lowp_weights: TrainableWeights,
    lowp_y: torch.Tensor,
    lowp_state: ForwardState,
) -> tuple[
    dict[str, Any],
    ForwardState,
    ForwardState,
    ForwardState,
    torch.Tensor,
]:
    """Decompose the same-weight low-precision forward error by stage."""
    bf16_y, bf16_state = _bf16_forward(problem, x, lowp_weights)
    bf16_on_lowp_qkv_out, bf16_on_lowp_qkv_lse = b300_mha_fwd(
        lowp_state.q,
        lowp_state.k,
        lowp_state.v,
        causal=True,
        return_lse=True,
    )
    qk_fp4_v_bf16_out = None
    qk_fp4_v_bf16_lse = None
    if lowp_state.forward_operands is not None:
        qk_fp4_v_bf16_out, qk_fp4_v_bf16_lse_bhs = _run_qk_fp4_v_bf16(
            *lowp_state.forward_operands[:6],
            lowp_state.v,
            launch_mode="persistent",
        )
        qk_fp4_v_bf16_lse = qk_fp4_v_bf16_lse_bhs.permute(
            0, 2, 1
        ).contiguous()
    bf16_on_lowp_out_y = torch.mm(
        lowp_state.out.reshape(problem.rows, problem.v_width),
        lowp_state.out_weight.T,
    ) * lowp_state.attention_output_gain
    attention_reference_f = bf16_on_lowp_qkv_out.float()
    attention_actual_f = lowp_state.out.float()
    least_squares_gain = float(
        torch.sum(attention_reference_f * attention_actual_f)
        / torch.sum(attention_actual_f.square()).clamp_min(1.0e-30)
    )
    bf16_loss, common_dy = _relative_mse_and_gradient(bf16_y, target)
    lowp_loss = _relative_mse_and_gradient(lowp_y, target)[0]
    bf16_gain_state = ForwardState(
        bf16_state.x,
        bf16_state.q,
        bf16_state.k,
        bf16_state.v,
        bf16_state.out,
        bf16_state.lse,
        bf16_state.out_weight,
        attention_output_gain=lowp_state.attention_output_gain,
        attention_backward_policy=lowp_state.attention_backward_policy,
    )
    bf16_on_lowp_qkv_state = ForwardState(
        lowp_state.x,
        lowp_state.q,
        lowp_state.k,
        lowp_state.v,
        bf16_on_lowp_qkv_out,
        bf16_on_lowp_qkv_lse,
        lowp_state.out_weight,
        attention_output_gain=lowp_state.attention_output_gain,
        attention_backward_policy=lowp_state.attention_backward_policy,
    )
    return (
        {
            "same_weight_losses": {
                "bf16_relative_mse": bf16_loss,
                "lowp_native_relative_mse": lowp_loss,
                "lowp_to_bf16_ratio": lowp_loss / max(bf16_loss, 1.0e-30),
            },
            "qkv_projection_plus_rope": {
                "q": metrics(bf16_state.q, lowp_state.q),
                "k": metrics(bf16_state.k, lowp_state.k),
                "v": metrics(bf16_state.v, lowp_state.v),
            },
            "qkv_projection_effect_on_attention": {
                "out": metrics(bf16_state.out, bf16_on_lowp_qkv_out),
                "lse": metrics(bf16_state.lse, bf16_on_lowp_qkv_lse),
            },
            "fp4_attention_kernel_given_same_lowp_qkv": {
                "out": metrics(bf16_on_lowp_qkv_out, lowp_state.out),
                "lse": metrics(bf16_on_lowp_qkv_lse, lowp_state.lse),
            },
            "qk_fp4_v_bf16_control_given_same_lowp_qkv": (
                {
                    "out": metrics(
                        bf16_on_lowp_qkv_out,
                        qk_fp4_v_bf16_out,
                    ),
                    "lse": metrics(
                        bf16_on_lowp_qkv_lse,
                        qk_fp4_v_bf16_lse,
                    ),
                }
                if qk_fp4_v_bf16_out is not None
                and qk_fp4_v_bf16_lse is not None
                else None
            ),
            "attention_output_gain_diagnostic": {
                "configured_gain": lowp_state.attention_output_gain,
                "configured_gain_out": metrics(
                    bf16_on_lowp_qkv_out,
                    lowp_state.out * lowp_state.attention_output_gain,
                ),
                "least_squares_gain": least_squares_gain,
                "least_squares_gain_out": metrics(
                    bf16_on_lowp_qkv_out,
                    lowp_state.out * least_squares_gain,
                ),
            },
            "nvfp4_output_projection_given_same_attention_output": metrics(
                bf16_on_lowp_out_y,
                lowp_y,
            ),
            "end_to_end_same_weights": metrics(bf16_y, lowp_y),
            "activation_distributions": {
                "bf16_q": _sampled_tensor_statistics(bf16_state.q),
                "lowp_q": _sampled_tensor_statistics(lowp_state.q),
                "bf16_k": _sampled_tensor_statistics(bf16_state.k),
                "lowp_k": _sampled_tensor_statistics(lowp_state.k),
                "bf16_v": _sampled_tensor_statistics(bf16_state.v),
                "lowp_v": _sampled_tensor_statistics(lowp_state.v),
                "bf16_attention_out": _sampled_tensor_statistics(
                    bf16_state.out
                ),
                "lowp_attention_out": _sampled_tensor_statistics(
                    lowp_state.out
                ),
            },
        },
        bf16_state,
        bf16_gain_state,
        bf16_on_lowp_qkv_state,
        common_dy,
    )


def _dout_publication_diagnostics(
    problem: Problem,
    state: ForwardState,
    dy: torch.Tensor,
) -> dict[str, Any]:
    """Split dO projection error from its fixed-scale FP8 publication error."""
    dout_gain, backward_out = _attention_backward_inputs(state)
    exact_dout = (
        torch.mm(dy, state.out_weight)
        * dout_gain
    ).reshape_as(state.out)
    dy_operand = list(b300_prepare_nvfp4_projection_operand(dy))
    if dout_gain != 1.0:
        dy_operand[2] = (
            dy_operand[2].float() * dout_gain
        ).contiguous()
    bundle = b300_project_dout_unified_lowp_nvfp4(
        tuple(dy_operand),
        tuple(
            b300_prepare_nvfp4_projection_operand(
                state.out_weight.T.contiguous()
            )
        ),
        backward_out,
        state.lse,
        batch=1,
        seqlen=problem.sequence,
        heads=problem.heads,
        store_bf16=True,
        publish_fp8_backward=True,
        publish_stats=False,
    )
    assert bundle.dout is not None and bundle.dout_backward_fp8 is not None
    published_dout = bundle.dout_backward_fp8.float() * 0.25
    return {
        "nvfp4_projection_bf16_output": metrics(exact_dout, bundle.dout),
        "fixed_scale_fp8_publication": metrics(bundle.dout, published_dout),
        "combined_projection_and_fp8": metrics(exact_dout, published_dout),
        "exact_dout_distribution": _sampled_tensor_statistics(exact_dout),
        "published_dout_distribution": _sampled_tensor_statistics(
            published_dout
        ),
    }


def _pre_update_diagnostics(
    problem: Problem,
    train_x: torch.Tensor,
    train_target: torch.Tensor,
    bf16_weights: TrainableWeights,
    lowp_weights: TrainableWeights,
    bf16_gradients: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    lowp_gradients: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    bf16_dy: torch.Tensor,
    lowp_dy: torch.Tensor,
    lowp_y: torch.Tensor,
    lowp_state: ForwardState,
    loss_scale: float,
) -> dict[str, Any]:
    """Localize route divergence before the optimizer mutates either student."""
    (
        forward,
        bf16_same_weight_state,
        bf16_gain_state,
        bf16_on_lowp_qkv_state,
        common_dy,
    ) = _forward_diagnostics(
        problem,
        train_x,
        train_target,
        lowp_weights,
        lowp_y,
        lowp_state,
    )
    common_dy_scaled = (common_dy.float() * loss_scale).bfloat16()
    bf16_same_weight_gradients = _unscale_gradients(
        _bf16_backward(problem, bf16_same_weight_state, common_dy_scaled),
        loss_scale,
    )
    bf16_gain_gradients = _unscale_gradients(
        _bf16_backward(problem, bf16_gain_state, common_dy_scaled),
        loss_scale,
    )
    bf16_on_lowp_qkv_gradients = _unscale_gradients(
        _bf16_backward(problem, bf16_on_lowp_qkv_state, common_dy_scaled),
        loss_scale,
    )
    bf16_on_lowp_state_gradients = _unscale_gradients(
        _bf16_backward(problem, lowp_state, common_dy_scaled),
        loss_scale,
    )
    lowp_on_lowp_state_gradients = _unscale_gradients(
        _lowp_backward(problem, lowp_state, common_dy_scaled),
        loss_scale,
    )
    return {
        "forward": forward,
        "loss_gradient": {
            "native_bf16_vs_lowp": metrics(bf16_dy, lowp_dy),
            "common_same_weight_bf16_dy_distribution": (
                _sampled_tensor_statistics(common_dy_scaled)
            ),
        },
        "dout_projection_and_publication": _dout_publication_diagnostics(
            problem,
            lowp_state,
            common_dy_scaled,
        ),
        "weight_gradient_quality": {
            "native_routes_different_current_weights_and_losses": (
                _gradient_quality(bf16_gradients, lowp_gradients)
            ),
            "forward_state_effect_with_bf16_backward": _gradient_quality(
                bf16_same_weight_gradients,
                bf16_on_lowp_state_gradients,
            ),
            "configured_gain_effect_with_bf16_attention": _gradient_quality(
                bf16_same_weight_gradients,
                bf16_gain_gradients,
            ),
            "qkv_projection_effect_with_bf16_attention": _gradient_quality(
                bf16_gain_gradients,
                bf16_on_lowp_qkv_gradients,
            ),
            "fp4_attention_effect_at_same_lowp_qkv": _gradient_quality(
                bf16_on_lowp_qkv_gradients,
                bf16_on_lowp_state_gradients,
            ),
            "backward_quantization_effect_on_same_lowp_state": (
                _gradient_quality(
                    bf16_on_lowp_state_gradients,
                    lowp_on_lowp_state_gradients,
                )
            ),
            "combined_lowp_forward_and_backward_same_weights": (
                _gradient_quality(
                    bf16_same_weight_gradients,
                    lowp_on_lowp_state_gradients,
                )
            ),
        },
        "weight_gradient_distributions": {
            "bf16_same_weights": _named_tensor_statistics(
                bf16_same_weight_gradients
            ),
            "lowp_same_weights_common_upstream": _named_tensor_statistics(
                lowp_on_lowp_state_gradients
            ),
            "lowp_native_training_signal": _named_tensor_statistics(
                lowp_gradients
            ),
        },
    }


def _optimizer_update_diagnostics(
    bf16_before: tuple[torch.Tensor, ...],
    lowp_before: tuple[torch.Tensor, ...],
    bf16_weights: TrainableWeights,
    lowp_weights: TrainableWeights,
    teacher: TrainableWeights,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, bf16_old, lowp_old, bf16_new, lowp_new, teacher_weight in zip(
        _WEIGHT_NAMES,
        bf16_before,
        lowp_before,
        bf16_weights.parameters(),
        lowp_weights.parameters(),
        teacher.parameters(),
    ):
        bf16_update = bf16_new.detach() - bf16_old
        lowp_update = lowp_new.detach() - lowp_old
        bf16_target_direction = teacher_weight.detach() - bf16_old
        lowp_target_direction = teacher_weight.detach() - lowp_old
        bf16_before_error = torch.linalg.vector_norm(bf16_target_direction)
        lowp_before_error = torch.linalg.vector_norm(lowp_target_direction)
        bf16_after_error = torch.linalg.vector_norm(
            teacher_weight.detach() - bf16_new.detach()
        )
        lowp_after_error = torch.linalg.vector_norm(
            teacher_weight.detach() - lowp_new.detach()
        )
        result[name] = {
            "bf16_vs_lowp_update": metrics(bf16_update, lowp_update),
            "bf16_update": {
                "statistics": _sampled_tensor_statistics(bf16_update),
                "alignment_with_teacher_direction": metrics(
                    bf16_target_direction,
                    bf16_update,
                ),
                "update_to_weight_norm": float(
                    torch.linalg.vector_norm(bf16_update)
                    / torch.linalg.vector_norm(bf16_old).clamp_min(1.0e-30)
                ),
                "teacher_distance_after_to_before": float(
                    bf16_after_error / bf16_before_error.clamp_min(1.0e-30)
                ),
            },
            "lowp_update": {
                "statistics": _sampled_tensor_statistics(lowp_update),
                "alignment_with_teacher_direction": metrics(
                    lowp_target_direction,
                    lowp_update,
                ),
                "update_to_weight_norm": float(
                    torch.linalg.vector_norm(lowp_update)
                    / torch.linalg.vector_norm(lowp_old).clamp_min(1.0e-30)
                ),
                "teacher_distance_after_to_before": float(
                    lowp_after_error / lowp_before_error.clamp_min(1.0e-30)
                ),
            },
            "weights_after_update": {
                "bf16_vs_lowp": metrics(bf16_new, lowp_new),
                "bf16_vs_teacher": metrics(teacher_weight, bf16_new),
                "lowp_vs_teacher": metrics(teacher_weight, lowp_new),
            },
        }
    return result


def _validation_loss(
    problem: Problem,
    x: torch.Tensor,
    target: torch.Tensor,
    weights: TrainableWeights,
) -> float:
    y, _ = _bf16_forward(problem, x, weights)
    return _relative_mse_and_gradient(y, target)[0]


def _lowp_validation_loss(
    problem: Problem,
    x: torch.Tensor,
    target: torch.Tensor,
    weights: TrainableWeights,
    qk_scales: torch.Tensor,
    *,
    forward_p_quant_mode: str,
    forward_attention_mode: str,
    attention_output_gain: float,
    attention_backward_policy: str,
) -> float:
    y, _, _ = _lowp_forward(
        problem,
        x,
        weights,
        qk_scales,
        forward_p_quant_mode=forward_p_quant_mode,
        forward_attention_mode=forward_attention_mode,
        attention_output_gain=attention_output_gain,
        attention_backward_policy=attention_backward_policy,
    )
    return _relative_mse_and_gradient(y, target)[0]


def run_convergence(
    sequence: int,
    heads: int,
    hidden: int,
    *,
    steps: int,
    learning_rate: float,
    relative_noise: float,
    seed: int,
    eval_interval: int,
    convergence_tolerance: float,
    loss_scale: float,
    forward_p_quant_mode: str = "rte",
    forward_attention_mode: str = "mxfp4",
    attention_output_gain: float = 1.0,
    attention_backward_policy: str = "chain_rule",
    diagnostic_steps: set[int] | None = None,
) -> dict[str, Any]:
    problem = build_problem(
        sequence,
        heads,
        hidden,
        seed,
        use_rope=True,
    )
    teacher = TrainableWeights(
        torch.nn.Parameter(problem.q_native_weight.float()),
        torch.nn.Parameter(problem.k_native_weight.float()),
        torch.nn.Parameter(problem.v_weight.float()),
        torch.nn.Parameter(problem.out_weight.float()),
    )
    initial = _make_student_weights(
        problem,
        seed=seed + 11,
        relative_noise=relative_noise,
    )
    bf16_weights = _clone_weights(initial)
    lowp_weights = _clone_weights(initial)
    bf16_optimizer = torch.optim.AdamW(
        bf16_weights.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1.0e-8,
        weight_decay=0.0,
    )
    lowp_optimizer = torch.optim.AdamW(
        lowp_weights.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1.0e-8,
        weight_decay=0.0,
    )

    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed + 29)
    validation_x = (
        torch.randn(
            sequence,
            hidden,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        * 0.1
    ).bfloat16()
    validation_target, _ = _bf16_forward(problem, validation_x, teacher)
    validation_target = validation_target.detach()

    qk_scales = problem.qk_scales.clone()
    checkpoints: list[dict[str, Any]] = []
    diagnostic_checkpoints: list[dict[str, Any]] = []
    diagnostic_steps = diagnostic_steps or set()
    first_gradient_quality: dict[str, Any] | None = None
    initial_bf16_validation = _validation_loss(
        problem, validation_x, validation_target, bf16_weights
    )
    initial_lowp_validation = _validation_loss(
        problem, validation_x, validation_target, lowp_weights
    )
    initial_lowp_native_validation = _lowp_validation_loss(
        problem,
        validation_x,
        validation_target,
        lowp_weights,
        qk_scales,
        forward_p_quant_mode=forward_p_quant_mode,
        forward_attention_mode=forward_attention_mode,
        attention_output_gain=attention_output_gain,
        attention_backward_policy=attention_backward_policy,
    )
    initial_lowp_teacher_floor = _lowp_validation_loss(
        problem,
        validation_x,
        validation_target,
        teacher,
        qk_scales,
        forward_p_quant_mode=forward_p_quant_mode,
        forward_attention_mode=forward_attention_mode,
        attention_output_gain=attention_output_gain,
        attention_backward_policy=attention_backward_policy,
    )
    initial_lowp_excess = max(
        initial_lowp_native_validation - initial_lowp_teacher_floor,
        0.0,
    )
    checkpoints.append(
        {
            "step": 0,
            "bf16_validation_relative_mse": initial_bf16_validation,
            "lowp_native_validation_relative_mse": (
                initial_lowp_native_validation
            ),
            "lowp_bf16_evaluator_validation_relative_mse": (
                initial_lowp_validation
            ),
            "lowp_teacher_quantization_floor": initial_lowp_teacher_floor,
            "lowp_excess_relative_mse": initial_lowp_excess,
        }
    )

    for step in range(1, steps + 1):
        bf16_optimizer.zero_grad(set_to_none=True)
        lowp_optimizer.zero_grad(set_to_none=True)

        # A fresh deterministic activation batch each step prevents the two
        # very large students from merely memorizing one 4096-token matrix.
        train_x = (
            torch.randn(
                sequence,
                hidden,
                device="cuda",
                dtype=torch.float32,
                generator=generator,
            )
            * 0.1
        ).bfloat16()
        train_target, _ = _bf16_forward(problem, train_x, teacher)
        train_target = train_target.detach()

        bf16_y, bf16_state = _bf16_forward(problem, train_x, bf16_weights)
        bf16_loss, bf16_dy = _relative_mse_and_gradient(bf16_y, train_target)
        bf16_dy = (bf16_dy.float() * loss_scale).bfloat16()
        bf16_gradients = _unscale_gradients(
            _bf16_backward(problem, bf16_state, bf16_dy),
            loss_scale,
        )

        lowp_y, lowp_state, qk_scales = _lowp_forward(
            problem,
            train_x,
            lowp_weights,
            qk_scales,
            forward_p_quant_mode=forward_p_quant_mode,
            forward_attention_mode=forward_attention_mode,
            attention_output_gain=attention_output_gain,
            attention_backward_policy=attention_backward_policy,
        )
        lowp_loss, lowp_dy = _relative_mse_and_gradient(lowp_y, train_target)
        lowp_dy = (lowp_dy.float() * loss_scale).bfloat16()
        lowp_gradients = _unscale_gradients(
            _lowp_backward(problem, lowp_state, lowp_dy),
            loss_scale,
        )
        if first_gradient_quality is None:
            lowp_shared_upstream_gradients = _unscale_gradients(
                _lowp_backward(
                    problem,
                    lowp_state,
                    bf16_dy,
                ),
                loss_scale,
            )
            first_gradient_quality = {
                "native_training_signal": _gradient_quality(
                    bf16_gradients,
                    lowp_gradients,
                ),
                "shared_upstream_gradient": _gradient_quality(
                    bf16_gradients,
                    lowp_shared_upstream_gradients,
                ),
            }

        diagnostic: dict[str, Any] | None = None
        bf16_before: tuple[torch.Tensor, ...] | None = None
        lowp_before: tuple[torch.Tensor, ...] | None = None
        if step in diagnostic_steps:
            diagnostic = _pre_update_diagnostics(
                problem,
                train_x,
                train_target,
                bf16_weights,
                lowp_weights,
                bf16_gradients,
                lowp_gradients,
                bf16_dy,
                lowp_dy,
                lowp_y,
                lowp_state,
                loss_scale,
            )
            bf16_before = _snapshot_weights(bf16_weights)
            lowp_before = _snapshot_weights(lowp_weights)

        _assign_gradients(bf16_weights, bf16_gradients)
        _assign_gradients(lowp_weights, lowp_gradients)
        bf16_gradient_norm = torch.nn.utils.clip_grad_norm_(
            bf16_weights.parameters(),
            1.0,
        )
        lowp_gradient_norm = torch.nn.utils.clip_grad_norm_(
            lowp_weights.parameters(),
            1.0,
        )
        bf16_optimizer.step()
        lowp_optimizer.step()

        if diagnostic is not None:
            assert bf16_before is not None and lowp_before is not None
            diagnostic["step"] = step
            diagnostic["gradient_clipping"] = {
                "maximum_norm": 1.0,
                "bf16_preclip_total_norm": float(bf16_gradient_norm),
                "lowp_preclip_total_norm": float(lowp_gradient_norm),
                "bf16_was_clipped": bool(bf16_gradient_norm > 1.0),
                "lowp_was_clipped": bool(lowp_gradient_norm > 1.0),
            }
            diagnostic["optimizer_updates"] = _optimizer_update_diagnostics(
                bf16_before,
                lowp_before,
                bf16_weights,
                lowp_weights,
                teacher,
            )
            diagnostic_checkpoints.append(diagnostic)
            print(
                f"step={step:4d} captured gradient/update diagnostics",
                flush=True,
            )

        if step == 1 or step == steps or step % eval_interval == 0:
            bf16_validation = _validation_loss(
                problem,
                validation_x,
                validation_target,
                bf16_weights,
            )
            lowp_bf16_validation = _validation_loss(
                problem,
                validation_x,
                validation_target,
                lowp_weights,
            )
            lowp_native_validation = _lowp_validation_loss(
                problem,
                validation_x,
                validation_target,
                lowp_weights,
                qk_scales,
                forward_p_quant_mode=forward_p_quant_mode,
                forward_attention_mode=forward_attention_mode,
                attention_output_gain=attention_output_gain,
                attention_backward_policy=attention_backward_policy,
            )
            lowp_teacher_floor = _lowp_validation_loss(
                problem,
                validation_x,
                validation_target,
                teacher,
                qk_scales,
                forward_p_quant_mode=forward_p_quant_mode,
                forward_attention_mode=forward_attention_mode,
                attention_output_gain=attention_output_gain,
                attention_backward_policy=attention_backward_policy,
            )
            lowp_excess = max(
                lowp_native_validation - lowp_teacher_floor,
                0.0,
            )
            checkpoints.append(
                {
                    "step": step,
                    "bf16_train_relative_mse": bf16_loss,
                    "lowp_train_relative_mse": lowp_loss,
                    "bf16_validation_relative_mse": bf16_validation,
                    "lowp_native_validation_relative_mse": (
                        lowp_native_validation
                    ),
                    "lowp_bf16_evaluator_validation_relative_mse": (
                        lowp_bf16_validation
                    ),
                    "lowp_teacher_quantization_floor": lowp_teacher_floor,
                    "lowp_excess_relative_mse": lowp_excess,
                    "lowp_to_bf16_native_validation_ratio": (
                        lowp_native_validation / max(bf16_validation, 1.0e-30)
                    ),
                }
            )
            print(
                f"step={step:4d} train bf16={bf16_loss:.6e} "
                f"lowp={lowp_loss:.6e} validation bf16={bf16_validation:.6e} "
                f"lowp-native={lowp_native_validation:.6e} "
                f"lowp-bf16-eval={lowp_bf16_validation:.6e}",
                flush=True,
            )

    final = checkpoints[-1]
    bf16_reduction = (
        initial_bf16_validation
        / max(final["bf16_validation_relative_mse"], 1.0e-30)
    )
    lowp_reduction = (
        initial_lowp_native_validation
        / max(final["lowp_native_validation_relative_mse"], 1.0e-30)
    )
    ratio = final["lowp_to_bf16_native_validation_ratio"]
    bf16_remaining = (
        final["bf16_validation_relative_mse"]
        / max(initial_bf16_validation, 1.0e-30)
    )
    lowp_excess_remaining = (
        final["lowp_excess_relative_mse"]
        / max(initial_lowp_excess, 1.0e-30)
    )
    normalized_remaining_ratio = (
        lowp_excess_remaining / max(bf16_remaining, 1.0e-30)
    )
    bf16_absolute_improvement = (
        initial_bf16_validation - final["bf16_validation_relative_mse"]
    )
    lowp_absolute_improvement = (
        initial_lowp_native_validation
        - final["lowp_native_validation_relative_mse"]
    )
    return {
        "contract": {
            "task": "streaming teacher/student dense causal attention sublayer",
            "language_model_convergence": False,
            "forward": "projection-native NVFP4 QKV + FP4 FA4 + NVFP4 output",
            "backward": "adaptive FP4 QK + FP8 dP/dV",
            "learned_weights": ["Q", "K", "V", "output"],
            "validation": (
                "native-route loss plus a shared BF16 evaluator on held-out "
                "synthetic activations"
            ),
            "timing_included": False,
            "weight_repacking": "refreshed after every optimizer update",
        },
        "shape": {
            "batch": 1,
            "sequence": sequence,
            "heads": heads,
            "hidden": hidden,
            "qk_head_dim": QK_DIM,
            "v_head_dim": V_DIM,
        },
        "configuration": {
            "steps": steps,
            "learning_rate": learning_rate,
            "relative_initial_weight_noise": relative_noise,
            "seed": seed,
            "eval_interval": eval_interval,
            "optimizer": "AdamW(beta1=0.9,beta2=0.95,weight_decay=0)",
            "gradient_clip_norm": 1.0,
            "loss_scale": loss_scale,
            "forward_p_quant_mode": forward_p_quant_mode,
            "forward_attention_mode": forward_attention_mode,
            "attention_output_gain": attention_output_gain,
            "attention_backward_policy": attention_backward_policy,
            "convergence_tolerance": convergence_tolerance,
            "diagnostic_steps": sorted(diagnostic_steps),
        },
        "first_step_weight_gradient_quality": first_gradient_quality,
        "checkpoints": checkpoints,
        "diagnostic_checkpoints": diagnostic_checkpoints,
        "summary": {
            "bf16_validation_loss_reduction": bf16_reduction,
            "lowp_native_validation_loss_reduction": lowp_reduction,
            "lowp_to_bf16_final_native_validation_ratio": ratio,
            "lowp_to_bf16_final_shared_evaluator_ratio": (
                final["lowp_bf16_evaluator_validation_relative_mse"]
                / max(final["bf16_validation_relative_mse"], 1.0e-30)
            ),
            "bf16_normalized_loss_remaining": bf16_remaining,
            "lowp_normalized_excess_loss_remaining": lowp_excess_remaining,
            "lowp_to_bf16_normalized_remaining_ratio": (
                normalized_remaining_ratio
            ),
            "bf16_absolute_loss_improvement": bf16_absolute_improvement,
            "lowp_absolute_loss_improvement": lowp_absolute_improvement,
            "lowp_to_bf16_absolute_improvement_ratio": (
                lowp_absolute_improvement
                / max(bf16_absolute_improvement, 1.0e-30)
            ),
            "within_configured_tolerance": bool(
                math.isfinite(normalized_remaining_ratio)
                and normalized_remaining_ratio <= convergence_tolerance
                and lowp_absolute_improvement > 0.0
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", default="4096x24x3072")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=2.0e-5)
    parser.add_argument("--relative-noise", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=2026081417)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--convergence-tolerance", type=float, default=1.25)
    parser.add_argument("--loss-scale", type=float, default=2.0**20)
    parser.add_argument(
        "--forward-p-quant-mode",
        choices=("rte", "encode", "decode"),
        default="rte",
        help="MXFP4 probability block-scale rounding mode",
    )
    parser.add_argument(
        "--forward-attention-mode",
        choices=(
            "mxfp4",
            "qk_fp4_v_bf16_control",
            "bf16_on_lowp_qkv",
        ),
        default="mxfp4",
        help="attention forward route; BF16 V/PV is a diagnostic control",
    )
    parser.add_argument(
        "--attention-output-gain",
        type=float,
        default=1.0,
        help=(
            "diagnostic gain folded into the NVFP4 output-projection input "
            "scale; 1.0 preserves the production route"
        ),
    )
    parser.add_argument(
        "--attention-backward-policy",
        choices=("chain_rule", "identity_ste", "delta_corrected_ste"),
        default="chain_rule",
        help="backward surrogate used with a non-unit attention output gain",
    )
    parser.add_argument(
        "--diagnostic-steps",
        default="",
        help=(
            "comma-separated pre-update steps at which to decompose forward, "
            "gradient, and Adam-update error"
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    shapes = parse_shapes(args.shape)
    if len(shapes) != 1:
        raise ValueError("--shape accepts exactly one sequence x heads x hidden triple")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("evaluation requires exactly one visible GPU")
    torch.cuda.set_device(0)
    diagnostic_steps = {
        int(raw)
        for raw in args.diagnostic_steps.split(",")
        if raw.strip()
    }
    if any(step < 1 or step > args.steps for step in diagnostic_steps):
        raise ValueError("diagnostic steps must be within [1, --steps]")
    if not math.isfinite(args.attention_output_gain) or args.attention_output_gain <= 0:
        raise ValueError("--attention-output-gain must be finite and positive")
    with torch.no_grad():
        result = run_convergence(
            *shapes[0],
            steps=args.steps,
            learning_rate=args.learning_rate,
            relative_noise=args.relative_noise,
            seed=args.seed,
            eval_interval=args.eval_interval,
            convergence_tolerance=args.convergence_tolerance,
            loss_scale=args.loss_scale,
            forward_p_quant_mode=args.forward_p_quant_mode,
            forward_attention_mode=args.forward_attention_mode,
            attention_output_gain=args.attention_output_gain,
            attention_backward_policy=args.attention_backward_policy,
            diagnostic_steps=diagnostic_steps,
        )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
