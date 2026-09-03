#!/usr/bin/env python3
"""Localize numerical error in the current D64 low-precision Llama path."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from tk_fa4 import (
    b300_prepare_nvfp4_projection_operand,
    b300_prepare_nvfp4_projection_weight,
    b300_prepare_nvfp4_projection_operand_scaled,
    b300_project_dout_unified_lowp_nvfp4,
    b300_project_nvfp4,
    b300_project_qkv_gqa_d64_paired_unified_lowp_nvfp4,
)
from tk_fa4.lowp_fa4_bwd import benchmark_llama12b_e2e as benchmark
from tk_fa4.lowp_fa4_bwd.benchmark_llama12b_e2e import (
    Config,
    Llama12B,
    LowpAttentionRuntime,
    _load_forward,
    _make_rope,
)


def _prepare_weight(
    runtime: LowpAttentionRuntime,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prepare = (
        b300_prepare_nvfp4_projection_weight
        if runtime.projection_weight_scale_2d
        else b300_prepare_nvfp4_projection_operand
    )
    return tuple(prepare(weight))


def _sample(tensor: torch.Tensor, limit: int = 1 << 20) -> torch.Tensor:
    flat = tensor.detach().reshape(-1)
    stride = max(1, (flat.numel() + limit - 1) // limit)
    return flat[::stride][:limit].float()


def _metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float | bool]:
    ref = _sample(reference)
    out = _sample(actual)
    ref_norm = torch.linalg.vector_norm(ref)
    out_norm = torch.linalg.vector_norm(out)
    denominator = (ref_norm * out_norm).clamp_min(1.0e-30)
    difference = out - ref
    gain_denominator = torch.dot(out, out).clamp_min(1.0e-30)
    return {
        "cosine": float(torch.dot(ref, out) / denominator),
        "relative_l2": float(
            torch.linalg.vector_norm(difference) / ref_norm.clamp_min(1.0e-30)
        ),
        "norm_ratio": float(out_norm / ref_norm.clamp_min(1.0e-30)),
        "least_squares_gain": float(torch.dot(ref, out) / gain_denominator),
        "finite": bool(torch.isfinite(out).all()),
    }


def _hidden_trace(
    model: Llama12B,
    tokens: torch.Tensor,
) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
    hidden = F.embedding(tokens, model.embedding)
    records: list[dict[str, torch.Tensor]] = []
    for layer in model.layers:
        attention_norm = layer.attention_norm(hidden)
        attention_branch = layer.attention(attention_norm)
        after_attention = hidden + attention_branch
        ffn_norm = layer.ffn_norm(after_attention)
        mlp_branch = layer.mlp(ffn_norm)
        hidden = after_attention + mlp_branch
        records.append(
            {
                "attention_norm": attention_norm.detach(),
                "attention_branch": attention_branch.detach(),
                "after_attention": after_attention.detach(),
                "ffn_norm": ffn_norm.detach(),
                "mlp_branch": mlp_branch.detach(),
                "layer_output": hidden.detach(),
            }
        )
    return model.final_norm(hidden), records


def _first_layer_forward_decomposition(
    config: Config,
    runtime: LowpAttentionRuntime,
    normalized_input: torch.Tensor,
    lowp_model: Llama12B,
) -> tuple[dict[str, Any], dict[str, torch.Tensor | Any]]:
    weights = lowp_model.layers[0].attention.weights
    qk_scales = lowp_model.layers[0].attention.qk_scales
    rows = normalized_input.reshape(config.sequence, config.hidden).contiguous()
    exact_q = F.linear(rows, weights.q).reshape(
        1, config.sequence, config.q_heads, config.head_dim
    )
    exact_k = F.linear(rows, weights.k).reshape(
        1, config.sequence, config.kv_heads, config.head_dim
    )
    exact_v = F.linear(rows, weights.v).reshape(
        1, config.sequence, config.kv_heads, config.head_dim
    )
    exact_q = benchmark._apply_pair_rope(exact_q, *runtime.rope)
    exact_k = benchmark._apply_pair_rope(exact_k, *runtime.rope)

    qkv_weight = torch.cat((weights.q, weights.k, weights.v), dim=0).contiguous()
    qkv = b300_project_qkv_gqa_d64_paired_unified_lowp_nvfp4(
        tuple(b300_prepare_nvfp4_projection_operand(rows)),
        _prepare_weight(runtime, qkv_weight),
        qk_scales,
        runtime.paired_rope,
        batch=1,
        seqlen=config.sequence,
        q_heads=config.q_heads,
        kv_heads=config.kv_heads,
        store_bf16=True,
        publish_fp8_backward=True,
        interleave_causal_kv=bool(
            runtime.forward_topology.get("causal_interleaved_kv", False)
        ),
        v_mxfp4_scale_2d=runtime.v_mxfp4_scale_2d,
    )
    assert qkv.q is not None and qkv.k is not None and qkv.v is not None
    assert qkv.q_backward_fp8 is not None
    assert qkv.k_backward_fp8 is not None
    assert qkv.v_backward_fp8 is not None

    exact_attention = benchmark.flash_attn_func(
        exact_q, exact_k, exact_v, causal=True
    )
    projected_qkv_result = benchmark.flash_attn_func(
        qkv.q, qkv.k, qkv.v, causal=True, return_lse=True
    )
    if isinstance(exact_attention, tuple):
        exact_attention = exact_attention[0]
    if not isinstance(projected_qkv_result, tuple) or len(projected_qkv_result) != 2:
        raise RuntimeError("projected-QKV reference did not return LSE")
    projected_qkv_attention, projected_qkv_lse = projected_qkv_result

    lowp_attention = torch.empty_like(exact_attention)
    lowp_lse = torch.empty(
        1,
        config.q_heads,
        1,
        config.sequence,
        device="cuda",
        dtype=torch.float32,
    )
    benchmark.activate_forward_route(
        str(runtime.forward_topology["route"])
    )
    benchmark._run_lowp_forward_attention(
        runtime,
        qkv,
        lowp_attention,
        lowp_lse,
    )

    sequence_slices = {
        "tokens_0_127": slice(0, 128),
        "tokens_128_511": slice(128, 512),
        "tokens_512_1023": slice(512, 1024),
        "tokens_1024_2047": slice(1024, 2048),
        "tokens_2048_4095": slice(2048, config.sequence),
    }
    attention_by_sequence = {
        name: _metrics(
            projected_qkv_attention[:, token_slice],
            lowp_attention[:, token_slice],
        )
        for name, token_slice in sequence_slices.items()
        if token_slice.start < config.sequence
    }
    query_tile = torch.arange(config.sequence, device="cuda") // 128
    attention_by_query_stage = {
        f"stage_{stage}": _metrics(
            projected_qkv_attention[:, (query_tile & 1) == stage],
            lowp_attention[:, (query_tile & 1) == stage],
        )
        for stage in (0, 1)
    }

    exact_branch = F.linear(
        exact_attention.reshape(config.sequence, config.q_width),
        weights.o,
    )
    exact_projection_on_lowp_attention = F.linear(
        lowp_attention.reshape(config.sequence, config.q_width),
        weights.o,
    )
    lowp_branch = b300_project_nvfp4(
        tuple(
            b300_prepare_nvfp4_projection_operand(
                lowp_attention.reshape(config.sequence, config.q_width)
            )
        ),
        _prepare_weight(runtime, weights.o),
    )

    result = {
        "qkv_projection": {
            "q": _metrics(exact_q, qkv.q),
            "k": _metrics(exact_k, qkv.k),
            "v": _metrics(exact_v, qkv.v),
        },
        "qkv_projection_effect_on_exact_attention": _metrics(
            exact_attention, projected_qkv_attention
        ),
        "fp4_attention_given_same_projected_qkv": _metrics(
            projected_qkv_attention, lowp_attention
        ),
        "fp4_attention_by_sequence": attention_by_sequence,
        "fp4_attention_by_query_stage": attention_by_query_stage,
        "fp4_lse_given_same_projected_qkv": _metrics(
            projected_qkv_lse.unsqueeze(2), lowp_lse
        ),
        "nvfp4_output_projection_given_same_attention": _metrics(
            exact_projection_on_lowp_attention, lowp_branch
        ),
        "complete_attention_branch": _metrics(exact_branch, lowp_branch),
    }
    state: dict[str, torch.Tensor | Any] = {
        "qkv": qkv,
        "attention_output": lowp_attention,
        "lse": lowp_lse,
        "out_weight": weights.o,
    }
    return result, state


def _capture_first_attention_upstream(
    model: Llama12B,
    tokens: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    captured: dict[str, torch.Tensor] = {}

    def capture(_: Any, __: Any, output: torch.Tensor) -> None:
        output.register_hook(
            lambda gradient: captured.setdefault("gradient", gradient.detach())
        )

    handle = model.layers[0].attention.register_forward_hook(capture)
    model.zero_grad(set_to_none=True)
    logits = model(tokens)
    loss = F.cross_entropy(
        logits.reshape(-1, model.config.vocab),
        targets.reshape(-1),
        reduction="mean",
    )
    loss.backward()
    handle.remove()
    if "gradient" not in captured:
        raise RuntimeError("failed to capture the first attention upstream gradient")
    return captured["gradient"], float(loss.detach())


def _first_layer_backward_decomposition(
    config: Config,
    runtime: LowpAttentionRuntime,
    upstream: torch.Tensor,
    state: dict[str, torch.Tensor | Any],
) -> dict[str, Any]:
    qkv = state["qkv"]
    attention_output = state["attention_output"]
    lse = state["lse"]
    out_weight = state["out_weight"]
    assert isinstance(attention_output, torch.Tensor)
    assert isinstance(lse, torch.Tensor)
    assert isinstance(out_weight, torch.Tensor)
    assert qkv.q_backward_fp8 is not None
    assert qkv.k_backward_fp8 is not None
    assert qkv.v_backward_fp8 is not None

    upstream_matrix = upstream.reshape(config.sequence, config.hidden).contiguous()
    exact_dout_scaled = torch.mm(upstream_matrix, out_weight).reshape_as(
        attention_output
    ).float() * runtime.loss_scale

    runtime.backward.reset()
    bundle = b300_project_dout_unified_lowp_nvfp4(
        tuple(
            b300_prepare_nvfp4_projection_operand_scaled(
                upstream_matrix,
                runtime.loss_scale,
            )
        ),
        _prepare_weight(runtime, out_weight.T.contiguous()),
        attention_output,
        lse,
        batch=1,
        seqlen=config.sequence,
        heads=config.q_heads,
        store_bf16=True,
        publish_fp8_backward=True,
        publish_stats=True,
        stats_workspace=runtime.backward.workspace_torch,
    )
    assert bundle.dout is not None and bundle.dout_backward_fp8 is not None
    runtime.bind_backward_inputs(
        qkv.q_backward_fp8,
        qkv.k_backward_fp8,
        qkv.v_backward_fp8,
        bundle.dout_backward_fp8,
    )
    runtime.backward.run(reset=False)
    current_dq = runtime.backward.dq.clone()
    current_dk = runtime.backward.dk.clone()
    current_dv = runtime.backward.dv.clone()

    q_ref = (qkv.q_backward_fp8.float() * 0.25).bfloat16().requires_grad_(True)
    k_ref = (qkv.k_backward_fp8.float() * 0.25).bfloat16().requires_grad_(True)
    v_ref = (qkv.v_backward_fp8.float() * 0.25).bfloat16().requires_grad_(True)
    dout_ref = (bundle.dout_backward_fp8.float() * 0.25).bfloat16()
    exact_result = benchmark.flash_attn_func(
        q_ref,
        k_ref,
        v_ref,
        causal=True,
        return_lse=True,
    )
    if not isinstance(exact_result, tuple) or len(exact_result) != 2:
        raise RuntimeError("exact backward reference did not return LSE")
    exact_output, exact_lse = exact_result
    exact_output.backward(dout_ref)
    assert q_ref.grad is not None and k_ref.grad is not None and v_ref.grad is not None

    # Run the same low-precision backward kernel with statistics published
    # from the exact attention state of the decoded FP8 operands.  This keeps
    # Q/K/V/dO fixed and removes forward output/LSE error from the comparison.
    runtime.backward.reset()
    exact_bundle = b300_project_dout_unified_lowp_nvfp4(
        tuple(
            b300_prepare_nvfp4_projection_operand_scaled(
                upstream_matrix,
                runtime.loss_scale,
            )
        ),
        _prepare_weight(runtime, out_weight.T.contiguous()),
        exact_output.detach(),
        exact_lse.detach().unsqueeze(2),
        batch=1,
        seqlen=config.sequence,
        heads=config.q_heads,
        store_bf16=True,
        publish_fp8_backward=True,
        publish_stats=True,
        stats_workspace=runtime.backward.workspace_torch,
    )
    assert exact_bundle.dout is not None
    assert exact_bundle.dout_backward_fp8 is not None
    runtime.bind_backward_inputs(
        qkv.q_backward_fp8,
        qkv.k_backward_fp8,
        qkv.v_backward_fp8,
        exact_bundle.dout_backward_fp8,
    )
    runtime.backward.run(reset=False)

    for _ in range(3):
        runtime.backward.reset()
        runtime.backward.run(reset=False)
    timings_us: list[float] = []
    for _ in range(9):
        runtime.backward.reset()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        runtime.backward.run(reset=False)
        end.record()
        end.synchronize()
        timings_us.append(float(start.elapsed_time(end)) * 1000.0)

    def attention_gradient_metrics(
        dq: torch.Tensor,
        dk: torch.Tensor,
        dv: torch.Tensor,
        correction: float = 1.0,
    ) -> dict[str, Any]:
        # The fixed-scale E4M3 dO publication leaves all three outputs x4.
        decode = 0.25 * correction
        return {
            "dq": _metrics(q_ref.grad, dq.float() * decode),
            "dk": _metrics(k_ref.grad, dk.float() * decode),
            "dv": _metrics(v_ref.grad, dv.float() * decode),
        }

    represented_probability_mass = torch.exp(
        exact_lse.detach() - lse.detach().squeeze(2)
    ).float()
    mass_mean = float(represented_probability_mass.mean())
    mass_correction = 1.0 / mass_mean

    return {
        "nvfp4_dout_projection": _metrics(exact_dout_scaled, bundle.dout),
        "fixed_scale_fp8_dout_publication": _metrics(
            bundle.dout,
            bundle.dout_backward_fp8.float() * 0.25,
        ),
        "exact_state_dout_publication_matches_current": _metrics(
            bundle.dout_backward_fp8.float(),
            exact_bundle.dout_backward_fp8.float(),
        ),
        "current_forward_statistics": attention_gradient_metrics(
            current_dq,
            current_dk,
            current_dv,
        ),
        "current_forward_statistics_mass_corrected": attention_gradient_metrics(
            current_dq,
            current_dk,
            current_dv,
            mass_correction,
        ),
        "represented_probability_mass": {
            "mean": mass_mean,
            "median": float(represented_probability_mass.median()),
            "p10": float(torch.quantile(represented_probability_mass, 0.10)),
            "p90": float(torch.quantile(represented_probability_mass, 0.90)),
            "minimum": float(represented_probability_mass.min()),
            "maximum": float(represented_probability_mass.max()),
            "inverse_mean_correction": mass_correction,
        },
        "exact_decoded_forward_statistics": attention_gradient_metrics(
            runtime.backward.dq,
            runtime.backward.dk,
            runtime.backward.dv,
        ),
        "attention_backward_median_us": statistics.median(timings_us),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--loss-scale", type=float, default=2.0**16)
    parser.add_argument(
        "--backward-exp2-degree", type=int, choices=(1, 2), default=2
    )
    parser.add_argument(
        "--backward-exp2-period", type=int, choices=tuple(range(17)), default=2
    )
    parser.add_argument(
        "--backward-fp8-ds-lift",
        type=int,
        choices=(16, 32, 64, 128, 256),
        default=16,
    )
    parser.add_argument("--backward-reuse-quantized-p", action="store_true")
    parser.add_argument("--q-quant-scale", type=float)
    parser.add_argument("--k-quant-scale", type=float)
    parser.add_argument(
        "--projection-weight-scaling",
        choices=("1d", "2d"),
        default="2d",
    )
    parser.add_argument(
        "--v-mxfp4-scaling",
        choices=("1d", "2d"),
        default="2d",
    )
    parser.add_argument("--adaptive-qk-weight-scales", action="store_true")
    parser.add_argument(
        "--forward-extension",
        type=Path,
        default=Path(
            "/tmp/_C_tk_gb200_causal_s4096_h32_d64."
            "cpython-312-aarch64-linux-gnu.so"
        ),
    )
    parser.add_argument(
        "--forward-module", default="_C_tk_gb200_causal_s4096_h32_d64"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("expose exactly one GPU to the numerical diagnostic")
    if args.layers <= 0 or args.layers > 16:
        raise ValueError("--layers must be in [1,16]")
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    config = Config(layers=args.layers)
    extension, topology = _load_forward(
        args.forward_extension, args.forward_module, config
    )
    rope = _make_rope(config.sequence, config.head_dim)
    runtime = LowpAttentionRuntime(
        config,
        rope,
        forward_extension=extension,
        forward_topology=topology,
        loss_scale=args.loss_scale,
        gradient_global_scale=2.0**-8,
        projection_dgrad="bf16",
        backward_exp2_degree=args.backward_exp2_degree,
        backward_exp2_period=args.backward_exp2_period,
        backward_fp8_ds_lift=args.backward_fp8_ds_lift,
        backward_reuse_quantized_p=args.backward_reuse_quantized_p,
        projection_weight_scale_2d=(
            args.projection_weight_scaling == "2d"
        ),
        v_mxfp4_scale_2d=(args.v_mxfp4_scaling == "2d"),
        adaptive_qk_weight_scales=args.adaptive_qk_weight_scales,
    )
    if args.q_quant_scale is not None:
        if not math.isfinite(args.q_quant_scale) or args.q_quant_scale <= 0.0:
            raise ValueError("--q-quant-scale must be positive and finite")
        runtime.qk_scales[:, :, 0] = args.q_quant_scale
    if args.k_quant_scale is not None:
        if not math.isfinite(args.k_quant_scale) or args.k_quant_scale <= 0.0:
            raise ValueError("--k-quant-scale must be positive and finite")
        runtime.qk_scales[:, :, 1] = args.k_quant_scale

    bf16_model = Llama12B(config, rope, None)
    lowp_model = Llama12B(config, rope, runtime)
    lowp_model.load_state_dict(bf16_model.state_dict())
    if runtime.adaptive_qk_weight_scales:
        for layer in lowp_model.layers:
            layer.attention.refresh_qk_quant_scales()
    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed + 1)
    tokens = torch.randint(
        config.vocab,
        (1, config.sequence),
        generator=generator,
        device="cuda",
    )
    targets = torch.roll(tokens, shifts=-1, dims=1)

    with torch.no_grad():
        bf16_hidden, bf16_records = _hidden_trace(bf16_model, tokens)
        lowp_hidden, lowp_records = _hidden_trace(lowp_model, tokens)
        layerwise = []
        for index, (reference, actual) in enumerate(
            zip(bf16_records, lowp_records)
        ):
            layerwise.append(
                {
                    "layer": index,
                    **{
                        name: _metrics(reference[name], actual[name])
                        for name in reference
                    },
                }
            )
        sampled_bf16_logits = F.linear(
            bf16_hidden[:, :16].reshape(16, config.hidden),
            bf16_model.embedding[:1024],
        )
        sampled_lowp_logits = F.linear(
            lowp_hidden[:, :16].reshape(16, config.hidden),
            lowp_model.embedding[:1024],
        )
        first_layer_forward, first_layer_state = (
            _first_layer_forward_decomposition(
                config,
                runtime,
                bf16_records[0]["attention_norm"],
                lowp_model,
            )
        )

    upstream, bf16_loss = _capture_first_attention_upstream(
        bf16_model, tokens, targets
    )
    first_layer_backward = _first_layer_backward_decomposition(
        config,
        runtime,
        upstream,
        first_layer_state,
    )
    torch.cuda.synchronize()

    result = {
        "configuration": {
            **config.__dict__,
            "batch": 1,
            "seed": args.seed,
            "loss_scale": args.loss_scale,
            "backward_exp2_degree": args.backward_exp2_degree,
            "backward_exp2_period": args.backward_exp2_period,
            "backward_fp8_ds_lift": args.backward_fp8_ds_lift,
            "backward_reuse_quantized_p": args.backward_reuse_quantized_p,
            "q_quant_scale": float(
                lowp_model.layers[0].attention.qk_scales[0, 0, 0]
            ),
            "k_quant_scale": float(
                lowp_model.layers[0].attention.qk_scales[0, 0, 1]
            ),
            "projection_weight_scaling": args.projection_weight_scaling,
            "v_mxfp4_scaling": args.v_mxfp4_scaling,
            "adaptive_qk_weight_scales": args.adaptive_qk_weight_scales,
            "forward_topology": topology,
        },
        "bf16_loss": bf16_loss,
        "sampled_logits": _metrics(sampled_bf16_logits, sampled_lowp_logits),
        "final_hidden": _metrics(bf16_hidden, lowp_hidden),
        "layerwise": layerwise,
        "first_layer_forward_decomposition": first_layer_forward,
        "first_layer_backward_decomposition": first_layer_backward,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
