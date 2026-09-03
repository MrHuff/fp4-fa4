#!/usr/bin/env python3
"""Validate D128 projection QDQ, fused GEMM, and FA4 publications."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

from tk_fa4 import (
    b300_pair_interleave_gqa_d128_qk_projection_weights,
    b300_prepare_nvfp4_projection_operand,
    b300_prepare_nvfp4_projection_weight,
    b300_project_qkv_gqa_d128_unified_lowp_nvfp4,
    b300_stack_gqa_d128_qkv_projection_weights,
)
from tk_fa4.lowp_fa4_bwd.projection_quantization_reference import (
    decode_native_mxfp4_v,
    decode_native_nvfp4_qk,
    decode_prepared_nvfp4_matrix,
    tensor_error_metrics,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=11)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _byte_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return bool(torch.equal(left.view(torch.uint8), right.view(torch.uint8)))


def _time_rotated(
    functions: dict[str, Callable[[], Any]],
    *,
    warmups: int,
    samples: int,
) -> dict[str, float]:
    names = list(functions)
    for iteration in range(warmups):
        for offset in range(len(names)):
            functions[names[(iteration + offset) % len(names)]]()
    torch.cuda.synchronize()
    elapsed: dict[str, list[float]] = {name: [] for name in names}
    for iteration in range(samples):
        for offset in range(len(names)):
            name = names[(iteration + offset) % len(names)]
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            functions[name]()
            end.record()
            end.synchronize()
            elapsed[name].append(start.elapsed_time(end))
    return {name: statistics.median(values) for name, values in elapsed.items()}


def _projection_components(
    matrix: torch.Tensor,
    *,
    sequence: int,
    q_heads: int,
    kv_heads: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    depth = 128
    q_width = q_heads * depth
    kv_width = kv_heads * depth
    return (
        matrix[:, :q_width].reshape(1, sequence, q_heads, depth),
        matrix[:, q_width : q_width + kv_width].reshape(
            1, sequence, kv_heads, depth
        ),
        matrix[:, q_width + kv_width :].reshape(
            1, sequence, kv_heads, depth
        ),
    )


def _component_metrics(
    reference: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    candidate: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, dict[str, float]]:
    return {
        name: tensor_error_metrics(expected, actual)
        for name, expected, actual in zip(
            ("q", "k", "v"),
            reference,
            candidate,
            strict=True,
        )
    }


@torch.no_grad()
def main() -> None:
    arguments = _parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("validator requires exactly one visible GPU")
    if arguments.sequence % 128 or arguments.hidden % 128:
        raise ValueError("sequence and hidden must be divisible by 128")
    if arguments.q_heads % arguments.kv_heads:
        raise ValueError("q-heads must be divisible by kv-heads")

    torch.manual_seed(arguments.seed)
    torch.cuda.manual_seed_all(arguments.seed)
    depth = 128
    activation = torch.randn(
        arguments.sequence,
        arguments.hidden,
        device="cuda",
        dtype=torch.float32,
    )
    activation = (
        activation
        * torch.rsqrt(activation.square().mean(dim=1, keepdim=True) + 1.0e-5)
    ).bfloat16()
    q_weight = (
        torch.randn(
            arguments.q_heads * depth,
            arguments.hidden,
            device="cuda",
        )
        * 0.02
    ).bfloat16()
    k_weight = (
        torch.randn(
            arguments.kv_heads * depth,
            arguments.hidden,
            device="cuda",
        )
        * 0.02
    ).bfloat16()
    v_weight = torch.randn_like(k_weight.float()).mul_(0.02).bfloat16()
    q_weight, k_weight = b300_pair_interleave_gqa_d128_qk_projection_weights(
        q_weight,
        k_weight,
    )
    qkv_weight = b300_stack_gqa_d128_qkv_projection_weights(
        q_weight,
        k_weight,
        v_weight,
    )

    activation_operand = tuple(
        b300_prepare_nvfp4_projection_operand(activation)
    )
    # Learned weights retain a shared 16x16 scale in every arm.
    weight_operand = tuple(b300_prepare_nvfp4_projection_weight(qkv_weight))
    weight_transpose_operand = tuple(
        b300_prepare_nvfp4_projection_weight(qkv_weight.T.contiguous())
    )
    activation_qdq = decode_prepared_nvfp4_matrix(activation_operand)
    weight_qdq = decode_prepared_nvfp4_matrix(weight_operand)
    weight_transpose_qdq = decode_prepared_nvfp4_matrix(
        weight_transpose_operand
    )
    weight_transpose_bitwise_equal = torch.equal(
        weight_qdq.T,
        weight_transpose_qdq,
    )
    if not weight_transpose_bitwise_equal:
        raise RuntimeError(
            "2D NVFP4 learned-weight quantization is not transpose consistent"
        )
    dense_bf16_matrix = F.linear(activation, qkv_weight)
    dense_qdq_matrix = F.linear(
        activation_qdq.bfloat16(),
        weight_qdq.bfloat16(),
    )
    dense_bf16 = _projection_components(
        dense_bf16_matrix,
        sequence=arguments.sequence,
        q_heads=arguments.q_heads,
        kv_heads=arguments.kv_heads,
    )
    dense_qdq = _projection_components(
        dense_qdq_matrix,
        sequence=arguments.sequence,
        q_heads=arguments.q_heads,
        kv_heads=arguments.kv_heads,
    )
    qk_scales = torch.zeros(
        1,
        arguments.q_heads,
        7,
        device="cuda",
        dtype=torch.float32,
    )
    qk_scales[:, :, 0] = 2.25
    qk_scales[:, :, 1] = 2.0

    def project(v_scale_2d: bool, per_block_qk: bool) -> Any:
        return b300_project_qkv_gqa_d128_unified_lowp_nvfp4(
            activation_operand,
            weight_operand,
            qk_scales,
            batch=1,
            seqlen=arguments.sequence,
            q_heads=arguments.q_heads,
            kv_heads=arguments.kv_heads,
            store_bf16=True,
            publish_fp8_backward=True,
            v_mxfp4_scale_2d=v_scale_2d,
            per_block_qk_scales=per_block_qk,
            cluster_cap=0,
            cache_packed_rope=False,
            cache_adaptive_qk_scale=False,
        )

    bundle_1d = project(False, False)
    bundle_dynamic = project(False, True)
    bundle_2d = project(True, False)
    assert bundle_1d.q is not None and bundle_1d.k is not None
    assert bundle_1d.v is not None
    assert bundle_dynamic.q is not None and bundle_dynamic.k is not None
    assert bundle_dynamic.v is not None
    assert bundle_2d.q is not None and bundle_2d.k is not None
    assert bundle_2d.v is not None
    for name, bundle in (("fixed", bundle_1d), ("perblock", bundle_dynamic)):
        if (
            bundle.q_backward_fp8 is None
            or bundle.k_backward_fp8 is None
            or bundle.v_backward_fp8 is None
        ):
            raise RuntimeError(f"{name} projection omitted E4M3 backward tensors")
    fused_1d = (bundle_1d.q, bundle_1d.k, bundle_1d.v)
    fused_dynamic = (
        bundle_dynamic.q,
        bundle_dynamic.k,
        bundle_dynamic.v,
    )
    fused_2d = (bundle_2d.q, bundle_2d.k, bundle_2d.v)

    q_1d = decode_native_nvfp4_qk(
        bundle_1d.q_forward_fp4,
        bundle_1d.q_forward_scales,
        bundle_1d.q_forward_global_scale,
        scale_tile_rows=128,
    ).permute(0, 2, 1, 3)
    k_1d = decode_native_nvfp4_qk(
        bundle_1d.k_forward_fp4,
        bundle_1d.k_forward_scales[:, 0::2],
        bundle_1d.k_forward_global_scale,
        scale_tile_rows=128,
    ).permute(0, 2, 1, 3)
    q_dynamic = decode_native_nvfp4_qk(
        bundle_dynamic.q_forward_fp4,
        bundle_dynamic.q_forward_scales,
        bundle_dynamic.q_forward_global_scale,
        scale_tile_rows=128,
    ).permute(0, 2, 1, 3)
    k_dynamic = decode_native_nvfp4_qk(
        bundle_dynamic.k_forward_fp4,
        bundle_dynamic.k_forward_scales[:, 0::2],
        bundle_dynamic.k_forward_global_scale,
        scale_tile_rows=128,
    ).permute(0, 2, 1, 3)
    v_1d = decode_native_mxfp4_v(
        bundle_1d.v_forward_fp4,
        bundle_1d.v_forward_scales,
    )
    v_2d = decode_native_mxfp4_v(
        bundle_2d.v_forward_fp4,
        bundle_2d.v_forward_scales,
    )

    perblock_isolation = {
        "fused_q_equal": _byte_equal(bundle_1d.q, bundle_dynamic.q),
        "fused_k_equal": _byte_equal(bundle_1d.k, bundle_dynamic.k),
        "fused_v_equal": _byte_equal(bundle_1d.v, bundle_dynamic.v),
        "v_payload_equal": _byte_equal(
            bundle_1d.v_forward_fp4,
            bundle_dynamic.v_forward_fp4,
        ),
        "v_scales_equal": _byte_equal(
            bundle_1d.v_forward_scales,
            bundle_dynamic.v_forward_scales,
        ),
        "q_backward_fp8_equal": _byte_equal(
            bundle_1d.q_backward_fp8,
            bundle_dynamic.q_backward_fp8,
        ),
        "k_backward_fp8_equal": _byte_equal(
            bundle_1d.k_backward_fp8,
            bundle_dynamic.k_backward_fp8,
        ),
        "v_backward_fp8_equal": _byte_equal(
            bundle_1d.v_backward_fp8,
            bundle_dynamic.v_backward_fp8,
        ),
        "q_global_scales_all_one": bool(
            torch.all(bundle_dynamic.q_forward_global_scale == 1.0)
        ),
        "k_global_scales_all_one": bool(
            torch.all(bundle_dynamic.k_forward_global_scale == 1.0)
        ),
        "q_local_scale_code_count": int(
            torch.unique(
                bundle_dynamic.q_forward_scales.view(torch.uint8)
            ).numel()
        ),
        "k_local_scale_code_count": int(
            torch.unique(
                bundle_dynamic.k_forward_scales.view(torch.uint8)
            ).numel()
        ),
        "k_even_odd_scale_pages_equal": _byte_equal(
            bundle_dynamic.k_forward_scales[:, 0::2],
            bundle_dynamic.k_forward_scales[:, 1::2],
        ),
    }
    equality_invariants = (
        "fused_q_equal",
        "fused_k_equal",
        "fused_v_equal",
        "v_payload_equal",
        "v_scales_equal",
        "q_backward_fp8_equal",
        "k_backward_fp8_equal",
        "v_backward_fp8_equal",
        "q_global_scales_all_one",
        "k_global_scales_all_one",
        "k_even_odd_scale_pages_equal",
    )
    failed_invariants = [
        name for name in equality_invariants if not perblock_isolation[name]
    ]
    if perblock_isolation["q_local_scale_code_count"] <= 1:
        failed_invariants.append("q_local_scales_nonuniform")
    if perblock_isolation["k_local_scale_code_count"] <= 1:
        failed_invariants.append("k_local_scales_nonuniform")
    if failed_invariants:
        raise RuntimeError(
            "D128 per-block Q/K isolation failed: "
            + ", ".join(failed_invariants)
        )

    timing = _time_rotated(
        {
            "fixed_head_qk_v_1d": lambda: project(False, False),
            "perblock_qk_v_1d": lambda: project(False, True),
            "fixed_head_qk_v_2d": lambda: project(True, False),
        },
        warmups=arguments.warmups,
        samples=arguments.samples,
    )
    result = {
        "schema": "gqa_d128_projection_boundaries_v2",
        "configuration": {
            "sequence": arguments.sequence,
            "hidden": arguments.hidden,
            "q_heads": arguments.q_heads,
            "kv_heads": arguments.kv_heads,
            "head_dim": depth,
            "seed": arguments.seed,
            "projection_weight_scale": "shared_2d_16x16",
            "q_quant_scale": 2.25,
            "k_quant_scale": 2.0,
        },
        "input_qdq": tensor_error_metrics(activation, activation_qdq),
        "weight_qdq": tensor_error_metrics(qkv_weight, weight_qdq),
        "weight_transpose_invariant": {
            "block_shape": [16, 16],
            "decoded_transpose_bitwise_equal": (
                weight_transpose_bitwise_equal
            ),
        },
        "dense_qdq_vs_dense_bf16": _component_metrics(dense_bf16, dense_qdq),
        "fused_bf16_vs_dense_qdq": _component_metrics(dense_qdq, fused_1d),
        "fused_bf16_vs_dense_bf16": _component_metrics(dense_bf16, fused_1d),
        "publication_vs_fused_bf16": {
            "q_fixed_head_nvfp4": tensor_error_metrics(bundle_1d.q, q_1d),
            "k_fixed_head_nvfp4": tensor_error_metrics(bundle_1d.k, k_1d),
            "q_perblock_nvfp4": tensor_error_metrics(
                bundle_dynamic.q,
                q_dynamic,
            ),
            "k_perblock_nvfp4": tensor_error_metrics(
                bundle_dynamic.k,
                k_dynamic,
            ),
            "v_mxfp4_1d": tensor_error_metrics(bundle_1d.v, v_1d),
            "v_mxfp4_2d": tensor_error_metrics(bundle_2d.v, v_2d),
        },
        "perblock_qk_isolation": perblock_isolation,
        "v_geometry_isolation": {
            "fused_q_equal": _byte_equal(bundle_1d.q, bundle_2d.q),
            "fused_k_equal": _byte_equal(bundle_1d.k, bundle_2d.k),
            "fused_v_equal": _byte_equal(bundle_1d.v, bundle_2d.v),
            "q_payload_equal": _byte_equal(
                bundle_1d.q_forward_fp4,
                bundle_2d.q_forward_fp4,
            ),
            "k_payload_equal": _byte_equal(
                bundle_1d.k_forward_fp4,
                bundle_2d.k_forward_fp4,
            ),
            "v_payload_equal": _byte_equal(
                bundle_1d.v_forward_fp4,
                bundle_2d.v_forward_fp4,
            ),
            "v_scales_equal": _byte_equal(
                bundle_1d.v_forward_scales,
                bundle_2d.v_forward_scales,
            ),
        },
        "projection_timing_ms": timing,
        "memory": {
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
