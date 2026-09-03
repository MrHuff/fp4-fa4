#!/usr/bin/env python3
"""Validate and benchmark the projection-native shared FP4 Q/K/V bundle."""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable

import torch
import torch.nn.functional as F

from tk_fa4 import (
    _C_b300_lowp_bwd as lowp,
    b300_prepare_nvfp4_projection_operand,
    b300_prepare_nvfp4_projection_weight,
    b300_project_qkv_unified_lowp_nvfp4,
    b300_stack_qkv_projection_weights,
)
from tk_fa4.fp4_pv_experiments import _run_forward_streaming_live_mxfp4


def tensor_metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    reference_f = reference.float()
    actual_f = actual.float()
    difference = actual_f - reference_f
    reference_norm = torch.linalg.vector_norm(reference_f).clamp_min(1.0e-20)
    actual_norm = torch.linalg.vector_norm(actual_f).clamp_min(1.0e-20)
    return {
        "cosine": float(
            torch.sum(reference_f * actual_f) / (reference_norm * actual_norm)
        ),
        "relative_l2": float(torch.linalg.vector_norm(difference) / reference_norm),
        "max_abs": float(difference.abs().max()),
    }


def byte_equal(left: torch.Tensor, right: torch.Tensor) -> dict[str, int | bool]:
    left_bytes = left.view(torch.uint8)
    right_bytes = right.view(torch.uint8)
    mismatch = left_bytes != right_bytes
    return {
        "equal": bool(not mismatch.any()),
        "mismatches": int(mismatch.sum()),
        "bytes": int(mismatch.numel()),
    }


def unpack_e2m1_codes(tensor: torch.Tensor) -> torch.Tensor:
    packed = tensor.view(torch.uint8)
    return torch.stack((packed & 0x0F, packed >> 4), dim=-1).flatten(-2)


def shared_v_2d_contract(
    bundle: object,
    *,
    batch: int,
    sequence: int,
    heads: int,
) -> dict[str, int | bool]:
    """Check that forward/backward V are two views of one MXFP4 operand."""
    forward_codes = unpack_e2m1_codes(bundle.v_forward_fp4)
    backward_codes = unpack_e2m1_codes(bundle.v_backward_fp4)
    expected_backward_codes = forward_codes.permute(0, 3, 1, 2)
    payload_mismatch = backward_codes != expected_backward_codes

    sequence_tiles = sequence // 128
    forward_scales = bundle.v_forward_scales.view(torch.uint8).reshape(
        batch, heads, 2, sequence_tiles, 32, 16
    )
    backward_scales = bundle.v_backward_scales.view(torch.uint8).reshape(
        batch, sequence_tiles, heads, 32, 16
    )
    scale_mismatches = 0
    scale_values = 0
    replicated_mismatches = 0
    replicated_values = 0
    for sequence_quarter in range(4):
        for depth_group in range(4):
            forward_primary = forward_scales[
                :, :, 0, :, :, depth_group * 4 + sequence_quarter
            ]
            backward_primary = backward_scales[
                :, :, :, :, sequence_quarter * 4 + depth_group
            ].permute(0, 2, 1, 3)
            expected = forward_primary[..., :1]
            replicated_mismatches += int((forward_primary != expected).sum())
            replicated_mismatches += int((backward_primary != expected).sum())
            replicated_values += forward_primary.numel() + backward_primary.numel()
            scale_mismatches += int((expected != backward_primary[..., :1]).sum())
            scale_values += expected.numel()

            if depth_group >= 2:
                forward_upper = forward_scales[
                    :, :, 1, :, :,
                    (depth_group - 2) * 4 + sequence_quarter
                ]
                replicated_mismatches += int((forward_upper != expected).sum())
                replicated_values += forward_upper.numel()

    return {
        "payload_transpose_equal": bool(not payload_mismatch.any()),
        "payload_mismatches": int(payload_mismatch.sum()),
        "payload_codes": int(payload_mismatch.numel()),
        "scale_tiles_equal": scale_mismatches == 0,
        "scale_tile_mismatches": scale_mismatches,
        "scale_tiles": scale_values,
        "scales_replicated_over_32x32": replicated_mismatches == 0,
        "scale_replication_mismatches": replicated_mismatches,
        "scale_replication_values": replicated_values,
    }


def time_rotated(
    candidates: dict[str, Callable[[], object]],
    *,
    warmups: int,
    samples: int,
) -> dict[str, dict[str, float]]:
    names = list(candidates)
    for iteration in range(warmups):
        for offset in range(len(names)):
            candidates[names[(iteration + offset) % len(names)]]()
    torch.cuda.synchronize()

    elapsed: dict[str, list[float]] = {name: [] for name in names}
    for iteration in range(samples):
        for offset in range(len(names)):
            name = names[(iteration + offset) % len(names)]
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
        }
        for name, values in elapsed.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--skip-forward", action="store_true")
    parser.add_argument("--skip-timing", action="store_true")
    args = parser.parse_args()

    if args.sequence % 128:
        raise ValueError("sequence must be divisible by 128")
    if args.hidden % 512:
        raise ValueError("hidden must be divisible by 512")

    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    batch = 1
    qk_width = args.heads * 192
    v_width = args.heads * 128
    input_tensor = (
        torch.randn(batch, args.sequence, args.hidden, device="cuda") * 0.1
    ).bfloat16()
    q_weight = (
        torch.randn(qk_width, args.hidden, device="cuda") * 0.02
    ).bfloat16()
    k_weight = (
        torch.randn(qk_width, args.hidden, device="cuda") * 0.02
    ).bfloat16()
    v_weight = (
        torch.randn(v_width, args.hidden, device="cuda") * 0.02
    ).bfloat16()

    input_matrix = input_tensor.reshape(batch * args.sequence, args.hidden)
    q_reference = torch.mm(input_matrix, q_weight.T).reshape(
        batch, args.sequence, args.heads, 192
    )
    k_reference = torch.mm(input_matrix, k_weight.T).reshape(
        batch, args.sequence, args.heads, 192
    )
    v_reference = torch.mm(input_matrix, v_weight.T).reshape(
        batch, args.sequence, args.heads, 128
    )
    adaptive = lowp.quantize_fp4_dual_qk_adaptive(
        q_reference,
        k_reference,
        16.0,
        2.0**-12,
        0.325,
        2.75,
        float(192**-0.5),
        4096.0,
    )
    adaptive_scales = adaptive[4]
    packed_input = tuple(b300_prepare_nvfp4_projection_operand(input_matrix))
    stacked_weight = b300_stack_qkv_projection_weights(
        q_weight, k_weight, v_weight
    )
    packed_weight = tuple(b300_prepare_nvfp4_projection_weight(stacked_weight))

    bundle = b300_project_qkv_unified_lowp_nvfp4(
        packed_input,
        packed_weight,
        adaptive_scales,
        batch=batch,
        seqlen=args.sequence,
        heads=args.heads,
        store_bf16=True,
    )
    no_bf16 = b300_project_qkv_unified_lowp_nvfp4(
        packed_input,
        packed_weight,
        adaptive_scales,
        batch=batch,
        seqlen=args.sequence,
        heads=args.heads,
        store_bf16=False,
    )
    pure_bundle = b300_project_qkv_unified_lowp_nvfp4(
        packed_input,
        packed_weight,
        adaptive_scales,
        batch=batch,
        seqlen=args.sequence,
        heads=args.heads,
        store_bf16=True,
        publish_pure_qk=True,
    )
    pure_no_bf16 = b300_project_qkv_unified_lowp_nvfp4(
        packed_input,
        packed_weight,
        adaptive_scales,
        batch=batch,
        seqlen=args.sequence,
        heads=args.heads,
        store_bf16=False,
        publish_pure_qk=True,
    )
    pure_single_quant = b300_project_qkv_unified_lowp_nvfp4(
        packed_input,
        packed_weight,
        adaptive_scales,
        batch=batch,
        seqlen=args.sequence,
        heads=args.heads,
        store_bf16=True,
        publish_pure_qk=True,
        pure_qk_single_quant=True,
    )
    pure_single_quant_no_bf16 = b300_project_qkv_unified_lowp_nvfp4(
        packed_input,
        packed_weight,
        adaptive_scales,
        batch=batch,
        seqlen=args.sequence,
        heads=args.heads,
        store_bf16=False,
        publish_pure_qk=True,
        pure_qk_single_quant=True,
    )
    assert bundle.q is not None and bundle.k is not None and bundle.v is not None
    assert pure_bundle.q is not None and pure_bundle.k is not None

    repacked_qk = lowp.quantize_fp4_dual_qk_precomputed_scales(
        bundle.q, bundle.k, adaptive_scales
    )
    backward_layouts = (
        bundle.backward.q_fp4,
        bundle.backward.score_q_fp4,
        bundle.backward.k_fp4,
        bundle.backward.score_k_fp4,
    )
    no_bf16_layouts = (
        no_bf16.backward.q_fp4,
        no_bf16.backward.score_q_fp4,
        no_bf16.backward.k_fp4,
        no_bf16.backward.score_k_fp4,
    )
    layout_names = (
        "q_sequence_aligned",
        "q_depth_packed",
        "k_depth_aligned",
        "k_depth_packed",
    )
    layout_checks = {
        name: {
            "standalone": byte_equal(produced, standalone),
            "no_bf16_specialization": byte_equal(produced, no_store),
        }
        for name, produced, standalone, no_store in zip(
            layout_names,
            backward_layouts,
            repacked_qk[:4],
            no_bf16_layouts,
            strict=True,
        )
    }
    compact_reference = lowp.quantize_fp4_dual_qk_blockscale(
        pure_bundle.q,
        pure_bundle.k,
        16.0,
        16.0,
    )
    compact_checks = {
        "q_dk_standalone": byte_equal(
            pure_bundle.q_dk_fp4, compact_reference[4]
        ),
        "k_dq_standalone": byte_equal(
            pure_bundle.k_dq_fp4, compact_reference[5]
        ),
        "q_dk_no_bf16_specialization": byte_equal(
            pure_bundle.q_dk_fp4, pure_no_bf16.q_dk_fp4
        ),
        "k_dq_no_bf16_specialization": byte_equal(
            pure_bundle.k_dq_fp4, pure_no_bf16.k_dq_fp4
        ),
    }
    pure_layout_names = (*layout_names, "q_dk_compact", "k_dq_compact")
    pure_single_payloads = pure_single_quant.pure_backward_operands()[:6]
    pure_single_no_store_payloads = (
        pure_single_quant_no_bf16.pure_backward_operands()[:6]
    )
    pure_single_checks = {
        name: {
            "standalone_fixed_scale": byte_equal(produced, standalone),
            "no_bf16_specialization": byte_equal(produced, no_store),
        }
        for name, produced, standalone, no_store in zip(
            pure_layout_names,
            pure_single_payloads,
            compact_reference[:6],
            pure_single_no_store_payloads,
            strict=True,
        )
    }

    v_checks = {
        "shared_2d_contract": shared_v_2d_contract(
            bundle,
            batch=batch,
            sequence=args.sequence,
            heads=args.heads,
        ),
        "payload_no_bf16_specialization": byte_equal(
            bundle.v_forward_fp4, no_bf16.v_forward_fp4
        ),
        "scales_no_bf16_specialization": byte_equal(
            bundle.v_forward_scales, no_bf16.v_forward_scales
        ),
    }
    backward_v_checks = {
        "payload_no_bf16_specialization": byte_equal(
            bundle.v_backward_fp4, no_bf16.v_backward_fp4
        ),
        "scales_no_bf16_specialization": byte_equal(
            bundle.v_backward_scales, no_bf16.v_backward_scales
        ),
    }

    forward_operands = bundle.forward_operands()
    alias_checks = {
        "q_forward_backward_payload": (
            forward_operands[0].untyped_storage().data_ptr()
            == bundle.backward.score_q_fp4.untyped_storage().data_ptr()
        ),
        "k_forward_backward_payload": (
            forward_operands[3].untyped_storage().data_ptr()
            == bundle.backward.score_k_fp4.untyped_storage().data_ptr()
        ),
    }
    q_scale_bytes = bundle.q_forward_scales.view(torch.uint8)
    k_scale_bytes = bundle.k_forward_scales.view(torch.uint8)
    expected_q_global = 1.0 / (
        448.0 * adaptive_scales.reshape(batch, args.heads, 7)[:, :, 0]
    )
    expected_k_global = 1.0 / (
        448.0 * adaptive_scales.reshape(batch, args.heads, 7)[:, :, 1]
    )
    scale_checks = {
        "q_scale_raw_values": q_scale_bytes.unique().cpu().tolist(),
        "k_scale_raw_values": k_scale_bytes.unique().cpu().tolist(),
        "q_global_max_abs": float(
            (bundle.q_forward_global_scale - expected_q_global).abs().max()
        ),
        "k_global_max_abs": float(
            (bundle.k_forward_global_scale - expected_k_global).abs().max()
        ),
    }
    pure_expected_global = torch.full_like(
        pure_single_quant.q_forward_global_scale,
        1.0 / (448.0 * 16.0),
    )
    pure_scale_checks = {
        "q_scale_raw_values": (
            pure_single_quant.q_forward_scales.view(torch.uint8)
            .unique()
            .cpu()
            .tolist()
        ),
        "k_scale_raw_values": (
            pure_single_quant.k_forward_scales.view(torch.uint8)
            .unique()
            .cpu()
            .tolist()
        ),
        "q_global_max_abs": float(
            (
                pure_single_quant.q_forward_global_scale
                - pure_expected_global
            )
            .abs()
            .max()
        ),
        "k_global_max_abs": float(
            (
                pure_single_quant.k_forward_global_scale
                - pure_expected_global
            )
            .abs()
            .max()
        ),
    }

    result: dict[str, object] = {
        "shape": {
            "batch": batch,
            "sequence": args.sequence,
            "heads": args.heads,
            "hidden": args.hidden,
        },
        "projection_accuracy": {
            "q": tensor_metrics(q_reference, bundle.q),
            "k": tensor_metrics(k_reference, bundle.k),
            "v": tensor_metrics(v_reference, bundle.v),
        },
        "backward_layouts": layout_checks,
        "pure_qk_layouts": compact_checks,
        "pure_single_quant_layouts": pure_single_checks,
        "forward_v": v_checks,
        "backward_v": backward_v_checks,
        "payload_aliases": alias_checks,
        "forward_qk_scales": scale_checks,
        "pure_single_quant_forward_qk_scales": pure_scale_checks,
        "no_bf16_publication": {
            "q_is_none": no_bf16.q is None,
            "k_is_none": no_bf16.k is None,
            "v_is_none": no_bf16.v is None,
            "pure_q_is_none": pure_single_quant_no_bf16.q is None,
            "pure_k_is_none": pure_single_quant_no_bf16.k is None,
            "pure_v_is_none": pure_single_quant_no_bf16.v is None,
        },
    }

    if not args.skip_forward:
        lowp_output, _ = _run_forward_streaming_live_mxfp4(*forward_operands)
        reference_output = F.scaled_dot_product_attention(
            bundle.q.permute(0, 2, 1, 3),
            bundle.k.permute(0, 2, 1, 3),
            bundle.v.permute(0, 2, 1, 3),
            is_causal=True,
            scale=192**-0.5,
        ).permute(0, 2, 1, 3).contiguous()
        result["forward_accuracy"] = tensor_metrics(
            reference_output, lowp_output
        )

    if not args.skip_timing:
        def bf16_projection() -> object:
            return (
                torch.mm(input_matrix, q_weight.T),
                torch.mm(input_matrix, k_weight.T),
                torch.mm(input_matrix, v_weight.T),
            )

        def bf16_projection_and_qk_pack() -> object:
            q, k, v = bf16_projection()
            q = q.reshape(batch, args.sequence, args.heads, 192)
            k = k.reshape(batch, args.sequence, args.heads, 192)
            lowp.quantize_fp4_dual_qk_precomputed_scales(
                q, k, adaptive_scales
            )
            return v

        def unified_store_bf16() -> object:
            return b300_project_qkv_unified_lowp_nvfp4(
                packed_input,
                packed_weight,
                adaptive_scales,
                batch=batch,
                seqlen=args.sequence,
                heads=args.heads,
                store_bf16=True,
            )

        def unified_no_bf16_store() -> object:
            return b300_project_qkv_unified_lowp_nvfp4(
                packed_input,
                packed_weight,
                adaptive_scales,
                batch=batch,
                seqlen=args.sequence,
                heads=args.heads,
                store_bf16=False,
            )

        def unified_pure_qk_store_bf16() -> object:
            return b300_project_qkv_unified_lowp_nvfp4(
                packed_input,
                packed_weight,
                adaptive_scales,
                batch=batch,
                seqlen=args.sequence,
                heads=args.heads,
                store_bf16=True,
                publish_pure_qk=True,
            )

        def unified_pure_qk_no_bf16_store() -> object:
            return b300_project_qkv_unified_lowp_nvfp4(
                packed_input,
                packed_weight,
                adaptive_scales,
                batch=batch,
                seqlen=args.sequence,
                heads=args.heads,
                store_bf16=False,
                publish_pure_qk=True,
            )

        def unified_pure_single_quant_store_bf16() -> object:
            return b300_project_qkv_unified_lowp_nvfp4(
                packed_input,
                packed_weight,
                adaptive_scales,
                batch=batch,
                seqlen=args.sequence,
                heads=args.heads,
                store_bf16=True,
                publish_pure_qk=True,
                pure_qk_single_quant=True,
            )

        def unified_pure_single_quant_no_bf16_store() -> object:
            return b300_project_qkv_unified_lowp_nvfp4(
                packed_input,
                packed_weight,
                adaptive_scales,
                batch=batch,
                seqlen=args.sequence,
                heads=args.heads,
                store_bf16=False,
                publish_pure_qk=True,
                pure_qk_single_quant=True,
            )

        result["timing"] = time_rotated(
            {
                "three_bf16_gemms": bf16_projection,
                "three_bf16_gemms_plus_qk_pack": bf16_projection_and_qk_pack,
                "unified_nvfp4_store_bf16": unified_store_bf16,
                "unified_nvfp4_no_bf16_store": unified_no_bf16_store,
                "unified_nvfp4_pure_qk_store_bf16": (
                    unified_pure_qk_store_bf16
                ),
                "unified_nvfp4_pure_qk_no_bf16_store": (
                    unified_pure_qk_no_bf16_store
                ),
                "unified_nvfp4_pure_single_quant_store_bf16": (
                    unified_pure_single_quant_store_bf16
                ),
                "unified_nvfp4_pure_single_quant_no_bf16_store": (
                    unified_pure_single_quant_no_bf16_store
                ),
            },
            warmups=args.warmups,
            samples=args.samples,
        )

    all_layouts_exact = all(
        check[variant]["equal"]
        for check in layout_checks.values()
        for variant in ("standalone", "no_bf16_specialization")
    )
    shared_v_contract = v_checks["shared_2d_contract"]
    all_v_exact = bool(
        shared_v_contract["payload_transpose_equal"]
        and shared_v_contract["scale_tiles_equal"]
        and shared_v_contract["scales_replicated_over_32x32"]
        and v_checks["payload_no_bf16_specialization"]["equal"]
        and v_checks["scales_no_bf16_specialization"]["equal"]
    )
    all_compact_exact = all(check["equal"] for check in compact_checks.values())
    all_pure_single_exact = all(
        check[variant]["equal"]
        for check in pure_single_checks.values()
        for variant in ("standalone_fixed_scale", "no_bf16_specialization")
    )
    all_backward_v_exact = all(
        check["equal"] for check in backward_v_checks.values()
    )
    no_publication_ok = all(result["no_bf16_publication"].values())
    projection_cosines = [
        result["projection_accuracy"][name]["cosine"]
        for name in ("q", "k", "v")
    ]
    forward_ok = (
        args.skip_forward or result["forward_accuracy"]["cosine"] >= 0.98
    )
    result["passed"] = bool(
        all_layouts_exact
        and all_compact_exact
        and all_pure_single_exact
        and all_v_exact
        and all_backward_v_exact
        and all(alias_checks.values())
        and no_publication_ok
        and min(projection_cosines) >= 0.985
        and q_scale_bytes.unique().tolist() == [126]
        and k_scale_bytes.unique().tolist() == [126]
        and scale_checks["q_global_max_abs"] == 0.0
        and scale_checks["k_global_max_abs"] == 0.0
        and pure_scale_checks["q_scale_raw_values"] == [126]
        and pure_scale_checks["k_scale_raw_values"] == [126]
        and pure_scale_checks["q_global_max_abs"] == 0.0
        and pure_scale_checks["k_global_max_abs"] == 0.0
        and forward_ok
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
