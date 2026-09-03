#!/usr/bin/env python3
"""Validate the projection-native pure-FP4 producer/attention pipeline."""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable, Sequence

import torch

from tk_fa4 import (
    _C_b300_lowp_bwd as lowp,
    b300_mha_fwd,
    b300_prepare_nvfp4_projection_operand,
    b300_project_dout_unified_lowp_nvfp4,
    b300_project_qkv_unified_lowp_nvfp4,
    b300_stack_qkv_projection_weights,
)


def byte_equal(left: torch.Tensor, right: torch.Tensor) -> dict[str, int | bool]:
    mismatch = left.view(torch.uint8) != right.view(torch.uint8)
    return {
        "equal": bool(not mismatch.any()),
        "mismatches": int(mismatch.sum()),
        "bytes": int(mismatch.numel()),
    }


def metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float | bool]:
    reference_f = reference.float()
    actual_f = actual.float()
    difference = actual_f - reference_f
    reference_norm = torch.linalg.vector_norm(reference_f).clamp_min(1.0e-20)
    actual_norm = torch.linalg.vector_norm(actual_f).clamp_min(1.0e-20)
    return {
        "finite": bool(torch.isfinite(actual_f).all()),
        "cosine": float(
            torch.sum(reference_f * actual_f) / (reference_norm * actual_norm)
        ),
        "relative_l2": float(torch.linalg.vector_norm(difference) / reference_norm),
        "max_abs": float(difference.abs().max()),
    }


def time_rotated(
    candidates: dict[str, Callable[[], object]], warmups: int, samples: int
) -> dict[str, dict[str, float]]:
    names = list(candidates)
    for iteration in range(warmups):
        for offset in range(len(names)):
            candidates[names[(iteration + offset) % len(names)]]()
    torch.cuda.synchronize()
    elapsed = {name: [] for name in names}
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
    parser.add_argument("--sequence", type=int, default=8192)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--seed", type=int, default=2026081307)
    parser.add_argument("--skip-timing", action="store_true")
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("validation requires exactly one visible GPU")

    torch.manual_seed(args.seed)
    batch = 1
    rows = batch * args.sequence
    input_matrix = (
        torch.randn(rows, args.hidden, device="cuda") * 0.1
    ).bfloat16()
    q_weight = (
        torch.randn(args.heads * 192, args.hidden, device="cuda") * 0.02
    ).bfloat16()
    k_weight = torch.randn_like(q_weight.float()).mul_(0.02).bfloat16()
    v_weight = (
        torch.randn(args.heads * 128, args.hidden, device="cuda") * 0.02
    ).bfloat16()
    qkv_weight = b300_stack_qkv_projection_weights(
        q_weight, k_weight, v_weight
    )
    packed_input = tuple(b300_prepare_nvfp4_projection_operand(input_matrix))
    packed_qkv_weight = tuple(
        b300_prepare_nvfp4_projection_operand(qkv_weight)
    )
    # The aggressive pure specialization has a fixed x16 representation and
    # deliberately carries no adaptive scale record.
    qk_metadata = torch.empty(0, device="cuda", dtype=torch.float32)

    def project_pure() -> object:
        return b300_project_qkv_unified_lowp_nvfp4(
            packed_input,
            packed_qkv_weight,
            qk_metadata,
            batch=batch,
            seqlen=args.sequence,
            heads=args.heads,
            store_bf16=True,
            publish_pure_qk=True,
            pure_qk_single_quant=True,
        )

    def project_then_pack_separately() -> tuple[object, Sequence[torch.Tensor]]:
        bundle = b300_project_qkv_unified_lowp_nvfp4(
            packed_input,
            packed_qkv_weight,
            qk_metadata,
            batch=batch,
            seqlen=args.sequence,
            heads=args.heads,
            store_bf16=True,
        )
        assert bundle.q is not None and bundle.k is not None
        packed = lowp.quantize_fp4_dual_qk_blockscale(
            bundle.q, bundle.k, 16.0, 16.0
        )
        return bundle, packed

    qkv = project_pure()
    assert qkv.q is not None and qkv.k is not None and qkv.v is not None
    q, k, v = qkv.q, qkv.k, qkv.v
    out, lse = b300_mha_fwd(q, k, v, causal=True, return_lse=True)

    dout_input = (
        torch.randn(rows, args.hidden, device="cuda") * 0.1
    ).bfloat16()
    dout_weight = (
        torch.randn(args.heads * 128, args.hidden, device="cuda") * 0.02
    ).bfloat16()
    packed_dout_input = tuple(
        b300_prepare_nvfp4_projection_operand(dout_input)
    )
    packed_dout_weight = tuple(
        b300_prepare_nvfp4_projection_operand(dout_weight)
    )
    dout_bundle = b300_project_dout_unified_lowp_nvfp4(
        packed_dout_input,
        packed_dout_weight,
        out,
        lse,
        batch=batch,
        seqlen=args.sequence,
        heads=args.heads,
        store_bf16=True,
    )
    assert dout_bundle.dout is not None
    dout = dout_bundle.dout

    standalone_qk = tuple(
        lowp.quantize_fp4_dual_qk_blockscale(q, k, 16.0, 16.0)
    )
    fused_qk = qkv.pure_backward_operands()
    standalone_mxfp4 = tuple(
        lowp.prepare_mxfp4_backward_operands(out, dout, v, lse)
    )
    fused_mxfp4 = (
        dout_bundle.dout_dp_fp4,
        qkv.v_backward_fp4,
        dout_bundle.dout_dp_scales,
        qkv.v_backward_scales,
        dout_bundle.dout_dv_fp4,
        dout_bundle.dout_dv_scales,
        dout_bundle.dpsum,
        dout_bundle.lse_log2,
    )
    qk_names = (
        "q_sequence_aligned",
        "q_depth_packed",
        "k_depth_aligned",
        "k_depth_packed",
        "q_dk_compact",
        "k_dq_compact",
        "q_dk_scales",
        "k_dq_scales",
    )
    producer_names = (
        "dout_dp",
        "v_dp",
        "dout_dp_scale",
        "v_dp_scale",
        "dout_dv",
        "dout_dv_scale",
    )
    payload_checks = {
        **{
            f"qk_{name}": byte_equal(fused, reference)
            for name, fused, reference in zip(
                qk_names, fused_qk, standalone_qk, strict=True
            )
        },
        **{
            f"producer_{name}": byte_equal(fused, reference)
            for name, fused, reference in zip(
                producer_names,
                fused_mxfp4[:6],
                standalone_mxfp4[:6],
                strict=True,
            )
        },
    }
    producer_stats_quality = {
        "dpsum": metrics(standalone_mxfp4[6], fused_mxfp4[6]),
        "lse_log2": metrics(standalone_mxfp4[7], fused_mxfp4[7]),
    }

    scale = float(192**-0.5)

    def backward(
        qk_operands: Sequence[torch.Tensor],
        producer_operands: Sequence[torch.Tensor],
        q_control: torch.Tensor = q,
        k_control: torch.Tensor = k,
        v_control: torch.Tensor = v,
    ) -> tuple[torch.Tensor, ...]:
        return tuple(
            lowp.backward_fp4_mxfp4dpdvdsdqdk_producer_native_x32(
                q_control,
                k_control,
                v_control,
                out,
                lse,
                dout,
                *qk_operands,
                *producer_operands,
                16.0,
                16.0,
                4096.0,
                True,
                scale,
                False,
            )
        )

    reference_output = backward(standalone_qk, standalone_mxfp4)
    fused_output = backward(fused_qk, fused_mxfp4)
    output_quality = {
        name: metrics(reference, actual)
        for name, reference, actual in zip(
            ("dq", "dk", "dv"), reference_output, fused_output, strict=True
        )
    }

    result: dict[str, object] = {
        "shape": {
            "sequence": args.sequence,
            "heads": args.heads,
            "hidden": args.hidden,
        },
        "payload_checks": payload_checks,
        "producer_stats_quality": producer_stats_quality,
        "output_quality": output_quality,
    }
    if not args.skip_timing:
        def fused_projection_then_backward() -> object:
            bundle = project_pure()
            assert bundle.q is not None and bundle.k is not None
            assert bundle.v is not None
            return backward(
                bundle.pure_backward_operands(),
                fused_mxfp4,
                bundle.q,
                bundle.k,
                bundle.v,
            )

        def separate_projection_pack_then_backward() -> object:
            bundle, packed = project_then_pack_separately()
            assert bundle.q is not None and bundle.k is not None
            assert bundle.v is not None
            return backward(
                packed,
                fused_mxfp4,
                bundle.q,
                bundle.k,
                bundle.v,
            )

        timing = time_rotated(
            {
                "projection_separate_pack_then_backward": (
                    separate_projection_pack_then_backward
                ),
                "projection_fused_single_quant_then_backward": (
                    fused_projection_then_backward
                ),
                "prepared_standalone_backward": lambda: backward(
                    standalone_qk, standalone_mxfp4
                ),
                "prepared_projection_native_backward": lambda: backward(
                    fused_qk, fused_mxfp4
                ),
            },
            args.warmups,
            args.samples,
        )
        separate_ms = timing[
            "projection_separate_pack_then_backward"
        ]["median_ms"]
        fused_ms = timing[
            "projection_fused_single_quant_then_backward"
        ]["median_ms"]
        result["timing"] = timing
        result["fused_saved_ms"] = separate_ms - fused_ms
        result["fused_speedup"] = separate_ms / fused_ms

    result["passed"] = bool(
        all(check["equal"] for check in payload_checks.values())
        and producer_stats_quality["dpsum"]["relative_l2"] < 1.0e-6
        and producer_stats_quality["lse_log2"]["relative_l2"] < 1.0e-6
        and all(
            item["finite"] and item["cosine"] > 0.99999
            for item in output_quality.values()
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
