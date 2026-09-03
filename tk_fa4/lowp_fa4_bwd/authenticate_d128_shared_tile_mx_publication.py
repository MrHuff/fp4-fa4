#!/usr/bin/env python3
"""Authenticate the single-quantization D128 shared-tile MXFP4 V publisher."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path
from typing import Any


BATCH = 2
SEQUENCE = 4096
HIDDEN = 4096
Q_HEADS = 32
KV_HEADS = 8
DEPTH = 128
BASE_SYMBOL = (
    "project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered_"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--blocks", type=int, default=30)
    parser.add_argument("--conditioning-replays", type=int, default=2)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _byte_comparison(torch: Any, left: Any, right: Any) -> dict[str, Any]:
    left_bytes = left.contiguous().view(torch.uint8).reshape(-1)
    right_bytes = right.contiguous().view(torch.uint8).reshape(-1)
    same_metadata = (
        left.dtype == right.dtype
        and tuple(left.shape) == tuple(right.shape)
        and left.device == right.device
    )
    mismatches = (
        int((left_bytes != right_bytes).sum().item())
        if same_metadata
        else None
    )
    return {
        "same_metadata": same_metadata,
        "bytes": int(left_bytes.numel()) if same_metadata else None,
        "mismatches": mismatches,
        "passed": same_metadata and mismatches == 0,
    }


def _forward_codes(torch: Any, payload: Any) -> Any:
    raw = payload.contiguous().view(torch.uint8)
    return torch.stack((raw & 0x0F, raw >> 4), dim=-1).reshape(
        BATCH, KV_HEADS, DEPTH, SEQUENCE
    ).permute(0, 3, 1, 2).contiguous()


def _backward_codes(torch: Any, payload: Any) -> Any:
    raw = payload.contiguous().view(torch.uint8)
    return torch.stack((raw & 0x0F, raw >> 4), dim=-1).reshape(
        BATCH, SEQUENCE, KV_HEADS, DEPTH
    )


def _scale_contract(torch: Any, forward: Any, backward: Any) -> dict[str, Any]:
    sequence_tiles = SEQUENCE // 128
    f_pages = forward.contiguous().view(torch.uint8).reshape(
        BATCH, sequence_tiles, KV_HEADS, 32, 4, 4
    )
    b_pages = backward.contiguous().view(torch.uint8).reshape(
        BATCH, sequence_tiles, KV_HEADS, 32, 4, 4
    )
    mismatches = 0
    replication_mismatches = 0
    for sequence_quarter in range(4):
        for depth_group in range(4):
            f_codes = f_pages[..., :, depth_group, sequence_quarter]
            b_codes = b_pages[..., :, sequence_quarter, depth_group]
            f_anchor = f_codes[..., :1]
            b_anchor = b_codes[..., :1]
            replication_mismatches += int((f_codes != f_anchor).sum().item())
            replication_mismatches += int((b_codes != b_anchor).sum().item())
            mismatches += int((f_anchor != b_anchor).sum().item())
    return {
        "tile_anchor_mismatches": mismatches,
        "replication_mismatches": replication_mismatches,
        "passed": mismatches == 0 and replication_mismatches == 0,
    }


def _metrics(torch: Any, actual: Any, reference: Any) -> dict[str, float]:
    actual64 = actual.double().reshape(-1)
    reference64 = reference.double().reshape(-1)
    actual_norm = torch.linalg.vector_norm(actual64)
    reference_norm = torch.linalg.vector_norm(reference64)
    delta = torch.linalg.vector_norm(actual64 - reference64)
    return {
        "cosine": float(
            (torch.dot(actual64, reference64) / (actual_norm * reference_norm)).item()
        ),
        "relative_l2": float((delta / reference_norm).item()),
        "norm_ratio": float((actual_norm / reference_norm).item()),
        "max_abs": float((actual64 - reference64).abs().max().item()),
    }


def main() -> int:
    args = _parse_args()
    extension_path = args.extension.resolve(strict=True)
    output_path = args.output.resolve(strict=False)
    if output_path.exists():
        raise FileExistsError(output_path)
    os.environ["TK_FA4_LOWP_BWD_EXTENSION_SOURCE"] = str(extension_path)

    import torch

    import tk_fa4
    from tk_fa4 import interface
    from tk_fa4.lowp_fa4_bwd import (
        authenticate_d128_mx_backward_v_publication as auth,
    )
    from tk_fa4.lowp_fa4_bwd.projection_quantization_reference import (
        decode_native_mxfp4_v,
    )

    if torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one visible GPU is required")
    if torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("an SM100 GPU is required")

    extension = interface._C_b300_lowp_bwd
    retained = getattr(
        extension,
        BASE_SYMBOL + "mx_backward_v_mx_forward_out_unchecked",
    )
    candidate = getattr(
        extension,
        BASE_SYMBOL + "shared_tile_mx_backward_v_mx_forward_out_unchecked",
    )
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    input_operand, weight_operand, qk_scales, packed_rope = auth._prepare_operands(
        torch, tk_fa4, batch=BATCH, seed=args.seed
    )
    retained_workspace = auth._allocate_workspace(
        torch, tk_fa4, batch=BATCH, include_mx_backward_v=True
    )
    candidate_workspace = auth._allocate_workspace(
        torch, tk_fa4, batch=BATCH, include_mx_backward_v=True
    )
    binder = tk_fa4.b300_bind_qkv_gqa_d128_unified_lowp_nvfp4_projection(
        batch=BATCH,
        seqlen=SEQUENCE,
        hidden=HIDDEN,
        q_heads=Q_HEADS,
        kv_heads=KV_HEADS,
        publish_mxfp4_v=True,
        v_mxfp4_scale_2d=False,
        per_block_qk_scales=True,
        experimental_mx_backward_v=True,
    )

    def call_args(workspace: Any, *, scale_2d: bool) -> tuple[Any, ...]:
        return (
            *input_operand,
            *weight_operand,
            qk_scales,
            packed_rope,
            BATCH,
            SEQUENCE,
            Q_HEADS,
            KV_HEADS,
            scale_2d,
            True,
            binder.cluster_cap,
            binder.cache_packed_rope,
            binder.cache_adaptive_qk_scale,
            *workspace.compact_mx_backward_v_outputs(),
        )

    retained_args = call_args(retained_workspace, scale_2d=False)
    candidate_args = call_args(candidate_workspace, scale_2d=True)
    with torch.no_grad():
        retained(*retained_args)
        candidate(*candidate_args)
        torch.cuda.synchronize()

    neutral_names = (
        "q_payload",
        "k_payload",
        "q_scale_pages",
        "q_global_scale",
        "k_scale_pages",
        "k_global_scale",
        "q_backward_fp8",
        "k_backward_fp8",
    )
    neutral = {
        name: _byte_comparison(
            torch,
            getattr(retained_workspace, name),
            getattr(candidate_workspace, name),
        )
        for name in neutral_names
    }
    if not all(item["passed"] for item in neutral.values()):
        raise RuntimeError("shared-tile route changed Q/K publication")

    assert candidate_workspace.v_backward_mxfp4 is not None
    assert candidate_workspace.v_backward_mxfp4_scale_pages is not None
    code_matrix = _byte_comparison(
        torch,
        _forward_codes(torch, candidate_workspace.v_mxfp4_payload),
        _backward_codes(torch, candidate_workspace.v_backward_mxfp4),
    )
    scales = _scale_contract(
        torch,
        candidate_workspace.v_mxfp4_scale_pages,
        candidate_workspace.v_backward_mxfp4_scale_pages,
    )
    if not code_matrix["passed"] or not scales["passed"]:
        raise RuntimeError("forward/backward shared-tile ABI mismatch")

    reference = tk_fa4.b300_project_qkv_gqa_d128_unified_lowp_nvfp4(
        input_operand,
        weight_operand,
        qk_scales,
        batch=BATCH,
        seqlen=SEQUENCE,
        q_heads=Q_HEADS,
        kv_heads=KV_HEADS,
        store_bf16=True,
        publish_fp8_backward=False,
        v_mxfp4_scale_2d=True,
        per_block_qk_scales=False,
        rope_packed=packed_rope,
        cluster_cap=binder.cluster_cap,
        cache_packed_rope=binder.cache_packed_rope,
        cache_adaptive_qk_scale=binder.cache_adaptive_qk_scale,
    )
    assert reference.v is not None
    forward_reference = {
        "payload": _byte_comparison(
            torch, candidate_workspace.v_mxfp4_payload, reference.v_forward_fp4
        ),
        "scales": _byte_comparison(
            torch,
            candidate_workspace.v_mxfp4_scale_pages,
            reference.v_forward_scales,
        ),
    }
    if not all(item["passed"] for item in forward_reference.values()):
        raise RuntimeError("shared-tile forward publication changed")
    decoded = decode_native_mxfp4_v(
        candidate_workspace.v_mxfp4_payload,
        candidate_workspace.v_mxfp4_scale_pages,
    )

    functions = {
        "retained_dual_quantization": lambda: retained(*retained_args),
        "shared_tile_single_quantization": lambda: candidate(*candidate_args),
    }
    for iteration in range(12):
        for name in (tuple(functions) if iteration % 2 == 0 else tuple(functions)[::-1]):
            functions[name]()
    torch.cuda.synchronize()
    samples = {name: [] for name in functions}
    paired_deltas = []
    for block in range(args.blocks):
        names = tuple(functions)
        order = (
            (names[0], names[1], names[1], names[0])
            if block % 2 == 0
            else (names[1], names[0], names[0], names[1])
        )
        block_samples = {name: [] for name in functions}
        for name in order:
            for _ in range(args.conditioning_replays):
                functions[name]()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            functions[name]()
            end.record()
            end.synchronize()
            elapsed_us = float(start.elapsed_time(end) * 1000.0)
            samples[name].append(elapsed_us)
            block_samples[name].append(elapsed_us)
        paired_deltas.append(
            statistics.fmean(block_samples[names[1]])
            - statistics.fmean(block_samples[names[0]])
        )

    medians = {name: statistics.median(values) for name, values in samples.items()}
    receipt = {
        "schema": "tkfa4.d128_shared_tile_mx_publication.v1",
        "shape": {
            "batch": BATCH,
            "sequence": SEQUENCE,
            "hidden": HIDDEN,
            "q_heads": Q_HEADS,
            "kv_heads": KV_HEADS,
            "depth": DEPTH,
        },
        "hardware": torch.cuda.get_device_name(),
        "extension": {
            "path": str(extension_path),
            "sha256": _sha256(extension_path),
            "bytes": extension_path.stat().st_size,
        },
        "route_neutral_bitwise": neutral,
        "shared_code_matrix": code_matrix,
        "shared_scale_tiles": scales,
        "forward_reference_bitwise": forward_reference,
        "numerics_vs_bf16": _metrics(torch, decoded, reference.v),
        "timing": {
            "protocol": "self_conditioned_rotated_abba_baab_cuda_events",
            "blocks": args.blocks,
            "conditioning_replays": args.conditioning_replays,
            "medians_us": medians,
            "paired_median_candidate_minus_retained_us": statistics.median(
                paired_deltas
            ),
            "paired_mean_candidate_minus_retained_us": statistics.fmean(
                paired_deltas
            ),
            "candidate_faster": (
                medians["shared_tile_single_quantization"]
                < medians["retained_dual_quantization"]
            ),
            "samples_us": samples,
            "paired_deltas_us": paired_deltas,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(output_path),
        "medians_us": medians,
        "paired_median_delta_us": statistics.median(paired_deltas),
        "candidate_faster": receipt["timing"]["candidate_faster"],
        "numerics_vs_bf16": receipt["numerics_vs_bf16"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
