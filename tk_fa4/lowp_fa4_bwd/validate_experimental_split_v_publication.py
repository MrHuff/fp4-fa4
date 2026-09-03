#!/usr/bin/env python3
"""Validate the experimental split-V E4M3 projection publication.

The production MX projection publishes backward Q/K/V by lifting its retained
NVFP4/MXFP4 representations to E4M3.  The experimental split-V specialization
keeps represented, per-block Q/K but publishes backward V directly from the
projection accumulator.  This validator launches both specializations from
identical operands and uses the ordinary nonrepresented MX specialization as
an independent direct-accumulator V control.

The direct control intentionally has a different Q/K quantization policy.  Its
forward MXFP4 V publication must nevertheless match the split route before its
backward E4M3 V is accepted as the aligned accumulator-publication reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

import tk_fa4.interface as tk_interface
from tk_fa4 import (
    B300UnifiedLowpQKV,
    b300_pack_gqa_d64_paired_rope,
    b300_prepare_e4m3_projection_operand,
    b300_prepare_e4m3_projection_weight,
    b300_project_qkv_gqa_d64_paired_unified_lowp_e4m3,
)


SPLIT_V_SYMBOL = (
    "project_qkv_gqa_d64_paired_unified_fp8_e4m3_rope_packed_"
    "interleaved_causal_represented_backward_perblock_qk_"
    "split_v_backward"
)

# These publications are required to remain byte-identical when only the
# split-V switch changes.  Names are kept explicit in the JSON artifact.
FORWARD_FIELDS = (
    "q_payload",
    "q_scales",
    "q_global_scale",
    "k_payload",
    "k_scales",
    "k_global_scale",
    "v_payload",
    "v_scales",
)
BACKWARD_QK_FIELDS = (
    "q_backward_e4m3",
    "k_backward_e4m3",
)

# The D64 MX scale page reserves 512 bytes for the general D128 layout.  D64
# writes two depth groups: for each of 32 depth lanes, bytes [0:8] of its
# 16-byte slot are contractual and bytes [8:16] are padding.  CUDA allocates
# the page with at::empty, so comparing that unwritten padding would compare
# allocator history rather than a projection publication.
V_SCALE_VALID_INDICES = tuple(
    depth_lane * 16 + depth_group * 4 + sequence_quarter
    for depth_lane in range(32)
    for depth_group in range(2)
    for sequence_quarter in range(4)
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _byte_comparison(
    reference: torch.Tensor,
    actual: torch.Tensor,
) -> dict[str, int | float | bool | list[int]]:
    reference_bytes = reference.contiguous().view(torch.uint8)
    actual_bytes = actual.contiguous().view(torch.uint8)
    if reference_bytes.shape != actual_bytes.shape:
        return {
            "equal": False,
            "shape_equal": False,
            "reference_shape": list(reference_bytes.shape),
            "actual_shape": list(actual_bytes.shape),
            "mismatches": -1,
            "values": max(reference_bytes.numel(), actual_bytes.numel()),
            "mismatch_fraction": 1.0,
        }
    mismatch = reference_bytes != actual_bytes
    mismatches = int(mismatch.sum())
    values = int(mismatch.numel())
    return {
        "equal": mismatches == 0,
        "shape_equal": True,
        "reference_shape": list(reference_bytes.shape),
        "actual_shape": list(actual_bytes.shape),
        "mismatches": mismatches,
        "values": values,
        "mismatch_fraction": mismatches / values if values else 0.0,
    }


def _contract_view(name: str, tensor: torch.Tensor) -> torch.Tensor:
    """Return only bytes that are defined by a publication's D64 ABI."""
    if name != "v_scales":
        return tensor
    indices = torch.tensor(
        V_SCALE_VALID_INDICES,
        device=tensor.device,
        dtype=torch.long,
    )
    return tensor.index_select(-1, indices)


def _metrics(
    reference: torch.Tensor,
    actual: torch.Tensor,
) -> dict[str, float | bool]:
    reference_f = reference.detach().float().reshape(-1)
    actual_f = actual.detach().float().reshape(-1)
    reference_norm = reference_f.norm().clamp_min(1.0e-30)
    actual_norm = actual_f.norm().clamp_min(1.0e-30)
    difference = actual_f - reference_f
    return {
        "cosine": float(
            torch.dot(reference_f, actual_f)
            / (reference_norm * actual_norm)
        ),
        "relative_l2": float(difference.norm() / reference_norm),
        "max_abs": float(difference.abs().max()),
        "finite": bool(torch.isfinite(actual_f).all()),
    }


def _publications(bundle: B300UnifiedLowpQKV) -> dict[str, torch.Tensor]:
    required = {
        "q_payload": bundle.backward.score_q_fp4,
        "q_scales": bundle.q_forward_scales,
        "q_global_scale": bundle.q_forward_global_scale,
        "k_payload": bundle.backward.score_k_fp4,
        "k_scales": bundle.k_forward_scales,
        "k_global_scale": bundle.k_forward_global_scale,
        "v_payload": bundle.v_forward_fp4,
        "v_scales": bundle.v_forward_scales,
        "q_backward_e4m3": bundle.q_backward_fp8,
        "k_backward_e4m3": bundle.k_backward_fp8,
        "v_backward_e4m3": bundle.v_backward_fp8,
    }
    missing = [name for name, tensor in required.items() if tensor is None]
    if missing:
        raise RuntimeError(
            "projection omitted required publication(s): " + ", ".join(missing)
        )
    empty = [name for name, tensor in required.items() if not tensor.numel()]
    if empty:
        raise RuntimeError(
            "projection returned empty publication(s): " + ", ".join(empty)
        )
    return required  # type: ignore[return-value]


def _make_rope(sequence: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(sequence, device=device, dtype=torch.float32)
    frequencies = 1.0 / (
        10_000.0
        ** (
            torch.arange(32, device=device, dtype=torch.float32)
            / 32.0
        )
    )
    angles = positions[:, None] * frequencies[None, :]
    return b300_pack_gqa_d64_paired_rope(
        angles.cos()[None].bfloat16(),
        angles.sin()[None].bfloat16(),
    )


def _project(
    input_operand: tuple[torch.Tensor, torch.Tensor],
    weight_operand: tuple[torch.Tensor, torch.Tensor],
    qk_scales: torch.Tensor,
    rope: torch.Tensor,
    *,
    sequence: int,
    q_heads: int,
    kv_heads: int,
    represented_backward: bool,
    per_block_qk_scales: bool,
    experimental_split_v_backward: bool,
    v_mxfp4_scale_2d: bool,
) -> B300UnifiedLowpQKV:
    return b300_project_qkv_gqa_d64_paired_unified_lowp_e4m3(
        input_operand,
        weight_operand,
        qk_scales,
        rope,
        batch=1,
        seqlen=sequence,
        q_heads=q_heads,
        kv_heads=kv_heads,
        publish_mxfp4_v=True,
        represented_backward=represented_backward,
        per_block_qk_scales=per_block_qk_scales,
        experimental_split_v_backward=experimental_split_v_backward,
        v_mxfp4_scale_2d=v_mxfp4_scale_2d,
    )


def _extension_identity(expected: Path | None) -> dict[str, Any]:
    extension = tk_interface._C_b300_lowp_bwd
    if extension is None:
        raise RuntimeError("the low-precision backward extension is not loaded")
    if not hasattr(extension, SPLIT_V_SYMBOL):
        raise RuntimeError(
            "the loaded low-precision extension does not export the split-V "
            f"specialization {SPLIT_V_SYMBOL}"
        )
    path = Path(extension.__file__).resolve()
    if expected is not None and path != expected.resolve():
        raise RuntimeError(
            f"loaded projection extension {path} does not match expected "
            f"extension {expected.resolve()}"
        )
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "split_v_symbol": SPLIT_V_SYMBOL,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--sequence", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--q-quant-scale", type=float, default=2.25)
    parser.add_argument("--k-quant-scale", type=float, default=2.0)
    parser.add_argument(
        "--v-mxfp4-scale-2d",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--expected-projection-extension", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.sequence <= 0 or args.sequence % 256:
        parser.error("--sequence must be positive and divisible by 256")
    if args.hidden <= 0 or args.hidden % 128:
        parser.error("--hidden must be positive and divisible by 128")
    if (
        args.q_heads <= 0
        or args.kv_heads <= 0
        or args.q_heads % 2
        or args.kv_heads % 2
        or args.q_heads % args.kv_heads
        or (args.q_heads + 2 * args.kv_heads) % 4
    ):
        parser.error(
            "paired D64 requires positive even Hq/Hkv, Hq divisible by Hkv, "
            "and total QKV width divisible by 256"
        )

    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda", args.gpu)
    extension = _extension_identity(args.expected_projection_extension)
    torch.manual_seed(args.seed)

    rows = torch.randn(
        args.sequence,
        args.hidden,
        device=device,
        dtype=torch.float32,
    ).bfloat16()
    total_width = (args.q_heads + 2 * args.kv_heads) * 64
    weight = (
        torch.randn(
            total_width,
            args.hidden,
            device=device,
            dtype=torch.float32,
        )
        * 0.02
    ).bfloat16()
    input_operand = tuple(b300_prepare_e4m3_projection_operand(rows))
    weight_operand = tuple(b300_prepare_e4m3_projection_weight(weight))
    qk_scales = torch.zeros(
        1,
        args.q_heads // 2,
        7,
        device=device,
        dtype=torch.float32,
    )
    qk_scales[..., 0] = args.q_quant_scale
    qk_scales[..., 1] = args.k_quant_scale
    rope = _make_rope(args.sequence, device)

    common = {
        "input_operand": input_operand,
        "weight_operand": weight_operand,
        "qk_scales": qk_scales,
        "rope": rope,
        "sequence": args.sequence,
        "q_heads": args.q_heads,
        "kv_heads": args.kv_heads,
        "v_mxfp4_scale_2d": args.v_mxfp4_scale_2d,
    }
    baseline = _project(
        **common,
        represented_backward=True,
        per_block_qk_scales=True,
        experimental_split_v_backward=False,
    )
    split = _project(
        **common,
        represented_backward=True,
        per_block_qk_scales=True,
        experimental_split_v_backward=True,
    )
    direct = _project(
        **common,
        represented_backward=False,
        per_block_qk_scales=False,
        experimental_split_v_backward=False,
    )
    torch.cuda.synchronize(device)

    baseline_publications = _publications(baseline)
    split_publications = _publications(split)
    direct_publications = _publications(direct)
    baseline_vs_split = {
        field: _byte_comparison(
            _contract_view(field, baseline_publications[field]),
            _contract_view(field, split_publications[field]),
        )
        for field in (*FORWARD_FIELDS, *BACKWARD_QK_FIELDS)
    }
    split_vs_direct = {
        field: _byte_comparison(
            _contract_view(field, direct_publications[field]),
            _contract_view(field, split_publications[field]),
        )
        for field in ("v_payload", "v_scales", "v_backward_e4m3")
    }
    baseline_vs_split_v_backward = {
        "bytes": _byte_comparison(
            baseline_publications["v_backward_e4m3"],
            split_publications["v_backward_e4m3"],
        ),
        "metrics": _metrics(
            baseline_publications["v_backward_e4m3"],
            split_publications["v_backward_e4m3"],
        ),
    }
    checks = {
        "forward_qk_mxv_publications_unchanged": all(
            bool(baseline_vs_split[field]["equal"])
            for field in FORWARD_FIELDS
        ),
        "backward_qk_publications_unchanged": all(
            bool(baseline_vs_split[field]["equal"])
            for field in BACKWARD_QK_FIELDS
        ),
        "direct_control_has_aligned_forward_mxv_contract": all(
            bool(split_vs_direct[field]["equal"])
            for field in ("v_payload", "v_scales")
        ),
        "split_backward_v_matches_direct_accumulator_publication": bool(
            split_vs_direct["v_backward_e4m3"]["equal"]
        ),
    }
    result = {
        "schema": "experimental_split_v_publication_validator_v1",
        "configuration": {
            "seed": args.seed,
            "sequence": args.sequence,
            "hidden": args.hidden,
            "q_heads": args.q_heads,
            "kv_heads": args.kv_heads,
            "head_dim": 64,
            "q_quant_scale": args.q_quant_scale,
            "k_quant_scale": args.k_quant_scale,
            "v_mxfp4_scale_2d": args.v_mxfp4_scale_2d,
            "v_scale_storage_bytes_per_page": 512,
            "v_scale_valid_bytes_per_page": len(V_SCALE_VALID_INDICES),
            "v_scale_padding_compared": False,
            "baseline_policy": "represented_perblock_qk_and_represented_mxv",
            "split_policy": "represented_perblock_qk_direct_accumulator_v",
            "direct_control_policy": "nonrepresented_qkv",
        },
        "extension": extension,
        "baseline_vs_split": baseline_vs_split,
        "split_vs_direct_accumulator_control": split_vs_direct,
        "baseline_vs_split_backward_v": baseline_vs_split_v_backward,
        "checks": checks,
        "passed": all(checks.values()),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
