#!/usr/bin/env python3
"""Validate true-2D D64 QKV weights and both native publication layouts.

Set ``TK_FA4_LOWP_BWD_EXTENSION_SOURCE`` to the audited projection extension
before importing this module when the extension is not installed in-tree.
The validator intentionally launches each publisher only once: it checks the
projection/publication boundary and is not a timing benchmark.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

# Keep direct-file and ``python -m`` execution equivalent.
_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

import tk_fa4.interface as tk_interface
from tk_fa4 import (
    b300_pack_gqa_d64_paired_rope,
    b300_prepare_nvfp4_projection_operand,
    b300_prepare_nvfp4_projection_weight,
    b300_prepare_nvfp4_projection_weight_dual,
    b300_project_qkv_gqa_d64_paired_unified_lowp_nvfp4,
)
from tk_fa4.lowp_fa4_bwd.projection_quantization_reference import (
    decode_native_nvfp4_qk,
    decode_prepared_nvfp4_matrix,
    tensor_error_metrics,
    transpose_prepared_nvfp4_weight_reference,
)


NORMAL_SYMBOL = "project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed"
INTERLEAVED_SYMBOL = NORMAL_SYMBOL + "_interleaved_causal"
WEIGHT_QUANTIZATION_SYMBOL = "quantize_nvfp4_projection_weight"
DUAL_WEIGHT_QUANTIZATION_SYMBOL = "quantize_nvfp4_projection_weight_dual"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    reduction = (args.q_heads + 2 * args.kv_heads) * 64
    if args.sequence <= 0 or args.sequence % 128:
        parser.error("--sequence must be positive and divisible by 128")
    if args.hidden <= 0 or args.hidden % 128:
        parser.error("--hidden must be positive and divisible by 128")
    if (
        args.q_heads <= 0
        or args.kv_heads <= 0
        or args.q_heads % 2
        or args.kv_heads % 2
        or args.q_heads % args.kv_heads
    ):
        parser.error(
            "D64 paired publication requires positive even head counts and "
            "q-heads divisible by kv-heads"
        )
    if reduction % 128:
        parser.error("the concatenated QKV width must be divisible by 128")
    return args


def _byte_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return bool(torch.equal(left.view(torch.uint8), right.view(torch.uint8)))


def _operand_byte_equal(
    left: tuple[torch.Tensor, ...],
    right: tuple[torch.Tensor, ...],
) -> bool:
    return len(left) == len(right) and all(
        _byte_equal(left_tensor, right_tensor)
        for left_tensor, right_tensor in zip(left, right, strict=True)
    )


def _make_rope(sequence: int) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(sequence, device="cuda", dtype=torch.float32)
    frequencies = 1.0 / (
        10_000.0
        ** (
            torch.arange(32, device="cuda", dtype=torch.float32) / 32.0
        )
    )
    angles = positions[:, None] * frequencies[None, :]
    return angles.cos()[None].bfloat16(), angles.sin()[None].bfloat16()


def _apply_rope(
    tensor: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    pairs = tensor.float().reshape(*tensor.shape[:-1], 32, 2)
    first, second = pairs[..., 0], pairs[..., 1]
    cosine_f = cosine.float().unsqueeze(2)
    sine_f = sine.float().unsqueeze(2)
    return torch.stack(
        (
            first * cosine_f - second * sine_f,
            first * sine_f + second * cosine_f,
        ),
        dim=-1,
    ).flatten(-2).bfloat16()


def _split_qkv(
    matrix: torch.Tensor,
    *,
    sequence: int,
    q_heads: int,
    kv_heads: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_width = q_heads * 64
    kv_width = kv_heads * 64
    return (
        matrix[:, :q_width].reshape(1, sequence, q_heads, 64),
        matrix[:, q_width : q_width + kv_width].reshape(
            1, sequence, kv_heads, 64
        ),
        matrix[:, q_width + kv_width :].reshape(
            1, sequence, kv_heads, 64
        ),
    )


def _component_metrics(
    reference: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    candidate: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, dict[str, float]]:
    return {
        name: tensor_error_metrics(expected, actual)
        for name, expected, actual in zip(
            ("q", "k", "v"), reference, candidate, strict=True
        )
    }


@torch.no_grad()
def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("validator requires exactly one visible GPU")
    extension = tk_interface._C_b300_lowp_bwd
    if extension is None:
        raise RuntimeError(
            "the low-precision projection extension is unavailable; set "
            "TK_FA4_LOWP_BWD_EXTENSION_SOURCE before invoking the validator"
        )
    required_symbols = (
        NORMAL_SYMBOL,
        INTERLEAVED_SYMBOL,
        WEIGHT_QUANTIZATION_SYMBOL,
        DUAL_WEIGHT_QUANTIZATION_SYMBOL,
    )
    missing_symbols = [
        symbol for symbol in required_symbols if not hasattr(extension, symbol)
    ]
    if missing_symbols:
        raise RuntimeError(
            "projection extension is missing required D64 symbols: "
            + ", ".join(missing_symbols)
        )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    activation = torch.randn(
        args.sequence, args.hidden, device="cuda", dtype=torch.float32
    )
    activation = (
        activation
        * torch.rsqrt(activation.square().mean(dim=1, keepdim=True) + 1.0e-5)
    ).bfloat16()
    reduction = (args.q_heads + 2 * args.kv_heads) * 64
    qkv_weight = (
        torch.randn(
            reduction, args.hidden, device="cuda", dtype=torch.float32
        )
        * 0.02
    ).bfloat16()

    activation_operand = tuple(
        b300_prepare_nvfp4_projection_operand(activation)
    )
    dual_weight = b300_prepare_nvfp4_projection_weight_dual(qkv_weight)
    weight_operand = dual_weight.forward
    weight_transpose_operand = dual_weight.transpose
    independent_weight_operand = tuple(
        b300_prepare_nvfp4_projection_weight(qkv_weight)
    )
    independent_weight_transpose_operand = tuple(
        b300_prepare_nvfp4_projection_weight(qkv_weight.T.contiguous())
    )
    reference_weight_transpose_operand = (
        transpose_prepared_nvfp4_weight_reference(independent_weight_operand)
    )
    dual_weight_invariants = {
        "forward_bytes_equal_independent_quantization": _operand_byte_equal(
            weight_operand,
            independent_weight_operand,
        ),
        "transpose_bytes_equal_independent_quantization": (
            _operand_byte_equal(
                weight_transpose_operand,
                independent_weight_transpose_operand,
            )
        ),
        "transpose_bytes_equal_storage_reference": _operand_byte_equal(
            weight_transpose_operand,
            reference_weight_transpose_operand,
        ),
        "global_scale_storage_shared": (
            weight_operand[2].data_ptr()
            == weight_transpose_operand[2].data_ptr()
        ),
    }
    failed_dual_weight = [
        name for name, valid in dual_weight_invariants.items() if not valid
    ]
    if failed_dual_weight:
        raise RuntimeError(
            "dual NVFP4 weight preparation invariants failed: "
            + ", ".join(failed_dual_weight)
        )
    activation_qdq = decode_prepared_nvfp4_matrix(activation_operand)
    weight_qdq = decode_prepared_nvfp4_matrix(weight_operand)
    weight_transpose_qdq = decode_prepared_nvfp4_matrix(
        weight_transpose_operand
    )
    weight_transpose_equal = torch.equal(
        weight_qdq.T, weight_transpose_qdq
    )
    if not weight_transpose_equal:
        raise RuntimeError(
            "2D NVFP4 learned-weight quantization is not transpose consistent"
        )

    cosine, sine = _make_rope(args.sequence)
    packed_rope = b300_pack_gqa_d64_paired_rope(cosine, sine)
    paired_qk_scales = torch.zeros(
        1, args.q_heads // 2, 7, device="cuda", dtype=torch.float32
    )
    paired_qk_scales[..., 0] = 2.25
    paired_qk_scales[..., 1] = 2.0

    def project(interleaved: bool):
        return b300_project_qkv_gqa_d64_paired_unified_lowp_nvfp4(
            activation_operand,
            weight_operand,
            paired_qk_scales,
            packed_rope,
            batch=1,
            seqlen=args.sequence,
            q_heads=args.q_heads,
            kv_heads=args.kv_heads,
            store_bf16=True,
            publish_fp8_backward=True,
            interleave_causal_kv=interleaved,
            v_mxfp4_scale_2d=False,
        )

    normal = project(False)
    interleaved = project(True)
    torch.cuda.synchronize()
    if normal.q is None or normal.k is None or normal.v is None:
        raise RuntimeError("normal publisher omitted requested BF16 Q/K/V")
    if interleaved.q is None or interleaved.k is None or interleaved.v is None:
        raise RuntimeError("interleaved publisher omitted requested BF16 Q/K/V")
    backward_tensors = (
        normal.q_backward_fp8,
        normal.k_backward_fp8,
        normal.v_backward_fp8,
        interleaved.q_backward_fp8,
        interleaved.k_backward_fp8,
        interleaved.v_backward_fp8,
    )
    if any(tensor is None for tensor in backward_tensors):
        raise RuntimeError("a D64 publisher omitted an FP8 backward operand")
    assert normal.q_backward_fp8 is not None
    assert normal.k_backward_fp8 is not None
    assert normal.v_backward_fp8 is not None
    assert interleaved.q_backward_fp8 is not None
    assert interleaved.k_backward_fp8 is not None
    assert interleaved.v_backward_fp8 is not None

    route_invariants = {
        "bf16_q_equal": _byte_equal(normal.q, interleaved.q),
        "bf16_k_equal": _byte_equal(normal.k, interleaved.k),
        "bf16_v_equal": _byte_equal(normal.v, interleaved.v),
        "q_backward_fp8_equal": _byte_equal(
            normal.q_backward_fp8, interleaved.q_backward_fp8
        ),
        "k_backward_fp8_equal": _byte_equal(
            normal.k_backward_fp8, interleaved.k_backward_fp8
        ),
        "v_backward_fp8_equal": _byte_equal(
            normal.v_backward_fp8, interleaved.v_backward_fp8
        ),
        "q_forward_payload_equal": _byte_equal(
            normal.q_forward_fp4, interleaved.q_forward_fp4
        ),
        "q_forward_scales_equal": _byte_equal(
            normal.q_forward_scales, interleaved.q_forward_scales
        ),
        "normal_has_feature_major_fp8_v": normal.v_forward_fp8 is not None,
        "interleaved_omits_feature_major_fp8_v": (
            interleaved.v_forward_fp8 is None
        ),
    }
    failed = [name for name, valid in route_invariants.items() if not valid]
    if failed:
        raise RuntimeError(
            "normal/interleaved D64 publication invariants failed: "
            + ", ".join(failed)
        )

    dense_bf16 = _split_qkv(
        F.linear(activation, qkv_weight),
        sequence=args.sequence,
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
    )
    dense_qdq = _split_qkv(
        F.linear(activation_qdq.bfloat16(), weight_qdq.bfloat16()),
        sequence=args.sequence,
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
    )
    dense_bf16 = (
        _apply_rope(dense_bf16[0], cosine, sine),
        _apply_rope(dense_bf16[1], cosine, sine),
        dense_bf16[2],
    )
    dense_qdq = (
        _apply_rope(dense_qdq[0], cosine, sine),
        _apply_rope(dense_qdq[1], cosine, sine),
        dense_qdq[2],
    )
    decoded_normal_q = decode_native_nvfp4_qk(
        normal.q_forward_fp4,
        normal.q_forward_scales,
        normal.q_forward_global_scale,
        scale_tile_rows=128,
    ).movedim(1, 2)
    decoded_normal_k = decode_native_nvfp4_qk(
        normal.k_forward_fp4,
        normal.k_forward_scales,
        normal.k_forward_global_scale,
        scale_tile_rows=64,
    ).movedim(1, 2)

    result = {
        "schema": "gqa_d64_projection_boundaries_v1",
        "configuration": {
            "sequence": args.sequence,
            "hidden": args.hidden,
            "q_heads": args.q_heads,
            "kv_heads": args.kv_heads,
            "head_dim": 64,
            "seed": args.seed,
            "projection_weight_scaling": "shared_2d_16x16",
            "normal_symbol": NORMAL_SYMBOL,
            "interleaved_symbol": INTERLEAVED_SYMBOL,
            "extension": str(Path(extension.__file__).resolve()),
        },
        "input_qdq": tensor_error_metrics(activation, activation_qdq),
        "weight_qdq": tensor_error_metrics(qkv_weight, weight_qdq),
        "weight_transpose_invariant": {
            "block_shape": [16, 16],
            "decoded_transpose_bitwise_equal": weight_transpose_equal,
            "dual_preparation": dual_weight_invariants,
        },
        "dense_qdq_vs_dense_bf16": _component_metrics(
            dense_bf16, dense_qdq
        ),
        "fused_bf16_vs_dense_qdq": _component_metrics(
            dense_qdq, (normal.q, normal.k, normal.v)
        ),
        "normal_vs_interleaved_route_invariants": route_invariants,
        "route_publication_provenance": {
            "normal": {
                "extension_symbol": NORMAL_SYMBOL,
                "causal_interleaved_kv": False,
                "v_mxfp4_payload_shape": list(normal.v_forward_fp4.shape),
                "v_mxfp4_scale_shape": list(normal.v_forward_scales.shape),
                "v_backward_fp8_shape": list(normal.v_backward_fp8.shape),
                "v_forward_fp8_shape": (
                    list(normal.v_forward_fp8.shape)
                    if normal.v_forward_fp8 is not None
                    else None
                ),
            },
            "interleaved_causal": {
                "extension_symbol": INTERLEAVED_SYMBOL,
                "causal_interleaved_kv": True,
                "v_mxfp4_payload_shape": list(
                    interleaved.v_forward_fp4.shape
                ),
                "v_mxfp4_scale_shape": list(
                    interleaved.v_forward_scales.shape
                ),
                "v_backward_fp8_shape": list(
                    interleaved.v_backward_fp8.shape
                ),
                "v_forward_fp8_shape": (
                    list(interleaved.v_forward_fp8.shape)
                    if interleaved.v_forward_fp8 is not None
                    else None
                ),
            },
        },
        "normal_publication_vs_fused_bf16": {
            "q_nvfp4": tensor_error_metrics(normal.q, decoded_normal_q),
            "k_nvfp4": tensor_error_metrics(normal.k, decoded_normal_k),
        },
        "expected_layout_difference": {
            "k_forward_payload_direct_equal": _byte_equal(
                normal.k_forward_fp4, interleaved.k_forward_fp4
            ),
            "k_forward_scales_direct_equal": _byte_equal(
                normal.k_forward_scales, interleaved.k_forward_scales
            ),
            "v_forward_payload_direct_equal": _byte_equal(
                normal.v_forward_fp4, interleaved.v_forward_fp4
            ),
            "v_forward_scales_direct_equal": _byte_equal(
                normal.v_forward_scales, interleaved.v_forward_scales
            ),
        },
        "memory": {
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
