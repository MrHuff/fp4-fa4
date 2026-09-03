#!/usr/bin/env python3
"""Profile a projection-native causal D128 GQA training chain.

This harness connects the retained D128 GQA QKV projection, the read-only
causal NVFP4/MXFP4 forward artifact, the fused dO projection/statistics
producer, and the tuned CuTe backward.  It also compiles a BF16 backward on
the same projected state so gradient quality is measured without materializing
an attention matrix in PyTorch.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import stat
import statistics
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

import torch

from tk_fa4 import (
    b300_pack_gqa_d128_rope,
    b300_pack_gqa_d128_hierarchical_qkv_gradient_nvfp4_tiles,
    b300_pair_interleave_gqa_d128_qk_projection_weights,
    b300_prepare_nvfp4_projection_operand,
    b300_prepare_nvfp4_projection_weight,
    b300_project_dout_unified_lowp_nvfp4,
    b300_project_gqa_d128_hierarchical_qkv_gradient_nvfp4,
    b300_project_nvfp4,
    b300_project_qkv_gqa_d128_unified_lowp_nvfp4,
    b300_stack_gqa_d128_qkv_projection_weights,
)
from tk_fa4.lowp_fa4_bwd.backward_policy import (
    resolve_backward_exp2_policy,
    resolve_backward_raster_policy,
)
from tk_fa4.lowp_fa4_bwd.cutlass_dsl_toolchain import (
    d128_mxfp4_v_compilation_receipt,
)
from tk_fa4.lowp_fa4_bwd.tune_d64_gqa_cute import _load_control


FP8_INPUT_SCALE = 4.0
FP8_DOUT_SCALE = 4.0
FP8_DPSUM_SCALE = FP8_INPUT_SCALE * FP8_DOUT_SCALE
FP8_PROBABILITY_DV_LIFT = 256.0
D128_MXFP4_V_OPERAND_CACHE_CAPACITY = 32


def _extension_file_identity(path: Path) -> dict[str, int | str]:
    """Hash one stable regular, non-symlink extension image."""
    try:
        before = path.lstat()
    except OSError as error:
        raise RuntimeError(f"cannot stat extension artifact {path}") from error
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError(
            f"extension artifact must be a regular non-symlink file: {path}"
        )
    resolved = path.resolve(strict=True)
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    after = resolved.stat()
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(
        getattr(before, field) != getattr(after, field)
        for field in stable_fields
    ):
        raise RuntimeError(f"extension artifact changed while hashing: {path}")
    return {
        "path": str(resolved),
        "sha256": digest.hexdigest(),
        "bytes": after.st_size,
        "device": after.st_dev,
        "inode": after.st_ino,
        "mtime_ns": after.st_mtime_ns,
    }


def _load_extension(path: Path, module_name: str) -> Any:
    identity_before = _extension_file_identity(path)
    resolved = Path(str(identity_before["path"]))
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {resolved}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    identity_after = _extension_file_identity(resolved)
    if identity_after != identity_before:
        raise RuntimeError(
            f"extension artifact changed while loading: {resolved}"
        )
    module._tk_fa4_loaded_artifact_identity = identity_before
    return module


def _make_rope(
    sequence: int,
    depth: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    if depth not in (64, 128):
        raise ValueError("RoPE depth must be 64 or 128")
    pair_count = depth // 2
    positions = torch.arange(sequence, device="cuda", dtype=torch.float32)
    frequencies = 1.0 / (
        10_000.0
        ** (
            torch.arange(pair_count, device="cuda", dtype=torch.float32)
            / pair_count
        )
    )
    angles = positions[:, None] * frequencies[None, :]
    return angles.cos()[None].bfloat16(), angles.sin()[None].bfloat16()


def _bf16_gqa_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize an exact causal BF16-boundary GQA reference once."""
    _, sequence, q_heads, depth = q.shape
    kv_heads = k.shape[2]
    ratio = q_heads // kv_heads
    q_hsd = q[0].permute(1, 0, 2).float()
    k_hsd = k.repeat_interleave(ratio, dim=2)[0].permute(1, 0, 2).float()
    v_hsd = v.repeat_interleave(ratio, dim=2)[0].permute(1, 0, 2).float()
    scores = torch.bmm(q_hsd, k_hsd.transpose(1, 2)) * (depth**-0.5)
    scores.masked_fill_(
        torch.ones(
            sequence,
            sequence,
            device=q.device,
            dtype=torch.bool,
        ).triu_(1),
        float("-inf"),
    )
    lse = torch.logsumexp(scores, dim=-1)
    output = torch.bmm(torch.softmax(scores, dim=-1), v_hsd)
    return (
        output.permute(1, 0, 2).unsqueeze(0).contiguous().bfloat16(),
        lse.unsqueeze(0).unsqueeze(2).contiguous(),
    )


def _metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, Any]:
    reference_f = reference.float().reshape(-1)
    actual_f = actual.float().reshape(-1)
    difference = actual_f - reference_f
    reference_norm = reference_f.norm().clamp_min(1.0e-20)
    actual_norm = actual_f.norm().clamp_min(1.0e-20)
    return {
        "finite": bool(torch.isfinite(actual_f).all()),
        "cosine": float(
            torch.dot(reference_f, actual_f) / (reference_norm * actual_norm)
        ),
        "relative_l2": float(difference.norm() / reference_norm),
        "norm_ratio": float(actual_norm / reference_norm),
        "max_abs": float(difference.abs().max()),
    }


def _hadamard16_blocks(tensor: torch.Tensor) -> torch.Tensor:
    """Apply an orthonormal 16-point Hadamard along contiguous K blocks."""
    if tensor.shape[-1] % 16:
        raise ValueError("Hadamard projection diagnostic requires K % 16 == 0")
    values = tensor.float().reshape(*tensor.shape[:-1], -1, 16)
    for width in (1, 2, 4, 8):
        paired = values.reshape(*values.shape[:-1], -1, 2, width)
        values = torch.cat(
            (paired[..., 0, :] + paired[..., 1, :],
             paired[..., 0, :] - paired[..., 1, :]),
            dim=-1,
        ).reshape(*values.shape)
    return (values * 0.25).reshape(tensor.shape)


def _gqa_dv_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    dout: torch.Tensor,
    *,
    lse_bh1s: torch.Tensor | None = None,
    probability_dtype: torch.dtype | None = None,
    probability_lift: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute dV one query head at a time and retain its GQA partials."""
    batch, sequence, q_heads, depth = q.shape
    kv_heads = k.shape[2]
    if batch != 1 or q_heads % kv_heads:
        raise ValueError("the diagnostic requires batch one and integral GQA")
    ratio = q_heads // kv_heads
    causal_mask = torch.ones(
        sequence,
        sequence,
        device=q.device,
        dtype=torch.bool,
    ).triu_(1)
    partials = torch.empty(
        batch,
        q_heads,
        sequence,
        depth,
        device=q.device,
        dtype=torch.float32,
    )
    for q_head in range(q_heads):
        kv_head = q_head // ratio
        scores = torch.mm(
            q[0, :, q_head].float(),
            k[0, :, kv_head].float().T,
        ) * (depth**-0.5)
        scores.masked_fill_(causal_mask, float("-inf"))
        if lse_bh1s is None:
            probability = torch.softmax(scores, dim=-1)
        else:
            probability = torch.exp(
                scores - lse_bh1s[0, q_head, 0].float()[:, None]
            )
        if probability_dtype is not None:
            probability = (
                (probability * probability_lift)
                .to(probability_dtype)
                .float()
                / probability_lift
            )
        partials[0, q_head] = torch.mm(
            probability.T,
            dout[0, :, q_head].float(),
        )
    reduced = partials.view(
        batch,
        kv_heads,
        ratio,
        sequence,
        depth,
    ).sum(dim=2).permute(0, 2, 1, 3).contiguous()
    return reduced, partials


def _inverse_rope_pair_native(
    tensor: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
) -> torch.Tensor:
    """Apply inverse RoPE to adjacent pair-native D64/D128 Q/K gradients."""
    pair_count = tensor.shape[-1] // 2
    pairs = tensor.float().reshape(*tensor.shape[:-1], pair_count, 2)
    cosine = rope_cos[..., :pair_count].float().unsqueeze(2)
    sine = rope_sin[..., :pair_count].float().unsqueeze(2)
    x = pairs[..., 0]
    y = pairs[..., 1]
    return torch.stack(
        (x * cosine + y * sine, -x * sine + y * cosine),
        dim=-1,
    ).reshape_as(tensor).bfloat16()


def _time_cuda(
    function: Callable[[], object],
    *,
    warmups: int,
    samples: int,
) -> dict[str, Any]:
    for _ in range(warmups):
        function()
    torch.cuda.synchronize()
    values: list[float] = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end) * 1000.0))
    return {
        "median_us": statistics.median(values),
        "minimum_us": min(values),
        "samples_us": values,
    }


def _attention_storage(
    tensor: torch.Tensor,
    *,
    q_heads: int,
    kv_heads: int,
) -> torch.Tensor:
    batch, sequence, heads, depth = tensor.shape
    head_ratio = q_heads // kv_heads if heads == q_heads else 1
    expected_heads = kv_heads * head_ratio
    if heads != expected_heads:
        raise ValueError(f"unexpected attention tensor shape {tuple(tensor.shape)}")
    base = tensor.view(torch.int8) if tensor.dtype == torch.float8_e4m3fn else tensor
    return base.reshape(
        batch, sequence, kv_heads, head_ratio, depth
    ).permute(1, 4, 3, 2, 0)


def _attention_cute_tensor(
    control: Any,
    tensor: torch.Tensor,
    *,
    q_heads: int,
    kv_heads: int,
) -> Any:
    storage = _attention_storage(
        tensor,
        q_heads=q_heads,
        kv_heads=kv_heads,
    )
    cute_tensor = control.from_dlpack(storage, assumed_align=16)
    if tensor.dtype == torch.float8_e4m3fn:
        cute_tensor.element_type = control.Float8E4M3FN
    return cute_tensor.mark_layout_dynamic(
        leading_dim=1
    ).mark_compact_shape_dynamic(
        mode=1,
        stride_order=(4, 0, 3, 2, 1),
        divisibility=64,
    )


def _stats_cute_tensor(
    control: Any,
    tensor: torch.Tensor,
    *,
    q_heads: int,
    kv_heads: int,
) -> Any:
    batch, heads, singleton, sequence = tensor.shape
    if heads != q_heads or singleton != 1:
        raise ValueError(f"unexpected statistics shape {tuple(tensor.shape)}")
    head_ratio = q_heads // kv_heads
    storage = tensor.reshape(
        batch, kv_heads, head_ratio, sequence
    ).permute(3, 2, 1, 0)
    return control.from_dlpack(
        storage, assumed_align=16
    ).mark_layout_dynamic(leading_dim=0)


def _forward_mx_probability_scales_pointer(
    control: Any,
    tensor: torch.Tensor,
    *,
    batch: int,
    q_heads: int,
    sequence: int,
    device: torch.device,
) -> Any:
    expected_shape = (batch, q_heads, sequence // 128, sequence)
    if (
        tensor.dtype != torch.int32
        or not tensor.is_cuda
        or not tensor.is_contiguous()
        or tensor.shape != expected_shape
        or tensor.device != device
    ):
        raise ValueError(
            "forward MX probability scales must be contiguous CUDA int32 "
            f"[B,H,S/128,S] with shape {expected_shape}"
        )
    # A raw pointer keeps the runtime ABI to one address.  The kernel forms
    # its query-contiguous [B,H,Ktile] base once per CTA.
    from cutlass.cute.runtime import make_ptr

    return make_ptr(
        control.Int32,
        tensor.data_ptr(),
        control.cute.AddressSpace.gmem,
        assumed_align=16,
    )


def _require_d128_mxfp4_v_dp_tensor_abi(
    *,
    batch: int,
    sequence: int,
    q_heads: int,
    kv_heads: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v_payload: torch.Tensor,
    v_scale_pages: torch.Tensor | None,
    dout: torch.Tensor,
) -> None:
    """Authenticate the packed MX V ABI without accepting implicit copies."""
    if v_scale_pages is None:
        raise ValueError("D128 MXFP4 V dP requires physical V scale pages")
    expected = {
        "Q": (q, (batch, sequence, q_heads, 128), torch.float8_e4m3fn),
        "K": (k, (batch, sequence, kv_heads, 128), torch.float8_e4m3fn),
        "packed V": (
            v_payload,
            (batch, sequence, kv_heads, 64),
            torch.uint8,
        ),
        "V scale pages": (
            v_scale_pages,
            (batch, sequence // 128, kv_heads, 512),
            torch.uint8,
        ),
        "dO": (
            dout,
            (batch, sequence, q_heads, 128),
            torch.float8_e4m3fn,
        ),
    }
    common_device = q.device
    # The mixed MXF8F6F4 MMA consumes V through the unpacked FP4 TMA
    # representation, whose global payload base must be 32-byte aligned.
    # E8M0 scale pages remain byte-addressed and retain the 16-byte contract.
    required_alignment = {"packed V": 32}
    for name, (tensor, expected_shape, expected_dtype) in expected.items():
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"D128 MXFP4 V dP {name} must have shape "
                f"{expected_shape}, got {tuple(tensor.shape)}"
            )
        if tensor.dtype != expected_dtype:
            raise ValueError(
                f"D128 MXFP4 V dP {name} must have dtype "
                f"{expected_dtype}, got {tensor.dtype}"
            )
        if not tensor.is_cuda:
            raise ValueError(f"D128 MXFP4 V dP {name} must be a CUDA tensor")
        if not tensor.is_contiguous():
            raise ValueError(f"D128 MXFP4 V dP {name} must be contiguous")
        if tensor.device != common_device:
            raise ValueError(
                "D128 MXFP4 V dP operands must share one CUDA device; "
                f"{name} is on {tensor.device}, expected {common_device}"
            )
        alignment = required_alignment.get(name, 16)
        if tensor.data_ptr() % alignment:
            raise ValueError(
                f"D128 MXFP4 V dP {name} must be at least "
                f"{alignment}-byte aligned"
            )


def _require_d128_mxfp4_v_dp_runtime_capability(control: Any) -> None:
    """Fail closed unless the loaded DSL exposes mixed FP8-by-FP4 MMA."""
    tcgen05 = getattr(control, "tcgen05", None)
    if getattr(tcgen05, "MmaMXF8F6F4Op", None) is None:
        raise RuntimeError(
            "D128 MXFP4 V dP requires a CUTLASS DSL runtime exposing "
            "cutlass.cute.nvgpu.tcgen05.MmaMXF8F6F4Op; this capability is "
            "present in the verified CUTLASS DSL 4.5.2 runtime but absent "
            "from 4.4.2"
        )


VERIFIED_BATCHED_D64_CONTROL_SHA256 = (
    "cd57e3360082abe4bad7560c51a7793a4e9bfd4d16efc1259b92ce20238b99e1"
)
VERIFIED_BATCHED_D64_CONTROL_BYTES = 220_876
VERIFIED_BATCHED_D64_BATCHES = (2, 8, 16)
VERIFIED_BATCHED_D128_BATCHES = (2,)
VERIFIED_BATCHED_D128_CONTROL_SHA256 = (
    "cfbd3ad27e5188d39c475abc238b57b5331fc7e631054a7075c7993150c70764"
)
VERIFIED_BATCHED_D128_CONTROL_BYTES = 221_230


def _generated_control_identity(control: Any) -> dict[str, int | str] | None:
    declared = getattr(control, "TK_GENERATED_CONTROL_SOURCE_IDENTITY", None)
    if isinstance(declared, dict):
        return declared
    source = getattr(control, "__file__", None)
    if not isinstance(source, str):
        return None
    path = Path(source).resolve(strict=True)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"sha256": digest, "bytes": path.stat().st_size}


def _require_verified_batched_lowp_tensors(
    *,
    batch: int,
    sequence: int,
    q_heads: int,
    kv_heads: int,
    depth: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    dpsum: torch.Tensor,
    scaled_lse: torch.Tensor,
) -> None:
    """Require the exact physical tensor ABI of a verified batched lane."""
    expected = {
        "Q": (q, (batch, sequence, q_heads, depth), torch.float8_e4m3fn),
        "K": (k, (batch, sequence, kv_heads, depth), torch.float8_e4m3fn),
        "V": (v, (batch, sequence, kv_heads, depth), torch.float8_e4m3fn),
        "dO": (
            dout,
            (batch, sequence, q_heads, depth),
            torch.float8_e4m3fn,
        ),
        "dPsum": (
            dpsum,
            (batch, q_heads, 1, sequence),
            torch.float32,
        ),
        "scaled LSE": (
            scaled_lse,
            (batch, q_heads, 1, sequence),
            torch.float32,
        ),
    }
    common_device = None
    for name, (tensor, expected_shape, expected_dtype) in expected.items():
        actual_shape = tuple(tensor.shape)
        if actual_shape != expected_shape:
            raise ValueError(
                f"batched lowp {name} must have shape "
                f"{expected_shape}, got {actual_shape}"
            )
        if tensor.dtype != expected_dtype:
            raise ValueError(
                f"batched lowp {name} must have dtype "
                f"{expected_dtype}, got {tensor.dtype}"
            )
        if not tensor.is_cuda:
            raise ValueError(f"batched lowp {name} must be a CUDA tensor")
        if not tensor.is_contiguous():
            raise ValueError(
                f"batched lowp {name} must be contiguous"
            )
        if common_device is None:
            common_device = tensor.device
        elif tensor.device != common_device:
            raise ValueError(
                "batched lowp inputs must share one CUDA device; "
                f"{name} is on {tensor.device}, expected {common_device}"
            )


# Retain the established test/import name while the contract now also serves
# the exact D128 B2 lane.
_require_verified_batched_direct_d64_tensors = (
    _require_verified_batched_lowp_tensors
)


def _is_verified_batched_direct_d64_route(
    control: Any,
    *,
    batch: int,
    sequence: int,
    q_heads: int,
    kv_heads: int,
    depth: int,
    lowp: bool,
    precomputed_stats: bool,
    workspace_stats: bool,
    hierarchical_dq_lanes: int,
    signal_dq_tiles: bool,
    owner_output_operand: tuple[torch.Tensor, torch.Tensor] | None,
    owner_quantize_kv: bool,
    reuse_quantized_p: bool,
    forward_mx_probability_replay: bool,
    forward_mx_probability_scales: torch.Tensor | None,
    use_forward_mx_probability_scales: bool,
    reverse_query_tiles: bool,
    head_fast_raster: bool | None,
    direct_tma_dkdv: bool,
    exp2_degree: int,
    exp2_period: int,
    fp8_ds_lift: int | None,
    lowp_do_stages: int,
    scale_softmax: float,
) -> bool:
    """Return whether B2/B8/B16 exactly matches the verified lane."""
    control_provenance = getattr(
        control, "TK_PRECOMPOSED_CONTROL_PROVENANCE", None
    )
    control_source = (
        control_provenance.get("source")
        if isinstance(control_provenance, dict)
        else None
    )
    return (
        batch in VERIFIED_BATCHED_D64_BATCHES
        and sequence == 4096
        and q_heads == 32
        and kv_heads == 8
        and depth == 64
        and lowp
        and precomputed_stats
        and workspace_stats
        and hierarchical_dq_lanes == 1
        and not signal_dq_tiles
        and owner_output_operand is None
        and not owner_quantize_kv
        and not reuse_quantized_p
        and not forward_mx_probability_replay
        and forward_mx_probability_scales is None
        and not use_forward_mx_probability_scales
        and not reverse_query_tiles
        and (head_fast_raster is None or head_fast_raster is False)
        and direct_tma_dkdv is True
        and exp2_degree == 1
        and exp2_period == 2
        and fp8_ds_lift == 16
        and lowp_do_stages == 1
        and scale_softmax == (64**-0.5) / 16.0
        and getattr(control, "TK_FP8_P_STORAGE", None) == "tmem"
        and not bool(getattr(control, "TK_DETACHED_FP8_P_TMEM", False))
        and isinstance(control_provenance, dict)
        and control_provenance.get("mode") == "precomposed"
        and isinstance(control_source, dict)
        and control_source.get("sha256")
        == VERIFIED_BATCHED_D64_CONTROL_SHA256
        and control_source.get("bytes") == VERIFIED_BATCHED_D64_CONTROL_BYTES
    )


def _is_verified_batched_shared_d128_route(
    control: Any,
    *,
    batch: int,
    sequence: int,
    q_heads: int,
    kv_heads: int,
    depth: int,
    lowp: bool,
    precomputed_stats: bool,
    workspace_stats: bool,
    hierarchical_dq_lanes: int,
    signal_dq_tiles: bool,
    owner_output_operand: tuple[torch.Tensor, torch.Tensor] | None,
    owner_quantize_kv: bool,
    reuse_quantized_p: bool,
    forward_mx_probability_replay: bool,
    forward_mx_probability_scales: torch.Tensor | None,
    use_forward_mx_probability_scales: bool,
    reverse_query_tiles: bool,
    head_fast_raster: bool | None,
    direct_tma_dkdv: bool,
    exp2_degree: int,
    exp2_period: int,
    fp8_ds_lift: int | None,
    lowp_do_stages: int,
    scale_softmax: float,
) -> bool:
    """Return whether B2 exactly matches the retained D128 shared-P lane."""
    control_identity = _generated_control_identity(control)
    return (
        batch in VERIFIED_BATCHED_D128_BATCHES
        and sequence == 4096
        and q_heads == 32
        and kv_heads == 8
        and depth == 128
        and lowp
        and precomputed_stats
        and workspace_stats
        and hierarchical_dq_lanes == 1
        and not signal_dq_tiles
        and owner_output_operand is None
        and not owner_quantize_kv
        and reuse_quantized_p
        and not forward_mx_probability_replay
        and forward_mx_probability_scales is None
        and not use_forward_mx_probability_scales
        and not reverse_query_tiles
        and (head_fast_raster is None or head_fast_raster is False)
        and direct_tma_dkdv is False
        and exp2_degree == 1
        and exp2_period == 0
        and fp8_ds_lift == 256
        and lowp_do_stages == 2
        and scale_softmax == (128**-0.5) / 16.0
        and getattr(control, "TK_FP8_P_STORAGE", None) == "shared"
        and not bool(getattr(control, "TK_DETACHED_FP8_P_TMEM", False))
        and not bool(getattr(control, "TK_DIRECT_TMA_DKDV", False))
        and getattr(control, "TK_PRECOMPOSED_CONTROL_PROVENANCE", None) is None
        and isinstance(control_identity, dict)
        and control_identity.get("sha256")
        == VERIFIED_BATCHED_D128_CONTROL_SHA256
        and control_identity.get("bytes")
        == VERIFIED_BATCHED_D128_CONTROL_BYTES
    )
class CompiledGqaBackward:
    def __init__(
        self,
        control: Any,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o_or_sum: torch.Tensor,
        dout: torch.Tensor,
        lse_or_scaled_lse: torch.Tensor,
        q_heads: int,
        kv_heads: int,
        lowp: bool,
        precomputed_stats: bool,
        scale_softmax: float,
        workspace_stats: bool = False,
        workspace_torch: torch.Tensor | None = None,
        dq_torch: torch.Tensor | None = None,
        hierarchical_dq_lanes: int = 1,
        signal_dq_tiles: bool = False,
        owner_output_operand: tuple[torch.Tensor, torch.Tensor] | None = None,
        owner_gradient_global_scale: torch.Tensor | None = None,
        owner_rope: torch.Tensor | None = None,
        exp2_degree: int = 1,
        exp2_period: int | None = None,
        reuse_quantized_p: bool = False,
        forward_mx_probability_replay: bool = False,
        forward_mx_probability_scales: torch.Tensor | None = None,
        use_forward_mx_probability_scales: bool = False,
        fp8_ds_lift: int | None = None,
        lowp_do_stages: int | None = None,
        owner_quantize_kv: bool = False,
        reverse_query_tiles: bool = False,
        head_fast_raster: bool | None = None,
        direct_tma_dkdv: bool = False,
        use_d128_mxfp4_v_dp: bool = False,
        v_mxfp4_scale_pages: torch.Tensor | None = None,
    ) -> None:
        batch, sequence, _, depth = q.shape
        if batch <= 0 or depth not in (64, 128):
            raise ValueError(
                "the retained chain requires a positive batch and D64 or D128"
            )
        is_d64 = depth == 64
        exp2_policy = resolve_backward_exp2_policy(
            sequence=sequence,
            head_dim=depth,
            q_heads=q_heads,
            kv_heads=kv_heads,
            lowp=lowp,
            exp2_degree=exp2_degree,
            exp2_period=exp2_period,
        )
        effective_lowp_do_stages = (
            lowp_do_stages
            if lowp and lowp_do_stages is not None
            else ((1 if lowp else 2) if is_d64 else (2 if lowp else 1))
        )
        control_uses_d128_mxfp4_v_dp = bool(
            getattr(control, "TK_D128_MXFP4_V_DP", False)
        )
        if use_d128_mxfp4_v_dp != control_uses_d128_mxfp4_v_dp:
            raise ValueError(
                "use_d128_mxfp4_v_dp must match the loaded CuTe control module"
            )
        if not use_d128_mxfp4_v_dp and v_mxfp4_scale_pages is not None:
            raise ValueError(
                "V MXFP4 scale pages require use_d128_mxfp4_v_dp=True"
            )
        if (
            lowp
            and not use_d128_mxfp4_v_dp
            and v.dtype != torch.float8_e4m3fn
        ):
            raise ValueError(
                "packed MXFP4 V requires use_d128_mxfp4_v_dp=True and its "
                "physical scale pages; the retained lowp route requires E4M3 V"
            )
        batched_direct_d64 = _is_verified_batched_direct_d64_route(
            control,
            batch=batch,
            sequence=sequence,
            q_heads=q_heads,
            kv_heads=kv_heads,
            depth=depth,
            lowp=lowp,
            precomputed_stats=precomputed_stats,
            workspace_stats=workspace_stats,
            hierarchical_dq_lanes=hierarchical_dq_lanes,
            signal_dq_tiles=signal_dq_tiles,
            owner_output_operand=owner_output_operand,
            owner_quantize_kv=owner_quantize_kv,
            reuse_quantized_p=reuse_quantized_p,
            forward_mx_probability_replay=forward_mx_probability_replay,
            forward_mx_probability_scales=forward_mx_probability_scales,
            use_forward_mx_probability_scales=(
                use_forward_mx_probability_scales
            ),
            reverse_query_tiles=reverse_query_tiles,
            head_fast_raster=head_fast_raster,
            direct_tma_dkdv=direct_tma_dkdv,
            exp2_degree=exp2_policy.effective_degree,
            exp2_period=exp2_policy.effective_period,
            fp8_ds_lift=fp8_ds_lift,
            lowp_do_stages=effective_lowp_do_stages,
            scale_softmax=scale_softmax,
        )
        batched_shared_d128 = (
            not use_d128_mxfp4_v_dp
            and _is_verified_batched_shared_d128_route(
                control,
                batch=batch,
                sequence=sequence,
                q_heads=q_heads,
                kv_heads=kv_heads,
                depth=depth,
                lowp=lowp,
                precomputed_stats=precomputed_stats,
                workspace_stats=workspace_stats,
                hierarchical_dq_lanes=hierarchical_dq_lanes,
                signal_dq_tiles=signal_dq_tiles,
                owner_output_operand=owner_output_operand,
                owner_quantize_kv=owner_quantize_kv,
                reuse_quantized_p=reuse_quantized_p,
                forward_mx_probability_replay=forward_mx_probability_replay,
                forward_mx_probability_scales=forward_mx_probability_scales,
                use_forward_mx_probability_scales=(
                    use_forward_mx_probability_scales
                ),
                reverse_query_tiles=reverse_query_tiles,
                head_fast_raster=head_fast_raster,
                direct_tma_dkdv=direct_tma_dkdv,
                exp2_degree=exp2_policy.effective_degree,
                exp2_period=exp2_policy.effective_period,
                fp8_ds_lift=fp8_ds_lift,
                lowp_do_stages=effective_lowp_do_stages,
                scale_softmax=scale_softmax,
            )
        )
        d128_mxfp4_v_dp_route = (
            use_d128_mxfp4_v_dp
            and batch in (1, 2)
            and sequence == 4096
            and q_heads == 32
            and kv_heads == 8
            and depth == 128
            and lowp
            and precomputed_stats
            and hierarchical_dq_lanes == 1
            and not signal_dq_tiles
            and owner_output_operand is None
            and not owner_quantize_kv
            and reuse_quantized_p
            and not forward_mx_probability_replay
            and forward_mx_probability_scales is None
            and not use_forward_mx_probability_scales
            and not reverse_query_tiles
            and (head_fast_raster is None or head_fast_raster is False)
            and direct_tma_dkdv is False
            and exp2_policy.effective_degree == 1
            and exp2_policy.effective_period == 0
            and fp8_ds_lift in (None, 256)
            and effective_lowp_do_stages == 2
            and scale_softmax == (128**-0.5) / 16.0
            and getattr(control, "TK_FP8_P_STORAGE", None) == "shared"
            and not bool(getattr(control, "TK_DETACHED_FP8_P_TMEM", False))
            and not bool(getattr(control, "TK_DIRECT_TMA_DKDV", False))
            and getattr(control, "TK_PRECOMPOSED_CONTROL_PROVENANCE", None)
            is None
        )
        if use_d128_mxfp4_v_dp and not d128_mxfp4_v_dp_route:
            raise ValueError(
                "D128 MXFP4 V dP is gated to the B1/B2 S4096 H32/Hkv8 "
                "lowp shared-P route with reused P, two dO stages, native "
                "EX2, and the x16 statistics contract"
            )
        if batch != 1 and not (
            batched_direct_d64
            or batched_shared_d128
            or d128_mxfp4_v_dp_route
        ):
            raise ValueError(
                "batched lowp backward is retained only for the authenticated "
                "D64 direct-TMA B2/B8/B16 lane, D128 shared-P B2 lane, or "
                "D128 MXFP4-V B2 lane"
            )
        if batched_direct_d64 or batched_shared_d128:
            _require_verified_batched_lowp_tensors(
                batch=batch,
                sequence=sequence,
                q_heads=q_heads,
                kv_heads=kv_heads,
                depth=depth,
                q=q,
                k=k,
                v=v,
                dout=dout,
                dpsum=o_or_sum,
                scaled_lse=lse_or_scaled_lse,
            )
        if d128_mxfp4_v_dp_route:
            _require_d128_mxfp4_v_dp_tensor_abi(
                batch=batch,
                sequence=sequence,
                q_heads=q_heads,
                kv_heads=kv_heads,
                q=q,
                k=k,
                v_payload=v,
                v_scale_pages=v_mxfp4_scale_pages,
                dout=dout,
            )
            _require_d128_mxfp4_v_dp_runtime_capability(control)
        self.exp2_policy = exp2_policy.as_dict()
        self.exp2_degree = exp2_policy.effective_degree
        self.exp2_period = exp2_policy.effective_period
        if direct_tma_dkdv != bool(
            getattr(control, "TK_DIRECT_TMA_DKDV", False)
        ):
            raise ValueError(
                "direct_tma_dkdv must match the loaded CuTe control module"
            )
        self.detached_fp8_p_tmem = bool(
            getattr(control, "TK_DETACHED_FP8_P_TMEM", False)
        )
        if self.detached_fp8_p_tmem and (not lowp or not is_d64):
            raise ValueError("detached FP8 P TMEM requires the D64 lowp route")
        if forward_mx_probability_replay and (not lowp or not is_d64):
            raise ValueError(
                "forward MX probability replay requires the D64 FP8-input "
                "backward route"
            )
        if forward_mx_probability_replay and reuse_quantized_p:
            raise ValueError(
                "forward MX probability replay cannot reuse the E4M3 dV "
                "probability for dS"
            )
        if (
            use_forward_mx_probability_scales
            and not forward_mx_probability_replay
        ):
            raise ValueError(
                "forward MX probability scales require exact probability replay"
            )
        if use_forward_mx_probability_scales and sequence % 128:
            raise ValueError(
                "forward MX probability scales require S divisible by 128"
            )
        if (
            forward_mx_probability_scales is not None
            and not use_forward_mx_probability_scales
        ):
            raise ValueError(
                "set use_forward_mx_probability_scales=True when binding "
                "forward scales"
            )
        base = control.BlackwellFusedMultiHeadAttentionBackward
        do_stages = effective_lowp_do_stages
        dkdv_stages = 2 if is_d64 else (1 if lowp else 2)
        owner_quantize_dq = owner_output_operand is not None
        raster_auto_eligible = (
            lowp
            and is_d64
            and direct_tma_dkdv
            and hierarchical_dq_lanes == 1
            and not owner_quantize_dq
            and not owner_quantize_kv
            and precomputed_stats
            and workspace_stats
            and not reuse_quantized_p
            and not reverse_query_tiles
            and fp8_ds_lift == 16
            and do_stages == 1
            and self.exp2_degree == 1
            and self.exp2_period == 2
            and getattr(control, "TK_FP8_P_STORAGE", None) == "tmem"
        )
        raster_policy = resolve_backward_raster_policy(
            sequence=sequence,
            head_dim=depth,
            q_heads=q_heads,
            kv_heads=kv_heads,
            batch=batch,
            lowp=lowp,
            head_fast_raster=head_fast_raster,
            auto_eligible=raster_auto_eligible,
            force_head_fast=owner_quantize_dq,
        )
        self.raster_policy = raster_policy.as_dict()
        self.head_fast_raster = raster_policy.effective_head_fast

        class ExternalGqaBackward(base):
            def _setup_attributes(self):
                super()._setup_attributes()
                self.load_mma_Q_stage = 2
                self.load_mma_dO_stage = do_stages
                self.mma_compute_dKdV_stage = dkdv_stages

        element_dtype = control.Float8E4M3FN if lowp else control.BFloat16
        self.kernel = ExternalGqaBackward(
            element_dtype,
            control.Float32,
            (128, 128, depth),
            False,
            control.fmha_utils.MaskEnum.WINDOW_MASK_BWD,
        )
        self.kernel.num_regs_reduce = (
            136 if use_d128_mxfp4_v_dp else (136 if is_d64 else 152)
        )
        self.kernel.num_regs_compute = (
            136 if use_d128_mxfp4_v_dp else (136 if is_d64 else 128)
        )
        self.kernel.num_regs_mma = 96
        self.kernel.num_regs_load = 96
        self.kernel.split_gqa_heads = True
        self.kernel.fuse_gqa_reduce = True
        self.kernel.fused_convert_block_seq = (
            32 if is_d64 else (64 if lowp else 32)
        )
        self.kernel.gqa_reduce_vector = 4
        self.kernel.gqa_reduce_threads = 256
        self.kernel.compact_dq_acc = lowp
        if hierarchical_dq_lanes not in (1, 2):
            raise ValueError("hierarchical_dq_lanes must be one or two")
        if hierarchical_dq_lanes > 1 and not lowp:
            raise ValueError("hierarchical dQ lanes require the lowp route")
        if signal_dq_tiles and hierarchical_dq_lanes != 2:
            raise ValueError("dQ tile signaling requires two reduction lanes")
        if direct_tma_dkdv and (
            not lowp
            or not is_d64
            or hierarchical_dq_lanes != 1
            or owner_quantize_dq
            or owner_quantize_kv
        ):
            raise ValueError(
                "direct dK/dV TMA reduction requires the one-lane D64 lowp "
                "route without owner quantization"
            )
        if owner_quantize_kv and not owner_quantize_dq:
            raise ValueError("owner K/V quantization requires owner dQ output")
        if owner_quantize_dq:
            if (
                batch != 1
                or not lowp
                or hierarchical_dq_lanes != 1
                or owner_gradient_global_scale is None
                or owner_rope is None
            ):
                raise ValueError(
                    "owner dQ quantization requires batch 1, the one-lane "
                    "lowp route, a gradient scale, and packed inverse RoPE"
                )
        self.kernel.hierarchical_dq_lanes = hierarchical_dq_lanes
        self.kernel.signal_dq_tiles = signal_dq_tiles
        self.kernel.owner_quantize_dq = owner_quantize_dq
        self.kernel.owner_quantize_kv = owner_quantize_kv
        self.kernel.reverse_query_tiles = lowp and reverse_query_tiles
        self.kernel.head_fast_raster = self.head_fast_raster
        self.kernel.direct_compact_dq = (
            lowp and hierarchical_dq_lanes == 1 and not owner_quantize_dq
        )
        self.kernel.use_precomputed_stats = precomputed_stats and not workspace_stats
        self.kernel.skip_stats_preprocess = workspace_stats
        self.kernel.exp2_alu_degree = self.exp2_degree
        self.kernel.exp2_alu_period = self.exp2_period
        self.kernel.reuse_quantized_p = lowp and reuse_quantized_p
        self.kernel.forward_mx_probability_replay = bool(
            forward_mx_probability_replay
        )
        self.kernel.use_forward_mx_probability_scales = bool(
            use_forward_mx_probability_scales
        )
        self.kernel.use_d128_mxfp4_v_dp = bool(use_d128_mxfp4_v_dp)
        self.kernel.fuse_probability_lift = lowp and is_d64
        self.kernel.prelift_probability_lse = (
            lowp
            and is_d64
            and (not precomputed_stats or workspace_stats)
        )
        if (
            forward_mx_probability_replay
            and not self.kernel.prelift_probability_lse
        ):
            raise ValueError(
                "forward MX probability replay requires prelifted workspace LSE"
            )
        if fp8_ds_lift is None:
            if is_d64 and sequence == 4096:
                # Model-distributed S4096 gradients saturate badly at the
                # old 2^8 lift.  The measured accuracy optimum is 2^4:
                # dQ/dK cosine rises from about 0.857/0.846 to
                # 0.997/0.997 with no measurable kernel-time regression.
                fp8_ds_lift = 16
            elif sequence <= 512:
                fp8_ds_lift = 32
            elif sequence <= 1024:
                fp8_ds_lift = 64
            elif sequence <= 2048:
                fp8_ds_lift = 128
            else:
                fp8_ds_lift = 256
        if fp8_ds_lift not in (16, 32, 64, 128, 256):
            raise ValueError("fp8_ds_lift must be a power of two in [16, 256]")
        self.kernel.fp8_ds_lift = fp8_ds_lift

        self.dq = (
            dq_torch
            if dq_torch is not None
            else torch.zeros_like(q, dtype=torch.bfloat16)
        )
        if (
            self.dq.shape != q.shape
            or self.dq.dtype != torch.bfloat16
            or not self.dq.is_cuda
            or not self.dq.is_contiguous()
            or self.dq.device != q.device
        ):
            raise ValueError(
                "external dQ must be contiguous CUDA BF16 and match Q"
            )
        self.dk = torch.empty_like(k, dtype=torch.bfloat16)
        # Packed MX V has depth 64 in bytes.  dV remains a full-width BF16
        # gradient and follows K ownership irrespective of the dP V format.
        self.dv = torch.empty_like(k, dtype=torch.bfloat16)
        problem_shape = (
            sequence,
            sequence,
            depth,
            ((q_heads // kv_heads, kv_heads), batch),
        )
        q_cute = _attention_cute_tensor(
            control, q, q_heads=q_heads, kv_heads=kv_heads
        )
        k_cute = _attention_cute_tensor(
            control, k, q_heads=q_heads, kv_heads=kv_heads
        )
        mxfp4_v_scale_iter = None
        if use_d128_mxfp4_v_dp:
            assert v_mxfp4_scale_pages is not None
            v_cute = control.from_dlpack(v, assumed_align=32)
            v_cute.element_type = control.cutlass.Float4E2M1FN
            v_scale_cute = control.from_dlpack(
                v_mxfp4_scale_pages,
                assumed_align=16,
            )
            v_scale_cute.element_type = control.cutlass.Float8E8M0FNU
            mxfp4_v_scale_iter = v_scale_cute.iterator
        else:
            v_cute = _attention_cute_tensor(
                control, v, q_heads=q_heads, kv_heads=kv_heads
            )
        if precomputed_stats:
            o_cute = _stats_cute_tensor(
                control,
                o_or_sum,
                q_heads=q_heads,
                kv_heads=kv_heads,
            )
        else:
            o_cute = _attention_cute_tensor(
                control,
                o_or_sum,
                q_heads=q_heads,
                kv_heads=kv_heads,
            )
        dq_cute = _attention_cute_tensor(
            control, self.dq, q_heads=q_heads, kv_heads=kv_heads
        )
        dk_cute = _attention_cute_tensor(
            control, self.dk, q_heads=q_heads, kv_heads=kv_heads
        )
        dv_cute = _attention_cute_tensor(
            control, self.dv, q_heads=q_heads, kv_heads=kv_heads
        )
        dout_cute = _attention_cute_tensor(
            control, dout, q_heads=q_heads, kv_heads=kv_heads
        )
        lse_cute = _stats_cute_tensor(
            control,
            lse_or_scaled_lse,
            q_heads=q_heads,
            kv_heads=kv_heads,
        )
        workspace_size = self.kernel._get_workspace_size(
            sequence,
            sequence,
            depth,
            q_heads,
            batch,
            control.Float32,
            control.BFloat16,
        )
        workspace_prepopulated = workspace_torch is not None
        if workspace_torch is None:
            self.workspace_torch = torch.zeros(
                workspace_size,
                device=q.device,
                dtype=torch.uint8,
            )
        else:
            if (
                workspace_torch.dtype != torch.uint8
                or not workspace_torch.is_cuda
                or not workspace_torch.is_contiguous()
                or workspace_torch.ndim != 1
                or workspace_torch.device != q.device
                or workspace_torch.numel() < workspace_size
            ):
                raise ValueError(
                    "preallocated backward workspace has an invalid layout "
                    f"or fewer than {workspace_size} bytes"
                )
            self.workspace_torch = workspace_torch
        if workspace_stats and not workspace_prepopulated:
            stats_numel = batch * q_heads * sequence
            workspace_stats_view = self.workspace_torch[
                : 2 * stats_numel * 4
            ].view(torch.float32)
            workspace_stats_view[:stats_numel].copy_(o_or_sum.reshape(-1))
            workspace_stats_view[stats_numel:].copy_(
                lse_or_scaled_lse.reshape(-1)
            )
        stats_bytes = 2 * batch * q_heads * sequence * torch.float32.itemsize
        if self.kernel.direct_compact_dq:
            dq_acc_bytes = 0
        elif self.kernel.compact_dq_acc:
            dq_acc_bytes = (
                hierarchical_dq_lanes
                * batch
                * q_heads
                * sequence
                * depth
                * torch.bfloat16.itemsize
            )
        else:
            dq_acc_bytes = (
                batch
                * q_heads
                * sequence
                * depth
                * torch.float32.itemsize
            )
        partial_numel = batch * q_heads * sequence * depth
        partial_bytes = partial_numel * torch.bfloat16.itemsize
        self.dk_partials = self.workspace_torch.narrow(
            0,
            stats_bytes + dq_acc_bytes,
            partial_bytes,
        ).view(torch.bfloat16).view(batch, q_heads, sequence, depth)
        self.dv_partials = self.workspace_torch.narrow(
            0,
            stats_bytes + dq_acc_bytes + partial_bytes,
            partial_bytes,
        ).view(torch.bfloat16).view(batch, q_heads, sequence, depth)
        workspace = control.from_dlpack(
            self.workspace_torch, assumed_align=16
        ).mark_layout_dynamic()
        self.dq_lanes = None
        self.dq_tile_arrivals = None
        self.owner_dq_acc = None
        self.owner_dq_clear = None
        self.owner_dq_ready = None
        if hierarchical_dq_lanes > 1:
            stats_bytes = 2 * batch * q_heads * sequence * 4
            dq_lane_numel = (
                hierarchical_dq_lanes
                * batch
                * q_heads
                * sequence
                * depth
            )
            self.dq_lanes = self.workspace_torch.narrow(
                0,
                stats_bytes,
                dq_lane_numel * torch.bfloat16.itemsize,
            ).view(torch.bfloat16).view(
                hierarchical_dq_lanes,
                batch,
                q_heads,
                sequence,
                depth,
            )
            if signal_dq_tiles:
                arrival_numel = batch * q_heads * (sequence // 128)
                self.dq_tile_arrivals = self.workspace_torch.narrow(
                    0,
                    workspace_size - arrival_numel * torch.int32.itemsize,
                    arrival_numel * torch.int32.itemsize,
                ).view(torch.int32).view(batch, q_heads, sequence // 128)
        owner_dq_fp4_iter = None
        owner_dq_scale_iter = None
        owner_global_scale_cute = None
        owner_rope_packed_cute = None
        owner_ready_cute = None
        if owner_quantize_dq:
            assert owner_output_operand is not None
            assert owner_gradient_global_scale is not None
            assert owner_rope is not None
            owner_payload, owner_scales = owner_output_operand
            reduction = (q_heads + 2 * kv_heads) * depth
            if (
                owner_payload.dtype != torch.float4_e2m1fn_x2
                or owner_payload.shape != (batch * sequence, reduction // 2)
                or not owner_payload.is_cuda
                or not owner_payload.is_contiguous()
                or owner_payload.device != q.device
            ):
                raise ValueError("owner payload must be contiguous E2M1 [B*S,K/2]")
            if (
                owner_scales.dtype != torch.float8_e4m3fn
                or owner_scales.shape
                != (batch * sequence // 128, reduction // 64, 512)
                or not owner_scales.is_cuda
                or not owner_scales.is_contiguous()
                or owner_scales.device != q.device
            ):
                raise ValueError(
                    "owner scales must be contiguous E4M3 [B*S/128,K/64,512]"
                )
            if (
                owner_gradient_global_scale.dtype != torch.float32
                or not owner_gradient_global_scale.is_cuda
                or not owner_gradient_global_scale.is_contiguous()
                or owner_gradient_global_scale.numel() != 1
                or owner_gradient_global_scale.device != q.device
            ):
                raise ValueError("owner gradient scale must be one CUDA float32")
            expected_rope = (batch, sequence, depth // 2)
            if (
                owner_rope.dtype != torch.int32
                or not owner_rope.is_cuda
                or not owner_rope.is_contiguous()
                or owner_rope.shape != expected_rope
                or owner_rope.device != q.device
            ):
                raise ValueError(
                    "owner RoPE must be packed contiguous int32 [B,S,D/2]"
                )

            stats_bytes = 2 * batch * q_heads * sequence * 4
            dq_numel = batch * q_heads * sequence * depth
            self.owner_dq_acc = self.workspace_torch.narrow(
                0,
                stats_bytes,
                dq_numel * torch.bfloat16.itemsize,
            ).view(torch.bfloat16).view(batch, q_heads, sequence, depth)
            self.owner_dq_clear = self.owner_dq_acc.view(
                batch, sequence, q_heads, depth
            )
            self.owner_dq_ready = torch.zeros(
                batch,
                q_heads,
                sequence // 128,
                device=q.device,
                dtype=torch.int32,
            )

            from cutlass.cute.runtime import make_ptr

            owner_dq_fp4_iter = make_ptr(
                control.cutlass.Uint8,
                owner_payload.data_ptr(),
                control.cute.AddressSpace.gmem,
                assumed_align=16,
            )
            owner_dq_scale_iter = make_ptr(
                control.cutlass.Uint8,
                owner_scales.data_ptr(),
                control.cute.AddressSpace.gmem,
                assumed_align=16,
            )
            owner_global_scale_cute = control.from_dlpack(
                owner_gradient_global_scale,
                assumed_align=16,
            ).mark_layout_dynamic()
            owner_rope_packed_cute = control.from_dlpack(
                owner_rope,
                assumed_align=16,
            ).mark_layout_dynamic()
            owner_ready_cute = control.from_dlpack(
                self.owner_dq_ready,
                assumed_align=16,
            ).mark_layout_dynamic()
        self._control = control
        self._d128_mxfp4_v_dp_validation_tensors = (
            (q, k, dout) if use_d128_mxfp4_v_dp else None
        )
        self.v_mxfp4_payload = v if use_d128_mxfp4_v_dp else None
        self.v_mxfp4_scale_pages = (
            v_mxfp4_scale_pages if use_d128_mxfp4_v_dp else None
        )
        self._forward_mx_probability_scale_shape = (
            batch,
            q_heads,
            sequence // 128,
            sequence,
        )
        self._forward_mx_probability_scales_bound = False
        self.forward_mx_probability_scales = None
        forward_mx_probability_scales_ptr = None
        if use_forward_mx_probability_scales:
            if forward_mx_probability_scales is None:
                # Compilation needs the runtime pointer type before the
                # corresponding forward layer has produced its scales.
                forward_mx_probability_scales = torch.empty(
                    self._forward_mx_probability_scale_shape,
                    device=q.device,
                    dtype=torch.int32,
                )
            else:
                self._forward_mx_probability_scales_bound = True
            self.forward_mx_probability_scales = forward_mx_probability_scales
            forward_mx_probability_scales_ptr = (
                _forward_mx_probability_scales_pointer(
                    control,
                    forward_mx_probability_scales,
                    batch=batch,
                    q_heads=q_heads,
                    sequence=sequence,
                    device=q.device,
                )
            )
        stream = control.cutlass_torch.default_stream()
        self._forward_mx_probability_scales_argument_index = 16
        self.arguments = (
            problem_shape,
            q_cute,
            k_cute,
            v_cute,
            o_cute,
            dq_cute,
            dk_cute,
            dv_cute,
            dout_cute,
            lse_cute,
            None,
            None,
            scale_softmax,
            None,
            control.Int32(0),
            workspace,
            forward_mx_probability_scales_ptr,
            owner_dq_fp4_iter,
            owner_dq_scale_iter,
            owner_global_scale_cute,
            owner_rope_packed_cute,
            owner_ready_cute,
            stream,
        )
        if use_d128_mxfp4_v_dp:
            # The conditional patch adds one final optional argument.  The
            # retained control keeps its exact 0..22 positional ABI.
            self.arguments += (mxfp4_v_scale_iter,)
            self._d128_mxfp4_v_scale_argument_index = len(self.arguments) - 1
        else:
            self._d128_mxfp4_v_scale_argument_index = None
        self._initialize_d128_mxfp4_v_operand_cache(
            enabled=use_d128_mxfp4_v_dp
        )
        self.compiled = control.cute.compile(self.kernel, *self.arguments)
        self._d128_mxfp4_v_compilation_receipt = (
            d128_mxfp4_v_compilation_receipt(
                control=control,
                compiled=self.compiled,
                kernel=self.kernel,
            )
            if use_d128_mxfp4_v_dp
            else None
        )
        self.lowp = lowp
        self.direct_tma_dkdv = direct_tma_dkdv
        self.hierarchical_dq_lanes = hierarchical_dq_lanes
        self.arrival_epoch = 0

    def _initialize_d128_mxfp4_v_operand_cache(
        self,
        *,
        enabled: bool,
    ) -> None:
        """Create the candidate-only bounded cache used by layer workspaces."""
        self._d128_mxfp4_v_operand_cache = (
            OrderedDict() if enabled else None
        )
        self._d128_mxfp4_v_operand_cache_counters = (
            {
                "hits": 0,
                "misses": 0,
                "full_validations": 0,
                "dlpack_wrapper_builds": 0,
                "invalidations": 0,
                "evictions": 0,
            }
            if enabled
            else None
        )

    @staticmethod
    def _d128_mxfp4_v_tensor_binding_identity(
        tensor: torch.Tensor,
    ) -> tuple[Any, ...]:
        """Capture pointer and view ABI while deliberately ignoring contents."""
        return (
            int(tensor.data_ptr()),
            tuple(int(extent) for extent in tensor.shape),
            tuple(int(stride) for stride in tensor.stride()),
            int(tensor.storage_offset()),
            tensor.dtype,
            tensor.device,
            bool(tensor.is_cuda),
            bool(tensor.is_contiguous()),
        )

    def d128_mxfp4_v_operand_cache_receipt(
        self,
    ) -> dict[str, Any] | None:
        """Report mutable cache diagnostics outside the backward contract."""
        cache = self._d128_mxfp4_v_operand_cache
        counters = self._d128_mxfp4_v_operand_cache_counters
        if cache is None:
            if counters is not None:
                raise RuntimeError(
                    "retained D128 MXFP4 V operand cache state is malformed"
                )
            return None
        if counters is None or len(cache) > D128_MXFP4_V_OPERAND_CACHE_CAPACITY:
            raise RuntimeError("D128 MXFP4 V operand cache state is malformed")
        return {
            "schema": "d128_mxfp4_v_operand_cache_v1",
            "capacity": D128_MXFP4_V_OPERAND_CACHE_CAPACITY,
            "entries": len(cache),
            **{name: int(value) for name, value in counters.items()},
            "key_contract": "exact_tensor_identity_pointer_and_view_abi",
            "strong_reference_owners": True,
            "static_constructor_q_k_do_revalidated_on_hit": False,
            "live_q_k_do_rebind_path_unchanged": True,
        }

    def d128_mxfp4_v_compilation_receipt(self) -> dict[str, Any] | None:
        """Return immutable candidate compiler and generated-code identity."""
        receipt = self._d128_mxfp4_v_compilation_receipt
        enabled = bool(self.kernel.use_d128_mxfp4_v_dp)
        if enabled and receipt is None:
            raise RuntimeError(
                "D128 MXFP4 V backward is missing compilation provenance"
            )
        if not enabled and receipt is not None:
            raise RuntimeError(
                "retained backward unexpectedly carries MXFP4-V compiler "
                "provenance"
            )
        if receipt is None:
            return None
        # Do not let a result consumer mutate the runner's authenticated state.
        return json.loads(json.dumps(receipt, sort_keys=True))

    def bind_forward_mx_probability_scales(
        self, tensor: torch.Tensor
    ) -> None:
        """Rebind the packed forward scale page used by the next backward."""
        if not self.kernel.use_forward_mx_probability_scales:
            raise RuntimeError(
                "this backward was not compiled to consume forward MX scales"
            )
        batch, q_heads, _, sequence = (
            self._forward_mx_probability_scale_shape
        )
        pointer = _forward_mx_probability_scales_pointer(
            self._control,
            tensor,
            batch=batch,
            q_heads=q_heads,
            sequence=sequence,
            device=self.dq.device,
        )
        arguments = list(self.arguments)
        arguments[
            self._forward_mx_probability_scales_argument_index
        ] = pointer
        self.arguments = tuple(arguments)
        self.forward_mx_probability_scales = tensor
        self._forward_mx_probability_scales_bound = True

    def bind_d128_mxfp4_v_operands(
        self,
        v_payload: torch.Tensor,
        v_scale_pages: torch.Tensor,
    ) -> None:
        """Rebind one authenticated packed-V publication without copies."""
        if not self.kernel.use_d128_mxfp4_v_dp:
            raise RuntimeError(
                "this backward was not compiled to consume D128 MXFP4 V"
            )
        validation_tensors = self._d128_mxfp4_v_dp_validation_tensors
        scale_argument_index = self._d128_mxfp4_v_scale_argument_index
        if validation_tensors is None or scale_argument_index is None:
            raise RuntimeError("D128 MXFP4 V binding metadata is unavailable")
        cache = self._d128_mxfp4_v_operand_cache
        counters = self._d128_mxfp4_v_operand_cache_counters
        if cache is None or counters is None:
            raise RuntimeError("D128 MXFP4 V operand cache is unavailable")
        key = (id(v_payload), id(v_scale_pages))
        binding_identity = (
            self._d128_mxfp4_v_tensor_binding_identity(v_payload),
            self._d128_mxfp4_v_tensor_binding_identity(v_scale_pages),
        )
        cached = cache.get(key)
        if cached is not None and (
            cached["v_payload"] is v_payload
            and cached["v_scale_pages"] is v_scale_pages
            and cached["binding_identity"] == binding_identity
        ):
            counters["hits"] += 1
            cache.move_to_end(key)
            v_cute = cached["v_cute"]
            v_scale_iter = cached["v_scale_iter"]
        else:
            if cached is not None:
                cache.pop(key)
                counters["invalidations"] += 1
            counters["misses"] += 1
            counters["full_validations"] += 1
            # These are constructor-time compile placeholders, not the live
            # Q/K/dO tensors rebound by LowpAttentionRuntime immediately before
            # this method. Preserve their established full validation on every
            # new V publication while keeping repeat V hits off that dead path.
            q, k, dout = validation_tensors
            batch, sequence, q_heads, _ = q.shape
            kv_heads = int(k.shape[2])
            _require_d128_mxfp4_v_dp_tensor_abi(
                batch=int(batch),
                sequence=int(sequence),
                q_heads=int(q_heads),
                kv_heads=kv_heads,
                q=q,
                k=k,
                v_payload=v_payload,
                v_scale_pages=v_scale_pages,
                dout=dout,
            )
            v_cute = self._control.from_dlpack(
                v_payload,
                assumed_align=32,
            )
            counters["dlpack_wrapper_builds"] += 1
            v_cute.element_type = self._control.cutlass.Float4E2M1FN
            v_scale_cute = self._control.from_dlpack(
                v_scale_pages,
                assumed_align=16,
            )
            counters["dlpack_wrapper_builds"] += 1
            v_scale_cute.element_type = self._control.cutlass.Float8E8M0FNU
            v_scale_iter = v_scale_cute.iterator
            if len(cache) >= D128_MXFP4_V_OPERAND_CACHE_CAPACITY:
                cache.popitem(last=False)
                counters["evictions"] += 1
            cache[key] = {
                "v_payload": v_payload,
                "v_scale_pages": v_scale_pages,
                "binding_identity": binding_identity,
                "v_cute": v_cute,
                "v_scale_cute": v_scale_cute,
                "v_scale_iter": v_scale_iter,
            }
        arguments = list(self.arguments)
        arguments[3] = v_cute
        arguments[scale_argument_index] = v_scale_iter
        self.arguments = tuple(arguments)
        # The compiled argument wrappers are pointer-like. Keep both physical
        # publication owners alive through launch and make their identity
        # explicit for diagnostics.
        self.v_mxfp4_payload = v_payload
        self.v_mxfp4_scale_pages = v_scale_pages

    def reset(self) -> None:
        if self.direct_tma_dkdv:
            torch._foreach_zero_((self.dq, self.dk, self.dv))
        elif self.owner_dq_acc is not None:
            self.owner_dq_acc.zero_()
            assert self.owner_dq_ready is not None
            self.owner_dq_ready.zero_()
        elif self.dq_lanes is not None:
            self.dq_lanes.zero_()
            if self.dq_tile_arrivals is not None:
                self.dq_tile_arrivals.zero_()
                self.arrival_epoch = 0
        elif self.lowp:
            self.dq.zero_()
        else:
            self.workspace_torch.zero_()

    def run(self, *, reset: bool) -> None:
        if (
            self.kernel.use_forward_mx_probability_scales
            and not self._forward_mx_probability_scales_bound
        ):
            raise RuntimeError(
                "bind_forward_mx_probability_scales() before running backward"
            )
        if reset:
            self.reset()
        self.compiled(*self.arguments)
        if self.dq_tile_arrivals is not None:
            self.arrival_epoch += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=int, default=4096)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument(
        "--forward-extension",
        type=Path,
        default=Path(
            "/tmp/_C_tk_causal_fast_defer_final_"
            "s4096h32kv8d128.cpython-312-aarch64-linux-gnu.so"
        ),
    )
    parser.add_argument(
        "--forward-module",
        default="_C_tk_causal_fast_defer_final_s4096h32kv8d128",
    )
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--tile-ready-early-heads", type=int, default=4)
    parser.add_argument("--exp2-degree", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--exp2-period",
        type=int,
        choices=tuple(range(17)),
        default=0,
        help="D128 production policy uses native EX2 (period 0)",
    )
    parser.add_argument(
        "--reuse-quantized-p",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reuse the shared coordinate-preserving E4M3 P for dS",
    )
    parser.add_argument(
        "--lowp-do-stages",
        type=int,
        choices=(1, 2, 3),
        default=2,
        help="diagnostic FP8 dO shared-memory pipeline depth",
    )
    parser.add_argument(
        "--owner-fuse-kv",
        action="store_true",
        help="quantize reduced dK/dV directly into the owner projection operand",
    )
    parser.add_argument(
        "--reverse-query-tiles",
        action="store_true",
        help="accumulate long dV inner products from low to high magnitude",
    )
    parser.add_argument("--skip-bf16-control", action="store_true")
    parser.add_argument(
        "--diagnose-dv",
        action="store_true",
        help="compare dV and each pre-reduction GQA partial with PyTorch",
    )
    parser.add_argument(
        "--diagnose-projection-formats",
        action="store_true",
        help=(
            "measure row-scaled FP8 and block-Hadamard NVFP4 projection "
            "accuracy ceilings"
        ),
    )
    parser.add_argument(
        "--true-bf16-reference",
        action="store_true",
        help=(
            "compare gradients with BF16 Q/K/V using their own exact causal "
            "output and LSE instead of the low-precision forward statistics"
        ),
    )
    parser.add_argument(
        "--owner-only",
        action="store_true",
        help="skip the two-lane hierarchical controls while profiling owner dQ",
    )
    parser.add_argument(
        "--direct-workspace-stats",
        action="store_true",
        help=(
            "publish CuTe-native negative dPsum/log2-LSE directly from the "
            "dO projection into the backward workspace"
        ),
    )
    parser.add_argument(
        "--full-layer-boundaries",
        action="store_true",
        help=(
            "also time CuTe BF16 forward, the output projection, and the "
            "BF16 QKV dgrad projection needed for a complete layer sum"
        ),
    )
    parser.add_argument(
        "--training-layer-boundaries",
        action="store_true",
        help=(
            "also charge the QKV- and output-projection weight-gradient "
            "GEMMs; requires --full-layer-boundaries"
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise RuntimeError("profiling requires exactly one visible GPU")
    if args.q_heads <= 0 or args.q_heads % args.kv_heads:
        raise ValueError("q-heads must be divisible by kv-heads")
    if not 1 <= args.tile_ready_early_heads <= args.q_heads:
        raise ValueError("tile-ready-early-heads must be in [1, q-heads]")
    if args.sequence % 256 or args.hidden % 256:
        raise ValueError("sequence and hidden must be divisible by 256")
    if args.full_layer_boundaries and args.skip_bf16_control:
        raise ValueError(
            "full-layer boundaries require the BF16 backward control"
        )
    if args.training_layer_boundaries and not args.full_layer_boundaries:
        raise ValueError(
            "training-layer boundaries require --full-layer-boundaries"
        )
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)

    forward = _load_extension(args.forward_extension, args.forward_module)
    topology = dict(forward.read_hao_direct_topology())
    expected_topology = {
        "batch": 1,
        "heads": args.q_heads,
        "kv_heads": args.kv_heads,
        "seqlen": args.sequence,
        "dqk": 128,
        "dvo": 128,
    }
    for name, expected in expected_topology.items():
        if int(topology[name]) != expected:
            raise ValueError(
                f"forward topology {name}={topology[name]} != {expected}"
            )
    if not bool(topology["causal"]):
        raise ValueError("forward extension must be causal")
    forward_pv_format = str(topology["pv_format"])
    if forward_pv_format not in ("mxfp4_e8m0_block32", "e4m3_fp8"):
        raise ValueError(
            f"unsupported forward probability/V format {forward_pv_format}"
        )
    bf16_forward_func = None
    if args.full_layer_boundaries:
        flash_attention_root = (
            Path(__file__).resolve().parents[2] / "flash-attention"
        )
        sys.path.insert(0, str(flash_attention_root))
        from flash_attn.cute.interface import flash_attn_func

        bf16_forward_func = flash_attn_func

    rows = args.sequence
    depth = 128
    if args.exp2_period is None:
        args.exp2_period = 0 if depth == 64 and args.sequence < 2048 else 2
    x = (torch.randn(rows, args.hidden, device="cuda") * 0.1).bfloat16()
    dy = (torch.randn_like(x.float()) * 0.1).bfloat16()
    q_weight_raw = (
        torch.randn(args.q_heads * depth, args.hidden, device="cuda") * 0.02
    ).bfloat16()
    k_weight_raw = (
        torch.randn(args.kv_heads * depth, args.hidden, device="cuda") * 0.02
    ).bfloat16()
    v_weight = torch.randn_like(k_weight_raw.float()).mul_(0.02).bfloat16()
    q_weight, k_weight = b300_pair_interleave_gqa_d128_qk_projection_weights(
        q_weight_raw,
        k_weight_raw,
    )
    qkv_weight = b300_stack_gqa_d128_qkv_projection_weights(
        q_weight,
        k_weight,
        v_weight,
    )
    x_operand = tuple(b300_prepare_nvfp4_projection_operand(x))
    qkv_weight_operand = tuple(
        b300_prepare_nvfp4_projection_weight(qkv_weight)
    )
    qk_scales = torch.zeros(
        1,
        args.q_heads,
        7,
        device="cuda",
        dtype=torch.float32,
    )
    qk_scales[:, :, :2] = 16.0
    rope_cos, rope_sin = _make_rope(args.sequence)
    rope_packed = b300_pack_gqa_d128_rope(rope_cos, rope_sin)

    def project_qkv(*, store_bf16: bool) -> Any:
        return b300_project_qkv_gqa_d128_unified_lowp_nvfp4(
            x_operand,
            qkv_weight_operand,
            qk_scales,
            batch=1,
            seqlen=args.sequence,
            q_heads=args.q_heads,
            kv_heads=args.kv_heads,
            store_bf16=store_bf16,
            publish_fp8_backward=True,
            rope_packed=rope_packed,
        )

    def project_qkv_bf16_no_rope() -> tuple[torch.Tensor, ...]:
        # This deliberately omits RoPE, making it a favorable lower bound for
        # the BF16 projection rather than overstating the low-precision gain.
        return (
            torch.mm(x, q_weight_raw.T),
            torch.mm(x, k_weight_raw.T),
            torch.mm(x, v_weight.T),
        )

    qkv = project_qkv(store_bf16=True)
    assert qkv.q is not None and qkv.k is not None and qkv.v is not None
    assert qkv.q_backward_fp8 is not None
    assert qkv.k_backward_fp8 is not None
    assert qkv.v_backward_fp8 is not None
    out = torch.empty(
        (1, args.sequence, args.q_heads, depth),
        device="cuda",
        dtype=torch.bfloat16,
    )
    lse_bh1s = torch.empty(
        (1, args.q_heads, 1, args.sequence),
        device="cuda",
        dtype=torch.float32,
    )
    forward_operands = qkv.forward_operands()
    forward_v_fp8_bhds = (
        qkv.v_forward_fp8
        if forward_pv_format == "e4m3_fp8"
        else None
    )
    if forward_pv_format == "e4m3_fp8" and forward_v_fp8_bhds is None:
        raise RuntimeError(
            "the D128 FP8-PV path requires projection-native feature-major V; "
            "refusing an unfused permute/contiguous fallback"
        )

    def attention_forward() -> None:
        if forward_v_fp8_bhds is not None:
            forward.forward_hao_direct_fp8pv(
                *forward_operands[:6],
                forward_v_fp8_bhds,
                out,
                lse_bh1s,
                0,
                True,
                True,
            )
        else:
            forward.forward_hao_direct_fp4pv(
                *forward_operands,
                out,
                lse_bh1s,
                0,
                True,
                True,
            )

    def attention_forward_bf16() -> object:
        if bf16_forward_func is None:
            raise RuntimeError("BF16 forward boundary was not requested")
        return bf16_forward_func(qkv.q, qkv.k, qkv.v, causal=True)

    os.environ["TK_FA4_FP4PV_FWD_CONFIG"] = str(topology["route"])
    attention_forward()
    torch.cuda.synchronize()
    lse_bsh = lse_bh1s[:, :, 0].permute(0, 2, 1).contiguous()

    out_weight = (
        torch.randn(args.hidden, args.q_heads * depth, device="cuda") * 0.02
    ).bfloat16()
    dy_operand = tuple(b300_prepare_nvfp4_projection_operand(dy))
    out_backward_weight_operand = tuple(
        b300_prepare_nvfp4_projection_weight(out_weight.T.contiguous())
    )
    out_matrix = out.reshape(rows, args.q_heads * depth)
    out_forward_weight_operand = None
    out_forward_operand = None
    if args.full_layer_boundaries:
        out_forward_weight_operand = tuple(
            b300_prepare_nvfp4_projection_weight(out_weight)
        )
        out_forward_operand = tuple(
            b300_prepare_nvfp4_projection_operand(out_matrix)
        )

    def project_output_bf16() -> torch.Tensor:
        return torch.mm(out_matrix, out_weight.T)

    def project_output_lowp_prepacked() -> torch.Tensor:
        if out_forward_operand is None or out_forward_weight_operand is None:
            raise RuntimeError("output projection boundary was not requested")
        return b300_project_nvfp4(
            out_forward_operand, out_forward_weight_operand
        )

    def project_output_lowp_materialized() -> torch.Tensor:
        if out_forward_weight_operand is None:
            raise RuntimeError("output projection boundary was not requested")
        operand = tuple(b300_prepare_nvfp4_projection_operand(out_matrix))
        return b300_project_nvfp4(operand, out_forward_weight_operand)

    # Direct compact dQ needs only the two FP32 statistics pages followed by
    # BF16 dK/dV partials for every split query head.
    lowp_workspace_bytes = (
        2 * args.q_heads * args.sequence * 4
        + 2 * args.q_heads * args.sequence * depth * 2
    )
    lowp_workspace = torch.zeros(
        lowp_workspace_bytes,
        device="cuda",
        dtype=torch.uint8,
    )
    hierarchical_workspace_bytes = (
        2 * args.q_heads * args.sequence * 4
        + 2 * args.q_heads * args.sequence * depth * 2
        + 2 * args.q_heads * args.sequence * depth * 2
    )
    hierarchical_workspace = torch.zeros(
        hierarchical_workspace_bytes,
        device="cuda",
        dtype=torch.uint8,
    )
    tile_ready_workspace = torch.zeros(
        hierarchical_workspace_bytes
        + args.q_heads * (args.sequence // 128) * torch.int32.itemsize,
        device="cuda",
        dtype=torch.uint8,
    )
    fused_dq_target = torch.empty_like(qkv.q, dtype=torch.bfloat16)

    def project_dout(
        *,
        store_bf16: bool,
        stats_workspace: torch.Tensor | None = None,
        dq_clear: torch.Tensor | None = None,
    ) -> Any:
        return b300_project_dout_unified_lowp_nvfp4(
            dy_operand,
            out_backward_weight_operand,
            out,
            lse_bsh,
            batch=1,
            seqlen=args.sequence,
            heads=args.q_heads,
            store_bf16=store_bf16,
            publish_fp8_backward=True,
            publish_stats=True,
            stats_workspace=stats_workspace,
            dq_clear=dq_clear,
        )

    def project_dout_bf16() -> torch.Tensor:
        return torch.mm(dy, out_weight)

    direct_stats_workspace = (
        lowp_workspace if args.direct_workspace_stats else None
    )
    dout_bundle = project_dout(
        store_bf16=True,
        stats_workspace=direct_stats_workspace,
        dq_clear=(
            fused_dq_target if args.direct_workspace_stats else None
        ),
    )
    assert dout_bundle.dout is not None
    assert dout_bundle.dout_backward_fp8 is not None
    separate_stats_bundle = None
    stats_pages = None
    if args.direct_workspace_stats and not args.owner_only:
        # Retain a same-process control for the old topology: publish positive
        # statistics to standalone tensors, then copy/negate both pages into
        # the CuTe workspace.  This isolates the value of producer-native
        # publication from run-to-run GPU clock and load variation.
        separate_stats_bundle = project_dout(
            store_bf16=False,
            stats_workspace=None,
        )
        stats_elements = args.q_heads * args.sequence
        stats_pages = lowp_workspace.narrow(
            0,
            0,
            2 * stats_elements * torch.float32.itemsize,
        ).view(torch.float32)

        def copy_and_negate_statistics() -> None:
            assert separate_stats_bundle is not None
            assert stats_pages is not None
            torch.neg(
                separate_stats_bundle.dpsum.reshape(-1),
                out=stats_pages[:stats_elements],
            )
            torch.neg(
                separate_stats_bundle.lse_log2.reshape(-1),
                out=stats_pages[stats_elements:],
            )
    # dP uses dO' @ V'^T, so its row correction needs the product of the
    # two projection publication scales.  The dO epilogue already emits
    # dpsum at exactly that 4x4 scale.
    if args.direct_workspace_stats:
        precomputed_sum = dout_bundle.dpsum
        precomputed_lse = dout_bundle.lse_log2
    else:
        precomputed_sum = (-dout_bundle.dpsum).contiguous()
        precomputed_lse = (-dout_bundle.lse_log2).contiguous()

    control = _load_control()
    lowp_backward = CompiledGqaBackward(
        control,
        q=qkv.q_backward_fp8,
        k=qkv.k_backward_fp8,
        v=qkv.v_backward_fp8,
        o_or_sum=precomputed_sum,
        dout=dout_bundle.dout_backward_fp8,
        lse_or_scaled_lse=precomputed_lse,
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
        lowp=True,
        precomputed_stats=True,
        workspace_stats=args.direct_workspace_stats,
        scale_softmax=(depth**-0.5) / (FP8_INPUT_SCALE**2),
        exp2_degree=args.exp2_degree,
        exp2_period=args.exp2_period,
        reuse_quantized_p=args.reuse_quantized_p,
        lowp_do_stages=args.lowp_do_stages,
        reverse_query_tiles=args.reverse_query_tiles,
        workspace_torch=direct_stats_workspace,
        dq_torch=(fused_dq_target if args.direct_workspace_stats else None),
    )
    lowp_backward.run(reset=not args.direct_workspace_stats)
    torch.cuda.synchronize()

    hierarchical_backward = None
    tile_ready_backward = None
    if args.direct_workspace_stats and not args.owner_only:
        hierarchical_backward = CompiledGqaBackward(
            control,
            q=qkv.q_backward_fp8,
            k=qkv.k_backward_fp8,
            v=qkv.v_backward_fp8,
            o_or_sum=precomputed_sum,
            dout=dout_bundle.dout_backward_fp8,
            lse_or_scaled_lse=precomputed_lse,
            q_heads=args.q_heads,
            kv_heads=args.kv_heads,
            lowp=True,
            precomputed_stats=True,
            workspace_stats=True,
            scale_softmax=(depth**-0.5) / (FP8_INPUT_SCALE**2),
            exp2_degree=args.exp2_degree,
            exp2_period=args.exp2_period,
            reuse_quantized_p=args.reuse_quantized_p,
            lowp_do_stages=args.lowp_do_stages,
            reverse_query_tiles=args.reverse_query_tiles,
            workspace_torch=hierarchical_workspace,
            hierarchical_dq_lanes=2,
        )
        assert hierarchical_backward.dq_lanes is not None
        project_dout(
            store_bf16=False,
            stats_workspace=hierarchical_workspace,
            dq_clear=hierarchical_backward.dq_lanes,
        )
        hierarchical_backward.run(reset=False)
        torch.cuda.synchronize()

        tile_ready_backward = CompiledGqaBackward(
            control,
            q=qkv.q_backward_fp8,
            k=qkv.k_backward_fp8,
            v=qkv.v_backward_fp8,
            o_or_sum=precomputed_sum,
            dout=dout_bundle.dout_backward_fp8,
            lse_or_scaled_lse=precomputed_lse,
            q_heads=args.q_heads,
            kv_heads=args.kv_heads,
            lowp=True,
            precomputed_stats=True,
            workspace_stats=True,
            scale_softmax=(depth**-0.5) / (FP8_INPUT_SCALE**2),
            exp2_degree=args.exp2_degree,
            exp2_period=args.exp2_period,
            reuse_quantized_p=args.reuse_quantized_p,
            lowp_do_stages=args.lowp_do_stages,
            reverse_query_tiles=args.reverse_query_tiles,
            workspace_torch=tile_ready_workspace,
            hierarchical_dq_lanes=2,
            signal_dq_tiles=True,
        )
        assert tile_ready_backward.dq_lanes is not None
        assert tile_ready_backward.dq_tile_arrivals is not None
        project_dout(
            store_bf16=False,
            stats_workspace=tile_ready_workspace,
            dq_clear=tile_ready_backward.dq_lanes,
        )
        tile_ready_backward.run(reset=False)
        torch.cuda.synchronize()
        expected_arrivals = torch.ones(
            1,
            args.q_heads,
            args.sequence // 128,
            device="cuda",
            dtype=torch.int32,
        )
        if not torch.equal(
            tile_ready_backward.dq_tile_arrivals,
            expected_arrivals,
        ):
            raise RuntimeError("hierarchical dQ release counters are invalid")

    def project_dout_with_fused_dq_clear() -> None:
        project_dout(
            store_bf16=False,
            stats_workspace=direct_stats_workspace,
            dq_clear=lowp_backward.dq,
        )

    def fused_dout_projection_and_backward() -> None:
        project_dout_with_fused_dq_clear()
        lowp_backward.run(reset=False)

    def separate_clear_dout_projection_and_backward() -> None:
        project_dout(
            store_bf16=False,
            stats_workspace=direct_stats_workspace,
        )
        lowp_backward.run(reset=True)

    bf16_backward = None
    bf16_attention_out = None
    bf16_attention_lse = None
    if not args.skip_bf16_control:
        bf16_out = out
        bf16_lse = lse_bh1s
        if args.true_bf16_reference:
            bf16_out, bf16_lse = _bf16_gqa_attention_reference(
                qkv.q,
                qkv.k,
                qkv.v,
            )
            bf16_attention_out = bf16_out
            bf16_attention_lse = bf16_lse
        bf16_backward = CompiledGqaBackward(
            control,
            q=qkv.q,
            k=qkv.k,
            v=qkv.v,
            o_or_sum=bf16_out,
            dout=dout_bundle.dout,
            lse_or_scaled_lse=bf16_lse,
            q_heads=args.q_heads,
            kv_heads=args.kv_heads,
            lowp=False,
            precomputed_stats=False,
            scale_softmax=depth**-0.5,
        )
        bf16_backward.run(reset=True)
        torch.cuda.synchronize()

    qkv_backward_weight_operand = tuple(
        b300_prepare_nvfp4_projection_weight(
            qkv_weight.T.contiguous()
        )
    )
    lowp_dq_inverse = _inverse_rope_pair_native(
        lowp_backward.dq,
        rope_cos,
        rope_sin,
    )
    lowp_dk_inverse = _inverse_rope_pair_native(
        lowp_backward.dk,
        rope_cos,
        rope_sin,
    )
    lowp_stacked_gradient = torch.cat(
        (
            lowp_dq_inverse.reshape(rows, -1),
            lowp_dk_inverse.reshape(rows, -1),
            lowp_backward.dv.reshape(rows, -1),
        ),
        dim=1,
    ).contiguous()
    gradient_global_scale = (
        b300_prepare_nvfp4_projection_operand(lowp_stacked_gradient)[2]
    )
    fp8_projection_state: tuple[torch.Tensor, ...] | None = None
    hadamard_projection_state: tuple[
        tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]
    ] | None = None
    if args.diagnose_projection_formats:
        fp8_gradient_decode = (
            lowp_stacked_gradient.float()
            .abs()
            .amax(dim=1, keepdim=True)
            .clamp_min_(1.0e-12)
            .div_(448.0)
            .contiguous()
        )
        fp8_weight_decode = (
            qkv_weight.float()
            .abs()
            .amax(dim=0, keepdim=True)
            .clamp_min_(1.0e-12)
            .div_(448.0)
            .contiguous()
        )
        fp8_gradient = (
            lowp_stacked_gradient.float()
            .div(fp8_gradient_decode)
            .to(torch.float8_e4m3fn)
            .contiguous()
        )
        # scaled_mm requires the right operand in K-major/column-major form.
        fp8_weight = (
            qkv_weight.float()
            .div(fp8_weight_decode)
            .to(torch.float8_e4m3fn)
            .T.contiguous().T
        )
        fp8_projection_state = (
            fp8_gradient,
            fp8_weight,
            fp8_gradient_decode,
            fp8_weight_decode,
        )

        hadamard_gradient = _hadamard16_blocks(
            lowp_stacked_gradient
        ).bfloat16()
        hadamard_weight_t = _hadamard16_blocks(
            qkv_weight.T.contiguous()
        ).bfloat16()
        hadamard_projection_state = (
            tuple(b300_prepare_nvfp4_projection_operand(hadamard_gradient)),
            tuple(b300_prepare_nvfp4_projection_weight(hadamard_weight_t)),
        )
    qkv_gradient_reduction = (args.q_heads + 2 * args.kv_heads) * depth
    tile_ready_gradient_payload = torch.empty(
        rows,
        qkv_gradient_reduction // 2,
        device="cuda",
        dtype=torch.float4_e2m1fn_x2,
    )
    tile_ready_gradient_scales = torch.empty(
        rows // 128,
        qkv_gradient_reduction // 64,
        512,
        device="cuda",
        dtype=torch.float8_e4m3fn,
    )
    tile_ready_gradient_storage = (
        tile_ready_gradient_payload,
        tile_ready_gradient_scales,
    )
    tile_ready_gradient_operand = (
        tile_ready_gradient_payload,
        tile_ready_gradient_scales,
        gradient_global_scale,
    )
    owner_gradient_payload = torch.empty_like(tile_ready_gradient_payload)
    owner_gradient_scales = torch.empty_like(tile_ready_gradient_scales)
    # Physical NVFP4 scale pages contain padding positions that no logical
    # row owns. Initialize the persistent operand once; live Q/K/V positions
    # are overwritten on every iteration and need no per-step clear.
    owner_gradient_payload.view(torch.uint8).zero_()
    owner_gradient_scales.view(torch.uint8).zero_()
    owner_gradient_storage = (
        owner_gradient_payload,
        owner_gradient_scales,
    )
    owner_gradient_operand = (
        owner_gradient_payload,
        owner_gradient_scales,
        gradient_global_scale,
    )
    owner_backward = CompiledGqaBackward(
        control,
        q=qkv.q_backward_fp8,
        k=qkv.k_backward_fp8,
        v=qkv.v_backward_fp8,
        o_or_sum=precomputed_sum,
        dout=dout_bundle.dout_backward_fp8,
        lse_or_scaled_lse=precomputed_lse,
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
        lowp=True,
        precomputed_stats=True,
        workspace_stats=True,
        scale_softmax=(depth**-0.5) / (FP8_INPUT_SCALE**2),
        exp2_degree=args.exp2_degree,
        exp2_period=args.exp2_period,
        reuse_quantized_p=args.reuse_quantized_p,
        lowp_do_stages=args.lowp_do_stages,
        reverse_query_tiles=args.reverse_query_tiles,
        owner_output_operand=owner_gradient_storage,
        owner_gradient_global_scale=gradient_global_scale,
        owner_rope=rope_packed,
        owner_quantize_kv=args.owner_fuse_kv,
    )
    assert owner_backward.owner_dq_clear is not None
    assert owner_backward.owner_dq_ready is not None
    project_dout(
        store_bf16=False,
        stats_workspace=owner_backward.workspace_torch,
        dq_clear=owner_backward.owner_dq_clear,
    )
    owner_backward.owner_dq_ready.zero_()
    owner_backward.run(reset=False)
    torch.cuda.synchronize()
    expected_owner_ready = torch.arange(
        rows // 128,
        device="cuda",
        dtype=torch.int32,
    ).view(1, 1, -1).expand(1, args.q_heads, -1)
    if not torch.equal(owner_backward.owner_dq_ready, expected_owner_ready):
        raise RuntimeError("owner dQ causal readiness counters are invalid")
    tile_ready_pack_stream = torch.cuda.Stream(priority=0)
    tile_ready_attention_done = torch.cuda.Event()

    def pack_hierarchical_qkv_gradient_post_attention() -> None:
        if (
            tile_ready_backward is None
            or tile_ready_backward.dq_lanes is None
            or tile_ready_backward.dq_tile_arrivals is None
        ):
            raise RuntimeError("hierarchical tile-ready pack is not configured")
        b300_pack_gqa_d128_hierarchical_qkv_gradient_nvfp4_tiles(
            tile_ready_backward.dq_lanes,
            tile_ready_backward.dk,
            tile_ready_backward.dv,
            gradient_global_scale,
            rope_packed,
            tile_ready_gradient_storage,
            tile_ready_backward.dq_tile_arrivals,
            row_tile_begin=0,
            row_tile_end=rows // 128,
            col_tile_begin=0,
            col_tile_end=args.q_heads + 2 * args.kv_heads,
        )

    def project_tile_ready_qkv_gradient() -> torch.Tensor:
        return b300_project_nvfp4(
            tile_ready_gradient_operand,
            qkv_backward_weight_operand,
        )

    def pack_owner_kv_gradient_post_attention() -> None:
        if args.owner_fuse_kv:
            return
        if (
            owner_backward.owner_dq_clear is None
            or owner_backward.owner_dq_ready is None
        ):
            raise RuntimeError("owner dQ publication is not configured")
        # Q payload/scales were published by the diagonal owner CTA directly
        # from its completed reduction fragment.  Reuse the established pack
        # only for K/V; its dQ argument is deliberately outside this range.
        b300_pack_gqa_d128_hierarchical_qkv_gradient_nvfp4_tiles(
            owner_backward.owner_dq_clear,
            owner_backward.dk,
            owner_backward.dv,
            gradient_global_scale,
            rope_packed,
            owner_gradient_storage,
            owner_backward.owner_dq_ready,
            row_tile_begin=0,
            row_tile_end=rows // 128,
            col_tile_begin=args.q_heads,
            col_tile_end=args.q_heads + 2 * args.kv_heads,
        )

    def project_owner_qkv_gradient() -> torch.Tensor:
        return b300_project_nvfp4(
            owner_gradient_operand,
            qkv_backward_weight_operand,
        )

    def project_rowwise_fp8_qkv_gradient() -> torch.Tensor:
        if fp8_projection_state is None:
            raise RuntimeError("FP8 projection diagnostic is not configured")
        gradient, weight, gradient_decode, weight_decode = (
            fp8_projection_state
        )
        return torch._scaled_mm(
            gradient,
            weight,
            scale_a=gradient_decode,
            scale_b=weight_decode,
            out_dtype=torch.bfloat16,
        )

    def project_hadamard_nvfp4_qkv_gradient() -> torch.Tensor:
        if hadamard_projection_state is None:
            raise RuntimeError(
                "Hadamard NVFP4 projection diagnostic is not configured"
            )
        gradient_operand, weight_operand = hadamard_projection_state
        return b300_project_nvfp4(gradient_operand, weight_operand)

    def hierarchical_tile_ready_projection_chain() -> torch.Tensor:
        if (
            tile_ready_backward is None
            or tile_ready_backward.dq_lanes is None
            or tile_ready_backward.dq_tile_arrivals is None
        ):
            raise RuntimeError("hierarchical tile-ready chain is not configured")
        upcoming_epoch = tile_ready_backward.arrival_epoch + 1
        q_tiles = rows // 128
        # The attention grid linearizes K tiles inside each query head.  Wait
        # for one head prefix to retire, then publish that full-row Q slice in
        # one launch while later heads are still running.  This avoids both
        # per-row launch overhead and a full-grid competing producer.
        early_q_heads = args.tile_ready_early_heads
        with torch.cuda.stream(tile_ready_pack_stream):
            b300_pack_gqa_d128_hierarchical_qkv_gradient_nvfp4_tiles(
                tile_ready_backward.dq_lanes,
                tile_ready_backward.dk,
                tile_ready_backward.dv,
                gradient_global_scale,
                rope_packed,
                tile_ready_gradient_storage,
                tile_ready_backward.dq_tile_arrivals,
                row_tile_begin=0,
                row_tile_end=q_tiles,
                col_tile_begin=0,
                col_tile_end=early_q_heads,
                arrival_epoch=upcoming_epoch,
            )

        project_dout(
            store_bf16=False,
            stats_workspace=tile_ready_workspace,
            dq_clear=tile_ready_backward.dq_lanes,
        )
        tile_ready_backward.run(reset=False)
        tile_ready_attention_done.record()
        tile_ready_pack_stream.wait_event(tile_ready_attention_done)
        with torch.cuda.stream(tile_ready_pack_stream):
            b300_pack_gqa_d128_hierarchical_qkv_gradient_nvfp4_tiles(
                tile_ready_backward.dq_lanes,
                tile_ready_backward.dk,
                tile_ready_backward.dv,
                gradient_global_scale,
                rope_packed,
                tile_ready_gradient_storage,
                tile_ready_backward.dq_tile_arrivals,
                row_tile_begin=0,
                row_tile_end=q_tiles,
                col_tile_begin=early_q_heads,
                col_tile_end=args.q_heads + 2 * args.kv_heads,
            )
        torch.cuda.current_stream().wait_stream(tile_ready_pack_stream)
        return project_tile_ready_qkv_gradient()

    def project_materialized_qkv_gradient() -> torch.Tensor:
        return b300_project_gqa_d128_hierarchical_qkv_gradient_nvfp4(
            lowp_backward.dq,
            lowp_backward.dk,
            lowp_backward.dv,
            qkv_backward_weight_operand,
            gradient_global_scale,
            rope_packed,
        )

    def project_hierarchical_qkv_gradient() -> torch.Tensor:
        if (
            hierarchical_backward is None
            or hierarchical_backward.dq_lanes is None
        ):
            raise RuntimeError("hierarchical dQ projection is not configured")
        return b300_project_gqa_d128_hierarchical_qkv_gradient_nvfp4(
            hierarchical_backward.dq_lanes,
            hierarchical_backward.dk,
            hierarchical_backward.dv,
            qkv_backward_weight_operand,
            gradient_global_scale,
            rope_packed,
        )

    materialized_bundle = (
        b300_project_gqa_d128_hierarchical_qkv_gradient_nvfp4(
            lowp_backward.dq,
            lowp_backward.dk,
            lowp_backward.dv,
            qkv_backward_weight_operand,
            gradient_global_scale,
            rope_packed,
            return_operand=True,
        )
    )
    materialized_projection = materialized_bundle[0]
    materialized_gradient_payload = materialized_bundle[1]
    materialized_gradient_scales = materialized_bundle[2]
    hierarchical_projection = (
        project_hierarchical_qkv_gradient()
        if hierarchical_backward is not None
        else None
    )
    tile_ready_projection = None
    if tile_ready_backward is not None:
        pack_hierarchical_qkv_gradient_post_attention()
        tile_ready_projection = project_tile_ready_qkv_gradient()
    pack_owner_kv_gradient_post_attention()
    owner_projection = project_owner_qkv_gradient()
    fp8_diagnostic_projection = (
        project_rowwise_fp8_qkv_gradient()
        if fp8_projection_state is not None
        else None
    )
    hadamard_diagnostic_projection = (
        project_hadamard_nvfp4_qkv_gradient()
        if hadamard_projection_state is not None
        else None
    )
    torch.cuda.synchronize()

    def materialize_bf16_qkv_gradient() -> torch.Tensor:
        if bf16_backward is None:
            raise RuntimeError("BF16 backward control was not requested")
        bf16_dq_inverse = _inverse_rope_pair_native(
            bf16_backward.dq,
            rope_cos,
            rope_sin,
        )
        bf16_dk_inverse = _inverse_rope_pair_native(
            bf16_backward.dk,
            rope_cos,
            rope_sin,
        )
        return torch.cat(
            (
                bf16_dq_inverse.reshape(rows, -1),
                bf16_dk_inverse.reshape(rows, -1),
                bf16_backward.dv.reshape(rows, -1),
            ),
            dim=1,
        )

    bf16_stacked_gradient = (
        materialize_bf16_qkv_gradient()
        if bf16_backward is not None
        else None
    )

    def project_bf16_qkv_gradient_gemm_only() -> torch.Tensor:
        if bf16_stacked_gradient is None:
            raise RuntimeError("BF16 backward control was not requested")
        return torch.mm(bf16_stacked_gradient, qkv_weight)

    def project_bf16_qkv_gradient_materialized() -> torch.Tensor:
        return torch.mm(materialize_bf16_qkv_gradient(), qkv_weight)

    bf16_projection = (
        project_bf16_qkv_gradient_gemm_only()
        if bf16_backward is not None
        else None
    )

    def project_qkv_weight_gradient_lowp() -> torch.Tensor:
        return torch.mm(lowp_stacked_gradient.T, x)

    def project_qkv_weight_gradient_bf16() -> torch.Tensor:
        if bf16_stacked_gradient is None:
            raise RuntimeError("BF16 backward control was not requested")
        return torch.mm(bf16_stacked_gradient.T, x)

    def project_output_weight_gradient_lowp() -> torch.Tensor:
        return torch.mm(dy.T, out_matrix)

    def project_output_weight_gradient_bf16() -> torch.Tensor:
        if bf16_attention_out is None:
            raise RuntimeError(
                "true BF16 forward reference is required for the BF16 "
                "output weight gradient"
            )
        return torch.mm(dy.T, bf16_attention_out.reshape(rows, -1))

    def hierarchical_dout_projection_and_backward() -> None:
        if (
            hierarchical_backward is None
            or hierarchical_backward.dq_lanes is None
        ):
            raise RuntimeError("hierarchical dQ projection is not configured")
        project_dout(
            store_bf16=False,
            stats_workspace=hierarchical_workspace,
            dq_clear=hierarchical_backward.dq_lanes,
        )
        hierarchical_backward.run(reset=False)

    def materialized_dout_backward_and_qkv_projection() -> torch.Tensor:
        if args.direct_workspace_stats:
            project_dout_with_fused_dq_clear()
            lowp_backward.run(reset=False)
        else:
            # The standalone-statistics route has no workspace destination
            # through which the projection epilogue can clear dQ.  Retain
            # its already-published statistics and clear dQ in the backward
            # launch instead.
            project_dout(store_bf16=False)
            lowp_backward.run(reset=True)
        return project_materialized_qkv_gradient()

    def hierarchical_dout_backward_and_qkv_projection() -> torch.Tensor:
        hierarchical_dout_projection_and_backward()
        return project_hierarchical_qkv_gradient()

    def owner_dout_backward_and_qkv_projection() -> torch.Tensor:
        assert owner_backward.owner_dq_clear is not None
        assert owner_backward.owner_dq_ready is not None
        project_dout(
            store_bf16=False,
            stats_workspace=owner_backward.workspace_torch,
            dq_clear=owner_backward.owner_dq_clear,
        )
        owner_backward.owner_dq_ready.zero_()
        owner_backward.run(reset=False)
        pack_owner_kv_gradient_post_attention()
        return project_owner_qkv_gradient()

    timing = {
        "qkv_projection_bf16_no_rope": _time_cuda(
            project_qkv_bf16_no_rope,
            warmups=args.warmups,
            samples=args.samples,
        ),
        "qkv_projection_lowp": _time_cuda(
            lambda: project_qkv(store_bf16=False),
            warmups=args.warmups,
            samples=args.samples,
        ),
        "attention_forward_lowp": _time_cuda(
            attention_forward, warmups=args.warmups, samples=args.samples
        ),
        "dout_projection_lowp": _time_cuda(
            lambda: project_dout(
                store_bf16=False,
                stats_workspace=direct_stats_workspace,
            ),
            warmups=args.warmups,
            samples=args.samples,
        ),
        "dout_projection_bf16": _time_cuda(
            project_dout_bf16,
            warmups=args.warmups,
            samples=args.samples,
        ),
        "attention_backward_lowp_kernel": _time_cuda(
            lambda: lowp_backward.run(reset=False),
            warmups=args.warmups,
            samples=args.samples,
        ),
        "attention_backward_lowp_with_dq_clear": _time_cuda(
            lambda: lowp_backward.run(reset=True),
            warmups=args.warmups,
            samples=args.samples,
        ),
        "qkv_dgrad_projection_materialized_dq": _time_cuda(
            project_materialized_qkv_gradient,
            warmups=args.warmups,
            samples=args.samples,
        ),
        "dout_backward_qkv_projection_materialized_dq": _time_cuda(
            materialized_dout_backward_and_qkv_projection,
            warmups=args.warmups,
            samples=args.samples,
        ),
        "attention_backward_owner_dq_with_clear": _time_cuda(
            lambda: owner_backward.run(reset=True),
            warmups=args.warmups,
            samples=args.samples,
        ),
        "qkv_dgrad_owner_kv_pack_post_attention": _time_cuda(
            pack_owner_kv_gradient_post_attention,
            warmups=args.warmups,
            samples=args.samples,
        ),
        "qkv_dgrad_projection_owner_prepacked": _time_cuda(
            project_owner_qkv_gradient,
            warmups=args.warmups,
            samples=args.samples,
        ),
        "dout_backward_qkv_projection_owner_dq": _time_cuda(
            owner_dout_backward_and_qkv_projection,
            warmups=args.warmups,
            samples=args.samples,
        ),
    }
    if args.full_layer_boundaries:
        timing["attention_forward_bf16_cute"] = _time_cuda(
            attention_forward_bf16,
            warmups=args.warmups,
            samples=args.samples,
        )
        timing["output_projection_bf16"] = _time_cuda(
            project_output_bf16,
            warmups=args.warmups,
            samples=args.samples,
        )
        timing["output_projection_lowp_prepacked"] = _time_cuda(
            project_output_lowp_prepacked,
            warmups=args.warmups,
            samples=args.samples,
        )
        timing["output_projection_lowp_materialized"] = _time_cuda(
            project_output_lowp_materialized,
            warmups=args.warmups,
            samples=args.samples,
        )
        timing["qkv_dgrad_projection_bf16_gemm_only"] = _time_cuda(
            project_bf16_qkv_gradient_gemm_only,
            warmups=args.warmups,
            samples=args.samples,
        )
        timing["qkv_dgrad_projection_bf16_materialized"] = _time_cuda(
            project_bf16_qkv_gradient_materialized,
            warmups=args.warmups,
            samples=args.samples,
        )
    if args.training_layer_boundaries:
        if not args.true_bf16_reference:
            raise ValueError(
                "training-layer boundaries require --true-bf16-reference"
            )
        timing["qkv_weight_gradient_lowp"] = _time_cuda(
            project_qkv_weight_gradient_lowp,
            warmups=args.warmups,
            samples=args.samples,
        )
        timing["qkv_weight_gradient_bf16"] = _time_cuda(
            project_qkv_weight_gradient_bf16,
            warmups=args.warmups,
            samples=args.samples,
        )
        timing["output_weight_gradient_lowp"] = _time_cuda(
            project_output_weight_gradient_lowp,
            warmups=args.warmups,
            samples=args.samples,
        )
        timing["output_weight_gradient_bf16"] = _time_cuda(
            project_output_weight_gradient_bf16,
            warmups=args.warmups,
            samples=args.samples,
        )
    if args.diagnose_projection_formats:
        timing["qkv_dgrad_projection_rowwise_fp8_prepacked"] = _time_cuda(
            project_rowwise_fp8_qkv_gradient,
            warmups=args.warmups,
            samples=args.samples,
        )
        timing[
            "qkv_dgrad_projection_hadamard_nvfp4_prepacked"
        ] = _time_cuda(
            project_hadamard_nvfp4_qkv_gradient,
            warmups=args.warmups,
            samples=args.samples,
        )
    if hierarchical_backward is not None:
        timing["attention_backward_hierarchical_with_lane_clear"] = _time_cuda(
            lambda: hierarchical_backward.run(reset=True),
            warmups=args.warmups,
            samples=args.samples,
        )
        timing["qkv_dgrad_projection_hierarchical_dq"] = _time_cuda(
            project_hierarchical_qkv_gradient,
            warmups=args.warmups,
            samples=args.samples,
        )
        timing["qkv_dgrad_tile_ready_pack_post_attention"] = _time_cuda(
            pack_hierarchical_qkv_gradient_post_attention,
            warmups=args.warmups,
            samples=args.samples,
        )
        timing["qkv_dgrad_projection_prepacked"] = _time_cuda(
            project_tile_ready_qkv_gradient,
            warmups=args.warmups,
            samples=args.samples,
        )
        timing[
            "dout_backward_qkv_projection_hierarchical_dq"
        ] = _time_cuda(
            hierarchical_dout_backward_and_qkv_projection,
            warmups=args.warmups,
            samples=args.samples,
        )
        timing[
            "dout_backward_qkv_projection_tile_ready_hierarchical_dq"
        ] = _time_cuda(
            hierarchical_tile_ready_projection_chain,
            warmups=args.warmups,
            samples=args.samples,
        )
    if args.direct_workspace_stats:
        timing["dout_projection_with_fused_dq_clear"] = _time_cuda(
            project_dout_with_fused_dq_clear,
            warmups=args.warmups,
            samples=args.samples,
        )
        timing["dout_projection_and_backward_fused_clear"] = _time_cuda(
            fused_dout_projection_and_backward,
            warmups=args.warmups,
            samples=args.samples,
        )
        timing["dout_projection_and_backward_separate_clear"] = _time_cuda(
            separate_clear_dout_projection_and_backward,
            warmups=args.warmups,
            samples=args.samples,
        )
    if separate_stats_bundle is not None:
        timing["dout_projection_lowp_separate_stats"] = _time_cuda(
            lambda: project_dout(store_bf16=False, stats_workspace=None),
            warmups=args.warmups,
            samples=args.samples,
        )
        timing["dout_statistics_copy_and_sign"] = _time_cuda(
            copy_and_negate_statistics,
            warmups=args.warmups,
            samples=args.samples,
        )
    if bf16_backward is not None:
        timing["attention_backward_bf16_kernel"] = _time_cuda(
            lambda: bf16_backward.run(reset=False),
            warmups=args.warmups,
            samples=args.samples,
        )
        timing["attention_backward_bf16_with_workspace_clear"] = _time_cuda(
            lambda: bf16_backward.run(reset=True),
            warmups=args.warmups,
            samples=args.samples,
        )

    speedup = {
        "qkv_projection_vs_bf16_no_rope_lower_bound": (
            timing["qkv_projection_bf16_no_rope"]["median_us"]
            / timing["qkv_projection_lowp"]["median_us"]
        ),
        "dout_projection_vs_bf16": (
            timing["dout_projection_bf16"]["median_us"]
            / timing["dout_projection_lowp"]["median_us"]
        ),
    }
    if bf16_backward is not None:
        speedup["attention_backward_with_clear_vs_bf16"] = (
            timing["attention_backward_bf16_with_workspace_clear"][
                "median_us"
            ]
            / timing["attention_backward_lowp_with_dq_clear"]["median_us"]
        )
    if args.direct_workspace_stats:
        speedup["fused_dq_clear_projection_backward"] = (
            timing["dout_projection_and_backward_separate_clear"][
                "median_us"
            ]
            / timing["dout_projection_and_backward_fused_clear"]["median_us"]
        )
    speedup["owner_dq_vs_materialized_projection_chain"] = (
        timing["dout_backward_qkv_projection_materialized_dq"]["median_us"]
        / timing["dout_backward_qkv_projection_owner_dq"]["median_us"]
    )
    if hierarchical_backward is not None:
        speedup["hierarchical_dq_projection_chain"] = (
            timing[
                "dout_backward_qkv_projection_materialized_dq"
            ]["median_us"]
            / timing[
                "dout_backward_qkv_projection_hierarchical_dq"
            ]["median_us"]
        )
        speedup["tile_ready_hierarchical_vs_materialized_chain"] = (
            timing[
                "dout_backward_qkv_projection_materialized_dq"
            ]["median_us"]
            / timing[
                "dout_backward_qkv_projection_tile_ready_hierarchical_dq"
            ]["median_us"]
        )
        speedup["tile_ready_vs_post_attention_hierarchical_chain"] = (
            timing[
                "dout_backward_qkv_projection_hierarchical_dq"
            ]["median_us"]
            / timing[
                "dout_backward_qkv_projection_tile_ready_hierarchical_dq"
            ]["median_us"]
        )

    lowp_projection_backward_us = sum(
        timing[name]["median_us"]
        for name in (
            "qkv_projection_lowp",
            "dout_projection_lowp",
            "attention_backward_lowp_with_dq_clear",
        )
    )
    timing_sum_us = {
        "lowp_projection_backward": lowp_projection_backward_us,
        "lowp_projection_attention_boundary": (
            lowp_projection_backward_us
            + timing["attention_forward_lowp"]["median_us"]
        ),
    }
    if args.direct_workspace_stats:
        timing_sum_us["lowp_projection_backward_with_fused_dq_clear"] = sum(
            timing[name]["median_us"]
            for name in (
                "qkv_projection_lowp",
                "dout_projection_with_fused_dq_clear",
                "attention_backward_lowp_kernel",
            )
        )
        timing_sum_us[
            "lowp_qkv_projection_backward_fused_clear_event"
        ] = (
            timing["qkv_projection_lowp"]["median_us"]
            + timing["dout_projection_and_backward_fused_clear"]["median_us"]
        )
    timing_sum_us["lowp_owner_dq_projection_backward_event"] = (
        timing["qkv_projection_lowp"]["median_us"]
        + timing["dout_backward_qkv_projection_owner_dq"]["median_us"]
    )
    if hierarchical_backward is not None:
        timing_sum_us[
            "lowp_materialized_dq_projection_backward_event"
        ] = (
            timing["qkv_projection_lowp"]["median_us"]
            + timing[
                "dout_backward_qkv_projection_materialized_dq"
            ]["median_us"]
        )
        timing_sum_us[
            "lowp_hierarchical_dq_projection_backward_event"
        ] = (
            timing["qkv_projection_lowp"]["median_us"]
            + timing[
                "dout_backward_qkv_projection_hierarchical_dq"
            ]["median_us"]
        )
        timing_sum_us[
            "lowp_tile_ready_hierarchical_dq_projection_backward_event"
        ] = (
            timing["qkv_projection_lowp"]["median_us"]
            + timing[
                "dout_backward_qkv_projection_tile_ready_hierarchical_dq"
            ]["median_us"]
        )
    if bf16_backward is not None:
        bf16_projection_backward_us = sum(
            timing[name]["median_us"]
            for name in (
                "qkv_projection_bf16_no_rope",
                "dout_projection_bf16",
                "attention_backward_bf16_with_workspace_clear",
            )
        )
        timing_sum_us["bf16_projection_backward_no_rope_lower_bound"] = (
            bf16_projection_backward_us
        )
        speedup["projection_backward_vs_bf16_no_rope_lower_bound"] = (
            bf16_projection_backward_us / lowp_projection_backward_us
        )
    if args.full_layer_boundaries:
        lowp_full_layer_us = sum(
            timing[name]["median_us"]
            for name in (
                "qkv_projection_lowp",
                "attention_forward_lowp",
                "output_projection_lowp_materialized",
                "dout_backward_qkv_projection_materialized_dq",
            )
        )
        lowp_prepacked_output_ceiling_us = (
            lowp_full_layer_us
            - timing["output_projection_lowp_materialized"]["median_us"]
            + timing["output_projection_lowp_prepacked"]["median_us"]
        )
        bf16_full_layer_lower_bound_us = sum(
            timing[name]["median_us"]
            for name in (
                "qkv_projection_bf16_no_rope",
                "attention_forward_bf16_cute",
                "output_projection_bf16",
                "dout_projection_bf16",
                "attention_backward_bf16_with_workspace_clear",
                "qkv_dgrad_projection_bf16_gemm_only",
            )
        )
        bf16_full_layer_materialized_us = (
            bf16_full_layer_lower_bound_us
            - timing["qkv_dgrad_projection_bf16_gemm_only"]["median_us"]
            + timing["qkv_dgrad_projection_bf16_materialized"]["median_us"]
        )
        timing_sum_us["lowp_full_layer_component_sum"] = lowp_full_layer_us
        timing_sum_us[
            "lowp_full_layer_prepacked_output_ceiling"
        ] = lowp_prepacked_output_ceiling_us
        timing_sum_us[
            "bf16_full_layer_no_rope_no_inverse_rope_lower_bound"
        ] = bf16_full_layer_lower_bound_us
        timing_sum_us[
            "bf16_full_layer_materialized_component_sum"
        ] = bf16_full_layer_materialized_us
        speedup["full_layer_vs_bf16_no_rope_lower_bound"] = (
            bf16_full_layer_lower_bound_us / lowp_full_layer_us
        )
        speedup["full_layer_prepacked_output_ceiling_vs_bf16"] = (
            bf16_full_layer_lower_bound_us
            / lowp_prepacked_output_ceiling_us
        )
        speedup["full_layer_vs_bf16_materialized"] = (
            bf16_full_layer_materialized_us / lowp_full_layer_us
        )
        if args.training_layer_boundaries:
            lowp_training_layer_us = lowp_full_layer_us + sum(
                timing[name]["median_us"]
                for name in (
                    "qkv_weight_gradient_lowp",
                    "output_weight_gradient_lowp",
                )
            )
            bf16_training_layer_lower_bound_us = (
                bf16_full_layer_lower_bound_us
                + sum(
                    timing[name]["median_us"]
                    for name in (
                        "qkv_weight_gradient_bf16",
                        "output_weight_gradient_bf16",
                    )
                )
            )
            bf16_training_layer_materialized_us = (
                bf16_full_layer_materialized_us
                + sum(
                    timing[name]["median_us"]
                    for name in (
                        "qkv_weight_gradient_bf16",
                        "output_weight_gradient_bf16",
                    )
                )
            )
            timing_sum_us["lowp_training_layer_component_sum"] = (
                lowp_training_layer_us
            )
            timing_sum_us[
                "bf16_training_layer_no_rope_no_inverse_rope_lower_bound"
            ] = bf16_training_layer_lower_bound_us
            timing_sum_us[
                "bf16_training_layer_materialized_component_sum"
            ] = bf16_training_layer_materialized_us
            speedup["training_layer_vs_bf16_no_rope_lower_bound"] = (
                bf16_training_layer_lower_bound_us / lowp_training_layer_us
            )
            speedup["training_layer_vs_bf16_materialized"] = (
                bf16_training_layer_materialized_us / lowp_training_layer_us
            )
    if separate_stats_bundle is not None:
        separate_handoff_us = (
            timing["dout_projection_lowp_separate_stats"]["median_us"]
            + timing["dout_statistics_copy_and_sign"]["median_us"]
        )
        timing_sum_us["dout_separate_statistics_handoff"] = separate_handoff_us
        speedup["direct_statistics_handoff"] = (
            separate_handoff_us / timing["dout_projection_lowp"]["median_us"]
        )

    dpsum_reference = (
        out.float()
        * (dout_bundle.dout_backward_fp8.float() / FP8_DOUT_SCALE)
    ).sum(dim=-1).permute(0, 2, 1).unsqueeze(2)
    quality: dict[str, Any] = {
        "projection_q_fp8_decode": _metrics(
            qkv.q,
            qkv.q_backward_fp8.float() / FP8_INPUT_SCALE,
        ),
        "projection_k_fp8_decode": _metrics(
            qkv.k,
            qkv.k_backward_fp8.float() / FP8_INPUT_SCALE,
        ),
        "projection_v_fp8_decode": _metrics(
            qkv.v,
            qkv.v_backward_fp8.float() / FP8_INPUT_SCALE,
        ),
        "projection_dout_fp8_decode": _metrics(
            dout_bundle.dout,
            dout_bundle.dout_backward_fp8.float() / FP8_DOUT_SCALE,
        ),
        "projection_dpsum": _metrics(
            dpsum_reference,
            (-1.0 if args.direct_workspace_stats else 1.0)
            * dout_bundle.dpsum
            / FP8_DPSUM_SCALE,
        ),
        "projection_lse_log2": _metrics(
            lse_bh1s * math.log2(math.e),
            (-1.0 if args.direct_workspace_stats else 1.0)
            * dout_bundle.lse_log2,
        ),
        "owner_dq_projection_vs_materialized": _metrics(
            materialized_projection,
            owner_projection,
        ),
        "owner_q_payload_byte_match": float(
            (
                owner_gradient_payload.view(torch.uint8)[
                    :, : args.q_heads * depth // 2
                ]
                == materialized_gradient_payload.view(torch.uint8)[
                    :, : args.q_heads * depth // 2
                ]
            )
            .float()
            .mean()
        ),
        "owner_q_scale_byte_match": float(
            (
                owner_gradient_scales.view(torch.uint8)[
                    :, : 2 * args.q_heads
                ]
                == materialized_gradient_scales.view(torch.uint8)[
                    :, : 2 * args.q_heads
                ]
            )
            .float()
            .mean()
        ),
        "owner_kv_payload_byte_match": float(
            (
                owner_gradient_payload.view(torch.uint8)[
                    :, args.q_heads * depth // 2 :
                ]
                == materialized_gradient_payload.view(torch.uint8)[
                    :, args.q_heads * depth // 2 :
                ]
            )
            .float()
            .mean()
        ),
        "owner_kv_scale_byte_match": float(
            (
                owner_gradient_scales.view(torch.uint8)[
                    :, 2 * args.q_heads :
                ]
                == materialized_gradient_scales.view(torch.uint8)[
                    :, 2 * args.q_heads :
                ]
            )
            .float()
            .mean()
        ),
        "owner_q_scale_finite_fraction": float(
            torch.isfinite(
                owner_gradient_scales[:, : 2 * args.q_heads].float()
            )
            .float()
            .mean()
        ),
        "owner_q_scale_max_abs": float(
            owner_gradient_scales[:, : 2 * args.q_heads]
            .float()
            .abs()
            .max()
        ),
        "owner_q_scale_saturated_count": int(
            (
                owner_gradient_scales[:, : 2 * args.q_heads]
                .float()
                .abs()
                == 448.0
            ).sum()
        ),
        "owner_projection_nonfinite_count": int(
            (~torch.isfinite(owner_projection)).sum()
        ),
    }
    if not args.owner_fuse_kv:
        quality.update(
            {
                "owner_dk_vs_materialized": _metrics(
                    lowp_backward.dk,
                    owner_backward.dk,
                ),
                "owner_dv_vs_materialized": _metrics(
                    lowp_backward.dv,
                    owner_backward.dv,
                ),
            }
        )
    if args.diagnose_dv:
        if bf16_backward is None:
            raise RuntimeError("--diagnose-dv requires the BF16 control")
        true_dv, true_dv_partials = _gqa_dv_reference(
            qkv.q,
            qkv.k,
            dout_bundle.dout,
        )
        represented_dv, represented_dv_partials = _gqa_dv_reference(
            qkv.q_backward_fp8.float() / FP8_INPUT_SCALE,
            qkv.k_backward_fp8.float() / FP8_INPUT_SCALE,
            dout_bundle.dout_backward_fp8.float() / FP8_DOUT_SCALE,
            lse_bh1s=lse_bh1s,
            probability_dtype=torch.float8_e4m3fn,
            probability_lift=FP8_PROBABILITY_DV_LIFT,
        )
        quality.update(
            {
                "bf16_backward_dv_vs_torch": _metrics(
                    true_dv,
                    bf16_backward.dv,
                ),
                "bf16_backward_dv_partials_vs_torch": _metrics(
                    true_dv_partials,
                    bf16_backward.dv_partials,
                ),
                "analytic_represented_dv_vs_true_bf16": _metrics(
                    true_dv,
                    represented_dv,
                ),
                "lowp_backward_dv_vs_analytic_represented": _metrics(
                    represented_dv,
                    lowp_backward.dv.float() / FP8_DOUT_SCALE,
                ),
                "lowp_backward_dv_partials_vs_analytic_represented": (
                    _metrics(
                        represented_dv_partials,
                        lowp_backward.dv_partials.float()
                        / (
                            FP8_PROBABILITY_DV_LIFT
                            * FP8_DOUT_SCALE
                        ),
                    )
                ),
            }
        )
    if bf16_backward is not None:
        quality.update(
            {
                "backward_dq": _metrics(
                    bf16_backward.dq,
                    lowp_backward.dq.float() / FP8_DOUT_SCALE,
                ),
                "backward_dk": _metrics(
                    bf16_backward.dk,
                    lowp_backward.dk.float() / FP8_DOUT_SCALE,
                ),
                "backward_dv": _metrics(
                    bf16_backward.dv,
                    lowp_backward.dv.float() / FP8_DOUT_SCALE,
                ),
                "qkv_dgrad_projection_materialized_vs_bf16": _metrics(
                    bf16_projection,
                    materialized_projection.float() / FP8_DOUT_SCALE,
                ),
                "qkv_dgrad_projection_owner_vs_bf16": _metrics(
                    bf16_projection,
                    owner_projection.float() / FP8_DOUT_SCALE,
                ),
            }
        )
        if (
            fp8_diagnostic_projection is not None
            and hadamard_diagnostic_projection is not None
        ):
            quality.update(
                {
                    "qkv_dgrad_projection_rowwise_fp8_vs_bf16": _metrics(
                        bf16_projection,
                        fp8_diagnostic_projection.float()
                        / FP8_DOUT_SCALE,
                    ),
                    "qkv_dgrad_projection_hadamard_nvfp4_vs_bf16": (
                        _metrics(
                            bf16_projection,
                            hadamard_diagnostic_projection.float()
                            / FP8_DOUT_SCALE,
                        )
                    ),
                }
            )
    if bf16_attention_out is not None and bf16_attention_lse is not None:
        quality.update(
            {
                "forward_output_vs_true_bf16": _metrics(
                    bf16_attention_out,
                    out,
                ),
                "forward_lse_vs_true_bf16": _metrics(
                    bf16_attention_lse,
                    lse_bh1s,
                ),
            }
        )
    if (
        hierarchical_backward is not None
        and hierarchical_backward.dq_lanes is not None
        and hierarchical_projection is not None
    ):
        hierarchical_dq = (
            hierarchical_backward.dq_lanes[0]
            + hierarchical_backward.dq_lanes[1]
        ).permute(0, 2, 1, 3).contiguous()
        quality.update(
            {
                "hierarchical_dq_vs_materialized": _metrics(
                    lowp_backward.dq,
                    hierarchical_dq,
                ),
                "hierarchical_dk_vs_materialized": _metrics(
                    lowp_backward.dk,
                    hierarchical_backward.dk,
                ),
                "hierarchical_dv_vs_materialized": _metrics(
                    lowp_backward.dv,
                    hierarchical_backward.dv,
                ),
                "qkv_dgrad_projection_hierarchical_vs_materialized": _metrics(
                    materialized_projection,
                    hierarchical_projection,
                ),
                "qkv_dgrad_projection_tile_ready_vs_hierarchical": _metrics(
                    hierarchical_projection,
                    tile_ready_projection,
                ),
            }
        )
        if bf16_projection is not None:
            quality[
                "qkv_dgrad_projection_hierarchical_vs_bf16"
            ] = _metrics(
                bf16_projection,
                hierarchical_projection.float() / FP8_DOUT_SCALE,
            )

    if args.full_layer_boundaries:
        lowp_projected_output = project_output_lowp_prepacked()
        quality["output_projection_nvfp4_vs_bf16_same_attention"] = (
            _metrics(project_output_bf16(), lowp_projected_output)
        )
        if bf16_attention_out is not None:
            quality[
                "forward_through_output_projection_vs_true_bf16"
            ] = _metrics(
                torch.mm(
                    bf16_attention_out.reshape(rows, -1),
                    out_weight.T,
                ),
                lowp_projected_output,
            )
    if args.training_layer_boundaries:
        quality["qkv_weight_gradient_vs_bf16"] = _metrics(
            project_qkv_weight_gradient_bf16(),
            project_qkv_weight_gradient_lowp().float() / FP8_DOUT_SCALE,
        )
        quality["output_weight_gradient_vs_true_bf16"] = _metrics(
            project_output_weight_gradient_bf16(),
            project_output_weight_gradient_lowp(),
        )

    result = {
        "shape": {
            "batch": 1,
            "sequence": args.sequence,
            "q_heads": args.q_heads,
            "kv_heads": args.kv_heads,
            "head_dim": depth,
            "hidden": args.hidden,
        },
        "scales": {
            "projection_qkv_fp8": FP8_INPUT_SCALE,
            "projection_dout_fp8": FP8_DOUT_SCALE,
            "projection_dpsum": FP8_DPSUM_SCALE,
            "backward_output_decode": FP8_DOUT_SCALE,
            "backward_probability_dv_lift": FP8_PROBABILITY_DV_LIFT,
            "qkv_gradient_global_decode": float(gradient_global_scale),
        },
        "backward_policy": {
            "exp2_degree": args.exp2_degree,
            "exp2_period": args.exp2_period,
            "dout_pipeline_stages": args.lowp_do_stages,
            "query_tile_accumulation_order": (
                "reverse" if args.reverse_query_tiles else "forward"
            ),
            "reuse_quantized_probability_for_ds": args.reuse_quantized_p,
            "precomputed_stats": True,
            "statistics_storage": (
                "backward_workspace"
                if args.direct_workspace_stats
                else "standalone_tensors"
            ),
            "direct_statistics": args.direct_workspace_stats,
            "statistics_sign": "negative",
            "direct_compact_dq": True,
            "probability_dv_fp8_lift": FP8_PROBABILITY_DV_LIFT,
            "probability_dv_lift_compensation": (
                "fused_gqa_reduction_output"
            ),
            "hierarchical_dq_reduction_lanes": (
                2 if hierarchical_backward is not None else 1
            ),
            "hierarchical_dq_consumed_by_projection": (
                hierarchical_backward is not None
            ),
            "tile_ready_dq_release_counters": (
                hierarchical_backward is not None
            ),
            "tile_ready_early_q_rows": args.sequence // 128,
            "tile_ready_early_q_heads": args.tile_ready_early_heads,
            "owner_dq_quantized_from_onchip_shared_tile": True,
            "owner_dq_causal_owner": (
                "diagonal_deferred_shared_compute_wg_head_major_early_publish"
            ),
            "owner_dq_global_partial_reads": 1,
            "owner_dq_completed_bf16_global_writes": 0,
            "owner_dq_completed_bf16_global_reads": 0,
            "owner_dq_prior_lane_load": "coalesced_bf16_pairs_via_shared",
            "owner_dq_inverse_rope": "packed_bf16_pair_inline_ptx",
            "owner_dq_publication": "shared_pack_tma_payload_and_scales",
            "owner_kv_publication": (
                "fused_gqa_reducer_direct_nvfp4"
                if args.owner_fuse_kv
                else "standalone_bf16_tile_pack"
            ),
            "owner_dq_kv_pack_only": True,
            "standalone_final_bf16_dq_materialized": False,
            "reference_policy": (
                "true_bf16_forward"
                if args.true_bf16_reference
                else "shared_lowp_forward_boundary"
            ),
        },
        "forward_topology": dict(forward.read_hao_direct_topology()),
        "forward_publication": {
            "fp8_v_layout": (
                "projection_native_feature_major"
                if forward_v_fp8_bhds is not None
                else None
            ),
            "unfused_fp8_v_layout_conversion": False,
        },
        "timing": timing,
        "timing_sum_us": timing_sum_us,
        "speedup": speedup,
        "quality": quality,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
