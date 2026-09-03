"""Readable QDQ references for projection-format ablations.

These routines are deliberately not production quantizers.  They reconstruct
the values represented by NVFP4 or MXFP4 so projection accuracy can be ranked
before changing a fused CUDA path.  Learned weights always use a square 2D
block: one scale and one set of E2M1 codes can therefore be reused by both the
forward and transposed input-gradient GEMMs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_THRESHOLDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
E4M3_MAX = 448.0
SIGNED_E2M1_LEVELS = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


@dataclass(frozen=True)
class FakeQuantResult:
    """Decoded tensor plus the scale grid used to represent it."""

    values: torch.Tensor
    block_scales: torch.Tensor
    diagnostics: dict[str, Any]


def _require_matrix_and_blocks(
    tensor: torch.Tensor,
    block_shape: tuple[int, int],
) -> None:
    if tensor.ndim != 2 or not tensor.is_floating_point():
        raise ValueError("fake quantization requires a floating-point matrix")
    block_rows, block_columns = block_shape
    if block_rows <= 0 or block_columns <= 0:
        raise ValueError("block dimensions must be positive")
    if tensor.shape[0] % block_rows or tensor.shape[1] % block_columns:
        raise ValueError(
            f"matrix shape {tuple(tensor.shape)} is not divisible by "
            f"block shape {block_shape}"
        )


def _to_blocks(
    tensor: torch.Tensor,
    block_shape: tuple[int, int],
) -> torch.Tensor:
    block_rows, block_columns = block_shape
    rows, columns = tensor.shape
    return (
        tensor.float()
        .reshape(
            rows // block_rows,
            block_rows,
            columns // block_columns,
            block_columns,
        )
        .permute(0, 2, 1, 3)
        .contiguous()
    )


def _from_blocks(
    blocks: torch.Tensor,
    shape: torch.Size,
) -> torch.Tensor:
    return (
        blocks.permute(0, 2, 1, 3)
        .contiguous()
        .reshape(shape)
    )


def _quantize_e2m1(values: torch.Tensor) -> torch.Tensor:
    thresholds = torch.tensor(
        E2M1_THRESHOLDS,
        device=values.device,
        dtype=torch.float32,
    )
    levels = torch.tensor(
        E2M1_LEVELS,
        device=values.device,
        dtype=torch.float32,
    )
    magnitude = values.float().abs()
    indices = torch.bucketize(magnitude, thresholds)
    return levels[indices].copysign(values.float())


def _quantize_e2m1_rne(values: torch.Tensor) -> torch.Tensor:
    """Round to E2M1 with the PTX nearest-even tie policy.

    ``torch.bucketize`` is sufficient away from exact midpoints, but always
    selects the lower endpoint at a tie.  E2M1 ties instead select the code
    with an even least-significant bit, so the explicit neighbor comparison
    matters for deterministic backward diagnostics.
    """
    levels = torch.tensor(
        E2M1_LEVELS,
        device=values.device,
        dtype=torch.float32,
    )
    magnitude = values.float().abs()
    upper_index = torch.searchsorted(levels, magnitude).clamp_(1, 7)
    lower_index = upper_index - 1
    lower = levels[lower_index]
    upper = levels[upper_index]
    lower_distance = magnitude - lower
    upper_distance = upper - magnitude
    tie = lower_distance == upper_distance
    choose_upper = (upper_distance < lower_distance) | (
        tie & ((upper_index & 1) == 0)
    )
    selected_index = torch.where(
        choose_upper,
        upper_index,
        lower_index,
    )
    selected_index = torch.where(
        magnitude >= levels[-1],
        torch.full_like(selected_index, 7),
        selected_index,
    )
    return levels[selected_index].copysign(values.float())


def _quantize_e2m1_stochastic(
    values: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Unbiased stochastic rounding between adjacent finite E2M1 levels."""
    levels = torch.tensor(
        E2M1_LEVELS,
        device=values.device,
        dtype=torch.float32,
    )
    magnitude = values.float().abs()
    upper_index = torch.searchsorted(levels, magnitude).clamp_(1, 7)
    lower_index = upper_index - 1
    lower = levels[lower_index]
    upper = levels[upper_index]
    probability_upper = ((magnitude - lower) / (upper - lower)).clamp_(
        0.0,
        1.0,
    )
    choose_upper = torch.rand(
        magnitude.shape,
        device=magnitude.device,
        dtype=torch.float32,
        generator=generator,
    ) < probability_upper
    selected_index = lower_index + choose_upper.to(torch.long)
    selected_index = torch.where(
        magnitude >= levels[-1],
        torch.full_like(selected_index, 7),
        selected_index,
    )
    return levels[selected_index].copysign(values.float())


def _decode_packed_e2m1(payload: torch.Tensor) -> torch.Tensor:
    packed = payload.contiguous().view(torch.uint8)
    levels = torch.tensor(
        SIGNED_E2M1_LEVELS,
        device=payload.device,
        dtype=torch.float32,
    )
    return torch.stack(
        (
            levels[(packed & 0x0F).long()],
            levels[(packed >> 4).long()],
        ),
        dim=-1,
    ).flatten(-2)


def transpose_prepared_nvfp4_weight_reference(
    operand: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Transpose one true-2D prepared NVFP4 weight byte-for-byte.

    This readable implementation specifies the packed-nibble and tcgen05
    scale-page transformation used by the production CUDA helper.  It is a
    validator, not a production path: the scale replication check and tensor
    indexing may synchronize or allocate temporary logical matrices.
    """
    if len(operand) != 3:
        raise ValueError("an NVFP4 operand must contain payload, scales, global")
    payload, scales, global_scale = operand
    if payload.ndim != 2 or scales.ndim != 3 or global_scale.numel() != 1:
        raise ValueError("invalid prepared NVFP4 matrix operand")
    if payload.element_size() != 1 or scales.element_size() != 1:
        raise ValueError("prepared NVFP4 payload and scales must be byte-sized")
    rows, packed_columns = payload.shape
    columns = packed_columns * 2
    if rows % 128 or columns % 128:
        raise ValueError(
            "dual NVFP4 weight dimensions must be multiples of 128x128"
        )
    expected_scale_shape = (rows // 128, columns // 64, 512)
    if tuple(scales.shape) != expected_scale_shape:
        raise ValueError("prepared NVFP4 scale pages have an invalid shape")

    payload_bytes = payload.contiguous().view(torch.uint8)
    low = payload_bytes & 0x0F
    high = payload_bytes >> 4
    codes = torch.stack((low, high), dim=-1).reshape(rows, columns)
    transpose_codes = codes.T.contiguous()
    transpose_payload_bytes = (
        transpose_codes[:, 0::2]
        | (transpose_codes[:, 1::2] << 4)
    ).contiguous()

    scales_bytes = scales.contiguous().view(torch.uint8)
    source_row = torch.arange(rows, device=scales.device)
    source_group = torch.arange(columns // 16, device=scales.device)
    source_page_row = (source_row // 128)[:, None]
    source_page_column = (source_group // 4)[None, :]
    source_offset = (
        (source_row % 32)[:, None] * 16
        + ((source_row % 128) // 32)[:, None] * 4
        + (source_group % 4)[None, :]
    )
    logical_scales = scales_bytes[
        source_page_row,
        source_page_column,
        source_offset,
    ]
    scale_tiles = logical_scales.reshape(
        rows // 16,
        16,
        columns // 16,
    )
    tile_values = scale_tiles[:, 0, :]
    if not torch.equal(
        scale_tiles,
        tile_values[:, None, :].expand_as(scale_tiles),
    ):
        raise ValueError(
            "prepared NVFP4 weight scales are not replicated per 16x16 tile"
        )

    transpose_logical_scales = tile_values.T.repeat_interleave(16, dim=0)
    transpose_scales_bytes = torch.empty(
        (columns // 128, rows // 64, 512),
        device=scales.device,
        dtype=torch.uint8,
    )
    transpose_row = torch.arange(columns, device=scales.device)
    transpose_group = torch.arange(rows // 16, device=scales.device)
    transpose_page_row = (transpose_row // 128)[:, None]
    transpose_page_column = (transpose_group // 4)[None, :]
    transpose_offset = (
        (transpose_row % 32)[:, None] * 16
        + ((transpose_row % 128) // 32)[:, None] * 4
        + (transpose_group % 4)[None, :]
    )
    transpose_scales_bytes[
        transpose_page_row,
        transpose_page_column,
        transpose_offset,
    ] = transpose_logical_scales
    return (
        transpose_payload_bytes.view(payload.dtype),
        transpose_scales_bytes.view(scales.dtype),
        global_scale,
    )


def decode_prepared_nvfp4_matrix(
    operand: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Decode the packed matrix ABI produced by the projection quantizer."""
    if len(operand) != 3:
        raise ValueError("an NVFP4 operand must contain payload, scales, global")
    payload, scales, global_scale = operand
    if payload.ndim != 2 or scales.ndim != 3 or global_scale.numel() != 1:
        raise ValueError("invalid prepared NVFP4 matrix operand")
    rows, packed_columns = payload.shape
    columns = packed_columns * 2
    if rows % 128 or columns % 64:
        raise ValueError("prepared NVFP4 dimensions must be multiples of 128x64")
    if tuple(scales.shape) != (rows // 128, columns // 64, 512):
        raise ValueError("prepared NVFP4 scale pages have an invalid shape")

    decoded = _decode_packed_e2m1(payload).reshape(
        rows,
        columns // 16,
        16,
    )
    row = torch.arange(rows, device=payload.device)
    column_block = torch.arange(columns // 16, device=payload.device)
    page_row = (row // 128)[:, None]
    page_column = (column_block // 4)[None, :]
    scale_offset = (
        (row % 32)[:, None] * 16
        + ((row % 128) // 32)[:, None] * 4
        + (column_block % 4)[None, :]
    )
    local_scale = scales.float()[page_row, page_column, scale_offset]
    return (
        decoded
        * local_scale[..., None]
        * global_scale.float().reshape(1, 1, 1)
    ).reshape(rows, columns)


def decode_native_nvfp4_qk(
    payload: torch.Tensor,
    scales: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    scale_tile_rows: int,
) -> torch.Tensor:
    """Decode a native FA4 Q/K operand into logical ``[B,H,S,D]`` order."""
    if payload.ndim != 4:
        raise ValueError("Q/K payload must have shape [B,H,S,D/2]")
    batch, heads, rows, packed_columns = payload.shape
    columns = packed_columns * 2
    if columns % 64 or rows % scale_tile_rows:
        raise ValueError("Q/K dimensions do not match the scale geometry")
    chunks = columns // 64
    expected_scales = (
        batch,
        rows // scale_tile_rows,
        heads * chunks,
        512,
    )
    if tuple(scales.shape) != expected_scales:
        raise ValueError(
            f"expected Q/K scales {expected_scales}, got {tuple(scales.shape)}"
        )
    if tuple(global_scale.shape) != (batch, heads):
        raise ValueError("Q/K global scale must have shape [B,H]")

    decoded = _decode_packed_e2m1(payload).reshape(
        batch,
        heads,
        rows,
        columns // 16,
        16,
    )
    row = torch.arange(rows, device=payload.device)
    column_block = torch.arange(columns // 16, device=payload.device)
    tile = (row // scale_tile_rows)[:, None]
    page_in_head = (column_block // 4)[None, :]
    offset = (
        (row % 32)[:, None] * 16
        + ((row % scale_tile_rows) // 32)[:, None] * 4
        + (column_block % 4)[None, :]
    )
    local_scale = torch.empty(
        batch,
        heads,
        rows,
        columns // 16,
        device=payload.device,
        dtype=torch.float32,
    )
    scale_values = scales.float()
    for batch_index in range(batch):
        for head_index in range(heads):
            scale_page = head_index * chunks + page_in_head
            local_scale[batch_index, head_index] = scale_values[
                batch_index,
                tile,
                scale_page,
                offset,
            ]
    return (
        decoded
        * local_scale[..., None]
        * global_scale.float()[:, :, None, None, None]
    ).reshape(batch, heads, rows, columns)


def decode_native_mxfp4_v(
    payload: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """Decode native feature-major MXFP4 V into ``[B,S,H,D]`` order."""
    if payload.ndim != 4:
        raise ValueError("V payload must have shape [B,H,D,S/2]")
    batch, heads, depth, packed_sequence = payload.shape
    sequence = packed_sequence * 2
    expected_scales = (batch, sequence // 128, heads, 512)
    if depth != 128 or sequence % 128 or tuple(scales.shape) != expected_scales:
        raise ValueError("V payload or scale pages have an invalid D128 shape")

    decoded = _decode_packed_e2m1(payload).reshape(
        batch,
        heads,
        depth,
        sequence,
    )
    sequence_index = torch.arange(sequence, device=payload.device)
    depth_index = torch.arange(depth, device=payload.device)
    scale_tile = (sequence_index // 128)[None, :].expand(depth, -1)
    scale_offset = (
        (depth_index % 32)[:, None] * 16
        + (depth_index // 32)[:, None] * 4
        + ((sequence_index % 128) // 32)[None, :]
    )
    scale_bytes = scales.contiguous().view(torch.uint8)
    decode_scale = torch.empty_like(decoded)
    for batch_index in range(batch):
        for head_index in range(heads):
            exponent = scale_bytes[
                batch_index,
                scale_tile,
                head_index,
                scale_offset,
            ].float()
            decode_scale[batch_index, head_index] = torch.where(
                exponent > 0.0,
                torch.exp2(exponent - 127.0) / 6.0,
                torch.zeros_like(exponent),
            )
    return (decoded * decode_scale).permute(0, 3, 1, 2).contiguous()


def _round_e4m3_scale(scale: torch.Tensor) -> torch.Tensor:
    # PyTorch returns NaN above the finite E4M3 endpoint whereas the CUDA
    # producer fixes 0x7f to 0x7e.  Clamp first to model that satfinite path.
    return (
        scale.float()
        .clamp(min=0.0, max=E4M3_MAX)
        .to(torch.float8_e4m3fn)
        .float()
    )


def _candidate_nvfp4(
    blocks: torch.Tensor,
    block_amax: torch.Tensor,
    global_decode: torch.Tensor,
    endpoint: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    local_scale = _round_e4m3_scale(
        block_amax / endpoint / global_decode
    )
    decode = local_scale * global_decode
    safe_decode = decode.clamp_min(torch.finfo(torch.float32).tiny)
    normalized = blocks / safe_decode[..., None, None]
    payload = _quantize_e2m1(normalized)
    decoded = payload * decode[..., None, None]
    return decoded, payload, normalized, local_scale


def fake_quantize_nvfp4(
    tensor: torch.Tensor,
    *,
    block_shape: tuple[int, int],
    selector: str = "static6",
    scale_target: float = E4M3_MAX,
) -> FakeQuantResult:
    """Decode a reference NVFP4 representation.

    ``block_shape=(1, 16)`` models projection activations.  Projection weights
    must use ``(16, 16)``: the scale-selection objective is reduced over all
    256 values, so independently consuming the row and transpose layouts
    cannot change the represented weight.

    ``selector`` may be ``static6``, ``static4``, ``adaptive_mae``, or
    ``adaptive_mse``.  Adaptive modes compare endpoint-4 and endpoint-6
    reconstructions within the same block and retain one shared result.
    """
    _require_matrix_and_blocks(tensor, block_shape)
    if selector not in {
        "static6",
        "static4",
        "adaptive_mae",
        "adaptive_mse",
    }:
        raise ValueError(f"unknown NVFP4 selector {selector!r}")
    if not 0.0 < scale_target <= 1024.0:
        raise ValueError("scale_target must be in (0, 1024]")

    blocks = _to_blocks(tensor, block_shape)
    block_amax = blocks.abs().amax(dim=(-2, -1))
    global_amax = block_amax.amax()
    if float(global_amax) == 0.0:
        scale_grid = torch.zeros_like(block_amax)
        return FakeQuantResult(
            values=torch.zeros_like(tensor),
            block_scales=scale_grid,
            diagnostics={
                "block_shape": list(block_shape),
                "selector": selector,
                "scale_target": float(scale_target),
                "global_amax": 0.0,
                "global_decode": 0.0,
                "selected_endpoint4_fraction": 0.0,
                "scale_saturation_fraction": 0.0,
                "payload_saturation_fraction": 0.0,
                "payload_clipping_fraction": 0.0,
                "payload_zero_fraction": 1.0,
            },
        )

    # The production global tensor scale remains based on the E2M1 endpoint
    # six.  Targets above 448 intentionally stress local E4M3 saturation; they
    # are not an extension of the format's representable range.
    global_decode = global_amax / (6.0 * scale_target)
    decoded6, payload6, normalized6, scale6 = _candidate_nvfp4(
        blocks,
        block_amax,
        global_decode,
        6.0,
    )
    selected4 = torch.zeros_like(block_amax, dtype=torch.bool)

    if selector == "static6":
        decoded = decoded6
        payload = payload6
        normalized = normalized6
        scale = scale6
    else:
        decoded4, payload4, normalized4, scale4 = _candidate_nvfp4(
            blocks,
            block_amax,
            global_decode,
            4.0,
        )
        if selector == "static4":
            selected4 = torch.ones_like(block_amax, dtype=torch.bool)
        else:
            error6 = decoded6 - blocks
            error4 = decoded4 - blocks
            if selector == "adaptive_mae":
                objective6 = error6.abs().sum(dim=(-2, -1))
                objective4 = error4.abs().sum(dim=(-2, -1))
            else:
                objective6 = error6.square().sum(dim=(-2, -1))
                objective4 = error4.square().sum(dim=(-2, -1))
            selected4 = objective4 < objective6
        mask = selected4[..., None, None]
        decoded = torch.where(mask, decoded4, decoded6)
        payload = torch.where(mask, payload4, payload6)
        normalized = torch.where(mask, normalized4, normalized6)
        scale = torch.where(selected4, scale4, scale6)

    values = _from_blocks(decoded, tensor.shape).to(tensor.dtype)
    return FakeQuantResult(
        values=values,
        block_scales=scale,
        diagnostics={
            "block_shape": list(block_shape),
            "selector": selector,
            "scale_target": float(scale_target),
            "global_amax": float(global_amax),
            "global_decode": float(global_decode),
            "selected_endpoint4_fraction": float(selected4.float().mean()),
            "scale_saturation_fraction": float(
                (scale == E4M3_MAX).float().mean()
            ),
            "payload_saturation_fraction": float(
                (payload.abs() == 6.0).float().mean()
            ),
            "payload_clipping_fraction": float(
                (normalized.abs() > 6.0).float().mean()
            ),
            "payload_zero_fraction": float(
                (payload == 0.0).float().mean()
            ),
        },
    )


def fake_quantize_mxfp4(
    tensor: torch.Tensor,
    *,
    block_shape: tuple[int, int],
    scale_mode: str = "ceil",
) -> FakeQuantResult:
    """Decode an E8M0/E2M1 MXFP4 representation.

    Activations use ``(1, 32)``.  Projection weights use ``(32, 32)`` so the
    identical E2M1 codes and E8M0 scale can be emitted in both orientations.
    ``dense`` mirrors the retained 2D-weight heuristic that drops the safe
    exponent when the amax phase is below 0.87.
    """
    _require_matrix_and_blocks(tensor, block_shape)
    if scale_mode not in {"ceil", "rte", "dense"}:
        raise ValueError(f"unknown MXFP4 scale mode {scale_mode!r}")

    blocks = _to_blocks(tensor, block_shape)
    block_amax = blocks.abs().amax(dim=(-2, -1))
    safe_amax = block_amax.clamp_min(torch.finfo(torch.float32).tiny)
    logarithm = torch.log2(safe_amax)
    if scale_mode == "rte":
        exponent = torch.round(logarithm)
    else:
        exponent = torch.ceil(logarithm)
        if scale_mode == "dense":
            safe_scale = torch.exp2(exponent)
            phase = block_amax / safe_scale
            exponent = torch.where(
                (block_amax > 0.0) & (phase < 0.87),
                exponent - 1.0,
                exponent,
            )
    scale = torch.where(
        block_amax > 0.0,
        torch.exp2(exponent),
        torch.zeros_like(block_amax),
    )
    decode = scale / 6.0
    normalized = blocks / decode.clamp_min(torch.finfo(torch.float32).tiny)[
        ..., None, None
    ]
    payload = _quantize_e2m1(normalized)
    decoded = payload * decode[..., None, None]
    values = _from_blocks(decoded, tensor.shape).to(tensor.dtype)
    return FakeQuantResult(
        values=values,
        block_scales=scale,
        diagnostics={
            "block_shape": list(block_shape),
            "scale_mode": scale_mode,
            "payload_saturation_fraction": float(
                (payload.abs() == 6.0).float().mean()
            ),
            "payload_clipping_fraction": float(
                (normalized.abs() > 6.0).float().mean()
            ),
            "payload_zero_fraction": float(
                (payload == 0.0).float().mean()
            ),
        },
    )


def fake_quantize_mxfp4_v_1d(
    tensor: torch.Tensor,
    *,
    rounding: str = "rne",
    generator: torch.Generator | None = None,
) -> FakeQuantResult:
    """QDQ logical V with the production 1x32 sequence-group policy.

    ``tensor`` is logical ``[B,S,Hkv,D]`` V.  Each feature row receives one
    power-of-two scale for 32 consecutive sequence values, matching the fast
    forward MXFP4 publisher.  Scale selection uses the production BF16-exact
    1.203125 cutoff.  ``stochastic`` changes only E2M1 rounding; it deliberately
    leaves the deterministic E8M0 selector untouched so the experiment can
    isolate whether randomized payload rounding recovers small backward
    signal.

    This is a readable numerical model, not a timed production quantizer.
    """
    if tensor.ndim != 4 or not tensor.is_floating_point():
        raise ValueError("MXFP4 V QDQ requires floating [B,S,H,D]")
    batch, sequence, heads, depth = tensor.shape
    if batch <= 0 or heads <= 0 or depth <= 0 or sequence % 32:
        raise ValueError(
            "MXFP4 V QDQ requires positive B/H/D and S divisible by 32"
        )
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("MXFP4 V QDQ requires finite input")
    if rounding not in {"rne", "stochastic"}:
        raise ValueError("rounding must be 'rne' or 'stochastic'")
    if rounding == "stochastic" and generator is None:
        raise ValueError("stochastic rounding requires an explicit generator")

    blocks = (
        tensor.float()
        .permute(0, 2, 3, 1)
        .contiguous()
        .reshape(batch, heads, depth, sequence // 32, 32)
    )
    # The CUDA selector compares BF16 magnitude bits.  Casting before amax
    # preserves that contract even when a caller supplies an FP32 reference.
    block_amax = blocks.bfloat16().abs().amax(dim=-1).float()
    positive = block_amax > 0.0
    safe_amax = block_amax.clamp_min(torch.finfo(torch.float32).tiny)
    lower_exponent = torch.floor(torch.log2(safe_amax))
    normalized_amax = safe_amax / torch.exp2(lower_exponent)
    selected_exponent = lower_exponent + (
        normalized_amax >= 1.203125
    ).to(torch.float32)
    # E8M0 byte zero is the all-zero sentinel; finite nonzero BF16 amax values
    # map to biased exponents 1..254, i.e. unbiased exponents -126..127.
    selected_exponent = selected_exponent.clamp_(-126.0, 127.0)
    scale = torch.where(
        positive,
        torch.exp2(selected_exponent),
        torch.zeros_like(selected_exponent),
    )
    decode = scale / 6.0
    normalized = blocks / decode.clamp_min(
        torch.finfo(torch.float32).tiny
    )[..., None]
    if rounding == "rne":
        payload = _quantize_e2m1_rne(normalized)
    else:
        assert generator is not None
        payload = _quantize_e2m1_stochastic(
            normalized,
            generator=generator,
        )
    decoded_blocks = payload * decode[..., None]
    values = (
        decoded_blocks.reshape(batch, heads, depth, sequence)
        .permute(0, 3, 1, 2)
        .contiguous()
        .to(tensor.dtype)
    )
    nonzero_input = blocks != 0.0
    zeroed_nonzero = nonzero_input & (payload == 0.0)
    difference = decoded_blocks - blocks
    return FakeQuantResult(
        values=values,
        block_scales=scale,
        diagnostics={
            "block_shape": [1, 32],
            "group_axis": "sequence",
            "scale_selector": "bf16_amax_1d_mse_cutoff_1.203125",
            "rounding": rounding,
            "payload_saturation_fraction": float(
                (payload.abs() == 6.0).float().mean()
            ),
            "payload_clipping_fraction": float(
                (normalized.abs() > 6.0).float().mean()
            ),
            "payload_zero_fraction": float(
                (payload == 0.0).float().mean()
            ),
            "nonzero_input_zeroed_fraction": float(
                zeroed_nonzero.sum()
                / nonzero_input.sum().clamp_min(1)
            ),
            "signed_error_mean": float(difference.mean()),
            "error_rmse": float(difference.square().mean().sqrt()),
        },
    )


def tensor_error_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, float]:
    """Return stable scalar error metrics using FP32 reductions."""
    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate shapes differ")
    expected = reference.float().reshape(-1)
    actual = candidate.float().reshape(-1)
    difference = actual - expected
    expected_norm = torch.linalg.vector_norm(expected)
    actual_norm = torch.linalg.vector_norm(actual)
    denominator = expected_norm.clamp_min(torch.finfo(torch.float32).tiny)
    cosine_denominator = (expected_norm * actual_norm).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    return {
        "cosine": float(torch.dot(expected, actual) / cosine_denominator),
        "relative_l2": float(torch.linalg.vector_norm(difference) / denominator),
        "norm_ratio": float(actual_norm / denominator),
        "mae": float(difference.abs().mean()),
        "rmse": float(difference.square().mean().sqrt()),
        "max_abs_error": float(difference.abs().amax()),
    }
