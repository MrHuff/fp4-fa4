"""Isolated CuTe-DSL prototype for the D128 GQA ``dP = dO @ V.T`` GEMM.

The production backward kernel currently publishes E4M3 copies of ``dO`` and
``V`` for dP even when their MXFP4 payloads already exist.  This module tests
the narrow replacement: one SM100 block-scaled GEMM consumes the producer's
packed row-major MXFP4 payloads and E8M0 pages.  It deliberately does not
change the Q/K, P, dS, or dV contracts.

The prototype keeps the GQA broadcast in the CuTe layout.  ``V`` therefore
remains [B, S, Hkv, D/2] and is not repeated to Hq.  Scale pages need one
small metadata reorder because the producer writes [B, S/128, H, 512], while
the stock CUTLASS block-scaled mainloop expects its batch mode outside the
row-page mode.  ``prepare_scale_pages`` performs that reorder and repeats
only the 512-byte scale records for GQA, never the FP4 V payload.

Both backward operands use the standard width-six E2M1 reconstruction while
storing the E8M0 exponent of the unscaled block maximum.  Consequently each
raw block-scaled operand is 6x its represented value and the MMA result is
``dP_x36``.  The active backward retains its existing ``dPsum_x16`` domain,
so it must compute ``dP_x36 * (16 / 36) - dPsum_x16`` before the dS
quantization.  This prototype keeps that exact contract rather than rewriting
the scale bytes.

This is an experimental, Python-loaded extension.  Importing it does not load
CUTLASS or initialize CUDA; those dependencies are loaded only by ``compile``.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import torch


HEAD_DIM = 128
MXFP4_PACKED_HEAD_DIM = HEAD_DIM // 2
MXFP4_SCALE_VECTOR = 32
MXFP4_SCALE_PAGE_ROWS = 128
MXFP4_SCALE_PAGE_BYTES = 512
E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


@dataclass(frozen=True)
class D128GqaDpGeometry:
    """Static geometry compiled into one CuTe dP callable."""

    batch: int
    sequence: int
    q_heads: int
    kv_heads: int
    mma_tiler_mn: tuple[int, int] = (128, 128)
    cluster_shape_mn: tuple[int, int] = (1, 1)

    def __post_init__(self) -> None:
        if self.batch <= 0 or self.sequence <= 0:
            raise ValueError("batch and sequence must be positive")
        if self.sequence % MXFP4_SCALE_PAGE_ROWS:
            raise ValueError("sequence must be divisible by 128")
        if self.q_heads <= 0 or self.kv_heads <= 0:
            raise ValueError("head counts must be positive")
        if self.q_heads % self.kv_heads:
            raise ValueError("q_heads must be divisible by kv_heads")

    @property
    def group_size(self) -> int:
        return self.q_heads // self.kv_heads

    @property
    def scale_pages(self) -> int:
        return self.sequence // MXFP4_SCALE_PAGE_ROWS


def _load_dense_blockscaled_reference() -> ModuleType:
    root = Path(__file__).resolve().parents[2]
    source = (
        root
        / "cutlass"
        / "examples"
        / "python"
        / "CuTeDSL"
        / "blackwell"
        / "dense_blockscaled_gemm_persistent.py"
    )
    if not source.is_file():
        raise FileNotFoundError(f"CUTLASS block-scaled reference is missing: {source}")
    spec = importlib.util.spec_from_file_location(
        "_tk_fa4_dense_blockscaled_gemm_persistent",
        source,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load CUTLASS block-scaled reference: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_producer_tensors(
    geometry: D128GqaDpGeometry,
    dout_fp4: torch.Tensor,
    v_fp4: torch.Tensor,
    dout_scales: torch.Tensor,
    v_scales: torch.Tensor,
) -> None:
    """Validate the producer-facing packed payload and scale-page ABI."""

    expected_dout = (
        geometry.batch,
        geometry.sequence,
        geometry.q_heads,
        MXFP4_PACKED_HEAD_DIM,
    )
    expected_v = (
        geometry.batch,
        geometry.sequence,
        geometry.kv_heads,
        MXFP4_PACKED_HEAD_DIM,
    )
    expected_dout_scales = (
        geometry.batch,
        geometry.scale_pages,
        geometry.q_heads,
        MXFP4_SCALE_PAGE_BYTES,
    )
    expected_v_scales = (
        geometry.batch,
        geometry.scale_pages,
        geometry.kv_heads,
        MXFP4_SCALE_PAGE_BYTES,
    )
    named = {
        "dout_fp4": (dout_fp4, expected_dout),
        "v_fp4": (v_fp4, expected_v),
        "dout_scales": (dout_scales, expected_dout_scales),
        "v_scales": (v_scales, expected_v_scales),
    }
    for name, (tensor, expected) in named.items():
        if tensor.device.type != "cuda":
            raise ValueError(f"{name} must be CUDA")
        if tensor.dtype != torch.uint8:
            raise ValueError(f"{name} must use uint8 packed storage")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if tuple(tensor.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}, got {tuple(tensor.shape)}")


def prepare_scale_pages(
    geometry: D128GqaDpGeometry,
    dout_scales: torch.Tensor,
    v_scales: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Put producer scale records in the stock GEMM's outer-L order.

    Returned tensors are physically [B, Hkv, G, S/128, 512].  Their flat L
    order is the same nested ``((G, Hkv), B)`` order used by the payload
    layouts.  V scale records are repeated over G; the packed V values are
    broadcast with a zero stride in the CuTe layout and are never copied.
    """

    expected_dout = (
        geometry.batch,
        geometry.scale_pages,
        geometry.q_heads,
        MXFP4_SCALE_PAGE_BYTES,
    )
    expected_v = (
        geometry.batch,
        geometry.scale_pages,
        geometry.kv_heads,
        MXFP4_SCALE_PAGE_BYTES,
    )
    if tuple(dout_scales.shape) != expected_dout:
        raise ValueError(
            f"dout_scales must have shape {expected_dout}, got {tuple(dout_scales.shape)}"
        )
    if tuple(v_scales.shape) != expected_v:
        raise ValueError(
            f"v_scales must have shape {expected_v}, got {tuple(v_scales.shape)}"
        )
    group = geometry.group_size
    dout_mma = (
        dout_scales.permute(0, 2, 1, 3)
        .reshape(
            geometry.batch,
            geometry.kv_heads,
            group,
            geometry.scale_pages,
            MXFP4_SCALE_PAGE_BYTES,
        )
        .contiguous()
    )
    v_mma = (
        v_scales.permute(0, 2, 1, 3)
        .unsqueeze(2)
        .expand(-1, -1, group, -1, -1)
        .contiguous()
    )
    return dout_mma, v_mma


def quantize_backward_mxfp4_reference(
    values: torch.Tensor,
    *,
    scale_rows: int,
    scale_selector: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Readable reproduction of the production width-six backward MXFP4 ABI.

    This helper exists only for the isolated smoke test.  The real extension
    consumes payloads emitted by the fused projection/dO publishers.  It
    returns packed E2M1 bytes [B,S,H,D/2], physical E8M0 scale pages
    [B,S/128,H,512], the represented FP32 tensor, and the raw operand seen by
    tcgen05.  The last tensor is exactly six times the represented value in
    the mathematical contract; returning it avoids introducing FP32 1/6
    roundoff into the bit-exact raw-MMA smoke reference.

    Production V uses ``scale_rows=1, scale_selector="mse_1d"``: each
    sequence row and K32 slice gets the BF16-exact 1.203125 MSE selector.
    Production project_dout uses ``scale_rows=32, scale_selector="rte"``:
    one RTE-selected scale is shared by each 32-sequence x 32-depth tile.
    """

    if values.ndim != 4 or values.shape[-1] != HEAD_DIM:
        raise ValueError("values must have shape [B,S,H,128]")
    batch, sequence, heads, _ = values.shape
    if sequence % MXFP4_SCALE_PAGE_ROWS:
        raise ValueError("sequence must be divisible by 128")
    if scale_rows not in {1, 32}:
        raise ValueError("scale_rows must be 1 or 32")
    if sequence % scale_rows:
        raise ValueError("sequence must be divisible by scale_rows")
    if scale_selector not in {"rte", "mse_1d"}:
        raise ValueError("scale_selector must be 'rte' or 'mse_1d'")
    if scale_rows == 32 and scale_selector != "rte":
        raise ValueError("32x32 dO scale groups require the RTE selector")
    if scale_rows == 1 and scale_selector != "mse_1d":
        raise ValueError("1x32 V scale groups require the MSE selector")
    source = values.bfloat16().float().contiguous()
    blocks = source.reshape(batch, sequence, heads, HEAD_DIM // 32, 32)
    if scale_rows == 1:
        scale_amax = blocks.abs().amax(dim=-1)
    else:
        scale_amax = (
            blocks.reshape(
                batch,
                sequence // scale_rows,
                scale_rows,
                heads,
                HEAD_DIM // 32,
                32,
            )
            .abs()
            .amax(dim=(2, 5))
        )

    # Match the BF16-bit selectors in projection_fp4_epilogue.cuh.  RTE uses
    # the 1.5 midpoint with even-exponent ties; the rowwise 1-D MSE selector
    # switches to the upper exponent at normalized BF16 amax 1.203125.
    bits = scale_amax.contiguous().view(torch.int32)
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    if scale_selector == "rte":
        half = 1 << 22
        round_up = (mantissa > half) | (
            (mantissa == half) & ((exponent & 1) != 0)
        )
    else:
        # BF16 mantissa 0x1a is 1.203125.  The source was narrowed to BF16
        # before amax, so testing the high seven FP32 mantissa bits is exact.
        round_up = (mantissa >> 16) >= 0x1A
    rounded = exponent + (round_up & (exponent < 0xFE)).to(torch.int32)
    finite_positive = (scale_amax > 1.0e-38) & (exponent != 0xFF)
    selected_scale_codes = torch.where(
        finite_positive,
        rounded,
        torch.zeros_like(rounded),
    ).to(torch.uint8)
    if scale_rows == 1:
        scale_codes = selected_scale_codes
    else:
        scale_codes = (
            selected_scale_codes[:, :, None]
            .expand(-1, -1, scale_rows, -1, -1)
            .reshape(batch, sequence, heads, HEAD_DIM // 32)
        )

    decode = torch.where(
        scale_codes > 0,
        torch.exp2(scale_codes.float() - 127.0) / 6.0,
        torch.ones_like(scale_codes, dtype=torch.float32),
    )
    normalized = blocks / decode[..., None]
    levels = torch.tensor(
        E2M1_LEVELS,
        device=values.device,
        dtype=torch.float32,
    )
    magnitude_index = (
        (normalized.abs()[..., None] - levels).abs().argmin(dim=-1).to(torch.uint8)
    )
    nibbles = magnitude_index | ((normalized < 0).to(torch.uint8) << 3)
    logical_nibbles = nibbles.reshape(batch, sequence, heads, HEAD_DIM)
    payload = (
        logical_nibbles[..., 0::2] | (logical_nibbles[..., 1::2] << 4)
    ).contiguous()
    signed_levels = levels[magnitude_index.long()] * torch.where(
        normalized < 0,
        -1.0,
        1.0,
    )
    represented = (
        signed_levels
        * decode[..., None]
    ).reshape_as(source)
    raw_operand = (
        signed_levels
        * torch.where(
            scale_codes > 0,
            torch.exp2(scale_codes.float() - 127.0),
            torch.zeros_like(scale_codes, dtype=torch.float32),
        )[..., None]
    ).reshape_as(source)

    physical_scales = torch.zeros(
        batch,
        sequence // MXFP4_SCALE_PAGE_ROWS,
        heads,
        MXFP4_SCALE_PAGE_BYTES,
        device=values.device,
        dtype=torch.uint8,
    )
    rows = torch.arange(sequence, device=values.device)
    k_blocks = torch.arange(HEAD_DIM // 32, device=values.device)
    pages = rows // MXFP4_SCALE_PAGE_ROWS
    offsets = (
        (rows % 32)[:, None] * 16
        + ((rows // 32) % 4)[:, None] * 4
        + k_blocks[None, :]
    )
    for batch_index in range(batch):
        for head_index in range(heads):
            physical_scales[
                batch_index,
                pages[:, None],
                head_index,
                offsets,
            ] = scale_codes[batch_index, :, head_index]
    return payload, physical_scales, represented, raw_operand


def build_kernel_class() -> type[Any]:
    """Build the lazy CuTe wrapper class around CUTLASS's proven mainloop."""

    reference = _load_dense_blockscaled_reference()
    import cutlass  # type: ignore
    import cutlass.cute as cute  # type: ignore
    import cutlass.utils.blockscaled_layout as blockscaled_utils  # type: ignore

    dense_kernel = reference.Sm100BlockScaledPersistentDenseGemmKernel

    class D128GqaMxfp4DpKernel:
        def __init__(self, geometry: D128GqaDpGeometry, max_active_clusters: int):
            self.geometry = geometry
            self.max_active_clusters = max_active_clusters
            self.gemm = dense_kernel(
                MXFP4_SCALE_VECTOR,
                geometry.mma_tiler_mn,
                geometry.cluster_shape_mn,
            )

        @cute.jit
        def __call__(
            self,
            dout_iter: Any,
            v_iter: Any,
            dout_scale_iter: Any,
            v_scale_iter: Any,
            dp_iter: Any,
            stream: Any,
        ):
            b = self.geometry.batch
            s = self.geometry.sequence
            hq = self.geometry.q_heads
            hkv = self.geometry.kv_heads
            group = self.geometry.group_size

            # Logical L is ((G, Hkv), B).  dO selects q_head=hkv*G+g;
            # V gives g a zero stride and therefore remains physically Hkv.
            l_shape = ((group, hkv), b)
            dout_layout = cute.make_layout(
                (s, HEAD_DIM, l_shape),
                stride=(
                    hq * HEAD_DIM,
                    1,
                    ((HEAD_DIM, group * HEAD_DIM), s * hq * HEAD_DIM),
                ),
            )
            v_layout = cute.make_layout(
                (s, HEAD_DIM, l_shape),
                stride=(
                    hkv * HEAD_DIM,
                    1,
                    ((0, HEAD_DIM), s * hkv * HEAD_DIM),
                ),
            )
            dp_layout = cute.make_layout(
                (s, s, l_shape),
                stride=(
                    s,
                    1,
                    ((s * s, group * s * s), hq * s * s),
                ),
            )
            dout = cute.make_tensor(dout_iter, dout_layout)
            v = cute.make_tensor(v_iter, v_layout)
            dp = cute.make_tensor(dp_iter, dp_layout)

            # The dense kernel rebuilds these exact scale layouts internally;
            # constructing them here gives it typed iterators with the correct
            # E8M0 element type and nested L extent.
            dout_sf_layout = blockscaled_utils.tile_atom_to_shape_SF(
                dout.shape,
                MXFP4_SCALE_VECTOR,
            )
            v_sf_layout = blockscaled_utils.tile_atom_to_shape_SF(
                v.shape,
                MXFP4_SCALE_VECTOR,
            )
            dout_scales = cute.make_tensor(dout_scale_iter, dout_sf_layout)
            v_scales = cute.make_tensor(v_scale_iter, v_sf_layout)
            self.gemm(
                dout,
                v,
                dout_scales,
                v_scales,
                dp,
                self.max_active_clusters,
                stream,
            )

    return D128GqaMxfp4DpKernel


class CompiledD128GqaMxfp4Dp:
    """Torch-facing owner for one shape-specialized CuTe callable."""

    def __init__(self, geometry: D128GqaDpGeometry, compiled: Any):
        self.geometry = geometry
        self._compiled = compiled

    @staticmethod
    def _typed_iterator(tensor: torch.Tensor, element_type: Any) -> Any:
        from cutlass.cute.runtime import from_dlpack  # type: ignore

        runtime_tensor = from_dlpack(tensor, assumed_align=16)
        runtime_tensor.element_type = element_type
        return runtime_tensor.iterator

    def __call__(
        self,
        dout_fp4: torch.Tensor,
        v_fp4: torch.Tensor,
        dout_scales_mma: torch.Tensor,
        v_scales_mma: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        import cutlass  # type: ignore
        import cutlass.torch as cutlass_torch  # type: ignore

        geometry = self.geometry
        if output is None:
            output_storage = torch.empty(
                geometry.batch,
                geometry.q_heads,
                geometry.sequence,
                geometry.sequence,
                device=dout_fp4.device,
                dtype=torch.uint8,
            )
            output = output_storage.view(torch.float8_e4m3fn)
        elif output.dtype == torch.float8_e4m3fn:
            output_storage = output.view(torch.uint8)
        elif output.dtype == torch.uint8:
            output_storage = output
            output = output_storage.view(torch.float8_e4m3fn)
        else:
            raise ValueError("output must use uint8 storage or torch.float8_e4m3fn")
        expected_output = (
            geometry.batch,
            geometry.q_heads,
            geometry.sequence,
            geometry.sequence,
        )
        if tuple(output.shape) != expected_output or not output.is_contiguous():
            raise ValueError(f"output must be contiguous with shape {expected_output}")
        expected_scale_shape = (
            geometry.batch,
            geometry.kv_heads,
            geometry.group_size,
            geometry.scale_pages,
            MXFP4_SCALE_PAGE_BYTES,
        )
        for name, tensor in (
            ("dout_scales_mma", dout_scales_mma),
            ("v_scales_mma", v_scales_mma),
        ):
            if tensor.dtype != torch.uint8 or not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous uint8")
            if tuple(tensor.shape) != expected_scale_shape:
                raise ValueError(
                    f"{name} must have shape {expected_scale_shape}, got {tuple(tensor.shape)}"
                )

        stream = cutlass_torch.default_stream()
        self._compiled(
            self._typed_iterator(dout_fp4, cutlass.Float4E2M1FN),
            self._typed_iterator(v_fp4, cutlass.Float4E2M1FN),
            self._typed_iterator(dout_scales_mma, cutlass.Float8E8M0FNU),
            self._typed_iterator(v_scales_mma, cutlass.Float8E8M0FNU),
            self._typed_iterator(output_storage, cutlass.Float8E4M3FN),
            stream,
        )
        return output


def compile(
    geometry: D128GqaDpGeometry,
    dout_fp4: torch.Tensor,
    v_fp4: torch.Tensor,
    dout_scales_mma: torch.Tensor,
    v_scales_mma: torch.Tensor,
) -> CompiledD128GqaMxfp4Dp:
    """Compile one shape-specialized dP kernel using representative pointers."""

    import cutlass  # type: ignore
    import cutlass.cute as cute  # type: ignore
    import cutlass.torch as cutlass_torch  # type: ignore
    from cutlass.cute.runtime import from_dlpack  # type: ignore

    output_storage = torch.empty(
        geometry.batch,
        geometry.q_heads,
        geometry.sequence,
        geometry.sequence,
        device=dout_fp4.device,
        dtype=torch.uint8,
    )

    def iterator(tensor: torch.Tensor, element_type: Any) -> Any:
        runtime_tensor = from_dlpack(tensor, assumed_align=16)
        runtime_tensor.element_type = element_type
        return runtime_tensor.iterator

    hardware_info = cutlass.utils.HardwareInfo()
    cluster_size = geometry.cluster_shape_mn[0] * geometry.cluster_shape_mn[1]
    max_active_clusters = hardware_info.get_max_active_clusters(cluster_size)
    kernel_type = build_kernel_class()
    kernel = kernel_type(geometry, max_active_clusters)
    stream = cutlass_torch.default_stream()
    compiled = cute.compile(
        kernel,
        iterator(dout_fp4, cutlass.Float4E2M1FN),
        iterator(v_fp4, cutlass.Float4E2M1FN),
        iterator(dout_scales_mma, cutlass.Float8E8M0FNU),
        iterator(v_scales_mma, cutlass.Float8E8M0FNU),
        iterator(output_storage, cutlass.Float8E4M3FN),
        stream,
        options="--opt-level 2",
    )
    return CompiledD128GqaMxfp4Dp(geometry, compiled)
