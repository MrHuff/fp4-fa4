from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from cute_dsl_mxfp4_forward_scaffold import reference_entrypoints


@dataclass(frozen=True)
class Mxfp4D192PortGeometry:
    qk_head_dim: int = 192
    v_head_dim: int = 128
    seq_tile_n: int = 128
    qk_head_dim_padded: int = 256
    qk_cta_tiler: tuple[int, int, int] = (128, 128, 256)
    qk_mma_tiler: tuple[int, int, int] = (256, 128, 64)
    pv_cta_tiler: tuple[int, int, int] = (128, 128, 128)
    pv_mma_tiler: tuple[int, int, int] = (256, 128, 128)
    qk_scale_granularity: int = 64
    v_scale_granularity: int = 128
    qk_sf_vec_size: int = 32
    pv_sf_vec_size: int = 32
    cluster_shape_mn: tuple[int, int] = (2, 1)
    is_persistent: bool = True
    is_causal: bool = False


def default_d192_port_geometry() -> Mxfp4D192PortGeometry:
    return Mxfp4D192PortGeometry()


def validate_d192_port_geometry(
    geometry: Mxfp4D192PortGeometry | None = None,
) -> dict[str, Any]:
    geometry = geometry or default_d192_port_geometry()
    checks = {
        "qk_head_dim_fits_inside_qk_head_dim_padded": geometry.qk_head_dim
        <= geometry.qk_head_dim_padded,
        "qk_head_dim_padded_covered_by_qk_mma_k": geometry.qk_head_dim_padded
        % geometry.qk_mma_tiler[2]
        == 0,
        "v_head_dim_covered_by_pv_mma_n": geometry.v_head_dim == geometry.pv_mma_tiler[1],
        "seq_tile_matches_pv_k": geometry.seq_tile_n == geometry.pv_mma_tiler[2],
        "qk_scale_granularity_divides_qk_mma_k": geometry.qk_scale_granularity
        % geometry.qk_mma_tiler[2]
        == 0,
        "qk_scale_granularity_divides_qk_head_dim_padded": geometry.qk_head_dim_padded
        % geometry.qk_scale_granularity
        == 0,
        "v_scale_granularity_divides_pv_mma_n": geometry.v_scale_granularity
        % geometry.pv_mma_tiler[1]
        == 0,
        "qk_cta_depth_matches_padded_head_dim": geometry.qk_cta_tiler[2]
        == geometry.qk_head_dim_padded,
        "pv_cta_depth_matches_v_dim": geometry.pv_cta_tiler[2] == geometry.v_head_dim,
    }
    return {
        "geometry": asdict(geometry),
        "checks": checks,
        "all_pass": all(checks.values()),
    }


def d192_port_patch_points() -> tuple[str, ...]:
    return (
        "Start from `MixedInputFusedMultiHeadAttentionPrefillD256.__call__` and replace its dense `make_trivial_tiled_mma(...)` setup with blockscaled FP4 MMA setup.",
        "Split the current single large-D mixed-input path into separate QK and PV tilers: QK pads D=192 to 256, PV keeps V=128.",
        "Replace `prefill_helpers.get_scale_smem_layout(...)` usage with a D192-compatible Q/K/V scale layout plan driven by `qk_scale_granularity=64` and `v_scale_granularity=128`.",
        "Swap dense K/V dequant path for prequantized blockscaled FP4 payload + scale staging.",
        "Replace dense PV issue in `mma_pv(...)` with blockscaled P/V MMA using MXFP4 P payload + scales.",
        "Keep online softmax / correction structure from mixed-input FMHA; only the data representation and MMA path should change first.",
        "Mask or zero the padded Q/K lanes so the extra 64 FP4 channels do not contribute numerically.",
    )


def d192_reference_lineage() -> dict[str, str]:
    return {
        "scheduler_mainloop_base": reference_entrypoints()[
            "mixed_input_fmha_d256_class"
        ],
        "blockscaled_mma_layout_base": reference_entrypoints()[
            "blockscaled_gemm_class"
        ],
        "small_dense_reference": reference_entrypoints()["fmha_class"],
    }


def build_runtime_port_state(
    geometry: Mxfp4D192PortGeometry | None = None,
) -> dict[str, Any]:
    geometry = geometry or default_d192_port_geometry()
    # These values are intentionally symbolic. The CuTe DSL helper paths used to build the
    # actual blockscaled tiled-MMA and scale layouts are JIT-only and expect an MLIR context,
    # so they cannot be executed from a plain host Python process. The formulas below are taken
    # directly from:
    # - cutlass/utils/blackwell_helpers.py::make_blockscaled_trivial_tiled_mma
    # - blackwell/mixed_input_fmha/prefill_helpers.py::get_scale_smem_layout
    qk_mma_inst_k = 64
    pv_mma_inst_k = 64
    mma_inst_tile_k = 4
    derived_qk_cta_k = qk_mma_inst_k * mma_inst_tile_k
    derived_pv_reduction_k = pv_mma_inst_k * mma_inst_tile_k
    qk_d_r = geometry.qk_head_dim_padded // geometry.qk_scale_granularity
    v_d_r = geometry.v_head_dim // geometry.v_scale_granularity
    qk_scale_tiler = (geometry.qk_mma_tiler[1] * qk_d_r,)
    v_scale_tiler = (geometry.pv_mma_tiler[1] * v_d_r,)

    return {
        "geometry": asdict(geometry),
        "symbolic_runtime_notes": {
            "jit_only": True,
            "blockscaled_helper_source": (
                "For sf_vec_size=32, make_blockscaled_trivial_tiled_mma uses "
                "MmaMXF4Op(..., K=64), and blockscaled GEMM derives CTA K as 4 * 64 = 256."
            ),
            "scale_layout_source": (
                "For K-major operands, prefill_helpers.get_scale_smem_layout uses "
                "scale_tiler = (mma_tiler[1] * d_r,) and requires "
                "scale_granularity % mma_tiler[2] == 0."
            ),
        },
        "qk": {
            "mma_inst_k": qk_mma_inst_k,
            "derived_cta_reduction_k": derived_qk_cta_k,
            "scale_tiler": qk_scale_tiler,
            "d_r": qk_d_r,
        },
        "pv": {
            "mma_inst_k": pv_mma_inst_k,
            "derived_cta_reduction_k": derived_pv_reduction_k,
            "scale_tiler": v_scale_tiler,
            "d_r": v_d_r,
        },
    }


def summarize_d192_port_plan(
    geometry: Mxfp4D192PortGeometry | None = None,
) -> dict[str, Any]:
    geometry = geometry or default_d192_port_geometry()
    return {
        "geometry_validation": validate_d192_port_geometry(geometry),
        "reference_lineage": d192_reference_lineage(),
        "patch_points": d192_port_patch_points(),
    }
