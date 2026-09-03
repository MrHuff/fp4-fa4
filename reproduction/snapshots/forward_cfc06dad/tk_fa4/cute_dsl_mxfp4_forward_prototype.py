from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from cute_dsl_mxfp4_forward_scaffold import (
    load_reference_blackwell_fmha_module,
    load_reference_blockscaled_gemm_module,
    load_reference_mixed_input_fmha_d256_module,
    reference_entrypoints,
)


_REFERENCE_FMHA_SUPPORTED_HEAD_DIMS = (32, 64, 128)
_REFERENCE_MIXED_INPUT_FMHA_SUPPORTED_HEAD_DIMS = (256,)


@dataclass(frozen=True)
class Mxfp4CutePrototypeConfig:
    qk_head_dim: int = 128
    v_head_dim: int = 128
    mma_tiler_mn: tuple[int, int] = (128, 128)
    cluster_shape_mn: tuple[int, int] = (1, 1)
    scale_granularity: int = 128
    qk_sf_vec_size: int = 32
    pv_sf_vec_size: int = 32
    is_persistent: bool = True
    is_causal: bool = False


@dataclass(frozen=True)
class Mxfp4CutePrototypeBundle:
    config: Mxfp4CutePrototypeConfig
    fmha_kernel: Any
    qk_blockscaled_gemm: Any
    pv_blockscaled_gemm: Any
    fmha_variant: str
    fmha_class_name: str
    blockscaled_gemm_class_name: str


def supported_reference_head_dims() -> tuple[int, ...]:
    return _REFERENCE_FMHA_SUPPORTED_HEAD_DIMS


def supported_mixed_input_reference_head_dims() -> tuple[int, ...]:
    return _REFERENCE_MIXED_INPUT_FMHA_SUPPORTED_HEAD_DIMS


def prototype_gap_report(
    config: Mxfp4CutePrototypeConfig | None = None,
) -> dict[str, Any]:
    config = config or Mxfp4CutePrototypeConfig()
    qk_head_dim_supported = config.qk_head_dim in supported_reference_head_dims()
    qk_head_dim_supported_by_mixed_input = (
        config.qk_head_dim in supported_mixed_input_reference_head_dims()
    )
    return {
        "config": asdict(config),
        "reference_entrypoints": reference_entrypoints(),
        "reference_constraints": {
            "fmha_supported_qk_head_dims": supported_reference_head_dims(),
            "mixed_input_fmha_supported_qk_head_dims": supported_mixed_input_reference_head_dims(),
            "blockscaled_fp4_sf_vec_size": 32,
            "closest_large_d_reference": (
                "blackwell.mixed_input_fmha.mixed_input_fmha_prefill_d256."
                "MixedInputFusedMultiHeadAttentionPrefillD256"
            ),
            "notes": (
                "The Blackwell FMHA CuTe DSL reference only documents head dims 32/64/128. "
                "The mixed-input FMHA reference is hard-wired to D=256. "
                "A tk_fa4-style qk_head_dim=192 port will require adapting one of those references. "
                "The closest in-tree larger-D FMHA reference is the mixed-input D256 prefill kernel."
            ),
        },
        "gaps": {
            "qk_head_dim_supported_by_reference_fmha": qk_head_dim_supported,
            "qk_head_dim_supported_by_mixed_input_fmha_d256": qk_head_dim_supported_by_mixed_input,
            "requires_fmha_head_dim_extension": not (
                qk_head_dim_supported or qk_head_dim_supported_by_mixed_input
            ),
            "requires_online_mxfp4_p_quantization": True,
            "requires_prequantized_v_payload_and_scale_pipeline": True,
            "requires_blockscaled_qk_and_pv_mma_swap": True,
        },
    }


def build_reference_kernel_bundle(
    config: Mxfp4CutePrototypeConfig | None = None,
) -> Mxfp4CutePrototypeBundle:
    config = config or Mxfp4CutePrototypeConfig()
    if config.qk_head_dim in supported_reference_head_dims():
        fmha_mod = load_reference_blackwell_fmha_module()
        import cutlass  # type: ignore

        fmha_kernel = fmha_mod.BlackwellFusedMultiHeadAttentionForward(
            cutlass.Float32,
            cutlass.Float32,
            (*config.mma_tiler_mn, config.qk_head_dim),
            config.is_persistent,
            fmha_mod.fmha_utils.MaskEnum.WINDOW_MASK,
        )
        fmha_variant = "dense_fmha"
    elif config.qk_head_dim in supported_mixed_input_reference_head_dims():
        if config.v_head_dim != config.qk_head_dim:
            raise ValueError(
                "The mixed-input D256 FMHA reference expects qk_head_dim == v_head_dim == 256, "
                f"got qk_head_dim={config.qk_head_dim}, v_head_dim={config.v_head_dim}."
            )
        fmha_mod = load_reference_mixed_input_fmha_d256_module()
        import cutlass  # type: ignore

        fmha_kernel = fmha_mod.MixedInputFusedMultiHeadAttentionPrefillD256(
            config.scale_granularity,
            cutlass.Float32,
            cutlass.Float32,
            config.is_persistent,
            fmha_mod.fmha_utils.MaskEnum.WINDOW_MASK_INFERENCE
            if not config.is_causal
            else fmha_mod.fmha_utils.MaskEnum.WINDOW_MASK_INFERENCE,
        )
        fmha_variant = "mixed_input_fmha_d256"
    else:
        raise ValueError(
            "No direct CuTe DSL FMHA reference matches this qk_head_dim. "
            f"Dense FMHA supports {supported_reference_head_dims()}, "
            f"mixed-input FMHA supports {supported_mixed_input_reference_head_dims()}, "
            f"got qk_head_dim={config.qk_head_dim}. Extend one of those references before "
            f"attempting a tk_fa4 qk_head_dim={config.qk_head_dim} port."
        )
    if (
        config.qk_head_dim in supported_reference_head_dims()
        and config.v_head_dim not in supported_reference_head_dims()
    ):
        raise ValueError(
            f"Reference FMHA only supports v_head_dim in {supported_reference_head_dims()}, "
            f"got {config.v_head_dim}."
        )

    blockscaled_mod = load_reference_blockscaled_gemm_module()
    qk_blockscaled_gemm = blockscaled_mod.Sm100BlockScaledPersistentDenseGemmKernel(
        sf_vec_size=config.qk_sf_vec_size,
        mma_tiler_mn=config.mma_tiler_mn,
        cluster_shape_mn=config.cluster_shape_mn,
    )
    pv_blockscaled_gemm = blockscaled_mod.Sm100BlockScaledPersistentDenseGemmKernel(
        sf_vec_size=config.pv_sf_vec_size,
        mma_tiler_mn=config.mma_tiler_mn,
        cluster_shape_mn=config.cluster_shape_mn,
    )

    return Mxfp4CutePrototypeBundle(
        config=config,
        fmha_kernel=fmha_kernel,
        qk_blockscaled_gemm=qk_blockscaled_gemm,
        pv_blockscaled_gemm=pv_blockscaled_gemm,
        fmha_variant=fmha_variant,
        fmha_class_name=type(fmha_kernel).__name__,
        blockscaled_gemm_class_name=type(qk_blockscaled_gemm).__name__,
    )


def summarize_reference_kernel_bundle(
    config: Mxfp4CutePrototypeConfig | None = None,
) -> dict[str, Any]:
    config = config or Mxfp4CutePrototypeConfig()
    summary = prototype_gap_report(config)
    if summary["gaps"]["requires_fmha_head_dim_extension"]:
        summary["instantiation"] = {
            "ready": False,
            "reason": (
                f"qk_head_dim={config.qk_head_dim} is outside the direct reference support sets: "
                f"dense={supported_reference_head_dims()}, "
                f"mixed_input_d256={supported_mixed_input_reference_head_dims()}."
            ),
        }
        return summary

    bundle = build_reference_kernel_bundle(config)
    summary["instantiation"] = {
        "ready": True,
        "fmha_variant": bundle.fmha_variant,
        "fmha_class_name": bundle.fmha_class_name,
        "blockscaled_gemm_class_name": bundle.blockscaled_gemm_class_name,
        "qk_mma_tiler": (*config.mma_tiler_mn, config.qk_head_dim),
        "pv_mma_tiler": (*config.mma_tiler_mn, config.v_head_dim),
        "scale_granularity": config.scale_granularity,
        "qk_sf_vec_size": config.qk_sf_vec_size,
        "pv_sf_vec_size": config.pv_sf_vec_size,
    }
    return summary
