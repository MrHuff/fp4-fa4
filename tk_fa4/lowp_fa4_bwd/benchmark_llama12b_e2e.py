#!/usr/bin/env python3
"""Execute BF16 and low-precision Llama-3 training steps.

This is a single-GPU synthetic-token training benchmark of the complete
decoder, not a component sum.  The default remains the 16-layer
Llama-3.2-1B-like configuration; ``--model-preset llama3.1-8b`` selects the
full 32-layer, D128 Llama-3.1-8B configuration.  All unchanged RMSNorm,
SwiGLU, vocabulary, loss, and optimizer work is executed in both routes.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib.util
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn

from tk_fa4 import (
    B300E4M3QKVForwardWorkspace,
    MXFP4_V_SCALE_POLICY_ROWWISE_D32,
    MXFP4_V_SCALE_POLICY_SHARED_D32XS32,
    b300_bind_qkv_gqa_d128_unified_lowp_e4m3_projection,
    b300_bind_qkv_gqa_d128_unified_lowp_nvfp4_projection,
    b300_bind_qkv_gqa_d64_paired_unified_lowp_e4m3_projection,
    b300_bind_qkv_gqa_d64_paired_unified_lowp_nvfp4_projection,
    b300_lowp_bwd_extension_artifact_identity,
    b300_pack_gqa_d128_rope,
    b300_pack_gqa_d64_paired_rope,
    b300_pair_interleave_gqa_d128_qk_projection_weights,
    b300_prepare_e4m3_projection_operand,
    b300_prepare_e4m3_projection_weight,
    b300_prepare_nvfp4_projection_operand,
    b300_prepare_nvfp4_projection_operand_rmsnorm,
    b300_rmsnorm_backward,
    b300_prepare_gqa_d128_qkv_projection_weight_dual_out,
    b300_prepare_nvfp4_projection_weight,
    b300_prepare_nvfp4_projection_weight_dual_out,
    b300_prepare_nvfp4_projection_operand_scaled,
    b300_project_dout_unified_lowp_nvfp4,
    b300_project_dout_unified_lowp_nvfp4_v509_e5m2,
    b300_project_e4m3,
    b300_project_gqa_d128_hierarchical_qkv_gradient_nvfp4,
    b300_project_gqa_d64_paired_qkv_gradient_nvfp4,
    b300_project_nvfp4,
    b300_project_qkv_gqa_d128_unified_lowp_nvfp4,
    b300_project_qkv_gqa_d64_paired_unified_lowp_nvfp4,
    b300_stack_gqa_d128_qkv_projection_weights,
    b300_stitch_gqa_d128_inverse_rope_gradient,
    b300_stitch_gqa_d64_inverse_rope_gradient,
    b300_require_v509_e5m2_dout_route,
)
from tk_fa4.lowp_fa4_bwd.profile_gqa_d128_chain import (
    CompiledGqaBackward,
    _attention_cute_tensor,
    _inverse_rope_pair_native,
    _load_extension,
)
from tk_fa4.lowp_fa4_bwd.backward_policy import (
    resolve_backward_exp2_policy,
    resolve_backward_probability_tmem_policy,
)
from tk_fa4.lowp_fa4_bwd.backward_contract import (
    require_matching_backward_contracts,
    require_shared_backward_physical_identity,
)
from tk_fa4.lowp_fa4_bwd.forward_route import (
    activate_forward_route,
    require_active_forward_route,
)
from tk_fa4.lowp_fa4_bwd.native_tk_d64_backward import (
    BACKEND as NATIVE_TK_D64_BACKEND,
    NativeTkD64E4M3Backward,
)
from tk_fa4.lowp_fa4_bwd.native_tk_d128_backward import (
    BACKEND as NATIVE_TK_D128_BACKEND,
    NativeTkD128E4M3Backward,
)
from tk_fa4.lowp_fa4_bwd.native_tk_d128_nvfp4_score_backward import (
    BACKEND as NATIVE_TK_D128_NVFP4_SCORE_BACKEND,
    NativeTkD128NVFP4ScoreE4M3GradientBackward,
)
from tk_fa4.lowp_fa4_bwd.native_tk_d128_nvfp4_score_e5m2_dout_backward import (
    BACKEND as NATIVE_TK_D128_V509_E5M2_DOUT_BACKEND,
    NativeTkD128NVFP4ScoreE4M3QKVE5M2DoutBackward,
)
from tk_fa4.lowp_fa4_bwd.native_tk_d128_mxfp4_v_backward import (
    BACKEND as NATIVE_TK_D128_MX_BACKEND,
    SHARED_TILE_V503_BACKEND as NATIVE_TK_D128_SHARED_TILE_MX_BACKEND,
    NativeTkD128Mxfp4VBackward,
    NativeTkD128SharedTileProducerV503Backward,
)
from tk_fa4.lowp_fa4_bwd.packed_bf16_qkv import (
    PackedQKVAttentionWeights,
    PackedQKVLayout,
    project_packed_qkv,
)
from tk_fa4.lowp_fa4_bwd.tune_d64_gqa_cute import (
    _load_control,
    _require_d128_mxfp4_v_dp_patch_provenance,
)


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
_PROFILE_STAGE_RANGES = False
_PROFILE_STAGE_NVTX = False


def _source_content_identity(path: Path | str) -> dict[str, int | str]:
    """Fingerprint generated control code without binding to a temp path."""
    resolved = Path(path).resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return {
        "bytes": resolved.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _require_declared_artifact_identity(
    label: str,
    path: Path,
    sha256: str,
    byte_count: int,
) -> dict[str, int | str]:
    """Authenticate a caller-selected binary before importing it."""
    normalized_sha256 = sha256.lower()
    if (
        len(normalized_sha256) != 64
        or any(c not in "0123456789abcdef" for c in normalized_sha256)
        or type(byte_count) is not int
        or byte_count <= 0
    ):
        raise ValueError(
            f"{label} requires a 64-character SHA256 and positive byte count"
        )
    identity = _source_content_identity(path)
    if (
        identity["sha256"] != normalized_sha256
        or identity["bytes"] != byte_count
    ):
        raise ValueError(
            f"{label} artifact identity mismatch: observed {identity}, "
            f"expected SHA256 {normalized_sha256!r} and {byte_count} bytes"
        )
    return identity


D4ALL_FORWARD_PROBABILITY_REPLAY_TOPOLOGY = {
    "causal": True,
    "causal_interleaved_kv": True,
    "pv_format": "mxfp4_e8m0_block32",
    "mx_pwl_exp2": True,
    "mx_pwl_exp2_mode": 23,
    "mx_mode23_native_density": 4,
    "mx_mode23_native_quarter_mask": 15,
    "mx_mode23_native_stage_mask": 3,
    "mx_max_sample_stride": 1,
    "mx_max_sample_bias_x2": 0,
    "mx_q1_max_sample_stride": 1,
    "mx_q1_max_interleave": False,
    "mx_q1_pair_probe": 0,
    "mx_q1_pair_probe_bias": 0.0,
    "mx_lagged_q1_scale": False,
    "mx_stage0_q1_max_interleave": False,
    "mx_q2_max_interleave": False,
    "mx_q3_max_interleave": False,
    "mx_pair_scale_reuse": 1,
    "mx_pair_scale_period": 1,
    "mx_pair_scale_stage_mask": 1,
    "mx_pair_scale_delta": 0,
    "mx_q3_scale_reuse_stage_mask": 0,
    "mx_q1_self_max": 3,
    "mx_mode23_self_stage0_native": True,
    "mx_scale_select": 1,
    "mx_skip_zero_scale_mask": True,
    "mx_p_effective_max": 6,
    "mx_quantized_denom": True,
    "mx_denom_sample_stride": 1,
    "mx_denom_words": 4,
    "mx_shiftless_softmax": True,
    "mx_stored_scale_shift_log2": 16,
    "mx_stage0_affine_mask": 0,
    "mx_stage1_affine_mask": 0,
    "mx_log2_p_quant": True,
    "mx_direct_log_code": False,
    "mx_log_lut_bits": 0,
    "mx_fixed_anchor_log2": 0,
    "mx_global_anchor32": False,
    "mx_global_anchor128": False,
    "mx_global_anchor_bias": 0.0,
    "mx_anchor_affine_hoist": False,
    "rowmax_pack_ceiling": False,
    "score_pack_ceiling": False,
    "fixed_p_ceiling": False,
}


def _stage(name: str) -> Any:
    if _PROFILE_STAGE_RANGES:
        if _PROFILE_STAGE_NVTX:
            return torch.cuda.nvtx.range(name)
        return torch.profiler.record_function(name)
    return contextlib.nullcontext()


FLASH_ATTN_ROOT = REPO_ROOT / "flash-attention"
sys.path.insert(0, str(FLASH_ATTN_ROOT))
try:
    from flash_attn.cute.interface import flash_attn_func
finally:
    sys.path.pop(0)


@dataclass(frozen=True)
class Config:
    model_preset: str = "llama3.2-1b"
    batch: int = 1
    sequence: int = 4096
    hidden: int = 2048
    intermediate: int = 8192
    layers: int = 16
    full_model_layers: int = 16
    q_heads: int = 32
    kv_heads: int = 8
    head_dim: int = 64
    vocab: int = 128256
    rms_epsilon: float = 1.0e-5
    tie_word_embeddings: bool = True
    max_position_embeddings: int = 131072
    rope_theta: float = 500_000.0
    rope_factor: float = 32.0
    rope_low_frequency_factor: float = 1.0
    rope_high_frequency_factor: float = 4.0
    rope_original_context: int = 8192
    d128_forward_topology_variant: str = "production"

    @property
    def q_width(self) -> int:
        return self.q_heads * self.head_dim

    @property
    def kv_width(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def parameter_count(self) -> int:
        embedding_tables = 1 + int(not self.tie_word_embeddings)
        attention = (
            self.hidden * self.q_width
            + 2 * self.hidden * self.kv_width
            + self.hidden * self.q_width
        )
        mlp = 3 * self.hidden * self.intermediate
        decoder = self.layers * (attention + mlp + 2 * self.hidden)
        return (
            embedding_tables * self.vocab * self.hidden
            + decoder
            + self.hidden
        )


DEFAULT_MODEL_PRESET = "llama3.2-1b"
MODEL_PRESETS = (DEFAULT_MODEL_PRESET, "llama3.1-8b")
AUTHENTICATED_D64_EXACT_BATCHES = (2, 8, 16)
AUTHENTICATED_D128_EXACT_BATCHES = (2, 4)
SUPPORTED_LOWP_BATCHES = (1, 2, 4, 8, 16)
D128_EXACT_FORWARD_TOPOLOGIES = {
    (
        "real_fwd_tk_hao_direct_causal_gqa_nvfp4_fp8pv",
        "e4m3_fp8",
    ): {
        "route": "real_fwd_tk_hao_direct_causal_gqa_nvfp4_fp8pv",
        "schema": "tk_hao_direct_pipeline_v1",
        "pv_format": "e4m3_fp8",
        "shiftless_fp8_mode": 0,
        "causal_interleaved_kv": False,
        "fixed_route_fastpath": True,
        "fixed_p_ceiling": False,
        "score_pack_ceiling": False,
        "query_stages": 2,
        "p_first_chunks": 2,
        "ex2_emu_mask": 12696,
        "retain_q0": 1,
        "retain_q1": 1,
        "retain_q2_mode": 1,
        "retain_q3": 1,
        "p_ilp_pipeline": False,
        "early_q3": False,
        "two_chunk_pipeline": False,
        "cubic_quarter_mask": 15,
        "affine_a": 1.623300313949585,
        "affine_b": 0.9208354353904724,
    },
    (
        "real_fwd_tk_hao_direct_nvfp4_mxfp4pv",
        "mxfp4_e8m0_block32",
    ): {
        "route": "real_fwd_tk_hao_direct_nvfp4_mxfp4pv",
        "schema": "tk_hao_direct_pipeline_v1",
        "pv_format": "mxfp4_e8m0_block32",
        "causal_interleaved_kv": False,
        "fixed_route_fastpath": True,
        "mx_scale_select": 4,
        "mx_log2_p_quant": True,
        "mx_quantized_denom": True,
        "mx_p_effective_max": 6,
        "mx_pwl_exp2": True,
        "mx_pwl_exp2_mode": 23,
        "mx_global_anchor32": True,
        "mx_global_anchor128": False,
        "mx_global_anchor_margin_log2": 64,
        "mx_anchor_affine_hoist": False,
        "mx_stored_scale_shift_log2": 32,
        "nv_qk_folded_k64_scales": False,
        "nv_qk_folded_k64_scale_mask": 0,
        "nv_qk_compact_folded_scales": False,
        "nv_qk_preload_page_mask": 3,
        "fixed_p_ceiling": False,
        "score_pack_ceiling": False,
        "rowmax_pack_ceiling": False,
        "query_stages": 2,
        "p_first_chunks": 1,
        "early_p_k64": True,
        "kv_reuse_ceiling": False,
        "sparse_pv": False,
        "sparse_pattern": "dense",
        "sparse_selector_mode": 0,
        "sparse_fixed_selector": 0,
        "sparse_split_k128": False,
        "sparse_issue_count": 0,
        "sparse_logical_k_per_issue": 0,
        "sparse_early_half": False,
        "fused_pv_issue": False,
        "log2_p_quant": False,
        "fake_denom_ceiling": False,
        "p_ilp_pipeline": False,
        "scale_copy_ceiling_mask": 0,
        "task_order": 0,
        "ex2_emu_mask": 34952,
        "ex2_alu_degree": 3,
        "mx_shiftless_corr_bypass": True,
        "mx_shiftless_softmax": True,
        "mx_skip_zero_scale_mask": True,
        "mx_q3_corr_wg": False,
        "mx_dual_q3_corr_wg": False,
        "mx_dual_q3_smem_wg": False,
        "mx_dual_q3_tmem_wg": False,
        "mx_direct_log_code": False,
        "mx_direct_log_code_mode": 0,
        "mx_log_lut_bits": 0,
        "mx_log_lut_bit_code_mode": 0,
        "mx_denom_corr_wg": False,
        "mx_denom_load_width": 4,
        "mx_denom_pipelined_sum": False,
        "mx_denom_pair_overlap": False,
        "mx_local_denom_pipeline": 2,
        "mx_defer_denom_finalize": 2,
        "mx_denom_factor_hoist": False,
        "mx_denom_int_scale_fusion": False,
        "mx_denom_sum_lanes": 1,
        "mx_denom_words": 4,
        "mx_denom_known_max": False,
        "mx_denom_sample_layout": 0,
        "mx_denom_decode_mode": 1,
        "mx_denom_scale_smem": False,
        "mx_denom_payload_smem": False,
        "mx_denom_magic_i2f": False,
        "mx_denom_early_smem": False,
        "mx_denom_early_split": False,
        "mx_denom_split_prefetch": False,
        "mx_denom_sample_stride": 1,
        "mx_global_anchor_bias": 44.361419677734375,
        "mx_fixed_anchor_log2": 0,
        "mx_max_sample_stride": 1,
        "mx_max_sample_bias_x2": 0,
        "mx_q1_max_sample_stride": 1,
        "mx_pair_scale_reuse": 1,
        "mx_pair_scale_stage_mask": 1,
        "mx_q3_scale_reuse_stage_mask": 0,
        "mx_pair_scale_period": 1,
        "mx_pair_scale_delta": 0,
        "mx_q1_pair_probe": 0,
        "mx_q1_pair_probe_bias": 0.0,
        "mx_lagged_q1_scale": False,
        "mx_q1_scale_offload": False,
        "mx_q1_self_max": 3,
        "mx_scan_fused_p": False,
        "mx_local_native_deferred_scale": False,
        "mx_bit_exp2_mask": 0,
        "mx_mode23_native_density": 3,
        "mx_mode23_native_quarter_mask": 3,
        "mx_mode23_native_density2_quarter_mask": 3,
        "mx_mode23_native_density3_quarter_mask": 1,
        "mx_mode23_native_density3_stage_mask": 3,
        "mx_mode23_native_stage_mask": 3,
        "mx_mode23_self_stage0_native": True,
        "mx_mode23_early_native": False,
        "mx_mode23_early_native_stage_mask": 3,
        "mx_mode23_early_native_quarter_mask": 15,
        "mx_mode23_early_native_lookahead": 4,
        "mx_mode23_early_native_order": 0,
        "mx_pwl_four_sample_denom": False,
        "mx_full_approx_denom": False,
        "mx_full_approx_denom_mode": 0,
        "mx_affine_a": 1.5,
        "mx_affine_b": 1.2000000476837158,
        "mx_cubic_a": 0.07430709153413773,
        "mx_cubic_b": 0.28611862659454346,
        "mx_cubic_c": 0.6467000246047974,
        "mx_cubic_d": 0.9901078343391418,
        "mx_stage0_affine_mask": 0,
        "mx_stage1_affine_mask": 0,
        "mx_sampled_stabilizer": False,
        "mx_retain_q0": False,
        "mx_pair_load_scan": True,
        "mx_half_prefetch_q0": False,
        "mx_half_prefetch_q1": False,
        "mx_half_prefetch_q3": False,
        "mx_delayed_half_q2": True,
        "mx_delayed_early_q3": True,
        "mx_causal_q3_lookahead": 1,
        "mx_causal_q3_progressive_reuse": True,
        "mx_early_q2_reduce": True,
        "mx_stage1_early_q1": True,
        "mx_q1_max_interleave": True,
        "mx_stage0_q1_max_interleave": True,
        "mx_q2_max_interleave": True,
        "mx_q3_max_interleave": True,
        "mx_tree_max": True,
        "mx_max3_reduce": True,
        "mx_max3_wide_reduce": True,
        "mx_ex2_q1_mask": 34952,
        "mx_ex2_q2_mask": 34952,
        "mx_ex2_q3_mask": 0,
    },
}
D128_MX_TOPOLOGY_KEY = (
    "real_fwd_tk_hao_direct_nvfp4_mxfp4pv",
    "mxfp4_e8m0_block32",
)
D128_MX_FORWARD_TOPOLOGY_VARIANTS = {
    "production": D128_EXACT_FORWARD_TOPOLOGIES[D128_MX_TOPOLOGY_KEY],
    "anchor128-m64": {
        **D128_EXACT_FORWARD_TOPOLOGIES[D128_MX_TOPOLOGY_KEY],
        "mx_global_anchor32": False,
        "mx_global_anchor128": True,
    },
    "anchor128-m64-dual-lse": {
        **D128_EXACT_FORWARD_TOPOLOGIES[D128_MX_TOPOLOGY_KEY],
        "mx_global_anchor32": False,
        "mx_global_anchor128": True,
        "mx_causal_q3_progressive_reuse": False,
        "mx_full_approx_denom": True,
        "mx_full_approx_denom_mode": 1,
        "mx_dual_lse_denom": True,
    },
    "anchor128-m64-stable-represented-logsum": {
        **D128_EXACT_FORWARD_TOPOLOGIES[D128_MX_TOPOLOGY_KEY],
        "mx_global_anchor32": False,
        "mx_global_anchor128": True,
        "mx_stable_lse_logsum": True,
        "mx_alternate_lse_stat": True,
    },
    "anchor128-m64-stable-full-approx-lse": {
        **D128_EXACT_FORWARD_TOPOLOGIES[D128_MX_TOPOLOGY_KEY],
        "mx_global_anchor32": False,
        "mx_global_anchor128": True,
        "mx_causal_q3_progressive_reuse": False,
        "mx_full_approx_denom": True,
        "mx_full_approx_denom_mode": 1,
        "mx_dual_lse_denom": True,
        "mx_stable_lse_logsum": True,
        "mx_alternate_lse_stat": True,
    },
    "anchor128-m0": {
        **D128_EXACT_FORWARD_TOPOLOGIES[D128_MX_TOPOLOGY_KEY],
        "mx_global_anchor32": False,
        "mx_global_anchor128": True,
        "mx_global_anchor_bias": 0.0,
        "mx_global_anchor_margin_log2": 0,
    },
}
D128_FORWARD_TOPOLOGY_VARIANTS = tuple(
    D128_MX_FORWARD_TOPOLOGY_VARIANTS
)
DIAGNOSTIC_FP8_LSE_SUBSTITUTION_MODES = (
    "all_rows",
    "mx_nonfinite_only",
)


def _d128_forward_topology_recipe(
    config: Config,
    route_key: tuple[object, object],
) -> dict[str, object] | None:
    """Resolve one exact D128 recipe without broadening production auth."""
    if route_key == D128_MX_TOPOLOGY_KEY:
        recipe = D128_MX_FORWARD_TOPOLOGY_VARIANTS.get(
            getattr(config, "d128_forward_topology_variant", "production")
        )
        if recipe is None:
            raise ValueError(
                "unknown D128 MX forward topology variant "
                f"{getattr(config, 'd128_forward_topology_variant', None)!r}"
            )
        return recipe
    return D128_EXACT_FORWARD_TOPOLOGIES.get(route_key)
DEFAULT_BF16_ATTENTION_CONTROL = "split_qkv_three_linear"
BF16_ATTENTION_CONTROLS = (
    DEFAULT_BF16_ATTENTION_CONTROL,
    "packed_qkv_single_linear",
)
BF16_ATTENTION_ROUTES = {
    DEFAULT_BF16_ATTENTION_CONTROL: "bf16_cute",
    "packed_qkv_single_linear": "bf16_cute_packed_qkv_single_linear",
}


def config_from_model_preset(
    model_preset: str = DEFAULT_MODEL_PRESET,
    *,
    batch: int = 1,
    sequence: int = 4096,
    layers: int | None = None,
    d128_forward_topology_variant: str = "production",
) -> Config:
    """Build one audited Llama configuration with an optional smoke depth."""
    if model_preset == DEFAULT_MODEL_PRESET:
        preset = {
            "hidden": 2048,
            "intermediate": 8192,
            "full_model_layers": 16,
            "q_heads": 32,
            "kv_heads": 8,
            "head_dim": 64,
            "vocab": 128256,
            "tie_word_embeddings": True,
            "rope_factor": 32.0,
        }
    elif model_preset == "llama3.1-8b":
        preset = {
            "hidden": 4096,
            "intermediate": 14336,
            "full_model_layers": 32,
            "q_heads": 32,
            "kv_heads": 8,
            "head_dim": 128,
            "vocab": 128256,
            "tie_word_embeddings": False,
            "rope_factor": 8.0,
        }
    else:
        raise ValueError(
            f"unknown model preset {model_preset!r}; expected one of "
            f"{MODEL_PRESETS}"
        )
    # Keep this literal self-contained for the CPU-only preset extractor.
    if batch not in (1, 2, 4, 8, 16):
        raise ValueError("batch must be one of (1, 2, 4, 8, 16)")
    if model_preset == "llama3.1-8b" and batch not in (1, 2, 4):
        raise ValueError(
            "D128 model preset batch must be one of (1, 2, 4)"
        )
    if model_preset == DEFAULT_MODEL_PRESET and batch not in (1, 2, 8, 16):
        raise ValueError(
            "D64 model preset batch must be one of (1, 2, 8, 16)"
        )
    # Keep the default path self-contained for the CPU-only preset extractor.
    if (
        d128_forward_topology_variant != "production"
        and d128_forward_topology_variant
        not in D128_FORWARD_TOPOLOGY_VARIANTS
    ):
        raise ValueError(
            "unknown D128 forward topology variant "
            f"{d128_forward_topology_variant!r}"
        )
    if (
        model_preset != "llama3.1-8b"
        and d128_forward_topology_variant != "production"
    ):
        raise ValueError(
            "non-production D128 forward topology variants require the "
            "llama3.1-8b preset"
        )
    resolved_layers = (
        int(preset["full_model_layers"]) if layers is None else layers
    )
    if sequence <= 0 or sequence > 131072:
        raise ValueError("sequence must be in [1, 131072]")
    if (
        resolved_layers <= 0
        or resolved_layers > int(preset["full_model_layers"])
    ):
        raise ValueError(
            f"layers must be in [1, {preset['full_model_layers']}] for "
            f"{model_preset}"
        )
    return Config(
        model_preset=model_preset,
        batch=batch,
        sequence=sequence,
        layers=resolved_layers,
        d128_forward_topology_variant=d128_forward_topology_variant,
        **preset,
    )


def _make_llama3_rope(
    config: Config,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct the model preset's Llama-3 scaled RoPE tables."""
    pair_count = config.head_dim // 2
    positions = torch.arange(
        config.sequence,
        device="cuda",
        dtype=torch.float32,
    )
    frequencies = 1.0 / (
        config.rope_theta
        ** (
            torch.arange(
                pair_count,
                device="cuda",
                dtype=torch.float32,
            )
            / pair_count
        )
    )
    wavelengths = 2.0 * math.pi / frequencies
    low_frequency_wavelength = (
        config.rope_original_context / config.rope_low_frequency_factor
    )
    high_frequency_wavelength = (
        config.rope_original_context / config.rope_high_frequency_factor
    )
    scaled_frequencies = torch.where(
        wavelengths > low_frequency_wavelength,
        frequencies / config.rope_factor,
        frequencies,
    )
    smooth = (
        config.rope_original_context / wavelengths
        - config.rope_low_frequency_factor
    ) / (
        config.rope_high_frequency_factor
        - config.rope_low_frequency_factor
    )
    smoothed_frequencies = (
        (1.0 - smooth) * scaled_frequencies / config.rope_factor
        + smooth * scaled_frequencies
    )
    medium = ~(
        (wavelengths < high_frequency_wavelength)
        | (wavelengths > low_frequency_wavelength)
    )
    frequencies = torch.where(
        medium,
        smoothed_frequencies,
        scaled_frequencies,
    )
    angles = positions[:, None] * frequencies[None, :]
    cosine = (
        angles.cos()[None]
        .repeat(config.batch, 1, 1)
        .bfloat16()
        .contiguous()
    )
    sine = (
        angles.sin()[None]
        .repeat(config.batch, 1, 1)
        .bfloat16()
        .contiguous()
    )
    return cosine, sine


def _new_weight(rows: int, cols: int, *, std: float = 0.02) -> nn.Parameter:
    value = torch.empty(rows, cols, device="cuda", dtype=torch.bfloat16)
    torch.nn.init.normal_(value, mean=0.0, std=std)
    return nn.Parameter(value)


class RMSNorm(nn.Module):
    def __init__(self, hidden: int, epsilon: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(hidden, device="cuda", dtype=torch.bfloat16)
        )
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().square().mean(dim=-1, keepdim=True)
        normalized = x.float() * torch.rsqrt(variance + self.epsilon)
        return (normalized * self.weight.float()).bfloat16()


def _apply_pair_rope(
    tensor: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    depth = tensor.shape[-1]
    pairs = tensor.float().reshape(*tensor.shape[:-1], depth // 2, 2)
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


def _pair_interleave_d128_rotary_features(
    tensor: torch.Tensor,
) -> torch.Tensor:
    """Convert split-half D128 rotary features to adjacent physical pairs."""
    if tensor.shape[-1] != 128:
        raise ValueError("D128 rotary interleave requires depth 128")
    return torch.stack(
        (tensor[..., :64], tensor[..., 64:]),
        dim=-1,
    ).reshape_as(tensor).contiguous()


def _stack_lowp_qkv_weights(
    config: Config,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
) -> torch.Tensor:
    """Publish the physical QKV row order consumed by each projection."""
    if config.head_dim == 128:
        q_physical, k_physical = (
            b300_pair_interleave_gqa_d128_qk_projection_weights(
                q_weight,
                k_weight,
            )
        )
        return b300_stack_gqa_d128_qkv_projection_weights(
            q_physical,
            k_physical,
            v_weight,
        )
    return torch.cat((q_weight, k_weight, v_weight), dim=0).contiguous()


def _uses_direct_d128_dual_qkv_weight_prep(
    runtime: LowpAttentionRuntime,
) -> bool:
    """Select only the true-2D native-NVFP4 D128 dgrad route."""
    return bool(
        runtime.config.head_dim == 128
        and runtime.qkv_projection_format == "nvfp4"
        and runtime.projection_weight_scale_2d
        and runtime.projection_dgrad == "nvfp4"
    )


def _uses_direct_dual_output_weight_prep(
    runtime: LowpAttentionRuntime,
) -> bool:
    """Select dual NVFP4 O weights only when forward O also consumes NVFP4."""
    return bool(
        runtime.output_projection_format == "nvfp4"
        and runtime.projection_weight_scale_2d
    )


def _deinterleave_d128_weight_gradient(
    gradient: torch.Tensor,
    heads: int,
) -> torch.Tensor:
    """Map adjacent-pair physical rows back to split-half model rows."""
    if gradient.ndim != 2 or gradient.shape[0] != heads * 128:
        raise ValueError("D128 weight gradient has an invalid shape")
    hidden = gradient.shape[1]
    paired = gradient.reshape(heads, 64, 2, hidden)
    return torch.cat(
        (paired[:, :, 0], paired[:, :, 1]),
        dim=1,
    ).reshape_as(gradient).contiguous()


class AttentionWeights(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.q = _new_weight(config.q_width, config.hidden)
        self.k = _new_weight(config.kv_width, config.hidden)
        self.v = _new_weight(config.kv_width, config.hidden)
        self.o = _new_weight(config.hidden, config.q_width)


class BF16Attention(nn.Module):
    def __init__(
        self,
        config: Config,
        rope: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        super().__init__()
        self.config = config
        self.rope = rope
        self.weights = AttentionWeights(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self.config
        rows = x.reshape(c.batch * c.sequence, c.hidden)
        q = F.linear(rows, self.weights.q).reshape(
            c.batch, c.sequence, c.q_heads, c.head_dim
        )
        k = F.linear(rows, self.weights.k).reshape(
            c.batch, c.sequence, c.kv_heads, c.head_dim
        )
        v = F.linear(rows, self.weights.v).reshape(
            c.batch, c.sequence, c.kv_heads, c.head_dim
        )
        if c.head_dim == 128:
            # The fused D128 projection converts standard Llama split-half
            # rotary rows to adjacent physical pairs before applying RoPE.
            # Mirror that permutation in the BF16 control so matched weights
            # describe the same attention function.
            q = _pair_interleave_d128_rotary_features(q)
            k = _pair_interleave_d128_rotary_features(k)
        q = _apply_pair_rope(q, *self.rope)
        k = _apply_pair_rope(k, *self.rope)
        output = flash_attn_func(q, k, v, causal=True)
        if isinstance(output, tuple):
            output = output[0]
        return F.linear(
            output.reshape(c.batch * c.sequence, c.q_width),
            self.weights.o,
        ).reshape_as(x)


def packed_qkv_layout(config: Config) -> PackedQKVLayout:
    """Return the canonical packed projection layout for one model preset."""
    return PackedQKVLayout(
        hidden=config.hidden,
        q_heads=config.q_heads,
        kv_heads=config.kv_heads,
        head_dim=config.head_dim,
    )


PACKED_D64_LOWP_QKV_LAYOUT = "canonical_packed_qkv_parameter"
SPLIT_D128_LOWP_QKV_LAYOUT = "canonical_split_qkv_parameters"


def lowp_qkv_master_parameter_layout(config: Config) -> str:
    """Select the learned QKV schema without changing D128 row semantics."""
    if config.head_dim == 64:
        return PACKED_D64_LOWP_QKV_LAYOUT
    if config.head_dim == 128:
        return SPLIT_D128_LOWP_QKV_LAYOUT
    raise ValueError(
        "low-precision QKV parameter layout requires head_dim 64 or 128"
    )


class PackedQKVBF16Attention(nn.Module):
    """BF16 CuTe FA4 control with one packed QKV projection GEMM."""

    def __init__(
        self,
        config: Config,
        rope: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        super().__init__()
        self.config = config
        self.rope = rope
        self.layout = packed_qkv_layout(config)
        self.weights = PackedQKVAttentionWeights(
            self.layout,
            device="cuda",
            dtype=torch.bfloat16,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self.config
        rows = x.reshape(c.batch * c.sequence, c.hidden)
        q, k, v = project_packed_qkv(
            rows,
            self.weights.qkv,
            self.layout,
            batch=c.batch,
            sequence=c.sequence,
        )
        if c.head_dim == 128:
            # Keep the exact physical-pair convention used by the historical
            # BF16 control and the fused D128 low-precision projection.
            q = _pair_interleave_d128_rotary_features(q)
            k = _pair_interleave_d128_rotary_features(k)
        q = _apply_pair_rope(q, *self.rope)
        k = _apply_pair_rope(k, *self.rope)
        output = flash_attn_func(q, k, v, causal=True)
        if isinstance(output, tuple):
            output = output[0]
        return F.linear(
            output.reshape(c.batch * c.sequence, c.q_width),
            self.weights.o,
        ).reshape_as(x)


def _require_forward_topology(
    config: Config,
    topology: dict[str, Any],
    *,
    runtime_populated: bool = False,
) -> None:
    """Authenticate the fixed forward artifact against the model shape."""
    expected = {
        "batch": config.batch,
        "seqlen": config.sequence,
        "heads": config.q_heads,
        "kv_heads": config.kv_heads,
        "dqk": config.head_dim,
        "dvo": config.head_dim,
        "causal": True,
        "qk_format": "nvfp4_e4m3_block16",
    }
    for key, value in expected.items():
        actual = topology.get(key)
        if actual != value:
            raise ValueError(
                f"forward topology {key}={actual!r} does not match {value!r}"
            )
    if (
        config.head_dim == 64
        and config.batch in AUTHENTICATED_D64_EXACT_BATCHES
    ):
        route_key = (topology.get("route"), topology.get("pv_format"))
        authenticated_routes = {
            (
                "real_fwd_tk_hao_direct_causal_gqa_nvfp4_fp8pv",
                "e4m3_fp8",
            ): {
                "route": (
                    "real_fwd_tk_hao_direct_causal_gqa_nvfp4_fp8pv"
                ),
                "schema": "tk_hao_direct_pipeline_v1",
                "pv_format": "e4m3_fp8",
                "shiftless_fp8_mode": 0,
                "causal_interleaved_kv": False,
                "fixed_route_fastpath": True,
                "fixed_p_ceiling": False,
                "score_pack_ceiling": False,
            },
            (
                "real_fwd_tk_hao_direct_nvfp4_mxfp4pv",
                "mxfp4_e8m0_block32",
            ): {
                "route": "real_fwd_tk_hao_direct_nvfp4_mxfp4pv",
                "schema": "tk_hao_direct_pipeline_v1",
                "pv_format": "mxfp4_e8m0_block32",
                "causal_interleaved_kv": True,
                "fixed_route_fastpath": True,
                "fixed_p_ceiling": False,
                "score_pack_ceiling": False,
                "d64_detached_p": True,
                "mx_mode23_native_density": 4,
                "mx_mode23_native_quarter_mask": 3,
                "mx_global_anchor32": True,
                "mx_anchor_affine_hoist": True,
                "mx_global_anchor_margin_log2": 64,
                "mx_stored_scale_shift_log2": 16,
            },
        }
        exact_batched = authenticated_routes.get(route_key)
        if exact_batched is None:
            raise ValueError(
                "batched exact forward topology route/PV pair "
                f"{route_key!r} is not authenticated"
            )
        if runtime_populated:
            exact_batched["valid"] = 1
        for key, value in exact_batched.items():
            actual = topology.get(key)
            if actual != value:
                raise ValueError(
                    "batched exact forward topology "
                    f"{key}={actual!r} does not match {value!r}"
                )
        if not runtime_populated and topology.get("valid") not in (0, 1):
            raise ValueError(
                "batched exact forward topology must expose a zero- or "
                "one-valued validity field before its first launch"
            )
    elif config.head_dim == 128 and config.batch in (
        1,
        *AUTHENTICATED_D128_EXACT_BATCHES,
    ):
        route_key = (topology.get("route"), topology.get("pv_format"))
        exact_d128 = _d128_forward_topology_recipe(config, route_key)
        if exact_d128 is None:
            raise ValueError(
                "D128 exact forward topology route/PV pair "
                f"{route_key!r} is not authenticated"
            )
        if route_key == D128_MX_TOPOLOGY_KEY:
            variant = getattr(
                config,
                "d128_forward_topology_variant",
                "production",
            )
            expected_lse_semantics = {
                "mx_dual_lse_denom": variant in {
                    "anchor128-m64-dual-lse",
                    "anchor128-m64-stable-full-approx-lse",
                },
                "mx_stable_lse_logsum": variant in {
                    "anchor128-m64-stable-represented-logsum",
                    "anchor128-m64-stable-full-approx-lse",
                },
            }
            for key, value in expected_lse_semantics.items():
                # Legacy nonstable artifacts predate these explicit topology
                # fields; absence is equivalent to false.  A stable/dual
                # binary must never authenticate under a plain recipe.
                actual = topology.get(key, False)
                if actual != value or type(actual) is not bool:
                    raise ValueError(
                        f"D128 B{config.batch} exact forward topology "
                        f"{key}={actual!r} does not match {value!r}"
                    )
        if runtime_populated:
            exact_d128 = {**exact_d128, "valid": 1}
        for key, value in exact_d128.items():
            actual = topology.get(key)
            if actual != value or type(actual) is not type(value):
                raise ValueError(
                    f"D128 B{config.batch} exact forward topology "
                    f"{key}={actual!r} does not match {value!r}"
                )
        if not runtime_populated and topology.get("valid") not in (0, 1):
            raise ValueError(
                f"D128 B{config.batch} exact forward topology must expose a "
                "zero- or one-valued validity field before its first launch"
            )


def _require_precomposed_backward_control(
    config: Config,
    source: Path | str | None,
    sha256: str | None,
    byte_count: int | None,
    *,
    native_tk_d64_backward: bool = False,
    native_tk_d128_backward: bool = False,
) -> dict[str, int | str] | None:
    """Authenticate batched control before any benchmark route allocates."""
    supplied = (source is not None, sha256 is not None, byte_count is not None)
    if native_tk_d64_backward or native_tk_d128_backward:
        if any(supplied):
            raise ValueError(
                "native TK backward does not accept CuTe precomposed "
                "control source, SHA-256, or byte-count options"
            )
        return None
    if any(supplied) and not all(supplied):
        raise ValueError(
            "precomposed backward control requires source, SHA-256, and "
            "byte count together"
        )
    if not any(supplied):
        if (
            config.head_dim == 64
            and config.batch in AUTHENTICATED_D64_EXACT_BATCHES
        ):
            raise ValueError(
                "batched exact FA4 requires an authenticated precomposed "
                "backward control source, SHA-256, and byte count"
            )
        return None
    assert source is not None and sha256 is not None and byte_count is not None
    identity = _source_content_identity(source)
    if identity["sha256"] != sha256 or identity["bytes"] != byte_count:
        raise ValueError(
            "precomposed backward control identity mismatch: "
            f"observed {identity}, expected SHA-256 {sha256!r} and "
            f"{byte_count} bytes"
        )
    return identity


def _require_authenticated_native_tk_extension(
    extension: Any,
) -> dict[str, int | str]:
    """Require the stable artifact receipt installed by ``_load_extension``."""
    identity = getattr(
        extension,
        "_tk_fa4_loaded_artifact_identity",
        None,
    )
    required_fields = {
        "path",
        "sha256",
        "bytes",
        "device",
        "inode",
        "mtime_ns",
    }
    if not isinstance(identity, dict) or set(identity) != required_fields:
        raise ValueError(
            "native TK backward extension must be loaded through the "
            "authenticated _load_extension path"
        )
    if (
        not isinstance(identity["path"], str)
        or not isinstance(identity["sha256"], str)
        or len(identity["sha256"]) != 64
        or any(
            type(identity[field]) is not int or identity[field] < 0
            for field in ("bytes", "device", "inode", "mtime_ns")
        )
    ):
        raise ValueError("native TK extension artifact receipt is malformed")
    return dict(identity)


def _require_output_projection_contract(
    config: Config,
    *,
    qkv_projection_format: str,
    output_projection_format: str,
    projection_dgrad: str,
    projection_weight_scale_2d: bool,
) -> None:
    """Fail closed around the first dense-E4M3 learned O projection.

    The current E4M3 O GEMM is an allocating correctness canary for the exact
    D64 and D128 production topologies. Its forward operands are rowwise/
    channelwise E4M3, while the already-authenticated O input-gradient
    projection remains NVFP4. Keeping this gate narrow prevents the selector
    from implying an E4M3 learned-projection backward implementation that does
    not yet exist.
    """
    if output_projection_format not in ("nvfp4", "e4m3"):
        raise ValueError("output projection format must be nvfp4 or e4m3")
    if output_projection_format == "e4m3":
        shape = (
            config.sequence,
            config.hidden,
            config.q_heads,
            config.kv_heads,
            config.head_dim,
            config.q_width,
        )
        violations = []
        d64 = bool(
            shape == (4096, 2048, 32, 8, 64, 2048)
            and config.batch in (1, *AUTHENTICATED_D64_EXACT_BATCHES)
            and projection_dgrad == "bf16"
        )
        d128 = bool(
            shape == (4096, 4096, 32, 8, 128, 4096)
            and config.batch in (1, *AUTHENTICATED_D128_EXACT_BATCHES)
            and projection_dgrad == "nvfp4"
        )
        if not (d64 or d128):
            violations.append(
                "authenticated D64/BF16-dgrad or D128/NVFP4-dgrad shape"
            )
        if qkv_projection_format != "e4m3":
            violations.append("dense E4M3 QKV projection")
        if not projection_weight_scale_2d:
            violations.append("true-2D NVFP4 backward-weight scaling")
        if violations:
            raise ValueError(
                "E4M3 output projection is retained only for the exact D64/"
                "D128 "
                "production correctness canary; requires "
                + ", ".join(violations)
            )
    elif config.head_dim in (64, 128) and qkv_projection_format == "e4m3":
        raise ValueError(
            "the dense-E4M3 production route requires an E4M3 forward "
            "output projection"
        )


def _require_native_tk_d128_runtime_contract(
    config: Config,
    forward_topology: dict[str, Any],
    *,
    projection_dgrad: str,
    qkv_projection_format: str,
    experimental_native_nvfp4_projection_out: bool,
    backward_reuse_quantized_p: bool,
    backward_forward_mx_probability_replay: bool,
    backward_forward_mx_probability_scale_handoff: bool | None,
    backward_match_forward_operands: bool,
    per_block_qk_scales: bool,
    experimental_split_v_backward: bool,
    experimental_d128_mxfp4_v_backward: bool,
    backward_probability_correction: float | None,
    q_quant_scale: float,
    k_quant_scale: float,
    projection_weight_scale_2d: bool,
    v_mxfp4_scale_2d: bool,
    adaptive_qk_weight_scales: bool,
    shared_runtime: LowpAttentionRuntime | None,
) -> None:
    """Fail closed around the authenticated D128 native-TK ABIs."""
    shape = (
        config.sequence,
        config.hidden,
        config.q_heads,
        config.kv_heads,
        config.head_dim,
    )
    violations = []
    if config.batch not in (1, *AUTHENTICATED_D128_EXACT_BATCHES):
        violations.append("batch 1 or an authenticated D128 exact batch")
    if shape != (4096, 4096, 32, 8, 128):
        violations.append("S4096/H4096/Hq32/Hkv8/D128")
    pv_format = forward_topology.get("pv_format")
    if pv_format not in ("e4m3_fp8", "mxfp4_e8m0_block32"):
        violations.append("authenticated FP8-PV or max-safe MXFP4-PV forward")
    if pv_format == "e4m3_fp8" and int(
        forward_topology.get("shiftless_fp8_mode", -1)
    ) != 0:
        violations.append("shiftless_fp8_mode=0")
    if bool(forward_topology.get("causal_interleaved_kv", False)):
        violations.append("ordinary causal K/V order")
    if projection_dgrad != "nvfp4":
        violations.append("NVFP4 projection dgrad")
    if qkv_projection_format == "nvfp4":
        if not experimental_native_nvfp4_projection_out:
            violations.append("native NVFP4 caller-owned QKV projection")
    elif qkv_projection_format == "e4m3":
        if experimental_native_nvfp4_projection_out:
            violations.append("dense E4M3 QKV projection without NVFP4 flag")
        if backward_match_forward_operands:
            violations.append("direct-accumulator E4M3 backward Q/K/V")
        if experimental_d128_mxfp4_v_backward:
            violations.append("E4M3 backward V for both D128 PV routes")
    else:
        violations.append("native NVFP4 or dense E4M3 QKV projection")
    if not per_block_qk_scales:
        violations.append(
            "row-by-K16 forward Q/K scales"
        )
    if backward_match_forward_operands and pv_format != "e4m3_fp8":
        violations.append("represented NVFP4 Q/K backward only with FP8-PV")
    if experimental_split_v_backward:
        violations.append("shared projection-accumulator E4M3 V backward")
    if experimental_d128_mxfp4_v_backward:
        if config.batch != 2:
            violations.append("batch 2 for MXFP4 V backward")
        if pv_format != "mxfp4_e8m0_block32":
            violations.append("MXFP4-PV forward for MXFP4 V backward")
    if backward_reuse_quantized_p:
        violations.append("native fresh probability computation")
    if backward_forward_mx_probability_replay:
        violations.append("no MX probability replay")
    if backward_forward_mx_probability_scale_handoff not in (None, False):
        violations.append("no MX probability scale handoff")
    if backward_probability_correction not in (None, 1.0):
        violations.append("unit backward probability correction")
    if q_quant_scale != 2.25 or k_quant_scale != 2.0:
        violations.append("Q/K quantization scales 2.25/2.0")
    if not projection_weight_scale_2d:
        violations.append("true-2D projection-weight scaling")
    if v_mxfp4_scale_2d and not experimental_d128_mxfp4_v_backward:
        violations.append(
            "D32xS32 MXFP4 V scaling only with the shared-tile backward "
            "publication"
        )
    if adaptive_qk_weight_scales:
        violations.append("fixed authenticated Q/K scales")
    if shared_runtime is not None and not getattr(
        shared_runtime,
        "native_tk_d128_backward",
        False,
    ):
        violations.append("one shared native-TK D128 backward runtime")
    if (
        shared_runtime is not None
        and experimental_d128_mxfp4_v_backward
        and v_mxfp4_scale_2d
    ):
        violations.append(
            "one producer-owned shared-tile runtime without cross-runtime "
            "backward sharing"
        )
    if violations:
        raise ValueError(
            "native TK D128 backward is retained only for the exact "
            "production route; requires " + ", ".join(violations)
        )


def _require_native_tk_d128_v509_e5m2_dout_runtime_contract(
    config: Config,
    forward_topology: dict[str, Any],
    *,
    qkv_projection_format: str,
    output_projection_format: str,
    experimental_native_nvfp4_projection_out: bool,
    native_tk_d128_native_score_backward: bool,
    backward_match_forward_operands: bool,
    per_block_qk_scales: bool,
    experimental_split_v_backward: bool,
    experimental_output_shared_split_v: bool | None,
    experimental_d128_mxfp4_v_backward: bool,
    v_mxfp4_scale_2d: bool,
    shared_runtime: LowpAttentionRuntime | None,
) -> None:
    """Fail closed around native-score v509 with fused E5M2 dO.

    Both learned-projection families publish the same exact native-score
    workspace and retained E4M3 Q/K/V gradient operands.  The score path is
    therefore invariant across the two learned-projection formats and the two
    authenticated forward PV formats.  No represented-forward-QK or MXFP4-V
    backward ABI is admitted here.
    """
    shape = (
        config.batch,
        config.sequence,
        config.hidden,
        config.q_heads,
        config.kv_heads,
        config.head_dim,
    )
    violations = []
    # Keep this literal self-contained for the CPU-only contract extractor.
    if shape[0] not in (1, 2, 4) or shape[1:] != (
        4096,
        4096,
        32,
        8,
        128,
    ):
        violations.append("B1/B2/B4 S4096/H4096/Hq32/Hkv8/D128")
    if forward_topology.get("pv_format") not in (
        "e4m3_fp8",
        "mxfp4_e8m0_block32",
    ):
        violations.append("FP8-PV or max-safe MXFP4-PV")
    if (qkv_projection_format, output_projection_format) not in (
        ("e4m3", "e4m3"),
        ("nvfp4", "nvfp4"),
    ):
        violations.append("matched E4M3 or NVFP4 learned projections")
    if qkv_projection_format == "nvfp4":
        if not experimental_native_nvfp4_projection_out:
            violations.append("native NVFP4 caller-owned QKV publication")
    elif experimental_native_nvfp4_projection_out:
        violations.append("dense E4M3 projection without the NVFP4 flag")
    if not native_tk_d128_native_score_backward:
        violations.append("native NVFP4 score workspace")
    if backward_match_forward_operands:
        violations.append("retained projection-accumulator E4M3 Q/K")
    if not per_block_qk_scales:
        violations.append("forward row-by-K16 Q/K scales")
    if experimental_split_v_backward:
        violations.append("retained projection-accumulator E4M3 V")
    if experimental_output_shared_split_v is not False:
        violations.append("explicit output-shared split-V disable")
    if experimental_d128_mxfp4_v_backward:
        violations.append("no MXFP4 backward V")
    if v_mxfp4_scale_2d:
        violations.append("no MXFP4 V scale ABI")
    if shared_runtime is not None and (
        getattr(
            shared_runtime,
            "native_tk_d128_v509_e5m2_dout_backward",
            None,
        )
        is not True
        or getattr(
            shared_runtime,
            "native_tk_d128_native_score_backward",
            None,
        )
        is not True
    ):
        violations.append("one shared native-score v509 backward runtime")
    if violations:
        raise ValueError(
            "native TK D128 v509 native-score E5M2-dO backward is "
            "fail-closed; requires " + ", ".join(violations)
        )


def _require_d128_e4m3_v501_runtime_contract(
    config: Config,
    forward_topology: dict[str, Any],
    *,
    projection_dgrad: str,
    qkv_projection_format: str,
    backward_exp2_degree: int,
    backward_exp2_period: int | None,
    backward_fp8_ds_lift: int | None,
    backward_reuse_quantized_p: bool,
    backward_forward_mx_probability_replay: bool,
    backward_forward_mx_probability_scale_handoff: bool | None,
    backward_match_forward_operands: bool,
    per_block_qk_scales: bool,
    experimental_split_v_backward: bool,
    experimental_d128_mxfp4_v_backward: bool,
    backward_probability_correction: float | None,
    q_quant_scale: float,
    k_quant_scale: float,
    projection_weight_scale_2d: bool,
    v_mxfp4_scale_2d: bool,
    adaptive_qk_weight_scales: bool,
    shared_runtime: LowpAttentionRuntime | None,
) -> None:
    """Admit the exact D128 E4M3 producer with the retained v501 consumer.

    The v501 kernel owns its EX2 and dS approximation policy, so the three
    corresponding CuTe controls remain provenance-only here. All physical
    producer/consumer contracts are delegated to the native-D128 gate with
    the NVFP4-projection selector explicitly disabled.
    """
    _ = (
        backward_exp2_degree,
        backward_exp2_period,
        backward_fp8_ds_lift,
    )
    if experimental_d128_mxfp4_v_backward:
        raise ValueError(
            "D128 E4M3 v501 requires direct projection-accumulator E4M3 V "
            "backward, not the MXFP4-V v503 experiment"
        )
    _require_native_tk_d128_runtime_contract(
        config,
        forward_topology,
        projection_dgrad=projection_dgrad,
        qkv_projection_format=qkv_projection_format,
        experimental_native_nvfp4_projection_out=False,
        backward_reuse_quantized_p=backward_reuse_quantized_p,
        backward_forward_mx_probability_replay=(
            backward_forward_mx_probability_replay
        ),
        backward_forward_mx_probability_scale_handoff=(
            backward_forward_mx_probability_scale_handoff
        ),
        backward_match_forward_operands=backward_match_forward_operands,
        per_block_qk_scales=per_block_qk_scales,
        experimental_split_v_backward=experimental_split_v_backward,
        experimental_d128_mxfp4_v_backward=False,
        backward_probability_correction=backward_probability_correction,
        q_quant_scale=q_quant_scale,
        k_quant_scale=k_quant_scale,
        projection_weight_scale_2d=projection_weight_scale_2d,
        v_mxfp4_scale_2d=v_mxfp4_scale_2d,
        adaptive_qk_weight_scales=adaptive_qk_weight_scales,
        shared_runtime=shared_runtime,
    )


def _require_native_tk_d64_runtime_contract(
    config: Config,
    forward_topology: dict[str, Any],
    *,
    projection_dgrad: str,
    qkv_projection_format: str,
    backward_reuse_quantized_p: bool,
    backward_forward_mx_probability_replay: bool,
    backward_forward_mx_probability_scale_handoff: bool | None,
    backward_match_forward_operands: bool,
    per_block_qk_scales: bool,
    experimental_split_v_backward: bool,
    backward_probability_correction: float | None,
    q_quant_scale: float,
    k_quant_scale: float,
    projection_weight_scale_2d: bool,
    v_mxfp4_scale_2d: bool,
    adaptive_qk_weight_scales: bool,
    shared_runtime: LowpAttentionRuntime | None,
) -> None:
    """Fail closed around the production D64 E4M3 native-TK ABI.

    The native kernel owns its probability approximation and scheduling
    policy, so this gate deliberately does not authenticate CuTe EX2, dS-lift,
    or generated-control fields.  Every forward, projection-publication, and
    externally visible numerical policy remains identical to the retained
    exact D64 route.
    """
    required_shape = (
        config.sequence,
        config.hidden,
        config.q_heads,
        config.kv_heads,
        config.head_dim,
    )
    violations = []
    d64_batches = (1, *AUTHENTICATED_D64_EXACT_BATCHES)
    if config.batch not in d64_batches:
        violations.append(f"batch in {d64_batches}")
    if required_shape != (4096, 2048, 32, 8, 64):
        violations.append("S4096/H2048/Hq32/Hkv8/D64")
    pv_format = forward_topology.get("pv_format")
    fp8_pv = pv_format == "e4m3_fp8"
    mx_pv = pv_format == "mxfp4_e8m0_block32"
    if not (fp8_pv or mx_pv):
        violations.append("authenticated E4M3 FP8-PV or MXFP4-PV forward")
    if fp8_pv:
        if int(forward_topology.get("shiftless_fp8_mode", -1)) != 0:
            violations.append("shiftless_fp8_mode=0")
        if bool(forward_topology.get("causal_interleaved_kv", False)):
            violations.append("ordinary causal K/V order")
    if mx_pv and not bool(
        forward_topology.get("causal_interleaved_kv", False)
    ):
        violations.append("interleaved causal MXFP4 K/V order")
    if projection_dgrad != "bf16":
        violations.append("BF16 projection dgrad")
    if qkv_projection_format != "e4m3":
        violations.append("E4M3 QKV projection")
    if not backward_match_forward_operands or not per_block_qk_scales:
        violations.append(
            "projection-native represented E4M3 per-block Q/K publications"
        )
    if fp8_pv and experimental_split_v_backward:
        violations.append("unsplit projection-native E4M3 V backward")
    if mx_pv and not experimental_split_v_backward:
        violations.append(
            "split MX forward / projection-accumulator E4M3 V backward"
        )
    if backward_reuse_quantized_p:
        violations.append("native fresh probability computation")
    if backward_forward_mx_probability_replay:
        violations.append("no MX probability replay")
    if backward_forward_mx_probability_scale_handoff not in (None, False):
        violations.append("no MX probability scale handoff")
    if backward_probability_correction not in (None, 1.0):
        violations.append("unit backward probability correction")
    if q_quant_scale != 2.25 or k_quant_scale != 2.0:
        violations.append("Q/K quantization scales 2.25/2.0")
    if not projection_weight_scale_2d:
        violations.append("true-2D projection-weight scaling")
    if v_mxfp4_scale_2d:
        violations.append("row-wise MXFP4 V scaling")
    if adaptive_qk_weight_scales:
        violations.append("fixed authenticated Q/K scales")
    if shared_runtime is not None and not getattr(
        shared_runtime,
        "native_tk_d64_backward",
        False,
    ):
        violations.append("one shared native-TK backward runtime")
    if violations:
        raise ValueError(
            "native TK D64 backward is retained only for the exact production "
            "route; requires " + ", ".join(violations)
        )


def _require_batched_exact_runtime_contract(
    config: Config,
    forward_topology: dict[str, Any],
    *,
    projection_dgrad: str,
    qkv_projection_format: str,
    backward_exp2_degree: int,
    backward_exp2_period: int | None,
    backward_fp8_ds_lift: int | None,
    backward_reuse_quantized_p: bool,
    backward_forward_mx_probability_replay: bool,
    backward_forward_mx_probability_scale_handoff: bool | None,
    backward_match_forward_operands: bool,
    per_block_qk_scales: bool,
    experimental_split_v_backward: bool,
    backward_probability_correction: float | None,
    q_quant_scale: float,
    k_quant_scale: float,
    projection_weight_scale_2d: bool,
    v_mxfp4_scale_2d: bool,
    adaptive_qk_weight_scales: bool,
    shared_runtime: LowpAttentionRuntime | None,
) -> None:
    """Fail closed outside the locally verified batched exact routes."""
    if config.batch == 1:
        return
    required_shape = (
        config.sequence,
        config.hidden,
        config.q_heads,
        config.kv_heads,
        config.head_dim,
    )
    violations = []
    if config.batch not in AUTHENTICATED_D64_EXACT_BATCHES:
        violations.append(
            f"batch in {AUTHENTICATED_D64_EXACT_BATCHES}"
        )
    if required_shape != (4096, 2048, 32, 8, 64):
        violations.append("S4096/H2048/Hq32/Hkv8/D64")
    pv_format = forward_topology.get("pv_format")
    fp8_pv = pv_format == "e4m3_fp8"
    mx_pv = pv_format == "mxfp4_e8m0_block32"
    if not (fp8_pv or mx_pv):
        violations.append("authenticated E4M3 FP8-PV or MXFP4-PV forward")
    if fp8_pv:
        if int(forward_topology.get("shiftless_fp8_mode", -1)) != 0:
            violations.append("shiftless_fp8_mode=0")
        if bool(forward_topology.get("causal_interleaved_kv", False)):
            violations.append("ordinary causal K/V order")
    if mx_pv and not bool(
        forward_topology.get("causal_interleaved_kv", False)
    ):
        violations.append("interleaved causal MXFP4 K/V order")
    if projection_dgrad != "bf16":
        violations.append("BF16 projection dgrad")
    if qkv_projection_format != "e4m3":
        violations.append("E4M3 QKV projection")
    if not backward_match_forward_operands or not per_block_qk_scales:
        violations.append("represented per-block Q/K backward publications")
    if fp8_pv and experimental_split_v_backward:
        violations.append("unsplit projection-accumulator E4M3 V backward")
    if mx_pv and not experimental_split_v_backward:
        violations.append(
            "split MX forward / projection-accumulator E4M3 V backward"
        )
    if backward_reuse_quantized_p:
        violations.append("fresh aliased-TMEM probability")
    if backward_forward_mx_probability_replay:
        violations.append("no MX probability replay")
    if backward_forward_mx_probability_scale_handoff not in (None, False):
        violations.append("no MX probability scale handoff")
    if backward_fp8_ds_lift != 16:
        violations.append("FP8 dS lift 16")
    if backward_exp2_period != 2 or backward_exp2_degree != 1:
        violations.append("selective EX2 degree 1 / period 2")
    if backward_probability_correction not in (None, 1.0):
        violations.append("unit backward probability correction")
    if q_quant_scale != 2.25 or k_quant_scale != 2.0:
        violations.append("Q/K quantization scales 2.25/2.0")
    if not projection_weight_scale_2d:
        violations.append("true-2D projection-weight scaling")
    if v_mxfp4_scale_2d:
        violations.append("no inactive MXFP4-V 2D policy")
    if adaptive_qk_weight_scales:
        violations.append("fixed authenticated Q/K scales")
    if shared_runtime is not None:
        violations.append("one route-owned backward runtime")
    if violations:
        raise ValueError(
            "batched exact FA4 is retained only for the authenticated "
            "D64 route; requires " + ", ".join(violations)
        )


def _require_experimental_native_batched_runtime_contract(
    config: Config,
    forward_topology: dict[str, Any],
    *,
    projection_dgrad: str,
    qkv_projection_format: str,
    backward_exp2_degree: int,
    backward_exp2_period: int | None,
    backward_fp8_ds_lift: int | None,
    backward_reuse_quantized_p: bool,
    backward_forward_mx_probability_replay: bool,
    backward_forward_mx_probability_scale_handoff: bool | None,
    backward_match_forward_operands: bool,
    per_block_qk_scales: bool,
    experimental_split_v_backward: bool,
    backward_probability_correction: float | None,
    q_quant_scale: float,
    k_quant_scale: float,
    projection_weight_scale_2d: bool,
    v_mxfp4_scale_2d: bool,
    adaptive_qk_weight_scales: bool,
    shared_runtime: LowpAttentionRuntime | None,
) -> None:
    """Fail closed around authenticated native-NVFP4 batched routes."""
    if config.batch == 1:
        return
    if config.head_dim == 128:
        required_shape = (
            config.sequence,
            config.hidden,
            config.q_heads,
            config.kv_heads,
            config.head_dim,
        )
        violations = []
        if config.batch not in AUTHENTICATED_D128_EXACT_BATCHES:
            violations.append(f"batch in {AUTHENTICATED_D128_EXACT_BATCHES}")
        if required_shape != (4096, 4096, 32, 8, 128):
            violations.append("S4096/H4096/Hq32/Hkv8/D128")
        pv_format = forward_topology.get("pv_format")
        if pv_format not in ("e4m3_fp8", "mxfp4_e8m0_block32"):
            violations.append("E4M3 FP8-PV or max-safe MXFP4-PV forward")
        if bool(forward_topology.get("causal_interleaved_kv", False)):
            violations.append("ordinary causal K/V order")
        if projection_dgrad != "nvfp4":
            violations.append("NVFP4 projection dgrad")
        if qkv_projection_format != "nvfp4":
            violations.append("native NVFP4 QKV projection")
        if not per_block_qk_scales:
            violations.append(
                "row-by-K16 forward Q/K scales"
            )
        if backward_match_forward_operands and pv_format != "e4m3_fp8":
            violations.append(
                "represented NVFP4 Q/K backward only with FP8-PV"
            )
        if experimental_split_v_backward:
            violations.append("shared projection-accumulator E4M3 V backward")
        if not backward_reuse_quantized_p:
            violations.append("shared quantized-P backward replay")
        if backward_forward_mx_probability_replay:
            violations.append("no MX probability replay")
        if backward_forward_mx_probability_scale_handoff not in (None, False):
            violations.append("no MX probability scale handoff")
        if backward_fp8_ds_lift != 16:
            violations.append("requested FP8 dS lift 16")
        if backward_exp2_period != 0 or backward_exp2_degree != 1:
            violations.append("selective EX2 degree 1 / period 0")
        if backward_probability_correction not in (None, 1.0):
            violations.append("unit backward probability correction")
        if q_quant_scale != 2.25 or k_quant_scale != 2.0:
            violations.append("Q/K quantization scales 2.25/2.0")
        if not projection_weight_scale_2d:
            violations.append("true-2D projection-weight scaling")
        if v_mxfp4_scale_2d:
            violations.append("row-wise MXFP4 V scaling")
        if adaptive_qk_weight_scales:
            violations.append("fixed authenticated Q/K scales")
        if violations:
            raise ValueError(
                "experimental native NVFP4 D128 B2 FA4 requires "
                + ", ".join(violations)
            )
        return
    required_shape = (
        config.sequence,
        config.hidden,
        config.q_heads,
        config.kv_heads,
        config.head_dim,
    )
    violations = []
    if config.batch != 16:
        violations.append("B16")
    if required_shape != (4096, 2048, 32, 8, 64):
        violations.append("S4096/H2048/Hq32/Hkv8/D64")
    pv_format = forward_topology.get("pv_format")
    fp8_pv = pv_format == "e4m3_fp8"
    mx_pv = pv_format == "mxfp4_e8m0_block32"
    if not (fp8_pv or mx_pv):
        violations.append("authenticated E4M3 FP8-PV or MXFP4-PV forward")
    if fp8_pv:
        if int(forward_topology.get("shiftless_fp8_mode", -1)) != 0:
            violations.append("shiftless_fp8_mode=0")
        if bool(forward_topology.get("causal_interleaved_kv", False)):
            violations.append("ordinary causal K/V order")
    if mx_pv and not bool(
        forward_topology.get("causal_interleaved_kv", False)
    ):
        violations.append("interleaved causal MXFP4 K/V order")
    if projection_dgrad != "bf16":
        violations.append("BF16 projection dgrad")
    if qkv_projection_format != "nvfp4":
        violations.append("native NVFP4 QKV projection")
    if not backward_match_forward_operands or not per_block_qk_scales:
        violations.append("represented per-block Q/K backward publications")
    if fp8_pv and experimental_split_v_backward:
        violations.append("unsplit projection-accumulator E4M3 V backward")
    if mx_pv and not experimental_split_v_backward:
        violations.append(
            "split MX forward / projection-accumulator E4M3 V backward"
        )
    if backward_reuse_quantized_p:
        violations.append("fresh aliased-TMEM probability")
    if backward_forward_mx_probability_replay:
        violations.append("no MX probability replay")
    if backward_forward_mx_probability_scale_handoff not in (None, False):
        violations.append("no MX probability scale handoff")
    if backward_fp8_ds_lift != 16:
        violations.append("FP8 dS lift 16")
    if backward_exp2_period != 2 or backward_exp2_degree != 1:
        violations.append("selective EX2 degree 1 / period 2")
    if backward_probability_correction not in (None, 1.0):
        violations.append("unit backward probability correction")
    if q_quant_scale != 2.25 or k_quant_scale != 2.0:
        violations.append("Q/K quantization scales 2.25/2.0")
    if not projection_weight_scale_2d:
        violations.append("true-2D projection-weight scaling")
    if v_mxfp4_scale_2d:
        violations.append("no inactive MXFP4-V 2D policy")
    if adaptive_qk_weight_scales:
        violations.append("fixed authenticated Q/K scales")
    if violations:
        raise ValueError(
            "experimental native NVFP4 B16 FA4 requires "
            + ", ".join(violations)
        )


def _require_fused_attention_rmsnorm_nvfp4(
    config: Config,
    *,
    enabled: bool,
    qkv_projection_format: str,
    experimental_native_nvfp4_projection_out: bool,
    projection_weight_scale_2d: bool,
) -> None:
    """Fail closed around the first exact-dynamic fused RMSNorm gate."""
    if not enabled:
        return
    fused_shape = (
        config.batch,
        config.sequence,
        config.hidden,
        config.head_dim,
    )
    if (
        fused_shape != (16, 4096, 2048, 64)
        or qkv_projection_format != "nvfp4"
        or not experimental_native_nvfp4_projection_out
        or not projection_weight_scale_2d
    ):
        raise ValueError(
            "experimental fused attention RMSNorm NVFP4 requires B16 S4096 "
            "H2048 D64, native NVFP4 caller-owned QKV publication, and 2D "
            "learned-weight scaling"
        )


def _native_output_shared_v_eligible(
    config: Config,
    *,
    experimental_native_nvfp4_projection_out: bool,
    qkv_projection_format: str,
    publish_mxfp4_v: bool,
    experimental_split_v_backward: bool,
    backward_match_forward_operands: bool,
    per_block_qk_scales: bool,
    v_mxfp4_scale_2d: bool,
) -> bool:
    """Authenticate the two shape-specific output-shared V publishers."""
    d64_policy = bool(
        config.batch == 16
        and config.sequence == 4096
        and config.hidden == 2048
        and config.q_heads == 32
        and config.kv_heads == 8
        and config.head_dim == 64
        and experimental_split_v_backward
        and backward_match_forward_operands
    )
    d128_policy = bool(
        config.batch in (1, 2)
        and config.sequence == 4096
        and config.hidden == 4096
        and config.q_heads == 32
        and config.kv_heads == 8
        and config.head_dim == 128
        and not experimental_split_v_backward
        and not backward_match_forward_operands
    )
    return bool(
        (d64_policy or d128_policy)
        and experimental_native_nvfp4_projection_out
        and qkv_projection_format == "nvfp4"
        and publish_mxfp4_v
        and per_block_qk_scales
        and not v_mxfp4_scale_2d
    )


def _native_d128_mxfp4_v_backward_eligible(
    config: Config,
    *,
    experimental_native_nvfp4_projection_out: bool,
    qkv_projection_format: str,
    publish_mxfp4_v: bool,
    backward_match_forward_operands: bool,
    per_block_qk_scales: bool,
    experimental_split_v_backward: bool,
    v_mxfp4_scale_2d: bool,
) -> bool:
    """Admit the rowwise and shared-tile B2 D128 MXFP4-V ABIs."""
    return bool(
        config.batch in (1, 2)
        and config.sequence == 4096
        and config.hidden == 4096
        and config.q_heads == 32
        and config.kv_heads == 8
        and config.head_dim == 128
        and experimental_native_nvfp4_projection_out
        and qkv_projection_format == "nvfp4"
        and publish_mxfp4_v
        and not backward_match_forward_operands
        and per_block_qk_scales
        and not experimental_split_v_backward
        and type(v_mxfp4_scale_2d) is bool
    )


class LowpAttentionRuntime:
    def __init__(
        self,
        config: Config,
        rope: tuple[torch.Tensor, torch.Tensor],
        *,
        forward_extension: Any,
        forward_topology: dict[str, Any],
        loss_scale: float,
        gradient_global_scale: float,
        projection_dgrad: str,
        qkv_projection_format: str = "nvfp4",
        output_projection_format: str = "nvfp4",
        experimental_native_nvfp4_projection_out: bool = False,
        experimental_fused_attention_rmsnorm_nvfp4: bool = False,
        backward_exp2_degree: int = 2,
        backward_exp2_period: int | None = None,
        backward_fp8_ds_lift: int | None = 16,
        backward_reuse_quantized_p: bool = False,
        backward_control_source: Path | str | None = None,
        backward_control_sha256: str | None = None,
        backward_control_bytes: int | None = None,
        backward_forward_mx_probability_replay: bool = False,
        backward_forward_mx_probability_scale_handoff: bool | None = None,
        backward_match_forward_operands: bool = False,
        per_block_qk_scales: bool = False,
        experimental_split_v_backward: bool = False,
        experimental_output_shared_split_v: bool | None = False,
        experimental_d128_mxfp4_v_backward: bool = False,
        backward_probability_correction: float | None = None,
        q_quant_scale: float = 2.25,
        k_quant_scale: float = 2.0,
        projection_weight_scale_2d: bool = True,
        v_mxfp4_scale_2d: bool = False,
        adaptive_qk_weight_scales: bool = False,
        native_tk_d64_backward_extension: Any | None = None,
        native_tk_d128_backward_extension: Any | None = None,
        native_tk_d128_native_score_backward: bool = False,
        native_tk_d128_v509_e5m2_dout_backward: bool = False,
        shared_backward_runtime: LowpAttentionRuntime | None = None,
    ) -> None:
        if (
            experimental_output_shared_split_v is not None
            and type(experimental_output_shared_split_v) is not bool
        ):
            raise TypeError(
                "experimental_output_shared_split_v must be exactly bool "
                "or None"
            )
        if type(experimental_d128_mxfp4_v_backward) is not bool:
            raise TypeError(
                "experimental_d128_mxfp4_v_backward must be exactly bool"
            )
        if type(native_tk_d128_native_score_backward) is not bool:
            raise TypeError(
                "native_tk_d128_native_score_backward must be exactly bool"
            )
        if type(native_tk_d128_v509_e5m2_dout_backward) is not bool:
            raise TypeError(
                "native_tk_d128_v509_e5m2_dout_backward must be exactly bool"
            )
        if native_tk_d128_v509_e5m2_dout_backward and not (
            native_tk_d128_native_score_backward
        ):
            raise ValueError(
                "v509 E5M2-dO backward requires the native NVFP4-score "
                "selector"
            )
        if (
            native_tk_d128_native_score_backward
            and native_tk_d128_backward_extension is None
        ):
            raise ValueError(
                "native-score D128 backward requires its authenticated "
                "native TK D128 extension"
            )
        if native_tk_d128_native_score_backward and (
            native_tk_d64_backward_extension is not None
            or shared_backward_runtime is not None
        ):
            raise ValueError(
                "native-score D128 backward is a private B1 runner and "
                "cannot use D64 or shared backward state"
            )
        if (
            native_tk_d64_backward_extension is not None
            and native_tk_d128_backward_extension is not None
        ):
            raise ValueError("select exactly one native TK backward extension")
        requested_native_kind = (
            "d64"
            if native_tk_d64_backward_extension is not None
            else "d128"
            if native_tk_d128_backward_extension is not None
            else None
        )
        shared_native_kind = (
            getattr(shared_backward_runtime, "native_tk_backward_kind", None)
            if shared_backward_runtime is not None
            else None
        )
        if shared_native_kind is None and shared_backward_runtime is not None:
            if getattr(shared_backward_runtime, "native_tk_d64_backward", False):
                shared_native_kind = "d64"
            elif getattr(
                shared_backward_runtime,
                "native_tk_d128_backward",
                False,
            ):
                shared_native_kind = "d128"
        if shared_native_kind is not None:
            if shared_native_kind not in ("d64", "d128"):
                raise ValueError(
                    "shared native TK backward has an unknown shape kind"
                )
            shared_native_extension = getattr(
                shared_backward_runtime,
                f"native_tk_{shared_native_kind}_backward_extension",
                None,
            )
            if requested_native_kind is None:
                requested_native_kind = shared_native_kind
                if shared_native_kind == "d64":
                    native_tk_d64_backward_extension = shared_native_extension
                else:
                    native_tk_d128_backward_extension = shared_native_extension
            elif requested_native_kind != shared_native_kind:
                raise ValueError(
                    "shared native TK backward requires the same D64/D128 "
                    "shape specialization"
                )
            requested_extension = (
                native_tk_d64_backward_extension
                if requested_native_kind == "d64"
                else native_tk_d128_backward_extension
            )
            if requested_extension is not shared_native_extension:
                raise ValueError(
                    "shared native TK backward requires the exact same loaded "
                    "extension module object"
                )
        elif requested_native_kind is not None and shared_backward_runtime is not None:
            raise ValueError(
                "a native TK backward cannot share a non-native runner"
            )
        if (
            native_tk_d128_v509_e5m2_dout_backward
            and requested_native_kind != "d128"
        ):
            raise ValueError(
                "v509 E5M2-dO backward requires its authenticated native "
                "TK D128 extension"
            )
        self.native_tk_d64_backward = (
            native_tk_d64_backward_extension is not None
        )
        self.native_tk_d128_backward = (
            native_tk_d128_backward_extension is not None
        )
        self.native_tk_d128_native_score_backward = (
            native_tk_d128_native_score_backward
        )
        self.native_tk_d128_v509_e5m2_dout_backward = (
            native_tk_d128_v509_e5m2_dout_backward
        )
        self.v509_e5m2_dout_route: dict[str, Any] | None = None
        self.native_tk_backward = bool(
            self.native_tk_d64_backward or self.native_tk_d128_backward
        )
        self.native_tk_backward_kind = requested_native_kind
        self.native_tk_d64_backward_extension = (
            native_tk_d64_backward_extension
        )
        self.native_tk_d128_backward_extension = (
            native_tk_d128_backward_extension
        )
        self.native_tk_d64_backward_extension_identity = (
            _require_authenticated_native_tk_extension(
                native_tk_d64_backward_extension
            )
            if self.native_tk_d64_backward
            else None
        )
        self.native_tk_d128_backward_extension_identity = (
            _require_authenticated_native_tk_extension(
                native_tk_d128_backward_extension
            )
            if self.native_tk_d128_backward
            else None
        )
        self.native_tk_backward_extension = (
            native_tk_d64_backward_extension
            if self.native_tk_d64_backward
            else native_tk_d128_backward_extension
        )
        self.native_tk_backward_extension_identity = (
            self.native_tk_d64_backward_extension_identity
            if self.native_tk_d64_backward
            else self.native_tk_d128_backward_extension_identity
        )
        if self.native_tk_backward and any(
            value is not None
            for value in (
                backward_control_source,
                backward_control_sha256,
                backward_control_bytes,
            )
        ):
            raise ValueError(
                "native TK backward does not accept CuTe precomposed "
                "control options"
            )
        _require_forward_topology(config, forward_topology)
        if config.head_dim not in (64, 128):
            raise ValueError("low-precision attention requires D64 or D128")
        is_d128 = config.head_dim == 128
        if (
            is_d128
            and backward_match_forward_operands
            and not self.native_tk_d128_backward
        ):
            raise ValueError(
                "represented D128 Q/K backward is authenticated only with "
                "an authenticated native TK D128 backend"
            )
        _require_output_projection_contract(
            config,
            qkv_projection_format=qkv_projection_format,
            output_projection_format=output_projection_format,
            projection_dgrad=projection_dgrad,
            projection_weight_scale_2d=projection_weight_scale_2d,
        )
        requested_backward_policy = {
            "exp2_degree": int(backward_exp2_degree),
            "exp2_period": backward_exp2_period,
            "fp8_ds_lift": backward_fp8_ds_lift,
            "reuse_quantized_p": bool(backward_reuse_quantized_p),
        }
        if self.native_tk_d64_backward:
            _require_native_tk_d64_runtime_contract(
                config,
                forward_topology,
                projection_dgrad=projection_dgrad,
                qkv_projection_format=qkv_projection_format,
                backward_reuse_quantized_p=backward_reuse_quantized_p,
                backward_forward_mx_probability_replay=(
                    backward_forward_mx_probability_replay
                ),
                backward_forward_mx_probability_scale_handoff=(
                    backward_forward_mx_probability_scale_handoff
                ),
                backward_match_forward_operands=(
                    backward_match_forward_operands
                ),
                per_block_qk_scales=per_block_qk_scales,
                experimental_split_v_backward=(
                    experimental_split_v_backward
                ),
                backward_probability_correction=(
                    backward_probability_correction
                ),
                q_quant_scale=q_quant_scale,
                k_quant_scale=k_quant_scale,
                projection_weight_scale_2d=projection_weight_scale_2d,
                v_mxfp4_scale_2d=v_mxfp4_scale_2d,
                adaptive_qk_weight_scales=adaptive_qk_weight_scales,
                shared_runtime=shared_backward_runtime,
            )
        elif self.native_tk_d128_backward:
            _require_native_tk_d128_runtime_contract(
                config,
                forward_topology,
                projection_dgrad=projection_dgrad,
                qkv_projection_format=qkv_projection_format,
                experimental_native_nvfp4_projection_out=(
                    experimental_native_nvfp4_projection_out
                ),
                backward_reuse_quantized_p=backward_reuse_quantized_p,
                backward_forward_mx_probability_replay=(
                    backward_forward_mx_probability_replay
                ),
                backward_forward_mx_probability_scale_handoff=(
                    backward_forward_mx_probability_scale_handoff
                ),
                backward_match_forward_operands=(
                    backward_match_forward_operands
                ),
                per_block_qk_scales=per_block_qk_scales,
                experimental_split_v_backward=(
                    experimental_split_v_backward
                ),
                experimental_d128_mxfp4_v_backward=(
                    experimental_d128_mxfp4_v_backward
                ),
                backward_probability_correction=(
                    backward_probability_correction
                ),
                q_quant_scale=q_quant_scale,
                k_quant_scale=k_quant_scale,
                projection_weight_scale_2d=projection_weight_scale_2d,
                v_mxfp4_scale_2d=v_mxfp4_scale_2d,
                adaptive_qk_weight_scales=adaptive_qk_weight_scales,
                shared_runtime=shared_backward_runtime,
            )
            if self.native_tk_d128_v509_e5m2_dout_backward:
                _require_native_tk_d128_v509_e5m2_dout_runtime_contract(
                    config,
                    forward_topology,
                    qkv_projection_format=qkv_projection_format,
                    output_projection_format=output_projection_format,
                    experimental_native_nvfp4_projection_out=(
                        experimental_native_nvfp4_projection_out
                    ),
                    native_tk_d128_native_score_backward=(
                        self.native_tk_d128_native_score_backward
                    ),
                    backward_match_forward_operands=(
                        backward_match_forward_operands
                    ),
                    per_block_qk_scales=per_block_qk_scales,
                    experimental_split_v_backward=(
                        experimental_split_v_backward
                    ),
                    experimental_output_shared_split_v=(
                        experimental_output_shared_split_v
                    ),
                    experimental_d128_mxfp4_v_backward=(
                        experimental_d128_mxfp4_v_backward
                    ),
                    v_mxfp4_scale_2d=v_mxfp4_scale_2d,
                    shared_runtime=shared_backward_runtime,
                )
            if (
                self.native_tk_d128_native_score_backward
                and not self.native_tk_d128_v509_e5m2_dout_backward
            ):
                native_score_violations = []
                if config.batch != 1:
                    native_score_violations.append("B1")
                if forward_topology.get("pv_format") != "e4m3_fp8":
                    native_score_violations.append("FP8-PV")
                if qkv_projection_format != "nvfp4":
                    native_score_violations.append("NVFP4 QKV projection")
                if output_projection_format != "nvfp4":
                    native_score_violations.append("NVFP4 O projection")
                if not experimental_native_nvfp4_projection_out:
                    native_score_violations.append(
                        "native NVFP4 caller-owned QKV publication"
                    )
                if not backward_match_forward_operands:
                    native_score_violations.append(
                        "represented-E4 Q/K gradient operands"
                    )
                if not per_block_qk_scales:
                    native_score_violations.append(
                        "forward row-by-K16 Q/K score scales"
                    )
                if experimental_split_v_backward:
                    native_score_violations.append("retained E4M3 V")
                if experimental_d128_mxfp4_v_backward:
                    native_score_violations.append("no MXFP4 backward V")
                if experimental_output_shared_split_v is not False:
                    native_score_violations.append(
                        "explicit output-shared split-V disable"
                    )
                if v_mxfp4_scale_2d:
                    native_score_violations.append("no MXFP4 V scale ABI")
                if shared_backward_runtime is not None:
                    native_score_violations.append(
                        "private non-shared backward runner"
                    )
                if native_score_violations:
                    raise ValueError(
                        "native NVFP4-score D128 backward is fail-closed to "
                        "B1/S4096/H4096/Hq32/Hkv8/D128 represented-QK "
                        "FP8-PV; requires "
                        + ", ".join(native_score_violations)
                    )
        elif is_d128 and qkv_projection_format == "e4m3":
            _require_d128_e4m3_v501_runtime_contract(
                config,
                forward_topology,
                projection_dgrad=projection_dgrad,
                qkv_projection_format=qkv_projection_format,
                backward_exp2_degree=backward_exp2_degree,
                backward_exp2_period=backward_exp2_period,
                backward_fp8_ds_lift=backward_fp8_ds_lift,
                backward_reuse_quantized_p=backward_reuse_quantized_p,
                backward_forward_mx_probability_replay=(
                    backward_forward_mx_probability_replay
                ),
                backward_forward_mx_probability_scale_handoff=(
                    backward_forward_mx_probability_scale_handoff
                ),
                backward_match_forward_operands=(
                    backward_match_forward_operands
                ),
                per_block_qk_scales=per_block_qk_scales,
                experimental_split_v_backward=(
                    experimental_split_v_backward
                ),
                experimental_d128_mxfp4_v_backward=(
                    experimental_d128_mxfp4_v_backward
                ),
                backward_probability_correction=(
                    backward_probability_correction
                ),
                q_quant_scale=q_quant_scale,
                k_quant_scale=k_quant_scale,
                projection_weight_scale_2d=projection_weight_scale_2d,
                v_mxfp4_scale_2d=v_mxfp4_scale_2d,
                adaptive_qk_weight_scales=adaptive_qk_weight_scales,
                shared_runtime=shared_backward_runtime,
            )
        else:
            require_runtime_contract = (
                _require_experimental_native_batched_runtime_contract
                if experimental_native_nvfp4_projection_out
                else _require_batched_exact_runtime_contract
            )
            require_runtime_contract(
                config,
                forward_topology,
                projection_dgrad=projection_dgrad,
                qkv_projection_format=qkv_projection_format,
                backward_exp2_degree=backward_exp2_degree,
                backward_exp2_period=backward_exp2_period,
                backward_fp8_ds_lift=backward_fp8_ds_lift,
                backward_reuse_quantized_p=backward_reuse_quantized_p,
                backward_forward_mx_probability_replay=(
                    backward_forward_mx_probability_replay
                ),
                backward_forward_mx_probability_scale_handoff=(
                    backward_forward_mx_probability_scale_handoff
                ),
                backward_match_forward_operands=(
                    backward_match_forward_operands
                ),
                per_block_qk_scales=per_block_qk_scales,
                experimental_split_v_backward=(
                    experimental_split_v_backward
                ),
                backward_probability_correction=(
                    backward_probability_correction
                ),
                q_quant_scale=q_quant_scale,
                k_quant_scale=k_quant_scale,
                projection_weight_scale_2d=projection_weight_scale_2d,
                v_mxfp4_scale_2d=v_mxfp4_scale_2d,
                adaptive_qk_weight_scales=adaptive_qk_weight_scales,
                shared_runtime=shared_backward_runtime,
            )
        if is_d128:
            if qkv_projection_format not in ("nvfp4", "e4m3"):
                raise ValueError(
                    "D128 requires the native NVFP4 or dense E4M3 QKV "
                    "projection"
                )
            if not projection_weight_scale_2d:
                raise ValueError(
                    "D128 learned projection weights require true 16x16 "
                    "NVFP4 scaling"
                )
            if backward_match_forward_operands and (
                qkv_projection_format != "nvfp4"
                or forward_topology.get("pv_format") != "e4m3_fp8"
                or not per_block_qk_scales
                or not experimental_native_nvfp4_projection_out
            ):
                raise ValueError(
                    "represented D128 Q/K backward requires native caller-owned "
                    "NVFP4 publication, per-row-K16 Q/K scales, and FP8-PV"
                )
            if experimental_split_v_backward:
                raise ValueError("split-V backward is a D64-only policy")
            if backward_forward_mx_probability_replay:
                raise ValueError(
                    "forward MX probability replay is a D64-only policy"
                )
            if backward_forward_mx_probability_scale_handoff:
                raise ValueError(
                    "forward MX probability scale handoff is a D64-only policy"
                )
            if any(
                value is not None
                for value in (
                    backward_control_source,
                    backward_control_sha256,
                    backward_control_bytes,
                )
            ):
                raise ValueError(
                    "D128 requires the generated shared-P control; the "
                    "D64 precomposed direct-TMA control is unsupported"
                )
            if (
                v_mxfp4_scale_2d
                and not experimental_d128_mxfp4_v_backward
            ):
                raise ValueError(
                    "D128 D32xS32 MXFP4 V scales require the explicit "
                    "shared-tile MXFP4 V backward route"
                )
            if self.native_tk_d128_backward:
                backward_exp2_degree = 0
                backward_exp2_period = 0
                backward_fp8_ds_lift = None
                backward_reuse_quantized_p = False
            else:
                # Pin the measured CuTe D128 specialization as one indivisible
                # shared-P policy.
                backward_exp2_degree = 1
                backward_exp2_period = 0
                backward_fp8_ds_lift = 256
                backward_reuse_quantized_p = True
            backward_forward_mx_probability_scale_handoff = False
        if (
            shared_backward_runtime is not None
            and shared_backward_runtime.config != config
        ):
            raise ValueError("shared backward runtime has a different shape")
        self.config = config
        self.is_d128 = is_d128
        effective_native_backend = (
            NATIVE_TK_D64_BACKEND
            if self.native_tk_d64_backward
            else NATIVE_TK_D128_V509_E5M2_DOUT_BACKEND
            if (
                self.native_tk_d128_backward
                and self.native_tk_d128_v509_e5m2_dout_backward
            )
            else NATIVE_TK_D128_NVFP4_SCORE_BACKEND
            if (
                self.native_tk_d128_backward
                and self.native_tk_d128_native_score_backward
            )
            else NATIVE_TK_D128_SHARED_TILE_MX_BACKEND
            if (
                self.native_tk_d128_backward
                and experimental_d128_mxfp4_v_backward
                and v_mxfp4_scale_2d
            )
            else NATIVE_TK_D128_MX_BACKEND
            if (
                self.native_tk_d128_backward
                and experimental_d128_mxfp4_v_backward
            )
            else NATIVE_TK_D128_BACKEND
            if self.native_tk_d128_backward
            else None
        )
        effective_backward_shape_policy = (
            {
                "backend": effective_native_backend,
                "exp2_degree": 0,
                "exp2_period": 0,
                "fp8_ds_lift": None,
                "reuse_quantized_p": False,
                "probability_storage": "native_tk_internal",
                "direct_tma_dkdv": True,
                "lowp_do_stages": None,
                "workspace_stats": True,
            }
            if self.native_tk_backward
            else {
                "exp2_degree": int(backward_exp2_degree),
                "exp2_period": backward_exp2_period,
                "fp8_ds_lift": backward_fp8_ds_lift,
                "reuse_quantized_p": bool(backward_reuse_quantized_p),
                "probability_storage": (
                    "shared_coordinate_preserving_128b"
                    if is_d128
                    else "tmem"
                ),
                "direct_tma_dkdv": not is_d128,
                "lowp_do_stages": 2 if is_d128 else 1,
                "workspace_stats": True,
            }
        )
        self.backward_shape_policy = {
            "shape": f"d{config.head_dim}",
            "requested": requested_backward_policy,
            # The fused dO projection writes negative dPsum/log2-LSE directly
            # into the runner workspace before each launch in both backends.
            "effective": effective_backward_shape_policy,
        }
        self.rope = (
            rope
            if shared_backward_runtime is None
            else shared_backward_runtime.rope
        )
        self.paired_rope = (
            (
                b300_pack_gqa_d128_rope(*rope)
                if is_d128
                else b300_pack_gqa_d64_paired_rope(*rope)
            )
            if shared_backward_runtime is None
            else shared_backward_runtime.paired_rope
        )
        self.loss_scale = float(loss_scale)
        if projection_dgrad not in ("bf16", "nvfp4"):
            raise ValueError("projection_dgrad must be bf16 or nvfp4")
        self.projection_dgrad = projection_dgrad
        self.gradient_global_scale = (
            torch.tensor(
                [gradient_global_scale], device="cuda", dtype=torch.float32
            )
            if shared_backward_runtime is None
            else shared_backward_runtime.gradient_global_scale
        )
        self.gradient_global_scale_value = float(gradient_global_scale)
        # A matched forward crossover may retain two route-specific runtime
        # objects, but aggregate backward must execute through one canonical
        # owner. Saving this owner in autograd closes the maintenance gap
        # where a future backward-affecting runtime field could otherwise be
        # read from the route wrapper even though the physical FA4 runner and
        # its storage are shared.
        # Store ``None`` for the owner rather than a self-reference so an
        # otherwise dead runtime cannot retain its CUDA tensors until cyclic
        # garbage collection happens.
        self._backward_execution_runtime = (
            None
            if shared_backward_runtime is None
            else shared_backward_runtime.backward_execution_runtime
        )
        self.forward_extension = forward_extension
        self.forward_topology = forward_topology
        # A freshly loaded forward extension reports ``valid=0`` until its
        # first launch populates the process-local topology record. D128 B1/B2
        # always reread after this runtime's first launch, even if the module
        # was already populated by an earlier runtime in the same process.
        initial_topology_populated = int(forward_topology.get("valid", 0)) == 1
        exact_d128_forward = bool(
            config.head_dim == 128
            and config.batch in (1, *AUTHENTICATED_D128_EXACT_BATCHES)
        )
        self.forward_topology_runtime_authenticated = bool(
            initial_topology_populated and not exact_d128_forward
        )
        if (
            (
                config.head_dim == 64
                and config.batch in AUTHENTICATED_D64_EXACT_BATCHES
            )
        ) and initial_topology_populated:
            _require_forward_topology(
                config,
                forward_topology,
                runtime_populated=True,
            )
        pv_format = str(forward_topology.get("pv_format", ""))
        self.pv_format = pv_format
        self.publish_mxfp4_v = pv_format == "mxfp4_e8m0_block32"
        self.causal_interleaved_kv = bool(
            forward_topology.get("causal_interleaved_kv", False)
        )
        if is_d128 and self.causal_interleaved_kv:
            raise ValueError("the D128 projection requires ordinary causal K/V order")
        if (
            is_d128
            and per_block_qk_scales
            and pv_format == "mxfp4_e8m0_block32"
        ):
            folded_k64 = bool(
                forward_topology.get("nv_qk_folded_k64_scales", False)
            )
            folded_k64_mask = int(
                forward_topology.get(
                    "nv_qk_folded_k64_scale_mask",
                    3 if folded_k64 else 0,
                )
            )
            compact_folded = bool(
                forward_topology.get("nv_qk_compact_folded_scales", False)
            )
            preload_page_mask = int(
                forward_topology.get("nv_qk_preload_page_mask", 0)
            )
            if (
                folded_k64
                or folded_k64_mask != 0
                or compact_folded
                or preload_page_mask != 3
            ):
                raise ValueError(
                    "D128 per-block Q/K scales require a non-folded MXFP4-PV "
                    "consumer that reads both K64 scale pages"
                )
        if qkv_projection_format not in ("nvfp4", "e4m3"):
            raise ValueError("QKV projection format must be nvfp4 or e4m3")
        self.experimental_native_nvfp4_projection_out = bool(
            experimental_native_nvfp4_projection_out
        )
        self.experimental_fused_attention_rmsnorm_nvfp4 = bool(
            experimental_fused_attention_rmsnorm_nvfp4
        )
        _require_fused_attention_rmsnorm_nvfp4(
            config,
            enabled=self.experimental_fused_attention_rmsnorm_nvfp4,
            qkv_projection_format=qkv_projection_format,
            experimental_native_nvfp4_projection_out=(
                self.experimental_native_nvfp4_projection_out
            ),
            projection_weight_scale_2d=projection_weight_scale_2d,
        )
        if (
            self.experimental_native_nvfp4_projection_out
            and qkv_projection_format != "nvfp4"
        ):
            raise ValueError(
                "experimental native NVFP4 caller-owned publication "
                "requires the native NVFP4 QKV projection"
            )
        if self.experimental_native_nvfp4_projection_out:
            if pv_format not in (
                "e4m3_fp8",
                "mxfp4_e8m0_block32",
            ):
                raise ValueError(
                    "the experimental native NVFP4 projection requires the "
                    "exact FP8-PV or interleaved causal MXFP4-PV route"
                )
            if pv_format == "e4m3_fp8":
                if int(forward_topology.get("shiftless_fp8_mode", -1)) != 0:
                    raise ValueError(
                        "the experimental native NVFP4 projection requires "
                        "exact FP8-PV with shiftless_fp8_mode=0"
                    )
                if bool(
                    forward_topology.get("causal_interleaved_kv", False)
                ):
                    raise ValueError(
                        "the experimental native exact FP8-PV route requires "
                        "normal K/V order"
                    )
            elif is_d128 and bool(
                forward_topology.get("causal_interleaved_kv", False)
            ):
                raise ValueError(
                    "the D128 native MXFP4-PV projection requires ordinary "
                    "causal K/V order"
                )
            elif not is_d128 and not bool(
                forward_topology.get("causal_interleaved_kv", False)
            ):
                raise ValueError(
                    "the experimental native MXFP4-PV route requires "
                    "interleaved causal K/V"
                )
            if is_d128 and experimental_split_v_backward:
                raise ValueError(
                    "split-V backward is a D64-only native projection policy"
                )
            if not is_d128 and (
                not backward_match_forward_operands
                or not per_block_qk_scales
            ):
                raise ValueError(
                    "the experimental native NVFP4 projection requires "
                    "represented per-block Q/K backward publications"
                )
            if not is_d128 and (
                bool(experimental_split_v_backward) != self.publish_mxfp4_v
            ):
                raise ValueError(
                    "the experimental native NVFP4 projection requires "
                    "unsplit FP8 V or split interleaved MXFP4 V backward"
                )
        if qkv_projection_format == "e4m3":
            if pv_format not in (
                "e4m3_fp8",
                "mxfp4_e8m0_block32",
            ):
                raise ValueError(
                    "the E4M3 QKV projection requires the exact FP8-PV or "
                    "interleaved causal MXFP4-PV forward route"
                )
            if pv_format == "e4m3_fp8":
                if int(forward_topology.get("shiftless_fp8_mode", -1)) != 0:
                    raise ValueError(
                        "the E4M3 QKV projection requires the exact FP8-PV "
                        "forward (shiftless_fp8_mode=0)"
                    )
                # The exact-FP8 kernel applies its causal mask in ordinary
                # physical sequence order. Older artifacts omit this field,
                # which therefore means the required non-interleaved layout.
                if (
                    "causal_interleaved_kv" in forward_topology
                    and bool(forward_topology["causal_interleaved_kv"])
                ):
                    raise ValueError(
                        "the exact FP8-PV route requires normal K/V order"
                    )
            elif not is_d128 and not bool(
                forward_topology.get("causal_interleaved_kv", False)
            ):
                raise ValueError(
                    "the paired-D64 E4M3 MXFP4-PV route requires interleaved "
                    "causal K/V"
                )
        self.qkv_projection_format = qkv_projection_format
        self.output_projection_format = output_projection_format
        represented_projection = bool(
            backward_match_forward_operands or
            (per_block_qk_scales and not is_d128)
        )
        if represented_projection and not (
            qkv_projection_format == "e4m3"
            or self.experimental_native_nvfp4_projection_out
        ):
            raise ValueError(
                "matching backward operands to forward codes requires the "
                "projection-native E4M3 QKV path"
            )
        self.backward_match_forward_operands = represented_projection
        self.per_block_qk_scales = bool(per_block_qk_scales)
        self.experimental_split_v_backward = bool(
            experimental_split_v_backward
        )
        if self.experimental_split_v_backward and not (
            pv_format == "mxfp4_e8m0_block32"
            and (
                qkv_projection_format == "e4m3"
                or self.experimental_native_nvfp4_projection_out
            )
            and self.backward_match_forward_operands
            and self.per_block_qk_scales
        ):
            raise ValueError(
                "experimental split-V backward requires the MXFP4-PV E4M3 "
                "projection with represented per-block Q/K operands"
            )
        self.experimental_d128_mxfp4_v_backward = bool(
            experimental_d128_mxfp4_v_backward
        )
        self.d128_mxfp4_v_scale_policy = (
            MXFP4_V_SCALE_POLICY_SHARED_D32XS32
            if (
                self.experimental_d128_mxfp4_v_backward
                and bool(v_mxfp4_scale_2d)
            )
            else MXFP4_V_SCALE_POLICY_ROWWISE_D32
            if self.experimental_d128_mxfp4_v_backward
            else None
        )
        mx_backward_v_eligible = _native_d128_mxfp4_v_backward_eligible(
            config,
            experimental_native_nvfp4_projection_out=(
                self.experimental_native_nvfp4_projection_out
            ),
            qkv_projection_format=qkv_projection_format,
            publish_mxfp4_v=self.publish_mxfp4_v,
            backward_match_forward_operands=(
                self.backward_match_forward_operands
            ),
            per_block_qk_scales=self.per_block_qk_scales,
            experimental_split_v_backward=(
                self.experimental_split_v_backward
            ),
            v_mxfp4_scale_2d=bool(v_mxfp4_scale_2d),
        )
        if (
            self.experimental_d128_mxfp4_v_backward
            and not mx_backward_v_eligible
        ):
            raise ValueError(
                "experimental D128 MXFP4 V backward requires native NVFP4 "
                "projection, MXFP4-PV, row-by-K16 Q/K, a tagged rowwise or "
                "shared D32xS32 V-scale ABI, "
                "and B1/B2 S4096 H4096 Hq32/Hkv8/D128"
            )
        if (
            self.experimental_d128_mxfp4_v_backward
            and experimental_output_shared_split_v is not False
        ):
            raise ValueError(
                "experimental D128 MXFP4 V backward is mutually exclusive "
                "with output-shared dual-V/E4M3-backward publication; pass "
                "experimental_output_shared_split_v=False"
            )
        self.experimental_output_shared_split_v_requested = (
            experimental_output_shared_split_v
        )
        output_shared_eligible = _native_output_shared_v_eligible(
            config,
            experimental_native_nvfp4_projection_out=(
                self.experimental_native_nvfp4_projection_out
            ),
            qkv_projection_format=qkv_projection_format,
            publish_mxfp4_v=self.publish_mxfp4_v,
            experimental_split_v_backward=(
                self.experimental_split_v_backward
            ),
            backward_match_forward_operands=(
                self.backward_match_forward_operands
            ),
            per_block_qk_scales=self.per_block_qk_scales,
            v_mxfp4_scale_2d=bool(v_mxfp4_scale_2d),
        )
        if (
            experimental_output_shared_split_v is True
            and not output_shared_eligible
        ):
            raise ValueError(
                "experimental output-shared split V requires the native "
                "NVFP4-QK / direct rowwise MXFP4-PV split-V route"
            )
        self.experimental_output_shared_split_v = bool(
            output_shared_eligible
            if experimental_output_shared_split_v is None
            else experimental_output_shared_split_v
        )
        self.experimental_output_shared_split_v_resolved = (
            self.experimental_output_shared_split_v
        )
        self.output_shared_split_v_path = (
            "output_shared_split_v"
            if self.experimental_output_shared_split_v
            else "retained_split_v"
            if output_shared_eligible
            else "not_applicable"
        )
        self.qkv_projection = None
        self.qkv_projection_symbol = None
        self.qkv_projection_abi_validation_symbol = None
        self.qkv_projection_requires_vscale_out = False
        self.qkv_projection_requires_forward_workspace = False
        if qkv_projection_format == "e4m3":
            if is_d128:
                self.qkv_projection = (
                    b300_bind_qkv_gqa_d128_unified_lowp_e4m3_projection(
                        batch=config.batch,
                        seqlen=config.sequence,
                        hidden=config.hidden,
                        q_heads=config.q_heads,
                        kv_heads=config.kv_heads,
                        publish_mxfp4_v=self.publish_mxfp4_v,
                        v_mxfp4_scale_2d=bool(v_mxfp4_scale_2d),
                    )
                )
            else:
                self.qkv_projection = (
                    b300_bind_qkv_gqa_d64_paired_unified_lowp_e4m3_projection(
                        batch=config.batch,
                        seqlen=config.sequence,
                        q_heads=config.q_heads,
                        kv_heads=config.kv_heads,
                        publish_mxfp4_v=self.publish_mxfp4_v,
                        v_mxfp4_scale_2d=bool(v_mxfp4_scale_2d),
                        represented_backward=(
                            self.backward_match_forward_operands
                        ),
                        per_block_qk_scales=self.per_block_qk_scales,
                        experimental_split_v_backward=(
                        self.experimental_split_v_backward
                        ),
                    )
                )
        elif self.experimental_native_nvfp4_projection_out:
            if is_d128:
                self.qkv_projection = (
                    b300_bind_qkv_gqa_d128_unified_lowp_nvfp4_projection(
                        batch=config.batch,
                        seqlen=config.sequence,
                        hidden=config.hidden,
                        q_heads=config.q_heads,
                        kv_heads=config.kv_heads,
                        publish_mxfp4_v=self.publish_mxfp4_v,
                        v_mxfp4_scale_2d=bool(v_mxfp4_scale_2d),
                        per_block_qk_scales=self.per_block_qk_scales,
                        represented_backward=(
                            self.backward_match_forward_operands
                        ),
                        experimental_output_shared_dual_v=(
                            experimental_output_shared_split_v
                        ),
                        experimental_mx_backward_v=(
                            self.experimental_d128_mxfp4_v_backward
                            and not bool(v_mxfp4_scale_2d)
                        ),
                        experimental_shared_tile_mx_backward_v=(
                            self.experimental_d128_mxfp4_v_backward
                            and bool(v_mxfp4_scale_2d)
                        ),
                    )
                )
            else:
                self.qkv_projection = (
                    b300_bind_qkv_gqa_d64_paired_unified_lowp_nvfp4_projection(
                        batch=config.batch,
                        seqlen=config.sequence,
                        hidden=config.hidden,
                        q_heads=config.q_heads,
                        kv_heads=config.kv_heads,
                        publish_mxfp4_v=self.publish_mxfp4_v,
                        v_mxfp4_scale_2d=bool(v_mxfp4_scale_2d),
                        experimental_output_shared_split_v=(
                            experimental_output_shared_split_v
                        ),
                    )
                )
        if self.qkv_projection is not None:
            projection_requested = getattr(
                self.qkv_projection,
                "experimental_output_shared_split_v_requested",
                self.experimental_output_shared_split_v_requested,
            )
            projection_resolved = getattr(
                self.qkv_projection,
                "experimental_output_shared_split_v_resolved",
                self.experimental_output_shared_split_v_resolved,
            )
            if type(projection_resolved) is not bool:
                raise RuntimeError(
                    "native NVFP4 projection returned a non-bool "
                    "output-shared split-V resolution"
                )
            if projection_requested is not experimental_output_shared_split_v:
                raise RuntimeError(
                    "native NVFP4 projection changed output-shared split-V "
                    "requested provenance"
                )
            if (
                projection_resolved
                is not self.experimental_output_shared_split_v_resolved
            ):
                raise RuntimeError(
                    "native NVFP4 projection disagrees with the runtime "
                    "output-shared split-V resolution"
                )
            self.experimental_output_shared_split_v_requested = (
                projection_requested
            )
            self.experimental_output_shared_split_v = projection_resolved
            self.experimental_output_shared_split_v_resolved = (
                projection_resolved
            )
            self.qkv_projection_symbol = getattr(
                self.qkv_projection,
                "unchecked_symbol",
                self.qkv_projection.symbol,
            )
            # Keep the matching allocating authentication entrypoint explicit
            # in provenance.
            # The compact checked and unchecked symbols are reported
            # separately by ``forward_dispatch_contract``.
            self.qkv_projection_abi_validation_symbol = getattr(
                self.qkv_projection,
                "abi_validation_symbol",
                None,
            )
            self.qkv_projection_requires_vscale_out = bool(
                self.qkv_projection.requires_v_mxfp4_scales_out
            )
            self.qkv_projection_requires_forward_workspace = bool(
                getattr(
                    self.qkv_projection,
                    "requires_forward_workspace",
                    False,
                )
            )
            self.output_shared_split_v_path = getattr(
                self.qkv_projection,
                "output_shared_split_v_path",
                self.output_shared_split_v_path,
            )
            if is_d128 and self.experimental_native_nvfp4_projection_out:
                projection_represented = getattr(
                    self.qkv_projection,
                    "represented_backward",
                    None,
                )
                if type(projection_represented) is not bool:
                    raise RuntimeError(
                        "native D128 NVFP4 projection omitted its exact-bool "
                        "represented-backward provenance"
                    )
                if (
                    projection_represented
                    is not self.backward_match_forward_operands
                ):
                    raise RuntimeError(
                        "native D128 NVFP4 projection disagrees with the "
                        "runtime represented-backward selection"
                    )
                expected_backward_publication_semantics = (
                    "represented_nvfp4_qk_per_row_k16_with_"
                    "projection_accumulator_e4m3_v"
                    if self.backward_match_forward_operands
                    else (
                        "projection_accumulator_e4m3_qkv_shared_across_"
                        "pv_routes"
                    )
                )
                actual_backward_publication_semantics = getattr(
                    self.qkv_projection,
                    "backward_publication_semantics",
                    None,
                )
                if (
                    actual_backward_publication_semantics
                    != expected_backward_publication_semantics
                ):
                    raise RuntimeError(
                        "native D128 NVFP4 projection backward-publication "
                        "semantics disagree with the runtime selection: "
                        f"{actual_backward_publication_semantics!r} != "
                        f"{expected_backward_publication_semantics!r}"
                    )
        if (
            self.per_block_qk_scales
            and self.backward_match_forward_operands
        ):
            qk_backward_source = "represented_nvfp4_codes_per_row_k16"
        elif self.backward_match_forward_operands:
            qk_backward_source = "represented_nvfp4_codes_adaptive_head"
        else:
            qk_backward_source = "projection_accumulator_e4m3"
        if self.experimental_d128_mxfp4_v_backward:
            v_backward_source = (
                "shared_d32xs32_forward_anchor_mxfp4_v"
                if self.d128_mxfp4_v_scale_policy
                == MXFP4_V_SCALE_POLICY_SHARED_D32XS32
                else "rowwise_width6_mxfp4_v"
            )
        elif (
            pv_format == "mxfp4_e8m0_block32"
            and self.backward_match_forward_operands
            and not self.experimental_split_v_backward
        ):
            v_backward_source = "represented_mxfp4_codes"
        else:
            v_backward_source = "projection_accumulator_e4m3"
        e5m2_dout_backward = self.native_tk_d128_v509_e5m2_dout_backward
        e5m2_dout_publisher = (
            "b300_project_dout_unified_lowp_nvfp4_v509_e5m2"
        )
        self.output_projection_topology = {
            "forward_format": self.output_projection_format,
            "forward_activation_publication": (
                "functional_rowwise_e4m3_fp32_decode"
                if self.output_projection_format == "e4m3"
                else "functional_rowwise_nvfp4"
            ),
            "forward_weight_publication": (
                "functional_channelwise_e4m3_fp32_decode"
                if self.output_projection_format == "e4m3"
                else "caller_owned_dual_true_2d_nvfp4"
                if projection_weight_scale_2d
                else "functional_rowwise_nvfp4"
            ),
            "forward_kernel": (
                "b300_project_e4m3"
                if self.output_projection_format == "e4m3"
                else "b300_project_nvfp4"
            ),
            "forward_allocation": (
                "allocating_generic_correctness_canary_nonfinal_speed"
                if self.output_projection_format == "e4m3"
                else "caller_owned_weight_workspace_functional_activation_pack"
                if projection_weight_scale_2d
                else "functional_operand_preparation"
            ),
            "unused_nvfp4_forward_weight_publication": False,
            "backward_input_gradient_format": "nvfp4",
            "backward_input_gradient_kernel": (
                e5m2_dout_publisher
                if e5m2_dout_backward
                else "b300_project_dout_unified_lowp_nvfp4"
            ),
            **(
                {
                    "backward_attention_dout_format": "e5m2",
                    "backward_attention_dout_source": (
                        "projection_accumulator_e5m2_x4"
                    ),
                    "backward_attention_dout_encoding_scale": 4.0,
                }
                if e5m2_dout_backward
                else {}
            ),
            "backward_weight_publication": (
                "functional_true_2d_nvfp4_transpose_prepared_in_forward"
                if self.output_projection_format == "e4m3"
                else "caller_owned_dual_true_2d_nvfp4"
                if projection_weight_scale_2d
                else "functional_nvfp4_transpose_in_backward"
            ),
            "backward_weight_gradient_format": "bf16",
            "backward_weight_gradient_kernel": "torch.mm",
            "e4m3_backward_learned_projection_gemms": False,
            "asymmetric_forward_input_gradient": (
                self.output_projection_format == "e4m3"
            ),
        }
        self.projection_publication_topology = {
            "qkv_projection_format": qkv_projection_format,
            "output_projection_format": self.output_projection_format,
            "output_projection": dict(self.output_projection_topology),
            "forward_pv_format": pv_format,
            "native_tk_d128_native_score_backward": (
                self.native_tk_d128_native_score_backward
            ),
            "native_tk_d128_v509_e5m2_dout_backward": (
                self.native_tk_d128_v509_e5m2_dout_backward
            ),
            **(
                {
                    "dout_backward_format": "e5m2",
                    "dout_backward_source": (
                        "projection_accumulator_e5m2_x4"
                    ),
                    "dout_backward_kernel": e5m2_dout_publisher,
                    "v509_e5m2_dout_route": None,
                }
                if e5m2_dout_backward
                else {}
            ),
            "represented_backward": self.backward_match_forward_operands,
            "per_block_qk_scales": self.per_block_qk_scales,
            "qk_backward_source": qk_backward_source,
            "v_backward_source": v_backward_source,
            "experimental_split_v_backward": (
                self.experimental_split_v_backward
            ),
            "experimental_output_shared_split_v": (
                self.experimental_output_shared_split_v
            ),
            "experimental_d128_mxfp4_v_backward": (
                self.experimental_d128_mxfp4_v_backward
            ),
            "d128_mxfp4_v_scale_policy": (
                self.d128_mxfp4_v_scale_policy
            ),
            "experimental_output_shared_split_v_requested": (
                self.experimental_output_shared_split_v_requested
            ),
            "experimental_output_shared_split_v_resolved": (
                self.experimental_output_shared_split_v_resolved
            ),
            "output_shared_split_v_path": self.output_shared_split_v_path,
            "projection_forward_publication_path": getattr(
                self.qkv_projection,
                "projection_forward_publication_path",
                None,
            ),
            "output_shared_split_v_checked_symbol": (
                getattr(self.qkv_projection, "checked_symbol", None)
                if self.qkv_projection is not None
                else None
            ),
            "experimental_native_nvfp4_projection_out": (
                self.experimental_native_nvfp4_projection_out
            ),
            "experimental_fused_attention_rmsnorm_nvfp4": (
                self.experimental_fused_attention_rmsnorm_nvfp4
            ),
        }
        if backward_probability_correction is None:
            exported_attention_gain = forward_topology.get(
                "backward_attention_branch_gain",
                forward_topology.get("backward_probability_correction"),
            )
            if exported_attention_gain is not None:
                backward_probability_correction = float(
                    exported_attention_gain
                )
            else:
                # A route-wide gain changes dQ/dK/dV and attention dX but not
                # the output-projection weight gradient, so silently inferring
                # one from a softmax implementation unbalances learning rates.
                # Topologies that genuinely need calibration must export it.
                backward_probability_correction = 1.0
        if not math.isfinite(backward_probability_correction) or not (
            backward_probability_correction > 0.0
        ):
            raise ValueError("backward probability correction must be positive")
        self.backward_probability_correction = float(
            backward_probability_correction
        )
        self.backward_q_gain = float(
            forward_topology.get(
                "backward_q_gain", self.backward_probability_correction
            )
        )
        self.backward_k_gain = float(
            forward_topology.get(
                "backward_k_gain", self.backward_probability_correction
            )
        )
        self.backward_v_gain = float(
            forward_topology.get(
                "backward_v_gain", self.backward_probability_correction
            )
        )
        # Numerical diagnostics may separate the V projection-weight update
        # from V's contribution to attention-input dX.  They are identical in
        # every production configuration unless explicitly overridden.
        self.backward_v_weight_gain = self.backward_v_gain
        if not all(
            math.isfinite(value) and value > 0.0
            for value in (
                self.backward_q_gain,
                self.backward_k_gain,
                self.backward_v_gain,
            )
        ):
            raise ValueError("backward Q/K/V gains must be finite and positive")
        self.backward_exp2_requested_degree = int(
            requested_backward_policy["exp2_degree"]
        )
        self.backward_exp2_requested_period = (
            None
            if requested_backward_policy["exp2_period"] is None
            else int(requested_backward_policy["exp2_period"])
        )
        self.backward_fp8_ds_lift = backward_fp8_ds_lift
        self.backward_reuse_quantized_p = bool(backward_reuse_quantized_p)
        self.backward_forward_mx_probability_replay = bool(
            backward_forward_mx_probability_replay
        )
        scale_publication_binding = getattr(
            forward_extension,
            "forward_hao_direct_fp4pv_with_p_scales",
            None,
        )
        scale_publication_supported = bool(
            forward_topology.get("p_scale_publication_supported", False)
        )
        if backward_forward_mx_probability_scale_handoff is None:
            backward_forward_mx_probability_scale_handoff = (
                self.backward_forward_mx_probability_replay
                and scale_publication_binding is not None
                and scale_publication_supported
            )
        self.backward_forward_mx_probability_scale_handoff = bool(
            backward_forward_mx_probability_scale_handoff
        )
        if (
            self.backward_forward_mx_probability_scale_handoff
            and not self.backward_forward_mx_probability_replay
        ):
            raise ValueError(
                "forward MX probability scale handoff requires exact "
                "probability replay"
            )
        if (
            self.backward_forward_mx_probability_scale_handoff
            and scale_publication_binding is None
        ):
            raise ValueError(
                "forward MX probability scale handoff requires the "
                "forward_hao_direct_fp4pv_with_p_scales binding"
            )
        if (
            self.backward_forward_mx_probability_scale_handoff
            and not scale_publication_supported
        ):
            raise ValueError(
                "the selected forward topology does not support MX "
                "probability scale publication"
            )
        if self.backward_forward_mx_probability_replay:
            if config.head_dim != 64:
                raise ValueError(
                    "forward MX probability replay requires head dimension 64"
                )
            if not self.per_block_qk_scales:
                raise ValueError(
                    "forward MX probability replay requires represented "
                    "per-block Q/K operands"
                )
            if self.backward_reuse_quantized_p:
                raise ValueError(
                    "forward MX probability replay cannot reuse the E4M3 dV "
                    "probability for dS"
                )
            for key, expected_value in (
                D4ALL_FORWARD_PROBABILITY_REPLAY_TOPOLOGY.items()
            ):
                actual_value = forward_topology.get(key)
                if actual_value != expected_value:
                    raise ValueError(
                        "forward MX probability replay requires the exact "
                        f"d4all topology: {key}={actual_value!r}, expected "
                        f"{expected_value!r}"
                    )
        self.projection_weight_scale_2d = bool(projection_weight_scale_2d)
        self.v_mxfp4_scale_2d = bool(v_mxfp4_scale_2d)
        self.adaptive_qk_weight_scales = bool(adaptive_qk_weight_scales)
        self.qk_scales = torch.zeros(
            config.batch,
            config.q_heads if is_d128 else config.q_heads // 2,
            7,
            device="cuda",
            dtype=torch.float32,
        )
        if not math.isfinite(q_quant_scale) or q_quant_scale <= 0.0:
            raise ValueError("Q quantization scale must be positive and finite")
        if not math.isfinite(k_quant_scale) or k_quant_scale <= 0.0:
            raise ValueError("K quantization scale must be positive and finite")
        self.qk_scales[:, :, 0] = q_quant_scale
        self.qk_scales[:, :, 1] = k_quant_scale
        self.q_quant_scale = float(q_quant_scale)
        self.k_quant_scale = float(k_quant_scale)
        # The trainer may temporarily install this read-only sink for a
        # diagnostic step.  Keeping the default ``None`` adds no tensor work,
        # allocation, or synchronization to ordinary benchmark/training runs.
        self.forward_diagnostic_sink: (
            list[dict[str, torch.Tensor | None]] | None
        ) = None
        # Diagnostic-only D128 control: retain MX's attention output while
        # substituting LSE from the authenticated FP8-PV forward on the exact
        # same projection-native Q/K/V publication.  Production runtimes leave
        # this unset, so there is no extra launch, allocation, or dispatch.
        self.diagnostic_fp8_lse_extension: Any | None = None
        self.diagnostic_fp8_lse_entrypoint: Any | None = None
        self.diagnostic_fp8_lse_topology: dict[str, Any] | None = None
        self.diagnostic_fp8_lse_runtime_authenticated = False
        self.diagnostic_fp8_lse_loaded_artifact_identity: (
            dict[str, Any] | None
        ) = None
        self.diagnostic_fp8_lse_first_launch_receipt: dict[str, Any] | None = None
        self.diagnostic_fp8_lse_substitution_mode = "all_rows"
        self.diagnostic_fp8_lse_substitution_counts = {
            "control_launches": 0,
            "control_lse_entries_computed": 0,
            "mx_finite_entries_seen": 0,
            "mx_nonfinite_entries_seen": 0,
            "mx_nan_entries_seen": 0,
            "mx_posinf_entries_seen": 0,
            "mx_neginf_entries_seen": 0,
            "fp8_entries_substituted": 0,
            "mx_entries_retained": 0,
        }
        self.diagnostic_mx_qk_abi_identity: tuple[tuple[Any, ...], ...] | None = (
            None
        )
        self.forward_attention_dispatch = "construction_bound"
        if pv_format == "e4m3_fp8":
            self.forward_attention_symbol = "forward_hao_direct_fp8pv"
            self.forward_attention_entrypoint = getattr(
                forward_extension, self.forward_attention_symbol, None
            )
            self.launch_forward_attention = self._launch_forward_fp8
        elif (
            pv_format == "mxfp4_e8m0_block32"
            and self.backward_forward_mx_probability_scale_handoff
        ):
            self.forward_attention_symbol = (
                "forward_hao_direct_fp4pv_with_p_scales"
            )
            self.forward_attention_entrypoint = getattr(
                forward_extension,
                self.forward_attention_symbol,
                None,
            )
            self.launch_forward_attention = (
                self._launch_forward_mx_with_probability_scales
            )
        elif pv_format == "mxfp4_e8m0_block32":
            self.forward_attention_symbol = "forward_hao_direct_fp4pv"
            self.forward_attention_entrypoint = getattr(
                forward_extension, self.forward_attention_symbol, None
            )
            self.launch_forward_attention = self._launch_forward_mx
        else:
            raise RuntimeError(
                f"unsupported low-precision PV format: {pv_format!r}"
            )
        if self.forward_attention_entrypoint is None:
            raise RuntimeError(
                "the selected fixed forward artifact lacks its bound "
                f"{pv_format!r} entrypoint"
            )

        if shared_backward_runtime is not None:
            shared = shared_backward_runtime
            if self.native_tk_backward:
                if (
                    not getattr(shared, "native_tk_backward", False)
                    or getattr(shared, "native_tk_backward_kind", None)
                    != self.native_tk_backward_kind
                ):
                    raise ValueError(
                        "a native TK backward can share only the same native "
                        "shape specialization"
                    )
                if (
                    self.native_tk_d128_backward
                    and self.experimental_d128_mxfp4_v_backward
                    is not shared.experimental_d128_mxfp4_v_backward
                ):
                    raise ValueError(
                        "a native TK D128 backward can share only the same "
                        "E4M3-V or MXFP4-V operand ABI"
                    )
                if (
                    self.native_tk_d128_backward
                    and self.d128_mxfp4_v_scale_policy
                    != shared.d128_mxfp4_v_scale_policy
                ):
                    raise ValueError(
                        "a native TK D128 backward can share only the same "
                        "MXFP4 V producer-scale policy"
                    )
                self.control = shared.control
                self.backward_control_provenance = (
                    shared.backward_control_provenance
                )
                self.backward_control_generated_source = (
                    shared.backward_control_generated_source
                )
                self.d128_mxfp4_v_dp_patch_provenance = (
                    shared.d128_mxfp4_v_dp_patch_provenance
                )
                self.backward = shared.backward
                self.backward_exp2_degree = shared.backward_exp2_degree
                self.backward_exp2_period = shared.backward_exp2_period
                self.backward_exp2_policy = shared.backward_exp2_policy
                self.backward_fp8_ds_lift = shared.backward_fp8_ds_lift
                self.backward_detached_fp8_p_tmem = (
                    shared.backward_detached_fp8_p_tmem
                )
                self.backward_probability_tmem_policy = (
                    shared.backward_probability_tmem_policy
                )
                self.backward_head_fast_raster = (
                    shared.backward_head_fast_raster
                )
                self.backward_raster_policy = shared.backward_raster_policy
                self.backward_initialized_from_shared_runtime = True
                require_matching_backward_contracts(
                    {
                        "shared": shared.backward_contract(),
                        "candidate": self.backward_contract(),
                    }
                )
                require_shared_backward_physical_identity(shared, self)
                return
            if getattr(shared, "native_tk_backward", False):
                raise ValueError(
                    "a CuTe backward cannot share a native TK runner"
                )
            resolved_exp2 = resolve_backward_exp2_policy(
                sequence=config.sequence,
                head_dim=config.head_dim,
                q_heads=config.q_heads,
                kv_heads=config.kv_heads,
                lowp=True,
                exp2_degree=backward_exp2_degree,
                exp2_period=backward_exp2_period,
            )
            requested_runner_policy = {
                "exp2_degree": resolved_exp2.effective_degree,
                "exp2_period": resolved_exp2.effective_period,
                "fp8_ds_lift": backward_fp8_ds_lift,
                "reuse_quantized_p": bool(backward_reuse_quantized_p),
                "forward_mx_probability_replay": bool(
                    backward_forward_mx_probability_replay
                ),
                "forward_mx_probability_scale_handoff": bool(
                    self.backward_forward_mx_probability_scale_handoff
                ),
                "d128_mxfp4_v_backward": (
                    self.experimental_d128_mxfp4_v_backward
                ),
            }
            shared_runner_policy = {
                "exp2_degree": shared.backward_exp2_degree,
                "exp2_period": shared.backward_exp2_period,
                "fp8_ds_lift": int(shared.backward.kernel.fp8_ds_lift),
                "reuse_quantized_p": shared.backward_reuse_quantized_p,
                "forward_mx_probability_replay": (
                    shared.backward_forward_mx_probability_replay
                ),
                "forward_mx_probability_scale_handoff": (
                    shared.backward_forward_mx_probability_scale_handoff
                ),
                "d128_mxfp4_v_backward": (
                    shared.experimental_d128_mxfp4_v_backward
                ),
            }
            if requested_runner_policy != shared_runner_policy:
                raise ValueError(
                    "shared backward runner policy mismatch: requested "
                    f"{requested_runner_policy}, shared {shared_runner_policy}"
                )
            self.control = shared.control
            self.backward_control_provenance = (
                shared.backward_control_provenance
            )
            self.backward_control_generated_source = (
                shared.backward_control_generated_source
            )
            shared_patch_provenance = (
                _require_d128_mxfp4_v_dp_patch_provenance(
                    shared.control,
                    enabled=self.experimental_d128_mxfp4_v_backward,
                )
            )
            if getattr(
                shared,
                "d128_mxfp4_v_dp_patch_provenance",
                None,
            ) != shared_patch_provenance:
                raise RuntimeError(
                    "shared backward runtime changed D128 MXFP4 V dP patch "
                    "provenance"
                )
            self.d128_mxfp4_v_dp_patch_provenance = shared_patch_provenance
            self.backward = shared.backward
            self.backward_exp2_degree = shared.backward_exp2_degree
            self.backward_exp2_period = shared.backward_exp2_period
            self.backward_exp2_policy = shared.backward_exp2_policy
            self.backward_detached_fp8_p_tmem = (
                shared.backward_detached_fp8_p_tmem
            )
            self.backward_probability_tmem_policy = (
                shared.backward_probability_tmem_policy
            )
            self.backward_head_fast_raster = shared.backward_head_fast_raster
            self.backward_raster_policy = shared.backward_raster_policy
            self.backward_initialized_from_shared_runtime = True
            # A batched crossover is permitted only when the newly built
            # forward route resolves to the same complete logical backward
            # contract and retains the exact same physical runner state.  In
            # particular, this catches aggregate-autograd differences such as
            # fused versus eager attention RMSNorm before the runtime can be
            # attached to a decoder.
            require_matching_backward_contracts(
                {
                    "shared": shared.backward_contract(),
                    "candidate": self.backward_contract(),
                }
            )
            require_shared_backward_physical_identity(shared, self)
            return

        self.backward_initialized_from_shared_runtime = False

        if self.native_tk_backward:
            assert self.native_tk_backward_extension is not None
            assert self.native_tk_backward_extension_identity is not None
            self.control = self.native_tk_backward_extension
            self.backward_control_provenance = {
                "backend": (
                    NATIVE_TK_D64_BACKEND
                    if self.native_tk_d64_backward
                    else NATIVE_TK_D128_V509_E5M2_DOUT_BACKEND
                    if self.native_tk_d128_v509_e5m2_dout_backward
                    else NATIVE_TK_D128_NVFP4_SCORE_BACKEND
                    if self.native_tk_d128_native_score_backward
                    else NATIVE_TK_D128_SHARED_TILE_MX_BACKEND
                    if (
                        self.experimental_d128_mxfp4_v_backward
                        and self.d128_mxfp4_v_scale_policy
                        == MXFP4_V_SCALE_POLICY_SHARED_D32XS32
                    )
                    else NATIVE_TK_D128_MX_BACKEND
                    if self.experimental_d128_mxfp4_v_backward
                    else NATIVE_TK_D128_BACKEND
                ),
                "extension": dict(
                    self.native_tk_backward_extension_identity
                ),
            }
            self.backward_control_generated_source = None
            self.d128_mxfp4_v_dp_patch_provenance = None
            backward_type = (
                NativeTkD64E4M3Backward
                if self.native_tk_d64_backward
                else NativeTkD128NVFP4ScoreE4M3QKVE5M2DoutBackward
                if self.native_tk_d128_v509_e5m2_dout_backward
                else NativeTkD128NVFP4ScoreE4M3GradientBackward
                if self.native_tk_d128_native_score_backward
                else NativeTkD128SharedTileProducerV503Backward
                if (
                    self.experimental_d128_mxfp4_v_backward
                    and self.d128_mxfp4_v_scale_policy
                    == MXFP4_V_SCALE_POLICY_SHARED_D32XS32
                )
                else NativeTkD128Mxfp4VBackward
                if self.experimental_d128_mxfp4_v_backward
                else NativeTkD128E4M3Backward
            )
            backward_kwargs: dict[str, Any] = {
                "batch": config.batch,
                "device": torch.device(
                    "cuda", torch.cuda.current_device()
                ),
            }
            if (
                backward_type
                is NativeTkD128SharedTileProducerV503Backward
            ):
                if self.qkv_projection is None:
                    raise RuntimeError(
                        "shared D32xS32 native TK backward requires its "
                        "bound projection producer"
                    )
                backward_kwargs["producer"] = self.qkv_projection
            self.backward = backward_type(
                self.native_tk_backward_extension,
                **backward_kwargs,
            )
            if self.native_tk_d128_v509_e5m2_dout_backward:
                self.v509_e5m2_dout_route = dict(
                    b300_require_v509_e5m2_dout_route(
                        self.backward.extension_metadata
                    )
                )
                self.projection_publication_topology[
                    "v509_e5m2_dout_route"
                ] = dict(self.v509_e5m2_dout_route)
            # The shared-runtime physical-identity audit historically names
            # the executable specialization ``kernel``.  Retain that identity
            # surface without interpreting native scheduling through CuTe
            # fields; ``backward_contract`` uses the backend-neutral receipt.
            self.backward.kernel = self.native_tk_backward_extension
            self.backward_exp2_degree = self.backward.exp2_degree
            self.backward_exp2_period = self.backward.exp2_period
            self.backward_exp2_policy = dict(self.backward.exp2_policy)
            self.backward_fp8_ds_lift = None
            self.backward_detached_fp8_p_tmem = (
                self.backward.detached_fp8_p_tmem
            )
            self.backward_probability_tmem_policy = {
                "backend": self.backward.backend,
                "storage": "native_tk_internal",
                "detached_fp8_p_tmem": False,
            }
            self.backward_head_fast_raster = self.backward.head_fast_raster
            self.backward_raster_policy = dict(self.backward.raster_policy)
            return

        q = torch.empty(
            config.batch,
            config.sequence,
            config.q_heads,
            config.head_dim,
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )
        k = torch.empty(
            config.batch,
            config.sequence,
            config.kv_heads,
            config.head_dim,
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )
        if self.experimental_d128_mxfp4_v_backward:
            v = torch.empty(
                config.batch,
                config.sequence,
                config.kv_heads,
                config.head_dim // 2,
                device="cuda",
                dtype=torch.uint8,
            )
            v_mxfp4_scale_pages = torch.empty(
                config.batch,
                config.sequence // 128,
                config.kv_heads,
                512,
                device="cuda",
                dtype=torch.uint8,
            )
        else:
            v = torch.empty_like(k)
            v_mxfp4_scale_pages = None
        dout = torch.empty_like(q)
        stats = torch.zeros(
            config.batch,
            config.q_heads,
            1,
            config.sequence,
            device="cuda",
        )
        resolved_exp2 = resolve_backward_exp2_policy(
            sequence=config.sequence,
            head_dim=config.head_dim,
            q_heads=config.q_heads,
            kv_heads=config.kv_heads,
            lowp=True,
            exp2_degree=backward_exp2_degree,
            exp2_period=backward_exp2_period,
        )
        probability_tmem_auto_eligible = (
            config.head_dim == 64
            and backward_fp8_ds_lift == 16
            and not backward_reuse_quantized_p
            and not self.backward_forward_mx_probability_replay
            and resolved_exp2.effective_degree == 1
            and resolved_exp2.effective_period == 2
        )
        probability_tmem_policy = resolve_backward_probability_tmem_policy(
            sequence=config.sequence,
            head_dim=config.head_dim,
            q_heads=config.q_heads,
            kv_heads=config.kv_heads,
            batch=config.batch,
            lowp=True,
            detached_fp8_p_tmem=None,
            auto_eligible=probability_tmem_auto_eligible,
        )
        control = _load_control(
            fp8_p_storage="shared" if is_d128 else "tmem",
            direct_tma_dkdv=not is_d128,
            detached_fp8_p_tmem=(
                probability_tmem_policy.effective_detached
            ),
            use_d128_mxfp4_v_dp=(
                self.experimental_d128_mxfp4_v_backward
            ),
            precomposed_control_source=backward_control_source,
            precomposed_control_sha256=backward_control_sha256,
            precomposed_control_bytes=backward_control_bytes,
        )
        self.control = control
        self.backward_control_provenance = getattr(
            control, "TK_PRECOMPOSED_CONTROL_PROVENANCE", None
        )
        self.d128_mxfp4_v_dp_patch_provenance = (
            _require_d128_mxfp4_v_dp_patch_provenance(
                control,
                enabled=self.experimental_d128_mxfp4_v_backward,
            )
        )
        self.backward_control_generated_source = _source_content_identity(
            control.__file__
        )
        self.backward = CompiledGqaBackward(
            control,
            q=q,
            k=k,
            v=v,
            o_or_sum=stats,
            dout=dout,
            lse_or_scaled_lse=stats,
            q_heads=config.q_heads,
            kv_heads=config.kv_heads,
            lowp=True,
            precomputed_stats=True,
            workspace_stats=True,
            scale_softmax=(config.head_dim**-0.5) / 16.0,
            exp2_degree=backward_exp2_degree,
            exp2_period=backward_exp2_period,
            reuse_quantized_p=backward_reuse_quantized_p,
            forward_mx_probability_replay=(
                self.backward_forward_mx_probability_replay
            ),
            use_forward_mx_probability_scales=(
                self.backward_forward_mx_probability_scale_handoff
            ),
            fp8_ds_lift=backward_fp8_ds_lift,
            lowp_do_stages=2 if is_d128 else 1,
            direct_tma_dkdv=not is_d128,
            use_d128_mxfp4_v_dp=(
                self.experimental_d128_mxfp4_v_backward
            ),
            v_mxfp4_scale_pages=v_mxfp4_scale_pages,
        )
        self.backward_exp2_degree = self.backward.exp2_degree
        self.backward_exp2_period = self.backward.exp2_period
        self.backward_exp2_policy = self.backward.exp2_policy
        self.backward_detached_fp8_p_tmem = (
            self.backward.detached_fp8_p_tmem
        )
        self.backward_probability_tmem_policy = (
            probability_tmem_policy.as_dict()
        )
        self.backward_head_fast_raster = self.backward.head_fast_raster
        self.backward_raster_policy = self.backward.raster_policy
        if (
            config.head_dim == 64
            and config.batch in AUTHENTICATED_D64_EXACT_BATCHES
        ):
            observed_batched_backward = {
                "exp2_degree": self.backward_exp2_degree,
                "exp2_period": self.backward_exp2_period,
                "fp8_ds_lift": int(self.backward.kernel.fp8_ds_lift),
                "reuse_quantized_p": self.backward_reuse_quantized_p,
                "fp8_p_storage": getattr(
                    self.control, "TK_FP8_P_STORAGE", None
                ),
                "detached_fp8_p_tmem": (
                    self.backward_detached_fp8_p_tmem
                ),
                "direct_tma_dkdv": bool(self.backward.direct_tma_dkdv),
                "head_fast_raster": bool(self.backward_head_fast_raster),
                "batch": int(self.backward.dq.shape[0]),
                "probability_correction": (
                    self.backward_probability_correction
                ),
                "q_gain": self.backward_q_gain,
                "k_gain": self.backward_k_gain,
                "v_gain": self.backward_v_gain,
                "v_weight_gain": self.backward_v_weight_gain,
            }
            expected_batched_backward = {
                "exp2_degree": 1,
                "exp2_period": 2,
                "fp8_ds_lift": 16,
                "reuse_quantized_p": False,
                "fp8_p_storage": "tmem",
                "detached_fp8_p_tmem": False,
                "direct_tma_dkdv": True,
                "head_fast_raster": False,
                "batch": config.batch,
                "probability_correction": 1.0,
                "q_gain": 1.0,
                "k_gain": 1.0,
                "v_gain": 1.0,
                "v_weight_gain": 1.0,
            }
            if observed_batched_backward != expected_batched_backward:
                raise RuntimeError(
                    "batched exact FA4 compiled an unauthenticated "
                    "backward policy: "
                    f"{observed_batched_backward} != "
                    f"{expected_batched_backward}"
                )
        elif (
            config.head_dim == 128
            and config.batch in AUTHENTICATED_D128_EXACT_BATCHES
        ):
            observed_batched_backward = {
                "exp2_degree": self.backward_exp2_degree,
                "exp2_period": self.backward_exp2_period,
                "fp8_ds_lift": int(self.backward.kernel.fp8_ds_lift),
                "reuse_quantized_p": self.backward_reuse_quantized_p,
                "fp8_p_storage": getattr(
                    self.control, "TK_FP8_P_STORAGE", None
                ),
                "detached_fp8_p_tmem": self.backward_detached_fp8_p_tmem,
                "direct_tma_dkdv": bool(self.backward.direct_tma_dkdv),
                "head_fast_raster": bool(self.backward_head_fast_raster),
                "batch": int(self.backward.dq.shape[0]),
                "use_d128_mxfp4_v_dp": bool(
                    self.backward.kernel.use_d128_mxfp4_v_dp
                ),
                "probability_correction": self.backward_probability_correction,
                "q_gain": self.backward_q_gain,
                "k_gain": self.backward_k_gain,
                "v_gain": self.backward_v_gain,
                "v_weight_gain": self.backward_v_weight_gain,
            }
            expected_batched_backward = {
                "exp2_degree": 1,
                "exp2_period": 0,
                "fp8_ds_lift": 256,
                "reuse_quantized_p": True,
                "fp8_p_storage": "shared",
                "detached_fp8_p_tmem": False,
                "direct_tma_dkdv": False,
                "head_fast_raster": False,
                "batch": config.batch,
                "use_d128_mxfp4_v_dp": (
                    self.experimental_d128_mxfp4_v_backward
                ),
                "probability_correction": 1.0,
                "q_gain": 1.0,
                "k_gain": 1.0,
                "v_gain": 1.0,
                "v_weight_gain": 1.0,
            }
            if observed_batched_backward != expected_batched_backward:
                raise RuntimeError(
                    "D128 B2 FA4 compiled an unauthenticated backward policy: "
                    f"{observed_batched_backward} != "
                    f"{expected_batched_backward}"
                )

    @property
    def backward_execution_runtime(self) -> LowpAttentionRuntime:
        """Return the one route-independent runtime saved by autograd."""
        owner = self._backward_execution_runtime
        return self if owner is None else owner

    def _launch_forward_fp8(
        self,
        qkv: Any,
        output: torch.Tensor,
        lse: torch.Tensor,
        forward_mx_probability_scales: torch.Tensor | None,
    ) -> None:
        if forward_mx_probability_scales is not None:
            raise ValueError(
                "forward MX probability scales cannot be published by the "
                "FP8-PV route"
            )
        if qkv.v_backward_fp8 is None:
            raise RuntimeError(
                "the FP8-PV forward requires projection-native E4M3 V"
            )
        forward_v_fp8 = qkv.v_forward_fp8
        if forward_v_fp8 is None:
            raise RuntimeError(
                "the FP8-PV path requires projection-native feature-major V; "
                "refusing an unfused permute/contiguous fallback"
            )
        self.forward_attention_entrypoint(
            *qkv.qk_forward_operands(),
            forward_v_fp8,
            output,
            lse,
            0,
            True,
            True,
        )

    def _launch_forward_mx(
        self,
        qkv: Any,
        output: torch.Tensor,
        lse: torch.Tensor,
        forward_mx_probability_scales: torch.Tensor | None,
    ) -> None:
        if forward_mx_probability_scales is not None:
            raise ValueError(
                "forward MX probability scales were supplied to a runtime "
                "without scale handoff"
            )
        if self.diagnostic_fp8_lse_entrypoint is not None:
            self.diagnostic_mx_qk_abi_identity = tuple(
                _tensor_abi_identity(tensor)
                for tensor in qkv.qk_forward_operands()
            )
        self.forward_attention_entrypoint(
            *qkv.forward_operands(),
            output,
            lse,
            0,
            True,
            True,
        )

    def _launch_forward_mx_with_probability_scales(
        self,
        qkv: Any,
        output: torch.Tensor,
        lse: torch.Tensor,
        forward_mx_probability_scales: torch.Tensor | None,
    ) -> None:
        if forward_mx_probability_scales is None:
            raise RuntimeError(
                "forward MX probability scales were not allocated for the "
                "bound scale-handoff route"
            )
        self.forward_attention_entrypoint(
            *qkv.forward_operands(),
            output,
            lse,
            forward_mx_probability_scales,
            0,
            True,
            True,
        )

    def install_diagnostic_fp8_lse_control(
        self,
        extension: Any,
        topology: dict[str, Any],
        loaded_artifact_identity: dict[str, Any],
        substitution_mode: str = "all_rows",
    ) -> None:
        """Install a fail-closed FP8-LSE control for an MX D128 diagnostic."""
        if not self.is_d128 or self.pv_format != "mxfp4_e8m0_block32":
            raise ValueError(
                "FP8-LSE control requires the D128 MXFP4-PV runtime"
            )
        _require_forward_topology(self.config, topology)
        if topology.get("pv_format") != "e4m3_fp8":
            raise ValueError(
                "FP8-LSE control artifact must publish the E4M3 FP8-PV route"
            )
        entrypoint = getattr(extension, "forward_hao_direct_fp8pv", None)
        if entrypoint is None:
            raise RuntimeError(
                "FP8-LSE control artifact lacks forward_hao_direct_fp8pv"
            )
        if self.diagnostic_fp8_lse_entrypoint is not None:
            raise RuntimeError("FP8-LSE control is already installed")
        if substitution_mode not in DIAGNOSTIC_FP8_LSE_SUBSTITUTION_MODES:
            raise ValueError(
                "unsupported diagnostic FP8-LSE substitution mode "
                f"{substitution_mode!r}; expected one of "
                f"{DIAGNOSTIC_FP8_LSE_SUBSTITUTION_MODES!r}"
            )
        self.diagnostic_fp8_lse_extension = extension
        self.diagnostic_fp8_lse_entrypoint = entrypoint
        self.diagnostic_fp8_lse_topology = dict(topology)
        self.diagnostic_fp8_lse_loaded_artifact_identity = dict(
            loaded_artifact_identity
        )
        self.diagnostic_fp8_lse_first_launch_receipt = None
        self.diagnostic_fp8_lse_substitution_mode = substitution_mode
        self.diagnostic_fp8_lse_substitution_counts = {
            "control_launches": 0,
            "control_lse_entries_computed": 0,
            "mx_finite_entries_seen": 0,
            "mx_nonfinite_entries_seen": 0,
            "mx_nan_entries_seen": 0,
            "mx_posinf_entries_seen": 0,
            "mx_neginf_entries_seen": 0,
            "fp8_entries_substituted": 0,
            "mx_entries_retained": 0,
        }
        self.diagnostic_mx_qk_abi_identity = None

    def diagnostic_fp8_lse(
        self,
        qkv: Any,
        output_template: torch.Tensor,
        lse_template: torch.Tensor,
    ) -> torch.Tensor:
        """Select authenticated FP8-control LSE and discard its FP8 output."""
        entrypoint = self.diagnostic_fp8_lse_entrypoint
        if entrypoint is None:
            raise RuntimeError("FP8-LSE control is not installed")
        if qkv.v_forward_fp8 is not None:
            raise RuntimeError(
                "D128 MX projection unexpectedly published inactive FP8 V"
            )
        expected_output_shape = (
            self.config.batch,
            self.config.sequence,
            self.config.q_heads,
            self.config.head_dim,
        )
        expected_lse_shape = (
            self.config.batch,
            self.config.q_heads,
            1,
            self.config.sequence,
        )
        if (
            output_template.dtype != torch.bfloat16
            or tuple(output_template.shape) != expected_output_shape
            or not output_template.is_contiguous()
            or lse_template.dtype != torch.float32
            or tuple(lse_template.shape) != expected_lse_shape
            or not lse_template.is_contiguous()
            or lse_template.device != output_template.device
        ):
            raise RuntimeError(
                "FP8-LSE control requires contiguous MX BF16 O [B,S,H,D] "
                "and FP32 LSE [B,H,1,S] on one device"
            )
        backward_v_fp8 = qkv.v_backward_fp8
        expected_backward_v_shape = (
            self.config.batch,
            self.config.sequence,
            self.config.kv_heads,
            self.config.head_dim,
        )
        if (
            backward_v_fp8 is None
            or backward_v_fp8.dtype != torch.float8_e4m3fn
            or tuple(backward_v_fp8.shape) != expected_backward_v_shape
            or not backward_v_fp8.is_contiguous()
            or backward_v_fp8.device != output_template.device
        ):
            raise RuntimeError(
                "FP8-LSE control requires contiguous projection-native "
                "E4M3 backward V [B,S,KV,D]"
            )
        # The route-selective MX publisher deliberately omits inactive
        # feature-major FP8 V.  Reorder its exact E4M3 backward bytes only for
        # this diagnostic control. V cannot affect LSE, and the FP8 output is
        # discarded.
        forward_v_fp8 = backward_v_fp8.permute(0, 2, 3, 1).contiguous()
        expected_forward_v_shape = (
            self.config.batch,
            self.config.kv_heads,
            self.config.head_dim,
            self.config.sequence,
        )
        if tuple(forward_v_fp8.shape) != expected_forward_v_shape:
            raise RuntimeError("synthesized FP8 control V has the wrong shape")
        control_output = torch.empty_like(output_template)
        control_lse = torch.empty_like(lse_template)
        if (
            control_output.data_ptr() == output_template.data_ptr()
            or control_lse.data_ptr() == lse_template.data_ptr()
        ):
            raise RuntimeError("FP8-LSE control scratch unexpectedly aliases MX")
        first_launch = self.diagnostic_fp8_lse_first_launch_receipt is None
        if first_launch:
            mx_output_snapshot = output_template.clone()
            mx_lse_snapshot = lse_template.clone()
        qk_operands = qkv.qk_forward_operands()
        qk_abi_identity = tuple(
            _tensor_abi_identity(tensor) for tensor in qk_operands
        )
        if qk_abi_identity != self.diagnostic_mx_qk_abi_identity:
            raise RuntimeError(
                "FP8-LSE control did not receive the exact MX Q/K operands"
            )
        entrypoint(
            *qk_operands,
            forward_v_fp8,
            control_output,
            control_lse,
            0,
            True,
            True,
        )
        control_finite = torch.isfinite(control_lse)
        if not bool(control_finite.all()):
            raise RuntimeError(
                "FP8-LSE diagnostic requires finite FP8 control LSE"
            )
        mx_finite_mask = torch.isfinite(lse_template)
        mx_nan_count = int(torch.isnan(lse_template).sum())
        mx_posinf_count = int(torch.isposinf(lse_template).sum())
        mx_neginf_count = int(torch.isneginf(lse_template).sum())
        mx_nonfinite_count = mx_nan_count + mx_posinf_count + mx_neginf_count
        total_entries = lse_template.numel()
        mx_finite_count = total_entries - mx_nonfinite_count
        if self.diagnostic_fp8_lse_substitution_mode == "all_rows":
            selected_lse = control_lse
            substituted_count = total_entries
            retained_count = 0
            substitution_semantics = (
                "substitute_authenticated_fp8_control_lse_for_all_rows"
            )
        elif (
            self.diagnostic_fp8_lse_substitution_mode
            == "mx_nonfinite_only"
        ):
            selected_lse = (
                lse_template
                if mx_nonfinite_count == 0
                else torch.where(mx_finite_mask, lse_template, control_lse)
            )
            substituted_count = mx_nonfinite_count
            retained_count = mx_finite_count
            substitution_semantics = (
                "retain_finite_mx_lse_substitute_authenticated_fp8_control_"
                "lse_only_where_mx_lse_is_nonfinite"
            )
        else:
            raise RuntimeError(
                "diagnostic FP8-LSE substitution mode changed after install"
            )
        if not bool(torch.isfinite(selected_lse).all()):
            raise RuntimeError(
                "diagnostic FP8-LSE selection produced non-finite LSE"
            )
        substitution_counts = self.diagnostic_fp8_lse_substitution_counts
        substitution_counts["control_launches"] += 1
        substitution_counts["control_lse_entries_computed"] += total_entries
        substitution_counts["mx_finite_entries_seen"] += mx_finite_count
        substitution_counts["mx_nonfinite_entries_seen"] += mx_nonfinite_count
        substitution_counts["mx_nan_entries_seen"] += mx_nan_count
        substitution_counts["mx_posinf_entries_seen"] += mx_posinf_count
        substitution_counts["mx_neginf_entries_seen"] += mx_neginf_count
        substitution_counts["fp8_entries_substituted"] += substituted_count
        substitution_counts["mx_entries_retained"] += retained_count
        if not self.diagnostic_fp8_lse_runtime_authenticated:
            assert self.diagnostic_fp8_lse_extension is not None
            populated = dict(
                self.diagnostic_fp8_lse_extension.read_hao_direct_topology()
            )
            _require_forward_topology(
                self.config,
                populated,
                runtime_populated=True,
            )
            if populated.get("pv_format") != "e4m3_fp8":
                raise RuntimeError(
                    "runtime-populated FP8-LSE control changed PV format"
                )
            self.diagnostic_fp8_lse_topology = populated
            self.diagnostic_fp8_lse_runtime_authenticated = True
        if first_launch:
            mx_lse_float = mx_lse_snapshot.float()
            fp8_lse_float = control_lse.float()
            mx_finite = torch.isfinite(mx_lse_float)
            if mx_finite_count == 0:
                raise RuntimeError(
                    "FP8-LSE diagnostic found no finite MX LSE entries"
                )
            finite_mx_lse = mx_lse_float[mx_finite]
            finite_fp8_lse = fp8_lse_float[mx_finite]
            difference = finite_fp8_lse - finite_mx_lse
            absolute_difference = difference.abs()
            mx_denominator = finite_mx_lse.norm().clamp_min(1.0e-30)
            fp8_denominator = finite_fp8_lse.norm().clamp_min(1.0e-30)
            centered_mx = finite_mx_lse - finite_mx_lse.mean()
            centered_fp8 = finite_fp8_lse - finite_fp8_lse.mean()
            affine_slope = (
                (centered_mx * centered_fp8).sum()
                / centered_mx.square().sum().clamp_min(1.0e-30)
            )
            affine_intercept = (
                finite_fp8_lse.mean()
                - affine_slope * finite_mx_lse.mean()
            )
            affine_residual = (
                affine_slope * finite_mx_lse
                + affine_intercept
                - finite_fp8_lse
            )
            quantiles = torch.tensor(
                [0.5, 0.9, 0.99, 0.999],
                device=absolute_difference.device,
                dtype=torch.float32,
            )
            absolute_quantiles = torch.quantile(
                absolute_difference.flatten(), quantiles
            )
            self.diagnostic_fp8_lse_first_launch_receipt = {
                "same_qk_operand_storage_as_mx_launch": True,
                "qk_operands": [
                    {
                        "shape": list(tensor.shape),
                        "stride": list(tensor.stride()),
                        "dtype": str(tensor.dtype),
                    }
                    for tensor in qk_operands
                ],
                "synthesized_v": {
                    "source": "projection_accumulator_e4m3_backward_v",
                    "transform": "permute_B_S_KV_D_to_B_KV_D_S_contiguous",
                    "shape": list(forward_v_fp8.shape),
                    "dtype": str(forward_v_fp8.dtype),
                    "contiguous": bool(forward_v_fp8.is_contiguous()),
                },
                "scratch_output_distinct": bool(
                    control_output.data_ptr() != output_template.data_ptr()
                ),
                "scratch_lse_distinct": bool(
                    control_lse.data_ptr() != lse_template.data_ptr()
                ),
                "mx_output_bitwise_unchanged": bool(
                    torch.equal(output_template, mx_output_snapshot)
                ),
                "substitution": {
                    "mode": self.diagnostic_fp8_lse_substitution_mode,
                    "semantics": substitution_semantics,
                    "selection_policy_allowlisted_at_install": True,
                    "control_launch_computes_all_rows": True,
                    "total_entries": total_entries,
                    "mx_finite_entries": mx_finite_count,
                    "mx_nonfinite_entries": mx_nonfinite_count,
                    "fp8_entries_substituted": substituted_count,
                    "mx_entries_retained": retained_count,
                    "selected_lse_all_finite": True,
                },
                "mx_vs_fp8_lse": {
                    "mx_all_finite": bool(mx_finite.all()),
                    "mx_finite_count": mx_finite_count,
                    "mx_nonfinite_count": int((~mx_finite).sum()),
                    "mx_nan_count": int(torch.isnan(mx_lse_float).sum()),
                    "mx_posinf_count": int(
                        torch.isposinf(mx_lse_float).sum()
                    ),
                    "mx_neginf_count": int(
                        torch.isneginf(mx_lse_float).sum()
                    ),
                    "fp8_all_finite": True,
                    "relative_l2_over_mx": float(
                        difference.norm() / mx_denominator
                    ),
                    "relative_l2_over_fp8": float(
                        difference.norm() / fp8_denominator
                    ),
                    "max_abs": float(absolute_difference.max()),
                    "mean_abs": float(absolute_difference.mean()),
                    "mean_signed_fp8_minus_mx": float(difference.mean()),
                    "p50_abs": float(absolute_quantiles[0]),
                    "p90_abs": float(absolute_quantiles[1]),
                    "p99_abs": float(absolute_quantiles[2]),
                    "p999_abs": float(absolute_quantiles[3]),
                    "mx_finite_min": float(finite_mx_lse.min()),
                    "mx_finite_max": float(finite_mx_lse.max()),
                    "fp8_min": float(fp8_lse_float.min()),
                    "fp8_max": float(fp8_lse_float.max()),
                    "least_squares_affine_mx_to_fp8": {
                        "slope": float(affine_slope),
                        "intercept": float(affine_intercept),
                        "residual_relative_l2_over_fp8": float(
                            affine_residual.norm() / fp8_denominator
                        ),
                    },
                },
            }
        return selected_lse

    def d128_mxfp4_v_operand_cache_receipt(
        self,
    ) -> dict[str, Any] | None:
        """Return candidate-only host-wrapper cache diagnostics."""
        receipt = self.backward.d128_mxfp4_v_operand_cache_receipt()
        if self.experimental_d128_mxfp4_v_backward:
            if receipt is None:
                raise RuntimeError(
                    "D128 MXFP4 V backward is missing its operand cache"
                )
            return receipt
        if receipt is not None:
            raise RuntimeError(
                "retained backward unexpectedly exposes an MXFP4 V operand "
                "cache"
            )
        return None

    def d128_mxfp4_v_compilation_receipt(self) -> dict[str, Any] | None:
        """Expose candidate-only CUTLASS image and generated-code identity."""
        receipt = self.backward.d128_mxfp4_v_compilation_receipt()
        if self.experimental_d128_mxfp4_v_backward:
            if receipt is None:
                raise RuntimeError(
                    "D128 MXFP4 V backward is missing compiler provenance"
                )
            return receipt
        if receipt is not None:
            raise RuntimeError(
                "retained backward unexpectedly exposes MXFP4-V compiler "
                "provenance"
            )
        return None

    def forward_dispatch_contract(self) -> dict[str, Any]:
        """Describe the immutable call targets used by forward execution.

        This is deliberately separate from :meth:`backward_contract`: MXFP4
        and FP8-PV must expose different forward symbols while sharing exactly
        the same physical backward runner.  Reading this contract performs no
        tensor work, synchronization, dispatch, or state mutation.
        """
        projection = self.qkv_projection
        projection_bound = projection is not None
        projection_abi_validated = (
            bool(projection.abi_validated) if projection_bound else None
        )
        projection_vscale_out_abi_validated = (
            bool(projection.vscale_out_abi_validated)
            if projection_bound else None
        )
        projection_forward_workspace_abi_validated = (
            bool(projection.forward_workspace_abi_validated)
            if projection_bound
            and hasattr(projection, "forward_workspace_abi_validated")
            else None
        )
        validated_forward_workspace_count = (
            int(projection.validated_forward_workspace_count)
            if projection_bound
            and hasattr(projection, "validated_forward_workspace_count")
            else None
        )
        successful_full_abi_validation_count = (
            int(projection.successful_full_abi_validation_count)
            if projection_bound
            and hasattr(projection, "successful_full_abi_validation_count")
            else None
        )
        return {
            "schema": "lowp_forward_dispatch_contract_v2",
            "route": str(self.forward_topology.get("route", "")),
            "pv_format": self.pv_format,
            "shape": {
                "batch": self.config.batch,
                "sequence": self.config.sequence,
                "q_heads": self.config.q_heads,
                "kv_heads": self.config.kv_heads,
                "head_dim": self.config.head_dim,
            },
            "qkv_projection": {
                "format": self.qkv_projection_format,
                "experimental_native_nvfp4_caller_owned": (
                    self.experimental_native_nvfp4_projection_out
                ),
                "experimental_fused_attention_rmsnorm_nvfp4": (
                    self.experimental_fused_attention_rmsnorm_nvfp4
                ),
                "experimental_d128_mxfp4_v_backward": (
                    self.experimental_d128_mxfp4_v_backward
                ),
                "d128_mxfp4_v_scale_policy": (
                    self.d128_mxfp4_v_scale_policy
                ),
                "output_shared_split_v_requested": (
                    self.experimental_output_shared_split_v_requested
                ),
                "output_shared_split_v_resolved": (
                    self.experimental_output_shared_split_v_resolved
                ),
                "output_shared_split_v_path": self.output_shared_split_v_path,
                "projection_forward_publication_path": getattr(
                    projection,
                    "projection_forward_publication_path",
                    None,
                ),
                "backward_publication_semantics": getattr(
                    projection,
                    "backward_publication_semantics",
                    None,
                ),
                "dispatch": (
                    "construction_bound_exact_pybind_symbol"
                    if projection_bound
                    else "public_api_per_invocation"
                ),
                "symbol": self.qkv_projection_symbol,
                "abi_validation_symbol": (
                    self.qkv_projection_abi_validation_symbol
                ),
                "checked_symbol": (
                    getattr(projection, "checked_symbol", None)
                    if projection_bound else None
                ),
                "unchecked_symbol": (
                    getattr(projection, "unchecked_symbol", None)
                    if projection_bound else None
                ),
                "shape_bound_at_construction": projection_bound,
                "first_call_full_abi_validation_complete": (
                    projection_abi_validated
                ),
                "subsequent_call_path": (
                    "bound_exact_pybind_symbol_with_preallocated_"
                    "forward_workspace"
                    if projection_bound else "public_api"
                ),
                "preallocated_forward_workspace_required": (
                    self.qkv_projection_requires_forward_workspace
                ),
                "preallocated_forward_publication_slots": list(
                    _FORWARD_WORKSPACE_OWNER_SLOTS.values()
                ),
                "preallocated_forward_workspace_abi_validated": (
                    projection_forward_workspace_abi_validated
                ),
                "validated_forward_workspace_count": (
                    validated_forward_workspace_count
                ),
                "live_validated_forward_workspace_count": (
                    validated_forward_workspace_count
                ),
                "successful_full_abi_validation_count": (
                    successful_full_abi_validation_count
                ),
                "timed_forward_publication_allocation_fallback": (
                    False if projection_bound else True
                ),
                "preallocated_forward_workspace_ownership": (
                    "private_nonpersistent_layer_route_neutral_superset"
                    if projection_bound
                    else "allocated_publication_return_owned_by_autograd"
                ),
                "qk_payload_typed_alias_materialization": (
                    "construction_time"
                ),
                "runtime_crossover_reallocation": False,
                "preallocated_v_mxfp4_scales_required": (
                    self.qkv_projection_requires_vscale_out
                ),
                "preallocated_v_mxfp4_scales_slot13_identity_validated": (
                    projection_vscale_out_abi_validated
                ),
                "timed_vscale_allocation_fallback": (
                    False
                    if self.qkv_projection_requires_vscale_out
                    else None
                ),
                "preallocated_v_mxfp4_scales_ownership": (
                    "nonpersistent_layer_workspace"
                    if self.qkv_projection_requires_vscale_out
                    else "not_applicable"
                ),
            },
            "output_projection": dict(self.output_projection_topology),
            "attention": {
                "dispatch": "construction_bound_route_specific_entrypoint",
                "symbol": self.forward_attention_symbol,
                "launcher": self.launch_forward_attention.__name__,
                "entrypoint_bound_at_construction": bool(
                    callable(self.forward_attention_entrypoint)
                ),
                "launcher_bound_to_runtime": bool(
                    getattr(self.launch_forward_attention, "__self__", None)
                    is self
                ),
                "runtime_topology_authenticated_after_launch": (
                    self.forward_topology_runtime_authenticated
                ),
                "runtime_topology_valid": self.forward_topology.get("valid"),
            },
        }

    def backward_contract(self) -> dict[str, Any]:
        """Return every effective field that can change backward work."""
        publication = self.projection_publication_topology
        if self.native_tk_backward:
            backend_contract = (
                self.backward.contract(fused_publisher_precleared_dq=True)
                if self.native_tk_d128_v509_e5m2_dout_backward
                else self.backward.contract()
            )
            return {
                "schema": "lowp_backward_contract_v1",
                "backend": backend_contract,
                "autograd": {
                    "experimental_fused_attention_rmsnorm_nvfp4": (
                        self.experimental_fused_attention_rmsnorm_nvfp4
                    ),
                },
                "shape": {
                    "batch": self.config.batch,
                    "sequence": self.config.sequence,
                    "q_heads": self.config.q_heads,
                    "kv_heads": self.config.kv_heads,
                    "head_dim": self.config.head_dim,
                },
                "probability": {
                    "implementation": "native_tk_internal",
                    "forward_mx_probability_replay": False,
                    "forward_mx_probability_scale_handoff": False,
                    "reuse_quantized_p": False,
                    "exp2_policy": dict(self.backward_exp2_policy),
                    "fp8_ds_lift": None,
                },
                "projection": {
                    "qkv_projection_format": self.qkv_projection_format,
                    "output_projection_format": (
                        self.output_projection_format
                    ),
                    "output_projection_forward_backward": dict(
                        self.output_projection_topology
                    ),
                    "projection_dgrad": self.projection_dgrad,
                    "projection_weight_scale_2d": (
                        self.projection_weight_scale_2d
                    ),
                    "represented_backward": publication[
                        "represented_backward"
                    ],
                    "per_block_qk_scales": publication[
                        "per_block_qk_scales"
                    ],
                    "qk_backward_source": publication[
                        "qk_backward_source"
                    ],
                    "v_backward_source": publication["v_backward_source"],
                    "dout_backward_source": (
                        "projection_accumulator_e5m2_x4"
                        if self.native_tk_d128_v509_e5m2_dout_backward
                        else "projection_accumulator_e4m3"
                    ),
                    **(
                        {
                            "dout_backward_format": "e5m2",
                            "dout_backward_kernel": (
                                "b300_project_dout_unified_lowp_nvfp4_"
                                "v509_e5m2"
                            ),
                            "native_tk_d128_v509_e5m2_dout_backward": True,
                            "v509_e5m2_dout_route": dict(
                                self.v509_e5m2_dout_route or {}
                            ),
                        }
                        if self.native_tk_d128_v509_e5m2_dout_backward
                        else {}
                    ),
                    "native_tk_d128_native_score_backward": (
                        self.native_tk_d128_native_score_backward
                    ),
                    "experimental_d128_mxfp4_v_backward": (
                        self.experimental_d128_mxfp4_v_backward
                    ),
                    "d128_mxfp4_v_scale_policy": (
                        self.d128_mxfp4_v_scale_policy
                    ),
                    "q_quant_scale": self.q_quant_scale,
                    "k_quant_scale": self.k_quant_scale,
                },
                "shape_policy": self.backward_shape_policy,
                "scaling": {
                    "loss_scale": self.loss_scale,
                    "gradient_global_scale": (
                        self.gradient_global_scale_value
                    ),
                    "probability_correction": (
                        self.backward_probability_correction
                    ),
                    "q_gain": self.backward_q_gain,
                    "k_gain": self.backward_k_gain,
                    "v_gain": self.backward_v_gain,
                    "v_weight_gain": self.backward_v_weight_gain,
                },
            }
        kernel = self.backward.kernel
        return {
            "schema": "lowp_backward_contract_v1",
            "autograd": {
                # Fused attention RMSNorm replaces the eager RMSNorm autograd
                # graph with the exact-shape CUDA backward.  It does not alter
                # the FA4 runner, but it does alter aggregate decoder backward
                # work and therefore belongs in every matched-route contract.
                "experimental_fused_attention_rmsnorm_nvfp4": (
                    self.experimental_fused_attention_rmsnorm_nvfp4
                ),
            },
            "shape": {
                "batch": self.config.batch,
                "sequence": self.config.sequence,
                "q_heads": self.config.q_heads,
                "kv_heads": self.config.kv_heads,
                "head_dim": self.config.head_dim,
            },
            "control": {
                "provenance": self.backward_control_provenance,
                "generated_source": self.backward_control_generated_source,
                "fp8_p_storage": getattr(
                    self.control, "TK_FP8_P_STORAGE", None
                ),
                "direct_tma_dkdv": bool(self.backward.direct_tma_dkdv),
                "detached_fp8_p_tmem": (
                    self.backward_detached_fp8_p_tmem
                ),
                **(
                    {
                        "d128_mxfp4_v_dp_patch": dict(
                            self.d128_mxfp4_v_dp_patch_provenance
                        )
                    }
                    if self.d128_mxfp4_v_dp_patch_provenance is not None
                    else {}
                ),
            },
            "probability": {
                "forward_mx_probability_replay": (
                    self.backward_forward_mx_probability_replay
                ),
                "forward_mx_probability_scale_handoff": (
                    self.backward_forward_mx_probability_scale_handoff
                ),
                "reuse_quantized_p": self.backward_reuse_quantized_p,
                "exp2_degree": self.backward_exp2_degree,
                "exp2_period": self.backward_exp2_period,
                "fp8_ds_lift": int(kernel.fp8_ds_lift),
                "fuse_probability_lift": bool(
                    kernel.fuse_probability_lift
                ),
                "prelift_probability_lse": bool(
                    kernel.prelift_probability_lse
                ),
            },
            "schedule": {
                "head_fast_raster": self.backward_head_fast_raster,
                "load_mma_q_stages": int(kernel.load_mma_Q_stage),
                "load_mma_do_stages": int(kernel.load_mma_dO_stage),
                "mma_dkdv_stages": int(kernel.mma_compute_dKdV_stage),
                "split_gqa_heads": bool(kernel.split_gqa_heads),
                "fuse_gqa_reduce": bool(kernel.fuse_gqa_reduce),
                "compact_dq_acc": bool(kernel.compact_dq_acc),
                "direct_compact_dq": bool(kernel.direct_compact_dq),
                "skip_stats_preprocess": bool(
                    kernel.skip_stats_preprocess
                ),
            },
            "projection": {
                "qkv_projection_format": self.qkv_projection_format,
                "output_projection_format": self.output_projection_format,
                "output_projection_forward_backward": dict(
                    self.output_projection_topology
                ),
                "projection_dgrad": self.projection_dgrad,
                "projection_weight_scale_2d": (
                    self.projection_weight_scale_2d
                ),
                "represented_backward": publication[
                    "represented_backward"
                ],
                "per_block_qk_scales": publication[
                    "per_block_qk_scales"
                ],
                "qk_backward_source": publication["qk_backward_source"],
                "v_backward_source": publication["v_backward_source"],
                "experimental_d128_mxfp4_v_backward": (
                    self.experimental_d128_mxfp4_v_backward
                ),
                "d128_mxfp4_v_scale_policy": (
                    self.d128_mxfp4_v_scale_policy
                ),
                "q_quant_scale": self.q_quant_scale,
                "k_quant_scale": self.k_quant_scale,
            },
            "shape_policy": self.backward_shape_policy,
            "scaling": {
                "loss_scale": self.loss_scale,
                "gradient_global_scale": self.gradient_global_scale_value,
                "probability_correction": (
                    self.backward_probability_correction
                ),
                "q_gain": self.backward_q_gain,
                "k_gain": self.backward_k_gain,
                "v_gain": self.backward_v_gain,
                "v_weight_gain": self.backward_v_weight_gain,
            },
        }

    def bind_backward_inputs(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dout: torch.Tensor,
        forward_mx_probability_scales: torch.Tensor | None = None,
        *,
        v_mxfp4_scale_pages: torch.Tensor | None = None,
        producer_workspace: B300E4M3QKVForwardWorkspace | None = None,
        native_score_workspace: B300E4M3QKVForwardWorkspace | None = None,
    ) -> None:
        if self.native_tk_backward:
            if forward_mx_probability_scales is not None:
                raise ValueError(
                    "native TK backward does not accept forward MX "
                    "probability-scale handoff"
                )
            if self.native_tk_d128_native_score_backward:
                if v_mxfp4_scale_pages is not None:
                    raise ValueError(
                        "native-score D128 backward requires retained E4M3 V"
                    )
                if producer_workspace is not None:
                    raise ValueError(
                        "native-score D128 backward does not accept an MX "
                        "producer workspace"
                    )
                if native_score_workspace is None:
                    raise RuntimeError(
                        "native-score D128 backward requires its exact "
                        "forward publication workspace"
                    )
                self.backward.bind_inputs(
                    q,
                    k,
                    v,
                    dout,
                    native_score_workspace,
                )
                return
            if native_score_workspace is not None:
                raise ValueError(
                    "a native-score workspace is accepted only by the "
                    "native NVFP4-score D128 backward route"
                )
            if (
                self.native_tk_d128_backward
                and self.experimental_d128_mxfp4_v_backward
            ):
                if v_mxfp4_scale_pages is None:
                    raise RuntimeError(
                        "native TK D128 MXFP4 V backward requires retained "
                        "E8M0 scale pages"
                    )
                if (
                    self.d128_mxfp4_v_scale_policy
                    == MXFP4_V_SCALE_POLICY_SHARED_D32XS32
                ):
                    if producer_workspace is None:
                        raise RuntimeError(
                            "shared D32xS32 native TK backward requires the "
                            "authenticated projection workspace"
                        )
                    self.backward.bind_inputs(
                        q,
                        k,
                        v,
                        v_mxfp4_scale_pages,
                        dout,
                        producer_workspace=producer_workspace,
                    )
                else:
                    if producer_workspace is not None:
                        raise ValueError(
                            "legacy rowwise native TK MXFP4 V backward must "
                            "not receive a shared-tile producer workspace"
                        )
                    self.backward.bind_inputs(
                        q,
                        k,
                        v,
                        v_mxfp4_scale_pages,
                        dout,
                    )
                return
            if producer_workspace is not None:
                raise ValueError(
                    "a producer workspace is accepted only by the shared "
                    "D32xS32 native TK MXFP4 V backward route"
                )
            if v_mxfp4_scale_pages is not None:
                raise ValueError(
                    "native TK backward requires projection-native E4M3 "
                    "V, not MXFP4 V scale pages"
                )
            self.backward.bind_inputs(q, k, v, dout)
            return
        if native_score_workspace is not None:
            raise ValueError(
                "a native-score workspace is not accepted by CuTe backward"
            )
        c = self.config
        arguments = list(self.backward.arguments)
        arguments[1] = _attention_cute_tensor(
            self.control, q, q_heads=c.q_heads, kv_heads=c.kv_heads
        )
        arguments[2] = _attention_cute_tensor(
            self.control, k, q_heads=c.q_heads, kv_heads=c.kv_heads
        )
        arguments[8] = _attention_cute_tensor(
            self.control, dout, q_heads=c.q_heads, kv_heads=c.kv_heads
        )
        self.backward.arguments = tuple(arguments)
        if self.experimental_d128_mxfp4_v_backward:
            if v_mxfp4_scale_pages is None:
                raise RuntimeError(
                    "D128 MXFP4 V backward requires retained E8M0 scale pages"
                )
            self.backward.bind_d128_mxfp4_v_operands(
                v,
                v_mxfp4_scale_pages,
            )
        else:
            if v_mxfp4_scale_pages is not None:
                raise ValueError(
                    "MXFP4 V scale pages were supplied to the retained E4M3 "
                    "backward route"
                )
            arguments = list(self.backward.arguments)
            arguments[3] = _attention_cute_tensor(
                self.control,
                v,
                q_heads=c.q_heads,
                kv_heads=c.kv_heads,
            )
            self.backward.arguments = tuple(arguments)
        if self.backward_forward_mx_probability_scale_handoff:
            if forward_mx_probability_scales is None:
                raise RuntimeError(
                    "forward MX probability scales were not retained for "
                    "backward"
                )
            self.backward.bind_forward_mx_probability_scales(
                forward_mx_probability_scales
            )
        elif forward_mx_probability_scales is not None:
            raise ValueError(
                "forward MX probability scales were supplied without "
                "enabling scale handoff"
            )


def _run_lowp_forward_attention(
    runtime: LowpAttentionRuntime,
    qkv: Any,
    output: torch.Tensor,
    lse: torch.Tensor,
    forward_mx_probability_scales: torch.Tensor | None = None,
) -> None:
    """Launch the construction-bound projection-native attention route."""
    runtime.launch_forward_attention(
        qkv,
        output,
        lse,
        forward_mx_probability_scales,
    )
    if not runtime.forward_topology_runtime_authenticated:
        populated_topology = dict(
            runtime.forward_extension.read_hao_direct_topology()
        )
        _require_forward_topology(
            runtime.config,
            populated_topology,
            runtime_populated=True,
        )
        runtime.forward_topology = populated_topology
        runtime.forward_topology_runtime_authenticated = True


@dataclass(slots=True)
class _WorkspacePublicationState:
    """Fail-closed ownership for publications retained until backward."""

    current_generation: int = -1
    in_flight_generation: int | None = None

    def begin_forward(self, *, requires_backward: bool) -> int:
        if type(requires_backward) is not bool:
            raise TypeError("requires_backward must be exactly bool")
        if self.in_flight_generation is not None:
            raise RuntimeError(
                "low-precision attention workspace generation "
                f"{self.in_flight_generation} is still awaiting backward; "
                "only one forward may be in flight per layer"
            )
        self.current_generation += 1
        generation = self.current_generation
        if requires_backward:
            self.in_flight_generation = generation
        return generation

    def require_backward(self, generation: int) -> None:
        if self.in_flight_generation != generation:
            raise RuntimeError(
                "low-precision attention backward does not own the active "
                "workspace publication generation; "
                f"requested {generation}, active "
                f"{self.in_flight_generation}"
            )

    def finish_backward(self, generation: int) -> None:
        self.require_backward(generation)
        self.in_flight_generation = None

    def abort_forward(self, generation: int) -> None:
        if generation != self.current_generation:
            raise RuntimeError(
                "cannot abort a stale low-precision attention workspace "
                f"generation {generation}; current generation is "
                f"{self.current_generation}"
            )
        if self.in_flight_generation not in (None, generation):
            raise RuntimeError(
                "cannot abort a low-precision attention workspace owned by "
                f"generation {self.in_flight_generation}"
            )
        self.in_flight_generation = None


@dataclass(slots=True)
class _DualWeightPackPublicationState:
    """Fail-closed lifecycle for one layer's rolling weight publication."""

    generation: int = -1
    weight_versions: tuple[int, int, int, int] | None = None
    qkv_published: bool = False
    output_published: bool = False
    qkv_consumed: bool = False
    output_consumed: bool = False
    backward_enqueued: bool = True

    def require_can_begin(self, generation: int) -> None:
        if generation <= self.generation:
            raise RuntimeError(
                "dual-weight forward generations must increase: "
                f"{generation} <= {self.generation}"
            )
        if self.generation >= 0 and not self.backward_enqueued:
            raise RuntimeError(
                "cannot overwrite a dual-weight publication before its "
                "backward consumer has been enqueued"
            )

    def begin(
        self,
        generation: int,
        weight_versions: tuple[int, int, int, int],
    ) -> None:
        self.require_can_begin(generation)
        if len(weight_versions) != 4:
            raise ValueError(
                "dual-weight publication requires Q/K/V/O versions"
            )
        self.generation = generation
        self.weight_versions = weight_versions
        self.qkv_published = False
        self.output_published = False
        self.qkv_consumed = False
        self.output_consumed = False
        self.backward_enqueued = False

    def publish_qkv(
        self,
        generation: int,
        weight_versions: tuple[int, int, int, int],
    ) -> None:
        self._require_current(generation, weight_versions)
        if self.qkv_published:
            raise RuntimeError("QKV dual-weight publication was enqueued twice")
        self.qkv_published = True

    def publish_output(
        self,
        generation: int,
        weight_versions: tuple[int, int, int, int],
    ) -> None:
        self._require_current(generation, weight_versions)
        if not self.qkv_published:
            raise RuntimeError("O publication was enqueued before QKV")
        if self.output_published:
            raise RuntimeError("O dual-weight publication was enqueued twice")
        self.output_published = True

    def consume_qkv(
        self,
        generation: int,
        weight_versions: tuple[int, int, int, int],
    ) -> None:
        self._require_current(generation, weight_versions)
        if not self.qkv_published:
            raise RuntimeError("QKV consumer reached an unpublished operand")
        if self.qkv_consumed:
            raise RuntimeError("QKV dual-weight operand was consumed twice")
        self.qkv_consumed = True

    def consume_output(
        self,
        generation: int,
        weight_versions: tuple[int, int, int, int],
    ) -> None:
        self._require_current(generation, weight_versions)
        if not self.qkv_consumed:
            raise RuntimeError("O consumer reached the publication before QKV")
        if not self.output_published:
            raise RuntimeError("O consumer reached an unpublished operand")
        if self.output_consumed:
            raise RuntimeError("O dual-weight operand was consumed twice")
        self.output_consumed = True

    def enqueue_backward(
        self,
        generation: int,
        weight_versions: tuple[int, int, int, int],
    ) -> None:
        self._require_current(generation, weight_versions)
        if not self.qkv_consumed or not self.output_consumed:
            raise RuntimeError(
                "backward reached an incompletely consumed dual-weight forward"
            )
        if self.backward_enqueued:
            raise RuntimeError("dual-weight backward was enqueued twice")
        self.backward_enqueued = True

    def release_without_backward(
        self,
        generation: int,
        weight_versions: tuple[int, int, int, int],
    ) -> None:
        self._require_current(generation, weight_versions)
        if not self.qkv_consumed or not self.output_consumed:
            raise RuntimeError(
                "no-grad release reached an incompletely consumed "
                "dual-weight forward"
            )
        if self.backward_enqueued:
            raise RuntimeError("dual-weight no-grad release was enqueued twice")
        self.backward_enqueued = True

    def _require_current(
        self,
        generation: int,
        weight_versions: tuple[int, int, int, int],
    ) -> None:
        if generation != self.generation:
            raise RuntimeError(
                "stale dual-weight generation: "
                f"{generation} != {self.generation}"
            )
        if weight_versions != self.weight_versions:
            raise RuntimeError(
                "stale dual-weight parameter versions: "
                f"{weight_versions} != {self.weight_versions}"
            )


@dataclass(slots=True)
class _LowpAttentionForwardWorkspace:
    """Opaque carrier for one layer's complete forward publications."""

    outputs: B300E4M3QKVForwardWorkspace
    qkv_weight_forward_packed: torch.Tensor | None
    qkv_weight_forward_scales: torch.Tensor | None
    qkv_weight_backward_packed: torch.Tensor | None
    qkv_weight_backward_scales: torch.Tensor | None
    qkv_weight_global_scale: torch.Tensor | None
    output_weight_forward_packed: torch.Tensor | None
    output_weight_forward_scales: torch.Tensor | None
    output_weight_backward_packed: torch.Tensor | None
    output_weight_backward_scales: torch.Tensor | None
    output_weight_global_scale: torch.Tensor | None
    allocation_data_ptrs: dict[str, int]
    cuda_stream: int | None
    d128_dual_qkv_weight_authenticated: bool = False
    output_dual_weight_authenticated: bool = False
    d128_dual_qkv_weight_abi_identity: tuple[Any, ...] | None = None
    output_dual_weight_abi_identity: tuple[Any, ...] | None = None
    weight_prep_authenticated: bool = False
    dual_weight_pack_controller: _DualWeightPackLayerController | None = None
    publication_state: _WorkspacePublicationState = field(
        default_factory=_WorkspacePublicationState
    )


_D128_DUAL_QKV_WEIGHT_FIELDS = (
    "qkv_weight_forward_packed",
    "qkv_weight_forward_scales",
    "qkv_weight_backward_packed",
    "qkv_weight_backward_scales",
    "qkv_weight_global_scale",
)

_DUAL_OUTPUT_WEIGHT_FIELDS = (
    "output_weight_forward_packed",
    "output_weight_forward_scales",
    "output_weight_backward_packed",
    "output_weight_backward_scales",
    "output_weight_global_scale",
)


_FORWARD_WORKSPACE_OWNER_SLOTS = {
    "q_payload": 4,
    "k_payload": 6,
    "q_scale_pages": 8,
    "q_global_scale": 9,
    "k_scale_pages": 10,
    "k_global_scale": 11,
    "v_mxfp4_payload": 12,
    "v_mxfp4_scale_pages": 13,
    "v_backward_fp8": 20,
    "q_backward_fp8": 21,
    "k_backward_fp8": 22,
    "v_fp8_payload": 23,
}
_FORWARD_WORKSPACE_COMMON_ROUTES = (
    "mxfp4_e8m0_block32",
    "e4m3_fp8",
)
_FORWARD_WORKSPACE_ACTIVE_ROUTES = {
    "q_payload": _FORWARD_WORKSPACE_COMMON_ROUTES,
    "k_payload": _FORWARD_WORKSPACE_COMMON_ROUTES,
    "q_scale_pages": _FORWARD_WORKSPACE_COMMON_ROUTES,
    "q_global_scale": _FORWARD_WORKSPACE_COMMON_ROUTES,
    "k_scale_pages": _FORWARD_WORKSPACE_COMMON_ROUTES,
    "k_global_scale": _FORWARD_WORKSPACE_COMMON_ROUTES,
    "v_backward_fp8": _FORWARD_WORKSPACE_COMMON_ROUTES,
    "q_backward_fp8": _FORWARD_WORKSPACE_COMMON_ROUTES,
    "k_backward_fp8": _FORWARD_WORKSPACE_COMMON_ROUTES,
    "v_mxfp4_payload": ("mxfp4_e8m0_block32",),
    "v_mxfp4_scale_pages": ("mxfp4_e8m0_block32",),
    "v_fp8_payload": ("e4m3_fp8",),
}
_FORWARD_WORKSPACE_ALIAS_OWNERS = {
    "q_payload_fp4": "q_payload",
    "k_payload_fp4": "k_payload",
}
_FORWARD_WORKSPACE_SENTINELS = (
    "empty_bf16",
    "empty_byte",
    "empty_fp8",
    "empty_fp4",
)
_FORWARD_WORKSPACE_OPTIONAL_OWNERS = (
    "v_backward_mxfp4",
    "v_backward_mxfp4_scale_pages",
)
_FORWARD_WORKSPACE_OPTIONAL_OWNER_SLOTS = {
    "v_backward_mxfp4": 24,
    "v_backward_mxfp4_scale_pages": 25,
}


def _forward_workspace_owner_tensors(
    workspace: _LowpAttentionForwardWorkspace,
) -> tuple[tuple[str, torch.Tensor], ...]:
    outputs = workspace.outputs
    return tuple(
        (name, getattr(outputs, name))
        for name in _FORWARD_WORKSPACE_OWNER_SLOTS
    )


def _forward_workspace_all_tensors(
    workspace: _LowpAttentionForwardWorkspace,
) -> tuple[tuple[str, torch.Tensor], ...]:
    outputs = workspace.outputs
    aliases = tuple(
        (name, getattr(outputs, name))
        for name in _FORWARD_WORKSPACE_ALIAS_OWNERS
    )
    sentinels = tuple(
        (name, getattr(outputs, name))
        for name in _FORWARD_WORKSPACE_SENTINELS
    )
    optional_owners = tuple(
        (name, tensor)
        for name in _FORWARD_WORKSPACE_OPTIONAL_OWNERS
        if (tensor := getattr(outputs, name)) is not None
    )
    dual_qkv_weight = _d128_dual_qkv_weight_tensors(workspace)
    dual_output_weight = _dual_output_weight_tensors(workspace)
    return (
        *_forward_workspace_owner_tensors(workspace),
        *optional_owners,
        *aliases,
        *sentinels,
        *dual_qkv_weight,
        *dual_output_weight,
    )


def _d128_dual_qkv_weight_tensors(
    workspace: _LowpAttentionForwardWorkspace,
) -> tuple[tuple[str, torch.Tensor], ...]:
    fields = tuple(
        (name, getattr(workspace, name))
        for name in _D128_DUAL_QKV_WEIGHT_FIELDS
    )
    present = tuple(tensor is not None for _name, tensor in fields)
    if not any(present):
        return ()
    if not all(present):
        raise RuntimeError(
            "D128 dual QKV weight workspace must be fully allocated or absent"
        )
    return tuple(
        (name, tensor)
        for name, tensor in fields
        if tensor is not None
    )


def _tensor_abi_identity(tensor: torch.Tensor) -> tuple[Any, ...]:
    """Return cheap storage/metadata identity, deliberately excluding version."""
    return (
        int(tensor.data_ptr()),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        int(tensor.storage_offset()),
        tensor.dtype,
        tensor.device,
    )


def _refresh_dual_weight_prep_authentication(
    workspace: _LowpAttentionForwardWorkspace,
) -> bool:
    """Refresh the aggregate only after both checked producers succeeded."""
    authenticated = bool(
        getattr(
            workspace,
            "d128_dual_qkv_weight_authenticated",
            False,
        )
        and getattr(
            workspace,
            "output_dual_weight_authenticated",
            False,
        )
    )
    # Lightweight unit stand-ins predate the aggregate slot. Real workspaces
    # always own it; do not require unrelated host-only mocks to grow it.
    if hasattr(workspace, "weight_prep_authenticated"):
        workspace.weight_prep_authenticated = authenticated
    return authenticated


def _prepare_direct_d128_dual_qkv_weight(
    workspace: _LowpAttentionForwardWorkspace,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """Refresh one layer's stable dual operand, authenticating first use."""
    tensors = _d128_dual_qkv_weight_tensors(workspace)
    if len(tensors) != len(_D128_DUAL_QKV_WEIGHT_FIELDS):
        raise RuntimeError(
            "direct D128 dual QKV weight preparation has no private storage"
        )
    destinations = tuple(tensor for _name, tensor in tensors)
    abi_identity = tuple(
        _tensor_abi_identity(tensor)
        for tensor in (q_weight, k_weight, v_weight, *destinations)
    )
    authenticate = bool(
        not workspace.d128_dual_qkv_weight_authenticated
        or workspace.d128_dual_qkv_weight_abi_identity != abi_identity
    )
    forward, backward = (
        b300_prepare_gqa_d128_qkv_projection_weight_dual_out(
            q_weight,
            k_weight,
            v_weight,
            *destinations,
            checked=authenticate,
            authenticate=authenticate,
        )
    )
    workspace.d128_dual_qkv_weight_authenticated = True
    workspace.d128_dual_qkv_weight_abi_identity = abi_identity
    _refresh_dual_weight_prep_authentication(workspace)
    return forward, backward


def _dual_output_weight_tensors(
    workspace: _LowpAttentionForwardWorkspace,
) -> tuple[tuple[str, torch.Tensor], ...]:
    fields = tuple(
        (name, getattr(workspace, name))
        for name in _DUAL_OUTPUT_WEIGHT_FIELDS
    )
    present = tuple(tensor is not None for _name, tensor in fields)
    if not any(present):
        return ()
    if not all(present):
        raise RuntimeError(
            "dual output-weight workspace must be fully allocated or absent"
        )
    return tuple(
        (name, tensor)
        for name, tensor in fields
        if tensor is not None
    )


def _prepare_direct_dual_output_weight(
    workspace: _LowpAttentionForwardWorkspace,
    out_weight: torch.Tensor,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """Refresh stable output-weight operands, authenticating first use."""
    tensors = _dual_output_weight_tensors(workspace)
    if len(tensors) != len(_DUAL_OUTPUT_WEIGHT_FIELDS):
        raise RuntimeError(
            "direct dual output-weight preparation has no private storage"
        )
    destinations = tuple(tensor for _name, tensor in tensors)
    abi_identity = tuple(
        _tensor_abi_identity(tensor)
        for tensor in (out_weight, *destinations)
    )
    authenticate = bool(
        not workspace.output_dual_weight_authenticated
        or workspace.output_dual_weight_abi_identity != abi_identity
    )
    forward, backward = b300_prepare_nvfp4_projection_weight_dual_out(
        out_weight,
        *destinations,
        checked=authenticate,
        authenticate=authenticate,
    )
    workspace.output_dual_weight_authenticated = True
    workspace.output_dual_weight_abi_identity = abi_identity
    _refresh_dual_weight_prep_authentication(workspace)
    return forward, backward


def _attention_weight_versions(
    weights: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
) -> tuple[int, int, int, int]:
    """Snapshot optimizer-visible Q/K/V/O parameter versions."""
    return tuple(int(weight._version) for weight in weights)


class _DualWeightPackLayerController:
    """Publish one layer's stable dual-weight operands on a private stream."""

    def __init__(
        self,
        *,
        layer_index: int,
        workspace: _LowpAttentionForwardWorkspace,
        weights: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        producer_stream: Any,
        consumer_stream: Any,
    ) -> None:
        self.layer_index = layer_index
        self.workspace = workspace
        self.producer_stream = producer_stream
        self.device = weights[0].device
        if self.device.type != "cuda":
            raise RuntimeError("rolling dual-weight preparation requires CUDA")
        if any(weight.device != self.device for weight in weights):
            raise RuntimeError("attention projection weights span CUDA devices")
        self.consumer_stream_id = int(consumer_stream.cuda_stream)
        self.producer_stream_id = int(producer_stream.cuda_stream)
        if self.consumer_stream_id == self.producer_stream_id:
            raise RuntimeError(
                "rolling dual-weight producer must use a private CUDA stream"
            )
        if workspace.cuda_stream != self.consumer_stream_id:
            raise RuntimeError(
                "rolling dual-weight consumer does not own the workspace "
                f"stream: {self.consumer_stream_id} != "
                f"{workspace.cuda_stream}"
            )
        if workspace.dual_weight_pack_controller is not None:
            raise RuntimeError(
                "low-precision attention workspace already has a rolling "
                "dual-weight controller"
            )
        self._bound_weights = weights
        self._bound_qkv_destinations = tuple(
            tensor
            for _name, tensor in _d128_dual_qkv_weight_tensors(workspace)
        )
        self._bound_output_destinations = tuple(
            tensor
            for _name, tensor in _dual_output_weight_tensors(workspace)
        )
        expected_destinations = len(
            _D128_DUAL_QKV_WEIGHT_FIELDS + _DUAL_OUTPUT_WEIGHT_FIELDS
        )
        if (
            len(self._bound_qkv_destinations)
            + len(self._bound_output_destinations)
            != expected_destinations
        ):
            raise RuntimeError(
                "rolling dual-weight preparation requires complete QKV/O "
                "destination storage"
            )
        self.qkv_ready_event = torch.cuda.Event(
            enable_timing=False,
            blocking=False,
            interprocess=False,
        )
        self.output_ready_event = torch.cuda.Event(
            enable_timing=False,
            blocking=False,
            interprocess=False,
        )
        self.backward_consumed_event = torch.cuda.Event(
            enable_timing=False,
            blocking=False,
            interprocess=False,
        )
        self.state = _DualWeightPackPublicationState()
        self._record_tensor_lifetimes()
        workspace.dual_weight_pack_controller = self

    def _record_tensor_lifetimes(self) -> None:
        for tensor in (
            *self._bound_weights,
            *self._bound_qkv_destinations,
            *self._bound_output_destinations,
        ):
            tensor.record_stream(self.producer_stream)

    def _require_bound_weights(
        self,
        weights: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> None:
        if len(weights) != len(self._bound_weights) or any(
            current is not bound
            for current, bound in zip(
                weights,
                self._bound_weights,
                strict=True,
            )
        ):
            raise RuntimeError(
                "rolling dual-weight preparation received replacement Q/K/V/O "
                "tensor objects"
            )

    def _require_bound_objects(
        self,
        weights: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> None:
        self._require_bound_weights(weights)
        for names, destinations in (
            (_D128_DUAL_QKV_WEIGHT_FIELDS, self._bound_qkv_destinations),
            (_DUAL_OUTPUT_WEIGHT_FIELDS, self._bound_output_destinations),
        ):
            for name, bound in zip(names, destinations, strict=True):
                if getattr(self.workspace, name) is not bound:
                    raise RuntimeError(
                        "rolling dual-weight preparation received a "
                        f"replacement destination tensor for {name}"
                    )

    def _authenticated_abi_matches(
        self,
        weights: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> bool:
        q_weight, k_weight, v_weight, out_weight = weights
        qkv_identity = tuple(
            _tensor_abi_identity(tensor)
            for tensor in (
                q_weight,
                k_weight,
                v_weight,
                *self._bound_qkv_destinations,
            )
        )
        output_identity = tuple(
            _tensor_abi_identity(tensor)
            for tensor in (out_weight, *self._bound_output_destinations)
        )
        return bool(
            self.workspace.d128_dual_qkv_weight_authenticated
            and self.workspace.output_dual_weight_authenticated
            and self.workspace.d128_dual_qkv_weight_abi_identity
            == qkv_identity
            and self.workspace.output_dual_weight_abi_identity
            == output_identity
        )

    def authenticate(
        self,
        weights: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> None:
        """Run checked byte authentication before the first model forward."""
        self._require_consumer_stream()
        if self.state.generation >= 0:
            raise RuntimeError(
                "dual-weight authentication must precede the first forward"
            )
        # This composite is consumed by the LBT rolling scheduler. Clear it
        # before either checked producer so a partial or failed refresh cannot
        # inherit an earlier successful aggregate state.
        self.workspace.weight_prep_authenticated = False
        self.workspace.d128_dual_qkv_weight_authenticated = False
        self.workspace.output_dual_weight_authenticated = False
        self._require_bound_objects(weights)
        q_weight, k_weight, v_weight, out_weight = weights
        _prepare_direct_d128_dual_qkv_weight(
            self.workspace,
            q_weight,
            k_weight,
            v_weight,
        )
        _prepare_direct_dual_output_weight(
            self.workspace,
            out_weight,
        )
        authenticated = self._authenticated_abi_matches(weights)
        self.workspace.weight_prep_authenticated = authenticated
        if not authenticated:
            raise RuntimeError(
                "rolling dual-weight authentication did not bind the current "
                "Q/K/V/O and destination ABI"
            )

    def begin(
        self,
        generation: int,
        weight_versions: tuple[int, int, int, int],
    ) -> None:
        self.state.begin(generation, weight_versions)

    def enqueue(
        self,
        weights: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        *,
        generation: int,
        weight_versions: tuple[int, int, int, int],
    ) -> None:
        if not self.workspace.weight_prep_authenticated:
            raise RuntimeError(
                "rolling dual-weight preparation requires activation-time "
                "byte authentication"
            )
        if not (
            self.workspace.d128_dual_qkv_weight_authenticated
            and self.workspace.output_dual_weight_authenticated
        ):
            raise RuntimeError(
                "rolling dual-weight preparation lost an authenticated "
                "producer"
            )
        # Full pointer/shape/stride/dtype/device ABI authentication occurs
        # once above, outside measured steps. Steady state retains those exact
        # objects and checks only object identity plus Parameter versions.
        self._require_bound_objects(weights)
        self.state._require_current(generation, weight_versions)
        if self.state.qkv_published or self.state.output_published:
            raise RuntimeError(
                f"decoder layer {self.layer_index} was scheduled twice"
            )
        if _attention_weight_versions(weights) != weight_versions:
            raise RuntimeError(
                "attention weights changed after the model-forward version "
                "snapshot"
            )
        if generation > 0:
            self.producer_stream.wait_event(self.backward_consumed_event)
        q_weight, k_weight, v_weight, out_weight = weights
        with torch.cuda.stream(self.producer_stream):
            b300_prepare_gqa_d128_qkv_projection_weight_dual_out(
                q_weight,
                k_weight,
                v_weight,
                *self._bound_qkv_destinations,
                checked=False,
                authenticate=False,
            )
            self.qkv_ready_event.record(self.producer_stream)
            self.state.publish_qkv(generation, weight_versions)
            b300_prepare_nvfp4_projection_weight_dual_out(
                out_weight,
                *self._bound_output_destinations,
                checked=False,
                authenticate=False,
            )
            self.output_ready_event.record(self.producer_stream)
            self.state.publish_output(generation, weight_versions)

    def consume_qkv(
        self,
        weights: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        int,
        tuple[int, int, int, int],
    ]:
        consumer_stream = self._require_consumer_stream()
        self._require_bound_weights(weights)
        generation = self.state.generation
        weight_versions = _attention_weight_versions(weights)
        self.state.consume_qkv(generation, weight_versions)
        consumer_stream.wait_event(self.qkv_ready_event)
        (
            forward_packed,
            forward_scales,
            backward_packed,
            backward_scales,
            global_scale,
        ) = self._bound_qkv_destinations
        return (
            (
                forward_packed,
                forward_scales,
                global_scale,
            ),
            (
                backward_packed,
                backward_scales,
                global_scale,
            ),
            generation,
            weight_versions,
        )

    def consume_output(
        self,
        weights: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        *,
        generation: int,
        weight_versions: tuple[int, int, int, int],
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        consumer_stream = self._require_consumer_stream()
        self._require_bound_weights(weights)
        if _attention_weight_versions(weights) != weight_versions:
            raise RuntimeError("attention weights changed during forward")
        self.state.consume_output(generation, weight_versions)
        consumer_stream.wait_event(self.output_ready_event)
        (
            forward_packed,
            forward_scales,
            backward_packed,
            backward_scales,
            global_scale,
        ) = self._bound_output_destinations
        return (
            (
                forward_packed,
                forward_scales,
                global_scale,
            ),
            (
                backward_packed,
                backward_scales,
                global_scale,
            ),
        )

    def enqueue_backward_consumed(
        self,
        *,
        generation: int,
        weight_versions: tuple[int, int, int, int],
    ) -> None:
        consumer_stream = self._require_consumer_stream()
        self.state._require_current(generation, weight_versions)
        if not self.state.qkv_consumed or not self.state.output_consumed:
            raise RuntimeError(
                "cannot release an incompletely consumed dual-weight forward"
            )
        if self.state.backward_enqueued:
            raise RuntimeError("dual-weight backward was enqueued twice")
        self.backward_consumed_event.record(consumer_stream)
        self.state.enqueue_backward(generation, weight_versions)

    def release_without_backward(
        self,
        *,
        generation: int,
        weight_versions: tuple[int, int, int, int],
    ) -> None:
        """Fence reuse after an explicit no-grad forward."""
        consumer_stream = self._require_consumer_stream()
        self.state._require_current(generation, weight_versions)
        if not self.state.qkv_consumed or not self.state.output_consumed:
            raise RuntimeError(
                "cannot release an incompletely consumed no-grad "
                "dual-weight forward"
            )
        if self.state.backward_enqueued:
            raise RuntimeError("dual-weight no-grad release was enqueued twice")
        self.backward_consumed_event.record(consumer_stream)
        self.state.release_without_backward(generation, weight_versions)

    def require_can_begin(self, generation: int) -> None:
        self.state.require_can_begin(generation)

    def require_bound_consumer_stream(self, stream_id: int) -> None:
        if stream_id != self.consumer_stream_id:
            raise RuntimeError(
                "rolling dual-weight consumer stream changed: "
                f"{stream_id} != {self.consumer_stream_id}"
            )

    def detach(self) -> None:
        if self.workspace.dual_weight_pack_controller is self:
            self.workspace.dual_weight_pack_controller = None

    def _require_consumer_stream(self) -> Any:
        stream = torch.cuda.current_stream(self.device)
        self.require_bound_consumer_stream(int(stream.cuda_stream))
        if self.workspace.cuda_stream != self.consumer_stream_id:
            raise RuntimeError(
                "rolling dual-weight workspace stream binding changed"
            )
        return stream


def _require_forward_workspace_same_stream(
    workspace: _LowpAttentionForwardWorkspace,
    tensor: torch.Tensor,
    *,
    phase: str,
) -> None:
    """Require every publication and consumer to use the bound CUDA stream."""
    device = workspace.outputs.q_payload.device
    if tensor.device != device:
        raise RuntimeError(
            "low-precision attention workspace device mismatch during "
            f"{phase}: tensor is on {tensor.device}, workspace is on {device}"
        )
    if device.type != "cuda":
        if workspace.cuda_stream is not None:
            raise RuntimeError(
                "non-CUDA low-precision attention workspace unexpectedly "
                "records a CUDA stream"
            )
        return
    if workspace.cuda_stream is None:
        raise RuntimeError(
            "CUDA low-precision attention workspace has no bound stream"
        )
    current_stream = int(torch.cuda.current_stream(device).cuda_stream)
    if current_stream != workspace.cuda_stream:
        raise RuntimeError(
            "low-precision attention workspace stream mismatch during "
            f"{phase}: current stream {current_stream}, bound stream "
            f"{workspace.cuda_stream}"
        )


class _LowpAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        attention_norm_weight: torch.Tensor,
        packed_qkv_weight: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        v_weight: torch.Tensor,
        out_weight: torch.Tensor,
        qk_scales: torch.Tensor,
        forward_workspace: _LowpAttentionForwardWorkspace,
        runtime: LowpAttentionRuntime,
    ) -> torch.Tensor:
        c = runtime.config
        publication_generation = (
            forward_workspace.publication_state.current_generation
        )
        if publication_generation < 0:
            raise RuntimeError(
                "low-precision attention custom forward must be entered "
                "through its generation-guarded module"
            )
        token_rows = c.batch * c.sequence
        raw_rows = x.reshape(token_rows, c.hidden).contiguous()
        fused_attention_rmsnorm = (
            runtime.experimental_fused_attention_rmsnorm_nvfp4
        )
        rows = raw_rows
        use_direct_d128_dual_qkv_weight = (
            _uses_direct_d128_dual_qkv_weight_prep(runtime)
        )
        use_direct_dual_out_weight = (
            _uses_direct_dual_output_weight_prep(runtime)
        )
        dual_weight_pack_controller = (
            forward_workspace.dual_weight_pack_controller
        )
        if dual_weight_pack_controller is not None and not (
            use_direct_d128_dual_qkv_weight
            and use_direct_dual_out_weight
        ):
            raise RuntimeError(
                "rolling dual-weight controller was attached to an "
                "ineligible attention route"
            )
        attention_weights = (
            q_weight,
            k_weight,
            v_weight,
            out_weight,
        )
        dual_weight_pack_generation = None
        dual_weight_pack_weight_versions = None
        qkv_weight_backward_operand = None
        if c.head_dim == 64:
            # D64 owns the projection in the exact canonical row order the
            # kernels consume. This is a direct Parameter alias: no timed
            # Q/K/V concatenation or dispatcher-side publication is needed.
            qkv_weight = packed_qkv_weight
        elif use_direct_d128_dual_qkv_weight:
            # The direct producer reads canonical Q/K/V Parameters and emits
            # both physical orientations. No BF16 aggregate exists on this
            # true-2D native-NVFP4 route.
            qkv_weight = forward_workspace.outputs.empty_bf16
        else:
            # Non-native or non-2D experiments retain their established BF16
            # pair-interleave/concatenate composition unchanged.
            with _stage("lowp/fwd/qkv_weight_pair_interleave_concat"):
                qkv_weight = _stack_lowp_qkv_weights(
                    c,
                    q_weight,
                    k_weight,
                    v_weight,
                )
        prepare_weight = (
            b300_prepare_nvfp4_projection_weight
            if runtime.projection_weight_scale_2d
            else b300_prepare_nvfp4_projection_operand
        )
        if runtime.qkv_projection_format == "e4m3":
            with _stage("lowp/fwd/input_e4m3_rowwise_pack"):
                rows_operand = tuple(
                    b300_prepare_e4m3_projection_operand(rows)
                )
            # Parameters are mutated in-place by AdamW.  Reprepare the current
            # value on every invocation rather than caching by tensor identity.
            with _stage("lowp/fwd/qkv_weight_e4m3_channelwise_pack"):
                qkv_weight_operand = tuple(
                    b300_prepare_e4m3_projection_weight(qkv_weight)
                )
        else:
            if fused_attention_rmsnorm:
                with _stage("lowp/fwd/attention_rmsnorm_nvfp4_pack"):
                    prepared_rows = tuple(
                        b300_prepare_nvfp4_projection_operand_rmsnorm(
                            raw_rows,
                            attention_norm_weight,
                            c.rms_epsilon,
                        )
                    )
                    rows_operand = prepared_rows[:3]
                    inv_rms = prepared_rows[3]
                    rows = prepared_rows[4]
            else:
                with _stage("lowp/fwd/input_nvfp4_pack"):
                    rows_operand = tuple(
                        b300_prepare_nvfp4_projection_operand(rows)
                    )
            with _stage("lowp/fwd/qkv_weight_nvfp4_pack"):
                if use_direct_d128_dual_qkv_weight:
                    if dual_weight_pack_controller is not None:
                        (
                            qkv_weight_operand,
                            qkv_weight_backward_operand,
                            dual_weight_pack_generation,
                            dual_weight_pack_weight_versions,
                        ) = dual_weight_pack_controller.consume_qkv(
                            attention_weights
                        )
                    else:
                        (
                            qkv_weight_operand,
                            qkv_weight_backward_operand,
                        ) = _prepare_direct_d128_dual_qkv_weight(
                            forward_workspace,
                            q_weight,
                            k_weight,
                            v_weight,
                        )
                else:
                    qkv_weight_operand = tuple(prepare_weight(qkv_weight))
        with _stage("lowp/fwd/qkv_projection_rope_publish"):
            if runtime.qkv_projection is not None:
                qkv = runtime.qkv_projection(
                    rows_operand,
                    qkv_weight_operand,
                    qk_scales,
                    runtime.paired_rope,
                    forward_workspace=forward_workspace.outputs,
                )
            else:
                if c.head_dim == 128:
                    qkv = b300_project_qkv_gqa_d128_unified_lowp_nvfp4(
                        rows_operand,
                        qkv_weight_operand,
                        qk_scales,
                        batch=c.batch,
                        seqlen=c.sequence,
                        q_heads=c.q_heads,
                        kv_heads=c.kv_heads,
                        store_bf16=False,
                        publish_fp8_backward=True,
                        v_mxfp4_scale_2d=runtime.v_mxfp4_scale_2d,
                        per_block_qk_scales=runtime.per_block_qk_scales,
                        rope_packed=runtime.paired_rope,
                    )
                else:
                    qkv = b300_project_qkv_gqa_d64_paired_unified_lowp_nvfp4(
                        rows_operand,
                        qkv_weight_operand,
                        qk_scales,
                        runtime.paired_rope,
                        batch=c.batch,
                        seqlen=c.sequence,
                        q_heads=c.q_heads,
                        kv_heads=c.kv_heads,
                        store_bf16=False,
                        publish_fp8_backward=True,
                        interleave_causal_kv=bool(
                            runtime.forward_topology.get(
                                "causal_interleaved_kv", False
                            )
                        ),
                        v_mxfp4_scale_2d=runtime.v_mxfp4_scale_2d,
                    )
        output = torch.empty(
            c.batch,
            c.sequence,
            c.q_heads,
            c.head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        lse = torch.empty(
            c.batch,
            c.q_heads,
            1,
            c.sequence,
            device="cuda",
            dtype=torch.float32,
        )
        forward_mx_probability_scales = None
        if runtime.backward_forward_mx_probability_scale_handoff:
            forward_mx_probability_scales = torch.empty(
                c.batch,
                c.q_heads,
                c.sequence // 128,
                c.sequence,
                device="cuda",
                dtype=torch.int32,
            )
        with _stage("lowp/fwd/attention"):
            _run_lowp_forward_attention(
                runtime,
                qkv,
                output,
                lse,
                forward_mx_probability_scales,
            )
        if runtime.diagnostic_fp8_lse_entrypoint is not None:
            with _stage("lowp/fwd/diagnostic_fp8_lse_control"):
                lse = runtime.diagnostic_fp8_lse(qkv, output, lse)
        if runtime.forward_diagnostic_sink is not None:
            runtime.forward_diagnostic_sink.append(
                {
                    "qk_policy_scales": qk_scales,
                    "backward_qk_scales": qkv.backward.qk_scales,
                    "q_forward_scales": qkv.q_forward_scales,
                    "q_forward_global_scale": qkv.q_forward_global_scale,
                    "k_forward_scales": qkv.k_forward_scales,
                    "k_forward_global_scale": qkv.k_forward_global_scale,
                    "v_forward_scales": qkv.v_forward_scales,
                    "q_backward_fp8": qkv.q_backward_fp8,
                    "k_backward_fp8": qkv.k_backward_fp8,
                    "v_backward_fp8": qkv.v_backward_fp8,
                    "attention_output": output,
                    "lse": lse,
                    "forward_mx_probability_scales": (
                        forward_mx_probability_scales
                    ),
                }
            )
        output_matrix = output.reshape(token_rows, c.q_width)
        out_weight_backward_operand = None
        if runtime.output_projection_format == "e4m3":
            with _stage("lowp/fwd/output_e4m3_rowwise_pack"):
                output_operand = tuple(
                    b300_prepare_e4m3_projection_operand(output_matrix)
                )
            with _stage(
                "lowp/fwd/output_weight_e4m3_forward_nvfp4_backward_pack"
            ):
                # The first D128 correctness canary allocates both publications.
                # Forward consumes only E4M3; the retained dX projection consumes
                # only this NVFP4 physical transpose. No unused NVFP4 forward-O
                # operand is prepared.
                out_weight_operand = tuple(
                    b300_prepare_e4m3_projection_weight(out_weight)
                )
                out_weight_backward_operand = tuple(
                    b300_prepare_nvfp4_projection_weight(
                        out_weight.T.contiguous()
                    )
                )
            with _stage("lowp/fwd/output_projection_e4m3"):
                projected = b300_project_e4m3(
                    output_operand,
                    out_weight_operand,
                )
        else:
            with _stage("lowp/fwd/output_nvfp4_pack"):
                output_operand = tuple(
                    b300_prepare_nvfp4_projection_operand(output_matrix)
                )
            if use_direct_dual_out_weight:
                with _stage("lowp/fwd/output_weight_nvfp4_dual_pack"):
                    if dual_weight_pack_controller is not None:
                        if (
                            dual_weight_pack_generation is None
                            or dual_weight_pack_weight_versions is None
                        ):
                            raise RuntimeError(
                                "rolling O publication has no matching QKV "
                                "generation"
                            )
                        (
                            out_weight_operand,
                            out_weight_backward_operand,
                        ) = dual_weight_pack_controller.consume_output(
                            attention_weights,
                            generation=dual_weight_pack_generation,
                            weight_versions=dual_weight_pack_weight_versions,
                        )
                    else:
                        (
                            out_weight_operand,
                            out_weight_backward_operand,
                        ) = _prepare_direct_dual_output_weight(
                            forward_workspace,
                            out_weight,
                        )
            else:
                with _stage("lowp/fwd/output_weight_nvfp4_pack"):
                    out_weight_operand = tuple(
                        prepare_weight(out_weight)
                    )
            with _stage("lowp/fwd/output_projection_nvfp4"):
                projected = b300_project_nvfp4(
                    output_operand,
                    out_weight_operand,
                )
        assert qkv.q_backward_fp8 is not None
        assert qkv.k_backward_fp8 is not None
        if runtime.experimental_d128_mxfp4_v_backward:
            saved_v_operands = qkv.mxfp4_backward_v_operands(
                required_scale_policy=(
                    runtime.d128_mxfp4_v_scale_policy
                )
            )
        else:
            assert qkv.v_backward_fp8 is not None
            saved_v_operands = (qkv.v_backward_fp8,)
        saved_tensors = (
            rows,
            packed_qkv_weight,
            q_weight,
            k_weight,
            v_weight,
            out_weight,
            output,
            lse,
            qkv.q_backward_fp8,
            qkv.k_backward_fp8,
            *saved_v_operands,
        )
        if fused_attention_rmsnorm:
            ctx.save_for_backward(
                raw_rows,
                attention_norm_weight,
                inv_rms,
                *saved_tensors,
            )
        else:
            ctx.save_for_backward(*saved_tensors)
        # Both retained forward routes save the exact same runtime object for
        # backward. Route-natural saved tensors still differ, as they must,
        # but there is no route-specific backward wrapper or dispatch state.
        ctx.runtime = runtime.backward_execution_runtime
        ctx.fused_attention_rmsnorm = fused_attention_rmsnorm
        ctx.qkv_weight_backward_operand = qkv_weight_backward_operand
        ctx.output_weight_backward_operand = out_weight_backward_operand
        ctx.forward_mx_probability_scales = forward_mx_probability_scales
        ctx.forward_workspace = forward_workspace
        ctx.publication_generation = publication_generation
        ctx.dual_weight_pack_controller = dual_weight_pack_controller
        ctx.dual_weight_pack_generation = dual_weight_pack_generation
        ctx.dual_weight_pack_weight_versions = (
            dual_weight_pack_weight_versions
        )
        return projected.reshape_as(x)

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...]:
        forward_workspace: _LowpAttentionForwardWorkspace = (
            ctx.forward_workspace
        )
        publication_generation: int = ctx.publication_generation
        forward_workspace.publication_state.require_backward(
            publication_generation
        )
        _require_forward_workspace_same_stream(
            forward_workspace,
            grad_output,
            phase="backward",
        )
        saved_tensors = ctx.saved_tensors
        if ctx.fused_attention_rmsnorm:
            raw_rows, attention_norm_weight, inv_rms, *saved_tensors = (
                saved_tensors
            )
        else:
            raw_rows = None
            attention_norm_weight = None
            inv_rms = None
        runtime: LowpAttentionRuntime = ctx.runtime
        (
            rows,
            packed_qkv_weight,
            q_weight,
            k_weight,
            v_weight,
            out_weight,
            attention_output,
            lse,
            q_fp8,
            k_fp8,
            *saved_v_operands,
        ) = saved_tensors
        expected_v_operand_count = (
            2 if runtime.experimental_d128_mxfp4_v_backward else 1
        )
        if len(saved_v_operands) != expected_v_operand_count:
            raise RuntimeError(
                "saved backward V publication does not match the runtime "
                "representation"
            )
        v_backward = saved_v_operands[0]
        v_mxfp4_scale_pages = (
            saved_v_operands[1]
            if runtime.experimental_d128_mxfp4_v_backward
            else None
        )
        c = runtime.config
        token_rows = c.batch * c.sequence
        scale = 1.0 / runtime.loss_scale
        if c.head_dim == 64:
            runtime.backward.reset()
        with _stage("lowp/bwd/dy_scaled_nvfp4_pack"):
            dy_operand = tuple(
                b300_prepare_nvfp4_projection_operand_scaled(
                    grad_output.reshape(token_rows, c.hidden).contiguous(),
                    runtime.loss_scale,
                )
            )
        prepare_weight = (
            b300_prepare_nvfp4_projection_weight
            if runtime.projection_weight_scale_2d
            else b300_prepare_nvfp4_projection_operand
        )
        out_weight_backward_operand = ctx.output_weight_backward_operand
        if out_weight_backward_operand is None:
            with _stage(
                "lowp/bwd/output_weight_transpose_nvfp4_pack"
            ):
                out_weight_backward_operand = tuple(
                    prepare_weight(out_weight.T.contiguous())
                )
        with _stage("lowp/bwd/dout_projection_publish"):
            if runtime.native_tk_d128_v509_e5m2_dout_backward:
                dout_bundle = (
                    b300_project_dout_unified_lowp_nvfp4_v509_e5m2(
                        dy_operand,
                        out_weight_backward_operand,
                        attention_output,
                        lse,
                        stats_workspace=runtime.backward.workspace_torch,
                        dq_clear=runtime.backward.dq,
                    )
                )
                dout_backward = dout_bundle.dout_backward_e5m2
            else:
                dout_bundle = b300_project_dout_unified_lowp_nvfp4(
                    dy_operand,
                    out_weight_backward_operand,
                    attention_output,
                    lse,
                    batch=c.batch,
                    seqlen=c.sequence,
                    heads=c.q_heads,
                    store_bf16=False,
                    publish_fp8_backward=True,
                    publish_stats=True,
                    stats_workspace=runtime.backward.workspace_torch,
                    dq_clear=(
                        runtime.backward.dq if c.head_dim == 128 else None
                    ),
                    probability_log2_lift=(
                        8.0 if runtime.native_tk_d128_backward else 0.0
                    ),
                )
                assert dout_bundle.dout_backward_fp8 is not None
                dout_backward = dout_bundle.dout_backward_fp8
        with _stage("lowp/bwd/attention"):
            runtime.bind_backward_inputs(
                q_fp8,
                k_fp8,
                v_backward,
                dout_backward,
                ctx.forward_mx_probability_scales,
                v_mxfp4_scale_pages=v_mxfp4_scale_pages,
                producer_workspace=(
                    forward_workspace.outputs
                    if runtime.d128_mxfp4_v_scale_policy
                    == MXFP4_V_SCALE_POLICY_SHARED_D32XS32
                    else None
                ),
                native_score_workspace=(
                    forward_workspace.outputs
                    if runtime.native_tk_d128_native_score_backward
                    else None
                ),
            )
            if runtime.native_tk_d128_v509_e5m2_dout_backward:
                runtime.backward.run_publisher_precleared_dq(reset=False)
            else:
                runtime.backward.run(reset=False)

        fused_d128_weight_gradient = (
            c.head_dim == 128 and runtime.projection_dgrad == "nvfp4"
        )
        with _stage("lowp/bwd/inverse_rope_decode_scale"):
            if fused_d128_weight_gradient:
                combined_gradient = (
                    b300_stitch_gqa_d128_inverse_rope_gradient(
                        runtime.backward.dq,
                        runtime.backward.dk,
                        runtime.backward.dv,
                        *runtime.rope,
                        q_gradient_scale=(
                            scale * 0.25 * runtime.backward_q_gain
                        ),
                        k_gradient_scale=(
                            scale * 0.25 * runtime.backward_k_gain
                        ),
                        v_gradient_scale=(
                            scale * 0.25 * runtime.backward_v_weight_gain
                        ),
                    )
                )
            elif c.head_dim == 128:
                dq_inverse = _inverse_rope_pair_native(
                    runtime.backward.dq,
                    *runtime.rope,
                )
                dk_inverse = _inverse_rope_pair_native(
                    runtime.backward.dk,
                    *runtime.rope,
                )
                dq_inverse.mul_(scale * 0.25 * runtime.backward_q_gain)
                dk_inverse.mul_(scale * 0.25 * runtime.backward_k_gain)
                dv_decoded = (
                    runtime.backward.dv.float()
                    .mul_(scale * 0.25 * runtime.backward_v_gain)
                    .bfloat16()
                )
                combined_gradient = torch.cat(
                    (
                        dq_inverse.reshape(token_rows, -1),
                        dk_inverse.reshape(token_rows, -1),
                        dv_decoded.reshape(token_rows, -1),
                    ),
                    dim=1,
                ).contiguous()
            else:
                combined_gradient = (
                    b300_stitch_gqa_d64_inverse_rope_gradient(
                        runtime.backward.dq,
                        runtime.backward.dk,
                        runtime.backward.dv,
                        *runtime.rope,
                        # The fixed-scale E4M3 dO publication is x4.
                        # Decode all three fields before projection.
                        q_gradient_scale=(
                            scale * 0.25 * runtime.backward_q_gain
                        ),
                        k_gradient_scale=(
                            scale * 0.25 * runtime.backward_k_gain
                        ),
                        v_gradient_scale=(
                            scale * 0.25 * runtime.backward_v_gain
                        ),
                    )
                )
        projection_weight_operand = ctx.qkv_weight_backward_operand
        if c.head_dim == 64:
            qkv_weight = packed_qkv_weight
        elif projection_weight_operand is None:
            with _stage("lowp/bwd/qkv_weight_pair_interleave_concat"):
                qkv_weight = _stack_lowp_qkv_weights(
                    c,
                    q_weight,
                    k_weight,
                    v_weight,
                )
        else:
            # The layer-private direct producer retained the physical
            # transpose consumed by D128 NVFP4 dgrad. No BF16 aggregate is
            # needed in this forward/backward pair.
            qkv_weight = None
        if runtime.projection_dgrad == "nvfp4":
            with _stage("lowp/bwd/qkv_dgrad_nvfp4"):
                if projection_weight_operand is None:
                    assert qkv_weight is not None
                    projection_weight_operand = tuple(
                        prepare_weight(qkv_weight.T.contiguous())
                    )
                project_dgrad = (
                    b300_project_gqa_d128_hierarchical_qkv_gradient_nvfp4
                    if c.head_dim == 128
                    else b300_project_gqa_d64_paired_qkv_gradient_nvfp4
                )
                dx_scaled = project_dgrad(
                    runtime.backward.dq,
                    runtime.backward.dk,
                    runtime.backward.dv,
                    projection_weight_operand,
                    runtime.gradient_global_scale,
                    runtime.paired_rope,
                    dq_decode_scale=(0.25 * runtime.backward_q_gain),
                    dk_decode_scale=(0.25 * runtime.backward_k_gain),
                    dv_decode_scale=(0.25 * runtime.backward_v_gain),
                )
                # The projection result is already BF16 and dead after this
                # rescale.  Scaling it in place produces the same BF16
                # rounding while avoiding an HBM-sized FP32 temporary and a
                # second dtype-conversion kernel.
                dx_matrix = dx_scaled.mul_(scale)
        else:
            with _stage("lowp/bwd/qkv_dgrad_bf16"):
                assert qkv_weight is not None
                dx_matrix = torch.mm(combined_gradient, qkv_weight)
        dattention_norm_weight = None
        if ctx.fused_attention_rmsnorm:
            assert raw_rows is not None
            assert attention_norm_weight is not None
            assert inv_rms is not None
            with _stage("lowp/bwd/attention_rmsnorm"):
                dx_matrix, dattention_norm_weight = b300_rmsnorm_backward(
                    raw_rows,
                    attention_norm_weight,
                    inv_rms,
                    dx_matrix,
                )
        dx = dx_matrix.reshape(c.batch, c.sequence, c.hidden)

        q_end = c.q_width
        k_end = q_end + c.kv_width
        assert combined_gradient is not None
        weight_gradient_input = combined_gradient
        if (
            not fused_d128_weight_gradient
            and runtime.backward_v_weight_gain != runtime.backward_v_gain
        ):
            weight_gradient_input = combined_gradient.clone()
            weight_gradient_input[:, k_end:].mul_(
                runtime.backward_v_weight_gain / runtime.backward_v_gain
            )
        with _stage("lowp/bwd/qkv_weight_gradient"):
            weight_gradient_lhs = weight_gradient_input.T
            if not fused_d128_weight_gradient:
                weight_gradient_lhs = weight_gradient_lhs.contiguous()
            qkv_weight_gradient = torch.mm(
                weight_gradient_lhs,
                rows,
            )
        if c.head_dim == 64:
            # The physical D64 projection rows are canonical Q/K/V rows, so
            # its one weight-gradient GEMM is already the logical gradient of
            # the packed leaf Parameter.
            dpacked_qkv_weight = qkv_weight_gradient
            dq_weight = None
            dk_weight = None
            dv_weight = None
        else:
            dpacked_qkv_weight = None
            dq_weight = qkv_weight_gradient[:q_end]
            dk_weight = qkv_weight_gradient[q_end:k_end]
            dv_weight = qkv_weight_gradient[k_end:]
            # The projection consumes pair-interleaved physical Q/K rows,
            # while model parameters retain the standard split-half layout.
            dq_weight = _deinterleave_d128_weight_gradient(
                dq_weight,
                c.q_heads,
            )
            dk_weight = _deinterleave_d128_weight_gradient(
                dk_weight,
                c.kv_heads,
            )
        with _stage("lowp/bwd/output_weight_gradient"):
            dout_weight = torch.mm(
                grad_output.reshape(token_rows, c.hidden).T,
                attention_output.reshape(token_rows, c.q_width),
            )
        result = (
            dx,
            dattention_norm_weight,
            dpacked_qkv_weight,
            dq_weight,
            dk_weight,
            dv_weight,
            dout_weight,
            None,
            None,
            None,
        )
        dual_weight_pack_controller = ctx.dual_weight_pack_controller
        if dual_weight_pack_controller is not None:
            dual_weight_pack_generation = ctx.dual_weight_pack_generation
            dual_weight_pack_weight_versions = (
                ctx.dual_weight_pack_weight_versions
            )
            if (
                dual_weight_pack_generation is None
                or dual_weight_pack_weight_versions is None
            ):
                raise RuntimeError(
                    "rolling dual-weight backward is missing forward "
                    "provenance"
                )
            dual_weight_pack_controller.enqueue_backward_consumed(
                generation=dual_weight_pack_generation,
                weight_versions=dual_weight_pack_weight_versions,
            )
        # All consumers of the caller-owned Q/K/V and dual-weight
        # publications have now been enqueued on the authenticated stream.
        # Clear ownership only on successful completion; any exception leaves
        # the workspace blocked rather than risking a silent overwrite.
        forward_workspace.publication_state.finish_backward(
            publication_generation
        )
        return result


class LowpAttention(nn.Module):
    def __init__(self, config: Config, runtime: LowpAttentionRuntime) -> None:
        super().__init__()
        self.qkv_layout = packed_qkv_layout(config)
        self.qkv_master_parameter_layout = (
            lowp_qkv_master_parameter_layout(config)
        )
        if self.qkv_master_parameter_layout == PACKED_D64_LOWP_QKV_LAYOUT:
            self.weights = PackedQKVAttentionWeights(
                self.qkv_layout,
                device="cuda",
                dtype=torch.bfloat16,
            )
        else:
            self.weights = AttentionWeights(config)
        self.runtime = runtime
        # Scale state belongs to the layer whose projected Q/K distribution
        # it describes.  A single runtime-wide tensor silently coupled all 16
        # layers and made delayed/adaptive scaling impossible.
        # This is quantizer policy state derived from the current projection
        # weights, not a learned model tensor.  Keeping it non-persistent lets
        # an ordinary BF16 checkpoint load strictly into either route.
        self.register_buffer(
            "qk_scales", runtime.qk_scales.clone(), persistent=False
        )
        # Projection publications are deliberately private tensors, not
        # registered buffers: DDP broadcasts registered buffers before every
        # forward. The opaque non-Tensor carrier also keeps these mutable
        # publication targets out of custom-autograd Tensor inputs. Every
        # layer owns the superset needed by both retained routes so runtime
        # crossover never changes allocator topology.
        self._forward_workspace = self._allocate_forward_workspace()
        if runtime.adaptive_qk_weight_scales:
            self.refresh_qk_quant_scales()

    def _allocate_forward_workspace(self) -> _LowpAttentionForwardWorkspace:
        config = self.runtime.config
        device = self.weights.o.device
        q_payload = torch.empty(
            config.batch,
            config.q_heads,
            config.sequence,
            config.head_dim // 2,
            device=device,
            dtype=torch.uint8,
        )
        k_payload = torch.empty(
            config.batch,
            config.kv_heads,
            config.sequence,
            config.head_dim // 2,
            device=device,
            dtype=torch.uint8,
        )
        q_scale_pages = torch.empty(
            config.batch,
            config.sequence // 128,
            config.q_heads * (2 if config.head_dim == 128 else 1),
            512,
            device=device,
            dtype=torch.float8_e4m3fn,
        )
        q_global_scale = torch.empty(
            config.batch,
            config.q_heads,
            device=device,
            dtype=torch.float32,
        )
        k_scale_pages = torch.empty(
            config.batch,
            config.sequence // 64,
            config.kv_heads * (2 if config.head_dim == 128 else 1),
            512,
            device=device,
            dtype=torch.float8_e4m3fn,
        )
        k_global_scale = torch.empty(
            config.batch,
            config.kv_heads,
            device=device,
            dtype=torch.float32,
        )
        v_mxfp4_payload = torch.empty(
            config.batch,
            config.kv_heads,
            config.head_dim,
            config.sequence // 2,
            device=device,
            dtype=torch.float4_e2m1fn_x2,
        )
        v_mxfp4_scale_pages = torch.empty(
            config.batch,
            config.sequence // 128,
            config.kv_heads,
            512,
            device=device,
            dtype=torch.float8_e4m3fn,
        )
        v_fp8_payload = torch.empty(
            config.batch,
            config.kv_heads,
            config.head_dim,
            config.sequence,
            device=device,
            dtype=torch.float8_e4m3fn,
        )
        v_backward_fp8 = torch.empty(
            config.batch,
            config.sequence,
            config.kv_heads,
            config.head_dim,
            device=device,
            dtype=torch.float8_e4m3fn,
        )
        q_backward_fp8 = torch.empty(
            config.batch,
            config.sequence,
            config.q_heads,
            config.head_dim,
            device=device,
            dtype=torch.float8_e4m3fn,
        )
        k_backward_fp8 = torch.empty(
            config.batch,
            config.sequence,
            config.kv_heads,
            config.head_dim,
            device=device,
            dtype=torch.float8_e4m3fn,
        )
        if self.runtime.experimental_d128_mxfp4_v_backward:
            v_backward_mxfp4 = torch.empty(
                config.batch,
                config.sequence,
                config.kv_heads,
                config.head_dim // 2,
                device=device,
                dtype=torch.uint8,
            )
            v_backward_mxfp4_scale_pages = torch.empty(
                config.batch,
                config.sequence // 128,
                config.kv_heads,
                512,
                device=device,
                dtype=torch.uint8,
            )
        else:
            v_backward_mxfp4 = None
            v_backward_mxfp4_scale_pages = None
        # Construct the typed Q/K aliases once. The retained attention path
        # consumes these directly and performs no timed dtype view or narrow.
        q_payload_fp4 = q_payload.view(torch.float4_e2m1fn_x2)
        k_payload_fp4 = k_payload.view(torch.float4_e2m1fn_x2)
        outputs = B300E4M3QKVForwardWorkspace(
            q_payload=q_payload,
            k_payload=k_payload,
            q_scale_pages=q_scale_pages,
            q_global_scale=q_global_scale,
            k_scale_pages=k_scale_pages,
            k_global_scale=k_global_scale,
            v_mxfp4_payload=v_mxfp4_payload,
            v_mxfp4_scale_pages=v_mxfp4_scale_pages,
            v_fp8_payload=v_fp8_payload,
            v_backward_fp8=v_backward_fp8,
            q_backward_fp8=q_backward_fp8,
            k_backward_fp8=k_backward_fp8,
            q_payload_fp4=q_payload_fp4,
            k_payload_fp4=k_payload_fp4,
            empty_bf16=torch.empty(
                (0,), device=device, dtype=torch.bfloat16
            ),
            empty_byte=torch.empty(
                (0,), device=device, dtype=torch.uint8
            ),
            empty_fp8=torch.empty(
                (0,), device=device, dtype=torch.float8_e4m3fn
            ),
            empty_fp4=torch.empty(
                (0,), device=device, dtype=torch.float4_e2m1fn_x2
            ),
            v_backward_mxfp4=v_backward_mxfp4,
            v_backward_mxfp4_scale_pages=(
                v_backward_mxfp4_scale_pages
            ),
        )
        if _uses_direct_d128_dual_qkv_weight_prep(self.runtime):
            qkv_rows = config.q_width + 2 * config.kv_width
            qkv_weight_forward_packed = torch.empty(
                qkv_rows,
                config.hidden // 2,
                device=device,
                dtype=torch.float4_e2m1fn_x2,
            )
            qkv_weight_forward_scales = torch.empty(
                qkv_rows // 128,
                config.hidden // 64,
                512,
                device=device,
                dtype=torch.float8_e4m3fn,
            )
            qkv_weight_backward_packed = torch.empty(
                config.hidden,
                qkv_rows // 2,
                device=device,
                dtype=torch.float4_e2m1fn_x2,
            )
            qkv_weight_backward_scales = torch.empty(
                config.hidden // 128,
                qkv_rows // 64,
                512,
                device=device,
                dtype=torch.float8_e4m3fn,
            )
            qkv_weight_global_scale = torch.empty(
                1,
                device=device,
                dtype=torch.float32,
            )
        else:
            qkv_weight_forward_packed = None
            qkv_weight_forward_scales = None
            qkv_weight_backward_packed = None
            qkv_weight_backward_scales = None
            qkv_weight_global_scale = None
        if _uses_direct_dual_output_weight_prep(self.runtime):
            output_weight_forward_packed = torch.empty(
                config.hidden,
                config.q_width // 2,
                device=device,
                dtype=torch.float4_e2m1fn_x2,
            )
            output_weight_forward_scales = torch.empty(
                config.hidden // 128,
                config.q_width // 64,
                512,
                device=device,
                dtype=torch.float8_e4m3fn,
            )
            output_weight_backward_packed = torch.empty(
                config.q_width,
                config.hidden // 2,
                device=device,
                dtype=torch.float4_e2m1fn_x2,
            )
            output_weight_backward_scales = torch.empty(
                config.q_width // 128,
                config.hidden // 64,
                512,
                device=device,
                dtype=torch.float8_e4m3fn,
            )
            output_weight_global_scale = torch.empty(
                1,
                device=device,
                dtype=torch.float32,
            )
        else:
            output_weight_forward_packed = None
            output_weight_forward_scales = None
            output_weight_backward_packed = None
            output_weight_backward_scales = None
            output_weight_global_scale = None
        cuda_stream = (
            int(torch.cuda.current_stream(device).cuda_stream)
            if device.type == "cuda"
            else None
        )
        owner_tensors = {
            name: getattr(outputs, name)
            for name in _FORWARD_WORKSPACE_OWNER_SLOTS
        }
        owner_tensors.update(
            {
                name: tensor
                for name in _FORWARD_WORKSPACE_OPTIONAL_OWNERS
                if (tensor := getattr(outputs, name)) is not None
            }
        )
        owner_tensors.update(
            {
                name: tensor
                for name, tensor in (
                    (
                        "qkv_weight_forward_packed",
                        qkv_weight_forward_packed,
                    ),
                    (
                        "qkv_weight_forward_scales",
                        qkv_weight_forward_scales,
                    ),
                    (
                        "qkv_weight_backward_packed",
                        qkv_weight_backward_packed,
                    ),
                    (
                        "qkv_weight_backward_scales",
                        qkv_weight_backward_scales,
                    ),
                    ("qkv_weight_global_scale", qkv_weight_global_scale),
                    (
                        "output_weight_forward_packed",
                        output_weight_forward_packed,
                    ),
                    (
                        "output_weight_forward_scales",
                        output_weight_forward_scales,
                    ),
                    (
                        "output_weight_backward_packed",
                        output_weight_backward_packed,
                    ),
                    (
                        "output_weight_backward_scales",
                        output_weight_backward_scales,
                    ),
                    (
                        "output_weight_global_scale",
                        output_weight_global_scale,
                    ),
                )
                if tensor is not None
            }
        )
        return _LowpAttentionForwardWorkspace(
            outputs=outputs,
            qkv_weight_forward_packed=qkv_weight_forward_packed,
            qkv_weight_forward_scales=qkv_weight_forward_scales,
            qkv_weight_backward_packed=qkv_weight_backward_packed,
            qkv_weight_backward_scales=qkv_weight_backward_scales,
            qkv_weight_global_scale=qkv_weight_global_scale,
            output_weight_forward_packed=output_weight_forward_packed,
            output_weight_forward_scales=output_weight_forward_scales,
            output_weight_backward_packed=output_weight_backward_packed,
            output_weight_backward_scales=output_weight_backward_scales,
            output_weight_global_scale=output_weight_global_scale,
            allocation_data_ptrs={
                name: int(tensor.data_ptr())
                for name, tensor in owner_tensors.items()
            },
            cuda_stream=cuda_stream,
        )

    def _apply(self, fn: Any, recurse: bool = True) -> LowpAttention:
        existing_workspace = getattr(self, "_forward_workspace", None)
        if (
            existing_workspace is not None
            and existing_workspace.publication_state.in_flight_generation
            is not None
        ):
            raise RuntimeError(
                "cannot migrate low-precision attention while workspace "
                "publications are awaiting backward"
            )
        result = super()._apply(fn, recurse)
        # Module._apply cannot see private tensors. Rebuild this dtype-fixed
        # scratch on the migrated parameter device and bind it to that
        # device's current stream.
        result._forward_workspace = result._allocate_forward_workspace()
        return result

    def forward_workspace_contract(self) -> dict[str, Any]:
        workspace = self._forward_workspace
        outputs = workspace.outputs
        buffer_ids = {
            id(buffer) for _name, buffer in self.named_buffers(recurse=True)
        }
        parameter_ids = {
            id(parameter)
            for _name, parameter in self.named_parameters(recurse=True)
        }
        owners: dict[str, dict[str, Any]] = {}
        for name, tensor in _forward_workspace_owner_tensors(workspace):
            allocation_data_ptr = workspace.allocation_data_ptrs[name]
            owners[name] = {
                "slot": _FORWARD_WORKSPACE_OWNER_SLOTS[name],
                "data_ptr": int(tensor.data_ptr()),
                "allocation_data_ptr": allocation_data_ptr,
                "pointer_stable_since_allocation": (
                    int(tensor.data_ptr()) == allocation_data_ptr
                ),
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "bytes": tensor.numel() * tensor.element_size(),
                "active_routes": list(
                    _FORWARD_WORKSPACE_ACTIVE_ROUTES[name]
                ),
                "listed_in_named_buffers": id(tensor) in buffer_ids,
                "listed_in_named_parameters": id(tensor) in parameter_ids,
                "optimizer_visible_parameter": id(tensor) in parameter_ids,
            }
        for name in _FORWARD_WORKSPACE_OPTIONAL_OWNERS:
            tensor = getattr(outputs, name)
            if tensor is None:
                continue
            allocation_data_ptr = workspace.allocation_data_ptrs[name]
            owners[name] = {
                "slot": _FORWARD_WORKSPACE_OPTIONAL_OWNER_SLOTS[name],
                "data_ptr": int(tensor.data_ptr()),
                "allocation_data_ptr": allocation_data_ptr,
                "pointer_stable_since_allocation": (
                    int(tensor.data_ptr()) == allocation_data_ptr
                ),
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "bytes": tensor.numel() * tensor.element_size(),
                "active_routes": ["mxfp4_e8m0_block32"],
                "listed_in_named_buffers": id(tensor) in buffer_ids,
                "listed_in_named_parameters": id(tensor) in parameter_ids,
                "optimizer_visible_parameter": id(tensor) in parameter_ids,
            }
        aliases: dict[str, dict[str, Any]] = {}
        for alias_name, owner_name in _FORWARD_WORKSPACE_ALIAS_OWNERS.items():
            alias = getattr(outputs, alias_name)
            owner = getattr(outputs, owner_name)
            aliases[alias_name] = {
                "owner": owner_name,
                "data_ptr": int(alias.data_ptr()),
                "owner_data_ptr": int(owner.data_ptr()),
                "pointer_matches_owner": (
                    int(alias.data_ptr()) == int(owner.data_ptr())
                ),
                "shape": list(alias.shape),
                "dtype": str(alias.dtype),
                "device": str(alias.device),
                "construction_time_dtype_alias": True,
                "listed_in_named_buffers": id(alias) in buffer_ids,
                "listed_in_named_parameters": id(alias) in parameter_ids,
                "optimizer_visible_parameter": id(alias) in parameter_ids,
            }
        sentinels: dict[str, dict[str, Any]] = {}
        for name in _FORWARD_WORKSPACE_SENTINELS:
            tensor = getattr(outputs, name)
            sentinels[name] = {
                "data_ptr": int(tensor.data_ptr()),
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "numel": tensor.numel(),
                "listed_in_named_buffers": id(tensor) in buffer_ids,
                "listed_in_named_parameters": id(tensor) in parameter_ids,
                "optimizer_visible_parameter": id(tensor) in parameter_ids,
            }
        dual_qkv_weight_owners: dict[str, dict[str, Any]] = {}
        for name, tensor in _d128_dual_qkv_weight_tensors(workspace):
            allocation_data_ptr = workspace.allocation_data_ptrs[name]
            dual_qkv_weight_owners[name] = {
                "data_ptr": int(tensor.data_ptr()),
                "allocation_data_ptr": allocation_data_ptr,
                "pointer_stable_since_allocation": (
                    int(tensor.data_ptr()) == allocation_data_ptr
                ),
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "bytes": tensor.numel() * tensor.element_size(),
                "listed_in_named_buffers": id(tensor) in buffer_ids,
                "listed_in_named_parameters": id(tensor) in parameter_ids,
                "optimizer_visible_parameter": id(tensor) in parameter_ids,
            }
        dual_output_weight_owners: dict[str, dict[str, Any]] = {}
        for name, tensor in _dual_output_weight_tensors(workspace):
            allocation_data_ptr = workspace.allocation_data_ptrs[name]
            dual_output_weight_owners[name] = {
                "data_ptr": int(tensor.data_ptr()),
                "allocation_data_ptr": allocation_data_ptr,
                "pointer_stable_since_allocation": (
                    int(tensor.data_ptr()) == allocation_data_ptr
                ),
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "bytes": tensor.numel() * tensor.element_size(),
                "listed_in_named_buffers": id(tensor) in buffer_ids,
                "listed_in_named_parameters": id(tensor) in parameter_ids,
                "optimizer_visible_parameter": id(tensor) in parameter_ids,
            }
        owner_pointers = [entry["data_ptr"] for entry in owners.values()]
        all_tensors = _forward_workspace_all_tensors(workspace)
        all_outputs_private = all(
            id(tensor) not in buffer_ids and id(tensor) not in parameter_ids
            for _name, tensor in all_tensors
        )
        projection = self.runtime.qkv_projection
        requires_vscale_output = bool(
            projection is not None
            and projection.requires_v_mxfp4_scales_out
        )
        requires_forward_workspace = bool(
            projection is not None
            and getattr(projection, "requires_forward_workspace", False)
        )
        bound_projection_symbol = (
            getattr(
                projection,
                "unchecked_symbol",
                getattr(projection, "symbol", None),
            )
            if projection is not None
            else None
        )
        vscale = outputs.v_mxfp4_scale_pages
        publication_state = workspace.publication_state
        rolling_controller_attached = (
            getattr(workspace, "dual_weight_pack_controller", None)
            is not None
        )
        weight_pack_schedule = (
            "rolling_private_stream"
            if rolling_controller_attached
            else "synchronous_same_stream"
        )
        return {
            "schema": "lowp_layer_forward_workspace_v2",
            "publication_slots": list(
                _FORWARD_WORKSPACE_OWNER_SLOTS.values()
            )
            + [
                _FORWARD_WORKSPACE_OPTIONAL_OWNER_SLOTS[name]
                for name in _FORWARD_WORKSPACE_OPTIONAL_OWNERS
                if getattr(outputs, name) is not None
            ],
            "owners": owners,
            "aliases": aliases,
            "sentinels": sentinels,
            "publication_lifecycle": {
                "generation_guard_enforced": True,
                "same_stream_enforced": True,
                "one_forward_in_flight_per_layer": True,
                "current_generation": publication_state.current_generation,
                "in_flight": (
                    publication_state.in_flight_generation is not None
                ),
            },
            "d128_dual_qkv_weight": {
                "eligible": _uses_direct_d128_dual_qkv_weight_prep(
                    self.runtime
                ),
                "authenticated": (
                    workspace.d128_dual_qkv_weight_authenticated
                ),
                "schedule": weight_pack_schedule,
                "rolling_controller_attached": (
                    rolling_controller_attached
                ),
                "composite_weight_prep_authenticated": (
                    getattr(
                        workspace,
                        "weight_prep_authenticated",
                        False,
                    )
                ),
                "one_forward_in_flight_per_layer": True,
                "generation_guard_enforced": True,
                "same_stream_enforced": True,
                "abi_identity_bound": bool(
                    workspace.d128_dual_qkv_weight_authenticated
                    and workspace.d128_dual_qkv_weight_abi_identity
                    is not None
                ),
                "abi_identity_tensor_count": (
                    len(workspace.d128_dual_qkv_weight_abi_identity)
                    if workspace.d128_dual_qkv_weight_abi_identity
                    is not None
                    else 0
                ),
                "abi_identity_excludes_tensor_version": True,
                "checked_symbol": (
                    "quantize_gqa_d128_qkv_projection_weight_dual_out"
                ),
                "unchecked_symbol": (
                    "quantize_gqa_d128_qkv_projection_weight_dual_out_"
                    "unchecked"
                ),
                "owners": dual_qkv_weight_owners,
                "all_pointers_stable_since_allocation": all(
                    entry["pointer_stable_since_allocation"]
                    for entry in dual_qkv_weight_owners.values()
                ),
                "all_pointers_unique": (
                    len(
                        {
                            entry["data_ptr"]
                            for entry in dual_qkv_weight_owners.values()
                        }
                    )
                    == len(dual_qkv_weight_owners)
                ),
                "total_bytes": sum(
                    entry["bytes"]
                    for entry in dual_qkv_weight_owners.values()
                ),
            },
            "dual_output_weight": {
                "eligible": _uses_direct_dual_output_weight_prep(
                    self.runtime
                ),
                "authenticated": workspace.output_dual_weight_authenticated,
                "schedule": weight_pack_schedule,
                "rolling_controller_attached": (
                    rolling_controller_attached
                ),
                "composite_weight_prep_authenticated": (
                    getattr(
                        workspace,
                        "weight_prep_authenticated",
                        False,
                    )
                ),
                "one_forward_in_flight_per_layer": True,
                "generation_guard_enforced": True,
                "same_stream_enforced": True,
                "abi_identity_bound": bool(
                    workspace.output_dual_weight_authenticated
                    and workspace.output_dual_weight_abi_identity is not None
                ),
                "abi_identity_tensor_count": (
                    len(workspace.output_dual_weight_abi_identity)
                    if workspace.output_dual_weight_abi_identity is not None
                    else 0
                ),
                "abi_identity_excludes_tensor_version": True,
                "checked_symbol": (
                    "quantize_nvfp4_projection_weight_dual_out"
                ),
                "unchecked_symbol": (
                    "quantize_nvfp4_projection_weight_dual_out_unchecked"
                ),
                "owners": dual_output_weight_owners,
                "all_pointers_stable_since_allocation": all(
                    entry["pointer_stable_since_allocation"]
                    for entry in dual_output_weight_owners.values()
                ),
                "all_pointers_unique": (
                    len(
                        {
                            entry["data_ptr"]
                            for entry in dual_output_weight_owners.values()
                        }
                    )
                    == len(dual_output_weight_owners)
                ),
                "total_bytes": sum(
                    entry["bytes"]
                    for entry in dual_output_weight_owners.values()
                ),
            },
            "owner_count": len(owners),
            "owner_pointers_unique_within_layer": (
                len(set(owner_pointers)) == len(owner_pointers)
            ),
            "owner_pointers_stable_since_allocation": all(
                entry["pointer_stable_since_allocation"]
                for entry in owners.values()
            ),
            "typed_aliases_match_owners": all(
                entry["pointer_matches_owner"]
                for entry in aliases.values()
            ),
            "total_owner_bytes": sum(
                entry["bytes"] for entry in owners.values()
            ),
            "all_outputs_private_nonpersistent": all_outputs_private,
            "supports_both_retained_routes": (
                self.runtime.config.batch == 1
                or self.runtime.experimental_native_nvfp4_projection_out
            ),
            "supported_routes": (
                list(_FORWARD_WORKSPACE_COMMON_ROUTES)
                if (
                    self.runtime.config.batch == 1
                    or self.runtime.experimental_native_nvfp4_projection_out
                )
                else ["e4m3_fp8"]
            ),
            "active_route": self.runtime.pv_format,
            "active_owner_fields": [
                name
                for name, routes in _FORWARD_WORKSPACE_ACTIVE_ROUTES.items()
                if self.runtime.pv_format in routes
            ]
            + [
                name
                for name in _FORWARD_WORKSPACE_OPTIONAL_OWNERS
                if getattr(outputs, name) is not None
            ],
            "single_stream_cuda_stream": workspace.cuda_stream,
            "bound_projection_symbol": bound_projection_symbol,
            "bound_projection_checked_symbol": (
                getattr(
                    projection,
                    "checked_symbol",
                    getattr(projection, "symbol", None),
                )
                if projection is not None
                else None
            ),
            "requires_forward_workspace": requires_forward_workspace,
            "forward_workspace_abi_validated": (
                bool(projection.forward_workspace_abi_validated)
                if projection is not None
                and hasattr(projection, "forward_workspace_abi_validated")
                else None
            ),
            "validated_forward_workspace_count": (
                int(projection.validated_forward_workspace_count)
                if projection is not None
                and hasattr(projection, "validated_forward_workspace_count")
                else None
            ),
            # Compatibility aliases for readers of the original slot-13-only
            # workspace schema. New gates authenticate every owner above.
            "data_ptr": int(vscale.data_ptr()),
            "allocation_data_ptr": workspace.allocation_data_ptrs[
                "v_mxfp4_scale_pages"
            ],
            "pointer_stable_since_allocation": (
                int(vscale.data_ptr())
                == workspace.allocation_data_ptrs["v_mxfp4_scale_pages"]
            ),
            "shape": list(vscale.shape),
            "dtype": str(vscale.dtype),
            "device": str(vscale.device),
            "bytes": vscale.numel() * vscale.element_size(),
            "listed_in_named_buffers": id(vscale) in buffer_ids,
            "listed_in_named_parameters": id(vscale) in parameter_ids,
            "optimizer_visible_parameter": id(vscale) in parameter_ids,
            "active_for_bound_projection": requires_vscale_output,
        }

    @torch.no_grad()
    def refresh_qk_quant_scales(self) -> None:
        """Estimate projection-native Q/K scales from projection row energy."""
        c = self.runtime.config

        def scales(weight: torch.Tensor, heads: int, rms_clip: float) -> torch.Tensor:
            row_variance = weight.float().square().sum(dim=-1)
            heads_per_scale = 1 if c.head_dim == 128 else 2
            paired_rms = row_variance.view(
                heads // heads_per_scale,
                heads_per_scale,
                c.head_dim,
            ).mean(dim=(1, 2)).sqrt()
            multiplier = (6.0 / (paired_rms * rms_clip)).clamp(0.5, 4.0)
            # Match the E4M3 dequantizer consumed downstream so packing and
            # score correction use exact reciprocal values.
            decode = (1.0 / multiplier).to(torch.float8_e4m3fn).float()
            return decode.reciprocal()

        self.qk_scales[:, :, 0] = scales(
            self.weights.q, c.q_heads, 3.0
        )
        kv_scale_heads = c.kv_heads if c.head_dim == 128 else c.kv_heads // 2
        self.qk_scales[:, :kv_scale_heads, 1] = scales(
            self.weights.k, c.kv_heads, 3.3
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_norm_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        config = self.runtime.config
        expected_shape = (config.batch, config.sequence, config.hidden)
        if tuple(x.shape) != expected_shape:
            raise ValueError(
                "low-precision attention input must match its fixed "
                f"{expected_shape} runtime shape, got {tuple(x.shape)}"
            )
        workspace = self._forward_workspace
        empty_weight = workspace.outputs.empty_bf16
        if self.runtime.experimental_fused_attention_rmsnorm_nvfp4:
            if (
                attention_norm_weight is None
                or attention_norm_weight.dtype != torch.bfloat16
                or attention_norm_weight.device != x.device
                or not attention_norm_weight.is_contiguous()
                or tuple(attention_norm_weight.shape) != (config.hidden,)
            ):
                raise ValueError(
                    "experimental fused attention RMSNorm requires a "
                    "contiguous BF16 norm weight [hidden] on the input device"
                )
            norm_weight = attention_norm_weight
        else:
            if attention_norm_weight is not None:
                raise ValueError(
                    "attention_norm_weight is accepted only by the "
                    "experimental fused attention RMSNorm route"
                )
            norm_weight = empty_weight
        if self.qkv_master_parameter_layout == PACKED_D64_LOWP_QKV_LAYOUT:
            packed_qkv_weight = self.weights.qkv
            q_weight = empty_weight
            k_weight = empty_weight
            v_weight = empty_weight
        else:
            packed_qkv_weight = empty_weight
            q_weight = self.weights.q
            k_weight = self.weights.k
            v_weight = self.weights.v
        # The raw CUDA publishers do not participate in PyTorch tensor
        # versioning. Authenticate both lifetime and stream on every use so a
        # second forward cannot silently overwrite values retained by an
        # earlier graph.
        _require_forward_workspace_same_stream(
            workspace,
            x,
            phase="forward",
        )
        autograd_tensors = (
            x,
            norm_weight,
            packed_qkv_weight,
            q_weight,
            k_weight,
            v_weight,
            self.weights.o,
            self.qk_scales,
        )
        requires_backward = bool(
            torch.is_grad_enabled()
            and any(tensor.requires_grad for tensor in autograd_tensors)
        )
        generation = workspace.publication_state.begin_forward(
            requires_backward=requires_backward
        )
        try:
            return _LowpAttentionFunction.apply(
                x,
                norm_weight,
                packed_qkv_weight,
                q_weight,
                k_weight,
                v_weight,
                self.weights.o,
                self.qk_scales,
                workspace,
                self.runtime,
            )
        except Exception:
            workspace.publication_state.abort_forward(generation)
            raise


class MLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.gate = _new_weight(config.intermediate, config.hidden)
        self.up = _new_weight(config.intermediate, config.hidden)
        self.down = _new_weight(config.hidden, config.intermediate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(F.silu(F.linear(x, self.gate)) * F.linear(x, self.up), self.down)


class DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Config,
        rope: tuple[torch.Tensor, torch.Tensor],
        runtime: LowpAttentionRuntime | None,
        bf16_attention_control: str = DEFAULT_BF16_ATTENTION_CONTROL,
    ) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden, config.rms_epsilon)
        self.ffn_norm = RMSNorm(config.hidden, config.rms_epsilon)
        self.fused_attention_rmsnorm_nvfp4 = bool(
            runtime is not None
            and runtime.experimental_fused_attention_rmsnorm_nvfp4
        )
        if runtime is None:
            if bf16_attention_control == DEFAULT_BF16_ATTENTION_CONTROL:
                self.attention = BF16Attention(config, rope)
            elif bf16_attention_control == "packed_qkv_single_linear":
                self.attention = PackedQKVBF16Attention(config, rope)
            else:
                raise ValueError(
                    "bf16_attention_control must be one of "
                    f"{BF16_ATTENTION_CONTROLS}, got "
                    f"{bf16_attention_control!r}"
                )
        else:
            if bf16_attention_control != DEFAULT_BF16_ATTENTION_CONTROL:
                raise ValueError(
                    "bf16_attention_control applies only to a BF16 decoder"
                )
            self.attention = LowpAttention(config, runtime)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused_attention_rmsnorm_nvfp4:
            attention_output = self.attention(x, self.attention_norm.weight)
        else:
            attention_output = self.attention(self.attention_norm(x))
        x = x + attention_output
        return x + self.mlp(self.ffn_norm(x))


class Llama12B(nn.Module):
    def __init__(
        self,
        config: Config,
        rope: tuple[torch.Tensor, torch.Tensor],
        runtime: LowpAttentionRuntime | None,
        bf16_attention_control: str = DEFAULT_BF16_ATTENTION_CONTROL,
    ) -> None:
        super().__init__()
        if bf16_attention_control not in BF16_ATTENTION_CONTROLS:
            raise ValueError(
                "bf16_attention_control must be one of "
                f"{BF16_ATTENTION_CONTROLS}, got "
                f"{bf16_attention_control!r}"
            )
        if (
            runtime is not None
            and bf16_attention_control != DEFAULT_BF16_ATTENTION_CONTROL
        ):
            raise ValueError(
                "bf16_attention_control applies only to a BF16 decoder"
            )
        self.config = config
        # Keep the route available at the model boundary. Timed callers
        # activate and validate it before recording their first CUDA event;
        # forward then performs one constant-time cached-route assertion.
        self.lowp_attention_runtime = runtime
        self.bf16_attention_control = (
            bf16_attention_control if runtime is None else None
        )
        self.embedding = nn.Parameter(
            torch.empty(
                config.vocab,
                config.hidden,
                device="cuda",
                dtype=torch.bfloat16,
            )
        )
        torch.nn.init.normal_(self.embedding, mean=0.0, std=0.02)
        self.lm_head: nn.Parameter | None = None
        if not config.tie_word_embeddings:
            self.lm_head = _new_weight(config.vocab, config.hidden)
        self.layers = nn.ModuleList(
            DecoderLayer(
                config,
                rope,
                runtime,
                bf16_attention_control=bf16_attention_control,
            )
            for _ in range(config.layers)
        )
        self.final_norm = RMSNorm(config.hidden, config.rms_epsilon)

    @property
    def attention_route(self) -> str:
        """Return the currently bound attention route without stale caching."""
        runtime = self.lowp_attention_runtime
        if runtime is not None:
            return str(runtime.forward_topology["route"])
        control = self.bf16_attention_control
        if control is None:
            raise RuntimeError("BF16 decoder is missing its attention control")
        return BF16_ATTENTION_ROUTES[control]

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        runtime = self.lowp_attention_runtime
        if runtime is not None:
            require_active_forward_route(
                str(runtime.forward_topology["route"])
            )
        hidden = F.embedding(tokens, self.embedding)
        for layer in self.layers:
            hidden = layer(hidden)
        hidden = self.final_norm(hidden)
        output_weight = (
            self.embedding if self.lm_head is None else self.lm_head
        )
        return F.linear(hidden, output_weight)

    def bind_lowp_attention_runtime(
        self,
        runtime: LowpAttentionRuntime,
    ) -> int:
        """Atomically bind one retained runtime to the complete decoder."""
        attentions: list[LowpAttention] = []
        for layer in self.layers:
            attention = layer.attention
            if not isinstance(attention, LowpAttention):
                raise TypeError(
                    "cannot bind a low-precision runtime to a BF16 decoder"
                )
            attentions.append(attention)
        if len(attentions) != self.config.layers:
            raise RuntimeError(
                f"found {len(attentions)} low-precision layers, expected "
                f"{self.config.layers}"
            )
        if runtime.config != self.config:
            raise ValueError("runtime and decoder configurations do not match")
        runtime_scale_policy = (
            runtime.qkv_projection_format,
            runtime.q_quant_scale,
            runtime.k_quant_scale,
            runtime.per_block_qk_scales,
            runtime.adaptive_qk_weight_scales,
        )
        for layer_index, attention in enumerate(attentions):
            bound = attention.runtime
            bound_scale_policy = (
                bound.qkv_projection_format,
                bound.q_quant_scale,
                bound.k_quant_scale,
                bound.per_block_qk_scales,
                bound.adaptive_qk_weight_scales,
            )
            if bound_scale_policy != runtime_scale_policy:
                raise ValueError(
                    "runtime crossover would reuse incompatible layer Q/K "
                    f"scale state at decoder layer {layer_index}"
                )
            try:
                require_matching_backward_contracts(
                    {
                        "bound": bound.backward_contract(),
                        "candidate": runtime.backward_contract(),
                    }
                )
                require_shared_backward_physical_identity(bound, runtime)
            except RuntimeError as error:
                raise ValueError(
                    "runtime crossover requires one identical logical and "
                    "physical backward at decoder layer "
                    f"{layer_index}: {error}"
                ) from error
        for attention in attentions:
            attention.runtime = runtime
        self.lowp_attention_runtime = runtime
        return len(attentions)

    def lowp_forward_workspace_contract(self) -> dict[str, Any]:
        """Authenticate every layer-owned forward workspace without CUDA IO."""
        entries: list[dict[str, Any]] = []
        for layer_index, layer in enumerate(self.layers):
            attention = layer.attention
            if not isinstance(attention, LowpAttention):
                raise TypeError(
                    f"decoder layer {layer_index} is not low precision"
                )
            entries.append(
                {
                    "layer": layer_index,
                    **attention.forward_workspace_contract(),
                }
            )
        legacy_pointers = [entry["data_ptr"] for entry in entries]
        owner_pointers = [
            owner["data_ptr"]
            for entry in entries
            for owner in entry["owners"].values()
        ]
        return {
            "schema": "lowp_model_forward_workspaces_v2",
            "layer_count": len(entries),
            "owner_count": len(owner_pointers),
            "owner_pointers_globally_unique": (
                len(set(owner_pointers)) == len(owner_pointers)
            ),
            "owner_pointers_unique_across_layers": (
                len(set(owner_pointers)) == len(owner_pointers)
            ),
            "owner_pointers_stable_since_allocation": all(
                entry["owner_pointers_stable_since_allocation"]
                for entry in entries
            ),
            "typed_aliases_match_owners": all(
                entry["typed_aliases_match_owners"] for entry in entries
            ),
            "all_outputs_private_nonpersistent": all(
                entry["all_outputs_private_nonpersistent"]
                for entry in entries
            ),
            "supports_both_retained_routes": all(
                entry["supports_both_retained_routes"] for entry in entries
            ),
            "total_owner_bytes": sum(
                entry["total_owner_bytes"] for entry in entries
            ),
            # Compatibility fields retain the former slot-13 view while the
            # v2 fields above cover all nine independently owned outputs.
            "pointers_unique_across_layers": (
                len(set(legacy_pointers)) == len(legacy_pointers)
            ),
            "pointers_stable_since_allocation": all(
                entry["pointer_stable_since_allocation"]
                for entry in entries
            ),
            "layers": entries,
        }

    def require_lowp_forward_workspace_stream(self) -> int | None:
        """Validate one stream for all layer scratch before model timing."""
        workspaces: list[_LowpAttentionForwardWorkspace] = []
        for layer_index, layer in enumerate(self.layers):
            attention = layer.attention
            if not isinstance(attention, LowpAttention):
                raise TypeError(
                    f"decoder layer {layer_index} is not low precision"
                )
            workspaces.append(attention._forward_workspace)
        if not workspaces:
            return None
        device = workspaces[0].outputs.q_payload.device
        if device.type != "cuda":
            return None
        current_stream = int(
            torch.cuda.current_stream(device).cuda_stream
        )
        for layer_index, workspace in enumerate(workspaces):
            for name, tensor in _forward_workspace_all_tensors(workspace):
                if tensor.device != device:
                    raise RuntimeError(
                        "layer-owned forward-publication workspaces span "
                        f"devices; layer {layer_index} field {name!r} is "
                        f"on {tensor.device}, expected {device}"
                    )
            for name, tensor in _forward_workspace_owner_tensors(workspace):
                if (
                    int(tensor.data_ptr())
                    != workspace.allocation_data_ptrs[name]
                ):
                    raise RuntimeError(
                        "layer-owned forward-publication pointer changed "
                        f"after allocation at layer {layer_index}, field "
                        f"{name!r}"
                    )
            direct_weight_tensors = (
                *_d128_dual_qkv_weight_tensors(workspace),
                *_dual_output_weight_tensors(workspace),
            )
            for name, tensor in direct_weight_tensors:
                if (
                    int(tensor.data_ptr())
                    != workspace.allocation_data_ptrs[name]
                ):
                    raise RuntimeError(
                        "layer-owned direct-weight publication pointer "
                        f"changed after allocation at layer {layer_index}, "
                        f"field {name!r}"
                    )
            for alias_name, owner_name in (
                _FORWARD_WORKSPACE_ALIAS_OWNERS.items()
            ):
                alias = getattr(workspace.outputs, alias_name)
                owner = getattr(workspace.outputs, owner_name)
                if int(alias.data_ptr()) != int(owner.data_ptr()):
                    raise RuntimeError(
                        "construction-time typed forward alias diverged "
                        f"from its owner at layer {layer_index}: "
                        f"{alias_name!r} vs {owner_name!r}"
                    )
            if workspace.cuda_stream != current_stream:
                raise RuntimeError(
                    "persistent forward-publication workspaces are "
                    "single-stream; "
                    f"layer {layer_index} is bound to CUDA stream "
                    f"{workspace.cuda_stream}, current stream is "
                    f"{current_stream}"
                )
        return current_stream


def activate_model_forward_route(model: nn.Module) -> bool:
    """Validate workspaces and activate a retained route before timing."""
    runtime = getattr(model, "lowp_attention_runtime", None)
    layers = getattr(model, "layers", None)
    if layers is None:
        raise TypeError("low-precision model does not expose decoder layers")
    if runtime is None:
        if any(
            isinstance(getattr(layer, "attention", None), LowpAttention)
            for layer in layers
        ):
            raise RuntimeError(
                "model-level runtime is missing from a low-precision decoder"
            )
        return False
    for layer_index, layer in enumerate(layers):
        attention = getattr(layer, "attention", None)
        if not isinstance(attention, LowpAttention):
            raise TypeError(
                f"decoder layer {layer_index} is not low precision"
            )
        if attention.runtime is not runtime:
            raise RuntimeError(
                "model-level and layer-level low-precision runtimes diverged "
                f"at decoder layer {layer_index}; bind the runtime through "
                "Llama12B.bind_lowp_attention_runtime()"
            )
    if not isinstance(model, Llama12B):
        raise TypeError("low-precision route activation requires Llama12B")
    model.require_lowp_forward_workspace_stream()
    return activate_forward_route(str(runtime.forward_topology["route"]))


def _useful_flops(config: Config) -> float:
    attention_weights = (
        config.hidden * config.q_width
        + 2 * config.hidden * config.kv_width
        + config.hidden * config.q_width
    )
    mlp_weights = 3 * config.hidden * config.intermediate
    trained_matmul_weights = (
        config.layers * (attention_weights + mlp_weights)
        + config.hidden * config.vocab
    )
    linear = 6.0 * config.sequence * trained_matmul_weights
    causal_pairs = config.sequence * (config.sequence + 1) / 2.0
    attention = (
        12.0
        * config.layers
        * config.q_heads
        * causal_pairs
        * config.head_dim
    )
    return config.batch * (linear + attention)


def _sample_gradients(model: nn.Module, elements: int = 8192) -> dict[str, torch.Tensor]:
    parameters = dict(model.named_parameters())
    wanted = [
        "embedding",
        "layers.0.attention.weights.q",
        "layers.0.attention.weights.k",
        "layers.0.attention.weights.v",
        "layers.0.attention.weights.o",
        "layers.0.mlp.down",
    ]
    if "lm_head" in parameters:
        wanted.insert(1, "lm_head")
    result: dict[str, torch.Tensor] = {}
    for name in wanted:
        if name not in parameters and name.rsplit(".", 1)[-1] in {
            "q",
            "k",
            "v",
        }:
            prefix, projection = name.rsplit(".", 1)
            packed = parameters.get(f"{prefix}.qkv")
            if packed is not None and packed.grad is not None:
                layout = packed_qkv_layout(model.config)
                q_grad, k_grad, v_grad = torch.split(
                    packed.grad,
                    (
                        layout.q_width,
                        layout.kv_width,
                        layout.kv_width,
                    ),
                    dim=0,
                )
                gradient = {"q": q_grad, "k": k_grad, "v": v_grad}[
                    projection
                ]
                result[name] = (
                    gradient.detach().reshape(-1)[:elements].float().cpu()
                )
            continue
        gradient = parameters[name].grad
        if gradient is not None:
            result[name] = gradient.detach().reshape(-1)[:elements].float().cpu()
    return result


def _cosine(reference: torch.Tensor, actual: torch.Tensor) -> float | None:
    denominator = reference.norm() * actual.norm()
    if float(denominator) == 0.0:
        return None
    return float(torch.dot(reference, actual) / denominator)


def _strict_json_value(value: Any) -> Any:
    """Replace non-finite diagnostics so receipts are strict RFC JSON."""

    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _strict_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json_value(item) for item in value]
    return value


def _qk_scale_summary(model: "Llama12B", config: Config) -> dict[str, Any]:
    """Return effective scales without dropping nonzero batch indices."""
    q_per_layer: list[list[list[float]]] = []
    k_per_layer: list[list[list[float]]] = []
    kv_scale_heads = (
        config.kv_heads
        if config.head_dim == 128
        else config.kv_heads // 2
    )
    for layer in model.layers:
        attention = layer.attention
        if not isinstance(attention, LowpAttention):
            continue
        q_per_layer.append(
            attention.qk_scales[
                : config.batch, :, 0
            ].detach().float().cpu().tolist()
        )
        k_per_layer.append(
            attention.qk_scales[
                : config.batch, :kv_scale_heads, 1
            ].detach().float().cpu().tolist()
        )

    def distribution(values: list[list[list[float]]]) -> dict[str, Any]:
        flat = sorted(
            value
            for layer_values in values
            for batch_values in layer_values
            for value in batch_values
        )
        per_batch = {
            str(batch_index): [
                layer_values[batch_index] for layer_values in values
            ]
            for batch_index in range(config.batch)
        }
        if not flat:
            return {
                "represented_batches": [],
                "per_batch": {},
                "per_layer": [],
                "minimum": None,
                "median": None,
                "maximum": None,
            }
        return {
            "represented_batches": list(range(config.batch)),
            "per_batch": per_batch,
            # Preserve the historical batch-zero view for existing readers;
            # aggregate statistics and ``per_batch`` cover every batch.
            "per_layer": per_batch["0"],
            "minimum": flat[0],
            "median": statistics.median(flat),
            "maximum": flat[-1],
        }

    return {"q": distribution(q_per_layer), "k": distribution(k_per_layer)}


def _requested_backward_approximation_policy(
    config: Config,
) -> tuple[int, int | None, bool]:
    """Return the shape-authenticated policy requested from the runtime."""
    if config.head_dim == 128:
        return 1, 0, True
    if config.batch != 1:
        return 1, 2, False
    return 2, None, False


LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _standard_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Return the shared dense language-model loss for both routes."""
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="mean",
    )


def _make_loss_function(
    *,
    compile_loss: bool,
) -> tuple[LossFunction, dict[str, Any]]:
    """Build one standard-CE callable and its audit receipt."""
    receipt: dict[str, Any] = {
        "route": "standard_dense_cross_entropy",
        "implementation": "torch.nn.functional.cross_entropy",
        "reduction": "mean",
        "compiled": compile_loss,
        "compiler": "torch.compile" if compile_loss else None,
        "backend": "inductor" if compile_loss else None,
        "fullgraph": True if compile_loss else None,
        "cut_cross_entropy": False,
        "shared_between_bf16_and_lowp": True,
    }
    if not compile_loss:
        return _standard_cross_entropy, receipt
    return (
        torch.compile(
            _standard_cross_entropy,
            backend="inductor",
            fullgraph=True,
        ),
        receipt,
    )


def _benchmark_route(
    name: str,
    config: Config,
    rope: tuple[torch.Tensor, torch.Tensor],
    tokens: torch.Tensor,
    targets: torch.Tensor,
    *,
    runtime: LowpAttentionRuntime | None,
    seed: int,
    warmups: int,
    samples: int,
    learning_rate: float,
    loss_function: LossFunction,
    loss_receipt: dict[str, Any],
    bf16_attention_control: str = DEFAULT_BF16_ATTENTION_CONTROL,
    reference_logits: torch.Tensor | None = None,
    reference_gradients: dict[str, torch.Tensor] | None = None,
) -> tuple[
    dict[str, Any],
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[str, Any] | None,
]:
    torch.manual_seed(seed)
    model = Llama12B(
        config,
        rope,
        runtime,
        bf16_attention_control=(
            bf16_attention_control
            if runtime is None
            else DEFAULT_BF16_ATTENTION_CONTROL
        ),
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != config.parameter_count:
        raise RuntimeError(
            f"constructed {parameter_count} parameters, expected "
            f"{config.parameter_count} for {config.model_preset}"
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1.0e-8,
        weight_decay=0.0,
        fused=True,
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline_allocated = torch.cuda.memory_allocated()
    records: list[dict[str, float | bool]] = []
    sampled_logits: torch.Tensor | None = None
    sampled_gradients: dict[str, torch.Tensor] | None = None

    for step in range(warmups + samples):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        forward_done = torch.cuda.Event(enable_timing=True)
        backward_done = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        start.record()
        optimizer.zero_grad(set_to_none=True)
        logits = model(tokens)
        loss = loss_function(logits, targets)
        if step == 0:
            # Match the production trainer's peak-memory lifetime: retain
            # only the small diagnostic slice and release the full
            # [B,S,V] logits before compiled CE allocates its gradient.
            sampled_logits = logits.detach()[0, :16, :1024].float().cpu()
        forward_done.record()
        del logits
        loss.backward()
        backward_done.record()
        optimizer.step()
        end.record()
        end.synchronize()
        wall_ms = (time.perf_counter() - wall_start) * 1000.0
        forward_ms = float(start.elapsed_time(forward_done))
        backward_ms = float(forward_done.elapsed_time(backward_done))
        optimizer_ms = float(backward_done.elapsed_time(end))
        step_ms = float(start.elapsed_time(end))
        loss_value = float(loss.detach())
        finite = math.isfinite(loss_value)
        records.append(
            {
                "step": float(step),
                "warmup": step < warmups,
                "loss": loss_value,
                "finite": finite,
                "forward_ms": forward_ms,
                "backward_ms": backward_ms,
                "optimizer_ms": optimizer_ms,
                "step_ms": step_ms,
                "wall_ms": wall_ms,
            }
        )
        print(
            f"{name} step={step} warmup={step < warmups} "
            f"loss={loss_value:.6f} step={step_ms:.3f} ms",
            flush=True,
        )
        if step == 0:
            sampled_gradients = _sample_gradients(model)
        del loss
        if not finite:
            break

    assert sampled_logits is not None and sampled_gradients is not None
    measured = [record for record in records if not bool(record["warmup"])]
    if not measured:
        measured = records[-1:]
    medians = {
        key: statistics.median(float(record[key]) for record in measured)
        for key in ("forward_ms", "backward_ms", "optimizer_ms", "step_ms", "wall_ms")
    }
    useful_flops = _useful_flops(config)
    step_seconds = medians["step_ms"] / 1000.0
    result: dict[str, Any] = {
        "route": name,
        "attention_route": model.attention_route,
        "loss": dict(loss_receipt),
        "parameter_count": parameter_count,
        "records": records,
        "steady_state": {
            **medians,
            "tokens_per_second": (
                config.batch * config.sequence / step_seconds
            ),
            "useful_tflops": useful_flops / step_seconds / 1.0e12,
            "mfu_at_2250_tflops": useful_flops / step_seconds / 2.25e15,
        },
        "memory": {
            "baseline_allocated_gib": baseline_allocated / 2.0**30,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2.0**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2.0**30,
        },
    }
    forward_dispatch: dict[str, Any] | None = None
    if runtime is not None:
        result["effective_qk_scales"] = _qk_scale_summary(model, config)
        # Capture this while the model still owns every layer workspace.
        # The bound projection tracks authenticated workspaces weakly, so a
        # post-GC receipt would incorrectly report zero validated workspaces.
        forward_dispatch = {
            **runtime.forward_dispatch_contract(),
            "captured_after_warmup_and_measured_steps": True,
            "captured_before_model_workspace_release": True,
        }
    if reference_logits is not None:
        result["initial_logits_vs_bf16"] = {
            "cosine": _cosine(reference_logits.flatten(), sampled_logits.flatten()),
            "relative_l2": float(
                (reference_logits - sampled_logits).norm()
                / reference_logits.norm().clamp_min(1.0e-20)
            ),
        }
    if reference_gradients is not None:
        result["initial_gradient_cosines_vs_bf16"] = {
            key: _cosine(reference_gradients[key], sampled_gradients[key])
            for key in reference_gradients.keys() & sampled_gradients.keys()
        }
        result["initial_gradient_quality_vs_bf16"] = {}
        for key in reference_gradients.keys() & sampled_gradients.keys():
            reference = reference_gradients[key]
            actual = sampled_gradients[key]
            reference_norm = reference.norm().clamp_min(1.0e-20)
            actual_norm = actual.norm()
            result["initial_gradient_quality_vs_bf16"][key] = {
                "cosine": _cosine(reference, actual),
                "relative_l2": float((actual - reference).norm() / reference_norm),
                "norm_ratio": float(actual_norm / reference_norm),
            }
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return result, sampled_logits, sampled_gradients, forward_dispatch


def _load_forward(path: Path, module: str, config: Config) -> tuple[Any, dict[str, Any]]:
    extension = _load_extension(path, module)
    topology = dict(extension.read_hao_direct_topology())
    _require_forward_topology(config, topology)
    activate_forward_route(str(topology["route"]))
    return extension, topology


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-preset",
        choices=MODEL_PRESETS,
        default=DEFAULT_MODEL_PRESET,
        help=(
            "model architecture; llama3.1-8b selects L32/H4096/D128 and "
            "an untied language-model head"
        ),
    )
    parser.add_argument(
        "--batch",
        type=int,
        choices=SUPPORTED_LOWP_BATCHES,
        default=1,
        help="select an authenticated D64 exact-FP8 batch specialization",
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument(
        "--bf16-attention-control",
        choices=BF16_ATTENTION_CONTROLS,
        default=DEFAULT_BF16_ATTENTION_CONTROL,
        help=(
            "retain the historical three-GEMM BF16 projection or use the "
            "same one-GEMM packed-QKV topology as the fused low-precision "
            "routes"
        ),
    )
    parser.add_argument(
        "--layers",
        type=int,
        help="override the preset depth only for integration smoke tests",
    )
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument(
        "--compile-loss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "compile the shared standard dense cross-entropy with "
            "torch.compile/Inductor; disabled by default to preserve the "
            "historical eager-loss benchmark"
        ),
    )
    parser.add_argument("--loss-scale", type=float, default=2.0**16)
    parser.add_argument("--gradient-global-scale", type=float, default=2.0**-8)
    parser.add_argument("--q-quant-scale", type=float, default=2.25)
    parser.add_argument("--k-quant-scale", type=float, default=2.0)
    parser.add_argument(
        "--qkv-projection-format",
        choices=("nvfp4", "e4m3"),
        default="nvfp4",
        help=(
            "opt into rowwise-activation/channelwise-weight dense E4M3 for "
            "the fused QKV projection; e4m3 supports the exact FP8-PV and "
            "interleaved causal MXFP4-PV forward extensions"
        ),
    )
    parser.add_argument(
        "--output-projection-format",
        choices=("nvfp4", "e4m3"),
        default="nvfp4",
        help=(
            "select the learned attention output projection forward format; "
            "e4m3 is fail-closed to the exact D128 E4M3-QKV correctness "
            "canary and retains NVFP4 dgrad"
        ),
    )
    parser.add_argument(
        "--experimental-native-nvfp4-projection-out",
        action="store_true",
        help=(
            "use the caller-owned native-NVFP4 projection ABI; D64 publishes "
            "represented per-row-K16 Q/K, while D128 publishes one shared "
            "projection-accumulator E4M3 Q/K/V backward representation"
        ),
    )
    parser.add_argument(
        "--experimental-fused-attention-rmsnorm-nvfp4",
        action="store_true",
        help=(
            "fuse attention RMSNorm with exact-dynamic native NVFP4 input "
            "packing; restricted to the experimental B16 S4096 H2048 D64 "
            "caller-owned native projection route"
        ),
    )
    parser.add_argument(
        "--experimental-output-shared-split-v",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "select output-shared direct MXFP4/E4M3 split-V publication; "
            "the explicit opt-in is accepted only for the eligible native "
            "NVFP4-QK + MXFP4-PV route; the default retains the established "
            "publisher"
        ),
    )
    parser.add_argument(
        "--backward-match-forward-operands",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "publish backward Q/K (and MX V) from the exact low-precision "
            "codes used by forward"
        ),
    )
    parser.add_argument(
        "--per-block-qk-scales",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "publish the row-by-K16 Q/K scales required by exact forward "
            "score reconstruction; represented-forward backward routes "
            "also consume these scales"
        ),
    )
    parser.add_argument(
        "--projection-weight-scaling",
        choices=("1d", "2d"),
        default="2d",
        help="use transpose-consistent 16x16 scaling for learned weights",
    )
    parser.add_argument(
        "--v-mxfp4-scaling",
        choices=("1d", "2d"),
        default="1d",
        help=(
            "use one MXFP4 V scale per depth-row x 32 sequence values; "
            "2d retains the coarser 32x32 compatibility policy"
        ),
    )
    parser.add_argument(
        "--adaptive-qk-weight-scales",
        action="store_true",
        help="initialize independent per-layer/per-head Q/K scales from weights",
    )
    parser.add_argument(
        "--backward-probability-correction",
        type=float,
        default=None,
        help=(
            "explicit attention-backward branch gain (historical probability "
            "correction name); overrides topology/default calibration"
        ),
    )
    parser.add_argument(
        "--projection-dgrad",
        choices=("auto", "bf16", "nvfp4"),
        default="auto",
        help=(
            "auto selects exact BF16 for D64 and the hierarchical NVFP4 "
            "projection for D128"
        ),
    )
    parser.add_argument(
        "--backward-control-source",
        type=Path,
        help=(
            "precomposed D64 backward control; batched routes require this "
            "together with its SHA-256 and byte count"
        ),
    )
    parser.add_argument(
        "--backward-control-sha256",
        help="expected SHA-256 for --backward-control-source",
    )
    parser.add_argument(
        "--backward-control-bytes",
        type=int,
        help="expected byte count for --backward-control-source",
    )
    parser.add_argument(
        "--native-tk-d64-backward-extension",
        type=Path,
        help=(
            "authenticated native TK D64 E4M3 backward extension; requires "
            "--native-tk-d64-backward-module and is mutually exclusive with "
            "all --backward-control-* options"
        ),
    )
    parser.add_argument(
        "--native-tk-d64-backward-module",
        help=(
            "Python module identity exported by the native TK D64 backward "
            "extension; requires --native-tk-d64-backward-extension"
        ),
    )
    parser.add_argument("--native-tk-d128-backward-extension", type=Path)
    parser.add_argument("--native-tk-d128-backward-module")
    parser.add_argument("--native-tk-d128-backward-sha256")
    parser.add_argument("--native-tk-d128-backward-bytes", type=int)
    parser.add_argument(
        "--native-tk-d128-native-score-backward",
        action="store_true",
        help=(
            "select exact-batch native-NVFP4 score reconstruction; "
            "alone this selects the v508 represented-Q/K diagnostic, while "
            "--native-tk-d128-v509-e5m2-dout-backward composes the v509 "
            "accumulator-E4 Q/K/V + E5M2-dO route"
        ),
    )
    parser.add_argument(
        "--native-tk-d128-v509-e5m2-dout-backward",
        action="store_true",
        help=(
            "select the exact-batch v509 native-NVFP4-score backward "
            "with retained E4M3 Q/K/V and fused E5M2 dO publication; "
            "requires --native-tk-d128-native-score-backward and the exact "
            "v509 D128 extension"
        ),
    )
    parser.add_argument(
        "--forward-extension",
        type=Path,
        default=Path(
            "/tmp/_C_tk_gb200_causal_s4096_h32_d64."
            "cpython-312-aarch64-linux-gnu.so"
        ),
    )
    parser.add_argument(
        "--forward-module",
        default="_C_tk_gb200_causal_s4096_h32_d64",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    native_tk_backward_options = (
        args.native_tk_d64_backward_extension is not None,
        args.native_tk_d64_backward_module is not None,
    )
    if any(native_tk_backward_options) and not all(
        native_tk_backward_options
    ):
        raise ValueError(
            "--native-tk-d64-backward-extension and "
            "--native-tk-d64-backward-module must be supplied together"
        )
    native_tk_d64_backward = all(native_tk_backward_options)
    if (
        args.native_tk_d64_backward_module is not None
        and not args.native_tk_d64_backward_module.strip()
    ):
        raise ValueError("native TK D64 backward module must be non-empty")
    native_tk_d128_backward_options = (
        args.native_tk_d128_backward_extension is not None,
        args.native_tk_d128_backward_module is not None,
        args.native_tk_d128_backward_sha256 is not None,
        args.native_tk_d128_backward_bytes is not None,
    )
    if any(native_tk_d128_backward_options) and not all(
        native_tk_d128_backward_options
    ):
        raise ValueError(
            "native TK D128 backward requires extension, module, SHA256, and "
            "byte count together"
        )
    native_tk_d128_backward = all(native_tk_d128_backward_options)
    if (
        args.native_tk_d128_native_score_backward
        and not native_tk_d128_backward
    ):
        raise ValueError(
            "--native-tk-d128-native-score-backward requires a complete "
            "native TK D128 extension identity"
        )
    if (
        args.native_tk_d128_v509_e5m2_dout_backward
        and not native_tk_d128_backward
    ):
        raise ValueError(
            "--native-tk-d128-v509-e5m2-dout-backward requires a complete "
            "native TK D128 extension identity"
        )
    if (
        args.native_tk_d128_v509_e5m2_dout_backward
        and not args.native_tk_d128_native_score_backward
    ):
        raise ValueError(
            "--native-tk-d128-v509-e5m2-dout-backward requires "
            "--native-tk-d128-native-score-backward"
        )
    if native_tk_d64_backward and native_tk_d128_backward:
        raise ValueError(
            "native TK D64 and D128 backward extensions are mutually exclusive"
        )
    if (
        args.native_tk_d128_backward_module is not None
        and not args.native_tk_d128_backward_module.strip()
    ):
        raise ValueError("native TK D128 backward module must be non-empty")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("expose exactly one GPU to the benchmark")
    if args.warmups < 1 or args.samples < 1:
        raise ValueError("at least one warmup and one measured sample are required")
    torch.cuda.set_device(0)
    config = config_from_model_preset(
        args.model_preset,
        batch=args.batch,
        layers=args.layers,
    )
    if native_tk_d128_backward and config.head_dim != 128:
        raise ValueError("native TK D128 backward requires a D128 model preset")
    if native_tk_d64_backward and config.head_dim != 64:
        raise ValueError("native TK D64 backward requires a D64 model preset")
    (
        requested_exp2_degree,
        requested_exp2_period,
        requested_reuse_quantized_p,
    ) = _requested_backward_approximation_policy(config)
    authenticated_backward_control = _require_precomposed_backward_control(
        config,
        args.backward_control_source,
        args.backward_control_sha256,
        args.backward_control_bytes,
        native_tk_d64_backward=native_tk_d64_backward,
        native_tk_d128_backward=native_tk_d128_backward,
    )
    effective_projection_dgrad = (
        "nvfp4" if config.head_dim == 128 else "bf16"
    ) if args.projection_dgrad == "auto" else args.projection_dgrad
    forward_extension_path = args.forward_extension.resolve(strict=True)
    forward_extension_identity = {
        "path": str(forward_extension_path),
        "module": args.forward_module,
        **_source_content_identity(forward_extension_path),
    }
    extension, topology = _load_forward(
        args.forward_extension, args.forward_module, config
    )
    native_tk_d128_declared_identity = None
    if native_tk_d128_backward:
        assert args.native_tk_d128_backward_extension is not None
        assert args.native_tk_d128_backward_sha256 is not None
        assert args.native_tk_d128_backward_bytes is not None
        native_tk_d128_declared_identity = _require_declared_artifact_identity(
            "native TK D128 backward",
            args.native_tk_d128_backward_extension,
            args.native_tk_d128_backward_sha256,
            args.native_tk_d128_backward_bytes,
        )
    native_tk_backward_extension = (
        _load_extension(
            args.native_tk_d64_backward_extension,
            args.native_tk_d64_backward_module,
        )
        if native_tk_d64_backward
        else _load_extension(
            args.native_tk_d128_backward_extension,
            args.native_tk_d128_backward_module,
        )
        if native_tk_d128_backward
        else None
    )
    if native_tk_d128_backward:
        assert native_tk_backward_extension is not None
        assert native_tk_d128_declared_identity is not None
        assert args.native_tk_d128_backward_extension is not None
        loaded_identity = _require_authenticated_native_tk_extension(
            native_tk_backward_extension
        )
        if (
            loaded_identity["path"]
            != str(args.native_tk_d128_backward_extension.resolve())
            or loaded_identity["sha256"]
            != native_tk_d128_declared_identity["sha256"]
            or loaded_identity["bytes"]
            != native_tk_d128_declared_identity["bytes"]
        ):
            raise RuntimeError(
                "loaded native TK D128 image does not match the declared "
                "artifact identity"
            )
    batched_mx_split_v_backward = (
        config.head_dim == 64
        and topology.get("pv_format") == "mxfp4_e8m0_block32"
        and (
            config.batch != 1
            or args.experimental_native_nvfp4_projection_out
            or native_tk_d64_backward
        )
    )
    rope = _make_llama3_rope(config)

    token_generator = torch.Generator(device="cuda")
    token_generator.manual_seed(args.seed + 101)
    tokens = torch.randint(
        config.vocab,
        (config.batch, config.sequence),
        generator=token_generator,
        device="cuda",
    )
    targets = torch.roll(tokens, shifts=-1, dims=1)
    loss_function, loss_receipt = _make_loss_function(
        compile_loss=args.compile_loss
    )

    bf16, bf16_logits, bf16_gradients, bf16_forward_dispatch = (
        _benchmark_route(
        BF16_ATTENTION_ROUTES[args.bf16_attention_control],
        config,
        rope,
        tokens,
        targets,
        runtime=None,
        seed=args.seed,
        warmups=args.warmups,
        samples=args.samples,
        learning_rate=args.learning_rate,
        loss_function=loss_function,
        loss_receipt=loss_receipt,
        bf16_attention_control=args.bf16_attention_control,
        )
    )
    if bf16_forward_dispatch is not None:
        raise RuntimeError("BF16 benchmark unexpectedly emitted lowp dispatch")
    runtime = LowpAttentionRuntime(
        config,
        rope,
        forward_extension=extension,
        forward_topology=topology,
        loss_scale=args.loss_scale,
        gradient_global_scale=args.gradient_global_scale,
        projection_dgrad=effective_projection_dgrad,
        qkv_projection_format=args.qkv_projection_format,
        output_projection_format=args.output_projection_format,
        experimental_native_nvfp4_projection_out=(
            args.experimental_native_nvfp4_projection_out
        ),
        experimental_fused_attention_rmsnorm_nvfp4=(
            args.experimental_fused_attention_rmsnorm_nvfp4
        ),
        backward_exp2_degree=requested_exp2_degree,
        backward_exp2_period=requested_exp2_period,
        backward_fp8_ds_lift=16,
        backward_reuse_quantized_p=(
            requested_reuse_quantized_p and not native_tk_d128_backward
        ),
        backward_control_source=args.backward_control_source,
        backward_control_sha256=args.backward_control_sha256,
        backward_control_bytes=args.backward_control_bytes,
        backward_forward_mx_probability_replay=False,
        backward_forward_mx_probability_scale_handoff=False,
        backward_match_forward_operands=(
            args.backward_match_forward_operands
        ),
        per_block_qk_scales=args.per_block_qk_scales,
        experimental_split_v_backward=batched_mx_split_v_backward,
        experimental_output_shared_split_v=(
            args.experimental_output_shared_split_v
        ),
        backward_probability_correction=(
            args.backward_probability_correction
        ),
        q_quant_scale=args.q_quant_scale,
        k_quant_scale=args.k_quant_scale,
        projection_weight_scale_2d=(
            args.projection_weight_scaling == "2d"
        ),
        v_mxfp4_scale_2d=(args.v_mxfp4_scaling == "2d"),
        adaptive_qk_weight_scales=args.adaptive_qk_weight_scales,
        native_tk_d64_backward_extension=(
            native_tk_backward_extension
            if native_tk_d64_backward
            else None
        ),
        native_tk_d128_backward_extension=(
            native_tk_backward_extension
            if native_tk_d128_backward
            else None
        ),
        native_tk_d128_native_score_backward=(
            args.native_tk_d128_native_score_backward
        ),
        native_tk_d128_v509_e5m2_dout_backward=(
            args.native_tk_d128_v509_e5m2_dout_backward
        ),
    )
    lowp, _, _, forward_dispatch = _benchmark_route(
        "fp4_fa4_fused_qkv_rope",
        config,
        rope,
        tokens,
        targets,
        runtime=runtime,
        seed=args.seed,
        warmups=args.warmups,
        samples=args.samples,
        learning_rate=args.learning_rate,
        loss_function=loss_function,
        loss_receipt=loss_receipt,
        reference_logits=bf16_logits,
        reference_gradients=bf16_gradients,
    )
    if forward_dispatch is None:
        raise RuntimeError("lowp benchmark omitted its live-workspace dispatch")
    bf16_ms = bf16["steady_state"]["step_ms"]
    lowp_ms = lowp["steady_state"]["step_ms"]
    result = {
        "configuration": {
            **config.__dict__,
            "batch": config.batch,
            "warmups": args.warmups,
            "samples": args.samples,
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "optimizer": "fused AdamW",
            "loss": dict(loss_receipt),
            "bf16_attention_control": args.bf16_attention_control,
            "bf16_attention_route": BF16_ATTENTION_ROUTES[
                args.bf16_attention_control
            ],
            "loss_scale": args.loss_scale,
            "gradient_global_scale": args.gradient_global_scale,
            "q_quant_scale": args.q_quant_scale,
            "k_quant_scale": args.k_quant_scale,
            "qkv_projection_format": args.qkv_projection_format,
            "output_projection_format": args.output_projection_format,
            "output_projection_topology": dict(
                runtime.output_projection_topology
            ),
            "experimental_native_nvfp4_projection_out": (
                args.experimental_native_nvfp4_projection_out
            ),
            "experimental_fused_attention_rmsnorm_nvfp4": (
                args.experimental_fused_attention_rmsnorm_nvfp4
            ),
            "experimental_output_shared_split_v_requested": (
                args.experimental_output_shared_split_v
            ),
            "experimental_output_shared_split_v_resolved": (
                runtime.experimental_output_shared_split_v
            ),
            "output_shared_split_v_path": (
                runtime.output_shared_split_v_path
            ),
            "output_shared_split_v_checked_symbol": (
                getattr(runtime.qkv_projection, "checked_symbol", None)
            ),
            "backward_match_forward_operands": (
                args.backward_match_forward_operands
            ),
            "authenticated_backward_control": (
                authenticated_backward_control
            ),
            "native_tk_d64_backward": runtime.native_tk_d64_backward,
            "native_tk_d64_backward_extension": (
                runtime.native_tk_d64_backward_extension_identity
            ),
            "native_tk_d128_backward": runtime.native_tk_d128_backward,
            "native_tk_d128_backward_extension": (
                runtime.native_tk_d128_backward_extension_identity
            ),
            "native_tk_d128_native_score_backward": (
                runtime.native_tk_d128_native_score_backward
            ),
            "native_tk_d128_v509_e5m2_dout_backward": (
                runtime.native_tk_d128_v509_e5m2_dout_backward
            ),
            "v509_e5m2_dout_route": runtime.v509_e5m2_dout_route,
            "per_block_qk_scales": args.per_block_qk_scales,
            "projection_weight_scaling": args.projection_weight_scaling,
            "v_mxfp4_scaling": args.v_mxfp4_scaling,
            "adaptive_qk_weight_scales": args.adaptive_qk_weight_scales,
            "projection_dgrad_requested": args.projection_dgrad,
            "projection_dgrad": effective_projection_dgrad,
            "backward_probability_correction": (
                runtime.backward_probability_correction
            ),
            "backward_attention_branch_gain": (
                runtime.backward_probability_correction
            ),
            "backward_q_gain": runtime.backward_q_gain,
            "backward_k_gain": runtime.backward_k_gain,
            "backward_v_gain": runtime.backward_v_gain,
            "backward_exp2_degree": runtime.backward_exp2_degree,
            "backward_exp2_period": runtime.backward_exp2_period,
            "backward_exp2_requested_degree": (
                runtime.backward_exp2_requested_degree
            ),
            "backward_exp2_requested_period": (
                runtime.backward_exp2_requested_period
            ),
            "backward_exp2_policy": runtime.backward_exp2_policy,
            "backward_detached_fp8_p_tmem": (
                runtime.backward_detached_fp8_p_tmem
            ),
            "backward_probability_tmem_policy": (
                runtime.backward_probability_tmem_policy
            ),
            "backward_head_fast_raster": runtime.backward_head_fast_raster,
            "backward_raster_policy": runtime.backward_raster_policy,
            "backward_contract": runtime.backward_contract(),
            "backward_shape_policy": runtime.backward_shape_policy,
            # The extension mutates its live topology receipt after the first
            # authenticated launch (for example ``valid`` and descriptor-cache
            # counters).  Serialize that post-launch state rather than the
            # stale pre-launch dictionary returned by ``_load_forward``.
            "forward_topology": dict(runtime.forward_topology),
            "forward_extension": forward_extension_identity,
            "lowp_runtime_extension": (
                b300_lowp_bwd_extension_artifact_identity()
            ),
            "forward_dispatch": forward_dispatch,
            "useful_flops_per_sequence": (
                _useful_flops(config) / config.batch
            ),
        },
        **(
            {
                "d128_mxfp4_v_operand_cache": (
                    runtime.d128_mxfp4_v_operand_cache_receipt()
                )
            }
            if runtime.experimental_d128_mxfp4_v_backward
            else {}
        ),
        "bf16": bf16,
        "lowp": lowp,
        "comparison": {
            "speedup_lowp_over_bf16": bf16_ms / lowp_ms,
            "step_time_reduction_percent": (1.0 - lowp_ms / bf16_ms) * 100.0,
            "tokens_per_second_speedup": (
                lowp["steady_state"]["tokens_per_second"]
                / bf16["steady_state"]["tokens_per_second"]
            ),
            "mfu_delta_percentage_points": 100.0
            * (
                lowp["steady_state"]["mfu_at_2250_tflops"]
                - bf16["steady_state"]["mfu_at_2250_tflops"]
            ),
        },
    }
    result = _strict_json_value(result)
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
