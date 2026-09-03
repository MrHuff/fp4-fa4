#
# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
#
"""Fail-closed Llama adapter for the validated causal low-precision FA4 route.

This converter deliberately exposes one narrow training contract:

* sequence length 4096 with local batch exactly matching the authenticated
  compiled artifact batch;
* local batch one, the authenticated local batch 16 D64 route, or the current
  local batch 1/2/4 D128 route when the complete source/runtime/artifact bundle
  exposes the matching batched ABI;
* NVFP4 Q/K attention operands with either FP8-PV or MXFP4-PV forward;
* native ThunderKittens v416 backward for the D64/B16 profile; the retained
  CuTe backward control remains a distinct legacy/control route;
* the current packed D64 B16 dense-E4M3 route and split-QKV D128 B1/B2/B4 paired
  E4M3 or NVFP4 learned QKV/O projections, plus the authenticated legacy
  split-QKV D64/D128 FP8 routes; output-projection dgrad remains explicitly
  NVFP4 in the current runtime;
* replicated data parallelism only.

The implementation is an adapter, not a second copy of the kernels.  It loads
the authenticated runtime from an fp4_matmul source capsule lazily on the first
CUDA forward.  Consequently model conversion remains safe while TorchTitan's
parameters are still on ``meta``.  Current D64 B16 owns one packed QKV leaf
and can import a complete split-Q/K/V model state; legacy D64 and D128 retain
their split parameters.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import os
import re
import sys
import threading
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from types import MethodType
from typing import Any, List, Union

import torch
from torch import nn

from torchtitan.distributed import ParallelDims
from torchtitan.models.llama3.model.model import apply_rotary_emb
from torchtitan.protocols.model_converter import (
    ModelConverter,
    register_model_converter,
)
from torchtitan.tools.logging import logger

from .converters import _safe_trunc_normal_
from .job_config import JobConfig


_BACKWARD_EXTENSION_ENV = "TK_FA4_LOWP_BWD_EXTENSION_SOURCE"
_EXACT_CONVERTER = "fa4_exact_lowp_attention"
_LEGACY_EXACT_CONVERTER = "fa4_exact_nvfp4_qk_fp8_pv"
_EXACT_CONVERTERS = (_EXACT_CONVERTER, _LEGACY_EXACT_CONVERTER)
_EXACT_FP8_PV = "e4m3_fp8"
_EXACT_MXFP4_PV = "mxfp4_e8m0_block32"
_EXACT_PV_FORMATS = (_EXACT_FP8_PV, _EXACT_MXFP4_PV)
_EXACT_E4M3_PROJECTIONS = "e4m3"
_EXACT_NVFP4_PROJECTIONS = "nvfp4"
_EXACT_LEARNED_PROJECTION_FORMATS = (
    _EXACT_E4M3_PROJECTIONS,
    _EXACT_NVFP4_PROJECTIONS,
)
_EXACT_RETAINED_SPLIT_V = "retained_split"
_EXACT_OUTPUT_SHARED_SPLIT_V = "output_shared_split"
_EXACT_D128_SHARED_D32XS32_V = "shared_d32xs32_forward_anchors"
_EXACT_MX_V_PUBLICATIONS = (
    _EXACT_RETAINED_SPLIT_V,
    _EXACT_OUTPUT_SHARED_SPLIT_V,
    _EXACT_D128_SHARED_D32XS32_V,
)
_EXACT_ROUTE_BY_FORMATS = {
    (_EXACT_NVFP4_PROJECTIONS, _EXACT_FP8_PV): "nvfp4_qk_fp8_pv",
    (_EXACT_NVFP4_PROJECTIONS, _EXACT_MXFP4_PV): "nvfp4_qk_mxfp4_pv",
    (
        _EXACT_E4M3_PROJECTIONS,
        _EXACT_FP8_PV,
    ): "e4m3_proj_nvfp4_qk_fp8_pv",
    (
        _EXACT_E4M3_PROJECTIONS,
        _EXACT_MXFP4_PV,
    ): "e4m3_proj_nvfp4_qk_mxfp4_pv",
}
_CURRENT_AUTOGRAD_ABI = "packed_qkv_fused_rmsnorm_v2"
_LEGACY_AUTOGRAD_ABI = "split_qkv_v1"
_CURRENT_AUTOGRAD_PARAMETERS = (
    "ctx",
    "x",
    "attention_norm_weight",
    "packed_qkv_weight",
    "q_weight",
    "k_weight",
    "v_weight",
    "out_weight",
    "qk_scales",
    "forward_workspace",
    "runtime",
)
_LEGACY_AUTOGRAD_PARAMETERS = (
    "ctx",
    "x",
    "q_weight",
    "k_weight",
    "v_weight",
    "out_weight",
    "qk_scales",
    "forward_workspace",
    "runtime",
)
_BF16_TOPOLOGY_CONVERTER = "fa4_exact_bf16_topology"
_BF16_ROUTE = "bf16_fa4"
_EXACT_SEQUENCE = 4096
_EXACT_ARTIFACT_BATCH = 1
_D64_AUTHENTICATED_BATCHES = (1, 16)
_D128_AUTHENTICATED_BATCHES = (1, 2, 4)
_D128_V509_AUTHENTICATED_BATCHES = (1, 4)
_D128_CURRENT_BATCH = 2
_D128_CURRENT_BATCHES = _D128_AUTHENTICATED_BATCHES
_D64_BUILD_PROFILE = "llama1p2b-d64-b16"
_D128_BUILD_PROFILE_PREFIX = "llama8b-d128-b"
_NATIVE_TK_D64_BACKEND = "native_tk_d64_e4m3"
_NATIVE_TK_D64_V416_SOURCE = "v416_d64_gqa_e4m3_production_bshd_dq_first_vec2_ds_v1"
_NATIVE_TK_D64_V416_MODULE = "_C_sm100_gqa_tk_v416_d64_e4m3_production_bshd_dq_first"
_NATIVE_TK_D128_E4M3_BACKEND = "native_tk_d128_e4m3"
_NATIVE_TK_D128_V501_SOURCE = "v501_d128_gqa_e4m3_unified_best_route_production_bshd_v1"
_NATIVE_TK_D128_NVFP4_SCORE_BACKEND = "native_tk_d128_nvfp4_score_e4m3_gradient"
_NATIVE_TK_D128_V508_SOURCE = (
    "v508_native_nvfp4_score_e4m3_gradient_b1_s4096_experimental_v1"
)
_NATIVE_TK_D128_NVFP4_SCORE_E5M2_BACKEND = (
    "native_tk_d128_nvfp4_score_e4m3_qkv_e5m2_dout"
)
_NATIVE_TK_D128_V509_SOURCE = (
    "v509_native_nvfp4_score_e4m3_qkv_e5m2_dout_b1_s4096_experimental_v1"
)


def _native_tk_d128_v509_source(batch: int) -> str:
    if batch == 1:
        return _NATIVE_TK_D128_V509_SOURCE
    return (
        "v509_native_nvfp4_score_e4m3_qkv_e5m2_dout_" f"b{batch}_s4096_experimental_v1"
    )


_D128_V509_E5M2_DOUT_SOURCE = "projection_accumulator_e5m2_x4"
_D128_V509_E5M2_DOUT_KERNEL = "b300_project_dout_unified_lowp_nvfp4_v509_e5m2"
_D128_V509_E5M2_DOUT_ROUTE = "v509_only_fail_closed"
_D128_V509_E5M2_DOUT_PUBLISHER_SOURCE = (
    "v509_fused_nvfp4_output_projection_e5m2_dout_b1_b2_b4_s4096_v1"
)
_D128_V509_E5M2_DOUT_LEGACY_B1_PUBLISHER_SOURCE = (
    "v509_fused_nvfp4_output_projection_e5m2_dout_b1_s4096_v1"
)
_D128_E4M3_PROJECTION_SYMBOL_PREFIX = (
    "project_qkv_gqa_d128_unified_fp8_e4m3_rope_packed_clustered"
)
_D128_NVFP4_PROJECTION_SYMBOL_PREFIX = (
    "project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered"
)
_D128_NVFP4_REPRESENTED_QK_FP8_SYMBOL = (
    _D128_NVFP4_PROJECTION_SYMBOL_PREFIX
    + "_represented_backward_perblock_qk_fp8_forward_out"
)
# The batch-aware D64 runtime owns twelve dense publication allocations per
# layer.  Their logical storage is 21,889,184 bytes per sample and layer.  This
# exact floor rejects a fake batched view over B=1 storage before any kernel is
# dispatched; sixteen D64 decoder layers therefore own 5,603,631,104 bytes at
# B=16.
_D64_FORWARD_WORKSPACE_OWNER_COUNT = 12
_D64_FORWARD_WORKSPACE_BYTES_PER_BATCH_PER_LAYER = 21_889_184
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_absolute_file(
    value: str,
    *,
    label: str,
    expected_sha256: str,
    expected_bytes: int | None = None,
) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path, got {value!r}")
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    normalized_sha256 = expected_sha256.lower()
    if _SHA256_PATTERN.fullmatch(normalized_sha256) is None:
        raise ValueError(f"{label} requires an exact lowercase SHA256 identity")
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"{label} byte identity mismatch: {path.stat().st_size} != "
            f"{expected_bytes}"
        )
    actual_sha256 = _sha256(path)
    if actual_sha256 != normalized_sha256:
        raise RuntimeError(
            f"{label} SHA256 mismatch: {actual_sha256} != " f"{normalized_sha256}"
        )
    return path


def _module_is_below(module: Any, root: Path) -> bool:
    module_file = getattr(module, "__file__", None)
    if not module_file:
        return False
    try:
        return Path(module_file).resolve().is_relative_to(root)
    except OSError:
        return False


@dataclass(frozen=True)
class _ExactSettings:
    source_root: Path
    runtime_source_sha256: str
    flash_attn_root: Path
    flash_attn_source_sha256: str
    cutlass_dsl_root: Path
    cutlass_dsl_version: str
    cutlass_dsl_native_sha256: str
    artifact_profile: str
    forward_extension: Path
    forward_module: str
    forward_sha256: str
    forward_batch_size: int
    pv_format: str
    learned_projection_format: str
    d128_represented_qk_backward: bool
    d128_native_score_backward: bool
    d128_e5m2_dout_backward: bool
    mx_v_publication: str
    backward_extension: Path
    backward_sha256: str
    backward_control_source: Path | None
    backward_control_sha256: str | None
    backward_control_bytes: int | None
    allow_fp32_master_shadows: bool
    native_tk_d64_backward_extension: Path | None = None
    native_tk_d64_backward_module: str | None = None
    native_tk_d64_backward_sha256: str | None = None
    native_tk_d64_backward_bytes: int | None = None
    native_tk_d128_backward_extension: Path | None = None
    native_tk_d128_backward_module: str | None = None
    native_tk_d128_backward_sha256: str | None = None
    native_tk_d128_backward_bytes: int | None = None

    @classmethod
    def from_job_config(cls, job_config: JobConfig) -> "_ExactSettings":
        cfg = job_config.fa4
        source_root = Path(cfg.exact_source_root)
        if not source_root.is_absolute():
            raise ValueError("fa4.exact_source_root must be an absolute path")
        source_root = source_root.resolve()
        runtime_source = (
            source_root / "tk_fa4" / "lowp_fa4_bwd" / "benchmark_llama12b_e2e.py"
        )
        flash_attn_root = Path(
            cfg.exact_flash_attn_root or str(source_root / "flash-attention")
        )
        if not flash_attn_root.is_absolute():
            raise ValueError("fa4.exact_flash_attn_root must be an absolute path")
        flash_attn_root = flash_attn_root.resolve()
        flash_attn_source = flash_attn_root / "flash_attn" / "cute" / "interface.py"
        required_sources = (runtime_source, flash_attn_source)
        for required_source in required_sources:
            if not required_source.is_file():
                raise FileNotFoundError(
                    "exact FA4 source capsule is incomplete: "
                    f"{required_source} is missing"
                )
        runtime_source_sha256 = cfg.exact_runtime_source_sha256.lower()
        if _SHA256_PATTERN.fullmatch(runtime_source_sha256) is None:
            raise ValueError(
                "fa4.exact_runtime_source_sha256 requires an exact lowercase "
                "SHA256 identity"
            )
        actual_runtime_source_sha256 = _sha256(runtime_source)
        if actual_runtime_source_sha256 != runtime_source_sha256:
            raise RuntimeError(
                "exact LowpAttention runtime source SHA256 mismatch: "
                f"{actual_runtime_source_sha256} != {runtime_source_sha256}"
            )
        flash_attn_source_sha256 = cfg.exact_flash_attn_source_sha256.lower()
        if _SHA256_PATTERN.fullmatch(flash_attn_source_sha256) is None:
            raise ValueError(
                "fa4.exact_flash_attn_source_sha256 requires an exact "
                "lowercase SHA256 identity"
            )
        actual_flash_attn_source_sha256 = _sha256(flash_attn_source)
        if actual_flash_attn_source_sha256 != flash_attn_source_sha256:
            raise RuntimeError(
                "exact FlashAttention interface source SHA256 mismatch: "
                f"{actual_flash_attn_source_sha256} != "
                f"{flash_attn_source_sha256}"
            )
        cutlass_dsl_root = Path(cfg.exact_cutlass_dsl_root)
        if not cutlass_dsl_root.is_absolute():
            raise ValueError("fa4.exact_cutlass_dsl_root must be an absolute path")
        cutlass_dsl_root = cutlass_dsl_root.resolve()
        cutlass_init = cutlass_dsl_root / "cutlass" / "__init__.py"
        if not cutlass_init.is_file():
            raise FileNotFoundError(
                f"exact CUTLASS DSL package is incomplete: {cutlass_init}"
            )
        native_candidates = tuple(
            (cutlass_dsl_root / "cutlass" / "_mlir" / "_mlir_libs").glob(
                "_cutlass_ir*.so"
            )
        )
        if len(native_candidates) != 1:
            raise RuntimeError(
                "exact CUTLASS DSL requires one _cutlass_ir native library; "
                f"found {len(native_candidates)}"
            )
        cutlass_native_sha256 = cfg.exact_cutlass_dsl_native_sha256.lower()
        if _SHA256_PATTERN.fullmatch(cutlass_native_sha256) is None:
            raise ValueError(
                "fa4.exact_cutlass_dsl_native_sha256 requires an exact "
                "lowercase SHA256 identity"
            )
        actual_cutlass_native_sha256 = _sha256(native_candidates[0])
        if actual_cutlass_native_sha256 != cutlass_native_sha256:
            raise RuntimeError(
                "exact CUTLASS DSL native SHA256 mismatch: "
                f"{actual_cutlass_native_sha256} != {cutlass_native_sha256}"
            )
        if cfg.exact_cutlass_dsl_version != "4.5.2":
            raise ValueError("exact FA4 currently authenticates CUTLASS DSL 4.5.2")
        artifact_profile = str(getattr(cfg, "exact_artifact_profile", ""))
        valid_artifact_profiles = {
            "",
            _D64_BUILD_PROFILE,
            *(
                f"{_D128_BUILD_PROFILE_PREFIX}{batch}"
                for batch in _D128_AUTHENTICATED_BATCHES
            ),
        }
        if artifact_profile not in valid_artifact_profiles:
            raise ValueError(
                "fa4.exact_artifact_profile is not a supported build profile: "
                f"{artifact_profile!r}"
            )
        if not cfg.exact_forward_module.isidentifier():
            raise ValueError(
                "fa4.exact_forward_module must be the extension's Python " "identifier"
            )
        # Pre-selector exact configs are the authenticated FP8-PV route. Keep
        # those historical Dolma/D128 recipes loadable while requiring every
        # new matched SlimPajama render to bind an explicit route selector.
        pv_format = (
            _EXACT_FP8_PV if cfg.exact_pv_format is None else str(cfg.exact_pv_format)
        )
        if pv_format not in _EXACT_PV_FORMATS:
            raise ValueError(
                "fa4.exact_pv_format must be 'e4m3_fp8' or " "'mxfp4_e8m0_block32'"
            )
        learned_projection_format = str(
            getattr(
                cfg,
                "exact_learned_projection_format",
                _EXACT_E4M3_PROJECTIONS,
            )
        )
        if learned_projection_format not in _EXACT_LEARNED_PROJECTION_FORMATS:
            raise ValueError(
                "fa4.exact_learned_projection_format must be 'e4m3' or " "'nvfp4'"
            )
        d128_represented_qk_backward = getattr(
            cfg,
            "exact_d128_represented_qk_backward",
            False,
        )
        if type(d128_represented_qk_backward) is not bool:
            raise TypeError(
                "fa4.exact_d128_represented_qk_backward must be exactly bool"
            )
        d128_native_score_backward = getattr(
            cfg,
            "exact_d128_native_score_backward",
            False,
        )
        if type(d128_native_score_backward) is not bool:
            raise TypeError("fa4.exact_d128_native_score_backward must be exactly bool")
        d128_e5m2_dout_backward = getattr(
            cfg,
            "exact_d128_e5m2_dout_backward",
            False,
        )
        if type(d128_e5m2_dout_backward) is not bool:
            raise TypeError("fa4.exact_d128_e5m2_dout_backward must be exactly bool")
        mx_v_publication = str(
            getattr(
                cfg,
                "exact_mx_v_publication",
                _EXACT_RETAINED_SPLIT_V,
            )
        )
        if mx_v_publication not in _EXACT_MX_V_PUBLICATIONS:
            raise ValueError(
                "fa4.exact_mx_v_publication must be 'retained_split', "
                "'output_shared_split', or "
                "'shared_d32xs32_forward_anchors'"
            )
        forward_extension = _require_absolute_file(
            cfg.exact_forward_extension,
            label=f"exact {pv_format} forward artifact",
            expected_sha256=cfg.exact_forward_sha256,
        )
        backward_extension = _require_absolute_file(
            cfg.exact_backward_extension,
            label="exact low-precision projection artifact",
            expected_sha256=cfg.exact_backward_sha256,
        )
        forward_batch_size = int(cfg.exact_forward_batch_size)
        authenticated_batches = tuple(
            sorted(set(_D64_AUTHENTICATED_BATCHES) | set(_D128_AUTHENTICATED_BATCHES))
        )
        if forward_batch_size not in authenticated_batches:
            raise ValueError(
                "fa4.exact_forward_batch_size must be one of {1, 2, 4, 16}"
            )
        if mx_v_publication == _EXACT_OUTPUT_SHARED_SPLIT_V and not (
            pv_format == _EXACT_MXFP4_PV and forward_batch_size == 16
        ):
            raise ValueError(
                "fa4.exact_mx_v_publication='output_shared_split' requires "
                "the D64 B16 NVFP4-QK/MXFP4-PV split-V route"
            )

        d64_native_path_value = str(
            getattr(cfg, "exact_native_tk_d64_backward_extension", "")
        )
        d64_native_module_value = str(
            getattr(cfg, "exact_native_tk_d64_backward_module", "")
        )
        d64_native_sha_value = str(
            getattr(cfg, "exact_native_tk_d64_backward_sha256", "")
        )
        d64_native_bytes_value = int(
            getattr(cfg, "exact_native_tk_d64_backward_bytes", 0)
        )
        d64_native_supplied = (
            bool(d64_native_path_value),
            bool(d64_native_module_value),
            bool(d64_native_sha_value),
            d64_native_bytes_value != 0,
        )
        if any(d64_native_supplied) and not all(d64_native_supplied):
            raise ValueError(
                "fa4 native-TK D64 backward requires extension, module, "
                "SHA256, and byte identity together"
            )
        d64_native_extension: Path | None = None
        d64_native_module: str | None = None
        d64_native_sha256: str | None = None
        d64_native_bytes: int | None = None
        if all(d64_native_supplied):
            if d64_native_module_value != _NATIVE_TK_D64_V416_MODULE:
                raise ValueError(
                    "fa4 native-TK D64 backward module must identify the "
                    "authenticated v416 image"
                )
            if d64_native_bytes_value <= 0:
                raise ValueError(
                    "fa4.exact_native_tk_d64_backward_bytes must be positive"
                )
            d64_native_extension = _require_absolute_file(
                d64_native_path_value,
                label="exact native TK D64 v416 backward artifact",
                expected_sha256=d64_native_sha_value,
                expected_bytes=d64_native_bytes_value,
            )
            if d64_native_extension == backward_extension:
                raise ValueError(
                    "exact projection publisher and native D64 backward "
                    "artifacts must be distinct files"
                )
            d64_native_module = d64_native_module_value
            d64_native_sha256 = d64_native_sha_value.lower()
            d64_native_bytes = d64_native_bytes_value

        native_path_value = str(
            getattr(cfg, "exact_native_tk_d128_backward_extension", "")
        )
        native_module_value = str(
            getattr(cfg, "exact_native_tk_d128_backward_module", "")
        )
        native_sha_value = str(getattr(cfg, "exact_native_tk_d128_backward_sha256", ""))
        native_bytes_value = int(getattr(cfg, "exact_native_tk_d128_backward_bytes", 0))
        native_supplied = (
            bool(native_path_value),
            bool(native_module_value),
            bool(native_sha_value),
            native_bytes_value != 0,
        )
        if any(native_supplied) and not all(native_supplied):
            raise ValueError(
                "fa4 current D128 native-TK backward requires extension, "
                "module, SHA256, and byte identity together"
            )
        native_extension: Path | None = None
        native_module: str | None = None
        native_sha256: str | None = None
        native_bytes: int | None = None
        if all(native_supplied):
            if not native_module_value.isidentifier():
                raise ValueError(
                    "fa4.exact_native_tk_d128_backward_module must be the "
                    "extension's Python identifier"
                )
            if native_bytes_value <= 0:
                raise ValueError(
                    "fa4.exact_native_tk_d128_backward_bytes must be positive"
                )
            native_extension = _require_absolute_file(
                native_path_value,
                label="exact native TK D128 backward artifact",
                expected_sha256=native_sha_value,
                expected_bytes=native_bytes_value,
            )
            if native_extension == backward_extension:
                raise ValueError(
                    "exact projection and native D128 backward artifacts must "
                    "be distinct files"
                )
            native_module = native_module_value
            native_sha256 = native_sha_value.lower()
            native_bytes = native_bytes_value

        if d64_native_extension is not None and native_extension is not None:
            raise ValueError("select exactly one native-TK backward shape profile")

        current_d128_artifacts = native_extension is not None
        if forward_batch_size in (2, 4) and not (current_d128_artifacts):
            raise ValueError(
                "the current D128 B2/B4 route requires an authenticated "
                "native-TK backward artifact"
            )
        if current_d128_artifacts:
            if forward_batch_size not in _D128_CURRENT_BATCHES:
                raise ValueError(
                    "the current D128 native-TK route requires local batch "
                    "1, 2, or 4"
                )
            # The paired D128 learned-projection routes change only their
            # forward QKV/O format and forward V payload. Both PV routes
            # publish the same direct-accumulator E4M3 V for v501 backward.
            expected_publication = _EXACT_RETAINED_SPLIT_V
            if mx_v_publication != expected_publication:
                raise ValueError(
                    "the current D128 B1/B2/B4 route requires "
                    f"fa4.exact_mx_v_publication={expected_publication!r}"
                )
        elif learned_projection_format != _EXACT_E4M3_PROJECTIONS:
            raise ValueError(
                "NVFP4 learned QKV/O projections are authenticated only for "
                "the current D128 B1/B2/B4 route"
            )
        elif mx_v_publication == _EXACT_D128_SHARED_D32XS32_V:
            raise ValueError(
                "shared_d32xs32_forward_anchors is not authenticated by the "
                "current paired-projection D128/v501 route"
            )
        if d128_represented_qk_backward:
            if not current_d128_artifacts:
                raise ValueError(
                    "represented D128 Q/K backward requires the authenticated "
                    "current native-TK D128 artifact"
                )
            if d128_e5m2_dout_backward:
                raise ValueError(
                    "E5M2-dO D128 backward is incompatible with represented "
                    "Q/K backward and requires ordinary retained E4M3 Q/K/V"
                )
            if learned_projection_format != _EXACT_NVFP4_PROJECTIONS:
                raise ValueError(
                    "represented D128 Q/K backward requires NVFP4 learned "
                    "QKV/O projections"
                )
            if pv_format != _EXACT_FP8_PV:
                raise ValueError(
                    "represented D128 Q/K backward is authenticated only for " "FP8-PV"
                )
            if mx_v_publication != _EXACT_RETAINED_SPLIT_V:
                raise ValueError(
                    "represented D128 Q/K backward requires retained_split V"
                )
        if d128_native_score_backward:
            if not d128_e5m2_dout_backward and not d128_represented_qk_backward:
                raise ValueError(
                    "native-score D128 backward requires represented Q/K "
                    "gradient operands"
                )
            if forward_batch_size not in _D128_V509_AUTHENTICATED_BATCHES:
                raise ValueError(
                    "native-score D128 backward is authenticated only for " "B1/B4"
                )
        if d128_e5m2_dout_backward:
            if not current_d128_artifacts:
                raise ValueError(
                    "E5M2-dO D128 backward requires the authenticated current "
                    "native-TK D128 artifact"
                )
            if forward_batch_size not in _D128_V509_AUTHENTICATED_BATCHES:
                raise ValueError(
                    "E5M2-dO D128 backward is authenticated only for B1/B4"
                )
            if d128_represented_qk_backward:
                raise ValueError(
                    "E5M2-dO D128 backward is incompatible with represented "
                    "Q/K backward"
                )
            if not d128_native_score_backward:
                raise ValueError("E5M2-dO D128 backward requires native-score backward")
            if mx_v_publication != _EXACT_RETAINED_SPLIT_V:
                raise ValueError("E5M2-dO D128 backward requires retained_split E4M3 V")
        control_source: Path | None = None
        control_sha256: str | None = None
        control_bytes: int | None = None
        any_control = bool(
            cfg.exact_backward_control_source
            or cfg.exact_backward_control_sha256
            or cfg.exact_backward_control_bytes
        )
        if any_control and d64_native_extension is not None:
            raise ValueError(
                "native D64 v416 and the CuTe backward control are distinct "
                "backends and cannot be selected together"
            )
        if any_control:
            if cfg.exact_backward_control_bytes <= 0:
                raise ValueError(
                    "fa4.exact_backward_control_bytes must be positive when "
                    "a D64 control is configured"
                )
            control_source = _require_absolute_file(
                cfg.exact_backward_control_source,
                label="exact D64 backward control",
                expected_sha256=cfg.exact_backward_control_sha256,
                expected_bytes=cfg.exact_backward_control_bytes,
            )
            control_sha256 = cfg.exact_backward_control_sha256.lower()
            control_bytes = int(cfg.exact_backward_control_bytes)

        return cls(
            source_root=source_root,
            runtime_source_sha256=runtime_source_sha256,
            flash_attn_root=flash_attn_root,
            flash_attn_source_sha256=flash_attn_source_sha256,
            cutlass_dsl_root=cutlass_dsl_root,
            cutlass_dsl_version=cfg.exact_cutlass_dsl_version,
            cutlass_dsl_native_sha256=cutlass_native_sha256,
            artifact_profile=artifact_profile,
            forward_extension=forward_extension,
            forward_module=cfg.exact_forward_module,
            forward_sha256=cfg.exact_forward_sha256.lower(),
            forward_batch_size=forward_batch_size,
            pv_format=pv_format,
            learned_projection_format=learned_projection_format,
            d128_represented_qk_backward=d128_represented_qk_backward,
            d128_native_score_backward=d128_native_score_backward,
            d128_e5m2_dout_backward=d128_e5m2_dout_backward,
            mx_v_publication=mx_v_publication,
            backward_extension=backward_extension,
            backward_sha256=cfg.exact_backward_sha256.lower(),
            native_tk_d64_backward_extension=d64_native_extension,
            native_tk_d64_backward_module=d64_native_module,
            native_tk_d64_backward_sha256=d64_native_sha256,
            native_tk_d64_backward_bytes=d64_native_bytes,
            native_tk_d128_backward_extension=native_extension,
            native_tk_d128_backward_module=native_module,
            native_tk_d128_backward_sha256=native_sha256,
            native_tk_d128_backward_bytes=native_bytes,
            backward_control_source=control_source,
            backward_control_sha256=control_sha256,
            backward_control_bytes=control_bytes,
            allow_fp32_master_shadows=bool(cfg.exact_allow_fp32_master_shadows),
        )


@dataclass(frozen=True)
class _ExactShapeProfile:
    name: str
    contract_name: str
    model_preset: str
    hidden: int
    q_heads: int
    kv_heads: int
    head_dim: int
    layers: int
    tied_embeddings: bool
    expected_unique_parameters: int
    rope_scaling_factor: float
    qkv_projection_format: str
    output_projection_format: str
    projection_dgrad: str
    per_block_qk_scales: bool
    backward_match_forward_operands: bool

    @property
    def q_width(self) -> int:
        return self.q_heads * self.head_dim

    @property
    def kv_width(self) -> int:
        return self.kv_heads * self.head_dim


_D64_PROFILE = _ExactShapeProfile(
    name="llama3.2-1b-d64-represented-fp8pv",
    contract_name="d64",
    model_preset="llama3.2-1b",
    hidden=2048,
    q_heads=32,
    kv_heads=8,
    head_dim=64,
    layers=16,
    tied_embeddings=True,
    expected_unique_parameters=1_235_814_400,
    rope_scaling_factor=32.0,
    qkv_projection_format="e4m3",
    output_projection_format="e4m3",
    projection_dgrad="bf16",
    per_block_qk_scales=True,
    backward_match_forward_operands=True,
)

_D128_PROFILE = _ExactShapeProfile(
    name="llama3.1-8b-d128-e4m3-qkv",
    contract_name="d128",
    model_preset="llama3.1-8b",
    hidden=4096,
    q_heads=32,
    kv_heads=8,
    head_dim=128,
    layers=32,
    tied_embeddings=False,
    expected_unique_parameters=8_030_261_248,
    rope_scaling_factor=8.0,
    qkv_projection_format=_EXACT_E4M3_PROJECTIONS,
    output_projection_format=_EXACT_E4M3_PROJECTIONS,
    projection_dgrad="nvfp4",
    per_block_qk_scales=True,
    backward_match_forward_operands=False,
)

_D128_NVFP4_PROFILE = replace(
    _D128_PROFILE,
    name="llama3.1-8b-d128-nvfp4-qkv-o",
    qkv_projection_format=_EXACT_NVFP4_PROJECTIONS,
    output_projection_format=_EXACT_NVFP4_PROJECTIONS,
)


def _d128_profile_for_learned_projection(
    learned_projection_format: str,
    represented_qk_backward: bool = False,
) -> _ExactShapeProfile:
    if type(represented_qk_backward) is not bool:
        raise TypeError("represented_qk_backward must be exactly bool")
    if learned_projection_format == _EXACT_E4M3_PROJECTIONS:
        if represented_qk_backward:
            raise ValueError(
                "represented D128 Q/K backward requires NVFP4 learned " "projections"
            )
        return _D128_PROFILE
    if learned_projection_format == _EXACT_NVFP4_PROJECTIONS:
        if represented_qk_backward:
            return replace(
                _D128_NVFP4_PROFILE,
                name="llama3.1-8b-d128-nvfp4-qkv-o-represented-qk-backward",
                backward_match_forward_operands=True,
            )
        return _D128_NVFP4_PROFILE
    raise ValueError("fa4.exact_learned_projection_format must be 'e4m3' or 'nvfp4'")


def _packed_qkv_shape(
    profile: _ExactShapeProfile,
) -> tuple[int, int]:
    return (profile.q_width + 2 * profile.kv_width, profile.hidden)


def _is_d64_profile(profile: _ExactShapeProfile) -> bool:
    return profile.contract_name == "d64" and profile.head_dim == 64


def _is_d128_profile(profile: _ExactShapeProfile) -> bool:
    return profile.contract_name == "d128" and profile.head_dim == 128


def _uses_rolling_d128_weight_pack(
    profile: _ExactShapeProfile,
    qkv_projection_format: str,
) -> bool:
    """Keep the dual-NVFP4 packer off the dense-E4M3 production route."""
    return bool(_is_d128_profile(profile) and qkv_projection_format == "nvfp4")


def _require_d64_packed_profile(profile: _ExactShapeProfile) -> None:
    if not _is_d64_profile(profile):
        raise ValueError("packed exact QKV parameters are D64-only")


def _pack_qkv_weights(
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    profile: _ExactShapeProfile,
) -> torch.Tensor:
    """Pack canonical TorchTitan Q/K/V rows without changing row semantics."""
    _require_d64_packed_profile(profile)
    expected = (
        (profile.q_width, profile.hidden),
        (profile.kv_width, profile.hidden),
        (profile.kv_width, profile.hidden),
    )
    weights = (q_weight, k_weight, v_weight)
    for name, weight, shape in zip(("Q", "K", "V"), weights, expected):
        if tuple(weight.shape) != shape:
            raise ValueError(
                f"exact D64 {name} weight must have shape {shape}, got "
                f"{tuple(weight.shape)}"
            )
    if len({weight.dtype for weight in weights}) != 1:
        raise TypeError("exact D64 Q/K/V weights must share one dtype")
    if len({weight.device for weight in weights}) != 1:
        raise RuntimeError("exact D64 Q/K/V weights must share one device")
    return torch.cat(weights, dim=0).contiguous()


def _unpack_qkv_weight(
    packed_qkv: torch.Tensor,
    profile: _ExactShapeProfile,
    *,
    clone: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expose canonical Q/K/V row slices from the one optimizer-owned leaf."""
    _require_d64_packed_profile(profile)
    expected = _packed_qkv_shape(profile)
    if tuple(packed_qkv.shape) != expected:
        raise ValueError(
            "exact D64 packed QKV weight must have shape "
            f"{expected}, got {tuple(packed_qkv.shape)}"
        )
    values = torch.split(
        packed_qkv,
        (profile.q_width, profile.kv_width, profile.kv_width),
        dim=0,
    )
    if clone:
        return tuple(value.clone() for value in values)  # type: ignore[return-value]
    return values  # type: ignore[return-value]


def export_exact_fa4_split_qkv_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    """Export native D64 packed checkpoints under canonical BF16 Q/K/V keys.

    Native low-precision save/resume deliberately keeps ``packed_qkv`` as the
    model and optimizer FQN.  This explicit, model-only export is for numerical
    comparisons or initializing an ordinary TorchTitan BF16 model; it clones
    the three row ranges so the exported tensors own independent storage.
    """
    result: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in state_dict.items():
        if key == "packed_qkv":
            prefix = ""
        elif key.endswith(".packed_qkv"):
            prefix = key[: -len("packed_qkv")]
        else:
            result[key] = value
            continue
        split_keys = tuple(f"{prefix}{name}.weight" for name in ("wq", "wk", "wv"))
        if any(split_key in state_dict for split_key in split_keys):
            raise KeyError(
                f"checkpoint contains both packed and split QKV at {prefix!r}"
            )
        q_weight, k_weight, v_weight = _unpack_qkv_weight(
            value,
            _D64_PROFILE,
            clone=True,
        )
        for split_key, split_weight in zip(
            split_keys,
            (q_weight, k_weight, v_weight),
            strict=True,
        ):
            result[split_key] = split_weight
    metadata = getattr(state_dict, "_metadata", None)
    if metadata is not None:
        result._metadata = metadata.copy()  # type: ignore[attr-defined]
    return result


def _validated_local_batch_size(
    profile: _ExactShapeProfile,
    value: int,
    *,
    label: str,
    allow_current_d128: bool = False,
) -> int:
    """Return a local batch that preserves an authenticated batch contract."""
    local_batch_size = int(value)
    supported = (
        _D64_AUTHENTICATED_BATCHES
        if _is_d64_profile(profile)
        else _D128_AUTHENTICATED_BATCHES if allow_current_d128 else (1,)
    )
    if local_batch_size not in supported:
        supported_text = ", ".join(str(batch) for batch in supported)
        raise ValueError(
            f"{label} supports local batch {{{supported_text}}} for "
            f"{profile.contract_name}; got {local_batch_size}"
        )
    return local_batch_size


def _exact_backward_policy(
    profile: _ExactShapeProfile,
    *,
    current_d128_route: bool = False,
    native_d64_route: bool = False,
) -> dict[str, Any]:
    if profile.head_dim == 64:
        if native_d64_route:
            return {
                "backward_exp2_degree": 0,
                "backward_exp2_period": 0,
                "backward_fp8_ds_lift": None,
                "backward_reuse_quantized_p": False,
            }
        return {
            "backward_exp2_degree": 1,
            "backward_exp2_period": 2,
            "backward_fp8_ds_lift": 16,
            "backward_reuse_quantized_p": False,
        }
    if profile.head_dim == 128:
        if current_d128_route:
            return {
                "backward_exp2_degree": 0,
                "backward_exp2_period": 0,
                "backward_fp8_ds_lift": None,
                "backward_reuse_quantized_p": False,
            }
        return {
            "backward_exp2_degree": 1,
            "backward_exp2_period": 0,
            "backward_fp8_ds_lift": 256,
            "backward_reuse_quantized_p": True,
        }
    raise ValueError(f"no exact backward policy for D{profile.head_dim}")


def _rope_value(config: Any, name: str) -> Any:
    if isinstance(config, dict):
        return config.get(name)
    return getattr(config, name, None)


def _validate_rope_scaling(
    scaling: Any,
    profile: _ExactShapeProfile,
    *,
    label: str,
) -> None:
    if scaling is None:
        raise ValueError(f"{label} must explicitly configure exact RoPE scaling")
    expected = {
        "scaling_factor": profile.rope_scaling_factor,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
    }
    for name, expected_value in expected.items():
        actual = _rope_value(scaling, name)
        if actual != expected_value:
            raise ValueError(
                f"{label}.{name}={actual!r} does not match {expected_value!r}"
            )


def _profile_from_job_flavor(job_config: JobConfig) -> _ExactShapeProfile:
    flavor = str(job_config.model.flavor)
    learned_projection_format = str(
        getattr(
            job_config.fa4,
            "exact_learned_projection_format",
            _EXACT_E4M3_PROJECTIONS,
        )
    )
    represented_qk_backward = getattr(
        job_config.fa4,
        "exact_d128_represented_qk_backward",
        False,
    )
    if type(represented_qk_backward) is not bool:
        raise TypeError("fa4.exact_d128_represented_qk_backward must be exactly bool")
    e5m2_dout_backward = getattr(
        job_config.fa4,
        "exact_d128_e5m2_dout_backward",
        False,
    )
    if type(e5m2_dout_backward) is not bool:
        raise TypeError("fa4.exact_d128_e5m2_dout_backward must be exactly bool")
    if flavor == "1B" or flavor.startswith("1B_override_"):
        if represented_qk_backward:
            raise ValueError("fa4.exact_d128_represented_qk_backward is D128-only")
        if e5m2_dout_backward:
            raise ValueError("fa4.exact_d128_e5m2_dout_backward is D128-only")
        if learned_projection_format != _EXACT_E4M3_PROJECTIONS:
            raise ValueError(
                "D64 exact FA4 retains paired E4M3 learned QKV/O projections"
            )
        return _D64_PROFILE
    if (
        flavor == "8B"
        or flavor == "8B_llama3_blog"
        or flavor.startswith("8B_llama3_blog_override_")
    ):
        return _d128_profile_for_learned_projection(
            learned_projection_format,
            represented_qk_backward,
        )
    raise ValueError(
        "exact FA4 supports model.flavor='1B', '8B', or "
        "'8B_llama3_blog' only; "
        f"got {flavor!r}"
    )


def _validate_job_rope_contract(job_config: JobConfig) -> _ExactShapeProfile:
    profile = _profile_from_job_flavor(job_config)
    _validate_rope_scaling(
        job_config.model.rope_scaling_args,
        profile,
        label="model.rope_scaling_args",
    )
    configured_theta = job_config.model.rope_theta
    if configured_theta not in (None, 500_000, 500_000.0):
        raise ValueError("exact FA4 requires model.rope_theta=500000")
    configured_max_sequence = job_config.model.max_seq_len
    if configured_max_sequence not in (None, _EXACT_SEQUENCE):
        raise ValueError("exact FA4 requires model.max_seq_len=4096")
    return profile


def _validate_model_rope_contract(
    model: nn.Module,
    profile: _ExactShapeProfile,
) -> None:
    model_args = getattr(model, "model_args", None)
    if model_args is None:
        raise TypeError("exact FA4 requires root model.model_args")
    if getattr(model_args, "rope_theta", None) != 500_000:
        raise ValueError("exact FA4 resolved model rope_theta must be 500000")
    if getattr(model_args, "max_seq_len", None) != _EXACT_SEQUENCE:
        raise ValueError("exact FA4 resolved model max_seq_len must be 4096")
    _validate_rope_scaling(
        getattr(model_args, "rope_scaling_args", None),
        profile,
        label="model.model_args.rope_scaling_args",
    )


def _linear_shape(linear: nn.Module) -> tuple[int, int, bool]:
    weight = getattr(linear, "weight", None)
    if not isinstance(weight, nn.Parameter) or weight.ndim != 2:
        raise TypeError("exact FA4 requires ordinary two-dimensional Linear weights")
    return (
        int(weight.shape[0]),
        int(weight.shape[1]),
        getattr(linear, "bias", None) is not None,
    )


def _select_shape_profile(attention: nn.Module) -> _ExactShapeProfile:
    required_attributes = (
        "wq",
        "wk",
        "wv",
        "wo",
        "n_heads",
        "n_kv_heads",
        "head_dim",
    )
    missing = [name for name in required_attributes if not hasattr(attention, name)]
    if missing:
        raise TypeError(f"attention module is missing exact FA4 fields: {missing}")

    q_shape = _linear_shape(attention.wq)
    k_shape = _linear_shape(attention.wk)
    v_shape = _linear_shape(attention.wv)
    o_shape = _linear_shape(attention.wo)
    if any(shape[2] for shape in (q_shape, k_shape, v_shape, o_shape)):
        raise ValueError("exact FA4 projection linears must not have biases")
    observed = (
        q_shape[1],
        int(attention.n_heads),
        int(attention.n_kv_heads),
        int(attention.head_dim),
        q_shape[0],
        k_shape[0],
        v_shape[0],
        o_shape[0],
        o_shape[1],
    )
    for profile in (_D64_PROFILE, _D128_PROFILE):
        expected = (
            profile.hidden,
            profile.q_heads,
            profile.kv_heads,
            profile.head_dim,
            profile.q_width,
            profile.kv_width,
            profile.kv_width,
            profile.hidden,
            profile.q_width,
        )
        if observed == expected:
            return profile
    raise ValueError(
        "exact FA4 supports only the audited Llama 1.2B D64 and Llama 8B "
        f"D128 attention shapes; observed {observed}"
    )


def _bf16_fa4_native_gqa_forward(
    attention: nn.Module,
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    attention_masks: Any | None,
) -> torch.Tensor:
    """TorchTitan Llama attention without its eager KV-head expansion."""
    if attention_masks is not None:
        raise ValueError(
            "BF16 FA4 native GQA supports causal attention_masks=None only"
        )
    if not getattr(attention, "_bf16_fa4_native_gqa", False):
        raise RuntimeError("BF16 FA4 native GQA forward is not authenticated")
    batch, sequence, hidden = x.shape
    expected_hidden = int(attention.n_heads) * int(attention.head_dim)
    if hidden != expected_hidden:
        raise ValueError(
            f"BF16 FA4 expected hidden width {expected_hidden}, got {hidden}"
        )

    q = attention.wq(x).view(
        batch,
        sequence,
        int(attention.n_heads),
        int(attention.head_dim),
    )
    k = attention.wk(x).view(
        batch,
        sequence,
        int(attention.n_kv_heads),
        int(attention.head_dim),
    )
    v = attention.wv(x).view(
        batch,
        sequence,
        int(attention.n_kv_heads),
        int(attention.head_dim),
    )
    q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

    # The authenticated CuTe FA4 forward and backward both implement native
    # grouped-query attention.  Pass Q32/KV8 directly; TorchTitan's stock
    # forward repeats K/V to 32 heads before calling inner_attention.
    output = attention.inner_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
    )
    expected_output = (
        batch,
        int(attention.n_heads),
        sequence,
        int(attention.head_dim),
    )
    if tuple(output.shape) != expected_output:
        raise RuntimeError(
            "BF16 FA4 native GQA returned an unexpected shape: "
            f"{tuple(output.shape)} != {expected_output}"
        )
    output = (
        output.transpose(1, 2)
        .contiguous()
        .view(
            batch,
            sequence,
            expected_hidden,
        )
    )
    return attention.wo(output)


def _install_bf16_native_gqa_forward(
    attention: nn.Module,
    profile: _ExactShapeProfile,
) -> None:
    if getattr(attention, "_bf16_fa4_native_gqa", False):
        raise RuntimeError("BF16 FA4 native GQA forward is already installed")
    if bool(getattr(attention, "use_flex_attn", False)):
        raise ValueError("BF16 FA4 native GQA is incompatible with flex attention")
    if int(attention.n_heads) != profile.q_heads:
        raise RuntimeError("BF16 FA4 query-head topology changed during conversion")
    if int(attention.n_kv_heads) != profile.kv_heads:
        raise RuntimeError("BF16 FA4 KV-head topology changed during conversion")
    if int(attention.head_dim) != profile.head_dim:
        raise RuntimeError("BF16 FA4 head dimension changed during conversion")
    if int(attention.n_rep) != profile.q_heads // profile.kv_heads:
        raise RuntimeError("BF16 FA4 GQA replication factor is inconsistent")
    if not isinstance(getattr(attention, "inner_attention", None), nn.Module):
        raise TypeError("BF16 FA4 requires an inner_attention module")
    attention._bf16_fa4_native_gqa = True
    attention.forward = MethodType(_bf16_fa4_native_gqa_forward, attention)


def _root_embedding_weights(
    model: nn.Module,
    profile: _ExactShapeProfile,
) -> tuple[nn.Parameter, nn.Parameter]:
    tok_embeddings = getattr(model, "tok_embeddings", None)
    output = getattr(model, "output", None)
    embedding_weight = getattr(tok_embeddings, "weight", None)
    output_weight = getattr(output, "weight", None)
    if not isinstance(embedding_weight, nn.Parameter):
        raise TypeError("exact FA4 requires tok_embeddings.weight Parameter")
    if not isinstance(output_weight, nn.Parameter):
        raise TypeError("exact FA4 requires output.weight Parameter")
    expected_shape = (128_256, profile.hidden)
    if tuple(embedding_weight.shape) != expected_shape:
        raise ValueError(
            "exact FA4 token embedding shape mismatch: "
            f"{tuple(embedding_weight.shape)} != {expected_shape}"
        )
    if tuple(output_weight.shape) != expected_shape:
        raise ValueError(
            "exact FA4 output shape mismatch: "
            f"{tuple(output_weight.shape)} != {expected_shape}"
        )
    return embedding_weight, output_weight


def _restore_root_embedding_alias(
    model: nn.Module,
    profile: _ExactShapeProfile,
) -> None:
    embedding_weight, output_weight = _root_embedding_weights(model, profile)
    if profile.tied_embeddings:
        if output_weight is not embedding_weight:
            model.output.weight = embedding_weight
    elif output_weight is embedding_weight:
        raise RuntimeError("the exact D128/8B contract requires untied embeddings")


def _validate_root_parameter_contract(
    model: nn.Module,
    profile: _ExactShapeProfile,
) -> None:
    embedding_weight, output_weight = _root_embedding_weights(model, profile)
    tied = output_weight is embedding_weight
    if tied != profile.tied_embeddings:
        raise RuntimeError(
            f"exact {profile.contract_name} embedding alias mismatch: "
            f"{tied} != {profile.tied_embeddings}"
        )
    unique_parameters = sum(parameter.numel() for parameter in model.parameters())
    if unique_parameters != profile.expected_unique_parameters:
        raise RuntimeError(
            f"exact {profile.contract_name} unique parameter count mismatch: "
            f"{unique_parameters} != {profile.expected_unique_parameters}"
        )


def _contract_log_line(
    profile: _ExactShapeProfile,
    pv_format: str = _EXACT_FP8_PV,
    e5m2_dout_backward: bool = False,
    learned_projection_format: str = _EXACT_NVFP4_PROJECTIONS,
) -> str:
    if type(e5m2_dout_backward) is not bool:
        raise TypeError("e5m2_dout_backward must be exactly bool")
    tied = str(profile.tied_embeddings).lower()
    try:
        route = _EXACT_ROUTE_BY_FORMATS[(learned_projection_format, pv_format)]
    except KeyError as error:
        raise ValueError(
            "unsupported exact learned-projection/PV format pair: "
            f"{learned_projection_format!r}/{pv_format!r}"
        ) from error
    backward_format = "native_tk_e4m3_qkv_dout"
    if _is_d128_profile(profile) and profile.backward_match_forward_operands:
        route += "_represented_qk_backward"
        backward_format = "represented_nvfp4_qk_e4m3_v_dout"
    if e5m2_dout_backward:
        if not _is_d128_profile(profile):
            raise ValueError("E5M2-dO backward contract logging is D128-only")
        if profile.backward_match_forward_operands:
            raise ValueError(
                "E5M2-dO backward contract logging requires ordinary retained "
                "E4M3 Q/K/V"
            )
        route += "_e5m2_dout_backward"
        backward_format = "nvfp4_score_e4m3_qkv_e5m2_dout"
    return (
        f"[EXACT FA4 CONTRACT] profile={profile.contract_name} "
        f"tied_embeddings={tied} "
        f"unique_parameters={profile.expected_unique_parameters} "
        f"route={route} "
        f"learned_projection_format={learned_projection_format} "
        "qk_format=nvfp4_e4m3_block16 "
        f"pv_format={pv_format} "
        f"backward_format={backward_format}"
    )


def _bf16_contract_log_line(profile: _ExactShapeProfile) -> str:
    tied = str(profile.tied_embeddings).lower()
    return (
        f"[BF16 FA4 CONTRACT] profile={profile.contract_name} "
        f"tied_embeddings={tied} "
        f"unique_parameters={profile.expected_unique_parameters} "
        f"route={_BF16_ROUTE} "
        "learned_projection_format=bf16 qk_format=bf16 pv_format=bf16 "
        "backward_format=bf16_fa4"
    )


def _install_root_parameter_contract(
    model: nn.Module,
    profile: _ExactShapeProfile,
    runtime_context: "_ExactRuntimeContext",
) -> None:
    if getattr(model, "_exact_fa4_root_contract", None) is not None:
        raise RuntimeError("exact FA4 root parameter contract is already installed")

    _restore_root_embedding_alias(model, profile)
    _validate_root_parameter_contract(model, profile)
    original_apply = model._apply
    original_init_weights = getattr(model, "init_weights", None)
    if not callable(original_init_weights):
        raise TypeError("exact FA4 requires a callable model.init_weights")

    def exact_apply(
        root: nn.Module,
        fn: Any,
        recurse: bool = True,
    ) -> nn.Module:
        result = original_apply(fn, recurse=recurse)
        if result is not root:
            raise RuntimeError("model._apply unexpectedly replaced the exact root")
        # Module._apply/to_empty does not preserve Parameters shared by sibling
        # modules. Restore the embedding/output alias before DDP's lazy first
        # forward and before optimizer construction.
        _restore_root_embedding_alias(root, profile)
        _validate_root_parameter_contract(root, profile)
        return result

    def exact_init_weights(root: nn.Module, *args: Any, **kwargs: Any) -> Any:
        _restore_root_embedding_alias(root, profile)
        result = original_init_weights(*args, **kwargs)
        _restore_root_embedding_alias(root, profile)
        if profile.tied_embeddings:
            # The verified standalone D64 anchor initializes its shared token
            # embedding/output matrix from N(0, 0.02). TorchTitan normally
            # initializes the untied output separately, so make this explicit.
            with torch.no_grad():
                nn.init.normal_(
                    root.tok_embeddings.weight,
                    mean=0.0,
                    std=0.02,
                )
        _validate_root_parameter_contract(root, profile)
        freqs_cis = getattr(root, "freqs_cis", None)
        if not isinstance(freqs_cis, torch.Tensor):
            raise TypeError("exact FA4 requires the root complex RoPE buffer")
        runtime_context.eager_initialize(freqs_cis)
        settings = getattr(runtime_context, "settings", None)
        pv_format = getattr(settings, "pv_format", _EXACT_FP8_PV)
        e5m2_dout_backward = getattr(
            settings,
            "d128_e5m2_dout_backward",
            False,
        )
        learned_projection_format = getattr(
            settings,
            "learned_projection_format",
            _EXACT_NVFP4_PROJECTIONS,
        )
        logger.info(
            _contract_log_line(
                profile,
                pv_format,
                e5m2_dout_backward,
                learned_projection_format,
            )
        )
        return result

    model._apply = MethodType(exact_apply, model)
    model.init_weights = MethodType(exact_init_weights, model)
    model._exact_fa4_root_contract = profile.contract_name


def _install_bf16_root_parameter_contract(
    model: nn.Module,
    profile: _ExactShapeProfile,
) -> None:
    """Install the exact root topology contract without a low-precision runtime."""
    if getattr(model, "_bf16_fa4_root_contract", None) is not None:
        raise RuntimeError("BF16 FA4 root parameter contract is already installed")
    if getattr(model, "_exact_fa4_root_contract", None) is not None:
        raise RuntimeError(
            "BF16 FA4 topology cannot be installed over exact low-precision FA4"
        )

    _restore_root_embedding_alias(model, profile)
    _validate_root_parameter_contract(model, profile)
    original_apply = model._apply
    original_init_weights = getattr(model, "init_weights", None)
    if not callable(original_init_weights):
        raise TypeError("BF16 FA4 requires a callable model.init_weights")

    def bf16_apply(
        root: nn.Module,
        fn: Any,
        recurse: bool = True,
    ) -> nn.Module:
        result = original_apply(fn, recurse=recurse)
        if result is not root:
            raise RuntimeError("model._apply unexpectedly replaced the BF16 root")
        _restore_root_embedding_alias(root, profile)
        _validate_root_parameter_contract(root, profile)
        return result

    def bf16_init_weights(root: nn.Module, *args: Any, **kwargs: Any) -> Any:
        _restore_root_embedding_alias(root, profile)
        result = original_init_weights(*args, **kwargs)
        _restore_root_embedding_alias(root, profile)
        if profile.tied_embeddings:
            # Match the authenticated D64 low-precision route exactly rather
            # than inheriting TorchTitan's normally untied output init.
            with torch.no_grad():
                nn.init.normal_(
                    root.tok_embeddings.weight,
                    mean=0.0,
                    std=0.02,
                )
        _validate_root_parameter_contract(root, profile)
        logger.info(_bf16_contract_log_line(profile))
        return result

    model._apply = MethodType(bf16_apply, model)
    model.init_weights = MethodType(bf16_init_weights, model)
    model._bf16_fa4_root_contract = profile.contract_name


def _as_bf16_compute_tensor(
    tensor: torch.Tensor,
    *,
    allow_fp32_master_shadows: bool,
    label: str,
) -> torch.Tensor:
    if tensor.dtype == torch.bfloat16:
        return tensor
    if tensor.dtype == torch.float32 and allow_fp32_master_shadows:
        # This cast is intentionally differentiable. CastBackward converts the
        # BF16 gradient returned by the native custom autograd function into a
        # FP32 gradient on the optimizer-owned leaf parameter.
        return tensor.to(dtype=torch.bfloat16)
    raise TypeError(
        f"{label} must be CUDA BF16"
        + (
            " or FP32 with fa4.exact_allow_fp32_master_shadows=true"
            if not allow_fp32_master_shadows
            else ""
        )
        + f", got {tensor.dtype}"
    )


def _adapter_weights(
    adapter: "ExactLowpFA4Attention",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not _is_d128_profile(adapter._profile):
        raise ValueError("split attention weights are retained only for D128")
    return (
        adapter.wq.weight,
        adapter.wk.weight,
        adapter.wv.weight,
        adapter.wo.weight,
    )


def _weight_versions(
    weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[int, int, int, int]:
    return tuple(int(weight._version) for weight in weights)


class _RollingD128WeightPackSchedule:
    """Exact source-owned dual-weight controllers with model-level ordering."""

    def __init__(
        self,
        exact_module: Any,
        adapters: list["ExactLowpFA4Attention"],
    ) -> None:
        if not adapters:
            raise ValueError("rolling D128 schedule requires attention layers")
        self.exact_module = exact_module
        self.adapters = adapters
        device = adapters[0].wq.weight.device
        if device.type != "cuda":
            raise RuntimeError("rolling D128 schedule requires CUDA weights")
        for adapter in adapters:
            if not _is_d128_profile(adapter._profile):
                raise ValueError("rolling weight packing is D128-only")
            weights = _adapter_weights(adapter)
            if any(weight.device != device for weight in weights):
                raise RuntimeError("rolling D128 attention weights span devices")
            if any(weight.dtype != torch.bfloat16 for weight in weights):
                raise TypeError(
                    "rolling D128 weight packing requires BF16 parameter storage"
                )
            if adapter._forward_workspace is None:
                raise RuntimeError("rolling D128 layer has no forward workspace")

        self.device = device
        self.consumer_stream = torch.cuda.current_stream(device)
        self.consumer_stream_id = int(self.consumer_stream.cuda_stream)
        self.producer_stream = torch.cuda.Stream(device=device)
        if int(self.producer_stream.cuda_stream) == self.consumer_stream_id:
            raise RuntimeError("rolling D128 producer stream is not private")
        self.weights_ready_event = torch.cuda.Event(
            enable_timing=False,
            blocking=False,
            interprocess=False,
        )
        controller_type = getattr(
            exact_module,
            "_DualWeightPackLayerController",
            None,
        )
        if controller_type is None:
            raise RuntimeError(
                "authenticated D128 source has no dual-weight controller"
            )
        self.controllers = [
            controller_type(
                layer_index=layer_index,
                workspace=adapter._forward_workspace,
                weights=_adapter_weights(adapter),
                producer_stream=self.producer_stream,
                consumer_stream=self.consumer_stream,
            )
            for layer_index, adapter in enumerate(adapters)
        ]
        self.generation = -1
        self._active_generation: int | None = None
        self._next_layer_to_enqueue = 0
        self._active_weight_versions: list[tuple[int, int, int, int]] = []

    def authenticate(self) -> None:
        self._require_consumer_stream()
        for adapter, controller in zip(
            self.adapters,
            self.controllers,
            strict=True,
        ):
            controller.authenticate(_adapter_weights(adapter))
        # Authentication creates checked reference publications. Drain them
        # during model initialization, before the first measured train step.
        self.consumer_stream.synchronize()
        if not all(
            adapter._forward_workspace.weight_prep_authenticated
            for adapter in self.adapters
        ):
            raise RuntimeError("rolling D128 authentication did not publish")

    def begin_forward(self) -> None:
        consumer_stream = self._require_consumer_stream()
        if self._active_generation is not None:
            raise RuntimeError("rolling D128 forward generations overlapped")
        generation = self.generation + 1
        for controller in self.controllers:
            controller.require_can_begin(generation)
        weight_versions = [
            _weight_versions(_adapter_weights(adapter)) for adapter in self.adapters
        ]
        if not all(
            adapter._forward_workspace.weight_prep_authenticated
            for adapter in self.adapters
        ):
            raise RuntimeError("rolling D128 schedule lost authentication")
        self.weights_ready_event.record(consumer_stream)
        self.producer_stream.wait_event(self.weights_ready_event)
        for controller, versions in zip(
            self.controllers,
            weight_versions,
            strict=True,
        ):
            controller.begin(generation, versions)
        self.generation = generation
        self._active_generation = generation
        self._next_layer_to_enqueue = 0
        self._active_weight_versions = weight_versions
        self._enqueue_layer(0)

    def complete_layer(
        self,
        layer_index: int,
        *,
        requires_backward: bool,
    ) -> None:
        generation = self._active_generation
        if generation is None:
            raise RuntimeError("rolling D128 layer ran outside a model forward")
        if type(requires_backward) is not bool:
            raise TypeError("requires_backward must be exactly bool")
        controller = self.controllers[layer_index]
        if not controller.state.qkv_consumed or not controller.state.output_consumed:
            raise RuntimeError(
                f"rolling D128 layer {layer_index} ended before operand " "consumption"
            )
        if not requires_backward:
            controller.release_without_backward(
                generation=generation,
                weight_versions=self._active_weight_versions[layer_index],
            )
        if layer_index + 1 < len(self.adapters):
            self._enqueue_layer(layer_index + 1)
            return
        if self._next_layer_to_enqueue != len(self.adapters):
            raise RuntimeError("rolling D128 did not publish every layer")
        if any(
            not controller.state.qkv_consumed or not controller.state.output_consumed
            for controller in self.controllers
        ):
            raise RuntimeError("rolling D128 ended before operand consumption")
        self._active_generation = None
        self._next_layer_to_enqueue = 0
        self._active_weight_versions = []

    def _enqueue_layer(self, layer_index: int) -> None:
        generation = self._active_generation
        if generation is None:
            raise RuntimeError("rolling D128 has no active generation")
        if layer_index != self._next_layer_to_enqueue:
            raise RuntimeError(
                "rolling D128 layers must execute once in order: "
                f"{layer_index} != {self._next_layer_to_enqueue}"
            )
        adapter = self.adapters[layer_index]
        self.controllers[layer_index].enqueue(
            _adapter_weights(adapter),
            generation=generation,
            weight_versions=self._active_weight_versions[layer_index],
        )
        self._next_layer_to_enqueue += 1

    def _require_consumer_stream(self) -> Any:
        stream = torch.cuda.current_stream(self.device)
        stream_id = int(stream.cuda_stream)
        if stream_id != self.consumer_stream_id:
            raise RuntimeError(
                "rolling D128 consumer stream changed: "
                f"{stream_id} != {self.consumer_stream_id}"
            )
        for controller in self.controllers:
            controller.require_bound_consumer_stream(stream_id)
        return stream


def _runtime_config_for_local_batch(
    exact_module: Any,
    profile: _ExactShapeProfile,
    *,
    layer_count: int,
    local_batch_size: int,
) -> Any:
    """Build a source-owned config and require explicit batched-runtime ABI."""
    builder = exact_module.config_from_model_preset
    parameters = inspect.signature(builder).parameters
    kwargs: dict[str, int] = {"layers": layer_count}
    if "batch" in parameters:
        kwargs["batch"] = local_batch_size
    elif local_batch_size != _EXACT_ARTIFACT_BATCH:
        raise RuntimeError(
            "authenticated exact runtime source has no batch-aware Config ABI"
        )
    config = builder(profile.model_preset, **kwargs)
    configured_batch = int(getattr(config, "batch", _EXACT_ARTIFACT_BATCH))
    if configured_batch != local_batch_size:
        raise RuntimeError(
            "exact runtime Config batch does not match requested local batch: "
            f"{configured_batch} != {local_batch_size}"
        )
    return config


def _require_forward_workspace_batch(
    exact_module: Any,
    workspace: Any,
    expected_batch: int,
    *,
    expected_owner_count: int | None = None,
    expected_total_bytes: int | None = None,
) -> int:
    """Authenticate real, independent dense storage for a batched workspace."""
    owner_tensors = getattr(
        exact_module,
        "_forward_workspace_owner_tensors",
        None,
    )
    if not callable(owner_tensors):
        raise RuntimeError(
            "authenticated exact runtime cannot enumerate forward workspace " "owners"
        )
    owners = tuple(owner_tensors(workspace))
    if not owners:
        raise RuntimeError("exact forward workspace has no owner tensors")
    if expected_owner_count is not None and len(owners) != expected_owner_count:
        raise RuntimeError(
            "exact forward workspace owner count mismatch: "
            f"{len(owners)} != {expected_owner_count}"
        )
    names: set[str] = set()
    pointers: set[int] = set()
    total_bytes = 0
    for name, tensor in owners:
        if not isinstance(name, str) or not name or name in names:
            raise RuntimeError(
                f"exact forward workspace has invalid/duplicate owner {name!r}"
            )
        names.add(name)
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(f"exact forward workspace owner {name} is not a tensor")
        if tensor.ndim == 0 or int(tensor.shape[0]) != expected_batch:
            raise RuntimeError(
                f"exact forward workspace owner {name} has batch shape "
                f"{tuple(tensor.shape)}, expected leading {expected_batch}"
            )
        if tensor.layout != torch.strided or not tensor.is_contiguous():
            raise RuntimeError(
                f"exact forward workspace owner {name} is not dense contiguous "
                "storage"
            )
        if tensor.numel() == 0 or tensor.stride(0) <= 0:
            raise RuntimeError(
                f"exact forward workspace owner {name} has no independent "
                "per-sample storage"
            )
        logical_bytes = tensor.numel() * tensor.element_size()
        storage_bytes = tensor.untyped_storage().nbytes()
        if storage_bytes < logical_bytes:
            raise RuntimeError(
                f"exact forward workspace owner {name} aliases only "
                f"{storage_bytes} bytes for {logical_bytes} logical bytes"
            )
        pointer = int(tensor.data_ptr())
        if pointer in pointers:
            raise RuntimeError(
                f"exact forward workspace owner {name} reuses another owner "
                "allocation"
            )
        pointers.add(pointer)
        total_bytes += logical_bytes
    allocation_data_ptrs = getattr(workspace, "allocation_data_ptrs", None)
    if allocation_data_ptrs is not None:
        if set(allocation_data_ptrs) != names:
            raise RuntimeError(
                "exact forward workspace allocation pointer record is not "
                "closed over owner tensors"
            )
        for name, tensor in owners:
            if int(allocation_data_ptrs[name]) != int(tensor.data_ptr()):
                raise RuntimeError(
                    f"exact forward workspace owner {name} changed allocation"
                )
    if expected_total_bytes is not None and total_bytes != expected_total_bytes:
        raise RuntimeError(
            "exact forward workspace byte identity mismatch: "
            f"{total_bytes} != {expected_total_bytes}"
        )
    return total_bytes


def _forward_workspace_allocator_proxy(
    adapter: "ExactLowpFA4Attention",
    runtime: Any,
) -> SimpleNamespace:
    """Expose the minimal weight/device contract used by runtime allocation."""
    return SimpleNamespace(
        runtime=runtime,
        weights=SimpleNamespace(
            # Retain the historical Q anchor for compatible source capsules.
            q=adapter._workspace_anchor_weight(),
            # The current allocator places publications beside the output
            # projection and therefore resolves its device through weights.o.
            o=adapter.wo.weight,
        ),
    )


def _apply_exact_artifact_once(
    exact_module: Any,
    x: torch.Tensor,
    attention_norm_weight: torch.Tensor,
    packed_qkv_weight: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    out_weight: torch.Tensor,
    qk_scales: torch.Tensor,
    workspace: Any,
    runtime: Any,
    autograd_abi: str = _CURRENT_AUTOGRAD_ABI,
) -> torch.Tensor:
    """Invoke exactly one artifact whose compiled batch matches ``x``."""
    publication_state = (
        getattr(workspace, "publication_state", None)
        if autograd_abi == _CURRENT_AUTOGRAD_ABI
        else None
    )
    publication_generation = None
    if publication_state is not None:
        requires_backward = bool(
            torch.is_grad_enabled()
            and any(
                tensor.requires_grad
                for tensor in (
                    x,
                    attention_norm_weight,
                    packed_qkv_weight,
                    q_weight,
                    k_weight,
                    v_weight,
                    out_weight,
                    qk_scales,
                )
            )
        )
        publication_generation = publication_state.begin_forward(
            requires_backward=requires_backward
        )
    try:
        if autograd_abi == _CURRENT_AUTOGRAD_ABI:
            output = exact_module._LowpAttentionFunction.apply(
                x,
                attention_norm_weight,
                packed_qkv_weight,
                q_weight,
                k_weight,
                v_weight,
                out_weight,
                qk_scales,
                workspace,
                runtime,
            )
        elif autograd_abi == _LEGACY_AUTOGRAD_ABI:
            output = exact_module._LowpAttentionFunction.apply(
                x,
                q_weight,
                k_weight,
                v_weight,
                out_weight,
                qk_scales,
                workspace,
                runtime,
            )
        else:
            raise RuntimeError(f"unsupported exact autograd ABI {autograd_abi!r}")
        if tuple(output.shape) != tuple(x.shape):
            raise RuntimeError(
                "exact artifact returned an unexpected local-batch shape: "
                f"{tuple(output.shape)} != {tuple(x.shape)}"
            )
    except Exception:
        if publication_generation is not None:
            publication_state.abort_forward(publication_generation)
        raise
    return output


def _classify_exact_autograd_parameters(parameters: tuple[str, ...]) -> str:
    if parameters == _CURRENT_AUTOGRAD_PARAMETERS:
        return _CURRENT_AUTOGRAD_ABI
    if parameters == _LEGACY_AUTOGRAD_PARAMETERS:
        return _LEGACY_AUTOGRAD_ABI
    raise RuntimeError(
        "exact LowpAttention autograd ABI mismatch: unsupported parameters "
        f"{parameters!r}"
    )


def _require_exact_autograd_abi(exact_module: Any) -> str:
    """Authenticate and identify the exact custom-autograd tensor ABI."""
    function = getattr(exact_module, "_LowpAttentionFunction", None)
    forward = getattr(function, "forward", None)
    if not callable(forward):
        raise RuntimeError("exact runtime has no _LowpAttentionFunction.forward")
    return _classify_exact_autograd_parameters(
        tuple(inspect.signature(forward).parameters)
    )


def _read_exact_autograd_abi(source_root: Path) -> str:
    """Read the authenticated runtime's ABI without importing CUDA code."""
    source = source_root / "tk_fa4" / "lowp_fa4_bwd" / "benchmark_llama12b_e2e.py"
    try:
        tree = ast.parse(source.read_text())
    except (OSError, SyntaxError) as error:
        raise RuntimeError(
            f"could not inspect exact runtime autograd ABI at {source}"
        ) from error
    function: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "_LowpAttentionFunction":
            function = next(
                (
                    child
                    for child in node.body
                    if isinstance(
                        child,
                        (ast.FunctionDef, ast.AsyncFunctionDef),
                    )
                    and child.name == "forward"
                ),
                None,
            )
            break
    if function is None:
        raise RuntimeError("exact runtime has no _LowpAttentionFunction.forward")
    if function.args.vararg is not None or function.args.kwarg is not None:
        raise RuntimeError("exact runtime autograd ABI must not use variadic args")
    parameters = tuple(
        argument.arg
        for argument in (
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
        )
    )
    return _classify_exact_autograd_parameters(parameters)


def _uses_current_d64_route(
    profile: _ExactShapeProfile,
    local_batch_size: int,
    autograd_abi: str,
) -> bool:
    return (
        _is_d64_profile(profile)
        and local_batch_size == 16
        and autograd_abi == _CURRENT_AUTOGRAD_ABI
    )


def _uses_current_d128_route(
    profile: _ExactShapeProfile,
    local_batch_size: int,
    autograd_abi: str,
) -> bool:
    return (
        _is_d128_profile(profile)
        and local_batch_size in _D128_CURRENT_BATCHES
        and autograd_abi == _CURRENT_AUTOGRAD_ABI
    )


def _require_compatible_autograd_route(
    profile: _ExactShapeProfile,
    local_batch_size: int,
    pv_format: str,
    autograd_abi: str,
) -> bool:
    """Return whether this uses the current autograd ABI, else fail closed."""
    current_d64 = _uses_current_d64_route(
        profile,
        local_batch_size,
        autograd_abi,
    )
    current_d128 = _uses_current_d128_route(
        profile,
        local_batch_size,
        autograd_abi,
    )
    if current_d64 or current_d128:
        return True
    legacy_shape = (_is_d64_profile(profile) and local_batch_size in (1, 16)) or (
        _is_d128_profile(profile) and local_batch_size == 1
    )
    if (
        autograd_abi == _LEGACY_AUTOGRAD_ABI
        and legacy_shape
        and pv_format == _EXACT_FP8_PV
    ):
        return False
    raise RuntimeError(
        "exact FA4 runtime/model route is not authenticated: "
        f"profile={profile.contract_name} batch={local_batch_size} "
        f"pv_format={pv_format} autograd_abi={autograd_abi}"
    )


def _current_runtime_only_kwargs(
    current_d64_route: bool,
    pv_format: str,
    mx_v_publication: str = _EXACT_RETAINED_SPLIT_V,
    *,
    current_d128_route: bool = False,
    learned_projection_format: str = _EXACT_E4M3_PROJECTIONS,
    native_score_backward: bool = False,
    represented_qk_backward: bool = False,
    e5m2_dout_backward: bool = False,
) -> dict[str, bool]:
    """Return kwargs absent from authenticated legacy runtime constructors."""
    if mx_v_publication not in _EXACT_MX_V_PUBLICATIONS:
        raise ValueError(f"unsupported exact MX V publication {mx_v_publication!r}")
    output_shared_split_v = mx_v_publication == _EXACT_OUTPUT_SHARED_SPLIT_V
    shared_d128_mx_v = mx_v_publication == _EXACT_D128_SHARED_D32XS32_V
    if current_d64_route and current_d128_route:
        raise ValueError("one exact runtime cannot be both current D64 and D128")
    if type(native_score_backward) is not bool:
        raise TypeError("native_score_backward must be exactly bool")
    if type(represented_qk_backward) is not bool:
        raise TypeError("represented_qk_backward must be exactly bool")
    if type(e5m2_dout_backward) is not bool:
        raise TypeError("e5m2_dout_backward must be exactly bool")
    if native_score_backward and not current_d128_route:
        raise ValueError("native-score backward is authenticated only for D128")
    if e5m2_dout_backward and not current_d128_route:
        raise ValueError("E5M2-dO backward is authenticated only for D128")
    if e5m2_dout_backward and not native_score_backward:
        raise ValueError("E5M2-dO backward requires native-score backward")
    if e5m2_dout_backward and represented_qk_backward:
        raise ValueError(
            "E5M2-dO backward is incompatible with represented Q/K backward"
        )
    if learned_projection_format not in _EXACT_LEARNED_PROJECTION_FORMATS:
        raise ValueError("exact learned projection format must be 'e4m3' or 'nvfp4'")
    if not current_d128_route and (
        learned_projection_format != _EXACT_E4M3_PROJECTIONS
    ):
        raise ValueError(
            "NVFP4 learned projections are authenticated only for current D128"
        )
    if output_shared_split_v:
        raise ValueError(
            "output_shared_split is a historical native-NVFP4-QKV concession "
            "and is not accepted by the current paired-projection routes"
        )
    if shared_d128_mx_v:
        raise ValueError(
            "shared_d32xs32_forward_anchors is not authenticated by the "
            "current paired-projection D128/v501 route"
        )
    if current_d128_route:
        if pv_format not in _EXACT_PV_FORMATS:
            raise ValueError(f"unsupported exact PV format {pv_format!r}")
        expected_publication = _EXACT_RETAINED_SPLIT_V
        if mx_v_publication != expected_publication:
            raise ValueError(
                "current D128 route/publication mismatch: "
                f"{mx_v_publication!r} != {expected_publication!r}"
            )
        runtime_kwargs = {
            # D128 owns distinct construction-bound binders for the paired
            # learned E4M3 and NVFP4 QKV/O forward routes. Both publish the
            # same direct-accumulator E4M3 Q/K/V for v501 backward.
            "experimental_native_nvfp4_projection_out": (
                learned_projection_format == _EXACT_NVFP4_PROJECTIONS
            ),
            "experimental_fused_attention_rmsnorm_nvfp4": False,
            "experimental_split_v_backward": False,
            "experimental_output_shared_split_v": False,
            # Forward MXFP4 V and backward V are intentionally split: v501
            # consumes the direct projection-accumulator E4M3 V for both PV
            # formats. The v503/shared-D32xS32 path remains experimental.
            "experimental_d128_mxfp4_v_backward": False,
            "v_mxfp4_scale_2d": False,
            "native_tk_d128_native_score_backward": native_score_backward,
        }
        if e5m2_dout_backward:
            runtime_kwargs["native_tk_d128_v509_e5m2_dout_backward"] = True
        return runtime_kwargs
    if not current_d64_route:
        return {}
    if pv_format not in _EXACT_PV_FORMATS:
        raise ValueError(f"unsupported exact PV format {pv_format!r}")
    runtime_kwargs = {
        "experimental_native_nvfp4_projection_out": False,
        "experimental_fused_attention_rmsnorm_nvfp4": False,
        "experimental_split_v_backward": pv_format == _EXACT_MXFP4_PV,
    }
    return runtime_kwargs


def _expected_current_d64_projection_symbols(
    pv_format: str,
    mx_v_publication: str = _EXACT_RETAINED_SPLIT_V,
) -> tuple[str, str]:
    """Return the exact checked/unchecked symbols for one D64 B16 route."""
    if mx_v_publication not in _EXACT_MX_V_PUBLICATIONS:
        raise ValueError(f"unsupported exact MX V publication {mx_v_publication!r}")
    prefix = (
        "project_qkv_gqa_d64_paired_unified_fp8_e4m3_rope_packed_"
        "interleaved_causal_represented_backward_perblock_qk"
    )
    if pv_format == _EXACT_FP8_PV:
        if mx_v_publication != _EXACT_RETAINED_SPLIT_V:
            raise ValueError("FP8-PV cannot select an MX V publication")
        checked = prefix + "_fp8_forward_out"
    elif pv_format == _EXACT_MXFP4_PV:
        if mx_v_publication != _EXACT_RETAINED_SPLIT_V:
            raise ValueError("production D64 E4M3 QKV requires retained_split MX V")
        checked = prefix + "_split_v_backward_mx_forward_out"
    else:
        raise ValueError(f"unsupported exact PV format {pv_format!r}")
    return checked, checked + "_unchecked"


def _expected_current_d128_projection_symbols(
    pv_format: str,
    learned_projection_format: str = _EXACT_E4M3_PROJECTIONS,
    represented_backward: bool = False,
) -> tuple[str, str]:
    """Return one D128 paired-projection checked/unchecked symbol tuple."""
    if type(represented_backward) is not bool:
        raise TypeError("represented_backward must be exactly bool")
    if represented_backward:
        if learned_projection_format != _EXACT_NVFP4_PROJECTIONS:
            raise ValueError(
                "represented D128 Q/K backward requires NVFP4 learned " "projections"
            )
        if pv_format != _EXACT_FP8_PV:
            raise ValueError(
                "represented D128 Q/K backward is authenticated only for " "FP8-PV"
            )
        return (
            _D128_NVFP4_REPRESENTED_QK_FP8_SYMBOL,
            _D128_NVFP4_REPRESENTED_QK_FP8_SYMBOL + "_unchecked",
        )
    suffix = {
        _EXACT_FP8_PV: "_fp8_forward_out",
        _EXACT_MXFP4_PV: "_mx_forward_out",
    }.get(pv_format)
    if suffix is None:
        raise ValueError(f"unsupported exact PV format {pv_format!r}")
    prefix = {
        _EXACT_E4M3_PROJECTIONS: _D128_E4M3_PROJECTION_SYMBOL_PREFIX,
        _EXACT_NVFP4_PROJECTIONS: _D128_NVFP4_PROJECTION_SYMBOL_PREFIX,
    }.get(learned_projection_format)
    if prefix is None:
        raise ValueError("exact learned projection format must be 'e4m3' or 'nvfp4'")
    checked = prefix + suffix
    return checked, checked + "_unchecked"


def _fuses_attention_rmsnorm(
    profile: _ExactShapeProfile,
    local_batch_size: int,
    autograd_abi: str = _CURRENT_AUTOGRAD_ABI,
) -> bool:
    # The historical fusion is a distinct D64 native-NVFP4 ABI. It is not the
    # current D128 paired-NVFP4 projection route, so eager RMSNorm remains the
    # only authenticated choice for both paired D128 formats.
    _ = (profile, local_batch_size, autograd_abi)
    return False


def _require_selected_forward_topology(
    profile: _ExactShapeProfile,
    local_batch_size: int,
    pv_format: str,
    topology: Mapping[str, Any],
    *,
    current_d64_route: bool | None = None,
    current_d128_route: bool | None = None,
) -> None:
    if int(topology.get("batch", -1)) != local_batch_size:
        raise RuntimeError(
            "exact forward artifact batch does not match requested local "
            f"batch {local_batch_size}"
        )
    actual_pv_format = topology.get("pv_format")
    if actual_pv_format != pv_format:
        raise RuntimeError(
            "exact forward artifact PV format mismatch: "
            f"{actual_pv_format!r} != {pv_format!r}"
        )
    causal_interleaved = bool(topology.get("causal_interleaved_kv", False))
    if pv_format == _EXACT_FP8_PV:
        if int(topology.get("shiftless_fp8_mode", -1)) != 0:
            raise RuntimeError("exact FP8-PV requires shiftless_fp8_mode=0")
        if causal_interleaved:
            raise RuntimeError("exact FP8-PV requires ordinary causal K/V order")
    elif pv_format == _EXACT_MXFP4_PV:
        if current_d64_route is None:
            current_d64_route = _is_d64_profile(profile) and local_batch_size == 16
        if current_d128_route is None:
            current_d128_route = (
                _is_d128_profile(profile) and local_batch_size in _D128_CURRENT_BATCHES
            )
        if not (current_d64_route or current_d128_route):
            raise RuntimeError(
                "exact MXFP4-PV is authenticated only by the current D64 B16 "
                "or D128 B1/B2/B4 runtime"
            )
        if current_d64_route and not causal_interleaved:
            raise RuntimeError(
                "exact D64 MXFP4-PV requires interleaved causal K/V order"
            )
        if current_d128_route and causal_interleaved:
            raise RuntimeError("exact D128 MXFP4-PV requires ordinary causal K/V order")
    else:
        raise RuntimeError(f"unsupported exact PV format {pv_format!r}")


def _require_v509_e5m2_dout_route_receipt(receipt: Any) -> None:
    """Authenticate the fused E5M2 publisher/native-score v509 pairing."""
    if not isinstance(receipt, dict) or set(receipt) != {
        "route",
        "publisher",
        "backward",
    }:
        raise RuntimeError(
            "exact D128 v509 route receipt has a noncanonical top-level ABI"
        )
    if receipt.get("route") != _D128_V509_E5M2_DOUT_ROUTE:
        raise RuntimeError("exact D128 v509 route receipt identity mismatch")
    publisher = receipt.get("publisher")
    backward = receipt.get("backward")
    if not isinstance(publisher, dict) or not isinstance(backward, dict):
        raise RuntimeError(
            "exact D128 v509 route receipt lacks producer/consumer metadata"
        )
    legacy_b1_publisher = publisher.get("source_identity") == (
        _D128_V509_E5M2_DOUT_LEGACY_B1_PUBLISHER_SOURCE
    )
    expected_publisher = {
        "schema": "tkfa4.v509_e5m2_dout_publisher.v1",
        "source_identity": (
            _D128_V509_E5M2_DOUT_LEGACY_B1_PUBLISHER_SOURCE
            if legacy_b1_publisher
            else _D128_V509_E5M2_DOUT_PUBLISHER_SOURCE
        ),
        "experimental": True,
        "production_dispatch_connected": False,
        "dispatch": (
            "fail_closed_B1_S4096_H32_D128_native_score_only"
            if legacy_b1_publisher
            else "B1_B2_B4_S4096_H32_D128_native_score_only"
        ),
        "selected_epilogue": "kernel_v509_native_score_e5m2_dout",
        "payload_dtype": "float8_e5m2",
        "payload_layout": "BSHD_contiguous",
        "encode": "(BF16.float()*4).to(float8_e5m2)",
        "encode_scale": 4.0,
        "logical_decode_scale": 0.25,
        "dstat_physical_abi": "-4*sum(O*raw_E5M2_dO)",
        "lstat_abi": "8-LSE*log2(e)",
        "probability_log2_lift": 8.0,
        "sequence": _EXACT_SEQUENCE,
        "query_heads": 32,
        "head_dim": 128,
        "store_bf16_dout": False,
        "publish_e4m3_dout": False,
        "publish_stats": True,
        "clear_dq": True,
        "raw_output_slots": 8,
        "e5m2_payload_slot": 7,
    }
    expected_publisher["batch" if legacy_b1_publisher else "batch_values"] = (
        1 if legacy_b1_publisher else (1, 2, 4)
    )
    backward_batch = backward.get("batch")
    if type(backward_batch) is not int or backward_batch not in (
        _D128_V509_AUTHENTICATED_BATCHES
    ):
        raise RuntimeError(
            "exact D128 v509 route receipt has an unsupported backward batch"
        )
    expected_backward = {
        "schema": "tkfa4.native_tk_d128_backward.v1",
        "backend": "thunderkittens_sm100a",
        "source_identity": _native_tk_d128_v509_source(backward_batch),
        "experimental": True,
        "production_dispatch_connected": False,
        "dispatch": (
            "fail_closed_B1_S4096_only_no_fallback"
            if backward_batch == 1
            else f"fail_closed_B{backward_batch}_S4096_only_no_fallback"
        ),
        "selected_kernel": (
            "v509::b1_native_nvfp4_score_e4m3_qkv_e5m2_dout_exact_s4096_kernel"
        ),
        "score_qk_dtype": "float4_e2m1fn_x2",
        "score_qk_layout": "BHSD_packed",
        "score_scale_dtype": "float8_e4m3fn",
        "score_scale_layout": (
            "forward_row_K16_pages_Q_B_S128_Hx2_512_K_B_S64_Hkvx2_512"
        ),
        "score_global_scale": "per_head_q_times_k",
        "score_mma": "two_K64_mxf4nvf4_block_scale_scale_vec_4X",
        "gradient_qkv_dtype": "float8_e4m3fn_represented_x4",
        "dout_dtype": "float8_e5m2_represented_x4",
        "dout_encode_scale": 4.0,
        "dout_decode_scale": 0.25,
        "mixed_mma_b_format_mask": 1024,
        "score_internal_beta_divisor": 1.0,
        "ds_internal_beta_divisor": 16.0,
        "lstat_abi": "8-LSE*log2(e)",
        "dstat_abi": "-16*sum(O*dO)",
        "dstat_physical_abi": "-4*sum(O*raw_E5M2_dO)",
        "output_dtype": "bfloat16_additive",
        "batch": backward_batch,
        "sequence": _EXACT_SEQUENCE,
        "query_heads": 32,
        "kv_heads": 8,
        "head_dim": 128,
        "threads": 512,
        "user_shared_storage_bytes": 193536,
        "score_scale_tmem_alias": "dP_dQ_columns_0_15",
        "score_schedule": (
            "wait_dq_tmem_drained_then_native_score_wait_complete_then_dense_dp"
        ),
        "caller_owned_output_api": True,
        "main_requires_precleared_outputs": True,
        "backward_out_clears_dq_dk_dv": True,
        "precleared_dq_out_clears_dk_dv": True,
    }
    if legacy_b1_publisher:
        if backward_batch != 1:
            raise RuntimeError("legacy exact D128 v509 publisher receipt is B1-only")
        expected_backward.pop("precleared_dq_out_clears_dk_dv")
    mismatches = {}
    for side, actual, expected in (
        ("publisher", publisher, expected_publisher),
        ("backward", backward, expected_backward),
    ):
        side_mismatches = {
            field: {"actual": actual.get(field), "expected": value}
            for field, value in expected.items()
            if actual.get(field) != value or type(actual.get(field)) is not type(value)
        }
        missing = set(expected) - set(actual)
        if missing:
            side_mismatches["missing"] = sorted(missing)
        unexpected = set(actual) - set(expected)
        if unexpected:
            side_mismatches["unexpected"] = sorted(unexpected)
        if side_mismatches:
            mismatches[side] = side_mismatches
    if mismatches:
        raise RuntimeError(
            f"exact D128 v509 route receipt metadata mismatch: {mismatches}"
        )


def _require_runtime_route_contract(
    runtime: Any,
    profile: _ExactShapeProfile,
    local_batch_size: int,
    pv_format: str,
    mx_v_publication: str = _EXACT_RETAINED_SPLIT_V,
    native_score_backward: bool = False,
    e5m2_dout_backward: bool = False,
) -> None:
    """Close over forward publication and backward representation semantics."""
    dispatch_builder = getattr(runtime, "forward_dispatch_contract", None)
    backward_builder = getattr(runtime, "backward_contract", None)
    if not callable(dispatch_builder) or not callable(backward_builder):
        raise RuntimeError(
            "exact runtime must expose forward and backward route contracts"
        )
    dispatch = dispatch_builder()
    backward = backward_builder()
    if type(native_score_backward) is not bool:
        raise TypeError("native_score_backward must be exactly bool")
    if type(e5m2_dout_backward) is not bool:
        raise TypeError("e5m2_dout_backward must be exactly bool")
    if native_score_backward and (
        not _is_d128_profile(profile)
        or local_batch_size not in _D128_V509_AUTHENTICATED_BATCHES
    ):
        raise RuntimeError("native-score backward is authenticated only for D128 B1/B4")
    if e5m2_dout_backward and (
        not _is_d128_profile(profile)
        or local_batch_size not in _D128_V509_AUTHENTICATED_BATCHES
    ):
        raise RuntimeError("E5M2-dO backward is authenticated only for D128 B1/B4")
    if e5m2_dout_backward and not native_score_backward:
        raise RuntimeError("E5M2-dO backward requires native-score backward")
    if e5m2_dout_backward and profile.backward_match_forward_operands:
        raise RuntimeError(
            "E5M2-dO backward is incompatible with represented Q/K backward"
        )
    if e5m2_dout_backward and mx_v_publication != _EXACT_RETAINED_SPLIT_V:
        raise RuntimeError("E5M2-dO backward requires retained_split E4M3 V")
    expected_shape = {
        "batch": local_batch_size,
        "sequence": _EXACT_SEQUENCE,
        "q_heads": profile.q_heads,
        "kv_heads": profile.kv_heads,
        "head_dim": profile.head_dim,
    }
    if dispatch.get("shape") != expected_shape:
        raise RuntimeError(
            "exact forward dispatch shape mismatch: "
            f"{dispatch.get('shape')!r} != {expected_shape!r}"
        )
    if dispatch.get("pv_format") != pv_format:
        raise RuntimeError("exact forward dispatch lost its selected PV format")
    projection_dispatch = dispatch.get("qkv_projection", {})
    expected_fused = _fuses_attention_rmsnorm(profile, local_batch_size)
    current_d64_route = _is_d64_profile(profile) and local_batch_size == 16
    expected_qkv_format = profile.qkv_projection_format
    expected_native_nvfp4 = bool(
        _is_d128_profile(profile) and expected_qkv_format == _EXACT_NVFP4_PROJECTIONS
    )
    represented_qk_backward = profile.backward_match_forward_operands
    if type(represented_qk_backward) is not bool:
        raise TypeError(
            "exact profile backward_match_forward_operands must be exactly bool"
        )
    if _is_d128_profile(profile) and represented_qk_backward:
        if not expected_native_nvfp4:
            raise RuntimeError(
                "represented D128 Q/K backward requires native NVFP4 QKV"
            )
        if pv_format != _EXACT_FP8_PV:
            raise RuntimeError("represented D128 Q/K backward requires FP8-PV")
    if projection_dispatch.get("format") != expected_qkv_format:
        raise RuntimeError("exact forward dispatch QKV format mismatch")
    output_projection_dispatch = dispatch.get("output_projection", {})
    expected_output_format = profile.output_projection_format
    if _is_d128_profile(profile) and (expected_qkv_format != expected_output_format):
        raise RuntimeError(
            "exact D128 requires paired QKV/O learned projection formats"
        )
    if output_projection_dispatch.get("forward_format") != expected_output_format:
        raise RuntimeError("exact forward dispatch O format mismatch")
    if (
        bool(projection_dispatch.get("experimental_native_nvfp4_caller_owned"))
        is not expected_native_nvfp4
    ):
        raise RuntimeError("exact forward dispatch native-NVFP4 publication mismatch")
    if (
        bool(projection_dispatch.get("experimental_fused_attention_rmsnorm_nvfp4"))
        != expected_fused
    ):
        raise RuntimeError("exact forward dispatch fused RMSNorm mismatch")
    if _is_d64_profile(profile) and local_batch_size == 16:
        expected_checked, expected_unchecked = _expected_current_d64_projection_symbols(
            pv_format,
            mx_v_publication,
        )
        expected_symbols = {
            "checked_symbol": expected_checked,
            "unchecked_symbol": expected_unchecked,
            "symbol": expected_unchecked,
        }
        for field, expected in expected_symbols.items():
            if projection_dispatch.get(field) != expected:
                raise RuntimeError(
                    f"exact forward dispatch {field} mismatch: "
                    f"{projection_dispatch.get(field)!r} != {expected!r}"
                )
        if projection_dispatch.get("dispatch") != (
            "construction_bound_exact_pybind_symbol"
        ):
            raise RuntimeError("exact forward dispatch is not construction-bound")
        if bool(projection_dispatch.get("shape_bound_at_construction")) is not True:
            raise RuntimeError("exact forward dispatch is not shape-bound")
    elif _is_d128_profile(profile) and local_batch_size in _D128_CURRENT_BATCHES:
        expected_checked, expected_unchecked = (
            _expected_current_d128_projection_symbols(
                pv_format,
                expected_qkv_format,
                represented_qk_backward,
            )
        )
        for field, expected in {
            "checked_symbol": expected_checked,
            "unchecked_symbol": expected_unchecked,
            "symbol": expected_unchecked,
        }.items():
            if projection_dispatch.get(field) != expected:
                raise RuntimeError(
                    f"exact D128 forward dispatch {field} mismatch: "
                    f"{projection_dispatch.get(field)!r} != {expected!r}"
                )
        if projection_dispatch.get("dispatch") != (
            "construction_bound_exact_pybind_symbol"
        ):
            raise RuntimeError("exact D128 forward dispatch is not bound")
        if bool(projection_dispatch.get("shape_bound_at_construction")) is not True:
            raise RuntimeError("exact D128 forward dispatch is not shape-bound")
        expected_projection_path = (
            "caller_owned_represented_qk_fp8_pv_d128"
            if represented_qk_backward
            else (
                "caller_owned_route_selective_d128"
                if expected_native_nvfp4
                else "caller_owned_dense_e4m3_d128"
            )
        )
        if projection_dispatch.get("projection_forward_publication_path") != (
            expected_projection_path
        ):
            raise RuntimeError(
                "exact D128 forward dispatch lost its selected projection "
                "publication"
            )
        for field in (
            "output_shared_split_v_requested",
            "output_shared_split_v_resolved",
        ):
            if projection_dispatch.get(field) is not False:
                raise RuntimeError(f"exact D128 forward dispatch {field} must be false")
        expected_backward_publication_semantics = (
            "represented_nvfp4_qk_per_row_k16_with_" "projection_accumulator_e4m3_v"
            if represented_qk_backward
            else "projection_accumulator_e4m3_qkv_shared_across_pv_routes"
        )
        if projection_dispatch.get("backward_publication_semantics") != (
            expected_backward_publication_semantics
        ):
            raise RuntimeError(
                "exact D128 forward dispatch lost selected backward "
                "publication semantics"
            )
        expected_output_projection = {
            "forward_activation_publication": (
                "functional_rowwise_nvfp4"
                if expected_native_nvfp4
                else "functional_rowwise_e4m3_fp32_decode"
            ),
            "forward_weight_publication": (
                "caller_owned_dual_true_2d_nvfp4"
                if expected_native_nvfp4
                else "functional_channelwise_e4m3_fp32_decode"
            ),
            "forward_kernel": (
                "b300_project_nvfp4" if expected_native_nvfp4 else "b300_project_e4m3"
            ),
            "forward_allocation": (
                "caller_owned_weight_workspace_functional_activation_pack"
                if expected_native_nvfp4
                else "allocating_generic_correctness_canary_nonfinal_speed"
            ),
            "unused_nvfp4_forward_weight_publication": False,
            "backward_input_gradient_format": "nvfp4",
            "backward_input_gradient_kernel": ("b300_project_dout_unified_lowp_nvfp4"),
            "backward_weight_publication": (
                "caller_owned_dual_true_2d_nvfp4"
                if expected_native_nvfp4
                else "functional_true_2d_nvfp4_transpose_prepared_in_forward"
            ),
            "backward_weight_gradient_format": "bf16",
            "backward_weight_gradient_kernel": "torch.mm",
            "e4m3_backward_learned_projection_gemms": False,
            "asymmetric_forward_input_gradient": not expected_native_nvfp4,
        }
        if e5m2_dout_backward:
            expected_output_projection.update(
                backward_input_gradient_kernel=(_D128_V509_E5M2_DOUT_KERNEL),
                backward_attention_dout_format="e5m2",
                backward_attention_dout_source=(_D128_V509_E5M2_DOUT_SOURCE),
                backward_attention_dout_encoding_scale=4.0,
            )
        for name, expected in expected_output_projection.items():
            if output_projection_dispatch.get(name) != expected:
                raise RuntimeError(
                    f"exact D128 forward O projection {name} mismatch: "
                    f"{output_projection_dispatch.get(name)!r} != {expected!r}"
                )
    if backward.get("shape") != expected_shape:
        raise RuntimeError("exact backward contract shape mismatch")
    if (
        bool(
            backward.get("autograd", {}).get(
                "experimental_fused_attention_rmsnorm_nvfp4"
            )
        )
        != expected_fused
    ):
        raise RuntimeError("exact backward contract fused RMSNorm mismatch")

    projection_backward = backward.get("projection", {})
    expected_projection = {
        "qkv_projection_format": expected_qkv_format,
        "output_projection_format": expected_output_format,
        "projection_dgrad": profile.projection_dgrad,
        "projection_weight_scale_2d": True,
        "represented_backward": profile.backward_match_forward_operands,
        "per_block_qk_scales": profile.per_block_qk_scales,
        "native_tk_d128_native_score_backward": native_score_backward,
    }
    if e5m2_dout_backward:
        expected_projection.update(
            native_tk_d128_v509_e5m2_dout_backward=True,
            dout_backward_format="e5m2",
            dout_backward_kernel=_D128_V509_E5M2_DOUT_KERNEL,
        )
    for name, expected in expected_projection.items():
        if projection_backward.get(name) != expected:
            raise RuntimeError(
                f"exact backward projection {name} mismatch: "
                f"{projection_backward.get(name)!r} != {expected!r}"
            )
    output_backward = projection_backward.get(
        "output_projection_forward_backward",
        {},
    )
    if output_backward != output_projection_dispatch:
        raise RuntimeError(
            "exact backward O projection provenance does not match forward"
        )
    if _is_d64_profile(profile):
        if projection_backward.get("qk_backward_source") != (
            "represented_nvfp4_codes_per_row_k16"
        ):
            raise RuntimeError("exact D64 backward lost represented Q/K codes")
        # MX forward V deliberately splits its publication: its backward V is
        # the same direct projection-accumulator E4M3 representation as FP8.
        if projection_backward.get("v_backward_source") != (
            "projection_accumulator_e4m3"
        ):
            raise RuntimeError("exact D64 backward V must use direct E4M3")

        publication = runtime.projection_publication_topology
        expected_split_v = pv_format == _EXACT_MXFP4_PV
        if bool(publication.get("experimental_split_v_backward")) != (expected_split_v):
            raise RuntimeError("exact D64 split-V backward policy mismatch")
        expected_output_shared = mx_v_publication == _EXACT_OUTPUT_SHARED_SPLIT_V
        if (
            bool(publication.get("experimental_output_shared_split_v"))
            != expected_output_shared
        ):
            raise RuntimeError("exact D64 MX V publication topology mismatch")
        if (
            bool(publication.get("experimental_native_nvfp4_projection_out"))
            is not False
        ):
            raise RuntimeError("exact D64 production must not select native NVFP4 QKV")
        backend = backward.get("backend", {})
        if backend.get("backend") != _NATIVE_TK_D64_BACKEND:
            raise RuntimeError("exact D64 native TK backend mismatch")
        extension_metadata = backend.get("extension_metadata", {})
        if extension_metadata.get("source_identity") != _NATIVE_TK_D64_V416_SOURCE:
            raise RuntimeError("exact D64 native TK source identity is not v416")
        runtime_identity = getattr(
            runtime,
            "native_tk_d64_backward_extension_identity",
            {},
        )
        backend_identity = backend.get("extension", {})
        required_identity_fields = {
            "path",
            "sha256",
            "bytes",
            "device",
            "inode",
            "mtime_ns",
        }
        if (
            not isinstance(runtime_identity, dict)
            or set(runtime_identity) != required_identity_fields
            or not isinstance(backend_identity, dict)
            or set(backend_identity) != required_identity_fields
            or backend_identity != runtime_identity
        ):
            raise RuntimeError(
                "exact D64 native TK backend changed its loaded artifact identity"
            )
    elif _is_d128_profile(profile):
        expected_publication = _EXACT_RETAINED_SPLIT_V
        if mx_v_publication != expected_publication:
            raise RuntimeError(
                "exact D128 V publication selector mismatch: "
                f"{mx_v_publication!r} != {expected_publication!r}"
            )
        expected_scale_policy = None
        expected_v_source = "projection_accumulator_e4m3"
        expected_qk_source = (
            "represented_nvfp4_codes_per_row_k16"
            if represented_qk_backward
            else "projection_accumulator_e4m3"
        )
        if projection_backward.get("qk_backward_source") != (expected_qk_source):
            raise RuntimeError("exact D128 backward Q/K publication mismatch")
        if projection_backward.get("v_backward_source") != expected_v_source:
            raise RuntimeError("exact D128 backward V publication mismatch")
        expected_dout_source = (
            _D128_V509_E5M2_DOUT_SOURCE
            if e5m2_dout_backward
            else "projection_accumulator_e4m3"
        )
        if projection_backward.get("dout_backward_source") != (expected_dout_source):
            raise RuntimeError("exact D128 backward dO publication mismatch")
        if projection_backward.get("experimental_d128_mxfp4_v_backward") is not False:
            raise RuntimeError("exact D128 must retain v501 E4M3 V backward")
        if (
            projection_backward.get("d128_mxfp4_v_scale_policy")
            != expected_scale_policy
        ):
            raise RuntimeError("exact D128 native MX scale policy mismatch")

        publication = runtime.projection_publication_topology
        expected_projection_path = (
            "caller_owned_represented_qk_fp8_pv_d128"
            if represented_qk_backward
            else (
                "caller_owned_route_selective_d128"
                if expected_native_nvfp4
                else "caller_owned_dense_e4m3_d128"
            )
        )
        expected_publication_fields = {
            "experimental_d128_mxfp4_v_backward": False,
            "d128_mxfp4_v_scale_policy": expected_scale_policy,
            "qk_backward_source": expected_qk_source,
            "v_backward_source": expected_v_source,
            "experimental_native_nvfp4_projection_out": (expected_native_nvfp4),
            "experimental_output_shared_split_v": False,
            "experimental_output_shared_split_v_requested": False,
            "experimental_output_shared_split_v_resolved": False,
            "projection_forward_publication_path": expected_projection_path,
            "native_tk_d128_native_score_backward": native_score_backward,
        }
        if e5m2_dout_backward:
            expected_publication_fields.update(
                native_tk_d128_v509_e5m2_dout_backward=True,
                dout_backward_format="e5m2",
                dout_backward_source=_D128_V509_E5M2_DOUT_SOURCE,
                dout_backward_kernel=_D128_V509_E5M2_DOUT_KERNEL,
            )
        for field, expected in expected_publication_fields.items():
            actual = publication.get(field)
            if actual != expected or type(actual) is not type(expected):
                raise RuntimeError(
                    f"exact D128 publication {field} mismatch: "
                    f"{actual!r} != {expected!r}"
                )

        if e5m2_dout_backward:
            publication_receipt = publication.get("v509_e5m2_dout_route")
            projection_receipt = projection_backward.get("v509_e5m2_dout_route")
            if projection_receipt != publication_receipt:
                raise RuntimeError(
                    "exact D128 v509 projection/publication route receipts "
                    "do not match"
                )
            _require_v509_e5m2_dout_route_receipt(publication_receipt)
        else:
            forbidden_projection_fields = {
                "native_tk_d128_v509_e5m2_dout_backward",
                "dout_backward_format",
                "dout_backward_kernel",
                "v509_e5m2_dout_route",
            }
            forbidden_publication_fields = {
                "dout_backward_format",
                "dout_backward_source",
                "dout_backward_kernel",
                "v509_e5m2_dout_route",
            }
            default_selector = publication.get("native_tk_d128_v509_e5m2_dout_backward")
            if default_selector is not None and default_selector is not False:
                raise RuntimeError(
                    "exact D128 default publication unexpectedly selects v509"
                )
            if forbidden_projection_fields & set(projection_backward):
                raise RuntimeError(
                    "exact D128 default route unexpectedly exposes v509 "
                    "projection fields"
                )
            if forbidden_publication_fields & set(publication):
                raise RuntimeError(
                    "exact D128 default route unexpectedly exposes v509 "
                    "publication fields"
                )

        backend = backward.get("backend", {})
        expected_backend = (
            _NATIVE_TK_D128_NVFP4_SCORE_E5M2_BACKEND
            if e5m2_dout_backward
            else (
                _NATIVE_TK_D128_NVFP4_SCORE_BACKEND
                if native_score_backward
                else _NATIVE_TK_D128_E4M3_BACKEND
            )
        )
        expected_source = (
            _native_tk_d128_v509_source(local_batch_size)
            if e5m2_dout_backward
            else (
                _NATIVE_TK_D128_V508_SOURCE
                if native_score_backward
                else _NATIVE_TK_D128_V501_SOURCE
            )
        )
        if backend.get("backend") != expected_backend:
            raise RuntimeError("exact D128 native TK backend mismatch")
        if (
            backend.get("extension_metadata", {}).get("source_identity")
            != expected_source
        ):
            raise RuntimeError("exact D128 native TK source identity mismatch")
        runtime_identity = getattr(
            runtime,
            "native_tk_d128_backward_extension_identity",
            {},
        )
        backend_identity = backend.get("extension", {})
        required_identity_fields = {
            "path",
            "sha256",
            "bytes",
            "device",
            "inode",
            "mtime_ns",
        }
        if (
            not isinstance(runtime_identity, dict)
            or set(runtime_identity) != required_identity_fields
            or not isinstance(backend_identity, dict)
            or set(backend_identity) != required_identity_fields
            or backend_identity != runtime_identity
        ):
            raise RuntimeError(
                "exact D128 native TK backend changed its loaded artifact " "identity"
            )


def _load_native_tk_d64_backward(
    exact_module: Any,
    settings: _ExactSettings,
) -> Any:
    """Load D64 v416 through its authenticated runner and binary identity."""
    path = settings.native_tk_d64_backward_extension
    module_name = settings.native_tk_d64_backward_module
    expected_sha256 = settings.native_tk_d64_backward_sha256
    expected_bytes = settings.native_tk_d64_backward_bytes
    if any(
        value is None for value in (path, module_name, expected_sha256, expected_bytes)
    ):
        raise RuntimeError("current D64 route has no complete native v416 identity")
    if module_name != _NATIVE_TK_D64_V416_MODULE:
        raise RuntimeError("current D64 route selected a non-v416 module")
    runner = importlib.import_module("tk_fa4.lowp_fa4_bwd.native_tk_d64_backward")
    if not _module_is_below(runner, settings.source_root):
        raise RuntimeError(
            "native D64 backward runner resolved outside the authenticated "
            "source root"
        )
    loader = getattr(exact_module, "_load_extension", None)
    if not callable(loader):
        raise RuntimeError(
            "exact runtime does not expose its authenticated extension loader"
        )
    extension = loader(path, module_name)
    if getattr(extension, "__name__", None) != module_name:
        raise RuntimeError("native D64 loader returned the wrong module identity")
    identity = getattr(extension, "_tk_fa4_loaded_artifact_identity", None)
    required_fields = {
        "path",
        "sha256",
        "bytes",
        "device",
        "inode",
        "mtime_ns",
    }
    if not isinstance(identity, dict) or set(identity) != required_fields:
        raise RuntimeError("native D64 loader returned no complete artifact receipt")
    expected = {
        "path": str(path.resolve()),
        "sha256": expected_sha256,
        "bytes": expected_bytes,
    }
    for field, value in expected.items():
        observed = identity.get(field)
        if observed != value or type(observed) is not type(value):
            raise RuntimeError(
                "loaded native D64 backward artifact identity mismatch: "
                f"{field}={observed!r} != {value!r}"
            )
    metadata_validator = getattr(runner, "_require_extension_metadata", None)
    if not callable(metadata_validator):
        raise RuntimeError("native D64 runner lacks metadata authentication")
    metadata = metadata_validator(extension)
    if metadata.get("source_identity") != _NATIVE_TK_D64_V416_SOURCE:
        raise RuntimeError("native D64 extension is not the authenticated v416 image")
    return extension


def _load_native_tk_d128_backward(
    exact_module: Any,
    settings: _ExactSettings,
) -> Any:
    """Load and re-authenticate the distinct current D128 consumer image."""
    path = settings.native_tk_d128_backward_extension
    module_name = settings.native_tk_d128_backward_module
    expected_sha256 = settings.native_tk_d128_backward_sha256
    expected_bytes = settings.native_tk_d128_backward_bytes
    if any(
        value is None for value in (path, module_name, expected_sha256, expected_bytes)
    ):
        raise RuntimeError(
            "current D128 route has no complete native-TK backward identity"
        )
    loader = getattr(exact_module, "_load_extension", None)
    if not callable(loader):
        raise RuntimeError(
            "exact runtime does not expose its authenticated extension loader"
        )
    extension = loader(path, module_name)
    if getattr(extension, "__name__", None) != module_name:
        raise RuntimeError(
            "native D128 backward loader returned the wrong module identity"
        )
    identity = getattr(extension, "_tk_fa4_loaded_artifact_identity", None)
    required_fields = {
        "path",
        "sha256",
        "bytes",
        "device",
        "inode",
        "mtime_ns",
    }
    if not isinstance(identity, dict) or set(identity) != required_fields:
        raise RuntimeError(
            "native D128 backward loader returned no complete artifact receipt"
        )
    expected = {
        "path": str(path.resolve()),
        "sha256": expected_sha256,
        "bytes": expected_bytes,
    }
    for field, value in expected.items():
        observed = identity.get(field)
        if observed != value or type(observed) is not type(value):
            raise RuntimeError(
                "loaded native D128 backward artifact identity mismatch: "
                f"{field}={observed!r} != {value!r}"
            )
    return extension


class _ExactRuntimeContext:
    """One process-local runtime shared by every converted attention layer."""

    def __init__(
        self,
        settings: _ExactSettings,
        profile: _ExactShapeProfile,
        layer_count: int,
        local_batch_size: int,
    ) -> None:
        self.settings = settings
        self.profile = profile
        if self.profile.qkv_projection_format != self.profile.output_projection_format:
            raise ValueError(
                "exact FA4 requires paired QKV/O learned projection formats"
            )
        if (
            self.settings.learned_projection_format
            != self.profile.qkv_projection_format
        ):
            raise ValueError(
                "exact learned projection setting/profile mismatch: "
                f"{self.settings.learned_projection_format!r} != "
                f"{self.profile.qkv_projection_format!r}"
            )
        if _is_d128_profile(self.profile):
            if (
                self.settings.d128_represented_qk_backward
                is not self.profile.backward_match_forward_operands
            ):
                raise ValueError("exact represented D128 Q/K setting/profile mismatch")
        elif self.settings.d128_represented_qk_backward:
            raise ValueError("represented D128 Q/K backward is D128-only")
        if self.settings.d128_native_score_backward and not _is_d128_profile(
            self.profile
        ):
            raise ValueError("native-score backward is D128-only")
        if self.settings.d128_e5m2_dout_backward and not _is_d128_profile(self.profile):
            raise ValueError("E5M2-dO backward is D128-only")
        self.layer_count = layer_count
        self.local_batch_size = _validated_local_batch_size(
            profile,
            local_batch_size,
            label="exact FA4",
            allow_current_d128=True,
        )
        if self.settings.forward_batch_size != self.local_batch_size:
            raise ValueError(
                "exact forward artifact batch identity does not match local "
                f"batch: {self.settings.forward_batch_size} != "
                f"{self.local_batch_size}"
            )
        if _is_d64_profile(self.profile) and self.local_batch_size == 16:
            if self.settings.artifact_profile.startswith(_D128_BUILD_PROFILE_PREFIX):
                raise ValueError("a D128 artifact profile cannot configure a D64 model")
            if self.settings.artifact_profile != _D64_BUILD_PROFILE:
                raise ValueError(
                    "the native D64 B16 route requires "
                    f"fa4.exact_artifact_profile={_D64_BUILD_PROFILE!r}"
                )
            if self.settings.native_tk_d64_backward_extension is None:
                raise ValueError(
                    "the native D64 B16 route requires its authenticated "
                    "v416 backward artifact"
                )
        elif self.settings.native_tk_d64_backward_extension is not None:
            raise ValueError(
                "the native D64 v416 artifact is valid only for D64 local batch 16"
            )
        if _is_d128_profile(self.profile):
            expected_profile = f"{_D128_BUILD_PROFILE_PREFIX}{self.local_batch_size}"
            if self.settings.artifact_profile not in ("", expected_profile):
                raise ValueError(
                    "D128 artifact profile does not match the selected local "
                    f"batch: {self.settings.artifact_profile!r} != "
                    f"{expected_profile!r}"
                )
        elif self.settings.artifact_profile.startswith(_D128_BUILD_PROFILE_PREFIX):
            raise ValueError("a D128 artifact profile cannot configure a D64 model")
        self.autograd_abi = _read_exact_autograd_abi(self.settings.source_root)
        current_autograd_route = _require_compatible_autograd_route(
            self.profile,
            self.local_batch_size,
            self.settings.pv_format,
            self.autograd_abi,
        )
        self.current_d64_route = bool(
            current_autograd_route
            and _is_d64_profile(self.profile)
            and self.local_batch_size == 16
        )
        self.current_d128_route = bool(
            current_autograd_route
            and _is_d128_profile(self.profile)
            and self.local_batch_size in _D128_CURRENT_BATCHES
        )
        if self.settings.mx_v_publication == _EXACT_OUTPUT_SHARED_SPLIT_V and not (
            self.current_d64_route and self.settings.pv_format == _EXACT_MXFP4_PV
        ):
            raise RuntimeError(
                "output_shared_split is authenticated only for the current "
                "D64 B16 native NVFP4-QK/MXFP4-PV split-V route"
            )
        if self.current_d128_route:
            expected_publication = _EXACT_RETAINED_SPLIT_V
            if self.settings.mx_v_publication != expected_publication:
                raise RuntimeError(
                    "current D128 route lost its authenticated V publication "
                    f"policy {expected_publication!r}"
                )
            if self.settings.native_tk_d128_backward_extension is None:
                raise RuntimeError(
                    "current D128 route requires its native-TK backward image"
                )
            if self.profile.backward_match_forward_operands:
                if self.profile.qkv_projection_format != _EXACT_NVFP4_PROJECTIONS:
                    raise RuntimeError(
                        "represented D128 Q/K backward requires NVFP4 QKV"
                    )
                if self.settings.pv_format != _EXACT_FP8_PV:
                    raise RuntimeError("represented D128 Q/K backward requires FP8-PV")
            if self.settings.d128_native_score_backward:
                if self.local_batch_size not in _D128_V509_AUTHENTICATED_BATCHES:
                    raise RuntimeError(
                        "native-score D128 backward is authenticated only "
                        "for local batch 1 or 4"
                    )
                if (
                    not self.settings.d128_e5m2_dout_backward
                    and not self.profile.backward_match_forward_operands
                ):
                    raise RuntimeError(
                        "native-score D128 backward requires represented-E4 "
                        "Q/K gradient operands"
                    )
            if self.settings.d128_e5m2_dout_backward:
                if self.local_batch_size not in _D128_V509_AUTHENTICATED_BATCHES:
                    raise RuntimeError(
                        "E5M2-dO D128 backward is authenticated only for "
                        "local batch 1 or 4"
                    )
                if self.profile.backward_match_forward_operands:
                    raise RuntimeError(
                        "E5M2-dO D128 backward requires ordinary retained " "E4M3 Q/K/V"
                    )
                if not self.settings.d128_native_score_backward:
                    raise RuntimeError(
                        "E5M2-dO D128 backward requires native-score backward"
                    )
        elif self.settings.d128_native_score_backward:
            raise RuntimeError(
                "native-score D128 backward requires the current D128 route"
            )
        elif self.settings.d128_e5m2_dout_backward:
            raise RuntimeError("E5M2-dO D128 backward requires the current D128 route")
        self.uses_packed_qkv = self.current_d64_route
        # Neither paired D128 learned-projection route uses the historical D64
        # native-NVFP4 RMSNorm fusion; retaining it would change the selected
        # projection ABI rather than merely fuse an equivalent operation.
        self.fuses_attention_rmsnorm = False
        self.qkv_projection_format = self.profile.qkv_projection_format
        self.output_projection_format = self.profile.output_projection_format
        self._lock = threading.Lock()
        self._exact_module: Any | None = None
        self._runtime: Any | None = None
        self._rope_identity: tuple[Any, ...] | None = None
        self._adapters: list[ExactLowpFA4Attention] = []
        self._rolling_schedule: _RollingD128WeightPackSchedule | None = None

    def register_adapter(self, adapter: "ExactLowpFA4Attention") -> int:
        if self._runtime is not None:
            raise RuntimeError("cannot register an exact layer after activation")
        layer_index = len(self._adapters)
        if layer_index >= self.layer_count:
            raise RuntimeError("registered more exact layers than configured")
        self._adapters.append(adapter)
        return layer_index

    def eager_initialize(self, freqs_cis: torch.Tensor) -> None:
        if len(self._adapters) != self.layer_count:
            raise RuntimeError(
                f"exact runtime registered {len(self._adapters)} layers, "
                f"expected {self.layer_count}"
            )
        self._initialize(self._adapters[0], freqs_cis)
        self._require_rope_identity(freqs_cis)

    def _load_exact_module(self) -> Any:
        existing_override = os.environ.get(_BACKWARD_EXTENSION_ENV)
        expected_override = str(self.settings.backward_extension)
        if existing_override is not None:
            if str(Path(existing_override).resolve()) != expected_override:
                raise RuntimeError(
                    f"{_BACKWARD_EXTENSION_ENV} already selects a different " "artifact"
                )
        else:
            os.environ[_BACKWARD_EXTENSION_ENV] = expected_override

        for prefix, expected_root in (
            ("tk_fa4", self.settings.source_root),
            ("flash_attn", self.settings.flash_attn_root),
            ("cutlass", self.settings.cutlass_dsl_root),
        ):
            stale_flash_modules: list[str] = []
            for name, module in tuple(sys.modules.items()):
                if name == prefix or name.startswith(prefix + "."):
                    if name == "tk_fa4._C_b300_lowp_bwd":
                        loaded_path = getattr(module, "__file__", None)
                        if (
                            loaded_path is not None
                            and Path(loaded_path).resolve()
                            == self.settings.backward_extension
                        ):
                            continue
                    if not _module_is_below(module, expected_root):
                        if prefix == "flash_attn":
                            stale_flash_modules.append(name)
                            continue
                        raise RuntimeError(
                            f"{name} was already loaded outside the authenticated "
                            f"exact FA4 source root {expected_root}"
                        )
            # LBT dependencies may import a wheel-installed FlashAttention
            # before model conversion. No exact route has run yet, so remove
            # those stale module-cache entries and authenticate the pinned
            # CuTe source below. This mirrors the established FA4 converter.
            for name in stale_flash_modules:
                del sys.modules[name]

        source = str(self.settings.source_root)
        if source not in sys.path:
            sys.path.insert(0, source)
        flash_source = str(self.settings.flash_attn_root)
        if flash_source not in sys.path:
            sys.path.insert(0, flash_source)
        cutlass_source = str(self.settings.cutlass_dsl_root)
        if cutlass_source not in sys.path:
            sys.path.insert(0, cutlass_source)
        cutlass = importlib.import_module("cutlass")
        if not _module_is_below(cutlass, self.settings.cutlass_dsl_root):
            raise RuntimeError(
                "CUTLASS DSL resolved outside its authenticated package root"
            )
        if getattr(cutlass, "__version__", None) != self.settings.cutlass_dsl_version:
            raise RuntimeError(
                "CUTLASS DSL version mismatch: "
                f"{getattr(cutlass, '__version__', None)!r} != "
                f"{self.settings.cutlass_dsl_version!r}"
            )
        flash_interface = importlib.import_module("flash_attn.cute.interface")
        if not _module_is_below(flash_interface, self.settings.flash_attn_root):
            raise RuntimeError(
                "FlashAttention CuTe interface resolved outside its "
                "authenticated source root"
            )
        exact_module = importlib.import_module(
            "tk_fa4.lowp_fa4_bwd.benchmark_llama12b_e2e"
        )
        if not _module_is_below(exact_module, self.settings.source_root):
            raise RuntimeError(
                "exact FA4 trainer module resolved outside source capsule"
            )
        loaded_autograd_abi = _require_exact_autograd_abi(exact_module)
        if loaded_autograd_abi != self.autograd_abi:
            raise RuntimeError(
                "exact runtime autograd ABI changed between source inspection "
                f"and import: {loaded_autograd_abi} != {self.autograd_abi}"
            )

        interface = importlib.import_module("tk_fa4.interface")
        loaded_extension = getattr(interface, "_C_b300_lowp_bwd", None)
        loaded_path = getattr(loaded_extension, "__file__", None)
        if (
            loaded_path is None
            or Path(loaded_path).resolve() != self.settings.backward_extension
        ):
            raise RuntimeError(
                "tk_fa4 loaded a projection artifact other than the "
                "authenticated exact backward extension"
            )
        return exact_module

    def _initialize(
        self,
        adapter: "ExactLowpFA4Attention",
        freqs_cis: torch.Tensor,
    ) -> None:
        if self._runtime is not None:
            return
        with self._lock:
            if self._runtime is not None:
                return
            weight = adapter._workspace_anchor_weight()
            if weight.device.type != "cuda":
                raise RuntimeError("exact FA4 runtime can only initialize on CUDA")
            capability = torch.cuda.get_device_capability(weight.device)
            if capability != (10, 0):
                raise RuntimeError(
                    "exact FA4 requires SM100/GB200; observed compute capability "
                    f"{capability}"
                )
            if not freqs_cis.is_complex():
                raise TypeError("exact FA4 requires TorchTitan's complex RoPE table")
            if freqs_cis.device != weight.device:
                raise RuntimeError(
                    "RoPE table and attention weights are on different devices"
                )
            expected_freq_shape = (self.profile.head_dim // 2,)
            if freqs_cis.ndim != 2 or tuple(freqs_cis.shape[1:]) != expected_freq_shape:
                raise ValueError(
                    "unexpected RoPE table shape for exact FA4: "
                    f"{tuple(freqs_cis.shape)}"
                )
            if freqs_cis.shape[0] != _EXACT_SEQUENCE:
                raise ValueError("exact FA4 RoPE table length must be 4096")

            exact_module = self._load_exact_module()
            config = _runtime_config_for_local_batch(
                exact_module,
                self.profile,
                layer_count=self.layer_count,
                local_batch_size=self.local_batch_size,
            )
            expected_config = {
                "batch": self.local_batch_size,
                "sequence": _EXACT_SEQUENCE,
                "hidden": self.profile.hidden,
                "q_heads": self.profile.q_heads,
                "kv_heads": self.profile.kv_heads,
                "head_dim": self.profile.head_dim,
            }
            for name, expected in expected_config.items():
                actual = getattr(
                    config,
                    name,
                    _EXACT_ARTIFACT_BATCH if name == "batch" else None,
                )
                if actual != expected:
                    raise RuntimeError(
                        f"exact source config {name}={actual} does not match {expected}"
                    )

            extension, topology = exact_module._load_forward(
                self.settings.forward_extension,
                self.settings.forward_module,
                config,
            )
            _require_selected_forward_topology(
                self.profile,
                self.local_batch_size,
                self.settings.pv_format,
                topology,
                current_d64_route=self.current_d64_route,
                current_d128_route=self.current_d128_route,
            )

            rope_values = freqs_cis[:_EXACT_SEQUENCE]
            rope = tuple(
                component.unsqueeze(0)
                .expand(self.local_batch_size, -1, -1)
                .to(torch.bfloat16)
                .contiguous()
                for component in (rope_values.real, rope_values.imag)
            )
            runtime_kwargs: dict[str, Any] = {
                "forward_extension": extension,
                "forward_topology": topology,
                "loss_scale": 2.0**16,
                "gradient_global_scale": 2.0**-8,
                "projection_dgrad": self.profile.projection_dgrad,
                "qkv_projection_format": self.qkv_projection_format,
                "output_projection_format": self.output_projection_format,
                "backward_match_forward_operands": (
                    self.profile.backward_match_forward_operands
                ),
                "per_block_qk_scales": self.profile.per_block_qk_scales,
                "backward_probability_correction": 1.0,
                "q_quant_scale": 2.25,
                "k_quant_scale": 2.0,
                "projection_weight_scale_2d": True,
                "v_mxfp4_scale_2d": False,
                "adaptive_qk_weight_scales": False,
            }
            runtime_kwargs.update(
                _current_runtime_only_kwargs(
                    self.current_d64_route,
                    self.settings.pv_format,
                    self.settings.mx_v_publication,
                    current_d128_route=self.current_d128_route,
                    learned_projection_format=(self.qkv_projection_format),
                    native_score_backward=(self.settings.d128_native_score_backward),
                    represented_qk_backward=(
                        self.profile.backward_match_forward_operands
                    ),
                    e5m2_dout_backward=(self.settings.d128_e5m2_dout_backward),
                )
            )
            backward_policy = _exact_backward_policy(
                self.profile,
                current_d128_route=self.current_d128_route,
                native_d64_route=self.current_d64_route,
            )
            runtime_kwargs.update(backward_policy)
            if self.profile.head_dim == 64:
                if self.current_d64_route:
                    if self.settings.backward_control_source is not None:
                        raise RuntimeError(
                            "native D64 v416 cannot be represented by a CuTe "
                            "backward control"
                        )
                else:
                    if self.settings.backward_control_source is None:
                        raise RuntimeError(
                            "the legacy D64 route requires a precomposed "
                            "CuTe backward control"
                        )
                    runtime_kwargs.update(
                        backward_control_source=self.settings.backward_control_source,
                        backward_control_sha256=(self.settings.backward_control_sha256),
                        backward_control_bytes=self.settings.backward_control_bytes,
                    )
            elif self.settings.backward_control_source is not None:
                raise RuntimeError("D128 exact FA4 must not receive a D64 control")

            native_tk_d64_backward_extension = None
            if self.current_d64_route:
                native_tk_d64_backward_extension = _load_native_tk_d64_backward(
                    exact_module,
                    self.settings,
                )
                runtime_kwargs["native_tk_d64_backward_extension"] = (
                    native_tk_d64_backward_extension
                )

            native_tk_d128_backward_extension = None
            if self.current_d128_route:
                native_tk_d128_backward_extension = _load_native_tk_d128_backward(
                    exact_module,
                    self.settings,
                )
                runtime_kwargs["native_tk_d128_backward_extension"] = (
                    native_tk_d128_backward_extension
                )

            runtime = exact_module.LowpAttentionRuntime(
                config,
                rope,
                **runtime_kwargs,
            )
            runtime_config_batch = int(
                getattr(
                    runtime.config,
                    "batch",
                    _EXACT_ARTIFACT_BATCH,
                )
            )
            if runtime_config_batch != self.local_batch_size:
                raise RuntimeError(
                    "exact LowpAttentionRuntime lost the authenticated batch: "
                    f"{runtime_config_batch} != {self.local_batch_size}"
                )
            if self.current_d64_route or self.current_d128_route:
                for contract_name in (
                    "forward_dispatch_contract",
                    "backward_contract",
                ):
                    contract_builder = getattr(runtime, contract_name, None)
                    if not callable(contract_builder):
                        raise RuntimeError(
                            f"exact runtime has no {contract_name} " "authentication"
                        )
                    contract = contract_builder()
                    contract_batch = int(
                        contract.get("shape", {}).get(
                            "batch",
                            _EXACT_ARTIFACT_BATCH,
                        )
                    )
                    if contract_batch != self.local_batch_size:
                        raise RuntimeError(
                            f"exact runtime {contract_name} batch mismatch: "
                            f"{contract_batch} != {self.local_batch_size}"
                        )
            if self.current_d64_route or self.current_d128_route:
                projection_batch = int(
                    getattr(
                        runtime.qkv_projection,
                        "batch",
                        _EXACT_ARTIFACT_BATCH,
                    )
                )
                if projection_batch != self.local_batch_size:
                    raise RuntimeError(
                        "exact current QKV projection batch mismatch: "
                        f"{projection_batch} != {self.local_batch_size}"
                    )
            if int(runtime.qk_scales.shape[0]) != self.local_batch_size:
                raise RuntimeError(
                    "exact Q/K policy scales do not cover the local batch"
                )
            for name in ("dq", "dk", "dv"):
                gradient = getattr(runtime.backward, name, None)
                if (
                    not isinstance(gradient, torch.Tensor)
                    or gradient.ndim == 0
                    or int(gradient.shape[0]) != self.local_batch_size
                ):
                    raise RuntimeError(
                        f"exact backward {name} does not cover local batch "
                        f"{self.local_batch_size}"
                    )
            actual_backward_policy = {
                "backward_exp2_degree": runtime.backward_exp2_degree,
                "backward_exp2_period": runtime.backward_exp2_period,
                "backward_fp8_ds_lift": (
                    None
                    if self.current_d64_route or self.current_d128_route
                    else int(runtime.backward.kernel.fp8_ds_lift)
                ),
                "backward_reuse_quantized_p": bool(runtime.backward_reuse_quantized_p),
            }
            if actual_backward_policy != backward_policy:
                raise RuntimeError(
                    "exact FA4 backward policy mismatch: "
                    f"{actual_backward_policy} != {backward_policy}"
                )
            if self.current_d64_route or self.current_d128_route:
                _require_runtime_route_contract(
                    runtime,
                    self.profile,
                    self.local_batch_size,
                    self.settings.pv_format,
                    self.settings.mx_v_publication,
                    self.settings.d128_native_score_backward,
                    self.settings.d128_e5m2_dout_backward,
                )
            publication = runtime.projection_publication_topology
            expected_publication = {
                "qkv_projection_format": self.qkv_projection_format,
                "output_projection_format": self.output_projection_format,
                "forward_pv_format": self.settings.pv_format,
                "represented_backward": (self.profile.backward_match_forward_operands),
                "per_block_qk_scales": self.profile.per_block_qk_scales,
                "native_tk_d128_native_score_backward": (
                    self.settings.d128_native_score_backward
                ),
            }
            if self.settings.d128_e5m2_dout_backward:
                v509_receipt = publication.get("v509_e5m2_dout_route")
                _require_v509_e5m2_dout_route_receipt(v509_receipt)
                expected_publication.update(
                    native_tk_d128_v509_e5m2_dout_backward=True,
                    dout_backward_format="e5m2",
                    dout_backward_source=_D128_V509_E5M2_DOUT_SOURCE,
                    dout_backward_kernel=_D128_V509_E5M2_DOUT_KERNEL,
                    v509_e5m2_dout_route=v509_receipt,
                )
            if self.current_d64_route:
                expected_publication.update(
                    v_backward_source="projection_accumulator_e4m3",
                    experimental_split_v_backward=(
                        self.settings.pv_format == _EXACT_MXFP4_PV
                    ),
                    experimental_output_shared_split_v=(
                        self.settings.mx_v_publication == _EXACT_OUTPUT_SHARED_SPLIT_V
                    ),
                    experimental_native_nvfp4_projection_out=False,
                    experimental_fused_attention_rmsnorm_nvfp4=False,
                    output_projection=(runtime.output_projection_topology),
                )
            elif self.current_d128_route:
                native_nvfp4_projection = (
                    self.qkv_projection_format == _EXACT_NVFP4_PROJECTIONS
                )
                represented_qk_backward = self.profile.backward_match_forward_operands
                expected_publication.update(
                    qk_backward_source=(
                        "represented_nvfp4_codes_per_row_k16"
                        if represented_qk_backward
                        else "projection_accumulator_e4m3"
                    ),
                    v_backward_source="projection_accumulator_e4m3",
                    experimental_split_v_backward=False,
                    experimental_output_shared_split_v=False,
                    experimental_d128_mxfp4_v_backward=False,
                    d128_mxfp4_v_scale_policy=None,
                    experimental_native_nvfp4_projection_out=(native_nvfp4_projection),
                    experimental_fused_attention_rmsnorm_nvfp4=False,
                    projection_forward_publication_path=(
                        "caller_owned_represented_qk_fp8_pv_d128"
                        if represented_qk_backward
                        else (
                            "caller_owned_route_selective_d128"
                            if native_nvfp4_projection
                            else "caller_owned_dense_e4m3_d128"
                        )
                    ),
                    output_projection=(runtime.output_projection_topology),
                )
            for name, expected in expected_publication.items():
                actual = publication.get(name)
                if actual != expected:
                    raise RuntimeError(
                        f"exact FA4 publication {name}={actual!r} does not "
                        f"match {expected!r}"
                    )
            self._rope_identity = (
                str(freqs_cis.device),
                tuple(freqs_cis.shape),
                int(freqs_cis.data_ptr()),
                int(freqs_cis._version),
            )
            self._exact_module = exact_module
            self._runtime = runtime
            authenticated_workspace_bytes = 0
            expected_workspace_bytes_per_layer = None
            if self.current_d64_route:
                expected_workspace_bytes_per_layer = (
                    _D64_FORWARD_WORKSPACE_BYTES_PER_BATCH_PER_LAYER
                    * self.local_batch_size
                )
            for layer_adapter in self._adapters:
                if layer_adapter._forward_workspace is not None:
                    raise RuntimeError(
                        "exact FA4 workspace was allocated before activation"
                    )
                proxy = _forward_workspace_allocator_proxy(
                    layer_adapter,
                    runtime,
                )
                layer_adapter._forward_workspace = (
                    exact_module.LowpAttention._allocate_forward_workspace(proxy)
                )
                if expected_workspace_bytes_per_layer is not None:
                    authenticated_workspace_bytes += _require_forward_workspace_batch(
                        exact_module,
                        layer_adapter._forward_workspace,
                        self.local_batch_size,
                        expected_owner_count=(_D64_FORWARD_WORKSPACE_OWNER_COUNT),
                        expected_total_bytes=(expected_workspace_bytes_per_layer),
                    )
                layer_adapter._qk_scales = runtime.qk_scales.clone()
            if expected_workspace_bytes_per_layer is not None:
                expected_workspace_bytes = (
                    expected_workspace_bytes_per_layer * self.layer_count
                )
                if authenticated_workspace_bytes != expected_workspace_bytes:
                    raise RuntimeError(
                        "exact D64 batched workspace total byte mismatch: "
                        f"{authenticated_workspace_bytes} != "
                        f"{expected_workspace_bytes}"
                    )

            schedule = "serial"
            if _uses_rolling_d128_weight_pack(
                self.profile,
                self.qkv_projection_format,
            ):
                if self.settings.allow_fp32_master_shadows:
                    raise RuntimeError(
                        "D128 rolling weight packing rejects FP32 master shadows"
                    )
                rolling_schedule = _RollingD128WeightPackSchedule(
                    exact_module,
                    self._adapters,
                )
                rolling_schedule.authenticate()
                self._rolling_schedule = rolling_schedule
                schedule = "rolling"
            logger.info(
                "Initialized exact %s-QKV/%s-O/%s-PV FA4 profile=%s layers=%d "
                "source_sha256=%s flash_source_sha256=%s forward_sha256=%s "
                "backward_sha256=%s cutlass_dsl=%s/%s "
                "fp32_master_shadows=%s local_batch=%d artifact_batch=%d "
                "mx_v_publication=%s projection_checked_symbol=%s "
                "projection_unchecked_symbol=%s workspace_bytes=%d",
                self.qkv_projection_format,
                self.output_projection_format,
                self.settings.pv_format,
                self.profile.name,
                self.layer_count,
                self.settings.runtime_source_sha256,
                self.settings.flash_attn_source_sha256,
                self.settings.forward_sha256,
                self.settings.backward_sha256,
                self.settings.cutlass_dsl_version,
                self.settings.cutlass_dsl_native_sha256,
                self.settings.allow_fp32_master_shadows,
                self.local_batch_size,
                self.settings.forward_batch_size,
                self.settings.mx_v_publication,
                getattr(runtime.qkv_projection, "checked_symbol", None),
                getattr(runtime.qkv_projection, "unchecked_symbol", None),
                authenticated_workspace_bytes,
            )
            logger.info(
                "[EXACT FA4 SCHEDULE] profile=%s schedule=%s "
                "authenticated=true layers=%d",
                self.profile.contract_name,
                schedule,
                self.layer_count,
            )

    def _require_rope_identity(self, freqs_cis: torch.Tensor) -> None:
        identity = (
            str(freqs_cis.device),
            tuple(freqs_cis.shape),
            int(freqs_cis.data_ptr()),
            int(freqs_cis._version),
        )
        if identity != self._rope_identity:
            raise RuntimeError(
                "exact FA4 runtime cannot change RoPE storage after initialization"
            )

    def forward(
        self,
        adapter: "ExactLowpFA4Attention",
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        self._initialize(adapter, freqs_cis)
        self._require_rope_identity(freqs_cis)
        assert self._runtime is not None and self._exact_module is not None
        runtime = self._runtime
        exact_module = self._exact_module
        exact_module.require_active_forward_route(
            str(runtime.forward_topology["route"])
        )

        if adapter._forward_workspace is None:
            raise RuntimeError("exact FA4 adapter has no activated workspace")
        if adapter._qk_scales is None:
            raise RuntimeError("exact FA4 layer has no Q/K scale publication")

        rolling_schedule = self._rolling_schedule
        if rolling_schedule is not None and adapter._layer_index == 0:
            rolling_schedule.begin_forward()

        allow_fp32 = self.settings.allow_fp32_master_shadows
        x_compute = _as_bf16_compute_tensor(
            x,
            allow_fp32_master_shadows=allow_fp32,
            label="attention input",
        )
        out_weight = _as_bf16_compute_tensor(
            adapter.wo.weight,
            allow_fp32_master_shadows=allow_fp32,
            label="attention O weight",
        )
        if self.current_d64_route:
            empty_weight = adapter._forward_workspace.outputs.empty_bf16
            packed_qkv_weight = _as_bf16_compute_tensor(
                adapter.packed_qkv,
                allow_fp32_master_shadows=allow_fp32,
                label="attention packed QKV weight",
            )
            q_weight = k_weight = v_weight = empty_weight
            if self.fuses_attention_rmsnorm:
                attention_norm_weight = _as_bf16_compute_tensor(
                    adapter._attention_norm_weight(),
                    allow_fp32_master_shadows=allow_fp32,
                    label="attention RMSNorm weight",
                )
            else:
                # Stock TorchTitan already applied attention_norm before this
                # module. The current autograd ABI retains this positional slot
                # but reads it only when the native-NVFP4 RMSNorm fusion is on.
                attention_norm_weight = packed_qkv_weight
        else:
            q_weight, k_weight, v_weight = tuple(
                _as_bf16_compute_tensor(
                    linear.weight,
                    allow_fp32_master_shadows=allow_fp32,
                    label=f"attention {name} weight",
                )
                for name, linear in (
                    ("Q", adapter.wq),
                    ("K", adapter.wk),
                    ("V", adapter.wv),
                )
            )
            # These placeholders are not forwarded through the authenticated
            # legacy split-QKV ABI; avoid assuming its workspace has current
            # source sentinels.
            attention_norm_weight = q_weight
            packed_qkv_weight = q_weight
        output = _apply_exact_artifact_once(
            exact_module,
            x_compute,
            attention_norm_weight,
            packed_qkv_weight,
            q_weight,
            k_weight,
            v_weight,
            out_weight,
            adapter._qk_scales,
            adapter._forward_workspace,
            runtime,
            self.autograd_abi,
        )
        if rolling_schedule is not None:
            rolling_schedule.complete_layer(
                adapter._layer_index,
                requires_backward=bool(output.requires_grad),
            )
        return output


class ExactLowpFA4Attention(nn.Module):
    """Checkpoint-compatible replacement for TorchTitan Llama Attention."""

    def __init__(
        self,
        original: nn.Module,
        runtime_context: _ExactRuntimeContext,
        profile: _ExactShapeProfile,
        attention_norm_owner: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.n_heads = int(original.n_heads)
        self.n_kv_heads = int(original.n_kv_heads)
        self.n_rep = int(original.n_rep)
        self.head_dim = int(original.head_dim)
        self.wo = original.wo
        self.use_flex_attn = False
        self._runtime_context = runtime_context
        self._profile = profile
        uses_packed_qkv = bool(runtime_context.uses_packed_qkv)
        self._uses_packed_qkv = uses_packed_qkv
        if uses_packed_qkv:
            if not _is_d64_profile(profile):
                raise ValueError("packed exact QKV is authenticated only for D64")
            split_parameters = (
                original.wq.weight,
                original.wk.weight,
                original.wv.weight,
            )
            requires_grad = {parameter.requires_grad for parameter in split_parameters}
            if len(requires_grad) != 1:
                raise ValueError("exact D64 Q/K/V weights must share requires_grad")
            packed = _pack_qkv_weights(*split_parameters, profile)
            self.packed_qkv = nn.Parameter(
                packed,
                requires_grad=requires_grad.pop(),
            )
        else:
            # Preserve the legacy D64 and established D128 split schemas
            # byte-for-byte.
            self.wq = original.wq
            self.wk = original.wk
            self.wv = original.wv

        fused_norm = bool(runtime_context.fuses_attention_rmsnorm)
        if fused_norm:
            _require_attention_norm_owner(attention_norm_owner, profile)
        elif attention_norm_owner is not None:
            raise ValueError("attention_norm_owner is accepted only by fused D64 B16")
        # Deliberately bypass Module.__setattr__: the block remains the sole
        # registered owner of attention_norm and its Parameter. Always
        # dereference owner.weight so meta->to_empty Parameter replacement is
        # observed by this adapter instead of leaving a stale alias.
        object.__setattr__(self, "_attention_norm_owner", attention_norm_owner)
        # These CUDA scratch objects are intentionally neither parameters nor
        # buffers. DDP must not broadcast mutable per-layer publications.
        self._forward_workspace: Any | None = None
        self._qk_scales: torch.Tensor | None = None
        self._layer_index = runtime_context.register_adapter(self)

    def init_weights(self, init_std: float) -> None:
        if self._uses_packed_qkv:
            # Preserve stock TorchTitan's three RNG call boundaries exactly:
            # wq, then wk, then wv. This keeps seeded BF16/FP8/MX initial
            # conditions paired while retaining one optimizer-owned leaf.
            for weight in _unpack_qkv_weight(
                self.packed_qkv,
                self._profile,
            ):
                _safe_trunc_normal_(weight, mean=0.0, std=0.02)
        else:
            for linear in (self.wq, self.wk, self.wv):
                _safe_trunc_normal_(linear.weight, mean=0.0, std=0.02)
        _safe_trunc_normal_(self.wo.weight, mean=0.0, std=init_std)

    def _workspace_anchor_weight(self) -> torch.Tensor:
        if self._uses_packed_qkv:
            return self.packed_qkv
        return self.wq.weight

    def _attention_norm_weight(self) -> nn.Parameter:
        owner = object.__getattribute__(self, "_attention_norm_owner")
        if owner is None:
            raise RuntimeError("exact D64 B16 adapter lost its RMSNorm owner")
        weight = getattr(owner, "weight", None)
        if not isinstance(weight, nn.Parameter):
            raise RuntimeError("exact D64 B16 RMSNorm owner lost its Parameter")
        if tuple(weight.shape) != (self._profile.hidden,):
            raise RuntimeError("exact D64 B16 RMSNorm weight shape changed")
        return weight

    def split_qkv_weights(
        self,
        *,
        clone: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expose the D64 logical Q/K/V rows for model-only comparisons."""
        if not self._uses_packed_qkv:
            raise ValueError("this exact route retains native split Q/K/V modules")
        return _unpack_qkv_weight(
            self.packed_qkv,
            self._profile,
            clone=clone,
        )

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        if self._uses_packed_qkv:
            packed_key = f"{prefix}packed_qkv"
            split_keys = tuple(f"{prefix}{name}.weight" for name in ("wq", "wk", "wv"))
            split_present = tuple(key in state_dict for key in split_keys)
            if packed_key in state_dict and any(split_present):
                error_msgs.append(
                    "checkpoint contains both packed and split exact D64 QKV "
                    f"at {prefix!r}"
                )
            elif any(split_present) and not all(split_present):
                error_msgs.append(
                    "checkpoint contains incomplete split exact D64 QKV at "
                    f"{prefix!r}"
                )
            elif packed_key not in state_dict and all(split_present):
                try:
                    state_dict[packed_key] = _pack_qkv_weights(
                        state_dict[split_keys[0]],
                        state_dict[split_keys[1]],
                        state_dict[split_keys[2]],
                        self._profile,
                    )
                except (TypeError, ValueError, RuntimeError) as error:
                    error_msgs.append(str(error))
                else:
                    for key in split_keys:
                        del state_dict[key]
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _apply(self, fn: Any, recurse: bool = True) -> "ExactLowpFA4Attention":
        if self._forward_workspace is not None or self._qk_scales is not None:
            raise RuntimeError(
                "exact FA4 attention cannot migrate after lazy CUDA runtime "
                "initialization"
            )
        return super()._apply(fn, recurse)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: Any,
    ) -> torch.Tensor:
        if attention_masks is not None:
            raise ValueError("exact FA4 supports causal attention_masks=None only")
        expected_shape = (
            self._runtime_context.local_batch_size,
            _EXACT_SEQUENCE,
            self._profile.hidden,
        )
        if tuple(x.shape) != expected_shape:
            raise ValueError(
                f"exact FA4 expected attention input {expected_shape}, got "
                f"{tuple(x.shape)}"
            )
        if x.device.type != "cuda":
            raise RuntimeError("exact FA4 attention input must be on CUDA")
        return self._runtime_context.forward(self, x, freqs_cis)


def _require_attention_norm_owner(
    norm: nn.Module | None,
    profile: _ExactShapeProfile,
) -> None:
    if not isinstance(norm, nn.Module):
        raise TypeError("fused exact D64 B16 requires block.attention_norm")
    weight = getattr(norm, "weight", None)
    if not isinstance(weight, nn.Parameter):
        raise TypeError("fused exact D64 B16 requires RMSNorm.weight Parameter")
    if tuple(weight.shape) != (profile.hidden,):
        raise ValueError(
            "fused exact D64 B16 RMSNorm weight must have shape "
            f"{(profile.hidden,)}, got {tuple(weight.shape)}"
        )
    normalized_shape = tuple(getattr(norm, "normalized_shape", ()))
    if normalized_shape != (profile.hidden,):
        raise ValueError(
            "fused exact D64 B16 requires RMSNorm over the hidden dimension"
        )
    if float(getattr(norm, "eps", float("nan"))) != 1.0e-5:
        raise ValueError("fused exact D64 B16 requires RMSNorm eps=1e-5")


def _fused_exact_attention_block_forward(
    block: nn.Module,
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    attention_masks: Any | None,
) -> torch.Tensor:
    if not getattr(block, "_exact_fa4_fused_attention_rmsnorm", False):
        raise RuntimeError("fused exact attention block is not authenticated")
    # The custom attention autograd function now owns both forward RMSNorm and
    # its exact CUDA derivative. FFN RMSNorm remains the ordinary block module.
    h = x + block.attention(x, freqs_cis, attention_masks)
    return h + block.feed_forward(block.ffn_norm(h))


def _install_fused_attention_rmsnorm_block_forward(
    block: nn.Module,
    profile: _ExactShapeProfile,
) -> None:
    if getattr(block, "_exact_fa4_fused_attention_rmsnorm", False):
        raise RuntimeError("fused exact attention block is already installed")
    if not isinstance(getattr(block, "attention", None), ExactLowpFA4Attention):
        raise TypeError("fused exact attention block requires its exact adapter")
    norm = getattr(block, "attention_norm", None)
    _require_attention_norm_owner(norm, profile)
    if block.attention._attention_norm_weight() is not norm.weight:
        raise RuntimeError("exact adapter does not reference the block RMSNorm owner")
    block._exact_fa4_fused_attention_rmsnorm = True
    block.forward = MethodType(_fused_exact_attention_block_forward, block)


def _allowed_exact_converter_chains(
    *,
    allow_fp32: bool,
) -> tuple[tuple[str, ...], ...]:
    suffix = ("fp32_master",) if allow_fp32 else ()
    return tuple(
        chain
        for converter in _EXACT_CONVERTERS
        for chain in (
            ("bfloat16", converter, *suffix),
            (
                "bfloat16",
                "fuse_mlp_linear",
                "spline_mlp",
                converter,
                *suffix,
            ),
        )
    )


def _validate_converter_contract(
    job_config: JobConfig,
    parallel_dims: ParallelDims,
) -> None:
    if not job_config.fa4.enabled:
        raise ValueError("fa4.enabled must be true for the exact FA4 converter")
    if job_config.model.name != "llama3_gc":
        raise ValueError("exact FA4 currently supports model.name='llama3_gc' only")
    profile = _validate_job_rope_contract(job_config)
    if int(job_config.training.seq_len) != _EXACT_SEQUENCE:
        raise ValueError("exact FA4 requires training.seq_len=4096")
    _validated_local_batch_size(
        profile,
        job_config.training.local_batch_size,
        label="exact FA4",
        allow_current_d128=True,
    )
    if int(job_config.fa4.exact_forward_batch_size) != int(
        job_config.training.local_batch_size
    ):
        raise ValueError(
            "exact FA4 training.local_batch_size must match "
            "fa4.exact_forward_batch_size"
        )
    if job_config.training.dtype != "bfloat16":
        raise ValueError("exact FA4 requires training.dtype='bfloat16'")
    allow_fp32 = bool(job_config.fa4.exact_allow_fp32_master_shadows)
    fp32_master_enabled = bool(job_config.training.enable_fp32_master_params)
    converters = tuple(job_config.model.converters or ())
    if fp32_master_enabled != allow_fp32:
        raise ValueError(
            "training.enable_fp32_master_params must equal "
            "fa4.exact_allow_fp32_master_shadows; this makes master-weight "
            "casts explicit"
        )
    if ("fp32_master" in converters) != allow_fp32:
        raise ValueError(
            "model.converters must include fp32_master exactly when exact "
            "FP32 master shadows are enabled"
        )
    allowed_converters = _allowed_exact_converter_chains(
        allow_fp32=allow_fp32,
    )
    if converters not in allowed_converters:
        raise ValueError(
            "exact FA4 requires model.converters exactly equal to one "
            "authenticated chain: "
            f"{[list(chain) for chain in allowed_converters]!r}; got "
            f"{list(converters)!r}"
        )
    if bool(getattr(job_config.training, "compile", False)):
        raise ValueError("exact FA4 has not been authenticated under torch.compile")
    if bool(getattr(job_config.training, "enable_cce", False)) or bool(
        getattr(getattr(job_config, "fp4_cce", None), "enabled", False)
    ):
        raise ValueError(
            "exact FA4 requires the ordinary output head and external loss; "
            "CCE must be disabled"
        )
    compile_config = getattr(job_config, "compile", None)
    if compile_config is not None and bool(getattr(compile_config, "enable", False)):
        components = tuple(getattr(compile_config, "components", ()))
        backend = getattr(compile_config, "backend", None)
        if components != ("loss",) or backend != "inductor":
            raise ValueError(
                "exact FA4 torch.compile is authenticated only for the "
                "ordinary loss component with the inductor backend"
            )
    activation_checkpoint = getattr(job_config, "activation_checkpoint", None)
    if (
        activation_checkpoint is not None
        and getattr(activation_checkpoint, "mode", "none") != "none"
    ):
        raise ValueError(
            "exact FA4 activation checkpointing is not yet an authenticated path"
        )

    required_one = {
        "dp_shard": int(parallel_dims.dp_shard),
        "tp": int(parallel_dims.tp),
        "pp": int(parallel_dims.pp),
        "cp": int(parallel_dims.cp),
    }
    invalid = {name: value for name, value in required_one.items() if value != 1}
    if invalid:
        raise ValueError(
            "exact FA4 supports replicated DDP only; invalid parallel dims "
            f"{invalid}"
        )
    if int(parallel_dims.dp_replicate) < 1:
        raise ValueError("exact FA4 requires a positive DDP replicate degree")


class ExactLowpFA4AttentionConverter(ModelConverter):
    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        _validate_converter_contract(job_config, parallel_dims)
        self.job_profile = _profile_from_job_flavor(job_config)
        self.local_batch_size = int(job_config.training.local_batch_size)
        self.settings = _ExactSettings.from_job_config(job_config)

    def convert(self, model: nn.Module) -> None:
        layers = getattr(model, "layers", None)
        if not isinstance(layers, (nn.ModuleDict, nn.ModuleList)):
            raise TypeError("exact FA4 requires TorchTitan Llama model.layers")
        blocks = (
            list(layers.values()) if isinstance(layers, nn.ModuleDict) else list(layers)
        )
        if not blocks:
            raise ValueError("exact FA4 cannot convert a model with no layers")
        attentions = [getattr(block, "attention", None) for block in blocks]
        if any(attention is None for attention in attentions):
            raise TypeError("every Llama block must expose an attention module")
        observed_profiles = [
            _select_shape_profile(attention) for attention in attentions
        ]
        observed_profile = observed_profiles[0]
        if observed_profile.contract_name != self.job_profile.contract_name:
            raise ValueError(
                "exact FA4 model flavor and attention shape profiles disagree: "
                f"{self.job_profile.contract_name} != "
                f"{observed_profile.contract_name}"
            )
        profile = self.job_profile
        _validate_model_rope_contract(model, profile)
        if any(candidate != observed_profile for candidate in observed_profiles[1:]):
            raise ValueError(
                "exact FA4 does not support heterogeneous attention shapes"
            )
        if len(blocks) != profile.layers:
            raise ValueError(
                f"exact {profile.contract_name} requires {profile.layers} "
                f"layers, got {len(blocks)}"
            )
        if profile.head_dim == 64:
            if self.local_batch_size == 16:
                if self.settings.native_tk_d64_backward_extension is None:
                    raise ValueError(
                        "D64 B16 exact FA4 requires its authenticated native "
                        "v416 backward"
                    )
                if self.settings.backward_control_source is not None:
                    raise ValueError(
                        "D64 B16 native v416 must not configure a CuTe "
                        "backward control"
                    )
            elif self.settings.backward_control_source is None:
                raise ValueError(
                    "legacy D64 exact FA4 requires its authenticated CuTe "
                    "control source"
                )
        if profile.head_dim == 128 and (
            self.settings.backward_control_source is not None
            or self.settings.native_tk_d64_backward_extension is not None
        ):
            raise ValueError("D128 exact FA4 must not configure D64 backward inputs")
        if profile.head_dim == 128:
            if self.local_batch_size == 1:
                if self.settings.native_tk_d128_backward_extension is None:
                    if self.settings.pv_format != _EXACT_FP8_PV:
                        raise ValueError(
                            "legacy D128 B1 exact FA4 authenticates FP8-PV " "only"
                        )
                    if (
                        self.settings.learned_projection_format
                        != _EXACT_E4M3_PROJECTIONS
                    ):
                        raise ValueError(
                            "legacy D128 B1 exact FA4 authenticates E4M3 "
                            "learned projections only"
                        )
                elif self.settings.pv_format not in _EXACT_PV_FORMATS:
                    raise ValueError("current D128 B1 PV format is unsupported")
            elif self.local_batch_size in (2, 4):
                if self.settings.pv_format not in _EXACT_PV_FORMATS:
                    raise ValueError("current D128 B2/B4 PV format is unsupported")
                if self.settings.native_tk_d128_backward_extension is None:
                    raise ValueError("current D128 B2/B4 requires native TK backward")
            else:
                raise ValueError("unsupported D128 exact local batch")
        if profile.head_dim == 128 and self.settings.allow_fp32_master_shadows:
            raise ValueError(
                "D128 exact FA4 requires BF16 parameter storage for rolling "
                "dual-weight packing"
            )

        runtime_context = _ExactRuntimeContext(
            self.settings,
            profile,
            layer_count=len(blocks),
            local_batch_size=self.local_batch_size,
        )
        fused_attention_rmsnorm = runtime_context.fuses_attention_rmsnorm
        for block, attention in zip(blocks, attentions, strict=True):
            block.attention = ExactLowpFA4Attention(
                attention,
                runtime_context,
                profile,
                attention_norm_owner=(
                    block.attention_norm if fused_attention_rmsnorm else None
                ),
            )
            if fused_attention_rmsnorm:
                _install_fused_attention_rmsnorm_block_forward(block, profile)
        _install_root_parameter_contract(model, profile, runtime_context)
        logger.info(
            "Converted %d Llama attention layers to lazy exact FA4 profile=%s",
            len(blocks),
            profile.name,
        )

    def post_optimizer_hook(self, model: Union[nn.Module, List[nn.Module]]) -> None:
        pass


def _validate_bf16_topology_converter_contract(
    job_config: JobConfig,
    parallel_dims: ParallelDims,
) -> None:
    if not job_config.fa4.enabled:
        raise ValueError("fa4.enabled must be true for the BF16 FA4 comparator")
    if job_config.fa4.mode != "softmax":
        raise ValueError("BF16 FA4 comparator requires fa4.mode='softmax'")
    if job_config.model.name != "llama3_gc":
        raise ValueError(
            "BF16 FA4 comparator currently supports model.name='llama3_gc' only"
        )
    profile = _validate_job_rope_contract(job_config)
    if int(job_config.training.seq_len) != _EXACT_SEQUENCE:
        raise ValueError("BF16 FA4 comparator requires training.seq_len=4096")
    _validated_local_batch_size(
        profile,
        job_config.training.local_batch_size,
        label="BF16 FA4 comparator",
        allow_current_d128=True,
    )
    if job_config.training.dtype != "bfloat16":
        raise ValueError("BF16 FA4 comparator requires training.dtype='bfloat16'")
    if bool(getattr(job_config.training, "enable_fp32_master_params", False)):
        raise ValueError("BF16 FA4 comparator requires BF16 parameter storage")

    converters = tuple(job_config.model.converters or ())
    allowed_converters = (
        (
            "bfloat16",
            _BF16_TOPOLOGY_CONVERTER,
            "fa4_attention",
        ),
        (
            "bfloat16",
            "fuse_mlp_linear",
            "spline_mlp",
            _BF16_TOPOLOGY_CONVERTER,
            "fa4_attention",
        ),
    )
    if converters not in allowed_converters:
        raise ValueError(
            "BF16 FA4 comparator requires model.converters exactly equal "
            "to one authenticated chain: "
            f"{[list(chain) for chain in allowed_converters]!r}; got "
            f"{list(converters)!r}"
        )
    if any(converter in converters for converter in _EXACT_CONVERTERS):
        raise ValueError("BF16 and exact low-precision FA4 are mutually exclusive")

    if bool(getattr(job_config.training, "compile", False)):
        raise ValueError("BF16 FA4 comparator has not been authenticated with compile")
    if bool(getattr(job_config.training, "enable_cce", False)) or bool(
        getattr(getattr(job_config, "fp4_cce", None), "enabled", False)
    ):
        raise ValueError(
            "BF16 FA4 comparator requires the ordinary output head and "
            "external loss; CCE must be disabled"
        )
    compile_config = getattr(job_config, "compile", None)
    if compile_config is not None and bool(getattr(compile_config, "enable", False)):
        components = tuple(getattr(compile_config, "components", ()))
        backend = getattr(compile_config, "backend", None)
        if components != ("loss",) or backend != "inductor":
            raise ValueError(
                "BF16 FA4 comparator torch.compile is authenticated only "
                "for the ordinary loss component with the inductor backend"
            )
    activation_checkpoint = getattr(job_config, "activation_checkpoint", None)
    if (
        activation_checkpoint is not None
        and getattr(
            activation_checkpoint,
            "mode",
            "none",
        )
        != "none"
    ):
        raise ValueError(
            "BF16 FA4 comparator activation checkpointing is not authenticated"
        )

    required_one = {
        "dp_shard": int(parallel_dims.dp_shard),
        "tp": int(parallel_dims.tp),
        "pp": int(parallel_dims.pp),
        "cp": int(parallel_dims.cp),
    }
    invalid = {name: value for name, value in required_one.items() if value != 1}
    if invalid:
        raise ValueError(
            "BF16 FA4 comparator supports replicated DDP only; invalid "
            f"parallel dims {invalid}"
        )
    if int(parallel_dims.dp_replicate) < 1:
        raise ValueError("BF16 FA4 comparator requires positive DDP replication")


class ExactBF16FA4TopologyConverter(ModelConverter):
    """Authenticate exact Llama topology while leaving BF16 FA4 independent."""

    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        _validate_bf16_topology_converter_contract(job_config, parallel_dims)
        self.job_profile = _profile_from_job_flavor(job_config)

    def convert(self, model: nn.Module) -> None:
        layers = getattr(model, "layers", None)
        if not isinstance(layers, (nn.ModuleDict, nn.ModuleList)):
            raise TypeError("BF16 FA4 comparator requires TorchTitan Llama layers")
        blocks = (
            list(layers.values()) if isinstance(layers, nn.ModuleDict) else list(layers)
        )
        if not blocks:
            raise ValueError("BF16 FA4 comparator cannot validate an empty model")
        attentions = [getattr(block, "attention", None) for block in blocks]
        if any(attention is None for attention in attentions):
            raise TypeError("every Llama block must expose an attention module")
        if any(
            isinstance(attention, ExactLowpFA4Attention) for attention in attentions
        ):
            raise RuntimeError(
                "BF16 FA4 comparator refuses low-precision attention adapters"
            )

        profiles = [_select_shape_profile(attention) for attention in attentions]
        profile = profiles[0]
        if profile.contract_name != self.job_profile.contract_name:
            raise ValueError(
                "BF16 FA4 model flavor and attention shape profiles disagree: "
                f"{self.job_profile.contract_name} != {profile.contract_name}"
            )
        if any(candidate != profile for candidate in profiles[1:]):
            raise ValueError("BF16 FA4 does not support heterogeneous attention shapes")
        if len(blocks) != profile.layers:
            raise ValueError(
                f"BF16 FA4 {profile.contract_name} requires {profile.layers} "
                f"layers, got {len(blocks)}"
            )
        _validate_model_rope_contract(model, profile)

        for attention in attentions:
            _install_bf16_native_gqa_forward(attention, profile)

        # The separately registered fa4_attention converter replaces only the
        # inner attention implementation.  This converter owns the outer
        # q32/kv8 projection/RoPE topology so TorchTitan cannot repeat KV heads.
        _install_bf16_root_parameter_contract(model, profile)
        logger.info(
            "Validated BF16 FA4 Llama topology profile=%s with native GQA",
            profile.name,
        )

    def post_optimizer_hook(self, model: Union[nn.Module, List[nn.Module]]) -> None:
        pass


register_model_converter(
    ExactLowpFA4AttentionConverter,
    _EXACT_CONVERTER,
)
register_model_converter(
    ExactLowpFA4AttentionConverter,
    _LEGACY_EXACT_CONVERTER,
)
register_model_converter(
    ExactBF16FA4TopologyConverter,
    _BF16_TOPOLOGY_CONVERTER,
)


__all__ = [
    "ExactBF16FA4TopologyConverter",
    "ExactLowpFA4Attention",
    "ExactLowpFA4AttentionConverter",
]
