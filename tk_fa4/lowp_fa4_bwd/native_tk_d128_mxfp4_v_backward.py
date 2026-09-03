"""Fail-closed runtimes for experimental D128 MXFP4-V TK backward.

Q, K, and dO are contiguous BSHD E4M3 encodings of ``4 * x``.  V is the
projection producer's row-major packed E2M1 publication plus its physical
E8M0 scale pages.  The native kernel collapses each row's four D32 scales to
one maximum scale, requantizes E2M1 by the exact power-of-two ratio, and uses
an unscaled E2M1 x E4M3 dP MMA.  This adapter deliberately accepts only the
authenticated B2/S4096/Hq32/Hkv8/D128 experiment.

``NativeTkD128SharedTileProducerV503Backward`` is the explicit composed path:
it requires the producer that quantizes one D32xS32 tile once and publishes
the exact code matrix in both physical orientations.  Shape-compatible legacy
rowwise tensors are not accepted by that adapter, and its extension receipt is
still exactly v503 rather than the slower v502/v507 block-scale consumers.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any
from weakref import WeakValueDictionary

import torch

from tk_fa4.lowp_fa4_bwd.native_tk_d128_backward import (
    HEAD_DIM,
    KV_HEADS,
    Q_HEADS,
    SEQUENCE,
    SOFTMAX_SCALE,
    _require_e4m3_bshd,
)


BATCH = 2
BACKEND = "native_tk_d128_rowscale_mxfp4_v"
V503_SOURCE_IDENTITY = (
    "v503_d128_gqa_mxfp4v_rowscale_e4m3do_b2_s4096_"
    "owner4_experimental_bshd_v1"
)
SOURCE_SUFFIX = (
    "/v503_d128_gqa_mxfp4v_rowscale_e4m3do_b2_s4096_"
    "owner4_experimental_bshd.cu"
)

SHARED_TILE_V503_BACKEND = (
    "native_tk_d128_shared_tile_mxfp4_v_v503_consumer"
)
SHARED_TILE_PRODUCER_ABI_SYMBOL = (
    "project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered"
)
SHARED_TILE_PRODUCER_CHECKED_SYMBOL = (
    SHARED_TILE_PRODUCER_ABI_SYMBOL
    + "_shared_tile_mx_backward_v_mx_forward_out"
)
SHARED_TILE_PRODUCER_UNCHECKED_SYMBOL = (
    SHARED_TILE_PRODUCER_CHECKED_SYMBOL + "_unchecked"
)
SHARED_TILE_PRODUCER_SEMANTICS = (
    "single_quantized_d32xs32_mxfp4_v_with_projection_accumulator_e4m3_qk"
)
SHARED_TILE_PRODUCER_CONTRACT_SCHEMA = (
    "tkfa4.d128_mxfp4_v.shared_tile_producer.v1"
)
SHARED_TILE_V503_COMPOSITION_SCHEMA = (
    "tkfa4.d128_mxfp4_v.shared_tile_v503_composition.v1"
)

# These are observable attributes of the shape-bound projection object.  Keep
# exact type checks below: ``1`` must not authenticate a boolean selector.
EXPECTED_SHARED_TILE_PRODUCER_ATTRIBUTES = MappingProxyType(
    {
        "batch": BATCH,
        "seqlen": SEQUENCE,
        "hidden": 4096,
        "q_heads": Q_HEADS,
        "kv_heads": KV_HEADS,
        "publish_mxfp4_v": True,
        "v_mxfp4_scale_2d": True,
        "per_block_qk_scales": True,
        "experimental": True,
        "experimental_output_shared_dual_v": False,
        "experimental_rowwise_mx_backward_v": False,
        "experimental_shared_tile_mx_backward_v": True,
        "experimental_mx_backward_v": True,
        "requires_forward_workspace": True,
        "requires_v_mxfp4_scales_out": False,
        "abi_validation_symbol": SHARED_TILE_PRODUCER_ABI_SYMBOL,
        "checked_symbol": SHARED_TILE_PRODUCER_CHECKED_SYMBOL,
        "unchecked_symbol": SHARED_TILE_PRODUCER_UNCHECKED_SYMBOL,
        "symbol": SHARED_TILE_PRODUCER_UNCHECKED_SYMBOL,
        "output_shared_dual_v_path": "shared_tile_mx_backward_v",
        "projection_forward_publication_path": (
            "caller_owned_shared_tile_mx_backward_v_d128"
        ),
        "backward_publication_semantics": SHARED_TILE_PRODUCER_SEMANTICS,
    }
)

# Keep this contract separate from v501's production metadata.  In
# particular, accepting v503 must never broaden NativeTkD128E4M3Backward.
EXPECTED_EXTENSION_METADATA = {
    "schema": "tkfa4.native_tk_d128_backward.experimental.v2",
    "backend": "thunderkittens_sm100a",
    "source_identity": V503_SOURCE_IDENTITY,
    "experimental": True,
    "production_data_abi_compatible": False,
    "fail_closed": "B2_S4096_D128_GQA_only",
    "batch_values": (BATCH,),
    "sequence": SEQUENCE,
    "causal": True,
    "threads": 512,
    "query_heads": Q_HEADS,
    "kv_heads": KV_HEADS,
    "head_dim": HEAD_DIM,
    "q_k_dout_dtype": "float8_e4m3fn_x4_encoding",
    "q_k_dout_layout": "BSHD_contiguous",
    "v_dtype": "packed_mxfp4_e2m1_uint8",
    "v_layout": "B,S,Hkv,D_over_2_row_major",
    "v_shape": "[2,4096,8,64]",
    "v_scale_dtype": "e8m0_bytes",
    "v_scale_layout": "B,S_over_128,Hkv,physical_32x16_page",
    "v_scale_shape": "[2,32,8,512]",
    "requires_second_row_major_v_orientation": True,
    "eliminates_backward_e4m3_v_publication": True,
    "eliminates_all_duplicate_v_publication": False,
    "v_row_scale_reduction": "common_e8m0=max(four_D32_codes)",
    "v_row_requantization": (
        "deterministic_E2M1_RNE_LUT_exact_power_of_two_ratio"
    ),
    "dp_opcode": "tcgen05_mma_cta_group_1_kind_f8f6f4_unscaled",
    "dp_instruction_descriptor": "0x08200290",
    "dp_a_format": "E2M1_encoding_5",
    "dp_b_format": "E4M3_encoding_0",
    "dp_b_major": "K_major_for_ABt",
    "dp_b_descriptor_chunk_stride": 1,
    "dp_scale_format": "none_in_mma",
    "dp_reduction_chunk": 32,
    "dp_row_factor": "(2/3)*2^(common_e8m0-127)",
    "dp_row_factor_application": (
        "fmaf(raw_dp,row_factor,dstat_x16)_before_probability_and_beta"
    ),
    "dstat_abi": "-16*sum(O*dO)",
    "lstat_abi": "8-LSE*log2(e)",
    "public_softmax_scale": "natural",
    "internal_beta_divisor": 16.0,
    "gradient_epilogue_scale": 1.0 / 256.0,
    "output_dtype": "bfloat16",
    "output_layout": "BSHD_contiguous",
    "caller_owned_output_api": True,
    "backward_out_clears_outputs": True,
    "backward_out_physical_clear_policy": (
        "memset_dq_only_unique_direct_overwrite_dk_dv"
    ),
    "dv_route": "unchanged_v490_probability_times_e4m3_dout",
    "scale_tmem": False,
    "score_tmem_alias": False,
    "tensor_issue_schedule": (
        "exact_v490_score_dp_overlap_except_mixed_unscaled_dp_opcode"
    ),
}


def require_shared_tile_v503_producer_contract(
    producer: Any,
) -> dict[str, Any]:
    """Authenticate the static shared-D32xS32 projection route.

    This deliberately does not claim that a workspace has completed first-use
    byte authentication.  ``bind_inputs`` performs that dynamic check against
    the exact workspace whose Q/K/V views are being bound.
    """
    # Import locally to avoid coupling ordinary legacy-v503 module import to
    # projection extension initialization.  Exact type/callable checks prevent
    # a mutable duck-typed receipt from authenticating this composed route.
    import tk_fa4.interface as interface

    if type(producer) is not interface.B300BoundD128NVFP4QKVProjection:
        raise TypeError(
            "shared D32xS32 MXFP4-V producer must be exactly "
            "B300BoundD128NVFP4QKVProjection"
        )
    missing = [
        field
        for field in EXPECTED_SHARED_TILE_PRODUCER_ATTRIBUTES
        if not hasattr(producer, field)
    ]
    mismatches = {
        field: {"actual": getattr(producer, field), "expected": expected}
        for field, expected in EXPECTED_SHARED_TILE_PRODUCER_ATTRIBUTES.items()
        if hasattr(producer, field)
        and (
            getattr(producer, field) != expected
            or type(getattr(producer, field)) is not type(expected)
        )
    }
    if missing or mismatches:
        raise RuntimeError(
            "shared D32xS32 MXFP4-V producer contract mismatch: "
            f"missing={missing}, mismatches={mismatches}"
        )
    extension = getattr(interface, "_C_b300_lowp_bwd", None)
    checked_callable = getattr(
        extension, SHARED_TILE_PRODUCER_CHECKED_SYMBOL, None
    )
    unchecked_callable = getattr(
        extension, SHARED_TILE_PRODUCER_UNCHECKED_SYMBOL, None
    )
    registry = getattr(producer, "_validated_forward_workspaces", None)
    if (
        not callable(checked_callable)
        or not callable(unchecked_callable)
        or getattr(producer, "_project_checked", None) is not checked_callable
        or getattr(producer, "_project_unchecked", None)
        is not unchecked_callable
        or type(registry) is not WeakValueDictionary
    ):
        raise RuntimeError(
            "shared D32xS32 MXFP4-V producer is not bound to the exact "
            "checked/unchecked extension callables and weak workspace registry"
        )
    return {
        "schema": SHARED_TILE_PRODUCER_CONTRACT_SCHEMA,
        "shape": "B2_S4096_H4096_Hq32_Hkv8_D128",
        "checked_symbol": SHARED_TILE_PRODUCER_CHECKED_SYMBOL,
        "unchecked_symbol": SHARED_TILE_PRODUCER_UNCHECKED_SYMBOL,
        "publication_path": (
            "caller_owned_shared_tile_mx_backward_v_d128"
        ),
        "producer_semantics": SHARED_TILE_PRODUCER_SEMANTICS,
        "shared_tile_shape": "D32xS32",
        "forward_backward_code_matrix": "bitwise_identical",
        "backward_payload_layout": "B,S,Hkv,D_over_2_row_major",
        "backward_scale_layout": (
            "B,S_over_128,Hkv,physical_32x16_page"
        ),
        "anchor_semantics": (
            "four_independent_D32_anchors_per_D128_row"
        ),
        "producer_quantization_passes_per_tile": 1,
    }


def _require_extension_metadata(extension: Any) -> dict[str, Any]:
    metadata_fn = getattr(extension, "native_tk_d128_backward_metadata", None)
    if not callable(metadata_fn):
        raise RuntimeError(
            "native TK D128 MXFP4-V extension lacks "
            "native_tk_d128_backward_metadata"
        )
    metadata = dict(metadata_fn())
    source_identity = metadata.get("source_identity")
    if isinstance(source_identity, str) and source_identity.startswith(
        ("v502_", "v507_")
    ):
        raise RuntimeError(
            "the v503 row-scale consumer adapter refuses v502/v507 "
            f"block-scale artifacts: {source_identity!r}"
        )
    missing = {*EXPECTED_EXTENSION_METADATA, "source_file"} - set(metadata)
    if missing:
        raise RuntimeError(
            "native TK D128 MXFP4-V extension metadata is incomplete: "
            f"missing {sorted(missing)}"
        )
    mismatches = {
        field: {"actual": metadata[field], "expected": expected}
        for field, expected in EXPECTED_EXTENSION_METADATA.items()
        if metadata[field] != expected
        or type(metadata[field]) is not type(expected)
    }
    source_file = metadata["source_file"]
    normalized_source = (
        source_file.replace("\\", "/")
        if isinstance(source_file, str)
        else ""
    )
    if not (
        normalized_source == SOURCE_SUFFIX.removeprefix("/")
        or normalized_source.endswith(SOURCE_SUFFIX)
    ):
        mismatches["source_file"] = {
            "actual": source_file,
            "expected_suffix": SOURCE_SUFFIX,
        }
    if mismatches:
        raise RuntimeError(
            "native TK D128 MXFP4-V extension does not match the "
            f"experimental ABI: {mismatches}"
        )
    return metadata


def _require_uint8_cuda(
    tensor: torch.Tensor,
    *,
    name: str,
    shape: tuple[int, ...],
    device: torch.device,
) -> None:
    if (
        tensor.dtype != torch.uint8
        or not tensor.is_cuda
        or not tensor.is_contiguous()
        or tuple(tensor.shape) != shape
        or tensor.device != device
    ):
        raise ValueError(
            f"{name} must be contiguous CUDA uint8 {shape} on {device}; "
            f"got {tensor.dtype} {tuple(tensor.shape)} on {tensor.device}"
        )


def _same_tensor_view(actual: Any, expected: Any) -> bool:
    if actual is expected:
        return True
    actual_data_ptr = getattr(actual, "data_ptr", None)
    expected_data_ptr = getattr(expected, "data_ptr", None)
    if not callable(actual_data_ptr) or not callable(expected_data_ptr):
        return False
    try:
        return bool(
            int(actual_data_ptr()) == int(expected_data_ptr())
            and tuple(actual.shape) == tuple(expected.shape)
            and actual.dtype == expected.dtype
            and actual.device == expected.device
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _require_validated_shared_tile_publication(
    producer: Any,
    producer_workspace: Any,
    *,
    q: Any,
    k: Any,
    v: Any,
    v_scales: Any,
) -> dict[str, Any]:
    """Bind views only from one first-use-authenticated shared workspace."""
    producer_contract = require_shared_tile_v503_producer_contract(producer)
    abi_validated = getattr(
        producer, "forward_workspace_abi_validated", None
    )
    validated_count = getattr(
        producer, "validated_forward_workspace_count", None
    )
    validated_workspaces = getattr(
        producer, "_validated_forward_workspaces", None
    )
    workspace_authenticated = bool(
        type(validated_workspaces) is WeakValueDictionary
        and validated_workspaces.get(id(producer_workspace))
        is producer_workspace
    )
    if (
        type(abi_validated) is not bool
        or not abi_validated
        or type(validated_count) is not int
        or validated_count <= 0
        or not workspace_authenticated
    ):
        raise RuntimeError(
            "shared D32xS32 MXFP4-V workspace has not completed first-use "
            "producer authentication"
        )

    expected_views = {
        "q": getattr(producer_workspace, "q_backward_fp8", None),
        "k": getattr(producer_workspace, "k_backward_fp8", None),
        "v": getattr(producer_workspace, "v_backward_mxfp4", None),
        "v_scales": getattr(
            producer_workspace, "v_backward_mxfp4_scale_pages", None
        ),
    }
    actual_views = {"q": q, "k": k, "v": v, "v_scales": v_scales}
    mismatches = [
        name
        for name, actual in actual_views.items()
        if expected_views[name] is None
        or not _same_tensor_view(actual, expected_views[name])
    ]
    if mismatches:
        raise RuntimeError(
            "shared D32xS32 MXFP4-V binding is not backed by the "
            "authenticated producer workspace: "
            + ", ".join(mismatches)
        )
    return {
        "schema": SHARED_TILE_V503_COMPOSITION_SCHEMA,
        "producer_contract_schema": producer_contract["schema"],
        "producer_checked_symbol": producer_contract["checked_symbol"],
        "producer_workspace_abi_validated": True,
        "producer_workspace_strongly_retained": True,
        "tensor_binding": "exact_data_ptr_shape_dtype_device",
        "producer_semantics": SHARED_TILE_PRODUCER_SEMANTICS,
        "consumer_source_identity": V503_SOURCE_IDENTITY,
        "consumer_transform": (
            "common_e8m0_max_then_exact_power_of_two_e2m1_requantization"
        ),
        "consumer_dp_opcode": (
            "tcgen05_mma_cta_group_1_kind_f8f6f4_unscaled"
        ),
    }


class NativeTkD128Mxfp4VBackward:
    """Preallocated B2/S4096 adapter for the experimental native v503 ABI."""

    backend = BACKEND
    detached_fp8_p_tmem = False
    head_fast_raster = False
    exp2_degree = 0
    exp2_period = 0
    exp2_policy = {
        "backend": backend,
        "implementation": "native_ex2",
        "polynomial_degree": 0,
        "selective_period": 0,
    }

    def __init__(
        self,
        extension: Any,
        *,
        batch: int,
        device: torch.device | str,
    ) -> None:
        if type(batch) is not int or batch != BATCH:
            raise ValueError(
                "native TK D128 MXFP4-V backward requires batch 2"
            )
        self.extension = extension
        self.extension_metadata = _require_extension_metadata(extension)
        self.compiled_out = getattr(
            extension,
            "backward_mxfp4v_e4m3do_bshd_precomputed_out",
            None,
        )
        if not callable(self.compiled_out):
            raise RuntimeError(
                "native TK D128 MXFP4-V extension lacks "
                "backward_mxfp4v_e4m3do_bshd_precomputed_out"
            )
        self.compiled_main = getattr(
            extension,
            "main_mxfp4v_e4m3do_bshd_precomputed",
            None,
        )
        if not callable(self.compiled_main):
            raise RuntimeError(
                "native TK D128 MXFP4-V extension lacks "
                "main_mxfp4v_e4m3do_bshd_precomputed"
            )
        self.compiled = self.compiled_out
        self.loaded_artifact_identity = dict(
            getattr(extension, "_tk_fa4_loaded_artifact_identity", {})
        )
        self.batch = batch
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError(
                "native TK D128 MXFP4-V backward requires a CUDA device"
            )
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())

        q_shape = (BATCH, SEQUENCE, Q_HEADS, HEAD_DIM)
        kv_shape = (BATCH, SEQUENCE, KV_HEADS, HEAD_DIM)
        stats_numel = BATCH * Q_HEADS * SEQUENCE
        self.workspace_torch = torch.empty(
            2 * stats_numel * torch.float32.itemsize,
            device=self.device,
            dtype=torch.uint8,
        )
        stats = self.workspace_torch.view(torch.float32)
        self.dstat = stats[:stats_numel].view(BATCH, Q_HEADS, 1, SEQUENCE)
        self.lstat = stats[stats_numel:].view(BATCH, Q_HEADS, 1, SEQUENCE)
        self.dq = torch.empty(q_shape, device=self.device, dtype=torch.bfloat16)
        self.dk = torch.empty(kv_shape, device=self.device, dtype=torch.bfloat16)
        self.dv = torch.empty_like(self.dk)
        unused_partial_sentinel = torch.empty(
            0,
            device=self.device,
            dtype=torch.float32,
        )
        self.dk_partials = unused_partial_sentinel
        self.dv_partials = unused_partial_sentinel
        self.direct_tma_dkdv = True
        self.raster_policy = {
            "backend": self.backend,
            "owner_order": "key_tile_major_head_owner",
            "host_dispatch_per_launch": False,
            "heads_per_owner": 4,
        }
        self._q: torch.Tensor | None = None
        self._k: torch.Tensor | None = None
        self._v: torch.Tensor | None = None
        self._v_scales: torch.Tensor | None = None
        self._dout: torch.Tensor | None = None
        self._bind_generation = 0
        self._run_generation = 0

    def bind_inputs(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        v_scales: torch.Tensor,
        dout: torch.Tensor,
    ) -> None:
        for tensor, name, heads in (
            (q, "q", Q_HEADS),
            (k, "k", KV_HEADS),
            (dout, "dout", Q_HEADS),
        ):
            _require_e4m3_bshd(
                tensor,
                name=name,
                batch=BATCH,
                heads=heads,
                device=self.device,
            )
        _require_uint8_cuda(
            v,
            name="v_backward_mxfp4",
            shape=(BATCH, SEQUENCE, KV_HEADS, HEAD_DIM // 2),
            device=self.device,
        )
        _require_uint8_cuda(
            v_scales,
            name="v_backward_mxfp4_scale_pages",
            shape=(BATCH, SEQUENCE // 128, KV_HEADS, 512),
            device=self.device,
        )
        self._q = q
        self._k = k
        self._v = v
        self._v_scales = v_scales
        self._dout = dout
        self._bind_generation += 1

    def reset(self) -> None:
        """No-op: the authenticated clearing entrypoint owns logical reset."""

    def d128_mxfp4_v_operand_cache_receipt(self) -> dict[str, Any]:
        return {
            "schema": "native_tk_d128_mxfp4_v_direct_bind_v1",
            "implementation": "direct_prebound_torch_tensor_arguments",
            "host_wrapper_cache_required": False,
            "bind_generation": self._bind_generation,
        }

    def d128_mxfp4_v_compilation_receipt(self) -> dict[str, Any]:
        return {
            "schema": "native_tk_d128_mxfp4_v_extension_v1",
            "extension": dict(self.loaded_artifact_identity),
            "source_identity": self.extension_metadata["source_identity"],
            "instruction_descriptor": self.extension_metadata[
                "dp_instruction_descriptor"
            ],
        }

    def run(self, *, reset: bool) -> None:
        if type(reset) is not bool:
            raise TypeError("reset must be exactly bool")
        if any(
            operand is None
            for operand in (
                self._q,
                self._k,
                self._v,
                self._v_scales,
                self._dout,
            )
        ):
            raise RuntimeError(
                "bind_inputs() must precede native TK MXFP4-V backward"
            )
        assert self._q is not None
        assert self._k is not None
        assert self._v is not None
        assert self._v_scales is not None
        assert self._dout is not None
        compiled = self.compiled_out if reset else self.compiled_main
        compiled(
            self._q,
            self._k,
            self._v,
            self._v_scales,
            self._dout,
            self.lstat,
            self.dstat,
            self.dq,
            self.dk,
            self.dv,
            SOFTMAX_SCALE,
        )
        self._run_generation += 1

    def contract(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "extension": dict(self.loaded_artifact_identity),
            "extension_metadata": dict(self.extension_metadata),
            "shape": {
                "batch": BATCH,
                "sequence": SEQUENCE,
                "q_heads": Q_HEADS,
                "kv_heads": KV_HEADS,
                "head_dim": HEAD_DIM,
            },
            "input": {
                "dtype": "mixed_e4m3fn_and_packed_mxfp4_e8m0",
                "layout": "BSHD_contiguous_with_physical_scale_pages",
                "q_k_dout": {
                    "dtype": "torch.float8_e4m3fn",
                    "layout": "BSHD_contiguous",
                    "encoding_scale": 4.0,
                },
                "v": {
                    "payload_dtype": "torch.uint8_packed_e2m1",
                    "payload_layout": "B,S,Hkv,D_over_2_contiguous",
                    "scale_dtype": "torch.uint8_e8m0",
                    "scale_layout": "B,S_over_128,Hkv,physical_32x16_page",
                    "row_scale_reduction": "max_four_D32_codes",
                },
            },
            "statistics": {
                "workspace_page_0": "-16_sum_O_dO",
                "workspace_page_1": "8_minus_LSE_log2e",
                "producer_native": True,
            },
            "output": {
                "dtype": "torch.bfloat16",
                "layout": "BSHD_contiguous",
                "encoding_scale": 4.0,
                "logical_reset_per_run": True,
            },
            "schedule": {
                "dispatch": "B2_S4096_v503_only",
                "owner_order": "key_tile_major_head_owner",
                "direct_dkdv_unique_writer": True,
                "gradient_publisher": "dedicated_warp14",
                "dv_route": "unchanged_v490",
            },
            "publication": {
                "backward_e4m3_v_required": False,
                "backward_row_major_mxfp4_v_required": True,
                "forward_feature_major_mxfp4_v_still_required": True,
            },
            "allocation": {
                "scope": "native_backward_runner_only",
                "caller_owned_runner_storage": True,
                "native_run_allocations": False,
                "native_run_dlpack_wrappers": False,
                "external_projection_publication": (
                    "authenticated_rowwise_mxfp4_v_and_e8m0_pages"
                ),
            },
        }


class NativeTkD128SharedTileProducerV503Backward(
    NativeTkD128Mxfp4VBackward
):
    """Explicit shared-D32xS32 producer plus v503 consumer composition.

    Construction authenticates the bound projection route.  Binding then
    requires the exact workspace recorded by that producer after its checked
    first call, so the shape-compatible legacy rowwise publication cannot be
    substituted.  The inherited extension check remains exact to v503.
    """

    backend = SHARED_TILE_V503_BACKEND
    exp2_policy = {
        "backend": backend,
        "implementation": "native_ex2",
        "polynomial_degree": 0,
        "selective_period": 0,
    }

    def __init__(
        self,
        extension: Any,
        *,
        producer: Any,
        batch: int,
        device: torch.device | str,
    ) -> None:
        producer_contract = require_shared_tile_v503_producer_contract(
            producer
        )
        super().__init__(extension, batch=batch, device=device)
        self._shared_tile_producer = producer
        self._shared_tile_producer_contract = producer_contract
        self._shared_tile_producer_workspace: Any | None = None
        self._shared_tile_publication_receipt: dict[str, Any] | None = None

    @property
    def shared_tile_producer_contract(self) -> dict[str, Any]:
        return dict(self._shared_tile_producer_contract)

    def bind_inputs(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        v_scales: torch.Tensor,
        dout: torch.Tensor,
        *,
        producer_workspace: Any,
    ) -> None:
        receipt = _require_validated_shared_tile_publication(
            self._shared_tile_producer,
            producer_workspace,
            q=q,
            k=k,
            v=v,
            v_scales=v_scales,
        )
        super().bind_inputs(q, k, v, v_scales, dout)
        # The producer registry intentionally owns weak values.  Retain the
        # authenticated workspace for exactly as long as these bound tensor
        # views may be executed, replacing it only after every check succeeds.
        self._shared_tile_producer_workspace = producer_workspace
        self._shared_tile_publication_receipt = {
            **receipt,
            "bind_generation": self._bind_generation,
        }

    def shared_tile_v503_publication_receipt(self) -> dict[str, Any]:
        if self._shared_tile_publication_receipt is None:
            raise RuntimeError(
                "bind_inputs() with an authenticated shared-tile producer "
                "workspace must precede the composition receipt"
            )
        return dict(self._shared_tile_publication_receipt)

    def d128_mxfp4_v_compilation_receipt(self) -> dict[str, Any]:
        receipt = super().d128_mxfp4_v_compilation_receipt()
        return {
            **receipt,
            "composition_schema": SHARED_TILE_V503_COMPOSITION_SCHEMA,
            "producer_contract_schema": (
                SHARED_TILE_PRODUCER_CONTRACT_SCHEMA
            ),
            "consumer_role": "v503_commonrow_requantizing_consumer",
        }

    def contract(self) -> dict[str, Any]:
        result = super().contract()
        result["input"]["v"].update(
            {
                "producer_semantics": SHARED_TILE_PRODUCER_SEMANTICS,
                "producer_shared_tile_shape": "D32xS32",
                "producer_quantization_passes_per_tile": 1,
                "producer_anchor_count_per_d128_row": 4,
                "consumer_row_scale_reduction": "max_four_D32_codes",
                "consumer_requantization": (
                    "deterministic_E2M1_RNE_LUT_exact_power_of_two_ratio"
                ),
            }
        )
        result["schedule"].update(
            {
                "dispatch": (
                    "B2_S4096_shared_tile_producer_v503_consumer_only"
                ),
                "dp_consumer": "v503_unscaled_commonrow_requantizing",
            }
        )
        result["publication"].update(
            {
                "producer_checked_symbol": (
                    SHARED_TILE_PRODUCER_CHECKED_SYMBOL
                ),
                "producer_workspace_authentication_required": True,
                "legacy_rowwise_producer_compatible": False,
            }
        )
        result["allocation"]["external_projection_publication"] = (
            "authenticated_shared_D32xS32_single_quantization_dual_orientation"
        )
        result["composition"] = {
            "schema": SHARED_TILE_V503_COMPOSITION_SCHEMA,
            "producer": dict(self._shared_tile_producer_contract),
            "consumer_source_identity": V503_SOURCE_IDENTITY,
            "consumer_dp_opcode": EXPECTED_EXTENSION_METADATA["dp_opcode"],
            "forbidden_consumer_source_prefixes": ("v502_", "v507_"),
        }
        result["publication_binding"] = (
            {
                "bound": True,
                **self._shared_tile_publication_receipt,
            }
            if self._shared_tile_publication_receipt is not None
            else {
                "bound": False,
                "required": (
                    "first_use_authenticated_shared_tile_producer_workspace"
                ),
            }
        )
        return result
