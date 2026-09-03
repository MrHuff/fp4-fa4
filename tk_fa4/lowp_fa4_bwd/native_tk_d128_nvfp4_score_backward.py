"""Fail-closed B1/S4096 adapter for the experimental v508 TK backward.

The score/probability path consumes the exact NVFP4 Q/K payload, row-K16
scale pages, and per-head global scales published by the D128 QKV projection.
The gradient MMAs deliberately retain the projection's represented E4M3 Q/K;
V and dO are represented E4M3 as well.  The native kernel publishes additive
BF16 gradients, so this adapter always calls its clearing ``*_out`` wrapper.

Binding retains the exact :class:`B300E4M3QKVForwardWorkspace` and its tensor
views.  It performs all metadata, shape, dtype, device, contiguity, and storage
identity checks before changing the bound state; ``run`` then performs no
allocation, dtype view construction, or workspace attribute lookup.
"""

from __future__ import annotations

from typing import Any

import torch

from tk_fa4.lowp_fa4_bwd.native_tk_d128_backward import (
    HEAD_DIM,
    KV_HEADS,
    Q_HEADS,
    SEQUENCE,
    SOFTMAX_SCALE,
    _require_e4m3_bshd,
)


BATCH = 1
BACKEND = "native_tk_d128_nvfp4_score_e4m3_gradient"
V508_SOURCE_IDENTITY = (
    "v508_native_nvfp4_score_e4m3_gradient_b1_s4096_experimental_v1"
)
SOURCE_SUFFIX = (
    "/v508_d128_gqa_nvfp4_score_e4m3_gradient_b1_exact_s4096_"
    "experimental_bshd.cu"
)
OUT_ENTRYPOINT = (
    "backward_nvfp4_score_e4m3_gradient_bshd_precomputed_out"
)
MAIN_ENTRYPOINT = "main_nvfp4_score_e4m3_gradient_bshd_precomputed"

# Keep this receipt byte-for-byte separate from the production v501 adapter.
# The shared schema is emitted by the extension, while the identity and
# fail-closed dispatch authenticate the experimental candidate specifically.
EXPECTED_EXTENSION_METADATA = {
    "schema": "tkfa4.native_tk_d128_backward.v1",
    "backend": "thunderkittens_sm100a",
    "source_identity": V508_SOURCE_IDENTITY,
    "experimental": True,
    "production_dispatch_connected": False,
    "dispatch": "fail_closed_B1_S4096_only_no_fallback",
    "selected_kernel": (
        "v508::b1_native_nvfp4_score_e4m3_gradient_exact_s4096_kernel"
    ),
    "score_qk_dtype": "float4_e2m1fn_x2",
    "score_qk_layout": "BHSD_packed",
    "score_scale_dtype": "float8_e4m3fn",
    "score_scale_layout": (
        "forward_row_K16_pages_Q_B_S128_Hx2_512_K_B_S64_Hkvx2_512"
    ),
    "score_global_scale": "per_head_q_times_k",
    "score_mma": "two_K64_mxf4nvf4_block_scale_scale_vec_4X",
    "gradient_qk_dtype": "float8_e4m3fn_represented_x4",
    "v_dout_dtype": "float8_e4m3fn_represented_x4",
    "score_internal_beta_divisor": 1.0,
    "ds_internal_beta_divisor": 16.0,
    "lstat_abi": "8-LSE*log2(e)",
    "dstat_abi": "-16*sum(O*dO)",
    "output_dtype": "bfloat16_additive",
    "batch": BATCH,
    "sequence": SEQUENCE,
    "query_heads": Q_HEADS,
    "kv_heads": KV_HEADS,
    "head_dim": HEAD_DIM,
    "threads": 512,
    "user_shared_storage_bytes": 193536,
    "score_scale_tmem_alias": "dP_dQ_columns_0_15",
    "score_schedule": (
        "wait_dq_tmem_drained_then_native_score_wait_complete_then_dense_dp"
    ),
    "caller_owned_output_api": True,
    "main_requires_precleared_outputs": True,
    "backward_out_clears_dq_dk_dv": True,
}


def _require_extension_metadata(extension: Any) -> dict[str, Any]:
    metadata_fn = getattr(extension, "native_tk_d128_backward_metadata", None)
    if not callable(metadata_fn):
        raise RuntimeError(
            "native TK D128 v508 extension lacks "
            "native_tk_d128_backward_metadata"
        )
    metadata = dict(metadata_fn())
    missing = {*EXPECTED_EXTENSION_METADATA, "source_file"} - set(metadata)
    if missing:
        raise RuntimeError(
            "native TK D128 v508 extension metadata is incomplete: "
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
            "native TK D128 v508 extension does not match the experimental "
            f"ABI: {mismatches}"
        )
    return metadata


def _same_tensor_view(actual: Any, expected: Any) -> bool:
    """Check exact storage/view ABI without accepting shape-only tensors."""
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


def _same_storage_alias(actual: Any, expected: Any) -> bool:
    """Authenticate a dtype-reinterpreted alias of the same byte layout."""
    actual_data_ptr = getattr(actual, "data_ptr", None)
    expected_data_ptr = getattr(expected, "data_ptr", None)
    if not callable(actual_data_ptr) or not callable(expected_data_ptr):
        return False
    try:
        return bool(
            int(actual_data_ptr()) == int(expected_data_ptr())
            and tuple(actual.shape) == tuple(expected.shape)
            and actual.device == expected.device
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _require_cuda_tensor(
    tensor: Any,
    *,
    name: str,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    device: torch.device,
) -> None:
    if (
        getattr(tensor, "dtype", None) != dtype
        or not bool(getattr(tensor, "is_cuda", False))
        or not callable(getattr(tensor, "is_contiguous", None))
        or not tensor.is_contiguous()
        or tuple(getattr(tensor, "shape", ())) != shape
        or getattr(tensor, "device", None) != device
    ):
        raise ValueError(
            f"{name} must be contiguous CUDA {dtype} {shape} on {device}"
        )


def _require_native_score_workspace(
    workspace: Any,
    *,
    q: Any,
    k: Any,
    v: Any,
    device: torch.device,
) -> tuple[Any, ...]:
    # Local import prevents the ordinary adapter import from initializing the
    # projection extension and also makes the exact-class check unambiguous.
    import tk_fa4.interface as interface

    if type(workspace) is not interface.B300E4M3QKVForwardWorkspace:
        raise TypeError(
            "native-score v508 requires exactly "
            "B300E4M3QKVForwardWorkspace"
        )

    q_native = workspace.q_payload_fp4
    k_native = workspace.k_payload_fp4
    q_scale_pages = workspace.q_scale_pages
    k_scale_pages = workspace.k_scale_pages
    q_global_scale = workspace.q_global_scale
    k_global_scale = workspace.k_global_scale

    for tensor, name, dtype, shape in (
        (
            workspace.q_payload,
            "workspace.q_payload",
            torch.uint8,
            (BATCH, Q_HEADS, SEQUENCE, HEAD_DIM // 2),
        ),
        (
            workspace.k_payload,
            "workspace.k_payload",
            torch.uint8,
            (BATCH, KV_HEADS, SEQUENCE, HEAD_DIM // 2),
        ),
        (
            q_native,
            "workspace.q_payload_fp4",
            torch.float4_e2m1fn_x2,
            (BATCH, Q_HEADS, SEQUENCE, HEAD_DIM // 2),
        ),
        (
            k_native,
            "workspace.k_payload_fp4",
            torch.float4_e2m1fn_x2,
            (BATCH, KV_HEADS, SEQUENCE, HEAD_DIM // 2),
        ),
        (
            q_scale_pages,
            "workspace.q_scale_pages",
            torch.float8_e4m3fn,
            (BATCH, SEQUENCE // 128, Q_HEADS * 2, 512),
        ),
        (
            k_scale_pages,
            "workspace.k_scale_pages",
            torch.float8_e4m3fn,
            (BATCH, SEQUENCE // 64, KV_HEADS * 2, 512),
        ),
        (
            q_global_scale,
            "workspace.q_global_scale",
            torch.float32,
            (BATCH, Q_HEADS),
        ),
        (
            k_global_scale,
            "workspace.k_global_scale",
            torch.float32,
            (BATCH, KV_HEADS),
        ),
    ):
        _require_cuda_tensor(
            tensor,
            name=name,
            dtype=dtype,
            shape=shape,
            device=device,
        )

    alias_mismatches = []
    if not _same_storage_alias(q_native, workspace.q_payload):
        alias_mismatches.append("q_payload_fp4")
    if not _same_storage_alias(k_native, workspace.k_payload):
        alias_mismatches.append("k_payload_fp4")
    if alias_mismatches:
        raise RuntimeError(
            "native-score typed FP4 aliases do not share their workspace "
            "payload storage: " + ", ".join(alias_mismatches)
        )

    represented_mismatches = []
    for name, actual, expected in (
        ("q", q, workspace.q_backward_fp8),
        ("k", k, workspace.k_backward_fp8),
        ("v", v, workspace.v_backward_fp8),
    ):
        if not _same_tensor_view(actual, expected):
            represented_mismatches.append(name)
    if represented_mismatches:
        raise RuntimeError(
            "represented E4M3 gradient operands are not the exact views from "
            "the native-score forward workspace: "
            + ", ".join(represented_mismatches)
        )

    return (
        q_native,
        k_native,
        q_scale_pages,
        k_scale_pages,
        q_global_scale,
        k_global_scale,
    )


class NativeTkD128NVFP4ScoreE4M3GradientBackward:
    """Preallocated strict adapter for v508 B1/S4096 only."""

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
            raise ValueError("native-score v508 backward requires batch 1")
        self.extension = extension
        self.extension_metadata = _require_extension_metadata(extension)
        self.compiled_out = getattr(extension, OUT_ENTRYPOINT, None)
        if not callable(self.compiled_out):
            raise RuntimeError(
                f"native TK D128 v508 extension lacks {OUT_ENTRYPOINT}"
            )
        authenticated_main = getattr(extension, MAIN_ENTRYPOINT, None)
        if not callable(authenticated_main):
            raise RuntimeError(
                f"native TK D128 v508 extension lacks {MAIN_ENTRYPOINT}"
            )
        # Authenticate, but do not retain or expose, the unsafe precleared
        # entrypoint. Every run below deliberately uses ``compiled_out``.
        self.loaded_artifact_identity = dict(
            getattr(extension, "_tk_fa4_loaded_artifact_identity", {})
        )
        self.batch = batch
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("native-score v508 backward requires a CUDA device")
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
            "heads_per_owner": 2,
        }

        self._q: torch.Tensor | None = None
        self._k: torch.Tensor | None = None
        self._v: torch.Tensor | None = None
        self._dout: torch.Tensor | None = None
        self._q_native: torch.Tensor | None = None
        self._k_native: torch.Tensor | None = None
        self._q_native_scale: torch.Tensor | None = None
        self._k_native_scale: torch.Tensor | None = None
        self._q_global_scale: torch.Tensor | None = None
        self._k_global_scale: torch.Tensor | None = None
        self._native_score_workspace: Any | None = None
        self._bind_generation = 0
        self._run_generation = 0

    def bind_inputs(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dout: torch.Tensor,
        native_score_workspace: Any,
    ) -> None:
        for tensor, name, heads in (
            (q, "q", Q_HEADS),
            (k, "k", KV_HEADS),
            (v, "v", KV_HEADS),
            (dout, "dout", Q_HEADS),
        ):
            _require_e4m3_bshd(
                tensor,
                name=name,
                batch=BATCH,
                heads=heads,
                device=self.device,
            )
        native_operands = _require_native_score_workspace(
            native_score_workspace,
            q=q,
            k=k,
            v=v,
            device=self.device,
        )

        # Commit the new binding only after every validation succeeds.
        self._q = q
        self._k = k
        self._v = v
        self._dout = dout
        (
            self._q_native,
            self._k_native,
            self._q_native_scale,
            self._k_native_scale,
            self._q_global_scale,
            self._k_global_scale,
        ) = native_operands
        self._native_score_workspace = native_score_workspace
        self._bind_generation += 1

    def reset(self) -> None:
        """No-op: every run uses the extension's clearing output wrapper."""

    def d128_mxfp4_v_operand_cache_receipt(self) -> None:
        return None

    def d128_mxfp4_v_compilation_receipt(self) -> None:
        return None

    def run(self, *, reset: bool) -> None:
        if type(reset) is not bool:
            raise TypeError("reset must be exactly bool")
        operands = (
            self._q,
            self._k,
            self._v,
            self._dout,
            self._q_native,
            self._k_native,
            self._q_native_scale,
            self._k_native_scale,
            self._q_global_scale,
            self._k_global_scale,
            self._native_score_workspace,
        )
        if any(operand is None for operand in operands):
            raise RuntimeError(
                "bind_inputs() must precede native-score v508 backward"
            )
        assert self._q is not None
        assert self._k is not None
        assert self._v is not None
        assert self._dout is not None
        assert self._q_native is not None
        assert self._k_native is not None
        assert self._q_native_scale is not None
        assert self._k_native_scale is not None
        assert self._q_global_scale is not None
        assert self._k_global_scale is not None

        # ``reset`` is accepted for the shared runner protocol, but v508 is B1
        # additive and therefore always enters through the clearing wrapper.
        self.compiled_out(
            self._q,
            self._k,
            self._v,
            self._dout,
            self.lstat,
            self.dstat,
            self.dq,
            self.dk,
            self.dv,
            self._q_native,
            self._k_native,
            self._q_native_scale,
            self._k_native_scale,
            self._q_global_scale,
            self._k_global_scale,
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
                "score_qk": {
                    "payload_dtype": "torch.float4_e2m1fn_x2",
                    "payload_layout": "BHSD_packed_contiguous",
                    "scale_dtype": "torch.float8_e4m3fn",
                    "scale_layout": (
                        "forward_row_K16_Q_S128_K_S64_physical_pages"
                    ),
                    "global_scale": "torch.float32_per_head",
                    "source": "exact_B300E4M3QKVForwardWorkspace",
                },
                "gradient_qk_v_dout": {
                    "dtype": "torch.float8_e4m3fn",
                    "layout": "BSHD_contiguous",
                    "encoding_scale": 4.0,
                    "semantics": "represented_E4M3_STE_gradient_operands",
                },
                "composition": "native_NVFP4_score_represented_E4M3_gradient",
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
                "kernel_store_semantics": "additive",
                "entrypoint": OUT_ENTRYPOINT,
                "logical_reset_per_run": True,
            },
            "schedule": {
                "dispatch": "B1_S4096_v508_only_fail_closed",
                "owner_order": "key_tile_major_head_owner",
                "direct_dkdv_unique_writer": False,
                "gradient_publisher": "owner_reducer",
                "always_clearing_out_entrypoint": True,
            },
            "publication": {
                "workspace_type": "B300E4M3QKVForwardWorkspace",
                "workspace_strongly_retained": True,
                "binding": "exact_data_ptr_shape_dtype_device",
                "native_score_forward_payload_reused": True,
                "second_qk_quantization": False,
            },
            "allocation": {
                "scope": "native_backward_runner_only",
                "caller_owned_runner_storage": True,
                "native_run_allocations": False,
                "native_run_dtype_views": False,
                "native_run_workspace_attribute_lookups": False,
            },
        }
