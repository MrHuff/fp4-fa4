"""Caller-owned runtime for the production D64 GQA TK backward.

The projection epilogue and the native TK attention kernel share a compact
statistics ABI.  Page zero of ``workspace_torch`` is ``-16 * sum(O * dO)``;
page one is ``8 - LSE * log2(e)``.  Q, K, V, and dO are fixed-scale E4M3
encodings of ``4 * x``.  The native kernel returns BF16 encodings of
``4 * dX``, matching the existing inverse-RoPE/projection consumer.

This wrapper owns its statistics workspace and outputs for its lifetime.  The
legacy v382 ABI additionally owns FP32 reduction storage; direct-output
production kernels replace it with a zero-size identity sentinel.  Binding new
operands changes only Python tensor references; it never creates a CuTe/DLPack
wrapper or copies an operand in the training loop.
"""

from __future__ import annotations

from typing import Any

import torch


SEQUENCE = 4096
Q_HEADS = 32
KV_HEADS = 8
HEAD_DIM = 64
SOFTMAX_SCALE = HEAD_DIM**-0.5
PARTIAL_HEADS = 16
BACKEND = "native_tk_d64_e4m3"
V382_SOURCE_IDENTITY = "v382_d64_gqa_e4m3_hkv2_register_pd_v1"
V414_SOURCE_IDENTITY = (
    "v414_d64_gqa_e4m3_production_bshd_dq_first_v1"
)
V416_SOURCE_IDENTITY = (
    "v416_d64_gqa_e4m3_production_bshd_dq_first_vec2_ds_v1"
)
EXPECTED_EXTENSION_METADATA = {
    "schema": "tkfa4.native_tk_d64_backward.v1",
    "backend": "thunderkittens_sm100a",
    "topology": "v382_owner_major_hkv2_register_p_ds",
    "source_identity": V382_SOURCE_IDENTITY,
    "sequence": SEQUENCE,
    "query_heads": Q_HEADS,
    "kv_heads": KV_HEADS,
    "head_dim": HEAD_DIM,
    "heads_per_owner": 2,
    "partial_heads": PARTIAL_HEADS,
    "operand_dtype": "float8_e4m3fn",
    "operand_layout": "BSHD_contiguous",
    "encoding_scale": 4.0,
    "lstat_abi": "8-LSE*log2(e)",
    "dstat_abi": "-16*sum(O*dO)",
    "stats_layout": "B,Hq,1,S_fp32_contiguous",
    "public_softmax_scale": "natural",
    "internal_beta_divisor": 16.0,
    "gradient_epilogue_scale": 1.0 / 256.0,
    "output_dtype": "bfloat16",
    "output_encoding_scale": 4.0,
    "caller_owned_output_api": True,
}
V414_EXPECTED_EXTENSION_METADATA = {
    "schema": "tkfa4.native_tk_d64_backward.v1",
    "backend": "thunderkittens_sm100a",
    "source_identity": V414_SOURCE_IDENTITY,
    "topology": (
        "split_qhead_cta_k128_q128_async_owner_aligned_tmem_p_half_"
        "overlap_shared_ds_tma_prelifted_stats_bshd_gradient_store_"
        "pipeline_dq_first"
    ),
    "aggregate_candidate": True,
    "attribution_valid": True,
    "production_data_abi_compatible": True,
    "existing_runner_drop_in_compatible": False,
    "gradient_mma_issue_order": "dq_then_dk",
    "threads": 512,
    "key_tile": 128,
    "query_tile": 128,
    "ds_shared_store": "owner_aligned_b32",
    "probability_exp": "native_ex2_clamp_production_log2_pscaled_le_8",
    "lossy_probability_alu": False,
    "sequence": "dynamic_positive_multiple_of_128",
    "production_sequence": SEQUENCE,
    "query_heads": Q_HEADS,
    "kv_heads": KV_HEADS,
    "head_dim": HEAD_DIM,
    "causal": True,
    "operand_dtype": "float8_e4m3fn",
    "operand_layout": "BSHD_contiguous",
    "encoding_scale": 4.0,
    "lstat_abi": "8-LSE*log2(e)",
    "dstat_abi": "-16*sum(O*dO)",
    "stats_layout": "B,Hq,1,S_fp32_contiguous",
    "public_softmax_scale": "natural",
    "internal_beta_divisor": 16.0,
    "gradient_epilogue_scale": 1.0 / 256.0,
    "output_dtype": "bfloat16_additive",
    "output_layout": "BSHD_contiguous",
    "output_encoding_scale": 4.0,
    "caller_owned_output_api": True,
    "caller_zeros_outputs_for_main": True,
    "backward_out_clears_outputs": True,
}
V416_EXPECTED_EXTENSION_METADATA = {
    **V414_EXPECTED_EXTENSION_METADATA,
    "source_identity": V416_SOURCE_IDENTITY,
    "ds_shared_store": "owner_aligned_v2_b32",
}
PRODUCTION_EXPECTED_METADATA = {
    V414_SOURCE_IDENTITY: (
        V414_EXPECTED_EXTENSION_METADATA,
        "/v414_d64_gqa_e4m3_production_bshd_dq_first.cu",
    ),
    V416_SOURCE_IDENTITY: (
        V416_EXPECTED_EXTENSION_METADATA,
        "/v416_d64_gqa_e4m3_production_bshd_dq_first_vec2_ds.cu",
    ),
}


def _require_extension_metadata(extension: Any) -> dict[str, Any]:
    metadata_fn = getattr(
        extension,
        "native_tk_d64_backward_metadata",
        None,
    )
    if not callable(metadata_fn):
        raise RuntimeError(
            "native TK extension lacks native_tk_d64_backward_metadata"
        )
    metadata = dict(metadata_fn())
    source_identity = metadata.get("source_identity")
    if source_identity == V382_SOURCE_IDENTITY:
        expected = EXPECTED_EXTENSION_METADATA
        expected_keys = {*expected, "source_file"}
        if set(metadata) != expected_keys:
            raise RuntimeError(
                "native TK extension metadata fields do not match the "
                "retained ABI: observed "
                f"{sorted(metadata)}, expected {sorted(expected_keys)}"
            )
        source_suffix = "/v382_d64_gqa_e4m3_hkv2_register_pd.cu"
    elif source_identity in PRODUCTION_EXPECTED_METADATA:
        expected, source_suffix = PRODUCTION_EXPECTED_METADATA[
            source_identity
        ]
        missing = {*expected, "source_file"} - set(metadata)
        if missing:
            raise RuntimeError(
                "native TK extension metadata fields do not match the "
                f"production ABI: missing {sorted(missing)}"
            )
    else:
        raise RuntimeError(
            "native TK extension metadata does not identify a supported "
            f"kernel: {source_identity!r}"
        )
    mismatches = {
        field: {"actual": metadata[field], "expected": expected_value}
        for field, expected_value in expected.items()
        if (
            metadata[field] != expected_value
            or type(metadata[field]) is not type(expected_value)
        )
    }
    source_file = metadata["source_file"]
    normalized_source = (
        source_file.replace("\\", "/")
        if isinstance(source_file, str)
        else ""
    )
    if (
        not isinstance(source_file, str)
        or not (
            normalized_source == source_suffix.removeprefix("/")
            or normalized_source.endswith(source_suffix)
        )
    ):
        mismatches["source_file"] = {
            "actual": source_file,
            "expected_suffix": source_suffix,
        }
    if mismatches:
        raise RuntimeError(
            "native TK extension metadata does not match the retained ABI: "
            f"{mismatches}"
        )
    return metadata


def _require_e4m3_bshd(
    tensor: torch.Tensor,
    *,
    name: str,
    batch: int,
    heads: int,
    device: torch.device,
) -> None:
    expected_shape = (batch, SEQUENCE, heads, HEAD_DIM)
    if (
        tensor.dtype != torch.float8_e4m3fn
        or not tensor.is_cuda
        or not tensor.is_contiguous()
        or tuple(tensor.shape) != expected_shape
        or tensor.device != device
    ):
        raise ValueError(
            f"{name} must be contiguous CUDA float8_e4m3fn "
            f"{expected_shape} on {device}; got {tensor.dtype} "
            f"{tuple(tensor.shape)} on {tensor.device}"
        )


class NativeTkD64E4M3Backward:
    """Preallocated production-shape adapter around the native TK extension."""

    backend = BACKEND
    direct_tma_dkdv = True
    detached_fp8_p_tmem = False
    head_fast_raster = False
    raster_policy = {
        "backend": backend,
        "owner_order": "causal_longest_first",
        "host_dispatch_per_launch": False,
    }
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
        if type(batch) is not int or batch <= 0 or batch > 65535:
            raise ValueError("batch must be an integer in [1, 65535]")
        self.extension = extension
        self.extension_metadata = _require_extension_metadata(extension)
        self.direct_bf16_outputs = (
            self.extension_metadata["source_identity"]
            in PRODUCTION_EXPECTED_METADATA
        )
        entrypoint = (
            "backward_e4m3_bshd_precomputed_out"
            if self.direct_bf16_outputs
            else "backward_e4m3_precomputed_out"
        )
        self.compiled = getattr(extension, entrypoint, None)
        if not callable(self.compiled):
            raise RuntimeError(
                f"native TK extension lacks {entrypoint}"
            )
        self.loaded_artifact_identity = dict(
            getattr(extension, "_tk_fa4_loaded_artifact_identity", {})
        )
        self.raster_policy = {
            "backend": self.backend,
            "owner_order": (
                "key_tile_major_query_head"
                if self.direct_bf16_outputs
                else "causal_longest_first"
            ),
            "host_dispatch_per_launch": False,
        }
        self.batch = batch
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("native TK backward requires a CUDA device")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())

        q_shape = (batch, SEQUENCE, Q_HEADS, HEAD_DIM)
        kv_shape = (batch, SEQUENCE, KV_HEADS, HEAD_DIM)
        stats_numel = batch * Q_HEADS * SEQUENCE
        self.workspace_torch = torch.empty(
            2 * stats_numel * torch.float32.itemsize,
            device=self.device,
            dtype=torch.uint8,
        )
        stats = self.workspace_torch.view(torch.float32)
        self.dstat = stats[:stats_numel].view(batch, Q_HEADS, 1, SEQUENCE)
        self.lstat = stats[stats_numel:].view(
            batch,
            Q_HEADS,
            1,
            SEQUENCE,
        )

        if self.direct_bf16_outputs:
            # The shared-runtime identity audit retains these public names.
            # One zero-size sentinel preserves that audit without allocating
            # v382's 1 GiB of FP32 accumulator/partial storage at B16/S4096.
            unused_partial_sentinel = torch.empty(
                0,
                device=self.device,
                dtype=torch.float32,
            )
            self.dq_accum = unused_partial_sentinel
            self.dk_partials = unused_partial_sentinel
            self.dv_partials = unused_partial_sentinel
        else:
            partial_shape = (batch, SEQUENCE, PARTIAL_HEADS, HEAD_DIM)
            self.dq_accum = torch.empty(
                q_shape,
                device=self.device,
                dtype=torch.float32,
            )
            self.dk_partials = torch.empty(
                partial_shape,
                device=self.device,
                dtype=torch.float32,
            )
            self.dv_partials = torch.empty_like(self.dk_partials)
        self.dq = torch.empty(
            q_shape,
            device=self.device,
            dtype=torch.bfloat16,
        )
        self.dk = torch.empty(
            kv_shape,
            device=self.device,
            dtype=torch.bfloat16,
        )
        self.dv = torch.empty_like(self.dk)

        self._q: torch.Tensor | None = None
        self._k: torch.Tensor | None = None
        self._v: torch.Tensor | None = None
        self._dout: torch.Tensor | None = None
        self._bind_generation = 0
        self._run_generation = 0

    def bind_inputs(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dout: torch.Tensor,
    ) -> None:
        """Bind exact live publications without copying or wrapping them."""
        _require_e4m3_bshd(
            q,
            name="q",
            batch=self.batch,
            heads=Q_HEADS,
            device=self.device,
        )
        _require_e4m3_bshd(
            k,
            name="k",
            batch=self.batch,
            heads=KV_HEADS,
            device=self.device,
        )
        _require_e4m3_bshd(
            v,
            name="v",
            batch=self.batch,
            heads=KV_HEADS,
            device=self.device,
        )
        _require_e4m3_bshd(
            dout,
            name="dout",
            batch=self.batch,
            heads=Q_HEADS,
            device=self.device,
        )
        self._q = q
        self._k = k
        self._v = v
        self._dout = dout
        self._bind_generation += 1

    def reset(self) -> None:
        """No-op: each native ``backward_*_out`` entrypoint clears outputs."""

    def d128_mxfp4_v_operand_cache_receipt(self) -> None:
        """Report that the retained D64 E4M3-V route has no MX-V cache."""
        return None

    def d128_mxfp4_v_compilation_receipt(self) -> None:
        """Report that the retained D64 route has no D128 MX-V image."""
        return None

    def run(self, *, reset: bool) -> None:
        if type(reset) is not bool:
            raise TypeError("reset must be exactly bool")
        if any(
            operand is None
            for operand in (self._q, self._k, self._v, self._dout)
        ):
            raise RuntimeError("bind_inputs() must precede native TK backward")
        assert self._q is not None
        assert self._k is not None
        assert self._v is not None
        assert self._dout is not None
        common_arguments = (
            self._q,
            self._k,
            self._v,
            self._dout,
            self.lstat,
            self.dstat,
        )
        if self.direct_bf16_outputs:
            self.compiled(
                *common_arguments,
                self.dq,
                self.dk,
                self.dv,
                SOFTMAX_SCALE,
            )
        else:
            self.compiled(
                *common_arguments,
                self.dq_accum,
                self.dk_partials,
                self.dv_partials,
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
                "batch": self.batch,
                "sequence": SEQUENCE,
                "q_heads": Q_HEADS,
                "kv_heads": KV_HEADS,
                "head_dim": HEAD_DIM,
            },
            "input": {
                "dtype": "torch.float8_e4m3fn",
                "layout": "BSHD_contiguous",
                "encoding_scale": 4.0,
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
            },
            "schedule": (
                {
                    "main": (
                        "split_qhead_cta_k128_q128_owner_aligned_dq_first"
                    ),
                    "owner_order": "key_tile_major_query_head",
                    "finalize_dq": False,
                    "merge_dk_dv": False,
                }
                if self.direct_bf16_outputs
                else {
                    "main": (
                        "two_CTA_two_query_heads_per_owner_register_p_ds"
                    ),
                    "owner_order": "causal_longest_first",
                    "finalize_dq": True,
                    "merge_dk_dv": True,
                }
            ),
            "allocation": {
                "scope": "native_backward_runner_only",
                "caller_owned_runner_storage": True,
                "native_run_allocations": False,
                "native_run_dlpack_wrappers": False,
                "external_projection_publication": "not_claimed",
            },
            "workspace": (
                {
                    "gradient_accumulation": "kernel_tmem",
                    "direct_output_dtype": "torch.bfloat16",
                    "host_visible_partials": False,
                }
                if self.direct_bf16_outputs
                else {
                    "dq_accum_dtype": "torch.float32",
                    "dk_dv_partial_dtype": "torch.float32",
                    "dk_dv_partial_heads": PARTIAL_HEADS,
                }
            ),
        }
