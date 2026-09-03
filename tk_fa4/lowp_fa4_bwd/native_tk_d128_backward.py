"""Caller-owned runtime for the production D128 GQA TK backward.

Q, K, V, and dO are contiguous BSHD E4M3 encodings of ``4 * x``.  The
projection epilogue writes ``-16 * sum(O * dO)`` followed by
``8 - LSE * log2(e)`` into ``workspace_torch``.  The native kernel consumes
those pages and returns contiguous BSHD BF16 encodings of ``4 * dX``.

The wrapper owns its statistics workspace and outputs for its lifetime.
Binding new operands changes only Python tensor references and never creates
CuTe/DLPack wrappers or allocates inside the measured training loop.
"""

from __future__ import annotations

from typing import Any

import torch


SEQUENCE = 4096
Q_HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128
SOFTMAX_SCALE = HEAD_DIM**-0.5
BACKEND = "native_tk_d128_e4m3"
V501_SOURCE_IDENTITY = (
    "v501_d128_gqa_e4m3_unified_best_route_production_bshd_v1"
)
EXPECTED_EXTENSION_METADATA = {
    "schema": "tkfa4.native_tk_d128_backward.v1",
    "backend": "thunderkittens_sm100a",
    "source_identity": V501_SOURCE_IDENTITY,
    "production_data_abi_compatible": True,
    "direct_output_entrypoint": "backward_e4m3_bshd_precomputed_out",
    "caller_owned_output_api": True,
    "caller_zeros_outputs_for_main": True,
    "backward_out_clears_outputs": True,
    "backward_out_logical_reset_policy": "dq_dk_dv_all_routes",
    "backward_out_physical_clear_policy": (
        "B2_S4096_memset_dq_only_complete_unique_direct_overwrite_dkdv;"
        "B1_S4096_B1_nonexact_B2_nonexact_memset_dq_dk_dv"
    ),
    "threads": 512,
    "key_tile": 128,
    "query_tile": 128,
    "query_heads": Q_HEADS,
    "kv_heads": KV_HEADS,
    "head_ratio": 4,
    "head_dim": HEAD_DIM,
    "batch_values": (1, 2),
    "sequence": "dynamic_positive_multiple_of_128",
    "exact_sequence_specialization": SEQUENCE,
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
    "dispatch": (
        "B1_S4096_v488;B2_S4096_v490;B1_other_v436;B2_other_v437"
    ),
    "selected_exact_kernels": (
        "v488::b1_owner2_exact_s4096_compact_p_reuse_kernel",
        "v490::owner4_kernel",
    ),
    "b1_s4096_route": (
        "v488_owner2_compact_p_split_dq_tmem_release_before_shared_"
        "publication_additive_dq_dk_dv"
    ),
    "b2_s4096_route": (
        "v490_owner4_compact_p_first_half_quarter_dp_fence_before_thread_"
        "sync_split_dv_dv_ready_gated_probability_operand_consumed_gated_"
        "stats_later_dk_commit_gated_operand_stage_reuse_later_dk_commit_"
        "gated_shared_ds_reuse_head_boundary_score_before_dq_drain_deferred_"
        "packed_d0_d1_register_payload_paired_final_x32_loads_dq_tmem_"
        "release_before_all_d0_d3_shared_stores_dedicated_warp14_gradient_"
        "publisher_additive_dq_unique_direct_dkdv"
    ),
    "b2_s4096_dkdv_unique_writer": True,
    "b2_s4096_dkdv_destination_preclear_required": False,
}
SOURCE_SUFFIX = "/v501_d128_gqa_e4m3_unified_best_route_production_bshd.cu"


def _require_extension_metadata(extension: Any) -> dict[str, Any]:
    metadata_fn = getattr(extension, "native_tk_d128_backward_metadata", None)
    if not callable(metadata_fn):
        raise RuntimeError(
            "native TK D128 extension lacks "
            "native_tk_d128_backward_metadata"
        )
    metadata = dict(metadata_fn())
    missing = {*EXPECTED_EXTENSION_METADATA, "source_file", "topology"} - set(
        metadata
    )
    if missing:
        raise RuntimeError(
            "native TK D128 extension metadata is incomplete: "
            f"missing {sorted(missing)}"
        )
    mismatches = {
        field: {"actual": metadata[field], "expected": expected}
        for field, expected in EXPECTED_EXTENSION_METADATA.items()
        if metadata[field] != expected or type(metadata[field]) is not type(expected)
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
    if not isinstance(metadata["topology"], str) or not metadata["topology"]:
        mismatches["topology"] = {
            "actual": metadata["topology"],
            "expected": "non-empty authenticated topology",
        }
    if mismatches:
        raise RuntimeError(
            "native TK D128 extension metadata does not match the production "
            f"ABI: {mismatches}"
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


class NativeTkD128E4M3Backward:
    """Preallocated B1/B2 S4096 adapter around the native TK extension."""

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
        if type(batch) is not int or batch not in (1, 2):
            raise ValueError("native TK D128 backward requires batch 1 or 2")
        self.extension = extension
        self.extension_metadata = _require_extension_metadata(extension)
        self.compiled_out = getattr(
            extension,
            "backward_e4m3_bshd_precomputed_out",
            None,
        )
        if not callable(self.compiled_out):
            raise RuntimeError(
                "native TK D128 extension lacks "
                "backward_e4m3_bshd_precomputed_out"
            )
        self.compiled_main = getattr(
            extension,
            "main_e4m3_bshd_precomputed",
            None,
        )
        if not callable(self.compiled_main):
            raise RuntimeError(
                "native TK D128 extension lacks "
                "main_e4m3_bshd_precomputed"
            )
        self.loaded_artifact_identity = dict(
            getattr(extension, "_tk_fa4_loaded_artifact_identity", {})
        )
        self.batch = batch
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("native TK D128 backward requires a CUDA device")
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
        self.lstat = stats[stats_numel:].view(batch, Q_HEADS, 1, SEQUENCE)
        self.dq = torch.empty(q_shape, device=self.device, dtype=torch.bfloat16)
        self.dk = torch.empty(kv_shape, device=self.device, dtype=torch.bfloat16)
        self.dv = torch.empty_like(self.dk)
        # The backend-neutral shared-runtime identity audit retains these
        # historical CuTe partial-workspace names. Native D128 writes BF16
        # outputs directly, so one zero-size sentinel authenticates their
        # shared identity without allocating unused accumulator storage.
        unused_partial_sentinel = torch.empty(
            0,
            device=self.device,
            dtype=torch.float32,
        )
        self.dk_partials = unused_partial_sentinel
        self.dv_partials = unused_partial_sentinel

        # Both exact routes publish dK/dV directly to caller-owned outputs.
        # B2 has unique writers; B1 retains additive direct TMA stores.
        self.direct_tma_dkdv = True
        self.raster_policy = {
            "backend": self.backend,
            "owner_order": "key_tile_major_head_owner",
            "host_dispatch_per_launch": False,
            "heads_per_owner": 2 if batch == 1 else 4,
        }
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
        for tensor, name, heads in (
            (q, "q", Q_HEADS),
            (k, "k", KV_HEADS),
            (v, "v", KV_HEADS),
            (dout, "dout", Q_HEADS),
        ):
            _require_e4m3_bshd(
                tensor,
                name=name,
                batch=self.batch,
                heads=heads,
                device=self.device,
            )
        self._q = q
        self._k = k
        self._v = v
        self._dout = dout
        self._bind_generation += 1

    def reset(self) -> None:
        """No-op: the authenticated direct-output entrypoint resets outputs."""

    def d128_mxfp4_v_operand_cache_receipt(self) -> None:
        return None

    def d128_mxfp4_v_compilation_receipt(self) -> None:
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
        # The exact B2/S4096 v490 route has unique complete dK/dV writers.
        # When reset=False, the fused projection has already cleared dQ, so
        # the unchecked main entrypoint can safely elide the redundant dQ
        # memset.  Every other path remains on the authenticated clearing
        # wrapper: B1 accumulates dQ/dK/dV and reset=True explicitly requests
        # the defensive logical reset.
        compiled = (
            self.compiled_main
            if self.batch == 2 and not reset
            else self.compiled_out
        )
        compiled(
            self._q,
            self._k,
            self._v,
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
                "logical_reset_per_run": True,
            },
            "schedule": {
                "dispatch": self.extension_metadata["dispatch"],
                "owner_order": "key_tile_major_head_owner",
                "direct_dkdv_unique_writer": self.batch == 2,
                "gradient_publisher": (
                    "dedicated_warp14" if self.batch == 2 else "owner_reducer"
                ),
            },
            "allocation": {
                "scope": "native_backward_runner_only",
                "caller_owned_runner_storage": True,
                "native_run_allocations": False,
                "native_run_dlpack_wrappers": False,
                "external_projection_publication": "authenticated_e4m3_bshd",
            },
        }
