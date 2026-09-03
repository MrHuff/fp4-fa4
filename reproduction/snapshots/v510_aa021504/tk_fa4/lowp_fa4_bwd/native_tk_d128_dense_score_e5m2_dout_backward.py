"""Caller-owned runtime for the experimental v510 D128 GQA TK backward.

Q, K, and V are contiguous BSHD E4M3 encodings of ``4 * x``. dO alone is a
contiguous BSHD E5M2 encoding of ``4 * dO``. The caller supplies physically
matched ``-4 * sum(O * raw_E5M2_dO)`` (logically ``-16 * sum(O * dO)``) and
``8 - LSE * log2(e)`` statistic pages. The kernel returns additive contiguous
BSHD BF16 encodings of ``4 * dX``.

The wrapper owns its statistics workspace and outputs for its lifetime.
Binding new operands changes only Python tensor references and never creates
CuTe/DLPack wrappers or allocates inside the measured training loop.
"""

from __future__ import annotations

from typing import Any

import torch


BATCH = 1
SEQUENCE = 4096
Q_HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128
SOFTMAX_SCALE = HEAD_DIM**-0.5
BACKEND = "native_tk_d128_dense_e4m3_score_qkv_e5m2_dout"
V510_SOURCE_IDENTITY = (
    "v510_dense_e4m3_score_qkv_e5m2_dout_b1_s4096_experimental_v1"
)
OUT_ENTRYPOINT = (
    "backward_e4m3_score_qkv_e5m2_dout_bshd_precomputed_out"
)
MAIN_ENTRYPOINT = "main_e4m3_score_qkv_e5m2_dout_bshd_precomputed"
EXPECTED_EXTENSION_METADATA = {
    "schema": "tkfa4.native_tk_d128_backward.v1",
    "backend": "thunderkittens_sm100a",
    "source_identity": V510_SOURCE_IDENTITY,
    "experimental": True,
    "production_dispatch_connected": False,
    "dispatch": "fail_closed_B1_S4096_only_no_fallback",
    "selected_kernel": (
        "v510::b1_dense_e4m3_score_qkv_e5m2_dout_exact_s4096_kernel"
    ),
    "score_qk_dtype": "float8_e4m3fn_represented_x4",
    "score_qk_layout": "BSHD_contiguous",
    "score_mma": "dense_E4M3_A_times_E4M3_B_transpose",
    "gradient_qkv_dtype": "float8_e4m3fn_represented_x4",
    "dout_dtype": "float8_e5m2_represented_x4",
    "dout_encode_scale": 4.0,
    "dout_decode_scale": 0.25,
    "mixed_mma_b_format_mask": 1024,
    "score_internal_beta_divisor": 16.0,
    "ds_internal_beta_divisor": 16.0,
    "lstat_abi": "8-LSE*log2(e)",
    "dstat_abi": "-16*sum(O*dO)",
    "dstat_physical_abi": "-4*sum(O*raw_E5M2_dO)",
    "output_dtype": "bfloat16_additive",
    "batch": BATCH,
    "sequence": SEQUENCE,
    "query_heads": Q_HEADS,
    "kv_heads": KV_HEADS,
    "head_dim": HEAD_DIM,
    "threads": 512,
    "user_shared_storage_bytes": 162 * 1024,
    "score_schedule": "dense_E4M3_score_then_mixed_E4M3_E5M2_dP_dV",
    "caller_owned_output_api": True,
    "main_requires_precleared_outputs": True,
    "backward_out_clears_dq_dk_dv": True,
}
SOURCE_SUFFIX = "/v510_d128_gqa_e4m3_score_qkv_e5m2_dout_b1_exact_s4096_experimental_bshd.cu"


def _require_extension_metadata(extension: Any) -> dict[str, Any]:
    metadata_fn = getattr(extension, "native_tk_d128_backward_metadata", None)
    if not callable(metadata_fn):
        raise RuntimeError(
            "native TK D128 extension lacks "
            "native_tk_d128_backward_metadata"
        )
    metadata = dict(metadata_fn())
    missing = {*EXPECTED_EXTENSION_METADATA, "source_file"} - set(metadata)
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
    if mismatches:
        raise RuntimeError(
            "native TK D128 v510 extension does not match the experimental "
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


def _require_e5m2_bshd(
    tensor: torch.Tensor,
    *,
    name: str,
    batch: int,
    heads: int,
    device: torch.device,
) -> None:
    expected_shape = (batch, SEQUENCE, heads, HEAD_DIM)
    if (
        tensor.dtype != torch.float8_e5m2
        or not tensor.is_cuda
        or not tensor.is_contiguous()
        or tuple(tensor.shape) != expected_shape
        or tensor.device != device
    ):
        raise ValueError(
            f"{name} must be contiguous CUDA float8_e5m2 "
            f"{expected_shape} on {device}; got {tensor.dtype} "
            f"{tuple(tensor.shape)} on {tensor.device}"
        )


class NativeTkD128DenseE4M3ScoreQKVE5M2DoutBackward:
    """Preallocated strict B1/S4096 adapter around the v510 extension."""

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
            raise ValueError("native TK D128 v510 backward requires batch 1")
        self.extension = extension
        self.extension_metadata = _require_extension_metadata(extension)
        self.compiled_out = getattr(extension, OUT_ENTRYPOINT, None)
        if not callable(self.compiled_out):
            raise RuntimeError(
                f"native TK D128 v510 extension lacks {OUT_ENTRYPOINT}"
            )
        self.compiled_main = getattr(extension, MAIN_ENTRYPOINT, None)
        if not callable(self.compiled_main):
            raise RuntimeError(
                f"native TK D128 v510 extension lacks {MAIN_ENTRYPOINT}"
            )
        # Keep the shared backward physical-identity audit compatible while
        # ensuring every public run enters through the clearing wrapper.
        self.compiled = self.compiled_out
        self.kernel = None
        self.loaded_artifact_identity = dict(
            getattr(extension, "_tk_fa4_loaded_artifact_identity", {})
        )
        self.batch = batch
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("native TK D128 v510 backward requires CUDA")
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

        # v510 retains v488's B1 additive direct TMA publication.
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
        ):
            _require_e4m3_bshd(
                tensor,
                name=name,
                batch=self.batch,
                heads=heads,
                device=self.device,
            )
        _require_e5m2_bshd(
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
            raise RuntimeError(
                "bind_inputs() must precede native TK D128 v510 backward"
            )
        assert self._q is not None
        assert self._k is not None
        assert self._v is not None
        assert self._dout is not None
        # v510 publishes additive B1 gradients. Accept ``reset`` for the
        # shared runner protocol, but always call the authenticated clearing
        # wrapper so caller state cannot silently violate that ABI.
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
                "score_and_gradient_qkv": {
                    "dtype": "torch.float8_e4m3fn",
                    "layout": "BSHD_contiguous",
                    "encoding_scale": 4.0,
                    "semantics": "represented_E4M3_STE_operands",
                },
                "dout": {
                    "dtype": "torch.float8_e5m2",
                    "layout": "BSHD_contiguous",
                    "encoding_scale": 4.0,
                    "decode_scale": 0.25,
                    "semantics": "represented_E5M2_STE_gradient_operand",
                    "matched_dout_dstat_authentication": (
                        "external_caller_responsibility"
                    ),
                },
            },
            "statistics": {
                "workspace_page_0": "-16_sum_O_dO",
                "workspace_page_0_physical": "-4_sum_O_raw_E5M2_dO",
                "workspace_page_1": "8_minus_LSE_log2e",
                "producer_native": False,
                "dstat_population": "external_producer_required",
                "lstat_population": "external_forward_population_required",
            },
            "output": {
                "dtype": "torch.bfloat16",
                "layout": "BSHD_contiguous",
                "encoding_scale": 4.0,
                "logical_reset_per_run": True,
                "kernel_store_semantics": "additive",
                "entrypoint": OUT_ENTRYPOINT,
                "storage_owner": "runner",
            },
            "schedule": {
                "dispatch": self.extension_metadata["dispatch"],
                "owner_order": "key_tile_major_head_owner",
                "direct_dkdv_unique_writer": False,
                "gradient_publisher": "owner_reducer",
                "always_clearing_out_entrypoint": True,
            },
            "allocation": {
                "scope": "native_backward_runner_only",
                "caller_owned_runner_storage": True,
                "extension_outputs_runner_owned": True,
                "e5m2_dout_external_caller_owned": True,
                "native_run_allocations": False,
                "native_run_dlpack_wrappers": False,
                "external_projection_publication": (
                    "authenticated_e4m3_qkv_and_e5m2_dout_bshd"
                ),
            },
        }
