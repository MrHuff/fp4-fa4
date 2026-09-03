#!/usr/bin/env python3
"""Adapter from the frozen native HAO input bundle to the isolated TK port.

The adapter preparation step is intentionally outside timing.  The first
kernel milestone is a full-grid correctness bring-up; it is not authorized as
an apples-to-apples performance provider until the HAO-style persistent work
assignment is implemented and separately frozen.
"""

from __future__ import annotations

import os
from typing import Any, NamedTuple


PROVIDER_NAME = "tk_hao_native_topology_port"
ADAPTER_IMPLEMENTED = True
GPU_AUTHORIZED = True
PORT_STAGE = "fullgrid_correctness_bringup"
ROUTE = "real_fwd_tk_hao_native_b1s4096h24d128_fullgrid"
ARTIFACT_SHA256 = (
    "70ad3daee7ece16a0fc4e05b9b19136b30eff8ac739882c1e944c93260dd6b56"
)

BATCH = 1
SEQLEN = 4096
HEADS = 24
DQK = 128
DVO = 128
QUERY_STAGES = 2
LOGICAL_JOBS = BATCH * HEADS * (SEQLEN // (QUERY_STAGES * 128))
FULLGRID_CTAS = LOGICAL_JOBS


class PreparedNativeTkInputs(NamedTuple):
    q_fp4_bhsd: Any
    q_scale_prepared: Any
    q_global_scale: Any
    k_fp4_bhsd: Any
    k_scale_prepared: Any
    k_global_scale: Any
    v_fp8_bhds: Any


def _require_shape(tensor: Any, expected: tuple[int, ...], name: str) -> None:
    actual = tuple(int(value) for value in tensor.shape)
    if actual != expected:
        raise ValueError(f"{name} shape {actual} != {expected}")


def native_scale_to_tk_prepared(scale: Any, *, duplicate_k_depth: bool) -> Any:
    """Convert HAO's logical 7-D scale view to TK's prepared 4-D view.

    HAO exposes `(32, 4, M/128, 4, D/64, H, B)`.  TK consumes
    `(B, M/128, H*(D/64), 512)`.  K currently uses the inherited local
    N64-depth descriptor, so each native N128 scale tile is duplicated in the
    depth dimension outside timing.
    """

    expected = (32, 4, SEQLEN // 128, 4, DQK // 64, HEADS, BATCH)
    _require_shape(scale, expected, "native Q/K scale")
    prepared = (
        scale.permute(6, 2, 5, 4, 0, 1, 3)
        .contiguous()
        .reshape(BATCH, SEQLEN // 128, HEADS * (DQK // 64), 512)
    )
    # FlashInfer exposes the native E4M3 scale payload through byte-backed
    # uint8 storage.  Preserve every encoded bit; a numeric .to(float8) would
    # corrupt the TCGEN scale operand.
    if str(prepared.dtype) == "torch.uint8":
        import torch

        prepared = prepared.view(torch.float8_e4m3fn)
    if duplicate_k_depth:
        prepared = prepared.repeat_interleave(2, dim=1).contiguous()
    return prepared


def prepare_native_tk_inputs(
    q_fp4: Any,
    k_fp4: Any,
    v_fp8: Any,
    q_scale: Any,
    k_scale: Any,
) -> PreparedNativeTkInputs:
    """Prepare the exact native bundle for TK, outside the timed closure."""

    import torch

    packed_d = DQK // 2
    _require_shape(q_fp4, (BATCH, SEQLEN, HEADS, packed_d), "q_fp4")
    _require_shape(k_fp4, (BATCH, SEQLEN, HEADS, packed_d), "k_fp4")
    _require_shape(v_fp8, (BATCH, SEQLEN, HEADS, DVO), "v_fp8")
    if q_fp4.dtype != torch.float4_e2m1fn_x2:
        raise TypeError("q_fp4 must be torch.float4_e2m1fn_x2")
    if k_fp4.dtype != torch.float4_e2m1fn_x2:
        raise TypeError("k_fp4 must be torch.float4_e2m1fn_x2")
    if v_fp8.dtype != torch.float8_e4m3fn:
        raise TypeError("v_fp8 must be torch.float8_e4m3fn")
    native_scale_dtypes = (torch.uint8, torch.float8_e4m3fn)
    if q_scale.dtype not in native_scale_dtypes:
        raise TypeError("q_scale must be byte-backed or typed E4M3")
    if k_scale.dtype not in native_scale_dtypes:
        raise TypeError("k_scale must be byte-backed or typed E4M3")

    # PyTorch does not implement copy_ for the packed Float4 dtype.  Move the
    # one-byte packed payload through a uint8 view and reinterpret afterward;
    # no nibble is decoded or requantized.
    q_local = (
        q_fp4.view(torch.uint8)
        .permute(0, 2, 1, 3)
        .contiguous()
        .view(torch.float4_e2m1fn_x2)
    )
    k_local = (
        k_fp4.view(torch.uint8)
        .permute(0, 2, 1, 3)
        .contiguous()
        .view(torch.float4_e2m1fn_x2)
    )
    v_local = v_fp8.permute(0, 2, 3, 1).contiguous()
    q_sc_local = native_scale_to_tk_prepared(
        q_scale, duplicate_k_depth=False
    )
    k_sc_local = native_scale_to_tk_prepared(
        k_scale, duplicate_k_depth=True
    )
    # Native FlashInfer scale factors are absolute TCGEN operands.  The local
    # kernel's extra per-head factor is therefore the multiplicative identity.
    q_sg = torch.ones((BATCH, HEADS), device=q_fp4.device, dtype=torch.float32)
    k_sg = torch.ones((BATCH, HEADS), device=k_fp4.device, dtype=torch.float32)
    return PreparedNativeTkInputs(
        q_local,
        q_sc_local,
        q_sg,
        k_local,
        k_sc_local,
        k_sg,
        v_local,
    )


def invoke_native_tk_provider(
    extension: Any,
    prepared: PreparedNativeTkInputs,
) -> tuple[Any, Any]:
    """Invoke only an explicitly frozen port extension.

    The benchmark harness owns extension identity and correctness
    authorization.  This adapter never loads or installs an extension.
    """

    import torch

    if not GPU_AUTHORIZED:
        raise RuntimeError(
            "TK native topology port is not GPU-authorized; finish static and "
            "correctness gates before enabling it in the native benchmark"
        )
    if not hasattr(extension, "forward_streaming_live_mxfp4"):
        raise AttributeError("port extension lacks forward binding")
    if not hasattr(extension, "read_upstream_mxfp4_pipeline_topology"):
        raise AttributeError("port extension lacks topology reader")
    output = torch.empty(
        (BATCH, SEQLEN, HEADS, DVO),
        device=prepared.q_fp4_bhsd.device,
        dtype=torch.bfloat16,
    )
    lse = torch.empty(
        (BATCH, HEADS, 1, SEQLEN),
        device=prepared.q_fp4_bhsd.device,
        dtype=torch.float32,
    )
    previous_route = os.environ.get("TK_FA4_FP4PV_FWD_CONFIG")
    try:
        os.environ["TK_FA4_FP4PV_FWD_CONFIG"] = ROUTE
        extension.forward_streaming_live_mxfp4(
            prepared.q_fp4_bhsd,
            prepared.q_scale_prepared,
            prepared.q_global_scale,
            prepared.k_fp4_bhsd,
            prepared.k_scale_prepared,
            prepared.k_global_scale,
            prepared.v_fp8_bhds,
            output,
            lse,
            0,
            False,
        )
    finally:
        if previous_route is None:
            os.environ.pop("TK_FA4_FP4PV_FWD_CONFIG", None)
        else:
            os.environ["TK_FA4_FP4PV_FWD_CONFIG"] = previous_route
    topology = dict(extension.read_upstream_mxfp4_pipeline_topology())
    expected = {
        "route": ROUTE,
        "batch": BATCH,
        "heads": HEADS,
        "seqlen": SEQLEN,
        "dqk": DQK,
        "dvo": DVO,
        "logical_jobs": LOGICAL_JOBS,
        "physical_grid_ctas": FULLGRID_CTAS,
        "threads_per_cta": 512,
        "query_stages": QUERY_STAGES,
        "tmem_columns": 512,
    }
    for key, value in expected.items():
        if topology.get(key) != value:
            raise RuntimeError(
                f"TK port topology mismatch for {key}: "
                f"{topology.get(key)!r} != {value!r}"
            )
    return output, lse.squeeze(2)
