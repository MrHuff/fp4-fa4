#!/usr/bin/env python3
"""Regress finite D64 backward probability reconstruction at the FP8/LSE boundary."""

from __future__ import annotations

import json
import math

import torch

from tk_fa4.lowp_fa4_bwd.profile_gqa_d128_chain import CompiledGqaBackward
from tk_fa4.lowp_fa4_bwd.tune_d64_gqa_cute import _load_control


def _health(tensor: torch.Tensor) -> dict[str, float | int | bool]:
    values = tensor.float()
    finite = torch.isfinite(values)
    return {
        "all_finite": bool(finite.all()),
        "nonfinite": int((~finite).sum()),
        "max_abs_finite": (
            float(values[finite].abs().max()) if bool(finite.any()) else float("nan")
        ),
    }


def main() -> None:
    if torch.cuda.device_count() != 1:
        raise RuntimeError("validation requires exactly one visible GPU")
    torch.cuda.set_device(0)

    batch = 1
    sequence = 128
    q_heads = 4
    kv_heads = 1
    depth = 64

    # These values are exactly representable in E4M3.  With the backward
    # fixed-scale score correction (1 / 16), the selected head reconstructs
    # a lifted log2 probability just above 128.  The old polynomial exp2 bit
    # construction returned a NaN there even though a probability may never
    # exceed 1 (or 256 after the fused lift).
    q = torch.zeros(
        batch,
        sequence,
        q_heads,
        depth,
        device="cuda",
        dtype=torch.float8_e4m3fn,
    )
    k = torch.zeros(
        batch,
        sequence,
        kv_heads,
        depth,
        device="cuda",
        dtype=torch.float8_e4m3fn,
    )
    q[:, :, 0, :63] = 12.0
    q[:, :, 0, 63] = 8.0
    k[:, :, 0, :63] = 14.0
    k[:, :, 0, 63] = 9.0
    v = torch.zeros_like(k)
    dout = torch.ones(
        batch,
        sequence,
        q_heads,
        depth,
        device="cuda",
        dtype=torch.float8_e4m3fn,
    )

    dpsum = torch.zeros(batch, q_heads, 1, sequence, device="cuda", dtype=torch.float32)
    prelifted_lse = torch.full_like(dpsum, -127.0)
    prelifted_lse[:, 0] = 8.0

    lifted_log2_probability = ((63.0 * 12.0 * 14.0 + 8.0 * 9.0) / 128.0) * math.log2(
        math.e
    ) + 8.0
    if not 128.0 < lifted_log2_probability < 129.0:
        raise AssertionError(lifted_log2_probability)

    control = _load_control(fp8_p_storage="tmem", direct_tma_dkdv=True)
    backward = CompiledGqaBackward(
        control,
        q=q,
        k=k,
        v=v,
        o_or_sum=dpsum,
        dout=dout,
        lse_or_scaled_lse=prelifted_lse,
        q_heads=q_heads,
        kv_heads=kv_heads,
        lowp=True,
        precomputed_stats=True,
        workspace_stats=True,
        scale_softmax=(depth**-0.5) / 16.0,
        exp2_degree=2,
        exp2_period=2,
        fp8_ds_lift=16,
        direct_tma_dkdv=True,
    )
    backward.run(reset=True)
    torch.cuda.synchronize()

    gradients = {
        "dq": _health(backward.dq),
        "dk": _health(backward.dk),
        "dv": _health(backward.dv),
    }
    expected_dv = (
        torch.arange(
            sequence,
            0,
            -1,
            device="cuda",
            dtype=torch.float32,
        )
        .view(1, sequence, 1, 1)
        .expand_as(backward.dv)
    )
    dv_max_abs_error = float((backward.dv.float() - expected_dv).abs().max())
    result = {
        "lifted_log2_probability_before_ceiling": lifted_log2_probability,
        "probability_log2_ceiling": 8.0,
        "exp2_degree": 2,
        "exp2_period": 2,
        "gradients": gradients,
        "dv_max_abs_error_vs_clipped_probability": dv_max_abs_error,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    failures = [name for name, health in gradients.items() if not health["all_finite"]]
    if failures:
        raise AssertionError(
            f"non-finite gradients after probability ceiling: {failures}"
        )
    if backward.dq.count_nonzero() or backward.dk.count_nonzero():
        raise AssertionError("zero dP and V must produce exactly zero dQ/dK")
    if dv_max_abs_error != 0.0:
        raise AssertionError(
            "dV does not match the exact causal result after clipping: "
            f"max_abs_error={dv_max_abs_error}"
        )


if __name__ == "__main__":
    main()
