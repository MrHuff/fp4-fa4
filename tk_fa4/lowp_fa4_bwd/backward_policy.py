"""Pure dispatch policies for the causal FA4 backward implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


BACKWARD_EXP2_POLICY_VERSION = "d64_verified_shape_selective_exp2_v3"
BACKWARD_RASTER_POLICY_VERSION = "d64_verified_shape_head_fast_raster_v2"
BACKWARD_P_TMEM_POLICY_VERSION = "d64_verified_shape_detached_fp8_p_tmem_v2"
D64_SELECTIVE_EXP2_MIN_SEQUENCE = 4096
D64_SELECTIVE_EXP2_VERIFIED_SHAPES = (
    (4096, 16, 4),
    (4096, 32, 8),
    (4096, 64, 16),
    (8192, 32, 8),
    (16384, 32, 8),
)
D64_HEAD_FAST_RASTER_VERIFIED_SHAPES = (
    (8192, 32, 8),
    (16384, 32, 8),
)
D64_DETACHED_FP8_P_TMEM_VERIFIED_SHAPES = (
    (8192, 32, 8),
    (16384, 32, 8),
)


@dataclass(frozen=True)
class BackwardExp2Policy:
    """Requested and effective compile-time EX2 settings."""

    requested_degree: int
    requested_period: int | None
    effective_degree: int
    effective_period: int
    mode: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": BACKWARD_EXP2_POLICY_VERSION,
            "mode": self.mode,
            "requested": {
                "degree": self.requested_degree,
                "period": self.requested_period,
            },
            "effective": {
                "degree": self.effective_degree,
                "period": self.effective_period,
            },
            "reason": self.reason,
        }


@dataclass(frozen=True)
class BackwardRasterPolicy:
    """Requested and effective compile-time CTA raster order."""

    requested_head_fast: bool | None
    effective_head_fast: bool
    auto_eligible: bool
    forced_head_fast: bool
    mode: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": BACKWARD_RASTER_POLICY_VERSION,
            "mode": self.mode,
            "requested_head_fast": self.requested_head_fast,
            "effective_head_fast": self.effective_head_fast,
            "auto_eligible": self.auto_eligible,
            "forced_head_fast": self.forced_head_fast,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class BackwardProbabilityTmemPolicy:
    """Requested and effective FP8 P placement in tensor memory."""

    requested_detached: bool | None
    effective_detached: bool
    auto_eligible: bool
    mode: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": BACKWARD_P_TMEM_POLICY_VERSION,
            "mode": self.mode,
            "requested_detached": self.requested_detached,
            "effective_detached": self.effective_detached,
            "auto_eligible": self.auto_eligible,
            "reason": self.reason,
        }


def resolve_backward_probability_tmem_policy(
    *,
    sequence: int,
    head_dim: int,
    q_heads: int,
    kv_heads: int,
    batch: int,
    lowp: bool,
    detached_fp8_p_tmem: bool | None,
    auto_eligible: bool = False,
) -> BackwardProbabilityTmemPolicy:
    """Resolve whether FP8 P aliases score TMEM or uses the D64 tail."""

    if sequence <= 0:
        raise ValueError("sequence must be positive")
    if head_dim not in (64, 128):
        raise ValueError("head_dim must be 64 or 128")
    if q_heads <= 0 or kv_heads <= 0 or q_heads % kv_heads:
        raise ValueError("head counts must be positive with q_heads % kv_heads == 0")
    if batch <= 0:
        raise ValueError("batch must be positive")

    if not lowp:
        return BackwardProbabilityTmemPolicy(
            requested_detached=detached_fp8_p_tmem,
            effective_detached=False,
            auto_eligible=False,
            mode="bf16_alias_p",
            reason="the BF16 comparison does not publish an FP8 P operand",
        )

    if detached_fp8_p_tmem is not None:
        return BackwardProbabilityTmemPolicy(
            requested_detached=detached_fp8_p_tmem,
            effective_detached=detached_fp8_p_tmem,
            auto_eligible=auto_eligible,
            mode="explicit",
            reason="an explicit P-placement override wins over measured dispatch",
        )

    verified_shape = (
        batch == 1
        and head_dim == 64
        and (sequence, q_heads, kv_heads)
        in D64_DETACHED_FP8_P_TMEM_VERIFIED_SHAPES
    )
    if verified_shape and auto_eligible:
        return BackwardProbabilityTmemPolicy(
            requested_detached=None,
            effective_detached=True,
            auto_eligible=True,
            mode="auto_verified_shape",
            reason=(
                "the measured long-sequence D64 retained route stores FP8 P "
                "in unused TMEM so score retirement does not wait for P "
                "publication"
            ),
        )

    if verified_shape:
        mode = "auto_alias_p_ineligible_route"
        reason = (
            "the shape is measured, but this route does not match the retained "
            "one-lane direct-TMA TMEM configuration"
        )
    else:
        mode = "auto_alias_p"
        reason = "detached FP8 P is not yet verified for this backward shape"
    return BackwardProbabilityTmemPolicy(
        requested_detached=None,
        effective_detached=False,
        auto_eligible=auto_eligible,
        mode=mode,
        reason=reason,
    )


def resolve_backward_raster_policy(
    *,
    sequence: int,
    head_dim: int,
    q_heads: int,
    kv_heads: int,
    batch: int,
    lowp: bool,
    head_fast_raster: bool | None,
    auto_eligible: bool = False,
    force_head_fast: bool = False,
) -> BackwardRasterPolicy:
    """Resolve the physical CTA raster without importing Torch or CuTe."""

    if sequence <= 0:
        raise ValueError("sequence must be positive")
    if head_dim not in (64, 128):
        raise ValueError("head_dim must be 64 or 128")
    if q_heads <= 0 or kv_heads <= 0 or q_heads % kv_heads:
        raise ValueError("head counts must be positive with q_heads % kv_heads == 0")
    if batch <= 0:
        raise ValueError("batch must be positive")

    if not lowp:
        if force_head_fast:
            raise ValueError("a forced head-fast raster requires the lowp route")
        return BackwardRasterPolicy(
            requested_head_fast=head_fast_raster,
            effective_head_fast=False,
            auto_eligible=False,
            forced_head_fast=False,
            mode="bf16_key_fast",
            reason="the BF16 comparison retains its established key-fast raster",
        )

    if force_head_fast:
        return BackwardRasterPolicy(
            requested_head_fast=head_fast_raster,
            effective_head_fast=True,
            auto_eligible=auto_eligible,
            forced_head_fast=True,
            mode="owner_required",
            reason=(
                "owner dQ publication requires query heads in physical grid-x"
            ),
        )

    if head_fast_raster is not None:
        return BackwardRasterPolicy(
            requested_head_fast=head_fast_raster,
            effective_head_fast=head_fast_raster,
            auto_eligible=auto_eligible,
            forced_head_fast=False,
            mode="explicit",
            reason="an explicit raster override wins over measured-shape dispatch",
        )

    verified_shape = (
        batch == 1
        and head_dim == 64
        and (sequence, q_heads, kv_heads)
        in D64_HEAD_FAST_RASTER_VERIFIED_SHAPES
    )
    if verified_shape and auto_eligible:
        return BackwardRasterPolicy(
            requested_head_fast=None,
            effective_head_fast=True,
            auto_eligible=True,
            forced_head_fast=False,
            mode="auto_verified_shape",
            reason=(
                "this measured one-lane direct-TMA D64 causal route launches "
                "all heads for each key tile so the longest CTAs retire before "
                "the tail"
            ),
        )

    if verified_shape:
        reason = (
            "the shape is measured, but this route does not match the retained "
            "one-lane direct-TMA TMEM configuration"
        )
        mode = "auto_key_fast_ineligible_route"
    else:
        reason = "head-fast scheduling is not yet verified for this backward shape"
        mode = "auto_key_fast"
    return BackwardRasterPolicy(
        requested_head_fast=None,
        effective_head_fast=False,
        auto_eligible=auto_eligible,
        forced_head_fast=False,
        mode=mode,
        reason=reason,
    )


def resolve_backward_exp2_policy(
    *,
    sequence: int,
    head_dim: int,
    q_heads: int | None = None,
    kv_heads: int | None = None,
    lowp: bool,
    exp2_degree: int,
    exp2_period: int | None,
) -> BackwardExp2Policy:
    """Resolve the compile-time EX2 pair without importing Torch or CuTe.

    An explicit integer period always wins, including zero.  ``None`` opts
    D64 low-precision backward into the measured sequence dispatch.  D128's
    previous implicit d1/p2 behavior is retained because the new measurements
    cover only D64.
    """

    if sequence <= 0:
        raise ValueError("sequence must be positive")
    if head_dim not in (64, 128):
        raise ValueError("head_dim must be 64 or 128")
    if (q_heads is None) != (kv_heads is None):
        raise ValueError("q_heads and kv_heads must be supplied together")
    if q_heads is not None and (
        q_heads <= 0 or kv_heads is None or kv_heads <= 0
    ):
        raise ValueError("head counts must be positive")
    if exp2_degree not in (1, 2):
        raise ValueError("exp2_degree must be one or two")
    if exp2_period is not None and not 0 <= exp2_period <= 16:
        raise ValueError("exp2_period must be in [0, 16] or None")

    if not lowp:
        return BackwardExp2Policy(
            requested_degree=exp2_degree,
            requested_period=exp2_period,
            effective_degree=exp2_degree,
            effective_period=0,
            mode="bf16_disabled",
            reason="the BF16 control does not use the low-precision EX2 shortcut",
        )

    if exp2_period is not None:
        return BackwardExp2Policy(
            requested_degree=exp2_degree,
            requested_period=exp2_period,
            effective_degree=exp2_degree,
            effective_period=exp2_period,
            mode="explicit",
            reason="an explicit period overrides automatic sequence dispatch",
        )

    if head_dim == 128:
        return BackwardExp2Policy(
            requested_degree=exp2_degree,
            requested_period=None,
            effective_degree=exp2_degree,
            effective_period=2,
            mode="legacy_d128_auto",
            reason="D128 retains its pre-existing implicit period-2 policy",
        )

    verified_shape = (
        sequence,
        q_heads,
        kv_heads,
    ) in D64_SELECTIVE_EXP2_VERIFIED_SHAPES
    if verified_shape:
        return BackwardExp2Policy(
            requested_degree=exp2_degree,
            requested_period=None,
            effective_degree=1,
            effective_period=2,
            mode="auto_verified_shape",
            reason=(
                "this measured D64 sequence/head shape uses the "
                "degree-1/period-2 selective packed-ALU EX2 path"
            ),
        )

    if sequence >= D64_SELECTIVE_EXP2_MIN_SEQUENCE:
        return BackwardExp2Policy(
            requested_degree=exp2_degree,
            requested_period=None,
            effective_degree=2,
            effective_period=0,
            mode="auto_unverified_shape",
            reason=(
                "selective EX2 is not yet verified for this long-sequence "
                "D64 sequence/head shape"
            ),
        )

    return BackwardExp2Policy(
        requested_degree=exp2_degree,
        requested_period=None,
        effective_degree=2,
        effective_period=0,
        mode="auto_native",
        reason="D64 S<4096 retains native EX2; shorter selective use is unverified",
    )
