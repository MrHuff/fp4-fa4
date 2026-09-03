from __future__ import annotations

import pytest

from tk_fa4.lowp_fa4_bwd.backward_policy import (
    BACKWARD_EXP2_POLICY_VERSION,
    BACKWARD_P_TMEM_POLICY_VERSION,
    BACKWARD_RASTER_POLICY_VERSION,
    resolve_backward_exp2_policy,
    resolve_backward_probability_tmem_policy,
    resolve_backward_raster_policy,
)


@pytest.mark.parametrize(
    ("sequence", "expected"),
    (
        (2048, (2, 0)),
        (4096, (1, 2)),
        (8192, (1, 2)),
        (16384, (1, 2)),
    ),
)
def test_d64_auto_dispatch_boundaries(
    sequence: int,
    expected: tuple[int, int],
) -> None:
    policy = resolve_backward_exp2_policy(
        sequence=sequence,
        head_dim=64,
        q_heads=32,
        kv_heads=8,
        lowp=True,
        exp2_degree=2,
        exp2_period=None,
    )
    assert (policy.effective_degree, policy.effective_period) == expected
    assert policy.as_dict()["version"] == BACKWARD_EXP2_POLICY_VERSION


@pytest.mark.parametrize(
    ("degree", "period"),
    ((2, 0), (1, 2), (1, 3)),
)
def test_explicit_period_overrides_d64_dispatch(
    degree: int,
    period: int,
) -> None:
    policy = resolve_backward_exp2_policy(
        sequence=8192,
        head_dim=64,
        lowp=True,
        exp2_degree=degree,
        exp2_period=period,
    )
    assert policy.mode == "explicit"
    assert (policy.effective_degree, policy.effective_period) == (
        degree,
        period,
    )


def test_d128_auto_policy_is_unchanged() -> None:
    policy = resolve_backward_exp2_policy(
        sequence=4096,
        head_dim=128,
        lowp=True,
        exp2_degree=1,
        exp2_period=None,
    )
    assert policy.mode == "legacy_d128_auto"
    assert (policy.effective_degree, policy.effective_period) == (1, 2)


@pytest.mark.parametrize(("q_heads", "kv_heads"), ((16, 4), (64, 16)))
def test_additional_measured_s4096_topologies_use_selective_exp2(
    q_heads: int,
    kv_heads: int,
) -> None:
    policy = resolve_backward_exp2_policy(
        sequence=4096,
        head_dim=64,
        q_heads=q_heads,
        kv_heads=kv_heads,
        lowp=True,
        exp2_degree=2,
        exp2_period=None,
    )
    assert policy.mode == "auto_verified_shape"
    assert (policy.effective_degree, policy.effective_period) == (1, 2)


@pytest.mark.parametrize(
    ("sequence", "q_heads", "kv_heads"),
    (
        (4096, 8, 2),
        (6144, 32, 8),
        (8192, 64, 16),
        (16384, 16, 4),
        (16384, 64, 16),
        (32768, 32, 8),
    ),
)
def test_unverified_long_sequence_shapes_remain_native(
    sequence: int,
    q_heads: int,
    kv_heads: int,
) -> None:
    policy = resolve_backward_exp2_policy(
        sequence=sequence,
        head_dim=64,
        q_heads=q_heads,
        kv_heads=kv_heads,
        lowp=True,
        exp2_degree=2,
        exp2_period=None,
    )
    assert policy.mode == "auto_unverified_shape"
    assert (policy.effective_degree, policy.effective_period) == (2, 0)


def test_bf16_disables_selective_exp2() -> None:
    policy = resolve_backward_exp2_policy(
        sequence=8192,
        head_dim=64,
        lowp=False,
        exp2_degree=1,
        exp2_period=2,
    )
    assert policy.mode == "bf16_disabled"
    assert policy.effective_period == 0


@pytest.mark.parametrize(
    ("sequence", "q_heads", "kv_heads", "expected"),
    (
        (4096, 32, 8, False),
        (8192, 16, 4, False),
        (8192, 32, 8, True),
        (8192, 64, 16, False),
        (16384, 16, 4, False),
        (16384, 32, 8, True),
        (16384, 64, 16, False),
        (32768, 32, 8, False),
    ),
)
def test_d64_head_fast_raster_is_measured_shape_only(
    sequence: int,
    q_heads: int,
    kv_heads: int,
    expected: bool,
) -> None:
    policy = resolve_backward_raster_policy(
        sequence=sequence,
        head_dim=64,
        q_heads=q_heads,
        kv_heads=kv_heads,
        batch=1,
        lowp=True,
        head_fast_raster=None,
        auto_eligible=True,
    )
    assert policy.effective_head_fast is expected
    assert policy.as_dict()["version"] == BACKWARD_RASTER_POLICY_VERSION


def test_explicit_raster_override_and_bf16_control() -> None:
    explicit = resolve_backward_raster_policy(
        sequence=4096,
        head_dim=64,
        q_heads=32,
        kv_heads=8,
        batch=1,
        lowp=True,
        head_fast_raster=True,
    )
    bf16 = resolve_backward_raster_policy(
        sequence=8192,
        head_dim=64,
        q_heads=32,
        kv_heads=8,
        batch=1,
        lowp=False,
        head_fast_raster=True,
    )
    assert explicit.mode == "explicit"
    assert explicit.effective_head_fast is True
    assert bf16.mode == "bf16_key_fast"
    assert bf16.effective_head_fast is False


def test_measured_raster_requires_exact_route_eligibility() -> None:
    policy = resolve_backward_raster_policy(
        sequence=8192,
        head_dim=64,
        q_heads=32,
        kv_heads=8,
        batch=1,
        lowp=True,
        head_fast_raster=None,
        auto_eligible=False,
    )
    assert policy.mode == "auto_key_fast_ineligible_route"
    assert policy.effective_head_fast is False


def test_owner_raster_reports_physical_head_fast_requirement() -> None:
    policy = resolve_backward_raster_policy(
        sequence=4096,
        head_dim=64,
        q_heads=32,
        kv_heads=8,
        batch=1,
        lowp=True,
        head_fast_raster=False,
        force_head_fast=True,
    )
    assert policy.mode == "owner_required"
    assert policy.effective_head_fast is True
    assert policy.forced_head_fast is True


@pytest.mark.parametrize(
    ("sequence", "q_heads", "kv_heads", "expected"),
    (
        (4096, 32, 8, False),
        (8192, 16, 4, False),
        (8192, 32, 8, True),
        (8192, 64, 16, False),
        (16384, 16, 4, False),
        (16384, 32, 8, True),
        (16384, 64, 16, False),
        (32768, 32, 8, False),
    ),
)
def test_detached_p_tmem_is_measured_shape_only(
    sequence: int,
    q_heads: int,
    kv_heads: int,
    expected: bool,
) -> None:
    policy = resolve_backward_probability_tmem_policy(
        sequence=sequence,
        head_dim=64,
        q_heads=q_heads,
        kv_heads=kv_heads,
        batch=1,
        lowp=True,
        detached_fp8_p_tmem=None,
        auto_eligible=True,
    )
    assert policy.effective_detached is expected
    assert policy.as_dict()["version"] == BACKWARD_P_TMEM_POLICY_VERSION


def test_detached_p_requires_route_eligibility_but_allows_explicit_control() -> None:
    automatic = resolve_backward_probability_tmem_policy(
        sequence=8192,
        head_dim=64,
        q_heads=32,
        kv_heads=8,
        batch=1,
        lowp=True,
        detached_fp8_p_tmem=None,
        auto_eligible=False,
    )
    explicit = resolve_backward_probability_tmem_policy(
        sequence=8192,
        head_dim=64,
        q_heads=32,
        kv_heads=8,
        batch=1,
        lowp=True,
        detached_fp8_p_tmem=True,
        auto_eligible=False,
    )
    assert automatic.mode == "auto_alias_p_ineligible_route"
    assert automatic.effective_detached is False
    assert explicit.mode == "explicit"
    assert explicit.effective_detached is True


@pytest.mark.parametrize(
    "overrides",
    (
        {"sequence": 0},
        {"head_dim": 32},
        {"exp2_degree": 3},
        {"exp2_period": 17},
    ),
)
def test_invalid_policy_inputs_are_rejected(overrides: dict[str, int]) -> None:
    arguments = {
        "sequence": 4096,
        "head_dim": 64,
        "lowp": True,
        "exp2_degree": 2,
        "exp2_period": None,
    }
    arguments.update(overrides)
    with pytest.raises(ValueError):
        resolve_backward_exp2_policy(**arguments)
