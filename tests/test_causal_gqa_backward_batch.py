from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from tk_fa4.lowp_fa4_bwd.profile_gqa_d128_chain import (
    VERIFIED_BATCHED_D128_BATCHES,
    VERIFIED_BATCHED_D128_CONTROL_BYTES,
    VERIFIED_BATCHED_D128_CONTROL_SHA256,
    VERIFIED_BATCHED_D64_BATCHES,
    VERIFIED_BATCHED_D64_CONTROL_BYTES,
    VERIFIED_BATCHED_D64_CONTROL_SHA256,
    _is_verified_batched_direct_d64_route,
    _is_verified_batched_shared_d128_route,
    _require_verified_batched_direct_d64_tensors,
)
from tk_fa4.lowp_fa4_bwd.validate_causal_gqa_exact_backward_batch import (
    REPEATABILITY_RELATIVE_L2_LIMIT,
    RepresentedState,
    _repeatability_within_limit,
)


def _route_kwargs() -> dict[str, object]:
    return {
        "batch": 2,
        "sequence": 4096,
        "q_heads": 32,
        "kv_heads": 8,
        "depth": 64,
        "lowp": True,
        "precomputed_stats": True,
        "workspace_stats": True,
        "hierarchical_dq_lanes": 1,
        "signal_dq_tiles": False,
        "owner_output_operand": None,
        "owner_quantize_kv": False,
        "reuse_quantized_p": False,
        "forward_mx_probability_replay": False,
        "forward_mx_probability_scales": None,
        "use_forward_mx_probability_scales": False,
        "reverse_query_tiles": False,
        "head_fast_raster": False,
        "direct_tma_dkdv": True,
        "exp2_degree": 1,
        "exp2_period": 2,
        "fp8_ds_lift": 16,
        "lowp_do_stages": 1,
        "scale_softmax": (64**-0.5) / 16.0,
    }


def _authenticated_control() -> SimpleNamespace:
    return SimpleNamespace(
        TK_FP8_P_STORAGE="tmem",
        TK_DETACHED_FP8_P_TMEM=False,
        TK_PRECOMPOSED_CONTROL_PROVENANCE={
            "mode": "precomposed",
            "source": {
                "sha256": VERIFIED_BATCHED_D64_CONTROL_SHA256,
                "bytes": VERIFIED_BATCHED_D64_CONTROL_BYTES,
            },
        },
    )


def _d128_control() -> SimpleNamespace:
    return SimpleNamespace(
        TK_FP8_P_STORAGE="shared",
        TK_DETACHED_FP8_P_TMEM=False,
        TK_DIRECT_TMA_DKDV=False,
        TK_PRECOMPOSED_CONTROL_PROVENANCE=None,
        TK_GENERATED_CONTROL_SOURCE_IDENTITY={
            "sha256": VERIFIED_BATCHED_D128_CONTROL_SHA256,
            "bytes": VERIFIED_BATCHED_D128_CONTROL_BYTES,
        },
    )


def _d128_route_kwargs() -> dict[str, object]:
    kwargs = _route_kwargs()
    kwargs.update(
        {
            "depth": 128,
            "reuse_quantized_p": True,
            "direct_tma_dkdv": False,
            "exp2_period": 0,
            "fp8_ds_lift": 256,
            "lowp_do_stages": 2,
            "scale_softmax": (128**-0.5) / 16.0,
        }
    )
    return kwargs


@dataclass
class _FakeTensor:
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device = torch.device("cuda:0")
    is_cuda: bool = True
    contiguous: bool = True

    def is_contiguous(self) -> bool:
        return self.contiguous


def _batched_tensor_kwargs() -> dict[str, object]:
    return {
        "batch": 2,
        "sequence": 4096,
        "q_heads": 32,
        "kv_heads": 8,
        "depth": 64,
        "q": _FakeTensor((2, 4096, 32, 64), torch.float8_e4m3fn),
        "k": _FakeTensor((2, 4096, 8, 64), torch.float8_e4m3fn),
        "v": _FakeTensor((2, 4096, 8, 64), torch.float8_e4m3fn),
        "dout": _FakeTensor((2, 4096, 32, 64), torch.float8_e4m3fn),
        "dpsum": _FakeTensor((2, 32, 1, 4096), torch.float32),
        "scaled_lse": _FakeTensor((2, 32, 1, 4096), torch.float32),
    }


def test_represented_b1_sample_owns_disjoint_storage() -> None:
    tensors = {
        name: torch.arange(24, dtype=torch.float32).reshape(3, 2, 4).clone()
        for name in RepresentedState.__dataclass_fields__
    }
    state = RepresentedState(**tensors)
    source_before = {
        name: value.clone() for name, value in tensors.items()
    }

    sample = state.sample(1)
    for name in RepresentedState.__dataclass_fields__:
        source = getattr(state, name)
        copied = getattr(sample, name)
        assert copied.is_contiguous()
        assert (
            copied.untyped_storage().data_ptr()
            != source.untyped_storage().data_ptr()
        )
        copied.zero_()
        assert torch.equal(source, source_before[name])


def test_b2_b8_b16_d64_direct_lane_is_the_only_verified_batched_route() -> None:
    control = _authenticated_control()
    assert VERIFIED_BATCHED_D64_BATCHES == (2, 8, 16)
    assert _is_verified_batched_direct_d64_route(control, **_route_kwargs())
    batch8 = _route_kwargs()
    batch8["batch"] = 8
    assert _is_verified_batched_direct_d64_route(control, **batch8)
    batch16 = _route_kwargs()
    batch16["batch"] = 16
    assert _is_verified_batched_direct_d64_route(control, **batch16)

    rejected_overrides = {
        "batch": (1, 3, 4, 32),
        "sequence": (2048, 8192, 16384),
        "q_heads": (16, 64),
        "kv_heads": (4, 16),
        "depth": (128,),
        "lowp": (False,),
        "precomputed_stats": (False,),
        "workspace_stats": (False,),
        "hierarchical_dq_lanes": (2,),
        "signal_dq_tiles": (True,),
        "owner_output_operand": ((object(), object()),),
        "owner_quantize_kv": (True,),
        "reuse_quantized_p": (True,),
        "forward_mx_probability_replay": (True,),
        "forward_mx_probability_scales": (object(),),
        "use_forward_mx_probability_scales": (True,),
        "reverse_query_tiles": (True,),
        "head_fast_raster": (True, 1, object()),
        "direct_tma_dkdv": (False, 1),
        "exp2_degree": (2,),
        "exp2_period": (0, 1, 3),
        "fp8_ds_lift": (None, 32, 256),
        "lowp_do_stages": (2,),
        "scale_softmax": (64**-0.5, 1.0),
    }
    for name, values in rejected_overrides.items():
        for value in values:
            kwargs = _route_kwargs()
            kwargs[name] = value
            assert not _is_verified_batched_direct_d64_route(
                control, **kwargs
            ), (name, value)


def test_b2_d128_shared_p_lane_is_exact_and_fail_closed() -> None:
    control = _d128_control()
    assert VERIFIED_BATCHED_D128_BATCHES == (2,)
    assert VERIFIED_BATCHED_D128_CONTROL_SHA256 == (
        "cfbd3ad27e5188d39c475abc238b57b5331fc7e631054a7075c7993150c70764"
    )
    assert VERIFIED_BATCHED_D128_CONTROL_BYTES == 221_230
    assert _is_verified_batched_shared_d128_route(
        control, **_d128_route_kwargs()
    )
    rejected_overrides = {
        "batch": (1, 4, 8, 16),
        "sequence": (2048, 8192),
        "depth": (64,),
        "reuse_quantized_p": (False,),
        "direct_tma_dkdv": (True,),
        "exp2_period": (1, 2),
        "lowp_do_stages": (1,),
        "scale_softmax": ((64**-0.5) / 16.0,),
    }
    for name, values in rejected_overrides.items():
        for value in values:
            kwargs = _d128_route_kwargs()
            kwargs[name] = value
            assert not _is_verified_batched_shared_d128_route(
                control, **kwargs
            ), (name, value)
    for broken_control in (
        SimpleNamespace(
            TK_FP8_P_STORAGE="tmem",
            TK_DETACHED_FP8_P_TMEM=False,
            TK_DIRECT_TMA_DKDV=False,
            TK_PRECOMPOSED_CONTROL_PROVENANCE=None,
        ),
        SimpleNamespace(
            TK_FP8_P_STORAGE="shared",
            TK_DETACHED_FP8_P_TMEM=True,
            TK_DIRECT_TMA_DKDV=False,
            TK_PRECOMPOSED_CONTROL_PROVENANCE=None,
            TK_GENERATED_CONTROL_SOURCE_IDENTITY={
                "sha256": VERIFIED_BATCHED_D128_CONTROL_SHA256,
                "bytes": VERIFIED_BATCHED_D128_CONTROL_BYTES,
            },
        ),
        SimpleNamespace(
            TK_FP8_P_STORAGE="shared",
            TK_DETACHED_FP8_P_TMEM=False,
            TK_DIRECT_TMA_DKDV=False,
            TK_PRECOMPOSED_CONTROL_PROVENANCE=None,
            TK_GENERATED_CONTROL_SOURCE_IDENTITY={
                "sha256": "0" * 64,
                "bytes": VERIFIED_BATCHED_D128_CONTROL_BYTES,
            },
        ),
    ):
        assert not _is_verified_batched_shared_d128_route(
            broken_control, **_d128_route_kwargs()
        )


def test_verified_batched_tensor_contract_accepts_the_exact_cuda_abi() -> None:
    _require_verified_batched_direct_d64_tensors(**_batched_tensor_kwargs())


@pytest.mark.parametrize(
    "name",
    ("q", "k", "v", "dout", "dpsum", "scaled_lse"),
)
def test_verified_batched_tensor_contract_rejects_wrong_dtype(name: str) -> None:
    kwargs = _batched_tensor_kwargs()
    tensor = kwargs[name]
    assert isinstance(tensor, _FakeTensor)
    tensor.dtype = (
        torch.bfloat16 if tensor.dtype == torch.float8_e4m3fn else torch.float16
    )
    with pytest.raises(ValueError, match="must have dtype"):
        _require_verified_batched_direct_d64_tensors(**kwargs)


@pytest.mark.parametrize(
    "name",
    ("q", "k", "v", "dout", "dpsum", "scaled_lse"),
)
def test_verified_batched_tensor_contract_rejects_cpu(name: str) -> None:
    kwargs = _batched_tensor_kwargs()
    tensor = kwargs[name]
    assert isinstance(tensor, _FakeTensor)
    tensor.is_cuda = False
    tensor.device = torch.device("cpu")
    with pytest.raises(ValueError, match="must be a CUDA tensor"):
        _require_verified_batched_direct_d64_tensors(**kwargs)


@pytest.mark.parametrize(
    "name",
    ("q", "k", "v", "dout", "dpsum", "scaled_lse"),
)
def test_verified_batched_tensor_contract_rejects_noncontiguous(
    name: str,
) -> None:
    kwargs = _batched_tensor_kwargs()
    tensor = kwargs[name]
    assert isinstance(tensor, _FakeTensor)
    tensor.contiguous = False
    with pytest.raises(ValueError, match="must be contiguous"):
        _require_verified_batched_direct_d64_tensors(**kwargs)


def test_verified_batched_tensor_contract_rejects_mixed_cuda_devices() -> None:
    kwargs = _batched_tensor_kwargs()
    tensor = kwargs["scaled_lse"]
    assert isinstance(tensor, _FakeTensor)
    tensor.device = torch.device("cuda:1")
    with pytest.raises(ValueError, match="must share one CUDA device"):
        _require_verified_batched_direct_d64_tensors(**kwargs)


def _repeatability(relative_l2: float) -> dict[str, object]:
    return {
        "sample_0": {
            "aggregate": {
                "reference_finite": True,
                "actual_finite": True,
                "relative_l2": relative_l2,
            }
        }
    }


def test_repeatability_gate_has_a_fixed_finite_ceiling() -> None:
    assert REPEATABILITY_RELATIVE_L2_LIMIT == 0.005
    assert _repeatability_within_limit(
        _repeatability(REPEATABILITY_RELATIVE_L2_LIMIT)
    )
    assert not _repeatability_within_limit(
        _repeatability(REPEATABILITY_RELATIVE_L2_LIMIT + 1.0e-6)
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (("reference_finite", False), ("actual_finite", False)),
)
def test_repeatability_gate_rejects_nonfinite_launches(
    field: str,
    value: bool,
) -> None:
    repeatability = _repeatability(0.0)
    repeatability["sample_0"]["aggregate"][field] = value
    assert not _repeatability_within_limit(repeatability)


def test_repeatability_gate_rejects_invalid_limits() -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        _repeatability_within_limit(_repeatability(0.0), limit=-1.0)
    assert not _repeatability_within_limit({})


def test_batched_route_rejects_non_tmem_or_detached_probability_storage() -> None:
    kwargs = _route_kwargs()
    assert not _is_verified_batched_direct_d64_route(
        SimpleNamespace(
            TK_FP8_P_STORAGE="shared",
            TK_DETACHED_FP8_P_TMEM=False,
            TK_PRECOMPOSED_CONTROL_PROVENANCE=(
                _authenticated_control().TK_PRECOMPOSED_CONTROL_PROVENANCE
            ),
        ),
        **kwargs,
    )
    assert not _is_verified_batched_direct_d64_route(
        SimpleNamespace(
            TK_FP8_P_STORAGE="tmem",
            TK_DETACHED_FP8_P_TMEM=True,
            TK_PRECOMPOSED_CONTROL_PROVENANCE=(
                _authenticated_control().TK_PRECOMPOSED_CONTROL_PROVENANCE
            ),
        ),
        **kwargs,
    )


def test_batched_route_requires_authenticated_precomposed_control() -> None:
    kwargs = _route_kwargs()
    for provenance in (None, {}, {"mode": "generated_patch_chain"}):
        control = SimpleNamespace(
            TK_FP8_P_STORAGE="tmem",
            TK_DETACHED_FP8_P_TMEM=False,
            TK_PRECOMPOSED_CONTROL_PROVENANCE=provenance,
        )
        assert not _is_verified_batched_direct_d64_route(control, **kwargs)


def test_batched_route_requires_the_pinned_control_identity() -> None:
    kwargs = _route_kwargs()
    valid_source = {
        "sha256": VERIFIED_BATCHED_D64_CONTROL_SHA256,
        "bytes": VERIFIED_BATCHED_D64_CONTROL_BYTES,
    }
    rejected_sources = (
        None,
        {},
        {**valid_source, "sha256": "0" * 64},
        {**valid_source, "bytes": VERIFIED_BATCHED_D64_CONTROL_BYTES + 1},
    )
    for source in rejected_sources:
        control = SimpleNamespace(
            TK_FP8_P_STORAGE="tmem",
            TK_DETACHED_FP8_P_TMEM=False,
            TK_PRECOMPOSED_CONTROL_PROVENANCE={
                "mode": "precomposed",
                "source": source,
            },
        )
        assert not _is_verified_batched_direct_d64_route(control, **kwargs)
