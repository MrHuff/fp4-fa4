from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import pytest
import torch

from tk_fa4.lowp_fa4_bwd import native_tk_d64_backward as native_module
from tk_fa4.lowp_fa4_bwd.native_tk_d64_backward import (
    BACKEND,
    EXPECTED_EXTENSION_METADATA,
    HEAD_DIM,
    KV_HEADS,
    PARTIAL_HEADS,
    Q_HEADS,
    SEQUENCE,
    V414_EXPECTED_EXTENSION_METADATA,
    V414_SOURCE_IDENTITY,
    V416_EXPECTED_EXTENSION_METADATA,
    V416_SOURCE_IDENTITY,
    NativeTkD64E4M3Backward,
    _require_extension_metadata,
)


def _bare_runner() -> NativeTkD64E4M3Backward:
    runner = object.__new__(NativeTkD64E4M3Backward)
    runner.batch = 1
    runner.device = torch.device("cuda:0")
    runner.loaded_artifact_identity = {
        "path": "/tmp/native.so",
        "sha256": "a" * 64,
        "bytes": 123,
        "device": 1,
        "inode": 2,
        "mtime_ns": 3,
    }
    runner.extension_metadata = {
        **EXPECTED_EXTENSION_METADATA,
        "source_file": (
            "/src/v382_d64_gqa_e4m3_hkv2_register_pd.cu"
        ),
    }
    runner.direct_bf16_outputs = False
    runner._q = None
    runner._k = None
    runner._v = None
    runner._dout = None
    runner._bind_generation = 0
    runner._run_generation = 0
    return runner


class _FakeExtension:
    def __init__(self, metadata: dict[str, Any]) -> None:
        self._metadata = metadata

    def native_tk_d64_backward_metadata(self) -> dict[str, Any]:
        return dict(self._metadata)


def _exact_metadata() -> dict[str, Any]:
    return {
        **EXPECTED_EXTENSION_METADATA,
        "source_file": (
            "../native_gqa_tk_bwd/"
            "v382_d64_gqa_e4m3_hkv2_register_pd.cu"
        ),
    }


def _v414_metadata() -> dict[str, Any]:
    return {
        **V414_EXPECTED_EXTENSION_METADATA,
        "source_file": (
            "../native_gqa_tk_bwd/"
            "v414_d64_gqa_e4m3_production_bshd_dq_first.cu"
        ),
    }


def _v416_metadata() -> dict[str, Any]:
    return {
        **V416_EXPECTED_EXTENSION_METADATA,
        "source_file": (
            "../native_gqa_tk_bwd/"
            "v416_d64_gqa_e4m3_production_bshd_dq_first_vec2_ds.cu"
        ),
    }


def test_native_extension_metadata_is_exact() -> None:
    metadata = _exact_metadata()
    observed = _require_extension_metadata(_FakeExtension(metadata))
    assert observed == metadata
    assert observed is not metadata


def test_v414_extension_metadata_authenticates_production_abi() -> None:
    metadata = _v414_metadata()
    observed = _require_extension_metadata(_FakeExtension(metadata))
    assert observed == metadata
    assert observed["source_identity"] == V414_SOURCE_IDENTITY


def test_v416_extension_metadata_authenticates_vec2_production_abi() -> None:
    metadata = _v416_metadata()
    observed = _require_extension_metadata(_FakeExtension(metadata))
    assert observed == metadata
    assert observed["source_identity"] == V416_SOURCE_IDENTITY
    assert observed["ds_shared_store"] == "owner_aligned_v2_b32"

    metadata["ds_shared_store"] = "owner_aligned_b32"
    with pytest.raises(RuntimeError, match="retained ABI"):
        _require_extension_metadata(_FakeExtension(metadata))


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("gradient_mma_issue_order", "dk_then_dq"),
        ("probability_exp", "native_ex2_clamp_log2_p_le_0"),
        ("backward_out_clears_outputs", False),
        ("source_file", "wrong.cu"),
    ),
)
def test_v414_extension_metadata_rejects_abi_drift(
    field: str,
    value: Any,
) -> None:
    metadata = _v414_metadata()
    metadata[field] = value
    with pytest.raises(RuntimeError, match="production ABI|retained ABI"):
        _require_extension_metadata(_FakeExtension(metadata))


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("topology", "wrong"),
        ("sequence", 2048),
        ("encoding_scale", 2.0),
        ("caller_owned_output_api", 1),
        ("source_file", "wrong.cu"),
    ),
)
def test_native_extension_metadata_rejects_abi_drift(
    field: str,
    value: Any,
) -> None:
    metadata = _exact_metadata()
    metadata[field] = value
    with pytest.raises(RuntimeError, match="retained ABI"):
        _require_extension_metadata(_FakeExtension(metadata))


def test_native_extension_metadata_rejects_missing_receipt() -> None:
    with pytest.raises(RuntimeError, match="lacks"):
        _require_extension_metadata(object())


class _FakeAllocation:
    def __init__(
        self,
        shape: int | tuple[int, ...],
        *,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        self.shape = (shape,) if isinstance(shape, int) else tuple(shape)
        self.dtype = dtype
        self.device = torch.device(device)

    def view(self, *_shape_or_dtype: Any) -> _FakeAllocation:
        return self

    def __getitem__(self, _index: Any) -> _FakeAllocation:
        return self


def test_native_runner_allocates_hkv2_partial_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allocations: list[_FakeAllocation] = []

    def fake_empty(
        shape: int | tuple[int, ...],
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> _FakeAllocation:
        allocation = _FakeAllocation(shape, dtype=dtype, device=device)
        allocations.append(allocation)
        return allocation

    def fake_empty_like(
        source: _FakeAllocation,
        *,
        dtype: torch.dtype | None = None,
    ) -> _FakeAllocation:
        allocation = _FakeAllocation(
            source.shape,
            dtype=source.dtype if dtype is None else dtype,
            device=source.device,
        )
        allocations.append(allocation)
        return allocation

    monkeypatch.setattr(native_module.torch, "empty", fake_empty)
    monkeypatch.setattr(native_module.torch, "empty_like", fake_empty_like)
    extension = _FakeExtension(_exact_metadata())
    extension.backward_e4m3_precomputed_out = lambda *_args: None
    extension._tk_fa4_loaded_artifact_identity = {
        "path": "/tmp/native.so",
        "sha256": "a" * 64,
        "bytes": 123,
        "device": 1,
        "inode": 2,
        "mtime_ns": 3,
    }

    runner = NativeTkD64E4M3Backward(
        extension,
        batch=16,
        device="cuda:0",
    )

    assert runner.dk_partials.shape == (
        16,
        SEQUENCE,
        PARTIAL_HEADS,
        HEAD_DIM,
    )
    assert runner.dv_partials.shape == runner.dk_partials.shape
    assert runner.dk_partials.dtype == torch.float32
    assert runner.dv_partials.dtype == torch.float32
    assert sum(
        allocation.shape == runner.dk_partials.shape
        and allocation.dtype == torch.float32
        for allocation in allocations
    ) == 2


def test_v414_runner_replaces_gib_workspace_with_empty_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allocations: list[_FakeAllocation] = []

    def fake_empty(
        shape: int | tuple[int, ...],
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> _FakeAllocation:
        allocation = _FakeAllocation(shape, dtype=dtype, device=device)
        allocations.append(allocation)
        return allocation

    def fake_empty_like(
        source: _FakeAllocation,
        *,
        dtype: torch.dtype | None = None,
    ) -> _FakeAllocation:
        allocation = _FakeAllocation(
            source.shape,
            dtype=source.dtype if dtype is None else dtype,
            device=source.device,
        )
        allocations.append(allocation)
        return allocation

    monkeypatch.setattr(native_module.torch, "empty", fake_empty)
    monkeypatch.setattr(native_module.torch, "empty_like", fake_empty_like)
    extension = _FakeExtension(_v414_metadata())
    extension.backward_e4m3_bshd_precomputed_out = lambda *_args: None

    runner = NativeTkD64E4M3Backward(
        extension,
        batch=16,
        device="cuda:0",
    )

    assert runner.direct_bf16_outputs is True
    assert runner.dq_accum is runner.dk_partials
    assert runner.dk_partials is runner.dv_partials
    assert runner.dk_partials.shape == (0,)
    assert runner.dk_partials.dtype == torch.float32
    assert not any(
        allocation.shape
        in {
            (16, SEQUENCE, Q_HEADS, HEAD_DIM),
            (16, SEQUENCE, PARTIAL_HEADS, HEAD_DIM),
        }
        and allocation.dtype == torch.float32
        for allocation in allocations
    )


def test_native_runner_calls_exact_out_abi() -> None:
    runner = _bare_runner()
    values = [object() for _ in range(12)]
    (
        runner._q,
        runner._k,
        runner._v,
        runner._dout,
        runner.lstat,
        runner.dstat,
        runner.dq_accum,
        runner.dk_partials,
        runner.dv_partials,
        runner.dq,
        runner.dk,
        runner.dv,
    ) = values
    calls: list[tuple[Any, ...]] = []
    runner.compiled = lambda *args: calls.append(args)

    runner.run(reset=False)

    assert calls == [(*values, HEAD_DIM**-0.5)]
    assert runner._run_generation == 1


def test_v414_runner_calls_direct_bf16_out_abi() -> None:
    runner = _bare_runner()
    runner.direct_bf16_outputs = True
    values = [object() for _ in range(9)]
    (
        runner._q,
        runner._k,
        runner._v,
        runner._dout,
        runner.lstat,
        runner.dstat,
        runner.dq,
        runner.dk,
        runner.dv,
    ) = values
    calls: list[tuple[Any, ...]] = []
    runner.compiled = lambda *args: calls.append(args)

    runner.run(reset=True)

    assert calls == [(*values, HEAD_DIM**-0.5)]
    assert runner._run_generation == 1


@pytest.mark.parametrize("reset", (None, 0, 1, "false"))
def test_native_runner_run_requires_exact_bool(reset: Any) -> None:
    runner = _bare_runner()
    runner.compiled = pytest.fail
    with pytest.raises(TypeError, match="exactly bool"):
        runner.run(reset=reset)


def test_native_runner_run_requires_bound_operands() -> None:
    runner = _bare_runner()
    calls: list[tuple[Any, ...]] = []
    runner.compiled = lambda *args: calls.append(args)
    with pytest.raises(RuntimeError, match=r"bind_inputs\(\)"):
        runner.run(reset=False)
    assert calls == []
    assert runner._run_generation == 0


def test_native_runner_reset_is_storage_preserving_noop() -> None:
    runner = _bare_runner()
    sentinels = [object() for _ in range(4)]
    runner._q, runner._k, runner._v, runner._dout = sentinels
    runner._bind_generation = 7
    runner._run_generation = 5

    runner.reset()

    assert [runner._q, runner._k, runner._v, runner._dout] == sentinels
    assert runner._bind_generation == 7
    assert runner._run_generation == 5


def test_native_runner_exposes_empty_d128_mxfp4_v_receipts() -> None:
    runner = _bare_runner()

    assert runner.d128_mxfp4_v_operand_cache_receipt() is None
    assert runner.d128_mxfp4_v_compilation_receipt() is None


@dataclass(frozen=True)
class _FakeTensor:
    shape: tuple[int, ...]
    dtype: torch.dtype = torch.float8_e4m3fn
    is_cuda: bool = True
    contiguous: bool = True
    device: torch.device = torch.device("cuda:0")

    def is_contiguous(self) -> bool:
        return self.contiguous


def _fake_operands() -> tuple[_FakeTensor, ...]:
    return (
        _FakeTensor((1, SEQUENCE, Q_HEADS, HEAD_DIM)),
        _FakeTensor((1, SEQUENCE, KV_HEADS, HEAD_DIM)),
        _FakeTensor((1, SEQUENCE, KV_HEADS, HEAD_DIM)),
        _FakeTensor((1, SEQUENCE, Q_HEADS, HEAD_DIM)),
    )


def test_native_runner_bind_is_reference_only() -> None:
    runner = _bare_runner()
    q, k, v, dout = _fake_operands()

    runner.bind_inputs(q, k, v, dout)  # type: ignore[arg-type]

    assert runner._q is q
    assert runner._k is k
    assert runner._v is v
    assert runner._dout is dout
    assert runner._bind_generation == 1


@pytest.mark.parametrize(
    "invalid",
    (
        {"dtype": torch.bfloat16},
        {"is_cuda": False},
        {"contiguous": False},
        {"shape": (1, SEQUENCE, Q_HEADS, HEAD_DIM + 1)},
        {"device": torch.device("cuda:1")},
    ),
)
def test_native_runner_invalid_bind_is_atomic(invalid: dict[str, Any]) -> None:
    runner = _bare_runner()
    existing = [object() for _ in range(4)]
    runner._q, runner._k, runner._v, runner._dout = existing
    q, k, v, dout = _fake_operands()
    q = replace(q, **invalid)

    with pytest.raises(ValueError, match="q must be"):
        runner.bind_inputs(q, k, v, dout)  # type: ignore[arg-type]

    assert [runner._q, runner._k, runner._v, runner._dout] == existing
    assert runner._bind_generation == 0


def test_native_runner_contract_is_exact_and_value_copied() -> None:
    runner = _bare_runner()

    contract = runner.contract()

    assert contract["backend"] == BACKEND
    assert contract["shape"] == {
        "batch": 1,
        "sequence": 4096,
        "q_heads": 32,
        "kv_heads": 8,
        "head_dim": 64,
    }
    assert contract["input"] == {
        "dtype": "torch.float8_e4m3fn",
        "layout": "BSHD_contiguous",
        "encoding_scale": 4.0,
    }
    assert contract["statistics"] == {
        "workspace_page_0": "-16_sum_O_dO",
        "workspace_page_1": "8_minus_LSE_log2e",
        "producer_native": True,
    }
    assert contract["output"] == {
        "dtype": "torch.bfloat16",
        "layout": "BSHD_contiguous",
        "encoding_scale": 4.0,
    }
    assert contract["schedule"] == {
        "main": "two_CTA_two_query_heads_per_owner_register_p_ds",
        "owner_order": "causal_longest_first",
        "finalize_dq": True,
        "merge_dk_dv": True,
    }
    assert contract["allocation"] == {
        "scope": "native_backward_runner_only",
        "caller_owned_runner_storage": True,
        "native_run_allocations": False,
        "native_run_dlpack_wrappers": False,
        "external_projection_publication": "not_claimed",
    }
    assert contract["workspace"] == {
        "dq_accum_dtype": "torch.float32",
        "dk_dv_partial_dtype": "torch.float32",
        "dk_dv_partial_heads": PARTIAL_HEADS,
    }
    contract["extension"]["sha256"] = "b" * 64
    assert runner.loaded_artifact_identity["sha256"] == "a" * 64
    contract["extension_metadata"]["topology"] = "wrong"
    assert runner.extension_metadata["topology"] == (
        "v382_owner_major_hkv2_register_p_ds"
    )


def test_v414_runner_contract_reports_direct_kernel_storage() -> None:
    runner = _bare_runner()
    runner.direct_bf16_outputs = True
    runner.extension_metadata = _v414_metadata()

    contract = runner.contract()

    assert contract["backend"] == BACKEND
    assert contract["schedule"] == {
        "main": "split_qhead_cta_k128_q128_owner_aligned_dq_first",
        "owner_order": "key_tile_major_query_head",
        "finalize_dq": False,
        "merge_dk_dv": False,
    }
    assert contract["workspace"] == {
        "gradient_accumulation": "kernel_tmem",
        "direct_output_dtype": "torch.bfloat16",
        "host_visible_partials": False,
    }
