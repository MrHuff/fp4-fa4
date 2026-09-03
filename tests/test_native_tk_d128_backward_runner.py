from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import pytest
import torch

from tk_fa4.lowp_fa4_bwd import native_tk_d128_backward as native_module
from tk_fa4.lowp_fa4_bwd.native_tk_d128_backward import (
    BACKEND,
    EXPECTED_EXTENSION_METADATA,
    HEAD_DIM,
    KV_HEADS,
    Q_HEADS,
    SEQUENCE,
    SOFTMAX_SCALE,
    V501_SOURCE_IDENTITY,
    NativeTkD128E4M3Backward,
    _require_extension_metadata,
)


def _exact_metadata() -> dict[str, Any]:
    return {
        **EXPECTED_EXTENSION_METADATA,
        "topology": (
            "unified_b1_v488_b2_v490_key_tile_major_head_owner"
        ),
        "source_file": (
            "../native_gqa_tk_bwd/"
            "v501_d128_gqa_e4m3_unified_best_route_production_bshd.cu"
        ),
    }


class _FakeExtension:
    def __init__(
        self,
        metadata: dict[str, Any] | None = None,
        *,
        identity: dict[str, Any] | None = None,
    ) -> None:
        self.metadata = _exact_metadata() if metadata is None else metadata
        self.calls: list[tuple[Any, ...]] = []
        self._tk_fa4_loaded_artifact_identity = (
            {
                "path": "/tmp/native-d128.so",
                "sha256": "a" * 64,
                "bytes": 123,
                "device": 1,
                "inode": 2,
                "mtime_ns": 3,
            }
            if identity is None
            else identity
        )

    def native_tk_d128_backward_metadata(self) -> dict[str, Any]:
        return dict(self.metadata)

    def backward_e4m3_bshd_precomputed_out(self, *args: Any) -> None:
        self.calls.append(("out", *args))

    def main_e4m3_bshd_precomputed(self, *args: Any) -> None:
        self.calls.append(("main", *args))


def test_native_extension_metadata_is_exact_and_copied() -> None:
    metadata = _exact_metadata()

    observed = _require_extension_metadata(_FakeExtension(metadata))

    assert observed == metadata
    assert observed is not metadata
    assert observed["source_identity"] == V501_SOURCE_IDENTITY
    assert observed["batch_values"] == (1, 2)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("schema", "tkfa4.native_tk_d128_backward.v2"),
        ("batch_values", [1, 2]),
        ("caller_owned_output_api", 1),
        ("exact_sequence_specialization", 2048),
        ("dispatch", "B1_S4096_v488;B2_S4096_v488"),
        ("output_dtype", "bfloat16"),
        ("source_file", "wrong.cu"),
        ("topology", ""),
        ("topology", 501),
    ),
)
def test_native_extension_metadata_rejects_production_abi_drift(
    field: str,
    value: Any,
) -> None:
    metadata = _exact_metadata()
    metadata[field] = value

    with pytest.raises(RuntimeError, match="production ABI"):
        _require_extension_metadata(_FakeExtension(metadata))


@pytest.mark.parametrize("missing", ("dispatch", "source_file", "topology"))
def test_native_extension_metadata_rejects_missing_fields(
    missing: str,
) -> None:
    metadata = _exact_metadata()
    del metadata[missing]

    with pytest.raises(RuntimeError, match="incomplete"):
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
        storage: object | None = None,
    ) -> None:
        self.shape = (shape,) if isinstance(shape, int) else tuple(shape)
        self.dtype = dtype
        self.device = torch.device(device)
        self.storage = object() if storage is None else storage

    @property
    def numel(self) -> int:
        result = 1
        for extent in self.shape:
            result *= extent
        return result

    def view(self, *shape_or_dtype: Any) -> _FakeAllocation:
        if len(shape_or_dtype) == 1 and isinstance(
            shape_or_dtype[0], torch.dtype
        ):
            dtype = shape_or_dtype[0]
            nbytes = self.numel * self.dtype.itemsize
            assert nbytes % dtype.itemsize == 0
            shape = (nbytes // dtype.itemsize,)
        else:
            dtype = self.dtype
            shape = tuple(int(extent) for extent in shape_or_dtype)
            assert self.numel == _numel(shape)
        return _FakeAllocation(
            shape,
            dtype=dtype,
            device=self.device,
            storage=self.storage,
        )

    def __getitem__(self, index: Any) -> _FakeAllocation:
        assert isinstance(index, slice)
        start, stop, step = index.indices(self.numel)
        assert step == 1
        return _FakeAllocation(
            (max(0, stop - start),),
            dtype=self.dtype,
            device=self.device,
            storage=self.storage,
        )


def _numel(shape: tuple[int, ...]) -> int:
    result = 1
    for extent in shape:
        result *= extent
    return result


def _patch_allocations(
    monkeypatch: pytest.MonkeyPatch,
) -> list[_FakeAllocation]:
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
    return allocations


@pytest.mark.parametrize(
    ("batch", "heads_per_owner", "unique_writer", "publisher"),
    (
        (1, 2, False, "owner_reducer"),
        (2, 4, True, "dedicated_warp14"),
    ),
)
def test_native_runner_allocates_direct_b1_b2_storage(
    monkeypatch: pytest.MonkeyPatch,
    batch: int,
    heads_per_owner: int,
    unique_writer: bool,
    publisher: str,
) -> None:
    allocations = _patch_allocations(monkeypatch)
    identity = {
        "path": "/tmp/v501.so",
        "sha256": "5" * 64,
        "bytes": 501,
    }
    extension = _FakeExtension(identity=identity)

    runner = NativeTkD128E4M3Backward(
        extension,
        batch=batch,
        device="cuda:0",
    )

    stats_numel = batch * Q_HEADS * SEQUENCE
    assert runner.workspace_torch.shape == (
        2 * stats_numel * torch.float32.itemsize,
    )
    assert runner.workspace_torch.dtype == torch.uint8
    assert runner.dstat.shape == (batch, Q_HEADS, 1, SEQUENCE)
    assert runner.lstat.shape == (batch, Q_HEADS, 1, SEQUENCE)
    assert runner.dstat.dtype == runner.lstat.dtype == torch.float32
    assert runner.dstat.storage is runner.workspace_torch.storage
    assert runner.lstat.storage is runner.workspace_torch.storage
    assert runner.dq.shape == (batch, SEQUENCE, Q_HEADS, HEAD_DIM)
    assert runner.dk.shape == (batch, SEQUENCE, KV_HEADS, HEAD_DIM)
    assert runner.dv.shape == runner.dk.shape
    assert runner.dq.dtype == runner.dk.dtype == runner.dv.dtype
    assert runner.dq.dtype == torch.bfloat16
    assert runner.dk_partials is runner.dv_partials
    assert runner.dk_partials.shape == (0,)
    assert runner.dk_partials.dtype == torch.float32
    assert runner.direct_tma_dkdv is True
    assert runner.raster_policy == {
        "backend": BACKEND,
        "owner_order": "key_tile_major_head_owner",
        "host_dispatch_per_launch": False,
        "heads_per_owner": heads_per_owner,
    }
    assert not any(
        allocation.dtype == torch.float32 and allocation.shape != (0,)
        for allocation in allocations
    )
    contract = runner.contract()
    assert contract["schedule"]["direct_dkdv_unique_writer"] is unique_writer
    assert contract["schedule"]["gradient_publisher"] == publisher

    identity["sha256"] = "6" * 64
    assert runner.loaded_artifact_identity["sha256"] == "5" * 64


@pytest.mark.parametrize("batch", (True, 1.0, 0, 3, -1))
def test_native_runner_rejects_unsupported_batch(batch: Any) -> None:
    with pytest.raises(ValueError, match="batch 1 or 2"):
        NativeTkD128E4M3Backward(
            _FakeExtension(),
            batch=batch,
            device="cuda:0",
        )


def test_native_runner_rejects_non_cuda_device() -> None:
    with pytest.raises(ValueError, match="CUDA device"):
        NativeTkD128E4M3Backward(
            _FakeExtension(),
            batch=1,
            device="cpu",
        )


def test_native_runner_requires_direct_output_entrypoint() -> None:
    extension = _FakeExtension()
    extension.backward_e4m3_bshd_precomputed_out = None  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="precomputed_out"):
        NativeTkD128E4M3Backward(
            extension,
            batch=1,
            device="cuda:0",
        )


def test_native_runner_requires_precleared_main_entrypoint() -> None:
    extension = _FakeExtension()
    extension.main_e4m3_bshd_precomputed = None  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="main_e4m3_bshd_precomputed"):
        NativeTkD128E4M3Backward(
            extension,
            batch=2,
            device="cuda:0",
        )


def _bare_runner(*, batch: int = 1) -> NativeTkD128E4M3Backward:
    runner = object.__new__(NativeTkD128E4M3Backward)
    runner.batch = batch
    runner.device = torch.device("cuda:0")
    runner.loaded_artifact_identity = {
        "path": "/tmp/native-d128.so",
        "sha256": "a" * 64,
        "bytes": 123,
        "device": 1,
        "inode": 2,
        "mtime_ns": 3,
    }
    runner.extension_metadata = _exact_metadata()
    runner.direct_tma_dkdv = True
    runner._q = None
    runner._k = None
    runner._v = None
    runner._dout = None
    runner._bind_generation = 0
    runner._run_generation = 0
    return runner


@pytest.mark.parametrize(
    ("batch", "reset", "expected_entrypoint"),
    (
        (1, False, "out"),
        (1, True, "out"),
        (2, True, "out"),
        (2, False, "main"),
    ),
)
def test_native_runner_selects_safe_entrypoint_and_exact_abi(
    batch: int,
    reset: bool,
    expected_entrypoint: str,
) -> None:
    runner = _bare_runner(batch=batch)
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
    calls: list[tuple[str, tuple[Any, ...]]] = []
    runner.compiled_out = lambda *args: calls.append(("out", args))
    runner.compiled_main = lambda *args: calls.append(("main", args))

    runner.run(reset=reset)

    assert calls == [(expected_entrypoint, (*values, SOFTMAX_SCALE))]
    assert runner._run_generation == 1


@pytest.mark.parametrize("reset", (None, 0, 1, "false"))
def test_native_runner_run_requires_exact_bool(reset: Any) -> None:
    runner = _bare_runner()
    runner.compiled_out = pytest.fail
    runner.compiled_main = pytest.fail

    with pytest.raises(TypeError, match="exactly bool"):
        runner.run(reset=reset)


def test_native_runner_run_requires_bound_operands() -> None:
    runner = _bare_runner()
    calls: list[tuple[Any, ...]] = []
    runner.compiled_out = lambda *args: calls.append(args)
    runner.compiled_main = lambda *args: calls.append(args)

    with pytest.raises(RuntimeError, match=r"bind_inputs\(\)"):
        runner.run(reset=False)

    assert calls == []
    assert runner._run_generation == 0


def test_native_runner_reset_and_mx_receipts_are_noops() -> None:
    runner = _bare_runner()
    sentinels = [object() for _ in range(4)]
    runner._q, runner._k, runner._v, runner._dout = sentinels
    runner._bind_generation = 7
    runner._run_generation = 5

    runner.reset()

    assert [runner._q, runner._k, runner._v, runner._dout] == sentinels
    assert runner._bind_generation == 7
    assert runner._run_generation == 5
    assert runner.d128_mxfp4_v_operand_cache_receipt() is None
    assert runner.d128_mxfp4_v_compilation_receipt() is None


@dataclass(frozen=True)
class _FakeTensor:
    shape: tuple[int, ...]
    dtype: torch.dtype = torch.float8_e4m3fn
    is_cuda: bool = True
    contiguous_value: bool = True
    device: torch.device = torch.device("cuda:0")

    def is_contiguous(self) -> bool:
        return self.contiguous_value


def _fake_operands(batch: int = 1) -> tuple[_FakeTensor, ...]:
    return (
        _FakeTensor((batch, SEQUENCE, Q_HEADS, HEAD_DIM)),
        _FakeTensor((batch, SEQUENCE, KV_HEADS, HEAD_DIM)),
        _FakeTensor((batch, SEQUENCE, KV_HEADS, HEAD_DIM)),
        _FakeTensor((batch, SEQUENCE, Q_HEADS, HEAD_DIM)),
    )


def test_native_runner_bind_is_reference_only() -> None:
    runner = _bare_runner(batch=2)
    q, k, v, dout = _fake_operands(batch=2)

    runner.bind_inputs(q, k, v, dout)  # type: ignore[arg-type]

    assert runner._q is q
    assert runner._k is k
    assert runner._v is v
    assert runner._dout is dout
    assert runner._bind_generation == 1


@pytest.mark.parametrize(
    ("operand_index", "name", "changes"),
    (
        (0, "q", {"dtype": torch.bfloat16}),
        (1, "k", {"is_cuda": False}),
        (2, "v", {"contiguous_value": False}),
        (3, "dout", {"shape": (1, SEQUENCE, Q_HEADS, HEAD_DIM + 1)}),
        (0, "q", {"device": torch.device("cuda:1")}),
    ),
)
def test_native_runner_invalid_bind_is_atomic(
    operand_index: int,
    name: str,
    changes: dict[str, Any],
) -> None:
    runner = _bare_runner()
    existing = [object() for _ in range(4)]
    runner._q, runner._k, runner._v, runner._dout = existing
    operands = list(_fake_operands())
    operands[operand_index] = replace(operands[operand_index], **changes)

    with pytest.raises(ValueError, match=rf"{name} must be"):
        runner.bind_inputs(*operands)  # type: ignore[arg-type]

    assert [runner._q, runner._k, runner._v, runner._dout] == existing
    assert runner._bind_generation == 0


@pytest.mark.parametrize(
    ("batch", "unique_writer", "publisher"),
    (
        (1, False, "owner_reducer"),
        (2, True, "dedicated_warp14"),
    ),
)
def test_native_runner_contract_is_exact_and_value_copied(
    batch: int,
    unique_writer: bool,
    publisher: str,
) -> None:
    runner = _bare_runner(batch=batch)

    contract = runner.contract()

    assert contract == {
        "backend": BACKEND,
        "extension": runner.loaded_artifact_identity,
        "extension_metadata": runner.extension_metadata,
        "shape": {
            "batch": batch,
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
            "dispatch": EXPECTED_EXTENSION_METADATA["dispatch"],
            "owner_order": "key_tile_major_head_owner",
            "direct_dkdv_unique_writer": unique_writer,
            "gradient_publisher": publisher,
        },
        "allocation": {
            "scope": "native_backward_runner_only",
            "caller_owned_runner_storage": True,
            "native_run_allocations": False,
            "native_run_dlpack_wrappers": False,
            "external_projection_publication": "authenticated_e4m3_bshd",
        },
    }
    assert contract["extension"] is not runner.loaded_artifact_identity
    assert contract["extension_metadata"] is not runner.extension_metadata
    contract["extension"]["sha256"] = "b" * 64
    contract["extension_metadata"]["topology"] = "wrong"
    assert runner.loaded_artifact_identity["sha256"] == "a" * 64
    assert runner.extension_metadata["topology"] == (
        "unified_b1_v488_b2_v490_key_tile_major_head_owner"
    )
