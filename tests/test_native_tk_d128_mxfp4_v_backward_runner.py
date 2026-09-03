from __future__ import annotations

import gc
import weakref
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import tk_fa4.interface as interface
from tk_fa4.lowp_fa4_bwd import (
    native_tk_d128_mxfp4_v_backward as native_module,
)
from tk_fa4.lowp_fa4_bwd.native_tk_d128_mxfp4_v_backward import (
    BACKEND,
    BATCH,
    EXPECTED_EXTENSION_METADATA,
    EXPECTED_SHARED_TILE_PRODUCER_ATTRIBUTES,
    HEAD_DIM,
    KV_HEADS,
    Q_HEADS,
    SEQUENCE,
    SHARED_TILE_PRODUCER_CONTRACT_SCHEMA,
    SHARED_TILE_V503_BACKEND,
    SHARED_TILE_V503_COMPOSITION_SCHEMA,
    SOFTMAX_SCALE,
    V503_SOURCE_IDENTITY,
    NativeTkD128Mxfp4VBackward,
    NativeTkD128SharedTileProducerV503Backward,
    _require_extension_metadata,
    require_shared_tile_v503_producer_contract,
)


def _exact_metadata() -> dict[str, Any]:
    return {
        **EXPECTED_EXTENSION_METADATA,
        "source_file": (
            "../native_gqa_tk_bwd/"
            "v503_d128_gqa_mxfp4v_rowscale_e4m3do_b2_s4096_"
            "owner4_experimental_bshd.cu"
        ),
    }


class _FakeExtension:
    def __init__(self, metadata: dict[str, Any] | None = None) -> None:
        self.metadata = _exact_metadata() if metadata is None else metadata
        self.calls: list[tuple[Any, ...]] = []
        self._tk_fa4_loaded_artifact_identity = {
            "path": "/tmp/v503.so",
            "sha256": "5" * 64,
            "bytes": 503,
            "device": 1,
            "inode": 2,
            "mtime_ns": 3,
        }

    def native_tk_d128_backward_metadata(self) -> dict[str, Any]:
        return dict(self.metadata)

    def backward_mxfp4v_e4m3do_bshd_precomputed_out(
        self, *args: Any
    ) -> None:
        self.calls.append(("out", *args))

    def main_mxfp4v_e4m3do_bshd_precomputed(self, *args: Any) -> None:
        self.calls.append(("main", *args))


def test_v503_metadata_is_exact_and_separate_from_v501() -> None:
    metadata = _exact_metadata()

    observed = _require_extension_metadata(_FakeExtension(metadata))

    assert observed == metadata
    assert observed is not metadata
    assert observed["source_identity"] == V503_SOURCE_IDENTITY
    assert observed["batch_values"] == (2,)
    assert observed["production_data_abi_compatible"] is False
    assert observed["eliminates_backward_e4m3_v_publication"] is True


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("schema", "tkfa4.native_tk_d128_backward.v1"),
        ("source_identity", "v501"),
        ("batch_values", (1, 2)),
        ("experimental", 1),
        ("production_data_abi_compatible", True),
        ("v_layout", "BSHD_contiguous"),
        ("dp_instruction_descriptor", "0x0"),
        ("source_file", "wrong.cu"),
    ),
)
def test_v503_metadata_rejects_abi_drift(field: str, value: Any) -> None:
    metadata = _exact_metadata()
    metadata[field] = value

    with pytest.raises(RuntimeError, match="experimental ABI"):
        _require_extension_metadata(_FakeExtension(metadata))


def test_v503_metadata_rejects_missing_receipt() -> None:
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
            shape = (self.numel * self.dtype.itemsize // dtype.itemsize,)
        else:
            dtype = self.dtype
            shape = tuple(int(extent) for extent in shape_or_dtype)
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


def test_v503_runner_allocates_only_direct_output_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allocations: list[_FakeAllocation] = []

    def fake_empty(
        shape: int | tuple[int, ...],
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> _FakeAllocation:
        result = _FakeAllocation(shape, device=device, dtype=dtype)
        allocations.append(result)
        return result

    def fake_empty_like(source: _FakeAllocation) -> _FakeAllocation:
        result = _FakeAllocation(
            source.shape,
            device=source.device,
            dtype=source.dtype,
        )
        allocations.append(result)
        return result

    monkeypatch.setattr(native_module.torch, "empty", fake_empty)
    monkeypatch.setattr(native_module.torch, "empty_like", fake_empty_like)

    runner = NativeTkD128Mxfp4VBackward(
        _FakeExtension(), batch=BATCH, device="cuda:0"
    )

    stats_numel = BATCH * Q_HEADS * SEQUENCE
    assert runner.workspace_torch.shape == (
        2 * stats_numel * torch.float32.itemsize,
    )
    assert runner.dstat.shape == (BATCH, Q_HEADS, 1, SEQUENCE)
    assert runner.lstat.shape == (BATCH, Q_HEADS, 1, SEQUENCE)
    assert runner.dq.shape == (BATCH, SEQUENCE, Q_HEADS, HEAD_DIM)
    assert runner.dk.shape == (BATCH, SEQUENCE, KV_HEADS, HEAD_DIM)
    assert runner.dv.shape == runner.dk.shape
    assert runner.dk_partials is runner.dv_partials
    assert runner.dk_partials.shape == (0,)
    assert not any(
        allocation.dtype == torch.float32 and allocation.shape != (0,)
        for allocation in allocations
    )


@pytest.mark.parametrize("batch", (True, 1, 3, 2.0))
def test_v503_runner_rejects_nonexact_batch(batch: Any) -> None:
    with pytest.raises(ValueError, match="batch 2"):
        NativeTkD128Mxfp4VBackward(
            _FakeExtension(), batch=batch, device="cuda:0"
        )


def _bare_runner() -> NativeTkD128Mxfp4VBackward:
    runner = object.__new__(NativeTkD128Mxfp4VBackward)
    runner.batch = BATCH
    runner.device = torch.device("cuda:0")
    runner.loaded_artifact_identity = {
        "path": "/tmp/v503.so",
        "sha256": "5" * 64,
        "bytes": 503,
    }
    runner.extension_metadata = _exact_metadata()
    runner._q = None
    runner._k = None
    runner._v = None
    runner._v_scales = None
    runner._dout = None
    runner._bind_generation = 0
    runner._run_generation = 0
    return runner


@dataclass(frozen=True)
class _FakeTensor:
    shape: tuple[int, ...]
    dtype: torch.dtype
    is_cuda: bool = True
    contiguous_value: bool = True
    device: torch.device = torch.device("cuda:0")
    pointer: int = 0

    def is_contiguous(self) -> bool:
        return self.contiguous_value

    def data_ptr(self) -> int:
        return self.pointer


def _fake_operands() -> tuple[_FakeTensor, ...]:
    return (
        _FakeTensor(
            (BATCH, SEQUENCE, Q_HEADS, HEAD_DIM),
            torch.float8_e4m3fn,
            pointer=1,
        ),
        _FakeTensor(
            (BATCH, SEQUENCE, KV_HEADS, HEAD_DIM),
            torch.float8_e4m3fn,
            pointer=2,
        ),
        _FakeTensor(
            (BATCH, SEQUENCE, KV_HEADS, HEAD_DIM // 2),
            torch.uint8,
            pointer=3,
        ),
        _FakeTensor(
            (BATCH, SEQUENCE // 128, KV_HEADS, 512),
            torch.uint8,
            pointer=4,
        ),
        _FakeTensor(
            (BATCH, SEQUENCE, Q_HEADS, HEAD_DIM),
            torch.float8_e4m3fn,
            pointer=5,
        ),
    )


def test_v503_bind_is_reference_only_and_atomic() -> None:
    runner = _bare_runner()
    q, k, v, scales, dout = _fake_operands()

    runner.bind_inputs(q, k, v, scales, dout)  # type: ignore[arg-type]

    assert (runner._q, runner._k, runner._v, runner._v_scales, runner._dout) == (
        q,
        k,
        v,
        scales,
        dout,
    )
    assert runner._bind_generation == 1

    existing = (runner._q, runner._k, runner._v, runner._v_scales, runner._dout)
    wrong_scales = replace(scales, shape=(BATCH, 31, KV_HEADS, 512))
    with pytest.raises(ValueError, match="v_backward_mxfp4_scale_pages"):
        runner.bind_inputs(  # type: ignore[arg-type]
            q, k, v, wrong_scales, dout
        )
    assert (runner._q, runner._k, runner._v, runner._v_scales, runner._dout) == existing
    assert runner._bind_generation == 1


@pytest.mark.parametrize(
    ("reset", "entrypoint"), ((False, "main"), (True, "out"))
)
def test_v503_runner_selects_exact_entrypoint_and_abi(
    reset: bool,
    entrypoint: str,
) -> None:
    runner = _bare_runner()
    values = [object() for _ in range(10)]
    (
        runner._q,
        runner._k,
        runner._v,
        runner._v_scales,
        runner._dout,
        runner.lstat,
        runner.dstat,
        runner.dq,
        runner.dk,
        runner.dv,
    ) = values
    calls: list[tuple[str, tuple[Any, ...]]] = []
    runner.compiled_main = lambda *args: calls.append(("main", args))
    runner.compiled_out = lambda *args: calls.append(("out", args))

    runner.run(reset=reset)

    assert calls == [(entrypoint, (*values, SOFTMAX_SCALE))]
    assert runner._run_generation == 1


def test_v503_receipts_and_contract_are_candidate_specific() -> None:
    runner = _bare_runner()
    runner._bind_generation = 3

    assert runner.d128_mxfp4_v_operand_cache_receipt() == {
        "schema": "native_tk_d128_mxfp4_v_direct_bind_v1",
        "implementation": "direct_prebound_torch_tensor_arguments",
        "host_wrapper_cache_required": False,
        "bind_generation": 3,
    }
    compilation = runner.d128_mxfp4_v_compilation_receipt()
    assert compilation["source_identity"] == V503_SOURCE_IDENTITY
    assert compilation["instruction_descriptor"] == "0x08200290"
    contract = runner.contract()
    assert contract["backend"] == BACKEND
    assert contract["publication"] == {
        "backward_e4m3_v_required": False,
        "backward_row_major_mxfp4_v_required": True,
        "forward_feature_major_mxfp4_v_still_required": True,
    }
    assert contract["schedule"]["dv_route"] == "unchanged_v490"


def _install_projection_extension(
    monkeypatch: pytest.MonkeyPatch,
    *,
    shared_tile: bool,
) -> None:
    base = (
        "project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered"
    )
    route = (
        "_shared_tile_mx_backward_v_mx_forward_out"
        if shared_tile
        else "_mx_backward_v_mx_forward_out"
    )
    checked_name = base + route
    unchecked_name = checked_name + "_unchecked"

    def checked(*_args: Any) -> None:
        return None

    def unchecked(*_args: Any) -> None:
        return None

    extension = SimpleNamespace(__file__="<fake-projection-extension>")
    setattr(extension, checked_name, checked)
    setattr(extension, unchecked_name, unchecked)
    monkeypatch.setattr(interface, "_C_b300_lowp_bwd", extension)
    monkeypatch.setattr(
        interface,
        "b300_require_qkv_gqa_d128_unified_lowp_nvfp4_projection",
        lambda: base,
    )


def _shared_tile_producer(
    monkeypatch: pytest.MonkeyPatch,
    **overrides: Any,
) -> interface.B300BoundD128NVFP4QKVProjection:
    _install_projection_extension(monkeypatch, shared_tile=True)
    producer = interface.B300BoundD128NVFP4QKVProjection(
        batch=BATCH,
        seqlen=SEQUENCE,
        hidden=4096,
        q_heads=Q_HEADS,
        kv_heads=KV_HEADS,
        publish_mxfp4_v=True,
        v_mxfp4_scale_2d=True,
        per_block_qk_scales=True,
        experimental_output_shared_dual_v=False,
        experimental_mx_backward_v=False,
        experimental_shared_tile_mx_backward_v=True,
    )
    for name, value in overrides.items():
        setattr(producer, name, value)
    return producer


def _legacy_rowwise_producer(
    monkeypatch: pytest.MonkeyPatch,
) -> interface.B300BoundD128NVFP4QKVProjection:
    _install_projection_extension(monkeypatch, shared_tile=False)
    return interface.B300BoundD128NVFP4QKVProjection(
        batch=BATCH,
        seqlen=SEQUENCE,
        hidden=4096,
        q_heads=Q_HEADS,
        kv_heads=KV_HEADS,
        publish_mxfp4_v=True,
        v_mxfp4_scale_2d=False,
        per_block_qk_scales=True,
        experimental_output_shared_dual_v=False,
        experimental_mx_backward_v=True,
        experimental_shared_tile_mx_backward_v=False,
    )


def _install_fake_allocations(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_empty(
        shape: int | tuple[int, ...],
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> _FakeAllocation:
        return _FakeAllocation(shape, device=device, dtype=dtype)

    def fake_empty_like(source: _FakeAllocation) -> _FakeAllocation:
        return _FakeAllocation(
            source.shape,
            device=source.device,
            dtype=source.dtype,
        )

    monkeypatch.setattr(native_module.torch, "empty", fake_empty)
    monkeypatch.setattr(native_module.torch, "empty_like", fake_empty_like)


def _authenticated_shared_workspace(
    producer: interface.B300BoundD128NVFP4QKVProjection,
    operands: tuple[_FakeTensor, ...],
) -> _FakeWorkspace:
    q, k, v, scales, _ = operands
    workspace = _FakeWorkspace()
    workspace.q_backward_fp8 = q
    workspace.k_backward_fp8 = k
    workspace.v_backward_mxfp4 = v
    workspace.v_backward_mxfp4_scale_pages = scales
    producer._validated_forward_workspaces[id(workspace)] = workspace
    producer._successful_full_abi_validation_count = 1
    return workspace


class _FakeWorkspace:
    pass


def test_shared_tile_v503_producer_contract_is_exact_and_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = require_shared_tile_v503_producer_contract(
        _shared_tile_producer(monkeypatch)
    )

    assert contract["schema"] == SHARED_TILE_PRODUCER_CONTRACT_SCHEMA
    assert contract["shared_tile_shape"] == "D32xS32"
    assert contract["forward_backward_code_matrix"] == "bitwise_identical"
    assert contract["anchor_semantics"] == (
        "four_independent_D32_anchors_per_D128_row"
    )
    assert contract["producer_quantization_passes_per_tile"] == 1

    with pytest.raises(RuntimeError, match="shared D32xS32.*mismatch"):
        require_shared_tile_v503_producer_contract(
            _legacy_rowwise_producer(monkeypatch)
        )
    with pytest.raises(RuntimeError, match="shared D32xS32.*mismatch"):
        require_shared_tile_v503_producer_contract(
            _shared_tile_producer(monkeypatch, publish_mxfp4_v=1)
        )

    forged = SimpleNamespace(**EXPECTED_SHARED_TILE_PRODUCER_ATTRIBUTES)
    with pytest.raises(TypeError, match="must be exactly"):
        require_shared_tile_v503_producer_contract(forged)


@pytest.mark.parametrize(
    "source_identity",
    (
        "v502_d128_gqa_mxfp4v_e4m3do_b2_s4096_owner4_"
        "experimental_bshd_v3",
        "v507_d128_gqa_mxfp4v_sharedtile_e4m3do_b2_s4096_owner4_"
        "experimental_bshd_v1_earlyscore_exact_four_anchor",
    ),
)
def test_v503_adapter_explicitly_refuses_v502_v507_artifacts(
    source_identity: str,
) -> None:
    metadata = _exact_metadata()
    metadata["source_identity"] = source_identity

    with pytest.raises(RuntimeError, match="refuses v502/v507"):
        _require_extension_metadata(_FakeExtension(metadata))


def test_shared_tile_v503_adapter_requires_authenticated_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_allocations(monkeypatch)
    producer = _shared_tile_producer(monkeypatch)
    runner = NativeTkD128SharedTileProducerV503Backward(
        _FakeExtension(),
        producer=producer,
        batch=BATCH,
        device="cuda:0",
    )
    q, k, v, scales, dout = _fake_operands()

    assert runner.backend == SHARED_TILE_V503_BACKEND
    assert runner.contract()["publication_binding"] == {
        "bound": False,
        "required": "first_use_authenticated_shared_tile_producer_workspace",
    }
    with pytest.raises(RuntimeError, match="must precede"):
        runner.shared_tile_v503_publication_receipt()
    with pytest.raises(TypeError, match="producer_workspace"):
        runner.bind_inputs(  # type: ignore[call-arg, arg-type]
            q, k, v, scales, dout
        )

    unauthenticated_workspace = _FakeWorkspace()
    unauthenticated_workspace.q_backward_fp8 = q
    unauthenticated_workspace.k_backward_fp8 = k
    unauthenticated_workspace.v_backward_mxfp4 = v
    unauthenticated_workspace.v_backward_mxfp4_scale_pages = scales
    producer._validated_forward_workspaces.clear()
    with pytest.raises(RuntimeError, match="first-use producer"):
        runner.bind_inputs(  # type: ignore[arg-type]
            q,
            k,
            v,
            scales,
            dout,
            producer_workspace=unauthenticated_workspace,
        )
    assert runner._bind_generation == 0


def test_shared_tile_v503_adapter_binds_only_exact_authenticated_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_allocations(monkeypatch)
    producer = _shared_tile_producer(monkeypatch)
    runner = NativeTkD128SharedTileProducerV503Backward(
        _FakeExtension(),
        producer=producer,
        batch=BATCH,
        device="cuda:0",
    )
    operands = _fake_operands()
    q, k, v, scales, dout = operands
    workspace = _authenticated_shared_workspace(producer, operands)

    wrong_v = replace(v, pointer=999)
    with pytest.raises(RuntimeError, match="authenticated producer workspace"):
        runner.bind_inputs(  # type: ignore[arg-type]
            q,
            k,
            wrong_v,
            scales,
            dout,
            producer_workspace=workspace,
        )
    assert runner._bind_generation == 0

    runner.bind_inputs(  # type: ignore[arg-type]
        q,
        k,
        v,
        scales,
        dout,
        producer_workspace=workspace,
    )
    receipt = runner.shared_tile_v503_publication_receipt()
    assert receipt["schema"] == SHARED_TILE_V503_COMPOSITION_SCHEMA
    assert receipt["consumer_source_identity"] == V503_SOURCE_IDENTITY
    assert receipt["producer_workspace_abi_validated"] is True
    assert receipt["producer_workspace_strongly_retained"] is True
    assert receipt["bind_generation"] == 1

    contract = runner.contract()
    assert contract["backend"] == SHARED_TILE_V503_BACKEND
    assert contract["publication_binding"]["bound"] is True
    assert contract["publication"]["legacy_rowwise_producer_compatible"] is False
    assert contract["composition"]["consumer_source_identity"] == (
        V503_SOURCE_IDENTITY
    )
    assert contract["composition"]["forbidden_consumer_source_prefixes"] == (
        "v502_",
        "v507_",
    )
    compilation = runner.d128_mxfp4_v_compilation_receipt()
    assert compilation["source_identity"] == V503_SOURCE_IDENTITY
    assert compilation["consumer_role"] == (
        "v503_commonrow_requantizing_consumer"
    )


def test_shared_tile_v503_adapter_retains_authenticated_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_allocations(monkeypatch)
    producer = _shared_tile_producer(monkeypatch)
    runner = NativeTkD128SharedTileProducerV503Backward(
        _FakeExtension(),
        producer=producer,
        batch=BATCH,
        device="cuda:0",
    )
    operands = _fake_operands()
    workspace = _authenticated_shared_workspace(producer, operands)
    workspace_id = id(workspace)
    workspace_ref = weakref.ref(workspace)

    runner.bind_inputs(  # type: ignore[arg-type]
        *operands[:4],
        operands[4],
        producer_workspace=workspace,
    )
    del workspace
    gc.collect()

    retained = workspace_ref()
    assert retained is not None
    assert retained is runner._shared_tile_producer_workspace
    assert producer._validated_forward_workspaces[workspace_id] is retained
