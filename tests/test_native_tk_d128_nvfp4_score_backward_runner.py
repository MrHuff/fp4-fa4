from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import tk_fa4.interface as interface
from tk_fa4.lowp_fa4_bwd import (
    native_tk_d128_nvfp4_score_backward as native_module,
)
from tk_fa4.lowp_fa4_bwd.native_tk_d128_nvfp4_score_backward import (
    BACKEND,
    BATCH,
    EXPECTED_EXTENSION_METADATA,
    HEAD_DIM,
    KV_HEADS,
    MAIN_ENTRYPOINT,
    OUT_ENTRYPOINT,
    Q_HEADS,
    SEQUENCE,
    SOFTMAX_SCALE,
    V508_SOURCE_IDENTITY,
    NativeTkD128NVFP4ScoreE4M3GradientBackward,
    _require_extension_metadata,
)


def _exact_metadata() -> dict[str, Any]:
    return {
        **EXPECTED_EXTENSION_METADATA,
        "source_file": (
            "../native_gqa_tk_bwd/"
            "v508_d128_gqa_nvfp4_score_e4m3_gradient_b1_exact_s4096_"
            "experimental_bshd.cu"
        ),
    }


class _FakeExtension:
    def __init__(self, metadata: dict[str, Any] | None = None) -> None:
        self.metadata = _exact_metadata() if metadata is None else metadata
        self.calls: list[tuple[Any, ...]] = []
        self._tk_fa4_loaded_artifact_identity = {
            "path": "/tmp/v508.so",
            "sha256": "5" * 64,
            "bytes": 508,
            "device": 1,
            "inode": 2,
            "mtime_ns": 3,
        }

    def native_tk_d128_backward_metadata(self) -> dict[str, Any]:
        return dict(self.metadata)

    def backward_nvfp4_score_e4m3_gradient_bshd_precomputed_out(
        self, *args: Any
    ) -> None:
        self.calls.append(("out", *args))

    def main_nvfp4_score_e4m3_gradient_bshd_precomputed(
        self, *args: Any
    ) -> None:
        self.calls.append(("main", *args))


def test_v508_metadata_is_exact_and_copied() -> None:
    metadata = _exact_metadata()

    observed = _require_extension_metadata(_FakeExtension(metadata))

    assert observed == metadata
    assert observed is not metadata
    assert observed["source_identity"] == V508_SOURCE_IDENTITY
    assert observed["dispatch"] == "fail_closed_B1_S4096_only_no_fallback"
    assert observed["production_dispatch_connected"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("schema", "tkfa4.native_tk_d128_backward.v2"),
        ("source_identity", "v501"),
        ("experimental", 1),
        ("batch", 2),
        ("score_qk_dtype", "float8_e4m3fn"),
        ("score_internal_beta_divisor", 16.0),
        ("backward_out_clears_dq_dk_dv", False),
        ("source_file", "wrong.cu"),
    ),
)
def test_v508_metadata_rejects_abi_drift(field: str, value: Any) -> None:
    metadata = _exact_metadata()
    metadata[field] = value

    with pytest.raises(RuntimeError, match="experimental ABI"):
        _require_extension_metadata(_FakeExtension(metadata))


def test_v508_metadata_rejects_missing_receipt_and_fields() -> None:
    with pytest.raises(RuntimeError, match="lacks"):
        _require_extension_metadata(object())

    metadata = _exact_metadata()
    del metadata["score_schedule"]
    with pytest.raises(RuntimeError, match="incomplete"):
        _require_extension_metadata(_FakeExtension(metadata))


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


def test_v508_runner_allocates_caller_owned_stats_and_outputs(
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

    runner = NativeTkD128NVFP4ScoreE4M3GradientBackward(
        _FakeExtension(), batch=BATCH, device="cuda:0"
    )

    stats_numel = BATCH * Q_HEADS * SEQUENCE
    assert runner.workspace_torch.shape == (
        2 * stats_numel * torch.float32.itemsize,
    )
    assert runner.dstat.shape == (BATCH, Q_HEADS, 1, SEQUENCE)
    assert runner.lstat.shape == (BATCH, Q_HEADS, 1, SEQUENCE)
    assert runner.dstat.storage is runner.workspace_torch.storage
    assert runner.lstat.storage is runner.workspace_torch.storage
    assert runner.dq.shape == (BATCH, SEQUENCE, Q_HEADS, HEAD_DIM)
    assert runner.dk.shape == (BATCH, SEQUENCE, KV_HEADS, HEAD_DIM)
    assert runner.dv.shape == runner.dk.shape
    assert runner.dq.dtype == runner.dk.dtype == runner.dv.dtype
    assert runner.dq.dtype == torch.bfloat16
    assert runner.dk_partials is runner.dv_partials
    assert runner.dk_partials.shape == (0,)
    assert not any(
        allocation.dtype == torch.float32 and allocation.shape != (0,)
        for allocation in allocations
    )


@pytest.mark.parametrize("batch", (True, 0, 2, 1.0))
def test_v508_runner_rejects_nonexact_batch(batch: Any) -> None:
    with pytest.raises(ValueError, match="batch 1"):
        NativeTkD128NVFP4ScoreE4M3GradientBackward(
            _FakeExtension(), batch=batch, device="cuda:0"
        )


def test_v508_runner_rejects_non_cuda_device() -> None:
    with pytest.raises(ValueError, match="CUDA device"):
        NativeTkD128NVFP4ScoreE4M3GradientBackward(
            _FakeExtension(), batch=BATCH, device="cpu"
        )


@pytest.mark.parametrize("missing_entrypoint", (OUT_ENTRYPOINT, MAIN_ENTRYPOINT))
def test_v508_runner_authenticates_both_entrypoints(
    missing_entrypoint: str,
) -> None:
    extension = _FakeExtension()
    setattr(extension, missing_entrypoint, None)

    with pytest.raises(RuntimeError, match=missing_entrypoint):
        NativeTkD128NVFP4ScoreE4M3GradientBackward(
            extension, batch=BATCH, device="cuda:0"
        )


@dataclass(frozen=True)
class _FakeTensor:
    shape: tuple[int, ...]
    dtype: torch.dtype
    pointer: int
    is_cuda: bool = True
    contiguous_value: bool = True
    device: torch.device = torch.device("cuda:0")

    def is_contiguous(self) -> bool:
        return self.contiguous_value

    def data_ptr(self) -> int:
        return self.pointer

    def numel(self) -> int:
        result = 1
        for extent in self.shape:
            result *= extent
        return result


def _tensor(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    pointer: int,
) -> _FakeTensor:
    return _FakeTensor(shape, dtype, pointer)


def _workspace() -> interface.B300E4M3QKVForwardWorkspace:
    q_payload = _tensor(
        (BATCH, Q_HEADS, SEQUENCE, HEAD_DIM // 2), torch.uint8, 100
    )
    k_payload = _tensor(
        (BATCH, KV_HEADS, SEQUENCE, HEAD_DIM // 2), torch.uint8, 200
    )
    return interface.B300E4M3QKVForwardWorkspace(
        q_payload=q_payload,  # type: ignore[arg-type]
        k_payload=k_payload,  # type: ignore[arg-type]
        q_scale_pages=_tensor(
            (BATCH, SEQUENCE // 128, Q_HEADS * 2, 512),
            torch.float8_e4m3fn,
            300,
        ),  # type: ignore[arg-type]
        q_global_scale=_tensor(
            (BATCH, Q_HEADS), torch.float32, 400
        ),  # type: ignore[arg-type]
        k_scale_pages=_tensor(
            (BATCH, SEQUENCE // 64, KV_HEADS * 2, 512),
            torch.float8_e4m3fn,
            500,
        ),  # type: ignore[arg-type]
        k_global_scale=_tensor(
            (BATCH, KV_HEADS), torch.float32, 600
        ),  # type: ignore[arg-type]
        v_mxfp4_payload=_tensor(
            (1,), torch.float4_e2m1fn_x2, 700
        ),  # type: ignore[arg-type]
        v_mxfp4_scale_pages=_tensor(
            (1,), torch.float8_e4m3fn, 800
        ),  # type: ignore[arg-type]
        v_fp8_payload=_tensor(
            (1,), torch.float8_e4m3fn, 900
        ),  # type: ignore[arg-type]
        v_backward_fp8=_tensor(
            (BATCH, SEQUENCE, KV_HEADS, HEAD_DIM),
            torch.float8_e4m3fn,
            1000,
        ),  # type: ignore[arg-type]
        q_backward_fp8=_tensor(
            (BATCH, SEQUENCE, Q_HEADS, HEAD_DIM),
            torch.float8_e4m3fn,
            1100,
        ),  # type: ignore[arg-type]
        k_backward_fp8=_tensor(
            (BATCH, SEQUENCE, KV_HEADS, HEAD_DIM),
            torch.float8_e4m3fn,
            1200,
        ),  # type: ignore[arg-type]
        q_payload_fp4=_tensor(
            q_payload.shape, torch.float4_e2m1fn_x2, q_payload.pointer
        ),  # type: ignore[arg-type]
        k_payload_fp4=_tensor(
            k_payload.shape, torch.float4_e2m1fn_x2, k_payload.pointer
        ),  # type: ignore[arg-type]
        empty_bf16=_tensor((0,), torch.bfloat16, 1300),  # type: ignore[arg-type]
        empty_byte=_tensor((0,), torch.uint8, 1400),  # type: ignore[arg-type]
        empty_fp8=_tensor(
            (0,), torch.float8_e4m3fn, 1500
        ),  # type: ignore[arg-type]
        empty_fp4=_tensor(
            (0,), torch.float4_e2m1fn_x2, 1600
        ),  # type: ignore[arg-type]
    )


def _bare_runner() -> NativeTkD128NVFP4ScoreE4M3GradientBackward:
    runner = object.__new__(NativeTkD128NVFP4ScoreE4M3GradientBackward)
    runner.batch = BATCH
    runner.device = torch.device("cuda:0")
    runner.loaded_artifact_identity = {
        "path": "/tmp/v508.so",
        "sha256": "5" * 64,
        "bytes": 508,
    }
    runner.extension_metadata = _exact_metadata()
    for name in (
        "_q",
        "_k",
        "_v",
        "_dout",
        "_q_native",
        "_k_native",
        "_q_native_scale",
        "_k_native_scale",
        "_q_global_scale",
        "_k_global_scale",
        "_native_score_workspace",
    ):
        setattr(runner, name, None)
    runner._bind_generation = 0
    runner._run_generation = 0
    return runner


def _operands(
    workspace: interface.B300E4M3QKVForwardWorkspace,
) -> tuple[_FakeTensor, ...]:
    return (
        workspace.q_backward_fp8,  # type: ignore[return-value]
        workspace.k_backward_fp8,  # type: ignore[return-value]
        workspace.v_backward_fp8,  # type: ignore[return-value]
        _tensor(
            (BATCH, SEQUENCE, Q_HEADS, HEAD_DIM),
            torch.float8_e4m3fn,
            1700,
        ),
    )


def test_v508_bind_retains_exact_workspace_and_prebound_operands() -> None:
    runner = _bare_runner()
    workspace = _workspace()
    q, k, v, dout = _operands(workspace)

    runner.bind_inputs(q, k, v, dout, workspace)  # type: ignore[arg-type]

    assert runner._q is q
    assert runner._k is k
    assert runner._v is v
    assert runner._dout is dout
    assert runner._q_native is workspace.q_payload_fp4
    assert runner._k_native is workspace.k_payload_fp4
    assert runner._q_native_scale is workspace.q_scale_pages
    assert runner._k_native_scale is workspace.k_scale_pages
    assert runner._q_global_scale is workspace.q_global_scale
    assert runner._k_global_scale is workspace.k_global_scale
    assert runner._native_score_workspace is workspace
    assert runner._bind_generation == 1


def test_v508_bind_requires_exact_workspace_type() -> None:
    runner = _bare_runner()
    workspace = _workspace()
    q, k, v, dout = _operands(workspace)

    with pytest.raises(TypeError, match="exactly"):
        runner.bind_inputs(  # type: ignore[arg-type]
            q, k, v, dout, SimpleNamespace()
        )


def test_v508_bind_rejects_nonworkspace_represented_view_atomically() -> None:
    runner = _bare_runner()
    workspace = _workspace()
    q, k, v, dout = _operands(workspace)
    wrong_q = replace(q, pointer=q.pointer + 1)
    existing = [object() for _ in range(4)]
    runner._q, runner._k, runner._v, runner._dout = existing

    with pytest.raises(RuntimeError, match="exact views"):
        runner.bind_inputs(  # type: ignore[arg-type]
            wrong_q, k, v, dout, workspace
        )

    assert [runner._q, runner._k, runner._v, runner._dout] == existing
    assert runner._bind_generation == 0


def test_v508_bind_rejects_mutated_native_alias_atomically() -> None:
    runner = _bare_runner()
    workspace = _workspace()
    q, k, v, dout = _operands(workspace)
    object.__setattr__(
        workspace,
        "q_payload_fp4",
        replace(workspace.q_payload_fp4, pointer=101),
    )

    with pytest.raises(RuntimeError, match="do not share"):
        runner.bind_inputs(q, k, v, dout, workspace)  # type: ignore[arg-type]

    assert runner._native_score_workspace is None
    assert runner._bind_generation == 0


@pytest.mark.parametrize(
    ("field", "replacement", "match"),
    (
        (
            "q_scale_pages",
            _tensor((1,), torch.float8_e4m3fn, 300),
            "q_scale_pages",
        ),
        (
            "k_global_scale",
            _tensor((BATCH, KV_HEADS), torch.bfloat16, 600),
            "k_global_scale",
        ),
        (
            "q_payload_fp4",
            replace(
                _tensor(
                    (BATCH, Q_HEADS, SEQUENCE, HEAD_DIM // 2),
                    torch.float4_e2m1fn_x2,
                    100,
                ),
                contiguous_value=False,
            ),
            "q_payload_fp4",
        ),
    ),
)
def test_v508_bind_rejects_workspace_tensor_abi_drift(
    field: str,
    replacement: _FakeTensor,
    match: str,
) -> None:
    runner = _bare_runner()
    workspace = _workspace()
    q, k, v, dout = _operands(workspace)
    object.__setattr__(workspace, field, replacement)

    with pytest.raises(ValueError, match=match):
        runner.bind_inputs(q, k, v, dout, workspace)  # type: ignore[arg-type]

    assert runner._bind_generation == 0


@pytest.mark.parametrize("reset", (False, True))
def test_v508_run_always_calls_clearing_out_with_exact_abi(reset: bool) -> None:
    runner = _bare_runner()
    values = [object() for _ in range(16)]
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
        runner._q_native,
        runner._k_native,
        runner._q_native_scale,
        runner._k_native_scale,
        runner._q_global_scale,
        runner._k_global_scale,
        runner._native_score_workspace,
    ) = values
    calls: list[tuple[Any, ...]] = []
    runner.compiled_out = lambda *args: calls.append(args)

    runner.run(reset=reset)

    assert calls == [(*values[:15], SOFTMAX_SCALE)]
    assert runner._run_generation == 1


@pytest.mark.parametrize("reset", (None, 0, 1, "false"))
def test_v508_run_requires_exact_bool(reset: Any) -> None:
    runner = _bare_runner()
    runner.compiled_out = pytest.fail

    with pytest.raises(TypeError, match="exactly bool"):
        runner.run(reset=reset)


def test_v508_run_requires_bound_workspace() -> None:
    runner = _bare_runner()
    runner.compiled_out = pytest.fail

    with pytest.raises(RuntimeError, match=r"bind_inputs\(\)"):
        runner.run(reset=False)

    assert runner._run_generation == 0


def test_v508_contract_declares_hybrid_score_gradient_and_additive_output() -> None:
    runner = _bare_runner()

    contract = runner.contract()

    assert contract["backend"] == BACKEND
    assert contract["shape"] == {
        "batch": BATCH,
        "sequence": SEQUENCE,
        "q_heads": Q_HEADS,
        "kv_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
    }
    assert contract["input"]["composition"] == (
        "native_NVFP4_score_represented_E4M3_gradient"
    )
    assert contract["input"]["score_qk"]["source"] == (
        "exact_B300E4M3QKVForwardWorkspace"
    )
    assert contract["input"]["gradient_qk_v_dout"]["encoding_scale"] == 4.0
    assert contract["output"]["kernel_store_semantics"] == "additive"
    assert contract["output"]["entrypoint"] == OUT_ENTRYPOINT
    assert contract["schedule"]["always_clearing_out_entrypoint"] is True
    assert contract["publication"]["second_qk_quantization"] is False
    assert contract["allocation"]["native_run_allocations"] is False

    contract["extension"]["sha256"] = "6" * 64
    contract["extension_metadata"]["source_identity"] = "wrong"
    assert runner.loaded_artifact_identity["sha256"] == "5" * 64
    assert runner.extension_metadata["source_identity"] == V508_SOURCE_IDENTITY


def test_e2e_cli_exposes_only_an_explicit_v508_selector() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "tk_fa4"
        / "lowp_fa4_bwd"
        / "benchmark_llama12b_e2e.py"
    ).read_text(encoding="utf-8")

    cli = source.split(
        '"--native-tk-d128-native-score-backward"', 1
    )[1].split("parser.add_argument", 1)[0]
    assert 'action="store_true"' in cli
    assert "and not native_tk_d128_backward" in source
    assert "native_tk_d128_native_score_backward=(" in source
    assert "args.native_tk_d128_native_score_backward" in source
