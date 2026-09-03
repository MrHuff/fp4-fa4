from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import pytest
import torch

from tk_fa4.lowp_fa4_bwd import (
    native_tk_d128_dense_score_e5m2_dout_backward as native_module,
)
from tk_fa4.lowp_fa4_bwd.native_tk_d128_dense_score_e5m2_dout_backward import (
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
    V510_SOURCE_IDENTITY,
    NativeTkD128DenseE4M3ScoreQKVE5M2DoutBackward,
    _require_e5m2_bshd,
    _require_extension_metadata,
)


def _exact_metadata() -> dict[str, Any]:
    return {
        **EXPECTED_EXTENSION_METADATA,
        "source_file": (
            "../native_gqa_tk_bwd/"
            "v510_d128_gqa_e4m3_score_qkv_e5m2_dout_b1_exact_s4096_"
            "experimental_bshd.cu"
        ),
    }


class _FakeExtension:
    def __init__(self, metadata: dict[str, Any] | None = None) -> None:
        self.metadata = _exact_metadata() if metadata is None else metadata
        self.calls: list[tuple[Any, ...]] = []
        self._tk_fa4_loaded_artifact_identity = {
            "path": "/tmp/v510.so",
            "sha256": "a" * 64,
            "bytes": 510,
        }

    def native_tk_d128_backward_metadata(self) -> dict[str, Any]:
        return dict(self.metadata)

    def backward_e4m3_score_qkv_e5m2_dout_bshd_precomputed_out(
        self, *args: Any
    ) -> None:
        self.calls.append(("out", *args))

    def main_e4m3_score_qkv_e5m2_dout_bshd_precomputed(
        self, *args: Any
    ) -> None:
        self.calls.append(("main", *args))


def test_v510_metadata_authenticates_the_dense_score_e5m2_route() -> None:
    metadata = _exact_metadata()

    observed = _require_extension_metadata(_FakeExtension(metadata))

    assert observed == metadata
    assert observed is not metadata
    assert observed["source_identity"] == V510_SOURCE_IDENTITY
    assert observed["score_qk_dtype"] == "float8_e4m3fn_represented_x4"
    assert observed["dout_dtype"] == "float8_e5m2_represented_x4"
    assert observed["mixed_mma_b_format_mask"] == 0x400
    assert observed["production_dispatch_connected"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("source_identity", "v509"),
        ("selected_kernel", "v488::kernel"),
        ("score_qk_dtype", "float4_e2m1fn_x2"),
        ("dout_dtype", "float8_e4m3fn_represented_x4"),
        ("mixed_mma_b_format_mask", 0),
        ("dstat_physical_abi", "-16*sum(O*raw_dO)"),
        ("source_file", "wrong.cu"),
    ),
)
def test_v510_metadata_rejects_any_abi_drift(
    field: str, value: Any
) -> None:
    metadata = _exact_metadata()
    metadata[field] = value

    with pytest.raises(RuntimeError, match="experimental ABI"):
        _require_extension_metadata(_FakeExtension(metadata))


@dataclass(frozen=True)
class _FakeTensor:
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device = torch.device("cuda:0")
    is_cuda: bool = True
    contiguous: bool = True

    def is_contiguous(self) -> bool:
        return self.contiguous


def _dout(dtype: torch.dtype = torch.float8_e5m2) -> _FakeTensor:
    return _FakeTensor((BATCH, SEQUENCE, Q_HEADS, HEAD_DIM), dtype)


def test_v510_dout_requires_exact_e5m2_b1_bshd() -> None:
    _require_e5m2_bshd(
        _dout(),
        name="dout",
        batch=BATCH,
        heads=Q_HEADS,
        device=torch.device("cuda:0"),
    )

    for invalid in (
        _dout(torch.float8_e4m3fn),
        replace(_dout(), contiguous=False),
        replace(_dout(), shape=(BATCH, SEQUENCE, Q_HEADS, 64)),
        replace(_dout(), device=torch.device("cuda:1")),
    ):
        with pytest.raises(ValueError, match="float8_e5m2"):
            _require_e5m2_bshd(
                invalid,  # type: ignore[arg-type]
                name="dout",
                batch=BATCH,
                heads=Q_HEADS,
                device=torch.device("cuda:0"),
            )


def _bare_runner() -> NativeTkD128DenseE4M3ScoreQKVE5M2DoutBackward:
    runner = object.__new__(
        NativeTkD128DenseE4M3ScoreQKVE5M2DoutBackward
    )
    runner.batch = BATCH
    runner.device = torch.device("cuda:0")
    runner.loaded_artifact_identity = {
        "path": "/tmp/v510.so",
        "sha256": "a" * 64,
        "bytes": 510,
    }
    runner.extension_metadata = _exact_metadata()
    for name in ("_q", "_k", "_v", "_dout"):
        setattr(runner, name, None)
    runner._bind_generation = 0
    runner._run_generation = 0
    return runner


def test_v510_bind_validates_e4_qkv_separately_from_e5_dout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _bare_runner()
    q, k, v, dout = (object() for _ in range(4))
    e4_calls: list[tuple[Any, str, int]] = []
    e5_calls: list[tuple[Any, str, int]] = []
    monkeypatch.setattr(
        native_module,
        "_require_e4m3_bshd",
        lambda tensor, *, name, batch, heads, device: e4_calls.append(
            (tensor, name, heads)
        ),
    )
    monkeypatch.setattr(
        native_module,
        "_require_e5m2_bshd",
        lambda tensor, *, name, batch, heads, device: e5_calls.append(
            (tensor, name, heads)
        ),
    )

    runner.bind_inputs(q, k, v, dout)  # type: ignore[arg-type]

    assert e4_calls == [(q, "q", Q_HEADS), (k, "k", KV_HEADS), (v, "v", KV_HEADS)]
    assert e5_calls == [(dout, "dout", Q_HEADS)]
    assert runner._dout is dout
    assert runner._bind_generation == 1


@pytest.mark.parametrize("reset", (False, True))
def test_v510_run_always_calls_clearing_out_with_exact_abi(reset: bool) -> None:
    runner = _bare_runner()
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
    runner.compiled_out = lambda *args: calls.append(args)

    runner.run(reset=reset)

    assert calls == [(*values, SOFTMAX_SCALE)]
    assert runner._run_generation == 1


def test_v510_contract_is_fail_closed_and_split_precision() -> None:
    runner = _bare_runner()

    contract = runner.contract()

    assert contract["backend"] == BACKEND
    assert contract["input"]["score_and_gradient_qkv"]["dtype"] == (
        "torch.float8_e4m3fn"
    )
    assert contract["input"]["dout"]["dtype"] == "torch.float8_e5m2"
    assert contract["statistics"]["workspace_page_0_physical"] == (
        "-4_sum_O_raw_E5M2_dO"
    )
    assert contract["statistics"]["producer_native"] is False
    assert contract["schedule"]["dispatch"] == (
        "fail_closed_B1_S4096_only_no_fallback"
    )
    assert contract["schedule"]["direct_dkdv_unique_writer"] is False
    assert contract["allocation"]["native_run_allocations"] is False
    assert contract["allocation"]["e5m2_dout_external_caller_owned"] is True
    assert contract["output"]["entrypoint"] == OUT_ENTRYPOINT
    assert "e5m2_dout" in MAIN_ENTRYPOINT
