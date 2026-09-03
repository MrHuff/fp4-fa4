from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import pytest
import torch

from tk_fa4.lowp_fa4_bwd import (
    native_tk_d128_nvfp4_score_e5m2_dout_backward as native_module,
)
from tk_fa4.lowp_fa4_bwd.native_tk_d128_nvfp4_score_e5m2_dout_backward import (
    BACKEND,
    BATCH,
    EXPECTED_EXTENSION_METADATA,
    HEAD_DIM,
    KV_HEADS,
    MAIN_ENTRYPOINT,
    OUT_ENTRYPOINT,
    PRECLEARED_DQ_OUT_ENTRYPOINT,
    Q_HEADS,
    SEQUENCE,
    SOFTMAX_SCALE,
    SUPPORTED_BATCHES,
    V509_SOURCE_IDENTITY,
    NativeTkD128NVFP4ScoreE4M3QKVE5M2DoutBackward,
    _require_e5m2_bshd,
    _require_extension_metadata,
    expected_extension_metadata,
)


def _exact_metadata(batch: int = BATCH) -> dict[str, Any]:
    return {
        **expected_extension_metadata(batch),
        "source_file": (
            "../native_gqa_tk_bwd/"
            "v509_d128_gqa_nvfp4_score_e4m3_qkv_e5m2_dout_b1_exact_"
            "s4096_experimental_bshd.cu"
        ),
    }


class _FakeExtension:
    def __init__(
        self,
        metadata: dict[str, Any] | None = None,
        *,
        batch: int = BATCH,
    ) -> None:
        self.metadata = (
            _exact_metadata(batch) if metadata is None else metadata
        )
        self.calls: list[tuple[Any, ...]] = []
        self._tk_fa4_loaded_artifact_identity = {
            "path": "/tmp/v509.so",
            "sha256": "9" * 64,
            "bytes": 509,
        }

    def native_tk_d128_backward_metadata(self) -> dict[str, Any]:
        return dict(self.metadata)

    def backward_nvfp4_score_e4m3_qkv_e5m2_dout_bshd_precomputed_out(
        self, *args: Any
    ) -> None:
        self.calls.append(("out", *args))

    def backward_nvfp4_score_e4m3_qkv_e5m2_dout_bshd_precleared_dq_out(
        self, *args: Any
    ) -> None:
        self.calls.append(("precleared_dq", *args))

    def main_nvfp4_score_e4m3_qkv_e5m2_dout_bshd_precomputed(
        self, *args: Any
    ) -> None:
        self.calls.append(("main", *args))


@pytest.mark.parametrize("batch", SUPPORTED_BATCHES)
def test_v509_metadata_is_exact_and_separate_from_v508(batch: int) -> None:
    metadata = _exact_metadata(batch)

    observed = _require_extension_metadata(
        _FakeExtension(metadata), batch=batch
    )

    assert observed == metadata
    assert observed is not metadata
    assert observed["source_identity"] == expected_extension_metadata(batch)[
        "source_identity"
    ]
    if batch == BATCH:
        assert observed["source_identity"] == V509_SOURCE_IDENTITY
        assert expected_extension_metadata(batch) == EXPECTED_EXTENSION_METADATA
    assert observed["batch"] == batch
    assert observed["dispatch"] == (
        f"fail_closed_B{batch}_S4096_only_no_fallback"
    )
    assert observed["gradient_qkv_dtype"] == (
        "float8_e4m3fn_represented_x4"
    )
    assert observed["dout_dtype"] == "float8_e5m2_represented_x4"
    assert observed["mixed_mma_b_format_mask"] == 0x400
    assert observed["production_dispatch_connected"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("source_identity", "v508"),
        ("selected_kernel", "v508::kernel"),
        ("dout_dtype", "float8_e4m3fn_represented_x4"),
        ("dout_decode_scale", 4.0),
        ("mixed_mma_b_format_mask", 0),
        ("dstat_physical_abi", "-16*sum(O*raw_dO)"),
        ("source_file", "wrong.cu"),
    ),
)
@pytest.mark.parametrize("batch", SUPPORTED_BATCHES)
def test_v509_metadata_rejects_abi_drift(
    field: str,
    value: Any,
    batch: int,
) -> None:
    metadata = _exact_metadata(batch)
    metadata[field] = value

    with pytest.raises(RuntimeError, match="experimental ABI"):
        _require_extension_metadata(_FakeExtension(metadata), batch=batch)


@pytest.mark.parametrize(
    ("expected_batch", "artifact_batch"),
    ((1, 2), (1, 4), (2, 1), (2, 4), (4, 1), (4, 2)),
)
def test_v509_metadata_rejects_a_different_exact_batch_artifact(
    expected_batch: int,
    artifact_batch: int,
) -> None:
    with pytest.raises(RuntimeError, match="experimental ABI"):
        _require_extension_metadata(
            _FakeExtension(batch=artifact_batch),
            batch=expected_batch,
        )


@pytest.mark.parametrize("batch", (False, 0, 3, 8))
def test_v509_metadata_rejects_unsupported_batch(batch: int) -> None:
    with pytest.raises(ValueError, match="supports batches"):
        expected_extension_metadata(batch)


@dataclass(frozen=True)
class _FakeTensor:
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device = torch.device("cuda:0")
    is_cuda: bool = True
    contiguous: bool = True

    def is_contiguous(self) -> bool:
        return self.contiguous


def _dout(
    dtype: torch.dtype = torch.float8_e5m2,
    *,
    batch: int = BATCH,
) -> _FakeTensor:
    return _FakeTensor((batch, SEQUENCE, Q_HEADS, HEAD_DIM), dtype)


@pytest.mark.parametrize("batch", SUPPORTED_BATCHES)
def test_v509_dout_requires_exact_e5m2_bshd(batch: int) -> None:
    _require_e5m2_bshd(
        _dout(batch=batch),
        name="dout",
        batch=batch,
        heads=Q_HEADS,
        device=torch.device("cuda:0"),
    )

    for invalid in (
        _dout(torch.float8_e4m3fn, batch=batch),
        replace(_dout(batch=batch), contiguous=False),
        replace(_dout(batch=batch), shape=(batch, SEQUENCE, Q_HEADS, 64)),
        replace(_dout(batch=batch), device=torch.device("cuda:1")),
    ):
        with pytest.raises(ValueError, match="float8_e5m2"):
            _require_e5m2_bshd(
                invalid,
                name="dout",
                batch=batch,
                heads=Q_HEADS,
                device=torch.device("cuda:0"),
            )


def _bare_runner(
    batch: int = BATCH,
) -> NativeTkD128NVFP4ScoreE4M3QKVE5M2DoutBackward:
    runner = object.__new__(
        NativeTkD128NVFP4ScoreE4M3QKVE5M2DoutBackward
    )
    runner.batch = batch
    runner.device = torch.device("cuda:0")
    runner.loaded_artifact_identity = {
        "path": "/tmp/v509.so",
        "sha256": "9" * 64,
        "bytes": 509,
    }
    runner.extension_metadata = _exact_metadata(batch)
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


@pytest.mark.parametrize("batch", SUPPORTED_BATCHES)
def test_v509_bind_splits_qkv_and_dout_validation(
    monkeypatch: pytest.MonkeyPatch,
    batch: int,
) -> None:
    runner = _bare_runner(batch)
    q, k, v, dout, workspace = (object() for _ in range(5))
    native_operands = tuple(object() for _ in range(6))
    e4_calls: list[tuple[Any, str, int, int]] = []
    e5_calls: list[tuple[Any, str, int, int]] = []

    monkeypatch.setattr(
        native_module,
        "_require_e4m3_bshd",
        lambda tensor, *, name, batch, heads, device: e4_calls.append(
            (tensor, name, batch, heads)
        ),
    )
    monkeypatch.setattr(
        native_module,
        "_require_e5m2_bshd",
        lambda tensor, *, name, batch, heads, device: e5_calls.append(
            (tensor, name, batch, heads)
        ),
    )
    monkeypatch.setattr(
        native_module,
        "_require_native_score_workspace",
        lambda native_workspace, **kwargs: native_operands,
    )

    runner.bind_inputs(q, k, v, dout, workspace)  # type: ignore[arg-type]

    assert e4_calls == [
        (q, "q", batch, Q_HEADS),
        (k, "k", batch, KV_HEADS),
        (v, "v", batch, KV_HEADS),
    ]
    assert e5_calls == [(dout, "dout", batch, Q_HEADS)]
    assert runner._dout is dout
    assert runner._native_score_workspace is workspace
    assert runner._bind_generation == 1


@pytest.mark.parametrize("batch", SUPPORTED_BATCHES)
@pytest.mark.parametrize("reset", (False, True))
def test_v509_run_always_calls_clearing_out_with_exact_abi(
    reset: bool,
    batch: int,
) -> None:
    runner = _bare_runner(batch)
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


@pytest.mark.parametrize("batch", SUPPORTED_BATCHES)
def test_v509_publisher_precleared_dq_run_uses_dkdv_clearing_abi(
    batch: int,
) -> None:
    runner = _bare_runner(batch)
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
    runner.compiled_precleared_dq_out = lambda *args: calls.append(args)

    runner.run_publisher_precleared_dq(reset=False)

    assert calls == [(*values[:15], SOFTMAX_SCALE)]
    assert runner._run_generation == 1


@pytest.mark.parametrize("batch", SUPPORTED_BATCHES)
def test_v509_contract_is_fail_closed_and_split_precision(batch: int) -> None:
    runner = _bare_runner(batch)

    contract = runner.contract()

    assert contract["backend"] == BACKEND
    assert contract["input"]["gradient_qkv"]["dtype"] == (
        "torch.float8_e4m3fn"
    )
    assert contract["input"]["dout"] == {
        "dtype": "torch.float8_e5m2",
        "layout": "BSHD_contiguous",
        "encoding_scale": 4.0,
        "decode_scale": 0.25,
        "semantics": "represented_E5M2_STE_gradient_operand",
        "initial_producer": (
            "authenticated_standalone_BF16_to_E5M2_producer"
        ),
        "matched_dout_dstat_authentication": (
            "external_caller_responsibility"
        ),
    }
    assert contract["statistics"]["workspace_page_0_physical"] == (
        "-4_sum_O_raw_E5M2_dO"
    )
    assert contract["statistics"]["producer_native"] is False
    assert contract["statistics"]["dstat_population"] == (
        "external_producer_required"
    )
    assert contract["statistics"]["lstat_population"] == (
        "external_forward_population_required"
    )
    assert contract["shape"]["batch"] == batch
    assert contract["extension_metadata"] == _exact_metadata(batch)
    assert contract["schedule"]["dispatch"] == (
        f"B{batch}_S4096_v509_only_fail_closed"
    )
    assert contract["publication"]["second_qk_quantization"] is False
    assert contract["allocation"]["native_run_allocations"] is False
    assert contract["allocation"]["e5m2_dout_storage"] == (
        "external_caller_owned_correctness_rung"
    )
    assert contract["allocation"]["extension_outputs_runner_owned"] is True
    assert (
        contract["allocation"]["e5m2_dout_external_caller_owned"] is True
    )
    assert "caller_owned_runner_storage" not in contract["allocation"]
    assert contract["allocation"]["e5m2_producer_integrated"] is False
    assert contract["output"]["storage_owner"] == "runner"
    assert contract["output"]["entrypoint"] == OUT_ENTRYPOINT
    assert contract["output"]["clear_ownership"] == {
        "dq": "selected_backward_entrypoint",
        "dk": "selected_backward_entrypoint",
        "dv": "selected_backward_entrypoint",
    }
    assert contract["schedule"]["direct_dkdv_unique_writer"] is True
    assert contract["schedule"]["always_clearing_out_entrypoint"] is True
    assert contract["schedule"]["fused_publisher_precleared_dq"] is False
    assert OUT_ENTRYPOINT.endswith("_out")
    assert PRECLEARED_DQ_OUT_ENTRYPOINT.endswith("_precleared_dq_out")
    assert "e5m2_dout" in MAIN_ENTRYPOINT


@pytest.mark.parametrize("batch", SUPPORTED_BATCHES)
def test_v509_contract_records_fused_publisher_precleared_path(
    batch: int,
) -> None:
    runner = _bare_runner(batch)

    contract = runner.contract(fused_publisher_precleared_dq=True)

    assert contract["input"]["dout"]["initial_producer"] == (
        "authenticated_fused_NVFP4_output_projection_E5M2_dout_"
        "dstat_lstat_dq_clear_publisher"
    )
    assert contract["input"]["dout"][
        "matched_dout_dstat_authentication"
    ] == "authenticated_v509_route_pair"
    assert contract["statistics"]["producer_source"] == (
        "fused_NVFP4_output_projection_E5M2_publisher"
    )
    assert contract["statistics"]["dstat_population"] == (
        "fused_projection_publisher"
    )
    assert contract["statistics"]["lstat_population"] == (
        "fused_projection_publisher"
    )
    assert contract["output"]["entrypoint"] == PRECLEARED_DQ_OUT_ENTRYPOINT
    assert contract["output"]["clear_ownership"] == {
        "dq": "fused_projection_publisher",
        "dk": "selected_backward_entrypoint",
        "dv": "selected_backward_entrypoint",
    }
    assert contract["schedule"]["always_clearing_out_entrypoint"] is False
    assert contract["schedule"]["fused_publisher_precleared_dq"] is True
    assert contract["allocation"]["e5m2_producer_integrated"] is True


def test_v509_contract_rejects_non_boolean_path_selector() -> None:
    with pytest.raises(TypeError, match="must be exactly bool"):
        _bare_runner().contract(
            fused_publisher_precleared_dq=1,  # type: ignore[arg-type]
        )
