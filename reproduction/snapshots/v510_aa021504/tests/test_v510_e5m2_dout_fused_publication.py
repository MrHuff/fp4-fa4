from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import tk_fa4
import tk_fa4.interface as interface
from tk_fa4.lowp_fa4_bwd.native_tk_d128_dense_score_e5m2_dout_backward import (
    EXPECTED_EXTENSION_METADATA as EXPECTED_V510_BACKWARD_METADATA,
)
from tk_fa4.lowp_fa4_bwd.native_tk_d128_nvfp4_score_e5m2_dout_backward import (
    EXPECTED_EXTENSION_METADATA as EXPECTED_V509_BACKWARD_METADATA,
)


ROOT = Path(__file__).resolve().parents[1]
CUDA = ROOT / "tk_fa4/lowp_fa4_bwd/lowp_fa4_bwd.cu"


class _FakeTensor:
    def __init__(
        self,
        dtype: object,
        shape: tuple[int, ...],
        *,
        pointer: int = 0,
        contiguous: bool = True,
        device: str = "cuda:0",
    ) -> None:
        self.dtype = dtype
        self.shape = shape
        self.ndim = len(shape)
        self.is_cuda = True
        self.device = device
        self._pointer = pointer
        self._contiguous = contiguous

    def is_contiguous(self) -> bool:
        return self._contiguous

    def numel(self) -> int:
        elements = 1
        for extent in self.shape:
            elements *= extent
        return elements

    def data_ptr(self) -> int:
        return self._pointer


def _publisher_extension(
    *,
    source_file: str = str(CUDA),
    **overrides: object,
) -> SimpleNamespace:
    metadata = dict(interface.V510_E5M2_DOUT_PUBLISHER_METADATA)
    metadata.update(overrides)
    metadata["source_file"] = source_file
    return SimpleNamespace(
        project_dout_unified_fp4_nvfp4_v510_e5m2_metadata=(
            lambda: dict(metadata)
        )
    )


def test_v510_fused_publication_is_separate_from_v509() -> None:
    assert (
        tk_fa4.B300V510E5M2DoutPublication
        is interface.B300V510E5M2DoutPublication
    )
    assert (
        tk_fa4.b300_project_dout_unified_lowp_nvfp4_v510_e5m2
        is interface.b300_project_dout_unified_lowp_nvfp4_v510_e5m2
    )
    assert (
        tk_fa4.b300_require_v510_e5m2_dout_route
        is interface.b300_require_v510_e5m2_dout_route
    )
    assert (
        interface.B300V510E5M2DoutPublication
        is not interface.B300V509E5M2DoutPublication
    )
    assert (
        interface.b300_project_dout_unified_lowp_nvfp4_v510_e5m2
        is not interface.b300_project_dout_unified_lowp_nvfp4_v509_e5m2
    )
    assert (
        interface.b300_require_v510_e5m2_dout_route
        is not interface.b300_require_v509_e5m2_dout_route
    )


def test_v510_route_pairing_accepts_only_exact_publisher_and_backward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        interface,
        "_C_b300_lowp_bwd",
        _publisher_extension(),
    )

    receipt = interface.b300_require_v510_e5m2_dout_route(
        dict(EXPECTED_V510_BACKWARD_METADATA)
    )

    assert receipt == {
        "route": "v510_only_fail_closed",
        "publisher": dict(interface.V510_E5M2_DOUT_PUBLISHER_METADATA),
        "backward": dict(EXPECTED_V510_BACKWARD_METADATA),
    }


@pytest.mark.parametrize(
    "source_file",
    (
        "lowp_fa4_bwd.cu",
        "tk_fa4/lowp_fa4_bwd/lowp_fa4_bwd.cu",
        "/workspace/fp4_matmul/tk_fa4/lowp_fa4_bwd/lowp_fa4_bwd.cu",
    ),
)
def test_v510_route_accepts_only_canonical_publisher_source(
    monkeypatch: pytest.MonkeyPatch,
    source_file: str,
) -> None:
    monkeypatch.setattr(
        interface,
        "_C_b300_lowp_bwd",
        _publisher_extension(source_file=source_file),
    )
    interface.b300_require_v510_e5m2_dout_route(
        dict(EXPECTED_V510_BACKWARD_METADATA)
    )


@pytest.mark.parametrize(
    "source_file",
    (
        "/tmp/lowp_fa4_bwd.cu",
        "projection_fp4_epilogue.cuh",
        "lowp_fa4_bwd.cu.bak",
    ),
)
def test_v510_route_rejects_noncanonical_publisher_source(
    monkeypatch: pytest.MonkeyPatch,
    source_file: str,
) -> None:
    monkeypatch.setattr(
        interface,
        "_C_b300_lowp_bwd",
        _publisher_extension(source_file=source_file),
    )
    with pytest.raises(RuntimeError, match="v510 E5M2 dO route"):
        interface.b300_require_v510_e5m2_dout_route(
            dict(EXPECTED_V510_BACKWARD_METADATA)
        )


@pytest.mark.parametrize(
    ("side", "field", "value"),
    (
        ("publisher", "payload_dtype", "float8_e4m3fn"),
        ("publisher", "batch", 2),
        ("publisher", "probability_log2_lift", 0.0),
        ("publisher", "dstat_physical_abi", "-16*sum(O*dO)"),
        ("backward", "dout_dtype", "float8_e4m3fn_represented_x4"),
        ("backward", "score_qk_dtype", "float4_e2m1fn_x2"),
        ("backward", "batch", 2),
        ("backward", "lstat_abi", "-LSE*log2(e)"),
    ),
)
def test_v510_route_pairing_rejects_any_metadata_drift(
    monkeypatch: pytest.MonkeyPatch,
    side: str,
    field: str,
    value: object,
) -> None:
    publisher_overrides = {field: value} if side == "publisher" else {}
    monkeypatch.setattr(
        interface,
        "_C_b300_lowp_bwd",
        _publisher_extension(**publisher_overrides),
    )
    backward = dict(EXPECTED_V510_BACKWARD_METADATA)
    if side == "backward":
        backward[field] = value

    with pytest.raises(RuntimeError, match="v510 E5M2 dO route"):
        interface.b300_require_v510_e5m2_dout_route(backward)


def test_v509_and_v510_backward_receipts_cannot_cross_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v509_metadata = dict(interface.V509_E5M2_DOUT_PUBLISHER_METADATA)
    v509_metadata["source_file"] = str(CUDA)
    v510_metadata = dict(interface.V510_E5M2_DOUT_PUBLISHER_METADATA)
    v510_metadata["source_file"] = str(CUDA)
    monkeypatch.setattr(
        interface,
        "_C_b300_lowp_bwd",
        SimpleNamespace(
            project_dout_unified_fp4_nvfp4_v509_e5m2_metadata=(
                lambda: dict(v509_metadata)
            ),
            project_dout_unified_fp4_nvfp4_v510_e5m2_metadata=(
                lambda: dict(v510_metadata)
            ),
        ),
    )

    with pytest.raises(RuntimeError, match="v510 E5M2 dO route"):
        interface.b300_require_v510_e5m2_dout_route(
            dict(EXPECTED_V509_BACKWARD_METADATA)
        )
    with pytest.raises(RuntimeError, match="v509 E5M2 dO route"):
        interface.b300_require_v509_e5m2_dout_route(
            dict(EXPECTED_V510_BACKWARD_METADATA)
        )


def test_v510_raw_symbol_reuses_proven_e5_epilogue_with_separate_identity() -> None:
    source = CUDA.read_text()

    assert '"project_dout_unified_fp4_nvfp4_v509_e5m2"' in source
    assert '"project_dout_unified_fp4_nvfp4_v510_e5m2"' in source
    assert "&project_dout_unified_fp4_nvfp4_v509_e5m2," in source
    assert "&project_dout_unified_fp4_nvfp4_v510_e5m2," in source
    assert '"tkfa4.v510_e5m2_dout_publisher.v1"' in source
    assert (
        '"v510_fused_nvfp4_output_projection_e5m2_dout_b1_s4096_v1"'
        in source
    )
    assert (
        source.count(
            'result["selected_epilogue"] =\n'
            '        "kernel_v509_native_score_e5m2_dout";'
        )
        == 2
    )


def test_v510_python_api_exposes_only_real_e5m2_slot_seven() -> None:
    source = Path(interface.__file__).read_text()
    function = source.split(
        "def b300_project_dout_unified_lowp_nvfp4_v510_e5m2(", 1
    )[1].split("\ndef ", 1)[0]

    assert (
        "_C_b300_lowp_bwd.project_dout_unified_fp4_nvfp4_v510_e5m2("
        in function
    )
    assert "dout_backward_e5m2=projected[7]" in function
    assert "torch.float8_e5m2" in function
    assert "stats_workspace.numel() != expected_stats_bytes" in function
    assert "with exactly {expected_stats_bytes}" in function
    assert "store_bf16" not in function
    assert "publish_fp8_backward" not in function


def test_v510_python_api_returns_its_own_typed_exact_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shape = (1, 4096, 32, 128)
    stats_shape = (1, 32, 1, 4096)
    attention_output = _FakeTensor(torch.bfloat16, shape, pointer=0xA0)
    lse = object()
    stats_workspace = _FakeTensor(torch.uint8, (2 * 32 * 4096 * 4,))
    dq_clear = _FakeTensor(torch.bfloat16, shape)
    dout_storage = _FakeTensor(torch.bfloat16, shape, pointer=0xA0)
    empty = _FakeTensor(torch.uint8, (0,))
    dpsum = _FakeTensor(torch.float32, stats_shape)
    lse_log2 = _FakeTensor(torch.float32, stats_shape)
    dout_e5m2 = _FakeTensor(torch.float8_e5m2, shape)
    calls: list[tuple[object, ...]] = []

    def project(*args: object) -> list[_FakeTensor]:
        calls.append(args)
        return [
            dout_storage,
            empty,
            empty,
            empty,
            empty,
            dpsum,
            lse_log2,
            dout_e5m2,
        ]

    monkeypatch.setattr(
        interface,
        "_C_b300_lowp_bwd",
        SimpleNamespace(
            project_dout_unified_fp4_nvfp4_v510_e5m2=project,
        ),
    )
    input_operand = (object(), object(), object())
    weight_operand = (object(), object(), object())

    publication = interface.b300_project_dout_unified_lowp_nvfp4_v510_e5m2(
        input_operand,
        weight_operand,
        attention_output,
        lse,
        stats_workspace=stats_workspace,
        dq_clear=dq_clear,
    )

    assert type(publication) is interface.B300V510E5M2DoutPublication
    assert publication.dout_storage is dout_storage
    assert publication.backward_operands() == (dout_e5m2, dpsum, lse_log2)
    assert calls == [
        (
            *input_operand,
            *weight_operand,
            attention_output,
            lse,
            stats_workspace,
            dq_clear,
        )
    ]


@pytest.mark.parametrize(
    ("attention_shape", "stats_bytes", "dq_shape"),
    (
        ((2, 4096, 32, 128), 2 * 32 * 4096 * 4, (1, 4096, 32, 128)),
        ((1, 4096, 32, 128), 2 * 32 * 4096 * 4 - 1, (1, 4096, 32, 128)),
        ((1, 4096, 32, 128), 2 * 32 * 4096 * 4, (2, 4096, 32, 128)),
    ),
)
def test_v510_python_api_fails_closed_before_raw_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    attention_shape: tuple[int, ...],
    stats_bytes: int,
    dq_shape: tuple[int, ...],
) -> None:
    dispatched = False

    def project(*args: object) -> list[_FakeTensor]:
        nonlocal dispatched
        dispatched = True
        return []

    monkeypatch.setattr(
        interface,
        "_C_b300_lowp_bwd",
        SimpleNamespace(
            project_dout_unified_fp4_nvfp4_v510_e5m2=project,
        ),
    )
    attention_output = _FakeTensor(torch.bfloat16, attention_shape)
    stats_workspace = _FakeTensor(torch.uint8, (stats_bytes,))
    dq_clear = _FakeTensor(torch.bfloat16, dq_shape)

    with pytest.raises(ValueError, match="v510 E5M2 dO publication"):
        interface.b300_project_dout_unified_lowp_nvfp4_v510_e5m2(
            (object(), object(), object()),
            (object(), object(), object()),
            attention_output,
            object(),
            stats_workspace=stats_workspace,
            dq_clear=dq_clear,
        )
    assert not dispatched
