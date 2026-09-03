from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import tk_fa4
import tk_fa4.interface as interface
from tk_fa4.lowp_fa4_bwd.native_tk_d128_nvfp4_score_e5m2_dout_backward import (
    SUPPORTED_BATCHES,
    expected_extension_metadata,
)


ROOT = Path(__file__).resolve().parents[1]
CUDA = ROOT / "tk_fa4/lowp_fa4_bwd/lowp_fa4_bwd.cu"
EPILOGUE = ROOT / "tk_fa4/lowp_fa4_bwd/projection_fp4_epilogue.cuh"


def _publisher_extension(
    *,
    source_file: str = str(CUDA),
    **overrides: object,
) -> SimpleNamespace:
    metadata = dict(interface.V509_E5M2_DOUT_PUBLISHER_METADATA)
    metadata.update(overrides)
    metadata["source_file"] = source_file
    return SimpleNamespace(
        project_dout_unified_fp4_nvfp4_v509_e5m2_metadata=(
            lambda: dict(metadata)
        )
    )


def test_v509_fused_publication_is_a_separate_public_api() -> None:
    assert (
        tk_fa4.B300V509E5M2DoutPublication
        is interface.B300V509E5M2DoutPublication
    )
    assert (
        tk_fa4.b300_project_dout_unified_lowp_nvfp4_v509_e5m2
        is interface.b300_project_dout_unified_lowp_nvfp4_v509_e5m2
    )
    assert (
        tk_fa4.b300_require_v509_e5m2_dout_route
        is interface.b300_require_v509_e5m2_dout_route
    )


@pytest.mark.parametrize("batch", SUPPORTED_BATCHES)
def test_v509_route_pairing_accepts_only_exact_publisher_and_backward(
    monkeypatch: pytest.MonkeyPatch,
    batch: int,
) -> None:
    publisher = _publisher_extension()
    monkeypatch.setattr(interface, "_C_b300_lowp_bwd", publisher)
    backward = expected_extension_metadata(batch)

    receipt = interface.b300_require_v509_e5m2_dout_route(
        dict(backward)
    )

    assert receipt["route"] == "v509_only_fail_closed"
    assert receipt["publisher"] == dict(
        interface.V509_E5M2_DOUT_PUBLISHER_METADATA
    )
    assert receipt["backward"] == backward
    assert receipt["backward"]["batch"] == batch
    assert receipt["publisher"]["batch_values"] == SUPPORTED_BATCHES
    assert "batch" not in receipt["publisher"]


@pytest.mark.parametrize(
    "source_file",
    (
        "lowp_fa4_bwd.cu",
        "tk_fa4/lowp_fa4_bwd/lowp_fa4_bwd.cu",
        "/workspace/fp4_matmul/tk_fa4/lowp_fa4_bwd/lowp_fa4_bwd.cu",
    ),
)
def test_v509_route_accepts_only_bare_or_canonical_publisher_source(
    monkeypatch: pytest.MonkeyPatch,
    source_file: str,
) -> None:
    monkeypatch.setattr(
        interface,
        "_C_b300_lowp_bwd",
        _publisher_extension(source_file=source_file),
    )

    interface.b300_require_v509_e5m2_dout_route(
        expected_extension_metadata(1)
    )


@pytest.mark.parametrize(
    "source_file",
    (
        "/tmp/lowp_fa4_bwd.cu",
        "projection_fp4_epilogue.cuh",
        "lowp_fa4_bwd.cu.bak",
    ),
)
def test_v509_route_rejects_noncanonical_publisher_source(
    monkeypatch: pytest.MonkeyPatch,
    source_file: str,
) -> None:
    monkeypatch.setattr(
        interface,
        "_C_b300_lowp_bwd",
        _publisher_extension(source_file=source_file),
    )

    with pytest.raises(RuntimeError, match="v509 E5M2 dO route"):
        interface.b300_require_v509_e5m2_dout_route(
            expected_extension_metadata(1)
        )


@pytest.mark.parametrize(
    ("side", "field", "value"),
    (
        ("publisher", "payload_dtype", "float8_e4m3fn"),
        ("publisher", "batch_values", (1, 2)),
        ("publisher", "batch_values", [1, 2, 4]),
        ("publisher", "probability_log2_lift", 0.0),
        ("publisher", "dstat_physical_abi", "-16*sum(O*dO)"),
        ("backward", "dout_dtype", "float8_e4m3fn_represented_x4"),
        ("backward", "batch", 3),
        ("backward", "lstat_abi", "-LSE*log2(e)"),
    ),
)
def test_v509_route_pairing_rejects_any_metadata_drift(
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
    backward = expected_extension_metadata(1)
    if side == "backward":
        backward[field] = value

    with pytest.raises(RuntimeError, match="v509 E5M2 dO route"):
        interface.b300_require_v509_e5m2_dout_route(backward)


@pytest.mark.parametrize(
    ("metadata_batch", "claimed_batch"),
    ((1, 2), (1, 4), (2, 1), (2, 4), (4, 1), (4, 2)),
)
def test_v509_route_rejects_cross_batch_backward_metadata(
    monkeypatch: pytest.MonkeyPatch,
    metadata_batch: int,
    claimed_batch: int,
) -> None:
    monkeypatch.setattr(
        interface,
        "_C_b300_lowp_bwd",
        _publisher_extension(),
    )
    backward = expected_extension_metadata(metadata_batch)
    backward["batch"] = claimed_batch

    with pytest.raises(RuntimeError, match="v509 E5M2 dO route"):
        interface.b300_require_v509_e5m2_dout_route(backward)


def test_v509_epilogue_selects_e5_only_for_output_dgrad() -> None:
    source = EPILOGUE.read_text()

    assert "bool PUBLISH_DOUT_E5M2 = false" in source
    assert "encode_e5m2_pair_x4" in source
    assert "decode_e5m2_pair" in source
    assert "g.v_backward_fp8 + output_base" in source
    assert "PUBLISH_DOUT_E5M2" in source.split(
        "void publish_v_fp8_from_output_shared(", 1
    )[1].split("template <", 1)[0]

    qkv_publisher = source.split("void publish_v_fp8(", 1)[1].split(
        "template <typename C>", 1
    )[0]
    assert "E5M2" not in qkv_publisher
    assert "e5m2" not in qkv_publisher


def test_v509_raw_symbol_keeps_old_e4_eight_slot_abi_separate() -> None:
    source = CUDA.read_text()

    assert source.count('"project_dout_unified_fp4_nvfp4"') == 1
    assert "&project_dout_unified_fp4_nvfp4," in source
    assert (
        '"project_dout_unified_fp4_nvfp4_v509_e5m2"' in source
    )
    assert (
        "&project_dout_unified_fp4_nvfp4_v509_e5m2," in source
    )
    assert "at::ScalarType::Float8_e5m2" in source
    assert (
        "v509 E5M2 dO publication is restricted to B1/B2/B4 at "
        in source
    )
    assert "batch == 1 || batch == 2 || batch == 4" in source


def test_v509_raw_symbol_authenticates_all_real_write_ranges() -> None:
    source = CUDA.read_text()
    implementation = source.split(
        "project_dout_unified_fp4_nvfp4_impl(", 1
    )[1].split(
        "std::vector<at::Tensor> project_dout_unified_fp4_nvfp4(", 1
    )[0]

    assert "if constexpr (PublishE5M2Dout)" in implementation
    for output in (
        '"dout_backward_e5m2", &dout_backward_fp8',
        '"stats_workspace", &stats_workspace.value()',
        '"dq_clear", &dq_clear.value()',
    ):
        assert output in implementation
    for read in (
        '"input_fp4", &input_fp4',
        '"input_scales", &input_scales',
        '"input_global_scale", &input_global_scale',
        '"weight_fp4", &weight_fp4',
        '"weight_scales", &weight_scales',
        '"weight_global_scale", &weight_global_scale',
        '"attention_output", &attention_output',
        '"lse", &lse',
    ):
        assert read in implementation
    assert "v509 E5M2 write destinations must use disjoint storage" in source
    assert "must not overlap v509 E5M2 write destination" in source


def test_v509_python_api_exposes_real_e5m2_slot_seven() -> None:
    source = Path(interface.__file__).read_text()
    function = source.split(
        "def b300_project_dout_unified_lowp_nvfp4_v509_e5m2(", 1
    )[1].split("\ndef ", 1)[0]

    assert (
        "_C_b300_lowp_bwd.project_dout_unified_fp4_nvfp4_v509_e5m2("
        in function
    )
    assert "dout_backward_e5m2=projected[7]" in function
    assert "torch.float8_e5m2" in function
    assert "stats_workspace.numel() != expected_stats_bytes" in function
    assert "with exactly {expected_stats_bytes}" in function
    assert "batch not in (1, 2, 4)" in function
    assert "expected_shape = (batch, 4096, 32, 128)" in function
    assert "expected_stats_shape = (batch, 32, 1, 4096)" in function
    assert "store_bf16" not in function
    assert "publish_fp8_backward" not in function
