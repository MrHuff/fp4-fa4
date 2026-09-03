from __future__ import annotations

from pathlib import Path

import pytest

import tk_fa4
import tk_fa4.interface as interface


ROOT = Path(__file__).resolve().parents[1]
CUDA = ROOT / "tk_fa4" / "lowp_fa4_bwd" / "lowp_fa4_bwd.cu"
EPILOGUE = (
    ROOT / "tk_fa4" / "lowp_fa4_bwd" / "projection_fp4_epilogue.cuh"
)


class _RecordingExtension:
    __file__ = "/tmp/recording-e4m3-output-projection.so"

    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.output = object()

    def project_e4m3_generic(self, *arguments: object) -> object:
        self.calls.append(arguments)
        return self.output


def test_e4m3_output_projection_is_public_and_dispatches_both_operands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert tk_fa4.b300_project_e4m3 is interface.b300_project_e4m3
    extension = _RecordingExtension()
    monkeypatch.setattr(interface, "_C_b300_lowp_bwd", extension)
    input_operand = (object(), object())
    weight_operand = (object(), object())

    output = interface.b300_project_e4m3(input_operand, weight_operand)

    assert output is extension.output
    assert extension.calls == [(*input_operand, *weight_operand)]


@pytest.mark.parametrize(
    ("input_operand", "weight_operand"),
    (
        ((object(),), (object(), object())),
        ((object(), object(), object()), (object(), object())),
        ((object(), object()), (object(),)),
        ((object(), object()), (object(), object(), object())),
    ),
)
def test_e4m3_output_projection_rejects_malformed_operand_tuples(
    monkeypatch: pytest.MonkeyPatch,
    input_operand: tuple[object, ...],
    weight_operand: tuple[object, ...],
) -> None:
    extension = _RecordingExtension()
    monkeypatch.setattr(interface, "_C_b300_lowp_bwd", extension)

    with pytest.raises(ValueError, match="E4M3 operands must contain"):
        interface.b300_project_e4m3(input_operand, weight_operand)

    assert extension.calls == []


def test_native_e4m3_output_projection_uses_dense_k128_single_output() -> None:
    source = CUDA.read_text()
    implementation = source.split(
        "at::Tensor project_e4m3_generic(",
        1,
    )[1].split(
        "// Caller-owned output ABI shared by the D64 and D128",
        1,
    )[0]

    assert "tkfa4_projection::config<4, 4, 128, 128, true>" in implementation
    assert "input_row_decode.numel() == rows64" in implementation
    assert "weight_channel_decode.numel() == output_width64" in implementation
    assert ".D = kittens::py::tensor_to_gl<typename G::D_gl>" in implementation
    assert ".batch = 1" in implementation
    assert ".seq_len = rows" in implementation
    assert ".output_width = output_width" in implementation
    assert "tkfa4_projection::launch_on_stream<" in implementation
    for launch_contract in (
        "true,   // STORE_BF16",
        "true,   // SINGLE_OUTPUT",
        "false   // APPLY_ROPE",
    ):
        assert launch_contract in implementation
    assert source.count('"project_e4m3_generic"') == 1
    assert "&project_e4m3_generic" in source


def test_dense_e4m3_bf16_store_is_narrowed_to_plain_single_output() -> None:
    source = EPILOGUE.read_text()
    assertion = source.split(
        "dense E4M3 BF16 storage requires a publication-free single output",
        1,
    )[0].rsplit("static_assert(", 1)[1]

    for required_clause in (
        "!C::DENSE_FP8 || !STORE_BF16",
        "SINGLE_OUTPUT && !OUTPUT_IS_DOUT && !PUBLISH_FP4",
        "!PUBLISH_FORWARD_QK && !PUBLISH_V_MXFP4",
        "!PUBLISH_V_FP8 && !PUBLISH_QK_FP8",
    ):
        assert required_clause in assertion
