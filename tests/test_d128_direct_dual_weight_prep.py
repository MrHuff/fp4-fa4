from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

import tk_fa4
import tk_fa4.interface as interface


def _arguments() -> tuple[torch.Tensor, ...]:
    q = torch.zeros((128, 128), dtype=torch.bfloat16)
    k = torch.zeros((128, 128), dtype=torch.bfloat16)
    v = torch.zeros((128, 128), dtype=torch.bfloat16)
    return (
        q,
        k,
        v,
        torch.zeros((384, 64), dtype=torch.uint8),
        torch.zeros((3, 2, 512), dtype=torch.uint8),
        torch.zeros((128, 192), dtype=torch.uint8),
        torch.zeros((1, 6, 512), dtype=torch.uint8),
        torch.zeros((1,), dtype=torch.float32),
    )


class _RecordingExtension:
    __file__ = "/tmp/recording-lowp-extension.so"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def quantize_gqa_d128_qkv_projection_weight_dual_out(
        self,
        *arguments: torch.Tensor,
    ) -> None:
        self.calls.append("checked")

    def quantize_gqa_d128_qkv_projection_weight_dual_out_unchecked(
        self,
        *arguments: torch.Tensor,
    ) -> None:
        self.calls.append("unchecked")


def test_direct_dual_api_is_exported() -> None:
    assert (
        tk_fa4.b300_prepare_gqa_d128_qkv_projection_weight_dual_out
        is interface.b300_prepare_gqa_d128_qkv_projection_weight_dual_out
    )


def test_direct_dual_selects_checked_and_unchecked_symbols_and_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extension = _RecordingExtension()
    monkeypatch.setattr(interface, "_C_b300_lowp_bwd", extension)
    arguments = _arguments()

    forward, backward = (
        interface.b300_prepare_gqa_d128_qkv_projection_weight_dual_out(
            *arguments,
            checked=True,
        )
    )
    assert extension.calls == ["checked"]
    assert forward == (arguments[3], arguments[4], arguments[7])
    assert backward == (arguments[5], arguments[6], arguments[7])

    interface.b300_prepare_gqa_d128_qkv_projection_weight_dual_out(
        *arguments,
        checked=False,
    )
    assert extension.calls == ["checked", "unchecked"]


def test_direct_dual_authentication_requires_checked_path() -> None:
    with pytest.raises(
        ValueError,
        match="bitwise authentication requires the checked path",
    ):
        interface.b300_prepare_gqa_d128_qkv_projection_weight_dual_out(
            *_arguments(),
            checked=False,
            authenticate=True,
        )


def test_checked_direct_dual_rejects_overlapping_storage_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extension = _RecordingExtension()
    monkeypatch.setattr(interface, "_C_b300_lowp_bwd", extension)
    arguments = list(_arguments())
    arguments[5] = arguments[3]

    with pytest.raises(ValueError, match="must use disjoint storage"):
        interface.b300_prepare_gqa_d128_qkv_projection_weight_dual_out(
            *arguments,
            checked=True,
        )
    assert extension.calls == []


def test_direct_dual_fails_closed_when_selected_extension_lacks_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _MissingExtension:
        __file__ = "/tmp/stale-lowp-extension.so"

    monkeypatch.setattr(interface, "_C_b300_lowp_bwd", _MissingExtension())
    with pytest.raises(
        RuntimeError,
        match="does not provide required direct D128 dual-weight",
    ):
        interface.b300_prepare_gqa_d128_qkv_projection_weight_dual_out(
            *_arguments(),
        )


def test_authentication_reference_uses_pair_interleaved_qk_and_canonical_v(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extension = _RecordingExtension()
    monkeypatch.setattr(interface, "_C_b300_lowp_bwd", extension)
    arguments = list(_arguments())
    row = torch.arange(128, dtype=torch.float32).to(torch.bfloat16)
    arguments[0] = row[:, None].expand(128, 128).contiguous()
    arguments[1] = (row + 256)[:, None].expand(128, 128).contiguous()
    arguments[2] = (row + 512)[:, None].expand(128, 128).contiguous()
    references: list[torch.Tensor] = []

    def prepare_reference(
        tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        references.append(tensor.clone())
        if len(references) == 1:
            return arguments[3], arguments[4], arguments[7]
        return arguments[5], arguments[6], arguments[7]

    monkeypatch.setattr(
        interface,
        "b300_prepare_nvfp4_projection_weight",
        prepare_reference,
    )
    interface.b300_prepare_gqa_d128_qkv_projection_weight_dual_out(
        *arguments,
        checked=True,
        authenticate=True,
    )

    expected_rows = torch.cat(
        (
            torch.stack((row[:64], row[64:]), dim=1).reshape(-1),
            torch.stack((row[:64] + 256, row[64:] + 256), dim=1).reshape(-1),
            row + 512,
        )
    ).to(torch.bfloat16)
    expected = expected_rows[:, None].expand(384, 128)
    assert len(references) == 2
    assert torch.equal(references[0], expected)
    assert torch.equal(references[1], expected.T.contiguous())


def test_native_source_contains_hardened_direct_dual_contract() -> None:
    source_path = (
        Path(interface.__file__).resolve().parent
        / "lowp_fa4_bwd"
        / "lowp_fa4_bwd.cu"
    )
    source = source_path.read_text()
    assert "struct nvfp4_dual_weight_globals" in source
    assert "quantize_nvfp4_dual_weight_kernel" in source
    assert "source_row = interleave_rows" in source
    assert "pair_interleave_qk = pair_interleave_qk" in source
    assert "kMaxCudaGridY = 65535" in source
    assert "must have a 16-byte-aligned base" in source
    assert "right_begin - left_begin < left_bytes" in source
    for symbol in (
        "quantize_gqa_d128_qkv_projection_weight_dual_out",
        "quantize_gqa_d128_qkv_projection_weight_dual_out_unchecked",
    ):
        assert source.count(f'"{symbol}"') == 1


@pytest.mark.skipif(
    os.environ.get("TK_FA4_RUN_GB200_D128_DUAL_AUTH") != "1",
    reason="production-shape GB200 authentication is opt-in",
)
def test_gb200_production_shape_direct_dual_is_bitwise_authenticated() -> None:
    assert torch.cuda.is_available()
    assert torch.cuda.get_device_capability()[0] == 10
    assert interface._C_b300_lowp_bwd is not None
    for symbol in (
        "quantize_gqa_d128_qkv_projection_weight_dual_out",
        "quantize_gqa_d128_qkv_projection_weight_dual_out_unchecked",
    ):
        assert getattr(interface._C_b300_lowp_bwd, symbol, None) is not None

    device = torch.device("cuda")
    hidden = 4096
    q_rows = 32 * 128
    kv_rows = 8 * 128
    total_rows = q_rows + 2 * kv_rows
    q = torch.randn((q_rows, hidden), device=device, dtype=torch.bfloat16)
    k = torch.randn((kv_rows, hidden), device=device, dtype=torch.bfloat16)
    v = torch.randn((kv_rows, hidden), device=device, dtype=torch.bfloat16)
    forward_packed = torch.empty(
        (total_rows, hidden // 2),
        device=device,
        dtype=torch.float4_e2m1fn_x2,
    )
    forward_scales = torch.empty(
        (total_rows // 128, hidden // 64, 512),
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    backward_packed = torch.empty(
        (hidden, total_rows // 2),
        device=device,
        dtype=torch.float4_e2m1fn_x2,
    )
    backward_scales = torch.empty(
        (hidden // 128, total_rows // 64, 512),
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    global_scale = torch.empty((1,), device=device, dtype=torch.float32)

    forward, backward = (
        interface.b300_prepare_gqa_d128_qkv_projection_weight_dual_out(
            q,
            k,
            v,
            forward_packed,
            forward_scales,
            backward_packed,
            backward_scales,
            global_scale,
            checked=True,
            authenticate=True,
        )
    )
    torch.cuda.synchronize()
    assert forward == (forward_packed, forward_scales, global_scale)
    assert backward == (backward_packed, backward_scales, global_scale)
