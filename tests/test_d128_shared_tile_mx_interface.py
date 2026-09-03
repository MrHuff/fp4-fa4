from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import tk_fa4.interface as interface


ROOT = Path(__file__).resolve().parents[1]
INTERFACE = ROOT / "tk_fa4/interface.py"


def _shared_tile_publications(
    *,
    batch: int = 1,
    sequence: int = 128,
    heads: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    depth = 128
    codes = (
        torch.arange(batch * sequence * heads * depth, dtype=torch.int64)
        .reshape(batch, sequence, heads, depth)
        .mul(7)
        .remainder(16)
        .to(torch.uint8)
    )

    feature_major = codes.permute(0, 2, 3, 1).contiguous()
    forward_bytes = (
        feature_major[..., 0::2] | (feature_major[..., 1::2] << 4)
    ).contiguous()
    forward_payload = forward_bytes.view(torch.float4_e2m1fn_x2)
    backward_payload = (
        codes[..., 0::2] | (codes[..., 1::2] << 4)
    ).contiguous()

    sequence_tiles = sequence // 128
    anchors = (
        torch.arange(
            batch * sequence_tiles * heads * 4 * 4,
            dtype=torch.int64,
        )
        .reshape(batch, sequence_tiles, heads, 4, 4)
        .add(1)
        .to(torch.uint8)
    )
    # Forward physical order is [depth_lane, depth_group, sequence_quarter].
    forward_scale_bytes = (
        anchors.permute(0, 1, 2, 4, 3)
        .unsqueeze(3)
        .expand(batch, sequence_tiles, heads, 32, 4, 4)
        .contiguous()
        .reshape(batch, sequence_tiles, heads, 512)
    )
    # Backward physical order is [sequence_lane, sequence_quarter, depth_group].
    backward_scales = (
        anchors.unsqueeze(3)
        .expand(batch, sequence_tiles, heads, 32, 4, 4)
        .contiguous()
        .reshape(batch, sequence_tiles, heads, 512)
    )
    forward_scales = forward_scale_bytes.view(torch.float8_e4m3fn)
    return forward_payload, forward_scales, backward_payload, backward_scales


def test_shared_tile_authenticator_accepts_exact_dual_orientation_abi() -> None:
    interface._b300_require_shared_tile_mxfp4_v(
        *_shared_tile_publications(sequence=256)
    )


def test_shared_tile_authenticator_rejects_payload_or_scale_divergence() -> None:
    forward, forward_scales, backward, backward_scales = (
        _shared_tile_publications()
    )

    corrupt_backward = backward.clone()
    corrupt_backward[0, 0, 0, 0] ^= 0x1
    with pytest.raises(RuntimeError, match="code matrix"):
        interface._b300_require_shared_tile_mxfp4_v(
            forward,
            forward_scales,
            corrupt_backward,
            backward_scales,
        )

    corrupt_forward_scale_bytes = forward_scales.view(torch.uint8).clone()
    corrupt_forward_scale_bytes.reshape(1, 1, 2, 32, 4, 4)[
        0, 0, 0, 1, 0, 0
    ] ^= 0x1
    with pytest.raises(RuntimeError, match="forward MXFP4 V scale replication"):
        interface._b300_require_shared_tile_mxfp4_v(
            forward,
            corrupt_forward_scale_bytes.view(torch.float8_e4m3fn),
            backward,
            backward_scales,
        )

    corrupt_backward_scales = backward_scales.clone()
    corrupt_backward_scales.reshape(1, 1, 2, 32, 4, 4)[
        0, 0, 0, :, 0, 0
    ] ^= 0x1
    with pytest.raises(RuntimeError, match="forward/backward MXFP4 V anchors"):
        interface._b300_require_shared_tile_mxfp4_v(
            forward,
            forward_scales,
            backward,
            corrupt_backward_scales,
        )


def _mock_shared_tile_extension(monkeypatch: pytest.MonkeyPatch) -> tuple[object, object]:
    base = "project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered"
    checked_name = base + "_shared_tile_mx_backward_v_mx_forward_out"
    unchecked_name = checked_name + "_unchecked"
    checked = object()
    unchecked = object()
    extension = SimpleNamespace(__file__="<mock-shared-tile-extension>")
    setattr(extension, checked_name, checked)
    setattr(extension, unchecked_name, unchecked)
    monkeypatch.setattr(interface, "_C_b300_lowp_bwd", extension)
    monkeypatch.setattr(
        interface,
        "b300_require_qkv_gqa_d128_unified_lowp_nvfp4_projection",
        lambda: base,
    )
    return checked, unchecked


def test_bound_shared_tile_route_selects_only_explicit_symbols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checked, unchecked = _mock_shared_tile_extension(monkeypatch)
    bound = interface.B300BoundD128NVFP4QKVProjection(
        batch=2,
        seqlen=4096,
        hidden=4096,
        q_heads=32,
        kv_heads=8,
        publish_mxfp4_v=True,
        v_mxfp4_scale_2d=True,
        per_block_qk_scales=True,
        experimental_output_shared_dual_v=False,
        experimental_mx_backward_v=False,
        experimental_shared_tile_mx_backward_v=True,
    )

    assert bound._project_checked is checked
    assert bound._project_unchecked is unchecked
    assert bound.experimental_shared_tile_mx_backward_v
    assert not bound.experimental_rowwise_mx_backward_v
    assert bound.experimental_mx_backward_v
    assert bound.v_backward_mxfp4_scale_policy == (
        interface.MXFP4_V_SCALE_POLICY_SHARED_D32XS32
    )
    assert bound.output_shared_dual_v_path == "shared_tile_mx_backward_v"
    assert bound.projection_forward_publication_path == (
        "caller_owned_shared_tile_mx_backward_v_d128"
    )
    assert bound.backward_publication_semantics.startswith(
        "single_quantized_d32xs32_mxfp4_v"
    )


def test_bound_shared_tile_route_rejects_conflicting_scale_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_shared_tile_extension(monkeypatch)
    common = dict(
        batch=2,
        seqlen=4096,
        hidden=4096,
        q_heads=32,
        kv_heads=8,
        publish_mxfp4_v=True,
        v_mxfp4_scale_2d=True,
        per_block_qk_scales=True,
        experimental_output_shared_dual_v=False,
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        interface.B300BoundD128NVFP4QKVProjection(
            **common,
            experimental_mx_backward_v=True,
            experimental_shared_tile_mx_backward_v=True,
        )

    with pytest.raises(ValueError, match="D32xS32 scales"):
        interface.B300BoundD128NVFP4QKVProjection(
            **(common | {"v_mxfp4_scale_2d": False}),
            experimental_shared_tile_mx_backward_v=True,
        )

    with pytest.raises(ValueError, match="publish_mxfp4_v=True"):
        interface.B300BoundD128NVFP4QKVProjection(
            **(common | {"publish_mxfp4_v": False}),
            experimental_shared_tile_mx_backward_v=True,
        )

    with pytest.raises(ValueError, match="authenticated only"):
        interface.B300BoundD128NVFP4QKVProjection(
            **(common | {"batch": 3}),
            experimental_shared_tile_mx_backward_v=True,
        )

    with pytest.raises(ValueError, match="per-row-K16"):
        interface.B300BoundD128NVFP4QKVProjection(
            **(common | {"per_block_qk_scales": False}),
            experimental_shared_tile_mx_backward_v=True,
        )


def test_shared_tile_compact_abi_appends_backward_payload_then_scales() -> None:
    source = INTERFACE.read_text(encoding="utf-8")
    compact_method = source.split(
        "def compact_mx_backward_v_outputs", 1
    )[1].split("def _b300_typed_fp4_alias", 1)[0]
    assert "*self.compact_outputs()," in compact_method
    assert compact_method.index("self.v_backward_mxfp4,") < compact_method.index(
        "self.v_backward_mxfp4_scale_pages,"
    )

    bound_call = source.split(
        "class B300BoundD128NVFP4QKVProjection", 1
    )[1].split("def b300_bind_qkv_gqa_d128", 1)[0]
    assert "expected_backward_count = 4 if self.experimental_mx_backward_v else 3" in bound_call
    assert "compact_backward = backward_tuple[2:]" in bound_call
    assert "mx_backward_v=self.experimental_mx_backward_v" in bound_call
