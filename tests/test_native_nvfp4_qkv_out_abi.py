from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import tk_fa4
from tk_fa4 import interface


def _fake_forward_workspace() -> interface.B300E4M3QKVForwardWorkspace:
    workspace = object.__new__(interface.B300E4M3QKVForwardWorkspace)
    for field in interface.B300E4M3QKVForwardWorkspace.__dataclass_fields__:
        object.__setattr__(workspace, field, object())
    return workspace


@pytest.mark.parametrize(
    ("publish_mxfp4_v", "base_symbol", "compact_suffix"),
    (
        (
            False,
            "project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_"
            "represented_backward_perblock_qk",
            "_fp8_forward_out",
        ),
        (
            True,
            "project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_"
            "interleaved_causal_represented_backward_perblock_qk_"
            "split_v_backward",
            "_mx_forward_out",
        ),
    ),
)
def test_bound_native_projection_authenticates_once_then_uses_unchecked(
    monkeypatch: pytest.MonkeyPatch,
    publish_mxfp4_v: bool,
    base_symbol: str,
    compact_suffix: str,
) -> None:
    calls: list[tuple[str, tuple[object, ...]]] = []
    legacy_kwargs: list[dict[str, object]] = []
    checked_symbol = base_symbol + compact_suffix
    unchecked_symbol = checked_symbol + "_unchecked"
    workspace = _fake_forward_workspace()
    backward_publications = tuple(
        SimpleNamespace(data_ptr=lambda pointer=pointer: pointer)
        for pointer in (101, 102, 103)
    )
    for field, publication in zip(
        ("v_backward_fp8", "q_backward_fp8", "k_backward_fp8"),
        backward_publications,
        strict=True,
    ):
        object.__setattr__(workspace, field, publication)

    def checked(*args: object) -> tuple[object, object, object]:
        calls.append(("checked", args))
        return backward_publications

    def unchecked(*args: object) -> tuple[object, object, object]:
        calls.append(("unchecked", args))
        return backward_publications

    extension = SimpleNamespace(__file__="/tmp/native-projection.so")
    setattr(extension, checked_symbol, checked)
    setattr(extension, unchecked_symbol, unchecked)
    monkeypatch.setattr(interface, "_C_b300_lowp_bwd", extension)
    monkeypatch.setattr(
        interface,
        "b300_require_qkv_gqa_d64_paired_unified_lowp_nvfp4_projection",
        lambda **_kwargs: base_symbol,
    )
    legacy_bundle = SimpleNamespace(
        backward=SimpleNamespace(
            score_q_fp4=object(),
            score_k_fp4=object(),
        ),
        q_forward_scales=object(),
        q_forward_global_scale=object(),
        k_forward_scales=object(),
        k_forward_global_scale=object(),
        v_forward_fp4=object(),
        v_forward_scales=object(),
        v_forward_fp8=object(),
        v_backward_fp8=object(),
        q_backward_fp8=object(),
        k_backward_fp8=object(),
    )

    def legacy(*_args: object, **kwargs: object) -> object:
        calls.append(("legacy", _args))
        legacy_kwargs.append(kwargs)
        return legacy_bundle

    monkeypatch.setattr(
        interface,
        "b300_project_qkv_gqa_d64_paired_unified_lowp_nvfp4",
        legacy,
    )
    authenticated: list[str] = []
    monkeypatch.setattr(
        interface,
        "_b300_require_bitwise_equal",
        lambda name, *_args, **_kwargs: authenticated.append(name),
    )
    returned_bundle = object()
    monkeypatch.setattr(
        interface,
        "_b300_compact_e4m3_qkv_bundle",
        lambda *_args, **_kwargs: returned_bundle,
    )
    bound = interface.B300BoundNVFP4QKVProjection(
        batch=16,
        seqlen=4096,
        q_heads=32,
        kv_heads=8,
        publish_mxfp4_v=publish_mxfp4_v,
        v_mxfp4_scale_2d=False,
        experimental_output_shared_split_v=False,
    )
    operands = (
        (object(), object(), object()),
        (object(), object(), object()),
        object(),
        object(),
    )
    with pytest.raises(
        TypeError, match="requires a B300E4M3QKVForwardWorkspace"
    ):
        bound(*operands, forward_workspace=object())

    assert bound(*operands, forward_workspace=workspace) is returned_bundle
    assert [name for name, _args in calls] == ["legacy", "checked"]
    assert legacy_kwargs == [
        {
            "batch": 16,
            "seqlen": 4096,
            "q_heads": 32,
            "kv_heads": 8,
            "store_bf16": False,
            "publish_fp8_backward": True,
            "interleave_causal_kv": publish_mxfp4_v,
            "v_mxfp4_scale_2d": False,
            "represented_backward": True,
            "per_block_qk_scales": True,
            "experimental_split_v_backward": publish_mxfp4_v,
        }
    ]
    checked_args = calls[1][1]
    assert checked_args[8:13] == (16, 4096, 32, 8, False)
    assert bound.abi_validated is True
    assert bound.validated_forward_workspace_count == 1
    assert bound.experimental is True
    assert (
        bound.backward_publication_semantics
        == "represented_nvfp4_qk_per_row_k16_with_"
        "projection_accumulator_e4m3_v"
    )
    assert bound.represented_backward is True
    assert bound.per_block_qk_scales is True
    assert bound.experimental_split_v_backward is publish_mxfp4_v

    assert bound(*operands, forward_workspace=workspace) is returned_bundle
    assert [name for name, _args in calls] == [
        "legacy",
        "checked",
        "unchecked",
    ]
    assert calls[2][1][8:13] == (16, 4096, 32, 8, False)
    assert bound.symbol == unchecked_symbol
    expected_v_name = "MXFP4 V payload" if publish_mxfp4_v else "FP8 V payload"
    assert expected_v_name in authenticated
    assert authenticated[-3:] == ["backward V", "backward Q", "backward K"]


def test_bound_native_projection_rejects_backward_storage_outside_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_symbol = (
        "project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_"
        "represented_backward_perblock_qk"
    )
    checked_symbol = base_symbol + "_fp8_forward_out"
    unchecked_symbol = checked_symbol + "_unchecked"
    workspace = _fake_forward_workspace()
    for pointer, field in zip(
        (101, 102, 103),
        ("v_backward_fp8", "q_backward_fp8", "k_backward_fp8"),
        strict=True,
    ):
        object.__setattr__(
            workspace,
            field,
            SimpleNamespace(data_ptr=lambda pointer=pointer: pointer),
        )
    stale_publications = tuple(
        SimpleNamespace(data_ptr=lambda pointer=pointer: pointer)
        for pointer in (201, 202, 203)
    )
    extension = SimpleNamespace(__file__="/tmp/stale-native-projection.so")
    setattr(extension, checked_symbol, lambda *_args: stale_publications)
    setattr(extension, unchecked_symbol, lambda *_args: stale_publications)
    monkeypatch.setattr(interface, "_C_b300_lowp_bwd", extension)
    monkeypatch.setattr(
        interface,
        "b300_require_qkv_gqa_d64_paired_unified_lowp_nvfp4_projection",
        lambda **_kwargs: base_symbol,
    )
    monkeypatch.setattr(
        interface,
        "b300_project_qkv_gqa_d64_paired_unified_lowp_nvfp4",
        lambda *_args, **_kwargs: object(),
    )
    bound = interface.B300BoundNVFP4QKVProjection(
        batch=2,
        seqlen=4096,
        q_heads=32,
        kv_heads=8,
        publish_mxfp4_v=False,
        v_mxfp4_scale_2d=False,
    )

    with pytest.raises(
        RuntimeError, match="backward V outside its caller-owned workspace"
    ):
        bound(
            (object(), object(), object()),
            (object(), object(), object()),
            object(),
            object(),
            forward_workspace=workspace,
        )


def test_native_projection_binder_fails_closed_on_missing_out_symbols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale_base_symbol = (
        "project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed"
    )
    extension = SimpleNamespace(__file__="/tmp/legacy-only.so")
    for symbol in (
        stale_base_symbol,
        stale_base_symbol + "_fp8_forward_out",
        stale_base_symbol + "_fp8_forward_out_unchecked",
    ):
        setattr(extension, symbol, object())
    monkeypatch.setattr(interface, "_C_b300_lowp_bwd", extension)

    with pytest.raises(
        RuntimeError,
        match="legacy-only.*represented_backward_perblock_qk",
    ):
        interface.B300BoundNVFP4QKVProjection(
            batch=1,
            seqlen=4096,
            q_heads=32,
            kv_heads=8,
            publish_mxfp4_v=False,
            v_mxfp4_scale_2d=False,
        )


def test_native_cuda_has_checked_unchecked_route_specific_out_symbols() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "tk_fa4"
        / "lowp_fa4_bwd"
        / "lowp_fa4_bwd.cu"
    ).read_text()
    exact_base = (
        "project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_"
        "represented_backward_perblock_qk"
    )
    mx_base = (
        "project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_"
        "interleaved_causal_represented_backward_perblock_qk_"
        "split_v_backward"
    )
    for symbol in (
        exact_base,
        mx_base,
        exact_base + "_fp8_forward_out",
        exact_base + "_fp8_forward_out_unchecked",
        mx_base + "_mx_forward_out",
        mx_base + "_mx_forward_out_unchecked",
    ):
        assert symbol in source
    assert "constexpr bool kPublishMxfp4V" in source
    assert "kPublishRepresentedBackwardFp8" in source
    assert "? kInterleaveCausalKv" in source
    assert "kCompactForwardOut" in source
    assert "? kCompactPublishesMxV" in source
    assert (
        ": (kPublishRepresentedBackwardFp8 ? kInterleaveCausalKv : true)"
        in source
    )
    assert "kExperimentalSplitVBackward" in source
    assert "kPerBlockQkScales" in source
    assert (
        "kInterleaveCausalKv, kPublishForwardFp8,\n"
        "                kPublishRepresentedBackwardFp8, PerBlockQkScales,\n"
        "                kExperimentalSplitVBackward"
    ) in source
    assert (
        "!kPublishRepresentedBackwardFp8 ||\n"
        "            (\n"
        "                kQkDepth == 128 && kPerBlockQkScales"
    ) in source
    assert "kCompactForwardOut && !kInterleaveCausalKv" in source
    assert "!kCompactPublishesMxV" in source
    assert (
        "!kExperimentalSplitVBackward ||\n"
        "            (kPublishRepresentedBackwardFp8 && kPerBlockQkScales &&\n"
        "             kInterleaveCausalKv)"
    ) in source
    assert "return {v_backward_fp8, q_backward_fp8, k_backward_fp8};" in source
    assert "input_scales.size(0) == rows / 128" in source
    assert "qkv_weight_scales.size(0) == total_width / 128" in source


def test_native_projection_is_explicit_and_batched_gate_stays_closed() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "tk_fa4"
        / "lowp_fa4_bwd"
        / "benchmark_llama12b_e2e.py"
    ).read_text()
    assert "--experimental-native-nvfp4-projection-out" in source
    assert "elif self.experimental_native_nvfp4_projection_out:" in source
    native_bind = source.split(
        "elif self.experimental_native_nvfp4_projection_out:", 1
    )[1].split("if self.qkv_projection is not None:", 1)[0]
    assert (
        "b300_bind_qkv_gqa_d64_paired_unified_lowp_nvfp4_projection("
        in native_bind
    )
    assert "batch=config.batch" in native_bind
    projection_stage = source.split(
        'with _stage("lowp/fwd/qkv_projection_rope_publish"):', 1
    )[1].split('with _stage("lowp/fwd/attention"):', 1)[0]
    assert "if runtime.qkv_projection is not None:" in projection_stage
    batched_gate = source.split(
        "def _require_batched_exact_runtime_contract(", 1
    )[1].split(
        "def _require_experimental_native_batched_runtime_contract(", 1
    )[0]
    assert 'if qkv_projection_format != "e4m3":' in batched_gate
    assert 'violations.append("E4M3 QKV projection")' in batched_gate
    native_gate = source.split(
        "def _require_experimental_native_batched_runtime_contract(", 1
    )[1].split("class LowpAttentionRuntime:", 1)[0]
    assert 'if config.batch != 16:' in native_gate
    assert 'if qkv_projection_format != "nvfp4":' in native_gate
    assert "experimental native NVFP4 B16 FA4 requires" in native_gate
    contract_selection = source.split(
        "require_runtime_contract = (", 1
    )[1].split("require_runtime_contract(", 1)[0]
    assert "if experimental_native_nvfp4_projection_out" in contract_selection


def test_native_projection_binder_is_exported() -> None:
    assert (
        tk_fa4.b300_bind_qkv_gqa_d64_paired_unified_lowp_nvfp4_projection
        is interface.b300_bind_qkv_gqa_d64_paired_unified_lowp_nvfp4_projection
    )
    assert "B300BoundNVFP4QKVProjection" in tk_fa4.__all__
