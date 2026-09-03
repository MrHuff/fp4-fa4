from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import tk_fa4
from tk_fa4 import interface


class _PackedOperand:
    def __init__(self, shape: tuple[int, int]) -> None:
        self.ndim = 2
        self.shape = shape


def _fake_forward_workspace() -> interface.B300E4M3QKVForwardWorkspace:
    workspace = object.__new__(interface.B300E4M3QKVForwardWorkspace)
    for field in interface.B300E4M3QKVForwardWorkspace.__dataclass_fields__:
        object.__setattr__(workspace, field, object())
    object.__setattr__(workspace, "v_mxfp4_payload", torch.arange(8))
    object.__setattr__(workspace, "v_mxfp4_scale_pages", torch.arange(4))
    object.__setattr__(workspace, "v_fp8_payload", torch.arange(6))
    return workspace


@pytest.mark.parametrize(
    (
        "publish_mxfp4_v",
        "output_shared_dual_v",
        "represented_backward",
        "route_suffix",
    ),
    (
        (False, False, False, "_fp8_forward_out"),
        (True, False, False, "_mx_forward_out"),
        (True, True, False, "_output_shared_dual_v_mx_forward_out"),
        (
            False,
            False,
            True,
            "_represented_backward_perblock_qk_fp8_forward_out",
        ),
    ),
)
def test_bound_d128_projection_authenticates_route_once_then_unchecked(
    monkeypatch: pytest.MonkeyPatch,
    publish_mxfp4_v: bool,
    output_shared_dual_v: bool,
    represented_backward: bool,
    route_suffix: str,
) -> None:
    base_symbol = (
        "project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered"
    )
    checked_symbol = base_symbol + route_suffix
    unchecked_symbol = checked_symbol + "_unchecked"
    calls: list[str] = []
    workspace = _fake_forward_workspace()
    backward_publications = tuple(
        SimpleNamespace(data_ptr=lambda pointer=pointer: pointer)
        for pointer in (301, 302, 303)
    )
    for field, publication in zip(
        ("v_backward_fp8", "q_backward_fp8", "k_backward_fp8"),
        backward_publications,
        strict=True,
    ):
        object.__setattr__(workspace, field, publication)

    def checked(*_args: object) -> tuple[object, object, object]:
        calls.append("checked")
        return backward_publications

    def unchecked(*_args: object) -> tuple[object, object, object]:
        calls.append("unchecked")
        return backward_publications

    extension = SimpleNamespace(__file__="/tmp/native-d128-projection.so")
    setattr(extension, checked_symbol, checked)
    setattr(extension, unchecked_symbol, unchecked)
    monkeypatch.setattr(interface, "_C_b300_lowp_bwd", extension)
    monkeypatch.setattr(
        interface,
        "b300_require_qkv_gqa_d128_unified_lowp_nvfp4_projection",
        lambda: base_symbol,
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
    legacy_kwargs: list[dict[str, object]] = []

    def legacy(*_args: object, **kwargs: object) -> object:
        calls.append("legacy")
        legacy_kwargs.append(kwargs)
        return legacy_bundle

    monkeypatch.setattr(
        interface,
        "b300_project_qkv_gqa_d128_unified_lowp_nvfp4",
        legacy,
    )
    authenticated: list[str] = []
    monkeypatch.setattr(
        interface,
        "_b300_require_bitwise_equal",
        lambda name, *_args: authenticated.append(name),
    )
    represented_authentications: list[tuple[object, object, object]] = []
    monkeypatch.setattr(
        interface,
        "_b300_require_represented_d128_nvfp4_qk_backward",
        lambda candidate_workspace, q_backward, k_backward: (
            represented_authentications.append(
                (candidate_workspace, q_backward, k_backward)
            )
        ),
    )
    returned_bundle = object()
    compact_kwargs: list[dict[str, object]] = []

    def compact(*_args: object, **kwargs: object) -> object:
        compact_kwargs.append(kwargs)
        return returned_bundle

    monkeypatch.setattr(interface, "_b300_compact_e4m3_qkv_bundle", compact)
    bound = interface.b300_bind_qkv_gqa_d128_unified_lowp_nvfp4_projection(
        batch=1,
        seqlen=4096,
        hidden=4096,
        q_heads=32,
        kv_heads=8,
        publish_mxfp4_v=publish_mxfp4_v,
        v_mxfp4_scale_2d=False,
        per_block_qk_scales=True,
        represented_backward=represented_backward,
        experimental_output_shared_dual_v=output_shared_dual_v,
    )
    operands = (
        (_PackedOperand((4096, 2048)), object(), object()),
        (_PackedOperand((6144, 2048)), object(), object()),
        object(),
        object(),
    )

    assert bound(*operands, forward_workspace=workspace) is returned_bundle
    assert calls == ["legacy", "checked"]
    assert legacy_kwargs == [
        {
            "batch": 1,
            "seqlen": 4096,
            "q_heads": 32,
            "kv_heads": 8,
            "store_bf16": False,
            "publish_fp8_backward": True,
            "v_mxfp4_scale_2d": False,
            "per_block_qk_scales": True,
            "rope_packed": operands[3],
            "cluster_cap": 68,
            "cache_packed_rope": True,
            "cache_adaptive_qk_scale": True,
        }
    ]
    inactive_names = (
        ["inactive FP8 V payload"]
        if publish_mxfp4_v
        else ["inactive MXFP4 V payload", "inactive MXFP4 V scale pages"]
    )
    assert authenticated[: len(inactive_names)] == inactive_names
    active_name = "MXFP4 V payload" if publish_mxfp4_v else "FP8 V payload"
    assert active_name in authenticated
    if represented_backward:
        assert authenticated[-1:] == ["backward V"]
        assert represented_authentications == [
            (workspace, backward_publications[1], backward_publications[2])
        ]
    else:
        assert authenticated[-3:] == [
            "backward V",
            "backward Q",
            "backward K",
        ]
        assert represented_authentications == []
    assert bound.abi_validated is True
    assert bound.validated_forward_workspace_count == 1
    assert (
        bound.experimental_output_shared_dual_v
        is output_shared_dual_v
    )
    assert bound.output_shared_dual_v_path == (
        "output_shared_dual_v"
        if output_shared_dual_v
        else "retained_dual_v"
        if publish_mxfp4_v
        else "fp8"
    )
    assert bound.represented_backward is represented_backward
    assert bound.qk_backward_source == (
        "represented_nvfp4_codes_per_row_k16"
        if represented_backward
        else "projection_accumulator_e4m3"
    )
    assert bound.projection_forward_publication_path == (
        "caller_owned_represented_qk_fp8_pv_d128"
        if represented_backward
        else "caller_owned_output_shared_dual_v_d128"
        if output_shared_dual_v
        else "caller_owned_route_selective_d128"
    )
    assert bound.backward_publication_semantics == (
        "represented_nvfp4_qk_per_row_k16_with_"
        "projection_accumulator_e4m3_v"
        if represented_backward
        else "projection_accumulator_e4m3_qkv_shared_across_pv_routes"
    )

    assert bound(*operands, forward_workspace=workspace) is returned_bundle
    assert calls == ["legacy", "checked", "unchecked"]
    assert len(represented_authentications) == int(represented_backward)
    assert bound.symbol == unchecked_symbol
    assert compact_kwargs == [
        {
            "q_heads": 32,
            "kv_heads": 8,
            "publish_mxfp4_v": publish_mxfp4_v,
            "head_dim": 128,
            "mx_backward_v": False,
            "mx_backward_v_scale_policy": None,
        },
        {
            "q_heads": 32,
            "kv_heads": 8,
            "publish_mxfp4_v": publish_mxfp4_v,
            "head_dim": 128,
            "mx_backward_v": False,
            "mx_backward_v_scale_policy": None,
        },
    ]


@pytest.mark.parametrize(
    ("seqlen", "cluster_cap", "cache_adaptive_qk_scale"),
    ((1024, 0, False), (4096, 68, True), (8192, 72, True)),
)
def test_bound_d128_projection_uses_cluster_and_cache_policy(
    monkeypatch: pytest.MonkeyPatch,
    seqlen: int,
    cluster_cap: int,
    cache_adaptive_qk_scale: bool,
) -> None:
    base_symbol = (
        "project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered"
    )
    checked_symbol = base_symbol + "_fp8_forward_out"
    extension = SimpleNamespace(__file__="/tmp/native-d128-policy.so")
    setattr(extension, checked_symbol, object())
    setattr(extension, checked_symbol + "_unchecked", object())
    monkeypatch.setattr(interface, "_C_b300_lowp_bwd", extension)
    monkeypatch.setattr(
        interface,
        "b300_require_qkv_gqa_d128_unified_lowp_nvfp4_projection",
        lambda: base_symbol,
    )
    bound = interface.B300BoundD128NVFP4QKVProjection(
        batch=1,
        seqlen=seqlen,
        hidden=4096,
        q_heads=32,
        kv_heads=8,
        publish_mxfp4_v=False,
        v_mxfp4_scale_2d=False,
        per_block_qk_scales=False,
    )
    assert bound.cluster_cap == cluster_cap
    assert bound.cache_packed_rope is True
    assert bound.cache_adaptive_qk_scale is cache_adaptive_qk_scale


def test_d128_projection_fails_closed_without_route_symbols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_symbol = (
        "project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered"
    )
    monkeypatch.setattr(
        interface,
        "_C_b300_lowp_bwd",
        SimpleNamespace(__file__="/tmp/d128-legacy-only.so"),
    )
    monkeypatch.setattr(
        interface,
        "b300_require_qkv_gqa_d128_unified_lowp_nvfp4_projection",
        lambda: base_symbol,
    )
    with pytest.raises(RuntimeError, match="d128-legacy-only.*mx_forward_out"):
        interface.B300BoundD128NVFP4QKVProjection(
            batch=1,
            seqlen=4096,
            hidden=4096,
            q_heads=32,
            kv_heads=8,
            publish_mxfp4_v=True,
            v_mxfp4_scale_2d=False,
            per_block_qk_scales=True,
        )


def test_d128_represented_qk_selection_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_symbol = (
        "project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered"
    )
    retained_fp8 = base_symbol + "_fp8_forward_out"
    extension = SimpleNamespace(__file__="/tmp/d128-direct-e4m3-only.so")
    setattr(extension, retained_fp8, object())
    setattr(extension, retained_fp8 + "_unchecked", object())
    monkeypatch.setattr(interface, "_C_b300_lowp_bwd", extension)
    monkeypatch.setattr(
        interface,
        "b300_require_qkv_gqa_d128_unified_lowp_nvfp4_projection",
        lambda: base_symbol,
    )
    common = {
        "batch": 1,
        "seqlen": 4096,
        "hidden": 4096,
        "q_heads": 32,
        "kv_heads": 8,
        "publish_mxfp4_v": False,
        "v_mxfp4_scale_2d": False,
        "per_block_qk_scales": True,
        "represented_backward": True,
    }

    with pytest.raises(
        RuntimeError,
        match=(
            "d128-direct-e4m3-only.*"
            "represented_backward_perblock_qk_fp8_forward_out"
        ),
    ):
        interface.b300_bind_qkv_gqa_d128_unified_lowp_nvfp4_projection(
            **common
        )

    with pytest.raises(TypeError, match="represented_backward.*exactly bool"):
        interface.b300_bind_qkv_gqa_d128_unified_lowp_nvfp4_projection(
            **{**common, "represented_backward": 1},  # type: ignore[arg-type]
        )

    invalid_shapes = (
        ({"batch": 3}, "B1/B2"),
        ({"seqlen": 2048}, "S4096"),
        ({"hidden": 8192}, "H4096"),
        ({"q_heads": 16}, "Hq32"),
        ({"kv_heads": 4}, "Hkv8"),
    )
    for override, match in invalid_shapes:
        with pytest.raises(ValueError, match=match):
            interface.b300_bind_qkv_gqa_d128_unified_lowp_nvfp4_projection(
                **{**common, **override}
            )

    invalid_policies = (
        ({"publish_mxfp4_v": True}, "requires FP8-PV"),
        ({"per_block_qk_scales": False}, "requires per-row-K16"),
        ({"v_mxfp4_scale_2d": True}, "does not accept an MXFP4"),
        (
            {"experimental_output_shared_dual_v": None},
            "experimental_output_shared_dual_v=False",
        ),
        (
            {"experimental_output_shared_dual_v": True},
            "experimental_output_shared_dual_v=False",
        ),
        (
            {"experimental_mx_backward_v": True},
            "incompatible with MX backward-V",
        ),
        (
            {"experimental_shared_tile_mx_backward_v": True},
            "incompatible with MX backward-V",
        ),
    )
    for override, match in invalid_policies:
        with pytest.raises(ValueError, match=match):
            interface.b300_bind_qkv_gqa_d128_unified_lowp_nvfp4_projection(
                **{**common, **override},
            )


def test_d128_output_shared_dual_v_selection_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_symbol = (
        "project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered"
    )
    retained = base_symbol + "_mx_forward_out"
    candidate = base_symbol + "_output_shared_dual_v_mx_forward_out"
    extension = SimpleNamespace(__file__="/tmp/d128-output-shared.so")
    for symbol in (retained, candidate):
        setattr(extension, symbol, object())
        setattr(extension, symbol + "_unchecked", object())
    monkeypatch.setattr(interface, "_C_b300_lowp_bwd", extension)
    monkeypatch.setattr(
        interface,
        "b300_require_qkv_gqa_d128_unified_lowp_nvfp4_projection",
        lambda: base_symbol,
    )
    common = {
        "batch": 1,
        "seqlen": 4096,
        "hidden": 4096,
        "q_heads": 32,
        "kv_heads": 8,
        "publish_mxfp4_v": True,
        "v_mxfp4_scale_2d": False,
        "per_block_qk_scales": True,
    }

    retained_bound = interface.B300BoundD128NVFP4QKVProjection(**common)
    assert retained_bound.checked_symbol == retained
    assert retained_bound.experimental_output_shared_dual_v is False

    automatic = (
        interface.b300_bind_qkv_gqa_d128_unified_lowp_nvfp4_projection(
            **common,
            experimental_output_shared_dual_v=None,
        )
    )
    assert automatic.checked_symbol == candidate
    assert automatic.experimental_output_shared_dual_v_requested is None
    assert automatic.experimental_output_shared_dual_v_resolved is True
    assert automatic.experimental_output_shared_split_v_resolved is True

    automatic_b2 = (
        interface.b300_bind_qkv_gqa_d128_unified_lowp_nvfp4_projection(
            **{**common, "batch": 2},
            experimental_output_shared_dual_v=None,
        )
    )
    assert automatic_b2.checked_symbol == candidate
    assert automatic_b2.experimental_output_shared_dual_v_requested is None
    assert automatic_b2.experimental_output_shared_dual_v_resolved is True

    delattr(extension, candidate)
    delattr(extension, candidate + "_unchecked")
    with pytest.raises(RuntimeError, match="output_shared_dual_v"):
        interface.B300BoundD128NVFP4QKVProjection(
            **common,
            experimental_output_shared_dual_v=True,
        )

    with pytest.raises(TypeError, match="exactly bool or None"):
        interface.B300BoundD128NVFP4QKVProjection(
            **common,
            experimental_output_shared_dual_v=1,  # type: ignore[arg-type]
        )
    for override, match in (
        ({"batch": 3}, "authenticated B1/B2/S4096"),
        ({"publish_mxfp4_v": False}, "publish_mxfp4_v=True"),
        ({"v_mxfp4_scale_2d": True}, "rowwise 1x32"),
        ({"per_block_qk_scales": False}, "per-row-K16"),
    ):
        invalid = dict(common)
        invalid.update(override)
        with pytest.raises(ValueError, match=match):
            interface.B300BoundD128NVFP4QKVProjection(
                **invalid,
                experimental_output_shared_dual_v=True,
            )


def test_d128_output_shared_dual_v_cuda_contract_is_source_explicit() -> None:
    root = Path(__file__).resolve().parents[1]
    cuda = (
        root / "tk_fa4" / "lowp_fa4_bwd" / "lowp_fa4_bwd.cu"
    ).read_text()
    epilogue = (
        root
        / "tk_fa4"
        / "lowp_fa4_bwd"
        / "projection_fp4_epilogue.cuh"
    ).read_text()
    symbol = (
        "project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered_"
        "output_shared_dual_v_mx_forward_out"
    )
    for suffix in ("", "_unchecked"):
        assert symbol + suffix in cuda
        assert f"&{symbol}{suffix}," in cuda
    candidate = cuda.split(
        f"TKFA4_DEFINE_D128_NVFP4_FORWARD_OUT(\n    {symbol},",
        1,
    )[1].split(")", 1)[0]
    assert candidate.replace(" ", "").replace("\n", "") == (
        "true,true,true,false,true"
    )
    checked = cuda.split(
        "if constexpr (ExperimentalOutputSharedDualV) {",
        1,
    )[1].split("}", 1)[0]
    for contract in (
        "(batch == 1 || batch == 2)",
        "seq_len == 4096",
        "q_heads == 32",
        "kv_heads == 8",
        "input_fp4.size(1) == 2048",
        "input_fp4.size(1) == 2048",
        "!v_mxfp4_scale_2d",
        "per_block_qk_scales",
    ):
        assert contract in checked
    branch = epilogue.split(
        "if constexpr (EXPERIMENTAL_OUTPUT_SHARED_SPLIT_V) {",
        1,
    )[1].split("output_rt registers;", 1)[0]
    assert "publish_v_fp8_from_output_shared<C, false>(" in branch
    assert "publish_v_mxfp4_from_output_shared<" in branch
    assert "INTERLEAVE_CAUSAL_KV" in branch
    assert "continue;" in branch
    assert "warpgroup::load" not in branch
    assert "stage_bf16_pairs" not in branch


def test_d128_represented_qk_cuda_contract_is_source_explicit() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "tk_fa4"
        / "lowp_fa4_bwd"
        / "lowp_fa4_bwd.cu"
    ).read_text()
    symbol = (
        "project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered_"
        "represented_backward_perblock_qk_fp8_forward_out"
    )
    candidate = source.split(
        f"TKFA4_DEFINE_D128_NVFP4_FORWARD_OUT(\n    {symbol},",
        1,
    )[1].split(")", 1)[0]
    assert candidate.replace(" ", "").replace("\n", "") == (
        "false,true,false,true,true"
    )
    represented_policy = source.split(
        "!PublishRepresentedBackwardFp8 ||",
        1,
    )[1].split(");", 1)[0]
    for contract in (
        "!PublishMxV",
        "!ExperimentalOutputSharedDualV",
        "!PublishMxBackwardV",
        "!ExperimentalCommonRowscaleMxBackwardV",
        "!ExperimentalSharedTileMxBackwardV",
        "PerBlockQkScales",
    ):
        assert contract in represented_policy
    checked_policy = source.split(
        "if constexpr (PublishRepresentedBackwardFp8) {",
        1,
    )[1].split("if constexpr (PublishMxBackwardV)", 1)[0]
    for contract in (
        "batch == 1 || batch == 2",
        "seq_len == 4096",
        "q_heads == 32",
        "kv_heads == 8",
        "B1/B2/S4096/H4096/Hq32/Hkv8/D128",
    ):
        assert contract in checked_policy
    for suffix in ("", "_unchecked"):
        pybind = (
            "m.def(\n"
            "        \"project_qkv_gqa_d128_unified_fp4_nvfp4_rope_"
            "packed_clustered_\"\n"
            f"        \"represented_backward_perblock_qk_fp8_forward_out"
            f"{suffix}\",\n"
            f"        &{symbol}{suffix},"
        )
        assert pybind in source


def test_represented_qk_first_use_oracle_models_satfinite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    represented = torch.tensor(
        [[[[200.0, -200.0]]]],
        dtype=torch.float32,
    )
    from tk_fa4.lowp_fa4_bwd import projection_quantization_reference

    monkeypatch.setattr(
        projection_quantization_reference,
        "decode_native_nvfp4_qk",
        lambda *_args, **_kwargs: represented.clone(),
    )
    references: list[torch.Tensor] = []
    monkeypatch.setattr(
        interface,
        "_b300_require_bitwise_equal",
        lambda _name, reference, _publication: references.append(reference),
    )
    workspace = SimpleNamespace(
        q_payload=object(),
        q_scale_pages=object(),
        q_global_scale=object(),
        k_payload=object(),
        k_scale_pages=torch.empty(1, 2, 1, 1),
        k_global_scale=object(),
    )

    interface._b300_require_represented_d128_nvfp4_qk_backward(
        workspace,
        object(),
        object(),
    )

    assert len(references) == 2
    for reference in references:
        assert reference.dtype is torch.float8_e4m3fn
        assert reference.float().flatten().tolist() == [448.0, -448.0]


def test_d128_cuda_and_python_export_route_selective_abi() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "tk_fa4"
        / "lowp_fa4_bwd"
        / "lowp_fa4_bwd.cu"
    ).read_text()
    base_symbol = (
        "project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered"
    )
    for suffix in (
        "_fp8_forward_out",
        "_fp8_forward_out_unchecked",
        "_represented_backward_perblock_qk_fp8_forward_out",
        "_represented_backward_perblock_qk_fp8_forward_out_unchecked",
        "_mx_forward_out",
        "_mx_forward_out_unchecked",
        "_output_shared_dual_v_mx_forward_out",
        "_output_shared_dual_v_mx_forward_out_unchecked",
    ):
        assert base_symbol + suffix in source
    assert "check_nvfp4_qkv_forward_outputs<kLogicalDepth>" in source
    assert (
        tk_fa4.b300_bind_qkv_gqa_d128_unified_lowp_nvfp4_projection
        is interface.b300_bind_qkv_gqa_d128_unified_lowp_nvfp4_projection
    )
    assert "B300BoundD128NVFP4QKVProjection" in tk_fa4.__all__
