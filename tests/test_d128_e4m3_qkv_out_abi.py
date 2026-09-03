from __future__ import annotations

import ast
import gc
from pathlib import Path
from types import SimpleNamespace

import pytest

import tk_fa4
from tk_fa4 import interface


ROOT = Path(__file__).resolve().parents[1]
CUDA = ROOT / "tk_fa4" / "lowp_fa4_bwd" / "lowp_fa4_bwd.cu"
RUNTIME = ROOT / "tk_fa4" / "lowp_fa4_bwd" / "benchmark_llama12b_e2e.py"


def _v501_contract_namespace() -> dict[str, object]:
    selected = []
    for node in ast.parse(RUNTIME.read_text()).body:
        name = getattr(node, "name", None)
        targets = (
            [target.id for target in node.targets if isinstance(target, ast.Name)]
            if isinstance(node, ast.Assign)
            else []
        )
        if name in {
            "_require_native_tk_d128_runtime_contract",
            "_require_d128_e4m3_v501_runtime_contract",
        } or "AUTHENTICATED_D128_EXACT_BATCHES" in targets:
            selected.append(node)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            *selected,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {}
    exec(compile(module, str(RUNTIME), "exec"), namespace)
    return namespace


class _FakeTensor:
    def __init__(self, shape: tuple[int, ...], pointer: int) -> None:
        self.shape = shape
        self._pointer = pointer

    def data_ptr(self) -> int:
        return self._pointer

    def clone(self) -> _FakeTensor:
        return _FakeTensor(self.shape, self._pointer + 100_000)


def _fake_workspace() -> interface.B300E4M3QKVForwardWorkspace:
    workspace = object.__new__(interface.B300E4M3QKVForwardWorkspace)
    for index, field in enumerate(
        interface.B300E4M3QKVForwardWorkspace.__dataclass_fields__,
        start=1,
    ):
        object.__setattr__(workspace, field, _FakeTensor((1,), index))
    return workspace


@pytest.mark.parametrize("publish_mxfp4_v", (False, True))
def test_bound_d128_e4m3_authenticates_once_then_uses_unchecked(
    monkeypatch: pytest.MonkeyPatch,
    publish_mxfp4_v: bool,
) -> None:
    base = "project_qkv_gqa_d128_unified_fp8_e4m3_rope_packed_clustered"
    suffix = "_mx_forward_out" if publish_mxfp4_v else "_fp8_forward_out"
    checked_symbol = base + suffix
    unchecked_symbol = checked_symbol + "_unchecked"
    calls: list[str] = []
    workspace = _fake_workspace()
    backward = (
        workspace.v_backward_fp8,
        workspace.q_backward_fp8,
        workspace.k_backward_fp8,
    )

    legacy = [_FakeTensor((1,), 1_000 + index) for index in range(24)]
    legacy[4] = _FakeTensor((1,), 2_004)
    legacy[6] = _FakeTensor((1,), 2_006)
    legacy[8] = _FakeTensor((1,), 2_008)
    legacy[9] = _FakeTensor((1,), 2_009)
    legacy[10] = _FakeTensor((1,), 2_010)
    legacy[11] = _FakeTensor((1,), 2_011)
    legacy[12] = _FakeTensor((1,), 2_012)
    legacy[13] = _FakeTensor((1,), 2_013)
    legacy[20] = _FakeTensor((1,), 2_020)
    legacy[21] = _FakeTensor((1,), 2_021)
    legacy[22] = _FakeTensor((1,), 2_022)
    legacy[23] = _FakeTensor((1,), 2_023)

    def allocating(*_args: object) -> tuple[object, ...]:
        calls.append("allocating")
        return tuple(legacy)

    def checked(*_args: object) -> tuple[object, ...]:
        calls.append("checked")
        return backward

    def unchecked(*_args: object) -> tuple[object, ...]:
        calls.append("unchecked")
        return backward

    extension = SimpleNamespace(__file__="/tmp/d128-e4m3-qkv.so")
    setattr(extension, base, allocating)
    setattr(extension, checked_symbol, checked)
    setattr(extension, unchecked_symbol, unchecked)
    monkeypatch.setattr(interface, "_C_b300_lowp_bwd", extension)
    monkeypatch.setattr(
        interface,
        "b300_require_qkv_gqa_d128_unified_lowp_e4m3_projection",
        lambda: base,
    )
    authenticated: list[str] = []
    monkeypatch.setattr(
        interface,
        "_b300_require_bitwise_equal",
        lambda name, *_args: authenticated.append(name),
    )
    bundle = object()
    compact_calls: list[dict[str, object]] = []

    def compact(*_args: object, **kwargs: object) -> object:
        compact_calls.append(kwargs)
        return bundle

    monkeypatch.setattr(interface, "_b300_compact_e4m3_qkv_bundle", compact)
    bound = interface.B300BoundD128E4M3QKVProjection(
        batch=2,
        seqlen=4096,
        hidden=4096,
        q_heads=32,
        kv_heads=8,
        publish_mxfp4_v=publish_mxfp4_v,
    )
    operands = (
        (
            _FakeTensor((8192, 4096), 101),
            _FakeTensor((8192,), 102),
        ),
        (
            _FakeTensor((6144, 4096), 103),
            _FakeTensor((6144,), 104),
        ),
        _FakeTensor((2, 32, 7), 105),
        _FakeTensor((2, 4096, 64), 106),
    )

    assert bound(*operands, forward_workspace=workspace) is bundle
    assert calls == ["allocating", "checked"]
    inactive = (
        ["inactive FP8 V payload"]
        if publish_mxfp4_v
        else ["inactive MXFP4 V payload", "inactive MXFP4 V scale pages"]
    )
    assert authenticated[: len(inactive)] == inactive
    assert authenticated[-3:] == ["backward V", "backward Q", "backward K"]
    assert compact_calls[-1] == {
        "q_heads": 32,
        "kv_heads": 8,
        "publish_mxfp4_v": publish_mxfp4_v,
        "head_dim": 128,
    }
    assert bound.per_block_qk_scales is True
    assert bound.represented_backward is False
    assert bound.interleave_causal_kv is False
    assert bound.backward_publication_semantics == (
        "projection_accumulator_e4m3_qkv_shared_across_pv_routes"
    )

    assert bound(*operands, forward_workspace=workspace) is bundle
    assert calls == ["allocating", "checked", "unchecked"]
    assert bound.abi_validated is True
    assert bound.validated_forward_workspace_count == 1
    assert bound.successful_full_abi_validation_count == 1

    del workspace
    gc.collect()
    assert bound.validated_forward_workspace_count == 0
    assert bound.abi_validated is True
    assert bound.forward_workspace_abi_validated is True
    assert bound.successful_full_abi_validation_count == 1


def test_d128_e4m3_cuda_route_is_native_direct_and_route_selective() -> None:
    source = CUDA.read_text()
    base = "project_qkv_gqa_d128_unified_fp8_e4m3_rope_packed_clustered"
    for suffix in (
        "",
        "_fp8_forward_out",
        "_fp8_forward_out_unchecked",
        "_mx_forward_out",
        "_mx_forward_out_unchecked",
    ):
        symbol = base + suffix
        assert symbol in source
        assert f"&{symbol}," in source

    normalized = " ".join(source.split())
    assert "using C = tkfa4_projection::config<4, 4, 128, 128, true>;" in source
    assert ".paired_d64 = !NativeD128" in source
    assert (
        "!NativeD128 || (!PublishRepresentedBackwardFp8 && "
        "PerBlockQkScales"
    ) in normalized
    assert (
        "false, // never derive backward Q/K from represented NVFP4 codes "
        "true, // one dynamic scale per logical row x K16 block"
    ) in normalized
    assert (
        "false, // D128 forward and backward both use ordinary K/V order"
    ) in normalized
    assert "check_d128_e4m3_forward_outputs(" in source
    assert "q_heads * kQkChunks" in source
    assert "kv_heads * kQkChunks" in source


def test_d128_e4m3_python_exports_and_runtime_select_native_binder() -> None:
    runtime = RUNTIME.read_text()
    assert (
        tk_fa4.b300_bind_qkv_gqa_d128_unified_lowp_e4m3_projection
        is interface.b300_bind_qkv_gqa_d128_unified_lowp_e4m3_projection
    )
    assert "B300BoundD128E4M3QKVProjection" in tk_fa4.__all__
    assert "if qkv_projection_format == \"e4m3\":" in runtime
    assert "if is_d128:" in runtime
    assert "b300_bind_qkv_gqa_d128_unified_lowp_e4m3_projection(" in runtime
    assert "elif not is_d128 and not bool(" in runtime
    assert "dense E4M3 QKV projection without NVFP4 flag" in runtime


def _d128_e4m3_v501_kwargs() -> dict[str, object]:
    return {
        "projection_dgrad": "nvfp4",
        "qkv_projection_format": "e4m3",
        "backward_exp2_degree": 1,
        "backward_exp2_period": 0,
        "backward_fp8_ds_lift": 16,
        "backward_reuse_quantized_p": False,
        "backward_forward_mx_probability_replay": False,
        "backward_forward_mx_probability_scale_handoff": False,
        "backward_match_forward_operands": False,
        "per_block_qk_scales": True,
        "experimental_split_v_backward": False,
        "experimental_d128_mxfp4_v_backward": False,
        "backward_probability_correction": 1.0,
        "q_quant_scale": 2.25,
        "k_quant_scale": 2.0,
        "projection_weight_scale_2d": True,
        "v_mxfp4_scale_2d": False,
        "adaptive_qk_weight_scales": False,
        "shared_runtime": None,
    }


@pytest.mark.parametrize(
    "topology",
    (
        {
            "pv_format": "e4m3_fp8",
            "shiftless_fp8_mode": 0,
            "causal_interleaved_kv": False,
        },
        {
            "pv_format": "mxfp4_e8m0_block32",
            "causal_interleaved_kv": False,
        },
    ),
)
def test_d128_e4m3_v501_contract_accepts_fp8_and_ordinary_mx(
    topology: dict[str, object],
) -> None:
    require = _v501_contract_namespace()[
        "_require_d128_e4m3_v501_runtime_contract"
    ]
    require(
        SimpleNamespace(
            batch=2,
            sequence=4096,
            hidden=4096,
            q_heads=32,
            kv_heads=8,
            head_dim=128,
        ),
        topology,
        **_d128_e4m3_v501_kwargs(),
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("projection_dgrad", "bf16"),
        ("qkv_projection_format", "nvfp4"),
        ("backward_reuse_quantized_p", True),
        ("backward_match_forward_operands", True),
        ("per_block_qk_scales", False),
        ("experimental_split_v_backward", True),
        ("experimental_d128_mxfp4_v_backward", True),
        ("v_mxfp4_scale_2d", True),
    ),
)
def test_d128_e4m3_v501_contract_rejects_incompatible_publication(
    field: str,
    value: object,
) -> None:
    require = _v501_contract_namespace()[
        "_require_d128_e4m3_v501_runtime_contract"
    ]
    kwargs = _d128_e4m3_v501_kwargs()
    kwargs[field] = value
    with pytest.raises(ValueError):
        require(
            SimpleNamespace(
                batch=2,
                sequence=4096,
                hidden=4096,
                q_heads=32,
                kv_heads=8,
                head_dim=128,
            ),
            {
                "pv_format": "mxfp4_e8m0_block32",
                "causal_interleaved_kv": False,
            },
            **kwargs,
        )


def test_d128_e4m3_v501_contract_rejects_interleaved_mx_and_is_selected() -> None:
    require = _v501_contract_namespace()[
        "_require_d128_e4m3_v501_runtime_contract"
    ]
    with pytest.raises(ValueError, match="ordinary causal K/V order"):
        require(
            SimpleNamespace(
                batch=2,
                sequence=4096,
                hidden=4096,
                q_heads=32,
                kv_heads=8,
                head_dim=128,
            ),
            {
                "pv_format": "mxfp4_e8m0_block32",
                "causal_interleaved_kv": True,
            },
            **_d128_e4m3_v501_kwargs(),
        )
    source = RUNTIME.read_text()
    selector = source.split("if self.native_tk_d64_backward:", 1)[1].split(
        "if is_d128:",
        1,
    )[0]
    assert 'elif is_d128 and qkv_projection_format == "e4m3":' in selector
    assert "_require_d128_e4m3_v501_runtime_contract(" in selector


def test_d128_e4m3_binder_fails_closed_on_represented_shape_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = "project_qkv_gqa_d128_unified_fp8_e4m3_rope_packed_clustered"
    extension = SimpleNamespace(__file__="/tmp/d128-e4m3-qkv.so")
    for suffix in (
        "",
        "_fp8_forward_out",
        "_fp8_forward_out_unchecked",
        "_mx_forward_out",
        "_mx_forward_out_unchecked",
    ):
        setattr(extension, base + suffix, object())
    monkeypatch.setattr(interface, "_C_b300_lowp_bwd", extension)
    monkeypatch.setattr(interface, "_ensure_lowp_bwd_extension", lambda: None)

    with pytest.raises(ValueError, match="seqlen divisible by 256"):
        interface.B300BoundD128E4M3QKVProjection(
            batch=1,
            seqlen=255,
            hidden=4096,
            q_heads=32,
            kv_heads=8,
            publish_mxfp4_v=False,
        )
    with pytest.raises(ValueError, match="cluster_cap"):
        interface.B300BoundD128E4M3QKVProjection(
            batch=1,
            seqlen=4096,
            hidden=4096,
            q_heads=32,
            kv_heads=8,
            publish_mxfp4_v=True,
            cluster_cap=-1,
        )
