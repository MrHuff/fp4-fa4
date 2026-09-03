from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tk_fa4 import interface


ROOT = Path(__file__).resolve().parents[1]
CUDA = ROOT / "tk_fa4" / "lowp_fa4_bwd" / "lowp_fa4_bwd.cu"
EPILOGUE = (
    ROOT / "tk_fa4" / "lowp_fa4_bwd" / "projection_fp4_epilogue.cuh"
)
INTERFACE = ROOT / "tk_fa4" / "interface.py"
E2E = ROOT / "tk_fa4" / "lowp_fa4_bwd" / "benchmark_llama12b_e2e.py"

SYMBOL = (
    "project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_"
    "interleaved_causal_represented_backward_perblock_qk_"
    "output_shared_split_v_mx_forward_out"
)


class _RuntimeSelectionCaptured(RuntimeError):
    pass


class _PackedRank2:
    def __init__(self, width: int) -> None:
        self.width = width

    def dim(self) -> int:
        return 2

    def size(self, dimension: int) -> int:
        assert dimension == 1
        return self.width


class _RuntimeProjectionCapture:
    def __init__(
        self,
        *,
        requested: bool | None,
        resolved: bool,
        path: str,
        checked_symbol: str,
        publish_selection: bool = True,
    ) -> None:
        if publish_selection:
            self.experimental_output_shared_split_v_requested = requested
            self.experimental_output_shared_split_v_resolved = resolved
        self.output_shared_split_v_path = path
        self.checked_symbol = checked_symbol
        self.unchecked_symbol = checked_symbol + "_unchecked"
        self.symbol = self.unchecked_symbol
        self.abi_validation_symbol = "allocating_abi"
        self.requires_forward_workspace = True

    @property
    def requires_v_mxfp4_scales_out(self) -> bool:
        return False


def _runtime_class(**overrides: object) -> type[Any]:
    source = E2E.read_text()
    tree = ast.parse(source)
    eligibility = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_native_output_shared_v_eligible"
    )
    d128_mx_backward_v_eligibility = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_native_d128_mxfp4_v_backward_eligible"
    )
    runtime = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "LowpAttentionRuntime"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            eligibility,
            d128_mx_backward_v_eligibility,
            runtime,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)

    class FakeTorch:
        float32 = object()

        @staticmethod
        def tensor(*_args: object, **_kwargs: object) -> object:
            return object()

    noop = lambda *_args, **_kwargs: None

    def capture_runtime_selection(*_args: object, **_kwargs: object) -> bool:
        raise _RuntimeSelectionCaptured

    namespace: dict[str, object] = {
        "Any": Any,
        "Path": Path,
        "torch": FakeTorch,
        "math": SimpleNamespace(isfinite=capture_runtime_selection),
        "AUTHENTICATED_D64_EXACT_BATCHES": (),
        "AUTHENTICATED_D128_EXACT_BATCHES": (),
        "_require_forward_topology": noop,
        "_require_output_projection_contract": noop,
        "_require_experimental_native_batched_runtime_contract": noop,
        "_require_batched_exact_runtime_contract": noop,
        "_require_fused_attention_rmsnorm_nvfp4": noop,
        "b300_pack_gqa_d64_paired_rope": lambda *_args: object(),
        "b300_pack_gqa_d128_rope": lambda *_args: object(),
    }
    namespace.update(overrides)
    exec(compile(module, str(E2E), "exec"), namespace)
    return namespace["LowpAttentionRuntime"]  # type: ignore[return-value]


_OMITTED = object()


def _exercise_runtime_selector(
    selection: bool | None | object = _OMITTED,
    *,
    route: str = "mx",
    native: bool = True,
    qkv_format: str = "nvfp4",
    v_scale_2d: bool = False,
    shape: dict[str, int] | None = None,
    binder_requested: bool | None | object = _OMITTED,
    binder_resolved: bool | object = _OMITTED,
    expected_exception: type[BaseException] = _RuntimeSelectionCaptured,
    expected_match: str | None = None,
) -> tuple[object, list[bool | None]]:
    calls: list[bool | None] = []
    config_values = {
        "batch": 16,
        "sequence": 4096,
        "hidden": 2048,
        "q_heads": 32,
        "kv_heads": 8,
        "head_dim": 64,
    }
    if shape is not None:
        config_values.update(shape)
    config = SimpleNamespace(**config_values)
    is_mx = route == "mx"

    def bind_nvfp4(**kwargs: object) -> _RuntimeProjectionCapture:
        requested = kwargs["experimental_output_shared_split_v"]
        assert requested is None or type(requested) is bool
        calls.append(requested)
        eligible = bool(
            kwargs["batch"] == 16
            and kwargs["seqlen"] == 4096
            and kwargs["hidden"] == 2048
            and kwargs["q_heads"] == 32
            and kwargs["kv_heads"] == 8
            and kwargs["publish_mxfp4_v"]
            and not kwargs["v_mxfp4_scale_2d"]
        )
        requested_out = (
            requested if binder_requested is _OMITTED else binder_requested
        )
        resolved: object = bool(
            eligible if requested is None else requested
        )
        if binder_resolved is not _OMITTED:
            resolved = binder_resolved
        path = (
            "output_shared_split_v"
            if resolved
            else "retained_split_v"
            if kwargs["publish_mxfp4_v"]
            else "fp8"
        )
        return _RuntimeProjectionCapture(
            requested=requested_out,  # type: ignore[arg-type]
            resolved=resolved,  # type: ignore[arg-type]
            path=path,
            checked_symbol="candidate" if resolved else "retained",
        )

    def bind_e4m3(**_kwargs: object) -> _RuntimeProjectionCapture:
        return _RuntimeProjectionCapture(
            requested=None,
            resolved=False,
            path="not_applicable",
            checked_symbol="e4m3",
            publish_selection=False,
        )

    runtime_class = _runtime_class(
        b300_bind_qkv_gqa_d64_paired_unified_lowp_nvfp4_projection=(
            bind_nvfp4
        ),
        b300_bind_qkv_gqa_d64_paired_unified_lowp_e4m3_projection=(
            bind_e4m3
        ),
    )
    runtime = runtime_class.__new__(runtime_class)
    topology = {
        "pv_format": "mxfp4_e8m0_block32" if is_mx else "e4m3_fp8",
        "causal_interleaved_kv": is_mx,
        "shiftless_fp8_mode": 0,
        "valid": 0,
    }
    kwargs: dict[str, object] = {
        "forward_extension": object(),
        "forward_topology": topology,
        "loss_scale": 1.0,
        "gradient_global_scale": 1.0,
        "projection_dgrad": "bf16",
        "qkv_projection_format": qkv_format,
        "experimental_native_nvfp4_projection_out": native,
        "backward_match_forward_operands": True,
        "per_block_qk_scales": True,
        "experimental_split_v_backward": is_mx,
        "v_mxfp4_scale_2d": v_scale_2d,
    }
    if selection is not _OMITTED:
        kwargs["experimental_output_shared_split_v"] = selection
    with pytest.raises(expected_exception, match=expected_match):
        runtime_class.__init__(
            runtime,
            config,
            (object(), object()),
            **kwargs,
        )
    return runtime, calls


def test_runtime_selector_preserves_omitted_none_and_false_provenance() -> None:
    omitted, omitted_calls = _exercise_runtime_selector()
    assert omitted_calls == [False]
    assert omitted.experimental_output_shared_split_v_requested is False
    assert omitted.experimental_output_shared_split_v_resolved is False
    assert omitted.output_shared_split_v_path == "retained_split_v"

    automatic, automatic_calls = _exercise_runtime_selector(None)
    assert automatic_calls == [None]
    assert automatic.experimental_output_shared_split_v_requested is None
    assert automatic.experimental_output_shared_split_v_resolved is True
    assert automatic.experimental_output_shared_split_v is True
    assert automatic.output_shared_split_v_path == "output_shared_split_v"

    retained, retained_calls = _exercise_runtime_selector(False)
    assert retained_calls == [False]
    assert retained.experimental_output_shared_split_v_requested is False
    assert retained.experimental_output_shared_split_v_resolved is False
    assert retained.output_shared_split_v_path == "retained_split_v"


@pytest.mark.parametrize(
    "case",
    (
        {"route": "fp8"},
        {"v_scale_2d": True},
        {"native": False, "qkv_format": "e4m3"},
        {"shape": {"batch": 1}},
        {"shape": {"hidden": 4096}},
    ),
)
def test_runtime_none_falls_back_and_true_fails_for_ineligible_recipe(
    case: dict[str, object],
) -> None:
    fallback, calls = _exercise_runtime_selector(None, **case)
    assert fallback.experimental_output_shared_split_v_requested is None
    assert fallback.experimental_output_shared_split_v_resolved is False
    assert fallback.experimental_output_shared_split_v is False
    expected_path = (
        "fp8"
        if case.get("route") == "fp8"
        else "not_applicable"
        if case.get("qkv_format") == "e4m3"
        else "retained_split_v"
    )
    assert fallback.output_shared_split_v_path == expected_path
    if case.get("qkv_format") == "e4m3":
        assert calls == []
    else:
        assert calls == [None]

    _exercise_runtime_selector(
        True,
        **case,
        expected_exception=ValueError,
    )


def test_runtime_and_binder_reject_truthy_non_bool_selector() -> None:
    runtime, calls = _exercise_runtime_selector(
        1,
        expected_exception=TypeError,
    )
    assert calls == []
    assert not hasattr(
        runtime,
        "experimental_output_shared_split_v_requested",
    )


@pytest.mark.parametrize(
    ("binder_requested", "binder_resolved", "match"),
    (
        (False, _OMITTED, "requested provenance"),
        (_OMITTED, False, "disagrees with the runtime"),
        (_OMITTED, 1, "non-bool"),
    ),
)
def test_runtime_rejects_binder_selector_provenance_disagreement(
    binder_requested: bool | None | object,
    binder_resolved: bool | object,
    match: str,
) -> None:
    _exercise_runtime_selector(
        None,
        binder_requested=binder_requested,
        binder_resolved=binder_resolved,
        expected_exception=RuntimeError,
        expected_match=match,
    )


def _function_body(source: str, name: str, next_marker: str) -> str:
    return source.split(name, 1)[1].split(next_marker, 1)[0]


def _fake_forward_workspace() -> interface.B300E4M3QKVForwardWorkspace:
    workspace = object.__new__(interface.B300E4M3QKVForwardWorkspace)
    for field in interface.B300E4M3QKVForwardWorkspace.__dataclass_fields__:
        object.__setattr__(workspace, field, object())
    return workspace


def test_output_shared_route_is_compile_time_opt_in_and_constrained() -> None:
    cuda = CUDA.read_text()
    epilogue = EPILOGUE.read_text()
    assert "bool kExperimentalOutputSharedSplitV = false" in cuda
    assert "bool ExperimentalOutputSharedSplitV = false" in cuda
    assert "bool EXPERIMENTAL_OUTPUT_SHARED_SPLIT_V = false" in epilogue
    for contract in (
        "kCompactForwardOut && kQkDepth == 128",
        "kPairedD64 && kPublishRepresentedBackwardFp8",
        "kPerBlockQkScales && kInterleaveCausalKv",
        "kExperimentalSplitVBackward",
        "!kExperimentalE4m3DerivedMxfp4V",
    ):
        assert contract in cuda
    for contract in (
        "G::D_tile::rows == 128 && G::D_tile::cols == 32",
        "!C::DENSE_FP8 && C::QK_DEPTH == 128 && !SINGLE_OUTPUT",
        "PUBLISH_V_MXFP4 &&",
        "PUBLISH_REPRESENTED_BACKWARD_FP8 &&",
        "PER_BLOCK_QK_SCALES && INTERLEAVE_CAUSAL_KV",
        "EXPERIMENTAL_SPLIT_V_BACKWARD",
        "PUBLISH_V_FP8 != PUBLISH_V_BACKWARD_MXFP4",
        "!OUTPUT_IS_DOUT",
    ):
        assert contract in epilogue


def test_output_shared_mx_publisher_reads_resident_bf16_exactly() -> None:
    source = EPILOGUE.read_text()
    assert (
        "__device__ __noinline__ void "
        "publish_v_mxfp4_from_output_shared(" in source
    )
    publisher = _function_body(
        source,
        "publish_v_mxfp4_from_output_shared(",
        "template <typename C>\n"
        "__device__ __noinline__ void\n"
        "publish_v_common_rowscale_mxfp4_from_output_ring(",
    )
    assert "&tile[source_coord0]" in publisher
    assert "&tile[source_coord1]" in publisher
    assert "(row0 >> 3) * 32 + warp + (row0 & 7) * 4" in publisher
    assert "(row1 >> 3) * 32 + warp + (row1 & 7) * 4" in publisher
    assert "bf16_amax_to_e8m0_1d_mse(row_amax_bits)" in publisher
    assert "quantize_four_bf16_pairs(" in publisher
    assert "g.v_mxfp4 + payload_base" in publisher
    assert "g.v_mxfp4_scales[" in publisher
    for forbidden in (
        "stage_bf16_pairs",
        "convert_scaled_bf16_pair_to_fp8",
        "g.v_backward_fp8",
        "kittens::warpgroup::sync",
    ):
        assert forbidden not in publisher
    assert "if constexpr (SHARE_MXFP4_TILE_WITH_BACKWARD)" in publisher
    assert (
        "PUBLISH_BACKWARD_MXFP4 && !SHARE_MXFP4_TILE_WITH_BACKWARD"
        in publisher
    )
    assert "2 * depth_pair_index" in publisher
    assert "g.v_backward_mxfp4 + backward_payload_base" in publisher
    assert "g.v_backward_mxfp4_scales[" in publisher
    assert "bf16_amax_to_e8m0_1d_mse(row_amax_bits)" in publisher


def test_output_shared_branch_bypasses_reload_and_restaging() -> None:
    source = EPILOGUE.read_text()
    branch = source.split(
        "if constexpr (EXPERIMENTAL_OUTPUT_SHARED_SPLIT_V) {",
        1,
    )[1].split("output_rt registers;", 1)[0]
    backward = branch.index("if constexpr (PUBLISH_V_FP8)")
    forward = branch.index("publish_v_mxfp4_from_output_shared<", backward)
    skip = branch.index("continue;", forward)
    assert backward < forward < skip
    assert "stage_bf16_pairs" not in branch
    assert "warpgroup::load" not in branch
    assert "warpgroup::sync" not in branch


def test_checked_and_unchecked_symbols_enable_only_candidate_flag() -> None:
    source = CUDA.read_text()
    checked = source.split(
        f"TKFA4_DEFINE_D64_NVFP4_FORWARD_OUT(\n    {SYMBOL},",
        1,
    )[1].split(")", 1)[0]
    unchecked = source.split(
        "TKFA4_DEFINE_D64_NVFP4_FORWARD_OUT(\n"
        f"    {SYMBOL}_unchecked,",
        1,
    )[1].split(")", 1)[0]
    assert checked.replace(" ", "").replace("\n", "") == (
        "true,true,true,true,true,false,true"
    )
    assert unchecked.replace(" ", "").replace("\n", "") == (
        "true,false,true,true,true,false,true"
    )
    for exported in (SYMBOL, f"{SYMBOL}_unchecked"):
        assert f"&{exported}," in source
    assert '"output_shared_split_v_mx_forward_out",' in source
    assert '"output_shared_split_v_mx_forward_out_unchecked",' in source


def test_checked_candidate_symbol_enforces_authenticated_shape_and_scale() -> None:
    source = CUDA.read_text()
    implementation = _function_body(
        source,
        "project_qkv_gqa_d64_paired_unified_fp4_nvfp4_forward_out_impl(",
        "#define TKFA4_DEFINE_D64_NVFP4_FORWARD_OUT",
    )
    assert "if constexpr (ValidateContracts)" in implementation
    checked = implementation.split(
        "if constexpr (ValidateContracts) {", 1
    )[1]
    assert "if constexpr (ExperimentalOutputSharedSplitV)" in checked
    assert (
        "batch == 16 && seq_len == 4096 && q_heads == 32 &&\n"
        "                    kv_heads == 8 && input_fp4.dim() == 2 &&\n"
        "                    input_fp4.size(1) == 1024 && "
        "!v_mxfp4_scale_2d"
    ) in checked
    assert "B16/S4096/H2048/Hq32/Hkv8/D64" in checked


def test_python_and_e2e_dispatch_are_explicit_and_report_topology() -> None:
    binder = INTERFACE.read_text()
    e2e = E2E.read_text()
    assert "experimental_output_shared_split_v: bool | None = False" in binder
    assert "experimental_output_shared_split_v: bool | None = False" in e2e
    assert "output_shared_split_v_mx_forward_out" in binder
    assert "are mutually exclusive" in binder
    assert "--experimental-output-shared-split-v" in e2e
    cli = e2e.split(
        '"--experimental-output-shared-split-v"', 1
    )[1].split("parser.add_argument", 1)[0]
    assert "action=argparse.BooleanOptionalAction" in cli
    assert "default=False" in cli
    assert "experimental_output_shared_split_v_requested" in e2e
    assert "experimental_output_shared_split_v_resolved" in e2e
    assert "output_shared_split_v_path" in e2e
    assert "output_shared_split_v_checked_symbol" in e2e
    assert (
        '"experimental_output_shared_split_v": (\n'
        "                self.experimental_output_shared_split_v"
    ) in e2e
    native_bind = e2e.split(
        "elif self.experimental_native_nvfp4_projection_out:", 1
    )[1].split("if self.qkv_projection is not None:", 1)[0]
    assert "hidden=config.hidden" in native_bind
    assert "experimental_output_shared_split_v=(" in native_bind
    d64_native_bind = native_bind.split("else:", 1)[1]
    assert "experimental_output_shared_split_v" in d64_native_bind


def test_candidate_binder_authenticates_all_forward_and_backward_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    authenticated: list[str] = []
    workspace = _fake_forward_workspace()
    backward = tuple(
        SimpleNamespace(data_ptr=lambda pointer=pointer: pointer)
        for pointer in (101, 102, 103)
    )
    for field, tensor in zip(
        ("v_backward_fp8", "q_backward_fp8", "k_backward_fp8"),
        backward,
        strict=True,
    ):
        object.__setattr__(workspace, field, tensor)

    extension = SimpleNamespace(__file__="/tmp/output-shared-candidate.so")

    def checked(*_args: object) -> tuple[object, object, object]:
        calls.append("checked")
        return backward

    def unchecked(*_args: object) -> tuple[object, object, object]:
        calls.append("unchecked")
        return backward

    setattr(extension, SYMBOL, checked)
    setattr(extension, f"{SYMBOL}_unchecked", unchecked)
    monkeypatch.setattr(interface, "_C_b300_lowp_bwd", extension)
    monkeypatch.setattr(
        interface,
        "b300_require_qkv_gqa_d64_paired_unified_lowp_nvfp4_projection",
        lambda **_kwargs: "legacy_split_symbol",
    )
    legacy_bundle = SimpleNamespace(
        backward=SimpleNamespace(score_q_fp4=object(), score_k_fp4=object()),
        q_forward_scales=object(),
        q_forward_global_scale=object(),
        k_forward_scales=object(),
        k_forward_global_scale=object(),
        v_forward_fp4=object(),
        v_forward_scales=object(),
        v_forward_fp8=None,
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
        "b300_project_qkv_gqa_d64_paired_unified_lowp_nvfp4",
        legacy,
    )
    monkeypatch.setattr(
        interface,
        "_b300_require_bitwise_equal",
        lambda name, *_args, **_kwargs: authenticated.append(name),
    )
    returned = object()
    monkeypatch.setattr(
        interface,
        "_b300_compact_e4m3_qkv_bundle",
        lambda *_args, **_kwargs: returned,
    )

    bound = interface.B300BoundNVFP4QKVProjection(
        batch=16,
        seqlen=4096,
        hidden=2048,
        q_heads=32,
        kv_heads=8,
        publish_mxfp4_v=True,
        v_mxfp4_scale_2d=False,
        experimental_output_shared_split_v=True,
    )
    operands = (
        (_PackedRank2(1024), object(), object()),
        (_PackedRank2(1024), object(), object()),
        object(),
        object(),
    )
    assert bound(*operands, forward_workspace=workspace) is returned
    assert calls == ["legacy", "checked"]
    assert bound.checked_symbol == SYMBOL
    assert bound.experimental_output_shared_split_v is True
    assert legacy_kwargs[0]["experimental_split_v_backward"] is True
    assert authenticated == [
        "Q payload",
        "K payload",
        "Q scale pages",
        "Q global scale",
        "K scale pages",
        "K global scale",
        "MXFP4 V payload",
        "MXFP4 V scale pages",
        "backward V",
        "backward Q",
        "backward K",
    ]
    assert bound(*operands, forward_workspace=workspace) is returned
    assert calls == ["legacy", "checked", "unchecked"]


@pytest.mark.parametrize(
    ("input_width", "weight_width", "name"),
    (
        (512, 1024, "input"),
        (1024, 512, "QKV weight"),
    ),
)
def test_candidate_binder_rechecks_packed_k_width_before_every_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    input_width: int,
    weight_width: int,
    name: str,
) -> None:
    calls: list[str] = []
    extension = SimpleNamespace(__file__="/tmp/output-shared-shape.so")
    setattr(extension, SYMBOL, lambda *_args: calls.append("checked"))
    setattr(
        extension,
        SYMBOL + "_unchecked",
        lambda *_args: calls.append("unchecked"),
    )
    monkeypatch.setattr(interface, "_C_b300_lowp_bwd", extension)
    monkeypatch.setattr(
        interface,
        "b300_require_qkv_gqa_d64_paired_unified_lowp_nvfp4_projection",
        lambda **_kwargs: "allocating_abi",
    )
    bound = interface.B300BoundNVFP4QKVProjection(
        batch=16,
        seqlen=4096,
        hidden=2048,
        q_heads=32,
        kv_heads=8,
        publish_mxfp4_v=True,
        v_mxfp4_scale_2d=False,
        experimental_output_shared_split_v=True,
    )
    workspace = _fake_forward_workspace()
    bound._validated_forward_workspaces[id(workspace)] = workspace
    with pytest.raises(ValueError, match=f"packed {name} with K width 1024"):
        bound(
            (_PackedRank2(input_width), object(), object()),
            (_PackedRank2(weight_width), object(), object()),
            object(),
            object(),
            forward_workspace=workspace,
        )
    assert calls == []


def test_candidate_binder_rejects_invalid_combinations_before_lookup() -> None:
    with pytest.raises(TypeError, match="exactly bool or None"):
        interface.B300BoundNVFP4QKVProjection(
            batch=16,
            seqlen=4096,
            hidden=2048,
            q_heads=32,
            kv_heads=8,
            publish_mxfp4_v=True,
            v_mxfp4_scale_2d=False,
            experimental_output_shared_split_v=1,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="publish_mxfp4_v=True"):
        interface.B300BoundNVFP4QKVProjection(
            batch=1,
            seqlen=128,
            q_heads=2,
            kv_heads=2,
            publish_mxfp4_v=False,
            v_mxfp4_scale_2d=False,
            experimental_output_shared_split_v=True,
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        interface.B300BoundNVFP4QKVProjection(
            batch=16,
            seqlen=4096,
            hidden=2048,
            q_heads=32,
            kv_heads=8,
            publish_mxfp4_v=True,
            v_mxfp4_scale_2d=False,
            experimental_e4m3_derived_mxfp4_v=True,
            experimental_output_shared_split_v=True,
        )


@pytest.mark.parametrize(
    "shape_override",
    (
        {"batch": 1},
        {"batch": 8},
        {"seqlen": 2048},
        {"hidden": 4096},
        {"hidden": None},
        {"q_heads": 16},
        {"kv_heads": 4},
    ),
)
def test_candidate_binder_auto_falls_back_and_true_rejects_other_shapes(
    monkeypatch: pytest.MonkeyPatch,
    shape_override: dict[str, int | None],
) -> None:
    retained_base = (
        "project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_"
        "interleaved_causal_represented_backward_perblock_qk_"
        "split_v_backward"
    )
    retained = retained_base + "_mx_forward_out"
    extension = SimpleNamespace(__file__="/tmp/output-shared-shape.so")
    setattr(extension, retained, object())
    setattr(extension, retained + "_unchecked", object())
    monkeypatch.setattr(interface, "_C_b300_lowp_bwd", extension)
    monkeypatch.setattr(
        interface,
        "b300_require_qkv_gqa_d64_paired_unified_lowp_nvfp4_projection",
        lambda **_kwargs: retained_base,
    )
    kwargs = {
        "batch": 16,
        "seqlen": 4096,
        "hidden": 2048,
        "q_heads": 32,
        "kv_heads": 8,
        "publish_mxfp4_v": True,
        "v_mxfp4_scale_2d": False,
    }
    kwargs.update(shape_override)
    automatic = interface.B300BoundNVFP4QKVProjection(
        **kwargs,
        experimental_output_shared_split_v=None,
    )
    assert automatic.experimental_output_shared_split_v_requested is None
    assert automatic.experimental_output_shared_split_v_resolved is False
    assert automatic.output_shared_split_v_path == "retained_split_v"
    assert automatic.checked_symbol == retained
    with pytest.raises(ValueError, match="authenticated B16/S4096"):
        interface.B300BoundNVFP4QKVProjection(
            **kwargs,
            experimental_output_shared_split_v=True,
        )


def test_tri_state_auto_fallback_and_fail_closed_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained_base = (
        "project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_"
        "interleaved_causal_represented_backward_perblock_qk_"
        "split_v_backward"
    )
    fp8_base = (
        "project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_"
        "represented_backward_perblock_qk"
    )
    retained = retained_base + "_mx_forward_out"
    fp8 = fp8_base + "_fp8_forward_out"
    derived = (
        "project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_"
        "interleaved_causal_represented_backward_perblock_qk_"
        "e4m3_derived_mx_forward_out"
    )
    symbols = (SYMBOL, retained, fp8, derived)
    extension = SimpleNamespace(
        __file__="/tmp/output-shared-selection.so",
        convert_e4m3_x4_v_bhds_to_causal_mxfp4=object(),
    )
    for symbol in symbols:
        setattr(extension, symbol, object())
        setattr(extension, symbol + "_unchecked", object())
    monkeypatch.setattr(interface, "_C_b300_lowp_bwd", extension)
    monkeypatch.setattr(
        interface,
        "b300_require_qkv_gqa_d64_paired_unified_lowp_nvfp4_projection",
        lambda *, publish_mxfp4_v: (
            retained_base if publish_mxfp4_v else fp8_base
        ),
    )

    def bind(
        *,
        publish_mxfp4_v: bool = True,
        derived_mx: bool = False,
        scale_2d: bool = False,
        selection: bool | None = None,
    ) -> interface.B300BoundNVFP4QKVProjection:
        return interface.B300BoundNVFP4QKVProjection(
            batch=16,
            seqlen=4096,
            hidden=2048,
            q_heads=32,
            kv_heads=8,
            publish_mxfp4_v=publish_mxfp4_v,
            v_mxfp4_scale_2d=scale_2d,
            experimental_e4m3_derived_mxfp4_v=derived_mx,
            experimental_output_shared_split_v=selection,
        )

    historical_omitted = interface.B300BoundNVFP4QKVProjection(
        batch=16,
        seqlen=4096,
        q_heads=32,
        kv_heads=8,
        publish_mxfp4_v=True,
        v_mxfp4_scale_2d=False,
    )
    assert historical_omitted.experimental_output_shared_split_v_requested is False
    assert historical_omitted.experimental_output_shared_split_v_resolved is False
    assert historical_omitted.output_shared_split_v_path == "retained_split_v"
    assert historical_omitted.checked_symbol == retained

    automatic = bind()
    assert automatic.experimental_output_shared_split_v_requested is None
    assert automatic.experimental_output_shared_split_v_resolved is True
    assert automatic.output_shared_split_v_path == "output_shared_split_v"
    assert automatic.checked_symbol == SYMBOL

    retained_fallback = bind(selection=False)
    assert retained_fallback.experimental_output_shared_split_v_resolved is False
    assert retained_fallback.output_shared_split_v_path == "retained_split_v"
    assert retained_fallback.checked_symbol == retained

    fp8_automatic = bind(publish_mxfp4_v=False)
    assert fp8_automatic.experimental_output_shared_split_v_resolved is False
    assert fp8_automatic.output_shared_split_v_path == "fp8"
    assert fp8_automatic.checked_symbol == fp8
    with pytest.raises(ValueError, match="publish_mxfp4_v=True"):
        bind(publish_mxfp4_v=False, selection=True)

    derived_automatic = bind(derived_mx=True)
    assert derived_automatic.experimental_output_shared_split_v_resolved is False
    assert derived_automatic.output_shared_split_v_path == "e4m3_derived_mx"
    assert derived_automatic.checked_symbol == derived
    with pytest.raises(ValueError, match="mutually exclusive"):
        bind(derived_mx=True, selection=True)

    scale_fallback = bind(scale_2d=True)
    assert scale_fallback.experimental_output_shared_split_v_resolved is False
    assert scale_fallback.checked_symbol == retained
    with pytest.raises(ValueError, match="rowwise 1x32"):
        bind(scale_2d=True, selection=True)
