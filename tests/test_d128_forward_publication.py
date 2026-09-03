from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tk_fa4.lowp_fa4_bwd.benchmark_causal_forward_matrix import (
    _apply_pair_rope,
    _split_half_to_adjacent_pairs,
)


ROOT = Path(__file__).resolve().parents[1]
INTERFACE = ROOT / "tk_fa4" / "interface.py"
CUDA = ROOT / "tk_fa4" / "lowp_fa4_bwd" / "lowp_fa4_bwd.cu"
FORWARD_MATRIX = (
    ROOT / "tk_fa4" / "lowp_fa4_bwd" / "benchmark_causal_forward_matrix.py"
)
D128_CHAIN = (
    ROOT / "tk_fa4" / "lowp_fa4_bwd" / "profile_gqa_d128_chain.py"
)
E2E = ROOT / "tk_fa4" / "lowp_fa4_bwd" / "benchmark_llama12b_e2e.py"
D128_PROJECTION_PROFILE = (
    ROOT / "tk_fa4" / "lowp_fa4_bwd" / "profile_gqa_d128_projection.py"
)
CAUSAL_PAIR_PROFILE = (
    ROOT / "tk_fa4" / "lowp_fa4_bwd" / "profile_causal_forward_pair.py"
)
MATCHED_PROFILE = (
    ROOT / "tk_fa4" / "lowp_fa4_bwd" / "profile_llama12b_matched_routes.py"
)
HAO_DIRECT_MAKEFILE = (
    ROOT / "tk_fa4" / "fp4_fa4_fwd" / "Makefile.hao_direct_fp4pv"
)
PROJECTION_EPILOGUE = (
    ROOT / "tk_fa4" / "lowp_fa4_bwd" / "projection_fp4_epilogue.cuh"
)


def _function_source(path: Path, name: str, next_name: str) -> str:
    source = path.read_text()
    return source.split(f"def {name}(", 1)[1].split(f"def {next_name}(", 1)[0]


def _runtime_ast() -> ast.Module:
    return ast.parse(E2E.read_text())


def _runtime_init_ast() -> ast.FunctionDef:
    runtime = next(
        node
        for node in _runtime_ast().body
        if isinstance(node, ast.ClassDef)
        and node.name == "LowpAttentionRuntime"
    )
    return next(
        node
        for node in runtime.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )


def _compile_runtime_function(name: str):
    function = next(
        node
        for node in _runtime_ast().body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    executable = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            function,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(executable)
    namespace: dict[str, object] = {
        "AUTHENTICATED_D128_EXACT_BATCHES": (2,),
    }
    exec(compile(executable, str(E2E), "exec"), namespace)
    return namespace[name]


def _d128_runtime_contract_kwargs() -> dict[str, object]:
    return {
        "config": SimpleNamespace(
            batch=2,
            sequence=4096,
            hidden=4096,
            q_heads=32,
            kv_heads=8,
            head_dim=128,
        ),
        "forward_topology": {
            "pv_format": "e4m3_fp8",
            "shiftless_fp8_mode": 0,
            "causal_interleaved_kv": False,
        },
        "projection_dgrad": "nvfp4",
        "qkv_projection_format": "nvfp4",
        "experimental_native_nvfp4_projection_out": True,
        "backward_reuse_quantized_p": False,
        "backward_forward_mx_probability_replay": False,
        "backward_forward_mx_probability_scale_handoff": False,
        "backward_match_forward_operands": True,
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


def _d128_nvfp4_binder_invoker():
    binder_name = "b300_bind_qkv_gqa_d128_unified_lowp_nvfp4_projection"
    call = next(
        node
        for node in ast.walk(_runtime_init_ast())
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == binder_name
    )
    invoke = ast.FunctionDef(
        name="_invoke",
        args=ast.arguments(
            posonlyargs=[],
            args=[
                ast.arg(arg="self"),
                ast.arg(arg="config"),
                ast.arg(arg="v_mxfp4_scale_2d"),
                ast.arg(arg="experimental_output_shared_split_v"),
            ],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[ast.Return(value=call)],
        decorator_list=[],
    )
    executable = ast.Module(body=[invoke], type_ignores=[])
    ast.fix_missing_locations(executable)
    namespace: dict[str, object] = {
        binder_name: lambda **kwargs: kwargs,
    }
    exec(compile(executable, str(E2E), "exec"), namespace)
    return namespace["_invoke"]


def _d128_projection_provenance_validator():
    validation = next(
        node
        for node in ast.walk(_runtime_init_ast())
        if isinstance(node, ast.If)
        and ast.unparse(node.test)
        == "is_d128 and self.experimental_native_nvfp4_projection_out"
        and any(
            isinstance(child, ast.Constant)
            and isinstance(child.value, str)
            and "omitted its exact-bool" in child.value
            for child in ast.walk(node)
        )
    )
    validate = ast.FunctionDef(
        name="_validate",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="self"), ast.arg(arg="is_d128")],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[validation],
        decorator_list=[],
    )
    executable = ast.Module(body=[validate], type_ignores=[])
    ast.fix_missing_locations(executable)
    namespace: dict[str, object] = {}
    exec(compile(executable, str(E2E), "exec"), namespace)
    return namespace["_validate"]


def _d128_represented_native_backend_validator():
    validation = next(
        node
        for node in ast.walk(_runtime_init_ast())
        if isinstance(node, ast.If)
        and any(
            isinstance(child, ast.Constant)
            and child.value
            == "represented D128 Q/K backward is authenticated only with "
            "an authenticated native TK D128 backend"
            for child in ast.walk(node)
        )
    )
    validate = ast.FunctionDef(
        name="_validate",
        args=ast.arguments(
            posonlyargs=[],
            args=[
                ast.arg(arg="self"),
                ast.arg(arg="is_d128"),
                ast.arg(arg="backward_match_forward_operands"),
            ],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[validation],
        decorator_list=[],
    )
    executable = ast.Module(body=[validate], type_ignores=[])
    ast.fix_missing_locations(executable)
    namespace: dict[str, object] = {}
    exec(compile(executable, str(E2E), "exec"), namespace)
    return namespace["_validate"]


def test_d128_runtime_passes_represented_only_to_accepted_nvfp4_fp8_route(
) -> None:
    require_contract = _compile_runtime_function(
        "_require_native_tk_d128_runtime_contract"
    )
    contract = _d128_runtime_contract_kwargs()
    require_contract(**contract)

    invoke = _d128_nvfp4_binder_invoker()
    runtime = SimpleNamespace(
        publish_mxfp4_v=False,
        per_block_qk_scales=contract["per_block_qk_scales"],
        backward_match_forward_operands=(
            contract["backward_match_forward_operands"]
        ),
        experimental_d128_mxfp4_v_backward=False,
    )
    bound = invoke(runtime, contract["config"], False, False)

    assert bound["publish_mxfp4_v"] is False
    assert bound["per_block_qk_scales"] is True
    assert bound["represented_backward"] is True
    assert bound["experimental_output_shared_dual_v"] is False
    assert bound["experimental_mx_backward_v"] is False
    assert bound["experimental_shared_tile_mx_backward_v"] is False

    mx_contract = dict(contract)
    mx_contract["forward_topology"] = {
        **contract["forward_topology"],
        "pv_format": "mxfp4_e8m0_block32",
    }
    with pytest.raises(
        ValueError,
        match="represented NVFP4 Q/K backward only with FP8-PV",
    ):
        require_contract(**mx_contract)

    non_perblock_contract = dict(contract)
    non_perblock_contract["per_block_qk_scales"] = False
    with pytest.raises(ValueError, match="row-by-K16 forward Q/K scales"):
        require_contract(**non_perblock_contract)

    non_native_contract = dict(contract)
    non_native_contract["experimental_native_nvfp4_projection_out"] = False
    with pytest.raises(
        ValueError,
        match="native NVFP4 caller-owned QKV projection",
    ):
        require_contract(**non_native_contract)


def test_d128_represented_qk_requires_authenticated_native_d128_backend(
) -> None:
    validate = _d128_represented_native_backend_validator()
    with pytest.raises(ValueError, match="native TK D128 backend"):
        validate(
            SimpleNamespace(native_tk_d128_backward=False),
            True,
            True,
        )
    validate(
        SimpleNamespace(native_tk_d128_backward=True),
        True,
        True,
    )
    validate(
        SimpleNamespace(native_tk_d128_backward=False),
        True,
        False,
    )


def test_d128_runtime_preserves_direct_accumulator_binder_selection() -> None:
    require_contract = _compile_runtime_function(
        "_require_native_tk_d128_runtime_contract"
    )
    contract = _d128_runtime_contract_kwargs()
    contract["backward_match_forward_operands"] = False
    require_contract(**contract)

    invoke = _d128_nvfp4_binder_invoker()
    runtime = SimpleNamespace(
        publish_mxfp4_v=False,
        per_block_qk_scales=True,
        backward_match_forward_operands=False,
        experimental_d128_mxfp4_v_backward=False,
    )
    bound = invoke(runtime, contract["config"], False, False)

    assert bound["represented_backward"] is False


@pytest.mark.parametrize(
    ("represented", "semantics"),
    (
        (
            True,
            "represented_nvfp4_qk_per_row_k16_with_"
            "projection_accumulator_e4m3_v",
        ),
        (
            False,
            "projection_accumulator_e4m3_qkv_shared_across_pv_routes",
        ),
    ),
)
def test_d128_runtime_accepts_exact_binder_provenance(
    represented: bool,
    semantics: str,
) -> None:
    validate = _d128_projection_provenance_validator()
    runtime = SimpleNamespace(
        experimental_native_nvfp4_projection_out=True,
        backward_match_forward_operands=represented,
        qkv_projection=SimpleNamespace(
            represented_backward=represented,
            backward_publication_semantics=semantics,
        ),
    )

    validate(runtime, True)


@pytest.mark.parametrize(
    ("projection", "message"),
    (
        (
            SimpleNamespace(
                backward_publication_semantics=(
                    "represented_nvfp4_qk_per_row_k16_with_"
                    "projection_accumulator_e4m3_v"
                ),
            ),
            "omitted its exact-bool represented-backward provenance",
        ),
        (
            SimpleNamespace(
                represented_backward=1,
                backward_publication_semantics=(
                    "represented_nvfp4_qk_per_row_k16_with_"
                    "projection_accumulator_e4m3_v"
                ),
            ),
            "omitted its exact-bool represented-backward provenance",
        ),
        (
            SimpleNamespace(
                represented_backward=False,
                backward_publication_semantics=(
                    "represented_nvfp4_qk_per_row_k16_with_"
                    "projection_accumulator_e4m3_v"
                ),
            ),
            "disagrees with the runtime represented-backward selection",
        ),
        (
            SimpleNamespace(represented_backward=True),
            "backward-publication semantics disagree",
        ),
        (
            SimpleNamespace(
                represented_backward=True,
                backward_publication_semantics=(
                    "projection_accumulator_e4m3_qkv_shared_across_"
                    "pv_routes"
                ),
            ),
            "backward-publication semantics disagree",
        ),
    ),
)
def test_d128_runtime_fails_closed_on_missing_or_wrong_binder_provenance(
    projection: SimpleNamespace,
    message: str,
) -> None:
    validate = _d128_projection_provenance_validator()
    runtime = SimpleNamespace(
        experimental_native_nvfp4_projection_out=True,
        backward_match_forward_operands=True,
        qkv_projection=projection,
    )

    with pytest.raises(RuntimeError, match=message):
        validate(runtime, True)


def test_d128_bf16_reference_uses_the_projection_native_rotary_pairs() -> None:
    generator = torch.Generator().manual_seed(19)
    tensor = torch.randn(1, 5, 3, 128, generator=generator).bfloat16()
    cosine = torch.randn(1, 5, 64, generator=generator).bfloat16()
    sine = torch.randn(1, 5, 64, generator=generator).bfloat16()
    first, second = tensor.float().chunk(2, dim=-1)
    cosine_f = cosine.float().unsqueeze(2)
    sine_f = sine.float().unsqueeze(2)
    expected = torch.stack(
        (
            first * cosine_f - second * sine_f,
            first * sine_f + second * cosine_f,
        ),
        dim=-1,
    ).flatten(-2).bfloat16()

    actual = _apply_pair_rope(
        _split_half_to_adjacent_pairs(tensor),
        cosine,
        sine,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_d128_projection_returns_native_feature_major_fp8_v() -> None:
    source = _function_source(
        INTERFACE,
        "b300_project_qkv_gqa_d128_unified_lowp_nvfp4",
        "b300_project_dout_unified_lowp_nvfp4",
    )
    assert "projected[23]" in source
    assert "expected_forward_v_fp8 = (batch, kv_heads, 128, seqlen)" in source
    assert "not v_forward_fp8_raw.is_contiguous()" in source
    assert "v_forward_fp8=v_forward_fp8" in source


def test_native_tk_d128_probability_lift_is_fused_without_changing_cute() -> None:
    interface = _function_source(
        INTERFACE,
        "b300_project_dout_unified_lowp_nvfp4",
        "b300_mha_fwd",
    )
    assert "probability_log2_lift: float = 0.0" in interface
    assert "probability_log2_lift not in (0.0, 8.0)" in interface
    assert "stats_workspace is None" in interface

    epilogue = PROJECTION_EPILOGUE.read_text()
    assert "float dout_probability_log2_lift = 0.0f;" in epilogue
    assert "kDepth == 64" in epilogue
    assert ": g.dout_probability_log2_lift" in epilogue

    cuda = CUDA.read_text()
    assert (
        "probability_log2_lift == 0.0 || probability_log2_lift == 8.0"
        in cuda
    )
    assert "stats_workspace.has_value()" in cuda

    e2e = E2E.read_text()
    assert "8.0 if runtime.native_tk_d128_backward else 0.0" in e2e


def test_d128_causal_accurate_policy_selects_the_authenticated_safe_mx_route() -> None:
    source = HAO_DIRECT_MAKEFILE.read_text()
    policy = source.split(
        "ifeq ($(HAO_FP4PV_MX_POLICY),causal-accurate)", 1
    )[1].split(
        "ifeq ($(GPU)x$(HAO_HEAD_DIM),B200x64)", 1
    )[0]
    d128 = policy.split("ifeq ($(HAO_HEAD_DIM),128)", 1)[1]

    assert "override HAO_FP4PV_MX_SCALE_SELECT := 4" in d128
    assert "override HAO_FP4PV_MX_GLOBAL_ANCHOR32 := 1" in d128
    assert "override HAO_FP4PV_MX_GLOBAL_ANCHOR128 := 0" in d128
    assert "override HAO_FP4PV_MX_GLOBAL_ANCHOR_MARGIN_LOG2 := 64" in d128
    assert "override HAO_FP4PV_MX_ANCHOR_AFFINE_HOIST := 0" in d128
    assert "override HAO_FP4PV_MX_STORED_SCALE_SHIFT_LOG2 := 32" in d128
    assert "override HAO_FP4PV_NV_QK_FOLDED_K64_SCALES := 0" in d128
    assert "override HAO_FP4PV_NV_QK_PRELOAD_PAGE_MASK := 3" in d128


def test_d128_projection_forwards_explicit_mxfp4_v_scale_geometry() -> None:
    source = _function_source(
        INTERFACE,
        "b300_project_qkv_gqa_d128_unified_lowp_nvfp4",
        "b300_project_dout_unified_lowp_nvfp4",
    )
    assert "v_mxfp4_scale_2d: bool = False" in source
    assert "per_block_qk_scales: bool = False" in source
    assert "bool(v_mxfp4_scale_2d)," in source
    assert "bool(per_block_qk_scales)," in source
    cuda = CUDA.read_text()
    assert cuda.count(
        "bool v_mxfp4_scale_2d,\n    bool per_block_qk_scales\n) {"
    ) >= 3
    assert cuda.count(
        "v_mxfp4_scale_2d,\n        per_block_qk_scales"
    ) >= 4


def test_d128_perblock_qk_keeps_independent_e4m3_backward() -> None:
    cuda = CUDA.read_text()
    epilogue = (
        ROOT / "tk_fa4/lowp_fa4_bwd/projection_fp4_epilogue.cuh"
    ).read_text()
    perblock_publisher = epilogue.split(
        "void publish_forward_qk_perblock_scales(", 1
    )[1].split("template <", 1)[0]
    assert "bool kPublishRepresentedBackwardFp8 = false" in cuda
    assert (
        "kPublishForwardFp8,\n"
        "                kPublishRepresentedBackwardFp8, PerBlockQkScales"
    ) in cuda
    assert "template <bool STAGE_BF16_PAIRS, typename RT>" in epilogue
    assert (
        "PUBLISH_QK_FP8 &&\n"
        "                                        !PUBLISH_REPRESENTED_BACKWARD_FP8"
    ) in epilogue
    assert perblock_publisher.count(
        "g.paired_d64 ? chunk : head_idx"
    ) == 2
    assert perblock_publisher.count(
        "(g.paired_d64 || depth_base == 0)"
    ) == 2
    assert perblock_publisher.count("const int output_head_count") == 2
    assert perblock_publisher.count("const int output_head_idx") == 2
    assert (
        "PUBLISH_FP4 && PUBLISH_QK_FP8 &&\n"
        "            !PUBLISH_REPRESENTED_BACKWARD_FP8"
    ) in epilogue


def test_d128_cuda_projection_publishes_forward_fp8_without_interleave() -> None:
    source = CUDA.read_text().split(
        "project_qkv_unified_fp4_nvfp4_impl(", 1
    )[1].split(
        "project_qkv_unified_fp4_nvfp4(", 1
    )[0]
    assert "constexpr bool kPublishForwardFp8 =" in source
    assert "!kInterleaveCausalKv &&" in source
    assert "(!kCompactForwardOut || !kCompactPublishesMxV)" in source
    assert "k_backward_fp8,\n        v_forward_fp8\n    };" in source
    assert ".v_forward_fp8 = v_forward_fp8.numel()" in source


def test_forward_matrix_refuses_fp8_v_transpose_fallback() -> None:
    source = _function_source(FORWARD_MATRIX, "_make_projection_state", "_future_v_perturbation")
    assert ".permute(" not in source
    assert "refusing an unfused permute/contiguous fallback" in source
    assert '"exact_v_materialized_transpose": False' in source


def test_d128_chain_consumes_native_feature_major_fp8_v() -> None:
    source = D128_CHAIN.read_text()
    assert "forward_v_fp8_bhds = (\n        qkv.v_forward_fp8" in source
    assert "v_backward_fp8.permute" not in source
    assert 'timing["forward_v_fp8_layout_conversion"]' not in source
    assert '"unfused_fp8_v_layout_conversion": False' in source
    assert "refusing an unfused permute/contiguous fallback" in source


def test_d128_e2e_routes_every_learned_projection_through_2d_prep() -> None:
    source = E2E.read_text()
    runtime_init = source.split("class LowpAttentionRuntime", 1)[1].split(
        "class _LowpAttentionFunction", 1
    )[0]
    assert 'int(forward_topology.get("valid", 0)) == 1' in runtime_init
    assert "config.batch == 1 or" not in runtime_init
    assert "if not projection_weight_scale_2d:" in runtime_init
    assert "D128 learned projection weights require true 16x16" in runtime_init
    attention = source.split("class _LowpAttentionFunction", 1)[1].split(
        "class LowpAttention", 1
    )[0]
    assert (
        "b300_prepare_nvfp4_projection_weight\n"
        "            if runtime.projection_weight_scale_2d"
    ) in attention
    assert "qkv_weight_operand = tuple(prepare_weight(qkv_weight))" in attention
    assert (
        "out_weight_operand = tuple(\n"
        "                        prepare_weight(out_weight)"
    ) in attention
    assert "_prepare_direct_dual_output_weight(" in attention
    assert "_uses_direct_dual_output_weight_prep(runtime)" in attention
    assert (
        "ctx.output_weight_backward_operand = out_weight_backward_operand"
        in attention
    )
    assert "if out_weight_backward_operand is None:" in attention
    assert "prepare_weight(out_weight.T.contiguous())" in attention
    assert "prepare_weight(qkv_weight.T.contiguous())" in attention


def test_d128_isolated_profilers_use_2d_learned_weight_prep() -> None:
    projection = D128_PROJECTION_PROFILE.read_text()
    pair = CAUSAL_PAIR_PROFILE.read_text()
    chain = D128_CHAIN.read_text()
    assert "b300_prepare_nvfp4_projection_weight(qkv_weight)" in projection
    assert "b300_prepare_nvfp4_projection_operand(qkv_weight)" not in projection
    assert "b300_prepare_nvfp4_projection_weight(qkv_weight)" in pair
    assert "b300_prepare_nvfp4_projection_weight(output_weight)" in pair
    assert "b300_prepare_nvfp4_projection_weight(qkv_weight)" in chain
    assert "b300_prepare_nvfp4_projection_weight(out_weight)" in chain
    assert "b300_prepare_nvfp4_projection_weight(out_weight.T.contiguous())" in chain
    assert "b300_prepare_nvfp4_projection_weight(\n            qkv_weight.T" in chain
    assert "b300_prepare_nvfp4_projection_weight(hadamard_weight_t)" in chain


def test_matched_route_profiler_does_not_disable_d128_perblock_qk() -> None:
    source = MATCHED_PROFILE.read_text()
    assert '"per_block_qk_scales": args.per_block_qk_scales' in source
    assert '"per_block_qk_scales": not d128' not in source
