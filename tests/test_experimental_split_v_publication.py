from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
INTERFACE = ROOT / "tk_fa4" / "interface.py"
RUNTIME = ROOT / "tk_fa4" / "lowp_fa4_bwd" / "benchmark_llama12b_e2e.py"
COMPARE = ROOT / "tk_fa4" / "lowp_fa4_bwd" / "compare_llama12b_mx_fp8pv.py"
TRAIN = ROOT / "tk_fa4" / "lowp_fa4_bwd" / "train_llama12b_real_tokens.py"
CUDA = ROOT / "tk_fa4" / "lowp_fa4_bwd" / "lowp_fa4_bwd.cu"
EPILOGUE = (
    ROOT / "tk_fa4" / "lowp_fa4_bwd" / "projection_fp4_epilogue.cuh"
)


def _function(path: Path, name: str) -> ast.FunctionDef:
    module = ast.parse(path.read_text())
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing {name} in {path}")


def _method(path: Path, class_name: str, name: str) -> ast.FunctionDef:
    module = ast.parse(path.read_text())
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for member in node.body:
                if isinstance(member, ast.FunctionDef) and member.name == name:
                    return member
    raise AssertionError(f"missing {class_name}.{name} in {path}")


def _keyword_default(function: ast.FunctionDef, name: str) -> ast.expr | None:
    defaults = dict(zip(function.args.kwonlyargs, function.args.kw_defaults))
    for argument, default in defaults.items():
        if argument.arg == name:
            return default
    raise AssertionError(f"missing keyword {name} in {function.name}")


def test_python_opt_in_defaults_are_false() -> None:
    interface = _function(
        INTERFACE,
        "b300_project_qkv_gqa_d64_paired_unified_lowp_e4m3",
    )
    runtime = _method(RUNTIME, "LowpAttentionRuntime", "__init__")
    make_runtime = _function(COMPARE, "_make_runtime")
    for function in (interface, runtime, make_runtime):
        default = _keyword_default(function, "experimental_split_v_backward")
        assert isinstance(default, ast.Constant)
        assert default.value is False


def test_interface_selects_a_distinct_validated_projection_symbol() -> None:
    source = INTERFACE.read_text()
    assert (
        "publish_mxfp4_v and represented_backward and per_block_qk_scales"
        in source
    )
    assert 'project_name += "_split_v_backward"' in source
    assert (
        "experimental_split_v_backward requires MXFP4 V, represented "
        in source
    )


def test_kernel_splits_only_v_publication() -> None:
    source = EPILOGUE.read_text()
    normalized = " ".join(source.split())
    assert "bool EXPERIMENTAL_SPLIT_V_BACKWARD = false" in source
    assert (
        "PUBLISH_REPRESENTED_BACKWARD_FP8 && PER_BLOCK_QK_SCALES &&\n"
        "             PUBLISH_V_MXFP4 && PUBLISH_V_FP8"
        in source
    )
    assert (
        "PUBLISH_REPRESENTED_BACKWARD_FP8 && PUBLISH_V_MXFP4 && "
        "!EXPERIMENTAL_SPLIT_V_BACKWARD"
        in normalized
    )
    assert (
        "!PUBLISH_V_MXFP4 || EXPERIMENTAL_SPLIT_V_BACKWARD"
        in normalized
    )
    # Q/K represented-code publication is deliberately independent of the
    # split-V switch.
    assert "publish_qk_fp8_from_perblock_codes(" in source


def test_split_v_publishes_direct_backward_fp8_before_forward_mx() -> None:
    source = EPILOGUE.read_text()
    publication = source.split(
        "if constexpr (PUBLISH_V_MXFP4 || PUBLISH_V_FP8)",
        1,
    )[1].split(
        "if constexpr (\n"
        "                                OUTPUT_IS_DOUT && PUBLISH_DOUT_STATS",
        1,
    )[0]
    ordinary = publication.split(
        "} else {\n"
        "                                if constexpr (\n"
        "                                    PUBLISH_V_FP8 &&\n"
        "                                    EXPERIMENTAL_SPLIT_V_BACKWARD &&\n"
        "                                    !PUBLISH_FORWARD_FP8",
        1,
    )[1]

    direct_backward = ordinary.index("publish_v_fp8<C, false>(")
    forward_mx = ordinary.index("publish_v_mxfp4<")
    legacy_fp8 = ordinary.index(
        "publish_v_fp8<C, PUBLISH_FORWARD_FP8>("
    )
    assert direct_backward < forward_mx < legacy_fp8

    # The order probe reuses the one staged BF16 tile: the early FP8
    # specialization does not restage or overwrite it, and no new barrier or
    # publication helper is introduced between the two split-V consumers.
    before_mx = ordinary[direct_backward:forward_mx]
    assert "stage_bf16_pairs(" not in before_mx
    assert "kittens::warpgroup::sync(1);" not in before_mx
    assert ordinary.count("publish_v_fp8<C, false>(") == 1
    assert ordinary.count("publish_v_mxfp4<") == 1


def test_split_v_reorder_preserves_non_split_publication_contracts() -> None:
    source = EPILOGUE.read_text()
    publication = source.split(
        "if constexpr (PUBLISH_V_MXFP4 || PUBLISH_V_FP8)",
        1,
    )[1].split(
        "if constexpr (\n"
        "                                OUTPUT_IS_DOUT && PUBLISH_DOUT_STATS",
        1,
    )[0]
    normalized = " ".join(publication.split())

    # One shared BF16 staging operation and its original barrier still feed
    # all ordinary V publishers.  The derived-MX branch keeps its separate,
    # explicitly synchronized contract unchanged.
    assert publication.count("stage_bf16_pairs(") == 1
    assert publication.count("kittens::warpgroup::sync(1);") == 2
    assert publication.count("publish_v_fp8<C, false, true>(") == 1
    assert publication.count(
        "publish_v_mxfp4_from_backward_e4m3<"
    ) == 1

    # The native MX publisher retains all five template policy arguments;
    # the legacy FP8 publisher retains its policy argument and is merely
    # excluded from the split specialization after its early direct store.
    assert (
        "PUBLISH_REPRESENTED_BACKWARD_FP8 && PUBLISH_V_MXFP4 && "
        "!EXPERIMENTAL_SPLIT_V_BACKWARD"
    ) in normalized
    assert "publish_v_fp8<C, PUBLISH_FORWARD_FP8>(" in publication
    late_fp8_condition = publication.split(
        "publish_v_mxfp4<",
        1,
    )[1].split(
        "publish_v_fp8<C, PUBLISH_FORWARD_FP8>(",
        1,
    )[0]
    assert (
        "!EXPERIMENTAL_SPLIT_V_BACKWARD || PUBLISH_FORWARD_FP8"
        in " ".join(late_fp8_condition.split())
    )


def test_cuda_exports_legacy_and_preallocated_mx_specializations() -> None:
    source = CUDA.read_text()
    symbol = (
        "project_qkv_gqa_d64_paired_unified_fp8_e4m3_rope_packed_"
        "interleaved_causal_represented_backward_perblock_qk_"
        "split_v_backward"
    )
    out_symbol = symbol + "_vscale_out"
    assert f"\n{symbol}(\n" in source
    assert f"&{symbol},\n" in source
    assert f"\n{out_symbol}(\n" in source
    assert f"&{out_symbol},\n" in source
    assert "bool ExperimentalSplitVBackward = false" in source
    assert "PublishRepresentedBackwardFp8 && PerBlockQkScales" in source
    assert "publish_mxfp4_v && interleave_causal_kv" in source
    assert (
        "project_qkv_gqa_d64_paired_unified_fp8_e4m3_rope_packed_"
        "interleaved_causal_impl<\n        true,\n        true,\n        true"
        in source
    )


def test_preallocated_vscale_symbol_validates_and_reuses_exact_output() -> None:
    source = CUDA.read_text()
    assert (
        "std::optional<at::Tensor> v_mxfp4_scales_out = std::nullopt"
        in source
    )
    assert "output.scalar_type() == at::kFloat8_e4m3fn" in source
    assert "output.is_cuda() && output.is_contiguous()" in source
    assert "output.device() == input_fp8.device()" in source
    assert (
        "batch, seq_len / 128, kv_heads, 512" in source
    )
    publication = source.split(
        "v_mxfp4_scales = publish_mxfp4_v", 1
    )[1].split("v_forward_fp8 = publish_mxfp4_v", 1)[0]
    assert "v_mxfp4_scales_out.value()" in publication
    assert "at::empty(" in publication
    assert "_split_v_backward_vscale_out" in source


def test_compact_forward_out_symbols_preserve_exact_route_abi() -> None:
    source = CUDA.read_text()
    fp8_base = (
        "project_qkv_gqa_d64_paired_unified_fp8_e4m3_rope_packed_"
        "interleaved_causal_represented_backward_perblock_qk_"
    )
    mx_base = fp8_base + "split_v_backward_"
    for symbol in (
        fp8_base + "fp8_forward_out",
        fp8_base + "fp8_forward_out_unchecked",
        mx_base + "mx_forward_out",
        mx_base + "mx_forward_out_unchecked",
    ):
        assert f"\n{symbol}(\n" in source
        assert f"&{symbol},\n" in source
    assert "struct paired_d64_e4m3_forward_outputs" in source
    for output in (
        "q_depth_packed",
        "k_depth_packed",
        "q_forward_scales",
        "q_forward_global_scale",
        "k_forward_scales",
        "k_forward_global_scale",
        "v_mxfp4",
        "v_mxfp4_scales",
        "v_forward_fp8",
        "v_backward_fp8",
        "q_backward_fp8",
        "k_backward_fp8",
    ):
        assert f"at::Tensor {output};" in source
    for output in (
        "v_backward_fp8_out",
        "q_backward_fp8_out",
        "k_backward_fp8_out",
    ):
        assert f'"{output}"' in source
    assert "for (int lhs = 0; lhs < 12; ++lhs)" in source
    assert "for (int rhs = lhs + 1; rhs < 12; ++rhs)" in source
    assert "bool CompactForwardOut = false" in source
    assert "bool ValidateCompactContracts = false" in source
    assert "if constexpr (!CompactForwardOut || ValidateCompactContracts)" in source
    assert "return {v_backward_fp8, q_backward_fp8, k_backward_fp8};" in source
    assert "input_fp8.data_ptr()" in source
    publication_setup = source.split("at::Tensor empty_bf16;", 1)[1].split(
        "// K128 keeps dense FP8", 1
    )[0]
    caller_owned, allocating = publication_setup.split("} else {", 1)
    assert "at::empty(" not in caller_owned
    for output in (
        "v_backward_fp8",
        "q_backward_fp8",
        "k_backward_fp8",
    ):
        assert f"{output} = forward_outputs->{output};" in caller_owned
        assert f"{output} = at::empty(" in allocating


def test_runtime_rejects_non_mx_or_non_perblock_split_v() -> None:
    source = RUNTIME.read_text()
    assert 'pv_format == "mxfp4_e8m0_block32"' in source
    assert 'qkv_projection_format == "e4m3"' in source
    assert "and self.per_block_qk_scales" in source
    assert (
        '"v_backward_source": v_backward_source' in source
        and 'v_backward_source = "projection_accumulator_e4m3"' in source
    )
    assert (
        "experimental_split_v_backward=(\n"
        "                        self.experimental_split_v_backward"
        in source
    )


def test_training_cli_records_policy_and_does_not_enable_exact_fp8() -> None:
    source = TRAIN.read_text()
    assert '"--mx-experimental-split-v-backward"' in source
    assert '"mx_experimental_split_v_backward"' in source
    assert '"mx_projection_publication_topology"' in source
    assert '"fp8_projection_publication_topology"' in source
    fp8_call = source.split(
        'if "nvfp4_qk_fp8_pv_exact" in route_names:', 1
    )[1].split("torch.manual_seed(args.seed)", 1)[0]
    assert "experimental_split_v_backward" not in fp8_call


def test_training_source_serializes_runtime_binary_provenance() -> None:
    source = TRAIN.read_text()
    assert '"schema": "llama12b_real_tokens_training_v3"' in source
    assert '"command": [sys.executable, *sys.argv]' in source
    assert '"source": source_identity' in source
    assert '"python_executable": sys.executable' in source
    assert '"mx_forward_extension": mx_forward_extension' in source
    assert '"fp8_forward_extension": fp8_forward_extension' in source
    identity = _function(TRAIN, "_extension_identity")
    rendered = ast.unparse(identity)
    assert '**_file_identity(path)' in rendered


def test_training_gradient_clipping_is_opt_in_and_recorded() -> None:
    step = _function(TRAIN, "_step")
    default = _keyword_default(step, "gradient_clip_norm")
    assert isinstance(default, ast.Constant)
    assert default.value is None
    source = TRAIN.read_text()
    assert "torch.nn.utils.clip_grad_norm_(" in source
    assert "error_if_nonfinite=True" in source
    assert "foreach=True" in source
    assert '"gradient_preclip_total_norm"' in source
    assert '"gradient_clip_ms"' in source
    assert '"gradient_clipping"' in source


def test_extension_identity_hashes_the_selected_file(tmp_path: Path) -> None:
    payload = b"selected runtime extension\x00\x01"
    extension = tmp_path / "runtime.so"
    extension.write_bytes(payload)
    functions = ast.Module(
        body=[
            _function(TRAIN, "_file_sha256"),
            _function(TRAIN, "_file_identity"),
            _function(TRAIN, "_extension_identity"),
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(functions)
    namespace: dict[str, Any] = {
        "Any": Any,
        "Path": Path,
        "hashlib": hashlib,
    }
    exec(compile(functions, str(TRAIN), "exec"), namespace)
    identity = namespace["_extension_identity"](extension, "runtime_probe")
    assert identity == {
        "module": "runtime_probe",
        "path": str(extension.resolve()),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
