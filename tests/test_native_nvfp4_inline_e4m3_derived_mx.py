from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
CUDA = ROOT / "tk_fa4" / "lowp_fa4_bwd" / "lowp_fa4_bwd.cu"
INTERFACE = ROOT / "tk_fa4" / "interface.py"
BENCHMARK = (
    ROOT
    / "tk_fa4"
    / "lowp_fa4_bwd"
    / "benchmark_native_inline_e4m3_derived_mx.py"
)
EPILOGUE = (
    ROOT / "tk_fa4" / "lowp_fa4_bwd" / "projection_fp4_epilogue.cuh"
)

SYMBOL = (
    "project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_"
    "interleaved_causal_represented_backward_perblock_qk_"
    "e4m3_derived_mx_forward_out"
)


def _load_benchmark_module():
    spec = importlib.util.spec_from_file_location(
        "benchmark_native_inline_e4m3_derived_mx_test",
        BENCHMARK,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _function_body(source: str, name: str, next_marker: str) -> str:
    return source.split(name, 1)[1].split(next_marker, 1)[0]


def test_inline_e4m3_derived_route_is_compile_time_opt_in() -> None:
    cuda = CUDA.read_text()
    epilogue = EPILOGUE.read_text()
    assert "bool kExperimentalE4m3DerivedMxfp4V = false" in cuda
    assert "bool ExperimentalE4m3DerivedMxfp4V = false" in cuda
    assert "bool EXPERIMENTAL_E4M3_DERIVED_MXFP4_V = false" in epilogue
    assert (
        "kInterleaveCausalKv && !kExperimentalSplitVBackward" in cuda
    )
    assert (
        "PUBLISH_V_MXFP4 && PUBLISH_V_FP8 && INTERLEAVE_CAUSAL_KV &&\n"
        "             !PUBLISH_FORWARD_FP8 && !EXPERIMENTAL_SPLIT_V_BACKWARD"
        in epilogue
    )


def test_inline_route_consumes_exact_staged_backward_e4m3() -> None:
    source = EPILOGUE.read_text()
    branch = source.split(
        "if constexpr (\n"
        "                                EXPERIMENTAL_E4M3_DERIVED_MXFP4_V",
        1,
    )[1].split("} else {", 1)[0]
    backward = branch.index("publish_v_fp8<C, false, true>(")
    barrier = branch.index("kittens::warpgroup::sync(1);", backward)
    derived = branch.index("publish_v_mxfp4_from_backward_e4m3<", barrier)
    assert backward < barrier < derived
    assert "publish_v_mxfp4<" not in branch

    publisher = _function_body(
        source,
        "publish_v_mxfp4_from_backward_e4m3(",
        "template <\n    typename C,\n"
        "    bool SEQUENCE_MAJOR_COLUMN_SCALES = false,\n"
        "    bool INTERLEAVE_CAUSAL_KV = false,\n"
        "    bool PUBLISH_BACKWARD_MXFP4 = false,\n"
        "    bool SHARE_MXFP4_TILE_WITH_BACKWARD = false\n"
        ">\n"
        "__device__ __noinline__ void "
        "publish_v_mxfp4_from_output_shared(",
    )
    assert "g.v_mxfp4 + payload_base" in publisher
    assert "g.v_mxfp4_scales[" in publisher
    assert "convert_scaled_bf16_pair_to_fp8" not in publisher
    assert "pairs[source_warp0][source_row0][byte_word]" in publisher
    assert "selected_amax_code == 0x7fu" in publisher
    assert "? 0xffu" in publisher
    assert "finite_pair_mask = nan_group ? 0u : 0xffffu" in publisher
    for forbidden in (
        "g.v_backward_fp8",
        "g.q_backward_fp8",
        "g.k_backward_fp8",
        "g.q_depth_packed",
        "g.k_depth_packed",
    ):
        assert forbidden not in publisher


def test_backward_e4m3_store_precedes_optional_scratch_staging() -> None:
    source = EPILOGUE.read_text()
    publisher = _function_body(
        source,
        "void publish_v_fp8(",
        "template <typename C>\n"
        "__device__ __forceinline__ void publish_qk_fp8(",
    )
    first_store = publisher.index("g.v_backward_fp8 + output_base")
    staging = publisher.index(
        "PUBLISH_FORWARD_FP8 || STAGE_BACKWARD_FP8_FOR_MXFP4"
    )
    assert first_store < staging
    assert "pairs[warp][lane][word] = words[word];" in publisher


def test_inline_converter_uses_packed_native_half2_path() -> None:
    source = EPILOGUE.read_text()
    quantizer = _function_body(
        source,
        "quantize_four_e4m3_pairs_to_mxfp4(",
        "__device__ __forceinline__ uint16_t bf16_pair_amax_bits(",
    )
    assert quantizer.count("cvt.rn.f16x2.e4m3x2") == 4
    assert quantizer.count("mul.rn.f16x2") == 4
    assert quantizer.count("cvt.rn.satfinite.e2m1x2.f16x2") == 4
    assert "cvt.f32" not in quantizer

    publisher = _function_body(
        source,
        "publish_v_mxfp4_from_backward_e4m3(",
        "template <\n    typename C,\n"
        "    bool SEQUENCE_MAJOR_COLUMN_SCALES = false,\n"
        "    bool INTERLEAVE_CAUSAL_KV = false,\n"
        "    bool PUBLISH_BACKWARD_MXFP4 = false,\n"
        "    bool SHARE_MXFP4_TILE_WITH_BACKWARD = false\n"
        ">\n"
        "__device__ __noinline__ void "
        "publish_v_mxfp4_from_output_shared(",
    )
    assert "e4m3_x4_amax_to_logical_bf16_bits(" in publisher
    assert "e8m0_e4m3_x4_encode_multiplier_half2(e8m0)" in publisher
    assert "decode_fp8_pair_to_float2" not in publisher
    assert "pairs[source_warp0][source_row0][byte_word]" in publisher
    assert "pair_index" not in publisher


def test_checked_and_unchecked_symbols_have_only_new_flag_enabled() -> None:
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
        "true,true,true,true,false,true,false"
    )
    assert unchecked.replace(" ", "").replace("\n", "") == (
        "true,false,true,true,false,true,false"
    )
    for exported in (SYMBOL, f"{SYMBOL}_unchecked"):
        assert f"&{exported}," in source
    assert '"e4m3_derived_mx_forward_out",' in source
    assert '"e4m3_derived_mx_forward_out_unchecked",' in source


def test_existing_native_forward_out_routes_keep_derived_flag_disabled() -> None:
    source = CUDA.read_text()
    macro_region = source.split(
        "TKFA4_DEFINE_D64_NVFP4_FORWARD_OUT(\n",
        1,
    )[1].split("#undef TKFA4_DEFINE_D64_NVFP4_FORWARD_OUT", 1)[0]
    calls = macro_region.split("TKFA4_DEFINE_D64_NVFP4_FORWARD_OUT(")
    legacy_calls = [
        call
        for call in calls
        if "e4m3_derived" not in call and "output_shared_split_v" not in call
    ]
    assert len(legacy_calls) == 8
    for call in legacy_calls:
        arguments = call.split(")", 1)[0]
        assert arguments.rstrip().endswith("false")


def test_python_binder_keeps_derived_route_explicit_and_fail_closed() -> None:
    source = INTERFACE.read_text()
    binder = _function_body(
        source,
        "class B300BoundNVFP4QKVProjection:",
        "def b300_bind_qkv_gqa_d64_paired_unified_lowp_nvfp4_projection(",
    )
    assert "experimental_e4m3_derived_mxfp4_v: bool = False" in binder
    assert "and not self.publish_mxfp4_v" in binder
    assert "and self.v_mxfp4_scale_2d" in binder
    assert "supports only the" in binder
    assert "v_mxfp4_scale_2d=False" in binder
    assert "e4m3_derived_mx_forward_out" in binder
    assert (
        "self.experimental_split_v_backward = bool(\n"
        "            self.publish_mxfp4_v\n"
        "            and not self.experimental_e4m3_derived_mxfp4_v"
        in binder
    )
    assert (
        "convert_e4m3_x4_v_bhds_to_causal_mxfp4" in binder
    )
    assert "legacy_bundle.v_backward_fp8.permute(0, 2, 3, 1)" in binder
    assert "valid_last_dim_indices=tuple(" in binder


def test_public_native_binder_defaults_to_existing_dispatch() -> None:
    source = INTERFACE.read_text()
    public_binder = source.split(
        "def b300_bind_qkv_gqa_d64_paired_unified_lowp_nvfp4_projection(",
        1,
    )[1].split(
        "def b300_require_qkv_gqa_d64_paired_unified_lowp_e4m3_projection(",
        1,
    )[0]
    assert "experimental_e4m3_derived_mxfp4_v: bool = False" in public_binder
    assert (
        "experimental_e4m3_derived_mxfp4_v=(\n"
        "            experimental_e4m3_derived_mxfp4_v\n"
        "        )"
        in public_binder
    )


def test_checked_native_out_abi_rejects_aliases_and_misalignment() -> None:
    source = CUDA.read_text()
    checked = _function_body(
        source,
        "inline void check_nvfp4_qkv_forward_outputs(",
        "inline void check_paired_d64_nvfp4_forward_outputs(",
    )
    assert "constexpr std::uintptr_t kOutputAlignment = alignof(uint4)" in (
        checked
    )
    assert "static_assert(kOutputAlignment == 16)" in checked
    assert "address % kOutputAlignment == 0" in checked
    assert "must have a 16-byte-aligned base" in checked
    assert "const at::Tensor *read_tensors[]" in checked
    assert "const at::Tensor *output_tensors[]" in checked
    assert "ranges_overlap(" in checked
    assert "must occupy disjoint byte ranges" in checked
    assert "must not overlap read operand" in checked
    compact_check = source.split(
        "if constexpr (kCompactForwardOut) {", 1
    )[1].split("check_paired_d64_nvfp4_forward_outputs(", 1)[0]
    assert "rope_packed != nullptr" in compact_check


def test_projection_ab_uses_authenticated_shape_and_receipt_guards() -> None:
    source = BENCHMARK.read_text()
    assert "BATCH = 16" in source
    assert "SEQUENCE = 4096" in source
    assert "HIDDEN = 2048" in source
    assert "Q_HEADS = 32" in source
    assert "KV_HEADS = 8" in source
    assert "HEAD_DIM = 64" in source
    assert "MINIMUM_SAMPLES = 100" in source
    assert "MINIMUM_BOOTSTRAP_DRAWS = 1_000" in source
    assert "balanced_order_block_deltas" in source
    assert "bootstrap_mean_95_percent_us" in source
    assert "torch.cuda.device_count() != 1" in source
    assert "torch.cuda.list_gpu_processes(0)" in source
    assert source.count("_require_exclusive_visible_gpu(torch)") == 3
    assert "_authenticate_loaded_interface(" in source
    assert "loaded_interface_before" in source
    assert "loaded_interface_after" in source
    assert "extension_before != extension_after" in source
    assert "sources_before != sources_after" in source
    assert "os.O_EXCL" in source
    assert "stream.flush()" in source
    assert "os.fsync(stream.fileno())" in source
    assert '"report": report' not in source
    assert '"exclusive_visible_gpu_check_count"' in source
    assert '"foreign_gpu_process_ids_across_checks"' in source
    assert "def _require_checked_input_output_alias_rejection(" in source
    assert "projector._project_checked(" in source
    assert '"must not overlap read operand input_fp4"' in source
    assert '"checked_input_output_alias_rejection"' in source


def test_loaded_interface_authentication_rejects_shadowing(
    tmp_path: Path,
) -> None:
    benchmark = _load_benchmark_module()
    expected = tmp_path / "expected" / "tk_fa4" / "interface.py"
    expected.parent.mkdir(parents=True)
    expected.write_text("EXPECTED = True\n")
    matching = benchmark._authenticate_loaded_interface(
        SimpleNamespace(__file__=str(expected)),
        expected,
    )
    assert matching["resolved_path_matches_expected"]
    assert matching["sha256_matches_expected"]
    assert (
        matching["loaded_file_identity"]["sha256"]
        == matching["expected_file_identity"]["sha256"]
    )

    shadow = tmp_path / "shadow" / "tk_fa4" / "interface.py"
    shadow.parent.mkdir(parents=True)
    shadow.write_text(expected.read_text())
    with pytest.raises(RuntimeError, match="interface is shadowed"):
        benchmark._authenticate_loaded_interface(
            SimpleNamespace(__file__=str(shadow)),
            expected,
        )
    with pytest.raises(RuntimeError, match="no filesystem __file__"):
        benchmark._authenticate_loaded_interface(
            SimpleNamespace(__file__=None),
            expected,
        )


@pytest.mark.parametrize(
    "report",
    (
        "",
        "pynvml module not found, please install pynvml",
        "cuda driver can't be loaded, is cuda enabled?",
        "GPU:0",
        "GPU:0\nunknown process state",
        "GPU:0\nno processes are running\nprocess 7 uses 1.000 MB GPU memory",
        "GPU:0\nprocess 7 uses 1.000 MB GPU memory\n"
        "process 7 uses 2.000 MB GPU memory",
    ),
)
def test_gpu_process_parser_fails_closed_on_malformed_reports(
    report: str,
) -> None:
    benchmark = _load_benchmark_module()
    with pytest.raises(RuntimeError, match="CUDA process report"):
        benchmark._parse_gpu_process_report(report)


def test_gpu_process_parser_accepts_only_explicit_process_states() -> None:
    benchmark = _load_benchmark_module()
    empty = benchmark._parse_gpu_process_report(
        "GPU:1\nno processes are running"
    )
    assert empty == {
        "reported_gpu_index": 1,
        "observed_process_ids": [],
        "process_count": 0,
        "report_parsed": True,
    }
    populated = benchmark._parse_gpu_process_report(
        "GPU:3\nprocess       17 uses      680.000 MB GPU memory\n"
        "process       29 uses       12.500 MB GPU memory"
    )
    assert populated["reported_gpu_index"] == 3
    assert populated["observed_process_ids"] == [17, 29]
    assert populated["process_count"] == 2


def test_gpu_process_enumeration_failure_and_foreign_pid_are_fatal() -> None:
    benchmark = _load_benchmark_module()

    class UnavailableCuda:
        @staticmethod
        def list_gpu_processes(_device):
            raise OSError("NVML unavailable")

    with pytest.raises(RuntimeError, match="process enumeration failed"):
        benchmark._require_exclusive_visible_gpu(
            SimpleNamespace(cuda=UnavailableCuda())
        )

    foreign_pid = os.getpid() + 1

    class ForeignCuda:
        @staticmethod
        def list_gpu_processes(_device):
            return (
                "GPU:0\n"
                f"process {foreign_pid} uses 1.000 MB GPU memory"
            )

    with pytest.raises(RuntimeError, match="foreign compute PIDs"):
        benchmark._require_exclusive_visible_gpu(
            SimpleNamespace(cuda=ForeignCuda())
        )


def test_timing_checks_exclusivity_after_every_balanced_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = _load_benchmark_module()
    state = {"elapsed_calls": 0, "checks": 0}

    class FakeEvent:
        def __init__(self, *, enable_timing: bool):
            assert enable_timing

        def record(self) -> None:
            return None

        def synchronize(self) -> None:
            return None

        def elapsed_time(self, _end) -> float:
            state["elapsed_calls"] += 1
            return 1.0

    class FakeCuda:
        Event = FakeEvent

        @staticmethod
        def synchronize() -> None:
            return None

    def require_exclusive(_torch):
        # Each complementary block contains two iterations, and each
        # iteration reads two provider event intervals before this check.
        assert state["elapsed_calls"] == (state["checks"] + 1) * 4
        state["checks"] += 1
        return {
            "reported_gpu_index": 0,
            "observed_process_ids": [os.getpid()],
            "process_count": 1,
            "report_parsed": True,
            "own_pid": os.getpid(),
            "foreign_process_ids": [],
        }

    monkeypatch.setattr(
        benchmark,
        "_require_exclusive_visible_gpu",
        require_exclusive,
    )
    result = benchmark._measure(
        SimpleNamespace(cuda=FakeCuda()),
        {
            "direct_native_mx": lambda: None,
            "inline_e4m3_derived_native_mx": lambda: None,
        },
        warmups=0,
        samples=4,
        bootstrap_draws=1_000,
        seed=11,
    )
    exclusivity = result["periodic_gpu_exclusivity"]
    assert state["checks"] == 2
    assert exclusivity["check_count"] == 2
    assert exclusivity["expected_check_count"] == 2
    assert exclusivity["foreign_process_ids"] == []
    assert [check["balanced_order_block"] for check in exclusivity["checks"]] == [
        0,
        1,
    ]


def test_plan_records_reproduction_parameters_and_write_is_create_only(
    tmp_path: Path,
) -> None:
    benchmark = _load_benchmark_module()
    output = tmp_path / "receipt.json"
    args = benchmark._parse_args(
        [
            "--extension",
            str(tmp_path / "extension.so"),
            "--output",
            str(output),
            "--seed",
            "71",
            "--warmups",
            "14",
            "--samples",
            "100",
            "--bootstrap-draws",
            "1234",
        ]
    )
    assert benchmark._plan(args)["parameters"] == {
        "seed": 71,
        "warmups": 14,
        "samples": 100,
        "bootstrap_draws": 1234,
    }
    benchmark._write_create_only(output, "first\n")
    assert output.read_text() == "first\n"
    with pytest.raises(FileExistsError):
        benchmark._write_create_only(output, "second\n")


def test_balanced_order_bootstrap_is_deterministic_and_rejects_zero() -> None:
    benchmark = _load_benchmark_module()
    # These are already complementary-order block effects. Their strictly
    # positive resampled means must exclude zero despite within-block noise.
    values = [3.0, 5.0, 4.0, 6.0, 2.0, 4.0]
    first = benchmark._bootstrap_mean_interval(
        values,
        draws=2_000,
        seed=17,
    )
    second = benchmark._bootstrap_mean_interval(
        values,
        draws=2_000,
        seed=17,
    )
    assert first == second
    assert first[0] > 0.0


def test_derived_binder_rejects_2d_mx_scaling_before_extension_lookup() -> None:
    from tk_fa4.interface import B300BoundNVFP4QKVProjection

    with pytest.raises(ValueError, match="rowwise 1x32 scale policy"):
        B300BoundNVFP4QKVProjection(
            batch=1,
            seqlen=128,
            q_heads=2,
            kv_heads=2,
            publish_mxfp4_v=True,
            v_mxfp4_scale_2d=True,
            experimental_e4m3_derived_mxfp4_v=True,
        )
