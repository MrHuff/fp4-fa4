from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "tk_fa4"
    / "lowp_fa4_bwd"
    / "validate_native_tk_d128_backward.py"
)
NATIVE_D128_SOURCE = ROOT / "tk_fa4" / "native_gqa_tk_bwd"
SPEC = importlib.util.spec_from_file_location(
    "validate_native_tk_d128_backward",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
validator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validator
SPEC.loader.exec_module(validator)

DIRECT_OUTPUT_ENTRYPOINT = validator.DIRECT_OUTPUT_ENTRYPOINT
EXPECTED_SEMANTIC_METADATA = validator.EXPECTED_SEMANTIC_METADATA
CANDIDATE_SOURCE_IDENTITY = "v420_test_d128_gqa_exact_v1"
DirectOutputs = validator.DirectOutputs
Shape = validator.Shape
_finite_and_nontrivial = validator._finite_and_nontrivial
_stable_file_identity = validator._stable_file_identity
check_exact_zero_dout = validator.check_exact_zero_dout
launch_reset_inclusive = validator.launch_reset_inclusive
make_represented_state = validator.make_represented_state
represented_e4m3_causal_reference = (
    validator.represented_e4m3_causal_reference
)
require_direct_output_entrypoint = validator.require_direct_output_entrypoint
require_extension_metadata = validator.require_extension_metadata
time_reset_inclusive = validator.time_reset_inclusive


class _FakeExtension:
    def __init__(self, metadata: dict[str, Any]) -> None:
        self.metadata = metadata

    def native_tk_d128_backward_metadata(self) -> dict[str, Any]:
        return dict(self.metadata)


def _identity(path: Path) -> tuple[str, int]:
    payload = path.read_bytes()
    return hashlib.sha256(payload).hexdigest(), len(payload)


def _native_source(name: str) -> str:
    return (NATIVE_D128_SOURCE / name).read_text()


def _metadata(
    source_file: str,
    *,
    source_identity: str = CANDIDATE_SOURCE_IDENTITY,
) -> dict[str, Any]:
    return {
        **EXPECTED_SEMANTIC_METADATA,
        "source_identity": source_identity,
        "source_file": source_file,
        "topology": "test_d128_k128_q128_topology",
    }


@pytest.mark.parametrize("batch", (1, 2))
@pytest.mark.parametrize("sequence", (128, 4096, 16_384))
def test_shape_accepts_only_production_geometry(batch: int, sequence: int) -> None:
    shape = Shape(batch=batch, sequence=sequence)
    assert shape.q_shape == (batch, sequence, 32, 128)
    assert shape.kv_shape == (batch, sequence, 8, 128)
    assert shape.stats_shape == (batch, 32, 1, sequence)


@pytest.mark.parametrize(
    ("batch", "sequence"),
    ((0, 128), (3, 128), (True, 128), (1, 0), (1, 127), (1, 129)),
)
def test_shape_rejects_nonproduction_geometry(batch: Any, sequence: Any) -> None:
    with pytest.raises(ValueError):
        Shape(batch=batch, sequence=sequence)


def test_stable_file_identity_authenticates_exact_regular_file(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "candidate.so"
    artifact.write_bytes(b"native-candidate")
    digest, byte_count = _identity(artifact)

    observed = _stable_file_identity(
        artifact,
        expected_sha256=digest,
        expected_bytes=byte_count,
        label="test artifact",
    )

    assert observed["path"] == str(artifact.resolve())
    assert observed["sha256"] == digest
    assert observed["bytes"] == byte_count


def test_stable_file_identity_rejects_mismatch_and_symlink(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "candidate.so"
    artifact.write_bytes(b"native-candidate")
    digest, byte_count = _identity(artifact)
    with pytest.raises(RuntimeError, match="identity mismatch"):
        _stable_file_identity(
            artifact,
            expected_sha256="0" * 64,
            expected_bytes=byte_count,
            label="test artifact",
        )
    link = tmp_path / "candidate-link.so"
    link.symlink_to(artifact)
    with pytest.raises(RuntimeError, match="non-symlink"):
        _stable_file_identity(
            link,
            expected_sha256=digest,
            expected_bytes=byte_count,
            label="test artifact",
        )


def test_load_authenticated_extension_accepts_exact_absolute_path(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "_tkfa4_path_candidate.py"
    module_path.write_text("MARKER = 420\n")
    digest, byte_count = _identity(module_path)
    module_name = "_tkfa4_path_candidate"
    sys.modules.pop(module_name, None)
    try:
        module, observed = validator.load_authenticated_extension(
            str(module_path),
            expected_sha256=digest,
            expected_bytes=byte_count,
            module_name=module_name,
        )
    finally:
        sys.modules.pop(module_name, None)

    assert module.MARKER == 420
    assert observed["path"] == str(module_path.resolve())
    assert module._tk_fa4_loaded_artifact_identity == observed


def test_load_authenticated_extension_accepts_exact_importable_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "_tkfa4_import_candidate"
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text("MARKER = 128\n")
    digest, byte_count = _identity(module_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    sys.modules.pop(module_name, None)
    try:
        module, observed = validator.load_authenticated_extension(
            module_name,
            expected_sha256=digest,
            expected_bytes=byte_count,
        )
    finally:
        sys.modules.pop(module_name, None)

    assert module.MARKER == 128
    assert observed["sha256"] == digest


def test_extension_metadata_and_declared_source_are_authenticated(
    tmp_path: Path,
) -> None:
    source = tmp_path / "v420_candidate.cu"
    source.write_text("// exact source\n")
    digest, byte_count = _identity(source)
    extension = _FakeExtension(_metadata(source.name))

    metadata, source_identity = require_extension_metadata(
        extension,
        expected_source_identity=CANDIDATE_SOURCE_IDENTITY,
        expected_source_sha256=digest,
        expected_source_bytes=byte_count,
        source_root=tmp_path,
    )

    assert metadata == _metadata(source.name)
    assert source_identity["path"] == str(source.resolve())
    assert source_identity["sha256"] == digest


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("head_dim", 64),
        ("threads", 256),
        ("key_tile", 64),
        ("query_tile", 64),
        ("head_ratio", 8),
        ("batch_values", [1, 2]),
        ("causal", 1),
        ("encoding_scale", 4),
        ("lstat_abi", "LSE"),
        ("caller_zeros_outputs_for_main", False),
        ("backward_out_clears_outputs", False),
        ("direct_output_entrypoint", "backward"),
        ("topology", ""),
    ),
)
def test_extension_metadata_rejects_semantic_or_identity_drift(
    tmp_path: Path,
    field: str,
    value: Any,
) -> None:
    source = tmp_path / "v420_candidate.cu"
    source.write_text("// exact source\n")
    digest, byte_count = _identity(source)
    metadata = _metadata(str(source))
    metadata[field] = value

    with pytest.raises(RuntimeError, match="direct-output ABI"):
        require_extension_metadata(
            _FakeExtension(metadata),
            expected_source_identity=CANDIDATE_SOURCE_IDENTITY,
            expected_source_sha256=digest,
            expected_source_bytes=byte_count,
            source_root=tmp_path,
        )


def test_extension_metadata_rejects_missing_receipt(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="lacks"):
        require_extension_metadata(
            object(),
            expected_source_identity=CANDIDATE_SOURCE_IDENTITY,
            expected_source_sha256="0" * 64,
            expected_source_bytes=1,
            source_root=tmp_path,
        )


def test_extension_metadata_permits_additional_diagnostic_fields(
    tmp_path: Path,
) -> None:
    source = tmp_path / "v420_candidate.cu"
    source.write_text("// exact source\n")
    digest, byte_count = _identity(source)
    metadata = _metadata(str(source))
    metadata["unreviewed_schedule"] = True

    observed, _ = require_extension_metadata(
        _FakeExtension(metadata),
        expected_source_identity=CANDIDATE_SOURCE_IDENTITY,
        expected_source_sha256=digest,
        expected_source_bytes=byte_count,
        source_root=tmp_path,
    )

    assert observed["unreviewed_schedule"] is True


def test_extension_metadata_accepts_exact_alternate_candidate_identity(
    tmp_path: Path,
) -> None:
    source = tmp_path / "v421_candidate.cu"
    source.write_text("// exact source\n")
    digest, byte_count = _identity(source)
    alternate_identity = "v421_d128_gqa_pipelined_v1"

    observed, _ = require_extension_metadata(
        _FakeExtension(
            _metadata(str(source), source_identity=alternate_identity)
        ),
        expected_source_identity=alternate_identity,
        expected_source_sha256=digest,
        expected_source_bytes=byte_count,
        source_root=tmp_path,
    )

    assert observed["source_identity"] == alternate_identity


def test_extension_metadata_rejects_source_identity_mismatch(
    tmp_path: Path,
) -> None:
    source = tmp_path / "candidate.cu"
    source.write_text("// exact source\n")
    digest, byte_count = _identity(source)

    with pytest.raises(RuntimeError, match="direct-output ABI"):
        require_extension_metadata(
            _FakeExtension(
                _metadata(str(source), source_identity="v421_actual")
            ),
            expected_source_identity="v421_expected",
            expected_source_sha256=digest,
            expected_source_bytes=byte_count,
            source_root=tmp_path,
        )


def test_direct_output_entrypoint_is_named_and_callable(tmp_path: Path) -> None:
    extension = _FakeExtension(_metadata(str(tmp_path / "candidate.cu")))
    sentinel = lambda *_args: None
    setattr(extension, DIRECT_OUTPUT_ENTRYPOINT, sentinel)

    assert (
        require_direct_output_entrypoint(extension, extension.metadata)
        is sentinel
    )
    delattr(extension, DIRECT_OUTPUT_ENTRYPOINT)
    with pytest.raises(RuntimeError, match="lacks callable"):
        require_direct_output_entrypoint(extension, extension.metadata)


def test_direct_outputs_are_caller_zero_bf16_bshd() -> None:
    shape = Shape(batch=2, sequence=128)
    outputs = DirectOutputs.allocate(shape, device="cpu")

    assert outputs.dq.shape == shape.q_shape
    assert outputs.dk.shape == shape.kv_shape
    assert outputs.dv.shape == shape.kv_shape
    assert all(tensor.dtype == torch.bfloat16 for tensor in outputs.tensors())
    assert all(int(torch.count_nonzero(tensor)) == 0 for tensor in outputs.tensors())


def test_represented_reference_is_deterministic_finite_and_production_shaped() -> None:
    shape = Shape(batch=1, sequence=128)
    state = make_represented_state(shape, device="cpu", seed=20260829)

    first = represented_e4m3_causal_reference(state)
    second = represented_e4m3_causal_reference(state)

    assert state.q.dtype == torch.float8_e4m3fn
    assert state.k.dtype == torch.float8_e4m3fn
    assert state.v.dtype == torch.float8_e4m3fn
    assert state.dout.dtype == torch.float8_e4m3fn
    assert state.lstat.shape == shape.stats_shape
    assert state.dstat.shape == shape.stats_shape
    assert state.lstat.dtype == torch.float32
    assert state.dstat.dtype == torch.float32
    assert torch.isfinite(state.lstat).all()
    assert torch.isfinite(state.dstat).all()
    for lhs, rhs, expected_shape in zip(
        first.tensors(),
        second.tensors(),
        (shape.q_shape, shape.kv_shape, shape.kv_shape),
        strict=True,
    ):
        assert lhs.shape == expected_shape
        assert torch.equal(lhs, rhs)
        assert torch.isfinite(lhs).all()


def test_launch_delegates_clear_to_exact_direct_output_abi() -> None:
    shape = Shape(batch=1, sequence=128)
    state = make_represented_state(shape, device="cpu", seed=7)
    outputs = DirectOutputs.allocate(shape, device="cpu")
    for tensor in outputs.tensors():
        tensor.fill_(3.0)
    calls: list[tuple[Any, ...]] = []

    def entrypoint(*args: Any) -> None:
        calls.append(args)
        assert all(torch.all(tensor == 3) for tensor in args[6:9])
        for tensor in args[6:9]:
            tensor.zero_()
            tensor.add_(1.0)

    launch_reset_inclusive(entrypoint, state, outputs)

    assert len(calls) == 1
    assert calls[0][:6] == (
        state.q,
        state.k,
        state.v,
        state.dout,
        state.lstat,
        state.dstat,
    )
    assert calls[0][6:9] == outputs.tensors()
    assert calls[0][9] == pytest.approx(128**-0.5)
    assert all(torch.all(tensor == 1) for tensor in outputs.tensors())


def test_exact_zero_dout_gate_is_bitwise_and_reset_inclusive() -> None:
    state = make_represented_state(
        Shape(batch=1, sequence=128),
        device="cpu",
        seed=11,
    )

    def zero_preserving_entrypoint(*args: Any) -> None:
        assert int(torch.count_nonzero(args[3].float())) == 0
        assert int(torch.count_nonzero(args[5])) == 0
        assert all(torch.all(tensor == 1) for tensor in args[6:9])
        for tensor in args[6:9]:
            tensor.zero_()

    report = check_exact_zero_dout(zero_preserving_entrypoint, state)
    assert report["passed"] is True
    assert report["exact_nonzero_counts"] == {"dq": 0, "dk": 0, "dv": 0}

    def corrupting_entrypoint(*args: Any) -> None:
        for tensor in args[6:9]:
            tensor.zero_()
        args[7].reshape(-1)[0] = 1.0

    report = check_exact_zero_dout(corrupting_entrypoint, state)
    assert report["passed"] is False
    assert report["exact_nonzero_counts"]["dk"] == 1


def test_finite_nontrivial_gate_rejects_zero_and_nonfinite() -> None:
    outputs = DirectOutputs.allocate(Shape(batch=1, sequence=128), device="cpu")
    assert _finite_and_nontrivial(outputs)["passed"] is False
    for tensor in outputs.tensors():
        tensor.fill_(1.0)
    assert _finite_and_nontrivial(outputs)["passed"] is True
    outputs.dv.reshape(-1)[0] = float("nan")
    assert _finite_and_nontrivial(outputs)["passed"] is False


def test_cpu_timing_boundary_runs_warmups_and_samples() -> None:
    calls = {"native": 0, "cute": 0}

    def native() -> None:
        calls["native"] += 1

    def cute() -> None:
        calls["cute"] += 1

    report = time_reset_inclusive(
        {"native": native, "cute": cute},
        device=torch.device("cpu"),
        warmups=2,
        samples=3,
    )

    assert calls == {"native": 5, "cute": 5}
    assert len(report["native"]["samples_us"]) == 3
    assert len(report["cute"]["samples_us"]) == 3
    assert report["native"]["median_us"] >= 0.0


def test_optional_comparator_is_not_required_and_loads_explicit_callable() -> None:
    name = "_tkfa4_test_cute_adapter"
    module = ModuleType(name)
    sentinel = lambda *_args: None
    module.direct = sentinel
    sys.modules[name] = module
    try:
        comparator, identity = validator.load_optional_comparator(
            f"{name}:direct"
        )
    finally:
        sys.modules.pop(name, None)

    assert comparator is sentinel
    assert identity == {"module": name, "callable": "direct", "file": None}


def test_optional_comparator_rejects_implicit_or_missing_callable() -> None:
    with pytest.raises(ValueError, match="module:callable"):
        validator.load_optional_comparator("implicit")


@pytest.mark.parametrize(
    "header",
    (
        "v443_d128_gqa_e4m3_b2_s4096_owner4_compact_p_reuse_"
        "production_bshd.cuh",
        "v445_d128_gqa_e4m3_b1_owner2_exact_s4096_compact_p_reuse_"
        "production_bshd.cuh",
    ),
)
def test_compact_probability_reuses_exact_published_e4m3_words(
    header: str,
) -> None:
    source = _native_source(header)

    assert "uint32_t words[kCompactProbabilityWords];" in source
    assert "compact.words[first_word] = packed_first;" in source
    assert "compact.words[second_word] = packed_second;" in source
    assert '"st.shared.v2.b32 [%0], {%1, %2};\\n"' in source
    assert "const float4 probability = expand_compact_e4m3_word(" in source
    assert source.count("compact.words[first_word]") == 2
    assert source.count("compact.words[second_word]") == 2


def test_v447_preissues_first_dp_half_after_acquire_before_prior_waits() -> None:
    source = _native_source(
        "v447_d128_gqa_e4m3_b2_s4096_owner4_dp_half0_preissue_"
        "compact_p_reuse_production_bshd.cuh"
    )
    schedule_start = source.index("wait(dp_ready, phase);")
    schedule_stop = source.index("make_ds_half_compact(\n", schedule_start)
    schedule = source[schedule_start:schedule_stop]

    assert schedule.index("tensor_after_thread_sync();") < schedule.index(
        "issue_dp_half_compact("
    )
    assert schedule.index("issue_dp_half_compact(") < schedule.index(
        "wait(dq_ready, old_phase);"
    )
    assert schedule.index("wait(dk_ready, old_phase);") < schedule.index(
        "consume_ds_half_compact("
    )
    consume_start = source.index("void consume_ds_half_compact(")
    consume_stop = source.index("void make_ds_half_compact(", consume_start)
    consume = source[consume_start:consume_stop]
    assert "tensor_load_wait();" in consume


def test_v454_freezes_selected_routes_and_route_aware_reset() -> None:
    header = _native_source(
        "v454_d128_gqa_e4m3_unified_best_route_production_bshd.cuh"
    )
    binding = _native_source(
        "v454_d128_gqa_e4m3_unified_best_route_production_bshd.cu"
    )

    assert "static_assert(b1_exact::kHeadsPerOwner == 2);" in header
    assert (
        "static_assert(b2_exact_and_fallbacks::kHeadsPerOwner == 4);"
        in header
    )
    assert "batch == 1 && sequence == kExactSequence" in header
    assert "batch == 2 && sequence == kExactSequence" in header
    assert "b1_exact::launch(" in header
    assert "b2_exact_and_fallbacks::launch(" in header

    reset = binding[
        binding.index("if (clear_outputs)") : binding.index(
            "candidate::launch(", binding.index("if (clear_outputs)")
        )
    ]
    assert reset.count("cudaMemsetAsync") == 3
    assert "!candidate::is_b2_exact_direct_route" in reset
    assert 'result["output_dtype"] = "bfloat16_additive";' in binding
    assert (
        '"B1_S4096_v445;B2_S4096_v447;B1_other_v436;B2_other_v437"'
        in binding
    )


def test_v455_splits_probability_publication_and_dv_issue_in_order() -> None:
    source = _native_source(
        "v455_d128_gqa_e4m3_b2_s4096_owner4_split_dv_compact_p_"
        "reuse_production_bshd.cuh"
    )
    assert (
        "init_semaphore(probability_half_ready[0], 0, kComputeWarps);"
        in source
    )
    assert (
        "init_semaphore(probability_half_ready[1], 0, kComputeWarps);"
        in source
    )

    producer_start = source.index(
        "compact_probability_half probability_compact[2];"
    )
    producer_stop = source.index(
        "d64::owner_aligned_fp32_half dp_first_half;", producer_start
    )
    producer = source[producer_start:producer_stop]
    assert producer.index("probability_compact[0]") < producer.index(
        "arrive(probability_half_ready[0]);"
    )
    assert producer.index("arrive(probability_half_ready[0]);") < (
        producer.index("probability_compact[1]")
    )
    assert producer.index("probability_compact[1]") < producer.index(
        "arrive(probability_half_ready[1]);"
    )
    assert producer.count("tensor_before_thread_sync();") == 2
    assert producer.count("__syncwarp();") == 2
    assert producer.count('"fence.proxy.async.shared::cta;"') == 2

    reuse_start = source.index("if (work > 0) {", producer_start - 500)
    reuse_stop = source.index("wait(score_ready, phase);", reuse_start)
    reuse = source[reuse_start:reuse_stop]
    assert reuse.index("wait(dv_ready, old_phase);") < reuse.index(
        "wait(probability_consumed, old_phase);"
    )

    issue_start = source.index("wait(probability_half_ready[0], phase);")
    issue_stop = source.index("wait(ds_ready, phase);", issue_start)
    issue = source[issue_start:issue_stop]
    assert issue.index("tensor_after_thread_sync();") < issue.index(
        "issue_gradient_ab_runtime_accumulate_half<0>("
    )
    assert issue.index(
        "issue_gradient_ab_runtime_accumulate_half<0>("
    ) < issue.index("core::issue_score_or_dp(")
    assert issue.index("core::issue_score_or_dp(") < issue.index(
        "wait(probability_half_ready[1], phase);"
    )
    assert issue.index("wait(probability_half_ready[1], phase);") < (
        issue.index("issue_gradient_ab_runtime_accumulate_half<1>(")
    )

    helper_start = source.index(
        "void issue_gradient_ab_runtime_accumulate_half("
    )
    helper_stop = source.index(
        "void publish_gradient_full_direct(", helper_start
    )
    helper = source[helper_start:helper_stop]
    assert "constexpr int kFirstChunk = Half * 2;" in helper
    assert "constexpr int kSecondChunk = kFirstChunk + 1;" in helper
    assert helper.count("tensor_commit<1>(completion);") == 1
    assert (
        "if constexpr (Half == 1) {\n"
        "        tensor_commit<1>(completion);\n"
        "    }"
        in helper
    )


def test_v458_freezes_v455_route_and_route_aware_reset() -> None:
    header = _native_source(
        "v458_d128_gqa_e4m3_unified_best_route_production_bshd.cuh"
    )
    binding = _native_source(
        "v458_d128_gqa_e4m3_unified_best_route_production_bshd.cu"
    )

    assert (
        '#include "v445_d128_gqa_e4m3_b1_owner2_exact_s4096_compact_p_'
        'reuse_production_bshd.cuh"'
        in header
    )
    assert (
        '#include "v455_d128_gqa_e4m3_b2_s4096_owner4_split_dv_compact_p_'
        'reuse_production_bshd.cuh"'
        in header
    )
    assert "static_assert(b1_exact::kHeadsPerOwner == 2);" in header
    assert (
        "static_assert(b2_exact_and_fallbacks::kHeadsPerOwner == 4);"
        in header
    )
    assert "batch == 1 && sequence == kExactSequence" in header
    assert "batch == 2 && sequence == kExactSequence" in header
    assert "b1_exact::launch(" in header
    assert "b2_exact_and_fallbacks::launch(" in header
    b1_dispatch = header.index("b1_exact::launch(")
    b1_return = header.index("return;", b1_dispatch)
    b2_dispatch = header.index("b2_exact_and_fallbacks::launch(")
    assert b1_dispatch < b1_return < b2_dispatch
    assert header.count("b2_exact_and_fallbacks::launch(") == 1

    reset = binding[
        binding.index("if (clear_outputs)") : binding.index(
            "candidate::launch(", binding.index("if (clear_outputs)")
        )
    ]
    assert reset.count("cudaMemsetAsync") == 3
    assert "!candidate::is_b2_exact_direct_route" in reset
    dq_clear = reset.index("cudaMemsetAsync(dq.data_ptr()")
    conditional = reset.index("!candidate::is_b2_exact_direct_route")
    dk_clear = reset.index("cudaMemsetAsync(dk.data_ptr()")
    dv_clear = reset.index("cudaMemsetAsync(dv.data_ptr()")
    assert dq_clear < conditional < dk_clear < dv_clear
    assert 'result["output_dtype"] = "bfloat16_additive";' in binding
    assert (
        'result["direct_output_entrypoint"] =\n'
        '        "backward_e4m3_bshd_precomputed_out";'
        in binding
    )
    assert (
        '"B1_S4096_v445;B2_S4096_v455;B1_other_v436;B2_other_v437"'
        in binding
    )


def test_v460_pipelines_first_half_quarter_dp_in_owner_order() -> None:
    source = _native_source(
        "v460_d128_gqa_e4m3_b2_s4096_owner4_quarter_dp_split_dv_"
        "compact_p_reuse_production_bshd.cuh"
    )

    issue_helper_start = source.index(
        "void issue_dp_first_half_quarter("
    )
    issue_helper_stop = source.index(
        "uint32_t make_ds_quarter_compact_word(", issue_helper_start
    )
    issue_helper = source[issue_helper_start:issue_helper_stop]
    assert "static_assert(Chunk == 0 || Chunk == 1);" in issue_helper
    assert "Chunk * (kColumnHalf / 4)" in issue_helper
    assert "tcgen05.ld.sync.aligned.16x32bx2.x16.b32" in issue_helper

    consume_helper_start = source.index(
        "void consume_ds_first_half_quarter("
    )
    consume_helper_stop = source.index(
        "void issue_gradient_ab_runtime_accumulate_half(",
        consume_helper_start,
    )
    consume_helper = source[consume_helper_start:consume_helper_stop]
    assert (
        "lane_column_base + Chunk * kChunkColumns"
        in consume_helper
    )
    assert "Chunk * kWordsPerChunk" in consume_helper
    assert (
        "lane_column_base + Chunk * kChunkColumns + 4 * first_word"
        in consume_helper
    )

    schedule_start = source.index(
        "owner_aligned_fp32_quarter dp_first_quarter;"
    )
    schedule_stop = source.index(
        "make_ds_half_compact(", schedule_start
    )
    schedule = source[schedule_start:schedule_stop]
    issue_chunk0 = schedule.index("issue_dp_first_half_quarter<0>(")
    wait_dq = schedule.index("wait(dq_ready, old_phase);")
    wait_dk = schedule.index("wait(dk_ready, old_phase);")
    wait_chunk0 = schedule.index("tensor_load_wait();", issue_chunk0)
    issue_chunk1 = schedule.index("issue_dp_first_half_quarter<1>(")
    anchor = schedule.index("tensor_before_thread_sync();", issue_chunk1)
    consume_chunk0 = schedule.index("consume_ds_first_half_quarter<0>(")
    wait_chunk1 = schedule.index("tensor_load_wait();", consume_chunk0)
    consume_chunk1 = schedule.index("consume_ds_first_half_quarter<1>(")
    assert (
        issue_chunk0
        < wait_dq
        < wait_dk
        < wait_chunk0
        < issue_chunk1
        < anchor
        < consume_chunk0
        < wait_chunk1
        < consume_chunk1
    )
    assert "__syncwarp();" not in schedule[issue_chunk1:consume_chunk0]


def test_v463_freezes_v460_route_and_route_aware_reset() -> None:
    header = _native_source(
        "v463_d128_gqa_e4m3_unified_best_route_production_bshd.cuh"
    )
    binding = _native_source(
        "v463_d128_gqa_e4m3_unified_best_route_production_bshd.cu"
    )

    assert (
        '#include "v445_d128_gqa_e4m3_b1_owner2_exact_s4096_compact_p_'
        'reuse_production_bshd.cuh"'
        in header
    )
    assert (
        '#include "v460_d128_gqa_e4m3_b2_s4096_owner4_quarter_dp_split_'
        'dv_compact_p_reuse_production_bshd.cuh"'
        in header
    )
    assert "static_assert(b1_exact::kHeadsPerOwner == 2);" in header
    assert (
        "static_assert(b2_exact_and_fallbacks::kHeadsPerOwner == 4);"
        in header
    )
    assert "batch == 1 && sequence == kExactSequence" in header
    assert "batch == 2 && sequence == kExactSequence" in header
    b1_dispatch = header.index("b1_exact::launch(")
    b1_return = header.index("return;", b1_dispatch)
    b2_dispatch = header.index("b2_exact_and_fallbacks::launch(")
    assert b1_dispatch < b1_return < b2_dispatch
    assert header.count("b2_exact_and_fallbacks::launch(") == 1

    reset = binding[
        binding.index("if (clear_outputs)") : binding.index(
            "candidate::launch(", binding.index("if (clear_outputs)")
        )
    ]
    assert reset.count("cudaMemsetAsync") == 3
    assert "!candidate::is_b2_exact_direct_route" in reset
    dq_clear = reset.index("cudaMemsetAsync(dq.data_ptr()")
    conditional = reset.index("!candidate::is_b2_exact_direct_route")
    dk_clear = reset.index("cudaMemsetAsync(dk.data_ptr()")
    dv_clear = reset.index("cudaMemsetAsync(dv.data_ptr()")
    assert dq_clear < conditional < dk_clear < dv_clear
    assert 'result["output_dtype"] = "bfloat16_additive";' in binding
    assert (
        'result["direct_output_entrypoint"] =\n'
        '        "backward_e4m3_bshd_precomputed_out";'
        in binding
    )
    assert (
        '"B1_S4096_v445;B2_S4096_v460;B1_other_v436;B2_other_v437"'
        in binding
    )


def test_v465_uses_dv_completion_as_shared_probability_reuse_gate() -> None:
    header = _native_source(
        "v465_d128_gqa_e4m3_b2_s4096_owner4_dv_gated_compact_p_"
        "reuse_production_bshd.cuh"
    )
    binding = _native_source(
        "v465_d128_gqa_e4m3_b2_s4096_owner4_dv_gated_compact_p_"
        "reuse_production_bshd.cu"
    )

    assert "probability_consumed" not in header
    split_helper_start = header.index(
        "void issue_gradient_ab_runtime_accumulate_half("
    )
    split_helper_stop = header.index(
        "void publish_gradient_full_direct(", split_helper_start
    )
    split_helper = header[split_helper_start:split_helper_stop]
    assert split_helper.count("tensor_commit<1>(completion);") == 1
    assert (
        "if constexpr (Half == 1) {\n"
        "        tensor_commit<1>(completion);\n"
        "    }"
        in split_helper
    )
    reuse_start = header.index("if (work > 0) {")
    reuse_stop = header.index("wait(score_ready, phase);", reuse_start)
    reuse_gate = header[reuse_start:reuse_stop]
    assert "wait(dv_ready, old_phase);" in reuse_gate
    assert 'result["probability_shared_reuse_gate"]' in binding
    assert (
        '"dv_ready_after_all_four_k32_dv_commands_no_probability_barrier"'
        in binding
    )


def test_v466_uses_operand_completion_as_stats_reuse_gate() -> None:
    header = _native_source(
        "v466_d128_gqa_e4m3_b2_s4096_owner4_operand_gated_stats_"
        "reuse_production_bshd.cuh"
    )
    binding = _native_source(
        "v466_d128_gqa_e4m3_b2_s4096_owner4_operand_gated_stats_"
        "reuse_production_bshd.cu"
    )

    assert "stats_consumed" not in header
    assert "semaphore stats_ready[kInputStages];" in header
    loader_reuse_start = header.index("if (work >= kInputStages) {")
    loader_reuse_stop = header.index(
        "const int query_tile", loader_reuse_start
    )
    loader_reuse = header[loader_reuse_start:loader_reuse_stop]
    operand_wait = loader_reuse.index("wait(operand_consumed[stage], old_phase);")
    acquire = loader_reuse.index("tensor_after_thread_sync();")
    assert operand_wait < acquire

    issuer_start = header.index("wait(ds_ready, phase);")
    wait_dk = header.index("wait(dk_ready, phase);", issuer_start)
    wait_dv = header.index("wait(dv_ready, phase);", wait_dk)
    issuer_acquire = header.index("tensor_after_thread_sync();", wait_dv)
    operand_arrive = header.index(
        "arrive(operand_consumed[stage]);", issuer_acquire
    )
    assert wait_dk < wait_dv < issuer_acquire < operand_arrive
    assert 'result["stats_ready_barrier"] = true;' in binding
    assert 'result["stats_consumed_barrier"] = false;' in binding


def test_v467_freezes_v466_route_and_route_aware_reset() -> None:
    header = _native_source(
        "v467_d128_gqa_e4m3_unified_best_route_production_bshd.cuh"
    )
    binding = _native_source(
        "v467_d128_gqa_e4m3_unified_best_route_production_bshd.cu"
    )

    assert (
        '#include "v445_d128_gqa_e4m3_b1_owner2_exact_s4096_compact_p_'
        'reuse_production_bshd.cuh"'
        in header
    )
    assert (
        '#include "v466_d128_gqa_e4m3_b2_s4096_owner4_operand_gated_'
        'stats_reuse_production_bshd.cuh"'
        in header
    )
    assert "static_assert(b1_exact::kHeadsPerOwner == 2);" in header
    assert (
        "static_assert(b2_exact_and_fallbacks::kHeadsPerOwner == 4);"
        in header
    )
    b1_dispatch = header.index("b1_exact::launch(")
    b1_return = header.index("return;", b1_dispatch)
    b2_dispatch = header.index("b2_exact_and_fallbacks::launch(")
    assert b1_dispatch < b1_return < b2_dispatch
    assert header.count("b2_exact_and_fallbacks::launch(") == 1
    assert "batch == 1 && sequence == kExactSequence" in header
    assert "batch == 2 && sequence == kExactSequence" in header

    reset = binding[
        binding.index("if (clear_outputs)") : binding.index(
            "candidate::launch(", binding.index("if (clear_outputs)")
        )
    ]
    assert reset.count("cudaMemsetAsync") == 3
    assert "!candidate::is_b2_exact_direct_route" in reset
    dq_clear = reset.index("cudaMemsetAsync(dq.data_ptr()")
    conditional = reset.index("!candidate::is_b2_exact_direct_route")
    dk_clear = reset.index("cudaMemsetAsync(dk.data_ptr()")
    dv_clear = reset.index("cudaMemsetAsync(dv.data_ptr()")
    assert dq_clear < conditional < dk_clear < dv_clear
    assert (
        '"B1_S4096_v445;B2_S4096_v466;B1_other_v436;B2_other_v437"'
        in binding
    )
    assert 'result["b2_s4096_stats_ready_barrier"] = true;' in binding
    assert 'result["b2_s4096_stats_consumed_barrier"] = false;' in binding


def test_v468_uses_later_dk_commit_for_operand_stage_reuse() -> None:
    header = _native_source(
        "v468_d128_gqa_e4m3_b2_s4096_owner4_dk_commit_gated_operand_"
        "reuse_production_bshd.cuh"
    )
    binding = _native_source(
        "v468_d128_gqa_e4m3_b2_s4096_owner4_dk_commit_gated_operand_"
        "reuse_production_bshd.cu"
    )

    issuer_start = header.index("wait(ds_ready, phase);")
    issuer_stop = header.index(
        "arrive(operand_consumed[stage]);", issuer_start
    )
    issuer = header[issuer_start:issuer_stop]
    assert "wait(dk_ready, phase);" in issuer
    assert "wait(dv_ready, phase);" not in issuer
    assert "tensor_after_thread_sync();" in issuer
    assert "wait(dv_ready, old_phase);" in header
    assert "wait(dv_ready, last_phase);" in header
    assert 'result["issuer_explicit_dv_wait"] = false;' in binding
    assert 'result["dv_ready_barrier_retained"] = true;' in binding


def test_v469_uses_later_dk_commit_for_shared_ds_reuse() -> None:
    header = _native_source(
        "v469_d128_gqa_e4m3_b2_s4096_owner4_dk_commit_gated_ds_"
        "reuse_production_bshd.cuh"
    )
    binding = _native_source(
        "v469_d128_gqa_e4m3_b2_s4096_owner4_dk_commit_gated_ds_"
        "reuse_production_bshd.cu"
    )

    schedule_start = header.index(
        "owner_aligned_fp32_quarter dp_first_quarter;"
    )
    schedule_stop = header.index(
        "consume_ds_first_half_quarter<0>(", schedule_start
    )
    schedule = header[schedule_start:schedule_stop]
    assert "wait(dq_ready, old_phase);" not in schedule
    assert "wait(dk_ready, old_phase);" in schedule
    assert "wait(dq_ready, phase);" in header
    assert "wait(dq_drained" in header
    assert 'result["compute_old_dq_ready_wait"] = false;' in binding
    assert 'result["dq_ready_barrier_retained"] = true;' in binding
    assert 'result["dq_ready_reducer_wait_retained"] = true;' in binding
    assert 'result["dq_drained_reuse_edges_retained"] = true;' in binding
    assert "previous_dq_dk_waits" not in binding
    assert "previous_dk_wait_only" in binding


def test_v470_freezes_v469_route_and_route_aware_reset() -> None:
    header = _native_source(
        "v470_d128_gqa_e4m3_unified_best_route_production_bshd.cuh"
    )
    binding = _native_source(
        "v470_d128_gqa_e4m3_unified_best_route_production_bshd.cu"
    )

    assert (
        '#include "v445_d128_gqa_e4m3_b1_owner2_exact_s4096_compact_p_'
        'reuse_production_bshd.cuh"'
        in header
    )
    assert (
        '#include "v469_d128_gqa_e4m3_b2_s4096_owner4_dk_commit_gated_'
        'ds_reuse_production_bshd.cuh"'
        in header
    )
    assert "static_assert(b1_exact::kHeadsPerOwner == 2);" in header
    assert (
        "static_assert(b2_exact_and_fallbacks::kHeadsPerOwner == 4);"
        in header
    )
    b1_dispatch = header.index("b1_exact::launch(")
    b1_return = header.index("return;", b1_dispatch)
    b2_dispatch = header.index("b2_exact_and_fallbacks::launch(")
    assert b1_dispatch < b1_return < b2_dispatch
    assert header.count("b2_exact_and_fallbacks::launch(") == 1

    reset = binding[
        binding.index("if (clear_outputs)") : binding.index(
            "candidate::launch(", binding.index("if (clear_outputs)")
        )
    ]
    assert reset.count("cudaMemsetAsync") == 3
    assert "!candidate::is_b2_exact_direct_route" in reset
    assert (
        '"B1_S4096_v445;B2_S4096_v469;B1_other_v436;B2_other_v437"'
        in binding
    )
    assert (
        'result["b2_s4096_compute_old_dq_ready_wait"] = false;'
        in binding
    )
    assert (
        'result["b2_s4096_issuer_explicit_dv_wait"] = false;'
        in binding
    )


def test_v472_issues_head_score_before_aliased_dp_reuse() -> None:
    header = _native_source(
        "v472_d128_gqa_e4m3_b2_s4096_owner4_head_score_before_dq_"
        "drain_production_bshd.cuh"
    )
    binding = _native_source(
        "v472_d128_gqa_e4m3_b2_s4096_owner4_head_score_before_dq_"
        "drain_production_bshd.cu"
    )

    issuer_start = header.index(
        "physical_warp == kTensorIssueWarp && lane == 0"
    )
    head_start = header.index(
        "for (int local_head = 0; local_head < kHeadsPerOwner; ++local_head)",
        issuer_start,
    )
    inner_start = header.index("for (", head_start + 1)
    boundary = header[head_start:inner_start]
    score_wait = boundary.index("wait(score_consumed, previous_phase);")
    query_wait = boundary.index("query_ready[first_stage]")
    score_issue = boundary.index(
        "core::issue_score_or_dp(\n"
        "                score_tmem,"
    )
    dq_wait = boundary.index("wait(dq_drained, previous_phase);")
    dp_issue = boundary.index(
        "core::issue_score_or_dp(\n"
        "                dp_tmem,"
    )
    assert score_wait < query_wait < score_issue < dq_wait < dp_issue
    assert boundary.count("tensor_after_thread_sync();") == 2
    assert boundary.count("wait(dq_drained, previous_phase);") == 1
    assert (
        'result["head_boundary_score_schedule"] ='
        in binding
    )
    assert (
        'result["head_boundary_dp_reuse_gate"] ='
        in binding
    )


def test_v473_freezes_v472_route_and_route_aware_reset() -> None:
    header = _native_source(
        "v473_d128_gqa_e4m3_unified_best_route_production_bshd.cuh"
    )
    binding = _native_source(
        "v473_d128_gqa_e4m3_unified_best_route_production_bshd.cu"
    )

    assert (
        '#include "v445_d128_gqa_e4m3_b1_owner2_exact_s4096_compact_p_'
        'reuse_production_bshd.cuh"'
        in header
    )
    assert (
        '#include "v472_d128_gqa_e4m3_b2_s4096_owner4_head_score_before_'
        'dq_drain_production_bshd.cuh"'
        in header
    )
    assert "static_assert(b1_exact::kHeadsPerOwner == 2);" in header
    assert (
        "static_assert(b2_exact_and_fallbacks::kHeadsPerOwner == 4);"
        in header
    )
    b1_dispatch = header.index("b1_exact::launch(")
    b1_return = header.index("return;", b1_dispatch)
    b2_dispatch = header.index("b2_exact_and_fallbacks::launch(")
    assert b1_dispatch < b1_return < b2_dispatch
    assert header.count("b2_exact_and_fallbacks::launch(") == 1

    reset = binding[
        binding.index("if (clear_outputs)") : binding.index(
            "candidate::launch(", binding.index("if (clear_outputs)")
        )
    ]
    assert reset.count("cudaMemsetAsync") == 3
    assert "!candidate::is_b2_exact_direct_route" in reset
    assert (
        '"B1_S4096_v445;B2_S4096_v472;B1_other_v436;B2_other_v437"'
        in binding
    )
    assert (
        'result["b2_s4096_head_boundary_score_schedule"] ='
        in binding
    )
    assert (
        'result["b2_s4096_head_boundary_dp_reuse_gate"] ='
        in binding
    )


def test_v477_v478_v480_record_progressive_dq_tmem_release() -> None:
    bindings = {
        "v477": _native_source(
            "v477_d128_gqa_e4m3_b2_s4096_owner4_peeled_split_dq_"
            "tmem_release_production_bshd.cu"
        ),
        "v478": _native_source(
            "v478_d128_gqa_e4m3_b2_s4096_owner4_two_chunk_dq_"
            "tmem_release_production_bshd.cu"
        ),
        "v480": _native_source(
            "v480_d128_gqa_e4m3_b2_s4096_owner4_deferred_packed_d1_"
            "dq_release_production_bshd.cu"
        ),
    }

    assert (
        '"peeled_final_depth_chunk_no_in_loop_final_chunk_branch"'
        in bindings["v477"]
    )
    assert (
        '"paired_final_two_depth_chunks_one_tensor_load_wait"'
        in bindings["v478"]
    )
    assert (
        '"deferred_packed_d1_plus_paired_final_two_depth_chunks"'
        in bindings["v480"]
    )
    for binding in bindings.values():
        assert 'result["dq_tmem_release_barrier"] = "dq_tmem_drained";' in binding
        assert (
            'result["dq_shared_publication_ready_barrier"] = "dq_drained";'
            in binding
        )
        assert (
            '"previous_dq_tmem_drained_before_aliased_dp_issue"'
            in binding
        )


def test_v482_releases_dq_tmem_before_all_shared_publication() -> None:
    header = _native_source(
        "v482_d128_gqa_e4m3_b2_s4096_owner4_deferred_packed_d0_d1_"
        "dq_release_production_bshd.cuh"
    )
    binding = _native_source(
        "v482_d128_gqa_e4m3_b2_s4096_owner4_deferred_packed_d0_d1_"
        "dq_release_production_bshd.cu"
    )

    drain_start = header.index(
        "void drain_dq_full_owner_x32_split_release("
    )
    drain_stop = header.index("__global__ __launch_bounds__", drain_start)
    drain = header[drain_start:drain_stop]
    assert (
        "uint32_t deferred_packed_d0_values[prior::kDepthChunk / 2];"
        in drain
    )
    assert (
        "uint32_t deferred_packed_d1_values[prior::kDepthChunk / 2];"
        in drain
    )
    assert "uint32_t final_values[prior::kDepthChunk];" in drain

    paired_load = drain.index(
        "source_row + kPairedDepthChunk * prior::kDepthChunk"
    )
    final_load = drain.index(
        "source_row + kFinalDepthChunk * prior::kDepthChunk",
        paired_load,
    )
    final_wait = drain.index("tensor_load_wait();", final_load)
    tmem_arrival = drain.index("arrive(tmem_drained);", final_wait)
    first_shared_store = drain.index('"st.shared.v4.b32', tmem_arrival)
    proxy_fence = drain.rindex("fence.proxy.async.shared::cta;")
    shared_arrival = drain.index("arrive(shared_ready);", proxy_fence)
    assert paired_load < final_load < final_wait < tmem_arrival
    assert tmem_arrival < first_shared_store < proxy_fence < shared_arrival
    assert drain.count('"st.shared.v4.b32') == 2
    assert drain.count("x32::store_bf16_x8(") == 2

    assert "init_semaphore(dq_tmem_drained, 0, kReduceWarps);" in header
    assert "init_semaphore(dq_drained, 0, kReduceWarps);" in header
    assert (
        '"deferred_packed_d0_d1_plus_paired_final_two_depth_chunks"'
        in binding
    )
    assert "pybind11::make_tuple(0, 1)" in binding
    assert (
        'result["dq_deferred_packed_words_per_lane"] =\n'
        "        candidate::prior::kDepthChunk;"
        in binding
    )


def test_v483_freezes_v482_route_and_route_aware_reset() -> None:
    header = _native_source(
        "v483_d128_gqa_e4m3_unified_best_route_production_bshd.cuh"
    )
    binding = _native_source(
        "v483_d128_gqa_e4m3_unified_best_route_production_bshd.cu"
    )

    assert (
        '#include "v445_d128_gqa_e4m3_b1_owner2_exact_s4096_compact_p_'
        'reuse_production_bshd.cuh"'
        in header
    )
    assert (
        '#include "v482_d128_gqa_e4m3_b2_s4096_owner4_deferred_packed_'
        'd0_d1_dq_release_production_bshd.cuh"'
        in header
    )
    assert "static_assert(b1_exact::kHeadsPerOwner == 2);" in header
    assert (
        "static_assert(b2_exact_and_fallbacks::kHeadsPerOwner == 4);"
        in header
    )
    b1_dispatch = header.index("b1_exact::launch(")
    b1_return = header.index("return;", b1_dispatch)
    b2_dispatch = header.index("b2_exact_and_fallbacks::launch(")
    assert b1_dispatch < b1_return < b2_dispatch
    assert header.count("b2_exact_and_fallbacks::launch(") == 1

    reset_start = binding.index("if (clear_outputs)")
    reset = binding[
        reset_start : binding.index("candidate::launch(", reset_start)
    ]
    assert reset.count("cudaMemsetAsync") == 3
    assert "!candidate::is_b2_exact_direct_route" in reset
    assert (
        '"B1_S4096_v445;B2_S4096_v482;B1_other_v436;B2_other_v437"'
        in binding
    )
    assert (
        'result["b2_s4096_dq_tmem_release_barrier"] = "dq_tmem_drained";'
        in binding
    )
    assert "pybind11::make_tuple(0, 1)" in binding
    assert (
        '"previous_dq_tmem_drained_before_aliased_dp_issue"'
        in binding
    )
