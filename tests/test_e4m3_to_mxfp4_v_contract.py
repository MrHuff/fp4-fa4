from __future__ import annotations

import importlib.util
import json
import random
import subprocess
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "tk_fa4" / "lowp_fa4_bwd" / "e4m3_to_mxfp4_v.cuh"
CUDA = ROOT / "tk_fa4" / "lowp_fa4_bwd" / "lowp_fa4_bwd.cu"
MAKEFILE = ROOT / "tk_fa4" / "lowp_fa4_bwd" / "Makefile"
SCRIPT = (
    ROOT / "tk_fa4" / "lowp_fa4_bwd" / "benchmark_e4m3_to_mxfp4_v.py"
)
SPEC = importlib.util.spec_from_file_location(
    "benchmark_e4m3_to_mxfp4_v",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


def test_warp_word_permutation_is_exact_causal_quarter_order() -> None:
    visited = []
    for quarter in range(4):
        output_quarter = []
        for lane in range(32):
            source_sequence = 4 * lane + quarter
            output_sequence = quarter * 32 + lane
            output_quarter.append((output_sequence, source_sequence))
            visited.append(source_sequence)
        assert [source for _, source in output_quarter] == list(
            range(quarter, 128, 4)
        )
    assert sorted(visited) == list(range(128))


def test_scale_page_formula_covers_only_the_d64_addressable_bytes() -> None:
    offsets = {
        (depth & 31) * 16 + (depth >> 5) * 4 + quarter
        for depth in range(64)
        for quarter in range(4)
    }
    assert len(offsets) == 256
    assert min(offsets) == 0
    assert max(offsets) == 503
    assert all(offset < 512 for offset in offsets)


def test_kernel_is_register_only_and_compensates_e4m3_x4_scale() -> None:
    source = HEADER.read_text()
    assert "__shared__" not in source
    assert "reinterpret_cast<const uint32_t *>(" in source
    assert "input + input_base" in source
    assert "payload[payload_base + 48 + output_byte]" in source
    assert "scale_row * 16 + scale_group * 4" in source
    assert "exponent + 118u" in source
    assert "logical MX scale" in source
    assert "same E2M1 codes" not in source
    assert "kWarpsPerBlock = 8" in source
    assert "kThreadsPerBlock = kWarpsPerBlock * 32" in source


def test_packed_even_lanes_use_native_half2_conversion() -> None:
    source = HEADER.read_text()
    assert "const uint32_t next_word = __shfl_down_sync(" in source
    assert "if ((lane & 1) == 0)" in source
    assert "quantize_four_e4m3_pairs_to_mxfp4(" in source
    assert source.count("cvt.rn.f16x2.e4m3x2") == 4
    assert source.count("mul.rn.f16x2") == 4
    assert source.count("cvt.rn.satfinite.e2m1x2.f16x2") == 4
    assert "decode_fp8_pair_to_float2" not in source
    assert "make_float2" not in source


def test_integer_scale_selector_matches_prior_bf16_path() -> None:
    subnormal_logical = (0, 116, 117, 118, 118, 119, 119, 119)

    def prior_path(code: int) -> int:
        if code == 0x7F:
            return 0xFF
        exponent = code >> 3
        mantissa = code & 7
        if exponent:
            bf16_exponent = exponent + 120
            bf16_mantissa = mantissa << 4
        elif mantissa:
            highest_bit = mantissa.bit_length() - 1
            bf16_exponent = 118 + highest_bit
            bf16_mantissa = (
                (mantissa - (1 << highest_bit)) << (7 - highest_bit)
            )
        else:
            return 0
        physical_e8m0 = bf16_exponent + (bf16_mantissa >= 0x1A)
        return physical_e8m0 - 2

    def integer_path(code: int) -> int:
        if code == 0x7F:
            return 0xFF
        exponent = code >> 3
        mantissa = code & 7
        if exponent:
            return exponent + 118 + (mantissa >= 2)
        return subnormal_logical[mantissa]

    assert [integer_path(code) for code in range(0x80)] == [
        prior_path(code) for code in range(0x80)
    ]
    source = HEADER.read_text()
    assert "0x7777777676757400ull" in source
    assert "exponent + 118u" in source
    assert "mantissa >= 2u" in source
    assert "__float2bfloat16_rn" not in source


def test_kernel_uses_division_free_3d_cta_mapping() -> None:
    source = HEADER.read_text()
    assert "const dim3 grid(" in source
    assert "static_cast<unsigned int>(sequence / kSequenceTile)" in source
    assert "static_cast<unsigned int>(heads)" in source
    assert "static_cast<unsigned int>(batch)" in source
    assert "const int sequence_tile = static_cast<int>(blockIdx.x)" in source
    assert "const int head = static_cast<int>(blockIdx.y)" in source
    assert "const int batch_index = static_cast<int>(blockIdx.z)" in source
    assert "int depth = warp_in_block" in source
    assert "depth += kWarpsPerBlock" in source
    assert "task_count" not in source
    assert "remaining" not in source


def test_packed_warp_max_matches_four_independent_finite_reductions() -> None:
    generator = random.Random(20260826)
    for _ in range(100):
        words = []
        for _lane in range(32):
            word = 0
            for quarter in range(4):
                # 0x7f is E4M3FN NaN; finite magnitudes end at 0x7e.
                magnitude = generator.randrange(0x7F)
                sign = generator.randrange(2) << 7
                word |= (sign | magnitude) << (quarter * 8)
            words.append(word)
        packed_max = 0
        for quarter in range(4):
            expected = max(
                (word >> (quarter * 8)) & 0x7F for word in words
            )
            packed_max |= expected << (quarter * 8)
        reduced = 0
        for quarter in range(4):
            reduced |= max(
                ((word & 0x7F7F7F7F) >> (quarter * 8)) & 0xFF
                for word in words
            ) << (quarter * 8)
        assert reduced == packed_max

    source = HEADER.read_text()
    assert "word & kE4m3MagnitudeMask" in source
    assert "value = __vmaxu4(" in source
    assert source.count("__shfl_xor_sync(") == 1


def test_scale_quarters_are_packed_in_little_endian_store_order() -> None:
    scales = (0x71, 0x82, 0x93, 0xA4)
    packed = sum(scale << (quarter * 8) for quarter, scale in enumerate(scales))
    assert tuple(packed.to_bytes(4, "little")) == scales
    source = HEADER.read_text()
    assert "logical_scale_word |= static_cast<uint32_t>(logical_e8m0)" in source
    assert "*reinterpret_cast<uint32_t *>(" in source
    assert "scale_row * 16 + scale_group * 4" in source


def test_nan_groups_are_canonicalized_without_changing_finite_path() -> None:
    source = HEADER.read_text()
    assert "kE4m3NanMagnitude = 0x7fu" in source
    assert "kE8m0Nan = 0xffu" in source
    assert "physical_amax_code == kE4m3NanMagnitude" in source
    assert "pair0 = scale0 == kE8m0Nan ? 0 : pair0" in source
    assert "pair3 = scale3 == kE8m0Nan ? 0 : pair3" in source
    assert "This does not change any finite-byte result" in source
    benchmark = SCRIPT.read_text()
    assert "def _nan_correctness_preflight(" in benchmark
    assert "sentinel_offsets == affected_offsets" in benchmark
    assert "zero_payload_groups == set(affected_groups)" in benchmark
    assert '"nan_policy": nan_policy' in benchmark
    assert '"passed": finite_passed and nan_policy["passed"]' in benchmark


def test_extension_exports_allocating_and_caller_owned_opt_in_apis() -> None:
    source = CUDA.read_text()
    symbol = "convert_e4m3_x4_v_bhds_to_causal_mxfp4"
    assert f"std::vector<at::Tensor> {symbol}(" in source
    assert f"void {symbol}_out(" in source
    assert f'        "{symbol}",' in source
    assert f'        "{symbol}_out",' in source
    assert "input.size(2) == tkfa4_e4m3_to_mxfp4_v::kHeadDepth" in source
    assert "input.size(3) % tkfa4_e4m3_to_mxfp4_v::kSequenceTile == 0" in source
    assert "at::kFloat4_e2m1fn_x2" in source
    assert "{batch, sequence / 128, heads," in source
    assert "e4m3_to_mxfp4_v.cuh" in MAKEFILE.read_text()


def test_out_contract_requires_aligned_disjoint_buffers_and_grid_bounds() -> None:
    source = CUDA.read_text()
    assert "constexpr int64_t kMaxCudaGridYz = 65535" in source
    assert "kRequiredAlignment = alignof(uint32_t)" in source
    assert "is_aligned(input)" in source
    assert "is_aligned(*payload) && is_aligned(*scales)" in source
    assert "ranges_overlap(input, *payload)" in source
    assert "ranges_overlap(input, *scales)" in source
    assert "ranges_overlap(*payload, *scales)" in source
    assert "disjoint byte ranges" in source


def test_benchmark_uses_the_matched_factorial_premium() -> None:
    args = BENCHMARK._parse_args([])
    baseline = BENCHMARK._load_projection_baseline(
        BENCHMARK.DEFAULT_FACTORIAL,
        BENCHMARK._invocation_shape(args),
    )
    assert baseline["exact_fp8_projection_us"] == 848.92724
    assert baseline["direct_mx_projection_us"] == 886.552998
    assert baseline["direct_mx_premium_us"] == 37.62575800000002
    assert baseline["factorial_schema"] == BENCHMARK.FACTORIAL_SCHEMA
    assert baseline["factorial_shape"] == BENCHMARK.FACTORIAL_SHAPE
    baseline_identity = BENCHMARK._file_identity(BENCHMARK.DEFAULT_FACTORIAL)
    plan = BENCHMARK._plan(args, baseline, baseline_identity)
    assert plan["shape"] == {
        "batch": 16,
        "heads": 8,
        "depth": 64,
        "sequence": 4096,
    }
    assert plan["neutrality_rule"] == (
        "converter median <= direct MX projection premium"
    )
    assert plan["source_contract"] == "contiguous E4M3(x4) [B,H,64,S]"
    assert plan["protocol"] == {
        "seed": 20260826,
        "warmup": 20,
        "iterations": 200,
        "require_neutral": False,
        "process_check_interval_iterations": 25,
    }
    assert plan["baseline"]["identity"] == baseline_identity
    applicability = plan["baseline"]["applicability"]
    assert applicability["applies_to_this_invocation"] is True
    assert applicability["neutrality_gate_enforced"] is False
    assert applicability["invocation_shape"] == plan["shape"]
    assert "not an enforced exit-status gate" in applicability["interpretation"]
    source = SCRIPT.read_text()
    assert "torch.cuda.device_count() != 1" in source
    assert "capability != [10, 0]" in source
    assert "torch.cuda.list_gpu_processes(0)" in source
    assert "foreign compute PIDs" in source
    assert "def _parse_gpu_process_report(" in source
    assert "after_timed_chunk_" in source
    assert '"check_count"' in source
    assert '"foreign_process_ids"' in source
    assert '"extension_before": extension_identity_before' in source
    assert '"extension_after": _file_identity(extension_path)' in source
    assert '"sources_before": source_identities_before' in source
    assert '"sources_after": {' in source
    assert '"wrapper_source": _file_identity(' in source
    assert '"projection_epilogue": _file_identity(' in source
    assert '"interface_module_before_timing"' in source
    assert '"interface_module_after_timing"' in source
    assert '"interface_source": _file_identity(' in source
    assert "benchmark source artifact changed during measurement" in source
    assert "refusing to overwrite" in source
    assert "os.O_EXCL" in source
    assert "output_file.flush()" in source
    assert "os.fsync(output_file.fileno())" in source


def test_factorial_schema_shape_and_invocation_are_fail_closed(
    tmp_path: Path,
) -> None:
    args = BENCHMARK._parse_args([])
    invocation_shape = BENCHMARK._invocation_shape(args)
    document = json.loads(BENCHMARK.DEFAULT_FACTORIAL.read_text())

    wrong_schema = tmp_path / "wrong_schema.json"
    wrong_schema.write_text(json.dumps({**document, "schema": "wrong"}))
    with pytest.raises(ValueError, match="factorial schema mismatch"):
        BENCHMARK._load_projection_baseline(wrong_schema, invocation_shape)

    wrong_shape = tmp_path / "wrong_shape.json"
    wrong_shape.write_text(
        json.dumps(
            {
                **document,
                "shape": {**document["shape"], "kv_heads": 4},
            }
        )
    )
    with pytest.raises(ValueError, match="factorial shape mismatch"):
        BENCHMARK._load_projection_baseline(wrong_shape, invocation_shape)

    wrong_shape_type = tmp_path / "wrong_shape_type.json"
    wrong_shape_type.write_text(
        json.dumps(
            {
                **document,
                "shape": {**document["shape"], "causal": 1},
            }
        )
    )
    with pytest.raises(ValueError, match="factorial shape mismatch"):
        BENCHMARK._load_projection_baseline(
            wrong_shape_type,
            invocation_shape,
        )

    with pytest.raises(
        ValueError,
        match="converter invocation does not match",
    ):
        BENCHMARK._load_projection_baseline(
            BENCHMARK.DEFAULT_FACTORIAL,
            {**invocation_shape, "batch": 8},
        )


def test_gpu_process_inventory_parser_is_strict_and_exclusive() -> None:
    own_pid = BENCHMARK.os.getpid()
    valid = (
        "GPU:0\n"
        f"process {own_pid:>10d} uses {123.0:>12.3f} MB GPU memory"
    )
    assert BENCHMARK._parse_gpu_process_report(valid) == [own_pid]
    assert BENCHMARK._parse_gpu_process_report(
        "GPU:0\nno processes are running"
    ) == []
    for malformed in (
        "",
        "pynvml module not found, please install pynvml",
        "GPU:0",
        "GPU:0\nunknown process state",
        "GPU:0\nno processes are running\ntrailing data",
        (
            "GPU:0\n"
            "process         17 uses        1.000 MB GPU memory\n"
            "process         17 uses        1.000 MB GPU memory"
        ),
    ):
        with pytest.raises(RuntimeError):
            BENCHMARK._parse_gpu_process_report(malformed)

    class FakeCuda:
        @staticmethod
        def list_gpu_processes(_device: int) -> str:
            return "GPU:0\nprocess         17 uses        1.000 MB GPU memory"

    with pytest.raises(RuntimeError, match="foreign compute PIDs"):
        BENCHMARK._require_exclusive_visible_gpu(
            types.SimpleNamespace(cuda=FakeCuda())
        )

    class UnavailableCuda:
        @staticmethod
        def list_gpu_processes(_device: int) -> str:
            raise OSError("NVML failed")

    with pytest.raises(RuntimeError, match="inventory is unavailable"):
        BENCHMARK._require_exclusive_visible_gpu(
            types.SimpleNamespace(cuda=UnavailableCuda())
        )


def test_measurement_checks_processes_between_bounded_event_chunks() -> None:
    call_log: list[str] = []

    class FakeEvent:
        def __init__(self, *, enable_timing: bool) -> None:
            assert enable_timing is True

        def record(self) -> None:
            call_log.append("event")

        @staticmethod
        def elapsed_time(_other: object) -> float:
            return 0.010

    class FakeCuda:
        Event = FakeEvent

        @staticmethod
        def synchronize() -> None:
            call_log.append("synchronize")

        @staticmethod
        def list_gpu_processes(_device: int) -> str:
            call_log.append("process_check")
            return "GPU:0\nno processes are running"

    def invoke(*_operands: object) -> None:
        call_log.append("invoke")

    timing, checks = BENCHMARK._measure(
        types.SimpleNamespace(cuda=FakeCuda()),
        invoke,
        object(),
        object(),
        object(),
        warmup=2,
        iterations=51,
    )
    assert timing["median_us"] == 10.0
    assert [check["phase"] for check in checks] == [
        "after_warmup",
        "after_timed_chunk_0",
        "after_timed_chunk_1",
        "after_timed_chunk_2",
    ]
    assert [check.get("iteration_count") for check in checks] == [
        None,
        25,
        25,
        1,
    ]
    assert call_log.count("process_check") == 4
    assert call_log.count("invoke") == 53


def test_loaded_interface_must_match_expected_worktree_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = BENCHMARK._file_identity(ROOT / "tk_fa4" / "interface.py")
    monkeypatch.delitem(sys.modules, "tk_fa4.interface", raising=False)
    assert BENCHMARK._authenticate_loaded_interface(expected) == {
        "imported": False,
        "expected": expected,
    }

    monkeypatch.setitem(
        sys.modules,
        "tk_fa4.interface",
        types.SimpleNamespace(__file__=expected["path"]),
    )
    authenticated = BENCHMARK._authenticate_loaded_interface(expected)
    assert authenticated["imported"] is True
    assert authenticated["loaded"] == expected

    wrong = tmp_path / "interface.py"
    wrong.write_text("# wrong worktree\n")
    monkeypatch.setitem(
        sys.modules,
        "tk_fa4.interface",
        types.SimpleNamespace(__file__=str(wrong)),
    )
    with pytest.raises(RuntimeError, match="expected worktree source"):
        BENCHMARK._authenticate_loaded_interface(expected)


def test_e2m1_reference_uses_rne_tie_breaking() -> None:
    expected = {
        0.25: 0,
        0.75: 2,
        1.25: 2,
        1.75: 4,
        2.50: 4,
        3.50: 6,
        5.00: 6,
    }
    for value, code in expected.items():
        assert BENCHMARK._e2m1_code(value) == code
        assert BENCHMARK._e2m1_code(-value) == (code | 8)


def test_dry_run_does_not_import_torch_or_touch_cuda(tmp_path: Path) -> None:
    output = tmp_path / "never-created" / "result.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-S",
            str(SCRIPT),
            "--output",
            str(output),
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    plan = json.loads(completed.stdout)
    assert plan["dry_run"] is True
    assert plan["baseline"]["direct_mx_premium_us"] > 37.6
    assert not output.exists()
    assert not output.parent.exists()
