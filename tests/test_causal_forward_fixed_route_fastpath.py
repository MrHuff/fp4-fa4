from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from tk_fa4.lowp_fa4_bwd.validate_causal_gqa_fp8pv_batch import (
    AUTHENTICATED_BATCHES,
    CANONICAL_FORWARD_ARTIFACTS,
    CANONICAL_PROJECTION_ARTIFACT,
    SEQUENTIAL_RELATIVE_L2_LIMIT,
    _authenticate_projection_artifact,
    _authenticate_regular_artifact,
    _require_canonical_forward_identity,
    _require_exact_forward_topology,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FORWARD_ROOT = REPO_ROOT / "tk_fa4" / "fp4_fa4_fwd"
LOWP_ROOT = REPO_ROOT / "tk_fa4" / "lowp_fa4_bwd"

HOST_CASES = (
    ("hao_direct_host.inc", "hao_direct_fp8pv_initialized_device"),
    ("hao_direct_fp4pv_host.inc", "hao_direct_fp4pv_initialized_device"),
)
DESCRIPTOR_CACHE_CASES = (
    (
        "hao_direct_host.inc",
        "HAO_DIRECT_FP8PV_DESCRIPTOR_CACHE_CAPACITY",
        9,
    ),
    (
        "hao_direct_fp4pv_host.inc",
        "HAO_DIRECT_FP4PV_DESCRIPTOR_CACHE_CAPACITY",
        10,
    ),
)
MAKEFILES = (
    "Makefile.hao_direct",
    "Makefile.hao_direct_fp4pv",
)


def _exact_fp8_topology(batch: int = 2, valid: int = 0) -> dict[str, object]:
    return {
        "batch": batch,
        "seqlen": 4096,
        "heads": 32,
        "kv_heads": 8,
        "dqk": 64,
        "dvo": 64,
        "causal": True,
        "causal_interleaved_kv": False,
        "qk_format": "nvfp4_e4m3_block16",
        "pv_format": "e4m3_fp8",
        "shiftless_fp8_mode": 0,
        "route": "real_fwd_tk_hao_direct_causal_gqa_nvfp4_fp8pv",
        "schema": "tk_hao_direct_pipeline_v1",
        "fixed_route_fastpath": True,
        "fixed_p_ceiling": False,
        "score_pack_ceiling": False,
        "valid": valid,
    }


@pytest.mark.parametrize(("filename", "cache_name"), HOST_CASES)
def test_fixed_route_host_fastpath_is_opt_in_and_device_scoped(
    filename: str,
    cache_name: str,
) -> None:
    source = (FORWARD_ROOT / filename).read_text()

    assert (
        "#ifndef TK_HAO_DIRECT_FIXED_ROUTE_FASTPATH\n"
        "#define TK_HAO_DIRECT_FIXED_ROUTE_FASTPATH 0\n"
        "#endif"
    ) in source
    assert f"static thread_local int\n    {cache_name} = -1;" in source
    assert "const auto stream = at::cuda::getCurrentCUDAStream();" in source
    assert "const int device = static_cast<int>(stream.device_index());" in source
    assert f"if ({cache_name} != device)" in source
    assert f"{cache_name} = device;" in source


@pytest.mark.parametrize(("filename", "_cache_name"), HOST_CASES)
def test_fixed_route_compile_elides_route_guard_and_hoists_attributes(
    filename: str,
    _cache_name: str,
) -> None:
    source = (FORWARD_ROOT / filename).read_text()
    route_lookup = source.index(
        'std::getenv("TK_FA4_FP4PV_FWD_CONFIG")'
    )
    guard_open = source.rfind(
        "#if !TK_HAO_DIRECT_FIXED_ROUTE_FASTPATH",
        0,
        route_lookup,
    )
    guard_close = source.index("#endif", route_lookup)

    assert guard_open >= 0
    assert guard_open < route_lookup < guard_close
    assert source.count("cudaFuncSetAttribute(") == 1
    assert "const auto initialize_host_invariants = [&]" in source
    assert "#if TK_HAO_DIRECT_FIXED_ROUTE_FASTPATH" in source
    assert "initialize_host_invariants();" in source


@pytest.mark.parametrize(("filename", "_cache_name"), HOST_CASES)
def test_topology_reports_host_fastpath_contract(
    filename: str,
    _cache_name: str,
) -> None:
    source = (FORWARD_ROOT / filename).read_text()

    assert 'out["fixed_route_fastpath"]' in source
    assert 'out["route_env_guard_per_launch"]' in source
    assert 'out["kernel_attribute_init"]' in source
    assert '"once_per_host_thread_and_cuda_device"' in source
    assert '"per_launch"' in source


@pytest.mark.parametrize(
    ("filename", "capacity_name", "gl_slot_count"),
    DESCRIPTOR_CACHE_CASES,
)
def test_fixed_route_descriptor_cache_is_per_gl_and_pointer_keyed(
    filename: str,
    capacity_name: str,
    gl_slot_count: int,
) -> None:
    source = (FORWARD_ROOT / filename).read_text()
    prefix = (
        "hao_direct_fp8pv"
        if "FP8PV" in capacity_name
        else "hao_direct_fp4pv"
    )
    first_cache_call = source.index(f"{prefix}_cached_gl<0")

    assert "#include <array>" in source
    assert "#include <cstdint>" in source
    assert "#include <optional>" in source
    assert "#include <vector>" not in source
    assert f"constexpr std::size_t {capacity_name} = 256;" in source
    ways_name = capacity_name.replace("CAPACITY", "WAYS")
    sets_name = capacity_name.replace("CAPACITY", "SETS")
    gl_slots_name = capacity_name.replace("CAPACITY", "GL_SLOTS")
    assert f"constexpr std::size_t {ways_name} = 4;" in source
    assert (
        f"constexpr std::size_t {sets_name} =\n"
        f"    {capacity_name} /\n"
        f"    {ways_name};"
    ) in source
    assert (
        f"constexpr std::size_t {gl_slots_name} = {gl_slot_count};"
        in source
    )
    assert "template <std::size_t GlSlot, typename Descriptor" in source
    assert f"GlSlot < {gl_slots_name}" in source
    assert "int device;" in source
    assert "const void *data_pointer;" in source
    assert "device == other.device &&" in source
    assert "data_pointer == other.data_pointer" in source
    assert "reinterpret_cast<std::uintptr_t>(data_pointer)" in source
    assert "static_cast<unsigned int>(device)" in source
    assert "DESCRIPTOR_DEVICE_MIX" in source
    assert "DESCRIPTOR_HASH_MIX_1" in source
    assert "DESCRIPTOR_HASH_MIX_2" in source
    assert "hash_key >> 30" in source
    assert "hash_key >> 27" in source
    assert "hash_key >> 31" in source
    assert f"hash_key & ({sets_name} - 1)" in source
    assert "descriptor_cache.sets[set_index]" in source
    assert "for (const auto &slot : cache_set.entries)" in source
    assert "slot.has_value() && slot->key == key" in source
    assert "for (auto &slot : cache_set.entries)" in source
    assert "slot.emplace(descriptor_cache_entry{key, descriptor});" in source
    assert "cache_set.entries[cache_set.next_victim]" in source
    assert (
        f"(cache_set.next_victim + 1) %\n"
        f"        {ways_name};"
    ) in source
    assert "descriptor_cache.clear();" not in source
    assert "return build_descriptor();" in source
    for gl_slot in range(gl_slot_count):
        assert f"{prefix}_cached_gl<{gl_slot}, typename G::" in source

    # Cache hits must not bypass the public tensor metadata contracts.
    assert source.index("CHECK_INPUT(Q);") < first_cache_call
    validation_call = source.index("check_scalar(K_sg")
    validation_guard = source.rfind(
        "#if TK_HAO_DIRECT_FIXED_ROUTE_FASTPATH",
        0,
        validation_call,
    )
    validation_guard_end = source.index("#endif", validation_call)
    assert validation_guard < validation_call < validation_guard_end
    assert validation_guard_end < first_cache_call


@pytest.mark.parametrize(
    ("filename", "capacity_name", "_pointer_count"),
    DESCRIPTOR_CACHE_CASES,
)
def test_topology_reports_descriptor_cache_provenance(
    filename: str,
    capacity_name: str,
    _pointer_count: int,
) -> None:
    source = (FORWARD_ROOT / filename).read_text()

    assert 'out["tma_descriptor_cache"]' in source
    assert '"bounded_thread_local_gl_descriptors"' in source
    assert '"disabled"' in source
    assert 'out["tma_descriptor_cache_capacity"]' in source
    assert capacity_name in source
    assert 'out["tma_descriptor_cache_lookup"]' in source
    assert (
        '"splitmix64_device_pointer_four_way_set_associative"'
        in source
    )
    assert 'out["tma_descriptor_cache_set_hash"]' in source
    assert '"splitmix64_device_pointer_v1"' in source
    assert 'out["tma_descriptor_cache_sets"]' in source
    assert capacity_name.replace("CAPACITY", "SETS") in source
    assert 'out["tma_descriptor_cache_ways"]' in source
    assert capacity_name.replace("CAPACITY", "WAYS") in source
    assert 'out["tma_descriptor_cache_capacity_scope"]' in source
    assert '"per_compile_time_gl_slot"' in source
    assert 'out["tma_descriptor_cache_gl_slots"]' in source
    assert capacity_name.replace("CAPACITY", "GL_SLOTS") in source
    assert 'out["tma_descriptor_cache_total_entry_ceiling"]' in source
    assert 'out["tma_descriptor_cache_key"]' in source
    assert (
        '"cuda_device_data_ptr_and_compile_time_gl_slot"' in source
    )
    assert 'out["tma_descriptor_cache_owns_tensors"] = false;' in source
    assert (
        'out["tma_descriptor_cache_counter_scope"] =\n'
        '        "calling_host_thread";'
    ) in source
    for counter in ("hits", "misses", "evictions", "clears"):
        assert f'out["tma_descriptor_cache_{counter}"]' in source


@pytest.mark.parametrize(
    ("filename", "capacity_name", "_pointer_count"),
    DESCRIPTOR_CACHE_CASES,
)
def test_descriptor_cache_replacement_is_local_and_observable(
    filename: str,
    capacity_name: str,
    _pointer_count: int,
) -> None:
    source = (FORWARD_ROOT / filename).read_text()
    prefix = (
        "hao_direct_fp8pv"
        if "FP8PV" in capacity_name
        else "hao_direct_fp4pv"
    )

    assert f"++{prefix}_cache_counters.hits;" in source
    assert f"++{prefix}_cache_counters.misses;" in source
    assert f"++{prefix}_cache_counters.evictions;" in source
    assert f"{prefix}_cache_counters.clears" in source
    assert "descriptor_cache.clear();" not in source
    assert "Capacity overflow" not in source
    assert "Set-local replacement bounds work" in source
    assert "other GL slots or sets" in source


def test_mx_cache_hit_resets_non_descriptor_launch_fields() -> None:
    source = (FORWARD_ROOT / "hao_direct_fp4pv_host.inc").read_text()
    cache_end = source.index("G g = build_globals();")
    publication = source.index("if (P_scales != nullptr)", cache_end)

    for assignment in (
        "g.v_global_decode = V_sg;",
        "g.disable_lse_store = store_lse ? 0 : 1;",
        "g.lifecycle_debug = nullptr;",
        "g.group2_trap_stage = -1;",
        "g.p_scale_output = nullptr;",
    ):
        position = source.index(assignment, cache_end)
        assert cache_end < position < publication


def test_fp8_cache_hit_resets_non_descriptor_launch_fields() -> None:
    source = (FORWARD_ROOT / "hao_direct_host.inc").read_text()
    cache_end = source.index("G g = build_globals();")

    assert source.index(
        "g.disable_lse_store = store_lse ? 0 : 1;",
        cache_end,
    ) > cache_end


@pytest.mark.parametrize("filename", MAKEFILES)
def test_generic_makefiles_default_fastpath_off(filename: str) -> None:
    source = (FORWARD_ROOT / filename).read_text()

    assert "HAO_FIXED_ROUTE_FASTPATH ?= 0" in source
    assert (
        "-DTK_HAO_DIRECT_FIXED_ROUTE_FASTPATH="
        "$(HAO_FIXED_ROUTE_FASTPATH)"
    ) in source


def test_causal_candidate_builders_enable_fixed_route_fastpath() -> None:
    fp8_builder = (
        LOWP_ROOT / "build_causal_gqa_fp8pv_forward.py"
    ).read_text()
    mx_builder = (
        LOWP_ROOT / "build_causal_gqa_mxfp4pv_forward.py"
    ).read_text()
    d128_mx_builder = (
        LOWP_ROOT / "build_causal_gqa_d128_mxfp4pv_forward.py"
    ).read_text()
    matrix_driver = (LOWP_ROOT / "run_causal_forward_matrix.py").read_text()

    assert fp8_builder.count('"HAO_FIXED_ROUTE_FASTPATH=1"') == 1
    assert 'parser.add_argument("--batch", type=int, default=1)' in fp8_builder
    assert 'parser.error("--batch must be positive")' in fp8_builder
    assert 'f"HAO_BATCH={args.batch}"' in fp8_builder
    assert 'f"{args.probability_policy}_b{args.batch}s{args.sequence}h{args.q_heads}"' in fp8_builder
    assert mx_builder.count('"HAO_FIXED_ROUTE_FASTPATH=1"') == 1
    assert 'parser.add_argument("--batch", type=int, default=16)' in mx_builder
    assert 'parser.error("--batch must be positive")' in mx_builder
    assert 'f"HAO_BATCH={args.batch}"' in mx_builder
    assert '"HAO_CAUSAL_INTERLEAVED_KV=1"' in mx_builder
    assert '"HAO_FP4PV_MX_POLICY=causal-accurate"' in mx_builder
    assert (
        '"HAO_FP4PV_MX_MODE23_NATIVE_DENSITY_OVERRIDE=4"'
        in mx_builder
    )
    assert '"unanchored-splitmix-v6"' in mx_builder
    assert (
        '"HAO_FP4PV_MX_GLOBAL_ANCHOR32_OVERRIDE=0"'
        in mx_builder
    )
    assert (
        '"HAO_FP4PV_MX_GLOBAL_ANCHOR128_OVERRIDE=0"'
        in mx_builder
    )
    assert (
        '"HAO_FP4PV_MX_GLOBAL_ANCHOR_MARGIN_LOG2_OVERRIDE=0"'
        in mx_builder
    )
    assert (
        '"HAO_FP4PV_MX_ANCHOR_AFFINE_HOIST_OVERRIDE=0"'
        in mx_builder
    )
    assert (
        '"HAO_FP4PV_MX_STORED_SCALE_SHIFT_LOG2_OVERRIDE=16"'
        in mx_builder
    )
    assert 'default_variant_tag = "" if args.variant == "anchored"' in mx_builder
    assert 'args.variant.replace("-", "_")' in mx_builder
    assert d128_mx_builder.count('"HAO_FIXED_ROUTE_FASTPATH=1"') == 1
    assert 'choices=(1, 2, 4)' in d128_mx_builder
    assert 'args.batch in (2, 4)' in d128_mx_builder
    assert '(4096, 32, 8)' in d128_mx_builder
    assert '"B2/B4 are restricted to --sequence 4096' in d128_mx_builder
    assert '"HAO_HEAD_DIM=128"' in d128_mx_builder
    assert '"HAO_CAUSAL_INTERLEAVED_KV=0"' in d128_mx_builder
    assert '"HAO_FP4PV_MX_POLICY=causal-accurate"' in d128_mx_builder
    assert '"anchor128-m0"' in d128_mx_builder
    assert '"anchor128-m64"' in d128_mx_builder
    assert '"full-approx-mode1"' in d128_mx_builder
    assert '"stable-represented-logsum"' in d128_mx_builder
    assert '"stable-full-approx-mode1"' in d128_mx_builder
    assert (
        '"HAO_FP4PV_MX_GLOBAL_ANCHOR32_OVERRIDE="'
        in d128_mx_builder
    )
    assert (
        '"HAO_FP4PV_MX_GLOBAL_ANCHOR128_OVERRIDE="'
        in d128_mx_builder
    )
    assert (
        '"HAO_FP4PV_MX_GLOBAL_ANCHOR_MARGIN_LOG2_OVERRIDE="'
        in d128_mx_builder
    )
    assert (
        '"HAO_FP4PV_MX_FULL_APPROX_DENOM_OVERRIDE="'
        in d128_mx_builder
    )
    assert (
        '"HAO_FP4PV_MX_STABLE_LSE_LOGSUM_OVERRIDE="'
        in d128_mx_builder
    )
    assert (
        '"HAO_FP4PV_MX_CAUSAL_Q3_PROGRESSIVE_REUSE_OVERRIDE="'
        in d128_mx_builder
    )
    assert "MX_MODE23_NATIVE_DENSITY_OVERRIDE" not in d128_mx_builder
    assert 'target = f"{args.gpu.lower()}_sm{args.num_sm}"' in d128_mx_builder
    assert "refusing to overwrite an existing forward artifact" in d128_mx_builder
    assert 'parser.error("--kv-heads must be positive")' in d128_mx_builder
    assert '("--jobs", args.jobs)' in d128_mx_builder
    assert '("--nvcc-threads", args.nvcc_threads)' in d128_mx_builder
    assert '("--nvcc-split-compile", args.nvcc_split_compile)' in d128_mx_builder
    assert '"causal_gqa_d128_mxfp4pv_forward_build_v2"' in d128_mx_builder
    assert '"anchor_variant": args.anchor_variant' in d128_mx_builder
    assert '"saved_lse_denom": args.saved_lse_denom' in d128_mx_builder
    assert "_require_requested_topology(topology, args=args)" in d128_mx_builder
    assert '"tracked_diff_sha256"' in d128_mx_builder
    assert '"sources": build_sources' in d128_mx_builder
    assert "if build_sources != live_sources_before:" in d128_mx_builder
    assert "if _source_identities(workdir) != build_sources:" in d128_mx_builder
    assert "if _source_identities() != live_sources_before:" in d128_mx_builder
    assert "topology = _load_topology(output, module)" in d128_mx_builder
    assert '"topology": topology' in d128_mx_builder
    assert matrix_driver.count('"HAO_FIXED_ROUTE_FASTPATH=1"') == 1
    assert '"fixed_route_fastpath": True' in matrix_driver
    assert '"route_env_guard_per_launch": False' in matrix_driver
    assert (
        '"kernel_attribute_init": '
        '"once_per_host_thread_and_cuda_device"'
    ) in matrix_driver
    assert matrix_driver.count(
        '"tma_descriptor_cache": '
        '"bounded_thread_local_gl_descriptors"'
    ) == 2
    assert matrix_driver.count('"tma_descriptor_cache_capacity": 256') == 2
    assert matrix_driver.count(
        '"cuda_device_data_ptr_and_compile_time_gl_slot"'
    ) == 2
    assert matrix_driver.count(
        '"tma_descriptor_cache_owns_tensors": False'
    ) == 2


@pytest.mark.parametrize(
    ("arguments", "message"),
    (
        (("--kv-heads", "0"), "--kv-heads must be positive"),
        (("--sequence", "0"), "--sequence must be positive"),
        (("--jobs", "0"), "--jobs must be positive"),
        (("--nvcc-threads", "0"), "--nvcc-threads must be positive"),
        (
            ("--nvcc-split-compile", "0"),
            "--nvcc-split-compile must be positive",
        ),
    ),
)
def test_d128_mx_builder_rejects_unsafe_parallelism_and_shapes(
    arguments: tuple[str, ...],
    message: str,
) -> None:
    builder = LOWP_ROOT / "build_causal_gqa_d128_mxfp4pv_forward.py"
    completed = subprocess.run(
        [sys.executable, str(builder), *arguments],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert message in completed.stderr


def test_d128_mx_builder_applies_the_exact_shape_gate_to_b4() -> None:
    builder = LOWP_ROOT / "build_causal_gqa_d128_mxfp4pv_forward.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(builder),
            "--batch",
            "4",
            "--sequence",
            "2048",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "B2/B4 are restricted" in completed.stderr


def test_isolated_forward_validator_pins_only_reviewed_batch_identities() -> None:
    assert AUTHENTICATED_BATCHES == (2, 8, 16)
    assert CANONICAL_FORWARD_ARTIFACTS == {
        1: (
            "e7bb8e69625adf0e545c80d01b194c13af0ea9e12db8765150d2762267716c35",
            1_817_192,
        ),
        2: (
            "4e4c4c9b1afd7a751c3bae9d734f617a04b0b95778370deba9be3f131f5e05d1",
            1_817_192,
        ),
        8: (
            "34114089ab4631093dc2b4dbd38e01a597a6608c9cfb748bd927f8038271db9d",
            1_817_088,
        ),
        16: (
            "88d81d3783e5aa80f0e9cf259a2ea7c935da4c2a5dc3ba1868e63f802a2c6208",
            1_817_256,
        ),
    }
    for batch, identity in CANONICAL_FORWARD_ARTIFACTS.items():
        assert _require_canonical_forward_identity(
            batch, identity[0], identity[1]
        ) == identity
    with pytest.raises(ValueError, match="no canonical"):
        _require_canonical_forward_identity(4, "0" * 64, 1)
    with pytest.raises(ValueError, match="not the canonical"):
        _require_canonical_forward_identity(
            2,
            "0" * 64,
            CANONICAL_FORWARD_ARTIFACTS[2][1],
        )
    with pytest.raises(ValueError, match="byte count"):
        _require_canonical_forward_identity(
            2,
            CANONICAL_FORWARD_ARTIFACTS[2][0],
            CANONICAL_FORWARD_ARTIFACTS[2][1] + 1,
        )
    assert CANONICAL_PROJECTION_ARTIFACT == (
        "bfdec1e43a0a19acec5afbac3fa837e2f4d1b25be80ae7fb5ff3b5bc5e9e25ce",
        17_504_688,
    )
    with pytest.raises(ValueError, match="not the canonical"):
        _authenticate_projection_artifact(
            Path("not-opened.so"),
            supplied_sha256="0" * 64,
            supplied_bytes=CANONICAL_PROJECTION_ARTIFACT[1],
        )
    with pytest.raises(ValueError, match="byte count"):
        _authenticate_projection_artifact(
            Path("not-opened.so"),
            supplied_sha256=CANONICAL_PROJECTION_ARTIFACT[0],
            supplied_bytes=CANONICAL_PROJECTION_ARTIFACT[1] + 1,
        )


def test_regular_forward_artifact_authentication_hashes_exact_bytes(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "forward.so"
    payload = b"reviewed-forward-artifact\n"
    artifact.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    resolved, identity = _authenticate_regular_artifact(
        artifact,
        digest,
        len(payload),
    )
    assert resolved == artifact.resolve()
    assert identity == {
        "path": str(artifact.resolve()),
        "sha256": digest,
        "bytes": len(payload),
    }
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        _authenticate_regular_artifact(artifact, "0" * 64, len(payload))
    with pytest.raises(RuntimeError, match="byte-count mismatch"):
        _authenticate_regular_artifact(artifact, digest, len(payload) + 1)


def test_regular_forward_artifact_authentication_rejects_symlinks(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "forward.so"
    artifact.write_bytes(b"payload")
    link = tmp_path / "forward-link.so"
    link.symlink_to(artifact)
    with pytest.raises(RuntimeError, match="non-symlink"):
        _authenticate_regular_artifact(
            link,
            hashlib.sha256(b"payload").hexdigest(),
            len(b"payload"),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("route", "wrong"),
        ("schema", "wrong"),
        ("fixed_route_fastpath", False),
        ("fixed_p_ceiling", True),
        ("score_pack_ceiling", True),
        ("shiftless_fp8_mode", 5),
        ("causal_interleaved_kv", True),
    ),
)
def test_isolated_forward_validator_rejects_unauthenticated_topology(
    field: str,
    value: object,
) -> None:
    topology = _exact_fp8_topology()
    topology[field] = value
    with pytest.raises(RuntimeError, match=field):
        _require_exact_forward_topology(
            topology,
            batch=2,
            sequence=4096,
            q_heads=32,
            kv_heads=8,
        )


def test_isolated_forward_validator_requires_runtime_valid_after_launch() -> None:
    topology = _exact_fp8_topology(valid=0)
    topology.pop("causal_interleaved_kv")
    with pytest.raises(RuntimeError, match="causal_interleaved_kv"):
        _require_exact_forward_topology(
            topology,
            batch=2,
            sequence=4096,
            q_heads=32,
            kv_heads=8,
        )
    topology["causal_interleaved_kv"] = False
    with pytest.raises(RuntimeError, match="valid=0"):
        _require_exact_forward_topology(
            topology,
            batch=2,
            sequence=4096,
            q_heads=32,
            kv_heads=8,
            runtime_populated=True,
        )
    topology["valid"] = 1
    _require_exact_forward_topology(
        topology,
        batch=2,
        sequence=4096,
        q_heads=32,
        kv_heads=8,
        runtime_populated=True,
    )


def test_isolated_forward_validator_gates_sequential_relative_l2() -> None:
    source = (
        LOWP_ROOT / "validate_causal_gqa_fp8pv_batch.py"
    ).read_text()
    main = source.split("def main() -> None:", 1)[1]
    assert SEQUENTIAL_RELATIVE_L2_LIMIT == 0.01
    for flag in (
        "--batched-forward-sha256",
        "--batched-forward-bytes",
        "--batch1-forward-sha256",
        "--batch1-forward-bytes",
        "--projection-extension-sha256",
        "--projection-extension-bytes",
    ):
        assert flag in source
    assert main.index("_authenticate_projection_artifact(") < main.index(
        "import torch"
    )
    assert main.index("_authenticate_forward_artifact(") < main.index(
        "import torch"
    )
    assert '"projection_extension": projection_identity' in source
    assert '"sequential_output_relative_l2"' in source
    assert '"sequential_lse_relative_l2"' in source


def test_mx_p_tmem_consumers_fence_after_publication_barriers() -> None:
    source = (FORWARD_ROOT / "hao_direct_fp4pv_kernel.inc").read_text()
    first_wait = source.index(
        "hao_direct_fp4pv_wait(\n"
        "                    p_first_o_rescaled[Stage]"
    )
    first_fence = source.index("tensor_after_thread_sync();", first_wait)
    first_mma = source.index("hao_direct_issue_pv_mxfp4_n128", first_wait)
    assert first_wait < first_fence < first_mma
    assert (
        "C::HAO_DIRECT_MXFP4_PV ||\n"
        "                    ((C::HAO_DIRECT_NV_SOFTMAX_V_SCALE_PREFETCH_MASK"
        in source[first_wait:first_mma]
    )

    helper = source.index("hao_direct_issue_pv_mxfp4_n128(")
    tail_wait = source.index("hao_direct_fp4pv_wait(p_tail, p_phase);", helper)
    tail_fence = source.index("tensor_after_thread_sync();", tail_wait)
    tail_mma = source.index("tcgen05.mma.cta_group::1.kind::mxf4nvf4", tail_fence)
    assert tail_wait < tail_fence < tail_mma


def test_selected_mx_route_completes_tmem_stores_before_handoff() -> None:
    makefile = (FORWARD_ROOT / "Makefile.hao_direct_fp4pv").read_text()
    selected_policy = makefile.index(
        "override HAO_FP4PV_MX_SKIP_ZERO_SCALE_MASK := 1"
    )
    next_policy = makefile.index(
        "override HAO_FP4PV_MX_SHIFTLESS_CORR_BYPASS := 1",
        selected_policy,
    )
    handoff = makefile[selected_policy:next_policy]

    assert "override HAO_FP4PV_MX_EARLY_P := 1" in handoff
    assert "override HAO_FP4PV_MX_EARLY_ASYNC_SCALE := 0" in handoff
    assert "override HAO_FP4PV_MX_EARLY_ASYNC_SCALE := 1" not in handoff
    assert "Complete each store group" in handoff

    kernel = (FORWARD_ROOT / "hao_direct_fp4pv_kernel.inc").read_text()
    helper_start = kernel.index(
        "hao_direct_store_mxfp4_p_scale_half_packed("
    )
    helper_end = kernel.index("\n}\n", helper_start)
    helper = kernel[helper_start:helper_end]
    assert "if constexpr (Wait)" in helper
    assert "fp4pv_tmem_store_wait();" in helper

    reader = (
        FORWARD_ROOT / "hao_direct_fp4pv_softmax_reader.inc"
    ).read_text()
    assert reader.count("!C::HAO_DIRECT_MX_EARLY_ASYNC_SCALE>(") >= 4
    assert reader.count("!C::HAO_DIRECT_MX_EARLY_ASYNC_SCALE,") >= 2


def test_mx_epilogue_normalizes_positive_subnormal_denominators() -> None:
    source = (
        FORWARD_ROOT / "depth1_upstream_mxfp4_fp8pv_kernel.inc"
    ).read_text()

    assert "constexpr float SUBNORMAL_LIFT = 0x1p24f;" in source
    assert "sum < 0x1p-126f" in source
    assert "arithmetic_sum ?" not in source
    assert '"f"(invalid ? 1.0f : arithmetic_sum)' in source
    assert "(lift_subnormal_lse_sum ? 24.0f : 0.0f)" in source
    assert "SCALAR_SUBNORMAL_LIFT" in source


def test_mx_dual_lse_keeps_output_and_saved_denominators_separate() -> None:
    kernel = (FORWARD_ROOT / "hao_direct_fp4pv_kernel.inc").read_text()
    reader = (
        FORWARD_ROOT / "hao_direct_fp4pv_softmax_reader.inc"
    ).read_text()
    epilogue = (
        FORWARD_ROOT / "depth1_upstream_mxfp4_fp8pv_kernel.inc"
    ).read_text()

    assert "HAO_DIRECT_MX_DUAL_LSE_DENOM" in kernel
    assert "*lse_sum_out +=" in kernel
    assert kernel.index("*lse_sum_out +=") < kernel.index(
        "if constexpr (DeferDenomFinalize)",
        kernel.index("*lse_sum_out +="),
    )
    assert "lse_row_sum += tile_lse_sum;" in reader
    assert "lse_sum_smem[Stage]" in reader
    assert "const float sum = stats_load(row_sum_owned, row);" in epilogue
    assert "float lse_stat = sum;" in epilogue
    assert '"f"(invalid ? 1.0f : arithmetic_sum)' in epilogue
    assert '"f"(arithmetic_lse_sum)' in epilogue


def test_mx_stable_saved_lse_accumulates_carriers_in_log_domain() -> None:
    kernel = (FORWARD_ROOT / "hao_direct_fp4pv_kernel.inc").read_text()
    reader = (
        FORWARD_ROOT / "hao_direct_fp4pv_softmax_reader.inc"
    ).read_text()
    epilogue = (
        FORWARD_ROOT / "depth1_upstream_mxfp4_fp8pv_kernel.inc"
    ).read_text()

    assert "hao_direct_accumulate_mxfp4_denom_carrier_log_domain" in kernel
    assert "constexpr float LOG2_12 = 3.584962500721156f;" in kernel
    assert "lse_scaled_quantized_sum_x2" in reader
    assert reader.count(
        "hao_direct_accumulate_mxfp4_denom_carrier_log_domain("
    ) == 4
    assert "hao_direct_finalize_mxfp4_denom_carrier_log2(" in reader
    assert "if constexpr (StableLseLogsum)" in epilogue
    assert "log2_sum = lse_stat;" in epilogue
    assert "__fmul_rn(full_approx_sum, 2.0f)" in kernel
    assert "*lse_max_e8m0_out" in kernel
    assert "*lse_scaled_mass_out" in kernel
    assert "if constexpr (StableLseLogsum && !DualLse)" in reader


def test_shiftless_mx_stabilizes_unrepresentable_initial_causal_scales() -> None:
    source = (
        FORWARD_ROOT / "hao_direct_fp4pv_softmax_reader.inc"
    ).read_text()

    assert "const bool stabilize_initial_causal_quarter" in source
    assert "!C::HAO_DIRECT_GLOBAL_ANCHOR" in source
    assert "C::HAO_DIRECT_MX_FIXED_ANCHOR_LOG2 == 0" in source
    assert "query_tile == 0 && reader_warp == 0" in source
    assert "if (stabilize_initial_causal_quarter && n == 0)" in source
    assert "hao_direct_select_mxfp4_e8m0<\n                                PairScaleC," in source
    assert (
        "q0_shiftless_e8m0 <=\n"
        "                            "
        "C::HAO_DIRECT_MX_STORED_SCALE_SHIFT_LOG2"
        in source
    )
    assert "if (stabilize_initial_causal_row)" in source
    assert "row_max = quarter_max;" in source

    assert "if (q0_shiftless_e8m0 == 0)" not in source
    assert "constexpr bool exact_native_q0_probe" not in source


def test_mx_stored_scale_shift_is_uniform_over_full_code_domain() -> None:
    source = (FORWARD_ROOT / "hao_direct_fp4pv_kernel.inc").read_text()

    assert "C::HAO_DIRECT_MX_STORED_SCALE_SHIFT_LOG2 != 0 ||" in source
    assert (
        "constexpr int WORKING_E8M0_FLOOR =\n"
        "            C::HAO_DIRECT_MX_STORED_SCALE_SHIFT_LOG2 + 1;"
        in source
    )
    assert '"max.s32 %0, %0, %1;"' in source
    assert ': "n"(WORKING_E8M0_FLOOR)' in source

    for shift in range(121):
        for raw_e8m0 in range(255):
            working_e8m0 = max(raw_e8m0, shift + 1)
            stored_e8m0 = (
                raw_e8m0 - shift if raw_e8m0 > shift else 1
            )
            assert working_e8m0 - stored_e8m0 == shift


def test_b200_causal_accurate_d64_uses_one_hoisted_row_anchor() -> None:
    makefile = (FORWARD_ROOT / "Makefile.hao_direct_fp4pv").read_text()
    policy_start = makefile.index(
        "ifeq ($(HAO_FP4PV_MX_POLICY),causal-accurate)"
    )
    policy_end = makefile.index(
        "ifeq ($(HAO_FP4PV_MX_POLICY),causal-scale-reuse)",
        policy_start,
    )
    policy = makefile[policy_start:policy_end]

    assert "ifeq ($(GPU)x$(HAO_HEAD_DIM),B200x64)" in policy
    assert "override HAO_FP4PV_MX_GLOBAL_ANCHOR32 := 1" in policy
    assert "override HAO_FP4PV_MX_GLOBAL_ANCHOR128 := 0" in policy
    assert "override HAO_FP4PV_MX_GLOBAL_ANCHOR_MARGIN_LOG2 := 64" in policy
    assert "override HAO_FP4PV_MX_ANCHOR_AFFINE_HOIST := 1" in policy

    reader = (
        FORWARD_ROOT / "hao_direct_fp4pv_softmax_reader.inc"
    ).read_text()
    row_scope = reader.index("float row_max =")
    tile_loop = reader.index("for (int n =", row_scope)
    anchor = reader.index("C::HAO_DIRECT_GLOBAL_ANCHOR", tile_loop)
    assert row_scope < tile_loop < anchor
    assert "? row_max\n                            : 0.0f;" in reader[anchor:]
