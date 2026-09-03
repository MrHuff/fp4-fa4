import copy
import json

import pytest

from tk_fa4.lowp_fa4_bwd.merge_real_token_route_results import (
    BF16_ROUTE,
    FP8_ROUTE,
    INPUT_SCHEMA,
    MX_ROUTE,
    OUTPUT_SCHEMA,
    MergeValidationError,
    main,
    merge_route_results,
)


def _source_identity():
    return {
        "trainer": {
            "path": "/repo/train.py",
            "bytes": 1234,
            "sha256": "1" * 64,
        },
        "git": {
            "available": True,
            "repo_root": "/repo",
            "head": "a" * 40,
            "branch": "test",
            "tracked_dirty": False,
            "tracked_diff_bytes": 0,
            "tracked_diff_sha256": "e" * 64,
        },
    }


def _backward_contract():
    return {
        "schema": "lowp_backward_contract_v1",
        "shape": {
            "sequence": 4096,
            "q_heads": 32,
            "kv_heads": 8,
            "head_dim": 128,
        },
        "control": {
            "provenance": {"sha256": "2" * 64},
            "generated_source": True,
            "fp8_p_storage": "tmem",
            "direct_tma_dkdv": True,
            "detached_fp8_p_tmem": True,
        },
        "probability": {
            "forward_mx_probability_replay": False,
            "forward_mx_probability_scale_handoff": False,
            "reuse_quantized_p": True,
            "exp2_degree": 1,
            "exp2_period": 0,
            "fp8_ds_lift": 0,
            "fuse_probability_lift": False,
            "prelift_probability_lse": False,
        },
        "schedule": {"head_fast_raster": True},
        "projection": {
            "qkv_projection_format": "nvfp4",
            "represented_backward": True,
        },
        "shape_policy": {"name": "d128"},
        "scaling": {
            "loss_scale": 1.0,
            "gradient_global_scale": 1.0,
            "probability_correction": 1.0,
            "q_gain": 1.0,
            "k_gain": 1.0,
            "v_gain": 1.0,
            "v_weight_gain": 1.0,
        },
    }


def _forward_topology(route):
    if route == MX_ROUTE:
        extension_route = "real_fwd_tk_hao_direct_nvfp4_mxfp4pv"
        pv_format = "mxfp4_e8m0_block32"
    else:
        extension_route = "real_fwd_tk_hao_direct_causal_gqa_nvfp4_fp8pv"
        pv_format = "e4m3_fp8"
    return {
        "schema": "tk_hao_direct_pipeline_v1",
        "batch": 1,
        "seqlen": 4096,
        "heads": 32,
        "kv_heads": 8,
        "dqk": 128,
        "dvo": 128,
        "causal": True,
        "causal_interleaved_kv": False,
        "qk_format": "nvfp4_e4m3_block16",
        "route": extension_route,
        "pv_format": pv_format,
    }


def _forward_dispatch_contract(route):
    topology = _forward_topology(route)
    attention_symbol = (
        "forward_hao_direct_fp4pv"
        if route == MX_ROUTE
        else "forward_hao_direct_fp8pv"
    )
    return {
        "schema": "lowp_forward_dispatch_contract_v2",
        "route": topology["route"],
        "pv_format": topology["pv_format"],
        "qkv_projection": {
            "format": "nvfp4",
            "runtime_crossover_reallocation": False,
        },
        "attention": {
            "dispatch": "construction_bound_route_specific_entrypoint",
            "symbol": attention_symbol,
            "entrypoint_bound_at_construction": True,
            "launcher_bound_to_runtime": True,
        },
        "d128_projection_publication": {
            "schema": "d128_nvfp4_public_projection_preflight_v1",
            "native_publication_validated_by_interface": True,
        },
        "validated_after_compile_before_timing": True,
    }


def _configuration(route):
    contract = _backward_contract()
    return {
        "model_preset": "llama3.1-8b",
        "layers": 32,
        "hidden": 4096,
        "intermediate": 14336,
        "q_heads": 32,
        "kv_heads": 8,
        "head_dim": 128,
        "sequence": 4096,
        "vocab": 128256,
        "batch": 1,
        "hardware_identity": {
            "schema": "cuda_hardware_identity_v1",
            "visible_device_count": 1,
            "logical_device_index": 0,
            "name": "NVIDIA GB200",
            "uuid": "GPU-00000000-0000-0000-0000-000000000000",
            "compute_capability": [10, 0],
            "total_memory_bytes": 197897617408,
            "multiprocessor_count": 152,
            "l2_cache_bytes": 135266304,
            "pci_domain_id": 0,
            "pci_bus_id": 41,
            "pci_device_id": 0,
            "torch_version": "2.9.0a0+145a3a7bda.nv25.10",
            "torch_cuda_version": "13.0",
        },
        "rounds": 3,
        "routes": [route],
        "training_batches": 3,
        "validation_batches": 1,
        "eval_every": 3,
        "diagnostic_start": None,
        "diagnostic_every": 1,
        "diagnostic_routes": [BF16_ROUTE, MX_ROUTE, FP8_ROUTE],
        "diagnostic_on_drift_warning": False,
        "progress_output": f"/tmp/{route}.progress.json",
        "progress_every": 1,
        "mx_loss_drift_gate": None,
        "seed": 20260818,
        "learning_rate": 1.0e-4,
        "gradient_clip_norm": 1.0,
        "gradient_clip_norm_type": 2.0,
        "gradient_clip_foreach": True,
        "gradient_error_if_nonfinite": True,
        "optimizer": "fused AdamW",
        "loss_dtype": "fp32",
        "warmup_updates_model": False,
        "corpus": "/mnt/dolma.jsonl",
        "corpus_sha256": "3" * 64,
        "corpus_documents": 20000,
        "corpus_metadata": {"format": "JSON Lines", "unique_documents": 20000},
        "train_fraction": 0.8,
        "split_seed": 20261121,
        "train_documents": 16000,
        "validation_documents": 4000,
        "tokenizer": "/mnt/tokenizer.json",
        "tokenizer_sha256": "4" * 64,
        "tokenizer_vocab": 128256,
        "bos_token_id": 128000,
        "eos_token_id": 128001,
        "rope_theta": 500000.0,
        "rope_scaling": {"rope_type": "llama3", "factor": 8.0},
        "backward_extension": {
            "module": "tk_fa4._C_b300_lowp_bwd",
            "path": "/tmp/backward.so",
            "sha256": "5" * 64,
        },
        "projection_extension": {
            "module": "tk_fa4._C_b300_lowp_bwd",
            "path": "/tmp/backward.so",
            "bytes": 999,
            "sha256": "5" * 64,
            "required_symbols": [],
            "capabilities": {},
        },
        "mx_forward_extension": (
            {"module": "mx", "bytes": 600, "sha256": "6" * 64}
            if route == MX_ROUTE
            else None
        ),
        "fp8_forward_extension": (
            {"module": "fp8", "bytes": 700, "sha256": "7" * 64}
            if route == FP8_ROUTE
            else None
        ),
        "initialization": "strict clone of BF16 state_dict",
        "initial_state_probe": {
            "schema": "state_dict_sparse_probe_v1",
            "tensor_count": 291,
            "sampled_values": 1164,
            "sha256": "8" * 64,
        },
        "targets": "next token from S+1 chunks; no sequence wraparound",
        "train_tokens": {
            "batches": 3,
            "sequence": 4096,
            "sha256": "9" * 64,
            "document_order_sha256": "a" * 64,
        },
        "validation_tokens": {
            "batches": 1,
            "sequence": 4096,
            "sha256": "b" * 64,
            "document_order_sha256": "c" * 64,
        },
        "projection_weight_scaling": "2d",
        "v_mxfp4_scaling": "1d",
        "fp8_qkv_projection_format": "nvfp4",
        "mx_qkv_projection_format": "nvfp4",
        "q_quant_scale": 2.25,
        "k_quant_scale": 2.0,
        "qk_scale_refresh_every": 0,
        "effective_qk_scale_refresh_every": {route: 0},
        "mx_per_block_qk_scales": False,
        "fp8_per_block_qk_scales": False,
        "mx_experimental_split_v_backward": False,
        "mx_backward_forward_probability_replay_requested": False,
        "mx_backward_forward_probability_scale_handoff_requested": False,
        "mx_backward_forward_probability_replay": (
            False if route == MX_ROUTE else None
        ),
        "mx_backward_forward_probability_scale_handoff": (
            False if route == MX_ROUTE else None
        ),
        "mx_probability_replay_provenance": (
            {"enabled": False} if route == MX_ROUTE else None
        ),
        "mx_projection_publication_topology": (
            {"represented_backward": True} if route == MX_ROUTE else None
        ),
        "fp8_projection_publication_topology": (
            {"represented_backward": True} if route == FP8_ROUTE else None
        ),
        "backward_gain": 1.0,
        "mx_backward_gain": 1.0,
        "fp8_backward_gain": 1.0,
        "mx_backward_component_gains": (
            {"q": 1.0, "k": 1.0, "v": 1.0, "v_weight": 1.0}
            if route == MX_ROUTE
            else {"q": None, "k": None, "v": None, "v_weight": None}
        ),
        "fp8_backward_component_gains": (
            {"q": 1.0, "k": 1.0, "v": 1.0, "v_weight": 1.0}
            if route == FP8_ROUTE
            else {"q": None, "k": None, "v": None, "v_weight": None}
        ),
        "backward_exp2_degree": 1,
        "backward_exp2_period": 0,
        "backward_exp2_requested_degree": 1,
        "backward_exp2_requested_period": 0,
        "backward_control_provenance": (
            {"sha256": "2" * 64} if route != BF16_ROUTE else None
        ),
        "backward_control_route_provenance": {
            MX_ROUTE: {"sha256": "2" * 64} if route == MX_ROUTE else None,
            FP8_ROUTE: {"sha256": "2" * 64} if route == FP8_ROUTE else None,
        },
        "backward_route_contracts": (
            {route: contract} if route in (MX_ROUTE, FP8_ROUTE) else {}
        ),
        "matched_lowp_backward_contract": False,
        "shared_lowp_backward_runner": None,
        "timed_forward_dispatch_contracts": (
            {route: _forward_dispatch_contract(route)}
            if route in (MX_ROUTE, FP8_ROUTE)
            else None
        ),
        "backward_exp2_policy": {"effective_degree": 1, "effective_period": 0},
        "backward_detached_fp8_p_tmem": (
            True if route != BF16_ROUTE else None
        ),
        "backward_probability_tmem_policy": (
            "detached" if route != BF16_ROUTE else None
        ),
        "backward_head_fast_raster": True if route != BF16_ROUTE else None,
        "backward_raster_policy": (
            "head_fast" if route != BF16_ROUTE else None
        ),
        "backward_exp2_route_policies": {
            MX_ROUTE: {"degree": 1} if route == MX_ROUTE else None,
            FP8_ROUTE: {"degree": 1} if route == FP8_ROUTE else None,
        },
        "backward_detached_fp8_p_tmem_routes": {
            MX_ROUTE: True if route == MX_ROUTE else None,
            FP8_ROUTE: True if route == FP8_ROUTE else None,
        },
        "backward_probability_tmem_route_policies": {
            MX_ROUTE: "detached" if route == MX_ROUTE else None,
            FP8_ROUTE: "detached" if route == FP8_ROUTE else None,
        },
        "backward_head_fast_rasters": {
            MX_ROUTE: True if route == MX_ROUTE else None,
            FP8_ROUTE: True if route == FP8_ROUTE else None,
        },
        "backward_raster_route_policies": {
            MX_ROUTE: "head_fast" if route == MX_ROUTE else None,
            FP8_ROUTE: "head_fast" if route == FP8_ROUTE else None,
        },
        "mx_backward_reuse_quantized_p": True,
        "fp8_backward_reuse_quantized_p": True,
        "mx_backward_match_forward_operands": True,
        "fp8_backward_match_forward_operands": True,
        "mx_effective_backward_match_forward_operands": (
            True if route == MX_ROUTE else None
        ),
        "fp8_effective_backward_match_forward_operands": (
            True if route == FP8_ROUTE else None
        ),
        "parameter_count": 8030000000,
        "mx_forward_topology": (
            _forward_topology(MX_ROUTE) if route == MX_ROUTE else None
        ),
        "fp8_forward_topology": (
            _forward_topology(FP8_ROUTE) if route == FP8_ROUTE else None
        ),
    }


def _result(route, *, timing_scale, loss_offset):
    losses = [4.0 + loss_offset, 3.0 + loss_offset, 2.0 + loss_offset]
    validation_losses = [4.5 + loss_offset, 2.5 + loss_offset]
    timing = {
        "forward_ms": 10.0 * timing_scale,
        "backward_ms": 15.0 * timing_scale,
        "gradient_clip_ms": 1.0 * timing_scale,
        "optimizer_ms": 4.0 * timing_scale,
        "step_ms": 30.0 * timing_scale,
        "wall_ms": 31.0 * timing_scale,
        "timed_records": 3,
    }
    records = [
        {
            "route": route,
            "round": index,
            "batch": index,
            "loss": loss,
            "finite": True,
        }
        for index, loss in enumerate(losses)
    ]
    return {
        "schema": INPUT_SCHEMA,
        "source": _source_identity(),
        "configuration": _configuration(route),
        "validation_history": [
            {
                "round": round_index,
                "routes": {route: {"mean_loss": loss, "losses": [loss]}},
            }
            for round_index, loss in zip((-1, 2), validation_losses, strict=True)
        ],
        "records": {route: records},
        "routes": {
            route: {
                "timing": timing,
                "training": {
                    "losses": losses,
                    "first_loss": losses[0],
                    "last_loss": losses[-1],
                    "last_eight_mean": sum(losses) / len(losses),
                    "minimum_loss": min(losses),
                    "all_steps_finite": True,
                    "gradient_clipping": None,
                },
                "validation": {
                    "initial_loss": validation_losses[0],
                    "final_loss": validation_losses[-1],
                },
            }
        },
        "comparison_reference_route": BF16_ROUTE if route == BF16_ROUTE else None,
        "comparisons": {},
        "memory": {"peak_allocated_gib": 20.0},
    }


@pytest.fixture
def matched_results():
    # BF16 is the slow reference; MX is faster than FP8 with equal lowp bwd.
    bf16 = _result(BF16_ROUTE, timing_scale=4.0 / 3.0, loss_offset=0.0)
    mx = _result(MX_ROUTE, timing_scale=1.0, loss_offset=0.2)
    fp8 = _result(FP8_ROUTE, timing_scale=16.0 / 15.0, loss_offset=0.1)
    # Backward is intentionally exactly shared even though other components differ.
    mx["routes"][MX_ROUTE]["timing"]["backward_ms"] = 15.0
    fp8["routes"][FP8_ROUTE]["timing"]["backward_ms"] = 15.0
    return bf16, mx, fp8


def _as_d64_results(matched_results, *, projection_format, experimental=False):
    results = copy.deepcopy(matched_results)
    for result in results:
        configuration = result["configuration"]
        configuration.update(
            {
                "model_preset": "llama-1.2b",
                "layers": 16,
                "hidden": 2048,
                "intermediate": 8192,
                "head_dim": 64,
                "parameter_count": 1_235_800_000,
                "mx_qkv_projection_format": projection_format,
                "fp8_qkv_projection_format": projection_format,
            }
        )
        if experimental:
            configuration["experimental_d64_nvfp4_qkv"] = True
            configuration["mx_backward_match_forward_operands"] = False
            configuration["fp8_backward_match_forward_operands"] = False
            configuration["mx_per_block_qk_scales"] = False
            configuration["fp8_per_block_qk_scales"] = False
            configuration["mx_experimental_split_v_backward"] = False

        route = configuration["routes"][0]
        for topology_key in ("mx_forward_topology", "fp8_forward_topology"):
            topology = configuration[topology_key]
            if topology is not None:
                topology["dqk"] = 64
                topology["dvo"] = 64

        publication_topology_key = (
            "mx_projection_publication_topology"
            if route == MX_ROUTE
            else "fp8_projection_publication_topology"
        )
        if route in (MX_ROUTE, FP8_ROUTE):
            configuration[publication_topology_key][
                "qkv_projection_format"
            ] = projection_format
            dispatch = configuration["timed_forward_dispatch_contracts"][route]
            dispatch["qkv_projection"]["format"] = projection_format
            dispatch.pop("d128_projection_publication")
            if experimental:
                dispatch["d64_nvfp4_projection_publication"] = {
                    "schema": "d64_nvfp4_public_projection_preflight_v1",
                    "native_publication_validated_by_interface": True,
                }
            contract = configuration["backward_route_contracts"][route]
            contract["shape"]["head_dim"] = 64
            contract["projection"][
                "qkv_projection_format"
            ] = projection_format
            if experimental:
                contract["projection"]["represented_backward"] = False
                contract["projection"]["per_block_qk_scales"] = False
                contract["projection"][
                    "experimental_split_v_backward"
                ] = False

    return results


def test_merge_reports_component_e2e_and_loss_comparisons(matched_results):
    merged = merge_route_results(*matched_results)

    assert merged["schema"] == OUTPUT_SCHEMA
    assert merged["validation"]["matched"] is True
    assert merged["validation"]["lowp_forward_provenance_verified"] is True
    assert merged["validation"]["hardware_identity_verified"] is True
    assert (
        merged["validation"]["hardware_identity"]["name"]
        == "NVIDIA GB200"
    )
    assert merged["validation"]["lowp_backward_contract_matched"] is True
    assert (
        merged["forward_provenance"][MX_ROUTE]["validated_pv_format"]
        == "mxfp4_e8m0_block32"
    )

    mx_vs_bf16 = merged["timing_comparisons"]["by_route_vs_bf16"][MX_ROUTE]
    assert mx_vs_bf16["e2e"]["speedup"] == pytest.approx(4.0 / 3.0)
    assert mx_vs_bf16["components"]["forward_ms"]["speedup"] == pytest.approx(
        4.0 / 3.0
    )
    mx_vs_fp8 = merged["timing_comparisons"]["mxfp4_vs_fp8"]
    assert mx_vs_fp8["e2e"]["speedup"] == pytest.approx(16.0 / 15.0)
    assert mx_vs_fp8["components"]["backward_ms"]["speedup"] == 1.0

    mx_loss = merged["loss_comparisons"]["by_route_vs_bf16"][MX_ROUTE]
    assert mx_loss["training"]["last_loss_delta"] == pytest.approx(0.2)
    assert mx_loss["validation"]["final_loss_delta"] == pytest.approx(0.2)
    assert (
        merged["loss_comparisons"]["mxfp4_vs_fp8"]["validation"][
            "final_loss_delta"
        ]
        == pytest.approx(0.1)
    )
    assert set(merged["records"]) == {BF16_ROUTE, MX_ROUTE, FP8_ROUTE}
    assert [event["round"] for event in merged["validation_history"]] == [-1, 2]


def test_merge_keeps_d64_e4m3_as_default(matched_results):
    results = _as_d64_results(matched_results, projection_format="e4m3")

    merged = merge_route_results(*results)

    assert (
        merged["forward_provenance"][MX_ROUTE]["validated_projection_format"]
        == "e4m3"
    )
    assert "experimental_d64_nvfp4_qkv" not in merged[
        "forward_provenance"
    ][MX_ROUTE]


def test_merge_accepts_explicit_matched_d64_nvfp4_projection(matched_results):
    results = _as_d64_results(
        matched_results,
        projection_format="nvfp4",
        experimental=True,
    )

    merged = merge_route_results(*results)

    for route in (MX_ROUTE, FP8_ROUTE):
        provenance = merged["forward_provenance"][route]
        assert provenance["validated_projection_format"] == "nvfp4"
        assert provenance["experimental_d64_nvfp4_qkv"] is True


def test_merge_rejects_d64_nvfp4_without_explicit_experimental_flag(
    matched_results,
):
    results = _as_d64_results(matched_results, projection_format="nvfp4")

    with pytest.raises(
        MergeValidationError,
        match="experimental_d64_nvfp4_qkv=true",
    ):
        merge_route_results(*results)


def test_merge_rejects_mixed_d64_projection_formats(matched_results):
    results = _as_d64_results(
        matched_results,
        projection_format="nvfp4",
        experimental=True,
    )
    for result in results:
        result["configuration"]["fp8_qkv_projection_format"] = "e4m3"

    with pytest.raises(
        MergeValidationError,
        match="requires both MX and FP8 route projection formats",
    ):
        merge_route_results(*results)


def test_merge_rejects_d64_nvfp4_without_2d_projection_weights(
    matched_results,
):
    results = _as_d64_results(
        matched_results,
        projection_format="nvfp4",
        experimental=True,
    )
    for result in results:
        result["configuration"]["projection_weight_scaling"] = "1d"

    with pytest.raises(
        MergeValidationError,
        match="projection_weight_scaling='2d'",
    ):
        merge_route_results(*results)


def test_merge_rejects_d64_nvfp4_represented_backward_configuration(
    matched_results,
):
    results = _as_d64_results(
        matched_results,
        projection_format="nvfp4",
        experimental=True,
    )
    for result in results:
        result["configuration"][
            "mx_backward_match_forward_operands"
        ] = True

    with pytest.raises(
        MergeValidationError,
        match="flags must be false",
    ):
        merge_route_results(*results)


def test_merge_rejects_d64_nvfp4_non_accumulator_backward_contract(
    matched_results,
):
    results = _as_d64_results(
        matched_results,
        projection_format="nvfp4",
        experimental=True,
    )
    mx = results[1]
    mx["configuration"]["backward_route_contracts"][MX_ROUTE][
        "projection"
    ]["per_block_qk_scales"] = True

    with pytest.raises(
        MergeValidationError,
        match="required accumulator-E4M3 D64 publication contract",
    ):
        merge_route_results(*results)


def test_merge_rejects_d64_nvfp4_dispatch_format_mismatch(matched_results):
    results = _as_d64_results(
        matched_results,
        projection_format="nvfp4",
        experimental=True,
    )
    mx = results[1]
    mx["configuration"]["timed_forward_dispatch_contracts"][MX_ROUTE][
        "qkv_projection"
    ]["format"] = "e4m3"

    with pytest.raises(
        MergeValidationError,
        match="qkv_projection.format does not match",
    ):
        merge_route_results(*results)


def test_merge_rejects_d64_nvfp4_publication_topology_mismatch(
    matched_results,
):
    results = _as_d64_results(
        matched_results,
        projection_format="nvfp4",
        experimental=True,
    )
    mx = results[1]
    mx["configuration"]["mx_projection_publication_topology"][
        "qkv_projection_format"
    ] = "e4m3"

    with pytest.raises(
        MergeValidationError,
        match="projection_publication_topology.qkv_projection_format",
    ):
        merge_route_results(*results)


def test_merge_rejects_d64_nvfp4_backward_contract_format_mismatch(
    matched_results,
):
    results = _as_d64_results(
        matched_results,
        projection_format="nvfp4",
        experimental=True,
    )
    mx = results[1]
    mx["configuration"]["backward_route_contracts"][MX_ROUTE]["projection"][
        "qkv_projection_format"
    ] = "e4m3"

    with pytest.raises(
        MergeValidationError,
        match="projection.qkv_projection_format must match experimental D64",
    ):
        merge_route_results(*results)


def test_merge_requires_validated_native_d64_nvfp4_publication(
    matched_results,
):
    results = _as_d64_results(
        matched_results,
        projection_format="nvfp4",
        experimental=True,
    )
    mx = results[1]
    mx["configuration"]["timed_forward_dispatch_contracts"][MX_ROUTE][
        "d64_nvfp4_projection_publication"
    ]["native_publication_validated_by_interface"] = False

    with pytest.raises(
        MergeValidationError,
        match="lacks validated native experimental D64 NVFP4 publication",
    ):
        merge_route_results(*results)


def test_merge_rejects_different_data(matched_results):
    bf16, mx, fp8 = matched_results
    mx = copy.deepcopy(mx)
    mx["configuration"]["train_tokens"]["sha256"] = "f" * 64

    with pytest.raises(MergeValidationError, match="train_tokens.sha256"):
        merge_route_results(bf16, mx, fp8)


def test_merge_rejects_unsupported_hardware(matched_results):
    bf16, mx, fp8 = copy.deepcopy(matched_results)
    mx["configuration"]["hardware_identity"]["name"] = "Different GPU"
    with pytest.raises(MergeValidationError, match="must identify NVIDIA GB200"):
        merge_route_results(bf16, mx, fp8)


def test_merge_rejects_different_initial_state_probe(matched_results):
    bf16, mx, fp8 = matched_results
    fp8 = copy.deepcopy(fp8)
    fp8["configuration"]["initial_state_probe"]["sha256"] = "f" * 64

    with pytest.raises(MergeValidationError, match="initial_state_probe.sha256"):
        merge_route_results(bf16, mx, fp8)


def test_merge_rejects_different_lowp_backward_contract(matched_results):
    bf16, mx, fp8 = matched_results
    fp8 = copy.deepcopy(fp8)
    fp8["configuration"]["backward_route_contracts"][FP8_ROUTE][
        "probability"
    ]["reuse_quantized_p"] = False

    with pytest.raises(
        MergeValidationError, match="probability.reuse_quantized_p"
    ):
        merge_route_results(bf16, mx, fp8)


def test_merge_rejects_wrong_forward_pv_format(matched_results):
    bf16, mx, fp8 = matched_results
    mx = copy.deepcopy(mx)
    mx["configuration"]["mx_forward_topology"]["pv_format"] = "e4m3_fp8"

    with pytest.raises(MergeValidationError, match="pv_format must be"):
        merge_route_results(bf16, mx, fp8)


def test_merge_requires_post_compile_timed_dispatch_contract(matched_results):
    bf16, mx, fp8 = matched_results
    fp8 = copy.deepcopy(fp8)
    fp8["configuration"]["timed_forward_dispatch_contracts"] = None

    with pytest.raises(MergeValidationError, match="must be an object"):
        merge_route_results(bf16, mx, fp8)


def test_merge_rejects_different_shared_projection_artifact(matched_results):
    bf16, mx, fp8 = matched_results
    fp8 = copy.deepcopy(fp8)
    fp8["configuration"]["projection_extension"]["sha256"] = "d" * 64

    with pytest.raises(
        MergeValidationError, match="projection extension content identity"
    ):
        merge_route_results(bf16, mx, fp8)


def test_merge_rejects_non_single_route_input(matched_results):
    bf16, mx, fp8 = matched_results
    mx = copy.deepcopy(mx)
    mx["configuration"]["routes"] = [MX_ROUTE, FP8_ROUTE]

    with pytest.raises(MergeValidationError, match="must select only"):
        merge_route_results(bf16, mx, fp8)


def test_merge_rejects_inconsistent_record_loss(matched_results):
    bf16, mx, fp8 = matched_results
    bf16 = copy.deepcopy(bf16)
    bf16["records"][BF16_ROUTE][1]["loss"] += 0.5

    with pytest.raises(MergeValidationError, match="loss disagrees"):
        merge_route_results(bf16, mx, fp8)


def test_cli_writes_atomic_merged_json(tmp_path, matched_results):
    input_paths = []
    for index, result in enumerate(matched_results):
        path = tmp_path / f"route-{index}.json"
        path.write_text(json.dumps(result))
        input_paths.append(path)
    output = tmp_path / "merged.json"

    return_code = main(
        [
            "--bf16",
            str(input_paths[0]),
            "--mx",
            str(input_paths[1]),
            "--fp8",
            str(input_paths[2]),
            "--output",
            str(output),
        ]
    )

    assert return_code == 0
    merged = json.loads(output.read_text())
    assert merged["schema"] == OUTPUT_SCHEMA
    assert merged["inputs"][BF16_ROUTE]["sha256"]
