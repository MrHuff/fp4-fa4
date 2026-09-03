from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch

from tk_fa4.lowp_fa4_bwd import diagnose_llama12b_gradient_floor as diagnostic


def _identity(name: str) -> dict[str, Any]:
    return {
        "path": f"/authenticated/{name}",
        "sha256": hashlib.sha256(name.encode()).hexdigest(),
        "bytes": len(name),
    }


def _valid_benchmark_payloads() -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    model_configuration = {"batch": 16, "sequence": 4096, "layers": 16}
    data = {
        "kind": "pinned_local_dolma_jsonl",
        "updates_including_probe": 24,
        "batch": 16,
        "sequence": 4096,
    }
    checkpoint = _identity("final.pt")
    sample_identity = _identity("samples.pt")
    artifacts = {
        "forward": _identity("forward.so"),
        "projection": _identity("projection.so"),
        "control": _identity("control.py"),
    }
    sources = {
        "harness": _identity("benchmark.py"),
        "runtime": _identity("runtime.py"),
    }
    initial_checkpoint = _identity("initial.pt")
    payload = {
        "schema": diagnostic.BENCHMARK_SCHEMA,
        "route": "fp8",
        "configuration": {
            **model_configuration,
            "loss_scale": 65_536.0,
            "token_source": "dolma",
            "warmups": 3,
            "measured_updates": 20,
        },
        "data": data,
        "checkpoint": initial_checkpoint,
        "final_checkpoint": {
            **checkpoint,
            "kind": "post_trajectory_model_state",
            "serialized_state_layout": "canonical_split_qkv",
            "runtime_state_layout": "split_qkv",
        },
        "sample_artifact": sample_identity,
        "source_files": sources,
        "artifacts": artifacts,
        "forward_topology": {"valid": 1, "route": "fp8"},
        "backward_contract": {
            "schema": "lowp_backward_contract_v1",
            "scaling": {"loss_scale": 65_536.0},
        },
        "heldout_loss": {"initial": 12.25, "final": 8.0},
    }
    sample = {
        "schema": diagnostic.SAMPLE_SCHEMA,
        "route": "fp8",
        "comparison_identity": {
            "seed": 20260825,
            "configuration": model_configuration,
            "data": data,
            "checkpoint_sha256": initial_checkpoint["sha256"],
            "checkpoint_bytes": initial_checkpoint["bytes"],
        },
        "checkpoint": initial_checkpoint,
        "initial_diagnostic": {"loss": 12.25},
        "final_diagnostic": {"loss": 8.0},
        "losses": [12.0] * 23,
    }
    context = {
        "route": "fp8",
        "seed": 20260825,
        "model_configuration": model_configuration,
        "artifacts": artifacts,
        "checkpoint": checkpoint,
        "sources": sources,
        "sample_identity": sample_identity,
    }
    return payload, sample, context


def _statistics(sample_hash: str) -> dict[str, Any]:
    values = [1.0, 2.0, -3.0]
    if sample_hash == "different":
        values = [-3.0, 2.0, 1.0]
    return {
        "nonzero": len(values),
        "strided_sample": {
            "values_sha256": sample_hash,
            "values": values,
        },
    }


def _probe(
    scale: float,
    repeat: int,
    *,
    loss: float = 7.5,
    suffix: str = "stable",
) -> dict[str, Any]:
    return {
        "loss_scale": scale,
        "repeat": repeat,
        "loss": loss,
        "backward_contract": {
            "schema": "lowp_backward_contract_v1",
            "probability": {"fp8_ds_lift": 16},
            "scaling": {"loss_scale": scale, "gradient_global_scale": 2**-8},
        },
        "raw_first_reverse": {
            "decoder_layer": 15,
            "dq": _statistics(f"dq-{scale}-{suffix}"),
            "dk": _statistics(f"dk-{scale}-{suffix}"),
            "dv": _statistics(f"dv-{scale}-{suffix}"),
        },
        "projection_weight_gradients": {
            "layers.15.attention.weights.q": _statistics(
                f"q-{scale}-{suffix}"
            )
        },
    }


def test_loss_scale_grid_requires_distinct_positive_powers_of_two() -> None:
    diagnostic._validate_loss_scales([65_536.0, 131_072.0, 262_144.0])

    for invalid in (
        [65_536.0],
        [65_536.0, 65_536.0],
        [65_536.0, 98_304.0],
        [65_536.0, math.inf],
        [65_536.0, -131_072.0],
    ):
        with pytest.raises(ValueError):
            diagnostic._validate_loss_scales(invalid)


def test_tensor_statistics_are_full_chunked_and_strided_deterministically() -> None:
    tensor = torch.arange(-8, 8, dtype=torch.bfloat16).reshape(4, 4)
    first = diagnostic._tensor_statistics(
        tensor,
        chunk_elements=3,
        sample_elements=4,
    )
    second = diagnostic._tensor_statistics(
        tensor.clone(),
        chunk_elements=5,
        sample_elements=4,
    )

    assert first["shape"] == [4, 4]
    assert first["dtype"] == "torch.bfloat16"
    assert first["elements"] == 16
    assert first["nonzero"] == 15
    assert first["finite"] is True
    assert first["l2"] == pytest.approx(float(tensor.float().norm()))
    assert first["max_abs"] == 8.0
    assert first["mean_abs"] == pytest.approx(float(tensor.float().abs().mean()))
    assert first["full_statistics_chunk_elements"] == 3
    assert first["strided_sample"]["values"] == [-8.0, -4.0, 0.0, 4.0]
    assert (
        first["strided_sample"]["indices_sha256"]
        == second["strided_sample"]["indices_sha256"]
    )
    assert (
        first["strided_sample"]["values_sha256"]
        == second["strided_sample"]["values_sha256"]
    )

    with pytest.raises(RuntimeError, match="implicit full-tensor copy"):
        diagnostic._tensor_statistics(tensor.transpose(0, 1))


def test_benchmark_payload_binds_route_seed_config_checkpoint_and_heldout() -> None:
    payload, sample, context = _valid_benchmark_payloads()
    diagnostic._validate_benchmark_payload(payload, sample, **context)

    def isolated_copies() -> tuple[
        dict[str, Any], dict[str, Any], dict[str, Any]
    ]:
        # Keep the independently observed identities separate from the
        # untrusted receipts even though the fixture initially shares values.
        return copy.deepcopy(payload), copy.deepcopy(sample), copy.deepcopy(context)

    corruptions = []
    changed = isolated_copies()
    changed[0]["route"] = "mx"
    corruptions.append(changed)
    changed = isolated_copies()
    changed[1]["comparison_identity"]["seed"] += 1
    corruptions.append(changed)
    changed = isolated_copies()
    changed[0]["configuration"]["sequence"] = 2048
    corruptions.append(changed)
    changed = isolated_copies()
    changed[0]["final_checkpoint"]["sha256"] = "0" * 64
    corruptions.append(changed)
    changed = isolated_copies()
    changed[0]["heldout_loss"]["final"] += 0.25
    corruptions.append(changed)
    changed = isolated_copies()
    changed[0]["source_files"]["runtime"]["sha256"] = "f" * 64
    corruptions.append(changed)

    for bad_payload, bad_sample, bad_context in corruptions:
        with pytest.raises(RuntimeError):
            diagnostic._validate_benchmark_payload(
                bad_payload,
                bad_sample,
                **bad_context,
            )


def test_probe_grid_gates_losses_repeats_and_each_backward_contract() -> None:
    scales = [65_536.0, 131_072.0]
    probes = [
        _probe(scale, repeat)
        for scale in scales
        for repeat in range(2)
    ]
    benchmark_contract = copy.deepcopy(probes[0]["backward_contract"])
    gates = diagnostic._validate_probe_grid(
        probes,
        loss_scales=scales,
        repeats=2,
        benchmark_contract=benchmark_contract,
    )
    assert gates["loss_invariant"] is True
    assert gates["gradient_sample_hashes_exactly_repeatable"] is True
    assert gates["gradient_samples_directionally_stable"] is True
    assert gates["contracts_match_benchmark_except_loss_scale"] is True

    changed_loss = copy.deepcopy(probes)
    changed_loss[-1]["loss"] += 1.0e-3
    with pytest.raises(RuntimeError, match="forward loss changed"):
        diagnostic._validate_probe_grid(
            changed_loss,
            loss_scales=scales,
            repeats=2,
            benchmark_contract=benchmark_contract,
        )

    changed_repeat = copy.deepcopy(probes)
    changed_repeat[1]["raw_first_reverse"]["dq"] = _statistics("different")
    with pytest.raises(RuntimeError, match="not directionally stable"):
        diagnostic._validate_probe_grid(
            changed_repeat,
            loss_scales=scales,
            repeats=2,
            benchmark_contract=benchmark_contract,
        )

    changed_contract = copy.deepcopy(probes)
    changed_contract[-1]["backward_contract"]["probability"]["fp8_ds_lift"] = 32
    with pytest.raises(RuntimeError, match="differs from benchmark"):
        diagnostic._validate_probe_grid(
            changed_contract,
            loss_scales=scales,
            repeats=2,
            benchmark_contract=benchmark_contract,
        )


def test_regular_file_authentication_and_atomic_json_refuse_aliases_and_overwrite(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"authenticated bytes")
    identity = diagnostic._authenticate_regular_file(source, label="fixture")
    assert identity["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()

    alias = tmp_path / "alias.bin"
    alias.symlink_to(source)
    with pytest.raises(RuntimeError, match="non-symlink"):
        diagnostic._authenticate_regular_file(alias, label="fixture")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        diagnostic._authenticate_regular_file(
            source,
            label="fixture",
            expected_sha256="0" * 64,
        )

    output = tmp_path / "nested" / "diagnostic.json"
    diagnostic._atomic_write_json(output, {"complete": True, "value": 4})
    assert json.loads(output.read_text()) == {"complete": True, "value": 4}
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        diagnostic._atomic_write_json(output, {"complete": False})
    assert not list(output.parent.glob(".*.tmp.*"))
