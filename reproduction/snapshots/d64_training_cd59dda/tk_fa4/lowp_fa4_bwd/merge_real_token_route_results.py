#!/usr/bin/env python3
"""Validate and merge independent real-token training route results.

Full-depth models do not fit safely as three simultaneous replicas.  The
real-token trainer can therefore run BF16, MXFP4-PV, and FP8-PV in separate
processes.  This module refuses to compare those processes unless their
source, model/training configuration, data, initialization probe, and the two
effective low-precision backward contracts match.

The implementation intentionally uses only the Python standard library so it
can run in a lightweight result-collection container without CUDA or PyTorch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


INPUT_SCHEMA = "llama12b_real_tokens_training_v3"
OUTPUT_SCHEMA = "llama_real_token_independent_routes_merged_v1"
BF16_ROUTE = "bf16_cute"
MX_ROUTE = "nvfp4_qk_mxfp4_pv"
FP8_ROUTE = "nvfp4_qk_fp8_pv_exact"
EXPECTED_ROUTES = (BF16_ROUTE, MX_ROUTE, FP8_ROUTE)
_FORWARD_ROUTE_PROVENANCE = {
    MX_ROUTE: {
        "slot": "mx",
        "extension_key": "mx_forward_extension",
        "topology_key": "mx_forward_topology",
        "extension_route": "real_fwd_tk_hao_direct_nvfp4_mxfp4pv",
        "pv_format": "mxfp4_e8m0_block32",
        "attention_symbol": "forward_hao_direct_fp4pv",
    },
    FP8_ROUTE: {
        "slot": "fp8",
        "extension_key": "fp8_forward_extension",
        "topology_key": "fp8_forward_topology",
        "extension_route": "real_fwd_tk_hao_direct_causal_gqa_nvfp4_fp8pv",
        "pv_format": "e4m3_fp8",
        "attention_symbol": "forward_hao_direct_fp8pv",
    },
}
TIMING_COMPONENTS = (
    "forward_ms",
    "backward_ms",
    "gradient_clip_ms",
    "optimizer_ms",
    "step_ms",
    "wall_ms",
)

# These fields are either process locations or are necessarily populated only
# for the selected route.  Every other configuration field is compared
# recursively.  Effective backward behavior is checked separately using the
# complete contracts, not by ignoring its route-scoped fields.
_COMMON_CONFIG_IGNORED_KEYS = frozenset(
    {
        "routes",
        "python_executable",
        "python_executable_resolved",
        "progress_output",
        "mx_forward_extension",
        "fp8_forward_extension",
        "effective_qk_scale_refresh_every",
        "mx_backward_forward_probability_replay",
        "mx_backward_forward_probability_scale_handoff",
        "mx_probability_replay_provenance",
        "mx_projection_publication_topology",
        "fp8_projection_publication_topology",
        "mx_backward_component_gains",
        "fp8_backward_component_gains",
        "backward_control_provenance",
        "backward_control_route_provenance",
        "backward_route_contracts",
        "matched_lowp_backward_contract",
        "shared_lowp_backward_runner",
        "timed_forward_dispatch_contracts",
        "backward_detached_fp8_p_tmem",
        "backward_probability_tmem_policy",
        "backward_head_fast_raster",
        "backward_raster_policy",
        "backward_exp2_route_policies",
        "backward_detached_fp8_p_tmem_routes",
        "backward_probability_tmem_route_policies",
        "backward_head_fast_rasters",
        "backward_raster_route_policies",
        "mx_effective_backward_match_forward_operands",
        "fp8_effective_backward_match_forward_operands",
        "mx_forward_topology",
        "fp8_forward_topology",
    }
)

_REQUIRED_COMMON_CONFIG_KEYS = (
    "model_preset",
    "layers",
    "hidden",
    "intermediate",
    "q_heads",
    "kv_heads",
    "head_dim",
    "sequence",
    "vocab",
    "batch",
    "rounds",
    "training_batches",
    "validation_batches",
    "eval_every",
    "seed",
    "learning_rate",
    "gradient_clip_norm",
    "optimizer",
    "loss_dtype",
    "corpus_sha256",
    "corpus_documents",
    "corpus_metadata",
    "train_fraction",
    "split_seed",
    "train_documents",
    "validation_documents",
    "tokenizer_sha256",
    "tokenizer_vocab",
    "bos_token_id",
    "eos_token_id",
    "rope_theta",
    "rope_scaling",
    "initialization",
    "initial_state_probe",
    "targets",
    "train_tokens",
    "validation_tokens",
    "parameter_count",
    "hardware_identity",
)

_DATA_CONFIG_KEYS = (
    "corpus_sha256",
    "corpus_documents",
    "corpus_metadata",
    "train_fraction",
    "split_seed",
    "train_documents",
    "validation_documents",
    "tokenizer_sha256",
    "tokenizer_vocab",
    "bos_token_id",
    "eos_token_id",
    "targets",
    "train_tokens",
    "validation_tokens",
)


class MergeValidationError(ValueError):
    """A source result is incomplete or not safe to compare."""


def _reject_json_constant(value: str) -> None:
    raise MergeValidationError(f"JSON contains non-finite constant {value}")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream, parse_constant=_reject_json_constant)
    except (OSError, json.JSONDecodeError) as error:
        raise MergeValidationError(f"could not read {path}: {error}") from error
    if not isinstance(value, dict):
        raise MergeValidationError(f"{path} must contain a JSON object")
    return value


def _file_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        size = path.stat().st_size
    except OSError as error:
        raise MergeValidationError(f"could not fingerprint {path}: {error}") from error
    return {
        "path": str(path.resolve()),
        "bytes": size,
        "sha256": digest.hexdigest(),
    }


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MergeValidationError(f"{label} must be an object")
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise MergeValidationError(f"{label} must be an array")
    return value


def _require_finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MergeValidationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise MergeValidationError(f"{label} must be finite")
    return result


def _require_positive_number(value: Any, label: str) -> float:
    result = _require_finite_number(value, label)
    if result <= 0.0:
        raise MergeValidationError(f"{label} must be positive")
    return result


def _require_finite_tree(value: Any, label: str) -> None:
    """Reject overflow-to-infinity values accepted by the JSON decoder."""
    if isinstance(value, float) and not math.isfinite(value):
        raise MergeValidationError(f"{label} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require_finite_tree(item, f"{label}.{key}")
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, item in enumerate(value):
            _require_finite_tree(item, f"{label}[{index}]")


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise MergeValidationError(f"{label} must be a 64-digit SHA256 hex digest")
    return value.lower()


def _require_keys(
    value: Mapping[str, Any], keys: Sequence[str], label: str
) -> None:
    missing = sorted(set(keys).difference(value))
    if missing:
        raise MergeValidationError(
            f"{label} is missing required fields: {', '.join(missing)}"
        )


def _different_fields(
    reference: Any,
    candidate: Any,
    *,
    prefix: str = "",
) -> list[str]:
    if isinstance(reference, Mapping) and isinstance(candidate, Mapping):
        fields: list[str] = []
        for key in sorted(set(reference) | set(candidate), key=str):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in reference or key not in candidate:
                fields.append(path)
            else:
                fields.extend(
                    _different_fields(
                        reference[key], candidate[key], prefix=path
                    )
                )
        return fields
    if (
        isinstance(reference, Sequence)
        and not isinstance(reference, (str, bytes, bytearray))
        and isinstance(candidate, Sequence)
        and not isinstance(candidate, (str, bytes, bytearray))
    ):
        fields = []
        if len(reference) != len(candidate):
            return [prefix or "<root>"]
        for index, (reference_item, candidate_item) in enumerate(
            zip(reference, candidate, strict=True)
        ):
            path = f"{prefix}[{index}]" if prefix else f"[{index}]"
            fields.extend(
                _different_fields(reference_item, candidate_item, prefix=path)
            )
        return fields
    return [] if reference == candidate else [prefix or "<root>"]


def _require_equal(
    reference: Any,
    candidate: Any,
    *,
    label: str,
    candidate_label: str,
) -> None:
    fields = _different_fields(reference, candidate)
    if fields:
        shown = fields[:20]
        suffix = f" (+{len(fields) - len(shown)} more)" if len(fields) > 20 else ""
        raise MergeValidationError(
            f"{candidate_label} {label} differs at: {', '.join(shown)}{suffix}"
        )


def _canonical_sha256(value: Any) -> str:
    rendered = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


def _common_configuration(configuration: Mapping[str, Any]) -> dict[str, Any]:
    common = {
        key: value
        for key, value in configuration.items()
        if key not in _COMMON_CONFIG_IGNORED_KEYS
    }
    # Content hashes authenticate these files independently of per-container
    # mount paths and route-specific required-symbol lists.
    common.pop("corpus", None)
    common.pop("tokenizer", None)
    common.pop("backward_extension", None)
    common.pop("projection_extension", None)
    return common


def _content_identity(value: Any, label: str) -> dict[str, Any]:
    identity = _require_mapping(value, label)
    _require_keys(identity, ("module", "sha256"), label)
    if not isinstance(identity["module"], str) or not identity["module"]:
        raise MergeValidationError(f"{label}.module must be a nonempty string")
    result = {
        "module": identity["module"],
        "sha256": _require_sha256(identity["sha256"], f"{label}.sha256"),
    }
    if "bytes" in identity:
        if (
            isinstance(identity["bytes"], bool)
            or not isinstance(identity["bytes"], int)
            or identity["bytes"] < 1
        ):
            raise MergeValidationError(f"{label}.bytes must be a positive integer")
        result["bytes"] = identity["bytes"]
    return result


def _hardware_identity(value: Any, label: str) -> dict[str, Any]:
    identity = _require_mapping(value, label)
    required = (
        "schema",
        "visible_device_count",
        "logical_device_index",
        "name",
        "uuid",
        "compute_capability",
        "total_memory_bytes",
        "multiprocessor_count",
        "l2_cache_bytes",
        "pci_domain_id",
        "pci_bus_id",
        "pci_device_id",
        "torch_version",
        "torch_cuda_version",
    )
    _require_keys(identity, required, label)
    if identity["schema"] != "cuda_hardware_identity_v1":
        raise MergeValidationError(f"{label} has an unsupported schema")
    if identity["visible_device_count"] != 1:
        raise MergeValidationError(f"{label} must expose exactly one CUDA device")
    if identity["logical_device_index"] != 0:
        raise MergeValidationError(f"{label} must use logical CUDA device zero")
    if identity["name"] != "NVIDIA GB200":
        raise MergeValidationError(f"{label} must identify NVIDIA GB200")
    if identity["compute_capability"] != [10, 0]:
        raise MergeValidationError(f"{label} must identify SM100")
    if not isinstance(identity["uuid"], str) or not identity["uuid"]:
        raise MergeValidationError(f"{label}.uuid must be a nonempty string")
    for key in (
        "total_memory_bytes",
        "multiprocessor_count",
        "l2_cache_bytes",
    ):
        if (
            isinstance(identity[key], bool)
            or not isinstance(identity[key], int)
            or identity[key] <= 0
        ):
            raise MergeValidationError(f"{label}.{key} must be a positive integer")
    return dict(identity)


def _source_identity(value: Any, label: str) -> dict[str, Any]:
    source = _require_mapping(value, label)
    trainer = _require_mapping(source.get("trainer"), f"{label}.trainer")
    _require_keys(trainer, ("bytes", "sha256"), f"{label}.trainer")
    if (
        isinstance(trainer["bytes"], bool)
        or not isinstance(trainer["bytes"], int)
        or trainer["bytes"] < 1
    ):
        raise MergeValidationError(f"{label}.trainer.bytes must be positive")
    result: dict[str, Any] = {
        "trainer": {
            "bytes": trainer["bytes"],
            "sha256": _require_sha256(
                trainer["sha256"], f"{label}.trainer.sha256"
            ),
        }
    }
    git = _require_mapping(source.get("git"), f"{label}.git")
    if git.get("available") is False:
        return {
            **result,
            "git": {
                "available": False,
                "error_type": git.get("error_type"),
            },
        }
    if git.get("available") is not True:
        raise MergeValidationError(
            f"{label}.git.available must be a boolean"
        )
    _require_keys(
        git,
        (
            "head",
            "tracked_dirty",
            "tracked_diff_bytes",
            "tracked_diff_sha256",
        ),
        f"{label}.git",
    )
    result["git"] = {
        "available": True,
        "head": git["head"],
        "tracked_dirty": git["tracked_dirty"],
        "tracked_diff_bytes": git["tracked_diff_bytes"],
        "tracked_diff_sha256": _require_sha256(
            git["tracked_diff_sha256"], f"{label}.git.tracked_diff_sha256"
        ),
    }
    return result


def _validate_forward_provenance(
    configuration: Mapping[str, Any], route: str, label: str
) -> dict[str, Any]:
    """Authenticate the route-specific forward binary and timed dispatch."""
    if route == BF16_ROUTE:
        for key in (
            "mx_forward_extension",
            "fp8_forward_extension",
            "mx_forward_topology",
            "fp8_forward_topology",
        ):
            if configuration.get(key) is not None:
                raise MergeValidationError(
                    f"{label}.configuration.{key} must be null for BF16"
                )
        timed_contracts = configuration.get("timed_forward_dispatch_contracts")
        if timed_contracts not in (None, {}):
            raise MergeValidationError(
                f"{label} BF16 route must not publish a lowp timed dispatch contract"
            )
        return {
            "route": BF16_ROUTE,
            "implementation": "bf16_cute",
            "forward_extension": None,
            "forward_topology": None,
            "timed_forward_dispatch_contract": None,
        }

    expected = _FORWARD_ROUTE_PROVENANCE[route]
    other_route = FP8_ROUTE if route == MX_ROUTE else MX_ROUTE
    other = _FORWARD_ROUTE_PROVENANCE[other_route]
    if configuration.get(other["extension_key"]) is not None:
        raise MergeValidationError(
            f"{label} unexpectedly publishes {other['extension_key']}"
        )
    if configuration.get(other["topology_key"]) is not None:
        raise MergeValidationError(
            f"{label} unexpectedly publishes {other['topology_key']}"
        )

    extension_label = f"{label}.configuration.{expected['extension_key']}"
    extension = _require_mapping(
        configuration.get(expected["extension_key"]), extension_label
    )
    extension_content = _content_identity(extension, extension_label)
    if "bytes" not in extension_content:
        raise MergeValidationError(f"{extension_label} must record artifact bytes")

    topology_label = f"{label}.configuration.{expected['topology_key']}"
    topology = _require_mapping(
        configuration.get(expected["topology_key"]), topology_label
    )
    expected_topology = {
        "batch": configuration["batch"],
        "seqlen": configuration["sequence"],
        "heads": configuration["q_heads"],
        "kv_heads": configuration["kv_heads"],
        "dqk": configuration["head_dim"],
        "dvo": configuration["head_dim"],
        "causal": True,
        "qk_format": "nvfp4_e4m3_block16",
        "route": expected["extension_route"],
        "pv_format": expected["pv_format"],
    }
    for key, expected_value in expected_topology.items():
        if topology.get(key) != expected_value:
            raise MergeValidationError(
                f"{topology_label}.{key} must be {expected_value!r}, "
                f"got {topology.get(key)!r}"
            )
    if route == FP8_ROUTE and bool(topology.get("causal_interleaved_kv", False)):
        raise MergeValidationError(
            f"{topology_label} must use ordinary causal K/V order for FP8-PV"
        )
    if configuration["head_dim"] == 128 and bool(
        topology.get("causal_interleaved_kv", False)
    ):
        raise MergeValidationError(
            f"{topology_label} must use ordinary causal K/V order for D128"
        )

    projection_format = configuration[
        f"{expected['slot']}_qkv_projection_format"
    ]
    production_projection_format = (
        "e4m3" if configuration["head_dim"] == 64 else "nvfp4"
    )
    if projection_format != production_projection_format:
        raise MergeValidationError(
            f"{label} {route} projection format must be "
            f"{production_projection_format!r} for D{configuration['head_dim']}"
        )

    contracts_label = f"{label}.configuration.timed_forward_dispatch_contracts"
    contracts = _require_mapping(
        configuration.get("timed_forward_dispatch_contracts"), contracts_label
    )
    if set(contracts) != {route}:
        raise MergeValidationError(
            f"{contracts_label} must contain exactly {route!r}"
        )
    dispatch_label = f"{contracts_label}.{route}"
    dispatch = _require_mapping(contracts[route], dispatch_label)
    if dispatch.get("schema") != "lowp_forward_dispatch_contract_v2":
        raise MergeValidationError(
            f"{dispatch_label} has an unsupported dispatch-contract schema"
        )
    for key, expected_value in (
        ("route", expected["extension_route"]),
        ("pv_format", expected["pv_format"]),
        ("validated_after_compile_before_timing", True),
    ):
        if dispatch.get(key) != expected_value:
            raise MergeValidationError(
                f"{dispatch_label}.{key} must be {expected_value!r}, "
                f"got {dispatch.get(key)!r}"
            )
    projection = _require_mapping(
        dispatch.get("qkv_projection"), f"{dispatch_label}.qkv_projection"
    )
    if projection.get("format") != production_projection_format:
        raise MergeValidationError(
            f"{dispatch_label}.qkv_projection.format does not match the "
            "production projection format"
        )
    if projection.get("runtime_crossover_reallocation") is not False:
        raise MergeValidationError(
            f"{dispatch_label} permits runtime projection reallocation"
        )
    attention = _require_mapping(
        dispatch.get("attention"), f"{dispatch_label}.attention"
    )
    attention_symbol = expected["attention_symbol"]
    if route == MX_ROUTE and configuration.get(
        "mx_backward_forward_probability_scale_handoff"
    ):
        attention_symbol = "forward_hao_direct_fp4pv_with_p_scales"
    expected_attention = {
        "dispatch": "construction_bound_route_specific_entrypoint",
        "symbol": attention_symbol,
        "entrypoint_bound_at_construction": True,
        "launcher_bound_to_runtime": True,
    }
    for key, expected_value in expected_attention.items():
        if attention.get(key) != expected_value:
            raise MergeValidationError(
                f"{dispatch_label}.attention.{key} must be {expected_value!r}, "
                f"got {attention.get(key)!r}"
            )
    if configuration["head_dim"] == 128:
        publication = _require_mapping(
            dispatch.get("d128_projection_publication"),
            f"{dispatch_label}.d128_projection_publication",
        )
        if publication.get("native_publication_validated_by_interface") is not True:
            raise MergeValidationError(
                f"{dispatch_label} lacks validated native D128 publication"
            )

    return {
        "route": route,
        "extension_identity": dict(extension),
        "extension_content_identity": extension_content,
        "forward_topology": dict(topology),
        "forward_topology_sha256": _canonical_sha256(topology),
        "timed_forward_dispatch_contract": dict(dispatch),
        "timed_forward_dispatch_contract_sha256": _canonical_sha256(dispatch),
        "validated_route": expected["extension_route"],
        "validated_qk_format": "nvfp4_e4m3_block16",
        "validated_pv_format": expected["pv_format"],
        "validated_projection_format": production_projection_format,
    }


def _validate_route_result(
    result: Mapping[str, Any], expected_route: str, label: str
) -> dict[str, Any]:
    _require_finite_tree(result, label)
    if result.get("schema") != INPUT_SCHEMA:
        raise MergeValidationError(
            f"{label}.schema must be {INPUT_SCHEMA!r}, got {result.get('schema')!r}"
        )
    configuration = _require_mapping(
        result.get("configuration"), f"{label}.configuration"
    )
    _require_keys(
        configuration, _REQUIRED_COMMON_CONFIG_KEYS, f"{label}.configuration"
    )
    selected_routes = _require_list(
        configuration.get("routes"), f"{label}.configuration.routes"
    )
    if selected_routes != [expected_route]:
        raise MergeValidationError(
            f"{label} must select only {expected_route!r}; got {selected_routes!r}"
        )

    route_summaries = _require_mapping(result.get("routes"), f"{label}.routes")
    if set(route_summaries) != {expected_route}:
        raise MergeValidationError(
            f"{label}.routes must contain only {expected_route!r}"
        )
    summary = _require_mapping(
        route_summaries[expected_route], f"{label}.routes.{expected_route}"
    )
    timing = _require_mapping(
        summary.get("timing"), f"{label}.routes.{expected_route}.timing"
    )
    _require_keys(timing, TIMING_COMPONENTS, f"{label}.{expected_route}.timing")
    for component in TIMING_COMPONENTS:
        number_label = f"{label}.{expected_route}.timing.{component}"
        if component in {"gradient_clip_ms"}:
            value = _require_finite_number(timing[component], number_label)
            if value < 0.0:
                raise MergeValidationError(f"{number_label} must be non-negative")
        else:
            _require_positive_number(timing[component], number_label)
    timed_records = timing.get("timed_records")
    if isinstance(timed_records, bool) or not isinstance(timed_records, int):
        raise MergeValidationError(
            f"{label}.{expected_route}.timing.timed_records must be an integer"
        )
    if timed_records < 1:
        raise MergeValidationError(
            f"{label}.{expected_route}.timing has no timed records"
        )

    training = _require_mapping(
        summary.get("training"), f"{label}.routes.{expected_route}.training"
    )
    _require_keys(
        training,
        (
            "losses",
            "first_loss",
            "last_loss",
            "last_eight_mean",
            "minimum_loss",
            "all_steps_finite",
        ),
        f"{label}.{expected_route}.training",
    )
    if training["all_steps_finite"] is not True:
        raise MergeValidationError(
            f"{label}.{expected_route} did not keep all training steps finite"
        )
    losses = _require_list(
        training["losses"], f"{label}.{expected_route}.training.losses"
    )
    rounds = configuration["rounds"]
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 1:
        raise MergeValidationError(f"{label}.configuration.rounds is invalid")
    if len(losses) != rounds:
        raise MergeValidationError(
            f"{label}.{expected_route} has {len(losses)} losses for {rounds} rounds"
        )
    finite_losses = [
        _require_finite_number(loss, f"{label}.{expected_route}.losses[{index}]")
        for index, loss in enumerate(losses)
    ]
    for key in ("first_loss", "last_loss", "last_eight_mean", "minimum_loss"):
        _require_finite_number(training[key], f"{label}.{expected_route}.{key}")
    expected_loss_values = {
        "first_loss": finite_losses[0],
        "last_loss": finite_losses[-1],
        "last_eight_mean": sum(finite_losses[-8:]) / len(finite_losses[-8:]),
        "minimum_loss": min(finite_losses),
    }
    for key, expected in expected_loss_values.items():
        if not math.isclose(
            float(training[key]), expected, rel_tol=1.0e-12, abs_tol=1.0e-12
        ):
            raise MergeValidationError(
                f"{label}.{expected_route}.training.{key} is inconsistent with losses"
            )

    records_by_route = _require_mapping(result.get("records"), f"{label}.records")
    if set(records_by_route) != {expected_route}:
        raise MergeValidationError(
            f"{label}.records must contain only {expected_route!r}"
        )
    records = _require_list(
        records_by_route[expected_route], f"{label}.records.{expected_route}"
    )
    if len(records) != rounds:
        raise MergeValidationError(
            f"{label}.{expected_route} has {len(records)} records for {rounds} rounds"
        )
    training_batches = configuration["training_batches"]
    if (
        isinstance(training_batches, bool)
        or not isinstance(training_batches, int)
        or training_batches < 1
    ):
        raise MergeValidationError(
            f"{label}.configuration.training_batches is invalid"
        )
    for index, (record_value, loss) in enumerate(
        zip(records, finite_losses, strict=True)
    ):
        record = _require_mapping(
            record_value, f"{label}.records.{expected_route}[{index}]"
        )
        if record.get("round") != index:
            raise MergeValidationError(
                f"{label}.{expected_route} record {index} has wrong round"
            )
        if record.get("batch") != index % training_batches:
            raise MergeValidationError(
                f"{label}.{expected_route} record {index} has wrong batch"
            )
        record_loss = _require_finite_number(
            record.get("loss"), f"{label}.{expected_route}.records[{index}].loss"
        )
        if record_loss != loss:
            raise MergeValidationError(
                f"{label}.{expected_route} record {index} loss disagrees with summary"
            )
        if record.get("finite") is not True:
            raise MergeValidationError(
                f"{label}.{expected_route} record {index} is not finite"
            )

    validation = _require_mapping(
        summary.get("validation"), f"{label}.routes.{expected_route}.validation"
    )
    _require_keys(
        validation,
        ("initial_loss", "final_loss"),
        f"{label}.{expected_route}.validation",
    )
    initial_validation = _require_finite_number(
        validation["initial_loss"],
        f"{label}.{expected_route}.validation.initial_loss",
    )
    final_validation = _require_finite_number(
        validation["final_loss"],
        f"{label}.{expected_route}.validation.final_loss",
    )
    history = _require_list(
        result.get("validation_history"), f"{label}.validation_history"
    )
    if len(history) < 2:
        raise MergeValidationError(
            f"{label}.validation_history must contain initial and final evaluations"
        )
    history_rounds: list[int] = []
    history_losses: list[float] = []
    for index, event_value in enumerate(history):
        event = _require_mapping(event_value, f"{label}.validation_history[{index}]")
        round_index = event.get("round")
        if isinstance(round_index, bool) or not isinstance(round_index, int):
            raise MergeValidationError(
                f"{label}.validation_history[{index}].round must be an integer"
            )
        event_routes = _require_mapping(
            event.get("routes"), f"{label}.validation_history[{index}].routes"
        )
        if set(event_routes) != {expected_route}:
            raise MergeValidationError(
                f"{label}.validation_history[{index}] has unexpected routes"
            )
        route_event = _require_mapping(
            event_routes[expected_route],
            f"{label}.validation_history[{index}].routes.{expected_route}",
        )
        history_rounds.append(round_index)
        history_losses.append(
            _require_finite_number(
                route_event.get("mean_loss"),
                f"{label}.validation_history[{index}].mean_loss",
            )
        )
    if history_rounds[0] != -1 or history_rounds[-1] != rounds - 1:
        raise MergeValidationError(
            f"{label}.validation_history does not span initial through final round"
        )
    if initial_validation != history_losses[0] or final_validation != history_losses[-1]:
        raise MergeValidationError(
            f"{label}.{expected_route} validation summary disagrees with history"
        )

    initial_probe = _require_mapping(
        configuration["initial_state_probe"],
        f"{label}.configuration.initial_state_probe",
    )
    _require_keys(
        initial_probe,
        ("schema", "tensor_count", "sampled_values", "sha256"),
        f"{label}.configuration.initial_state_probe",
    )
    if initial_probe["schema"] != "state_dict_sparse_probe_v1":
        raise MergeValidationError(
            f"{label} has an unsupported initial-state probe schema"
        )
    _require_sha256(
        initial_probe["sha256"],
        f"{label}.configuration.initial_state_probe.sha256",
    )
    _require_sha256(
        configuration["corpus_sha256"],
        f"{label}.configuration.corpus_sha256",
    )
    _require_sha256(
        configuration["tokenizer_sha256"],
        f"{label}.configuration.tokenizer_sha256",
    )
    for stream_name in ("train_tokens", "validation_tokens"):
        stream_identity = _require_mapping(
            configuration[stream_name], f"{label}.configuration.{stream_name}"
        )
        _require_sha256(
            stream_identity.get("sha256"),
            f"{label}.configuration.{stream_name}.sha256",
        )
    hardware_identity = _hardware_identity(
        configuration["hardware_identity"],
        f"{label}.configuration.hardware_identity",
    )

    return {
        "configuration": configuration,
        "summary": summary,
        "records": records,
        "history": history,
        "history_rounds": history_rounds,
        "history_losses": history_losses,
        "source_identity": _source_identity(result.get("source"), f"{label}.source"),
        "hardware_identity": hardware_identity,
    }


def _timing_pair(
    subject: Mapping[str, Any], reference: Mapping[str, Any]
) -> dict[str, Any]:
    components: dict[str, Any] = {}
    for component in TIMING_COMPONENTS:
        subject_ms = float(subject[component])
        reference_ms = float(reference[component])
        if subject_ms <= 0.0 or reference_ms <= 0.0:
            # Gradient clipping is exactly zero when clipping is disabled.
            if component == "gradient_clip_ms" and subject_ms == reference_ms == 0.0:
                components[component] = {
                    "subject_ms": 0.0,
                    "reference_ms": 0.0,
                    "milliseconds_saved": 0.0,
                    "speedup": None,
                    "time_reduction_percent": None,
                }
                continue
            raise MergeValidationError(
                f"cannot calculate a timing ratio for {component}"
            )
        components[component] = {
            "subject_ms": subject_ms,
            "reference_ms": reference_ms,
            "milliseconds_saved": reference_ms - subject_ms,
            "speedup": reference_ms / subject_ms,
            "time_reduction_percent": 100.0 * (1.0 - subject_ms / reference_ms),
        }
    return {
        "e2e": components["step_ms"],
        "components": components,
    }


def _loss_pair(
    subject: Mapping[str, Any],
    reference: Mapping[str, Any],
    subject_losses: Sequence[float],
    reference_losses: Sequence[float],
    subject_history: Sequence[float],
    reference_history: Sequence[float],
) -> dict[str, Any]:
    if len(subject_losses) != len(reference_losses):
        raise MergeValidationError("matched training loss histories have unequal lengths")
    if len(subject_history) != len(reference_history):
        raise MergeValidationError("matched validation histories have unequal lengths")
    deltas = [
        float(subject_loss) - float(reference_loss)
        for subject_loss, reference_loss in zip(
            subject_losses, reference_losses, strict=True
        )
    ]
    validation_deltas = [
        float(subject_loss) - float(reference_loss)
        for subject_loss, reference_loss in zip(
            subject_history, reference_history, strict=True
        )
    ]
    subject_training = _require_mapping(subject["training"], "subject.training")
    reference_training = _require_mapping(
        reference["training"], "reference.training"
    )
    subject_validation = _require_mapping(subject["validation"], "subject.validation")
    reference_validation = _require_mapping(
        reference["validation"], "reference.validation"
    )
    reference_final_validation = float(reference_validation["final_loss"])
    return {
        "training": {
            "aligned_steps": len(deltas),
            "first_loss_delta": deltas[0],
            "last_loss_delta": deltas[-1],
            "mean_loss_delta": sum(deltas) / len(deltas),
            "mean_absolute_loss_delta": sum(abs(value) for value in deltas)
            / len(deltas),
            "maximum_absolute_loss_delta": max(abs(value) for value in deltas),
            "last_eight_mean_loss_delta": float(
                subject_training["last_eight_mean"]
            )
            - float(reference_training["last_eight_mean"]),
        },
        "validation": {
            "aligned_evaluations": len(validation_deltas),
            "initial_loss_delta": validation_deltas[0],
            "final_loss_delta": validation_deltas[-1],
            "mean_loss_delta": sum(validation_deltas) / len(validation_deltas),
            "maximum_absolute_loss_delta": max(
                abs(value) for value in validation_deltas
            ),
            "final_loss_ratio": (
                float(subject_validation["final_loss"])
                / reference_final_validation
                if reference_final_validation != 0.0
                else None
            ),
        },
    }


def merge_route_results(
    bf16_result: Mapping[str, Any],
    mx_result: Mapping[str, Any],
    fp8_result: Mapping[str, Any],
    *,
    input_identities: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and merge three independent single-route trainer results."""
    source_results = {
        BF16_ROUTE: bf16_result,
        MX_ROUTE: mx_result,
        FP8_ROUTE: fp8_result,
    }
    validated = {
        route: _validate_route_result(result, route, route)
        for route, result in source_results.items()
    }
    forward_provenance = {
        route: _validate_forward_provenance(
            validated[route]["configuration"], route, route
        )
        for route in EXPECTED_ROUTES
    }

    reference_common = _common_configuration(
        validated[BF16_ROUTE]["configuration"]
    )
    for route in (MX_ROUTE, FP8_ROUTE):
        candidate_common = _common_configuration(
            validated[route]["configuration"]
        )
        _require_equal(
            reference_common,
            candidate_common,
            label="common configuration",
            candidate_label=route,
        )

    reference_source = validated[BF16_ROUTE]["source_identity"]
    for route in (MX_ROUTE, FP8_ROUTE):
        _require_equal(
            reference_source,
            validated[route]["source_identity"],
            label="source identity",
            candidate_label=route,
        )

    reference_configuration = validated[BF16_ROUTE]["configuration"]
    data_identity = {
        key: reference_configuration[key] for key in _DATA_CONFIG_KEYS
    }
    initial_state_probe = reference_configuration["initial_state_probe"]

    backward_extension = _content_identity(
        reference_configuration.get("backward_extension"),
        f"{BF16_ROUTE}.configuration.backward_extension",
    )
    projection_extension = _content_identity(
        reference_configuration.get("projection_extension"),
        f"{BF16_ROUTE}.configuration.projection_extension",
    )
    for route in (MX_ROUTE, FP8_ROUTE):
        configuration = validated[route]["configuration"]
        _require_equal(
            backward_extension,
            _content_identity(
                configuration.get("backward_extension"),
                f"{route}.configuration.backward_extension",
            ),
            label="backward extension content identity",
            candidate_label=route,
        )
        _require_equal(
            projection_extension,
            _content_identity(
                configuration.get("projection_extension"),
                f"{route}.configuration.projection_extension",
            ),
            label="projection extension content identity",
            candidate_label=route,
        )

    lowp_contracts: dict[str, Mapping[str, Any]] = {}
    for route in (MX_ROUTE, FP8_ROUTE):
        contracts = _require_mapping(
            validated[route]["configuration"].get("backward_route_contracts"),
            f"{route}.configuration.backward_route_contracts",
        )
        if set(contracts) != {route}:
            raise MergeValidationError(
                f"{route} must publish exactly its own backward contract"
            )
        contract = _require_mapping(
            contracts[route], f"{route}.backward_route_contracts.{route}"
        )
        if contract.get("schema") != "lowp_backward_contract_v1":
            raise MergeValidationError(
                f"{route} has an unsupported low-precision backward contract"
            )
        lowp_contracts[route] = contract
    _require_equal(
        lowp_contracts[MX_ROUTE],
        lowp_contracts[FP8_ROUTE],
        label="effective low-precision backward contract",
        candidate_label=FP8_ROUTE,
    )

    reference_rounds = validated[BF16_ROUTE]["history_rounds"]
    for route in (MX_ROUTE, FP8_ROUTE):
        _require_equal(
            reference_rounds,
            validated[route]["history_rounds"],
            label="validation evaluation rounds",
            candidate_label=route,
        )

    route_summaries = {
        route: dict(validated[route]["summary"])
        for route in EXPECTED_ROUTES
    }
    timing_comparisons = {
        "by_route_vs_bf16": {
            route: _timing_pair(
                _require_mapping(route_summaries[route]["timing"], "timing"),
                _require_mapping(
                    route_summaries[BF16_ROUTE]["timing"], "bf16 timing"
                ),
            )
            for route in (MX_ROUTE, FP8_ROUTE)
        },
        "mxfp4_vs_fp8": _timing_pair(
            _require_mapping(route_summaries[MX_ROUTE]["timing"], "MX timing"),
            _require_mapping(route_summaries[FP8_ROUTE]["timing"], "FP8 timing"),
        ),
    }

    def loss_comparison(subject_route: str, reference_route: str) -> dict[str, Any]:
        return _loss_pair(
            route_summaries[subject_route],
            route_summaries[reference_route],
            validated[subject_route]["summary"]["training"]["losses"],
            validated[reference_route]["summary"]["training"]["losses"],
            validated[subject_route]["history_losses"],
            validated[reference_route]["history_losses"],
        )

    loss_comparisons = {
        "by_route_vs_bf16": {
            route: loss_comparison(route, BF16_ROUTE)
            for route in (MX_ROUTE, FP8_ROUTE)
        },
        "mxfp4_vs_fp8": loss_comparison(MX_ROUTE, FP8_ROUTE),
    }

    validation_history: list[dict[str, Any]] = []
    for index, round_index in enumerate(reference_rounds):
        validation_history.append(
            {
                "round": round_index,
                "routes": {
                    route: validated[route]["history"][index]["routes"][route]
                    for route in EXPECTED_ROUTES
                },
            }
        )

    return {
        "schema": OUTPUT_SCHEMA,
        "comparison_design": (
            "matched deterministic single-route processes; medians are "
            "independent-process, not interleaved paired samples"
        ),
        "inputs": dict(input_identities or {}),
        "validation": {
            "matched": True,
            "routes": list(EXPECTED_ROUTES),
            "source_identity": reference_source,
            "common_configuration_sha256": _canonical_sha256(reference_common),
            "data_identity_sha256": _canonical_sha256(data_identity),
            "initial_state_probe": initial_state_probe,
            "backward_extension": backward_extension,
            "projection_extension": projection_extension,
            "lowp_forward_provenance_verified": True,
            "lowp_backward_contract_matched": True,
            "lowp_backward_contract_sha256": _canonical_sha256(
                lowp_contracts[MX_ROUTE]
            ),
            "hardware_identity_verified": True,
            "hardware_identity": validated[BF16_ROUTE]["hardware_identity"],
        },
        "matched_configuration": reference_common,
        "data_identity": data_identity,
        "forward_provenance": forward_provenance,
        "lowp_backward_contract": lowp_contracts[MX_ROUTE],
        "routes": route_summaries,
        "timing_comparisons": timing_comparisons,
        "loss_comparisons": loss_comparisons,
        "validation_history": validation_history,
        "records": {
            route: validated[route]["records"] for route in EXPECTED_ROUTES
        },
        "memory_by_route_process": {
            route: source_results[route].get("memory") for route in EXPECTED_ROUTES
        },
    }


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf16", type=Path, required=True)
    parser.add_argument("--mx", type=Path, required=True)
    parser.add_argument("--fp8", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="write atomically to this path; omit to print JSON to stdout",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    paths = {
        BF16_ROUTE: args.bf16,
        MX_ROUTE: args.mx,
        FP8_ROUTE: args.fp8,
    }
    try:
        identities = {route: _file_identity(path) for route, path in paths.items()}
        merged = merge_route_results(
            _load_json(args.bf16),
            _load_json(args.mx),
            _load_json(args.fp8),
            input_identities=identities,
        )
        if args.output is None:
            json.dump(merged, sys.stdout, indent=2, sort_keys=True, allow_nan=False)
            sys.stdout.write("\n")
        else:
            _atomic_write_json(args.output, merged)
    except MergeValidationError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
