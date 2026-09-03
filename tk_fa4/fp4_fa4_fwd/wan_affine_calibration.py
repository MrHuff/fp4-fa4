#!/usr/bin/env python3
"""Fit a frozen per-layer Wan affine E2M1 code map from shadow runs.

Each candidate shadow run must follow the same BF16 trajectory.  The low-
precision result is observed at every self-attention layer, but BF16 is fed to
the next layer.  Candidate comparisons are therefore paired and do not drift
because an earlier low-precision error changed a later layer's input.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from eval_regular_attention import parse_layer_indices


@dataclass(frozen=True)
class Candidate:
    label: str
    affine_a: float
    affine_b: float
    shadow_path: Path
    manifest_path: Path
    shadow: dict[str, Any]
    manifest: dict[str, Any]


def parse_candidate(specification: str) -> tuple[str, float, float, Path, Path]:
    try:
        label, remainder = specification.split("=", 1)
        a_text, b_text, shadow_text, manifest_text = remainder.split(",", 3)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "candidate must be LABEL=A,B,SHADOW_JSON,MANIFEST_JSON"
        ) from error
    if not label:
        raise argparse.ArgumentTypeError("candidate label cannot be empty")
    return (
        label,
        float(a_text),
        float(b_text),
        Path(shadow_text).resolve(),
        Path(manifest_text).resolve(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        type=parse_candidate,
        metavar="LABEL=A,B,SHADOW_JSON,MANIFEST_JSON",
    )
    parser.add_argument(
        "--base",
        required=True,
        help="Candidate label used for layers without a retained override.",
    )
    parser.add_argument(
        "--max-codebook",
        type=int,
        default=4,
        help="Maximum number of distinct affine pairs, including the base.",
    )
    parser.add_argument(
        "--minimum-improvement",
        type=float,
        default=1e-4,
        help=(
            "Minimum absolute aggregate relative-L2 improvement required to "
            "route a layer away from the base candidate."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--policy", default="fast")
    return parser.parse_args()


def load_candidates(
    specifications: list[tuple[str, float, float, Path, Path]],
) -> list[Candidate]:
    candidates = []
    seen = set()
    for label, affine_a, affine_b, shadow_path, manifest_path in specifications:
        if label in seen:
            raise ValueError(f"duplicate candidate label: {label}")
        seen.add(label)
        candidates.append(
            Candidate(
                label=label,
                affine_a=affine_a,
                affine_b=affine_b,
                shadow_path=shadow_path,
                manifest_path=manifest_path,
                shadow=json.loads(shadow_path.read_text()),
                manifest=json.loads(manifest_path.read_text()),
            )
        )
    return candidates


def shadow_provider(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        provider = payload["providers"]["tk-shadow"]
    except KeyError as error:
        raise ValueError("shadow JSON is missing providers.tk-shadow") from error
    if provider.get("status") != "complete":
        raise ValueError(f"shadow provider did not complete: {provider}")
    return provider


def calibration_identity(payload: dict[str, Any]) -> dict[str, Any]:
    generation = payload["generation"]
    return {
        "model": payload["model"],
        "prompt": payload["prompt"],
        "negative_prompt": payload.get("negative_prompt"),
        "height": generation["height"],
        "width": generation["width"],
        "num_frames": generation["num_frames"],
        "steps": generation["steps"],
        "guidance_scale": generation["guidance_scale"],
        "seed": generation["seed"],
        "model_attention_shape": payload["model_attention_shape"],
    }


def layer_records(candidate: Candidate) -> dict[int, list[dict[str, float]]]:
    result = {}
    for layer in shadow_provider(candidate.shadow)["shadow_layers"]:
        metrics = [
            call["lowp_vs_bf16"]
            for call in layer["calls"]
            if "lowp_vs_bf16" in call
        ]
        if metrics:
            result[int(layer["layer"])] = metrics
    return result


def aggregate_metrics(records: list[dict[str, float]]) -> dict[str, float | int]:
    if not records:
        raise ValueError("cannot aggregate an empty metric list")
    reference_energy = sum(record["reference_rms"] ** 2 for record in records)
    error_energy = sum(record["rmse"] ** 2 for record in records)
    relative_l2 = math.sqrt(error_energy / max(reference_energy, 1e-30))
    return {
        "calls": len(records),
        "cosine_mean": sum(record["cosine"] for record in records) / len(records),
        "relative_l2": relative_l2,
        "rmse_rms": math.sqrt(
            sum(record["rmse"] ** 2 for record in records) / len(records)
        ),
        "reference_rms_rms": math.sqrt(reference_energy / len(records)),
    }


def guard_layers(candidate: Candidate) -> set[int]:
    specification = candidate.manifest.get("guard_layers")
    return parse_layer_indices(specification) if specification else set()


def choose_codebook(
    losses: dict[int, dict[str, float]],
    labels: list[str],
    base: str,
    maximum: int,
) -> list[str]:
    selected = [base]
    available = [label for label in labels if label != base]

    def objective(codebook: list[str]) -> float:
        return sum(
            min(layer_losses[label] ** 2 for label in codebook)
            for layer_losses in losses.values()
        )

    while available and len(selected) < maximum:
        current = objective(selected)
        winner = min(available, key=lambda label: objective([*selected, label]))
        updated = objective([*selected, winner])
        if updated >= current - 1e-15:
            break
        selected.append(winner)
        available.remove(winner)
    return selected


def compress_layers(layers: list[int]) -> str:
    if not layers:
        raise ValueError("cannot format an empty layer list")
    ranges = []
    first = previous = layers[0]
    for layer in layers[1:]:
        if layer == previous + 1:
            previous = layer
            continue
        ranges.append(str(first) if first == previous else f"{first}-{previous}")
        first = previous = layer
    ranges.append(str(first) if first == previous else f"{first}-{previous}")
    return ",".join(ranges)


def resolve_manifest_path(manifest_path: Path, path_text: str) -> str:
    path = Path(path_text)
    if not path.is_absolute():
        path = manifest_path.parent / path
    return str(path.resolve())


def make_routed_manifest(
    candidates: dict[str, Candidate],
    base: str,
    assignments: dict[int, str],
    policy: str,
    calibration_path: Path,
) -> dict[str, Any]:
    base_candidate = candidates[base]
    source = base_candidate.manifest
    selected_policy = source["policies"][policy]
    base_entry = dict(selected_policy["base"])
    base_entry["path"] = resolve_manifest_path(
        base_candidate.manifest_path, base_entry["path"]
    )
    layer_extensions = []
    for label in sorted(set(assignments.values())):
        if label == base:
            continue
        layers = sorted(layer for layer, assigned in assignments.items() if assigned == label)
        candidate = candidates[label]
        entry = candidate.manifest["policies"][policy]["base"]
        layer_extensions.append(
            {
                "layers": compress_layers(layers),
                "path": resolve_manifest_path(candidate.manifest_path, entry["path"]),
                "module": entry["module"],
                "purpose": f"layer-calibrated affine code map {label}",
            }
        )
    for entry in selected_policy.get("layer_extensions", []):
        retained = dict(entry)
        retained["path"] = resolve_manifest_path(
            base_candidate.manifest_path, retained["path"]
        )
        layer_extensions.append(retained)

    result = {
        key: value
        for key, value in source.items()
        if key not in ("policies", "fast_affine_code_map")
    }
    result["schema"] = "tk_wan_nv_mx_affine_routed_bundle_v1"
    result["affine_calibration"] = {
        "path": str(calibration_path.resolve()),
        "base": base,
        "runtime_overhead": "none; compile-time constants and static layer routing",
    }
    result["policies"] = {
        policy: {
            "base": base_entry,
            "layer_extensions": layer_extensions,
        }
    }
    return result


def main() -> None:
    args = parse_args()
    if args.max_codebook < 1:
        raise ValueError("--max-codebook must be positive")
    if args.minimum_improvement < 0.0:
        raise ValueError("--minimum-improvement cannot be negative")

    candidate_list = load_candidates(args.candidate)
    candidates = {candidate.label: candidate for candidate in candidate_list}
    if args.base not in candidates:
        raise ValueError(f"unknown base candidate: {args.base}")

    identity = calibration_identity(candidate_list[0].shadow)
    for candidate in candidate_list[1:]:
        if calibration_identity(candidate.shadow) != identity:
            raise ValueError(
                f"candidate {candidate.label} does not share the paired trajectory"
            )
    guards = guard_layers(candidates[args.base])
    if any(guard_layers(candidate) != guards for candidate in candidate_list):
        raise ValueError("candidate manifests use different guard layers")

    records = {candidate.label: layer_records(candidate) for candidate in candidate_list}
    expected_layers = set(records[args.base])
    for label, layers in records.items():
        if set(layers) != expected_layers:
            raise ValueError(f"candidate {label} has a different layer set")
        for layer in expected_layers:
            if len(layers[layer]) != len(records[args.base][layer]):
                raise ValueError(
                    f"candidate {label} has a different call count for layer {layer}"
                )

    per_layer: dict[int, dict[str, dict[str, float | int]]] = {}
    losses: dict[int, dict[str, float]] = {}
    for layer in sorted(expected_layers):
        per_layer[layer] = {
            label: aggregate_metrics(candidate_records[layer])
            for label, candidate_records in records.items()
        }
        if layer not in guards:
            losses[layer] = {
                label: float(metrics["relative_l2"])
                for label, metrics in per_layer[layer].items()
            }

    codebook = choose_codebook(
        losses,
        [candidate.label for candidate in candidate_list],
        args.base,
        args.max_codebook,
    )
    assignments = {}
    for layer, layer_losses in losses.items():
        winner = min(codebook, key=lambda label: layer_losses[label])
        improvement = layer_losses[args.base] - layer_losses[winner]
        assignments[layer] = (
            winner if improvement >= args.minimum_improvement else args.base
        )

    payload = {
        "schema": "tk_wan_affine_layer_calibration_v1",
        "calibration": identity,
        "objective": (
            "minimum aggregate layer-output relative L2 on a fixed BF16 trajectory"
        ),
        "warning": (
            "Layer-local loss is a calibration proxy; the routed policy must pass "
            "end-to-end held-out validation before promotion."
        ),
        "guard_layers_excluded": sorted(guards),
        "base": args.base,
        "maximum_codebook": args.max_codebook,
        "selected_codebook": codebook,
        "minimum_improvement": args.minimum_improvement,
        "candidates": {
            candidate.label: {
                "affine_a": candidate.affine_a,
                "affine_b": candidate.affine_b,
                "shadow_path": str(candidate.shadow_path),
                "manifest_path": str(candidate.manifest_path),
            }
            for candidate in candidate_list
        },
        "assignments": {str(layer): label for layer, label in assignments.items()},
        "assignment_groups": {
            label: compress_layers(
                sorted(layer for layer, assigned in assignments.items() if assigned == label)
            )
            for label in sorted(set(assignments.values()))
        },
        "layer_metrics": {
            str(layer): metrics for layer, metrics in per_layer.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")

    manifest = make_routed_manifest(
        candidates,
        args.base,
        assignments,
        args.policy,
        args.output,
    )
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(args.output)
    print(args.manifest_output)


if __name__ == "__main__":
    main()
