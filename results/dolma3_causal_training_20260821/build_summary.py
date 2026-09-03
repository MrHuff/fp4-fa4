#!/usr/bin/env python3
"""Validate the literal-Dolma3 scale gate and build its compact summary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SOURCE_COMMIT = "cd59dda37ebf22e0d77b9c9d6851ec164b86e3af"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
DATA_SHA256 = "860b33924dffd53f4c20b80abbcee96e1bf09c3c313290c15ea3a6ee418269ce"
TOKENIZER_SHA256 = (
    "76e48799b099d43365bd24ccd8ecc5aedac831718da780552f03b0a6eb4412aa"
)
GPU_UUID = "01355792-ef83-14f6-793b-b31a141c113a"
VALIDATION_ROUNDS = [-1, 63, 127, 191, 255]
BF16_ROUTE = "bf16_cute"
MX_ROUTE = "nvfp4_qk_mxfp4_pv"
FP8_ROUTE = "nvfp4_qk_fp8_pv_exact"

RAW_SHA256 = {
    "raw/1p2b/exit-status.txt": (
        "ce4522f4f1c3e83b9518ef738478c785e24757e12eb92a1e21c03f2bef0d3bac"
    ),
    "raw/1p2b/llama1p2b-matched.json": (
        "d012cc8d2531a1d9a24c8d3e27407befdd95f37fe534e4a42c3176b104133079"
    ),
    "raw/1p2b/runtime-preflight.json": (
        "72c69706cbb11ea5fd1731b7341c9aa0a921fd6c3b244ea3f406429c1c4ebf25"
    ),
    "raw/8b/bf16.json": (
        "e27208450ab4aad99c8b10c0665a360ee01c62c47db6d1f8c672b2dcd7bfd075"
    ),
    "raw/8b/bracket-summary.json": (
        "224c6bd1581729866dced2c738784ba0c3c9541cec3fa5a1942c6cc75f50233d"
    ),
    "raw/8b/dolma3-materialization-metadata.json": (
        "0a621a5dd546cc78127a52c25551cd03083daf1712c41c6b3276df8d2e85a79a"
    ),
    "raw/8b/exit-status.txt": (
        "1460b87de303106f05246236c2b125edbc9cdcacb37124755c54758c49ce46e9"
    ),
    "raw/8b/fp8.json": (
        "45b28e26d93a9a822062fdcdad9c2e0d6d7b0a6aed8107e9c30c474fa9e4fd41"
    ),
    "raw/8b/merged-mx-a.json": (
        "dddff5fb599a582237d1d93adfb34fcf703d6398d0039a0113f3015c24782580"
    ),
    "raw/8b/merged-mx-b.json": (
        "9182c71848a5a5b1e96b39b72ab2844a83c2f78e567693035b960a3201dc2f0b"
    ),
    "raw/8b/mx-a.json": (
        "270dbef5d72ba0c6330d7a9a62a6cbe037ad2ea34477abedc494d6dc036300cd"
    ),
    "raw/8b/mx-b.json": (
        "ae40f68658fa07f36a53a91989974cbb3378f7b8a868f82e98505969111d58e1"
    ),
    "raw/8b/runtime-preflight.json": (
        "f0c2e955db08af44b1ffdb2b37cb5b709a066cb73dd38aed50a21de0954360f1"
    ),
}

EXPECTED_CORPUS_METADATA = {
    "empty_rows_removed": 0,
    "exact_duplicate_rows_removed": 66,
    "format": "JSON Lines",
    "source_rows": 512,
    "unique_documents": 446,
}


def _fail(message: str) -> None:
    raise RuntimeError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(relative_path: str) -> dict[str, Any]:
    path = ROOT / relative_path

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value} in {relative_path}")

    value = json.loads(path.read_text(), parse_constant=reject_constant)
    _require(isinstance(value, dict), f"{relative_path}: top level must be an object")
    return value


def _validate_raw_hashes() -> None:
    for relative_path, expected in RAW_SHA256.items():
        path = ROOT / relative_path
        _require(path.is_file(), f"missing raw artifact: {relative_path}")
        actual = _sha256(path)
        _require(
            actual == expected,
            f"raw artifact hash mismatch for {relative_path}: {actual} != {expected}",
        )


def _validate_common_result(
    result: dict[str, Any],
    *,
    label: str,
    expected_routes: list[str],
    parameter_count: int,
    model_preset: str,
    layers: int,
    head_dim: int,
    initial_probe_sha256: str,
) -> None:
    _require(result.get("schema") == "llama12b_real_tokens_training_v3", label)
    source = result["source"]["git"]
    _require(source["head"] == SOURCE_COMMIT, f"{label}: source commit")
    _require(source["tracked_dirty"] is False, f"{label}: dirty source")
    _require(source["tracked_diff_bytes"] == 0, f"{label}: source diff bytes")
    _require(
        source["tracked_diff_sha256"] == EMPTY_SHA256,
        f"{label}: source diff hash",
    )

    config = result["configuration"]
    expected_config = {
        "batch": 1,
        "sequence": 4096,
        "rounds": 256,
        "training_batches": 256,
        "validation_batches": 8,
        "eval_every": 64,
        "seed": 20260818,
        "learning_rate": 0.0001,
        "gradient_clip_norm": None,
        "parameter_count": parameter_count,
        "model_preset": model_preset,
        "layers": layers,
        "head_dim": head_dim,
        "corpus_sha256": DATA_SHA256,
        "corpus_documents": 446,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "routes": expected_routes,
    }
    for key, expected in expected_config.items():
        _require(config.get(key) == expected, f"{label}: configuration.{key}")
    _require(
        config.get("corpus_metadata") == EXPECTED_CORPUS_METADATA,
        f"{label}: corpus metadata",
    )
    _require(
        config["train_tokens"]["tokens_with_boundary"] == 1_048_832,
        f"{label}: training token count",
    )
    _require(
        config["train_tokens"]["sha256"]
        == "f0c2a2a40fb77bc5f43b91aab0385cd05d8a663f69f88830978c69b76839478c",
        f"{label}: training token identity",
    )
    _require(
        config["validation_tokens"]["tokens_with_boundary"] == 32_776,
        f"{label}: validation token count",
    )
    _require(
        config["validation_tokens"]["sha256"]
        == "61e89010706c73acd6bcb37af531f87dd5f4db4c1f06d54f32a3514dfd10c88b",
        f"{label}: validation token identity",
    )
    _require(
        config["initial_state_probe"]["sha256"] == initial_probe_sha256,
        f"{label}: initialization probe",
    )
    hardware = config["hardware_identity"]
    _require(hardware["uuid"] == GPU_UUID, f"{label}: GPU UUID")
    _require(hardware["name"] == "NVIDIA GB200", f"{label}: GPU name")
    _require(
        hardware["compute_capability"] == [10, 0],
        f"{label}: compute capability",
    )
    _require(
        [entry["round"] for entry in result["validation_history"]]
        == VALIDATION_ROUNDS,
        f"{label}: validation schedule",
    )
    _require(set(result["routes"]) == set(expected_routes), f"{label}: route keys")
    _require(set(result["records"]) == set(expected_routes), f"{label}: record keys")

    for route in expected_routes:
        records = result["records"][route]
        _require(len(records) == 256, f"{label}/{route}: record count")
        _require(
            [record["round"] for record in records] == list(range(256)),
            f"{label}/{route}: record order",
        )
        for record in records:
            _require(record["finite"] is True, f"{label}/{route}: nonfinite record")
            _require(
                record["diagnostic"] is False,
                f"{label}/{route}: diagnostic record",
            )
            _require(
                record["timing_eligible"] is True,
                f"{label}/{route}: ineligible timing record",
            )
            _require(
                math.isfinite(record["loss"]),
                f"{label}/{route}: nonfinite loss",
            )
        route_result = result["routes"][route]
        timing = route_result["timing"]
        _require(timing["timed_records"] == 256, f"{label}/{route}: timing count")
        _require(
            timing["timing_fallback_used"] is False,
            f"{label}/{route}: timing fallback",
        )
        _require(
            timing["matched_round_records_ineligible"] == 0,
            f"{label}/{route}: ineligible summary",
        )
        _require(
            timing["route_diagnostic_records"] == 0,
            f"{label}/{route}: diagnostic summary",
        )
        for key in (
            "forward_ms",
            "backward_ms",
            "optimizer_ms",
            "step_ms",
            "wall_ms",
            "tokens_per_second",
        ):
            _require(math.isfinite(timing[key]), f"{label}/{route}: timing.{key}")
        training = route_result["training"]
        _require(training["all_steps_finite"] is True, f"{label}/{route}: finite")
        _require(len(training["losses"]) == 256, f"{label}/{route}: losses")
        _require(
            all(math.isfinite(loss) for loss in training["losses"]),
            f"{label}/{route}: training losses",
        )
        for key in ("initial_loss", "final_loss"):
            _require(
                math.isfinite(route_result["validation"][key]),
                f"{label}/{route}: validation.{key}",
            )


def _validate_merged(merged: dict[str, Any], label: str) -> None:
    _require(
        merged.get("schema") == "llama_real_token_independent_routes_merged_v1",
        f"{label}: schema",
    )
    validation = merged["validation"]
    expected = {
        "matched": True,
        "hardware_identity_verified": True,
        "lowp_backward_contract_matched": True,
        "lowp_forward_provenance_verified": True,
        "common_configuration_sha256": (
            "aae2ce1413cd08071efba84aee97564f1a095b0fa151a83b0f5c01e9d9139965"
        ),
        "data_identity_sha256": (
            "63d792196bf46eaf6870585d16ba0ddbe78a0ae3e86ecf1a8f9925e4c15225cd"
        ),
        "lowp_backward_contract_sha256": (
            "47c3244204c178b6bdc2c8d02e9a1e4cf10c17262829c4ff1c44d007df889d97"
        ),
        "routes": [BF16_ROUTE, MX_ROUTE, FP8_ROUTE],
    }
    for key, value in expected.items():
        _require(validation.get(key) == value, f"{label}: validation.{key}")
    _require(validation["hardware_identity"]["uuid"] == GPU_UUID, label)
    _require(validation["source_identity"]["git"]["head"] == SOURCE_COMMIT, label)
    _require(validation["source_identity"]["git"]["tracked_dirty"] is False, label)
    _require(
        validation["initial_state_probe"]["sha256"]
        == "15cdbd44942987f87eb94b37ffaff1abd2fa5be6a389fbf35396adb5ce5e61b8",
        label,
    )
    _require(
        validation["backward_extension"]["sha256"]
        == "bfdec1e43a0a19acec5afbac3fa837e2f4d1b25be80ae7fb5ff3b5bc5e9e25ce",
        label,
    )


def _route_metrics(result: dict[str, Any], route: str) -> dict[str, Any]:
    route_result = result["routes"][route]
    timing = route_result["timing"]
    return {
        "forward_ms": timing["forward_ms"],
        "backward_ms": timing["backward_ms"],
        "optimizer_ms": timing["optimizer_ms"],
        "step_ms": timing["step_ms"],
        "wall_ms": timing["wall_ms"],
        "tokens_per_second": timing["tokens_per_second"],
        "initial_validation_loss": route_result["validation"]["initial_loss"],
        "final_validation_loss": route_result["validation"]["final_loss"],
        "last_eight_training_loss_mean": route_result["training"]["last_eight_mean"],
    }


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator


def _build_summary() -> dict[str, Any]:
    _validate_raw_hashes()

    one = _load_json("raw/1p2b/llama1p2b-matched.json")
    bf16 = _load_json("raw/8b/bf16.json")
    mx_a = _load_json("raw/8b/mx-a.json")
    fp8 = _load_json("raw/8b/fp8.json")
    mx_b = _load_json("raw/8b/mx-b.json")
    merged_a = _load_json("raw/8b/merged-mx-a.json")
    merged_b = _load_json("raw/8b/merged-mx-b.json")
    bracket = _load_json("raw/8b/bracket-summary.json")

    _validate_common_result(
        one,
        label="1p2b",
        expected_routes=[BF16_ROUTE, MX_ROUTE, FP8_ROUTE],
        parameter_count=1_235_814_400,
        model_preset="llama3.2-1b",
        layers=16,
        head_dim=64,
        initial_probe_sha256=(
            "899954cf7f7cdfe734467e9fd9d17b15ecf9011ea69d8fc99215f9412f20c24c"
        ),
    )
    eight_inputs = (
        (bf16, "8b/bf16", [BF16_ROUTE]),
        (mx_a, "8b/mx-a", [MX_ROUTE]),
        (fp8, "8b/fp8", [FP8_ROUTE]),
        (mx_b, "8b/mx-b", [MX_ROUTE]),
    )
    for result, label, routes in eight_inputs:
        _validate_common_result(
            result,
            label=label,
            expected_routes=routes,
            parameter_count=8_030_261_248,
            model_preset="llama3.1-8b",
            layers=32,
            head_dim=128,
            initial_probe_sha256=(
                "15cdbd44942987f87eb94b37ffaff1abd2fa5be6a389fbf35396adb5ce5e61b8"
            ),
        )

    one_contracts = one["configuration"]["backward_route_contracts"]
    _require(
        one_contracts[MX_ROUTE] == one_contracts[FP8_ROUTE],
        "1p2b: MX/FP8 backward contracts differ",
    )
    _require(
        one["configuration"]["matched_lowp_backward_contract"] is True,
        "1p2b: matched backward proof",
    )
    _require(
        all(one["configuration"]["shared_lowp_backward_runner"].values()),
        "1p2b: backward runner/storage was not shared",
    )

    mx_a_contract = mx_a["configuration"]["backward_route_contracts"][MX_ROUTE]
    mx_b_contract = mx_b["configuration"]["backward_route_contracts"][MX_ROUTE]
    fp8_contract = fp8["configuration"]["backward_route_contracts"][FP8_ROUTE]
    _require(
        mx_a_contract == mx_b_contract == fp8_contract,
        "8b: MX/FP8 backward contracts differ",
    )
    _validate_merged(merged_a, "8b/merged-mx-a")
    _validate_merged(merged_b, "8b/merged-mx-b")
    _require(
        bracket["arm_order"] == ["bf16", "mx-a", "fp8", "mx-b"],
        "8b: bracket arm order",
    )
    _require(bracket["source_commit"] == SOURCE_COMMIT, "8b: bracket source")
    _require(bracket["dataset"]["sha256"] == DATA_SHA256, "8b: bracket data")

    one_exit = (ROOT / "raw/1p2b/exit-status.txt").read_text().splitlines()
    eight_exit = (ROOT / "raw/8b/exit-status.txt").read_text().splitlines()
    _require(one_exit[0] == "exit_status=1", "1p2b: expected preserved exit 1")
    _require(eight_exit[0] == "exit_status=0", "8b: expected exit 0")

    one_metrics = {
        "bf16": _route_metrics(one, BF16_ROUTE),
        "mx": _route_metrics(one, MX_ROUTE),
        "fp8": _route_metrics(one, FP8_ROUTE),
    }
    eight_metrics = {
        "bf16": _route_metrics(bf16, BF16_ROUTE),
        "mx_a": _route_metrics(mx_a, MX_ROUTE),
        "fp8": _route_metrics(fp8, FP8_ROUTE),
        "mx_b": _route_metrics(mx_b, MX_ROUTE),
    }
    mx_bracket = {
        key: statistics.median([eight_metrics["mx_a"][key], eight_metrics["mx_b"][key]])
        for key in (
            "forward_ms",
            "backward_ms",
            "optimizer_ms",
            "step_ms",
            "wall_ms",
            "tokens_per_second",
        )
    }

    one_fp8_mx_forward = _ratio(
        one_metrics["fp8"]["forward_ms"], one_metrics["mx"]["forward_ms"]
    )
    one_fp8_mx_step = _ratio(
        one_metrics["fp8"]["step_ms"], one_metrics["mx"]["step_ms"]
    )
    eight_fp8_mx_forward = _ratio(
        eight_metrics["fp8"]["forward_ms"], mx_bracket["forward_ms"]
    )
    eight_fp8_mx_step = _ratio(
        eight_metrics["fp8"]["step_ms"], mx_bracket["step_ms"]
    )

    mx_a_losses = mx_a["routes"][MX_ROUTE]["training"]["losses"]
    mx_b_losses = mx_b["routes"][MX_ROUTE]["training"]["losses"]
    mx_ab_deltas = [a - b for a, b in zip(mx_a_losses, mx_b_losses, strict=True)]
    first_divergence = next(
        (index for index, delta in enumerate(mx_ab_deltas) if delta != 0.0), None
    )

    summary = {
        "schema": "fa4_dolma3_causal_model_scale_gate_v1",
        "source_commit": SOURCE_COMMIT,
        "protocol": {
            "dataset": {
                "description": "Dolma3 Longmino len-8-16k first 512 physical MDS rows",
                "sha256": DATA_SHA256,
                "bytes": 21_911_537,
                "corpus_metadata": EXPECTED_CORPUS_METADATA,
                "scope": "one Longmino bucket, not the full ten-stream Dolma mixture",
            },
            "tokenizer_sha256": TOKENIZER_SHA256,
            "hardware": {"name": "NVIDIA GB200", "uuid": GPU_UUID},
            "seed": 20260818,
            "learning_rate": 0.0001,
            "sequence": 4096,
            "batch": 1,
            "updates": 256,
            "training_tokens": 1_048_832,
            "validation_tokens": 32_776,
            "validation_rounds": VALIDATION_ROUNDS,
            "gradient_clip_norm": None,
        },
        "models": {
            "1p2b_d64": {
                "parameter_count": 1_235_814_400,
                "layers": 16,
                "head_dim": 64,
                "execution": "three routes interleaved and rotated in one process",
                "routes": one_metrics,
                "comparisons": {
                    "bf16_over_mx_step_speedup": _ratio(
                        one_metrics["bf16"]["step_ms"], one_metrics["mx"]["step_ms"]
                    ),
                    "bf16_over_fp8_step_speedup": _ratio(
                        one_metrics["bf16"]["step_ms"], one_metrics["fp8"]["step_ms"]
                    ),
                    "fp8_over_mx_forward_speedup": one_fp8_mx_forward,
                    "fp8_over_mx_backward_speedup": _ratio(
                        one_metrics["fp8"]["backward_ms"],
                        one_metrics["mx"]["backward_ms"],
                    ),
                    "fp8_over_mx_step_speedup": one_fp8_mx_step,
                    "mx_final_validation_loss_delta_vs_bf16": (
                        one_metrics["mx"]["final_validation_loss"]
                        - one_metrics["bf16"]["final_validation_loss"]
                    ),
                    "fp8_final_validation_loss_delta_vs_bf16": (
                        one_metrics["fp8"]["final_validation_loss"]
                        - one_metrics["bf16"]["final_validation_loss"]
                    ),
                },
                "job_status": {
                    "training_complete": True,
                    "all_records_verified": True,
                    "kubernetes_exit_status": 1,
                    "failure_stage": "post-run document-count assertion",
                    "failure_explanation": (
                        "submitted gate expected 512 documents; trainer correctly "
                        "reported 446 unique documents after 66 exact duplicates"
                    ),
                },
            },
            "8b_d128": {
                "parameter_count": 8_030_261_248,
                "layers": 32,
                "head_dim": 128,
                "execution": "isolated sequential arms: bf16, mx-a, fp8, mx-b",
                "routes": eight_metrics,
                "mx_bracket_median": mx_bracket,
                "comparisons": {
                    "bf16_over_mx_bracket_step_speedup": _ratio(
                        eight_metrics["bf16"]["step_ms"], mx_bracket["step_ms"]
                    ),
                    "bf16_over_fp8_step_speedup": _ratio(
                        eight_metrics["bf16"]["step_ms"],
                        eight_metrics["fp8"]["step_ms"],
                    ),
                    "fp8_over_mx_bracket_forward_speedup": eight_fp8_mx_forward,
                    "fp8_over_mx_bracket_backward_speedup": _ratio(
                        eight_metrics["fp8"]["backward_ms"],
                        mx_bracket["backward_ms"],
                    ),
                    "fp8_over_mx_bracket_step_speedup": eight_fp8_mx_step,
                    "mx_forward_run_order_spread_ratio": _ratio(
                        max(
                            eight_metrics["mx_a"]["forward_ms"],
                            eight_metrics["mx_b"]["forward_ms"],
                        ),
                        min(
                            eight_metrics["mx_a"]["forward_ms"],
                            eight_metrics["mx_b"]["forward_ms"],
                        ),
                    ),
                    "mx_step_run_order_spread_ratio": _ratio(
                        max(
                            eight_metrics["mx_a"]["step_ms"],
                            eight_metrics["mx_b"]["step_ms"],
                        ),
                        min(
                            eight_metrics["mx_a"]["step_ms"],
                            eight_metrics["mx_b"]["step_ms"],
                        ),
                    ),
                    "mx_a_final_validation_loss_delta_vs_bf16": (
                        eight_metrics["mx_a"]["final_validation_loss"]
                        - eight_metrics["bf16"]["final_validation_loss"]
                    ),
                    "mx_b_final_validation_loss_delta_vs_bf16": (
                        eight_metrics["mx_b"]["final_validation_loss"]
                        - eight_metrics["bf16"]["final_validation_loss"]
                    ),
                    "fp8_final_validation_loss_delta_vs_bf16": (
                        eight_metrics["fp8"]["final_validation_loss"]
                        - eight_metrics["bf16"]["final_validation_loss"]
                    ),
                },
                "mx_a_b_reproducibility": {
                    "first_divergent_training_step": first_divergence,
                    "mean_absolute_training_loss_delta": statistics.fmean(
                        abs(delta) for delta in mx_ab_deltas
                    ),
                    "maximum_absolute_training_loss_delta": max(
                        abs(delta) for delta in mx_ab_deltas
                    ),
                    "final_validation_loss_delta": (
                        eight_metrics["mx_b"]["final_validation_loss"]
                        - eight_metrics["mx_a"]["final_validation_loss"]
                    ),
                },
                "job_status": {
                    "training_complete": True,
                    "all_records_verified": True,
                    "kubernetes_exit_status": 0,
                },
            },
        },
        "scale_observation": {
            "verified_direction": (
                "MXFP4-PV beats FP8-PV in both 8B bracket arms, while backward "
                "remains effectively equal"
            ),
            "fp8_over_mx_forward_speedup": {
                "1p2b": one_fp8_mx_forward,
                "8b_bracket": eight_fp8_mx_forward,
                "ratio_of_speedup_factors_8b_over_1p2b": _ratio(
                    eight_fp8_mx_forward, one_fp8_mx_forward
                ),
            },
            "fp8_over_mx_step_speedup": {
                "1p2b": one_fp8_mx_step,
                "8b_bracket": eight_fp8_mx_step,
                "ratio_of_speedup_factors_8b_over_1p2b": _ratio(
                    eight_fp8_mx_step, one_fp8_mx_step
                ),
            },
            "interpretation": (
                "directionally supports better MX scaling, but is not a pure "
                "model-size causal comparison because D64/D128, projection "
                "publication, layer count, and process protocol change together"
            ),
        },
        "blocking_findings": [
            (
                "D128 low-precision backward is about 8.3% slower than BF16 and "
                "cancels the larger MX forward saving"
            ),
            (
                "8B low-precision final validation loss drifts materially from "
                "BF16; MX-A/B divergence also exposes a reproducibility issue"
            ),
            (
                "the eight-sequence validation sample and 1.05M-token horizon are "
                "a short-run gate, not a convergence-quality verdict"
            ),
        ],
        "raw_artifact_sha256": RAW_SHA256,
        "validation": {
            "raw_hashes": "matched",
            "source_and_data_identity": "matched",
            "finite_timing_eligible_records_per_route": 256,
            "hardware_identity": "matched",
            "initialization_identity_within_model": "matched",
            "lowp_backward_contract_within_model": "matched",
            "8b_strict_merger_outputs": "matched",
        },
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "comparison_summary.json",
        help="summary output path",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate that the existing output already matches",
    )
    args = parser.parse_args()
    summary_text = json.dumps(_build_summary(), indent=2, sort_keys=True) + "\n"
    if args.check:
        _require(args.output.is_file(), f"missing summary output: {args.output}")
        _require(
            args.output.read_text() == summary_text,
            f"stale summary output: {args.output}",
        )
        print(f"validated {args.output}")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(summary_text)
    temporary.replace(args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
