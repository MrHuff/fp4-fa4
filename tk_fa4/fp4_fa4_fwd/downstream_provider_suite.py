#!/usr/bin/env python3
"""Run and summarize the fixed downstream provider comparison matrix."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

try:
    from .eval_regular_attention import (
        authenticate_asset_record as _authenticate_asset_record,
    )
except ImportError:  # direct script execution
    from eval_regular_attention import (
        authenticate_asset_record as _authenticate_asset_record,
    )


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_OUTPUT = Path(
    "../../results/fp4_fa4_downstream_matrix_20260801"
)
PROVIDERS = (
    "nvmx-fast",
    "nvmx-accurate",
    "hao-nvnv",
    "tk-nvnv-control",
)
TASKS: dict[str, dict[str, Any]] = {
    "vit-s256": {
        "kind": "classification",
        "script": "eval_regular_attention.py",
        "samples": 1000,
        "shape": "s256-h16",
        "arguments": ("--mask-value", "10"),
    },
    "vit-s1024": {
        "kind": "classification",
        "script": "eval_regular_attention.py",
        "samples": 200,
        "shape": "s1024-h24",
        "arguments": ("--mask-value", "10", "--image-size", "496"),
    },
    "vit-s4096": {
        "kind": "classification",
        "script": "eval_regular_attention.py",
        "samples": 200,
        "shape": "s4096-h24",
        "arguments": ("--mask-value", "10", "--image-size", "1008"),
    },
    "bert-mlm-s256": {
        "kind": "mlm",
        "script": "eval_bert_mlm_attention.py",
        "samples": 800,
        "shape": "s256-h16",
        "arguments": ("--sequence-length", "256"),
    },
    "bert-mlm-s512": {
        "kind": "mlm",
        "script": "eval_bert_mlm_attention.py",
        "samples": 200,
        "shape": "s1024-h24",
        "arguments": ("--sequence-length", "512"),
    },
    "bert-sst2-s256": {
        "kind": "classification",
        "script": "eval_bert_sequence_classification.py",
        "samples": 872,
        "shape": "s256-h16",
        "arguments": ("--sequence-length", "256"),
    },
}
SHAPES = {
    "s256-h16": {
        "prefix": "b1_s256_h16_d128",
    },
    "s1024-h24": {
        "prefix": "b1_s1024_h24_d128",
    },
    "s4096-h24": {
        "prefix": "b1_s4096_h24_d128",
    },
}
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", action="append", choices=PROVIDERS)
    parser.add_argument("--task", action="append", choices=tuple(TASKS))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--extension-root",
        type=Path,
        help="Directory containing the explicitly rebuilt unified extensions.",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        help="Authenticated local Hugging Face model snapshot.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Authenticated local Hugging Face dataset snapshot.",
    )
    parser.add_argument(
        "--asset-manifest",
        type=Path,
        help="fa4_external_assets_v1 manifest authenticated by the planner.",
    )
    parser.add_argument("--model-asset", help="Model key in --asset-manifest.")
    parser.add_argument("--dataset-asset", help="Dataset key in --asset-manifest.")
    parser.add_argument("--reuse-results", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def provider_extension(
    provider: str,
    shape: str,
    extension_root: Path,
) -> tuple[Path, list[str]]:
    shape_config = SHAPES[shape]
    variant = {
        "nvmx-fast": "nvmx-fast",
        "nvmx-accurate": "nvmx-accurate",
        "hao-nvnv": "nv-nv",
        "tk-nvnv-control": "nv-nv",
    }[provider]
    extension = extension_root / f"{shape_config['prefix']}_{variant}.so"
    arguments: list[str] = []
    if provider == "nvmx-accurate":
        arguments.extend(
            ["--global-anchor-kv", "--global-anchor-samples", "32"]
        )
    elif provider == "hao-nvnv":
        arguments.extend(["--attention-backend", "hao-native"])
    elif provider == "tk-nvnv-control":
        arguments.extend(
            [
                "--finite-diagnostics",
                "--stop-on-nonfinite",
                "--scale-sweep-samples",
                "1",
            ]
        )
    return extension, arguments


def authenticate_asset_record(
    assets: object,
    name: str,
    role: str,
    expected_root: Path,
) -> dict[str, str]:
    return _authenticate_asset_record(assets, name, role, expected_root)


def extension_identities(
    providers: list[str],
    task_name: str,
    extension_root: Path,
) -> dict[str, dict[str, str | int]]:
    identities: dict[str, dict[str, str | int]] = {}
    shape = TASKS[task_name]["shape"]
    for provider in providers:
        extension, _ = provider_extension(provider, shape, extension_root)
        if not extension.is_file():
            raise FileNotFoundError(f"missing extension: {extension}")
        identities[provider] = {
            "file": extension.name,
            "bytes": extension.stat().st_size,
            "sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
        }
    return identities


def asset_identity_arguments(
    selected_assets: dict[str, dict[str, str]],
) -> list[str]:
    arguments: list[str] = []
    for role in ("model", "dataset"):
        record = selected_assets[role]
        arguments.extend(
            (
                f"--{role}-identifier",
                record["identifier"],
                f"--{role}-revision",
                record["revision"],
                f"--{role}-tree-sha256",
                record["tree_sha256"],
            )
        )
    return arguments


def assert_reused_result_identity(
    path: Path,
    selected_assets: dict[str, dict[str, str]],
    extension_identity: dict[str, str | int],
) -> None:
    result = json.loads(path.read_text(encoding="utf-8"))
    for role in ("model", "dataset"):
        record = selected_assets[role]
        expected = {
            "identifier": record["identifier"],
            "revision": record["revision"],
            "tree_sha256": record["tree_sha256"],
            "source": "authenticated_local_snapshot",
        }
        if result.get(role) != expected:
            raise ValueError(
                f"refusing to reuse {path}: {role} identity does not match"
            )
    if result.get("extension") != extension_identity:
        raise ValueError(
            f"refusing to reuse {path}: extension identity does not match"
        )


def result_path(output_dir: Path, task: str, provider: str) -> Path:
    return output_dir / "raw" / f"{task}__{provider}.json"


def reproduction_path(output_dir: Path, task: str) -> Path:
    return output_dir / "reproduction" / f"{task}.json"


def write_task_reproduction(
    output_dir: Path,
    task: str,
    reproduction: dict[str, Any],
) -> None:
    path = reproduction_path(output_dir, task)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"task": task, **reproduction}
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(previous, dict) or previous.get("task") != task:
            raise ValueError(f"invalid task reproduction record: {path}")
        for field in ("asset_manifest_sha256", "assets"):
            if previous.get(field) != payload.get(field):
                raise ValueError(
                    f"refusing to mix {task} results with different {field}"
                )
        previous_extensions = previous.get("extensions")
        if not isinstance(previous_extensions, dict):
            raise ValueError(f"invalid task reproduction record: {path}")
        payload["extensions"] = {
            **previous_extensions,
            **payload.get("extensions", {}),
        }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_task_reproductions(output_dir: Path) -> dict[str, dict[str, Any]]:
    reproductions: dict[str, dict[str, Any]] = {}
    for task in TASKS:
        path = reproduction_path(output_dir, task)
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("task") != task:
            raise ValueError(f"invalid task reproduction record: {path}")
        reproductions[task] = payload
    return reproductions


def run_case(
    output_dir: Path,
    task_name: str,
    provider: str,
    gpu: str,
    extension_root: Path,
    model_root: Path,
    dataset_root: Path,
    selected_assets: dict[str, dict[str, str]],
) -> None:
    task = TASKS[task_name]
    extension, provider_arguments = provider_extension(
        provider,
        task["shape"],
        extension_root,
    )
    if not extension.exists():
        raise FileNotFoundError(f"missing extension: {extension}")
    output = result_path(output_dir, task_name, provider)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        task["script"],
        "--samples",
        str(task["samples"]),
        "--progress-every",
        str(max(1, task["samples"] // 20)),
        "--extension",
        str(extension),
        "--extension-module",
        "_C_tk_hao_direct_fp4pv",
        "--output",
        str(output.resolve()),
        "--model",
        str(model_root),
        "--dataset",
        str(dataset_root),
        "--hao-root",
        str(REPO_ROOT / "third_party/hao_flash_attention_fp4"),
        *asset_identity_arguments(selected_assets),
        *task["arguments"],
        *provider_arguments,
    ]
    if provider != "tk-nvnv-control" and task["script"] != (
        "eval_bert_sequence_classification.py"
    ):
        command.extend(["--scale-sweep-samples", "0"])
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    completed = subprocess.run(
        command,
        cwd=HERE,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log = output.with_suffix(".log")
    log.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(
            f"{task_name}/{provider} failed with code "
            f"{completed.returncode}; see {log}"
        )


def finite_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def mean_layer_metric(attention: dict[str, Any], metric: str) -> float | None:
    values = [
        finite_number(record.get(metric))
        for record in attention["layer_output_error"].values()
    ]
    finite_values = [value for value in values if value is not None]
    if not finite_values:
        return None
    return statistics.fmean(finite_values)


def baseline_payload(result: dict[str, Any]) -> Any:
    if "classification" in result:
        return [
            (
                record["label"],
                record["baseline_prediction"],
                record["baseline_logits"],
            )
            for record in result["records"]
        ]
    return [
        (
            record["masked_tokens"],
            record["baseline_correct"],
            record["baseline_nll"],
        )
        for record in result["records"]
    ]


def baseline_digest(result: dict[str, Any]) -> str:
    payload = json.dumps(
        baseline_payload(result),
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def classification_margin_metrics(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    ranked = []
    regressions = 0
    corrections = 0
    for record in records:
        logits = sorted(
            (float(value) for value in record["baseline_logits"]),
            reverse=True,
        )
        margin = logits[0] - logits[1]
        changed = record["baseline_prediction"] != record["fp4_prediction"]
        ranked.append((margin, int(record["index"]), changed))
        baseline_correct = record["baseline_prediction"] == record["label"]
        provider_correct = record["fp4_prediction"] == record["label"]
        regressions += int(baseline_correct and not provider_correct)
        corrections += int(not baseline_correct and provider_correct)

    ranked.sort()
    low_count = max(1, len(ranked) // 4)
    low = ranked[:low_count]
    upper = ranked[low_count:]
    changed_margins = [margin for margin, _, changed in ranked if changed]
    stable_margins = [margin for margin, _, changed in ranked if not changed]
    return {
        "prediction_changes": len(changed_margins),
        "baseline_correct_provider_wrong": regressions,
        "baseline_wrong_provider_correct": corrections,
        "lowest_margin_quartile_samples": len(low),
        "lowest_margin_quartile_changes": sum(changed for _, _, changed in low),
        "upper_three_quartiles_samples": len(upper),
        "upper_three_quartiles_changes": sum(
            changed for _, _, changed in upper
        ),
        "median_margin_changed": (
            statistics.median(changed_margins) if changed_margins else None
        ),
        "median_margin_stable": (
            statistics.median(stable_margins) if stable_margins else None
        ),
    }


def summarize_result(
    task_name: str,
    provider: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    attention = result["attention"]
    finite = bool(attention["all_outputs_finite"])
    requested = int(
        result.get("protocol", result.get("adapter", {})).get(
            "requested_samples",
            TASKS[task_name]["samples"],
        )
    )
    if "classification" in result:
        metrics = result["classification"]
        completed = int(metrics["samples"])
        task_metrics = {
            "baseline_accuracy": finite_number(
                metrics["baseline_accuracy"]
            ),
            "provider_accuracy": finite_number(metrics["fp4_accuracy"]),
            "top1_agreement": finite_number(metrics["top1_agreement"]),
            "logit_cosine": finite_number(
                metrics["logit_error"]["cosine"]
            ),
            "logit_relative_l2": finite_number(
                metrics["logit_error"]["relative_l2"]
            ),
            "margin_analysis": classification_margin_metrics(
                result["records"]
            ),
        }
    else:
        metrics = result["masked_language_modeling"]
        completed = len(result["records"])
        task_metrics = {
            "masked_tokens": int(metrics["masked_tokens"]),
            "baseline_loss": finite_number(metrics["baseline_loss"]),
            "provider_loss": finite_number(metrics["fp4_loss"]),
            "baseline_accuracy": finite_number(
                metrics["baseline_masked_accuracy"]
            ),
            "provider_accuracy": finite_number(
                metrics["fp4_masked_accuracy"]
            ),
            "top1_agreement": finite_number(
                metrics["masked_top1_agreement"]
            ),
            "logit_cosine": finite_number(
                metrics["logit_error"]["cosine"]
            ),
            "logit_relative_l2": finite_number(
                metrics["logit_error"]["relative_l2"]
            ),
        }
    if not finite:
        task_metrics = {
            key: value if key.startswith("baseline_") else None
            for key, value in task_metrics.items()
        }
    return {
        "task": task_name,
        "provider": provider,
        "status": "complete" if finite else "nonfinite",
        "requested_samples": requested,
        "completed_samples": completed,
        "baseline_digest": baseline_digest(result),
        "attention_backend": attention.get("attention_backend", "tk"),
        "all_outputs_finite": finite,
        "nonfinite_output_rows": int(attention["nonfinite_output_rows"]),
        "numeric_path": {
            key: attention["topology"].get(key)
            for key in (
                "nv_shiftless_softmax",
                "nv_scale_satfinite",
                "nv_sampled_denom",
                "nv_quarter_scale",
                "nv_scale_encode_mode",
            )
        },
        "mean_layer_cosine": mean_layer_metric(attention, "cosine"),
        "mean_layer_relative_l2": mean_layer_metric(
            attention,
            "relative_l2",
        ),
        "p_scale_distribution": attention.get("p_scale_distribution", {}),
        "task_metrics": task_metrics,
    }


def write_summary(output_dir: Path) -> dict[str, Any]:
    rows = []
    loaded_results: dict[tuple[str, str], dict[str, Any]] = {}
    for task_name in TASKS:
        for provider in PROVIDERS:
            path = result_path(output_dir, task_name, provider)
            if not path.exists():
                continue
            result = json.loads(path.read_text(encoding="utf-8"))
            loaded_results[(task_name, provider)] = result
            rows.append(summarize_result(task_name, provider, result))

    baseline_audit = {}
    for task_name in TASKS:
        task_rows = [row for row in rows if row["task"] == task_name]
        complete_digests = {
            row["baseline_digest"]
            for row in task_rows
            if row["completed_samples"] == row["requested_samples"]
        }
        full_result = next(
            (
                loaded_results[(task_name, row["provider"])]
                for row in task_rows
                if row["completed_samples"] == row["requested_samples"]
            ),
            None,
        )
        reference_payload = (
            baseline_payload(full_result) if full_result is not None else []
        )
        prefix_matches = {}
        for row in task_rows:
            provider = row["provider"]
            payload = baseline_payload(
                loaded_results[(task_name, provider)]
            )
            matched = payload == reference_payload[: len(payload)]
            prefix_matches[provider] = matched
            row["baseline_prefix_matched"] = matched
        baseline_audit[task_name] = {
            "provider_count": len(task_rows),
            "full_run_digest_count": len(complete_digests),
            "full_runs_matched": len(complete_digests) <= 1,
            "provider_prefix_matches": prefix_matches,
            "all_provider_prefixes_matched": all(prefix_matches.values()),
        }
    summary = {
        "schema": "tk_fp4_downstream_provider_matrix_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "providers": list(PROVIDERS),
        "tasks": TASKS,
        "baseline_audit": baseline_audit,
        "rows": rows,
        "reproduction_by_task": load_task_reproductions(output_dir),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / "summary.json.tmp"
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_dir / "summary.json")
    csv_fields = (
        "task",
        "provider",
        "status",
        "requested_samples",
        "completed_samples",
        "attention_backend",
        "all_outputs_finite",
        "nonfinite_output_rows",
        "mean_layer_cosine",
        "mean_layer_relative_l2",
        "baseline_accuracy",
        "provider_accuracy",
        "baseline_loss",
        "provider_loss",
        "top1_agreement",
        "logit_cosine",
        "logit_relative_l2",
        "shiftless_scale_overflow_fraction",
        "shiftless_scale_maximum",
        "stable_scale_overflow_fraction",
        "stable_scale_maximum",
    )
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            metrics = row["task_metrics"]
            scales = row["p_scale_distribution"]
            shiftless = scales.get("shiftless", {})
            stable = scales.get("stable", {})
            writer.writerow(
                {
                    **{
                        key: row.get(key)
                        for key in csv_fields
                        if key in row
                    },
                    **{
                        key: metrics.get(key)
                        for key in csv_fields
                        if key in metrics
                    },
                    "shiftless_scale_overflow_fraction": shiftless.get(
                        "fraction_above_e4m3_max"
                    ),
                    "shiftless_scale_maximum": shiftless.get("maximum"),
                    "stable_scale_overflow_fraction": stable.get(
                        "fraction_above_e4m3_max"
                    ),
                    "stable_scale_maximum": stable.get("maximum"),
                }
            )
    return summary


def main() -> None:
    args = parse_args()
    providers = args.provider or list(PROVIDERS)
    tasks = args.task or list(TASKS)
    reproduction = None
    if not args.summarize_only:
        if len(tasks) != 1:
            raise ValueError(
                "fresh replay requires exactly one --task so one authenticated "
                "model/dataset pair cannot be reused accidentally"
            )
        required = {
            "--extension-root": args.extension_root,
            "--model-root": args.model_root,
            "--dataset-root": args.dataset_root,
            "--asset-manifest": args.asset_manifest,
            "--model-asset": args.model_asset,
            "--dataset-asset": args.dataset_asset,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"fresh replay requires {', '.join(missing)}")
        extension_root = args.extension_root.resolve()
        model_root = args.model_root.resolve()
        dataset_root = args.dataset_root.resolve()
        asset_manifest = args.asset_manifest.resolve()
        for label, path in (
            ("extension root", extension_root),
            ("model root", model_root),
            ("dataset root", dataset_root),
        ):
            if not path.is_dir():
                raise FileNotFoundError(f"{label} is not a directory: {path}")
        if not asset_manifest.is_file():
            raise FileNotFoundError(f"asset manifest is missing: {asset_manifest}")
        manifest = json.loads(asset_manifest.read_text(encoding="utf-8"))
        if manifest.get("schema") != "fa4_external_assets_v1":
            raise ValueError("--asset-manifest must use fa4_external_assets_v1")
        assets = manifest.get("assets")
        selected_assets = {}
        for role, name, expected_root in (
            ("model", args.model_asset, model_root),
            ("dataset", args.dataset_asset, dataset_root),
        ):
            selected_assets[role] = authenticate_asset_record(
                assets,
                name,
                role,
                expected_root,
            )
        extensions = extension_identities(
            providers,
            tasks[0],
            extension_root,
        )
        reproduction = {
            "asset_manifest_sha256": hashlib.sha256(
                asset_manifest.read_bytes()
            ).hexdigest(),
            "assets": selected_assets,
            "extensions": extensions,
        }
        for task_name in tasks:
            for provider in providers:
                output = result_path(args.output_dir, task_name, provider)
                if args.reuse_results and output.exists():
                    assert_reused_result_identity(
                        output,
                        selected_assets,
                        extensions[provider],
                    )
                    continue
                print(f"running {task_name}/{provider}", flush=True)
                run_case(
                    args.output_dir,
                    task_name,
                    provider,
                    args.gpu,
                    extension_root,
                    model_root,
                    dataset_root,
                    selected_assets,
                )
        write_task_reproduction(args.output_dir, tasks[0], reproduction)
    summary = write_summary(args.output_dir)
    print(json.dumps(summary["baseline_audit"], indent=2), flush=True)
    print(f"wrote {args.output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
