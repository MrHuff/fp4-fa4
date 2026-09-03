#!/usr/bin/env python3
"""Run the named NVFP4 policies through the retained ViT and BERT evals."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
POLICY_IDS = {
    "fast": 1,
    "balanced": 2,
    "accurate": 3,
    "exact": 4,
    "hao-cosine": 5,
    "hao-l2": 6,
    "universal": 7,
    "fast-accurate": 8,
    "fast-corrected": 9,
    "fast-adaptive": 10,
}
POLICY_GLOBAL_ANCHOR_SAMPLES = {
    "fast-accurate": 64,
    "fast-corrected": 32,
    "fast-adaptive": 32,
}
DEFAULT_POLICIES = ("universal",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        action="append",
        choices=tuple(POLICY_IDS),
        dest="policies",
    )
    parser.add_argument("--vit-samples", type=int, default=1000)
    parser.add_argument("--bert-samples", type=int, default=200)
    parser.add_argument(
        "--build-root",
        type=Path,
        default=Path("/tmp/tk_hao_nv_policy_downstream"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--reuse-results", action="store_true")
    return parser.parse_args()


def environment() -> dict[str, str]:
    return dict(os.environ)


def run(
    command: list[str],
    *,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=HERE,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_policy(
    *,
    policy: str,
    build_root: Path,
    rebuild: bool,
    env: dict[str, str],
) -> tuple[Path, str, str]:
    slug = policy.replace("-", "_")
    module = f"_C_tk_nv_policy_{slug}_b1s256h16"
    extension = build_root / f"{module}.so"
    if extension.exists() and not rebuild:
        return extension, module, "reused existing extension\n"

    build_root.mkdir(parents=True, exist_ok=True)
    command = [
        "make",
        "-B",
        "-f",
        "Makefile.hao_direct_fp4pv",
        "-j1",
        f"OUT={extension}",
        f"MODULE={module}",
        "HAO_BATCH=1",
        "HAO_SEQ_LEN=256",
        "HAO_HEADS=16",
        "HAO_QK_SCALE_MODE=0",
        "HAO_PV_SCALE_MODE=0",
        f"HAO_FP4PV_NV_POLICY={policy}",
        "NVCC_SPLIT_COMPILE=2",
    ]
    completed = run(command, env=env)
    return extension, module, completed.stdout + completed.stderr


def run_eval(
    *,
    script: str,
    samples: int,
    extension: Path,
    module: str,
    output: Path,
    global_anchor_samples: int | None,
    env: dict[str, str],
) -> str:
    command = [
        sys.executable,
        script,
        "--samples",
        str(samples),
        "--scale-sweep-samples",
        "0",
        "--progress-every",
        str(max(1, samples // 10)),
        "--extension",
        str(extension),
        "--extension-module",
        module,
        "--output",
        str(output),
    ]
    if script == "eval_regular_attention.py":
        command.extend(["--mask-value", "10"])
    if global_anchor_samples is not None:
        command.extend(
            [
                "--global-anchor-kv",
                "--global-anchor-samples",
                str(global_anchor_samples),
            ]
        )
    completed = run(command, env=env)
    return completed.stdout + completed.stderr


def attention_summary(result: dict[str, Any]) -> dict[str, float]:
    layer_errors = result["attention"]["layer_output_error"].values()
    layer_errors = list(layer_errors)
    return {
        "mean_layer_cosine": statistics.fmean(
            layer["cosine"] for layer in layer_errors
        ),
        "mean_layer_relative_l2": statistics.fmean(
            layer["relative_l2"] for layer in layer_errors
        ),
        "max_layer_relative_l2": max(
            layer["relative_l2"] for layer in layer_errors
        ),
    }


def vit_summary(result: dict[str, Any]) -> dict[str, Any]:
    classification = result["classification"]
    return {
        "samples": classification["samples"],
        "baseline_accuracy": classification["baseline_accuracy"],
        "fp4_accuracy": classification["fp4_accuracy"],
        "top1_agreement": classification["top1_agreement"],
        "logit_cosine": classification["logit_error"]["cosine"],
        "logit_relative_l2": classification["logit_error"][
            "relative_l2"
        ],
        "mean_kl": classification["mean_kl_fp4_vs_baseline"],
        **attention_summary(result),
    }


def bert_summary(result: dict[str, Any]) -> dict[str, Any]:
    mlm = result["masked_language_modeling"]
    return {
        "masked_tokens": mlm["masked_tokens"],
        "baseline_loss": mlm["baseline_loss"],
        "fp4_loss": mlm["fp4_loss"],
        "baseline_masked_accuracy": mlm[
            "baseline_masked_accuracy"
        ],
        "fp4_masked_accuracy": mlm["fp4_masked_accuracy"],
        "top1_agreement": mlm["masked_top1_agreement"],
        "logit_cosine": mlm["logit_error"]["cosine"],
        "logit_relative_l2": mlm["logit_error"]["relative_l2"],
        "mean_kl": mlm["mean_kl_fp4_vs_baseline"],
        **attention_summary(result),
    }


def main() -> None:
    args = parse_args()
    policies = args.policies or list(DEFAULT_POLICIES)
    if args.vit_samples < 1 or args.bert_samples < 1:
        raise ValueError("sample counts must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = environment()
    summaries = []

    for policy in policies:
        global_anchor_samples = POLICY_GLOBAL_ANCHOR_SAMPLES.get(policy)
        extension, module, build_log = build_policy(
            policy=policy,
            build_root=args.build_root,
            rebuild=args.rebuild,
            env=env,
        )
        (args.output_dir / f"build_{policy}.log").write_text(
            build_log,
            encoding="utf-8",
        )

        vit_path = args.output_dir / f"vit_{policy}.json"
        bert_path = args.output_dir / f"bert_{policy}.json"
        if not (args.reuse_results and vit_path.exists()):
            vit_log = run_eval(
                script="eval_regular_attention.py",
                samples=args.vit_samples,
                extension=extension,
                module=module,
                output=vit_path,
                global_anchor_samples=global_anchor_samples,
                env=env,
            )
            (args.output_dir / f"vit_{policy}.log").write_text(
                vit_log,
                encoding="utf-8",
            )
        if not (args.reuse_results and bert_path.exists()):
            bert_log = run_eval(
                script="eval_bert_mlm_attention.py",
                samples=args.bert_samples,
                extension=extension,
                module=module,
                output=bert_path,
                global_anchor_samples=global_anchor_samples,
                env=env,
            )
            (args.output_dir / f"bert_{policy}.log").write_text(
                bert_log,
                encoding="utf-8",
            )

        vit = json.loads(vit_path.read_text(encoding="utf-8"))
        bert = json.loads(bert_path.read_text(encoding="utf-8"))
        vit_policy_id = vit["attention"]["topology"].get("nv_policy_id")
        bert_policy_id = bert["attention"]["topology"].get("nv_policy_id")
        if vit_policy_id != POLICY_IDS[policy]:
            raise RuntimeError(
                f"ViT {policy} emitted policy ID {vit_policy_id}"
            )
        if bert_policy_id != POLICY_IDS[policy]:
            raise RuntimeError(
                f"BERT {policy} emitted policy ID {bert_policy_id}"
            )
        summaries.append(
            {
                "policy": policy,
                "policy_id": POLICY_IDS[policy],
                "vit": vit_summary(vit),
                "bert": bert_summary(bert),
            }
        )

    summary = {
        "schema": "tk_hao_nv_policy_downstream_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "shape": [1, 256, 16, 128],
            "vit_samples": args.vit_samples,
            "bert_samples": args.bert_samples,
        },
        "policies": summaries,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
