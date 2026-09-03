#!/usr/bin/env python3
"""Evaluate an FP4 attention provider in a BERT classification model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval_bert_mlm_attention import install_bert_attention
from eval_regular_attention import (
    DEFAULT_HAO_ROOT,
    RegularAttentionRunner,
    add_asset_identity_arguments,
    classification_metrics,
    load_extension,
    portable_asset_identity,
    portable_file_identity,
)


DEFAULT_OUTPUT = Path(
    "results/fp4_fa4_downstream_20260728/bert_sst2.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="textattack/bert-base-uncased-SST-2",
    )
    parser.add_argument("--dataset", default="glue")
    add_asset_identity_arguments(parser)
    parser.add_argument("--dataset-config", default="sst2")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--text-field", default="sentence")
    parser.add_argument("--text-pair-field", default="")
    parser.add_argument("--label-field", default="label")
    parser.add_argument("--samples", type=int, default=872)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--extension-module", required=True)
    parser.add_argument(
        "--attention-backend",
        choices=("tk", "hao-native"),
        default="tk",
    )
    parser.add_argument("--hao-root", type=Path, default=DEFAULT_HAO_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--finite-diagnostics", action="store_true")
    parser.add_argument("--stop-on-nonfinite", action="store_true")
    parser.add_argument("--scale-sweep-samples", type=int, default=0)
    parser.add_argument("--global-anchor-kv", action="store_true")
    parser.add_argument(
        "--global-anchor-samples",
        type=int,
        choices=(32, 64, 128),
        default=32,
    )
    return parser.parse_args()


def main() -> None:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    args = parse_args()
    model_identity = portable_asset_identity(
        args.model,
        identifier=args.model_identifier,
        revision=args.model_revision,
        tree_sha256=args.model_tree_sha256,
    )
    dataset_identity = portable_asset_identity(
        args.dataset,
        identifier=args.dataset_identifier,
        revision=args.dataset_revision,
        tree_sha256=args.dataset_tree_sha256,
    )
    extension_identity = portable_file_identity(args.extension)
    if args.samples < 1:
        raise ValueError("--samples must be positive")
    if args.sequence_length < 2 or args.sequence_length > 512:
        raise ValueError("--sequence-length must be in [2, 512]")
    if args.progress_every < 1:
        raise ValueError("--progress-every must be positive")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.set_device(0)

    extension = load_extension(
        args.extension.resolve(),
        args.extension_module,
    )
    runner = RegularAttentionRunner(
        extension,
        attention_backend=args.attention_backend,
        hao_root=args.hao_root,
        mask_value=10.0,
        scale_factors=[1.0, 32.0, 448.0],
        scale_sweep_samples=args.scale_sweep_samples,
        finite_diagnostics=args.finite_diagnostics,
        global_anchor=args.global_anchor_kv,
        global_anchor_samples=args.global_anchor_samples,
    )
    if args.sequence_length > runner.target_seqlen:
        raise ValueError(
            f"sequence length {args.sequence_length} exceeds kernel S"
            f"{runner.target_seqlen}"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval().cuda()
    install_bert_attention(model, runner)
    dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.split,
    )
    if args.samples > len(dataset):
        raise ValueError(
            f"requested {args.samples} samples from {len(dataset)} records"
        )

    records: list[dict[str, Any]] = []
    baseline_finite = True
    fp4_finite = True
    with torch.inference_mode():
        for index in range(args.samples):
            item = dataset[index]
            text_pair = (
                item[args.text_pair_field]
                if args.text_pair_field
                else None
            )
            encoded = tokenizer(
                item[args.text_field],
                text_pair,
                truncation=True,
                padding="max_length",
                max_length=args.sequence_length,
                return_tensors="pt",
            )
            encoded = {
                key: value.cuda()
                for key, value in encoded.items()
            }

            runner.enabled = False
            baseline_logits = model(**encoded).logits.float()
            runner.begin_sample(index)
            runner.enabled = True
            fp4_logits = model(**encoded).logits.float()
            baseline_finite &= bool(torch.isfinite(baseline_logits).all())
            fp4_finite &= bool(torch.isfinite(fp4_logits).all())
            record = {
                "index": index,
                "label": int(item[args.label_field]),
                "valid_tokens": int(encoded["attention_mask"].sum().item()),
                "baseline_prediction": int(
                    baseline_logits.argmax(dim=-1).item()
                ),
                "fp4_prediction": int(fp4_logits.argmax(dim=-1).item()),
                "baseline_logits": baseline_logits[0].cpu().tolist(),
                "fp4_logits": fp4_logits[0].cpu().tolist(),
            }
            records.append(record)
            if args.stop_on_nonfinite and runner.nonfinite_output_count:
                print(
                    f"stopping after sample {index}: "
                    f"{runner.nonfinite_output_count} non-finite rows",
                    flush=True,
                )
                break
            if (
                (index + 1) % args.progress_every == 0
                or index + 1 == args.samples
            ):
                print(
                    f"[{index + 1}/{args.samples}] "
                    f"tokens={record['valid_tokens']} "
                    f"label={record['label']} "
                    f"bf16={record['baseline_prediction']} "
                    f"fp4={record['fp4_prediction']}",
                    flush=True,
                )

    metrics = classification_metrics(records)
    metrics.update(
        {
            "all_baseline_logits_finite": baseline_finite,
            "all_fp4_logits_finite": fp4_finite,
        }
    )
    result = {
        "schema": "tk_fp4_bert_sequence_classification_eval_v1",
        "model": model_identity,
        "dataset": dataset_identity,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "seed": args.seed,
        "extension": extension_identity,
        "protocol": {
            "sequence_length": args.sequence_length,
            "samples": args.samples,
            "right_padding": True,
            "full_noncausal_attention": True,
            "global_anchor_kv": args.global_anchor_kv,
            "global_anchor_samples": args.global_anchor_samples,
            "timing_scope": (
                "accuracy only; dynamic Q/K/V quantization is not timed"
            ),
            "requested_samples": args.samples,
            "stop_on_nonfinite": args.stop_on_nonfinite,
        },
        "classification": metrics,
        "attention": runner.summary(),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
