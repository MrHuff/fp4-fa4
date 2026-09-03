#!/usr/bin/env python3
"""Evaluate the stabilized FP4 attention kernel in BERT masked LM."""

from __future__ import annotations

import argparse
import json
import math
import types
from pathlib import Path
from typing import Any

from eval_regular_attention import (
    DEFAULT_HAO_ROOT,
    RegularAttentionRunner,
    add_asset_identity_arguments,
    load_extension,
    portable_asset_identity,
    portable_file_identity,
)


DEFAULT_OUTPUT = Path(
    "results/fp4_fa4_downstream_20260728/bert_wikitext_mlm.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="google-bert/bert-base-uncased",
    )
    parser.add_argument("--dataset", default="Salesforce/wikitext")
    add_asset_identity_arguments(parser)
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=256,
        help="Fixed BERT block length; must not exceed 512 or the kernel S.",
    )
    parser.add_argument("--scale-sweep-samples", type=int, default=4)
    parser.add_argument(
        "--scale-factors",
        default="1,2,4,8,12,16,24,32,320,448,1440,1536,2688",
    )
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
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--finite-diagnostics",
        action="store_true",
        help="Record per-layer tensor ranges when debugging non-finite output.",
    )
    parser.add_argument(
        "--stop-on-nonfinite",
        action="store_true",
        help="Stop after the first sample that produces a non-finite row.",
    )
    parser.add_argument(
        "--interleave-kv-quarters",
        action="store_true",
        help=(
            "Apply the same stride-4 permutation to K and V within each "
            "128-token tile so score quarter 0 samples the full tile."
        ),
    )
    parser.add_argument(
        "--global-anchor-kv",
        action="store_true",
        help=(
            "Place 32 keys sampled across the logical sequence in physical "
            "Q0 of the first N128 tile, with the same permutation for V."
        ),
    )
    parser.add_argument(
        "--global-anchor-samples",
        type=int,
        choices=(32, 64, 128),
        default=32,
        help="Number of globally distributed keys placed first.",
    )
    parser.add_argument(
        "--key-centering",
        choices=("none", "projection-bias", "sequence-mean"),
        default="none",
        help=(
            "Subtract a key vector shared by every token. This leaves exact "
            "softmax unchanged while controlling shiftless score offsets."
        ),
    )
    parser.add_argument(
        "--score-shift",
        type=float,
        default=0.0,
        help=(
            "Subtract this nonnegative constant from every normalized QK "
            "score through an otherwise-unused padded Q/K dimension."
        ),
    )
    parser.add_argument(
        "--score-shift-predictor",
        choices=(
            "fixed",
            "q-rms",
            "qk-rms",
            "sample32-rowmax",
            "sample-rowmax",
            "exact-rowmax",
        ),
        default="fixed",
        help="Statistic multiplied by --score-shift for each query row.",
    )
    parser.add_argument(
        "--score-shift-bias",
        type=float,
        default=0.0,
        help="Additive normalized-score margin after the row-shift predictor.",
    )
    return parser.parse_args()


def install_bert_attention(
    model: Any,
    runner: RegularAttentionRunner,
) -> None:
    for layer_index, layer in enumerate(model.bert.encoder.layer):
        attention = layer.attention.self
        original_forward = attention.forward

        def patched_forward(
            this: Any,
            hidden_states: Any,
            attention_mask: Any = None,
            head_mask: Any = None,
            encoder_hidden_states: Any = None,
            encoder_attention_mask: Any = None,
            past_key_value: Any = None,
            output_attentions: bool = False,
            *,
            _layer_index: int = layer_index,
            _original_forward: Any = original_forward,
        ) -> tuple[Any, ...]:
            if not runner.enabled:
                return _original_forward(
                    hidden_states,
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                )
            if (
                head_mask is not None
                or encoder_hidden_states is not None
                or encoder_attention_mask is not None
                or past_key_value is not None
                or output_attentions
            ):
                raise ValueError(
                    "the FP4 BERT adapter supports plain encoder "
                    "self-attention only"
                )
            if this.position_embedding_type != "absolute":
                raise ValueError("relative-position BERT is unsupported")

            key_valid_mask = None
            if attention_mask is not None:
                key_valid_mask = (attention_mask == 0).reshape(
                    attention_mask.shape[0],
                    attention_mask.shape[-1],
                )

            query = this.transpose_for_scores(this.query(hidden_states))
            key = this.transpose_for_scores(this.key(hidden_states))
            key = runner.center_key(key, this.key.bias)
            value = this.transpose_for_scores(this.value(hidden_states))
            context = runner(
                query,
                key,
                value,
                layer_index=_layer_index,
                key_valid_mask=key_valid_mask,
            )
            context = context.permute(0, 2, 1, 3).contiguous()
            context = context.view(
                *context.shape[:-2],
                this.all_head_size,
            )
            return (context,)

        attention.forward = types.MethodType(patched_forward, attention)


def make_blocks(
    dataset: Any,
    tokenizer: Any,
    samples: int,
    sequence_length: int,
) -> list[Any]:
    import torch

    token_ids: list[int] = []
    content_length = sequence_length - 2
    required = samples * content_length
    for text in dataset["text"]:
        if not text.strip():
            continue
        token_ids.extend(
            tokenizer(
                text,
                add_special_tokens=False,
            ).input_ids
        )
        if len(token_ids) >= required:
            break
    if len(token_ids) < required:
        raise RuntimeError(
            f"dataset supplied {len(token_ids)} tokens, need {required}"
        )
    blocks = []
    for index in range(samples):
        begin = index * content_length
        content = token_ids[begin : begin + content_length]
        blocks.append(
            torch.tensor(
                [tokenizer.cls_token_id, *content, tokenizer.sep_token_id],
                dtype=torch.long,
            )
        )
    return blocks


def mask_block(
    input_ids: Any,
    tokenizer: Any,
    *,
    seed: int,
) -> tuple[Any, Any]:
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    eligible = torch.zeros_like(input_ids, dtype=torch.bool)
    eligible[1:-1] = True
    selected = (torch.rand(input_ids.shape, generator=generator) < 0.15)
    selected &= eligible
    if not bool(selected.any()):
        selected[1] = True

    labels = input_ids.clone()
    labels[~selected] = -100
    masked = input_ids.clone()
    replacement_draw = torch.rand(input_ids.shape, generator=generator)
    replace_with_mask = selected & (replacement_draw < 0.8)
    replace_random = (
        selected
        & (replacement_draw >= 0.8)
        & (replacement_draw < 0.9)
    )
    masked[replace_with_mask] = tokenizer.mask_token_id
    random_ids = torch.randint(
        0,
        tokenizer.vocab_size,
        input_ids.shape,
        generator=generator,
    )
    masked[replace_random] = random_ids[replace_random]
    return masked, labels


class LogitAccumulator:
    def __init__(self) -> None:
        self.tokens = 0
        self.baseline_correct = 0
        self.fp4_correct = 0
        self.agree = 0
        self.baseline_nll = 0.0
        self.fp4_nll = 0.0
        self.kl = 0.0
        self.squared_error = 0.0
        self.squared_reference = 0.0
        self.dot = 0.0
        self.squared_actual = 0.0
        self.logit_values = 0
        self.max_abs = 0.0

    def add(
        self,
        baseline_logits: Any,
        fp4_logits: Any,
        labels: Any,
    ) -> dict[str, Any]:
        import torch
        import torch.nn.functional as functional

        selected = labels != -100
        target = labels[selected]
        baseline = baseline_logits[selected].float()
        fp4 = fp4_logits[selected].float()
        count = int(target.numel())
        baseline_prediction = baseline.argmax(dim=-1)
        fp4_prediction = fp4.argmax(dim=-1)
        baseline_nll = functional.cross_entropy(
            baseline,
            target,
            reduction="sum",
        )
        fp4_nll = functional.cross_entropy(
            fp4,
            target,
            reduction="sum",
        )
        baseline_probability = baseline.softmax(dim=-1)
        kl = functional.kl_div(
            fp4.log_softmax(dim=-1),
            baseline_probability,
            reduction="sum",
        )
        delta = fp4 - baseline

        self.tokens += count
        self.baseline_correct += int(
            (baseline_prediction == target).sum().item()
        )
        self.fp4_correct += int((fp4_prediction == target).sum().item())
        self.agree += int(
            (baseline_prediction == fp4_prediction).sum().item()
        )
        self.baseline_nll += float(baseline_nll.item())
        self.fp4_nll += float(fp4_nll.item())
        self.kl += float(kl.item())
        self.squared_error += float(delta.square().sum().item())
        self.squared_reference += float(baseline.square().sum().item())
        self.dot += float((fp4 * baseline).sum().item())
        self.squared_actual += float(fp4.square().sum().item())
        self.logit_values += int(baseline.numel())
        self.max_abs = max(
            self.max_abs,
            float(delta.abs().max().item()),
        )
        return {
            "masked_tokens": count,
            "baseline_correct": int(
                (baseline_prediction == target).sum().item()
            ),
            "fp4_correct": int((fp4_prediction == target).sum().item()),
            "prediction_agreement": int(
                (baseline_prediction == fp4_prediction).sum().item()
            ),
            "baseline_nll": float(baseline_nll.item()),
            "fp4_nll": float(fp4_nll.item()),
        }

    def summary(self) -> dict[str, Any]:
        baseline_loss = self.baseline_nll / self.tokens
        fp4_loss = self.fp4_nll / self.tokens
        rmse = math.sqrt(self.squared_error / self.logit_values)
        reference_rms = math.sqrt(
            self.squared_reference / self.logit_values
        )
        cosine = self.dot / math.sqrt(
            self.squared_actual * self.squared_reference
        )
        return {
            "masked_tokens": self.tokens,
            "baseline_loss": baseline_loss,
            "fp4_loss": fp4_loss,
            "baseline_perplexity": math.exp(baseline_loss),
            "fp4_perplexity": math.exp(fp4_loss),
            "baseline_masked_accuracy": (
                self.baseline_correct / self.tokens
            ),
            "fp4_masked_accuracy": self.fp4_correct / self.tokens,
            "masked_top1_agreement": self.agree / self.tokens,
            "mean_kl_fp4_vs_baseline": self.kl / self.tokens,
            "logit_error": {
                "cosine": cosine,
                "rmse": rmse,
                "reference_rms": reference_rms,
                "relative_l2": rmse / reference_rms,
                "max_abs": self.max_abs,
            },
        }


def main() -> None:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForMaskedLM, AutoTokenizer

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
    if not math.isfinite(args.score_shift) or args.score_shift < 0.0:
        raise ValueError("--score-shift must be finite and nonnegative")
    if not math.isfinite(args.score_shift_bias):
        raise ValueError("--score-shift-bias must be finite")
    scale_factors = [
        float(item) for item in args.scale_factors.split(",") if item
    ]
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
        scale_factors=scale_factors,
        scale_sweep_samples=args.scale_sweep_samples,
        finite_diagnostics=args.finite_diagnostics,
        interleave_quarters=args.interleave_kv_quarters,
        global_anchor=args.global_anchor_kv,
        global_anchor_samples=args.global_anchor_samples,
        key_centering=args.key_centering,
        score_shift=args.score_shift,
        score_shift_predictor=args.score_shift_predictor,
        score_shift_bias=args.score_shift_bias,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(
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
    if args.sequence_length > runner.target_seqlen:
        raise ValueError(
            f"sequence length {args.sequence_length} exceeds kernel S"
            f"{runner.target_seqlen}"
        )
    blocks = make_blocks(
        dataset,
        tokenizer,
        args.samples,
        args.sequence_length,
    )

    accumulator = LogitAccumulator()
    records = []
    with torch.inference_mode():
        for index, block in enumerate(blocks):
            masked, labels = mask_block(
                block,
                tokenizer,
                seed=args.seed + index,
            )
            input_ids = masked.unsqueeze(0).cuda()
            labels = labels.unsqueeze(0).cuda()
            attention_mask = torch.ones_like(input_ids)

            runner.enabled = False
            baseline_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits
            runner.begin_sample(index)
            runner.enabled = True
            fp4_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits
            record = {"index": index}
            record.update(
                accumulator.add(
                    baseline_logits,
                    fp4_logits,
                    labels,
                )
            )
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
                    f"masked_tokens={accumulator.tokens}",
                    flush=True,
                )

    result = {
        "schema": "tk_fp4_bert_mlm_eval_v1",
        "model": model_identity,
        "dataset": dataset_identity,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "seed": args.seed,
        "extension": extension_identity,
        "protocol": {
            "sequence_length": args.sequence_length,
            "content_tokens_per_block": args.sequence_length - 2,
            "mlm_probability": 0.15,
            "mlm_replacement": "80% mask, 10% random, 10% unchanged",
            "full_noncausal_attention": True,
            "interleave_kv_quarters": args.interleave_kv_quarters,
            "global_anchor_kv": args.global_anchor_kv,
            "global_anchor_samples": args.global_anchor_samples,
            "key_centering": args.key_centering,
            "score_shift": args.score_shift,
            "score_shift_predictor": args.score_shift_predictor,
            "score_shift_bias": args.score_shift_bias,
            "timing_scope": (
                "accuracy only; dynamic Q/K/V quantization is not timed"
            ),
            "requested_samples": args.samples,
            "stop_on_nonfinite": args.stop_on_nonfinite,
        },
        "masked_language_modeling": accumulator.summary(),
        "attention": runner.summary(),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.output)
    print(
        json.dumps(result["masked_language_modeling"], indent=2),
        flush=True,
    )
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
