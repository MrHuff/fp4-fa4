#!/usr/bin/env python3
"""Attribute downstream NVFP4 P error on real ViT and BERT tensors.

This is an arithmetic diagnostic, not a kernel benchmark.  It keeps the
shiftless policy's hybrid exp2/E2M1 packer fixed while independently changing
the row shift, denominator, and P-scale scope.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

from eval_bert_mlm_attention import (
    LogitAccumulator,
    install_bert_attention,
    make_blocks,
    mask_block,
)
from eval_regular_attention import (
    classification_metrics,
    install_vit_attention,
    mean_records,
    tensor_metrics,
)


E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
LOG2E = math.log2(math.e)
LOG2_E2M1_MAX = math.log2(6.0)
AFFINE_A = 1.61131608
AFFINE_B = 0.93574703


@dataclass(frozen=True)
class Mode:
    name: str
    stabilized: bool
    denominator: str
    scale_scope: str
    exp_mode: str = "fast"
    global_row_max: bool = False
    max_estimator: str = "exact"


MODES = {
    mode.name: mode
    for mode in (
        Mode("fast", False, "sampled", "quarter32"),
        Mode("denom", False, "represented", "quarter32"),
        Mode("rowmax", True, "sampled", "quarter32"),
        Mode("rowmax-denom", True, "represented", "quarter32"),
        Mode(
            "q2-denom",
            True,
            "represented",
            "quarter32",
            max_estimator="q2",
        ),
        Mode(
            "q0x8-denom",
            True,
            "represented",
            "quarter32",
            max_estimator="q0x8",
        ),
        Mode(
            "q0x16-denom",
            True,
            "represented",
            "quarter32",
            max_estimator="q0x16",
        ),
        Mode(
            "q0-denom",
            True,
            "represented",
            "quarter32",
            max_estimator="q0",
        ),
        Mode(
            "q0q2x8-denom",
            True,
            "represented",
            "quarter32",
            max_estimator="q0q2x8",
        ),
        Mode(
            "staggered-x8-denom",
            True,
            "represented",
            "quarter32",
            max_estimator="staggered-x8",
        ),
        Mode("block16", True, "represented", "block16"),
        Mode("tile128", True, "represented", "tile128"),
        Mode(
            "rowwide",
            True,
            "represented",
            "rowwide",
            global_row_max=True,
        ),
        Mode(
            "shiftless-exact-exp",
            False,
            "represented",
            "quarter32",
            exp_mode="exact",
        ),
        Mode(
            "exact-exp",
            True,
            "represented",
            "quarter32",
            exp_mode="exact",
        ),
    )
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modes",
        default=",".join(MODES),
        help="Comma-separated mode names",
    )
    parser.add_argument("--vit-samples", type=int, default=20)
    parser.add_argument("--bert-samples", type=int, default=10)
    parser.add_argument(
        "--vit-model",
        default="nateraw/vit-base-patch16-224-cifar10",
    )
    parser.add_argument("--vit-dataset", default="uoft-cs/cifar10")
    parser.add_argument(
        "--bert-model",
        default="google-bert/bert-base-uncased",
    )
    parser.add_argument("--bert-dataset", default="Salesforce/wikitext")
    parser.add_argument(
        "--bert-dataset-config",
        default="wikitext-2-raw-v1",
    )
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=1)
    return parser.parse_args()


def direct_e4m3_scale(log2_scale: Any) -> tuple[Any, Any]:
    """Emulate scale-encoder mode 4 and return actual and encoded log2."""
    import torch

    bounded = torch.maximum(
        log2_scale,
        torch.full_like(log2_scale, -100.0),
    )
    code = torch.round(bounded * 8.0 + 56.0).to(torch.int32)
    code = code.clamp(8, 126)
    encoded_log2 = (code.float() - 56.0) * 0.125
    scale = (
        code.to(torch.uint8)
        .contiguous()
        .view(torch.float8_e4m3fn)
        .float()
    )
    return scale, encoded_log2


def quantize_e2m1(value: Any, levels: Any, midpoints: Any) -> Any:
    code = (value.unsqueeze(-1) > midpoints).sum(dim=-1)
    return levels[code]


def native_pair_mask(device: Any) -> Any:
    import torch

    mask = torch.zeros((4, 16, 1), device=device, dtype=torch.bool)
    for quarter in range(4):
        for pair in (quarter, quarter + 8, 4, 12):
            mask[quarter, pair, 0] = True
    return mask


def expand_scale(
    normalized_log2: Any,
    *,
    scope: str,
    rowwide_max: Any | None,
) -> tuple[Any, Any]:
    """Return scale and payload log2-scale broadcast over N128 values."""
    if scope == "quarter32":
        group_max = normalized_log2.amax(dim=(-1, -2), keepdim=True)
        scale, encoded = direct_e4m3_scale(
            group_max - LOG2_E2M1_MAX
        )
        return scale, encoded
    if scope == "block16":
        shape = normalized_log2.shape
        groups = normalized_log2.reshape(*shape[:-2], 2, 8, 2)
        group_max = groups.amax(dim=(-1, -2), keepdim=True)
        scale, encoded = direct_e4m3_scale(
            group_max - LOG2_E2M1_MAX
        )
        scale = scale.expand_as(groups).reshape(shape)
        encoded = encoded.expand_as(groups).reshape(shape)
        return scale, encoded
    if scope == "tile128":
        group_max = normalized_log2.amax(
            dim=(-3, -2, -1),
            keepdim=True,
        )
        scale, encoded = direct_e4m3_scale(
            group_max - LOG2_E2M1_MAX
        )
        return scale, encoded
    if scope == "rowwide":
        if rowwide_max is None:
            raise ValueError("rowwide scale requires a row maximum")
        group_max = rowwide_max[..., None, None, None]
        scale, encoded = direct_e4m3_scale(
            group_max - LOG2_E2M1_MAX
        )
        return scale, encoded
    raise ValueError(f"unsupported scale scope: {scope}")


def quantize_tile(
    normalized_log2: Any,
    *,
    mode: Mode,
    levels: Any,
    midpoints: Any,
    pair_mask: Any,
    rowwide_max: Any | None,
) -> tuple[Any, Any, Any]:
    """Pack one N128 tile and return P, selected denominator, and P sum."""
    import torch

    scale, encoded_log2 = expand_scale(
        normalized_log2,
        scope=mode.scale_scope,
        rowwide_max=rowwide_max,
    )
    payload_log2 = normalized_log2 - encoded_log2
    if mode.exp_mode == "exact":
        transformed = torch.exp2(payload_log2)
    elif mode.exp_mode == "fast":
        native = torch.exp2(payload_log2)
        affine = payload_log2 * AFFINE_A + AFFINE_B
        transformed = torch.where(pair_mask, native, affine)
    else:
        raise ValueError(f"unsupported exp mode: {mode.exp_mode}")

    represented = quantize_e2m1(
        transformed.clamp_min(0.0),
        levels,
        midpoints,
    ) * scale
    represented_sum = represented.sum(dim=(-3, -2, -1))

    if mode.denominator == "represented":
        denominator = represented_sum
    elif mode.denominator == "sampled":
        if mode.scale_scope != "quarter32":
            raise ValueError("sampled denominator requires quarter32 scale")
        sampled = torch.where(
            pair_mask,
            transformed.clamp(max=6.0),
            torch.zeros_like(transformed),
        )
        sampled = sampled.sum(dim=(-1, -2)) * 4.0
        sampled = sampled.clamp_min(6.0)
        quarter_scale = scale[..., 0, 0]
        denominator = (sampled * quarter_scale).sum(dim=-1)
    else:
        raise ValueError(
            f"unsupported denominator: {mode.denominator}"
        )
    return represented, denominator, represented_sum


def approximate_attention(
    query: Any,
    key: Any,
    value: Any,
    mode: Mode,
) -> tuple[Any, Any, dict[str, float]]:
    import torch
    import torch.nn.functional as functional

    scores = torch.matmul(
        query.float(),
        key.float().transpose(-1, -2),
    ) / math.sqrt(query.shape[-1])
    exact_probability = torch.softmax(scores, dim=-1)
    exact_context = torch.matmul(exact_probability, value.float())

    key_length = scores.shape[-1]
    padded_length = math.ceil(key_length / 128) * 128
    padded_scores = functional.pad(
        scores,
        (0, padded_length - key_length),
        value=-math.inf,
    )
    padded_value = functional.pad(
        value.float(),
        (0, 0, 0, padded_length - key_length),
    )

    levels = torch.tensor(
        E2M1_LEVELS,
        device=scores.device,
        dtype=torch.float32,
    )
    midpoints = torch.tensor(
        E2M1_MIDPOINTS,
        device=scores.device,
        dtype=torch.float32,
    )
    pair_mask = native_pair_mask(scores.device)

    final_row_max = padded_scores.amax(dim=-1)
    running_max = torch.full_like(final_row_max, -math.inf)
    numerator = torch.zeros(
        (*scores.shape[:-1], value.shape[-1]),
        device=scores.device,
        dtype=torch.float32,
    )
    denominator = torch.zeros_like(final_row_max)
    represented_total = torch.zeros_like(final_row_max)
    chunks: list[Any] = []
    anchors: list[Any] = []

    if mode.global_row_max:
        rowwide_max = torch.zeros_like(final_row_max)
    else:
        rowwide_max = None

    def estimate_tile_max(tile_scores: Any) -> Any:
        if mode.max_estimator == "exact":
            sampled = tile_scores
        elif mode.max_estimator == "q2":
            sampled = tile_scores[..., 64:96]
        elif mode.max_estimator == "q0x8":
            sampled = tile_scores[..., :8]
        elif mode.max_estimator == "q0x16":
            sampled = tile_scores[..., :16]
        elif mode.max_estimator == "q0":
            sampled = tile_scores[..., :32]
        elif mode.max_estimator == "q0q2x8":
            sampled = torch.cat(
                (
                    tile_scores[..., 0:8],
                    tile_scores[..., 80:88],
                ),
                dim=-1,
            )
        elif mode.max_estimator == "staggered-x8":
            sampled = torch.cat(
                (
                    tile_scores[..., 0:8],
                    tile_scores[..., 40:48],
                    tile_scores[..., 80:88],
                    tile_scores[..., 120:128],
                ),
                dim=-1,
            )
        else:
            raise ValueError(
                f"unsupported max estimator: {mode.max_estimator}"
            )
        return sampled.amax(dim=-1)

    for start in range(0, padded_length, 128):
        tile_scores = padded_scores[..., start : start + 128]
        tile_value = padded_value[..., start : start + 128, :]
        if mode.stabilized:
            if mode.global_row_max:
                anchor = final_row_max
                correction = torch.ones_like(final_row_max)
            else:
                tile_max = estimate_tile_max(tile_scores)
                next_max = torch.maximum(running_max, tile_max)
                correction = torch.exp(running_max - next_max)
                correction = torch.where(
                    torch.isfinite(running_max),
                    correction,
                    torch.zeros_like(correction),
                )
                running_max = next_max
                anchor = running_max
            normalized_log2 = (
                tile_scores - anchor.unsqueeze(-1)
            ) * LOG2E
        else:
            anchor = torch.zeros_like(final_row_max)
            correction = torch.ones_like(final_row_max)
            normalized_log2 = tile_scores * LOG2E

        tile_log2 = normalized_log2.reshape(
            *normalized_log2.shape[:-1],
            4,
            16,
            2,
        )
        represented, tile_denominator, tile_represented_sum = (
            quantize_tile(
                tile_log2,
                mode=mode,
                levels=levels,
                midpoints=midpoints,
                pair_mask=pair_mask,
                rowwide_max=rowwide_max,
            )
        )
        represented_flat = represented.reshape(
            *represented.shape[:-3],
            128,
        )
        numerator = (
            numerator * correction.unsqueeze(-1)
            + torch.matmul(
                represented_flat.unsqueeze(-2),
                tile_value.unsqueeze(-3),
            ).squeeze(-2)
        )
        denominator = denominator * correction + tile_denominator
        represented_total = (
            represented_total * correction + tile_represented_sum
        )
        chunks.append(represented_flat)
        anchors.append(anchor)

    final_anchor = (
        running_max
        if mode.stabilized and not mode.global_row_max
        else (
            final_row_max
            if mode.stabilized
            else torch.zeros_like(final_row_max)
        )
    )
    corrected_chunks = []
    for chunk, anchor in zip(chunks, anchors, strict=True):
        correction = (
            torch.exp(anchor - final_anchor)
            if mode.stabilized
            else torch.ones_like(anchor)
        )
        corrected_chunks.append(chunk * correction.unsqueeze(-1))
    probability = torch.cat(corrected_chunks, dim=-1)
    safe_denominator = denominator.clamp_min(
        torch.finfo(torch.float32).tiny
    )
    probability = probability / safe_denominator.unsqueeze(-1)
    probability = probability[..., :key_length]
    context = numerator / safe_denominator.unsqueeze(-1)

    mass = probability.sum(dim=-1)
    represented_safe = represented_total.clamp_min(
        torch.finfo(torch.float32).tiny
    )
    diagnostics = {
        "mean_probability_mass": float(mass.mean().item()),
        "mean_abs_probability_mass_error": float(
            (mass - 1.0).abs().mean().item()
        ),
        "mean_denominator_over_represented_sum": float(
            (safe_denominator / represented_safe).mean().item()
        ),
        "finite_fraction": float(
            torch.isfinite(context).float().mean().item()
        ),
    }
    diagnostics.update(
        {
            f"probability_{key}": value
            for key, value in tensor_metrics(
                probability,
                exact_probability,
            ).items()
        }
    )
    diagnostics.update(
        {
            f"context_{key}": value
            for key, value in tensor_metrics(
                context,
                exact_context,
            ).items()
        }
    )
    return context, exact_context, diagnostics


class AblationRunner:
    def __init__(self, modes: list[Mode]) -> None:
        self.modes = {mode.name: mode for mode in modes}
        self.mode = modes[0].name
        self.enabled = False
        self.sample_index = -1
        self.layer_records: dict[
            str, dict[int, list[dict[str, float]]]
        ] = defaultdict(lambda: defaultdict(list))

    def set_mode(self, mode: str) -> None:
        if mode not in self.modes:
            raise ValueError(f"unknown mode: {mode}")
        self.mode = mode

    def begin_sample(self, sample_index: int) -> None:
        self.sample_index = sample_index

    def __call__(
        self,
        query: Any,
        key: Any,
        value: Any,
        *,
        layer_index: int,
    ) -> Any:
        context, _, diagnostics = approximate_attention(
            query,
            key,
            value,
            self.modes[self.mode],
        )
        self.layer_records[self.mode][layer_index].append(diagnostics)
        return context.to(query.dtype)

    def summary(self, mode: str) -> dict[str, Any]:
        layers = self.layer_records[mode]
        return {
            "layer_output_error": {
                str(layer): mean_records(records)
                for layer, records in sorted(layers.items())
            },
            "mean_context_relative_l2": sum(
                record["context_relative_l2"]
                for records in layers.values()
                for record in records
            )
            / sum(len(records) for records in layers.values()),
            "mean_probability_relative_l2": sum(
                record["probability_relative_l2"]
                for records in layers.values()
                for record in records
            )
            / sum(len(records) for records in layers.values()),
            "mean_abs_probability_mass_error": sum(
                record["mean_abs_probability_mass_error"]
                for records in layers.values()
                for record in records
            )
            / sum(len(records) for records in layers.values()),
        }


def run_vit(
    args: argparse.Namespace,
    modes: list[Mode],
) -> dict[str, Any]:
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoImageProcessor,
        AutoModelForImageClassification,
    )

    runner = AblationRunner(modes)
    processor = AutoImageProcessor.from_pretrained(args.vit_model)
    model = AutoModelForImageClassification.from_pretrained(
        args.vit_model,
        torch_dtype=torch.bfloat16,
    ).eval().cuda()
    install_vit_attention(model, runner)
    dataset = load_dataset(args.vit_dataset, split="test")
    records: dict[str, list[dict[str, Any]]] = {
        mode.name: [] for mode in modes
    }

    with torch.inference_mode():
        for index in range(args.vit_samples):
            item = dataset[index]
            pixels = processor(
                images=item["img"],
                return_tensors="pt",
            ).pixel_values.cuda().to(torch.bfloat16)
            runner.enabled = False
            baseline = model(pixel_values=pixels).logits.float()
            for mode in modes:
                runner.begin_sample(index)
                runner.set_mode(mode.name)
                runner.enabled = True
                actual = model(pixel_values=pixels).logits.float()
                records[mode.name].append(
                    {
                        "baseline_prediction": int(
                            baseline.argmax(dim=-1).item()
                        ),
                        "fp4_prediction": int(
                            actual.argmax(dim=-1).item()
                        ),
                        "label": int(item["label"]),
                        "baseline_logits": baseline[0].cpu().tolist(),
                        "fp4_logits": actual[0].cpu().tolist(),
                    }
                )
            if (
                (index + 1) % args.progress_every == 0
                or index + 1 == args.vit_samples
            ):
                print(
                    f"[ViT {index + 1}/{args.vit_samples}]",
                    flush=True,
                )

    result = {
        mode.name: {
            "classification": classification_metrics(records[mode.name]),
            "attention": runner.summary(mode.name),
        }
        for mode in modes
    }
    del model
    torch.cuda.empty_cache()
    return result


def run_bert(
    args: argparse.Namespace,
    modes: list[Mode],
) -> dict[str, Any]:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    runner = AblationRunner(modes)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    model = AutoModelForMaskedLM.from_pretrained(
        args.bert_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval().cuda()
    install_bert_attention(model, runner)
    dataset = load_dataset(
        args.bert_dataset,
        args.bert_dataset_config,
        split="test",
    )
    blocks = make_blocks(dataset, tokenizer, args.bert_samples)
    accumulators = {
        mode.name: LogitAccumulator() for mode in modes
    }

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
            baseline = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits
            for mode in modes:
                runner.begin_sample(index)
                runner.set_mode(mode.name)
                runner.enabled = True
                actual = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).logits
                accumulators[mode.name].add(
                    baseline,
                    actual,
                    labels,
                )
            if (
                (index + 1) % args.progress_every == 0
                or index + 1 == args.bert_samples
            ):
                print(
                    f"[BERT {index + 1}/{args.bert_samples}]",
                    flush=True,
                )

    result = {
        mode.name: {
            "masked_language_modeling": accumulators[
                mode.name
            ].summary(),
            "attention": runner.summary(mode.name),
        }
        for mode in modes
    }
    del model
    torch.cuda.empty_cache()
    return result


def main() -> None:
    import torch

    args = parse_args()
    names = [name for name in args.modes.split(",") if name]
    unknown = sorted(set(names) - set(MODES))
    if unknown:
        raise ValueError(f"unknown modes: {', '.join(unknown)}")
    if not names:
        raise ValueError("at least one mode is required")
    if args.vit_samples < 0 or args.bert_samples < 0:
        raise ValueError("sample counts cannot be negative")
    modes = [MODES[name] for name in names]

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.set_device(0)
    result: dict[str, Any] = {
        "schema": "tk_hao_nv_p_ablation_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "modes": {
            mode.name: {
                "stabilized": mode.stabilized,
                "denominator": mode.denominator,
                "scale_scope": mode.scale_scope,
                "exp_mode": mode.exp_mode,
                "global_row_max": mode.global_row_max,
                "max_estimator": mode.max_estimator,
            }
            for mode in modes
        },
    }
    if args.vit_samples:
        result["vit"] = run_vit(args, modes)
    if args.bert_samples:
        result["bert"] = run_bert(args, modes)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
