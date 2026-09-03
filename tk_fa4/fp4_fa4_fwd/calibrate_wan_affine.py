#!/usr/bin/env python3
"""Teacher-forced Wan sweep for affine E2M1 code-map candidates.

Each candidate is enabled in one video self-attention layer at a time while
every other self-attention layer remains BF16. This isolates local candidate
error without repeatedly loading the Wan checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from eval_regular_attention import load_extension, parse_layer_indices, tensor_metrics
from eval_wan_video import (
    DEFAULT_PROMPT,
    latent_token_count,
    make_runner,
    restore_processors,
    run_provider,
)


DEFAULT_MODEL = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="NAME=PATH:MODULE",
        help="Candidate extension. Names must be unique.",
    )
    parser.add_argument("--layers", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num-frames", type=int, default=17)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def parse_candidates(
    specifications: list[str],
) -> dict[str, tuple[Path, str]]:
    candidates: dict[str, tuple[Path, str]] = {}
    for specification in specifications:
        try:
            name, extension_text = specification.split("=", 1)
            path_text, module = extension_text.rsplit(":", 1)
        except ValueError as error:
            raise ValueError(
                "--candidate must use NAME=PATH:MODULE"
            ) from error
        if not name or name in candidates:
            raise ValueError(f"duplicate or empty candidate name: {name!r}")
        candidates[name] = (Path(path_text).resolve(), module)
    return candidates


def compact_record(record: dict[str, Any], candidate: str, layer: int) -> None:
    record.pop("runner", None)
    record["candidate"] = candidate
    record["active_layer"] = layer


def main() -> None:
    args = parse_args()
    candidates = parse_candidates(args.candidate)
    layers = sorted(parse_layer_indices(args.layers))

    from diffusers import WanPipeline

    runners = {}
    candidate_metadata = {}
    for name, (path, module) in candidates.items():
        extension = load_extension(path, module)
        runners[name] = make_runner(extension, "tk", "none")
        topology = dict(extension.read_hao_direct_topology())
        candidate_metadata[name] = {
            "path": str(path),
            "module": module,
            "affine_a": topology.get("mx_affine_a"),
            "affine_b": topology.get("mx_affine_b"),
        }

    pipeline = WanPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        local_files_only=args.local_files_only,
    )
    pipeline.to(args.device)
    pipeline.set_progress_bar_config(disable=False)
    original_processors = [
        block.attn1.processor for block in pipeline.transformer.blocks
    ]
    model_shape = {
        "layers": len(pipeline.transformer.blocks),
        "heads": int(pipeline.transformer.config.num_attention_heads),
        "head_dim": int(pipeline.transformer.config.attention_head_dim),
        "sequence_length": latent_token_count(
            pipeline, args.height, args.width, args.num_frames
        ),
    }
    if layers[-1] >= model_shape["layers"]:
        raise ValueError(
            f"layer {layers[-1]} is outside a {model_shape['layers']}-layer model"
        )
    expected = (
        model_shape["sequence_length"],
        model_shape["heads"],
        model_shape["head_dim"],
    )
    for name, runner in runners.items():
        actual = (runner.target_seqlen, runner.target_heads, runner.target_dim)
        if actual != expected:
            raise ValueError(
                f"candidate {name} has shape {actual}, expected {expected}"
            )

    args.output_type = "latent"
    args.key_centering = "none"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}

    payload = {
        "schema": "tk_fp4_fa4_wan_affine_teacher_forced_v2",
        "model": args.model,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "generation": {
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "seed": args.seed,
            "output_type": args.output_type,
        },
        "model_attention_shape": model_shape,
        "layers": layers,
        "candidates": candidate_metadata,
        "providers": records,
        "comparisons": comparisons,
    }

    def checkpoint() -> None:
        args.output.write_text(json.dumps(payload, indent=2) + "\n")

    reference, reference_record = run_provider(
        pipeline,
        "bf16",
        None,
        original_processors,
        args,
    )
    reference_record["status"] = "complete"
    records["bf16"] = reference_record
    checkpoint()

    for layer in layers:
        for candidate, runner in runners.items():
            provider = f"{candidate}_l{layer:02d}"
            try:
                output, record = run_provider(
                    pipeline,
                    "tk",
                    runner,
                    original_processors,
                    args,
                    lowp_layers={layer},
                )
                compact_record(record, candidate, layer)
                record["status"] = "complete"
                records[provider] = record
                comparisons[f"{provider}_vs_bf16"] = tensor_metrics(
                    output, reference
                )
                metric = comparisons[f"{provider}_vs_bf16"]
                print(
                    f"{provider}: cosine={metric['cosine']:.8f} "
                    f"relative_l2={metric['relative_l2']:.8f}",
                    flush=True,
                )
            except Exception as error:
                records[provider] = {
                    "status": "failed",
                    "candidate": candidate,
                    "active_layer": layer,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                print(f"{provider}: failed: {error}", flush=True)
            checkpoint()

    restore_processors(pipeline.transformer, original_processors)


if __name__ == "__main__":
    main()
