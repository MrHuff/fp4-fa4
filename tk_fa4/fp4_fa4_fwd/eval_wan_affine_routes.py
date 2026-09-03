#!/usr/bin/env python3
"""Evaluate several frozen Wan layer-routing manifests in one model load."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

try:
    from .eval_regular_attention import (
        authenticate_asset_manifest,
        load_extension,
        parse_layer_indices,
        portable_asset_identity,
        portable_file_identity,
        tensor_metrics,
    )
    from .eval_wan_video import (
        DEFAULT_HAO_ROOT,
        DEFAULT_MODEL,
        DEFAULT_PROMPT,
        authenticate_policy_extension,
        latent_token_count,
        load_hao_bf16_attention,
        make_runner,
        restore_processors,
        run_provider,
    )
except ImportError:  # direct script execution
    from eval_regular_attention import (
        authenticate_asset_manifest,
        load_extension,
        parse_layer_indices,
        portable_asset_identity,
        portable_file_identity,
        tensor_metrics,
    )
    from eval_wan_video import (
        DEFAULT_HAO_ROOT,
        DEFAULT_MODEL,
        DEFAULT_PROMPT,
        authenticate_policy_extension,
        latent_token_count,
        load_hao_bf16_attention,
        make_runner,
        restore_processors,
        run_provider,
    )


def parse_route(specification: str) -> tuple[str, Path, str]:
    try:
        name, remainder = specification.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "route must be NAME=MANIFEST_JSON[:POLICY]"
        ) from error
    policy = "fast"
    path_text = remainder
    if ":" in remainder:
        path_text, policy = remainder.rsplit(":", 1)
    if not name or not policy:
        raise argparse.ArgumentTypeError("route name and policy cannot be empty")
    return name, Path(path_text).resolve(), policy


def parse_override(specification: str) -> tuple[str, str, Path, str]:
    try:
        route, remainder = specification.split("=", 1)
        layers, path_text, module = remainder.split(",", 2)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "override must be ROUTE=LAYERS,EXTENSION_PATH,MODULE"
        ) from error
    parse_layer_indices(layers)
    if not route or not module:
        raise argparse.ArgumentTypeError("override route and module cannot be empty")
    return route, layers, Path(path_text).resolve(), module


def parse_active(specification: str) -> tuple[str, set[int]]:
    try:
        route, layers = specification.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "active layer selection must be ROUTE=LAYERS"
        ) from error
    if not route:
        raise argparse.ArgumentTypeError("active route cannot be empty")
    selected = parse_layer_indices(layers)
    if not selected:
        raise argparse.ArgumentTypeError("active layer selection cannot be empty")
    return route, selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Authenticated local snapshot used for model loading.",
    )
    parser.add_argument("--asset-manifest", type=Path)
    parser.add_argument("--model-asset")
    parser.add_argument("--route", action="append", required=True, type=parse_route)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        type=parse_override,
        metavar="ROUTE=LAYERS,EXTENSION_PATH,MODULE",
        help="Add a static layer extension to one route.",
    )
    parser.add_argument(
        "--active",
        action="append",
        default=[],
        type=parse_active,
        metavar="ROUTE=LAYERS",
        help="Run FP4 only in these layers; use BF16 attention elsewhere.",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num-frames", type=int, default=17)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--output-type", choices=("latent", "np"), default="latent")
    parser.add_argument("--key-centering", choices=("none", "sequence-mean"), default="none")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--hao-root",
        type=Path,
        default=DEFAULT_HAO_ROOT,
        help="HAO flash-attention-fp4 checkout used by the BF16 reference.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def resolve_path(manifest_path: Path, path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def load_route(
    name: str,
    manifest_path: Path,
    policy: str,
    model: str,
    key_centering: str,
    extension_cache: dict[tuple[str, str], Any],
    overrides: list[tuple[str, Path, str]],
) -> tuple[Any, dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text())
    authenticated_bundle = manifest.get("schema") == "tk_wan_nv_mx_policy_bundle_v2"
    if manifest.get("model") != model:
        raise ValueError(
            f"route {name} model {manifest.get('model')} does not match {model}"
        )
    try:
        selected = manifest["policies"][policy]
        base = selected["base"]
    except KeyError as error:
        raise ValueError(
            f"route {name} has no policy {policy!r}"
        ) from error

    def cached_extension(path: Path, module: str) -> Any:
        key = (str(path), module)
        if key not in extension_cache:
            extension_cache[key] = load_extension(path, module)
        return extension_cache[key]

    if authenticated_bundle:
        base_path, base_identity = authenticate_policy_extension(
            manifest_path,
            base,
            f"route {name} base",
        )
        base_identity["verification"] = "matched_manifest"
    else:
        base_path = resolve_path(manifest_path, base["path"])
        base_identity = {
            **portable_file_identity(base_path),
            "module": base["module"],
            "verification": "observed_at_load",
        }
    extension = cached_extension(base_path, base["module"])
    layer_extensions = {}
    layer_sources = {}
    extension_entries = list(selected.get("layer_extensions", []))
    extension_entries.extend(
        {
            "layers": layers,
            "path": str(path),
            "module": module,
            "purpose": "command-line affine route override",
        }
        for layers, path, module in overrides
    )
    for entry in extension_entries:
        if {"bytes", "sha256"} <= set(entry):
            path, identity = authenticate_policy_extension(
                manifest_path,
                entry,
                f"route {name} layer",
            )
            identity["verification"] = "matched_manifest"
        else:
            path = resolve_path(manifest_path, entry["path"])
            identity = {
                **portable_file_identity(path),
                "module": entry["module"],
                "verification": "observed_at_load",
            }
        layer_extension = cached_extension(path, entry["module"])
        for layer in parse_layer_indices(entry["layers"]):
            if layer in layer_extensions:
                raise ValueError(
                    f"route {name} assigns layer {layer} more than once"
                )
            layer_extensions[layer] = layer_extension
            layer_sources[layer] = {
                **identity,
                "purpose": entry.get("purpose"),
            }
    runner = make_runner(
        extension,
        "tk",
        key_centering,
        layer_extensions=layer_extensions,
    )
    metadata = {
        "manifest": portable_file_identity(manifest_path),
        "policy": policy,
        "base": base_identity,
        "layer_sources": {str(layer): source for layer, source in layer_sources.items()},
        "affine_calibration": manifest.get("affine_calibration"),
    }
    return runner, metadata


def main() -> None:
    args = parse_args()
    names = [name for name, _, _ in args.route]
    if len(set(names)) != len(names):
        raise ValueError("route names must be distinct")
    overrides_by_route: dict[str, list[tuple[str, Path, str]]] = {
        name: [] for name in names
    }
    for route, layers, path, module in args.override:
        if route not in overrides_by_route:
            raise ValueError(f"override references unknown route: {route}")
        overrides_by_route[route].append((layers, path, module))
    active_by_route: dict[str, set[int] | None] = {name: None for name in names}
    for route, layers in args.active:
        if route not in active_by_route:
            raise ValueError(f"active selection references unknown route: {route}")
        if active_by_route[route] is not None:
            raise ValueError(f"route {route} has more than one active selection")
        active_by_route[route] = layers

    from diffusers import WanPipeline

    asset_manifest_identity = None
    if args.model_path is not None:
        if args.asset_manifest is None or args.model_asset is None:
            raise ValueError(
                "--model-path requires --asset-manifest and --model-asset"
            )
        model_root = args.model_path.resolve()
        model_identity, asset_manifest_identity = authenticate_asset_manifest(
            args.asset_manifest,
            args.model_asset,
            "model",
            model_root,
        )
        if model_identity["identifier"] != args.model:
            raise ValueError(
                f"authenticated model identifier {model_identity['identifier']!r} "
                f"does not match {args.model!r}"
            )
        model_source = str(model_root)
    else:
        model_identity = portable_asset_identity(
            args.model,
            identifier=None,
            revision=None,
            tree_sha256=None,
        )
        model_source = args.model

    hao_bf16_attention = load_hao_bf16_attention(args.hao_root)

    routes = {}
    route_metadata = {}
    extension_cache: dict[tuple[str, str], Any] = {}
    for name, manifest_path, policy in args.route:
        runner, metadata = load_route(
            name,
            manifest_path,
            policy,
            args.model,
            args.key_centering,
            extension_cache,
            overrides_by_route[name],
        )
        routes[name] = runner
        route_metadata[name] = metadata

    pipeline = WanPipeline.from_pretrained(
        model_source,
        torch_dtype=torch.bfloat16,
        local_files_only=args.local_files_only,
    )
    pipeline.to(args.device)
    pipeline.set_progress_bar_config(disable=False)
    original_processors = [block.attn1.processor for block in pipeline.transformer.blocks]

    sequence_length = latent_token_count(
        pipeline, args.height, args.width, args.num_frames
    )
    model_shape = {
        "layers": len(pipeline.transformer.blocks),
        "heads": int(pipeline.transformer.config.num_attention_heads),
        "head_dim": int(pipeline.transformer.config.attention_head_dim),
        "sequence_length": sequence_length,
    }
    for name, active_layers in active_by_route.items():
        if active_layers is not None:
            invalid = sorted(
                layer
                for layer in active_layers
                if layer < 0 or layer >= model_shape["layers"]
            )
            if invalid:
                raise ValueError(f"route {name} has invalid active layers: {invalid}")
    for name, runner in routes.items():
        expected = (runner.target_seqlen, runner.target_heads, runner.target_dim)
        actual = (
            model_shape["sequence_length"],
            model_shape["heads"],
            model_shape["head_dim"],
        )
        if expected != actual:
            raise ValueError(f"route {name} shape {expected} does not match {actual}")

    outputs = {}
    records = {}
    try:
        output, record = run_provider(
            pipeline,
            "hao-bf16",
            None,
            original_processors,
            args,
            hao_bf16_attention=hao_bf16_attention,
        )
        outputs["hao-bf16"] = output
        record["status"] = "complete"
        records["hao-bf16"] = record
        for name, runner in routes.items():
            try:
                output, record = run_provider(
                    pipeline,
                    name,
                    runner,
                    original_processors,
                    args,
                    lowp_layers=active_by_route[name],
                )
            except Exception as error:
                records[name] = {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "route": route_metadata[name],
                }
                continue
            outputs[name] = output
            record["status"] = "complete"
            record["route"] = route_metadata[name]
            record["route"]["active_lowp_layers"] = (
                "all"
                if active_by_route[name] is None
                else sorted(active_by_route[name])
            )
            records[name] = record
    finally:
        restore_processors(pipeline.transformer, original_processors)

    comparisons = {
        f"{name}_vs_hao-bf16": tensor_metrics(output, outputs["hao-bf16"])
        for name, output in outputs.items()
        if name != "hao-bf16"
    }
    payload = {
        "schema": "tk_fp4_fa4_wan_affine_routes_v2",
        "model": model_identity,
        "asset_manifest": asset_manifest_identity,
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
            "key_centering": args.key_centering,
        },
        "model_attention_shape": model_shape,
        "reference_provider": "hao-bf16",
        "providers": records,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
