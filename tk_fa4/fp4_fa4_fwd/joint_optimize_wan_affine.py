#!/usr/bin/env python3
"""Jointly optimize Wan affine-kernel assignments with persistent GPU workers.

The search uses exact compiled kernels. Each worker loads one Wan pipeline and
the complete affine candidate bank once, caches the BF16 reference for every
prompt/step pair, and then evaluates complete low-precision layer routes.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import queue
import random
import statistics
from types import SimpleNamespace
from typing import Any


def parse_candidate(specification: str) -> tuple[str, str, str]:
    try:
        label, extension_text = specification.split("=", 1)
        path, module = extension_text.rsplit(":", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "candidate must use LABEL=PATH:MODULE"
        ) from error
    if not label or not path or not module:
        raise argparse.ArgumentTypeError("candidate fields cannot be empty")
    return label, str(Path(path).resolve()), module


def parse_guard(specification: str) -> tuple[str, str, str]:
    try:
        layers, extension_text = specification.split("=", 1)
        path, module = extension_text.rsplit(":", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "guard must use LAYERS=PATH:MODULE"
        ) from error
    return layers, str(Path(path).resolve()), module


def parse_assignment(specification: str) -> tuple[str, str]:
    try:
        label, layers = specification.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "initial assignment must use LABEL=LAYERS"
        ) from error
    return label, layers


def parse_prompt(specification: str) -> dict[str, Any]:
    try:
        name, remainder = specification.split("=", 1)
        seed_text, prompt = remainder.split(":", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "prompt must use NAME=SEED:TEXT"
        ) from error
    if not name or not prompt:
        raise argparse.ArgumentTypeError("prompt name and text cannot be empty")
    return {"name": name, "seed": int(seed_text), "prompt": prompt}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--candidate", action="append", required=True, type=parse_candidate
    )
    parser.add_argument("--base", required=True)
    parser.add_argument("--guard", required=True, type=parse_guard)
    parser.add_argument(
        "--initial", action="append", default=[], type=parse_assignment
    )
    parser.add_argument(
        "--prompt", action="append", required=True, type=parse_prompt
    )
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--search-steps", type=int, default=1)
    parser.add_argument("--validation-steps", type=int, default=4)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--validation-top-k", type=int, default=8)
    parser.add_argument("--elite-fraction", type=float, default=0.25)
    parser.add_argument("--update-rate", type=float, default=0.65)
    parser.add_argument("--probability-floor", type=float, default=0.03)
    parser.add_argument("--initial-confidence", type=float, default=0.70)
    parser.add_argument(
        "--regularization-grid",
        default="1,2,4,8,12,16,24,32",
        help="Comma-separated counts of favorable single-layer changes to combine.",
    )
    parser.add_argument(
        "--single-flip-temperature",
        type=float,
        default=0.0,
        help="CEM initialization temperature; zero selects it from observed deltas.",
    )
    parser.add_argument("--random-seed", type=int, default=20260806)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num-frames", type=int, default=17)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def compress_layers(layers: list[int]) -> str:
    if not layers:
        return ""
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


def route_groups(
    route: tuple[str, ...], searchable_layers: list[int]
) -> dict[str, str]:
    groups: dict[str, list[int]] = defaultdict(list)
    for layer, label in zip(searchable_layers, route, strict=True):
        groups[label].append(layer)
    return {
        label: compress_layers(layers)
        for label, layers in sorted(groups.items())
    }


def route_key(route: tuple[str, ...]) -> str:
    digest = hashlib.sha1("\0".join(route).encode()).hexdigest()[:12]
    return f"route-{digest}"


def reset_runner(runner: Any) -> None:
    runner.nonfinite_output_count = 0
    runner.layer_metrics.clear()
    runner.layer_finite_stats.clear()
    runner.scale_sweep.clear()
    runner.scale_stats.clear()
    runner.p_replay_ranges.clear()
    runner.p_replay_metrics.clear()


def aggregate_objective(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records or any(not record["finite"] for record in records):
        return {
            "score": 1e6,
            "mean_relative_l2": math.inf,
            "worst_relative_l2": math.inf,
            "mean_cosine": -1.0,
        }
    relative_l2 = [record["metrics"]["relative_l2"] for record in records]
    cosine = [record["metrics"]["cosine"] for record in records]
    mean_relative_l2 = sum(relative_l2) / len(relative_l2)
    worst_relative_l2 = max(relative_l2)
    mean_cosine = sum(cosine) / len(cosine)
    score = (
        mean_relative_l2
        + 0.25 * (worst_relative_l2 - mean_relative_l2)
        + 0.10 * (1.0 - mean_cosine)
    )
    return {
        "score": score,
        "mean_relative_l2": mean_relative_l2,
        "worst_relative_l2": worst_relative_l2,
        "mean_cosine": mean_cosine,
    }


def worker_main(
    worker_id: int,
    gpu: int,
    config: dict[str, Any],
    task_queue: Any,
    result_queue: Any,
) -> None:
    try:
        import torch

        torch.cuda.set_device(gpu)
        device = f"cuda:{gpu}"

        from diffusers import WanPipeline
        from eval_regular_attention import (
            load_extension,
            parse_layer_indices,
            tensor_metrics,
        )
        from eval_wan_video import (
            latent_token_count,
            make_runner,
            restore_processors,
            run_provider,
        )

        extensions = {
            label: load_extension(Path(path), module)
            for label, path, module in config["candidates"]
        }
        guard_layers_text, guard_path, guard_module = config["guard"]
        guard_layers = parse_layer_indices(guard_layers_text)
        guard_extension = load_extension(Path(guard_path), guard_module)
        topologies = {
            label: dict(extension.read_hao_direct_topology())
            for label, extension in extensions.items()
        }
        guard_topology = dict(guard_extension.read_hao_direct_topology())

        pipeline = WanPipeline.from_pretrained(
            config["model"],
            torch_dtype=torch.bfloat16,
            local_files_only=config["local_files_only"],
        )
        pipeline.to(device)
        pipeline.set_progress_bar_config(disable=True)
        original_processors = [
            block.attn1.processor for block in pipeline.transformer.blocks
        ]
        model_shape = {
            "layers": len(pipeline.transformer.blocks),
            "heads": int(pipeline.transformer.config.num_attention_heads),
            "head_dim": int(pipeline.transformer.config.attention_head_dim),
            "sequence_length": latent_token_count(
                pipeline,
                config["height"],
                config["width"],
                config["num_frames"],
            ),
        }
        runner = make_runner(
            extensions[config["base"]], "tk", "none"
        )
        expected = (
            runner.target_seqlen,
            runner.target_heads,
            runner.target_dim,
        )
        actual = (
            model_shape["sequence_length"],
            model_shape["heads"],
            model_shape["head_dim"],
        )
        if expected != actual:
            raise ValueError(f"extension shape {expected} != model shape {actual}")

        compatible_keys = (
            "batch", "seqlen", "heads", "dqk", "dvo", "route",
            "qk_format", "pv_format",
        )
        for label, topology in [*topologies.items(), ("guard", guard_topology)]:
            for key in compatible_keys:
                if topology.get(key) != runner.topology.get(key):
                    raise ValueError(
                        f"{label} topology {key}={topology.get(key)}; "
                        f"expected {runner.topology.get(key)}"
                    )

        references: dict[tuple[str, int], Any] = {}
        reference_records: dict[tuple[str, int], dict[str, Any]] = {}

        def generation_args(prompt: dict[str, Any], steps: int) -> Any:
            return SimpleNamespace(
                prompt=prompt["prompt"],
                negative_prompt=None,
                height=config["height"],
                width=config["width"],
                num_frames=config["num_frames"],
                steps=steps,
                guidance_scale=config["guidance_scale"],
                seed=prompt["seed"],
                output_type="latent",
                key_centering="none",
                device=device,
            )

        result_queue.put(
            {
                "type": "ready",
                "worker": worker_id,
                "gpu": gpu,
                "model_shape": model_shape,
            }
        )
        while True:
            task = task_queue.get()
            if task is None:
                break
            task_id = task["task_id"]
            route = tuple(task["route"])
            steps = int(task["steps"])
            try:
                layer_extensions = {
                    layer: guard_extension for layer in guard_layers
                }
                layer_topologies = {
                    layer: guard_topology for layer in guard_layers
                }
                for layer, label in zip(
                    task["searchable_layers"], route, strict=True
                ):
                    if label == config["base"]:
                        continue
                    layer_extensions[layer] = extensions[label]
                    layer_topologies[layer] = topologies[label]
                runner.layer_extensions = layer_extensions
                runner.layer_topologies = layer_topologies
                reset_runner(runner)

                prompt_records = []
                for prompt in config["prompts"]:
                    cache_key = (prompt["name"], steps)
                    args = generation_args(prompt, steps)
                    if cache_key not in references:
                        reference, reference_record = run_provider(
                            pipeline,
                            "bf16",
                            None,
                            original_processors,
                            args,
                        )
                        references[cache_key] = reference
                        reference_records[cache_key] = {
                            "elapsed_seconds": reference_record["elapsed_seconds"],
                            "finite": bool(reference.isfinite().all().item()),
                        }
                    output, record = run_provider(
                        pipeline,
                        "joint-route",
                        runner,
                        original_processors,
                        args,
                    )
                    finite = bool(output.isfinite().all().item())
                    prompt_record: dict[str, Any] = {
                        "name": prompt["name"],
                        "seed": prompt["seed"],
                        "finite": finite,
                        "elapsed_seconds": record["elapsed_seconds"],
                    }
                    if finite:
                        prompt_record["metrics"] = tensor_metrics(
                            output, references[cache_key]
                        )
                    prompt_records.append(prompt_record)
                result_queue.put(
                    {
                        "type": "result",
                        "worker": worker_id,
                        "gpu": gpu,
                        "task_id": task_id,
                        "route": route,
                        "steps": steps,
                        "prompts": prompt_records,
                        "objective": aggregate_objective(prompt_records),
                    }
                )
            except Exception as error:
                restore_processors(pipeline.transformer, original_processors)
                result_queue.put(
                    {
                        "type": "result",
                        "worker": worker_id,
                        "gpu": gpu,
                        "task_id": task_id,
                        "route": route,
                        "steps": steps,
                        "prompts": [],
                        "objective": aggregate_objective([]),
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
        restore_processors(pipeline.transformer, original_processors)
    except Exception as error:
        result_queue.put(
            {
                "type": "fatal",
                "worker": worker_id,
                "gpu": gpu,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )


def normalize_probabilities(
    probabilities: list[float], floor: float
) -> list[float]:
    clipped = [max(floor, value) for value in probabilities]
    total = sum(clipped)
    return [value / total for value in clipped]


def main() -> None:
    args = parse_args()
    candidates = {label: (path, module) for label, path, module in args.candidate}
    labels = list(candidates)
    if len(candidates) != len(args.candidate):
        raise ValueError("candidate labels must be distinct")
    if args.base not in candidates:
        raise ValueError(f"base candidate {args.base!r} is not defined")
    if len(labels) < 2:
        raise ValueError("joint search requires at least two candidates")
    if not 0.0 < args.elite_fraction <= 1.0:
        raise ValueError("--elite-fraction must be in (0, 1]")
    if not 0.0 < args.update_rate <= 1.0:
        raise ValueError("--update-rate must be in (0, 1]")
    if args.probability_floor * len(labels) >= 1.0:
        raise ValueError("--probability-floor is too large for the codebook")
    if not 0.0 < args.initial_confidence < 1.0:
        raise ValueError("--initial-confidence must be in (0, 1)")
    regularization_counts = sorted(
        {
            int(item)
            for item in args.regularization_grid.split(",")
            if item.strip()
        }
    )
    if any(count <= 0 for count in regularization_counts):
        raise ValueError("--regularization-grid counts must be positive")
    if args.single_flip_temperature < 0.0:
        raise ValueError("--single-flip-temperature cannot be negative")

    from eval_regular_attention import parse_layer_indices

    guard_layers = parse_layer_indices(args.guard[0])
    initial_specs = []
    for label, layers_text in args.initial:
        if label not in candidates:
            raise ValueError(f"initial assignment uses unknown label {label}")
        initial_specs.append((label, parse_layer_indices(layers_text)))
    gpus = [int(item) for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise ValueError("--gpus must select at least one GPU")

    config = {
        "model": args.model,
        "candidates": [
            (label, path, module)
            for label, (path, module) in candidates.items()
        ],
        "base": args.base,
        "guard": args.guard,
        "prompts": args.prompt,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "guidance_scale": args.guidance_scale,
        "local_files_only": args.local_files_only,
    }

    context = mp.get_context("spawn")
    task_queue = context.Queue(maxsize=max(8, len(gpus) * 4))
    result_queue = context.Queue()
    workers = [
        context.Process(
            target=worker_main,
            args=(index, gpu, config, task_queue, result_queue),
        )
        for index, gpu in enumerate(gpus)
    ]
    for worker in workers:
        worker.start()

    def get_result() -> dict[str, Any]:
        while True:
            try:
                message = result_queue.get(timeout=30.0)
            except queue.Empty:
                failed = [
                    (worker.pid, worker.exitcode)
                    for worker in workers
                    if not worker.is_alive()
                ]
                if failed:
                    raise RuntimeError(f"joint-search workers exited: {failed}")
                print("waiting for persistent workers...", flush=True)
                continue
            if message["type"] == "fatal":
                raise RuntimeError(
                    f"worker {message['worker']} on GPU {message['gpu']} "
                    f"failed: {message['error_type']}: {message['error']}"
                )
            return message

    try:
        ready = []
        while len(ready) < len(workers):
            message = get_result()
            if message["type"] != "ready":
                raise RuntimeError(f"unexpected startup message: {message}")
            ready.append(message)
            print(
                f"worker {message['worker']} ready on GPU {message['gpu']}",
                flush=True,
            )
        model_shape = ready[0]["model_shape"]
        if any(message["model_shape"] != model_shape for message in ready[1:]):
            raise ValueError("workers reported different model shapes")
        searchable_layers = sorted(
            set(range(model_shape["layers"])) - guard_layers
        )
        initial_by_layer = {layer: args.base for layer in searchable_layers}
        for label, layers in initial_specs:
            overlap = layers & guard_layers
            if overlap:
                raise ValueError(
                    f"initial assignment overlaps guard layers: {sorted(overlap)}"
                )
            invalid = layers - set(searchable_layers)
            if invalid:
                raise ValueError(f"invalid initial layers: {sorted(invalid)}")
            for layer in layers:
                if initial_by_layer[layer] != args.base:
                    raise ValueError(f"layer {layer} is assigned more than once")
                initial_by_layer[layer] = label
        initial_route = tuple(initial_by_layer[layer] for layer in searchable_layers)
        base_route = tuple(args.base for _ in searchable_layers)

        evaluations: dict[int, dict[tuple[str, ...], dict[str, Any]]] = defaultdict(dict)
        task_counter = 0

        def write_checkpoint(stage: str, history: list[dict[str, Any]]) -> None:
            payload = {
                "schema": "tk_wan_joint_affine_search_v1",
                "stage": stage,
                "model": args.model,
                "base": args.base,
                "model_shape": model_shape,
                "searchable_layers": searchable_layers,
                "guard": {
                    "layers": sorted(guard_layers),
                    "path": args.guard[1],
                    "module": args.guard[2],
                },
                "candidates": {
                    label: {"path": path, "module": module}
                    for label, (path, module) in candidates.items()
                },
                "prompts": args.prompt,
                "search_steps": args.search_steps,
                "validation_steps": args.validation_steps,
                "initial_route": route_groups(initial_route, searchable_layers),
                "history": history,
                "evaluations": {
                    str(steps): [
                        {
                            "id": route_key(route),
                            "groups": route_groups(route, searchable_layers),
                            **record,
                        }
                        for route, record in sorted(
                            step_records.items(),
                            key=lambda item: item[1]["objective"]["score"],
                        )
                    ]
                    for steps, step_records in sorted(evaluations.items())
                },
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2) + "\n")

        def evaluate(
            routes: list[tuple[str, ...]], steps: int
        ) -> list[dict[str, Any]]:
            nonlocal task_counter
            unique = []
            seen = set()
            for route in routes:
                if route in seen or route in evaluations[steps]:
                    continue
                seen.add(route)
                unique.append(route)
            next_route = 0
            max_inflight = max(8, len(workers) * 4)

            def submit(route: tuple[str, ...]) -> None:
                nonlocal task_counter
                task_counter += 1
                task_queue.put(
                    {
                        "task_id": task_counter,
                        "route": route,
                        "searchable_layers": searchable_layers,
                        "steps": steps,
                    }
                )

            while next_route < min(len(unique), max_inflight):
                submit(unique[next_route])
                next_route += 1
            for completed in range(len(unique)):
                message = get_result()
                if message["type"] != "result":
                    raise RuntimeError(f"unexpected worker message: {message}")
                route = tuple(message.pop("route"))
                message.pop("type", None)
                evaluations[steps][route] = message
                objective = message["objective"]
                print(
                    f"[{completed + 1}/{len(unique)} steps={steps}] "
                    f"{route_key(route)} score={objective['score']:.8f} "
                    f"relL2={objective['mean_relative_l2']:.8f} "
                    f"cos={objective['mean_cosine']:.8f}",
                    flush=True,
                )
                if next_route < len(unique):
                    submit(unique[next_route])
                    next_route += 1
            return [evaluations[steps][route] for route in routes]

        rng = random.Random(args.random_seed)
        history: list[dict[str, Any]] = []

        single_flips = [initial_route, base_route]
        for index in range(len(searchable_layers)):
            for label in labels:
                if label == initial_route[index]:
                    continue
                route = list(initial_route)
                route[index] = label
                single_flips.append(tuple(route))
        print(f"evaluating {len(set(single_flips))} initial/single-flip routes", flush=True)
        evaluate(single_flips, args.search_steps)
        ranked = sorted(
            evaluations[args.search_steps],
            key=lambda route: evaluations[args.search_steps][route]["objective"]["score"],
        )

        initial_score = evaluations[args.search_steps][initial_route]["objective"][
            "score"
        ]
        per_layer_scores: list[dict[str, float]] = []
        observed_deltas = []
        favorable_flips = []
        for index, layer in enumerate(searchable_layers):
            scores = {}
            for label in labels:
                route = list(initial_route)
                route[index] = label
                score = evaluations[args.search_steps][tuple(route)]["objective"][
                    "score"
                ]
                scores[label] = score
                if math.isfinite(score) and math.isfinite(initial_score):
                    delta = abs(score - initial_score)
                    if delta > 0.0:
                        observed_deltas.append(delta)
            per_layer_scores.append(scores)
            best_label = min(scores, key=scores.get)
            improvement = initial_score - scores[best_label]
            if best_label != initial_route[index] and improvement > 0.0:
                favorable_flips.append((improvement, index, layer, best_label))

        favorable_flips.sort(reverse=True)
        combined_routes = []
        for count in regularization_counts:
            if count > len(favorable_flips):
                continue
            route = list(initial_route)
            for _, index, _, label in favorable_flips[:count]:
                route[index] = label
            combined_routes.append(tuple(route))
        if favorable_flips:
            greedy_route = list(initial_route)
            for _, index, _, label in favorable_flips:
                greedy_route[index] = label
            combined_routes.append(tuple(greedy_route))
        combined_routes = list(dict.fromkeys(combined_routes))
        if combined_routes:
            print(
                f"evaluating {len(combined_routes)} regularized single-flip "
                "combinations",
                flush=True,
            )
            evaluate(combined_routes, args.search_steps)
            ranked = sorted(
                evaluations[args.search_steps],
                key=lambda route: evaluations[args.search_steps][route][
                    "objective"
                ]["score"],
            )

        temperature = args.single_flip_temperature
        if temperature == 0.0:
            temperature = (
                statistics.median(observed_deltas) if observed_deltas else 1e-4
            )
        temperature = max(temperature, 1e-6)
        probabilities = []
        for index, current in enumerate(initial_route):
            scores = per_layer_scores[index]
            best_score = min(scores.values())
            row = []
            for label in labels:
                score = scores[label]
                weight = (
                    math.exp(-(score - best_score) / temperature)
                    if math.isfinite(score)
                    else 0.0
                )
                if label == current:
                    weight *= 1.0 + args.initial_confidence
                row.append(weight)
            probabilities.append(normalize_probabilities(row, args.probability_floor))

        def update_probabilities(elites: list[tuple[str, ...]]) -> None:
            for index in range(len(searchable_layers)):
                empirical = [
                    sum(route[index] == label for route in elites) / len(elites)
                    for label in labels
                ]
                mixed = [
                    (1.0 - args.update_rate) * probabilities[index][label_index]
                    + args.update_rate * empirical[label_index]
                    for label_index in range(len(labels))
                ]
                probabilities[index] = normalize_probabilities(
                    mixed, args.probability_floor
                )

        history.append(
            {
                "stage": "single_flips",
                "routes": len(set(single_flips)),
                "best": route_key(ranked[0]),
                "best_score": evaluations[args.search_steps][ranked[0]]["objective"]["score"],
                "initial_score": initial_score,
                "single_flip_temperature": temperature,
                "favorable_flips": [
                    {
                        "layer": layer,
                        "label": label,
                        "improvement": improvement,
                    }
                    for improvement, _, layer, label in favorable_flips
                ],
                "combined_routes": [
                    {
                        "id": route_key(route),
                        "groups": route_groups(route, searchable_layers),
                        "score": evaluations[args.search_steps][route]["objective"][
                            "score"
                        ],
                    }
                    for route in combined_routes
                ],
            }
        )
        write_checkpoint("single_flips", history)

        for generation in range(args.generations):
            sampled = [ranked[0], initial_route, base_route]
            attempts = 0
            while len(set(sampled)) < args.population and attempts < args.population * 100:
                sampled.append(
                    tuple(
                        rng.choices(labels, weights=row, k=1)[0]
                        for row in probabilities
                    )
                )
                attempts += 1
            sampled = list(dict.fromkeys(sampled))
            print(
                f"generation {generation}: evaluating {len(sampled)} routes",
                flush=True,
            )
            evaluate(sampled, args.search_steps)
            generation_ranked = sorted(
                sampled,
                key=lambda route: evaluations[args.search_steps][route]["objective"]["score"],
            )
            elite_count = max(
                2, math.ceil(args.elite_fraction * len(generation_ranked))
            )
            update_probabilities(generation_ranked[:elite_count])
            ranked = sorted(
                evaluations[args.search_steps],
                key=lambda route: evaluations[args.search_steps][route]["objective"]["score"],
            )
            history.append(
                {
                    "stage": "generation",
                    "generation": generation,
                    "sampled": len(sampled),
                    "new_evaluations": len(evaluations[args.search_steps]),
                    "best": route_key(ranked[0]),
                    "best_score": evaluations[args.search_steps][ranked[0]]["objective"]["score"],
                    "mean_entropy": sum(
                        -sum(value * math.log(max(value, 1e-30)) for value in row)
                        for row in probabilities
                    ) / len(probabilities),
                }
            )
            write_checkpoint(f"generation-{generation}", history)

        validation_routes = ranked[: args.validation_top_k]
        for required in (initial_route, base_route):
            if required not in validation_routes:
                validation_routes.append(required)
        print(
            f"validating {len(validation_routes)} routes at "
            f"{args.validation_steps} steps",
            flush=True,
        )
        evaluate(validation_routes, args.validation_steps)
        validation_ranked = sorted(
            validation_routes,
            key=lambda route: evaluations[args.validation_steps][route]["objective"]["score"],
        )
        winner = validation_ranked[0]
        initial_score = evaluations[args.validation_steps][initial_route]["objective"]["score"]
        winner_score = evaluations[args.validation_steps][winner]["objective"]["score"]
        history.append(
            {
                "stage": "validation",
                "routes": len(validation_routes),
                "winner": route_key(winner),
                "winner_groups": route_groups(winner, searchable_layers),
                "winner_score": winner_score,
                "initial_score": initial_score,
                "improvement": initial_score - winner_score,
            }
        )
        write_checkpoint("complete", history)
        print(
            json.dumps(
                {
                    "winner": route_key(winner),
                    "groups": route_groups(winner, searchable_layers),
                    "score": winner_score,
                    "initial_score": initial_score,
                    "improvement": initial_score - winner_score,
                },
                indent=2,
            ),
            flush=True,
        )
    finally:
        for _ in workers:
            task_queue.put(None)
        for worker in workers:
            worker.join(timeout=30.0)
            if worker.is_alive():
                worker.terminate()
                worker.join()


if __name__ == "__main__":
    main()
