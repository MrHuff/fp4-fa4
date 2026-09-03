#!/usr/bin/env python3
"""Validate saved joint affine routes on a wider prompt set."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
import queue
from typing import Any

from joint_optimize_wan_affine import parse_prompt, route_groups, worker_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--route-id", action="append", default=[])
    parser.add_argument(
        "--include-regularized",
        action="store_true",
        help="Include every regularized single-flip combination.",
    )
    parser.add_argument("--prompt", action="append", required=True, type=parse_prompt)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--gpus", default="0,1,2,3")
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


def expand_groups(
    groups: dict[str, str], searchable_layers: list[int]
) -> tuple[str, ...]:
    from eval_regular_attention import parse_layer_indices

    labels_by_layer = {}
    for label, layers_text in groups.items():
        for layer in parse_layer_indices(layers_text):
            if layer in labels_by_layer:
                raise ValueError(f"layer {layer} occurs in more than one group")
            labels_by_layer[layer] = label
    missing = set(searchable_layers) - set(labels_by_layer)
    extra = set(labels_by_layer) - set(searchable_layers)
    if missing or extra:
        raise ValueError(f"route groups missing={sorted(missing)} extra={sorted(extra)}")
    return tuple(labels_by_layer[layer] for layer in searchable_layers)


def main() -> None:
    args = parse_args()
    search = json.loads(args.search.read_text())
    if search.get("stage") != "complete":
        raise ValueError("joint search is not complete")
    if args.base not in search["candidates"]:
        raise ValueError(f"base candidate {args.base!r} is not present")

    records_by_id = {
        record["id"]: record
        for records in search["evaluations"].values()
        for record in records
    }
    route_ids = list(args.route_id)
    if args.include_regularized:
        single_flip_history = next(
            record
            for record in search["history"]
            if record["stage"] == "single_flips"
        )
        route_ids.extend(
            record["id"] for record in single_flip_history["combined_routes"]
        )
    route_ids = list(dict.fromkeys(route_ids))
    if not route_ids:
        raise ValueError("select at least one route")
    unknown = set(route_ids) - set(records_by_id)
    if unknown:
        raise ValueError(f"unknown route IDs: {sorted(unknown)}")

    searchable_layers = search["searchable_layers"]
    routes = {
        route_id: expand_groups(records_by_id[route_id]["groups"], searchable_layers)
        for route_id in route_ids
    }
    gpus = [int(item) for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise ValueError("--gpus must select at least one GPU")
    candidates = search["candidates"]
    config = {
        "model": search["model"],
        "candidates": [
            (label, candidate["path"], candidate["module"])
            for label, candidate in candidates.items()
        ],
        "base": args.base,
        "guard": (
            ",".join(str(layer) for layer in search["guard"]["layers"]),
            search["guard"]["path"],
            search["guard"]["module"],
        ),
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
            args=(worker_id, gpu, config, task_queue, result_queue),
        )
        for worker_id, gpu in enumerate(gpus)
    ]
    for worker in workers:
        worker.start()

    def get_message() -> dict[str, Any]:
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
                    raise RuntimeError(f"validation workers exited: {failed}")
                print("waiting for persistent workers...", flush=True)
                continue
            if message["type"] == "fatal":
                raise RuntimeError(
                    f"worker {message['worker']} on GPU {message['gpu']} failed: "
                    f"{message['error_type']}: {message['error']}"
                )
            return message

    try:
        ready = 0
        while ready < len(workers):
            message = get_message()
            if message["type"] != "ready":
                raise RuntimeError(f"unexpected startup message: {message}")
            ready += 1
            print(
                f"worker {message['worker']} ready on GPU {message['gpu']}",
                flush=True,
            )
        for task_id, (route_id, route) in enumerate(routes.items(), start=1):
            task_queue.put(
                {
                    "task_id": task_id,
                    "route": route,
                    "searchable_layers": searchable_layers,
                    "steps": args.steps,
                }
            )

        results = {}
        route_id_by_task = {
            task_id: route_id
            for task_id, route_id in enumerate(routes, start=1)
        }
        for completed in range(len(routes)):
            message = get_message()
            if message["type"] != "result":
                raise RuntimeError(f"unexpected worker message: {message}")
            route_id = route_id_by_task[message["task_id"]]
            message.pop("type", None)
            message.pop("route", None)
            results[route_id] = message
            objective = message["objective"]
            print(
                f"[{completed + 1}/{len(routes)}] {route_id} "
                f"score={objective['score']:.8f} "
                f"relL2={objective['mean_relative_l2']:.8f} "
                f"cos={objective['mean_cosine']:.8f}",
                flush=True,
            )

        ranked = sorted(
            results,
            key=lambda route_id: results[route_id]["objective"]["score"],
        )
        payload = {
            "schema": "tk_wan_joint_route_validation_v1",
            "source": str(args.search.resolve()),
            "model": search["model"],
            "steps": args.steps,
            "prompts": args.prompt,
            "routes": {
                route_id: {
                    "groups": route_groups(routes[route_id], searchable_layers),
                    **results[route_id],
                }
                for route_id in ranked
            },
            "ranking": ranked,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps({"ranking": ranked}, indent=2), flush=True)
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
