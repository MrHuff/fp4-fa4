#!/usr/bin/env python3
"""Materialize a joint affine-search route as an evaluator manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument(
        "--route-id",
        help="Validation route ID; defaults to the recorded winner.",
    )
    return parser.parse_args()


def selected_route(search: dict[str, Any], route_id: str) -> dict[str, Any]:
    matches = [
        record
        for records in search["evaluations"].values()
        for record in records
        if record["id"] == route_id
    ]
    if not matches:
        raise ValueError(f"route {route_id!r} is not present")
    return max(matches, key=lambda record: record["steps"])


def main() -> None:
    args = parse_args()
    search = json.loads(args.search.read_text())
    if search.get("stage") != "complete":
        raise ValueError("joint search is not complete")
    if args.base not in search["candidates"]:
        raise ValueError(f"base candidate {args.base!r} is not present")

    winner_id = args.route_id or search["history"][-1]["winner"]
    route = selected_route(search, winner_id)
    candidates = search["candidates"]
    layer_extensions = []
    for label, layers in route["groups"].items():
        if label == args.base:
            continue
        candidate = candidates[label]
        layer_extensions.append(
            {
                "layers": layers,
                "path": candidate["path"],
                "module": candidate["module"],
                "purpose": f"joint affine search candidate {label}",
            }
        )
    layer_extensions.append(
        {
            "layers": ",".join(str(layer) for layer in search["guard"]["layers"]),
            "path": search["guard"]["path"],
            "module": search["guard"]["module"],
            "purpose": "wide global QK probe for extreme-logit layers",
        }
    )

    shape = search["model_shape"]
    manifest = {
        "schema": "tk_wan_joint_affine_candidate_v1",
        "model": search["model"],
        "shape": {
            "batch": 1,
            "seqlen": shape["sequence_length"],
            "heads": shape["heads"],
            "dim": shape["head_dim"],
        },
        "formats": {"qk": "nvfp4", "pv": "mxfp4"},
        "softmax": "shiftless/sampled; no stable-softmax fallback",
        "joint_search": {
            "source": str(args.search.resolve()),
            "route_id": winner_id,
            "groups": route["groups"],
            "objective": route["objective"],
        },
        "policies": {
            "fast": {
                "base": candidates[args.base],
                "layer_extensions": layer_extensions,
            }
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
