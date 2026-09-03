#!/usr/bin/env python3
"""Measure NVFP4 versus MXFP4 representability for softmax probabilities."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch


LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
THRESHOLDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seqlen", type=int, action="append")
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--query-rows", type=int, default=64)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--gpu", default="0")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "p_format_range.json",
    )
    return parser.parse_args()


def e2m1_reconstruct(values: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    levels = torch.tensor(LEVELS, device=values.device, dtype=torch.float32)
    thresholds = torch.tensor(
        THRESHOLDS, device=values.device, dtype=torch.float32
    )
    normalized = torch.where(
        scale > 0.0,
        values / scale,
        torch.zeros_like(values),
    )
    codes = (normalized.unsqueeze(-1) > thresholds).sum(dim=-1)
    return levels[codes] * scale


def normalize_rows(values: torch.Tensor) -> torch.Tensor:
    denominator = values.sum(dim=-1, keepdim=True)
    return torch.where(denominator > 0.0, values / denominator, values)


def nv_reconstruct(
    probabilities: torch.Tensor,
    global_encode_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    blocks = probabilities.reshape(*probabilities.shape[:-1], -1, 32)
    ideal_scale = blocks.amax(dim=-1, keepdim=True) / 6.0
    encoded = (ideal_scale * global_encode_scale).to(torch.float8_e4m3fn)
    scale = encoded.float() / global_encode_scale
    reconstructed = e2m1_reconstruct(blocks, scale)
    return reconstructed.reshape_as(probabilities), scale.squeeze(-1)


def mx_reconstruct(
    probabilities: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    blocks = probabilities.reshape(*probabilities.shape[:-1], -1, 32)
    ideal_scale = blocks.amax(dim=-1, keepdim=True).clamp_min(
        torch.finfo(torch.float32).tiny
    ) / 6.0
    log2_scale = torch.log2(ideal_scale)
    floor_scale = torch.exp2(torch.floor(log2_scale))
    ceil_scale = torch.exp2(torch.ceil(log2_scale))
    floor_reconstruction = e2m1_reconstruct(blocks, floor_scale)
    ceil_reconstruction = e2m1_reconstruct(blocks, ceil_scale)
    floor_error = (floor_reconstruction - blocks).square().sum(dim=-1)
    ceil_error = (ceil_reconstruction - blocks).square().sum(dim=-1)
    choose_ceil = (ceil_error <= floor_error).unsqueeze(-1)
    scale = torch.where(choose_ceil, ceil_scale, floor_scale)
    reconstructed = torch.where(
        choose_ceil, ceil_reconstruction, floor_reconstruction
    )
    return reconstructed.reshape_as(probabilities), scale.squeeze(-1)


def error_metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    candidate = candidate.float().reshape(-1)
    reference = reference.float().reshape(-1)
    error = candidate - reference
    rmse = error.square().mean().sqrt()
    reference_rms = reference.square().mean().sqrt()
    cosine = torch.nn.functional.cosine_similarity(
        candidate, reference, dim=0, eps=1e-12
    )
    return {
        "cosine": float(cosine.item()),
        "relative_l2": float((rmse / reference_rms).item()),
        "rmse": float(rmse.item()),
    }


def evaluate_format(
    *,
    name: str,
    probabilities: torch.Tensor,
    reconstructed: torch.Tensor,
    scales: torch.Tensor,
    values: torch.Tensor,
) -> dict[str, Any]:
    normalized = normalize_rows(reconstructed)
    exact_output = torch.einsum("hqk,khd->qhd", probabilities, values)
    quantized_output = torch.einsum("hqk,khd->qhd", normalized, values)
    zero_mask = reconstructed == 0.0
    return {
        "format": name,
        "probability_error": error_metrics(normalized, probabilities),
        "output_error": error_metrics(quantized_output, exact_output),
        "zero_payload_fraction": float(zero_mask.float().mean().item()),
        "zeroed_probability_mass": float(
            probabilities[zero_mask].sum().div(probabilities.sum()).item()
        ),
        "zero_scale_block_fraction": float((scales == 0.0).float().mean().item()),
        "minimum_nonzero_scale": float(
            scales[scales > 0.0].amin().item()
            if bool((scales > 0.0).any())
            else 0.0
        ),
    }


def run_shape(
    *,
    seqlen: int,
    heads: int,
    query_rows: int,
    dim: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(seed + seqlen)
    q = torch.randn(query_rows, heads, dim, device=device, dtype=torch.float32)
    k = torch.randn(seqlen, heads, dim, device=device, dtype=torch.float32)
    v = torch.randn(seqlen, heads, dim, device=device, dtype=torch.float32)
    scores = torch.einsum("qhd,khd->hqk", q, k) / math.sqrt(dim)
    probabilities = torch.softmax(scores, dim=-1)

    nv1, nv1_scales = nv_reconstruct(probabilities, 1.0)
    nv448, nv448_scales = nv_reconstruct(probabilities, 448.0)
    mx, mx_scales = mx_reconstruct(probabilities)
    return {
        "seqlen": seqlen,
        "heads": heads,
        "query_rows": query_rows,
        "dim": dim,
        "formats": [
            evaluate_format(
                name="NVFP4 G=1",
                probabilities=probabilities,
                reconstructed=nv1,
                scales=nv1_scales,
                values=v,
            ),
            evaluate_format(
                name="NVFP4 G=448",
                probabilities=probabilities,
                reconstructed=nv448,
                scales=nv448_scales,
                values=v,
            ),
            evaluate_format(
                name="MXFP4",
                probabilities=probabilities,
                reconstructed=mx,
                scales=mx_scales,
                values=v,
            ),
        ],
    }


def main() -> None:
    args = parse_args()
    torch.cuda.set_device(int(args.gpu))
    device = torch.device("cuda")
    seqlens = args.seqlen or [1024, 4096, 8192]
    result = {
        "schema": "tk_fp4_probability_format_range_v1",
        "protocol": {
            "seed": args.seed,
            "distribution": "Q,K,V iid N(0,1); softmax(QK^T/sqrt(D))",
            "block": "same N32 block for both scale formats",
            "nv_scale": "ideal amax/6 encoded as E4M3, with stated global encode factor",
            "mx_scale": "best-L2 adjacent E8M0 power of two",
            "normalization": "represented E2M1 payload renormalized by represented row sum",
            "scope": "format representability diagnostic; not a kernel timing",
        },
        "shapes": [
            run_shape(
                seqlen=seqlen,
                heads=args.heads,
                query_rows=args.query_rows,
                dim=args.dim,
                seed=args.seed,
                device=device,
            )
            for seqlen in seqlens
        ],
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        fields = (
            "seqlen",
            "format",
            "probability_cosine",
            "probability_relative_l2",
            "output_cosine",
            "output_relative_l2",
            "zero_payload_fraction",
            "zeroed_probability_mass",
            "zero_scale_block_fraction",
            "minimum_nonzero_scale",
        )
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for shape in result["shapes"]:
            for value in shape["formats"]:
                writer.writerow(
                    {
                        "seqlen": shape["seqlen"],
                        "format": value["format"],
                        "probability_cosine": value["probability_error"]["cosine"],
                        "probability_relative_l2": value["probability_error"]["relative_l2"],
                        "output_cosine": value["output_error"]["cosine"],
                        "output_relative_l2": value["output_error"]["relative_l2"],
                        "zero_payload_fraction": value["zero_payload_fraction"],
                        "zeroed_probability_mass": value["zeroed_probability_mass"],
                        "zero_scale_block_fraction": value["zero_scale_block_fraction"],
                        "minimum_nonzero_scale": value["minimum_nonzero_scale"],
                    }
                )

    table_dir = args.output.parent / "tables"
    table_dir.mkdir(exist_ok=True)
    table_lines = []
    for shape in result["shapes"]:
        for value in shape["formats"]:
            table_lines.append(
                " & ".join(
                    (
                        str(shape["seqlen"]),
                        value["format"].replace("=", "{=}"),
                        f'{value["zero_scale_block_fraction"]:.6f}',
                        f'{value["zero_payload_fraction"]:.6f}',
                        f'{value["zeroed_probability_mass"]:.6f}',
                        f'{value["probability_error"]["relative_l2"]:.6f}',
                        f'{value["output_error"]["cosine"]:.6f}',
                        f'{value["output_error"]["relative_l2"]:.6f}',
                    )
                )
                + r" \\"
            )
    (table_dir / "p_range_rows.tex").write_text(
        "\n".join((*table_lines, r"\bottomrule")) + "\n"
    )


if __name__ == "__main__":
    main()
