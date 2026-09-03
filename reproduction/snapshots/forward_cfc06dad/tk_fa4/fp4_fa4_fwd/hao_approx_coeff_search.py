#!/usr/bin/env python3
"""Search shiftless NVFP4 polynomial fits against attention-output error."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Iterable


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_HAO_REPO = REPO_ROOT.parents[1] / "flash-attention-fp4"

E2M1_THRESHOLDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
CURRENT_AFFINE = (1.62330034, 0.92083546)
CURRENT_QUADRATIC = (0.24022651, 0.69314718, 1.0)
CURRENT_CUBIC = (0.07839806, 0.28625049, 0.63145205, 0.99202336)
CURRENT_REFIT_CUBIC = (
    0.07430709,
    0.28611863,
    0.64670005,
    0.99010784,
)


@dataclass
class SimulationCase:
    seed: int
    scores_x: object
    block_scale: object
    denominator: object
    native_mask: object
    stage: object
    quarter: object
    value: object
    reference_probability: object
    reference_output: object


def parse_csv_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(","))
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hao-repo", type=Path, default=DEFAULT_HAO_REPO)
    parser.add_argument("--seqlen", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--rows-per-head", type=int, default=64)
    parser.add_argument("--seeds", type=parse_csv_ints, default=(0, 1, 2, 3))
    parser.add_argument("--sample-count", type=int, default=131072)
    parser.add_argument("--affine-candidates", type=int, default=8192)
    parser.add_argument("--quadratic-candidates", type=int, default=8192)
    parser.add_argument("--cubic-candidates", type=int, default=16384)
    parser.add_argument("--refine-rounds", type=int, default=2)
    parser.add_argument("--candidate-batch", type=int, default=128)
    parser.add_argument("--search-seed", type=int, default=20260729)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPO_ROOT
            / "results"
            / "fp4_fa4_approx_coeff_search_20260729"
            / "offline_search.json"
        ),
    )
    return parser.parse_args()


def e2m1_quantize(values: object, thresholds: object, grid: object) -> object:
    import torch

    code = torch.zeros_like(values, dtype=torch.uint8)
    for threshold in thresholds:
        code.add_(values >= threshold)
    return grid[code.to(torch.int64)]


def native_mask(device: object) -> object:
    import torch

    pairs = torch.arange(16, device=device).repeat_interleave(2)
    masks = []
    for quarter in range(4):
        masks.append(
            (pairs == quarter)
            | (pairs == quarter + 8)
            | (pairs == 4)
            | (pairs == 12)
        )
    return torch.stack(masks)


def encode_nvfp4_scale(log2_scale: object) -> tuple[object, object]:
    import torch

    magic = torch.tensor(
        12582912.0 + 56.0,
        device=log2_scale.device,
        dtype=torch.float32,
    )
    carrier = (log2_scale * 8.0 + magic).to(torch.float32)
    encoded_log2 = (carrier - magic) * 0.125
    byte = carrier.view(torch.int32) & 0xff
    fp32_bits = (byte << 20) + (120 << 23)
    return encoded_log2, fp32_bits.view(torch.float32)


def select_rows(
    seqlen: int,
    rows_per_head: int,
    seed: int,
    device: object,
) -> object:
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed * 1009 + 17)
    selected = []
    tiles = seqlen // 128
    base = rows_per_head // tiles
    remainder = rows_per_head % tiles
    for tile in range(tiles):
        count = base + int(tile < remainder)
        if count == 0:
            continue
        local = torch.randperm(128, generator=generator)[:count]
        selected.append(local + tile * 128)
    return torch.cat(selected).sort().values.to(device)


def make_case(
    *,
    seed: int,
    seqlen: int,
    heads: int,
    rows_per_head: int,
) -> SimulationCase:
    import torch
    from flash_attn.cute.benchmarks import bench_fp4
    from flashinfer.quantization import (
        SfLayout,
        nvfp4_kv_dequantize,
        nvfp4_quantize,
    )

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tensors = bench_fp4.create_nvfp4_attention_tensors(
        1,
        seqlen,
        seqlen,
        heads,
        heads,
        128,
        128,
        device="cuda",
        dtype_gen=torch.bfloat16,
        pv_mode="fp4",
    )
    q_ref, k_ref, v_ref = tensors[6:9]
    one = torch.ones(1, device="cuda", dtype=torch.float32)

    def nvfp4_roundtrip(matrix: object) -> object:
        packed, scale = nvfp4_quantize(
            matrix.contiguous(),
            one,
            sfLayout=SfLayout.layout_linear,
            do_shuffle=False,
        )
        return nvfp4_kv_dequantize(
            packed.view(torch.uint8).contiguous(),
            scale.view(torch.uint8).contiguous(),
            one,
        )

    q_bf16 = q_ref.to(torch.bfloat16)
    k_bf16 = k_ref.to(torch.bfloat16)
    v_bf16 = v_ref.to(torch.bfloat16)
    q = nvfp4_roundtrip(
        q_bf16.reshape(seqlen, heads * 128),
    ).reshape(1, seqlen, heads, 128)[0].permute(1, 0, 2).float()
    k = nvfp4_roundtrip(
        k_bf16.reshape(seqlen, heads * 128),
    ).reshape(1, seqlen, heads, 128)[0].permute(1, 0, 2).float()
    value = nvfp4_roundtrip(
        v_bf16.permute(0, 2, 3, 1).reshape(heads * 128, seqlen),
    ).reshape(1, heads, 128, seqlen).permute(0, 3, 1, 2)[0]
    value = value.permute(1, 0, 2).float()
    rows = select_rows(seqlen, rows_per_head, seed, q.device)
    selected_q = q[:, rows]
    scores = torch.matmul(selected_q, k.transpose(1, 2)) / math.sqrt(128.0)
    reference_q = q_bf16[0].permute(1, 0, 2).float()
    reference_k = k_bf16[0].permute(1, 0, 2).float()
    reference_v = v_bf16[0].permute(1, 0, 2).float()
    reference_scores = (
        torch.matmul(
            reference_q[:, rows],
            reference_k.transpose(1, 2),
        )
        / math.sqrt(128.0)
    )
    reference_probability = torch.softmax(reference_scores, dim=-1)
    reference_output = torch.matmul(reference_probability, reference_v)

    row_stage = ((rows // 128) & 1).view(1, -1).expand(heads, -1)
    blocks = scores.mul(math.log2(math.e)).reshape(
        heads,
        rows_per_head,
        seqlen // 32,
        32,
    )
    raw_max = blocks.amax(dim=-1)
    quant_log2_scale = raw_max - math.log2(6.0)
    encoded_log2, block_scale = encode_nvfp4_scale(quant_log2_scale)
    scores_x = blocks - encoded_log2[..., None]

    quarter = torch.arange(
        seqlen // 32,
        device=scores.device,
        dtype=torch.int64,
    ) & 3
    quarter_native = native_mask(scores.device)[quarter]
    native_exp = torch.where(
        quarter_native.view(1, 1, seqlen // 32, 32),
        torch.exp2(scores_x),
        torch.zeros_like(scores_x),
    )
    denominator = (
        block_scale * native_exp.sum(dim=-1) * 4.0
    ).sum(dim=-1)

    return SimulationCase(
        seed=seed,
        scores_x=scores_x,
        block_scale=block_scale,
        denominator=denominator,
        native_mask=quarter_native,
        stage=row_stage,
        quarter=quarter,
        value=value,
        reference_probability=reference_probability,
        reference_output=reference_output,
    )


def collect_search_samples(
    cases: Iterable[SimulationCase],
    sample_count: int,
    search_seed: int,
) -> tuple[object, object, object]:
    import torch

    xs = []
    gains = []
    targets = []
    for case in cases:
        heads, rows, blocks, width = case.scores_x.shape
        native = case.native_mask.view(1, 1, blocks, width)
        probability = case.reference_probability.reshape(
            heads,
            rows,
            blocks,
            width,
        )
        gain = (
            case.block_scale / case.denominator[..., None]
        )[..., None].expand_as(case.scores_x)
        keep = (~native).expand(heads, rows, blocks, width)
        xs.append(case.scores_x[keep])
        gains.append(gain[keep])
        targets.append(probability[keep])

    x = torch.cat(xs)
    gain = torch.cat(gains)
    target = torch.cat(targets)
    if x.numel() > sample_count:
        generator = torch.Generator(device=x.device)
        generator.manual_seed(search_seed)
        indices = torch.randperm(
            x.numel(),
            generator=generator,
            device=x.device,
        )[:sample_count]
        x = x[indices]
        gain = gain[indices]
        target = target[indices]
    return x, gain, target


def candidate_losses(
    coefficients: object,
    degree: int,
    x: object,
    gain: object,
    target: object,
    batch_size: int,
) -> object:
    import torch

    thresholds = torch.tensor(
        E2M1_THRESHOLDS,
        device=x.device,
        dtype=torch.float32,
    )
    grid = torch.tensor(
        E2M1_VALUES,
        device=x.device,
        dtype=torch.float32,
    )
    losses = []
    target_energy = target.square().mean().clamp_min(1e-30)
    for begin in range(0, coefficients.shape[0], batch_size):
        coeff = coefficients[begin : begin + batch_size]
        if degree == 1:
            transformed = coeff[:, 0, None] * x + coeff[:, 1, None]
        elif degree == 2:
            transformed = (
                coeff[:, 0, None] * x
                + coeff[:, 1, None]
            ) * x + coeff[:, 2, None]
        elif degree == 3:
            transformed = (
                (
                    coeff[:, 0, None] * x
                    + coeff[:, 1, None]
                )
                * x
                + coeff[:, 2, None]
            ) * x + coeff[:, 3, None]
        else:
            raise ValueError(f"unsupported degree {degree}")
        quantized = e2m1_quantize(transformed, thresholds, grid)
        error = gain * quantized - target
        losses.append(
            (error.square().mean(dim=1) / target_energy).cpu()
        )
    return torch.cat(losses)


def sobol_candidates(
    count: int,
    bounds: tuple[tuple[float, float], ...],
    seed: int,
    device: object,
) -> object:
    import torch

    engine = torch.quasirandom.SobolEngine(
        len(bounds),
        scramble=True,
        seed=seed,
    )
    unit = engine.draw(count).to(device)
    low = torch.tensor([item[0] for item in bounds], device=device)
    high = torch.tensor([item[1] for item in bounds], device=device)
    return low + unit * (high - low)


def monotonic_polynomial(
    coefficients: object,
    degree: int,
    x_min: float,
    x_max: float,
) -> object:
    import torch

    grid = torch.linspace(
        x_min,
        x_max,
        65,
        device=coefficients.device,
    )
    if degree == 2:
        derivative = (
            2.0 * coefficients[:, 0, None] * grid
            + coefficients[:, 1, None]
        )
    elif degree == 3:
        derivative = (
            3.0 * coefficients[:, 0, None] * grid.square()
            + 2.0 * coefficients[:, 1, None] * grid
            + coefficients[:, 2, None]
        )
    else:
        raise ValueError(f"unsupported monotonic degree {degree}")
    return (derivative > 0.0).all(dim=1)


def search_family(
    *,
    degree: int,
    initial: tuple[tuple[float, ...], ...],
    bounds: tuple[tuple[float, float], ...],
    count: int,
    refine_rounds: int,
    search_seed: int,
    x: object,
    gain: object,
    target: object,
    batch_size: int,
) -> list[dict[str, object]]:
    import torch

    candidates = sobol_candidates(
        count,
        bounds,
        search_seed + degree,
        x.device,
    )
    candidates = torch.cat(
        [
            torch.tensor(initial, device=x.device, dtype=torch.float32),
            candidates,
        ],
        dim=0,
    )
    if degree in (2, 3):
        candidates = candidates[
            monotonic_polynomial(
                candidates,
                degree,
                float(x.min()),
                float(x.max()),
            )
        ]

    for round_index in range(refine_rounds + 1):
        losses = candidate_losses(
            candidates,
            degree,
            x,
            gain,
            target,
            batch_size,
        )
        keep_count = min(32, losses.numel())
        best_indices = torch.topk(
            losses,
            keep_count,
            largest=False,
        ).indices
        best = candidates[best_indices.to(candidates.device)]
        if round_index == refine_rounds:
            records = []
            for coeff, loss in zip(
                best.cpu().tolist(),
                losses[best_indices].tolist(),
            ):
                records.append(
                    {
                        "coefficients": coeff,
                        "normalized_weight_mse": loss,
                    }
                )
            return records

        span = torch.tensor(
            [high - low for low, high in bounds],
            device=x.device,
        )
        radius = span * (0.08 / (2**round_index))
        children_per_parent = max(64, count // keep_count)
        engine = torch.quasirandom.SobolEngine(
            len(bounds) + 1,
            scramble=True,
            seed=search_seed + 100 * degree + round_index,
        )
        unit = engine.draw(keep_count * children_per_parent).to(x.device)
        parent_index = torch.floor(
            unit[:, 0] * keep_count
        ).to(torch.int64).clamp_max(keep_count - 1)
        jitter = (unit[:, 1:] * 2.0 - 1.0) * radius
        candidates = best[parent_index] + jitter
        low = torch.tensor([item[0] for item in bounds], device=x.device)
        high = torch.tensor([item[1] for item in bounds], device=x.device)
        candidates = torch.minimum(torch.maximum(candidates, low), high)
        candidates = torch.cat([best, candidates], dim=0)
        if degree in (2, 3):
            candidates = candidates[
                monotonic_polynomial(
                    candidates,
                    degree,
                    float(x.min()),
                    float(x.max()),
                )
            ]
    raise AssertionError("unreachable")


def metric_sums(candidate: object, reference: object) -> dict[str, float]:
    delta = candidate - reference
    return {
        "dot": float((candidate * reference).sum().item()),
        "candidate_sq": float(candidate.square().sum().item()),
        "reference_sq": float(reference.square().sum().item()),
        "error_sq": float(delta.square().sum().item()),
        "count": float(delta.numel()),
        "max_abs": float(delta.abs().max().item()),
    }


def merge_metrics(parts: Iterable[dict[str, float]]) -> dict[str, float]:
    totals = {
        key: sum(part[key] for part in parts)
        for key in ("dot", "candidate_sq", "reference_sq", "error_sq", "count")
    }
    max_abs = max(part["max_abs"] for part in parts)
    cosine = totals["dot"] / math.sqrt(
        totals["candidate_sq"] * totals["reference_sq"]
    )
    rmse = math.sqrt(totals["error_sq"] / totals["count"])
    reference_rms = math.sqrt(totals["reference_sq"] / totals["count"])
    return {
        "cosine": cosine,
        "rmse": rmse,
        "relative_l2": rmse / reference_rms,
        "max_abs": max_abs,
    }


def case_contributions(
    case: SimulationCase,
    affine: tuple[float, float],
    cubic: tuple[float, float, float, float],
) -> tuple[object, dict[tuple[int, int], object]]:
    import torch

    thresholds = torch.tensor(
        E2M1_THRESHOLDS,
        device=case.scores_x.device,
        dtype=torch.float32,
    )
    grid = torch.tensor(
        E2M1_VALUES,
        device=case.scores_x.device,
        dtype=torch.float32,
    )
    native = case.native_mask.view(
        1,
        1,
        case.scores_x.shape[2],
        32,
    )
    native_value = torch.exp2(case.scores_x)
    affine_value = affine[0] * case.scores_x + affine[1]
    cubic_value = (
        (
            cubic[0] * case.scores_x + cubic[1]
        )
        * case.scores_x
        + cubic[2]
    ) * case.scores_x + cubic[3]
    affine_quantized = e2m1_quantize(affine_value, thresholds, grid)
    cubic_quantized = e2m1_quantize(cubic_value, thresholds, grid)
    native_quantized = e2m1_quantize(native_value, thresholds, grid)
    affine_quantized = torch.where(
        native,
        native_quantized,
        affine_quantized,
    )
    cubic_quantized = torch.where(
        native,
        native_quantized,
        cubic_quantized,
    )
    affine_weight = (
        case.block_scale[..., None] * affine_quantized
    ) / case.denominator[..., None, None]
    cubic_weight = (
        case.block_scale[..., None] * cubic_quantized
    ) / case.denominator[..., None, None]

    heads, rows, blocks, width = case.scores_x.shape
    all_cubic = torch.matmul(
        cubic_weight.reshape(heads, rows, blocks * width),
        case.value,
    )
    deltas = {}
    for stage in range(2):
        row_mask = case.stage == stage
        for quarter in range(4):
            block_mask = case.quarter == quarter
            weight_delta = torch.where(
                row_mask[..., None, None]
                & block_mask.view(1, 1, blocks, 1),
                affine_weight - cubic_weight,
                torch.zeros_like(affine_weight),
            )
            deltas[(stage, quarter)] = torch.matmul(
                weight_delta.reshape(heads, rows, blocks * width),
                case.value,
            )
    return all_cubic, deltas


def placement_sweep(
    cases: Iterable[SimulationCase],
    affine: tuple[float, float],
    cubic: tuple[float, float, float, float],
) -> list[dict[str, object]]:
    prepared = [
        (case, *case_contributions(case, affine, cubic))
        for case in cases
    ]
    masks = (0, 2, 4, 6, 8, 10, 12, 14)
    records = []
    for stage0_mask in masks:
        for stage1_mask in masks:
            parts = []
            for case, all_cubic, deltas in prepared:
                output = all_cubic.clone()
                # Q0 always uses the fast affine in the production mode.
                output += deltas[(0, 0)] + deltas[(1, 0)]
                for stage, mask in (
                    (0, stage0_mask),
                    (1, stage1_mask),
                ):
                    for quarter in range(1, 4):
                        if mask & (1 << quarter):
                            output += deltas[(stage, quarter)]
                parts.append(metric_sums(output, case.reference_output))
            records.append(
                {
                    "stage0_affine_mask": stage0_mask,
                    "stage1_affine_mask": stage1_mask,
                    **merge_metrics(parts),
                }
            )
    records.sort(key=lambda item: item["relative_l2"])
    return records


def main() -> None:
    args = parse_args()
    if args.seqlen % 128 != 0:
        raise ValueError("--seqlen must be divisible by 128")
    if args.rows_per_head > args.seqlen:
        raise ValueError("--rows-per-head cannot exceed --seqlen")
    sys.path.insert(0, str(args.hao_repo))
    try:
        import torch

        torch.cuda.set_device(0)
        cases = [
            make_case(
                seed=seed,
                seqlen=args.seqlen,
                heads=args.heads,
                rows_per_head=args.rows_per_head,
            )
            for seed in args.seeds
        ]
    finally:
        sys.path.pop(0)

    x, gain, target = collect_search_samples(
        cases,
        args.sample_count,
        args.search_seed,
    )
    affine = search_family(
        degree=1,
        initial=(CURRENT_AFFINE,),
        bounds=((0.8, 2.4), (0.2, 1.6)),
        count=args.affine_candidates,
        refine_rounds=args.refine_rounds,
        search_seed=args.search_seed,
        x=x,
        gain=gain,
        target=target,
        batch_size=args.candidate_batch,
    )
    quadratic = search_family(
        degree=2,
        initial=(CURRENT_QUADRATIC,),
        bounds=(
            (0.02, 0.50),
            (0.35, 1.20),
            (0.70, 1.25),
        ),
        count=args.quadratic_candidates,
        refine_rounds=args.refine_rounds,
        search_seed=args.search_seed,
        x=x,
        gain=gain,
        target=target,
        batch_size=args.candidate_batch,
    )
    cubic = search_family(
        degree=3,
        initial=(CURRENT_CUBIC, CURRENT_REFIT_CUBIC),
        bounds=(
            (0.02, 0.14),
            (0.12, 0.48),
            (0.35, 0.95),
            (0.75, 1.20),
        ),
        count=args.cubic_candidates,
        refine_rounds=args.refine_rounds,
        search_seed=args.search_seed,
        x=x,
        gain=gain,
        target=target,
        batch_size=args.candidate_batch,
    )
    best_affine = tuple(float(v) for v in affine[0]["coefficients"])
    best_cubic = tuple(float(v) for v in cubic[0]["coefficients"])
    placements = placement_sweep(cases, best_affine, best_cubic)
    baseline_placements = placement_sweep(
        cases,
        CURRENT_AFFINE,
        CURRENT_CUBIC,
    )

    result = {
        "protocol": {
            "seqlen": args.seqlen,
            "heads": args.heads,
            "rows_per_head": args.rows_per_head,
            "seeds": list(args.seeds),
            "sample_count": int(x.numel()),
            "objective": (
                "normalized squared error between reconstructed attention "
                "weights and BF16 softmax weights"
            ),
            "operand_model": (
                "NVFP4 Q/K/V roundtrip with linear-layout E4M3 block "
                "scales; BF16 Q/K/V for the reference output"
            ),
            "native_pairs_per_quarter": 4,
            "denominator": "four rotating native float2 pairs, multiplied by 4",
        },
        "current": {
            "affine": list(CURRENT_AFFINE),
            "quadratic": list(CURRENT_QUADRATIC),
            "cubic": list(CURRENT_CUBIC),
            "refit_cubic": list(CURRENT_REFIT_CUBIC),
        },
        "affine_candidates": affine,
        "quadratic_candidates": quadratic,
        "cubic_candidates": cubic,
        "best_optimized_placements": placements,
        "current_coefficient_placements": baseline_placements,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
