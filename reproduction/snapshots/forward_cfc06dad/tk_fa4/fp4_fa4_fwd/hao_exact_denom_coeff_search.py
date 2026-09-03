#!/usr/bin/env python3
"""Fit the shiftless NVFP4 affine path with its represented denominator."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Iterable

from hao_approx_coeff_search import (
    E2M1_THRESHOLDS,
    E2M1_VALUES,
    metric_sums,
    merge_metrics,
    native_mask,
    select_rows,
    sobol_candidates,
)


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_HAO_REPO = REPO_ROOT.parents[1] / "flash-attention-fp4"
CURRENT_AFFINE = (1.61131608, 0.93574703)
EXACT_SEED_AFFINE = (1.5, 1.22)
PREVIOUS_AFFINE = (1.62330034, 0.92083546)
CURRENT_CUBIC = (0.07839806, 0.28625049, 0.63145205, 0.99202336)


@dataclass
class ExactCase:
    seed: int
    scores_x: object
    block_scale: object
    native: object
    native_code: object
    cubic_code: object
    affine_selector: object
    reference_probability: object
    value: object
    reference_output: object
    heads: int
    rows_per_head: int


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
    parser.add_argument("--rows-per-head", type=int, default=16)
    parser.add_argument("--seeds", type=parse_csv_ints, default=(0, 1, 2, 3))
    parser.add_argument("--anchor-samples", type=int, default=32)
    parser.add_argument("--anchor-bias", type=float, default=0.125)
    parser.add_argument(
        "--stage0-affine-mask",
        type=lambda value: int(value, 0),
        default=14,
    )
    parser.add_argument(
        "--stage1-affine-mask",
        type=lambda value: int(value, 0),
        default=14,
    )
    parser.add_argument("--candidates", type=int, default=4096)
    parser.add_argument("--refine-rounds", type=int, default=2)
    parser.add_argument("--candidate-batch", type=int, default=32)
    parser.add_argument("--search-seed", type=int, default=20260730)
    parser.add_argument(
        "--objective",
        choices=("probability", "output-l2", "output-cosine"),
        default="output-l2",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPO_ROOT
            / "results"
            / "fp4_fa4_exact_denom_coeff_search_20260730"
            / "search.json"
        ),
    )
    return parser.parse_args()


def global_anchor_permutation(
    seqlen: int,
    samples: int,
    device: object,
) -> object:
    import torch

    anchor = (
        torch.linspace(
            0,
            seqlen - 1,
            samples,
            device=device,
            dtype=torch.float32,
        )
        .round()
        .to(torch.long)
    )
    selected = torch.zeros(seqlen, device=device, dtype=torch.bool)
    selected[anchor] = True
    remainder = torch.arange(seqlen, device=device)[~selected]
    return torch.cat((anchor, remainder))


def e2m1_quantize(values: object, thresholds: object, grid: object) -> object:
    import torch

    code = torch.zeros_like(values, dtype=torch.uint8)
    for threshold in thresholds:
        code.add_(values >= threshold)
    return grid[code.to(torch.int64)]


def encode_mode4_scale(log2_scale: object) -> tuple[object, object]:
    """Reproduce direct E4M3 scale encoding used by scale mode 4."""
    import torch

    scale_code = torch.round(log2_scale * 8.0 + 56.0)
    scale_code = scale_code.clamp(8.0, 126.0).to(torch.int32)
    encoded_log2 = (scale_code.float() - 56.0) * 0.125
    fp32_bits = (scale_code << 20) + (120 << 23)
    return encoded_log2, fp32_bits.view(torch.float32)


def nvfp4_roundtrip(
    matrix: object,
    nvfp4_quantize: object,
    nvfp4_dequantize: object,
    sf_layout: object,
) -> object:
    import torch

    one = torch.ones(1, device=matrix.device, dtype=torch.float32)
    packed, scale = nvfp4_quantize(
        matrix.contiguous(),
        one,
        sfLayout=sf_layout.layout_linear,
        do_shuffle=False,
    )
    return nvfp4_dequantize(
        packed.view(torch.uint8).contiguous(),
        scale.view(torch.uint8).contiguous(),
        one,
    )


def make_case(
    *,
    seed: int,
    seqlen: int,
    heads: int,
    rows_per_head: int,
    anchor_samples: int,
    anchor_bias: float,
    stage0_affine_mask: int,
    stage1_affine_mask: int,
) -> ExactCase:
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
    permutation = global_anchor_permutation(
        seqlen,
        anchor_samples,
        q_ref.device,
    )
    k_ref = k_ref.index_select(1, permutation).contiguous()
    v_ref = v_ref.index_select(1, permutation).contiguous()
    q_bf16 = q_ref.to(torch.bfloat16)
    k_bf16 = k_ref.to(torch.bfloat16)
    v_bf16 = v_ref.to(torch.bfloat16)

    q = nvfp4_roundtrip(
        q_bf16.reshape(seqlen, heads * 128),
        nvfp4_quantize,
        nvfp4_kv_dequantize,
        SfLayout,
    ).reshape(1, seqlen, heads, 128)[0].permute(1, 0, 2).float()
    k = nvfp4_roundtrip(
        k_bf16.reshape(seqlen, heads * 128),
        nvfp4_quantize,
        nvfp4_kv_dequantize,
        SfLayout,
    ).reshape(1, seqlen, heads, 128)[0].permute(1, 0, 2).float()
    value = nvfp4_roundtrip(
        v_bf16.permute(0, 2, 3, 1).reshape(heads * 128, seqlen),
        nvfp4_quantize,
        nvfp4_kv_dequantize,
        SfLayout,
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

    blocks = scores.reshape(heads, rows_per_head, seqlen // 32, 32)
    row_anchor = blocks[:, :, 0].amax(dim=-1) + anchor_bias
    raw_group_max = blocks.amax(dim=-1)
    scale_log2 = math.log2(math.e)
    quant_log2_scale = (
        (raw_group_max - row_anchor[..., None]) * scale_log2
    ).clamp_min(-100.0) - math.log2(6.0)
    encoded_log2, block_scale = encode_mode4_scale(quant_log2_scale)
    quant_bias = (
        -row_anchor[..., None] * scale_log2 - encoded_log2
    )
    scores_x = blocks * scale_log2 + quant_bias[..., None]

    quarter = (
        torch.arange(
            seqlen // 32,
            device=scores.device,
            dtype=torch.int64,
        )
        & 3
    )
    native = native_mask(scores.device)[quarter]
    thresholds = torch.tensor(
        E2M1_THRESHOLDS,
        device=scores.device,
        dtype=torch.float32,
    )
    grid = torch.tensor(
        E2M1_VALUES,
        device=scores.device,
        dtype=torch.float32,
    )
    native_code = e2m1_quantize(torch.exp2(scores_x), thresholds, grid)
    cubic_value = (
        (
            CURRENT_CUBIC[0] * scores_x + CURRENT_CUBIC[1]
        ) * scores_x + CURRENT_CUBIC[2]
    ) * scores_x + CURRENT_CUBIC[3]
    cubic_code = e2m1_quantize(cubic_value, thresholds, grid)
    stage = ((rows // 128) & 1).view(1, -1).expand(heads, -1)
    stage_mask = torch.where(
        stage == 0,
        torch.full_like(stage, stage0_affine_mask),
        torch.full_like(stage, stage1_affine_mask),
    )
    affine_selector = (
        ((stage_mask[..., None] >> quarter) & 1) != 0
    )
    affine_selector[..., 0::4] = True
    return ExactCase(
        seed=seed,
        scores_x=scores_x,
        block_scale=block_scale,
        native=native,
        native_code=native_code,
        cubic_code=cubic_code,
        affine_selector=affine_selector,
        reference_probability=reference_probability,
        value=value,
        reference_output=reference_output,
        heads=heads,
        rows_per_head=rows_per_head,
    )


def represented_probability(
    case: ExactCase,
    coefficients: object,
) -> object:
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
    transformed = (
        coefficients[:, 0, None, None, None, None] *
        case.scores_x[None] +
        coefficients[:, 1, None, None, None, None]
    )
    affine_code = e2m1_quantize(transformed, thresholds, grid)
    selected_code = torch.where(
        case.affine_selector[None, ..., None],
        affine_code,
        case.cubic_code[None],
    )
    code = torch.where(
        case.native[None, None, None],
        case.native_code[None],
        selected_code,
    )
    represented = code * case.block_scale[None, ..., None]
    denominator = represented.sum(dim=(-1, -2), keepdim=True)
    return represented / denominator.clamp_min(1.0e-30)


def candidate_losses(
    coefficients: object,
    cases: Iterable[ExactCase],
    batch_size: int,
    objective: str,
) -> object:
    import torch

    cases = tuple(cases)
    if objective == "probability":
        target_energy = sum(
            case.reference_probability.square().sum()
            for case in cases
        ).clamp_min(1.0e-30)
    else:
        target_energy = sum(
            case.reference_output.square().sum()
            for case in cases
        ).clamp_min(1.0e-30)
    losses = []
    for begin in range(0, coefficients.shape[0], batch_size):
        coeff = coefficients[begin : begin + batch_size]
        error_sq = torch.zeros(
            coeff.shape[0],
            device=coeff.device,
            dtype=torch.float64,
        )
        candidate_sq = torch.zeros_like(error_sq)
        dot = torch.zeros_like(error_sq)
        for case in cases:
            probability = represented_probability(case, coeff)
            if objective == "probability":
                target = case.reference_probability.reshape(
                    case.heads,
                    case.rows_per_head,
                    -1,
                    32,
                )
                error_sq += (
                    probability - target[None]
                ).square().sum(dim=(1, 2, 3, 4)).double()
            else:
                output = probability.reshape(
                    coeff.shape[0],
                    case.heads,
                    case.rows_per_head,
                    -1,
                ) @ case.value
                target = case.reference_output[None]
                error_sq += (
                    output - target
                ).square().sum(dim=(1, 2, 3)).double()
                candidate_sq += output.square().sum(
                    dim=(1, 2, 3)
                ).double()
                dot += (output * target).sum(dim=(1, 2, 3)).double()
        if objective == "output-cosine":
            loss = 1.0 - dot / torch.sqrt(
                candidate_sq * target_energy.double()
            )
        else:
            loss = error_sq / target_energy.double()
        losses.append(loss.cpu())
    return torch.cat(losses)


def search_affine(
    *,
    cases: Iterable[ExactCase],
    count: int,
    refine_rounds: int,
    search_seed: int,
    batch_size: int,
    objective: str,
) -> tuple[object, object]:
    import torch

    bounds = ((0.8, 2.4), (0.2, 1.8))
    candidates = sobol_candidates(
        count,
        bounds,
        search_seed,
        cases[0].scores_x.device,
    )
    candidates = torch.cat(
        (
            torch.tensor(
                (CURRENT_AFFINE, EXACT_SEED_AFFINE, PREVIOUS_AFFINE),
                device=candidates.device,
                dtype=torch.float32,
            ),
            candidates,
        ),
        dim=0,
    )
    for round_index in range(refine_rounds + 1):
        losses = candidate_losses(
            candidates,
            cases,
            batch_size,
            objective,
        )
        keep_count = min(32, losses.numel())
        best_indices = torch.topk(
            losses,
            keep_count,
            largest=False,
        ).indices
        best = candidates[best_indices.to(candidates.device)]
        best_losses = losses[best_indices]
        if round_index == refine_rounds:
            return best, best_losses

        span = torch.tensor(
            [high - low for low, high in bounds],
            device=candidates.device,
        )
        radius = span * (0.08 / (2**round_index))
        children_per_parent = max(64, count // keep_count)
        engine = torch.quasirandom.SobolEngine(
            3,
            scramble=True,
            seed=search_seed + round_index + 1,
        )
        unit = engine.draw(keep_count * children_per_parent).to(
            candidates.device
        )
        parent_index = torch.floor(
            unit[:, 0] * keep_count
        ).to(torch.int64).clamp_max(keep_count - 1)
        children = best[parent_index] + (unit[:, 1:] * 2.0 - 1.0) * radius
        low = torch.tensor(
            [item[0] for item in bounds],
            device=candidates.device,
        )
        high = torch.tensor(
            [item[1] for item in bounds],
            device=candidates.device,
        )
        children = torch.minimum(torch.maximum(children, low), high)
        candidates = torch.cat((best, children), dim=0)
    raise AssertionError("unreachable")


def output_metrics(
    coefficients: object,
    cases: Iterable[ExactCase],
) -> list[dict[str, object]]:
    records = []
    for coefficient in coefficients:
        parts = []
        for case in cases:
            probability = represented_probability(
                case,
                coefficient[None],
            )[0].reshape(case.heads, case.rows_per_head, -1)
            output = probability @ case.value
            parts.append(metric_sums(output, case.reference_output))
        records.append(
            {
                "coefficients": [float(v) for v in coefficient.tolist()],
                "output": merge_metrics(parts),
            }
        )
    return records


def main() -> None:
    args = parse_args()
    if args.seqlen % 128:
        raise ValueError("--seqlen must be divisible by 128")
    if args.rows_per_head > args.seqlen:
        raise ValueError("--rows-per-head cannot exceed --seqlen")
    if args.anchor_samples not in (32, 64, 128):
        raise ValueError("--anchor-samples must be 32, 64, or 128")
    sys.path.insert(0, str(args.hao_repo))
    try:
        import torch

        torch.cuda.set_device(0)
        cases = tuple(
            make_case(
                seed=seed,
                seqlen=args.seqlen,
                heads=args.heads,
                rows_per_head=args.rows_per_head,
                anchor_samples=args.anchor_samples,
                anchor_bias=args.anchor_bias,
                stage0_affine_mask=args.stage0_affine_mask,
                stage1_affine_mask=args.stage1_affine_mask,
            )
            for seed in args.seeds
        )
        best, losses = search_affine(
            cases=cases,
            count=args.candidates,
            refine_rounds=args.refine_rounds,
            search_seed=args.search_seed,
            batch_size=args.candidate_batch,
            objective=args.objective,
        )
        records = output_metrics(best, cases)
    finally:
        sys.path.pop(0)

    for record, loss in zip(records, losses.tolist()):
        record["objective_loss"] = float(loss)
    records.sort(key=lambda item: item["output"]["relative_l2"])
    result = {
        "protocol": {
            "seqlen": args.seqlen,
            "heads": args.heads,
            "rows_per_head": args.rows_per_head,
            "seeds": list(args.seeds),
            "anchor_samples": args.anchor_samples,
            "anchor_bias": args.anchor_bias,
            "stage0_affine_mask": args.stage0_affine_mask,
            "stage1_affine_mask": args.stage1_affine_mask,
            "candidates": args.candidates,
            "refine_rounds": args.refine_rounds,
            "objective": args.objective,
            "normalization": (
                "exact represented E2M1 payload times E4M3 quarter scale"
            ),
            "reference": "BF16 Q/K/V attention output",
        },
        "candidates_by_output_relative_l2": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
