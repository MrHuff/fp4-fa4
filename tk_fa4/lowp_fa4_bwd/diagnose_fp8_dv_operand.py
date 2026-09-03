#!/usr/bin/env python3
"""Isolate the E4M3 P x dO path from the rest of attention backward."""

from __future__ import annotations

import argparse
import json
import math

import torch

from tk_fa4.lowp_fa4_bwd.profile_gqa_d128_chain import (
    CompiledGqaBackward,
    _gqa_dv_reference,
    _metrics,
)
from tk_fa4.lowp_fa4_bwd.tune_d64_gqa_cute import _load_control


def _exact_lse(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    _, sequence, q_heads, depth = q.shape
    kv_heads = k.shape[2]
    ratio = q_heads // kv_heads
    mask = torch.ones(
        sequence,
        sequence,
        device=q.device,
        dtype=torch.bool,
    ).triu_(1)
    lse = torch.empty(1, q_heads, 1, sequence, device=q.device)
    for q_head in range(q_heads):
        scores = torch.mm(
            q[0, :, q_head].float(),
            k[0, :, q_head // ratio].float().T,
        ) * (depth**-0.5)
        scores.masked_fill_(mask, float("-inf"))
        lse[0, q_head, 0] = torch.logsumexp(scores, dim=-1)
    return lse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=int, default=4096)
    parser.add_argument("--q-heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--reverse-query-tiles", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise RuntimeError("diagnostic requires exactly one visible GPU")
    if args.q_heads % args.kv_heads:
        raise ValueError("q-heads must be divisible by kv-heads")
    torch.manual_seed(args.seed)
    depth = 128
    shape_q = (1, args.sequence, args.q_heads, depth)
    shape_kv = (1, args.sequence, args.kv_heads, depth)
    q = (torch.randn(shape_q, device="cuda") * 0.25).to(
        torch.float8_e4m3fn
    )
    k = (torch.randn(shape_kv, device="cuda") * 0.25).to(
        torch.float8_e4m3fn
    )
    v = (torch.randn(shape_kv, device="cuda") * 0.25).to(
        torch.float8_e4m3fn
    )
    dout = (torch.randn(shape_q, device="cuda") * 0.25).to(
        torch.float8_e4m3fn
    )
    random_dout = dout.clone()
    lse = _exact_lse(q, k)
    precomputed_lse = (-lse * math.log2(math.e)).contiguous()
    precomputed_sum = torch.zeros_like(precomputed_lse)

    control = _load_control()
    backward = CompiledGqaBackward(
        control,
        q=q,
        k=k,
        v=v,
        o_or_sum=precomputed_sum,
        dout=dout,
        lse_or_scaled_lse=precomputed_lse,
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
        lowp=True,
        precomputed_stats=True,
        scale_softmax=depth**-0.5,
        exp2_period=0,
        reuse_quantized_p=True,
        reverse_query_tiles=args.reverse_query_tiles,
    )

    def run_case(name: str, values: torch.Tensor) -> dict[str, object]:
        dout.copy_(values)
        backward.run(reset=True)
        torch.cuda.synchronize()
        represented, represented_partials = _gqa_dv_reference(
            q.float(),
            k.float(),
            dout.float(),
            lse_bh1s=lse,
            probability_dtype=torch.float8_e4m3fn,
            probability_lift=256.0,
        )
        return {
            "final": _metrics(represented, backward.dv),
            "partials": _metrics(
                represented_partials,
                backward.dv_partials.float() / 256.0,
            ),
            "dimension_std_ratio": float(
                backward.dv.float().std(dim=-1).mean()
                / represented.std(dim=-1).mean().clamp_min(1.0e-20)
            ),
            "name": name,
        }

    repeated_column = random_dout[..., :1].expand_as(random_dout).contiguous()
    constant = torch.full_like(random_dout, 0.25)
    random_result = run_case("random", random_dout)
    repeated_result = run_case("repeated_column", repeated_column)
    constant_result = run_case("constant", constant)

    # With zero scores, every causal row is uniform and dV is a reverse
    # weighted prefix sum.  Its adjacent differences recover the effective
    # dO sequence seen by PdO, exposing any in-tile index permutation.
    q.zero_()
    k.zero_()
    positions = torch.arange(1, args.sequence + 1, device="cuda").float()
    lse.zero_()
    lse[0, :, 0] = positions.log()[None]
    precomputed_lse.copy_((-lse * math.log2(math.e)).contiguous())
    signs = torch.where(
        torch.rand(args.sequence, device="cuda") >= 0.5,
        1.0,
        -1.0,
    ).to(torch.float8_e4m3fn)
    signed_dout = signs.view(1, args.sequence, 1, 1).expand_as(dout)
    dout.copy_(signed_dout)
    backward.run(reset=True)
    torch.cuda.synchronize()
    partial = backward.dv_partials[0, 0, :, 0].float()
    next_partial = torch.zeros_like(partial)
    next_partial[:-1] = partial[1:]
    represented_probability_lifted = (
        (256.0 / positions).to(torch.float8_e4m3fn).float()
    )
    recovered = (partial - next_partial) / represented_probability_lifted
    expected = signs.float()
    tiles = args.sequence // 128
    recovered_by_residue = recovered.view(tiles, 128).T
    expected_by_residue = expected.view(tiles, 128).T
    recovered_unit = recovered_by_residue / recovered_by_residue.norm(
        dim=1, keepdim=True
    ).clamp_min(1.0e-20)
    expected_unit = expected_by_residue / expected_by_residue.norm(
        dim=1, keepdim=True
    ).clamp_min(1.0e-20)
    residue_correlation = recovered_unit @ expected_unit.T
    best_correlation, best_source = residue_correlation.abs().max(dim=1)
    zero_score_probe = {
        "recovered_vs_input": _metrics(expected, recovered),
        "mean_best_residue_correlation": float(best_correlation.mean()),
        "minimum_best_residue_correlation": float(best_correlation.min()),
        "source_residue_for_effective": best_source.cpu().tolist(),
        "signed_best_residue_correlation": residue_correlation[
            torch.arange(128, device="cuda"), best_source
        ].cpu().tolist(),
    }
    results = {
        "shape": {
            "sequence": args.sequence,
            "q_heads": args.q_heads,
            "kv_heads": args.kv_heads,
            "head_dim": depth,
            "reverse_query_tiles": args.reverse_query_tiles,
        },
        "random": random_result,
        "repeated_column": repeated_result,
        "constant": constant_result,
        "zero_score_index_probe": zero_score_probe,
    }
    rendered = json.dumps(results, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
        print(f"wrote {args.output}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
