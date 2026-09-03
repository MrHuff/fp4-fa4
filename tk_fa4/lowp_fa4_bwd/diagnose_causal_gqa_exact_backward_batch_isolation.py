#!/usr/bin/env python3
"""Isolate batched-backward cross-sample behavior without B1 controllers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tk_fa4.lowp_fa4_bwd.tune_d64_gqa_cute import _load_control
from tk_fa4.lowp_fa4_bwd.validate_causal_gqa_exact_backward_batch import (
    _authenticate_control,
    _build_lowp,
    _capture,
    _gradient_metrics,
    _make_state,
    _publish_workspace_statistics,
)


def _sample(
    values: tuple[torch.Tensor, ...], index: int
) -> tuple[torch.Tensor, ...]:
    return tuple(value[index : index + 1] for value in values)


def _cross_sample_aggregates(
    baseline: tuple[torch.Tensor, ...],
    actual: tuple[torch.Tensor, ...],
    batch: int,
) -> dict[str, dict[str, object]]:
    return {
        f"sample_{index}": _gradient_metrics(
            _sample(baseline, index), _sample(actual, index)
        )["aggregate"]
        for index in range(batch)
    }


def _sample_zero_mapping(
    baseline: tuple[torch.Tensor, ...],
    actual: tuple[torch.Tensor, ...],
    last_index: int,
) -> dict[str, dict[str, object]]:
    sample_zero = _sample(actual, 0)
    baseline_zero = _sample(baseline, 0)
    baseline_last = _sample(baseline, last_index)
    return {
        "baseline_sample_0": _gradient_metrics(
            baseline_zero, sample_zero
        )["aggregate"],
        "baseline_last_sample": _gradient_metrics(
            baseline_last, sample_zero
        )["aggregate"],
        "negative_baseline_last_sample": _gradient_metrics(
            tuple(-value for value in baseline_last), sample_zero
        )["aggregate"],
    }


def _workspace_stats_views(
    backward: object,
    state: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    stats_numel = state.direct_dpsum.numel()
    pages = backward.workspace_torch[: 2 * stats_numel * 4].view(
        torch.float32
    )
    return (
        pages[:stats_numel].view_as(state.direct_dpsum),
        pages[stats_numel:].view_as(state.direct_lse_log2),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, choices=(8, 16), required=True)
    parser.add_argument("--control-source", required=True, type=Path)
    parser.add_argument("--control-sha256", required=True)
    parser.add_argument("--control-bytes", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise RuntimeError("expose exactly one GPU to this diagnostic")
    torch.cuda.set_device(0)
    control_identity = _authenticate_control(
        args.control_source,
        args.control_sha256,
        args.control_bytes,
    )
    state = _make_state(
        batch=args.batch,
        sequence=4096,
        q_heads=32,
        kv_heads=8,
        seed=args.seed,
    )
    control = _load_control(
        fp8_p_storage="tmem",
        direct_tma_dkdv=True,
        precomposed_control_source=args.control_source,
        precomposed_control_sha256=args.control_sha256,
        precomposed_control_bytes=args.control_bytes,
    )
    batched = _build_lowp(control, state, q_heads=32, kv_heads=8)

    perturbed_index = args.batch - 1
    unperturbed_captures = [_capture(batched) for _ in range(6)]
    baseline = unperturbed_captures[0]
    unperturbed_repeats = [
        {
            "capture": index,
            "cross_sample": _cross_sample_aggregates(
                baseline, capture, args.batch
            ),
            "sample_zero_mapping": _sample_zero_mapping(
                baseline, capture, perturbed_index
            ),
        }
        for index, capture in enumerate(unperturbed_captures[1:], start=1)
    ]

    original_dout = state.dout_fp8[perturbed_index].clone()
    original_dpsum = state.direct_dpsum[perturbed_index].clone()
    workspace_dpsum, workspace_lse = _workspace_stats_views(batched, state)
    lse0_before_publication = workspace_lse[0].clone()

    # A: perturb dO only.  The prepublished dPsum and scaled-LSE pages remain
    # byte-for-byte untouched.
    state.dout_fp8[perturbed_index].copy_(
        (-original_dout.float()).to(torch.float8_e4m3fn)
    )
    dout_only = _capture(batched)
    state.dout_fp8[perturbed_index].copy_(original_dout)

    # B: perturb only the final dPsum page while dO remains original.
    state.direct_dpsum[perturbed_index].copy_(-original_dpsum)
    _publish_workspace_statistics(batched, state)
    lse0_after_dpsum_publication = workspace_lse[0].clone()
    dpsum_only = _capture(batched)

    # C: combine the consistent dO and dPsum sign flip used by the full
    # validator.  LSE is intentionally unchanged.
    state.dout_fp8[perturbed_index].copy_(
        (-original_dout.float()).to(torch.float8_e4m3fn)
    )
    combined = _capture(batched)

    experiments = {
        "dout_only": {
            "cross_sample": _cross_sample_aggregates(
                baseline, dout_only, args.batch
            ),
            "sample_zero_mapping": _sample_zero_mapping(
                baseline, dout_only, perturbed_index
            ),
        },
        "dpsum_only": {
            "cross_sample": _cross_sample_aggregates(
                baseline, dpsum_only, args.batch
            ),
            "sample_zero_mapping": _sample_zero_mapping(
                baseline, dpsum_only, perturbed_index
            ),
        },
        "dout_and_dpsum": {
            "cross_sample": _cross_sample_aggregates(
                baseline, combined, args.batch
            ),
            "sample_zero_mapping": _sample_zero_mapping(
                baseline, combined, perturbed_index
            ),
        },
    }
    stats_numel = state.direct_dpsum.numel()
    document = {
        "schema": "fp4_fa4_causal_exact_backward_isolation_only_v2",
        "control": control_identity,
        "shape": {"batch": args.batch, "sequence": 4096, "head_dim": 64},
        "protocol": (
            "six unperturbed captures, then dO-only, dPsum-only, and combined "
            "last-sample sign flips; no B1 or BF16 controller is compiled or "
            "launched"
        ),
        "unperturbed_repeats": unperturbed_repeats,
        "experiments": experiments,
        "workspace_statistics": {
            "workspace_base_data_ptr": int(batched.workspace_torch.data_ptr()),
            "dpsum_data_ptr": int(workspace_dpsum.data_ptr()),
            "scaled_lse_data_ptr": int(workspace_lse.data_ptr()),
            "scaled_lse_offset_bytes": stats_numel * 4,
            "expected_scaled_lse_offset_bytes": (
                args.batch * 32 * 4096 * 4
            ),
            "lse0_before_vs_after_dpsum_publication_byte_equal": bool(
                torch.equal(
                    lse0_before_publication,
                    lse0_after_dpsum_publication,
                )
            ),
            "lse0_after_dpsum_publication_matches_source_byte_equal": bool(
                torch.equal(
                    lse0_after_dpsum_publication,
                    state.direct_lse_log2[0],
                )
            ),
            "last_dpsum_after_publication_matches_source_byte_equal": bool(
                torch.equal(
                    workspace_dpsum[perturbed_index],
                    state.direct_dpsum[perturbed_index],
                )
            ),
        },
        "device": {
            "name": torch.cuda.get_device_name(0),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
    }
    rendered = json.dumps(document, indent=2, sort_keys=True)
    print(rendered)
    args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
