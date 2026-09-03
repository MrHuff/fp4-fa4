#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


STOCK_DEGREE2 = (0.6657850742340088, 0.33010703325271606)
E2M1_THRESHOLDS = np.asarray(
    [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
    dtype=np.float64,
)
E2M1_LOG_THRESHOLDS = np.log2(E2M1_THRESHOLDS)


def _float_from_bits(bits: int) -> float:
    return float(np.asarray([bits], dtype=np.uint32).view(np.float32)[0])


DEGREE3 = (
    _float_from_bits(0x3F31F519),
    _float_from_bits(0x3E6906A4),
    _float_from_bits(0x3D9DF09D),
)


def _representative_inputs(seed: int, count: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    positions = np.arange(count)
    typical = rng.uniform(-32.0, np.log2(6.0), count)
    tail = rng.uniform(-127.0, -32.0, count)
    return np.where((positions % 8) == 0, tail, typical).astype(np.float32)


def _degree2_values(inputs: np.ndarray, coefficients: tuple[float, float]) -> np.ndarray:
    floor_value = np.floor(inputs).astype(np.int32)
    fraction = inputs.astype(np.float64) - floor_value
    c1, c2 = coefficients
    polynomial = 1.0 + c1 * fraction + c2 * fraction * fraction
    return np.ldexp(polynomial, floor_value)


def _degree3_values(inputs: np.ndarray) -> np.ndarray:
    floor_value = np.floor(inputs).astype(np.int32)
    fraction = inputs.astype(np.float64) - floor_value
    c1, c2, c3 = DEGREE3
    polynomial = 1.0 + fraction * (c1 + fraction * (c2 + fraction * c3))
    return np.ldexp(polynomial, floor_value)


def _metrics(
    inputs: np.ndarray,
    values: np.ndarray,
    row_width: int,
) -> dict[str, float | int]:
    native = np.exp2(inputs.astype(np.float64))
    relative = values / native - 1.0
    native_codes = np.searchsorted(E2M1_THRESHOLDS, native, side="right")
    candidate_codes = np.searchsorted(E2M1_THRESHOLDS, values, side="right")
    mismatch = candidate_codes != native_codes
    threshold_distance = np.min(
        np.abs(inputs.astype(np.float64)[:, None] - E2M1_LOG_THRESHOLDS[None, :]),
        axis=1,
    )
    near_threshold = threshold_distance < 0.01
    rows = inputs.size // row_width
    native_rows = native[: rows * row_width].reshape(rows, row_width).sum(axis=1)
    candidate_rows = values[: rows * row_width].reshape(rows, row_width).sum(axis=1)
    row_relative = candidate_rows / native_rows - 1.0
    return {
        "scalars": int(inputs.size),
        "relative_exp_mean": float(relative.mean()),
        "relative_exp_rms": float(np.sqrt(np.mean(relative * relative))),
        "relative_exp_max_abs": float(np.max(np.abs(relative))),
        "payload_mismatch_rate": float(mismatch.mean()),
        "near_threshold_scalars": int(near_threshold.sum()),
        "near_threshold_mismatch_rate": float(mismatch[near_threshold].mean()),
        "row_sum_relative_mean": float(row_relative.mean()),
        "row_sum_relative_rms": float(np.sqrt(np.mean(row_relative * row_relative))),
        "row_sum_relative_max_abs": float(np.max(np.abs(row_relative))),
    }


def _fit_coefficients(inputs: np.ndarray) -> tuple[tuple[float, float], dict[str, float]]:
    floor_value = np.floor(inputs).astype(np.int32)
    fraction = inputs.astype(np.float64) - floor_value
    native = np.exp2(inputs.astype(np.float64))
    inverse_fraction_exp = np.exp2(-fraction)
    mean0 = np.mean(inverse_fraction_exp - 1.0)
    mean1 = np.mean(fraction * inverse_fraction_exp)
    mean2 = np.mean(fraction * fraction * inverse_fraction_exp)
    native_codes = np.searchsorted(E2M1_THRESHOLDS, native, side="right")
    threshold_distance = np.min(
        np.abs(inputs.astype(np.float64)[:, None] - E2M1_LOG_THRESHOLDS[None, :]),
        axis=1,
    )
    near_threshold = threshold_distance < 0.01

    candidates: dict[tuple[float, float], float] = {}
    for c1_raw in np.linspace(0.6655, 0.6665, 401):
        c2_raw = -(mean0 + c1_raw * mean1) / mean2
        c1 = float(np.float32(c1_raw))
        c2 = float(np.float32(c2_raw))
        polynomial = 1.0 + c1 * fraction + c2 * fraction * fraction
        values = np.ldexp(polynomial, floor_value)
        relative = values / native - 1.0
        mismatch = (
            np.searchsorted(E2M1_THRESHOLDS, values, side="right") != native_codes
        )
        objective = (
            np.sqrt(np.mean(relative * relative))
            + 2.0 * abs(relative.mean())
            + 0.5 * mismatch.mean()
            + 0.001 * mismatch[near_threshold].mean()
            + 0.1 * np.max(np.abs(relative))
        )
        candidates[(c1, c2)] = min(candidates.get((c1, c2), np.inf), objective)

    coefficients, objective = min(candidates.items(), key=lambda item: item[1])
    return coefficients, {"weighted_objective": float(objective)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--fit-scalars", type=int, default=500_000)
    parser.add_argument("--validation-rows", type=int, default=128)
    args = parser.parse_args()

    fit_inputs = _representative_inputs(94601, args.fit_scalars)
    tuned, fit_summary = _fit_coefficients(fit_inputs)
    validation: dict[str, dict[str, dict[str, float | int]]] = {}
    for seed in (127001, 20260711):
        for row_width in (1024, 2048, 4096):
            inputs = _representative_inputs(seed, args.validation_rows * row_width)
            key = f"seed{seed}_w{row_width}"
            validation[key] = {
                "stock_degree2": _metrics(
                    inputs,
                    _degree2_values(inputs, STOCK_DEGREE2),
                    row_width,
                ),
                "tuned_degree2": _metrics(
                    inputs,
                    _degree2_values(inputs, tuned),
                    row_width,
                ),
                "retained_degree3": _metrics(
                    inputs,
                    _degree3_values(inputs),
                    row_width,
                ),
            }

    tuned_dominates = all(
        record["tuned_degree2"][metric] <= record["stock_degree2"][metric]
        for record in validation.values()
        for metric in (
            "relative_exp_max_abs",
            "payload_mismatch_rate",
            "near_threshold_mismatch_rate",
            "row_sum_relative_rms",
            "row_sum_relative_max_abs",
        )
    )
    result = {
        "fit_seed": 94601,
        "fit_scalars": args.fit_scalars,
        "fit_summary": fit_summary,
        "stock_degree2": {
            "c0": 1.0,
            "c1": STOCK_DEGREE2[0],
            "c2": STOCK_DEGREE2[1],
        },
        "tuned_degree2": {
            "c0": 1.0,
            "c1": tuned[0],
            "c2": tuned[1],
            "c1_bits": hex(np.float32(tuned[0]).view(np.uint32).item()),
            "c2_bits": hex(np.float32(tuned[1]).view(np.uint32).item()),
        },
        "retained_degree3": {
            "c0": 1.0,
            "c1": DEGREE3[0],
            "c2": DEGREE3[1],
            "c3": DEGREE3[2],
        },
        "validation": validation,
        "tuned_dominates_stock_degree2": tuned_dominates,
        "decision": "reject" if not tuned_dominates else "kernel_validation_required",
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
