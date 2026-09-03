#!/usr/bin/env python3
"""Validate fail-closed v508 on immutable step-5000 layer captures.

This create-only diagnostic runs no optimizer/model update and touches no job.
It compares decoded v508 outputs with a chunked explicit native-score plus
represented-E4-gradient reference, and with the production-v501 outputs saved
in the same authenticated boundary captures.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import torch


RELEASE_ROOT = Path(__file__).resolve().parents[2]
V508_MODULE_NAME = "_C_sm100_gqa_tk_v508_d128_nvfp4_score_e4m3_gradient_b1_s4096"
EXPECTED_CAPTURE_HASHES = {
    "layer-10-exact-boundary.pt": (
        "546684335fabe807d924b484a7e39dc7ab91e9a53ca7077039faff1cfbfb203c"
    ),
    "layer-22-exact-boundary.pt": (
        "dea1357f2206f4a0051053d281f1fdfc88de4007c70fd9befd4721a798b7d1b8"
    ),
}
EXPECTED_SOURCE_HASHES = {
    "tk_fa4/lowp_fa4_bwd/projection_quantization_reference.py": (
        "69be3ca3fd26c11fe28ad8b2721b1d68bcdefd7bd9c1e2b67aa6bd01fd6e11fa"
    ),
    (
        "tk_fa4/native_gqa_tk_bwd/"
        "v508_d128_gqa_nvfp4_score_e4m3_gradient_b1_exact_s4096_"
        "experimental_bshd.cu"
    ): "809cad1db2f0cc053ba736372b854abcafa102fc65ed86512803c448feae3ba4",
    (
        "tk_fa4/native_gqa_tk_bwd/"
        "v508_d128_gqa_nvfp4_score_e4m3_gradient_b1_exact_s4096_"
        "experimental_bshd.cuh"
    ): "209e524ada3f45e0155bc36c7fcee0c60627a18f30c0d804250ff5b0056b64e6",
    "tk_fa4/native_gqa_tk_bwd/Makefile.v508": (
        "2b6272791f14e7b97e36a16926a4a6793d3da004e95753209d070d1fd42f53bc"
    ),
}
EXPECTED_BINARY_SHA256 = (
    "5c92ecd4588b193126c64defe29ba2fce52cc2a012d680ca0619e553c84b6fe9"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the authenticated v508 step-5000 diagnostic. The two "
            "historical capture tensors and exact historical extension are "
            "external inputs and must be named explicitly."
        )
    )
    parser.add_argument("output", type=Path, help="new create-only JSON receipt")
    parser.add_argument(
        "--capture-root",
        required=True,
        type=Path,
        help="directory containing the two authenticated layer capture files",
    )
    parser.add_argument(
        "--v508-binary",
        required=True,
        type=Path,
        help="exact historical v508 extension (.so); its SHA256 is verified",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=RELEASE_ROOT,
        help="release checkout containing tk_fa4 (default: inferred checkout root)",
    )
    return parser.parse_args()


def authenticated_inputs(
    *, repo_root: Path, capture_root: Path, v508_binary: Path
) -> dict[Path, str]:
    expected = {
        capture_root / name: digest for name, digest in EXPECTED_CAPTURE_HASHES.items()
    }
    expected.update(
        {repo_root / name: digest for name, digest in EXPECTED_SOURCE_HASHES.items()}
    )
    expected[v508_binary] = EXPECTED_BINARY_SHA256
    return expected


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def tensor_stats(tensor: torch.Tensor) -> dict[str, object]:
    values = tensor.float()
    return {
        "shape": list(values.shape),
        "numel": values.numel(),
        "finite": bool(torch.isfinite(values).all()),
        "nonzero": int(torch.count_nonzero(values)),
        "mean": float(values.mean()),
        "mean_abs": float(values.abs().mean()),
        "rms": float(values.square().mean().sqrt()),
        "max_abs": float(values.abs().max()),
    }


def comparison(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    actual_f = actual.float()
    reference_f = reference.float()
    difference = actual_f - reference_f
    tiny = torch.finfo(torch.float32).tiny
    actual_norm = torch.linalg.vector_norm(actual_f)
    reference_norm = torch.linalg.vector_norm(reference_f)
    return {
        "relative_l2_to_reference": float(
            torch.linalg.vector_norm(difference) / reference_norm.clamp_min(tiny)
        ),
        "cosine": float(
            (actual_f * reference_f).sum()
            / (actual_norm * reference_norm).clamp_min(tiny)
        ),
        "rms_ratio": float(
            actual_f.square().mean().sqrt()
            / reference_f.square().mean().sqrt().clamp_min(tiny)
        ),
        "max_abs_error": float(difference.abs().max()),
    }


@torch.no_grad()
def explicit_hybrid_reference(
    *,
    q_native: torch.Tensor,
    k_native: torch.Tensor,
    q_e4: torch.Tensor,
    k_e4: torch.Tensor,
    v_e4: torch.Tensor,
    dout_e4: torch.Tensor,
    forward_lse: torch.Tensor,
    dstat: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Chunked FP32 equations with native score and represented-E4 gradients."""
    batch, q_heads, sequence, depth = q_native.shape
    kv_heads = k_native.shape[1]
    if (batch, q_heads, kv_heads, sequence, depth) != (1, 32, 8, 4096, 128):
        raise RuntimeError("reference is exact B1/S4096/Hq32/Hkv8/D128 only")
    ratio = q_heads // kv_heads
    softmax_scale = depth**-0.5
    key_positions = torch.arange(sequence, device=q_native.device)
    dq = torch.empty((1, sequence, q_heads, depth), device=q_native.device)
    dk = torch.zeros((1, sequence, kv_heads, depth), device=q_native.device)
    dv = torch.zeros_like(dk)
    forward_dot = (-dstat.float() / 16.0).squeeze(2)
    mass_min = math.inf
    mass_max = -math.inf
    mass_sum = 0.0
    mass_count = 0

    for q_head in range(q_heads):
        kv_head = q_head // ratio
        kn = k_native[0, kv_head].float()
        ks = k_e4[0, :, kv_head].float()
        vs = v_e4[0, :, kv_head].float()
        for start in range(0, sequence, 128):
            stop = start + 128
            qn = q_native[0, q_head, start:stop].float()
            qs = q_e4[0, start:stop, q_head].float()
            scores = qn @ kn.T
            scores.mul_(softmax_scale)
            query_positions = torch.arange(start, stop, device=q_native.device)
            scores.masked_fill_(
                key_positions.unsqueeze(0) > query_positions.unsqueeze(1),
                float("-inf"),
            )
            probability = torch.exp(
                scores - forward_lse[0, q_head, 0, start:stop].float().unsqueeze(1)
            )
            mass = probability.sum(dim=1)
            mass_min = min(mass_min, float(mass.min()))
            mass_max = max(mass_max, float(mass.max()))
            mass_sum += float(mass.sum())
            mass_count += mass.numel()

            dout = dout_e4[0, start:stop, q_head].float()
            dp = dout @ vs.T
            ds = probability * (dp - forward_dot[0, q_head, start:stop].unsqueeze(1))
            ds.mul_(softmax_scale)
            dq[0, start:stop, q_head] = ds @ ks
            dk[0, :, kv_head].add_(ds.T @ qs)
            dv[0, :, kv_head].add_(probability.T @ dout)

    return {"dq": dq, "dk": dk, "dv": dv}, {
        "mean": mass_sum / mass_count,
        "min": mass_min,
        "max": mass_max,
    }


@torch.no_grad()
def run_v508(module, payload: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    q = payload["q_fp8"].cuda()
    k = payload["k_fp8"].cuda()
    v = payload["v_fp8"].cuda()
    dout = payload["dout_fp8"].cuda()
    lstat = payload["lstat"].cuda()
    dstat = payload["dstat"].cuda()
    q_native = payload["q_forward_payload_uint8"].cuda().view(torch.float4_e2m1fn_x2)
    k_native = payload["k_forward_payload_uint8"].cuda().view(torch.float4_e2m1fn_x2)
    q_scale = payload["q_forward_scale_pages_workspace"].cuda()
    k_scale = payload["k_forward_scale_pages_workspace"].cuda()
    q_global = payload["q_forward_global_scale_workspace"].cuda()
    k_global = payload["k_forward_global_scale_workspace"].cuda()
    outputs = {
        "dq": torch.empty(q.shape, device="cuda", dtype=torch.bfloat16),
        "dk": torch.empty(k.shape, device="cuda", dtype=torch.bfloat16),
        "dv": torch.empty(k.shape, device="cuda", dtype=torch.bfloat16),
    }
    module.backward_nvfp4_score_e4m3_gradient_bshd_precomputed_out(
        q,
        k,
        v,
        dout,
        lstat,
        dstat,
        outputs["dq"],
        outputs["dk"],
        outputs["dv"],
        q_native,
        k_native,
        q_scale,
        k_scale,
        q_global,
        k_global,
        128**-0.5,
    )
    torch.cuda.synchronize()
    return outputs


def layer_gate(
    actual: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
    production: dict[str, torch.Tensor],
    repeatability: dict[str, dict[str, float | bool]],
) -> tuple[dict[str, object], bool]:
    thresholds = {
        "dq": {"relative_l2": 0.10, "cosine": 0.995, "rms_lo": 0.94, "rms_hi": 1.06},
        "dk": {"relative_l2": 0.08, "cosine": 0.997, "rms_lo": 0.94, "rms_hi": 1.06},
        # dV consumes the kernel's rounded E4 probability publication while
        # the explicit oracle retains FP32 probability. Layer 22 establishes
        # a strict but format-aware 5% bound for that intentional difference.
        "dv": {"relative_l2": 0.05, "cosine": 0.999, "rms_lo": 0.95, "rms_hi": 1.05},
    }
    result: dict[str, object] = {}
    passed = True
    for name in ("dq", "dk", "dv"):
        stats = tensor_stats(actual[name])
        actual_cmp = comparison(actual[name], reference[name])
        production_cmp = comparison(production[name], reference[name])
        limit = thresholds[name]
        checks = {
            "finite": bool(stats["finite"]),
            "nontrivial": int(stats["nonzero"]) > 0 and float(stats["rms"]) > 0.0,
            # B1 dQ is additive across key-tile owners. Atomic/TMA arrival
            # order can change a handful of terminal BF16 roundings, so gate
            # numerical repeatability while retaining bitwise status below.
            "numerically_repeatable": (
                # At most 72 of 16,777,216 dQ elements varied in observed
                # repeats, by one terminal BF16 quantum; a 5e-5 relative bound
                # is strict while correctly classifying that additive order.
                float(repeatability[name]["relative_l2_between_runs"]) <= 5.0e-5
                and float(repeatability[name]["cosine_between_runs"]) >= 0.999999
            ),
            "relative_l2": actual_cmp["relative_l2_to_reference"]
            <= limit["relative_l2"],
            "cosine": actual_cmp["cosine"] >= limit["cosine"],
            "rms_ratio": limit["rms_lo"] <= actual_cmp["rms_ratio"] <= limit["rms_hi"],
            "improves_over_captured_v501": (
                actual_cmp["relative_l2_to_reference"]
                < production_cmp["relative_l2_to_reference"]
            ),
        }
        entry_passed = all(checks.values())
        passed = passed and entry_passed
        result[name] = {
            "v508_decoded_x0p25_stats": stats,
            "v508_vs_explicit_hybrid": actual_cmp,
            "captured_v501_decoded_x0p25_vs_explicit_hybrid": production_cmp,
            "relative_l2_improvement_factor_vs_v501": (
                production_cmp["relative_l2_to_reference"]
                / max(
                    actual_cmp["relative_l2_to_reference"],
                    torch.finfo(torch.float32).tiny,
                )
            ),
            "thresholds": limit,
            "repeatability": repeatability[name],
            "checks": checks,
            "passed": entry_passed,
        }
    return result, passed


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    capture_root = args.capture_root.expanduser().resolve()
    repo_root = args.repo_root.expanduser().resolve()
    v508_binary = args.v508_binary.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"output parent does not exist: {output.parent}")
    expected_inputs = authenticated_inputs(
        repo_root=repo_root,
        capture_root=capture_root,
        v508_binary=v508_binary,
    )
    for path, expected in expected_inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f"authenticated input is absent: {path}")
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(f"authenticated input changed: {path}: {observed}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("select exactly one idle local GPU")

    self_path = Path(__file__).resolve()
    self_sha_before = sha256(self_path)
    reference_decoder = load_module(
        repo_root / "tk_fa4/lowp_fa4_bwd/projection_quantization_reference.py",
        "fa4_projection_quantization_reference_v508_gate",
    )
    v508 = load_module(v508_binary, V508_MODULE_NAME)
    torch.backends.cuda.matmul.allow_tf32 = False
    receipt: dict[str, object] = {
        "schema": "fa4_v508_native_score_e4m3_gradient_nonzero_gate_v3",
        "scope": {
            "checkpoint_step": 5000,
            "batch": "exact fixed c4_test batch",
            "layers": [10, 22],
            "shape": "B1/S4096/Hq32/Hkv8/D128 causal",
            "kernel": "v508 native-NVFP4 score + represented-E4 Q/K gradients",
            "output_decode": "multiply BF16 dQ/dK/dV by 0.25",
            "reference": "chunked FP32 native-score/E4-gradient equations",
            "mutation": "create-only receipt; no model/optimizer/checkpoint/job mutation",
        },
        "inputs": {
            str(path): {"sha256": expected, "bytes": path.stat().st_size}
            for path, expected in expected_inputs.items()
        },
        "validator": {"path": str(self_path), "sha256": self_sha_before},
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "kernel_metadata": dict(v508.native_tk_d128_backward_metadata()),
        "layers": {},
    }
    all_passed = True

    for layer in (10, 22):
        payload = torch.load(
            capture_root / f"layer-{layer:02d}-exact-boundary.pt",
            map_location="cpu",
            weights_only=True,
        )
        first_encoded = run_v508(v508, payload)
        second_encoded = run_v508(v508, payload)
        repeatability = {}
        for name in ("dq", "dk", "dv"):
            first_decoded = first_encoded[name].float().mul(0.25)
            second_decoded = second_encoded[name].float().mul(0.25)
            repeat_cmp = comparison(first_decoded, second_decoded)
            repeatability[name] = {
                "bitwise_equal": bool(
                    torch.equal(first_encoded[name], second_encoded[name])
                ),
                "different_elements": int(
                    torch.count_nonzero(first_encoded[name] != second_encoded[name])
                ),
                "relative_l2_between_runs": repeat_cmp["relative_l2_to_reference"],
                "cosine_between_runs": repeat_cmp["cosine"],
                "rms_ratio_between_runs": repeat_cmp["rms_ratio"],
                "max_abs_difference": repeat_cmp["max_abs_error"],
            }
        actual = {
            name: first_encoded[name].float().mul(0.25) for name in ("dq", "dk", "dv")
        }
        production = {
            name: payload[f"{name}_v501"].cuda().float().mul(0.25)
            for name in ("dq", "dk", "dv")
        }
        q_native = reference_decoder.decode_native_nvfp4_qk(
            payload["q_forward_payload_uint8"].cuda(),
            payload["q_forward_scale_pages_workspace"].cuda(),
            payload["q_forward_global_scale_workspace"].cuda(),
            scale_tile_rows=128,
        ).float()
        k_native = reference_decoder.decode_native_nvfp4_qk(
            payload["k_forward_payload_uint8"].cuda(),
            payload["k_forward_scale_pages_workspace"][:, 0::2].cuda(),
            payload["k_forward_global_scale_workspace"].cuda(),
            scale_tile_rows=128,
        ).float()
        q_e4 = payload["q_fp8"].cuda().float().mul_(0.25)
        k_e4 = payload["k_fp8"].cuda().float().mul_(0.25)
        v_e4 = payload["v_fp8"].cuda().float().mul_(0.25)
        dout_e4 = payload["dout_fp8"].cuda().float().mul_(0.25)
        explicit, probability_mass = explicit_hybrid_reference(
            q_native=q_native,
            k_native=k_native,
            q_e4=q_e4,
            k_e4=k_e4,
            v_e4=v_e4,
            dout_e4=dout_e4,
            forward_lse=payload["forward_lse"].cuda(),
            dstat=payload["dstat"].cuda(),
        )
        gate, passed = layer_gate(actual, explicit, production, repeatability)
        all_passed = all_passed and passed
        receipt["layers"][str(layer)] = {
            "probability_row_mass": probability_mass,
            "explicit_reference_stats": {
                name: tensor_stats(explicit[name]) for name in ("dq", "dk", "dv")
            },
            "outputs": gate,
            "passed": passed,
        }
        del (
            payload,
            first_encoded,
            second_encoded,
            actual,
            production,
            q_native,
            k_native,
            q_e4,
            k_e4,
            v_e4,
            dout_e4,
            explicit,
        )
        torch.cuda.empty_cache()

    receipt["passed"] = all_passed
    if sha256(self_path) != self_sha_before:
        raise RuntimeError("validator changed while running")
    serialized = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(serialized)
    print(output)
    if not all_passed:
        raise SystemExit("v508 nonzero gate failed; inspect create-only receipt")


if __name__ == "__main__":
    main()
