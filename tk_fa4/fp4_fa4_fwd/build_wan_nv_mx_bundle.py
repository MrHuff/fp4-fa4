#!/usr/bin/env python3
"""Build the non-stable NV/MX policy bundle used by the Wan evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


HERE = Path(__file__).resolve().parent
MAKEFILE = HERE / "Makefile.hao_direct_fp4pv"
MODEL_CONFIGS = {
    "1.3b": {
        "model": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "heads": 12,
        "guard_layers": "27-29",
        "anchor_margin_log2": 110,
        "stored_scale_shift_log2": 16,
        "fast_affine_overrides": [
            {
                "layers": "0",
                "a": 1.625,
                "b": 0.95,
                "tag": "a1625_b095",
            },
            {
                "layers": "11",
                "a": 1.575,
                "b": 1.05,
                "tag": "a1575_b105",
            },
        ],
    },
    "14b": {
        "model": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        "heads": 40,
        "guard_layers": "33-34,38-39",
        "anchor_margin_log2": 112,
        "stored_scale_shift_log2": 14,
        "fast_affine_overrides": [
            {
                "layers": "1,3,6,8-12,15-17,22-27,30-31,35",
                "a": 1.575,
                "b": 1.05,
                "tag": "a1575_b105",
            },
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_CONFIGS, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--policies",
        default="fast,accurate",
        help="Comma-separated base policies to build (fast, accurate).",
    )
    parser.add_argument("--gpu", default="B200")
    parser.add_argument("--num-sm", type=int, default=152)
    parser.add_argument(
        "--anchor-margin-log2",
        type=int,
        default=None,
        help=(
            "Binary-exponent safety margin added to the sampled QK anchor "
            "(0-126). By default use the largest model-specific value that "
            "leaves at least E8M0 code 1 after the stored-scale shift."
        ),
    )
    parser.add_argument(
        "--stored-scale-shift-log2",
        type=int,
        default=None,
        help=(
            "Common binary shift subtracted from stored P scales "
            "(0-120). By default use the calibrated model-specific value "
            "(16 for 1.3B, 14 for 14B)."
        ),
    )
    parser.add_argument(
        "--fast-affine-a",
        type=float,
        default=1.60,
        help="Wan-calibrated slope for the fast affine E2M1 code map.",
    )
    parser.add_argument(
        "--fast-affine-b",
        type=float,
        default=0.95,
        help="Wan-calibrated intercept for the fast affine E2M1 code map.",
    )
    parser.add_argument(
        "--enable-layer-affine-overrides",
        action="store_true",
        help=(
            "Include the historical layer-local affine map. Disabled by "
            "default because its end-to-end effect is below repeat variance."
        ),
    )
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def build(
    output: Path,
    module: str,
    heads: int,
    policy: str,
    extra: list[str],
    args: argparse.Namespace,
) -> None:
    if args.skip_existing and output.exists():
        return
    symbol_tag = re.sub(r"[^A-Za-z0-9_]", "_", module)
    if not symbol_tag or symbol_tag[0].isdigit():
        symbol_tag = f"wan_{symbol_tag}"
    command = [
        "make",
        "-B",
        "-f",
        str(MAKEFILE),
        "-j1",
        f"GPU={args.gpu}",
        f"OUT={output}",
        f"MODULE={module}",
        f"HAO_KERNEL_SYMBOL_TAG={symbol_tag}",
        "HAO_BATCH=1",
        "HAO_SEQ_LEN=7680",
        f"HAO_HEADS={heads}",
        "HAO_HEAD_DIM=128",
        f"HAO_NUM_SM={args.num_sm}",
        "HAO_QK_SCALE_MODE=0",
        "HAO_PV_SCALE_MODE=1",
        f"HAO_FP4PV_MX_POLICY={policy}",
        "HAO_EXTENSION_SYMBOLIC_BIND=1",
        "NVCC_SPLIT_COMPILE=1",
        *extra,
    ]
    subprocess.run(command, cwd=HERE, check=True)


def extension_record(
    path: Path,
    module: str,
    output_dir: Path,
) -> dict[str, str | int]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(output_dir.resolve())
    except ValueError as error:
        raise ValueError(f"extension is outside its policy bundle: {resolved}") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"built extension is missing: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return {
        "path": relative.as_posix(),
        "module": module,
        "bytes": resolved.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def main() -> None:
    args = parse_args()
    config = MODEL_CONFIGS[args.model]
    policies = [item.strip() for item in args.policies.split(",") if item.strip()]
    unknown = set(policies) - {"fast", "accurate"}
    if not policies or unknown:
        raise ValueError(f"unsupported policies: {sorted(unknown)}")
    if (
        args.anchor_margin_log2 is not None
        and not 0 <= args.anchor_margin_log2 <= 126
    ):
        raise ValueError("--anchor-margin-log2 must be between 0 and 126")
    if (
        args.stored_scale_shift_log2 is not None
        and not 0 <= args.stored_scale_shift_log2 <= 120
    ):
        raise ValueError(
            "--stored-scale-shift-log2 must be between 0 and 120"
        )

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else Path(f"/tmp/wan_nv_mx_{args.model.replace('.', 'p')}_s7680")
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    heads = int(config["heads"])

    scale_shift = int(
        config["stored_scale_shift_log2"]
        if args.stored_scale_shift_log2 is None
        else args.stored_scale_shift_log2
    )
    margin = int(
        config["anchor_margin_log2"]
        if args.anchor_margin_log2 is None
        else args.anchor_margin_log2
    )
    if margin + scale_shift > 126:
        raise ValueError(
            "global-anchor margin plus stored-scale shift must be <= 126 "
            f"to preserve E8M0 code 1; got {margin}+{scale_shift}"
        )
    guard_module = (
        f"_C_tk_hao_direct_fp4pv_anchor128_"
        f"m{margin}_s{scale_shift}_h{heads}"
    )
    guard_path = output_dir / (
        f"nvmx-anchor128-m{margin}-s{scale_shift}.so"
    )
    build(
        guard_path,
        guard_module,
        heads,
        "balanced",
        [
            "HAO_FP4PV_MX_GLOBAL_ANCHOR32_OVERRIDE=0",
            "HAO_FP4PV_MX_GLOBAL_ANCHOR128_OVERRIDE=1",
            "HAO_FP4PV_MX_MODE23_NATIVE_DENSITY_OVERRIDE=1",
            f"HAO_FP4PV_MX_GLOBAL_ANCHOR_MARGIN_LOG2_OVERRIDE={margin}",
            "HAO_FP4PV_MX_STORED_SCALE_SHIFT_LOG2_OVERRIDE="
            f"{scale_shift}",
        ],
        args,
    )

    manifest_policies: dict[str, object] = {}
    for policy in policies:
        module = f"_C_tk_hao_direct_fp4pv_{policy}_h{heads}"
        path = output_dir / f"nvmx-{policy}.so"
        policy_overrides = []
        if policy == "fast":
            policy_overrides = [
                "HAO_FP4PV_MX_AFFINE_A_OVERRIDE="
                f"{args.fast_affine_a:.9g}f",
                "HAO_FP4PV_MX_AFFINE_B_OVERRIDE="
                f"{args.fast_affine_b:.9g}f",
            ]
        build(path, module, heads, policy, policy_overrides, args)
        layer_extensions = []
        if policy == "fast" and args.enable_layer_affine_overrides:
            for affine in config["fast_affine_overrides"]:
                affine_module = (
                    f"_C_tk_hao_direct_fp4pv_fast_{affine['tag']}_h{heads}"
                )
                affine_path = output_dir / f"nvmx-fast-{affine['tag']}.so"
                build(
                    affine_path,
                    affine_module,
                    heads,
                    policy,
                    [
                        "HAO_FP4PV_MX_AFFINE_A_OVERRIDE="
                        f"{affine['a']:.9g}f",
                        "HAO_FP4PV_MX_AFFINE_B_OVERRIDE="
                        f"{affine['b']:.9g}f",
                    ],
                    args,
                )
                layer_extensions.append(
                    {
                        "layers": affine["layers"],
                        **extension_record(
                            affine_path,
                            affine_module,
                            output_dir,
                        ),
                        "purpose": "model-calibrated affine E2M1 boundaries",
                    }
                )
        layer_extensions.append(
            {
                "layers": config["guard_layers"],
                **extension_record(guard_path, guard_module, output_dir),
                "purpose": "wide global QK probe for extreme-logit layers",
            }
        )
        manifest_policies[policy] = {
            "base": extension_record(path, module, output_dir),
            "layer_extensions": layer_extensions,
        }

    manifest = {
        "schema": "tk_wan_nv_mx_policy_bundle_v2",
        "model": config["model"],
        "shape": {"batch": 1, "seqlen": 7680, "heads": heads, "dim": 128},
        "formats": {"qk": "nvfp4", "pv": "mxfp4"},
        "softmax": "shiftless/sampled; no stable-softmax fallback",
        "guard_layers": config["guard_layers"],
        "qk_guard": {
            "samples": 128,
            "margin_log2": margin,
            "stored_scale_shift_log2": scale_shift,
        },
        "fast_affine_code_map": {
            "base": {"a": args.fast_affine_a, "b": args.fast_affine_b},
            "layer_overrides": (
                config["fast_affine_overrides"]
                if args.enable_layer_affine_overrides
                else []
            ),
            "calibration": (
                "historical layer-local candidates enabled explicitly"
                if args.enable_layer_affine_overrides
                else "global map retained; layer-local effect below repeat variance"
            ),
        },
        "affine_calibration": {
            "enabled": args.enable_layer_affine_overrides,
            "status": (
                "experimental"
                if args.enable_layer_affine_overrides
                else "rejected as non-robust"
            ),
        },
        "policies": manifest_policies,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(manifest_path)


if __name__ == "__main__":
    main()
