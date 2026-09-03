#!/usr/bin/env python3
"""Authenticate a fixed D64 causal NVFP4-QK/MXFP4-PV artifact.

The artifact bytes are checked before Python loads the extension.  Topology
checks are variant-specific so an anchor32 kernel cannot silently stand in for
the historical unanchored splitmix-v6 experiment (or vice versa).  Reading the
compile-time topology does not launch the CUDA kernel.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any


COMMON_D64_CAUSAL_MX_TOPOLOGY = {
    "seqlen": 4096,
    "heads": 32,
    "kv_heads": 8,
    "dqk": 64,
    "dvo": 64,
    "causal": True,
    "causal_interleaved_kv": True,
    "qk_format": "nvfp4_e4m3_block16",
    "pv_format": "mxfp4_e8m0_block32",
    "route": "real_fwd_tk_hao_direct_nvfp4_mxfp4pv",
    "schema": "tk_hao_direct_pipeline_v1",
    "fixed_route_fastpath": True,
    "route_env_guard_per_launch": False,
    "fixed_p_ceiling": False,
    "score_pack_ceiling": False,
    "rowmax_pack_ceiling": False,
    "d64_detached_p": True,
    "compiled_num_sm": 152,
    "persistent": True,
    "physical_grid_ctas": 152,
    "threads_per_cta": 512,
    "query_tile": 128,
    "key_tile": 128,
    "mx_pwl_exp2": True,
    "mx_pwl_exp2_mode": 23,
    "mx_mode23_native_density": 4,
    "mx_mode23_native_quarter_mask": 3,
    "mx_mode23_native_stage_mask": 3,
    "mx_mode23_self_stage0_native": True,
    "mx_quantized_denom": True,
    "mx_denom_decode_mode": 1,
    "mx_pair_scale_reuse": 1,
    "mx_pair_scale_stage_mask": 1,
    "mx_q1_self_max": 3,
    "mx_shiftless_softmax": True,
    "mx_stored_scale_shift_log2": 16,
    "mx_stage0_affine_mask": 0,
    "mx_stage1_affine_mask": 0,
    "mx_skip_zero_scale_mask": True,
    "mx_tree_max": True,
    "mx_max3_reduce": True,
    "mx_max3_wide_reduce": True,
    "mx_local_denom_pipeline": 2,
}

VARIANT_TOPOLOGY = {
    "anchored": {
        "mx_global_anchor32": True,
        "mx_global_anchor128": False,
        "mx_global_anchor_margin_log2": 64,
        "mx_anchor_affine_hoist": True,
        "mx_async_scale_handoff": False,
    },
    "unanchored-splitmix-v6": {
        "mx_global_anchor32": False,
        "mx_global_anchor128": False,
        "mx_global_anchor_bias": 0.0,
        "mx_global_anchor_margin_log2": 0,
        "mx_anchor_affine_hoist": False,
        # Retain the current explicit TMEM publication fence.  The historical
        # v6 binary predated that fix and reported an asynchronous handoff;
        # this rebuild intentionally changes only the anchor policy.
        "mx_async_scale_handoff": False,
    },
}

VARIANT_BATCHES = {
    "anchored": (2, 8, 16),
    "unanchored-splitmix-v6": (16,),
}

CANONICAL_ARTIFACTS = {
    ("unanchored-splitmix-v6", 16): {
        "module": (
            "_C_cfwd_mx_d4q01_unanchored_splitmix_v6_"
            "b16s4096h32kv8d64_sm100_20260825"
        ),
        "sha256": (
            "93488ece199812bbd001d9e1f662db79a"
            "c39ecc230d91e8f0de2c2e4321976d3"
        ),
        "bytes": 1_958_304,
    },
}


def authenticate_artifact(
    path: Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
) -> dict[str, int | str]:
    """Verify exact regular-file bytes before importing an extension."""
    if path.is_symlink():
        raise RuntimeError(f"artifact must be a non-symlink regular file: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise RuntimeError(f"artifact is not a regular file: {resolved}")
    byte_count = resolved.stat().st_size
    if byte_count != expected_bytes:
        raise RuntimeError(
            "artifact byte-count mismatch: "
            f"observed {byte_count}, expected {expected_bytes}"
        )
    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise RuntimeError(
            "artifact SHA-256 mismatch: "
            f"observed {digest}, expected {expected_sha256}"
        )
    return {
        "path": str(resolved),
        "sha256": digest,
        "bytes": byte_count,
    }


def require_topology(
    topology: dict[str, Any],
    *,
    variant: str,
    batch: int,
    runtime_populated: bool = False,
) -> None:
    """Fail closed unless every selected compile-time field is exact."""
    if variant not in VARIANT_TOPOLOGY:
        raise ValueError(f"unknown MXFP4 forward variant: {variant!r}")
    if batch not in VARIANT_BATCHES[variant]:
        raise ValueError(
            f"variant {variant!r} has no authenticated batch-{batch} artifact"
        )
    expected = {
        **COMMON_D64_CAUSAL_MX_TOPOLOGY,
        **VARIANT_TOPOLOGY[variant],
        "batch": batch,
        "logical_jobs": batch * 32 * (4096 // 256),
    }
    if not runtime_populated:
        # Grid, jobs, and thread count are populated by the first launch.  The
        # remaining fields must already be exact before any device work.
        expected.pop("physical_grid_ctas")
        expected.pop("threads_per_cta")
        expected.pop("logical_jobs")
    for key, value in expected.items():
        actual = topology.get(key)
        if actual != value:
            raise RuntimeError(
                f"{variant} topology {key}={actual!r}, expected {value!r}"
            )
    valid = topology.get("valid")
    expected_valid = 1 if runtime_populated else (0, 1)
    if runtime_populated:
        if valid != expected_valid:
            raise RuntimeError(
                f"{variant} runtime topology valid={valid!r}, expected 1"
            )
    elif valid not in expected_valid:
        raise RuntimeError(
            f"{variant} prelaunch topology valid={valid!r}, expected 0 or 1"
        )


def load_authenticated_extension(
    path: Path,
    *,
    module_name: str,
    expected_sha256: str,
    expected_bytes: int,
    variant: str,
    batch: int,
) -> tuple[ModuleType, dict[str, Any], dict[str, int | str]]:
    """Authenticate bytes, load the module, then check prelaunch topology."""
    identity = authenticate_artifact(
        path,
        expected_sha256=expected_sha256,
        expected_bytes=expected_bytes,
    )
    if not path.name.startswith(f"{module_name}."):
        raise RuntimeError(
            f"artifact basename {path.name!r} does not bind module "
            f"{module_name!r}"
        )
    # Import torch first so the extension's PyTorch symbols are globally
    # available.  This is a host-side load; no forward kernel is invoked.
    import torch  # noqa: F401

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load extension spec from {path}")
    extension = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extension)
    identity_after = authenticate_artifact(
        path,
        expected_sha256=expected_sha256,
        expected_bytes=expected_bytes,
    )
    if identity_after != identity:
        raise RuntimeError(f"artifact changed while loading: {path}")
    extension._tk_fa4_loaded_artifact_identity = identity
    topology = dict(extension.read_hao_direct_topology())
    require_topology(topology, variant=variant, batch=batch)
    return extension, topology, identity


def canonical_artifact_spec(variant: str, batch: int) -> dict[str, int | str]:
    """Return the immutable reviewed identity for one named artifact."""
    spec = CANONICAL_ARTIFACTS.get((variant, batch))
    if spec is None:
        raise ValueError(
            f"no canonical artifact is pinned for {variant!r} at batch {batch}"
        )
    return dict(spec)


def load_canonical_extension(
    path: Path,
    *,
    variant: str,
    batch: int,
) -> tuple[ModuleType, dict[str, Any], dict[str, int | str]]:
    """Load only the repository-pinned identity for a named variant."""
    artifact = canonical_artifact_spec(variant, batch)
    return load_authenticated_extension(
        path,
        module_name=str(artifact["module"]),
        expected_sha256=str(artifact["sha256"]),
        expected_bytes=int(artifact["bytes"]),
        variant=variant,
        batch=batch,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--variant", choices=tuple(VARIANT_TOPOLOGY), required=True)
    parser.add_argument("--batch", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    artifact = canonical_artifact_spec(args.variant, args.batch)
    _, topology, identity = load_canonical_extension(
        args.extension,
        variant=args.variant,
        batch=args.batch,
    )
    print(
        json.dumps(
            {
                "identity": identity,
                "module": artifact["module"],
                "variant": args.variant,
                "topology": topology,
                "status": "prelaunch_authenticated_no_kernel_executed",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
