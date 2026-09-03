#!/usr/bin/env python3
"""Run one memory-safe, saturated Llama-1.2B training route.

The historical three-route harness is useful for B1 diagnostics, but it keeps
three models resident.  This harness instead runs exactly one route per
process, uses MLCE's torch-compiled linear cross entropy, and records enough
seeded samples to compare independent route processes.  The installed MLCE
torch-compile backend still expresses the logical full logits matrix; it is not
the Triton Cut Cross Entropy implementation.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import random
import statistics
import string
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import cut_cross_entropy
from cut_cross_entropy import linear_cross_entropy
from tokenizers import Tokenizer

import tk_fa4.interface as tk_interface
from tk_fa4.lowp_fa4_bwd.authenticate_causal_gqa_mxfp4pv_forward import (
    load_authenticated_extension as load_authenticated_mx_extension,
    require_topology as require_mx_variant_topology,
)
from tk_fa4.lowp_fa4_bwd.backward_contract import (
    require_matching_backward_contracts,
)
from tk_fa4.lowp_fa4_bwd.benchmark_llama12b_e2e import (
    AUTHENTICATED_D128_EXACT_BATCHES,
    Config,
    D128_EXACT_FORWARD_TOPOLOGIES,
    D128_FORWARD_TOPOLOGY_VARIANTS,
    DIAGNOSTIC_FP8_LSE_SUBSTITUTION_MODES,
    DEFAULT_BF16_ATTENTION_CONTROL,
    DEFAULT_MODEL_PRESET,
    Llama12B,
    LowpAttentionRuntime,
    MODEL_PRESETS,
    _load_forward,
    _load_extension,
    _d128_forward_topology_recipe,
    _make_llama3_rope,
    _useful_flops,
    activate_model_forward_route,
    config_from_model_preset,
    packed_qkv_layout,
)
from tk_fa4.lowp_fa4_bwd.forward_route import require_active_forward_route
from tk_fa4.lowp_fa4_bwd.cutlass_dsl_toolchain import (
    configure_d128_mxfp4_v_compile_environment,
)
from tk_fa4.lowp_fa4_bwd.packed_bf16_qkv import (
    PackedQKVAttentionWeights,
    canonical_split_qkv_tensors,
    pack_qkv_state_dict,
    unpack_qkv_state_dict,
)
from tk_fa4.lowp_fa4_bwd.native_tk_d64_backward import (
    BACKEND as NATIVE_TK_D64_BACKEND,
)
from tk_fa4.lowp_fa4_bwd.native_tk_d128_backward import (
    BACKEND as NATIVE_TK_D128_BACKEND,
)
from tk_fa4.lowp_fa4_bwd.native_tk_d128_mxfp4_v_backward import (
    BACKEND as NATIVE_TK_D128_MX_BACKEND,
    SHARED_TILE_V503_BACKEND as NATIVE_TK_D128_SHARED_TILE_MX_BACKEND,
)


DEFAULT_FORWARDS = {
    "fp8": Path(
        "/tmp/_C_cfwd_fp8exact0_b16_s4096h32kv8d64_sm100_topofix_b200_20260825."
        "cpython-312-aarch64-linux-gnu.so"
    ),
    "mx": Path(
        "/tmp/_C_cfwd_mx_d4q01_i1_b16s4096h32kv8d64_20260825."
        "cpython-312-aarch64-linux-gnu.so"
    ),
    "mx_unanchored": Path(
        "/tmp/_C_cfwd_mx_d4q01_unanchored_splitmix_v6_"
        "b16s4096h32kv8d64_sm100_20260825."
        "cpython-312-aarch64-linux-gnu.so"
    ),
}
FORWARD_MODULES = {
    "fp8": (
        "_C_cfwd_fp8exact0_b16_s4096h32kv8d64_sm100_topofix_b200_20260825"
    ),
    "mx": "_C_cfwd_mx_d4q01_i1_b16s4096h32kv8d64_20260825",
    "mx_unanchored": (
        "_C_cfwd_mx_d4q01_unanchored_splitmix_v6_"
        "b16s4096h32kv8d64_sm100_20260825"
    ),
}
LOWP_ROUTES = ("fp8", "mx", "mx_unanchored")
MX_ROUTES = ("mx", "mx_unanchored")
BF16_ROUTES = ("bf16", "bf16_packed")
EXPERIMENTAL_NATIVE_NVFP4_ROUTES = ("fp8", "mx")
DEFAULT_PROJECTION = Path(
    "/tmp/fa4-dolma3-d64-assets.QZwFvk/assets/"
    "_C_b300_lowp_bwd.cpython-312-aarch64-linux-gnu.so"
)
DEFAULT_CONTROL = Path(
    "/tmp/fa4-dolma3-d64-assets.QZwFvk/assets/"
    "fmha_bwd_d64_gqa_aug19_exact.py"
)
DEFAULT_CORPUS = Path(
    "/tmp/fa4_8b_e2e_f3db4bf_20260822/data/"
    "dolma3-longmino-len-8-16k-first512.jsonl"
)
DEFAULT_TOKENIZER = Path(
    "/tmp/fa4-dolma3-d64-assets.QZwFvk/assets/tokenizer.json"
)
PINNED_ARTIFACTS = {
    "forward": {
        "fp8": (
            "88d81d3783e5aa80f0e9cf259a2ea7c935da4c2a5dc3ba1868e63f802a2c6208",
            1_817_256,
        ),
        "mx": (
            "cc06fe4337fdc3a7c900f81d68fabc4a8e0c375ea536fbe6405754237a393717",
            1_958_000,
        ),
        "mx_unanchored": (
            "93488ece199812bbd001d9e1f662db79ac39ecc230d91e8f0de2c2e4321976d3",
            1_958_304,
        ),
    },
    "projection": (
        "bfdec1e43a0a19acec5afbac3fa837e2f4d1b25be80ae7fb5ff3b5bc5e9e25ce",
        17_504_688,
    ),
    "control": (
        "cd57e3360082abe4bad7560c51a7793a4e9bfd4d16efc1259b92ce20238b99e1",
        220_876,
    ),
}
PINNED_DATA = {
    "corpus": "860b33924dffd53f4c20b80abbcee96e1bf09c3c313290c15ea3a6ee418269ce",
    "tokenizer": "76e48799b099d43365bd24ccd8ecc5aedac831718da780552f03b0a6eb4412aa",
}
SAMPLED_PARAMETER_NAMES = (
    "embedding",
    "layers.0.attention.weights.q",
    "layers.0.attention.weights.k",
    "layers.0.attention.weights.v",
    "layers.0.attention.weights.o",
    "layers.0.mlp.down",
    "layers.15.attention.weights.q",
)
HIDDEN_SAMPLE_BATCHES = (0, 8, 15)
HIDDEN_SAMPLE_POSITIONS = (0, 15, 63, 255, 1023, 2047, 3071, 4095)
SAMPLE_ELEMENTS_PER_PARAMETER = 8192
MINIMUM_MEASURED_UPDATES = 20
D128_PROJECTION_ABI_SYMBOL = (
    "project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered"
)


def _require_saturated_shape(config: Config) -> None:
    """Admit only full-depth shapes authenticated by this harness."""
    expected = {
        DEFAULT_MODEL_PRESET: {
            "layers": 16,
            "hidden": 2048,
            "head_dim": 64,
        },
        "llama3.1-8b": {
            "layers": 32,
            "hidden": 4096,
            "head_dim": 128,
        },
    }[config.model_preset]
    observed = {
        "layers": config.layers,
        "hidden": config.hidden,
        "head_dim": config.head_dim,
    }
    expected_batches = (
        (16,)
        if config.model_preset == DEFAULT_MODEL_PRESET
        else (1, *AUTHENTICATED_D128_EXACT_BATCHES)
    )
    if (
        observed != expected
        or config.batch not in expected_batches
        or config.sequence != 4096
    ):
        raise ValueError(
            "saturated benchmark requires exactly llama3.2-1b/B16/L16/D64 "
            "or llama3.1-8b/B1|B2/L32/D128 at sequence 4096; "
            f"observed {config.model_preset}: {observed}, "
            f"batch={config.batch}, sequence={config.sequence}"
        )


def _diagnostic_sample_layout(
    config: Config,
) -> tuple[tuple[str, ...], tuple[int, ...], tuple[int, ...]]:
    """Return route-independent, in-bounds samples for either model shape."""
    parameter_names = (
        *SAMPLED_PARAMETER_NAMES[:-1],
        f"layers.{config.layers - 1}.attention.weights.q",
    )
    batch_indices = tuple(
        dict.fromkeys((0, config.batch // 2, config.batch - 1))
    )
    positions = tuple(
        position
        for position in HIDDEN_SAMPLE_POSITIONS
        if position < config.sequence
    )
    if not positions:
        raise ValueError("diagnostic sampling requires a non-empty sequence")
    return parameter_names, batch_indices, positions


def _require_d128_runtime_populated_forward_topology(
    route: str,
    config: Config,
    runtime: LowpAttentionRuntime | None,
) -> None:
    """Reject a D128 forward outside the two authenticated final recipes."""
    if config.head_dim != 128 or runtime is None:
        return
    if route not in ("fp8", "mx"):
        raise RuntimeError(
            "D128 saturated forward topology requires the exact fp8 or mx "
            f"route, observed {route!r}"
        )
    if runtime.forward_topology_runtime_authenticated is not True:
        raise RuntimeError(
            "D128 saturated forward topology is not runtime-authenticated"
        )
    topology = runtime.forward_topology
    if not isinstance(topology, dict):
        raise RuntimeError(
            "D128 saturated forward topology must be a populated mapping"
        )

    route_keys = {
        "fp8": (
            "real_fwd_tk_hao_direct_causal_gqa_nvfp4_fp8pv",
            "e4m3_fp8",
        ),
        "mx": (
            "real_fwd_tk_hao_direct_nvfp4_mxfp4pv",
            "mxfp4_e8m0_block32",
        ),
    }
    recipe = _d128_forward_topology_recipe(config, route_keys[route])
    if recipe is None:
        raise RuntimeError(
            f"D128 {route} has no exact forward topology recipe"
        )
    expected = {
        **recipe,
        "valid": 1,
    }
    mismatches = {
        field: {"actual": topology.get(field), "expected": expected}
        for field, expected in expected.items()
        if (
            topology.get(field) != expected
            or type(topology.get(field)) is not type(expected)
        )
    }
    if mismatches:
        raise RuntimeError(
            f"D128 {route} saturated forward topology mismatch: {mismatches}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def _benchmark_source_identities() -> dict[str, dict[str, Any]]:
    """Snapshot every source that can define the measured model step."""
    return {
        "harness": _source_identity(Path(__file__)),
        "projection_interface": _source_identity(Path(tk_interface.__file__)),
        "runtime": _source_identity(
            Path(__file__).with_name("benchmark_llama12b_e2e.py")
        ),
        "backward_policy": _source_identity(
            Path(__file__).with_name("backward_policy.py")
        ),
        "backward_contract": _source_identity(
            Path(__file__).with_name("backward_contract.py")
        ),
        "forward_route": _source_identity(
            Path(__file__).with_name("forward_route.py")
        ),
        "backward_runner": _source_identity(
            Path(__file__).with_name("profile_gqa_d128_chain.py")
        ),
        "native_tk_d64_backward_runner": _source_identity(
            Path(__file__).with_name("native_tk_d64_backward.py")
        ),
        "native_tk_d128_backward_runner": _source_identity(
            Path(__file__).with_name("native_tk_d128_backward.py")
        ),
        "native_tk_d128_mxfp4_v_backward_runner": _source_identity(
            Path(__file__).with_name(
                "native_tk_d128_mxfp4_v_backward.py"
            )
        ),
        "backward_control_loader": _source_identity(
            Path(__file__).with_name("tune_d64_gqa_cute.py")
        ),
        "packed_bf16_qkv": _source_identity(
            Path(__file__).with_name("packed_bf16_qkv.py")
        ),
    }


def _file_identity(path: Path, expected: tuple[str, int]) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"artifact must be a regular non-symlink file: {path}")
    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    expected_sha, expected_bytes = expected
    if resolved.stat().st_size != expected_bytes or digest != expected_sha:
        raise RuntimeError(f"artifact identity mismatch: {resolved}")
    return {
        "path": str(resolved),
        "sha256": digest,
        "bytes": resolved.stat().st_size,
    }


def _require_loaded_artifact_identity(
    label: str,
    extension: Any,
    requested_path: Path,
    expected: tuple[str, int],
) -> dict[str, Any]:
    """Bind an authenticated request to the exact image imported by Python."""
    identity = getattr(
        extension,
        "_tk_fa4_loaded_artifact_identity",
        None,
    )
    if not isinstance(identity, dict):
        raise RuntimeError(
            f"{label} extension has no loaded-artifact identity receipt"
        )
    wanted = {
        "path": str(requested_path.resolve(strict=True)),
        "sha256": expected[0],
        "bytes": expected[1],
    }
    mismatches = {
        field: {"actual": identity.get(field), "expected": value}
        for field, value in wanted.items()
        if (
            identity.get(field) != value
            or type(identity.get(field)) is not type(value)
        )
    }
    if mismatches:
        raise RuntimeError(
            f"{label} loaded-artifact identity mismatch: {mismatches}"
        )
    return dict(identity)


def _projection_expected_identity(
    path: Path,
    declared_sha256: str | None,
    declared_bytes: int | None,
) -> tuple[tuple[str, int], str]:
    """Resolve a fail-closed identity for the projection extension.

    The archived production binary keeps its source-pinned identity. A local
    experimental build must declare both its digest and byte count; the normal
    file gate still re-reads and authenticates it before constructing a
    low-precision runtime.
    """
    is_default = path.resolve() == DEFAULT_PROJECTION.resolve()
    has_sha256 = declared_sha256 is not None
    has_bytes = declared_bytes is not None
    if is_default:
        if has_sha256 or has_bytes:
            raise ValueError(
                "the pinned projection extension does not accept an identity "
                "override"
            )
        return PINNED_ARTIFACTS["projection"], "source_pinned"
    return (
        _caller_declared_expected_identity(
            "projection",
            declared_sha256,
            declared_bytes,
        ),
        "caller_declared",
    )


def _caller_declared_expected_identity(
    artifact: str,
    declared_sha256: str | None,
    declared_bytes: int | None,
) -> tuple[str, int]:
    """Validate an explicit artifact digest/size pair without reading it."""
    has_sha256 = declared_sha256 is not None
    has_bytes = declared_bytes is not None
    if has_sha256 != has_bytes:
        raise ValueError(
            f"a custom --{artifact}-extension requires both "
            f"--{artifact}-sha256 and --{artifact}-bytes"
        )
    if not has_sha256:
        raise ValueError(
            f"a custom --{artifact}-extension requires an explicitly declared "
            "SHA256 and byte count"
        )
    assert declared_sha256 is not None and declared_bytes is not None
    normalized_sha256 = declared_sha256.lower()
    if (
        len(normalized_sha256) != 64
        or any(
            character not in string.hexdigits
            for character in normalized_sha256
        )
    ):
        raise ValueError(
            f"--{artifact}-sha256 must contain 64 hexadecimal digits"
        )
    if declared_bytes <= 0:
        raise ValueError(f"--{artifact}-bytes must be positive")
    return normalized_sha256, declared_bytes


def _forward_expected_identity(
    route: str,
    path: Path,
    module_name: str | None,
    declared_sha256: str | None,
    declared_bytes: int | None,
) -> tuple[tuple[str, int], str, str]:
    """Resolve a pinned D64 or caller-declared forward extension identity."""
    if route not in LOWP_ROUTES:
        raise ValueError(f"forward identity requires a low-precision route: {route}")
    default_path = DEFAULT_FORWARDS[route]
    default_module = FORWARD_MODULES[route]
    is_default = path.resolve() == default_path.resolve()
    if is_default:
        if declared_sha256 is not None or declared_bytes is not None:
            raise ValueError(
                "the pinned forward extension does not accept an identity override"
            )
        if module_name is not None and module_name != default_module:
            raise ValueError(
                "the pinned forward extension requires its pinned module name"
            )
        return (
            PINNED_ARTIFACTS["forward"][route],
            "source_pinned",
            default_module,
        )
    if not isinstance(module_name, str) or not module_name.strip():
        raise ValueError(
            "a custom --forward-extension requires --forward-module"
        )
    return (
        _caller_declared_expected_identity(
            "forward",
            declared_sha256,
            declared_bytes,
        ),
        "caller_declared",
        module_name,
    )


def _require_saturated_projection_selection(
    route: str,
    qkv_projection_format: str,
    experimental_native_nvfp4_projection_out: bool,
    experimental_fused_attention_rmsnorm_nvfp4: bool,
    experimental_output_shared_split_v: bool | None,
    projection_authentication: str | None,
    experimental_d128_mxfp4_v_backward: bool = False,
    experimental_d128_shared_tile_mxfp4_v: bool = False,
) -> None:
    """Keep native NVFP4 behind an explicit, provenance-bound B16 arm."""
    native_format = qkv_projection_format == "nvfp4"
    if native_format and not experimental_native_nvfp4_projection_out:
        raise ValueError(
            "--qkv-projection-format nvfp4 requires "
            "--experimental-native-nvfp4-projection-out"
        )
    if experimental_native_nvfp4_projection_out and not native_format:
        raise ValueError(
            "--experimental-native-nvfp4-projection-out requires "
            "--qkv-projection-format nvfp4"
        )
    if experimental_fused_attention_rmsnorm_nvfp4 and (
        route not in EXPERIMENTAL_NATIVE_NVFP4_ROUTES
        or not native_format
        or not experimental_native_nvfp4_projection_out
    ):
        raise ValueError(
            "--experimental-fused-attention-rmsnorm-nvfp4 requires the "
            "exact fp8 or mx native-NVFP4 saturated route"
        )
    if experimental_output_shared_split_v is True and (
        route != "mx"
        or not native_format
        or not experimental_native_nvfp4_projection_out
    ):
        raise ValueError(
            "--experimental-output-shared-split-v requires the exact MX "
            "native-NVFP4 saturated route"
        )
    if experimental_d128_mxfp4_v_backward and (
        route != "mx"
        or not native_format
        or not experimental_native_nvfp4_projection_out
    ):
        raise ValueError(
            "--experimental-d128-mxfp4-v-backward requires the exact D128 "
            "MX native-NVFP4 saturated route"
        )
    if (
        experimental_d128_mxfp4_v_backward
        and experimental_output_shared_split_v is not False
    ):
        raise ValueError(
            "--experimental-d128-mxfp4-v-backward is mutually exclusive "
            "with --experimental-output-shared-split-v"
        )
    if (
        experimental_d128_shared_tile_mxfp4_v
        and not experimental_d128_mxfp4_v_backward
    ):
        raise ValueError(
            "--experimental-d128-shared-tile-mxfp4-v requires "
            "--experimental-d128-mxfp4-v-backward"
        )
    if not native_format:
        return
    if route not in EXPERIMENTAL_NATIVE_NVFP4_ROUTES:
        raise ValueError(
            "experimental native NVFP4 saturated runs require the exact "
            "fp8 or mx route"
        )
    if projection_authentication != "caller_declared":
        raise ValueError(
            "experimental native NVFP4 saturated runs require a custom "
            "projection binary authenticated by caller-declared SHA256 and "
            "byte count"
        )


def _require_native_tk_d64_saturated_runtime(
    route: str,
    config: Config,
    runtime: LowpAttentionRuntime,
    binary_artifact: dict[str, Any],
) -> None:
    """Authenticate the complete B16 native-TK training publication."""
    if route not in LOWP_ROUTES:
        raise ValueError("native TK D64 backward requires a low-precision route")
    shape = (
        config.batch,
        config.sequence,
        config.hidden,
        config.q_heads,
        config.kv_heads,
        config.head_dim,
    )
    if shape != (16, 4096, 2048, 32, 8, 64):
        raise ValueError(
            "native TK saturated backward requires exactly "
            "B16/S4096/H2048/Hq32/Hkv8/D64"
        )
    expected_pv_format = (
        "mxfp4_e8m0_block32" if route in MX_ROUTES else "e4m3_fp8"
    )
    expected_split_v = route in MX_ROUTES
    expected_runtime = {
        "native_tk_d64_backward": True,
        "qkv_projection_format": "e4m3",
        "projection_dgrad": "bf16",
        "pv_format": expected_pv_format,
        "experimental_split_v_backward": expected_split_v,
        "experimental_output_shared_split_v": False,
        "backward_match_forward_operands": True,
        "per_block_qk_scales": True,
    }
    runtime_mismatches = {
        field: {"actual": getattr(runtime, field, None), "expected": expected}
        for field, expected in expected_runtime.items()
        if (
            getattr(runtime, field, None) != expected
            or type(getattr(runtime, field, None)) is not type(expected)
        )
    }
    publication = runtime.projection_publication_topology
    expected_publication = {
        "qkv_projection_format": "e4m3",
        "represented_backward": True,
        "per_block_qk_scales": True,
        "qk_backward_source": "represented_nvfp4_codes_per_row_k16",
        "v_backward_source": "projection_accumulator_e4m3",
        "experimental_split_v_backward": expected_split_v,
        "experimental_output_shared_split_v": False,
    }
    publication_mismatches = {
        field: {"actual": publication.get(field), "expected": expected}
        for field, expected in expected_publication.items()
        if (
            publication.get(field) != expected
            or type(publication.get(field)) is not type(expected)
        )
    }
    topology = runtime.forward_topology
    expected_topology = {
        "pv_format": expected_pv_format,
        "causal_interleaved_kv": expected_split_v,
    }
    if route == "fp8":
        expected_topology["shiftless_fp8_mode"] = 0
    topology_mismatches = {
        field: {"actual": topology.get(field), "expected": expected}
        for field, expected in expected_topology.items()
        if (
            topology.get(field) != expected
            or type(topology.get(field)) is not type(expected)
        )
    }
    loaded_image = binary_artifact.get("loaded_image")
    if not isinstance(loaded_image, dict):
        raise RuntimeError(
            "native TK backward binary has no loaded-image provenance"
        )
    identity_mismatches = {}
    for label, identity in (
        (
            "runtime",
            runtime.native_tk_d64_backward_extension_identity,
        ),
        ("runner", runtime.backward.contract().get("extension")),
    ):
        if identity != loaded_image:
            identity_mismatches[label] = {
                "actual": identity,
                "expected": loaded_image,
            }
    backward_contract = runtime.backward_contract()
    backend_contract = backward_contract.get("backend", {})
    projection_contract = backward_contract.get("projection", {})
    contract_mismatches = {}
    expected_contract = {
        "backend.backend": NATIVE_TK_D64_BACKEND,
        "backend.input.dtype": "torch.float8_e4m3fn",
        "backend.input.layout": "BSHD_contiguous",
        "projection.dout_backward_source": "projection_accumulator_e4m3",
    }
    observed_contract = {
        "backend.backend": backend_contract.get("backend"),
        "backend.input.dtype": backend_contract.get("input", {}).get("dtype"),
        "backend.input.layout": backend_contract.get("input", {}).get("layout"),
        "projection.dout_backward_source": projection_contract.get(
            "dout_backward_source"
        ),
    }
    for field, expected in expected_contract.items():
        if observed_contract[field] != expected:
            contract_mismatches[field] = {
                "actual": observed_contract[field],
                "expected": expected,
            }
    if (
        runtime_mismatches
        or publication_mismatches
        or topology_mismatches
        or identity_mismatches
        or contract_mismatches
    ):
        raise RuntimeError(
            "native TK saturated runtime contract mismatch: "
            f"runtime={runtime_mismatches}, "
            f"publication={publication_mismatches}, "
            f"topology={topology_mismatches}, "
            f"identity={identity_mismatches}, "
            f"backward={contract_mismatches}"
        )


def _require_native_tk_d128_saturated_runtime(
    route: str,
    config: Config,
    runtime: LowpAttentionRuntime,
    binary_artifact: dict[str, Any],
) -> None:
    """Authenticate the complete native-TK D128 operand publication."""
    if route not in ("fp8", "mx"):
        raise ValueError(
            "native TK D128 backward requires the FP8-PV or MXFP4-PV route"
        )
    shape = (
        config.batch,
        config.sequence,
        config.hidden,
        config.q_heads,
        config.kv_heads,
        config.head_dim,
    )
    if shape not in {
        (1, 4096, 4096, 32, 8, 128),
        (2, 4096, 4096, 32, 8, 128),
    }:
        raise ValueError(
            "native TK D128 saturated backward requires exactly B1/B2 "
            "S4096/H4096/Hq32/Hkv8/D128"
        )
    expected_pv_format = (
        "mxfp4_e8m0_block32" if route == "mx" else "e4m3_fp8"
    )
    mx_v_backward = bool(runtime.experimental_d128_mxfp4_v_backward)
    mx_scale_policy = runtime.d128_mxfp4_v_scale_policy
    allowed_mx_scale_policies = {
        tk_interface.MXFP4_V_SCALE_POLICY_ROWWISE_D32,
        tk_interface.MXFP4_V_SCALE_POLICY_SHARED_D32XS32,
    }
    if mx_v_backward and mx_scale_policy not in allowed_mx_scale_policies:
        raise RuntimeError("native TK MXFP4-V scale policy is unauthenticated")
    if not mx_v_backward and mx_scale_policy is not None:
        raise RuntimeError("E4M3-V backward carries an MXFP4 scale policy")
    shared_tile_mx = bool(
        mx_v_backward
        and mx_scale_policy
        == tk_interface.MXFP4_V_SCALE_POLICY_SHARED_D32XS32
    )
    expected_v_backward_source = (
        "shared_d32xs32_forward_anchor_mxfp4_v"
        if shared_tile_mx
        else "rowwise_width6_mxfp4_v"
        if mx_v_backward
        else "projection_accumulator_e4m3"
    )
    if mx_v_backward and (route != "mx" or config.batch != 2):
        raise ValueError(
            "native TK D128 MXFP4-V backward requires exactly the B2 "
            "MXFP4-PV route"
        )
    expected_runtime = {
        "native_tk_d128_backward": True,
        "qkv_projection_format": "nvfp4",
        "projection_dgrad": "nvfp4",
        "pv_format": expected_pv_format,
        "experimental_split_v_backward": False,
        "experimental_d128_mxfp4_v_backward": mx_v_backward,
        "d128_mxfp4_v_scale_policy": mx_scale_policy,
        "v_mxfp4_scale_2d": shared_tile_mx,
        "backward_match_forward_operands": False,
        "per_block_qk_scales": True,
    }
    runtime_mismatches = {
        field: {"actual": getattr(runtime, field, None), "expected": expected}
        for field, expected in expected_runtime.items()
        if (
            getattr(runtime, field, None) != expected
            or type(getattr(runtime, field, None)) is not type(expected)
        )
    }
    publication = runtime.projection_publication_topology
    expected_publication = {
        "qkv_projection_format": "nvfp4",
        "forward_pv_format": expected_pv_format,
        "represented_backward": False,
        "per_block_qk_scales": True,
        "qk_backward_source": "projection_accumulator_e4m3",
        "v_backward_source": expected_v_backward_source,
        "experimental_split_v_backward": False,
        "experimental_d128_mxfp4_v_backward": mx_v_backward,
        "d128_mxfp4_v_scale_policy": mx_scale_policy,
    }
    publication_mismatches = {
        field: {"actual": publication.get(field), "expected": expected}
        for field, expected in expected_publication.items()
        if (
            publication.get(field) != expected
            or type(publication.get(field)) is not type(expected)
        )
    }
    expected_topology = {
        "pv_format": expected_pv_format,
        "causal_interleaved_kv": False,
    }
    if route == "fp8":
        expected_topology["shiftless_fp8_mode"] = 0
    topology_mismatches = {
        field: {
            "actual": runtime.forward_topology.get(field),
            "expected": expected,
        }
        for field, expected in expected_topology.items()
        if (
            runtime.forward_topology.get(field) != expected
            or type(runtime.forward_topology.get(field)) is not type(expected)
        )
    }
    loaded_image = binary_artifact.get("loaded_image")
    if not isinstance(loaded_image, dict):
        raise RuntimeError(
            "native TK D128 backward binary has no loaded-image provenance"
        )
    identity_mismatches = {}
    for label, identity in (
        ("runtime", runtime.native_tk_d128_backward_extension_identity),
        ("runner", runtime.backward.contract().get("extension")),
    ):
        if identity != loaded_image:
            identity_mismatches[label] = {
                "actual": identity,
                "expected": loaded_image,
            }
    backward_contract = runtime.backward_contract()
    backend_contract = backward_contract.get("backend", {})
    projection_contract = backward_contract.get("projection", {})
    expected_backend = (
        NATIVE_TK_D128_SHARED_TILE_MX_BACKEND
        if shared_tile_mx
        else NATIVE_TK_D128_MX_BACKEND
        if mx_v_backward
        else NATIVE_TK_D128_BACKEND
    )
    expected_contract = {
        "backend.backend": expected_backend,
        "backend.input.dtype": (
            "mixed_e4m3fn_and_packed_mxfp4_e8m0"
            if mx_v_backward
            else "torch.float8_e4m3fn"
        ),
        "backend.input.layout": (
            "BSHD_contiguous_with_physical_scale_pages"
            if mx_v_backward
            else "BSHD_contiguous"
        ),
        "projection.qk_backward_source": "projection_accumulator_e4m3",
        "projection.v_backward_source": expected_v_backward_source,
        "projection.dout_backward_source": "projection_accumulator_e4m3",
        "projection.experimental_d128_mxfp4_v_backward": mx_v_backward,
        "projection.d128_mxfp4_v_scale_policy": mx_scale_policy,
    }
    observed_contract = {
        "backend.backend": backend_contract.get("backend"),
        "backend.input.dtype": backend_contract.get("input", {}).get("dtype"),
        "backend.input.layout": backend_contract.get("input", {}).get("layout"),
        "projection.qk_backward_source": projection_contract.get(
            "qk_backward_source"
        ),
        "projection.v_backward_source": projection_contract.get(
            "v_backward_source"
        ),
        "projection.dout_backward_source": projection_contract.get(
            "dout_backward_source"
        ),
        "projection.experimental_d128_mxfp4_v_backward": (
            projection_contract.get("experimental_d128_mxfp4_v_backward")
        ),
        "projection.d128_mxfp4_v_scale_policy": projection_contract.get(
            "d128_mxfp4_v_scale_policy"
        ),
    }
    contract_mismatches = {
        field: {"actual": observed_contract[field], "expected": expected}
        for field, expected in expected_contract.items()
        if observed_contract[field] != expected
    }
    if (
        runtime_mismatches
        or publication_mismatches
        or topology_mismatches
        or identity_mismatches
        or contract_mismatches
    ):
        raise RuntimeError(
            "native TK D128 saturated runtime contract mismatch: "
            f"runtime={runtime_mismatches}, "
            f"publication={publication_mismatches}, "
            f"topology={topology_mismatches}, "
            f"identity={identity_mismatches}, "
            f"backward={contract_mismatches}"
        )


def _d128_mxfp4_v_dp_patch_artifact(
    runtime: LowpAttentionRuntime,
) -> dict[str, int | str] | None:
    """Return the candidate-only loader receipt, failing closed on drift."""
    enabled = bool(runtime.experimental_d128_mxfp4_v_backward)
    receipt = getattr(
        runtime,
        "d128_mxfp4_v_dp_patch_provenance",
        None,
    )
    if not enabled:
        if receipt is not None:
            raise RuntimeError(
                "retained saturated route unexpectedly carries D128 MXFP4 "
                "V dP patch provenance"
            )
        return None
    if bool(getattr(runtime, "native_tk_d128_backward", False)):
        if receipt is not None:
            raise RuntimeError(
                "native TK D128 MXFP4-V backward unexpectedly carries a "
                "CuTe patch receipt"
            )
        return None
    if not isinstance(receipt, dict) or set(receipt) != {
        "path",
        "sha256",
        "bytes",
    }:
        raise RuntimeError(
            "candidate saturated route is missing D128 MXFP4 V dP patch "
            "provenance"
        )
    path = receipt.get("path")
    sha256 = receipt.get("sha256")
    byte_count = receipt.get("bytes")
    if not isinstance(path, str) or not path or not Path(path).is_absolute():
        raise RuntimeError("candidate patch artifact path is malformed")
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or sha256 != sha256.lower()
        or any(character not in string.hexdigits for character in sha256)
    ):
        raise RuntimeError("candidate patch artifact SHA256 is malformed")
    if type(byte_count) is not int or byte_count <= 0:
        raise RuntimeError("candidate patch artifact byte count is malformed")
    artifact = {"path": path, "sha256": sha256, "bytes": byte_count}
    control_receipt = runtime.backward_contract().get("control", {}).get(
        "d128_mxfp4_v_dp_patch"
    )
    if control_receipt != artifact:
        raise RuntimeError(
            "candidate patch artifact disagrees with backward contract"
        )
    return artifact


def _d128_qkv_projection_contract(
    runtime: LowpAttentionRuntime,
    projection_artifact: dict[str, Any],
) -> dict[str, Any]:
    """Authenticate the caller-owned D128 route-selective publication ABI."""
    config = runtime.config
    expected_shape = {
        "batch": config.batch,
        "sequence": 4096,
        "hidden": 4096,
        "q_heads": 32,
        "kv_heads": 8,
        "head_dim": 128,
    }
    observed_shape = {
        field: int(getattr(config, field)) for field in expected_shape
    }
    is_mx = runtime.pv_format == "mxfp4_e8m0_block32"
    mx_backward_v = bool(runtime.experimental_d128_mxfp4_v_backward)
    mx_scale_policy = runtime.d128_mxfp4_v_scale_policy
    shared_tile_mx = bool(
        mx_backward_v
        and mx_scale_policy
        == tk_interface.MXFP4_V_SCALE_POLICY_SHARED_D32XS32
    )
    expected_v_backward_source = (
        "shared_d32xs32_forward_anchor_mxfp4_v"
        if shared_tile_mx
        else "rowwise_width6_mxfp4_v"
        if mx_backward_v
        else "projection_accumulator_e4m3"
    )
    expected_backward_semantics = (
        "single_quantized_d32xs32_mxfp4_v_with_projection_accumulator_e4m3_qk"
        if shared_tile_mx
        else "rowwise_width6_mxfp4_v_with_projection_accumulator_e4m3_qk"
        if mx_backward_v
        else "projection_accumulator_e4m3_qkv_shared_across_pv_routes"
    )
    if runtime.pv_format not in ("mxfp4_e8m0_block32", "e4m3_fp8"):
        raise RuntimeError(
            f"unsupported D128 saturated PV format: {runtime.pv_format!r}"
        )
    requested_output_shared = (
        runtime.experimental_output_shared_split_v_requested
    )
    if (
        requested_output_shared is not None
        and type(requested_output_shared) is not bool
    ):
        raise RuntimeError(
            "D128 output-shared selector must be exactly bool or None"
        )
    output_shared_eligible = bool(
        is_mx
        and config.batch in (1, *AUTHENTICATED_D128_EXACT_BATCHES)
        and not mx_backward_v
    )
    if requested_output_shared is True and not output_shared_eligible:
        raise RuntimeError(
            "D128 output-shared dual V was explicitly requested outside "
            "the authenticated B1/B2 MXFP4-PV route"
        )
    resolved_output_shared = bool(
        output_shared_eligible
        if requested_output_shared is None
        else requested_output_shared
    )
    output_shared_path = (
        "shared_tile_mx_backward_v"
        if shared_tile_mx
        else "mx_backward_v"
        if mx_backward_v
        else "output_shared_dual_v"
        if resolved_output_shared
        else "retained_dual_v"
        if is_mx
        else "fp8"
    )
    projection_publication_path = (
        "caller_owned_shared_tile_mx_backward_v_d128"
        if shared_tile_mx
        else "caller_owned_mx_backward_v_d128"
        if mx_backward_v
        else "caller_owned_output_shared_dual_v_d128"
        if resolved_output_shared
        else "caller_owned_route_selective_d128"
    )
    checked_symbol = D128_PROJECTION_ABI_SYMBOL + (
        "_shared_tile_mx_backward_v_mx_forward_out"
        if shared_tile_mx
        else "_mx_backward_v_mx_forward_out"
        if mx_backward_v
        else "_output_shared_dual_v_mx_forward_out"
        if resolved_output_shared
        else "_mx_forward_out"
        if is_mx
        else "_fp8_forward_out"
    )
    unchecked_symbol = checked_symbol + "_unchecked"
    publication = dict(runtime.projection_publication_topology)
    projection_dispatch = runtime.forward_dispatch_contract()["qkv_projection"]
    expected_publication = {
        "qkv_projection_format": "nvfp4",
        "represented_backward": False,
        "per_block_qk_scales": True,
        "qk_backward_source": "projection_accumulator_e4m3",
        "v_backward_source": expected_v_backward_source,
        "experimental_split_v_backward": False,
        "experimental_output_shared_split_v": resolved_output_shared,
        "experimental_output_shared_split_v_requested": (
            requested_output_shared
        ),
        "experimental_output_shared_split_v_resolved": (
            resolved_output_shared
        ),
        "output_shared_split_v_path": output_shared_path,
        "projection_forward_publication_path": projection_publication_path,
        "experimental_native_nvfp4_projection_out": True,
        "experimental_fused_attention_rmsnorm_nvfp4": False,
        "experimental_d128_mxfp4_v_backward": mx_backward_v,
        "d128_mxfp4_v_scale_policy": mx_scale_policy,
    }
    expected_dispatch = {
        "format": "nvfp4",
        "experimental_native_nvfp4_caller_owned": True,
        "experimental_fused_attention_rmsnorm_nvfp4": False,
        "experimental_d128_mxfp4_v_backward": mx_backward_v,
        "output_shared_split_v_requested": requested_output_shared,
        "output_shared_split_v_resolved": resolved_output_shared,
        "output_shared_split_v_path": output_shared_path,
        "projection_forward_publication_path": projection_publication_path,
        "backward_publication_semantics": expected_backward_semantics,
        "dispatch": "construction_bound_exact_pybind_symbol",
        "symbol": unchecked_symbol,
        "abi_validation_symbol": D128_PROJECTION_ABI_SYMBOL,
        "checked_symbol": checked_symbol,
        "unchecked_symbol": unchecked_symbol,
        "shape_bound_at_construction": True,
        "preallocated_forward_workspace_required": True,
        "timed_forward_publication_allocation_fallback": False,
    }
    native_tk_d128_backward = bool(
        getattr(runtime, "native_tk_d128_backward", False)
    )
    expected_runtime = {
        "experimental_output_shared_split_v": resolved_output_shared,
        "experimental_d128_mxfp4_v_backward": mx_backward_v,
        "d128_mxfp4_v_scale_policy": mx_scale_policy,
        "experimental_output_shared_split_v_requested": (
            requested_output_shared
        ),
        "experimental_output_shared_split_v_resolved": (
            resolved_output_shared
        ),
        "output_shared_split_v_path": output_shared_path,
        "qkv_projection_symbol": unchecked_symbol,
        "projection_dgrad": "nvfp4",
        "backward_match_forward_operands": False,
        "per_block_qk_scales": True,
        "experimental_split_v_backward": False,
        "native_tk_d128_backward": native_tk_d128_backward,
        "backward_reuse_quantized_p": not native_tk_d128_backward,
        "backward_exp2_degree": 0 if native_tk_d128_backward else 1,
        "backward_exp2_period": 0,
    }
    projection = runtime.qkv_projection
    expected_bound_projection = {
        "abi_validation_symbol": D128_PROJECTION_ABI_SYMBOL,
        "checked_symbol": checked_symbol,
        "unchecked_symbol": unchecked_symbol,
        "projection_forward_publication_path": projection_publication_path,
        "backward_publication_semantics": expected_backward_semantics,
        "per_block_qk_scales": True,
        "requires_forward_workspace": True,
        "experimental_output_shared_split_v_requested": (
            requested_output_shared
        ),
        "experimental_output_shared_split_v_resolved": (
            resolved_output_shared
        ),
        "experimental_mx_backward_v": mx_backward_v,
        "experimental_shared_tile_mx_backward_v": shared_tile_mx,
        "v_backward_mxfp4_scale_policy": mx_scale_policy,
        "output_shared_split_v_path": output_shared_path,
    }
    mismatches = {
        "authenticated_batch": {
            "actual": config.batch,
            "expected": (1, *AUTHENTICATED_D128_EXACT_BATCHES),
        }
        if config.batch not in (1, *AUTHENTICATED_D128_EXACT_BATCHES)
        else None,
        "shape": {
            "actual": observed_shape,
            "expected": expected_shape,
        }
        if observed_shape != expected_shape
        else None,
        "projection_weight_scale_2d": {
            "actual": runtime.projection_weight_scale_2d,
            "expected": True,
        }
        if not runtime.projection_weight_scale_2d
        else None,
        "v_mxfp4_scale_2d": {
            "actual": runtime.v_mxfp4_scale_2d,
            "expected": shared_tile_mx,
        }
        if runtime.v_mxfp4_scale_2d is not shared_tile_mx
        else None,
        "projection_artifact.authentication": {
            "actual": projection_artifact.get("authentication"),
            "expected": "caller_declared",
        }
        if projection_artifact.get("authentication") != "caller_declared"
        else None,
    }
    mismatches = {key: value for key, value in mismatches.items() if value}
    publication_mismatches = {
        key: {"actual": publication.get(key), "expected": expected}
        for key, expected in expected_publication.items()
        if publication.get(key) != expected
    }
    dispatch_mismatches = {
        key: {"actual": projection_dispatch.get(key), "expected": expected}
        for key, expected in expected_dispatch.items()
        if projection_dispatch.get(key) != expected
    }
    runtime_mismatches = {
        key: {"actual": getattr(runtime, key, None), "expected": expected}
        for key, expected in expected_runtime.items()
        if getattr(runtime, key, None) != expected
    }
    bound_projection_mismatches = {
        key: {"actual": getattr(projection, key, None), "expected": expected}
        for key, expected in expected_bound_projection.items()
        if getattr(projection, key, None) != expected
    }
    if (
        mismatches
        or publication_mismatches
        or dispatch_mismatches
        or runtime_mismatches
        or bound_projection_mismatches
    ):
        raise RuntimeError(
            "D128 route-selective projection contract mismatch: "
            f"general={mismatches}, publication={publication_mismatches}, "
            f"dispatch={dispatch_mismatches}, runtime={runtime_mismatches}, "
            f"bound_projection={bound_projection_mismatches}"
        )
    sanitized_publication = {
        **publication,
        "output_shared_split_v_checked_symbol": (
            checked_symbol if resolved_output_shared else None
        ),
        "route_selective_checked_symbol": checked_symbol,
        "route_selective_unchecked_symbol": unchecked_symbol,
        "backward_publication_semantics": expected_backward_semantics,
    }
    return {
        "schema": "saturated_qkv_projection_contract_v4",
        "qkv_projection_format": "nvfp4",
        "experimental_native_nvfp4_projection_out": True,
        "experimental_fused_attention_rmsnorm_nvfp4": False,
        "experimental_d128_mxfp4_v_backward": mx_backward_v,
        "d128_mxfp4_v_scale_policy": mx_scale_policy,
        "projection_artifact": projection_artifact,
        "operand_preparation": {
            "input": {
                "function": "b300_prepare_nvfp4_projection_operand",
                "scale_layout": "row_by_k16",
                "fuses_attention_rmsnorm": False,
            },
            "learned_weight": {
                "function": (
                    "b300_prepare_gqa_d128_qkv_projection_weight_dual_out"
                ),
                "source": "canonical_split_qkv_parameters",
                "forward_operand": {
                    "format": "nvfp4",
                    "physical_layout": (
                        "pair_interleaved_qk_then_canonical_v"
                    ),
                    "scale_layout": "true_16x16",
                },
                "backward_operand": {
                    "format": "nvfp4",
                    "physical_layout": (
                        "transpose_of_pair_interleaved_qk_then_canonical_v"
                    ),
                    "scale_layout": "true_16x16",
                },
                "caller_owned": True,
                "shared_global_scale": True,
                "refresh": "every_forward",
                "first_use_authentication": {
                    "comparison": "bitwise_all_published_bytes",
                    "reference": (
                        "pair_interleave_concat_then_independent_true_2d_"
                        "quantization"
                    ),
                    "checked_symbol": (
                        "quantize_gqa_d128_qkv_projection_weight_dual_out"
                    ),
                    "hot_path_symbol": (
                        "quantize_gqa_d128_qkv_projection_weight_dual_out_"
                        "unchecked"
                    ),
                },
            },
        },
        "publication": sanitized_publication,
        "forward_dispatch": runtime.forward_dispatch_contract(),
        "d128_route_selective_publication": {
            "active_forward_v": "mxfp4" if is_mx else "e4m3_fp8",
            "inactive_forward_v_omitted": True,
            "qk_scale_geometry": "row_by_k16",
            "checked_symbol": checked_symbol,
            "unchecked_symbol": unchecked_symbol,
            "shared_backward_qkv": (
                "mxfp4_v_plus_e4m3_qk"
                if mx_backward_v
                else "e4m3_projection_accumulator"
            ),
            "output_shared_split_v": resolved_output_shared,
            "output_shared_candidate_eligible": output_shared_eligible,
            "publication_path": output_shared_path,
        },
    }


def _qkv_projection_contract(
    runtime: LowpAttentionRuntime,
    projection_artifact: dict[str, Any],
) -> dict[str, Any]:
    """Authenticate and describe the projection recipe recorded in results."""
    publication = dict(runtime.projection_publication_topology)
    dispatch = runtime.forward_dispatch_contract()
    native = runtime.experimental_native_nvfp4_projection_out
    fused_rmsnorm = runtime.experimental_fused_attention_rmsnorm_nvfp4
    if native and runtime.config.head_dim == 128:
        return _d128_qkv_projection_contract(runtime, projection_artifact)
    if native:
        requested = runtime.experimental_output_shared_split_v_requested
        if requested is not None and type(requested) is not bool:
            raise RuntimeError(
                "experimental native NVFP4 projection contract mismatch: "
                "output-shared selector must be exactly bool or None"
            )
        config = runtime.config
        direct_mx = runtime.pv_format == "mxfp4_e8m0_block32"
        output_shared_eligible = bool(
            direct_mx
            and runtime.qkv_projection_format == "nvfp4"
            and not runtime.v_mxfp4_scale_2d
            and config.batch == 16
            and config.sequence == 4096
            and config.hidden == 2048
            and config.q_heads == 32
            and config.kv_heads == 8
            and config.head_dim == 64
        )
        if requested is True and not output_shared_eligible:
            raise RuntimeError(
                "experimental native NVFP4 projection contract mismatch: "
                "output-shared split-V was explicitly requested for an "
                "ineligible route, scale policy, or shape"
            )
        expected_output_shared = bool(
            output_shared_eligible if requested is None else requested
        )
        expected_path = (
            "output_shared_split_v"
            if expected_output_shared
            else "retained_split_v"
            if direct_mx
            else "fp8"
        )
        symbol_prefix = (
            "project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_"
        )
        if expected_output_shared:
            expected_checked_symbol = (
                symbol_prefix
                + "interleaved_causal_represented_backward_perblock_qk_"
                "output_shared_split_v_mx_forward_out"
            )
        elif direct_mx:
            expected_checked_symbol = (
                symbol_prefix
                + "interleaved_causal_represented_backward_perblock_qk_"
                "split_v_backward_mx_forward_out"
            )
        else:
            expected_checked_symbol = (
                symbol_prefix
                + "represented_backward_perblock_qk_fp8_forward_out"
            )
        expected_unchecked_symbol = expected_checked_symbol + "_unchecked"
        expected_publication = {
            "qkv_projection_format": "nvfp4",
            "represented_backward": True,
            "per_block_qk_scales": True,
            "qk_backward_source": "represented_nvfp4_codes_per_row_k16",
            "v_backward_source": "projection_accumulator_e4m3",
            "experimental_split_v_backward": direct_mx,
            "experimental_output_shared_split_v": (
                expected_output_shared
            ),
            "experimental_output_shared_split_v_requested": (
                requested
            ),
            "experimental_output_shared_split_v_resolved": (
                expected_output_shared
            ),
            "output_shared_split_v_path": expected_path,
            "output_shared_split_v_checked_symbol": expected_checked_symbol,
            "experimental_native_nvfp4_projection_out": True,
            "experimental_fused_attention_rmsnorm_nvfp4": fused_rmsnorm,
        }
        mismatches = {
            key: {"actual": publication.get(key), "expected": expected}
            for key, expected in expected_publication.items()
            if publication.get(key) != expected
        }
        projection_dispatch = dispatch["qkv_projection"]
        expected_dispatch = {
            "format": "nvfp4",
            "experimental_native_nvfp4_caller_owned": True,
            "experimental_fused_attention_rmsnorm_nvfp4": fused_rmsnorm,
            "backward_publication_semantics": (
                "represented_nvfp4_qk_per_row_k16_with_"
                "projection_accumulator_e4m3_v"
            ),
            "shape_bound_at_construction": True,
            "preallocated_forward_workspace_required": True,
            "output_shared_split_v_requested": requested,
            "output_shared_split_v_resolved": expected_output_shared,
            "output_shared_split_v_path": expected_path,
            "symbol": expected_unchecked_symbol,
            "checked_symbol": expected_checked_symbol,
            "unchecked_symbol": expected_unchecked_symbol,
        }
        dispatch_mismatches = {
            key: {
                "actual": projection_dispatch.get(key),
                "expected": expected,
            }
            for key, expected in expected_dispatch.items()
            if projection_dispatch.get(key) != expected
        }
        expected_runtime = {
            "experimental_output_shared_split_v": expected_output_shared,
            "experimental_output_shared_split_v_resolved": (
                expected_output_shared
            ),
            "output_shared_split_v_path": expected_path,
            "qkv_projection_symbol": expected_unchecked_symbol,
        }
        runtime_mismatches = {
            key: {"actual": getattr(runtime, key, None), "expected": expected}
            for key, expected in expected_runtime.items()
            if getattr(runtime, key, None) != expected
        }
        projection = runtime.qkv_projection
        expected_bound_projection = {
            "checked_symbol": expected_checked_symbol,
            "unchecked_symbol": expected_unchecked_symbol,
        }
        bound_projection_mismatches = {
            key: {
                "actual": getattr(projection, key, None),
                "expected": expected,
            }
            for key, expected in expected_bound_projection.items()
            if getattr(projection, key, None) != expected
        }
        if not runtime.projection_weight_scale_2d:
            mismatches["projection_weight_scale_2d"] = {
                "actual": False,
                "expected": True,
            }
        if projection_artifact.get("authentication") != "caller_declared":
            mismatches["projection_artifact.authentication"] = {
                "actual": projection_artifact.get("authentication"),
                "expected": "caller_declared",
            }
        if (
            mismatches
            or dispatch_mismatches
            or runtime_mismatches
            or bound_projection_mismatches
        ):
            raise RuntimeError(
                "experimental native NVFP4 projection contract mismatch: "
                f"publication={mismatches}, dispatch={dispatch_mismatches}, "
                f"runtime={runtime_mismatches}, "
                f"bound_projection={bound_projection_mismatches}"
            )
    return {
        "schema": "saturated_qkv_projection_contract_v1",
        "qkv_projection_format": runtime.qkv_projection_format,
        "experimental_native_nvfp4_projection_out": native,
        "experimental_fused_attention_rmsnorm_nvfp4": fused_rmsnorm,
        "projection_artifact": projection_artifact,
        "operand_preparation": (
            {
                "input": {
                    "function": (
                        "b300_prepare_nvfp4_projection_operand_rmsnorm"
                        if fused_rmsnorm
                        else "b300_prepare_nvfp4_projection_operand"
                    ),
                    "scale_layout": "row_by_k16",
                    "fuses_attention_rmsnorm": fused_rmsnorm,
                },
                "learned_weight": {
                    "function": "b300_prepare_nvfp4_projection_weight",
                    "scale_layout": "true_16x16",
                    "refresh": "every_forward_after_optimizer_update",
                },
            }
            if native
            else {
                "input": {
                    "function": "b300_prepare_e4m3_projection_operand",
                    "scale_layout": "rowwise",
                },
                "learned_weight": {
                    "function": "b300_prepare_e4m3_projection_weight",
                    "scale_layout": "output_channelwise",
                    "refresh": "every_forward_after_optimizer_update",
                },
            }
        ),
        "publication": publication,
        "forward_dispatch": dispatch,
    }


def _d128_dual_qkv_weight_preparation_receipt(
    model: Llama12B,
    config: Config,
    runtime: LowpAttentionRuntime | None,
) -> dict[str, Any] | None:
    """Aggregate post-first-use D128 dual-weight evidence without pointers."""
    if runtime is None or config.head_dim != 128:
        return None

    contract = model.lowp_forward_workspace_contract()
    expected_schema = "lowp_model_forward_workspaces_v2"
    if contract.get("schema") != expected_schema:
        raise RuntimeError(
            "D128 direct-dual QKV receipt requires workspace schema "
            f"{expected_schema!r}; observed {contract.get('schema')!r}"
        )
    layers = contract.get("layers")
    if not isinstance(layers, list) or len(layers) != config.layers:
        observed_count = len(layers) if isinstance(layers, list) else None
        raise RuntimeError(
            "D128 direct-dual QKV receipt requires one workspace contract "
            f"per layer; expected {config.layers}, observed {observed_count}"
        )

    checked_symbol = "quantize_gqa_d128_qkv_projection_weight_dual_out"
    unchecked_symbol = checked_symbol + "_unchecked"
    owner_fields = (
        "qkv_weight_forward_packed",
        "qkv_weight_forward_scales",
        "qkv_weight_backward_packed",
        "qkv_weight_backward_scales",
        "qkv_weight_global_scale",
    )
    owner_pointers: list[int] = []
    total_bytes = 0
    eligible_layer_count = 0
    authenticated_layer_count = 0
    abi_identity_bound_layer_count = 0
    generation_guard_enforced_layer_count = 0
    same_stream_enforced_layer_count = 0
    for expected_layer, layer in enumerate(layers):
        if not isinstance(layer, dict) or layer.get("layer") != expected_layer:
            observed_layer = (
                layer.get("layer") if isinstance(layer, dict) else None
            )
            raise RuntimeError(
                "D128 direct-dual QKV workspace layer order mismatch at "
                f"position {expected_layer}; observed {observed_layer!r}"
            )
        lifecycle = layer.get("publication_lifecycle")
        expected_lifecycle = {
            "generation_guard_enforced": True,
            "same_stream_enforced": True,
            "one_forward_in_flight_per_layer": True,
            "in_flight": False,
        }
        lifecycle_mismatches = {
            field: {
                "actual": (
                    lifecycle.get(field)
                    if isinstance(lifecycle, dict)
                    else None
                ),
                "expected": expected,
            }
            for field, expected in expected_lifecycle.items()
            if not isinstance(lifecycle, dict)
            or lifecycle.get(field) != expected
        }
        current_generation = (
            lifecycle.get("current_generation")
            if isinstance(lifecycle, dict)
            else None
        )
        if type(current_generation) is not int or current_generation < 0:
            lifecycle_mismatches["current_generation"] = {
                "actual": current_generation,
                "expected": "non-negative int after initial diagnostic",
            }
        if lifecycle_mismatches:
            raise RuntimeError(
                "D128 direct-dual QKV lifecycle contract mismatch at "
                f"layer {expected_layer}: {lifecycle_mismatches}"
            )
        generation_guard_enforced_layer_count += 1
        same_stream_enforced_layer_count += 1
        preparation = layer.get("d128_dual_qkv_weight")
        if not isinstance(preparation, dict):
            raise RuntimeError(
                "D128 direct-dual QKV workspace evidence is missing at "
                f"layer {expected_layer}"
            )
        expected_fields = {
            "eligible": True,
            "authenticated": True,
            "schedule": "synchronous_same_stream",
            "one_forward_in_flight_per_layer": True,
            "generation_guard_enforced": True,
            "same_stream_enforced": True,
            "abi_identity_bound": True,
            "abi_identity_tensor_count": 8,
            "abi_identity_excludes_tensor_version": True,
            "checked_symbol": checked_symbol,
            "unchecked_symbol": unchecked_symbol,
            "all_pointers_stable_since_allocation": True,
            "all_pointers_unique": True,
        }
        mismatches = {
            field: {
                "actual": preparation.get(field),
                "expected": expected,
            }
            for field, expected in expected_fields.items()
            if preparation.get(field) != expected
        }
        if mismatches:
            raise RuntimeError(
                "D128 direct-dual QKV preparation contract mismatch at "
                f"layer {expected_layer}: {mismatches}"
            )
        eligible_layer_count += 1
        authenticated_layer_count += 1
        abi_identity_bound_layer_count += 1

        owners = preparation.get("owners")
        if not isinstance(owners, dict) or set(owners) != set(owner_fields):
            observed_fields = (
                sorted(owners) if isinstance(owners, dict) else None
            )
            raise RuntimeError(
                "D128 direct-dual QKV owner fields mismatch at layer "
                f"{expected_layer}; observed {observed_fields!r}"
            )
        layer_bytes = 0
        for owner_name in owner_fields:
            owner = owners[owner_name]
            if not isinstance(owner, dict):
                raise RuntimeError(
                    "D128 direct-dual QKV owner evidence must be a mapping; "
                    f"layer {expected_layer}, owner {owner_name!r}"
                )
            if owner.get("pointer_stable_since_allocation") is not True:
                raise RuntimeError(
                    "D128 direct-dual QKV owner pointer is not stable; "
                    f"layer {expected_layer}, owner {owner_name!r}"
                )
            visibility = {
                field: owner.get(field)
                for field in (
                    "listed_in_named_buffers",
                    "listed_in_named_parameters",
                    "optimizer_visible_parameter",
                )
            }
            if any(value is not False for value in visibility.values()):
                raise RuntimeError(
                    "D128 direct-dual QKV owner is not private scratch; "
                    f"layer {expected_layer}, owner {owner_name!r}, "
                    f"visibility={visibility}"
                )
            pointer = owner.get("data_ptr")
            byte_count = owner.get("bytes")
            if type(pointer) is not int or pointer <= 0:
                raise RuntimeError(
                    "D128 direct-dual QKV owner has no valid allocation "
                    f"identity; layer {expected_layer}, owner {owner_name!r}"
                )
            if type(byte_count) is not int or byte_count <= 0:
                raise RuntimeError(
                    "D128 direct-dual QKV owner has no positive byte count; "
                    f"layer {expected_layer}, owner {owner_name!r}"
                )
            owner_pointers.append(pointer)
            layer_bytes += byte_count
        if preparation.get("total_bytes") != layer_bytes:
            raise RuntimeError(
                "D128 direct-dual QKV byte accounting mismatch at layer "
                f"{expected_layer}; owners sum to {layer_bytes}, observed "
                f"{preparation.get('total_bytes')!r}"
            )
        total_bytes += layer_bytes

    if len(set(owner_pointers)) != len(owner_pointers):
        raise RuntimeError(
            "D128 direct-dual QKV owner allocations are not globally unique"
        )
    if eligible_layer_count != config.layers:
        raise RuntimeError(
            "D128 direct-dual QKV preparation is not eligible on every layer"
        )
    if authenticated_layer_count != config.layers:
        raise RuntimeError(
            "D128 direct-dual QKV preparation is not authenticated on every "
            "layer"
        )

    return {
        "schema": "d128_direct_dual_qkv_weight_preparation_v1",
        "observed_after": "initial_diagnostic_forward_backward",
        "source_contract_schema": expected_schema,
        "expected_layer_count": config.layers,
        "eligible_layer_count": eligible_layer_count,
        "authenticated_layer_count": authenticated_layer_count,
        "abi_identity_bound_layer_count": (
            abi_identity_bound_layer_count
        ),
        "generation_guard_enforced_layer_count": (
            generation_guard_enforced_layer_count
        ),
        "same_stream_enforced_layer_count": (
            same_stream_enforced_layer_count
        ),
        "in_flight_layer_count": 0,
        "schedule": "synchronous_same_stream",
        "source": "canonical_split_qkv_parameters",
        "caller_owned": True,
        "refresh": "every_forward",
        "abi_identity_excludes_tensor_version": True,
        "first_use_authentication": (
            "bitwise_against_pair_interleave_concat_then_independent_"
            "true_2d_quantization"
        ),
        "checked_symbol": checked_symbol,
        "unchecked_symbol": unchecked_symbol,
        "owner_fields": list(owner_fields),
        "owner_count_per_layer": len(owner_fields),
        "all_owner_tensors_private_nonpersistent": True,
        "all_pointers_stable_since_allocation": True,
        "owner_pointers_globally_unique": True,
        "total_bytes": total_bytes,
    }


def _dual_output_weight_preparation_receipt(
    model: Llama12B,
    config: Config,
    runtime: LowpAttentionRuntime | None,
) -> dict[str, Any] | None:
    """Prove generic caller-owned output dual weights on every layer."""
    if runtime is None or not runtime.projection_weight_scale_2d:
        return None
    if config.head_dim not in (64, 128):
        raise RuntimeError(
            "direct-dual output-weight receipt supports only D64 or D128; "
            f"observed D{config.head_dim}"
        )

    contract = model.lowp_forward_workspace_contract()
    expected_schema = "lowp_model_forward_workspaces_v2"
    if contract.get("schema") != expected_schema:
        raise RuntimeError(
            "direct-dual output-weight receipt requires workspace schema "
            f"{expected_schema!r}; observed {contract.get('schema')!r}"
        )
    layers = contract.get("layers")
    if not isinstance(layers, list) or len(layers) != config.layers:
        observed_count = len(layers) if isinstance(layers, list) else None
        raise RuntimeError(
            "direct-dual output-weight receipt requires one workspace "
            f"contract per layer; expected {config.layers}, observed "
            f"{observed_count}"
        )

    checked_symbol = "quantize_nvfp4_projection_weight_dual_out"
    unchecked_symbol = checked_symbol + "_unchecked"
    owner_fields = (
        "output_weight_forward_packed",
        "output_weight_forward_scales",
        "output_weight_backward_packed",
        "output_weight_backward_scales",
        "output_weight_global_scale",
    )
    owner_pointers: list[int] = []
    total_bytes = 0
    for expected_layer, layer in enumerate(layers):
        if not isinstance(layer, dict) or layer.get("layer") != expected_layer:
            observed_layer = (
                layer.get("layer") if isinstance(layer, dict) else None
            )
            raise RuntimeError(
                "direct-dual output-weight workspace layer order mismatch "
                f"at position {expected_layer}; observed {observed_layer!r}"
            )
        lifecycle = layer.get("publication_lifecycle")
        expected_lifecycle = {
            "generation_guard_enforced": True,
            "same_stream_enforced": True,
            "one_forward_in_flight_per_layer": True,
            "in_flight": False,
        }
        lifecycle_mismatches = {
            field: {
                "actual": (
                    lifecycle.get(field)
                    if isinstance(lifecycle, dict)
                    else None
                ),
                "expected": expected,
            }
            for field, expected in expected_lifecycle.items()
            if not isinstance(lifecycle, dict)
            or lifecycle.get(field) != expected
        }
        current_generation = (
            lifecycle.get("current_generation")
            if isinstance(lifecycle, dict)
            else None
        )
        if type(current_generation) is not int or current_generation < 0:
            lifecycle_mismatches["current_generation"] = {
                "actual": current_generation,
                "expected": "non-negative int after initial diagnostic",
            }
        if lifecycle_mismatches:
            raise RuntimeError(
                "direct-dual output-weight lifecycle contract mismatch at "
                f"layer {expected_layer}: {lifecycle_mismatches}"
            )

        preparation = layer.get("dual_output_weight")
        if not isinstance(preparation, dict):
            raise RuntimeError(
                "direct-dual output-weight workspace evidence is missing at "
                f"layer {expected_layer}"
            )
        expected_fields = {
            "eligible": True,
            "authenticated": True,
            "schedule": "synchronous_same_stream",
            "one_forward_in_flight_per_layer": True,
            "generation_guard_enforced": True,
            "same_stream_enforced": True,
            "abi_identity_bound": True,
            "abi_identity_tensor_count": 6,
            "abi_identity_excludes_tensor_version": True,
            "checked_symbol": checked_symbol,
            "unchecked_symbol": unchecked_symbol,
            "all_pointers_stable_since_allocation": True,
            "all_pointers_unique": True,
        }
        mismatches = {
            field: {
                "actual": preparation.get(field),
                "expected": expected,
            }
            for field, expected in expected_fields.items()
            if preparation.get(field) != expected
        }
        if mismatches:
            raise RuntimeError(
                "direct-dual output-weight preparation contract mismatch at "
                f"layer {expected_layer}: {mismatches}"
            )

        owners = preparation.get("owners")
        if not isinstance(owners, dict) or set(owners) != set(owner_fields):
            observed_fields = (
                sorted(owners) if isinstance(owners, dict) else None
            )
            raise RuntimeError(
                "direct-dual output-weight owner fields mismatch at layer "
                f"{expected_layer}; observed {observed_fields!r}"
            )
        layer_bytes = 0
        for owner_name in owner_fields:
            owner = owners[owner_name]
            if not isinstance(owner, dict):
                raise RuntimeError(
                    "direct-dual output-weight owner evidence must be a "
                    f"mapping; layer {expected_layer}, owner {owner_name!r}"
                )
            if owner.get("pointer_stable_since_allocation") is not True:
                raise RuntimeError(
                    "direct-dual output-weight owner pointer is not stable; "
                    f"layer {expected_layer}, owner {owner_name!r}"
                )
            visibility = {
                field: owner.get(field)
                for field in (
                    "listed_in_named_buffers",
                    "listed_in_named_parameters",
                    "optimizer_visible_parameter",
                )
            }
            if any(value is not False for value in visibility.values()):
                raise RuntimeError(
                    "direct-dual output-weight owner is not private scratch; "
                    f"layer {expected_layer}, owner {owner_name!r}, "
                    f"visibility={visibility}"
                )
            pointer = owner.get("data_ptr")
            byte_count = owner.get("bytes")
            if type(pointer) is not int or pointer <= 0:
                raise RuntimeError(
                    "direct-dual output-weight owner has no valid allocation "
                    f"identity; layer {expected_layer}, owner {owner_name!r}"
                )
            if type(byte_count) is not int or byte_count <= 0:
                raise RuntimeError(
                    "direct-dual output-weight owner has no positive byte "
                    f"count; layer {expected_layer}, owner {owner_name!r}"
                )
            owner_pointers.append(pointer)
            layer_bytes += byte_count
        if preparation.get("total_bytes") != layer_bytes:
            raise RuntimeError(
                "direct-dual output-weight byte accounting mismatch at "
                f"layer {expected_layer}; owners sum to {layer_bytes}, "
                f"observed {preparation.get('total_bytes')!r}"
            )
        total_bytes += layer_bytes

    if len(set(owner_pointers)) != len(owner_pointers):
        raise RuntimeError(
            "direct-dual output-weight owner allocations are not globally "
            "unique"
        )

    return {
        "schema": "direct_dual_output_weight_preparation_v1",
        "observed_after": "initial_diagnostic_forward_backward",
        "source_contract_schema": expected_schema,
        "observed_head_dim": config.head_dim,
        "eligible_head_dims": [64, 128],
        "expected_layer_count": config.layers,
        "eligible_layer_count": config.layers,
        "authenticated_layer_count": config.layers,
        "abi_identity_bound_layer_count": config.layers,
        "generation_guard_enforced_layer_count": config.layers,
        "same_stream_enforced_layer_count": config.layers,
        "in_flight_layer_count": 0,
        "function": "b300_prepare_nvfp4_projection_weight_dual_out",
        "source": "output_projection_weight_parameter",
        "forward_operand": {
            "format": "nvfp4",
            "physical_layout": "canonical_output_weight",
            "scale_layout": "true_16x16",
        },
        "backward_operand": {
            "format": "nvfp4",
            "physical_layout": "physical_transpose_output_weight",
            "scale_layout": "true_16x16",
        },
        "caller_owned": True,
        "shared_global_scale": True,
        "refresh": "every_forward",
        "abi_identity_excludes_tensor_version": True,
        "first_use_authentication": (
            "bitwise_against_independent_true_2d_forward_and_physical_"
            "transpose_quantization"
        ),
        "checked_symbol": checked_symbol,
        "unchecked_symbol": unchecked_symbol,
        "owner_fields": list(owner_fields),
        "owner_count_per_layer": len(owner_fields),
        "all_owner_tensors_private_nonpersistent": True,
        "all_pointers_stable_since_allocation": True,
        "owner_pointers_globally_unique": True,
        "total_bytes": total_bytes,
    }


def _cce_identity() -> dict[str, Any]:
    package_root = Path(cut_cross_entropy.__file__).resolve().parent
    compiled_source = package_root / "torch_compile.py"
    return {
        "distribution": "cut-cross-entropy",
        "version": importlib.metadata.version("cut-cross-entropy"),
        "package_root": str(package_root),
        "torch_compile_source": str(compiled_source),
        "torch_compile_source_sha256": _sha256(compiled_source),
        "implementation": "torch_compile",
        "logical_full_logits": True,
        "source_operation": "logits = e @ c.T",
        "inductor_logit_buffer_elision_proven": False,
    }


def _hardware_identity() -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    identity = {
        "name": properties.name,
        "uuid": str(properties.uuid),
        "compute_capability": [properties.major, properties.minor],
        "multiprocessor_count": properties.multi_processor_count,
        "total_memory_bytes": properties.total_memory,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pid": os.getpid(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "torch_cpu_threads": torch.get_num_threads(),
    }
    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,pstate,clocks.sm,clocks.mem,"
                "power.draw,power.limit,memory.total,memory.used,"
                "utilization.gpu,compute_mode",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        identity["nvidia_smi_snapshot"] = [
            line.strip() for line in query.stdout.splitlines() if line.strip()
        ]
    except (OSError, subprocess.CalledProcessError) as error:
        identity["nvidia_smi_error"] = type(error).__name__
    try:
        processes = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        identity["compute_processes"] = [
            line.strip()
            for line in processes.stdout.splitlines()
            if line.strip()
        ]
    except (OSError, subprocess.CalledProcessError) as error:
        identity["compute_process_query_error"] = type(error).__name__
    return identity


def _uses_packed_qkv(model: Llama12B) -> bool:
    """Require one physical QKV schema across every decoder layer."""
    packed_layers = tuple(
        isinstance(layer.attention.weights, PackedQKVAttentionWeights)
        for layer in model.layers
    )
    if not packed_layers:
        raise RuntimeError("packed-QKV schema inspection requires decoder layers")
    if any(packed_layers) and not all(packed_layers):
        raise RuntimeError("decoder layers mix packed and split QKV schemas")
    return all(packed_layers)


def _canonical_parameter_tensors(
    model: Llama12B,
) -> dict[str, torch.Tensor]:
    tensors = dict(model.named_parameters())
    if _uses_packed_qkv(model):
        tensors = dict(
            canonical_split_qkv_tensors(
                tensors,
                packed_qkv_layout(model.config),
            )
        )
    return tensors


def _canonical_gradient_tensors(
    model: Llama12B,
) -> dict[str, torch.Tensor]:
    gradients = {
        name: parameter.grad
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    if _uses_packed_qkv(model):
        gradients = dict(
            canonical_split_qkv_tensors(
                gradients,
                packed_qkv_layout(model.config),
            )
        )
    return gradients


def _canonical_state_tensors(
    model: Llama12B,
) -> dict[str, torch.Tensor]:
    tensors = dict(model.state_dict())
    if _uses_packed_qkv(model):
        tensors = dict(
            canonical_split_qkv_tensors(
                tensors,
                packed_qkv_layout(model.config),
            )
        )
    return tensors


def _parameter_schema(model: Llama12B) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "requires_grad": parameter.requires_grad,
        }
        for name, parameter in _canonical_parameter_tensors(model).items()
    ]


def _write_model_checkpoint(
    model: Llama12B,
    path: Path,
    *,
    kind: str,
) -> dict[str, Any]:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        state = model.state_dict()
        if _uses_packed_qkv(model):
            state = unpack_qkv_state_dict(
                state,
                packed_qkv_layout(model.config),
            )
        torch.save(state, temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "kind": kind,
        "serialized_state_layout": "canonical_split_qkv",
        "runtime_state_layout": (
            "packed_qkv" if _uses_packed_qkv(model) else "split_qkv"
        ),
    }


def _save_initial_checkpoint(model: Llama12B, path: Path) -> dict[str, Any]:
    receipt = _write_model_checkpoint(model, path, kind="shared_initial_state")
    receipt["created_by_route"] = True
    return receipt


def _load_initial_checkpoint(model: Llama12B, path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(
            f"initial checkpoint must be a regular non-symlink file: {path}"
        )
    state = torch.load(resolved, map_location="cpu", weights_only=True)
    expected_canonical = _canonical_state_tensors(model)
    if set(state) != set(expected_canonical):
        missing = sorted(set(expected_canonical) - set(state))
        extra = sorted(set(state) - set(expected_canonical))
        raise RuntimeError(
            "initial checkpoint must use the canonical split-QKV schema: "
            f"missing={missing[:4]}, extra={extra[:4]}"
        )
    for name, expected in expected_canonical.items():
        actual = state[name]
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise RuntimeError(
                f"initial checkpoint tensor {name} has shape/dtype "
                f"{tuple(actual.shape)}/{actual.dtype}, expected "
                f"{tuple(expected.shape)}/{expected.dtype}"
            )
    if _uses_packed_qkv(model):
        state = pack_qkv_state_dict(
            state,
            packed_qkv_layout(model.config),
        )
    if set(state) != set(model.state_dict()):
        missing = sorted(set(model.state_dict()) - set(state))
        extra = sorted(set(state) - set(model.state_dict()))
        raise RuntimeError(
            "initial checkpoint parameter schema mismatch: "
            f"missing={missing[:4]}, extra={extra[:4]}"
        )
    model.load_state_dict(state, strict=True)
    del state
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "bytes": resolved.stat().st_size,
        "created_by_route": False,
        "kind": "shared_initial_state",
        "serialized_state_layout": "canonical_split_qkv",
        "runtime_state_layout": (
            "packed_qkv" if _uses_packed_qkv(model) else "split_qkv"
        ),
    }


def _loss(hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return linear_cross_entropy(
        hidden.reshape(-1, hidden.shape[-1]),
        weight,
        targets.reshape(-1),
        reduction="mean",
        filter_eps=0.0,
        impl="torch_compile",
    )


def _hidden_and_weight(
    model: Llama12B,
    tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    runtime = model.lowp_attention_runtime
    if runtime is not None:
        require_active_forward_route(str(runtime.forward_topology["route"]))
    hidden = F.embedding(tokens, model.embedding)
    for layer in model.layers:
        hidden = layer(hidden)
    hidden = model.final_norm(hidden)
    weight = model.embedding if model.lm_head is None else model.lm_head
    assert weight is not None
    return hidden, weight


class _DistributedHiddenAndWeight(torch.nn.Module):
    """Expose the CCE hidden-state boundary through ``DDP.forward``.

    The saturated harness deliberately avoids materializing full vocabulary
    logits, so wrapping ``Llama12B.forward`` would benchmark a different loss
    path.  This adapter keeps the established hidden+weight contract while
    allowing DistributedDataParallel to prepare and overlap its reducer.
    """

    def __init__(self, model: Llama12B) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _hidden_and_weight(self.model, tokens)


def _ranked_output_path(path: Path, rank: int) -> Path:
    """Return a collision-free per-rank artifact path."""
    return path.with_name(f"{path.stem}.rank{rank}{path.suffix}")


def _batch(seed: int, step: int, batch: int, sequence: int, vocab: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed + 10_000 + step)
    packed = torch.randint(
        vocab,
        (batch, sequence + 1),
        generator=generator,
        device="cuda",
    )
    return packed[:, :-1].contiguous(), packed[:, 1:].contiguous()


def _dolma_batches(
    corpus_path: Path,
    tokenizer_path: Path,
    *,
    seed: int,
    count: int,
    batch: int,
    sequence: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    corpus_sha = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    tokenizer_sha = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    if corpus_sha != PINNED_DATA["corpus"]:
        raise RuntimeError("Dolma corpus SHA-256 mismatch")
    if tokenizer_sha != PINNED_DATA["tokenizer"]:
        raise RuntimeError("tokenizer SHA-256 mismatch")
    documents: list[str] = []
    for line_number, line in enumerate(corpus_path.read_text().splitlines(), start=1):
        try:
            text = str(json.loads(line).get("text", ""))
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"invalid JSON at {corpus_path}:{line_number}"
            ) from error
        if text.strip():
            documents.append(text)
    order = list(range(len(documents)))
    random.Random(seed).shuffle(order)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    required = count * batch * (sequence + 1)
    stream: list[int] = []
    consumed = 0
    for document_index in order:
        encoded = tokenizer.encode(
            documents[document_index], add_special_tokens=False
        ).ids
        stream.append(128000)
        stream.extend(int(token) for token in encoded)
        stream.append(128001)
        consumed += 1
        if len(stream) >= required:
            break
    if len(stream) < required:
        raise RuntimeError(
            f"Dolma shard supplied {len(stream):,} packed tokens; "
            f"need {required:,}"
        )
    packed_cpu = torch.tensor(stream[:required], dtype=torch.int64).reshape(
        count, batch, sequence + 1
    )
    packed_sha = hashlib.sha256(packed_cpu.numpy().tobytes()).hexdigest()
    packed = packed_cpu.cuda()
    return (
        packed[:, :, :-1].contiguous(),
        packed[:, :, 1:].contiguous(),
        {
            "kind": "pinned_local_dolma_jsonl",
            "corpus_path": str(corpus_path.resolve()),
            "corpus_sha256": corpus_sha,
            "tokenizer_path": str(tokenizer_path.resolve()),
            "tokenizer_sha256": tokenizer_sha,
            "source_documents": len(documents),
            "documents_consumed": consumed,
            "packed_sha256": packed_sha,
            "updates_including_probe": count,
            "batch": batch,
            "sequence": sequence,
        },
    )


def _sample_named_tensors(
    tensors: dict[str, torch.Tensor],
    names: tuple[str, ...],
    elements: int = SAMPLE_ELEMENTS_PER_PARAMETER,
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for name in names:
        if name not in tensors:
            raise RuntimeError(f"required sampled tensor is missing: {name}")
        flattened = tensors[name].detach().reshape(-1)
        if flattened.numel() <= elements:
            sample = flattened
        else:
            indices = (
                torch.arange(elements, device=flattened.device, dtype=torch.int64)
                * (flattened.numel() - 1)
                // (elements - 1)
            )
            sample = flattened.index_select(0, indices)
        result[name] = sample.float().cpu()
    return result


def _parameter_samples(
    model: Llama12B,
    names: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    return _sample_named_tensors(_canonical_parameter_tensors(model), names)


def _gradient_samples(
    model: Llama12B,
    names: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    return _sample_named_tensors(_canonical_gradient_tensors(model), names)


def _compare_samples(
    candidate: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
) -> dict[str, Any]:
    if set(candidate) != set(reference):
        raise RuntimeError(
            "sample key mismatch: "
            f"candidate_only={sorted(set(candidate) - set(reference))}, "
            f"reference_only={sorted(set(reference) - set(candidate))}"
        )
    result: dict[str, Any] = {}
    for name in sorted(candidate):
        if candidate[name].shape != reference[name].shape:
            raise RuntimeError(
                f"sample shape mismatch for {name}: "
                f"{tuple(candidate[name].shape)} != "
                f"{tuple(reference[name].shape)}"
            )
        actual = candidate[name].double().reshape(-1)
        wanted = reference[name].double().reshape(-1)
        denominator = wanted.norm().clamp_min(1.0e-30)
        cosine_denominator = (actual.norm() * wanted.norm()).clamp_min(1.0e-30)
        result[name] = {
            "cosine": float(torch.dot(actual, wanted) / cosine_denominator),
            "relative_l2": float((actual - wanted).norm() / denominator),
            "norm_ratio": float(actual.norm() / denominator),
            "max_abs": float((actual - wanted).abs().max()),
        }
    return result


def _subtract_samples(
    final: dict[str, torch.Tensor],
    initial: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if set(final) != set(initial):
        raise RuntimeError("cannot form update vectors from mismatched samples")
    return {name: final[name] - initial[name] for name in sorted(final)}


def _gradient_statistics(
    model: Llama12B,
    sampled_parameter_names: tuple[str, ...],
) -> dict[str, Any]:
    squared_norm = torch.zeros((), device="cuda", dtype=torch.float64)
    finite = torch.ones((), device="cuda", dtype=torch.bool)
    maximum = torch.zeros((), device="cuda", dtype=torch.float32)
    parameter_norms: dict[str, float] = {}
    parameters_with_gradients = 0
    elements = 0
    for name, gradient in _canonical_gradient_tensors(model).items():
        parameters_with_gradients += 1
        elements += gradient.numel()
        gradient_float = gradient.detach().float()
        squared_norm += gradient_float.double().square().sum()
        finite &= torch.isfinite(gradient_float).all()
        maximum = torch.maximum(maximum, gradient_float.abs().max())
        if name in sampled_parameter_names:
            parameter_norms[name] = float(gradient_float.norm())
    return {
        "global_l2": float(squared_norm.sqrt()),
        "max_abs": float(maximum),
        "finite": bool(finite),
        "parameters_with_gradients": parameters_with_gradients,
        "elements": elements,
        "sampled_parameter_l2": parameter_norms,
    }


def _selected_hidden_and_logits(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    batch_indices: tuple[int, ...],
    positions: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = hidden[
        torch.tensor(batch_indices, device=hidden.device)[:, None],
        torch.tensor(positions, device=hidden.device)[None, :],
    ].reshape(-1, hidden.shape[-1])
    logits = F.linear(selected, weight)
    return selected.detach().float().cpu(), logits.detach().float().cpu()


def _diagnostic_pass(
    model: Llama12B,
    optimizer: torch.optim.Optimizer,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    sampled_parameter_names: tuple[str, ...],
    hidden_sample_batches: tuple[int, ...],
    hidden_sample_positions: tuple[int, ...],
    distributed_forward: torch.nn.Module | None = None,
) -> dict[str, Any]:
    optimizer.zero_grad(set_to_none=True)
    hidden, weight = (
        distributed_forward(tokens)
        if distributed_forward is not None
        else _hidden_and_weight(model, tokens)
    )
    loss = _loss(hidden, weight, targets)
    selected_hidden, selected_logits = _selected_hidden_and_logits(
        hidden,
        weight,
        hidden_sample_batches,
        hidden_sample_positions,
    )
    loss.backward()
    result = {
        "loss": float(loss.detach()),
        "hidden": {"selected_hidden": selected_hidden},
        "logits": {"selected_logits": selected_logits},
        "gradients": _gradient_samples(model, sampled_parameter_names),
        "gradient_statistics": _gradient_statistics(
            model, sampled_parameter_names
        ),
    }
    optimizer.zero_grad(set_to_none=True)
    del hidden, loss, selected_hidden, selected_logits
    torch.cuda.synchronize()
    return result


def _compare_logits(
    candidate: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, Any]:
    if candidate.shape != reference.shape:
        raise RuntimeError(
            f"sampled-logit shape mismatch: {candidate.shape} != {reference.shape}"
        )
    candidate_float = candidate.float()
    reference_float = reference.float()
    reference_log_probability = F.log_softmax(reference_float, dim=-1)
    candidate_log_probability = F.log_softmax(candidate_float, dim=-1)
    reference_probability = reference_log_probability.exp()
    kl_rows = (
        reference_probability
        * (reference_log_probability - candidate_log_probability)
    ).sum(dim=-1)
    return {
        "reference_to_candidate_kl_mean": float(kl_rows.mean()),
        "reference_to_candidate_kl_max": float(kl_rows.max()),
        "top1_agreement": float(
            (
                candidate_float.argmax(dim=-1)
                == reference_float.argmax(dim=-1)
            ).float().mean()
        ),
        "sample_metrics": _compare_samples(
            {"selected_logits": candidate_float},
            {"selected_logits": reference_float},
        )["selected_logits"],
    }


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sample")
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _timing_statistics(
    records: list[dict[str, Any]],
    fields: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for field in fields:
        values = [float(record[field]) for record in records]
        mean = statistics.fmean(values)
        standard_deviation = statistics.stdev(values) if len(values) > 1 else 0.0
        result[field] = {
            "p10": _percentile(values, 0.10),
            "p50": _percentile(values, 0.50),
            "p90": _percentile(values, 0.90),
            "mean": mean,
            "standard_deviation": standard_deviation,
            "coefficient_of_variation": (
                standard_deviation / mean if mean != 0.0 else float("nan")
            ),
            "minimum": min(values),
            "maximum": max(values),
        }
    return result


def _comparison_scope(route: str, reference_route: str | None) -> str:
    if route == "bf16_packed":
        return (
            "BF16 projection-topology control: one packed QKV linear and "
            "BF16 CuTe FA4 versus the canonical split-QKV BF16 route"
        )
    if route in LOWP_ROUTES and reference_route == "bf16_packed":
        return (
            "single-QKV-projection control: packed BF16 projection and "
            "BF16 CuTe FA4 versus fused low-precision QKV publication and "
            "low-precision FA4"
        )
    if route in LOWP_ROUTES:
        return (
            "system recipe: fused low-precision QKV publication and FA4 "
            "versus eager BF16 Q/K/V projections and BF16 attention; use "
            "the isolated matrix for attention-format-only claims"
        )
    return "canonical split-QKV BF16 CuTe FA4 reference route"


def _require_reference_identity(
    candidate: dict[str, Any],
    reference: dict[str, Any],
) -> None:
    expected_keys = {
        "schema",
        "route",
        "comparison_identity",
        "checkpoint",
        "initial_parameters",
        "initial_diagnostic",
        "final_parameters",
        "parameter_updates",
        "final_diagnostic",
        "losses",
    }
    if set(reference) != expected_keys:
        raise RuntimeError(
            "reference sample schema keys mismatch: "
            f"missing={sorted(expected_keys - set(reference))}, "
            f"extra={sorted(set(reference) - expected_keys)}"
        )
    if reference["schema"] != "llama12b_saturated_route_samples_v2":
        raise RuntimeError("reference sample schema version mismatch")
    if reference["route"] not in BF16_ROUTES:
        raise RuntimeError("reference samples must come from a BF16 route")
    if candidate["comparison_identity"] != reference["comparison_identity"]:
        raise RuntimeError(
            "reference comparison identity mismatch; config, data, checkpoint, "
            "optimizer, update count, and sample layout must match exactly"
        )
    for key in ("sha256", "bytes"):
        if candidate["checkpoint"][key] != reference["checkpoint"][key]:
            raise RuntimeError(f"reference checkpoint {key} mismatch")
    if len(candidate["losses"]) != len(reference["losses"]):
        raise RuntimeError("reference loss trajectory length mismatch")


def _gradient_statistic_deltas(
    candidate: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    if set(candidate["sampled_parameter_l2"]) != set(
        reference["sampled_parameter_l2"]
    ):
        raise RuntimeError("gradient-statistic parameter keys mismatch")
    return {
        "global_l2_delta": (
            float(candidate["global_l2"]) - float(reference["global_l2"])
        ),
        "global_l2_ratio": (
            float(candidate["global_l2"])
            / max(float(reference["global_l2"]), 1.0e-30)
        ),
        "max_abs_delta": (
            float(candidate["max_abs"]) - float(reference["max_abs"])
        ),
        "both_finite": bool(candidate["finite"] and reference["finite"]),
        "sampled_parameter_l2_ratios": {
            name: (
                float(candidate["sampled_parameter_l2"][name])
                / max(
                    float(reference["sampled_parameter_l2"][name]),
                    1.0e-30,
                )
            )
            for name in sorted(candidate["sampled_parameter_l2"])
        },
    }


def _runtime(
    route: str,
    config: Any,
    forward_path: Path,
    forward_module: str,
    forward_expected_identity: tuple[str, int],
    control_path: Path | None,
    loss_scale: float,
    qkv_projection_format: str,
    experimental_native_nvfp4_projection_out: bool,
    experimental_fused_attention_rmsnorm_nvfp4: bool,
    experimental_output_shared_split_v: bool | None,
    experimental_d128_mxfp4_v_backward: bool = False,
    experimental_d128_shared_tile_mxfp4_v: bool = False,
    native_tk_d64_backward_extension: Any | None = None,
    native_tk_d128_backward_extension: Any | None = None,
) -> tuple[LowpAttentionRuntime, dict[str, Any]]:
    is_d128 = config.head_dim == 128
    native_tk_d64_backward = native_tk_d64_backward_extension is not None
    native_tk_d128_backward = native_tk_d128_backward_extension is not None
    if native_tk_d64_backward and native_tk_d128_backward:
        raise ValueError("select exactly one native TK backward extension")
    if native_tk_d64_backward:
        if is_d128 or route not in LOWP_ROUTES:
            raise ValueError(
                "native TK D64 backward requires a D64 low-precision route"
            )
        if control_path is not None:
            raise ValueError(
                "native TK D64 backward does not accept a CuTe backward control"
            )
    if native_tk_d128_backward:
        if not is_d128 or route not in ("fp8", "mx"):
            raise ValueError(
                "native TK D128 backward requires a D128 FP8-PV or MXFP4-PV "
                "route"
            )
        if (
            experimental_d128_shared_tile_mxfp4_v
            and not experimental_d128_mxfp4_v_backward
        ):
            raise ValueError(
                "shared-tile MXFP4 V publication requires MXFP4 V backward"
            )
        if control_path is not None:
            raise ValueError(
                "native TK D128 backward does not accept a CuTe backward control"
            )
        if experimental_d128_mxfp4_v_backward and (
            route != "mx" or config.batch != 2
        ):
            raise ValueError(
                "native TK D128 MXFP4-V backward requires the B2 MXFP4-PV "
                "route"
            )
    if is_d128 and route == "mx_unanchored":
        raise ValueError("mx_unanchored is authenticated only for D64/B16")
    if route == "mx_unanchored":
        extension, topology, _identity = load_authenticated_mx_extension(
            forward_path,
            module_name=forward_module,
            expected_sha256=forward_expected_identity[0],
            expected_bytes=forward_expected_identity[1],
            variant="unanchored-splitmix-v6",
            batch=config.batch,
        )
    else:
        extension, topology = _load_forward(
            forward_path, forward_module, config
        )
    forward_loaded_identity = _require_loaded_artifact_identity(
        "forward",
        extension,
        forward_path,
        forward_expected_identity,
    )
    expected_pv_format = (
        "mxfp4_e8m0_block32" if route in MX_ROUTES else "e4m3_fp8"
    )
    if topology.get("pv_format") != expected_pv_format:
        raise RuntimeError(
            f"{route} forward artifact publishes {topology.get('pv_format')!r}; "
            f"expected {expected_pv_format!r}"
        )
    if is_d128 and control_path is not None:
        raise ValueError("D128 must generate its shared-P backward control")
    use_generated_or_native_control = (
        is_d128 or native_tk_d64_backward or native_tk_d128_backward
    )
    control_sha256 = (
        None
        if use_generated_or_native_control
        else PINNED_ARTIFACTS["control"][0]
    )
    control_bytes = (
        None
        if use_generated_or_native_control
        else PINNED_ARTIFACTS["control"][1]
    )
    runtime = LowpAttentionRuntime(
        config,
        _make_llama3_rope(config),
        forward_extension=extension,
        forward_topology=topology,
        loss_scale=loss_scale,
        gradient_global_scale=2.0**-8,
        projection_dgrad="nvfp4" if is_d128 else "bf16",
        qkv_projection_format=qkv_projection_format,
        experimental_native_nvfp4_projection_out=(
            experimental_native_nvfp4_projection_out
        ),
        experimental_fused_attention_rmsnorm_nvfp4=(
            experimental_fused_attention_rmsnorm_nvfp4
        ),
        backward_exp2_degree=1,
        backward_exp2_period=0 if is_d128 else 2,
        backward_fp8_ds_lift=16,
        backward_reuse_quantized_p=(
            is_d128 and not native_tk_d128_backward
        ),
        backward_control_source=control_path,
        backward_control_sha256=control_sha256,
        backward_control_bytes=control_bytes,
        backward_forward_mx_probability_replay=False,
        backward_forward_mx_probability_scale_handoff=False,
        backward_match_forward_operands=not is_d128,
        # Fixed-head D128 Q/K scales pass isolated kernels but compound
        # materially across the 32-layer model. Row-by-K16 publication is
        # therefore part of the authenticated saturated recipe at both D64
        # and D128; it does not alter the shared E4M3 backward publication.
        per_block_qk_scales=True,
        experimental_split_v_backward=(route in MX_ROUTES) if not is_d128 else False,
        experimental_output_shared_split_v=(
            experimental_output_shared_split_v
        ),
        experimental_d128_mxfp4_v_backward=(
            experimental_d128_mxfp4_v_backward
        ),
        backward_probability_correction=1.0,
        q_quant_scale=2.25,
        k_quant_scale=2.0,
        projection_weight_scale_2d=True,
        v_mxfp4_scale_2d=experimental_d128_shared_tile_mxfp4_v,
        adaptive_qk_weight_scales=False,
        native_tk_d64_backward_extension=(
            native_tk_d64_backward_extension
        ),
        native_tk_d128_backward_extension=(
            native_tk_d128_backward_extension
        ),
    )
    runtime.forward_loaded_artifact_identity = forward_loaded_identity
    return runtime, topology


def _timed_update(
    model: Llama12B,
    optimizer: torch.optim.Optimizer,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    *,
    update: int,
    warmup: bool,
    max_grad_norm: float,
    profile: bool,
    distributed_forward: torch.nn.Module | None = None,
) -> dict[str, Any]:
    events = [torch.cuda.Event(enable_timing=True) for _ in range(6)]
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    profile_context = (
        torch.cuda.nvtx.range("profile_step")
        if profile
        else contextlib.nullcontext()
    )
    with profile_context:
        events[0].record()
        optimizer.zero_grad(set_to_none=True)
        decoder_context = (
            torch.cuda.nvtx.range("decoder_forward")
            if profile
            else contextlib.nullcontext()
        )
        with decoder_context:
            hidden, weight = (
                distributed_forward(tokens)
                if distributed_forward is not None
                else _hidden_and_weight(model, tokens)
            )
        events[1].record()
        ce_context = (
            torch.cuda.nvtx.range("ce_forward")
            if profile
            else contextlib.nullcontext()
        )
        with ce_context:
            loss = _loss(hidden, weight, targets)
        events[2].record()
        backward_context = (
            torch.cuda.nvtx.range("backward_total")
            if profile
            else contextlib.nullcontext()
        )
        with backward_context:
            loss.backward()
        events[3].record()
        clip_context = (
            torch.cuda.nvtx.range("gradient_clip")
            if profile
            else contextlib.nullcontext()
        )
        with clip_context:
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_grad_norm,
                error_if_nonfinite=True,
                foreach=True,
            )
        events[4].record()
        optimizer_context = (
            torch.cuda.nvtx.range("optimizer")
            if profile
            else contextlib.nullcontext()
        )
        with optimizer_context:
            optimizer.step()
        events[5].record()
    events[5].synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    loss_value = float(loss.detach())
    record = {
        "update": update,
        "warmup": warmup,
        "loss": loss_value,
        "finite": math.isfinite(loss_value),
        "decoder_forward_ms": float(events[0].elapsed_time(events[1])),
        "ce_forward_ms": float(events[1].elapsed_time(events[2])),
        "backward_ms": float(events[2].elapsed_time(events[3])),
        "gradient_clip_ms": float(events[3].elapsed_time(events[4])),
        "optimizer_ms": float(events[4].elapsed_time(events[5])),
        "step_ms": float(events[0].elapsed_time(events[5])),
        "wall_ms": wall_ms,
        "gradient_norm_before_clip": float(gradient_norm),
    }
    del hidden, loss
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-preset",
        choices=MODEL_PRESETS,
        default=DEFAULT_MODEL_PRESET,
        help=(
            "llama3.2-1b selects the authenticated B16/D64 shape; "
            "llama3.1-8b selects the full-depth B1/B2 D128 shapes"
        ),
    )
    parser.add_argument(
        "--route", choices=(*BF16_ROUTES, *LOWP_ROUTES), required=True
    )
    parser.add_argument("--batch", type=int, default=16, choices=(1, 2, 16))
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--updates", type=int, default=MINIMUM_MEASURED_UPDATES)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--learning-rate", type=float, default=0.00048828125)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--loss-scale", type=float, default=65_536.0)
    parser.add_argument("--max-hbm-gib", type=float, default=180.0)
    parser.add_argument("--trial-label", required=True)
    parser.add_argument(
        "--distributed-data-parallel",
        action="store_true",
        help=(
            "run one authenticated local microbatch per torchrun rank and "
            "synchronize gradients with NCCL DDP"
        ),
    )
    parser.add_argument(
        "--ddp-bucket-cap-mb",
        type=int,
        default=512,
        help="DDP gradient bucket size; large buckets limit launch overhead",
    )
    parser.add_argument(
        "--profile-update",
        type=int,
        help="absolute update index to annotate with nested NVTX ranges",
    )
    parser.add_argument("--forward-extension", type=Path)
    parser.add_argument("--forward-module")
    parser.add_argument("--forward-sha256")
    parser.add_argument("--forward-bytes", type=int)
    parser.add_argument("--diagnostic-fp8-lse-extension", type=Path)
    parser.add_argument("--diagnostic-fp8-lse-module")
    parser.add_argument("--diagnostic-fp8-lse-sha256")
    parser.add_argument("--diagnostic-fp8-lse-bytes", type=int)
    parser.add_argument(
        "--diagnostic-fp8-lse-substitution-mode",
        choices=DIAGNOSTIC_FP8_LSE_SUBSTITUTION_MODES,
        default="all_rows",
        help=(
            "diagnostic-only FP8 control selection; all_rows preserves the "
            "existing behavior, while mx_nonfinite_only retains every "
            "finite MX LSE entry"
        ),
    )
    parser.add_argument(
        "--d128-forward-topology-variant",
        choices=D128_FORWARD_TOPOLOGY_VARIANTS,
        default="production",
        help=(
            "explicit D128 MX topology recipe; non-production choices are "
            "authenticated single-pass diagnostic candidates"
        ),
    )
    parser.add_argument("--projection-extension", type=Path, default=DEFAULT_PROJECTION)
    parser.add_argument("--projection-sha256")
    parser.add_argument("--projection-bytes", type=int)
    parser.add_argument(
        "--qkv-projection-format",
        choices=("e4m3", "nvfp4"),
        default="e4m3",
        help=(
            "select the fused QKV projection operand format; native NVFP4 "
            "also requires the explicit experimental opt-in"
        ),
    )
    parser.add_argument(
        "--experimental-native-nvfp4-projection-out",
        action="store_true",
        help=(
            "enable the provenance-bound native-NVFP4 caller-owned projection; "
            "D128 publishes route-selective forward V with shared E4M3 Q/K/V "
            "backward"
        ),
    )
    parser.add_argument(
        "--experimental-fused-attention-rmsnorm-nvfp4",
        action="store_true",
        help=(
            "fuse attention RMSNorm with exact-dynamic native-NVFP4 input "
            "preparation; requires the explicit native-NVFP4 projection arm"
        ),
    )
    parser.add_argument(
        "--experimental-output-shared-split-v",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "select output-shared MXFP4 forward V and E4M3 backward V; the "
            "explicit opt-in is accepted only for the eligible native-NVFP4 "
            "MX route; omission keeps the validated D64/B16 automatic path "
            "but retains the established D128 publisher"
        ),
    )
    parser.add_argument(
        "--experimental-d128-mxfp4-v-backward",
        action="store_true",
        help=(
            "consume one rowwise MXFP4 V publication in D128 dP while "
            "retaining the E4M3 dO/P dV path; requires either the separately "
            "authenticated CuTe patch or authenticated native-TK v503 "
            "artifact"
        ),
    )
    parser.add_argument(
        "--experimental-d128-shared-tile-mxfp4-v",
        action="store_true",
        help=(
            "reuse each forward D32xS32 MXFP4 V quantization for the "
            "row-major backward publication; requires the explicit D128 "
            "MXFP4-V backward route and its scale-policy-aware consumer"
        ),
    )
    parser.add_argument("--backward-control", type=Path)
    parser.add_argument("--native-tk-d64-backward-extension", type=Path)
    parser.add_argument("--native-tk-d64-backward-module")
    parser.add_argument("--native-tk-d64-backward-sha256")
    parser.add_argument("--native-tk-d64-backward-bytes", type=int)
    parser.add_argument("--native-tk-d128-backward-extension", type=Path)
    parser.add_argument("--native-tk-d128-backward-module")
    parser.add_argument("--native-tk-d128-backward-sha256")
    parser.add_argument("--native-tk-d128-backward-bytes", type=int)
    parser.add_argument("--tokens", choices=("dolma", "synthetic"), default="dolma")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--reference-samples", type=Path)
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument("--initial-checkpoint", type=Path)
    checkpoint_group.add_argument("--save-initial-checkpoint", type=Path)
    parser.add_argument("--save-final-checkpoint", type=Path)
    parser.add_argument("--samples-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    native_tk_d64_backward_values = (
        args.native_tk_d64_backward_extension,
        args.native_tk_d64_backward_module,
        args.native_tk_d64_backward_sha256,
        args.native_tk_d64_backward_bytes,
    )
    if any(
        value is not None for value in native_tk_d64_backward_values
    ) and not all(
        value is not None for value in native_tk_d64_backward_values
    ):
        raise ValueError(
            "native TK D64 backward requires extension, module, SHA256, and "
            "byte count together"
        )
    native_tk_d64_backward = all(
        value is not None for value in native_tk_d64_backward_values
    )
    native_tk_d64_backward_expected: tuple[str, int] | None = None
    if native_tk_d64_backward:
        assert args.native_tk_d64_backward_module is not None
        if not args.native_tk_d64_backward_module.strip():
            raise ValueError(
                "--native-tk-d64-backward-module must be non-empty"
            )
        native_tk_d64_backward_expected = _caller_declared_expected_identity(
            "native-tk-d64-backward",
            args.native_tk_d64_backward_sha256,
            args.native_tk_d64_backward_bytes,
        )
    native_tk_d128_backward_values = (
        args.native_tk_d128_backward_extension,
        args.native_tk_d128_backward_module,
        args.native_tk_d128_backward_sha256,
        args.native_tk_d128_backward_bytes,
    )
    if any(
        value is not None for value in native_tk_d128_backward_values
    ) and not all(
        value is not None for value in native_tk_d128_backward_values
    ):
        raise ValueError(
            "native TK D128 backward requires extension, module, SHA256, and "
            "byte count together"
        )
    native_tk_d128_backward = all(
        value is not None for value in native_tk_d128_backward_values
    )
    native_tk_d128_backward_expected: tuple[str, int] | None = None
    if native_tk_d128_backward:
        assert args.native_tk_d128_backward_module is not None
        if not args.native_tk_d128_backward_module.strip():
            raise ValueError(
                "--native-tk-d128-backward-module must be non-empty"
            )
        native_tk_d128_backward_expected = _caller_declared_expected_identity(
            "native-tk-d128-backward",
            args.native_tk_d128_backward_sha256,
            args.native_tk_d128_backward_bytes,
        )
    if native_tk_d64_backward and native_tk_d128_backward:
        raise ValueError(
            "native TK D64 and D128 backward extensions are mutually exclusive"
        )

    diagnostic_fp8_lse_values = (
        args.diagnostic_fp8_lse_extension,
        args.diagnostic_fp8_lse_module,
        args.diagnostic_fp8_lse_sha256,
        args.diagnostic_fp8_lse_bytes,
    )
    if any(value is not None for value in diagnostic_fp8_lse_values) and not all(
        value is not None for value in diagnostic_fp8_lse_values
    ):
        raise ValueError(
            "diagnostic FP8-LSE control requires extension, module, SHA256, "
            "and byte count together"
        )
    if (
        args.diagnostic_fp8_lse_substitution_mode != "all_rows"
        and args.diagnostic_fp8_lse_extension is None
    ):
        raise ValueError(
            "non-default diagnostic FP8-LSE substitution mode requires the "
            "authenticated control artifact"
        )

    distributed_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed_rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if args.distributed_data_parallel:
        if distributed_world_size < 2:
            raise RuntimeError(
                "--distributed-data-parallel requires torchrun WORLD_SIZE >= 2"
            )
        if args.ddp_bucket_cap_mb <= 0:
            raise ValueError("--ddp-bucket-cap-mb must be positive")
        if not 0 <= local_rank < torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} is outside the visible CUDA devices"
            )
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        if torch.distributed.get_world_size() != distributed_world_size:
            raise RuntimeError("initialized process-group world size mismatch")
        if torch.distributed.get_rank() != distributed_rank:
            raise RuntimeError("initialized process-group rank mismatch")
        args.output = _ranked_output_path(args.output, distributed_rank)
        args.samples_output = _ranked_output_path(
            args.samples_output, distributed_rank
        )
        if args.save_final_checkpoint is not None:
            args.save_final_checkpoint = _ranked_output_path(
                args.save_final_checkpoint, distributed_rank
            )
        if args.save_initial_checkpoint is not None:
            args.save_initial_checkpoint = _ranked_output_path(
                args.save_initial_checkpoint, distributed_rank
            )
        args.trial_label = f"{args.trial_label}-rank{distributed_rank}"
    else:
        if distributed_world_size != 1:
            raise RuntimeError(
                "torchrun WORLD_SIZE > 1 requires --distributed-data-parallel"
            )
        if torch.cuda.device_count() != 1:
            raise RuntimeError("expose exactly one GPU to each route process")
        torch.cuda.set_device(0)
    if args.warmups < 1 or args.updates < MINIMUM_MEASURED_UPDATES:
        raise ValueError(
            f"require at least one warmup and {MINIMUM_MEASURED_UPDATES} "
            "measured updates"
        )
    if args.max_grad_norm <= 0.0 or not math.isfinite(args.max_grad_norm):
        raise ValueError("--max-grad-norm must be finite and positive")
    if args.loss_scale <= 0.0 or not math.isfinite(args.loss_scale):
        raise ValueError("--loss-scale must be finite and positive")
    if args.weight_decay < 0.0 or not math.isfinite(args.weight_decay):
        raise ValueError("--weight-decay must be finite and non-negative")
    total_updates = args.warmups + args.updates
    if args.profile_update is not None and not (
        0 <= args.profile_update < total_updates
    ):
        raise ValueError("--profile-update must select an executed update")
    if args.samples_output.exists() or args.output.exists():
        raise RuntimeError("refusing to overwrite an existing benchmark output")
    if (
        args.save_final_checkpoint is not None
        and args.save_final_checkpoint.exists()
    ):
        raise RuntimeError("refusing to overwrite an existing final checkpoint")
    if (
        args.save_initial_checkpoint is not None
        and args.route not in BF16_ROUTES
    ):
        raise RuntimeError(
            "only a BF16 route may create the canonical shared checkpoint"
        )
    config = config_from_model_preset(
        args.model_preset,
        batch=args.batch,
        d128_forward_topology_variant=(
            args.d128_forward_topology_variant
        ),
    )
    _require_saturated_shape(config)
    is_d128 = config.head_dim == 128
    if native_tk_d64_backward:
        if args.route not in LOWP_ROUTES or is_d128:
            raise ValueError(
                "native TK D64 backward is restricted to a D64 "
                "low-precision route"
            )
        if args.qkv_projection_format != "e4m3":
            raise ValueError(
                "native TK D64 backward requires --qkv-projection-format e4m3"
            )
        if args.experimental_native_nvfp4_projection_out:
            raise ValueError(
                "native TK D64 backward requires the E4M3 QKV projection, "
                "not the experimental native-NVFP4 projection"
            )
        if args.experimental_fused_attention_rmsnorm_nvfp4:
            raise ValueError(
                "native TK D64 backward does not admit the NVFP4-only fused "
                "attention RMSNorm arm"
            )
        if args.experimental_d128_mxfp4_v_backward:
            raise ValueError(
                "native TK D64 backward does not admit D128 MXFP4 V backward"
            )
        if args.experimental_d128_shared_tile_mxfp4_v:
            raise ValueError(
                "native TK D64 backward does not admit D128 shared-tile V"
            )
        if args.backward_control is not None:
            raise ValueError(
                "native TK D64 backward does not accept --backward-control"
            )
    if native_tk_d128_backward:
        if args.route not in ("fp8", "mx") or not is_d128:
            raise ValueError(
                "native TK D128 backward is restricted to a D128 FP8-PV or "
                "MXFP4-PV route"
            )
        if (
            args.qkv_projection_format != "nvfp4"
            or not args.experimental_native_nvfp4_projection_out
        ):
            raise ValueError(
                "native TK D128 backward requires the caller-owned native "
                "NVFP4 QKV projection"
            )
        if args.experimental_fused_attention_rmsnorm_nvfp4:
            raise ValueError(
                "native TK D128 backward does not admit the D64-only fused "
                "attention RMSNorm arm"
            )
        if args.experimental_d128_mxfp4_v_backward and (
            args.route != "mx" or config.batch != 2
        ):
            raise ValueError(
                "native TK D128 MXFP4-V backward requires the B2 MXFP4-PV "
                "route"
            )
        if args.backward_control is not None:
            raise ValueError(
                "native TK D128 backward does not accept --backward-control"
            )
    source_files_before = _benchmark_source_identities()
    output_shared_cli_request = args.experimental_output_shared_split_v
    if is_d128 and output_shared_cli_request is None:
        # The D128/B2 output-shared candidate was E2E-flat. Preserve the
        # validated D64/B16 automatic selector while requiring an explicit
        # opt-in for D128.
        args.experimental_output_shared_split_v = False
    if args.experimental_d128_mxfp4_v_backward:
        if not is_d128 or args.route != "mx":
            raise ValueError(
                "--experimental-d128-mxfp4-v-backward requires the D128 MX "
                "route"
            )
        if args.experimental_output_shared_split_v is not False:
            raise ValueError(
                "--experimental-d128-mxfp4-v-backward is mutually exclusive "
                "with --experimental-output-shared-split-v"
            )
        if args.backward_control is not None:
            raise ValueError(
                "--experimental-d128-mxfp4-v-backward does not accept "
                "--backward-control"
            )
        if args.diagnostic_fp8_lse_extension is not None:
            raise ValueError(
                "the FP8-LSE diagnostic requires the E4M3 backward V "
                "publication omitted by --experimental-d128-mxfp4-v-backward"
            )
    if args.experimental_d128_shared_tile_mxfp4_v:
        if not args.experimental_d128_mxfp4_v_backward:
            raise ValueError(
                "--experimental-d128-shared-tile-mxfp4-v requires "
                "--experimental-d128-mxfp4-v-backward"
            )
        if not native_tk_d128_backward:
            raise ValueError(
                "shared-tile MXFP4 V is authenticated only with the native "
                "TK D128 v503 consumer"
            )
    if is_d128 and args.route == "mx_unanchored":
        raise ValueError("mx_unanchored is authenticated only for D64/B16")
    if args.d128_forward_topology_variant != "production" and (
        not is_d128 or args.route != "mx"
    ):
        raise ValueError(
            "non-production D128 forward topology variants require the "
            "D128 MX route"
        )
    if args.diagnostic_fp8_lse_extension is not None and (
        not is_d128 or args.route != "mx"
    ):
        raise ValueError(
            "diagnostic FP8-LSE control is supported only by the D128 MX route"
        )
    if is_d128 and args.route in LOWP_ROUTES:
        if not (
            args.qkv_projection_format == "nvfp4"
            and args.experimental_native_nvfp4_projection_out
        ):
            raise ValueError(
                "D128 low-precision routes require --qkv-projection-format "
                "nvfp4 and --experimental-native-nvfp4-projection-out"
            )
        if args.experimental_fused_attention_rmsnorm_nvfp4:
            raise ValueError(
                "fused attention RMSNorm NVFP4 is authenticated only for D64/B16"
            )
        if args.backward_control is not None:
            raise ValueError(
                "D128 generates shared-P backward control; do not pass "
                "--backward-control"
            )
    elif (
        not is_d128
        and args.route in LOWP_ROUTES
        and not native_tk_d64_backward
    ):
        if args.backward_control is None:
            args.backward_control = DEFAULT_CONTROL
    if args.route in LOWP_ROUTES and args.forward_extension is None:
        if is_d128:
            raise ValueError(
                "D128 low-precision routes require a caller-declared custom "
                "--forward-extension"
            )
        args.forward_extension = DEFAULT_FORWARDS[args.route]
    if (
        args.experimental_d128_mxfp4_v_backward
        and not native_tk_d128_backward
    ):
        configured_dump = os.environ.get("CUTE_DSL_DUMP_DIR")
        if configured_dump is None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            compiler_artifact_dir = (
                args.output.parent
                / f"{args.output.name}.cutlass-compiler-artifacts"
            ).resolve()
            if compiler_artifact_dir.exists():
                raise RuntimeError(
                    "refusing to reuse an existing CUTLASS compiler artifact "
                    "directory"
                )
            compiler_artifact_dir.mkdir(mode=0o700)
        else:
            compiler_artifact_dir = Path(configured_dump)
        configure_d128_mxfp4_v_compile_environment(
            compiler_artifact_dir,
        )
    selected_projection = os.environ.get("TK_FA4_LOWP_BWD_EXTENSION_SOURCE")
    if args.route in LOWP_ROUTES and (
        selected_projection is None
        or Path(selected_projection).resolve() != args.projection_extension.resolve()
    ):
        raise RuntimeError(
            "TK_FA4_LOWP_BWD_EXTENSION_SOURCE must select --projection-extension"
        )

    torch.cuda.set_device(local_rank if args.distributed_data_parallel else 0)
    hardware_before = _hardware_identity()
    artifacts: dict[str, Any] = {}
    projection_authentication = None
    forward_expected: tuple[str, int] | None = None
    native_tk_backward_extension: Any | None = None
    if args.route in LOWP_ROUTES:
        assert args.forward_extension is not None
        (
            forward_expected,
            forward_authentication,
            args.forward_module,
        ) = _forward_expected_identity(
            args.route,
            args.forward_extension,
            args.forward_module,
            args.forward_sha256,
            args.forward_bytes,
        )
        projection_expected, projection_authentication = (
            _projection_expected_identity(
                args.projection_extension,
                args.projection_sha256,
                args.projection_bytes,
            )
        )
        artifacts["projection"] = _file_identity(
            args.projection_extension, projection_expected
        )
        artifacts["projection"]["authentication"] = projection_authentication
        if tk_interface._C_b300_lowp_bwd is None:
            raise RuntimeError("projection extension failed to import")
        artifacts["projection"]["loaded_image"] = (
            _require_loaded_artifact_identity(
                "projection",
                tk_interface._C_b300_lowp_bwd,
                args.projection_extension,
                projection_expected,
            )
        )
        if native_tk_d64_backward:
            assert args.native_tk_d64_backward_extension is not None
            assert args.native_tk_d64_backward_module is not None
            assert native_tk_d64_backward_expected is not None
            native_tk_artifact = _file_identity(
                args.native_tk_d64_backward_extension,
                native_tk_d64_backward_expected,
            )
            native_tk_artifact["authentication"] = "caller_declared"
            native_tk_artifact["module"] = (
                args.native_tk_d64_backward_module
            )
            native_tk_backward_extension = _load_extension(
                args.native_tk_d64_backward_extension,
                args.native_tk_d64_backward_module,
            )
            native_tk_artifact["loaded_image"] = (
                _require_loaded_artifact_identity(
                    "native TK D64 backward",
                    native_tk_backward_extension,
                    args.native_tk_d64_backward_extension,
                    native_tk_d64_backward_expected,
                )
            )
            artifacts["native_tk_d64_backward"] = native_tk_artifact
        elif native_tk_d128_backward:
            assert args.native_tk_d128_backward_extension is not None
            assert args.native_tk_d128_backward_module is not None
            assert native_tk_d128_backward_expected is not None
            native_tk_artifact = _file_identity(
                args.native_tk_d128_backward_extension,
                native_tk_d128_backward_expected,
            )
            native_tk_artifact["authentication"] = "caller_declared"
            native_tk_artifact["module"] = (
                args.native_tk_d128_backward_module
            )
            native_tk_backward_extension = _load_extension(
                args.native_tk_d128_backward_extension,
                args.native_tk_d128_backward_module,
            )
            native_tk_artifact["loaded_image"] = (
                _require_loaded_artifact_identity(
                    "native TK D128 backward",
                    native_tk_backward_extension,
                    args.native_tk_d128_backward_extension,
                    native_tk_d128_backward_expected,
                )
            )
            artifacts["native_tk_d128_backward"] = native_tk_artifact
        if not is_d128 and not native_tk_d64_backward:
            assert args.backward_control is not None
            artifacts["control"] = _file_identity(
                args.backward_control, PINNED_ARTIFACTS["control"]
            )
        artifacts["forward"] = _file_identity(
            args.forward_extension, forward_expected
        )
        artifacts["forward"]["authentication"] = forward_authentication
        artifacts["forward"]["module"] = args.forward_module
    _require_saturated_projection_selection(
        args.route,
        args.qkv_projection_format,
        args.experimental_native_nvfp4_projection_out,
        args.experimental_fused_attention_rmsnorm_nvfp4,
        args.experimental_output_shared_split_v,
        projection_authentication,
        args.experimental_d128_mxfp4_v_backward,
        args.experimental_d128_shared_tile_mxfp4_v,
    )

    runtime = None
    topology = None
    diagnostic_fp8_lse_topology = None
    if args.route in LOWP_ROUTES:
        assert args.forward_extension is not None
        assert args.forward_module is not None
        assert forward_expected is not None
        runtime, topology = _runtime(
            args.route,
            config,
            args.forward_extension,
            args.forward_module,
            forward_expected,
            args.backward_control,
            args.loss_scale,
            args.qkv_projection_format,
            args.experimental_native_nvfp4_projection_out,
            args.experimental_fused_attention_rmsnorm_nvfp4,
            args.experimental_output_shared_split_v,
            experimental_d128_mxfp4_v_backward=(
                args.experimental_d128_mxfp4_v_backward
            ),
            experimental_d128_shared_tile_mxfp4_v=(
                args.experimental_d128_shared_tile_mxfp4_v
            ),
            native_tk_d64_backward_extension=(
                native_tk_backward_extension
                if native_tk_d64_backward
                else None
            ),
            native_tk_d128_backward_extension=(
                native_tk_backward_extension
                if native_tk_d128_backward
                else None
            ),
        )
        artifacts["forward"]["loaded_image"] = dict(
            runtime.forward_loaded_artifact_identity
        )
        if native_tk_d64_backward:
            _require_native_tk_d64_saturated_runtime(
                args.route,
                config,
                runtime,
                artifacts["native_tk_d64_backward"],
            )
        elif native_tk_d128_backward:
            _require_native_tk_d128_saturated_runtime(
                args.route,
                config,
                runtime,
                artifacts["native_tk_d128_backward"],
            )
        d128_mxfp4_v_dp_patch_artifact = (
            _d128_mxfp4_v_dp_patch_artifact(runtime)
        )
        if d128_mxfp4_v_dp_patch_artifact is not None:
            artifacts["d128_mxfp4_v_dp_patch"] = (
                d128_mxfp4_v_dp_patch_artifact
            )
        d128_mxfp4_v_compilation = (
            runtime.d128_mxfp4_v_compilation_receipt()
        )
        if d128_mxfp4_v_compilation is not None:
            artifacts["d128_mxfp4_v_compilation"] = (
                d128_mxfp4_v_compilation
            )
        _qkv_projection_contract(runtime, artifacts["projection"])
        if args.diagnostic_fp8_lse_extension is not None:
            assert args.diagnostic_fp8_lse_module is not None
            assert args.diagnostic_fp8_lse_sha256 is not None
            assert args.diagnostic_fp8_lse_bytes is not None
            (
                diagnostic_expected,
                diagnostic_authentication,
                diagnostic_module,
            ) = _forward_expected_identity(
                "fp8",
                args.diagnostic_fp8_lse_extension,
                args.diagnostic_fp8_lse_module,
                args.diagnostic_fp8_lse_sha256,
                args.diagnostic_fp8_lse_bytes,
            )
            diagnostic_artifact = _file_identity(
                args.diagnostic_fp8_lse_extension,
                diagnostic_expected,
            )
            diagnostic_artifact["authentication"] = (
                diagnostic_authentication
            )
            diagnostic_artifact["module"] = diagnostic_module
            (
                diagnostic_extension,
                diagnostic_fp8_lse_topology,
            ) = _load_forward(
                args.diagnostic_fp8_lse_extension,
                diagnostic_module,
                config,
            )
            diagnostic_artifact["loaded_image"] = (
                _require_loaded_artifact_identity(
                    "diagnostic FP8-LSE forward",
                    diagnostic_extension,
                    args.diagnostic_fp8_lse_extension,
                    diagnostic_expected,
                )
            )
            runtime.install_diagnostic_fp8_lse_control(
                diagnostic_extension,
                diagnostic_fp8_lse_topology,
                diagnostic_artifact["loaded_image"],
                substitution_mode=(
                    args.diagnostic_fp8_lse_substitution_mode
                ),
            )
            artifacts["diagnostic_fp8_lse_control"] = diagnostic_artifact

    torch.manual_seed(args.seed)
    bf16_attention_control = (
        "packed_qkv_single_linear"
        if args.route == "bf16_packed"
        else DEFAULT_BF16_ATTENTION_CONTROL
    )
    model = Llama12B(
        config,
        _make_llama3_rope(config),
        runtime,
        bf16_attention_control=bf16_attention_control,
    )
    if runtime is not None:
        activate_model_forward_route(model)
        require_matching_backward_contracts(
            {args.route: runtime.backward_contract()}
        )
    parameter_schema = _parameter_schema(model)
    (
        sampled_parameter_names,
        hidden_sample_batches,
        hidden_sample_positions,
    ) = _diagnostic_sample_layout(config)
    physical_parameter_tensors = sum(1 for _ in model.parameters())
    if args.save_initial_checkpoint is not None:
        checkpoint = _save_initial_checkpoint(
            model, args.save_initial_checkpoint
        )
    else:
        assert args.initial_checkpoint is not None
        checkpoint = _load_initial_checkpoint(model, args.initial_checkpoint)
    initial_parameters = _parameter_samples(model, sampled_parameter_names)
    distributed_forward: torch.nn.Module | None = None
    if args.distributed_data_parallel:
        distributed_forward = torch.nn.parallel.DistributedDataParallel(
            _DistributedHiddenAndWeight(model),
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            bucket_cap_mb=args.ddp_bucket_cap_mb,
            gradient_as_bucket_view=True,
            static_graph=True,
        )
    optimizer_configuration = {
        "name": "AdamW",
        "implementation": "torch_fused",
        "learning_rate": args.learning_rate,
        "betas": [0.9, 0.95],
        "eps": 1.0e-8,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "gradient_clipping": "torch.nn.utils.clip_grad_norm_foreach",
    }
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1.0e-8,
        weight_decay=args.weight_decay,
        fused=True,
    )

    dolma_tokens = None
    dolma_targets = None
    data_receipt: dict[str, Any]
    data_seed = args.seed
    global_batch = config.batch * distributed_world_size
    rank_batch_start = distributed_rank * config.batch
    rank_batch_end = rank_batch_start + config.batch
    if args.tokens == "dolma":
        global_dolma_tokens, global_dolma_targets, data_receipt = _dolma_batches(
            args.corpus,
            args.tokenizer,
            seed=data_seed,
            count=total_updates + 1,
            batch=global_batch,
            sequence=config.sequence,
        )
        dolma_tokens = global_dolma_tokens[
            :, rank_batch_start:rank_batch_end
        ].contiguous()
        dolma_targets = global_dolma_targets[
            :, rank_batch_start:rank_batch_end
        ].contiguous()
        del global_dolma_tokens, global_dolma_targets
    else:
        data_receipt = {
            "kind": "deterministic_uniform_synthetic",
            "seed": data_seed,
            "updates_including_probe": total_updates + 1,
            "batch": global_batch,
            "sequence": config.sequence,
        }
    data_receipt["local_batch"] = config.batch
    data_receipt["rank_batch_slice"] = [rank_batch_start, rank_batch_end]
    data_receipt["distributed_rank"] = distributed_rank
    data_receipt["distributed_world_size"] = distributed_world_size
    data_receipt["global_batch"] = config.batch * distributed_world_size

    # Compile the CE and capture a fixed heldout diagnostic without changing
    # weights.  The same batch is repeated after the timed update trajectory.
    if dolma_tokens is None or dolma_targets is None:
        global_probe_tokens, global_probe_targets = _batch(
            data_seed, -1, global_batch, config.sequence, config.vocab
        )
        probe_tokens = global_probe_tokens[
            rank_batch_start:rank_batch_end
        ].contiguous()
        probe_targets = global_probe_targets[
            rank_batch_start:rank_batch_end
        ].contiguous()
        del global_probe_tokens, global_probe_targets
    else:
        probe_tokens, probe_targets = dolma_tokens[0], dolma_targets[0]
    initial_diagnostic = _diagnostic_pass(
        model,
        optimizer,
        probe_tokens,
        probe_targets,
        sampled_parameter_names,
        hidden_sample_batches,
        hidden_sample_positions,
        distributed_forward,
    )
    _require_d128_runtime_populated_forward_topology(
        args.route,
        config,
        runtime,
    )
    d128_dual_qkv_weight_preparation = (
        _d128_dual_qkv_weight_preparation_receipt(model, config, runtime)
    )
    dual_output_weight_preparation = (
        _dual_output_weight_preparation_receipt(model, config, runtime)
    )

    torch.cuda.reset_peak_memory_stats()
    records: list[dict[str, Any]] = []
    measured_loop_start: float | None = None
    for update in range(total_updates):
        if dolma_tokens is None or dolma_targets is None:
            global_tokens, global_targets = _batch(
                data_seed,
                update,
                global_batch,
                config.sequence,
                config.vocab,
            )
            tokens = global_tokens[rank_batch_start:rank_batch_end].contiguous()
            targets = global_targets[
                rank_batch_start:rank_batch_end
            ].contiguous()
            del global_tokens, global_targets
        else:
            tokens, targets = dolma_tokens[update + 1], dolma_targets[update + 1]
        if update == args.warmups:
            torch.cuda.synchronize()
            measured_loop_start = time.perf_counter()
        record = _timed_update(
            model,
            optimizer,
            tokens,
            targets,
            update=update,
            warmup=update < args.warmups,
            max_grad_norm=args.max_grad_norm,
            profile=(update == args.profile_update),
            distributed_forward=distributed_forward,
        )
        records.append(record)
        print(
            f"route={args.route} update={update} warmup={record['warmup']} "
            f"loss={record['loss']:.6f} step={record['step_ms']:.3f}ms",
            flush=True,
        )
        del tokens, targets
        if not record["finite"]:
            raise RuntimeError(f"non-finite loss at update {update}")
    assert measured_loop_start is not None
    torch.cuda.synchronize()
    measured_loop_wall_ms = (
        time.perf_counter() - measured_loop_start
    ) * 1000.0
    if args.distributed_data_parallel:
        global_losses = torch.tensor(
            [float(record["loss"]) for record in records],
            device="cuda",
            dtype=torch.float64,
        )
        torch.distributed.all_reduce(
            global_losses,
            op=torch.distributed.ReduceOp.SUM,
        )
        global_losses /= distributed_world_size
        for record, global_loss in zip(
            records, global_losses.cpu().tolist(), strict=True
        ):
            record["global_mean_loss"] = float(global_loss)
    else:
        for record in records:
            record["global_mean_loss"] = float(record["loss"])

    final_parameters = _parameter_samples(model, sampled_parameter_names)
    final_diagnostic = _diagnostic_pass(
        model,
        optimizer,
        probe_tokens,
        probe_targets,
        sampled_parameter_names,
        hidden_sample_batches,
        hidden_sample_positions,
        distributed_forward,
    )
    peak_allocated = torch.cuda.max_memory_allocated() / 2.0**30
    peak_reserved = torch.cuda.max_memory_reserved() / 2.0**30
    if peak_reserved > args.max_hbm_gib:
        raise RuntimeError(
            f"peak reserved HBM {peak_reserved:.3f} GiB exceeds "
            f"{args.max_hbm_gib:.3f} GiB gate"
        )
    final_checkpoint = None
    if args.save_final_checkpoint is not None:
        final_checkpoint = _write_model_checkpoint(
            model,
            args.save_final_checkpoint,
            kind="post_trajectory_model_state",
        )
    del probe_tokens, probe_targets

    measured = [record for record in records if not record["warmup"]]
    timing_fields = (
        "decoder_forward_ms",
        "ce_forward_ms",
        "backward_ms",
        "gradient_clip_ms",
        "optimizer_ms",
        "step_ms",
        "wall_ms",
    )
    timing_statistics = _timing_statistics(measured, timing_fields)
    p50_step_seconds = timing_statistics["step_ms"]["p50"] / 1000.0
    sustained_step_seconds = measured_loop_wall_ms / len(measured) / 1000.0
    useful_flops = _useful_flops(config)
    global_p50_step_seconds = p50_step_seconds
    global_sustained_step_seconds = sustained_step_seconds
    if args.distributed_data_parallel:
        distributed_durations = torch.tensor(
            [p50_step_seconds, sustained_step_seconds],
            device="cuda",
            dtype=torch.float64,
        )
        torch.distributed.all_reduce(
            distributed_durations,
            op=torch.distributed.ReduceOp.MAX,
        )
        global_p50_step_seconds = float(distributed_durations[0])
        global_sustained_step_seconds = float(distributed_durations[1])

    comparison_identity = {
        "seed": data_seed,
        "model_seed": args.seed,
        "configuration": config.__dict__,
        "data": data_receipt,
        "checkpoint_sha256": checkpoint["sha256"],
        "checkpoint_bytes": checkpoint["bytes"],
        "parameter_schema": parameter_schema,
        "optimizer": optimizer_configuration,
        "warmups": args.warmups,
        "measured_updates": args.updates,
        "sampled_parameter_names": list(sampled_parameter_names),
        "sample_elements_per_parameter": SAMPLE_ELEMENTS_PER_PARAMETER,
        "hidden_sample_batches": list(hidden_sample_batches),
        "hidden_sample_positions": list(hidden_sample_positions),
        "distributed": {
            "enabled": bool(args.distributed_data_parallel),
            "rank": distributed_rank,
            "world_size": distributed_world_size,
            "local_batch": config.batch,
            "global_batch": global_batch,
        },
    }
    samples: dict[str, Any] = {
        "schema": "llama12b_saturated_route_samples_v2",
        "route": args.route,
        "comparison_identity": comparison_identity,
        "checkpoint": checkpoint,
        "initial_parameters": initial_parameters,
        "initial_diagnostic": initial_diagnostic,
        "final_parameters": final_parameters,
        "parameter_updates": _subtract_samples(
            final_parameters, initial_parameters
        ),
        "final_diagnostic": final_diagnostic,
        "losses": [float(record["loss"]) for record in records],
    }
    comparisons = None
    reference_route = None
    reference_samples_identity = None
    if args.reference_samples is not None:
        reference_samples_identity = _source_identity(args.reference_samples)
        reference = torch.load(
            args.reference_samples, map_location="cpu", weights_only=True
        )
        _require_reference_identity(samples, reference)
        reference_route = str(reference["route"])
        comparisons = {
            "heldout_loss": {
                "initial_delta": (
                    float(samples["initial_diagnostic"]["loss"])
                    - float(reference["initial_diagnostic"]["loss"])
                ),
                "final_delta": (
                    float(samples["final_diagnostic"]["loss"])
                    - float(reference["final_diagnostic"]["loss"])
                ),
                "candidate_improvement": (
                    float(samples["initial_diagnostic"]["loss"])
                    - float(samples["final_diagnostic"]["loss"])
                ),
                "reference_improvement": (
                    float(reference["initial_diagnostic"]["loss"])
                    - float(reference["final_diagnostic"]["loss"])
                ),
            },
            "initial_parameters": _compare_samples(
                samples["initial_parameters"],
                reference["initial_parameters"],
            ),
            "initial_hidden": _compare_samples(
                samples["initial_diagnostic"]["hidden"],
                reference["initial_diagnostic"]["hidden"],
            ),
            "initial_gradients": _compare_samples(
                samples["initial_diagnostic"]["gradients"],
                reference["initial_diagnostic"]["gradients"],
            ),
            "initial_gradient_statistics": _gradient_statistic_deltas(
                samples["initial_diagnostic"]["gradient_statistics"],
                reference["initial_diagnostic"]["gradient_statistics"],
            ),
            "initial_sampled_logits": _compare_logits(
                samples["initial_diagnostic"]["logits"]["selected_logits"],
                reference["initial_diagnostic"]["logits"]["selected_logits"],
            ),
            "final_parameters": _compare_samples(
                samples["final_parameters"], reference["final_parameters"]
            ),
            "parameter_updates": _compare_samples(
                samples["parameter_updates"], reference["parameter_updates"]
            ),
            "final_hidden": _compare_samples(
                samples["final_diagnostic"]["hidden"],
                reference["final_diagnostic"]["hidden"],
            ),
            "final_gradients": _compare_samples(
                samples["final_diagnostic"]["gradients"],
                reference["final_diagnostic"]["gradients"],
            ),
            "final_gradient_statistics": _gradient_statistic_deltas(
                samples["final_diagnostic"]["gradient_statistics"],
                reference["final_diagnostic"]["gradient_statistics"],
            ),
            "final_sampled_logits": _compare_logits(
                samples["final_diagnostic"]["logits"]["selected_logits"],
                reference["final_diagnostic"]["logits"]["selected_logits"],
            ),
            "loss_deltas": [
                actual - wanted
                for actual, wanted in zip(
                    samples["losses"], reference["losses"], strict=True
                )
            ],
        }

    args.samples_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(samples, args.samples_output)
    samples_identity = {
        "path": str(args.samples_output.resolve()),
        "sha256": _sha256(args.samples_output),
        "bytes": args.samples_output.stat().st_size,
    }
    current_topology = (
        dict(runtime.forward_topology) if runtime is not None else None
    )
    if args.route == "mx_unanchored":
        assert current_topology is not None
        require_mx_variant_topology(
            current_topology,
            variant="unanchored-splitmix-v6",
            batch=config.batch,
            runtime_populated=True,
        )
    source_files_after = _benchmark_source_identities()
    if source_files_after != source_files_before:
        raise RuntimeError(
            "benchmark source files changed during execution; discard this "
            "run and repeat from a stable checkout"
        )
    source_files = dict(source_files_before)
    if not _uses_packed_qkv(model):
        source_files.pop("packed_bf16_qkv")
    if runtime is None or not runtime.native_tk_d64_backward:
        source_files.pop("native_tk_d64_backward_runner")
    if runtime is None or not runtime.native_tk_d128_backward:
        source_files.pop("native_tk_d128_backward_runner")
    if (
        runtime is None
        or not runtime.native_tk_d128_backward
        or not runtime.experimental_d128_mxfp4_v_backward
    ):
        source_files.pop("native_tk_d128_mxfp4_v_backward_runner")
    result = {
        "schema": "llama12b_saturated_route_benchmark_v2",
        "route": args.route,
        "trial_label": args.trial_label,
        "distributed": {
            "enabled": bool(args.distributed_data_parallel),
            "backend": (
                str(torch.distributed.get_backend())
                if args.distributed_data_parallel
                else None
            ),
            "rank": distributed_rank,
            "local_rank": local_rank,
            "world_size": distributed_world_size,
            "local_batch": config.batch,
            "global_batch": global_batch,
            "gradient_sync": (
                "pytorch_ddp_nccl_mean_with_backward_overlap"
                if args.distributed_data_parallel
                else None
            ),
            "bucket_cap_mb": (
                args.ddp_bucket_cap_mb
                if args.distributed_data_parallel
                else None
            ),
            "gradient_as_bucket_view": bool(
                args.distributed_data_parallel
            ),
            "static_graph": bool(args.distributed_data_parallel),
        },
        "configuration": {
            **config.__dict__,
            "warmups": args.warmups,
            "measured_updates": args.updates,
            "profile_update": args.profile_update,
            "loss_scale": args.loss_scale if runtime is not None else None,
            "qkv_projection_format": (
                runtime.qkv_projection_format if runtime is not None else None
            ),
            "native_tk_d64_backward": bool(
                runtime.native_tk_d64_backward
                if runtime is not None
                else False
            ),
            "native_tk_d128_backward": bool(
                runtime.native_tk_d128_backward
                if runtime is not None
                else False
            ),
            "experimental_native_nvfp4_projection_out": bool(
                args.experimental_native_nvfp4_projection_out
            ),
            "experimental_fused_attention_rmsnorm_nvfp4": bool(
                runtime.experimental_fused_attention_rmsnorm_nvfp4
                if runtime is not None
                else False
            ),
            "experimental_output_shared_split_v": bool(
                runtime.experimental_output_shared_split_v
                if runtime is not None
                else False
            ),
            "experimental_output_shared_split_v_requested": (
                args.experimental_output_shared_split_v
                if runtime is not None
                else None
            ),
            "experimental_output_shared_split_v_cli_requested": (
                output_shared_cli_request if runtime is not None else None
            ),
            "experimental_output_shared_split_v_resolved": bool(
                runtime.experimental_output_shared_split_v_resolved
                if runtime is not None
                else False
            ),
            "experimental_d128_mxfp4_v_backward": bool(
                runtime.experimental_d128_mxfp4_v_backward
                if runtime is not None
                else False
            ),
            "d128_mxfp4_v_scale_policy": (
                runtime.d128_mxfp4_v_scale_policy
                if runtime is not None
                else None
            ),
            "diagnostic_fp8_lse_control": bool(
                runtime is not None
                and runtime.diagnostic_fp8_lse_entrypoint is not None
            ),
            "diagnostic_fp8_lse_substitution_mode": (
                runtime.diagnostic_fp8_lse_substitution_mode
                if runtime is not None
                and runtime.diagnostic_fp8_lse_entrypoint is not None
                else None
            ),
            "output_shared_split_v_path": (
                runtime.output_shared_split_v_path
                if runtime is not None
                else "not_applicable"
            ),
            "output_shared_split_v_checked_symbol": (
                getattr(runtime.qkv_projection, "checked_symbol", None)
                if (
                    runtime is not None
                    and (
                        config.head_dim == 64
                        or (
                            config.head_dim == 128
                            and runtime.experimental_output_shared_split_v
                        )
                    )
                )
                else None
            ),
            "d128_route_selective_checked_symbol": (
                getattr(runtime.qkv_projection, "checked_symbol", None)
                if runtime is not None and config.head_dim == 128
                else None
            ),
            "d128_route_selective_unchecked_symbol": (
                getattr(runtime.qkv_projection, "unchecked_symbol", None)
                if runtime is not None and config.head_dim == 128
                else None
            ),
            "bf16_attention_control": (
                bf16_attention_control if runtime is None else None
            ),
            "attention_route": model.attention_route,
            "qkv_parameter_layout": (
                "packed_qkv" if _uses_packed_qkv(model) else "split_qkv"
            ),
            "physical_optimizer_parameter_tensors": physical_parameter_tensors,
            "optimizer": optimizer_configuration,
            "loss": _cce_identity(),
            "outer_model_compile": False,
            "token_source": args.tokens,
            "comparison_scope": _comparison_scope(
                args.route, reference_route
            ),
            "reference_route": reference_route,
        },
        "data": data_receipt,
        "checkpoint": checkpoint,
        "final_checkpoint": final_checkpoint,
        "sample_artifact": samples_identity,
        "reference_sample_artifact": reference_samples_identity,
        "source_files": source_files,
        "artifacts": artifacts,
        "hardware_before": hardware_before,
        "hardware_after": _hardware_identity(),
        "forward_topology": current_topology,
        "diagnostic_fp8_lse_control": (
            {
                "semantics": (
                    (
                        "retain_mx_attention_output_substitute_authenticated_"
                        "fp8_control_lse_for_all_rows_in_shared_backward"
                    )
                    if runtime.diagnostic_fp8_lse_substitution_mode
                    == "all_rows"
                    else (
                        "retain_mx_attention_output_and_finite_mx_lse_"
                        "substitute_authenticated_fp8_control_lse_only_"
                        "where_mx_lse_is_nonfinite_in_shared_backward"
                    )
                ),
                "substitution_mode": (
                    runtime.diagnostic_fp8_lse_substitution_mode
                ),
                "substitution_counts": dict(
                    runtime.diagnostic_fp8_lse_substitution_counts
                ),
                "substitution_count_scope": (
                    "all_control_launches_in_this_process_including_model_"
                    "diagnostics_and_training_updates"
                ),
                "production_route": False,
                "adds_one_discarded_fp8_forward_per_layer": True,
                "runtime_authenticated": bool(
                    runtime.diagnostic_fp8_lse_runtime_authenticated
                ),
                "topology": runtime.diagnostic_fp8_lse_topology,
                "loaded_artifact_identity": (
                    runtime.diagnostic_fp8_lse_loaded_artifact_identity
                ),
                "first_launch_receipt": (
                    runtime.diagnostic_fp8_lse_first_launch_receipt
                ),
                "backward_contract_unchanged": runtime.backward_contract(),
            }
            if runtime is not None
            and runtime.diagnostic_fp8_lse_entrypoint is not None
            else None
        ),
        "qkv_projection_contract": (
            _qkv_projection_contract(runtime, artifacts["projection"])
            if runtime is not None
            else None
        ),
        "d128_dual_qkv_weight_preparation": (
            d128_dual_qkv_weight_preparation
        ),
        "dual_output_weight_preparation": dual_output_weight_preparation,
        "backward_contract": runtime.backward_contract() if runtime else None,
        **(
            {
                "d128_mxfp4_v_operand_cache": (
                    runtime.d128_mxfp4_v_operand_cache_receipt()
                )
            }
            if runtime is not None
            and runtime.experimental_d128_mxfp4_v_backward
            else {}
        ),
        "records": records,
        "steady_state": {
            "timing_statistics_ms": timing_statistics,
            "measured_loop_wall_ms": measured_loop_wall_ms,
            "p50_tokens_per_second": (
                config.batch * config.sequence / p50_step_seconds
            ),
            "sustained_tokens_per_second": (
                config.batch * config.sequence / sustained_step_seconds
            ),
            "global_p50_step_ms": global_p50_step_seconds * 1000.0,
            "global_sustained_step_ms": (
                global_sustained_step_seconds * 1000.0
            ),
            "global_p50_tokens_per_second": (
                global_batch * config.sequence / global_p50_step_seconds
            ),
            "global_sustained_tokens_per_second": (
                global_batch
                * config.sequence
                / global_sustained_step_seconds
            ),
            "global_p50_bf16_equivalent_useful_tflops": (
                useful_flops
                * distributed_world_size
                / global_p50_step_seconds
                / 1.0e12
            ),
            "global_p50_bf16_equivalent_useful_mfu_at_2250_tflops_per_gpu": (
                useful_flops
                / global_p50_step_seconds
                / 2.25e15
            ),
            "p50_bf16_equivalent_useful_tflops": (
                useful_flops / p50_step_seconds / 1.0e12
            ),
            "p50_bf16_equivalent_useful_mfu_at_2250_tflops": (
                useful_flops / p50_step_seconds / 2.25e15
            ),
            "hardware_utilization_requires_external_profiler": True,
            "production_timing_valid": not bool(
                runtime is not None
                and runtime.diagnostic_fp8_lse_entrypoint is not None
            )
            and config.d128_forward_topology_variant == "production",
            "single_pass_candidate_timing_valid": bool(
                runtime is not None
                and runtime.diagnostic_fp8_lse_entrypoint is None
            ),
        },
        "memory": {
            "peak_allocated_gib": peak_allocated,
            "peak_reserved_gib": peak_reserved,
            "gate_gib": args.max_hbm_gib,
        },
        "heldout_loss": {
            "initial": float(initial_diagnostic["loss"]),
            "final": float(final_diagnostic["loss"]),
        },
        "comparisons_vs_bf16": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not args.distributed_data_parallel or distributed_rank == 0:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            json.dumps(
                {
                    "route": args.route,
                    "rank": distributed_rank,
                    "output": str(args.output.resolve()),
                    "global_p50_tokens_per_second": result["steady_state"][
                        "global_p50_tokens_per_second"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if args.distributed_data_parallel:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
