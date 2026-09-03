#!/usr/bin/env python3
"""Render an authenticated, credential-free FA4 TorchTitan config."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# Keep direct source-checkout invocation deterministic; installing TorchTitan is
# optional for rendering a config from this repository.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from torchtitan.experiments.fa4.artifacts import load_artifact_manifest
from torchtitan.experiments.fa4.optimizer.fused_adamw_bf16_sr import (
    PROVIDER,
    PROVIDER_VERSION,
    SOURCE_SHA256,
)
from torchtitan.experiments.fa4.optimizer.optimizer_sr_state import (
    CHECKPOINT_SCHEMA,
)
from tools.fa4_dataset_manifest import verify_dataset_manifest


_COMMIT = re.compile(r"[0-9a-f]{40}")
REQUIRED_ENVIRONMENT = {
    "FA4_DATALOADER_PIN_MEMORY": "0",
    "FA4_DATALOADER_PREFETCH_FACTOR": "8",
    "FA4_TRAIN_DATALOADER_NUM_WORKERS": "8",
    "FA4_VALIDATION_DATALOADER_NUM_WORKERS": "1",
    "LBT_ADAMW_BF16_SR_CHECKPOINT_SCHEMA": CHECKPOINT_SCHEMA,
    "LBT_ADAMW_BF16_SR_PROVIDER": PROVIDER,
    "LBT_ADAMW_BF16_SR_PROVIDER_VERSION": str(PROVIDER_VERSION),
    "LBT_ADAMW_BF16_SR_SEED": "0",
    "LBT_ADAMW_BF16_SR_SOURCE_SHA256": SOURCE_SHA256,
    "LBT_DCP_SYNC_CPU_PROCESS_GROUP": "1",
    "TORCHTITAN_FSDP_ACCUMULATE_WITHOUT_SYNC": "1",
}
_HISTORICAL_TOKENIZER = {
    "original/tokenizer.model": (
        2_183_982,
        "82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55",
    ),
    "special_tokens_map.json": (
        73,
        "462d91939dbc37178aa5a3eae7068d1990ccc92e09f288cc71f42cdf139d69cc",
    ),
    "tokenizer.json": (
        9_085_658,
        "76e48799b099d43365bd24ccd8ecc5aedac831718da780552f03b0a6eb4412aa",
    ),
    "tokenizer_config.json": (
        50_500,
        "8004530facf809ac432114de2a4dcc65fcb632da5ec16d666091aeb6a2ee444a",
    ),
}


@dataclass(frozen=True)
class TrainingProfile:
    """Model/training defaults that are independent of one compiled route."""

    name: str
    artifact_profile: str
    model_flavor: str
    rope_scaling_factor: float
    local_batch: int | None
    sequence: int
    query_heads: int
    key_value_heads: int
    head_dimension: int
    world_size: int
    global_batch_size: int
    steps: int
    target_tokens: int
    warmup_steps: int
    checkpoint_interval: int
    validation_frequency: int
    validation_local_batch: int | None
    learning_rate: float
    tied_embeddings: bool


TRAINING_PROFILES = {
    "llama8b-d128-100b": TrainingProfile(
        name="llama8b-d128-100b",
        artifact_profile="llama8b-d128-b{batch}",
        model_flavor="8B_llama3_blog",
        rope_scaling_factor=8.0,
        local_batch=None,
        sequence=4096,
        query_heads=32,
        key_value_heads=8,
        head_dimension=128,
        world_size=64,
        global_batch_size=1024,
        steps=23_842,
        target_tokens=100_000_595_968,
        warmup_steps=2_000,
        checkpoint_interval=239,
        validation_frequency=298,
        validation_local_batch=None,
        learning_rate=3e-4,
        tied_embeddings=False,
    ),
    "llama1p2b-d64-50b": TrainingProfile(
        name="llama1p2b-d64-50b",
        artifact_profile="llama1p2b-d64-b16",
        model_flavor="1B",
        rope_scaling_factor=32.0,
        local_batch=16,
        sequence=4096,
        query_heads=32,
        key_value_heads=8,
        head_dimension=64,
        world_size=16,
        global_batch_size=256,
        steps=47_684,
        target_tokens=50_000_297_984,
        warmup_steps=954,
        checkpoint_interval=954,
        validation_frequency=262,
        validation_local_batch=16,
        learning_rate=0.00048828125,
        tied_embeddings=True,
    ),
}


def _quote(value: str) -> str:
    # JSON string quoting is a valid subset of basic TOML string quoting.
    return json.dumps(value)


def _absolute(value: str, label: str, *, must_exist: bool) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute: {path}")
    path = path.resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _file_identity(path: Path) -> dict[str, object]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _tokenizer_identity(root: Path, *, allow_nonhistorical: bool) -> dict[str, object]:
    if not allow_nonhistorical:
        files: list[dict[str, object]] = []
        for relative_name, (
            expected_bytes,
            expected_sha256,
        ) in _HISTORICAL_TOKENIZER.items():
            path = root / relative_name
            if not path.is_file():
                raise FileNotFoundError(
                    f"historical tokenizer requires {relative_name}: {path}"
                )
            identity = _file_identity(path)
            identity["path"] = relative_name
            if (
                identity["bytes"] != expected_bytes
                or identity["sha256"] != expected_sha256
            ):
                raise ValueError(
                    "historical tokenizer identity mismatch for "
                    f"{relative_name}; run tools/verify_fa4_data.py for details"
                )
            files.append(identity)
        return {
            "root": str(root),
            "files": files,
            "historical_four_file_identity": True,
            "historical_tree_sha256": (
                "ba9162eb542cf6445c6a1c9cf997dc176458b1dcfa127aad434b563ec5d94718"
            ),
        }

    config = root / "tokenizer_config.json"
    if not config.is_file():
        raise FileNotFoundError(f"HF assets require tokenizer_config.json: {config}")
    tokenizer_json = root / "tokenizer.json"
    if tokenizer_json.is_file():
        selected = (tokenizer_json, config)
    else:
        vocabulary = tuple(
            path for path in (root / "vocab.json", root / "vocab.txt") if path.is_file()
        )
        if len(vocabulary) != 1:
            raise FileNotFoundError(
                "HF assets require tokenizer.json or exactly one of "
                "vocab.json/vocab.txt"
            )
        merges = root / "merges.txt"
        selected = (*vocabulary, *((merges,) if merges.is_file() else ()), config)
    files = tuple(_file_identity(path) for path in selected)
    digest = hashlib.sha256()
    for identity in files:
        digest.update(str(identity["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(identity["bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(identity["sha256"]).encode("ascii"))
        digest.update(b"\0")
    return {
        "root": str(root),
        "files": list(files),
        "tree_sha256": digest.hexdigest(),
        "historical_four_file_identity": False,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=tuple(TRAINING_PROFILES),
        default="llama8b-d128-100b",
        help="named model, topology, schedule, and token-budget defaults",
    )
    parser.add_argument("--artifact-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dump-folder", required=True)
    parser.add_argument("--hf-assets-path", required=True)
    parser.add_argument(
        "--allow-nonhistorical-tokenizer",
        action="store_true",
        help=(
            "permit a tokenizer other than the authenticated four-file paper "
            "identity; the receipt is marked nonhistorical"
        ),
    )
    parser.add_argument(
        "--dataset-path",
        default="cerebras/SlimPajama-627B",
        help="public HF dataset id or an absolute local snapshot",
    )
    parser.add_argument(
        "--dataset-revision",
        default="",
        help="immutable 40-hex HF revision; required for a remote dataset id",
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        help="required exhaustive manifest when --dataset-path is local",
    )
    parser.add_argument("--world-size", type=int)
    parser.add_argument("--global-batch-size", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--validation-frequency", type=int)
    parser.add_argument("--validation-steps", type=int, default=16)
    parser.add_argument("--no-validation", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    profile = TRAINING_PROFILES[args.profile]
    for name in (
        "world_size",
        "global_batch_size",
        "steps",
        "warmup_steps",
        "checkpoint_interval",
        "validation_frequency",
    ):
        if getattr(args, name) is None:
            setattr(args, name, getattr(profile, name))
    for name in (
        "world_size",
        "global_batch_size",
        "steps",
        "warmup_steps",
        "checkpoint_interval",
        "validation_frequency",
        "validation_steps",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    return args


def _source_roots(manifest):
    assert manifest.runtime_source is not None
    assert manifest.flash_interface is not None
    assert manifest.cutlass_dsl is not None
    source_root = manifest.runtime_source.path.parents[2]
    if manifest.runtime_source.path.relative_to(source_root).as_posix() != (
        "tk_fa4/lowp_fa4_bwd/benchmark_llama12b_e2e.py"
    ):
        raise ValueError("runtime source does not live in the expected source tree")
    flash_root = manifest.flash_interface.path.parents[2]
    if manifest.flash_interface.path.relative_to(flash_root).as_posix() != (
        "flash_attn/cute/interface.py"
    ):
        raise ValueError("FlashAttention interface does not live below its root")
    return source_root, flash_root


def _dataset(args: argparse.Namespace) -> tuple[str, dict[str, object]]:
    local = Path(args.dataset_path).expanduser()
    if local.exists():
        if not local.is_dir():
            raise ValueError(f"local dataset path must be a directory: {local}")
        if args.dataset_revision:
            raise ValueError("a local dataset snapshot must not set --dataset-revision")
        if args.dataset_manifest is None:
            raise ValueError("a local dataset snapshot requires --dataset-manifest")
        identity = verify_dataset_manifest(
            args.dataset_manifest,
            expected_root=local,
        )
        manifest_sha256 = hashlib.sha256(identity.path.read_bytes()).hexdigest()
        return str(identity.root), {
            "kind": "local_snapshot",
            "path": str(identity.root),
            "manifest": str(identity.path),
            "manifest_sha256": manifest_sha256,
            "tree_sha256": identity.tree_sha256,
            "file_count": len(identity.files),
        }
    if local.is_absolute():
        raise FileNotFoundError(f"local dataset path does not exist: {local}")
    if args.dataset_manifest is not None:
        raise ValueError("a remote dataset id must not set --dataset-manifest")
    if _COMMIT.fullmatch(args.dataset_revision) is None:
        raise ValueError(
            "a remote dataset requires --dataset-revision as an immutable "
            "lowercase 40-hex commit"
        )
    resolved = f"{args.dataset_path}@{args.dataset_revision}"
    return resolved, {
        "kind": "huggingface_revision",
        "identifier": args.dataset_path,
        "revision": args.dataset_revision,
        "resolved": resolved,
    }


def _render(args: argparse.Namespace) -> tuple[str, dict[str, object]]:
    manifest = load_artifact_manifest(args.artifact_manifest, require_training=True)
    profile = TRAINING_PROFILES[args.profile]
    expected_artifact_profile = profile.artifact_profile.format(batch=manifest.batch)
    if manifest.profile != expected_artifact_profile:
        raise ValueError(
            f"training profile {profile.name!r} requires artifact profile "
            f"{expected_artifact_profile!r}; got {manifest.profile!r}"
        )
    if profile.local_batch is not None and manifest.batch != profile.local_batch:
        raise ValueError(
            f"training profile {profile.name!r} requires local batch "
            f"{profile.local_batch}; got B{manifest.batch}"
        )
    if profile.name == "llama1p2b-d64-50b" and (
        args.world_size != profile.world_size
        or args.global_batch_size != profile.global_batch_size
    ):
        raise ValueError(
            "llama1p2b-d64-50b fixes world size 16 and global batch 256 "
            "(gradient accumulation one)"
        )
    observed_attention_shape = (
        manifest.sequence,
        manifest.q_heads,
        manifest.kv_heads,
        manifest.head_dim,
    )
    expected_attention_shape = (
        profile.sequence,
        profile.query_heads,
        profile.key_value_heads,
        profile.head_dimension,
    )
    if observed_attention_shape != expected_attention_shape:
        raise ValueError(
            f"training profile {profile.name!r} requires attention shape "
            f"{expected_attention_shape!r}; got {observed_attention_shape!r}"
        )
    if manifest.head_dim == 128 and manifest.batch not in (1, 4):
        raise ValueError("D128 native-score training supports B1 or B4 only")
    if manifest.head_dim == 64 and manifest.batch != 16:
        raise ValueError("D64 native-v416 training supports B16 only")
    denominator = manifest.batch * args.world_size
    if args.global_batch_size % denominator:
        raise ValueError(
            "global batch must be divisible by local batch times world size: "
            f"{args.global_batch_size} % {denominator}"
        )
    accumulation = args.global_batch_size // denominator
    dump_folder = _absolute(args.dump_folder, "dump folder", must_exist=False)
    hf_assets = _absolute(args.hf_assets_path, "HF assets path", must_exist=True)
    tokenizer_identity = _tokenizer_identity(
        hf_assets,
        allow_nonhistorical=args.allow_nonhistorical_tokenizer,
    )
    dataset_path, dataset_identity = _dataset(args)
    source_root, flash_root = _source_roots(manifest)
    manifest_sha256 = hashlib.sha256(manifest.path.read_bytes()).hexdigest()

    if manifest.route == "bf16_fa4":
        converters = [
            "bfloat16",
            "fa4_exact_bf16_topology",
            "fa4_attention",
        ]
        route_description = "BF16 FA4 control"
    else:
        converters = ["bfloat16", "fa4_exact_lowp_attention"]
        projection_name = (
            "NVFP4" if manifest.learned_projection_format == "nvfp4" else "E4M3"
        )
        pv_name = "FP8" if manifest.pv_format == "e4m3_fp8" else "MXFP4"
        route_description = (
            f"{projection_name} learned projections, NVFP4 Q/K, {pv_name} P/V"
        )

    validation_enabled = not args.no_validation
    rendered_tokens = args.steps * args.global_batch_size * manifest.sequence
    if (
        args.steps == profile.steps
        and args.global_batch_size == profile.global_batch_size
        and rendered_tokens != profile.target_tokens
    ):
        raise AssertionError(
            f"profile {profile.name!r} token-budget drift: "
            f"{rendered_tokens} != {profile.target_tokens}"
        )
    lines = [
        "# Generated by tools/render_fa4_training_config.py.",
        f"# Artifact manifest: {manifest.path}",
        f"# Artifact manifest SHA256: {manifest_sha256}",
        f"# Recipe profile: {profile.name}",
        f"# Token budget: {rendered_tokens}",
        f"# Gradient accumulation: {accumulation}",
        "# This portable recipe uses fused BF16-SR AdamW; the historical "
        "1.2B launch templates used ordinary AdamW.",
        "",
        "[job]",
        'custom_config_module = "torchtitan.experiments.fa4"',
        f"dump_folder = {_quote(str(dump_folder))}",
        f"description = {_quote(route_description)}",
        "",
        "[experimental]",
        'custom_import = "torchtitan.experiments.fa4"',
        "",
        "[metrics]",
        "log_freq = 25",
        "enable_tensorboard = true",
        'save_tb_folder = "tb"',
        "",
        "[model]",
        'name = "llama3_gc"',
        f"flavor = {_quote(profile.model_flavor)}",
        f"hf_assets_path = {_quote(str(hf_assets))}",
        f"converters = {json.dumps(converters)}",
        f"# Exact adapter requires tied_embeddings={str(profile.tied_embeddings).lower()}.",
        "rope_theta = 500000.0",
        "max_seq_len = 4096",
        "",
        "[model.rope_scaling_args]",
        f"scaling_factor = {profile.rope_scaling_factor}",
        "low_freq_factor = 1.0",
        "high_freq_factor = 4.0",
        "original_max_position_embeddings = 8192",
        "",
        "[optimizer]",
        'name = "AdamWBF16SR"',
        'implementation = "fused"',
        f"lr = {profile.learning_rate}",
        "beta1 = 0.9",
        "beta2 = 0.95",
        "eps = 1e-8",
        "weight_decay = 0.1",
        "",
        "[training]",
        f"local_batch_size = {manifest.batch}",
        f"global_batch_size = {args.global_batch_size}",
        "seq_len = 4096",
        "max_norm = 1.0",
        f"steps = {args.steps}",
        'dtype = "bfloat16"',
        'mixed_precision_param = "bfloat16"',
        'mixed_precision_reduce = "bfloat16"',
        "enable_fp32_master_params = false",
        "enable_cce = false",
        "compile = false",
        'dataset = "slimpajama"',
        f"dataset_path = {_quote(dataset_path)}",
        "",
        "[validation]",
        f"enable = {str(validation_enabled).lower()}",
        f"freq = {args.validation_frequency}",
        f"steps = {args.validation_steps}",
        "local_batch_size = " + str(profile.validation_local_batch or manifest.batch),
        "seq_len = 4096",
        'dataset = "slimpajama_val"',
        f"dataset_path = {_quote(dataset_path)}",
        "",
        "[compile]",
        "enable = true",
        'components = ["loss"]',
        'backend = "inductor"',
        "",
        "[lr_scheduler]",
        f"warmup_steps = {args.warmup_steps}",
        'decay_type = "cosine"',
        "min_lr_factor = 0.01",
        "",
        "[parallelism]",
        "pipeline_parallel_degree = 1",
        f"data_parallel_replicate_degree = {args.world_size}",
        "data_parallel_shard_degree = 1",
        "tensor_parallel_degree = 1",
        "context_parallel_degree = 1",
        "",
        "[checkpoint]",
        "enable = true",
        "enable_ft_dataloader_checkpoints = false",
        'folder = "checkpoint"',
        f"interval = {args.checkpoint_interval}",
        "load_step = -1",
        'export_dtype = "float32"',
        'async_mode = "disabled"',
        "keep_latest_k = 3",
        "last_save_model_only = false",
        "",
        "[comm]",
        "init_timeout_seconds = 1800",
        "train_timeout_seconds = 3600",
        "",
        "[activation_checkpoint]",
        'mode = "none"',
        "",
        "[debug]",
        f"seed = {args.seed}",
        "",
        "[fa4]",
        "enabled = true",
        "cuda_data_prefetch = true",
        "fail_on_nonfinite_metrics = true",
        "scan_nonfinite_gradients = false",
        "gradient_diagnostics_topk = 0",
        'mode = "softmax"',
        f"exact_source_root = {_quote(str(source_root))}",
        f"exact_runtime_source_sha256 = {_quote(manifest.runtime_source.sha256)}",
        f"exact_flash_attn_root = {_quote(str(flash_root))}",
        f"exact_flash_attn_source_sha256 = {_quote(manifest.flash_interface.sha256)}",
        f"exact_cutlass_dsl_root = {_quote(str(manifest.cutlass_dsl.root))}",
        f"exact_cutlass_dsl_version = {_quote(manifest.cutlass_dsl.version)}",
        "exact_cutlass_dsl_native_sha256 = "
        + _quote(manifest.cutlass_dsl.native.sha256),
        f"exact_forward_batch_size = {manifest.batch}",
        "exact_allow_fp32_master_shadows = false",
        f"exact_artifact_profile = {_quote(manifest.profile)}",
    ]
    if manifest.route != "bf16_fa4":
        assert manifest.forward is not None
        assert manifest.projection_publisher is not None
        assert manifest.native_backward is not None
        lines.extend(
            [
                f"exact_forward_extension = {_quote(str(manifest.forward.path))}",
                f"exact_forward_module = {_quote(manifest.forward.module or '')}",
                f"exact_forward_sha256 = {_quote(manifest.forward.sha256)}",
                f"exact_pv_format = {_quote(manifest.pv_format or '')}",
                "exact_learned_projection_format = "
                + _quote(manifest.learned_projection_format or ""),
                'exact_mx_v_publication = "retained_split"',
                "exact_backward_extension = "
                + _quote(str(manifest.projection_publisher.path)),
                "exact_backward_sha256 = "
                + _quote(manifest.projection_publisher.sha256),
            ]
        )
        if manifest.head_dim == 64:
            assert manifest.v416_backward is not None
            lines.extend(
                [
                    "exact_native_tk_d64_backward_extension = "
                    + _quote(str(manifest.v416_backward.path)),
                    "exact_native_tk_d64_backward_module = "
                    + _quote(manifest.v416_backward.module or ""),
                    "exact_native_tk_d64_backward_sha256 = "
                    + _quote(manifest.v416_backward.sha256),
                    "exact_native_tk_d64_backward_bytes = "
                    + str(manifest.v416_backward.bytes),
                ]
            )
        else:
            assert manifest.v509_backward is not None
            lines.extend(
                [
                    "exact_d128_represented_qk_backward = false",
                    "exact_d128_native_score_backward = true",
                    "exact_d128_e5m2_dout_backward = true",
                    "exact_native_tk_d128_backward_extension = "
                    + _quote(str(manifest.v509_backward.path)),
                    "exact_native_tk_d128_backward_module = "
                    + _quote(manifest.v509_backward.module or ""),
                    "exact_native_tk_d128_backward_sha256 = "
                    + _quote(manifest.v509_backward.sha256),
                    "exact_native_tk_d128_backward_bytes = "
                    + str(manifest.v509_backward.bytes),
                    'exact_backward_control_source = ""',
                    'exact_backward_control_sha256 = ""',
                    "exact_backward_control_bytes = 0",
                ]
            )
    lines.append("")
    rendered_config = "\n".join(lines)
    rendered_bytes = rendered_config.encode("utf-8")
    output = args.output.expanduser().resolve()
    receipt_path = Path(str(output) + ".receipt.json")
    receipt = {
        "schema": "fa4_training_config_receipt_v2",
        "config": str(output),
        "config_relative_to_receipt": os.path.relpath(output, receipt_path.parent),
        "config_bytes": len(rendered_bytes),
        "config_sha256": hashlib.sha256(rendered_bytes).hexdigest(),
        "artifact_manifest": str(manifest.path),
        "artifact_manifest_sha256": manifest_sha256,
        "route": manifest.route,
        "shape": {
            "batch": manifest.batch,
            "sequence": manifest.sequence,
            "q_heads": manifest.q_heads,
            "kv_heads": manifest.kv_heads,
            "head_dim": manifest.head_dim,
        },
        "world_size": args.world_size,
        "global_batch_size": args.global_batch_size,
        "gradient_accumulation_steps": accumulation,
        "trainer_module": "torchtitan.experiments.fa4.train",
        "training_integration": {
            "cuda_data_prefetch": True,
            "checkpoint_aligned_lookahead": True,
            "fail_on_nonfinite_metrics": True,
            "scan_nonfinite_gradients": False,
            "gradient_diagnostics_topk": 0,
        },
        "dataset": dataset_identity,
        "tokenizer": tokenizer_identity,
        "required_environment": REQUIRED_ENVIRONMENT,
    }
    return rendered_config, receipt


def main() -> None:
    args = _parse_args()
    output = args.output.expanduser().resolve()
    receipt_path = Path(str(output) + ".receipt.json")
    for path in (output, receipt_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    config, receipt = _render(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(config.encode("utf-8"))
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(output)
    print(receipt_path)


if __name__ == "__main__":
    main()
